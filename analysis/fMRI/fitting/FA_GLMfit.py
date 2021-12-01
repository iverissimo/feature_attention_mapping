################################################
#
#   Fit post-pRF GLM on FA runs  
#
################################################

import numpy as np
import os, sys
import os.path as op
import yaml
from pathlib import Path
import glob

sys.path.insert(0,'..') # add parent folder to path
from utils import * #import script to use relevante functions

import pandas as pd
import numpy as np

# requires pfpy to be installed - preferably with python setup.py develop
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter

from popeye import utilities

from nistats.design_matrix import make_first_level_design_matrix
from nistats.reporting import plot_design_matrix

from joblib import Parallel, delayed

import datetime
from tqdm import tqdm

# load settings from yaml
with open(op.join(str(Path(os.getcwd()).parents[1]),'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)


# define participant number and run 

if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')
  
elif len(sys.argv) < 3:
    raise NameError('Please add run to be fitted (ex: 1) '
                    'as 2nd argument in the command line!')

else:
    # fill subject number and chunk number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(3)
    run = str(sys.argv[2])

# print start time, for bookeeping
start_time = datetime.datetime.now()

# some settings
base_dir = params['general']['current_dir'] # which machine we run the data
acq = params['mri']['acq'] # if using standard files or nordic files
space = params['mri']['space'] # subject space
total_chunks = params['mri']['fitting']['pRF']['total_chunks'][space] # number of chunks that data was split in

TR = params['mri']['TR']

# type of pRF model to use, and run type (mean vs median)
model_type = params['mri']['fitting']['pRF']['fit_model']
run_type = 'mean'

# define file extension that we want to use, 
# should include processing key words
file_ext = '_cropped_{filt}_{stand}.npy'.format(filt = params['mri']['filtering']['type'],
                                                    stand = 'psc')

# set paths
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
postfmriprep_dir = op.join(derivatives_dir,'post_fmriprep','sub-{sj}'.format(sj=sj),space,'processed')

output_dir =  op.join(derivatives_dir,'FA_GLM_fit','sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run))

# check if path to save processed files exist
if not op.exists(output_dir): 
    os.makedirs(output_dir) 
    #also make gauss dir, to save intermediate estimates
    if model_type!='gauss':
        os.makedirs(output_dir.replace(model_type,'gauss')) 


# list with absolute file name to be fitted
proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-FA' in h and
                 'acq-{acq}'.format(acq=acq) in h and 'run-{run}'.format(run=run) in h and h.endswith(file_ext)]

## load functional data
file = proc_files[0]
data = np.load(file,allow_pickle=True) # will be (vertex, TR)

# path to pRF fits 
fits_pth =  op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), space, 'iterative_{model}'.format(model=model_type),'run-{run}'.format(run=run_type))

## Load pRF estimates 
    
# path to combined estimates
pRF_estimates_pth = op.join(fits_pth,'combined')

# combined estimates filename
est_name = [x for _,x in enumerate(os.listdir(fits_pth)) if 'chunk-001' in x][0]
est_name = est_name.replace('chunk-001_of_{ch}'.format(ch=str(total_chunks).zfill(3)),'chunk-combined')

# total path to estimates path
pRF_estimates_combi = op.join(pRF_estimates_pth,est_name)

if op.isfile(pRF_estimates_combi): # if combined estimates exists

    print('loading %s'%pRF_estimates_combi)
    pRF_estimates = np.load(pRF_estimates_combi) # load it

else: # if not join chunks and save file
    if not op.exists(pRF_estimates_pth):
        os.makedirs(pRF_estimates_pth) 

    pRF_estimates = join_chunks(fits_pth, pRF_estimates_combi,
                        chunk_num = total_chunks, fit_model = 'it{model}'.format(model=model_type)) #'{model}'.format(model=model_type)))#
    
# save estimates in specific variables
xx = pRF_estimates['x']
yy = pRF_estimates['y']

size = pRF_estimates['size']

beta = pRF_estimates['betas']
baseline = pRF_estimates['baseline']

if 'css' in model_type:
    ns = pRF_estimates['ns']

rsq = pRF_estimates['r2']

## load bar position for FA
## and make DM for run

# path to events csv, pickle etc
events_pth = glob.glob(op.join(derivatives_dir.replace('derivatives','sourcedata'),'sub-{sj}'.format(sj=sj), 'ses-*', 'func'))[0]

# get absolute path to pkl with bar positions for each run
bar_pos_files = [op.join(events_pth, h) for h in os.listdir(events_pth) if 'task-FA' in h and
                'run-{run}'.format(run=run) in h and h.endswith('_bar_positions.pkl')]

# load bar positions for run
bar_pos = pd.read_pickle(bar_pos_files[0])

# get absolute path to csv with general infos for each run
trial_info_files = [op.join(events_pth, h) for h in os.listdir(events_pth) if 'task-FA' in h and
                'run-{run}'.format(run=run) in h and h.endswith('_trial_info.csv')]

# load trial info dataframe
trial_info = pd.read_csv(trial_info_files[0])

### all 4 regressors to use in fit
all_regressors = {'ACAO': {'color': True, 'orientation': True},
                  'ACUO': {'color': True, 'orientation': False},
                  'UCAO': {'color': False, 'orientation': True},
                  'UCUO': {'color': False, 'orientation': False}
                    }

all_reg_predictions = [] # to append all regressor predictions

# make visual DM for each GLM regressor, and obtain prediction using pRF model
for reg in all_regressors.keys():
    
    # filename for regressor dm
    DM_reg_filename = op.join(output_dir,'DM_regressor-{reg}.npy'.format(reg=reg))
    # filename for regressor prediction
    reg_filename = DM_reg_filename.replace('DM','prediction')
    
    if not op.exists(reg_filename): # check if file already exists
        # make array with spatial position of bar of interest 
        DM_cond = get_FA_bar_stim(DM_reg_filename, 
                            params, bar_pos, trial_info, attend_cond = all_regressors[reg], 
                            save_imgs = False, downsample = 0.1, crop = params['feature']['crop'] , 
                            crop_TR = params['feature']['crop_TR'], overwrite=True)
        
        # make stimulus object, which takes an input design matrix and sets up its real-world dimensions
        prf_stim = PRFStimulus2D(screen_size_cm = params['monitor']['height'],
                                 screen_distance_cm = params['monitor']['distance'],
                                 design_matrix = DM_cond,
                                 TR = TR)

        # get prediction
        if model_type == 'css':

            # define CSS model 
            css_model = CSS_Iso2DGaussianModel(stimulus = prf_stim,
                                         filter_predictions = True,
                                         filter_type = params['mri']['filtering']['type'],
                                         filter_params = {'highpass': params['mri']['filtering']['highpass'],
                                                         'add_mean': params['mri']['filtering']['add_mean'],
                                                         'window_length': params['mri']['filtering']['window_length'],
                                                         'polyorder': params['mri']['filtering']['polyorder']}
                                        )

            model_fit = Parallel(n_jobs=16)(delayed(css_model.return_prediction)(xx[vert], 
                                                                                 yy[vert],
                                                                                 size[vert],
                                                                                 beta[vert],
                                                                                 baseline[vert],
                                                                                 ns[vert]) for vert in tqdm(range(len(xx))))

        else:
            # define gaussian model 
            gauss_model = Iso2DGaussianModel(stimulus = prf_stim,
                                         filter_predictions = True,
                                         filter_type = params['mri']['filtering']['type'],
                                         filter_params = {'highpass': params['mri']['filtering']['highpass'],
                                                         'add_mean': params['mri']['filtering']['add_mean'],
                                                         'window_length': params['mri']['filtering']['window_length'],
                                                         'polyorder': params['mri']['filtering']['polyorder']}
                                        )

            model_fit = Parallel(n_jobs=16)(delayed(gauss_model.return_prediction)(xx[vert], 
                                                                                   yy[vert],
                                                                                   size[vert],
                                                                                   beta[vert],
                                                                                   baseline[vert]) for vert in tqdm(range(len(xx))))

        # squeeze out single dimension that parallel creates
        prediction = np.squeeze(model_fit)

        # save prediction in folder, for later use/plotting
        np.save(reg_filename,prediction)

    else:
        # load prediction
        prediction = np.load(reg_filename)
    
    ## append predictions in array, to use for FA GLM DM
    all_reg_predictions.append(prediction[np.newaxis,...])

all_reg_predictions = np.vstack(all_reg_predictions)

