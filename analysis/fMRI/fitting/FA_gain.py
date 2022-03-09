################################################
#
#   Fit gain model on FA runs  
#
################################################

import numpy as np
import os, sys
import os.path as op
import yaml
from pathlib import Path
import glob

from FAM_utils import mri as mri_utils

import pandas as pd
import numpy as np


# requires pfpy to be installed - preferably with python setup.py develop
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D

from joblib import Parallel, delayed

import datetime
from tqdm import tqdm

from lmfit import Parameters

from feature_model import FA_GainModel


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

## some settings
base_dir = params['general']['current_dir'] # which machine we run the data
acq = params['mri']['acq'] # if using standard files or nordic files
space = params['mri']['space'] # subject space

mask_prf = True # if we're masking pRFs

## define file extension that we want to use, 
# should include processing key words
file_ext = ''
# if cropped first
if params['feature']['crop']:
    file_ext += '_{name}'.format(name='cropped')
# type of filtering/denoising
if params['feature']['regress_confounds']:
    file_ext += '_{name}'.format(name='confound')
else:
    file_ext += '_{name}'.format(name = params['mri']['filtering']['type'])
# type of standardization 
file_ext += '_{name}'.format(name = params['feature']['standardize'])
# don't forget its a numpy array
file_ext += '.npy'

### define model
fa_model = FA_GainModel(params)

## set paths
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
postfmriprep_dir = op.join(derivatives_dir, 'post_fmriprep',
                           'sub-{sj}'.format(sj=sj), space,'processed')

source_dir = glob.glob(op.join(params['mri']['paths'][base_dir]['root'], 'sourcedata', 
                               'sub-{sj}'.format(sj=sj), 'ses-*', 'func'))[0] 

output_dir =  op.join(derivatives_dir,'FA_gain','sub-{sj}'.format(sj=sj), space, 
                      fa_model.prf_model_type, 'run-{run}'.format(run=run))

## check if path to save fit estimates exists
if not op.exists(output_dir): 
    os.makedirs(output_dir) 

## list with absolute file name to be fitted
proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-FA' in h and
                 'acq-{acq}'.format(acq=acq) in h and 'run-{run}'.format(run=run) in h and h.endswith(file_ext)]

## load functional data
file = proc_files[0]
if len(proc_files)>1:
    raise ValueError('%s files found to fit, unsure of which to use'%len(proc_files))
else:
    print('Fitting %s'%file)
data = np.load(file,allow_pickle=True) # will be (vertex, TR)

##### Load pRF estimates ####
#### to use in FA model ####

# path to pRF fits 
prf_fits_pth =  op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), space, 
                        'iterative_{model}'.format(model = fa_model.prf_model_type),
                        'run-{run}'.format(run=fa_model.prf_run_type))

# load them into numpy dict
pRF_estimates = fa_model.get_pRF_estimates(prf_fits_pth, params['mri']['fitting']['pRF']['total_chunks'][space])

# if we want to mask pRFs, given screen limits and behavior responses
if mask_prf: 
    
    print('masking pRF estimates')
    
    ## make pRF DM mask, according to sub responses
    # list of behavior files
    behav_files = [op.join(source_dir, h) for h in os.listdir(source_dir) if 'task-pRF' in h and
                     h.endswith('events.tsv')]
    # behav boolean mask
    DM_mask_beh = mri_utils.get_beh_mask(behav_files,params)


    pRF_estimates = fa_model.mask_pRF_estimates(prf_fits_pth.split(space)[0], DM_mask_beh)
    fa_model.pRF_estimates = pRF_estimates


## rsq mask, get indices for vertices where pRF 
# rsq is greater than threshold
rsq_threshold = 0.12
mask_ind = np.array([ind for ind,val in enumerate(pRF_estimates['rsq']) if val > rsq_threshold])

# saved masked pRF rsq in output dir (not rsq thresholded), for plotting purposes
print('saving masked pRF rsq in %s'%op.split(output_dir)[0])
np.save(op.join(op.split(output_dir)[0], 'masked_pRF_rsq.npy'), pRF_estimates['rsq'])   

##### load bar position for FA #####
##### and make DM for run #####

# get absolute path to pkl with bar positions for each run
bar_pos_files = [op.join(source_dir, h) for h in os.listdir(source_dir) if 'task-FA' in h and
                'run-{run}'.format(run=run) in h and h.endswith('_bar_positions.pkl')]
print('getting bar positions from %s'%bar_pos_files[0])

# load bar positions for run
bar_pos = pd.read_pickle(bar_pos_files[0])

# get absolute path to csv with general infos for each run
trial_info_files = [op.join(source_dir, h) for h in os.listdir(source_dir) if 'task-FA' in h and
                'run-{run}'.format(run=run) in h and h.endswith('_trial_info.csv')]

# load trial info dataframe
trial_info = pd.read_csv(trial_info_files[0])

## make visual FA DM (spatial postions over time)
fa_model.bar_pos = bar_pos
fa_model.trial_info = trial_info

fa_model.make_FA_visual_DM(fa_model.unique_cond.keys(), crop = False, shift_TRs = False,
                          crop_unit = 'sec', oversampling_time = None)

# create upsampled hrf
hrf_params = np.ones((3, fa_model.pRF_estimates['rsq'].shape[0]))

if fa_model.fit_hrf: # use fitted hrf params
    hrf_params[1] = fa_model.pRF_estimates['hrf_derivative']
    hrf_params[2] = fa_model.pRF_estimates['hrf_dispersion']
    
    #hrf_oversampled = mri_utils.create_hrf(hrf_params = hrf_params, TR = fa_model.TR, osf = fa_model.osf) 
else:
    #hrf_oversampled = np.tile(mri_utils.create_hrf(TR = fa_model.TR, osf = fa_model.osf), 
    #                          (hrf_params.shape[-1],1))
    hrf_params[2] = 0


##set all necessary parameters used for 
# gain fit - also setting which ones we fit or not
fa_pars = Parameters()

# add pRF parameters - will not vary
fa_pars.add('pRF_x', value = 0, vary = False)
fa_pars.add('pRF_y', value = 0, vary = False)
fa_pars.add('pRF_size', value = 0, vary = False)
fa_pars.add('pRF_beta', value = 0, vary = False, min = -1.e-10)
fa_pars.add('pRF_baseline', value = 0, vary = False, min = -10, max = 10)
fa_pars.add('pRF_n', value = 1, vary = False)

# add gain params for each bar - will vary
fa_pars.add('gain_ACAO', value = 1, vary = False, min = -1.e-10, max = 1.00001) # attended condition doesnt vary - pRF task was attended to bar
fa_pars.add('gain_ACUO', value = 0, vary = True, min = -1.e-10, max = 1.00001)
fa_pars.add('gain_UCAO', value = 0, vary = True, min = -1.e-10, max = 1.00001)
fa_pars.add('gain_UCUO', value = 0, vary = True, min = -1.e-10, max = 1.00001)

# add params that will be filled by GLM - will not vary
fa_pars.add('beta_cue_0', value = 0, vary = False)
fa_pars.add('beta_cue_1', value = 0, vary = False)
fa_pars.add('beta_cue_2', value = 0, vary = False)
fa_pars.add('beta_cue_3', value = 0, vary = False)
fa_pars.add('beta_bar_stim', value = 0, vary = False)
fa_pars.add('intercept', value = 1, vary = False)
fa_pars.add('rsq', value = 0, vary = False)


## fit it!
results = fa_model.iterative_fit(data, fa_pars, 
                                     hrf_params = hrf_params, 
                                     mask_ind = mask_ind)

## save fitted params Dataframe
results.to_csv(op.join(output_dir,'iterative_params.csv'), index = False)

