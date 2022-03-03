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

screen_res = params['window']['size']
if params['window']['display'] == 'square': # if square display
    screen_res = np.array([screen_res[1], screen_res[1]])

TR = params['mri']['TR']

# type of pRF model to use, and run type (mean vs median)
model_type = params['mri']['fitting']['pRF']['fit_model']
run_type = params['mri']['fitting']['pRF']['run']
fit_hrf = params['mri']['fitting']['pRF']['fit_hrf'] 

# define file extension that we want to use, 
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

# set paths
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
postfmriprep_dir = op.join(derivatives_dir,'post_fmriprep','sub-{sj}'.format(sj=sj),space,'processed')

output_dir =  op.join(derivatives_dir,'FA_gain','sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run))

# check if path to save processed files exist
if not op.exists(output_dir): 
    os.makedirs(output_dir) 

# list with absolute file name to be fitted
proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-FA' in h and
                 'acq-{acq}'.format(acq=acq) in h and 'run-{run}'.format(run=run) in h and h.endswith(file_ext)]


## load functional data
file = proc_files[0]
if len(proc_files)>1:
    raise ValueError('%s files found to fit, unsure of which to use'%len(proc_files))
else:
    print('Fitting %s'%file)
data = np.load(file,allow_pickle=True) # will be (vertex, TR)

# path to pRF fits 
fits_pth =  op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), space, 'iterative_{model}'.format(model=model_type),'run-{run}'.format(run=run_type))

##### Load pRF estimates ####
    
# path to combined estimates
pRF_estimates_pth = op.join(fits_pth,'combined')

# combined estimates filename
est_name = [x for _,x in enumerate(os.listdir(fits_pth)) if 'chunk-001' in x]
if len(est_name)>1:
    raise ValueError('%s files found as pRF estimates of same chuck, unsure of which to use'%len(est_name))
else:
    est_name = est_name[0].replace('chunk-001_of_{ch}'.format(ch=str(total_chunks).zfill(3)),'chunk-combined')

# total path to estimates path
pRF_estimates_combi = op.join(pRF_estimates_pth,est_name)

if op.isfile(pRF_estimates_combi): # if combined estimates exists

    print('loading %s'%pRF_estimates_combi)
    pRF_estimates = np.load(pRF_estimates_combi) # load it

else: # if not join chunks and save file
    if not op.exists(pRF_estimates_pth):
        os.makedirs(pRF_estimates_pth) 

    pRF_estimates = mri_utils.join_chunks(fits_pth, pRF_estimates_combi, fit_hrf = params['mri']['fitting']['pRF']['fit_hrf'],
                        chunk_num = total_chunks, fit_model = 'it{model}'.format(model=model_type)) #'{model}'.format(model=model_type)))#

## make pRF DM mask, according to sub responses
source_dir = glob.glob(op.join(params['mri']['paths'][base_dir]['root'], 'sourcedata', 
                               'sub-{sj}'.format(sj=sj), 'ses-*', 'func'))[0] 
# list of behavior files
behav_files = [op.join(source_dir, h) for h in os.listdir(source_dir) if 'task-pRF' in h and
                 h.endswith('events.tsv')]
# behav boolean mask
DM_mask_beh = mri_utils.get_beh_mask(behav_files,params)

# define design matrix for pRF task
visual_dm = mri_utils.make_pRF_DM(op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), 'DMprf.npy'), params, 
                    save_imgs=False, res_scaling = 0.1, crop = params['prf']['crop'] , crop_TR = params['prf']['crop_TR'],
                    shift_TRs = True, shift_TR_num = 1, overwrite = True, mask = DM_mask_beh)

# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm = params['monitor']['height'],
                        screen_distance_cm = params['monitor']['distance'],
                        design_matrix = visual_dm,
                        TR = TR)

# get the ecc limits (in dva)
# to mask estimates
x_ecc_lim, y_ecc_lim = mri_utils.get_ecc_limits(visual_dm,params,screen_size_deg = [prf_stim.screen_size_degrees,prf_stim.screen_size_degrees])

# also compute limit in pixels
# to make spatial position mask for FA DM   
xy_lim_pix = {'x_lim': x_ecc_lim*screen_res[0]/prf_stim.screen_size_degrees,
              'y_lim': y_ecc_lim*screen_res[1]/prf_stim.screen_size_degrees}

# mask estimates, to be within screen boundaries
print('masking estimates')
masked_pRF_estimates = mri_utils.mask_estimates(pRF_estimates, fit_model = model_type,
                                        x_ecc_lim = x_ecc_lim, y_ecc_lim = y_ecc_lim)

# save estimates in specific variables
xx = masked_pRF_estimates['x']
yy = masked_pRF_estimates['y']

size = masked_pRF_estimates['size']

beta = masked_pRF_estimates['beta']
baseline = masked_pRF_estimates['baseline']

if 'css' in model_type:
    ns = masked_pRF_estimates['ns']

rsq = masked_pRF_estimates['rsq']

## rsq mask, get indices for vertices where pRF 
# rsq is greater than 0.05 
mask_ind = np.array([ind for ind,val in enumerate(rsq) if val>0.05])

# saved masked rsq, useful for FA plots
print('saving masked rsq in %s'%op.join(fits_pth,'combined'))
np.save(op.join(fits_pth,'combined','masked_rsq.npy'), rsq)

####

## load bar position for FA
## and make DM for run

# path to events csv, pickle etc
events_pth = glob.glob(op.join(derivatives_dir.replace('derivatives','sourcedata'),'sub-{sj}'.format(sj=sj), 'ses-*', 'func'))[0]

# get absolute path to pkl with bar positions for each run
bar_pos_files = [op.join(events_pth, h) for h in os.listdir(events_pth) if 'task-FA' in h and
                'run-{run}'.format(run=run) in h and h.endswith('_bar_positions.pkl')]
print('getting bar positions from %s'%bar_pos_files[0])

# load bar positions for run
bar_pos = pd.read_pickle(bar_pos_files[0])

# get absolute path to csv with general infos for each run
trial_info_files = [op.join(events_pth, h) for h in os.listdir(events_pth) if 'task-FA' in h and
                'run-{run}'.format(run=run) in h and h.endswith('_trial_info.csv')]

# load trial info dataframe
trial_info = pd.read_csv(trial_info_files[0])

## get DM for each of the 4 bar conditions
unique_cond = params['mri']['fitting']['FA']['condition_keys']

## get info on conditions in run (4 conditions x 4 miniblocks = 16)
all_conditions = pd.DataFrame(columns = ['reg_name', 'color','orientation','miniblock','run'])

for key in unique_cond.keys(): # for each condition
    
    for blk in range(params['feature']['mini_blocks']): # for each miniblock
        
        # name of attended condition in miniblock
        attended_cond = bar_pos.loc[(bar_pos['mini_block']==blk)&(bar_pos['attend_condition']==1)]['condition'].values[0]
        
        all_conditions = all_conditions.append(pd.DataFrame({'reg_name': '{cond}_mblk-{blk}_run-{run}'.format(cond=key,
                                                                                                             blk=blk,
                                                                                                             run=run),
                                                             'color': unique_cond[key]['color'],
                                                             'orientation': unique_cond[key]['orientation'],
                                                             'condition_name': mri_utils.get_cond_name(attended_cond,key),
                                                             'miniblock': blk,
                                                             'run': int(run)
                                                            }, index=[0]),ignore_index=True)

# task sampling rate might be different from trial
# so stimulus object TR needs to reflect that
FA_sampling_rate = TR if params['feature']['task_rate']=='TR' else params['feature']['task_rate']

## set oversampling factor
osf = 10 

# create upsampled hrf
hrf_params = np.ones((3, pRF_estimates['r2'].shape[0]))
if fit_hrf: # use fitted hrf params
    hrf_params[1] = pRF_estimates['hrf_derivative']
    hrf_params[2] = pRF_estimates['hrf_dispersion']
    
    hrf_oversampled = mri_utils.create_hrf(hrf_params = hrf_params, TR = TR, osf = osf) 
else:
    hrf_oversampled = np.tile(mri_utils.create_hrf(TR = TR, osf = osf), (pRF_estimates['r2'].shape[0],1))
    hrf_params[2] = 0

## get upsampled DM for all conditions (num bars, x, y, samples)

all_cond_DM = {} # store in dict, to keep track

for key in unique_cond.keys(): # for each condition
    
    # filename for condition dm
    DM_cond_filename = op.join(output_dir,'DM_condition-{cond}_run-{run}_osf-{osf}.npy'.format(cond = key, 
                                                                                              run = run,
                                                                                              osf = int(osf)))
    if not op.exists(DM_cond_filename): # check if file already exists
        
        # sub-select regressors for that condition 
        regs = [val for _,val in enumerate(all_conditions['reg_name'].values) if key in val]
        
        all_regs_dm = []
        
        for reg in regs:
            
            # make array with spatial position of bar of interest 
            reg_dm = mri_utils.get_FA_bar_stim(op.join(output_dir,'DM_regressor-{reg}.npy'.format(reg=reg)), 
                                params, bar_pos, trial_info, TR = TR,
                                attend_cond = all_conditions[all_conditions['reg_name']==reg].to_dict('r')[0], 
                                save_imgs = False, res_scaling = 0.1, oversampling_time = None, 
                                stim_dur_seconds = params['feature']['bars_phase_dur'], 
                                crop = False, crop_unit = 'sec', 
                                crop_TR = params['feature']['crop_TR'],
                                shift_TRs = False, shift_TR_num = 1.5, # to account for lack of slicetimecorrection
                                overwrite = True, save_DM = False)

            all_regs_dm.append(reg_dm)
        
        # save condition dm
        DM_cond = np.amax(np.array(all_regs_dm),axis=0)
        np.save(DM_cond_filename, DM_cond)
        
    else:
        # load condition dm
        DM_cond = np.load(DM_cond_filename)
        print('file already exists, loading %s'%DM_cond_filename)
        
    # append in all condition DM
    all_cond_DM[key] = DM_cond

 ## stack DM for efficiency
cond_DM_stacked = np.stack((all_cond_DM[k].astype(np.float32) for k in all_cond_DM.keys()),axis = 0)
   
## set starting params
starting_params = {'gain_ACAO': 1, 'gain_ACUO': 0, 'gain_UCAO': 0,'gain_UCUO':0}

## set all necessary parameters used for 
# gain fit - also setting which ones we fit or not
pars = Parameters()

# add pRF parameters - will not vary
pars.add('pRF_x', value = 0, vary = False)
pars.add('pRF_y', value = 0, vary = False)
pars.add('pRF_size', value = 0, vary = False)
pars.add('pRF_beta', value = 0, vary = False, min = -1.e-10)
pars.add('pRF_baseline', value = 0, vary = False, min = -10, max = 10)
pars.add('pRF_n', value = 1, vary = False)

# add gain params for each bar - will vary
pars.add('gain_ACAO', value = starting_params['gain_ACAO'], vary = False, min = -1.e-10, max = 1.00001) # attended condition doesnt vary - pRF task was attended to bar
pars.add('gain_ACUO', value = starting_params['gain_ACUO'], vary = True, min = -1.e-10, max = 1.00001)
pars.add('gain_UCAO', value = starting_params['gain_UCAO'], vary = True, min = -1.e-10, max = 1.00001)
pars.add('gain_UCUO', value = starting_params['gain_UCUO'], vary = True, min = -1.e-10, max = 1.00001)

# add params that will be filled by GLM - will not vary
pars.add('FA_beta', value = 0, vary = False)
pars.add('cue_beta', value = 0, vary = False)
pars.add('intercept', value = 1, vary = False)

## fit it!
results = np.array(Parallel(n_jobs=16)(delayed(mri_utils.get_gain_fit_params)(data[vertex], 
                                                                    pars,
                                                                   params = params, trial_info = trial_info,
                                                                   all_cond_DM = cond_DM_stacked,
                                        hrf_params = hrf_params[...,vertex], pRFmodel = model_type,
                                        xx = xx[vertex], yy = yy[vertex], size = size[vertex],
                                        betas = beta[vertex], baseline = baseline[vertex], ns = ns[vertex]) 
                                for _,vertex in enumerate(tqdm(mask_ind))))

## save fitted params list of dicts as Dataframe
fitted_params_df = pd.DataFrame(d for d in results)
# and add vertex number for bookeeping
fitted_params_df['vertex'] = mask_ind

## save in dir
fitted_params_df.to_csv(op.join(output_dir,'iterative_gain_params.csv'),index=False)
