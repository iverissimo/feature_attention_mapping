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

from FAM_utils import mri as mri_utils

import pandas as pd
import numpy as np

import scipy.signal as signal

# requires pfpy to be installed - preferably with python setup.py develop
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel

from joblib import Parallel, delayed

import datetime
from tqdm import tqdm
from scipy import interpolate

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

output_dir =  op.join(derivatives_dir,'FA_GLM_fit','sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run))

# check if path to save processed files exist
if not op.exists(output_dir): 
    os.makedirs(output_dir) 

# list with absolute file name to be fitted
proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-FA' in h and
                 'acq-{acq}'.format(acq=acq) in h and 'run-{run}'.format(run=run) in h and h.endswith(file_ext)]

# exception for sub 4, run 4 because nordic failed for FA
if sj=='004' and run=='4':
    proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-FA' in h and
                 'acq-{acq}'.format(acq='standard') in h and 'run-{run}'.format(run=run) in h and h.endswith(file_ext)]

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

# define design matrix for pRF task
visual_dm = mri_utils.make_pRF_DM(op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), 'DMprf.npy'), params, 
                    save_imgs=False, downsample=0.1, crop = params['prf']['crop'] , crop_TR = params['prf']['crop_TR'], overwrite=False)

# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm = params['monitor']['height'],
                        screen_distance_cm = params['monitor']['distance'],
                        design_matrix = visual_dm,
                        TR = TR)

# get the ecc limits (in dva)
# to mask estimates
x_ecc_lim, y_ecc_lim = mri_utils.get_ecc_limits(visual_dm,params,screen_size_deg = [prf_stim.screen_size_degrees,prf_stim.screen_size_degrees])

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

### use 4x4 regressors in fit - 4 conditions x 4 miniblocks
unique_cond = params['mri']['fitting']['FA']['condition_keys']

all_regressors = pd.DataFrame(columns = ['reg_name', 'color','orientation','miniblock','run'])

for key in unique_cond.keys(): # for each condition
    
    for blk in range(params['feature']['mini_blocks']): # for each miniblock
        
        # name of attended condition in miniblock
        attended_condition = bar_pos.loc[(bar_pos['mini_block']==blk)&(bar_pos['attend_condition']==1)]['condition'].values[0]
        
        all_regressors = all_regressors.append(pd.DataFrame({'reg_name': '{cond}_mblk-{blk}_run-{run}'.format(cond=key,
                                                                                                             blk=blk,
                                                                                                             run=run),
                                                             'color': unique_cond[key]['color'],
                                                             'orientation': unique_cond[key]['orientation'],
                                                             'condition_name': mri_utils.get_cond_name(attended_condition,key),
                                                             'miniblock': blk,
                                                             'run': int(run)
                                                            }, index=[0]),ignore_index=True)

all_reg_predictions = [] # to append all regressor predictions

# task sampling rate might be different from trial
# so stimulus object TR needs to reflect that
FA_sampling_rate = TR if params['feature']['task_rate']=='TR' else params['feature']['task_rate']

# set oversampling factor
osf = 10

# create upsampled hrf
if fit_hrf: # use fitted hrf params
    hrf_params = np.ones((3, pRF_estimates['hrf_derivative'].shape[0]))
    hrf_params[1] = pRF_estimates['hrf_derivative']
    hrf_params[2] = pRF_estimates['hrf_dispersion']
    
    hrf_oversampled = mri_utils.create_hrf(hrf_params = hrf_params, TR = TR, osf = osf) 
else:
    hrf_oversampled = np.tile(mri_utils.create_hrf(TR = TR, osf = osf), (pRF_estimates['r2'].shape[0],1))
    hrf_params = np.full((3, pRF_estimates['r2'].shape[0]), None)

# make visual DM for each GLM regressor, and obtain prediction using pRF model
for reg in all_regressors['reg_name'].values:
    
    # filename for regressor dm
    DM_reg_filename = op.join(output_dir,'DM_regressor-{reg}.npy'.format(reg=reg))
    # filename for regressor prediction
    reg_filename = DM_reg_filename.replace('DM','prediction')
    
    if not op.exists(reg_filename): # check if file already exists
        # make array with spatial position of bar of interest 
        DM_cond = mri_utils.get_FA_bar_stim(DM_reg_filename, 
                            params, bar_pos, trial_info, attend_cond = all_regressors[all_regressors['reg_name']==reg].to_dict('r')[0], 
                            save_imgs = False, downsample = 0.1, crop = False, shift_TRs = False, #params['feature']['crop'],
                            oversampling_time = osf, stim_dur_seconds = params['feature']['bars_phase_dur'], 
                            crop_TR = params['feature']['crop_TR'], overwrite=True)
        
        # make stimulus object, which takes an input design matrix and sets up its real-world dimensions
        prf_stim = PRFStimulus2D(screen_size_cm = params['monitor']['height'],
                                 screen_distance_cm = params['monitor']['distance'],
                                 design_matrix = DM_cond,
                                 TR = TR)#FA_sampling_rate)

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
            # set our oversampled hrf
            css_model.hrf = hrf_oversampled

            model_fit = Parallel(n_jobs=16)(delayed(css_model.return_prediction)(xx[vert], 
                                                                                 yy[vert],
                                                                                 size[vert],
                                                                                 beta[vert],
                                                                                 baseline[vert],
                                                                                 ns[vert],
                                                                                 hrf_1 = hrf_params[1][vert],
                                                                                 hrf_2 = hrf_params[2][vert],
                                                                                 osf = osf) for _,vert in enumerate(tqdm(mask_ind)))

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
            # set our oversampled hrf
            gauss_model.hrf = hrf_oversampled

            model_fit = Parallel(n_jobs=16)(delayed(gauss_model.return_prediction)(xx[vert], 
                                                                                   yy[vert],
                                                                                   size[vert],
                                                                                   beta[vert],
                                                                                   baseline[vert],
                                                                                   hrf_1 = hrf_params[1][vert],
                                                                                   hrf_2 = hrf_params[2][vert],
                                                                                   osf = osf) for _,vert in enumerate(tqdm(mask_ind)))

        # squeeze out single dimension that parallel creates
        model_fit = np.squeeze(model_fit)

        # original scale of data in seconds
        original_scale = np.arange(0, model_fit.shape[-1]/osf, 1/osf)

        # cubic interpolation of predictor
        interp = interpolate.interp1d(original_scale, 
                                    model_fit, 
                                    kind = "cubic", axis=-1)
        desired_scale = np.arange(0, model_fit.shape[-1]/osf, TR) # we want the predictor to be sampled in TR

        resampled_predictor = interp(desired_scale)

        ## need to crop out of function, to avoid rounding errors
        if params['feature']['crop']: # if cropping runs 
            resampled_predictor = resampled_predictor[...,params['feature']['crop_TR']::] 
            
        ## also shifting TR to the left (quick fix)
        # to account for first trigger that was "dummy" - in future change experiment settings to skip 1st TR
        new_resampled_predictor = resampled_predictor.copy()
        new_resampled_predictor[...,:-1] = resampled_predictor[...,1:]
        resampled_predictor = new_resampled_predictor.copy()

        # save surface prediction array 
        prediction = np.zeros(data.shape); prediction[:]=np.nan # not fitted voxels will be nan
        for i,ind in enumerate(mask_ind):
            prediction[ind] = resampled_predictor[i]

        # save prediction in folder, for later use/plotting
        np.save(reg_filename,prediction)

    else:
        # load prediction
        prediction = np.load(reg_filename)
        print('file already exists, loading %s'%reg_filename)
    
    ## append predictions in array, to use for FA GLM DM
    all_reg_predictions.append(prediction[np.newaxis,...])

all_reg_predictions = np.vstack(all_reg_predictions)

### make cue regressors

# array with cue regressors
cue_regs = np.zeros((params['feature']['mini_blocks'],all_reg_predictions.shape[1],all_reg_predictions.shape[-1]))
# array with cue regressors - UPSAMPLED
cue_regs_upsampled = np.zeros((params['feature']['mini_blocks'], all_reg_predictions.shape[1], 
                               len(trial_info['trial_num'].values)*osf))

for blk in range(params['feature']['mini_blocks']): # for each miniblock
    
    for trl in trial_info.loc[trial_info['trial_type']=='cue_%i'%blk]['trial_num'].values:
        
        ## fill upsampled array
        cue_regs_upsampled[blk,:,trl*osf:trl*osf+osf] = 1
    
    # convolve with spm hrf, similar to what prfpy is using
    cue_regs_upsampled[blk] = signal.fftconvolve(cue_regs_upsampled[blk], hrf_oversampled, 
                                            mode='full', axes=(-1))[..., :cue_regs_upsampled.shape[-1]]

    # original scale of data in seconds
    original_scale = np.arange(0, cue_regs_upsampled[blk].shape[-1]/osf, 1/osf)

    # cubic interpolation of predictor
    interp = interpolate.interp1d(original_scale, 
                                cue_regs_upsampled[blk], 
                                kind = "cubic", axis=-1)
    desired_scale = np.arange(0, cue_regs_upsampled[blk].shape[-1]/osf, TR) # we want the predictor to be sampled in TR

    cue_regs_RESAMPLED = interp(desired_scale)

    ## filter it, like we do to the data
    cue_regs_RESAMPLED =  mri_utils.dc_data(cue_regs_RESAMPLED,
                             first_modes_to_remove = params['mri']['filtering']['first_modes_to_remove'])

    ## need to crop out of function, to avoid rounding errors
    if params['feature']['crop']: # if cropping runs 
        cue_regs[blk] = cue_regs_RESAMPLED[...,params['feature']['crop_TR']::] 
        
    ## also shifting TR to the left (quick fix)
    # to account for first trigger that was "dummy" - in future change experiment settings to skip 1st TR
    new_cue_regs = cue_regs[blk].copy()
    new_cue_regs[...,:-1] = cue_regs[blk][...,1:]
    cue_regs[blk] = new_cue_regs.copy()
    
    ## also update regressors info to know name and order of regressors
    # basically including cues
    all_regressors = all_regressors.append(pd.DataFrame({'reg_name': 'cue_mblk-{blk}_run-{run}'.format(blk=blk,
                                                                                  run=run),
                                                             'color': np.nan,
                                                             'orientation': np.nan,
                                                             'condition_name': trial_info.loc[trial_info['trial_type']=='cue_%i'%blk]['attend_condition'].values[0],
                                                             'miniblock': blk,
                                                             'run': int(run)
                                                            }, index=[0]),ignore_index=True)
    
# save regressors dataframe, for later access
all_regressors.to_csv(op.join(output_dir, 'all_regressors_info.csv'),index=False)

## Make actual DM to be used in GLM fit (+ intercept)

# number of regressors
num_regs = all_reg_predictions.shape[0] + cue_regs.shape[0] + 1 # conditions, cues, and intercept

DM_FA = np.zeros((all_reg_predictions.shape[1], all_reg_predictions.shape[-1], num_regs)) # shape of DM is (vox,time,reg)

# iterate over vertex/voxel
for i in range(all_reg_predictions.shape[1]):

    DM_FA[i,:,0] = np.repeat(1,all_reg_predictions.shape[-1]) # intercept will be first always
    
    for w in range(all_reg_predictions.shape[0]): # add regressor (which will be the model from pRF estimates)
        DM_FA[i,:,w+1] = all_reg_predictions[w,i,:] 
        
    for k in range(cue_regs.shape[0]): # add cues
        DM_FA[i,:,w+2+k] = cue_regs[k,i,:]


# save DM, for later checking
np.save(op.join(output_dir,'DM_FA_run-{run}.npy'.format(run=run)),DM_FA)
print('saving %s'%op.join(output_dir,'DM_FA_run-{run}.npy'.format(run=run)))

## Actually fit GLM
FA_GLM_estimates_filename = op.join(output_dir, op.split(proc_files[0])[-1].replace('.npy','_estimates.npz'))

if not op.isfile(FA_GLM_estimates_filename): # if doesn't exist already
    
    print('fitting GLM to %d vertices'%data.shape[0])
    glm_outcome = Parallel(n_jobs=16)(delayed(mri_utils.fit_glm)(data[vert], DM_FA[vert]) for vert in tqdm(range(data.shape[0])))

    np.savez(FA_GLM_estimates_filename,
                prediction = np.array([glm_outcome[i][0] for i in range(data.shape[0])]),
                betas = np.array([glm_outcome[i][1] for i in range(data.shape[0])]),
                r2 = np.array([glm_outcome[i][2] for i in range(data.shape[0])]),
                mse = np.array([glm_outcome[i][3] for i in range(data.shape[0])])
            )
else:
    print('already exists %s'%FA_GLM_estimates_filename)
