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
use_nuisance_reg = True # if we are using nuisance regressor (to account for start block arousal)

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

## if doing leave-one-out, we are fitting all runs except left out one
if 'loo' in run:
    runs2fit = np.array([str(r+1) for r in np.arange(4) if str(r+1) != run[-1:]])
else:
    runs2fit = np.array([run]) 

## list with absolute file name to be fitted
proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) for r in runs2fit 
                       if 'task-FA' in h and 'acq-{acq}'.format(acq = acq) in h and 
                       'run-{run}'.format(run = r) in h and h.endswith(file_ext)]

## load functional data
if len(proc_files)>1:
    print(' fitting several files - %i files found for run %s'%(len(proc_files),run))

[print('Fitting %s'%p) for p in proc_files]

data = np.stack((np.load(file, allow_pickle=True) for file in proc_files))# will be (runs, vertex, TR)

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
bar_pos_files = [op.join(source_dir, h) for h in os.listdir(source_dir) for r in runs2fit if 'task-FA' in h and
                'run-{run}'.format(run = r) in h and h.endswith('_bar_positions.pkl')]
[print('getting bar positions from %s'%bpos) for bpos in bar_pos_files]

# load bar positions for runs
bar_pos = []
[bar_pos.append(pd.read_pickle(file)) for file in bar_pos_files]

# get absolute path to csv with general infos for each run
trial_info_files = [op.join(source_dir, h) for h in os.listdir(source_dir) for r in runs2fit if 'task-FA' in h and
                'run-{run}'.format(run = r) in h and h.endswith('_trial_info.csv')]

# load trial info dataframe for runs
trial_info = []
[trial_info.append(pd.read_csv(file)) for file in trial_info_files]

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
else:
    hrf_params[2] = 0

if use_nuisance_reg:
    fa_model.use_nuisance_reg = use_nuisance_reg
    nuisance_regressors = np.load(op.join(derivatives_dir, 'block_nuisance', 'sub-{sj}'.format(sj=sj), 
                                            space,'nuisance_regressor.npy'))

##set all necessary parameters used for 
# gain fit - also setting which ones we fit or not
pars = Parameters()

# add pRF parameters - will not vary
pars.add('pRF_x', value = 0, vary = False)
pars.add('pRF_y', value = 0, vary = False)
pars.add('pRF_size', value = 0, vary = False)
pars.add('pRF_beta', value = 0, vary = False)
pars.add('pRF_baseline', value = 0, vary = False)
pars.add('pRF_n', value = 1, vary = False)

# add gain params for each bar - will vary
pars.add('gain_ACAO', value = 1, vary = False) # attended condition doesnt vary - pRF task was attended to bar
pars.add('gain_ACUO', value = 0, vary = True, min = 0.1, max = 1, brute_step = .2)
pars.add('gain_UCAO', value = 0, vary = True, min = 0.1, max = 1, brute_step = .2)
pars.add('gain_UCUO', value = 0, vary = True, min = 0.1, max = 1, brute_step = .2)

# add params that will be filled by GLM - will not vary
pars.add('beta_cue_0', value = 0, vary = False)
pars.add('beta_cue_1', value = 0, vary = False)
pars.add('beta_cue_2', value = 0, vary = False)
pars.add('beta_cue_3', value = 0, vary = False)
pars.add('beta_bar_stim', value = 0, vary = False)
pars.add('beta_nuisance', value = 0, vary = False)
pars.add('intercept', value = 1, vary = False)
pars.add('rsq', value = 0, vary = False)

fa_pars = []
for i,r in enumerate(runs2fit):
    fa_pars.append(pars)

# some optimizer params
xtol = 1e-7
ftol = 1e-6
solver_type = 'trust-constr' #'lbfgsb' ##'trust-constr'
n_jobs = 16 # for paralell

## if already in dir, load
# otherwise fit
grid_filename = op.join(output_dir,'run-%s_grid_params.csv'%run)
                        
if op.exists(grid_filename):
    print('loading %s'%grid_filename)
    grid_results = [pd.read_csv(grid_filename)]
else:
    if len(runs2fit)>1 and op.exists(op.join(output_dir,'run-%s_grid_params.csv'%runs2fit[0])):
        grid_results = [pd.read_csv(op.join(output_dir,'run-%s_grid_params.csv'%r)) for r in runs2fit]
    else:
        print('grid fitting params')
        grid_results = fa_model.grid_fit(data, fa_pars, 
                                     hrf_params = hrf_params, 
                                     mask_ind = mask_ind,
                                     nuisance_regressors = nuisance_regressors, workers = 1, n_jobs = n_jobs)
        ## save fitted params Dataframe
        for i,r in enumerate(runs2fit):
            grid_results[i].to_csv(op.join(output_dir,'run-%s_grid_params.csv'%r), index = False)


## same logic for iterative fit
# 
it_filename = op.join(output_dir,'run-%s_iterative_params.csv'%run)
                        
if op.exists(it_filename):
    print('loading %s'%it_filename)
    it_results = [pd.read_csv(it_filename)]
else:
    if len(runs2fit)>1 and op.exists(op.join(output_dir,'run-%s_iterative_params.csv'%runs2fit[0])):
        it_results = [pd.read_csv(op.join(output_dir,'run-%s_iterative_params.csv'%r)) for r in runs2fit]
    else:
        print('iterative fitting params')
        it_results = fa_model.iterative_fit(data, fa_pars, 
                                     hrf_params = hrf_params, 
                                     mask_ind = mask_ind,
                                     nuisance_regressors = nuisance_regressors,
                                     xtol = xtol, ftol = ftol, method = solver_type, n_jobs = n_jobs,
                                     prev_fit_params = np.stack((np.array(grid_results[r].to_dict('r')) for r in range(len(runs2fit))), axis = 0)) # using grid fit outcome as starting point

        ## save fitted params Dataframe
        for i,r in enumerate(runs2fit):
            it_results[i].to_csv(op.join(output_dir,'run-%s_iterative_params.csv'%r), index = False)


# set cue regressors, in case we just loaded estimates
if not hasattr(fa_model, 'cue_regressors'):
    fa_model.bar_stim_regressors_keys = np.array(['bar_stim'])
    nr_cue_regs = 4
    fa_model.cue_regressors = np.stack((mri_utils.get_cue_regressor(fa_model.trial_info[0], 
                                                    hrf_params = hrf_params, cues = [i],
                                                    TR = fa_model.TR, oversampling_time = fa_model.osf, 
                                                    baseline = fa_model.pRF_estimates['baseline'],
                                                    crop_unit = 'sec', crop = fa_model.fa_crop, 
                                                    crop_TR = fa_model.fa_crop_TRs, 
                                                    shift_TRs = fa_model.fa_shift_TRs, 
                                                    shift_TR_num = fa_model.fa_shift_TR_num) for i in range(nr_cue_regs)), axis = 0)

## save DM given fitted params for all relevant vertices
# others will be nan
print('saving design matrix')
all_dms = [] # append for all runs, to use to model predictions

for i, r in enumerate(runs2fit):
    dm = np.array(Parallel(n_jobs = 16, backend ='threading')(delayed(fa_model.make_FA_DM)(it_results[i].to_dict('r')[ind],
                                                                                            hrf_params = hrf_params[..., vert], 
                                                                                            cue_regressors = {'cue_0': fa_model.cue_regressors[0][vert], 
                                                                                                            'cue_1': fa_model.cue_regressors[1][vert], 
                                                                                                            'cue_2': fa_model.cue_regressors[2][vert], 
                                                                                                            'cue_3': fa_model.cue_regressors[3][vert]},
                                                                                            nuisance_regressors = nuisance_regressors[vert],
                                                                                                visual_dm = fa_model.FA_visual_DM[i],
                                                                                            weight_stim = True)
                                                                                for ind, vert in enumerate(tqdm(it_results[i]['vertex'].values))))
    
    # get all regressor key names to save out as well
    all_regressor_keys = dm[0][1]

    # reshape dm 
    dm_reshape = np.stack((dm[ind][0] for ind in range(len(mask_ind))), axis = 0)

    dm_surf = np.zeros((data.shape[1], data.shape[2], len(all_regressor_keys))); dm_surf[:] = np.nan
    dm_surf[mask_ind, ...] = dm_reshape
    
    fa_dm_filename = op.join(output_dir,'DM_FA_iterative_gain_run-{r}.npz'.format(r = r))
    np.savez(fa_dm_filename,
            dm = dm_surf,
            reg_names = all_regressor_keys)

    all_dms.append(dm_surf)

## save also model predictions for whole surface
print('saving model timecourses')
for i, r in enumerate(runs2fit):
    ## save also model predictions for whole surface
    print('saving model timecourses')

    model_tc_surf = np.zeros((data.shape[1], data.shape[2])); model_tc_surf[:] = np.nan

    model_tc = np.array(Parallel(n_jobs = 16, backend ='threading')(delayed(mri_utils.get_fa_prediction_tc)(all_dms[i][vert],
                                                                                            np.concatenate(([it_results[i].iloc[ind]['intercept']],
                                                                                                                [it_results[i].iloc[ind]['beta_%s'%val] for val in all_regressor_keys if val != 'intercept']
                                                                                                            )),
                                                                                                            timecourse = data[i][vert],
                                                                                                            r2 = it_results[i].iloc[ind]['rsq'], viz_model = False)
                                                                                for ind, vert in enumerate(tqdm(it_results[i]['vertex'].values))))

    model_tc_surf[mask_ind, ...] = model_tc

    model_tc_filename = op.join(output_dir,'prediction_FA_iterative_gain_run-{r}.npy'.format(r = r))
    np.save(model_tc_filename, model_tc_surf)