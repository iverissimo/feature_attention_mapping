
## visualize pRF and FA estimates
# with interactive figure 
# that shows timecourse on click

import numpy as np
import os, sys
import os.path as op
import yaml
from pathlib import Path
import glob

import pandas as pd

import cortex
import matplotlib.pyplot as plt

from FAM_utils import mri as mri_utils
from FAM_utils import plotting as plot_utils

# inserting path to fitting, to get feature model objects
# should reorganize folder in future, to avoid this
sys.path.insert(1, op.join(str(Path(os.getcwd()).parents[0]), 'fitting'))
from feature_model import FA_GainModel
from prfpy.stimulus import PRFStimulus2D

# load settings from yaml
with open(op.join(str(Path(os.getcwd()).parents[1]),'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)


# define participant number and run 
if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')
elif len(sys.argv) < 3:
    raise NameError('Please add task to be visualized (ex: pRF, FA, both) '
                    'as 2nd argument in the command line!')
elif len(sys.argv) < 4:
    raise NameError('Please add run to be fitted (ex: 1) '
                    'as 3rd argument in the command line!')
else:
    # fill subject number and chunk number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(3)
    task2viz = str(sys.argv[2])
    run = str(sys.argv[3])

# set font type for plots globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# get pycortex sub
pysub = params['plotting']['pycortex_sub']+'_sub-{sj}'.format(sj=sj) # because subject specific borders 

# set threshold for plotting
rsq_threshold = params['plotting']['rsq_threshold']

# some settings
base_dir = params['general']['current_dir'] # which machine we run the data
acq = params['mri']['acq'] # if using standard files or nordic files
space = params['mri']['space'] # subject space
total_chunks = params['mri']['fitting']['pRF']['total_chunks'][space] # number of chunks that data was split in

TR = params['mri']['TR']

# type of model to fit
model_type = params['mri']['fitting']['pRF']['fit_model']

# if we fitted hrf
fit_hrf = params['mri']['fitting']['pRF']['fit_hrf']

# set estimate key names
estimate_keys = params['mri']['fitting']['pRF']['estimate_keys'][model_type]
if fit_hrf:
    estimate_keys = estimate_keys+['hrf_derivative','hrf_dispersion']

# if we're shifting TRs, to account for dummy scans or slicetime correction
shift_TRs = params['mri']['fitting']['pRF']['shift_DM'] 
shift_TR_num =  params['mri']['fitting']['pRF']['shift_DM_TRs']
if isinstance(shift_TR_num, int):
    osf = 1
    resample_pred = False
else:
    print('shift implies upsampling DM')
    osf = 10
    resample_pred = True
    
# if we are keeping baseline fixed at 0
fix_bold_baseline = params['mri']['fitting']['pRF']['fix_bold_baseline']
# if we want to do bold baseline correction
correct_baseline = params['mri']['fitting']['pRF']['correct_baseline'] 

## define file extension that we want to use, 
# should include processing key words

file_ext = {'prf': '', 'feature': ''}

for key in file_ext.keys():

    # if cropped first
    if params[key]['crop']:
        file_ext[key] += '_{name}'.format(name = 'cropped')
    # type of filtering/denoising
    if params[key]['regress_confounds']:
        file_ext[key] += '_{name}'.format(name = 'confound')
    else:
        file_ext[key] += '_{name}'.format(name = params['mri']['filtering']['type'])
    # type of standardization 
    file_ext[key] += '_{name}'.format(name = params[key]['standardize'])
    # don't forget its a numpy array
    file_ext[key] += '.npy'
    
# set paths
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
postfmriprep_dir = op.join(derivatives_dir,'post_fmriprep','sub-{sj}'.format(sj=sj),space,'processed')
source_dir = glob.glob(op.join(params['mri']['paths'][base_dir]['root'], 'sourcedata', 
                               'sub-{sj}'.format(sj=sj), 'ses-*', 'func'))[0] 

## get behavioral files 
# to mask DM according to behav response

behav_files = [op.join(source_dir, h) for h in os.listdir(source_dir) if 'task-pRF' in h and
                 h.endswith('events.tsv')]
# behav boolean mask
DM_mask_beh = mri_utils.get_beh_mask(behav_files,params)

## get onset of events from behavioral tsv files
event_onsets = mri_utils.get_event_onsets(behav_files, crop = params['prf']['crop'] , crop_TR = params['prf']['crop_TR'])


## get relevant estimates that we want to plot
# if both tasks or just feature
if task2viz in ['both', 'FA', 'feature']:
    
    print('not implemented yet') 
    
    ######## REVISE ALL THIS AFTER RESTRUCTURING FA #############
    
    ########### Load FA data and fits ##########
    ## load Feature model
    fa_model = FA_GainModel(params)
    
    ## load results from iterative fit
    FA_fit_dir = op.join(derivatives_dir,'FA_gain','sub-{sj}'.format(sj=sj), space, 
                          fa_model.prf_model_type, 'run-{run}'.format(run=run))
    if fit_hrf:
        FA_fit_dir = op.join(FA_fit_dir,'with_hrf')

    it_FA_results = pd.read_csv(op.join(FA_fit_dir,'run-{run}_iterative_params.csv'.format(run=run)))
    
    ## load model predictions
    FA_model_predictions = np.load(op.join(FA_fit_dir,'prediction_FA_iterative_gain_run-{run}.npy'.format(run=run)))

    ## list with absolute file name to be fitted
    proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-FA' in h and \
                     'acq-{acq}'.format(acq=acq) in h and \
                  'run-{run}'.format(run=run) in h and h.endswith(file_ext['feature'])]

    ## load functional data
    FA_data = np.load(proc_files[0], allow_pickle = True) # will be (vertex, TR)

    
    ######## Load pRF data and fits ###########
    ## load pRF fit 
    prf_fits_dir = op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), space, 
                                     'iterative_{model}'.format(model=model_type),
                                       'run-{run}'.format(run=fa_model.prf_run_type))
    if fit_hrf:
        prf_fits_dir = op.join(prf_fits_dir,'with_hrf')
    
    # load them into numpy dict
    pRF_estimates = fa_model.get_pRF_estimates(prf_fits_dir, total_chunks)
    
    # mask estimates
    print('masking estimates')
    pRF_estimates = fa_model.mask_pRF_estimates(prf_fits_dir.split(space)[0], DM_mask_beh, 
                                                estimate_keys = estimate_keys)
    fa_model.pRF_estimates = pRF_estimates
    
    ## list with absolute file name to be fitted
    proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-pRF' in h and \
                     'acq-{acq}'.format(acq = acq) in h and \
                   'run-{run}'.format(run = fa_model.prf_run_type) in h and h.endswith(file_ext['prf'])]

    ## load functional data
    pRF_data = np.load(proc_files[0], allow_pickle = True) # will be (vertex, TR)
    
    # set pRF DM
    visual_dm = fa_model.prf_dm

# if just pRF
elif task2viz in ['prf', 'pRF']:
    
    ## load pRF fit 
    prf_fits_dir = op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj = sj), space, 
                                     'iterative_{model}'.format(model = model_type),
                                       'run-{run}'.format(run = run))
    if fit_hrf:
        prf_fits_dir = op.join(prf_fits_dir,'with_hrf')
    
    # load them into numpy dict
    pRF_estimates = mri_utils.load_pRF_estimates(prf_fits_dir, params, 
                                                 total_chunks = total_chunks, model_type = model_type)
    
    # define design matrix 
    visual_dm = mri_utils.make_pRF_DM(op.join(derivatives_dir,'pRF_fit', 'sub-{sj}'.format(sj=sj), 'DMprf.npy'), params, 
                                save_imgs = False, res_scaling = 0.1, TR = params['mri']['TR'],
                                crop = params['prf']['crop'] , crop_TR = params['prf']['crop_TR'], 
                                shift_TRs = shift_TRs, shift_TR_num = shift_TR_num, oversampling_time = osf,
                                overwrite = False, mask = [], event_onsets = [])


    # make stimulus object, which takes an input design matrix and sets up its real-world dimensions
    prf_stim = PRFStimulus2D(screen_size_cm = params['monitor']['height'],
                            screen_distance_cm = params['monitor']['distance'],
                            design_matrix = visual_dm,
                            TR = TR)
    
    # get the ecc limits (in dva)
    x_ecc_lim = [- prf_stim.screen_size_degrees/2, prf_stim.screen_size_degrees/2]
    y_ecc_lim = [- prf_stim.screen_size_degrees/2, prf_stim.screen_size_degrees/2] 

    # mask estimates
    print('masking estimates')
    pRF_estimates = mri_utils.mask_estimates(pRF_estimates, 
                                          estimate_keys = estimate_keys,
                                          x_ecc_lim = x_ecc_lim, y_ecc_lim = y_ecc_lim)
    
    ## list with absolute file name to be fitted
    proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-pRF' in h and \
                     'acq-{acq}'.format(acq = acq) in h and \
                   'run-{run}'.format(run = run) in h and h.endswith(file_ext['prf'])]

    ## load functional data
    pRF_data = np.load(proc_files[0], allow_pickle = True) # will be (vertex, TR)
    
    
    # set FA data and predictions as empty
    FA_data = []
    FA_model_predictions = []


# if we want to keep baseline fix, we need to correct it!
if correct_baseline:
    pRF_data = mri_utils.baseline_correction(pRF_data, params, 
                            num_baseline_TRs = params['mri']['fitting']['pRF']['num_baseline_TRs'], 
                            baseline_interval = 'empty_long', 
                            avg_type = 'median', crop = params['prf']['crop'], 
                            crop_TR = params['prf']['crop_TR'])

##
## set cortex flatmaps to show

## make alpha level based on pRF rsquared 
alpha_level = mri_utils.normalize(np.clip(pRF_estimates['r2'], 0, .8)) #mask, 0, .8)) # normalize 
# number of bins for custom colormaps
n_bins_colors = 256
images = {}

## pRF rsq
# mask out 0 to nans, for prettier plot
prf_rsq4plot = np.zeros(pRF_estimates['r2'].shape); prf_rsq4plot[:] = np.nan
prf_rsq4plot[pRF_estimates['r2']>0] = pRF_estimates['r2'][pRF_estimates['r2']>0] 
images['pRF_rsq'] = cortex.Vertex(prf_rsq4plot, 
                                subject = pysub, 
                                vmin = 0, vmax = 1,
                                cmap = 'Reds')

## calculate pa + ecc + size
complex_location = pRF_estimates['x'] + pRF_estimates['y'] * 1j # calculate eccentricity values

polar_angle = np.angle(complex_location)
eccentricity = np.abs(complex_location)

if model_type in ['dn', 'dog']:
    size_fwhmax, fwatmin = mri_utils.fwhmax_fwatmin(model_type, pRF_estimates)
else: 
    size_fwhmax = mri_utils.fwhmax_fwatmin(model_type, pRF_estimates)


## pRF Eccentricity
ecc4plot = np.zeros(pRF_estimates['r2'].shape); ecc4plot[:] = np.nan
ecc4plot[pRF_estimates['r2']>0] = eccentricity[pRF_estimates['r2']>0]

ecc_cmap = mri_utils.make_colormap(colormap = ['#dd3933','#f3eb53','#7cb956','#82cbdb','#3d549f'],
                               bins = n_bins_colors, cmap_name = 'ECC_mackey_costum', 
                                   discrete = False, add_alpha = False, return_cmap = True)

images['ecc'] = mri_utils.make_raw_vertex_image(ecc4plot, 
                                               cmap = ecc_cmap, vmin = 0, vmax = 6, 
                                              data2 = alpha_level, vmin2 = 0, vmax2 = 1, 
                                               subject = pysub, data2D = True)
    
## pRF Size
size4plot = np.zeros(pRF_estimates['r2'].shape); size4plot[:] = np.nan
size4plot[pRF_estimates['r2']>0] = size_fwhmax[pRF_estimates['r2']>0]

images['size_fwhmax'] = mri_utils.make_raw_vertex_image(size4plot, 
                                               cmap = 'hot', vmin = 0, vmax = 7, 
                                              data2 = alpha_level, vmin2 = 0, vmax2 = 1, 
                                               subject = pysub, data2D = True)

## pRF Polar Angle
pa4plot = np.zeros(pRF_estimates['r2'].shape); pa4plot[:] = np.nan
pa4plot[pRF_estimates['r2']>0] = ((polar_angle + np.pi) / (np.pi * 2.0))[pRF_estimates['r2']>0]

# get matplotlib color map from segmented colors
PA_cmap = mri_utils.make_colormap(colormap = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb',
                              '#3d549f','#655099','#ad5a9b','#dd3933'], bins = n_bins_colors, 
                                    cmap_name = 'PA_mackey_costum',
                              discrete = False, add_alpha = False, return_cmap = True)

images['PA'] = mri_utils.make_raw_vertex_image(pa4plot, 
                                               cmap = PA_cmap, vmin = 0, vmax = 1, 
                                              data2 = alpha_level, vmin2 = 0, vmax2 = 1, 
                                               subject = pysub, data2D = True)

## pRF Exponent 
if model_type == 'css':
    images['ns'] = cortex.Vertex2D(pRF_estimates['ns'], alpha_level,
                            pysub,
                            vmin = 0, vmax = .5,
                            vmin2 = 0, vmax2 = 1,
                            cmap='plasma_alpha')

## initialize interactive figure

plot_obj = plot_utils.visualize_on_click(params, pRF_estimates, 
                                    prf_dm = visual_dm, 
                                    max_ecc_ext = np.max(x_ecc_lim),
                                    pRF_data = pRF_data,
                                    FA_data = FA_data, FA_model = FA_model_predictions)

# set flatmaps in object class
plot_obj.images = images

plot_obj.set_figure(task2viz = task2viz)

cortex.quickshow(images['pRF_rsq'], fig = plot_obj.flatmap_ax,
                 with_rois = False, with_curvature = True, with_colorbar=False)

plot_obj.full_fig.canvas.mpl_connect('button_press_event', plot_obj.onclick)
plot_obj.full_fig.canvas.mpl_connect('key_press_event', plot_obj.onkey)

plt.show()