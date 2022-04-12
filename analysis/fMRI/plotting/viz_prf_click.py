
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

## some settings
base_dir = params['general']['current_dir'] # which machine we run the data
acq = params['mri']['acq'] # if using standard files or nordic files
space = params['mri']['space'] # subject space
TR = params['mri']['TR']
# type of model to fit

model_type = params['mri']['fitting']['pRF']['fit_model']
fit_hrf = params['mri']['fitting']['pRF']['fit_hrf']

# set estimate key names
estimate_keys = params['mri']['fitting']['pRF']['estimate_keys'][model_type]

# if we are keeping baseline fixed at 0
fix_bold_baseline = params['mri']['fitting']['pRF']['fix_bold_baseline']
# if we want to do bold baseline correction
correct_baseline = params['mri']['fitting']['pRF']['correct_baseline'] 

# get pycortex sub
pysub = params['plotting']['pycortex_sub']+'_sub-{sj}'.format(sj=sj) # because subject specific borders 

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

########### Load FA data and fits ##########

FA_fit_dir = op.join(derivatives_dir,'FA_gain','sub-{sj}'.format(sj=sj), space, 
                          fa_model.prf_model_type, 'run-{run}'.format(run=run))

# load results from iterative fit
it_FA_results = pd.read_csv(op.join(FA_fit_dir,'run-{run}_iterative_params.csv'.format(run=run)))
   
# load model predictions
FA_model_predictions = np.load(op.join(FA_fit_dir,'prediction_FA_iterative_gain_run-{run}.npy'.format(run=run)))
    
## list with absolute file name to be fitted
proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-FA' in h and
                 'acq-{acq}'.format(acq=acq) in h and 'run-{run}'.format(run=run) in h and h.endswith(file_ext)]

## load functional data
FA_data = np.load(proc_files[0], allow_pickle = True) # will be (vertex, TR)

######## Load pRF data and fits ###########

prf_fits_pth =  op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), space, 
                        'iterative_{model}'.format(model = fa_model.prf_model_type),
                        'run-{run}'.format(run=fa_model.prf_run_type))

# load them into numpy dict
pRF_estimates = fa_model.get_pRF_estimates(prf_fits_pth, 
                                           params['mri']['fitting']['pRF']['total_chunks'][space])

# if we want to mask pRFs, given screen limits and behavior responses
if mask_prf: 

    print('masking pRF estimates')

    ## make pRF DM mask, according to sub responses
    # list of behavior files
    behav_files = [op.join(source_dir, h) for h in os.listdir(source_dir) if 'task-pRF' in h and
                     h.endswith('events.tsv')]
    # behav boolean mask
    DM_mask_beh = mri_utils.get_beh_mask(behav_files, params)

    # include hrf extra estimates
    if fa_model.fit_hrf:
        print('fitted hrf, using those estimates')
        estimate_keys = estimate_keys+['hrf_derivative','hrf_dispersion']

    # mask estimates
    pRF_estimates = fa_model.mask_pRF_estimates(prf_fits_pth.split(space)[0], DM_mask_beh, 
                                                estimate_keys = estimate_keys)
    fa_model.pRF_estimates = pRF_estimates
    
    
## list with absolute file name to be fitted
proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-pRF' in h and
                 'acq-{acq}'.format(acq=acq) in h and fa_model.prf_run_type]

## load functional data
pRF_data = np.load(proc_files[0], allow_pickle = True) # will be (vertex, TR)

# if we want to keep baseline fix, we need to correct it!
if correct_baseline:
    pRF_data = mri_utils.baseline_correction(pRF_data, params, num_baseline_TRs = 7, baseline_interval = 'empty_long', 
                            avg_type = 'median', crop = params['prf']['crop'], 
                            crop_TR = params['prf']['crop_TR'])

##
## set cortex flatmaps to show
# rsq_flat = cortex.Vertex(fa_model.pRF_estimates['r2'], 
#                   subject = pysub, 
#                   vmin = 0, vmax = .8,
#                     cmap = 'Reds')

## make alpha level based on pRF rsquared 
alpha_level = mri_utils.normalize(np.clip(fa_model.pRF_estimates['r2'], 0, .8))#mask, 0, .8)) # normalize 
# number of bins for custom colormaps
n_bins_colors = 256
images = {}

## pRF rsq
# mask out 0 to nans, for prettier plot
prf_rsq4plot = np.zeros(fa_model.pRF_estimates['r2'].shape); prf_rsq4plot[:] = np.nan
prf_rsq4plot[fa_model.pRF_estimates['r2']>0] = fa_model.pRF_estimates['r2'][fa_model.pRF_estimates['r2']>0] 
images['pRF_rsq'] = cortex.Vertex(prf_rsq4plot, 
                                subject = pysub, 
                                vmin = 0, vmax = .8,
                                cmap = 'Reds')

## FA rsq
FA_rsq4plot = np.zeros(fa_model.pRF_estimates['r2'].shape); FA_rsq4plot[:] = np.nan
FA_rsq4plot[it_FA_results['vertex'].values] = it_FA_results['rsq'].values
images['FA_rsq'] = cortex.Vertex(FA_rsq4plot, 
                                subject = pysub, 
                                vmin = 0, vmax = .3,
                                cmap = 'Reds')

## calculate pa + ecc + size
complex_location = fa_model.pRF_estimates['x'] + fa_model.pRF_estimates['y'] * 1j # calculate eccentricity values
polar_angle = np.angle(complex_location)
eccentricity = np.abs(complex_location)
size = fa_model.pRF_estimates['size']
# non-linearity interacts with the Gaussian standard deviation to make an effective pRF size of σ/sqr(n)
if fa_model.prf_model_type == 'css': 
    size = fa_model.pRF_estimates['size']/np.sqrt(fa_model.pRF_estimates['ns']) 

## pRF Eccentricity
ecc4plot = np.zeros(fa_model.pRF_estimates['r2'].shape); ecc4plot[:] = np.nan
ecc4plot[fa_model.pRF_estimates['r2']>0] = eccentricity[fa_model.pRF_estimates['r2']>0]

ecc_cmap = mri_utils.make_colormap(colormap = ['#dd3933','#f3eb53','#7cb956','#82cbdb','#3d549f'],
                               bins = n_bins_colors, cmap_name = 'ECC_mackey_costum', 
                                   discrete = False, add_alpha = False, return_cmap = True)

images['ecc'] = mri_utils.make_raw_vertex_image(ecc4plot, 
                                               cmap = ecc_cmap, vmin = 0, vmax = 6, 
                                              data2 = alpha_level, vmin2 = 0, vmax2 = 1, 
                                               subject = pysub, data2D = True)

## pRF Size
size4plot = np.zeros(fa_model.pRF_estimates['r2'].shape); size4plot[:] = np.nan
size4plot[fa_model.pRF_estimates['r2']>0] = size[fa_model.pRF_estimates['r2']>0]

images['size'] = mri_utils.make_raw_vertex_image(size4plot, 
                                               cmap = 'hot', vmin = 0, vmax = 7, 
                                              data2 = alpha_level, vmin2 = 0, vmax2 = 1, 
                                               subject = pysub, data2D = True)

## pRF Polar Angle
pa4plot = np.zeros(fa_model.pRF_estimates['r2'].shape); pa4plot[:] = np.nan
pa4plot[fa_model.pRF_estimates['r2']>0] = ((polar_angle + np.pi) / (np.pi * 2.0))[fa_model.pRF_estimates['r2']>0]

# get matplotlib color map from segmented colors
PA_cmap = mri_utils.make_colormap(colormap = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb',
                              '#3d549f','#655099','#ad5a9b','#dd3933'], bins = n_bins_colors, 
                                    cmap_name = 'PA_mackey_costum',
                              discrete = False, add_alpha = False, return_cmap = True)

images['PA'] = mri_utils.make_raw_vertex_image(pa4plot, 
                                               cmap = PA_cmap, vmin = 0, vmax = 1, 
                                              data2 = alpha_level, vmin2 = 0, vmax2 = 1, 
                                               subject = pysub, data2D = True)

## initialize interactive figure

plot_obj = plot_utils.visualize_on_click(params, fa_model.pRF_estimates, 
                                    prf_dm = fa_model.prf_dm, 
                                    max_ecc_ext = np.max(fa_model.x_ecc_lim),
                                    pRF_data = pRF_data)

# set flatmaps in object class
plot_obj.images = images

plot_obj.set_figure()

cortex.quickshow(images['pRF_rsq'], fig = plot_obj.flatmap_ax,
                 with_rois = False, with_curvature = True, with_colorbar=False)

plot_obj.full_fig.canvas.mpl_connect('button_press_event', plot_obj.onclick)
plot_obj.full_fig.canvas.mpl_connect('key_press_event', plot_obj.onkey)

plt.show()