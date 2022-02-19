################################################
#      Make some exponent plots 
#    To see distribution of fits
#    in surface and per ROI - ONLY CSS
################################################


import numpy as np
import os, sys
import os.path as op
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import cortex
import pandas as pd
import seaborn as sns
from prfpy.stimulus import PRFStimulus2D

from FAM_utils import mri as mri_utils

# load settings from yaml
with open(op.join(str(Path(os.getcwd()).parents[1]),'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')
  
elif len(sys.argv) < 3:
    raise NameError('Please add type of run to be fitted (ex: leave_01_out vs median) '
                    'as 2nd argument in the command line!')
    
else:
    # fill subject number and chunk number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(3)
    run_type = str(sys.argv[2])

# set font type for plots globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# some settings
base_dir = params['general']['current_dir'] # which machine we run the data
acq = params['mri']['acq'] # if using standard files or nordic files
space = params['mri']['space'] # subject space
total_chunks = params['mri']['fitting']['pRF']['total_chunks'][space] # number of chunks that data was split in

TR = params['mri']['TR']

# type of model to fit
model_type = 'css'

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

# output dir to save fit and plot
figures_pth = op.join(derivatives_dir,'plots','exponent',
                      'sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run_type)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

# path to pRF fits 
fits_pth =  op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), space, 'iterative_{model}'.format(model=model_type),'run-{run}'.format(run=run_type))

## Load pRF estimates 

# path to combined estimates
estimates_pth = op.join(fits_pth,'combined')

# combined estimates filename
est_name = [x for _,x in enumerate(os.listdir(fits_pth)) if 'chunk-001' in x][0]
est_name = est_name.replace('chunk-001_of_{ch}'.format(ch=str(total_chunks).zfill(3)),'chunk-combined')

# total path to estimates path
estimates_combi = op.join(estimates_pth,est_name)

if op.isfile(estimates_combi): # if combined estimates exists

    print('loading %s'%estimates_combi)
    estimates = np.load(estimates_combi) # load it

else: # if not join chunks and save file
    if not op.exists(estimates_pth):
        os.makedirs(estimates_pth) 

    estimates = mri_utils.join_chunks(fits_pth, estimates_combi, fit_hrf = params['mri']['fitting']['pRF']['fit_hrf'],
                            chunk_num = total_chunks, fit_model = 'it{model}'.format(model=model_type)) #'{model}'.format(model=model_type)))#


# define design matrix 
visual_dm = mri_utils.make_pRF_DM(op.join(derivatives_dir,'pRF_fit', 'sub-{sj}'.format(sj=sj), 'DMprf.npy'), params, save_imgs=False, downsample=0.1, 
                                        crop = params['prf']['crop'] , crop_TR = params['prf']['crop_TR'], overwrite=False)

# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm = params['monitor']['height'],
                        screen_distance_cm = params['monitor']['distance'],
                        design_matrix = visual_dm,
                        TR = TR)

# get the ecc limits (in dva)
# to mask estimates
x_ecc_lim, y_ecc_lim = mri_utils.get_ecc_limits(visual_dm,params,screen_size_deg = [prf_stim.screen_size_degrees,prf_stim.screen_size_degrees])

rsq = estimates['r2'] 

# mask estimates
print('masking estimates')
masked_est = mri_utils.mask_estimates(estimates, fit_model = model_type,
                            x_ecc_lim = x_ecc_lim, y_ecc_lim = y_ecc_lim)

masked_rsq = masked_est['rsq']
masked_ns = masked_est['ns']

# get vertices for subject fsaverage
ROIs = params['plotting']['ROIs'][space]

# dictionary with one specific color per group - similar to fig3 colors
ROI_pal = params['plotting']['ROI_pal']
color_codes = {key: ROI_pal[key] for key in ROIs}

# get pycortex sub
pysub = params['plotting']['pycortex_sub'] 

# get vertices for ROI
roi_verts = {} #empty dictionary  
for _,val in enumerate(ROIs):
    roi_verts[val] = cortex.get_roi_verts(pysub,val)[val]


## save flatmap
images = {}

alpha_level = mri_utils.normalize(np.clip(masked_rsq, 0, 0.8))

## plot rsq before masking
images['masked_ns'] = cortex.Vertex2D(masked_ns, alpha_level,
                            pysub,
                            vmin = 0, vmax = 1,
                            vmin2 = 0, vmax2 = 1,
                            cmap='plasma_alpha')
cortex.quickshow(images['masked_ns'],with_curvature=True,with_sulci=True)
filename = op.join(figures_pth,'flatmap_space-{space}_type-exponent_visual_masked.svg'.format(space=pysub))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['masked_ns'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

## plot rsq before masking
images['ns'] = cortex.Vertex2D(estimates['ns'], mri_utils.normalize(np.clip(rsq, 0, 0.8)),
                            pysub,
                            vmin = 0, vmax = 1,
                            vmin2 = 0, vmax2 = 1,
                            cmap='plasma_alpha')
cortex.quickshow(images['ns'],with_curvature=True,with_sulci=True)
filename = op.join(figures_pth,'flatmap_space-{space}_type-exponent_visual.svg'.format(space=pysub))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['ns'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

# set threshold for plotting
rsq_threshold = params['plotting']['rsq_threshold']

## plot distribution
for idx,rois_ks in enumerate(ROIs): 
    
    # mask estimates
    print('masking estimates for ROI %s'%rois_ks)
    
    roi_ns = masked_ns[roi_verts[rois_ks]]
    roi_rsq = masked_rsq[roi_verts[rois_ks]]
    
    roi_ns = roi_ns[roi_rsq >= rsq_threshold]

    if idx == 0:
        df_ns = pd.DataFrame({'roi': np.tile(rois_ks,len(roi_ns)),'exponent': roi_ns})
    else:
        df_ns = df_ns.append(pd.DataFrame({'roi': np.tile(rois_ks,len(roi_ns)),'exponent': roi_ns}),
                                                   ignore_index = True)

# make figures
fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

v1 = sns.boxplot(data = df_ns, x = 'roi', y = 'exponent', palette = color_codes)
                    #cut=0, inner='box', palette = color_codes) # palette ='Set3',linewidth=1.8)

v1.set(xlabel=None)
v1.set(ylabel=None)
plt.margins(y=0.025)
#sns.swarmplot(x = 'roi', y = 'exponent', data=df_ns,color=".25",alpha=0.5)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)

plt.xlabel('ROI',fontsize = 20,labelpad=18)
plt.ylabel('Exponent',fontsize = 20,labelpad=18)
plt.ylim(0,1.1)

fig.savefig(op.join(figures_pth,'exponent_visual_boxplot.svg'), dpi=100)


