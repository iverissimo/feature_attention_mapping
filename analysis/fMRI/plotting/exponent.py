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

# if we fitted hrf
fit_hrf = params['mri']['fitting']['pRF']['fit_hrf']
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

# set estimate key names
estimate_keys = params['mri']['fitting']['pRF']['estimate_keys'][model_type]

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

if fit_hrf:
    fits_pth = op.join(fits_pth,'with_hrf')
    estimate_keys = estimate_keys+['hrf_derivative','hrf_dispersion']

## Load pRF estimates
estimates = mri_utils.load_pRF_estimates(fits_pth, params, 
                                                 total_chunks = total_chunks, model_type = model_type) 

# define design matrix 
visual_dm = mri_utils.make_pRF_DM(op.join(derivatives_dir,'pRF_fit', 'sub-{sj}'.format(sj=sj), 'DMprf.npy'), params, 
                                save_imgs = False, res_scaling = 0.1, TR = params['mri']['TR'],
                                crop = params['prf']['crop'] , crop_TR = params['prf']['crop_TR'], 
                                shift_TRs = shift_TRs, shift_TR_num = shift_TR_num, 
                                oversampling_time = osf,
                                overwrite = False, mask = [], event_onsets = [])

# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm = params['monitor']['height'],
                        screen_distance_cm = params['monitor']['distance'],
                        design_matrix = visual_dm,
                        TR = TR)

# get the ecc limits (in dva)
# to mask estimates
#x_ecc_lim, y_ecc_lim = mri_utils.get_ecc_limits(visual_dm,params,screen_size_deg = [prf_stim.screen_size_degrees,prf_stim.screen_size_degrees])
x_ecc_lim = [- prf_stim.screen_size_degrees/2, prf_stim.screen_size_degrees/2]
y_ecc_lim = [- prf_stim.screen_size_degrees/2, prf_stim.screen_size_degrees/2] 

rsq = estimates['r2'] 

# mask estimates
print('masking estimates')
masked_est = mri_utils.mask_estimates(estimates, 
                                      estimate_keys = estimate_keys,
                                      x_ecc_lim = x_ecc_lim, y_ecc_lim = y_ecc_lim)

masked_rsq = masked_est['r2']
masked_ns = masked_est['ns']

# get pycortex sub
pysub = params['plotting']['pycortex_sub']+'_sub-{sj}'.format(sj=sj) # because subject specific borders 

ROIs, roi_verts, color_codes = mri_utils.get_rois4plotting(params, pysub = pysub, 
                                            use_atlas = False, atlas_pth = op.join(derivatives_dir,'glasser_atlas','59k_mesh'))

## make flatmap
## save flatmap
images = {}

alpha_level = mri_utils.normalize(np.clip(masked_rsq, 0, 0.8))

## plot masked estimate
images['masked_ns'] = cortex.Vertex2D(masked_ns, alpha_level,
                            pysub,
                            vmin = 0, vmax = 1,
                            vmin2 = 0, vmax2 = 1,
                            cmap='plasma_alpha')
#cortex.quickshow(images['masked_ns'],with_curvature=True,with_sulci=True)
filename = op.join(figures_pth,'flatmap_space-{space}_type-exponent_visual_masked_withHRF-{hrf}.png'.format(space = pysub,
                                                                                                            hrf = str(params['mri']['fitting']['pRF']['fit_hrf'])))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['masked_ns'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

## plot estimate before masking
images['ns'] = cortex.Vertex2D(estimates['ns'], mri_utils.normalize(np.clip(rsq, 0, 0.8)),
                            pysub,
                            vmin = 0, vmax = 1,
                            vmin2 = 0, vmax2 = 1,
                            cmap='plasma_alpha')
#cortex.quickshow(images['ns'],with_curvature=True,with_sulci=True)
filename = op.join(figures_pth,'flatmap_space-{space}_type-exponent_visual_withHRF-{hrf}.png'.format(space = pysub,
                                                                                                    hrf = str(params['mri']['fitting']['pRF']['fit_hrf'])))
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
        df_ns = pd.DataFrame({'roi': np.tile(rois_ks,len(roi_ns)),
                              'exponent': roi_ns,
                             'r2': roi_rsq[roi_rsq >= rsq_threshold]})
    else:
        df_ns = df_ns.append(pd.DataFrame({'roi': np.tile(rois_ks,len(roi_ns)),
                                           'exponent': roi_ns,
                                          'r2': roi_rsq[roi_rsq >= rsq_threshold]}),
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

fig.savefig(op.join(figures_pth,'exponent_visual_boxplot_withHRF-{hrf}.png'.format(hrf = str(params['mri']['fitting']['pRF']['fit_hrf']))), dpi=100)

## make value distribution plot
sns.set(font_scale = 1.5)
g = sns.FacetGrid(df_ns, #the dataframe to pull from
                  col = "roi", #define the column for each subplot row to be differentiated by
                  hue = "roi", #define the column for each subplot color to be differentiated by
                  aspect = .65, #aspect * height = width
                  height = 3.5, #height of each subplot
                  palette = color_codes,
                  col_wrap = 4
                 )
g.map(sns.histplot, "exponent", bins = 5, stat = 'percent')
#g.set(label='big')#xlim=(0, 1))
# flatten axes into a 1-d array
axes = g.axes.flatten()

# iterate through the axes
for i, ax in enumerate(axes):
    ax.axvline(df_ns.loc[df_ns['roi']==ROIs[i]]["exponent"].median(), ls='--', c='green')
    
g.set_titles("{col_name}")  # use this argument literally
g.savefig(op.join(figures_pth,'exponent_visual_histogram_withHRF-{hrf}.png'.format(hrf = str(params['mri']['fitting']['pRF']['fit_hrf']))), dpi=100)