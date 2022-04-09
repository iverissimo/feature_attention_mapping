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

# set estimate key names
estimate_keys = params['mri']['fitting']['pRF']['estimate_keys'][model_type]

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
visual_dm = mri_utils.make_pRF_DM(op.join(derivatives_dir,'pRF_fit', 'sub-{sj}'.format(sj=sj), 'DMprf.npy'), params, 
                                 save_imgs = False, res_scaling = 0.1, crop = params['prf']['crop'] , 
                                 crop_TR = params['prf']['crop_TR'], overwrite=False)

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
masked_est = mri_utils.mask_estimates(estimates, 
                                      estimate_keys = estimate_keys+['hrf_derivative','hrf_dispersion'],
                                      x_ecc_lim = x_ecc_lim, y_ecc_lim = y_ecc_lim)

masked_rsq = masked_est['r2']
masked_ns = masked_est['ns']

## make violin plots with exponent values per ROI

# if using atlas to get ROIs 
use_atlas = False 
# get pycortex sub
pysub = params['plotting']['pycortex_sub']+'_sub-{sj}'.format(sj=sj) # because subject specific borders 
# get vertices for ROI
roi_verts = {} #empty dictionary  

## get vertices and color palette, 
# for consistency
if use_atlas:
    # Get Glasser atlas
    atlas_df, atlas_array = mri_utils.create_glasser_df(op.join(derivatives_dir,'glasser_atlas','59k_mesh'))

    # ROI names
    ROIs = list(params['plotting']['ROIs']['glasser_atlas'].keys())
    # colors
    color_codes = {key: params['plotting']['ROIs']['glasser_atlas'][key]['color'] for key in ROIs}

    # get vertices for ROI
    for _,key in enumerate(ROIs):
        roi_verts[key] = np.hstack((np.where(atlas_array == ind)[0] for ind in atlas_df[atlas_df['ROI'].isin(params['plotting']['ROIs']['glasser_atlas'][key]['ROI'])]['index'].values))

else:
    # set ROI names
    ROIs = params['plotting']['ROIs'][space]

    # dictionary with one specific color per group - similar to fig3 colors
    ROI_pal = params['plotting']['ROI_pal']
    color_codes = {key: ROI_pal[key] for key in ROIs}

    # get vertices for ROI
    for _,val in enumerate(ROIs):
        roi_verts[val] = cortex.get_roi_verts(pysub,val)[val]


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
filename = op.join(figures_pth,'flatmap_space-{space}_type-exponent_visual_masked.svg'.format(space=pysub))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['masked_ns'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

## plot estimate before masking
images['ns'] = cortex.Vertex2D(estimates['ns'], mri_utils.normalize(np.clip(rsq, 0, 0.8)),
                            pysub,
                            vmin = 0, vmax = 1,
                            vmin2 = 0, vmax2 = 1,
                            cmap='plasma_alpha')
#cortex.quickshow(images['ns'],with_curvature=True,with_sulci=True)
filename = op.join(figures_pth,'flatmap_space-{space}_type-exponent_visual.svg'.format(space=pysub))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['ns'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

# set threshold for plotting
rsq_threshold = .1 #params['plotting']['rsq_threshold']

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

fig.savefig(op.join(figures_pth,'exponent_visual_boxplot.svg'), dpi=100)

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
g.savefig(op.join(figures_pth,'exponent_visual_histogram.svg'), dpi=100)