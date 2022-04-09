################################################
#      Make some rsq plots 
#    To see distribution of fits
#    in surface and per ROI
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

elif len(sys.argv) < 4:
    raise NameError('Please add task (ex: FA vs pRF) '
                    'as 3rd argument in the command line!')
    
else:
    # fill subject number and chunk number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(3)
    run_type = str(sys.argv[2])
    task = str(sys.argv[3]) 

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
model_type = params['mri']['fitting']['pRF']['fit_model']

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
figures_pth = op.join(derivatives_dir,'plots','rsq','{task}fit'.format(task=task),
                      'sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run_type)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

if task == 'pRF':

    # path to iterative and grid pRF fits 
    fits_pth = {'iterative': op.join(derivatives_dir,'{task}_fit'.format(task=task),'sub-{sj}'.format(sj=sj), space, 
                                     'iterative_{model}'.format(model=model_type),'run-{run}'.format(run=run_type)),
               'grid': op.join(derivatives_dir,'{task}_fit'.format(task=task),'sub-{sj}'.format(sj=sj), space, 
                               '{model}'.format(model=model_type),'run-{run}'.format(run=run_type))}
    
    # grid fitting doesnt include hrf
    fit_hrf = {'iterative': params['mri']['fitting']['pRF']['fit_hrf'], 'grid': False}

    ## Load pRF estimates 
        
    # combined estimates filename + np array dict
    estimates_combi = {'iterative': None, 'grid': None}
    estimates = {'iterative': None, 'grid': None}
    
    for k in fits_pth.keys():
        
        est_name = [x for _,x in enumerate(os.listdir(fits_pth[k])) if 'chunk-001' in x][0]
        est_name = est_name.replace('chunk-001_of_{ch}'.format(ch=str(total_chunks).zfill(3)),'chunk-combined')
        
        # total path to estimates path
        estimates_combi[k] = op.join(fits_pth[k],'combined', est_name)
    
        if op.isfile(estimates_combi[k]): # if combined estimates exists

                print('loading %s'%estimates_combi[k])
                estimates[k] = np.load(estimates_combi[k]) # load it

        else: # if not join chunks and save file
            if not op.exists(op.join(fits_pth[k],'combined')):
                os.makedirs(op.join(fits_pth[k],'combined')) 

            # model name to use as input for func
            mod_name = 'it{model}'.format(model=model_type) if k == 'iterative' else '{model}'.format(model=model_type)
            
            # combine estimate chunks
            estimates[k] = mri_utils.join_chunks(fits_pth[k], estimates_combi[k], fit_hrf = fit_hrf[k],
                                    chunk_num = total_chunks, fit_model = mod_name) 

    
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

    rsq = estimates['iterative']['r2'] 

    # mask estimates
    print('masking estimates')
    masked_est = mri_utils.mask_estimates(estimates['iterative'], 
                                          estimate_keys = estimate_keys+['hrf_derivative','hrf_dispersion'],
                                          x_ecc_lim = x_ecc_lim, y_ecc_lim = y_ecc_lim)

    masked_rsq = masked_est['r2']

    # saved masked rsq, useful for FA plots
    np.save(op.join(fits_pth['iterative'],'combined','masked_rsq.npy'), masked_rsq)

    plot_lims_dist = [0,1] # axis value for plotting
    plot_lims_flat = [0,.8] # axis value for plotting


elif task == 'FA':
    
    # mask rsq given masked rsq of pRF mean run (within screen boundaries etc)
    pRF_masked_rsq = np.load(op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), space, 
                      'iterative_{model}'.format(model=model_type),'run-mean','combined','masked_rsq.npy'))
    
    runs = ['1','2','3','4'] if run_type == 'mean' else [run_type]
        
    rsq = []
    
    for r in runs:
        
        # path to FA fits 
        fits_pth =  op.join(derivatives_dir,'FA_GLM_fit','sub-{sj}'.format(sj=sj), 
                            space, model_type,'run-{run}'.format(run=r))
        
        # get GLM estimates file
        estimates_filename = [op.join(fits_pth, val) for val in os.listdir(fits_pth) if val.endswith('_estimates.npz')]
        estimates = np.load(estimates_filename[0])

        # get rsquared
        tmp_arr = estimates['r2'].copy()
        tmp_arr[np.isnan(pRF_masked_rsq)] = np.nan

        rsq.append(tmp_arr)
        
    
    rsq = np.nanmean(rsq, axis=0)
    masked_rsq = rsq.copy()

    plot_lims_dist = [0,.5] # axis value for plotting
    plot_lims_flat = [0, .3] #[0,.4] # axis value for plotting 

# set threshold for plotting
rsq_threshold = params['plotting']['rsq_threshold']

## make violin plots with values per ROI

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



for idx,rois_ks in enumerate(ROIs): 
    
    # mask estimates
    print('masking estimates for ROI %s'%rois_ks)
    
    roi_rsq = masked_rsq[roi_verts[rois_ks]]
    roi_rsq = roi_rsq[roi_rsq >= rsq_threshold]

    if idx == 0:
        df_rsq = pd.DataFrame({'roi': np.tile(rois_ks,len(roi_rsq)),'rsq': roi_rsq})
    else:
        df_rsq = df_rsq.append(pd.DataFrame({'roi': np.tile(rois_ks,len(roi_rsq)),'rsq': roi_rsq}),
                                                   ignore_index = True)


# make figures
fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

v1 = sns.violinplot(data = df_rsq, x = 'roi', y = 'rsq', 
                    cut=0, inner='box', palette = color_codes, linewidth=1.8) # palette ='Set3',linewidth=1.8)

v1.set(xlabel=None)
v1.set(ylabel=None)
plt.margins(y=0.025)
#sns.swarmplot(x='ecc', y='cs', data=crwd_df4plot,color=".25",alpha=0.5)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)

plt.xlabel('ROI',fontsize = 20,labelpad=18)
plt.ylabel('RSQ',fontsize = 20,labelpad=18)
plt.ylim(plot_lims_dist[0],plot_lims_dist[1])

fig.savefig(op.join(figures_pth,'rsq_%s_violinplot.svg'%model_type), dpi=100)

images = {}

## plot rsq before masking
images['rsq'] = cortex.Vertex(rsq, 
                            pysub,
                            vmin = plot_lims_flat[0], vmax = plot_lims_flat[1],
                            cmap='Reds')
#cortex.quickshow(images['rsq'],with_curvature=True,with_sulci=True)

filename = op.join(figures_pth,'flatmap_space-{space}_type-rsq_{model}.svg'.format(space=pysub, model=model_type))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

## plot masked rsq 
# also mask out nans, makes it look nicer
new_rsq = np.array([x if np.isnan(x)==False else 0 for _,x in enumerate(masked_rsq)])

images['rsq_masked'] = cortex.Vertex(new_rsq, 
                                    pysub,
                                    vmin = plot_lims_flat[0], vmax = plot_lims_flat[1],
                                    cmap='Reds')
#cortex.quickshow(images['rsq_masked'],with_curvature=True,with_sulci=True)

filename = op.join(figures_pth,'flatmap_space-{space}_type-rsq_masked_{model}.svg'.format(space=pysub, model=model_type))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_masked'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

## compare pRF model rsq's
if task == 'pRF':
    
    ## compare grid and iterative
    # mask also
    grid_r2 = estimates['grid']['r2'].copy()
    grid_r2[new_rsq == 0] = 0
    
    ## plot percentage increase from grid to iterative
    images['rsq_grid_iter_change'] = cortex.Vertex((new_rsq - grid_r2)/grid_r2 * 100, 
                                    pysub,
                                    vmin = -100, vmax = 100,
                                    cmap='BuBkRd')
    #cortex.quickshow(images['rsq_grid_iter_change'],with_curvature=True,with_sulci=True)

    filename = op.join(figures_pth,'flatmap_space-{space}_percent_change_grid2iterative_type-rsq_masked_{model}.svg'.format(space=pysub, model=model_type))
    print('saving %s' %filename)
    _ = cortex.quickflat.make_png(filename, images['rsq_grid_iter_change'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

    # if not gauss, compare rsq of model with gauss rsq
    if model_type != 'gauss':
        
        gauss_est_filename = estimates_combi['iterative'].replace(model_type, 'gauss')
        
        if op.isfile(gauss_est_filename): # if combined estimates exists

                print('loading %s'%gauss_est_filename)
                estimates_gauss = np.load(gauss_est_filename) # load it

        else: # if not join chunks and save file
            if not op.exists(op.split(gauss_est_filename)[0]):
                os.makedirs(op.split(gauss_est_filename)[0]) 

            # combine estimate chunks
            estimates_gauss = mri_utils.join_chunks(op.split(gauss_est_filename)[0], gauss_est_filename, 
                                                    fit_hrf = fit_hrf['iterative'],
                                                chunk_num = total_chunks, fit_model = 'itgauss') 
        
        ## compare gauss and current model
        # mask also
        gauss_r2 = estimates_gauss['r2'].copy()
        gauss_r2[new_rsq == 0] = 0

        ## plot percentage increase from grid to iterative
        images['rsq_gauss_%s_change'%model_type] = cortex.Vertex((new_rsq - gauss_r2)/gauss_r2 * 100, 
                                        pysub,
                                        vmin = -100, vmax = 100,
                                        cmap='BuBkRd')
        #cortex.quickshow(images['rsq_gauss_%s_change'%model_type],with_curvature=True,with_sulci=True)
        
        filename = op.join(figures_pth,'flatmap_space-{space}_percent_change_gauss2{model}_type-rsq_masked_{model}.svg'.format(space=pysub, model=model_type))
        print('saving %s' %filename)
        _ = cortex.quickflat.make_png(filename, images['rsq_gauss_%s_change'%model_type], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)





