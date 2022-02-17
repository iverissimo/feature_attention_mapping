################################################
#      Make some ecc vs size plots 
#    To see distribution of fits
#    in surface and per region/roi 
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
from statsmodels.stats import weightstats
from prfpy.stimulus import PRFStimulus2D

sys.path.insert(0,'..') # add parent folder to path
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

task = 'pRF'

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

# define file extension that we want to use, 
# should include processing key words
file_ext = '_cropped_{filt}_{stand}.npy'.format(filt = params['mri']['filtering']['type'],
                                                    stand = 'psc')

# set paths
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
postfmriprep_dir = op.join(derivatives_dir,'post_fmriprep','sub-{sj}'.format(sj=sj),space,'processed')

# path to pRF fits 
fits_pth =  op.join(derivatives_dir,'{task}_fit'.format(task=task),'sub-{sj}'.format(sj=sj), space, 'iterative_{model}'.format(model=model_type),'run-{run}'.format(run=run_type))
#fits_pth =  op.join(derivatives_dir,'{task}_fit'.format(task=task),'sub-{sj}'.format(sj=sj), space, '{model}'.format(model=model_type),'run-{run}'.format(run=run_type))

# output dir to save fit and plot
figures_pth = op.join(derivatives_dir,'plots','size_ecc','{task}fit'.format(task=task),
                      'sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run_type)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

if task == 'pRF':
    
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

        estimates = mri_utils.join_chunks(fits_pth, estimates_combi,
                                chunk_num = total_chunks, fit_model = 'it{model}'.format(model=model_type)) #'{model}'.format(model=model_type)))#


# define design matrix 
visual_dm = mri_utils.make_pRF_DM(op.join(derivatives_dir,'pRF_fit', 'sub-{sj}'.format(sj=sj), 'DMprf.npy'), params, save_imgs=False, downsample=0.1, crop = params['prf']['crop'] , crop_TR = params['prf']['crop_TR'], overwrite=False)

# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm = params['monitor']['height'],
                         screen_distance_cm = params['monitor']['distance'],
                         design_matrix = visual_dm,
                         TR = TR)

# get the ecc limits (in dva)
# to mask estimates
x_ecc_lim, y_ecc_lim = mri_utils.get_ecc_limits(visual_dm,params,screen_size_deg = [prf_stim.screen_size_degrees,prf_stim.screen_size_degrees])

# mask estimates
print('masking estimates')
masked_est = mri_utils.mask_estimates(estimates, fit_model = model_type,
                                x_ecc_lim = x_ecc_lim, y_ecc_lim = y_ecc_lim)

rsq = masked_est['rsq']
x = masked_est['x']
y = masked_est['y']
size = masked_est['size']

complex_location = x + y * 1j # calculate eccentricity values
ecc = np.abs(complex_location)

# set threshold for plotting
rsq_threshold = params['plotting']['rsq_threshold']

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
    
regions = {'occipital': ['V1','V2','V3','V3AB','hV4','LO'],
            'parietal': ['IPS0','IPS1','IPS2+'],
            'frontal': ['sPCS','iPCS']}

# now select estimates per ROI

min_ecc = 0.25
max_ecc = 4 #3.3
n_bins = 10
    
for idx,roi in enumerate(ROIs): # go over ROIs

    # mask estimates
    print('masking estimates for ROI %s'%roi)

    # get datapoints for RF only belonging to roi
    new_size = size[roi_verts[roi]]

    new_complex_location = x[roi_verts[roi]] + y[roi_verts[roi]] * 1j # calculate eccentricity values
    new_ecc = np.abs(new_complex_location)

    new_rsq = rsq[roi_verts[roi]]

    # define indices of voxels within region to plot
    # with rsq > threshold, and where value not nan, ecc values between 0.25 and 3.3
    indices4plot = np.where((new_ecc >= min_ecc) & (new_ecc<= max_ecc) & (new_rsq >= rsq_threshold) & (np.logical_not(np.isnan(new_size))))[0]

    df = pd.DataFrame({'ecc': new_ecc[indices4plot],'size': new_size[indices4plot],
                        'rsq': new_rsq[indices4plot]})

    # sort values by eccentricity
    df = df.sort_values(by=['ecc'])  

    #divide in equally sized bins
    bin_size = int(len(df)/n_bins) 
    mean_ecc = []
    mean_ecc_std = []
    mean_size = []
    mean_size_std = []

    # for each bin calculate rsq-weighted means and errors of binned ecc/size 
    for j in range(n_bins): 
        mean_size.append(weightstats.DescrStatsW(df[bin_size*j:bin_size*(j+1)]['size'],weights=df[bin_size*j:bin_size*(j+1)]['rsq']).mean)
        mean_size_std.append(weightstats.DescrStatsW(df[bin_size*j:bin_size*(j+1)]['size'],weights=df[bin_size*j:bin_size*(j+1)]['rsq']).std_mean)
        mean_ecc.append(weightstats.DescrStatsW(df[bin_size*j:bin_size*(j+1)]['ecc'],weights=df[bin_size*j:bin_size*(j+1)]['rsq']).mean)
        mean_ecc_std.append(weightstats.DescrStatsW(df[bin_size*j:bin_size*(j+1)]['ecc'],weights=df[bin_size*j:bin_size*(j+1)]['rsq']).std_mean)

    if idx == 0:
        all_roi = pd.DataFrame({'mean_ecc': mean_ecc,'mean_ecc_std': mean_ecc_std,
                                'mean_size': mean_size,'mean_size_std': mean_size_std,
                                'ROI':np.tile(roi,n_bins)})
    else:
        all_roi = all_roi.append(pd.DataFrame({'mean_ecc': mean_ecc,'mean_ecc_std': mean_ecc_std,
                                               'mean_size': mean_size,'mean_size_std': mean_size_std,
                                               'ROI': np.tile(roi,n_bins)}),ignore_index=True)


### plot for Occipital Areas - V1 V2 V3 V3AB hV4 LO ###

sns.set(font_scale=1.3)
sns.set_style("ticks")

ax = sns.lmplot(x = 'mean_ecc', y = 'mean_size', hue = 'ROI', data = all_roi[all_roi.ROI.isin(regions['occipital'])].applymap(lambda x: x if isinstance(x, (float,str)) else x[0]),
                scatter=True, palette = color_codes, markers=['^','s','o'])

ax = plt.gca()
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
#ax.axes.tick_params(labelsize=16)
ax.axes.set_xlim(min_ecc,max_ecc)
ax.axes.set_ylim(0.5,4)

ax.set_xlabel('pRF eccentricity [dva]', fontsize = 20, labelpad = 15)
ax.set_ylabel('pRF size [dva]', fontsize = 20, labelpad = 15)
#ax.set_title('ecc vs size plot, %d bins from %.2f-%.2f ecc [dva]'%(n_bins,min_ecc,max_ecc),fontsize=12)
sns.despine(offset=15)
fig1 = plt.gcf()
fig1.savefig(op.join(figures_pth,'occipital_ecc_vs_size_binned_rsq-%0.2f.svg'%(rsq_threshold)), dpi=100,bbox_inches = 'tight')


# ### plot for Occipital Areas - V1 V2 V3 V3AB hV4 LO ###

# sns.set(font_scale=1.3)
# sns.set_style("ticks")

# ax = sns.lmplot(x = 'mean_ecc', y = 'mean_size', hue = 'ROI', data = all_roi[all_roi.ROI.isin(regions['occipital'])].applymap(lambda x: x if isinstance(x, (float,str)) else x[0]),
#                 scatter=True, palette="YlGnBu_r",markers=['^','s','o','v','D','h'])

# ax = plt.gca()
# plt.xticks(fontsize = 18)
# plt.yticks(fontsize = 18)
# #ax.axes.tick_params(labelsize=16)
# ax.axes.set_xlim(min_ecc,max_ecc)
# ax.axes.set_ylim(0,5)

# ax.set_xlabel('pRF eccentricity [dva]', fontsize = 20, labelpad = 15)
# ax.set_ylabel('pRF size [dva]', fontsize = 20, labelpad = 15)
# #ax.set_title('ecc vs size plot, %d bins from %.2f-%.2f ecc [dva]'%(n_bins,min_ecc,max_ecc),fontsize=12)
# sns.despine(offset=15)
# fig1 = plt.gcf()
# fig1.savefig(op.join(figures_pth,'occipital_ecc_vs_size_binned_rsq-%0.2f.svg'%(rsq_threshold)), dpi=100,bbox_inches = 'tight')

# ### plot for Parietal Areas - IPS0 IPS1 IPS2+ ###

# sns.set(font_scale=1.3)
# sns.set_style("ticks")

# ax = sns.lmplot(x = 'mean_ecc', y = 'mean_size', hue = 'ROI', data = all_roi[all_roi.ROI.isin(regions['parietal'])].applymap(lambda x: x if isinstance(x, (float,str)) else x[0]),
#                 scatter=True, palette="YlOrRd",markers=['^','s','o'])
# ax = plt.gca()
# plt.xticks(fontsize = 18)
# plt.yticks(fontsize = 18)
# #ax.axes.tick_params(labelsize=16)
# ax.axes.set_xlim(min_ecc,max_ecc)
# ax.axes.set_ylim(0,5)

# ax.set_xlabel('pRF eccentricity [dva]', fontsize = 20, labelpad = 15)
# ax.set_ylabel('pRF size [dva]', fontsize = 20, labelpad = 15)

# #ax.set_title('ecc vs size plot, %d bins from %.2f-%.2f ecc [dva]'%(n_bins,min_ecc,max_ecc),fontsize=12)
# sns.despine(offset=15)
# fig1 = plt.gcf()
# fig1.savefig(op.join(figures_pth,'parietal_ecc_vs_size_binned_rsq-%0.2f.svg'%(rsq_threshold)), dpi=100,bbox_inches = 'tight')

# ### plot for Frontal Areas - sPCS iPCS ###

# sns.set(font_scale=1.3)
# sns.set_style("ticks")

# ax = sns.lmplot(x = 'mean_ecc', y = 'mean_size', hue = 'ROI', data = all_roi[all_roi.ROI.isin(regions['frontal'])].applymap(lambda x: x if isinstance(x, (float,str)) else x[0]),
#                 scatter=True, palette="PuRd",markers=['^','s'])
# ax = plt.gca()
# plt.xticks(fontsize = 18)
# plt.yticks(fontsize = 18)
# #ax.axes.tick_params(labelsize=16)
# ax.axes.set_xlim(min_ecc,max_ecc)
# ax.axes.set_ylim(0,5)

# ax.set_xlabel('pRF eccentricity [dva]', fontsize = 20, labelpad = 15)
# ax.set_ylabel('pRF size [dva]', fontsize = 20, labelpad = 15)

# #ax.set_title('ecc vs size plot, %d bins from %.2f-%.2f ecc [dva]'%(n_bins,min_ecc,max_ecc),fontsize=12)
# sns.despine(offset=15)
# fig1 = plt.gcf()
# fig1.savefig(os.path.join(figures_pth,'frontal_ecc_vs_size_binned_rsq-%0.2f.svg'%(rsq_threshold)), dpi=100,bbox_inches = 'tight')

### flatmaps ####

# mask out nans

images = {}

# make alpha level based on rsquared
alpha_level = mri_utils.normalize(np.clip(rsq,rsq_threshold,.3))#

# make costum colormap, similar to mackey paper
n_bins = 256
ECC_colors = mri_utils.add_alpha2colormap(colormap = ['#dd3933','#f3eb53','#7cb956','#82cbdb','#3d549f'],
                               bins = n_bins, cmap_name = 'ECC_mackey_costum', discrete = False)

## Plot ecc
# create costume colormp rainbow_r
col2D_name = os.path.splitext(os.path.split(ECC_colors)[-1])[0]
print('created costum colormap %s'%col2D_name)


images['ecc'] = cortex.Vertex2D(ecc, alpha_level, 
                        subject = pysub, 
                        vmin = 0, vmax = 6,
                        vmin2 = 0, vmax2 = np.nanmax(alpha_level),
                        cmap = col2D_name)

cortex.quickshow(images['ecc'],with_curvature=True,with_sulci=True,with_labels=False,
                 curvature_brightness = 0.4, curvature_contrast = 0.1)

filename = op.join(figures_pth,'flatmap_space-fsaverage_type-ecc_visual.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['ecc'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

# make costum colormap viridis_r
n_bins = 256
SIZE_colors = mri_utils.add_alpha2colormap(colormap = 'viridis_r',
                               bins = n_bins, cmap_name = 'SIZE_costum', discrete = False)

col2D_name = os.path.splitext(os.path.split(SIZE_colors)[-1])[0]
print('created costum colormap %s'%col2D_name)

images['size'] = cortex.Vertex2D(size, alpha_level, 
                        subject = pysub,
                        vmin = 0, vmax = 7,
                        vmin2 = 0, vmax2 = np.nanmax(alpha_level),
                        cmap ='hot_alpha') #col2D_name)
cortex.quickshow(images['size'],with_curvature=True,with_sulci=True,with_labels=False,
                 curvature_brightness = 0.4, curvature_contrast = 0.1)

filename = op.join(figures_pth,'flatmap_space-fsaverage_type-size_visual.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['size'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)
