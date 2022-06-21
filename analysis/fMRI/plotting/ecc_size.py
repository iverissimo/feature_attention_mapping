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
model_type = params['mri']['fitting']['pRF']['fit_model']

# if we fitted hrf
fit_hrf = params['mri']['fitting']['pRF']['fit_hrf']

# set estimate key names
estimate_keys = params['mri']['fitting']['pRF']['estimate_keys'][model_type]

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

# set paths
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
postfmriprep_dir = op.join(derivatives_dir,'post_fmriprep','sub-{sj}'.format(sj=sj),space,'processed')

# path to pRF fits 
fits_pth =  op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), space, 'iterative_{model}'.format(model=model_type),'run-{run}'.format(run=run_type))

if fit_hrf:
    fits_pth = op.join(fits_pth,'with_hrf')
    estimate_keys = estimate_keys+['hrf_derivative','hrf_dispersion']

# output dir to save fit and plot
figures_pth = op.join(derivatives_dir,'plots','size_ecc','pRF_fit',
                      'sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run_type)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

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
# to mask estimates
#x_ecc_lim, y_ecc_lim = mri_utils.get_ecc_limits(visual_dm,params,screen_size_deg = [prf_stim.screen_size_degrees,prf_stim.screen_size_degrees])
x_ecc_lim = [- prf_stim.screen_size_degrees/2, prf_stim.screen_size_degrees/2]
y_ecc_lim = [- prf_stim.screen_size_degrees/2, prf_stim.screen_size_degrees/2] 

# mask estimates
print('masking estimates')
masked_est = mri_utils.mask_estimates(estimates, 
                                      estimate_keys = estimate_keys,
                                      x_ecc_lim = x_ecc_lim, y_ecc_lim = y_ecc_lim)

rsq = masked_est['r2']
x = masked_est['x']
y = masked_est['y']
# calculate eccentricity values
complex_location = x + y * 1j 
ecc = np.abs(complex_location)

# size estimates are different across models (due to non linearities)
# eg non-linearity interacts with the Gaussian standard deviation to make an effective pRF size of σ/sqr(n)
# best way is to compute the fwhm as a way to make size comparable across models 
if model_type in ['dn', 'dog']:
    size_fwhmax, fwatmin = mri_utils.fwhmax_fwatmin(model_type, masked_est)
else: 
    size_fwhmax = mri_utils.fwhmax_fwatmin(model_type, masked_est)

# get pycortex sub
pysub = params['plotting']['pycortex_sub']+'_sub-{sj}'.format(sj=sj) # because subject specific borders 

ROIs, roi_verts, color_codes = mri_utils.get_rois4plotting(params, pysub = pysub, 
                                            use_atlas = False, atlas_pth = op.join(derivatives_dir,'glasser_atlas','59k_mesh'))


# set threshold for plotting
rsq_threshold = params['plotting']['rsq_threshold']

# set region names 
regions = {'occipital': ['V1','V2','V3','V3AB','hV4','LO'],
            'parietal': ['IPS','IPS0','IPS1','IPS2+'],
            'frontal': ['sPCS','iPCS']}

# now select estimates per ROI

min_ecc = 0.25
max_ecc = 4 #3.3
n_bins = 10
max_size = 20

# to also save unbinned data and compare
unbinned_df = pd.DataFrame({'ecc': [], 'size': [], 'rsq': [], 'ROI': []})
    
for idx,roi in enumerate(ROIs): # go over ROIs

    # mask estimates
    print('masking estimates for ROI %s'%roi)

    # get datapoints for RF only belonging to roi
    new_size = size_fwhmax[roi_verts[roi]]

    new_complex_location = x[roi_verts[roi]] + y[roi_verts[roi]] * 1j # calculate eccentricity values
    new_ecc = np.abs(new_complex_location)

    new_rsq = rsq[roi_verts[roi]]

    # define indices of voxels within region to plot
    # with rsq > threshold, and where value not nan, ecc values between 0.25 and 3.3
    indices4plot = np.where((new_ecc >= min_ecc) & (new_ecc<= max_ecc) & (new_rsq >= rsq_threshold) & (new_size <= max_size) & (np.logical_not(np.isnan(new_size))))[0]

    df = pd.DataFrame({'ecc': new_ecc[indices4plot],'size': new_size[indices4plot],
                        'rsq': new_rsq[indices4plot]})
    
    ## save in unbinned dataframe
    unbinned_df = unbinned_df.append(pd.DataFrame({'ecc': new_ecc[indices4plot], 
                                                   'size': new_size[indices4plot], 
                                                   'rsq': new_rsq[indices4plot], 
                                                   'ROI': np.tile(roi,len(new_rsq[indices4plot]))
                                                }),ignore_index=True)

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



### plot binned regressions ###

sns.set(font_scale=1.3)
sns.set_style("ticks")

ax = sns.lmplot(x = 'mean_ecc', y = 'mean_size', hue = 'ROI', data = all_roi,
                scatter=True, palette = color_codes, markers=['^', 's', 'o', 'v', 'D', 'h', 'P', '.', ','])

ax = plt.gca()
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
#ax.axes.tick_params(labelsize=16)
ax.axes.set_xlim(min_ecc,max_ecc)
ax.axes.set_ylim(0.5,14)

ax.set_xlabel('pRF eccentricity [deg]', fontsize = 20, labelpad = 15)
ax.set_ylabel('pRF size FWHMax [deg]', fontsize = 20, labelpad = 15)
#ax.set_title('ecc vs size plot, %d bins from %.2f-%.2f ecc [dva]'%(n_bins,min_ecc,max_ecc),fontsize=12)
sns.despine(offset=15)
fig1 = plt.gcf()
fig1.savefig(op.join(figures_pth,'ecc_vs_size_binned_weighted_rsq-{thresh}_withHRF-{hrf}.png'.format(space = pysub,
                                                                                    thresh = rsq_threshold,
                                                                                    hrf = str(params['mri']['fitting']['pRF']['fit_hrf']))), dpi=100,bbox_inches = 'tight')

### plot UNbinned regressions ###

sns.set(font_scale=1.3)
sns.set_style("ticks")

g = sns.lmplot(x="ecc", y="size", hue = 'ROI', data = unbinned_df, scatter_kws={'alpha':0.15},
              scatter=True, palette = color_codes, markers=['^', 's', 'o', 'v', 'D', 'h', 'P', '.', ','])

ax = plt.gca()
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
#ax.axes.tick_params(labelsize=16)
ax.axes.set_xlim(min_ecc,max_ecc)
ax.axes.set_ylim(0.5,14)

ax.set_xlabel('pRF eccentricity [deg]', fontsize = 20, labelpad = 15)
ax.set_ylabel('pRF size FWHMax [deg]', fontsize = 20, labelpad = 15)
#ax.set_title('ecc vs size plot, %d bins from %.2f-%.2f ecc [dva]'%(n_bins,min_ecc,max_ecc),fontsize=12)
sns.despine(offset=15)
# to make legend full alpha
for lh in g._legend.legendHandles: 
    lh.set_alpha(1)
fig2 = plt.gcf()
fig2.savefig(op.join(figures_pth,'ecc_vs_size_UNbinned_rsq-{thresh}_withHRF-{hrf}.png'.format(space = pysub,
                                                                                    thresh = rsq_threshold,
                                                                                    hrf = str(params['mri']['fitting']['pRF']['fit_hrf']))), dpi=100,bbox_inches = 'tight')


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
alpha_level = mri_utils.normalize(np.clip(rsq, 0, .8)) #rsq_threshold,.8))#

# only used voxels where pRF rsq bigger than 0
ecc4plot = np.zeros(ecc.shape); ecc4plot[:] = np.nan
ecc4plot[rsq>0] = ecc[rsq>0]

# get matplotlib color map from segmented colors
n_bins_colors = 256
ecc_cmap = mri_utils.make_colormap(colormap = ['#dd3933','#f3eb53','#7cb956','#82cbdb','#3d549f'],
                               bins = n_bins_colors, cmap_name = 'ECC_mackey_costum', 
                                   discrete = False, add_alpha = False, return_cmap = True)

images['ecc'] = mri_utils.make_raw_vertex_image(ecc4plot, 
                                               cmap = ecc_cmap, vmin = 0, vmax = 6, 
                                              data2 = alpha_level, vmin2 = 0, vmax2 = 1, 
                                               subject = pysub, data2D = True)

#cortex.quickshow(images['ecc'],with_curvature=True,with_sulci=True,with_labels=False,
#                 curvature_brightness = 0.4, curvature_contrast = 0.1)
filename = op.join(figures_pth,'flatmap_space-fsaverage_type-ecc_visual_withHRF-{hrf}.png'.format(space = pysub,
                                                                                    hrf = str(params['mri']['fitting']['pRF']['fit_hrf'])))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['ecc'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

## plot pRF size ##

# only used voxels where pRF rsq bigger than 0
size4plot = np.zeros(size_fwhmax.shape); size4plot[:] = np.nan
size4plot[rsq>0] = size_fwhmax[rsq>0]

images['size'] = mri_utils.make_raw_vertex_image(size4plot, 
                                               cmap = 'hot', vmin = 0, vmax = 7, 
                                              data2 = alpha_level, vmin2 = 0, vmax2 = 1, 
                                               subject = pysub, data2D = True)

#cortex.quickshow(images['size'],with_curvature=True,with_sulci=True,with_labels=False,
#                 curvature_brightness = 0.4, curvature_contrast = 0.1)
filename = op.join(figures_pth,'flatmap_space-fsaverage_type-size_fwhmax_visual_withHRF-{hrf}.png'.format(space = pysub,
                                                                                    hrf = str(params['mri']['fitting']['pRF']['fit_hrf'])))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['size'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)
