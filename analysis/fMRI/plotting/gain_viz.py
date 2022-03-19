## script to plot gain outcomes

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

# inserting path to fitting, to get feature model objects
# should reorganize folder in future, to avoid this
sys.path.insert(1, op.join(str(Path(os.getcwd()).parents[0]), 'fitting'))
from feature_model import FA_GainModel

import cortex
import matplotlib.pyplot as plt
from statsmodels.stats import weightstats
import seaborn as sns

# load settings from yaml
with open(op.join(str(Path(os.getcwd()).parents[1]),'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')
  
elif len(sys.argv) < 3:
    raise NameError('Please add type of run to be fitted (ex: 1 vs all) '
                    'as 2nd argument in the command line!')
    
else:
    # fill subject number and chunk number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(3)
    run = str(sys.argv[2])

if run == 'all':
    all_runs = ['1', '2', '3', '4'] # all runs
else:
    all_runs = [run]

## some settings
base_dir = params['general']['current_dir'] # which machine we run the data
acq = params['mri']['acq'] # if using standard files or nordic files
space = params['mri']['space'] # subject space
TR = params['mri']['TR']
model_type = params['mri']['fitting']['pRF']['fit_model']

# get pycortex sub
pysub = params['plotting']['pycortex_sub'] 

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

# output dir to save fit and plot
figures_pth = op.join(derivatives_dir,'plots','FA_gain','sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

data = []
results = []
model_tc = []
dm = []

for i, run in enumerate(all_runs):
    
    #################### for now, because checking only iterative fit (no grid) #####################
    # later not needed 'SP_null' part
    FA_fit_dir = op.join(derivatives_dir,'FA_gain','sub-{sj}'.format(sj=sj), space, # 'SP_null', 
                          fa_model.prf_model_type, 'run-{run}'.format(run=run))

    ################################################################################################

    ## list with absolute file name to be fitted
    proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-FA' in h and
                     'acq-{acq}'.format(acq=acq) in h and 'run-{run}'.format(run=run) in h and h.endswith(file_ext)]

    ## load functional data
    file = proc_files[0]
    if len(proc_files)>1:
        raise ValueError('%s files found to fit, unsure of which to use'%len(proc_files))
    else:
        print('Fitting %s'%file)
    data.append(np.load(file,allow_pickle=True)) # will be (vertex, TR)

    
    if i == 0: 
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

        
    ## load results from iterative fit

    ## save fitted params Dataframe
    results.append(pd.read_csv(op.join(FA_fit_dir,'iterative_params.csv'))) 
    
    # load DM
    dm_filename = op.join(FA_fit_dir,'DM_FA_iterative_gain_run-{run}.npz'.format(run=run))

    if op.isfile(dm_filename):
        dm.append(np.load(dm_filename))
        
    
    ## load model predictions
    model_tc_filename = op.join(FA_fit_dir,'prediction_FA_iterative_gain_run-{run}.npy'.format(run=run))
    
    if op.isfile(model_tc_filename):
        model_tc.append(np.load(model_tc_filename))


#### plot RSQ of iterative gain fit
rsq_gain = []
for i in range(np.array(results).shape[0]):
    
    r2 = np.zeros(np.array(data).shape[1]); r2[:] = np.nan
    r2[results[i]['vertex'].values] = results[i]['rsq'].values

    rsq_gain.append(r2)
    
rsq_gain = np.array(rsq_gain)

images = {}

## plot rsq average accross runs
images['FA_rsq'] = cortex.Vertex(np.nanmean(rsq_gain, axis=0), 
                            pysub,
                            vmin = 0, vmax = .3,
                            cmap='Reds').raw
cortex.quickshow(images['FA_rsq'],with_curvature=True,with_sulci=True)

filename = op.join(figures_pth,'flatmap_space-{space}_type-rsq_FA.svg'.format(space=pysub))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['FA_rsq'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

### now plot gain values

## reunite them all in dict, according to condition
gain_all = {}

for i, cond in enumerate(fa_model.unique_cond.keys()):
    
    for r in range(len(all_runs)):
    
        if r == 0:
            gain_all[cond] = results[r]['gain_%s'%cond].values
        else:
            gain_all[cond] = np.vstack((gain_all[cond], results[r]['gain_%s'%cond].values))

surf_gain_all = {}

for i, cond in enumerate(fa_model.unique_cond.keys()):
    
    gc = np.zeros(np.array(data).shape[1]); gc[:] = np.nan
    gc[results[0]['vertex'].values] = np.nanmean(gain_all[cond], axis= 0)

    surf_gain_all[cond] = gc
    
# plot gain on surface
for _, cond in enumerate(fa_model.unique_cond.keys()):
    
    images['gain_%s'%cond] = cortex.Vertex(surf_gain_all[cond], #np.subtract(surf_gain_all['ACAO'], surf_gain_all['UCUO']), 
                                pysub,
                                vmin = 0, vmax = 1,
                                cmap='plasma').raw
    cortex.quickshow(images['gain_%s'%cond], with_curvature=True,with_sulci=True)

    filename = op.join(figures_pth,'flatmap_space-{space}_type-{cond}.svg'.format(space = pysub, cond = cond))
    print('saving %s' %filename)
    _ = cortex.quickflat.make_png(filename, images['gain_%s'%cond], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

## get vertices from ROIs
## of glasser atlas

# Get Glasser atlas
atlas_df, atlas_array = mri_utils.create_glasser_df(op.join(derivatives_dir,'glasser_atlas','59k_mesh'))

# ROI names
ROIs = list(params['plotting']['ROIs']['glasser_atlas'].keys())
# colors
color_codes = {key: params['plotting']['ROIs']['glasser_atlas'][key]['color'] for key in ROIs}

# get vertices for ROI
roi_verts = {} #empty dictionary  
for _,key in enumerate(ROIs):
    roi_verts[key] = np.hstack((np.where(atlas_array == ind)[0] for ind in atlas_df[atlas_df['ROI'].isin(params['plotting']['ROIs']['glasser_atlas'][key]['ROI'])]['index'].values))


### make bar chart, per ROI 
avg_gain_df = pd.DataFrame(columns = ['condition','run','gain','roi'])

for roi in ROIs:
    for cond in fa_model.unique_cond.keys():
        for r, run in enumerate(all_runs):
            
            # roi vertices 
            ind = np.array([i for i, vert in enumerate(results[r]['vertex'].values) if vert in roi_verts[roi]]) #and vert in mask_ind])
            
            avg_gain_df = avg_gain_df.append(pd.DataFrame({'condition': [cond],
                                             'run': [run],
                                             'gain': [weightstats.DescrStatsW(gain_all[cond][r][ind], 
                                                                        weights = mri_utils.normalize(results[r]['rsq'].values[ind])).mean], #pRF_estimates['rsq'][results[r]['vertex'].values[ind]]).mean], #
                                             'roi': [roi]
                                        }))

for roi in ROIs:
    
    sns.set(font_scale=1.3)
    sns.set_style("ticks")

    fig_dims = (20, 10)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.barplot(y='condition', x='gain', data = avg_gain_df[avg_gain_df['roi']==roi])

    ax = plt.gca()
    plt.xticks(fontsize = 20, fontweight = "bold")
    plt.yticks(fontsize = 20, fontweight = "bold")
    ax.axes.tick_params(labelsize=18)
    sns.despine(offset=15)
    fig1 = plt.gcf()
    
    fig1.savefig(op.join(figures_pth,'gain_barplot_ROI-%s.svg'%(roi)), dpi=100,bbox_inches = 'tight')

## make figure with all ROIs

sns.set(font_scale=1.3)
sns.set_style("ticks")

fig_dims = (20, 10)

fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x='roi', y='gain', data = avg_gain_df, hue='condition')

ax = plt.gca()
plt.xticks(fontsize = 20, fontweight = "bold")
plt.yticks(fontsize = 20, fontweight = "bold")
ax.axes.tick_params(labelsize=18)
sns.despine(offset=15)
fig1 = plt.gcf()

fig1.savefig(op.join(figures_pth,'gain_barplot_ROI-all.svg'), dpi=100,bbox_inches = 'tight')

## calculate ecc and polar angle

complex_location = pRF_estimates['x'] + pRF_estimates['y'] * 1j # calculate eccentricity values
polar_angle = np.angle(complex_location)
eccentricity = np.abs(complex_location)

# make ecc vs gain plots
# weighted by the model rsq

min_ecc = 0.25
max_ecc = 4 #3.3
min_size = .25
max_size = 6
n_bins = 10

all_roi = pd.DataFrame(columns = ['mean_ecc','mean_gain','ROI','run', 'condition'])

for cond in fa_model.unique_cond.keys():
    for idx, roi in enumerate(ROIs): # go over ROIs
        print('ROI %s'%roi)

        df = pd.DataFrame(columns = ['ecc','run','gain','pRF_rsq','FA_rsq'])

        for r, run in enumerate(all_runs):

            # get relevant indices for plot 
            indices4plot = np.where((eccentricity >= min_ecc) & \
                                    (eccentricity <= max_ecc) & \
                                    (pRF_estimates['rsq'] >= rsq_threshold))[0]

            # get roi indices
            ind = np.array([i for i, vert in enumerate(results[r]['vertex'].values) if vert in roi_verts[roi] and vert in indices4plot])

            df = pd.DataFrame({'ecc': eccentricity[results[r]['vertex'].values[ind]],
                               #'size': pRF_estimates['size'][results[r]['vertex'].values[ind]],
                               'gain': gain_all[cond][r][ind],
                                'pRF_rsq': pRF_estimates['rsq'][results[r]['vertex'].values[ind]],
                                'FA_rsq': results[r]['rsq'].values[ind],
                                 'run': np.tile(r+1, len(ind))})


            mean_ecc, _, mean_gain, _ = mri_utils.get_weighted_bins(df, x_key = 'ecc', y_key = 'gain', 
                                                          weight_key = 'FA_rsq', n_bins = n_bins)


            all_roi = all_roi.append(pd.DataFrame({'mean_ecc': mean_ecc,'mean_gain': mean_gain,
                                                   'ROI': np.tile(roi,n_bins),
                                                   'condition': np.tile(cond,n_bins),
                                                  'run': np.tile(r+1,n_bins)}), ignore_index=True)

## plot gain vs ecc

for cond in fa_model.unique_cond.keys():
    sns.set(font_scale=1.3)
    sns.set_style("ticks")

    ax = sns.lmplot(x = 'mean_ecc', y = 'mean_gain', hue = 'ROI', data = all_roi.loc[all_roi['condition']==cond],
                    scatter=False, palette = color_codes)#, markers=['^','s','o'])

    ax = plt.gca()
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    #ax.axes.tick_params(labelsize=16)
    ax.axes.set_xlim(min_ecc,max_ecc)
    ax.axes.set_ylim(0,1)

    ax.set_xlabel('pRF eccentricity [dva]', fontsize = 20, labelpad = 15)
    ax.set_ylabel('gain %s'%cond, fontsize = 20, labelpad = 15)
    #ax.set_title('ecc vs size plot, %d bins from %.2f-%.2f ecc [dva]'%(n_bins,min_ecc,max_ecc),fontsize=12)
    sns.despine(offset=15)
    fig1 = plt.gcf()
    fig1.savefig(op.join(figures_pth,'gain_vs_ecc_ROIs_cond-%s'%cond), dpi=100,bbox_inches = 'tight')

# make size vs gain plots
# weighted by the model rsq

all_roi = pd.DataFrame(columns = ['mean_size','mean_gain','ROI','run', 'condition'])

for cond in fa_model.unique_cond.keys():
    for idx, roi in enumerate(ROIs): # go over ROIs
        print('ROI %s'%roi)

        df = pd.DataFrame(columns = ['size','run','gain','pRF_rsq','FA_rsq'])

        for r, run in enumerate(all_runs):

            # get relevant indices for plot 
            indices4plot = np.where((pRF_estimates['size'] >= min_size) & \
                                    (pRF_estimates['size'] <= max_size) & \
                                    (pRF_estimates['rsq'] >= rsq_threshold))[0]

            # get roi indices
            ind = np.array([i for i, vert in enumerate(results[r]['vertex'].values) if vert in roi_verts[roi] and vert in indices4plot])

            df = pd.DataFrame({'size': pRF_estimates['size'][results[r]['vertex'].values[ind]],
                               'gain': gain_all[cond][r][ind],
                                'pRF_rsq': pRF_estimates['rsq'][results[r]['vertex'].values[ind]],
                                'FA_rsq': results[r]['rsq'].values[ind],
                                 'run': np.tile(r+1, len(ind))})


            mean_size, _, mean_gain, _ = mri_utils.get_weighted_bins(df, x_key = 'size', y_key = 'gain', 
                                                          weight_key = 'FA_rsq', n_bins = n_bins)


            all_roi = all_roi.append(pd.DataFrame({'mean_size': mean_size,'mean_gain': mean_gain,
                                                   'ROI': np.tile(roi,n_bins),
                                                   'condition': np.tile(cond,n_bins),
                                                  'run': np.tile(r+1,n_bins)}), ignore_index=True)

## plot gain vs size

for cond in fa_model.unique_cond.keys():
    sns.set(font_scale=1.3)
    sns.set_style("ticks")

    ax = sns.lmplot(x = 'mean_size', y = 'mean_gain', hue = 'ROI', data = all_roi.loc[all_roi['condition']==cond],
                    scatter=False, palette = color_codes)#, markers=['^','s','o'])

    ax = plt.gca()
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    #ax.axes.tick_params(labelsize=16)
    ax.axes.set_xlim(0,4)
    ax.axes.set_ylim(0,1)

    ax.set_xlabel('pRF size [dva]', fontsize = 20, labelpad = 15)
    ax.set_ylabel('gain %s'%cond, fontsize = 20, labelpad = 15)
    #ax.set_title('ecc vs size plot, %d bins from %.2f-%.2f ecc [dva]'%(n_bins,min_ecc,max_ecc),fontsize=12)
    sns.despine(offset=15)
    fig1 = plt.gcf()
    fig1.savefig(op.join(figures_pth,'gain_vs_size_ROIs_cond-%s'%cond), dpi=100,bbox_inches = 'tight')

# make alpha level based on rsquared
mask = np.zeros(pRF_estimates['rsq'].shape); mask[:] = np.nan
mask[mask_ind] = pRF_estimates['rsq'][mask_ind]

alpha_level = mri_utils.normalize(np.clip(mask, 0, .8)) #rsq_threshold,.8))#

# make costum colormap, similar to mackey paper
n_bins = 256
ECC_colors = mri_utils.make_colormap(colormap = ['#dd3933','#f3eb53','#7cb956','#82cbdb','#3d549f'],
                               bins = n_bins, cmap_name = 'ECC_mackey_costum', discrete = False, add_alpha = True)

## Plot ecc
# create costume colormp rainbow_r
col2D_name = os.path.splitext(os.path.split(ECC_colors)[-1])[0]
print('created costum colormap %s'%col2D_name)

ecc4plot = np.zeros(pRF_estimates['rsq'].shape); ecc4plot[:] = np.nan
ecc4plot[mask_ind] = eccentricity[mask_ind]


images['ecc'] = cortex.Vertex2D(eccentricity, alpha_level, 
                        subject = pysub, 
                        vmin = 0, vmax = 6,
                        vmin2 = 0, vmax2 = np.nanmax(alpha_level),
                        cmap = col2D_name).raw

cortex.quickshow(images['ecc'],with_curvature=True,with_sulci=True,with_labels=False,
                 curvature_brightness = 0.4, curvature_contrast = 0.1)

SIZE_colors = mri_utils.make_colormap(colormap = 'viridis_r',
                               bins = n_bins, cmap_name = 'SIZE_costum', discrete = False, add_alpha = True)

col2D_name = os.path.splitext(os.path.split(SIZE_colors)[-1])[0]
print('created costum colormap %s'%col2D_name)

size4plot = np.zeros(pRF_estimates['rsq'].shape); size4plot[:] = np.nan
size4plot[mask_ind] = (pRF_estimates['size']/np.sqrt(pRF_estimates['ns']))[mask_ind]

images['size'] = cortex.Vertex2D(size4plot, alpha_level, 
                        subject = pysub,
                        vmin = 0, vmax = 7,
                        vmin2 = 0, vmax2 = np.nanmax(alpha_level),
                        cmap ='hot_alpha').raw #col2D_name)
cortex.quickshow(images['size'],with_curvature=True,with_sulci=True,with_labels=False,
                 curvature_brightness = 0.4, curvature_contrast = 0.1)

PA_colors = mri_utils.make_colormap(colormap = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb',
                              '#3d549f','#655099','#ad5a9b','#dd3933'],bins = n_bins, cmap_name = 'PA_mackey_costum',
                              discrete = False, add_alpha = True)

# create costume colormp rainbow_r
col2D_name = op.splitext(op.split(PA_colors)[-1])[0]
print('created costum colormap %s'%col2D_name)

pa4plot = np.zeros(pRF_estimates['rsq'].shape); pa4plot[:] = np.nan
pa4plot[mask_ind] = ((polar_angle + np.pi) / (np.pi * 2.0))[mask_ind]

images['PA'] = cortex.Vertex2D(pa4plot, alpha_level,
                                subject = pysub, 
                                vmin = 0, vmax = 1,
                                vmin2 = 0, vmax2 = np.nanmax(alpha_level),
                                cmap = col2D_name).raw

cortex.quickshow(images['PA'],with_curvature=True,with_sulci=True,with_colorbar=True,
                 curvature_brightness = 0.4, curvature_contrast = 0.1)

## plot rsq before masking

n4plot = np.zeros(pRF_estimates['rsq'].shape); n4plot[:] = np.nan
n4plot[mask_ind] = pRF_estimates['ns'][mask_ind]

images['ns'] = mri_utils.make_raw_vertex_image(n4plot, cmap = 'plasma', vmin = 0, vmax = 1, 
                          data2 = alpha_level, vmin2 = 0, vmax2 = 1, subject = pysub, data2D = True)


cortex.quickshow(images['ns'],with_curvature=True,with_sulci=True,with_colorbar=True,
                 curvature_brightness = 0.4, curvature_contrast = 0.1)

#cortex.webshow(images)