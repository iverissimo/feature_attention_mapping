import numpy as np
import os, sys
import os.path as op
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import cortex
import pandas as pd
import seaborn as sns

from FAM_utils import mri as mri_utils

# load settings from yaml
with open(op.join(str(Path(os.getcwd()).parents[1]),'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')
else:
    sj = str(sys.argv[1]).zfill(3)

run_type = 'mean'

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
figures_pth = op.join(derivatives_dir,'plots','betas_FA',
                      'sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run_type)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

# mask rsq given masked rsq of pRF mean run (within screen boundaries etc)
pRF_masked_rsq = np.load(op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), space, 
                  'iterative_{model}'.format(model=model_type),'run-mean','combined','masked_rsq.npy'))

# set threshold for plotting
rsq_threshold = 0.2 #0.15 #0.2 #params['plotting']['rsq_threshold']

# make rsq mask, to only show betas with some pRF rsq
rsq_mask = pRF_masked_rsq < rsq_threshold

runs = ['1','2','3','4'] if run_type == 'mean' else [run_type]

## put betas from all runs in DataFrame
# for easier manipulation and control

betas_df = pd.DataFrame(columns = ['regressor', 'run','miniblock','betas','rsq','vertex'])
    
for r in runs:

    # path to FA fits 
    fits_pth =  op.join(derivatives_dir,'FA_GLM_fit','sub-{sj}'.format(sj=sj), #'OLD_STUFF','FA_GLM_fit','sub-{sj}'.format(sj=sj), 
                        space, model_type,'run-{run}'.format(run=r))

    # get GLM estimates file
    estimates_filename = [op.join(fits_pth, val) for val in os.listdir(fits_pth) if val.endswith('_estimates.npz')]
    estimates = np.load(estimates_filename[0])

    betas_no_intercept = estimates['betas'].copy()
    betas_no_intercept = betas_no_intercept[...,1::]

    num_vert = betas_no_intercept.shape[0]

    # get regressors dataframe
    all_regressors = pd.read_csv(op.join(fits_pth,'all_regressors_info.csv'))

    # loop over regressors that are not cue
    for ind in all_regressors.loc[all_regressors['reg_name'].str.contains('cue')==False].index.values:

        betas = betas_no_intercept[...,ind].copy()
        betas[rsq_mask] = np.nan

        betas_df = betas_df.append(pd.DataFrame({'regressor': np.tile('{cond}_{feat}'.format(cond=all_regressors.iloc[ind]['reg_name'].split('_')[0],
                                                                 feat = all_regressors.iloc[ind]['condition_name']),num_vert),
                             'condition': np.tile(all_regressors.iloc[ind]['reg_name'].split('_')[0],num_vert),
                             'run': np.tile(int(all_regressors.iloc[ind]['reg_name'].split('_')[-1].split('-')[-1]),num_vert),
                             'miniblock': np.tile(int(all_regressors.iloc[ind]['reg_name'].split('_')[1].split('-')[-1]),num_vert),
                             'betas': betas,
                             'rsq': estimates['r2'],
                             'vertex': np.arange(num_vert)}))


### now make new DF with average betas
## by condition

condition_names = betas_df['condition'].unique()

new_betas_df = pd.DataFrame(columns = ['condition','run','betas','rsq','vertex'])

for cond in condition_names:
    
    for r in [1,2,3,4]:
        
        avg_betas = betas_df[(betas_df['condition']==cond)&(betas_df['run']==r)].groupby('vertex').mean()['betas'].values
        avg_rsq = betas_df[(betas_df['condition']==cond)&(betas_df['run']==r)].groupby('vertex').mean()['rsq'].values
        
        
        new_betas_df = new_betas_df.append(pd.DataFrame({'condition': np.tile(cond,num_vert),
                                                         'run': np.tile(r,num_vert),
                                        'betas': avg_betas,
                                        'rsq': avg_rsq,
                                        'vertex':np.arange(num_vert)
                                    }))   

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

# do weighted average per ROI 
# (vertices with better model fit will weight more in the average)

avg_df = pd.DataFrame(columns = ['condition','run','betas','roi'])

for roi in ROIs:
    
    roi_df = new_betas_df.loc[new_betas_df['vertex'].isin(roi_verts[roi])] # sub select vertices that are in ROI
    
    for cond in condition_names:

        for r in [1,2,3,4]:

            avg_betas = roi_df[(roi_df['condition']==cond)&(roi_df['run']==r)]['betas'].values
            avg_rsq = roi_df[(roi_df['condition']==cond)&(roi_df['run']==r)]['rsq'].values

            avg_df = avg_df.append(pd.DataFrame({'condition': [cond],
                                                             'run': [r],
                                                            'betas': [np.average(avg_betas[np.where(~np.isnan(avg_betas))[0]],
                                                                                weights=avg_rsq[np.where(~np.isnan(avg_betas))[0]])],
                                                             'roi': [roi]
                                        }))#,ignore_index=True)  


## normalize by ACAO condition

norm_mean_betas_df = pd.DataFrame(columns = ['condition','run', 'betas','roi'])

for roi in ROIs:

    attention_betas = avg_df[(avg_df['condition']=='ACAO')&(avg_df['roi']==roi)]

    for cond in condition_names:

        for r in [1,2,3,4]:

            norm_mean_betas_df = norm_mean_betas_df.append(pd.DataFrame({'condition': [cond],
                                                             'run': [r],
                        'betas': avg_df[(avg_df['condition']==cond)&(avg_df['run']==r)&(avg_df['roi']==roi)]['betas'].values/attention_betas[attention_betas['run']==r]['betas'].values[0],
                        'roi': [roi]

        }))         


for roi in ROIs:
    
    sns.set(font_scale=1.3)
    sns.set_style("ticks")

    fig_dims = (20, 10)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.barplot(y='condition', x='betas', data = norm_mean_betas_df[norm_mean_betas_df['roi']==roi])

    ax = plt.gca()
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    ax.axes.tick_params(labelsize=16)
    sns.despine(offset=15)
    fig1 = plt.gcf()

    fig1.savefig(op.join(figures_pth,'betas_barplot_ROI-%s_rsq-%0.2f.svg'%(roi,rsq_threshold)), dpi=100,bbox_inches = 'tight')

