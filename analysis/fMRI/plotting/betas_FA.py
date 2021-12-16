import numpy as np
import os, sys
import os.path as op
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import cortex
import pandas as pd
import seaborn as sns

sys.path.insert(0,'..') # add parent folder to path
from utils import * #import script to use relevante functions

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
file_ext = '_cropped_{filt}_{stand}.npy'.format(filt = params['mri']['filtering']['type'],
                                                    stand = 'psc')

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
rsq_threshold = 0.2 #0.15 #params['plotting']['rsq_threshold']

# make rsq mask, to only show betas with some pRF rsq
rsq_mask = pRF_masked_rsq < rsq_threshold

runs = ['1','2','3','4'] if run_type == 'mean' else [run_type]

## put betas from all runs in DataFrame
# for easier manipulaiton and control

betas_df = pd.DataFrame(columns = ['regressor', 'run','miniblock','betas','vertex'])
    
for r in runs:
    #r = runs[0]

    # path to FA fits 
    fits_pth =  op.join(derivatives_dir,'FA_GLM_fit','sub-{sj}'.format(sj=sj), 
                        space, model_type,'run-{run}'.format(run=r))

    # get GLM estimates file
    estimates_filename = [op.join(fits_pth, val) for val in os.listdir(fits_pth) if val.endswith('_estimates.npz')]
    estimates = np.load(estimates_filename[0])

    # get regressors dataframe
    all_regressors = pd.read_csv(op.join(fits_pth,'all_regressors_info.csv'))

    # loop over regressors that are not cue
    for ind in all_regressors.loc[all_regressors['reg_name'].str.contains('cue')==False].index.values:

        num_vert = estimates['betas'][...,ind].shape[0]
        
        betas = estimates['betas'][...,ind].copy()
        betas[rsq_mask] = np.nan

        betas_df = betas_df.append(pd.DataFrame({'regressor': np.tile('{cond}_{feat}'.format(cond=all_regressors.iloc[ind]['reg_name'].split('_')[0],
                                                                 feat = all_regressors.iloc[ind]['condition_name']),
                                                   num_vert),
                             'run': np.tile(int(all_regressors.iloc[ind]['reg_name'].split('_')[-1].split('-')[-1]),num_vert),
                             'miniblock': np.tile(int(all_regressors.iloc[ind]['reg_name'].split('_')[1].split('-')[-1]),num_vert),
                             'betas': betas,
                             'vertex': np.arange(num_vert)}))


### now make new DF with average betas

regressor_names = betas_df['regressor'].unique()

mean_betas_df = betas_df.groupby(['regressor', 'vertex']).mean().reset_index()##.unstack()#.reset_index()#


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

for roi in ROIs:
    
    sns.set(font_scale=1.3)
    sns.set_style("ticks")

    fig_dims = (20, 10)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.barplot(y='regressor', x='betas', data = mean_betas_df.loc[mean_betas_df['vertex'].isin(roi_verts[roi])])

    ax = plt.gca()
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    ax.axes.tick_params(labelsize=16)
    sns.despine(offset=15)
    fig1 = plt.gcf()

    fig1.savefig(op.join(figures_pth,'betas_barplot_ROI-%s_rsq-%0.2f.svg'%(roi,rsq_threshold)), dpi=100,bbox_inches = 'tight')

