
## make contrasts for FA task

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

from tqdm import tqdm

sys.path.insert(0,'..') # add parent folder to path
from utils import * #import script to use relevante functions

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


# set paths
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
postfmriprep_dir = op.join(derivatives_dir,'post_fmriprep','sub-{sj}'.format(sj=sj),space,'processed')

# output dir to save contrasts and plot
output_dir =  op.join(derivatives_dir,'FA_GLM_fit','sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run))

# output dir to save fit and plot
figures_pth = op.join(derivatives_dir,'plots','FA_GLM_contrasts',
                      'sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

# list with absolute file name to be fitted
proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-FA' in h and
                 'acq-{acq}'.format(acq=acq) in h and 'run-{run}'.format(run=run) in h and h.endswith(file_ext)]

# exception for sub 4, run 4 because nordic failed for FA
if sj=='004' and run=='4':
    proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-FA' in h and
                 'acq-{acq}'.format(acq='standard') in h and 'run-{run}'.format(run=run) in h and h.endswith(file_ext)]

## load functional data
file = proc_files[0]
data = np.load(file,allow_pickle=True) # will be (vertex, TR)

# masked rsq of pRF mean run (within screen boundaries etc)
pRF_masked_rsq = np.load(op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), space, 
                  'iterative_{model}'.format(model=model_type),'run-mean','combined','masked_rsq.npy'))

# get FA GLM estimates file
estimates_filename = [op.join(output_dir, val) for val in os.listdir(output_dir) if val.endswith('_estimates.npz')]
estimates = np.load(estimates_filename[0])
print('loading estimates from %s'%estimates)

# Load DM
DM_FA = np.load(op.join(output_dir,'DM_FA_run-{run}.npy'.format(run=run)))

### all 4 regressors used in fit

all_regressors = {'ACAO': {'color': True, 'orientation': True},
                  'ACUO': {'color': True, 'orientation': False},
                  'UCAO': {'color': False, 'orientation': True},
                  'UCUO': {'color': False, 'orientation': False}
                    }
conditions = list(all_regressors.keys())

# load betas
betas = estimates['betas']

# now make simple contrasts
print('Computing simple contrasts')

stats_all = {} # save all computed stats, don't need to load again

## columns names, in order, used im GLM fit
dm_columns = np.array(['intercept'] + conditions)

# set contrast arrays
contrast_cond = {'ACAO_vs_others': set_contrast(dm_columns, [['ACAO'],['ACUO', 'UCAO', 'UCUO']], contrast_val = [1,-len(conditions)/(len(conditions)-1)], num_cond = 2),
                 'attended_vs_unattended': np.array([0,1,-1/4,-1/4,-1/2]),
                 'ACAO_vs_UCUO': set_contrast(dm_columns, [['ACAO'],['UCUO']], contrast_val = [1,-1], num_cond = 2),
                 'AC_vs_UC': set_contrast(dm_columns, [['ACAO','ACUO'],['UCAO', 'UCUO']], contrast_val = [1,-1], num_cond = 2),
                 'AO_vs_UO': set_contrast(dm_columns, [['ACAO','UCAO'],['ACUO', 'UCUO']], contrast_val = [1,-1], num_cond = 2),
                 'interaction': set_contrast(dm_columns, [['ACAO','UCUO'],['ACUO', 'UCAO']], contrast_val = [1,-1], num_cond = 2)
                }

## compute stats and make plots to visualize, in loop
for contrast_val in list(contrast_cond.keys()):
    
    ## save estimates in dir 
    stats_filename = os.path.join(output_dir,'glm_stats_%s_contrast.npz' %(contrast_val))

    #if not op.isfile(stats_filename): # if doesn't exist already
    print('making %s'%stats_filename)

    # compute contrast-related statistics
    soma_stats = Parallel(n_jobs=16)(delayed(compute_stats)(data[w], DM_FA[w],
                                                            contrast_cond[contrast_val],
                                                            betas[w]) for w in tqdm(range(data.shape[0])))
    soma_stats = np.vstack(soma_stats) # t_val,p_val,zscore

    np.savez(stats_filename,
            t_val = soma_stats[..., 0],
            p_val = soma_stats[..., 1],
            zscore = soma_stats[..., 2])

    ## mask t-values, according to pRF rsq
    masked_tval = soma_stats[..., 0].copy()
    masked_tval[np.isnan(pRF_masked_rsq)] = np.nan # mask for points where pRF in screen boundaries, etc
    masked_tval[pRF_masked_rsq < rsq_threshold] = np.nan # mask for points where pRF rsq > threshold (0.1)
    
    ## make flatmap
    images = {}
    images['t_stat'] = cortex.Vertex(masked_tval,
                                pysub,
                                vmin = -3, vmax = 3,
                                cmap='BuBkRd')
    cortex.quickshow(images['t_stat'],with_curvature=True,with_sulci=True)

    filename = op.join(figures_pth,
                       op.split(file)[-1].replace('.npy','_contrast-{c}_tval.png'.format(c=contrast_val)))
    print('saving %s' %filename)
    _ = cortex.quickflat.make_png(filename, images['t_stat'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

    ## make per ROI boxplot, to check distribution of values
    # threshold plot for p-value of a certain alpha level
    alpha = 0.05

    for idx,rois_ks in enumerate(ROIs): 

        # mask estimates
        print('masking t-val for ROI %s'%rois_ks)

        roi_tval = masked_tval.copy()
        #roi_tval[soma_stats[..., 1] >= alpha] = np.nan
        roi_tval = roi_tval[roi_verts[rois_ks]]

        if idx == 0:
            df_tval = pd.DataFrame({'roi': np.tile(rois_ks,len(roi_tval)),'t_val': roi_tval})
        else:
            df_tval = df_tval.append(pd.DataFrame({'roi': np.tile(rois_ks,len(roi_tval)),'t_val': roi_tval}),
                                                       ignore_index = True)
            
    fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

    v1 = sns.boxplot(data = df_tval, x = 'roi', y = 't_val', 
                    palette = color_codes, linewidth=1.8) # palette ='Set3',linewidth=1.8)

    v1.set(xlabel=None)
    v1.set(ylabel=None)
    plt.margins(y=0.025)
    #sns.swarmplot(x='ecc', y='cs', data=crwd_df4plot,color=".25",alpha=0.5)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)

    plt.xlabel('ROI',fontsize = 20,labelpad=18)
    plt.ylabel('t-val',fontsize = 20,labelpad=18)
    plt.ylim(1,5)
    fig.savefig(filename.replace('.png','-ROI.png'), dpi=100)




