# cross validate FA runs

import numpy as np
import os, sys
import os.path as op
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed

from FAM_utils import mri as mri_utils

# load settings from yaml
with open(op.join(str(Path(os.getcwd()).parents[1]),'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')
  
elif len(sys.argv) < 3:
    raise NameError('Please add run to be fitted (ex: leave_1_out) '
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
out_pth = op.join(derivatives_dir,'CV_FA',
                      'sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run_type)) # path to save plots
if not os.path.exists(out_pth):
    os.makedirs(out_pth) 

## put betas from all runs in DataFrame
# for easier manipulaiton and control

runs = ['1','2','3','4'] 

betas_df = pd.DataFrame(columns = ['regressor', 'run','miniblock','betas','vertex'])
    
for r in runs:

    # path to FA fits 
    fits_pth =  op.join(derivatives_dir,'FA_GLM_fit','sub-{sj}'.format(sj=sj), 
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
    for ind in all_regressors.index.values:
        
        if ind == 0:
            # add intercept
            
            betas_df = betas_df.append(pd.DataFrame({'regressor': np.tile('intercept',num_vert),
                             'run': np.tile(int(all_regressors.iloc[ind]['reg_name'].split('_')[-1].split('-')[-1]),num_vert),
                             'miniblock': np.tile(np.nan,num_vert),
                             'betas': estimates['betas'][...,0],
                             'vertex': np.arange(num_vert)}))
            

        betas = betas_no_intercept[...,ind].copy()

        betas_df = betas_df.append(pd.DataFrame({'regressor': np.tile('{cond}_{feat}'.format(cond=all_regressors.iloc[ind]['reg_name'].split('_')[0],
                                                                 feat = all_regressors.iloc[ind]['condition_name']),
                                                   num_vert),
                             'run': np.tile(int(all_regressors.iloc[ind]['reg_name'].split('_')[-1].split('-')[-1]),num_vert),
                             'miniblock': np.tile(int(all_regressors.iloc[ind]['reg_name'].split('_')[1].split('-')[-1]),num_vert),
                             'betas': betas,
                             'vertex': np.arange(num_vert)}))

### now make new DF with average betas
# excluding the left out run

regressor_names = betas_df['regressor'].unique()
loo_run = run_type.split('_')[1]

new_betas_df = betas_df.loc[betas_df['run'].isin([int(val) for _,val in enumerate(runs) if val!= loo_run])]

mean_betas_df = new_betas_df.groupby(['regressor', 'vertex']).mean().reset_index()##.unstack()#.reset_index()#

## now we have to obtain prediction
# given these betas and left out run DM
# and calculate rsq

# load left out data
# list with absolute file name to be fitted
proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-FA' in h and
                 'acq-{acq}'.format(acq=acq) in h and 'run-{run}'.format(run=loo_run) in h and h.endswith(file_ext)]

# exception for sub 4, run 4 because nordic failed for FA
if sj=='004' and loo_run=='4':
    proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-FA' in h and
                 'acq-{acq}'.format(acq='standard') in h and 'run-{run}'.format(run=loo_run) in h and h.endswith(file_ext)]

## load functional data
file = proc_files[0]
data = np.load(file,allow_pickle=True) # will be (vertex, TR)
print('loading data for %s'%file)

# load regressor info for left out run

# path to FA fits 
loo_fits_pth =  op.join(derivatives_dir,'FA_GLM_fit','sub-{sj}'.format(sj=sj), 
                        space, model_type,'run-{run}'.format(run=loo_run))

loo_all_regressors = pd.read_csv(op.join(loo_fits_pth,'all_regressors_info.csv'))

## load DM of left out run

loo_DM_FA = np.load(op.join(loo_fits_pth,'DM_FA_run-{run}.npy'.format(run=loo_run)))

## make dict with betas for each 
## reg_name of left out run

mean_betas_dict = {}

for ind in loo_all_regressors.index.values:
    
    if ind==0:
        # add intercept
        mean_betas_dict['intercept'] = mean_betas_df.loc[mean_betas_df['regressor']=='intercept']['betas'].values
    
    regressor = '{cond}_{feat}'.format(cond=loo_all_regressors.iloc[ind]['reg_name'].split('_')[0],
                        feat = loo_all_regressors.iloc[ind]['condition_name'])
    
    mean_betas_dict[loo_all_regressors.iloc[ind]['reg_name']] = mean_betas_df.loc[mean_betas_df['regressor']==regressor]['betas'].values

## make it into array
mean_betas_arr = np.array(list(mean_betas_dict.values())).T

### now get CV rsq

CV_FA_outcome = Parallel(n_jobs=16)(delayed(mri_utils.CV_FA)(data[vert], loo_DM_FA[vert], mean_betas_arr[vert]) for vert in tqdm(range(data.shape[0])))

# save in folder
CV_estimates_filename = op.join(out_pth, 'CV_run-%s_estimates.npz'%run_type)

np.savez(CV_estimates_filename,
            prediction = np.array([CV_FA_outcome[i][0] for i in range(data.shape[0])]),
            cv_r2 = np.array([CV_FA_outcome[i][1] for i in range(data.shape[0])])
            )

prediction = np.array([CV_FA_outcome[i][0] for i in range(data.shape[0])])
cv_r2 = np.array([CV_FA_outcome[i][1] for i in range(data.shape[0])])

#### plot a vertex where cv rsq is max
# to check how it looks

vertex = np.where(cv_r2==np.nanmax(cv_r2))[0][0]

# set figure name
fig_name = 'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_vertex-{vert}.png'.format(sj=sj,
                                                                                        acq=acq,
                                                                                        space=space,
                                                                                        run=run_type,
                                                                                        model=model_type,
                                                                                        vert=vertex) 
loo_timeseries = data[vertex]
cv_prediction = prediction[vertex]
loo_cv_r2 = cv_r2[vertex]

#%matplotlib inline
# plot data with model
fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)

# plot data with model
time_sec = np.linspace(0,len(loo_timeseries)*TR,num=len(loo_timeseries)) # array with timepoints, in seconds
 
plt.plot(time_sec, cv_prediction, c='#0040ff',lw=3,label='model CV-R$^2$ = %.2f'%loo_cv_r2,zorder=1)
plt.plot(time_sec, loo_timeseries,'k--',label='FA LOO data')
axis.set_xlabel('Time (s)',fontsize=20, labelpad=20)
axis.set_ylabel('BOLD signal change (%)',fontsize=20, labelpad=10)
axis.set_xlim(0,len(cv_prediction)*TR)
axis.legend(loc='upper left',fontsize=10)  # doing this to guarantee that legend is how I want it  

# times where bar is on screen [1st on, last on, 1st on, last on, etc] 
bar_onset = np.array([27,98,126,197,225,296,324,395])*TR

if params['feature']['crop']:
    bar_onset = bar_onset - params['feature']['crop_TR']

bar_directions = [val for _,val in enumerate(params['feature']['bar_pass_direction']) if 'empty' not in val and 'cue' not in val]
# plot axis vertical bar on background to indicate stimulus display time
ax_count = 0
for h in range(len(bar_directions)):
    
    plt.axvspan(bar_onset[ax_count], bar_onset[ax_count+1]+TR, facecolor='#0040ff', alpha=0.1)
    
    ax_count += 2
    
fig.savefig(op.join(out_pth,fig_name))
