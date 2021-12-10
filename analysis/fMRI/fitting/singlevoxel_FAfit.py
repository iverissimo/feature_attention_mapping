################################################
#      Do FA fit on single voxel, 
#    by loading estimates, getting fit OR
#    by fitting the timeseries
#    saving plot of fit on timeseries
################################################

import numpy as np
import os, sys
import os.path as op
import yaml
from pathlib import Path
import glob

sys.path.insert(0,'..') # add parent folder to path
from utils import * #import script to use relevante functions

import pandas as pd
import numpy as np

# requires pfpy to be installed - preferably with python setup.py develop
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel

from popeye import utilities

from nistats.design_matrix import make_first_level_design_matrix
from nistats.reporting import plot_design_matrix

from joblib import Parallel, delayed

import datetime
from tqdm import tqdm

# load settings from yaml
with open(op.join(str(Path(os.getcwd()).parents[1]),'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

# define participant number, ROI (if the case) and vertex number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex: 01) '	
                    'as 1st argument in the command line!')	

elif len(sys.argv)<3:   
    raise NameError('Please add ROI name (ex: V1) or "None" if looking at vertex from no specific ROI  '	
                    'as 2nd argument in the command line!')	

elif len(sys.argv)<4:   
    raise NameError('Please vertex index number of that ROI (or from whole brain)'	
                    'as 3rd argument in the command line!'
                    '(can also be "max" or "min" to fit vertex of max or min RSQ)')	
elif len(sys.argv)<5:   
    raise NameError('fit vs load estimates'	
                    'as 4th argument in the command line!')	
elif len(sys.argv)<6:   
    raise NameError('add run number'	
                    'as 5th argument in the command line!')	 

else:
    sj = str(sys.argv[1]).zfill(3) #fill subject number with 0 in case user forgets	

    roi = str(sys.argv[2]) # ROI or 'None'

    if str(sys.argv[3]) != 'max' and str(sys.argv[3]) != 'min': # if we actually get a number for the vertex
    
        vertex = int(sys.argv[3]) # vertex number
    else:
        vertex = str(sys.argv[3]) 
        
    fit_now = True if str(sys.argv[4])=='fit' else False

    run = int(sys.argv[5])

if fit_now == True and vertex in ['min','max']:
    raise NameError('Cannot fit vertex, need to load pre-fitted estimates')

# set font type for plots globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# some settings
base_dir = params['general']['current_dir'] # which machine we run the data
acq = params['mri']['acq'] # if using standard files or nordic files
space = params['mri']['space'] # subject space
total_chunks = params['mri']['fitting']['pRF']['total_chunks'][space] # number of chunks that data was split in

TR = params['mri']['TR']

# type of pRF model to use, and run type (mean vs median)
model_type = params['mri']['fitting']['pRF']['fit_model']
run_type = params['mri']['fitting']['pRF']['run']

# define file extension that we want to use, 
# should include processing key words
file_ext = '_cropped_{filt}_{stand}.npy'.format(filt = params['mri']['filtering']['type'],
                                                    stand = 'psc')

# set paths
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
postfmriprep_dir = op.join(derivatives_dir,'post_fmriprep','sub-{sj}'.format(sj=sj),space,'processed')

FA_estimates_dir =  op.join(derivatives_dir,'FA_GLM_fit','sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run))

# output dir to save fit and plot
figures_pth = op.join(derivatives_dir,'plots','single_vertex','FAfit','sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run)) # path to save plots
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

if roi != 'None' and vertex not in ['max','min']:
    print('masking data for ROI %s'%roi)
    roi_ind = cortex.get_roi_verts(params['plotting']['pycortex_sub'],roi) # get indices for that ROI
    
else:
    roi_ind = {'None': np.array(range(data.shape[0]))}
    
data = data[roi_ind[roi]]

timeseries = data[vertex][np.newaxis,...]

# path to pRF fits 
fits_pth =  op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), space, 'iterative_{model}'.format(model=model_type),'run-{run}'.format(run=run_type))

## Load pRF estimates 
    
# path to combined estimates
pRF_estimates_pth = op.join(fits_pth,'combined')

# combined estimates filename
est_name = [x for _,x in enumerate(os.listdir(fits_pth)) if 'chunk-001' in x][0]
est_name = est_name.replace('chunk-001_of_{ch}'.format(ch=str(total_chunks).zfill(3)),'chunk-combined')

# total path to estimates path
pRF_estimates_combi = op.join(pRF_estimates_pth,est_name)

if op.isfile(pRF_estimates_combi): # if combined estimates exists

    print('loading %s'%pRF_estimates_combi)
    pRF_estimates = np.load(pRF_estimates_combi) # load it

else: # if not join chunks and save file
    if not op.exists(pRF_estimates_pth):
        os.makedirs(pRF_estimates_pth) 

    pRF_estimates = join_chunks(fits_pth, pRF_estimates_combi,
                        chunk_num = total_chunks, fit_model = 'it{model}'.format(model=model_type)) #'{model}'.format(model=model_type)))#
    
# save estimates in specific variables
xx = pRF_estimates['x'][roi_ind[roi]][vertex]
yy = pRF_estimates['y'][roi_ind[roi]][vertex]

size = pRF_estimates['size'][roi_ind[roi]][vertex]

beta = pRF_estimates['betas'][roi_ind[roi]][vertex]
baseline = pRF_estimates['baseline'][roi_ind[roi]][vertex]

if 'css' in model_type:
    ns = pRF_estimates['ns'][roi_ind[roi]][vertex]

rsq = pRF_estimates['r2'][roi_ind[roi]][vertex]

## load bar position for FA
## and make DM for run

# path to events csv, pickle etc
events_pth = glob.glob(op.join(derivatives_dir.replace('derivatives','sourcedata'),'sub-{sj}'.format(sj=sj), 'ses-*', 'func'))[0]

# get absolute path to pkl with bar positions for each run
bar_pos_files = [op.join(events_pth, h) for h in os.listdir(events_pth) if 'task-FA' in h and
                'run-{run}'.format(run=run) in h and h.endswith('_bar_positions.pkl')]
print('getting bar positions from %s'%bar_pos_files[0])

# load bar positions for run
bar_pos = pd.read_pickle(bar_pos_files[0])

# get absolute path to csv with general infos for each run
trial_info_files = [op.join(events_pth, h) for h in os.listdir(events_pth) if 'task-FA' in h and
                'run-{run}'.format(run=run) in h and h.endswith('_trial_info.csv')]

# load trial info dataframe
trial_info = pd.read_csv(trial_info_files[0])

### use 4x4 regressors in fit - 4 conditions x 4 miniblocks
unique_cond = params['mri']['fitting']['FA']['condition_keys']

all_regressors = pd.DataFrame(columns = ['reg_name', 'color','orientation','miniblock','run'])

for key in unique_cond.keys(): # for each condition
    
    for blk in range(params['feature']['mini_blocks']): # for each miniblock
        
        all_regressors = all_regressors.append(pd.DataFrame({'reg_name': '{cond}_mblk-{blk}_run-{run}'.format(cond=key,
                                                                                                             blk=blk,
                                                                                                             run=run),
                                                             'color': unique_cond[key]['color'],
                                                             'orientation': unique_cond[key]['orientation'],
                                                             'miniblock': blk,
                                                             'run': int(run)
                                                            }, index=[0]),ignore_index=True)

all_reg_predictions = [] # to append all regressor predictions

# make visual DM for each GLM regressor, and obtain prediction using pRF model
for reg in all_regressors['reg_name'].values:
    
    # filename for regressor dm
    DM_reg_filename = op.join(figures_pth,'DM_regressor-{reg}.npy'.format(reg=reg))
    
    # make array with spatial position of bar of interest 
    DM_cond = get_FA_bar_stim(DM_reg_filename, 
                        params, bar_pos, trial_info, attend_cond = all_regressors[all_regressors['reg_name']==reg].to_dict('r')[0], 
                        save_imgs = False, downsample = 0.1, crop = params['feature']['crop'] , 
                        crop_TR = params['feature']['crop_TR'], overwrite=True, save_DM=False)

    # make stimulus object, which takes an input design matrix and sets up its real-world dimensions
    prf_stim = PRFStimulus2D(screen_size_cm = params['monitor']['height'],
                             screen_distance_cm = params['monitor']['distance'],
                             design_matrix = DM_cond,
                             TR = TR)

    # get prediction
    if model_type == 'css':

        # define CSS model 
        css_model = CSS_Iso2DGaussianModel(stimulus = prf_stim,
                                     filter_predictions = True,
                                     filter_type = params['mri']['filtering']['type'],
                                     filter_params = {'highpass': params['mri']['filtering']['highpass'],
                                                     'add_mean': params['mri']['filtering']['add_mean'],
                                                     'window_length': params['mri']['filtering']['window_length'],
                                                     'polyorder': params['mri']['filtering']['polyorder']}
                                    )

        model_fit = css_model.return_prediction(xx,yy,
                                        size, beta,
                                        baseline, ns)

    else:
        # define gaussian model 
        gauss_model = Iso2DGaussianModel(stimulus = prf_stim,
                                     filter_predictions = True,
                                     filter_type = params['mri']['filtering']['type'],
                                     filter_params = {'highpass': params['mri']['filtering']['highpass'],
                                                     'add_mean': params['mri']['filtering']['add_mean'],
                                                     'window_length': params['mri']['filtering']['window_length'],
                                                     'polyorder': params['mri']['filtering']['polyorder']}
                                    )

        model_fit = gauss_model.return_prediction(xx,yy,
                                        size, beta,
                                        baseline)   

    # squeeze out single dimension that parallel creates
    prediction =  model_fit

    
    ## append predictions in array, to use for FA GLM DM
    all_reg_predictions.append(prediction[np.newaxis,...])

all_reg_predictions = np.vstack(all_reg_predictions)

## Make actual DM to be used in GLM fit (4 regressors + intercept)

DM_FA = np.zeros((all_reg_predictions.shape[1], all_reg_predictions.shape[-1], all_reg_predictions.shape[0]+1)) # shape of DM is (vox,time,reg)

# iterate over vertex/voxel
for i in range(all_reg_predictions.shape[1]):

    DM_FA[i,:,0] = np.repeat(1,all_reg_predictions.shape[-1]) # add intercept
    
    for w in range(all_reg_predictions.shape[0]): # add regressor (which will be the model from pRF estimates)
        DM_FA[i,:,w+1] = all_reg_predictions[w,i,:] 

### plot vertex DM for sanity check
plot_DM(DM_FA, 0, op.join(figures_pth,'DM_FA_vertex_%i.png'%vertex), names=['intercept']+list(all_regressors['reg_name'].values))

if fit_now:
    
    glm_outcome = fit_glm(timeseries[0], DM_FA[0])
    
    prediction = glm_outcome[0]
    r2 = glm_outcome[-2]

### MODEL FIGURE ###

# set figure name
fig_name = 'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_roi-{roi}_vertex-{vert}.png'.format(sj=sj,
                                                                                        acq=acq,
                                                                                        space=space,
                                                                                        run=run,
                                                                                        model=model_type,
                                                                                        roi=roi,
                                                                                        vert=vertex) 
if fit_now == False:
    fig_name = fig_name.replace('.png','_loaded.png')
    
#%matplotlib inline
# plot data with model
fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)

# plot data with model
time_sec = np.linspace(0,len(timeseries[0])*TR,num=len(timeseries[0])) # array with timepoints, in seconds
 
#plt.plot(time_sec, all_reg_predictions[0,...][roi_ind['V1']][1811], c='#0040ff',lw=3,label='prediction %s'%'ACAO',zorder=1)
#plt.plot(time_sec, all_reg_predictions[-1,...][roi_ind['V1']][1811], c='#002db3',lw=3,label='prediction %s'%'UCUO',zorder=1)
plt.plot(time_sec, prediction, c='#0040ff',lw=3,label='model R$^2$ = %.2f'%r2,zorder=1)
plt.plot(time_sec, timeseries[0],'k--',label='FA data')
axis.set_xlabel('Time (s)',fontsize=20, labelpad=20)
axis.set_ylabel('BOLD signal change (%)',fontsize=20, labelpad=10)
axis.set_xlim(0,len(prediction)*TR)
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
    
fig.savefig(op.join(figures_pth,fig_name))

### PLOT REGRESSORS TO CHECK ###

for key in params['mri']['fitting']['FA']['condition_keys'].keys():
    
    # set figure name
    fig_name = 'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_roi-{roi}_reg-{key}_vertex-{vert}.png'.format(sj=sj,
                                                                                            acq=acq,
                                                                                            space=space,
                                                                                            run=run,
                                                                                            model=model_type,
                                                                                            roi=roi,
                                                                                            key=key,
                                                                                            vert=vertex) 
    if fit_now == False:
        fig_name = fig_name.replace('.png','_loaded.png')


    ind_predict = np.array([i for i, val in enumerate(all_regressors['reg_name'].values) if key in val]) 

    # plot data with model
    fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)

    # plot regressors
    plt.plot(time_sec, all_reg_predictions[ind_predict[0]][0], 
             c='#0040ff',lw=3,label='prediction %s'%all_regressors['reg_name'][ind_predict[0]],zorder=1)
    plt.plot(time_sec, all_reg_predictions[ind_predict[1]][0], 
             c='#4272ff',lw=3,label='prediction %s'%all_regressors['reg_name'][ind_predict[1]],zorder=1)
    plt.plot(time_sec, all_reg_predictions[ind_predict[2]][0], 
             c='#8ca9ff',lw=3,label='prediction %s'%all_regressors['reg_name'][ind_predict[2]],zorder=1)
    plt.plot(time_sec, all_reg_predictions[ind_predict[3]][0], 
             c='#c2d1ff',lw=3,label='prediction %s'%all_regressors['reg_name'][ind_predict[3]],zorder=1)

    # plot data and bars
    plt.plot(time_sec, timeseries[0],'k--',label='FA data')
    axis.set_xlabel('Time (s)',fontsize=20, labelpad=20)
    axis.set_ylabel('BOLD signal change (%)',fontsize=20, labelpad=10)
    axis.set_xlim(0,len(prediction)*TR)
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
        
    fig.savefig(op.join(figures_pth,fig_name))