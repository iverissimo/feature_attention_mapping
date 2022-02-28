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

from FAM_utils import mri as mri_utils

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cortex

# requires pfpy to be installed - preferably with python setup.py develop
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D

import seaborn as sns

from lmfit import Parameters

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
fit_hrf = params['mri']['fitting']['pRF']['fit_hrf'] 

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

FA_estimates_dir =  op.join(derivatives_dir,'FA_GLM_fit','sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run))

# output dir to save fit and plot
figures_pth = op.join(derivatives_dir,'plots','single_vertex','FAfit','sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

# list with absolute file name to be fitted
proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-FA' in h and
                 'acq-{acq}'.format(acq=acq) in h and 'run-{run}'.format(run=run) in h and h.endswith(file_ext)]

## load functional data
file = proc_files[0]
if len(proc_files)>1:
    raise ValueError('%s files found to fit, unsure of which to use'%len(proc_files))
else:
    print('Fitting %s'%file)
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
est_name = [x for _,x in enumerate(os.listdir(fits_pth)) if 'chunk-001' in x]
if len(est_name)>1:
    raise ValueError('%s files found as pRF estimates of same chuck, unsure of which to use'%len(est_name))
else:
    est_name = est_name[0].replace('chunk-001_of_{ch}'.format(ch=str(total_chunks).zfill(3)),'chunk-combined')

# total path to estimates path
pRF_estimates_combi = op.join(pRF_estimates_pth,est_name)

if op.isfile(pRF_estimates_combi): # if combined estimates exists

    print('loading %s'%pRF_estimates_combi)
    pRF_estimates = np.load(pRF_estimates_combi) # load it

else: # if not join chunks and save file
    if not op.exists(pRF_estimates_pth):
        os.makedirs(pRF_estimates_pth) 

    pRF_estimates = mri_utils.join_chunks(fits_pth, pRF_estimates_combi, fit_hrf = params['mri']['fitting']['pRF']['fit_hrf'],
                        chunk_num = total_chunks, fit_model = 'it{model}'.format(model=model_type)) #'{model}'.format(model=model_type)))#
    
# define design matrix for pRF task
visual_dm = mri_utils.make_pRF_DM(op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), 'DMprf.npy'), params, 
                    save_imgs=False, downsample=0.1, crop = params['prf']['crop'] , crop_TR = params['prf']['crop_TR'], overwrite=False)

# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm = params['monitor']['height'],
                        screen_distance_cm = params['monitor']['distance'],
                        design_matrix = visual_dm,
                        TR = TR)

# get the ecc limits (in dva)
# to mask estimates
x_ecc_lim, y_ecc_lim = mri_utils.get_ecc_limits(visual_dm,params,screen_size_deg = [prf_stim.screen_size_degrees,prf_stim.screen_size_degrees])

# mask estimates, to be within screen boundaries
print('masking estimates')
masked_pRF_estimates = mri_utils.mask_estimates(pRF_estimates, fit_model = model_type,
                                        x_ecc_lim = x_ecc_lim, y_ecc_lim = y_ecc_lim)

# save estimates in specific variables
# of parameters object
pRF_pars = Parameters()

pRF_pars.add('pRF_x', value = masked_pRF_estimates['x'][roi_ind[roi]][vertex])
pRF_pars.add('pRF_y', value = masked_pRF_estimates['y'][roi_ind[roi]][vertex])
pRF_pars.add('pRF_size', value = masked_pRF_estimates['size'][roi_ind[roi]][vertex])
pRF_pars.add('pRF_beta', value = masked_pRF_estimates['beta'][roi_ind[roi]][vertex])
pRF_pars.add('pRF_baseline', value = masked_pRF_estimates['baseline'][roi_ind[roi]][vertex])
if 'css' in model_type:
    pRF_pars.add('pRF_n', value = masked_pRF_estimates['ns'][roi_ind[roi]][vertex])

rsq = masked_pRF_estimates['rsq'][roi_ind[roi]][vertex]

if np.isnan(rsq) or rsq <=0.05:
    raise ValueError('pRF rsq is %s, not fitting vertice'%str(rsq))
else:
    print('pRF rsq for vertice is %.2f'%rsq)

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
        
        # name of attended condition in miniblock
        attended_condition = bar_pos.loc[(bar_pos['mini_block']==blk)&(bar_pos['attend_condition']==1)]['condition'].values[0]
        
        all_regressors = all_regressors.append(pd.DataFrame({'reg_name': '{cond}_mblk-{blk}_run-{run}'.format(cond=key,
                                                                                                             blk=blk,
                                                                                                             run=run),
                                                             'color': unique_cond[key]['color'],
                                                             'orientation': unique_cond[key]['orientation'],
                                                             'condition_name': mri_utils.get_cond_name(attended_condition,key),
                                                             'miniblock': blk,
                                                             'run': int(run)
                                                            }, index=[0]),ignore_index=True)

all_reg_predictions = [] # to append all regressor predictions

# task sampling rate might be different from trial
# so stimulus object TR needs to reflect that
FA_sampling_rate = TR if params['feature']['task_rate']=='TR' else params['feature']['task_rate']

# set oversampling factor
osf = 10

# create upsampled hrf
if fit_hrf: # use fitted hrf params
    hrf_params = np.ones((3, 1))
    hrf_params[1] = pRF_estimates['hrf_derivative'][roi_ind[roi]][vertex]
    hrf_params[2] = pRF_estimates['hrf_dispersion'][roi_ind[roi]][vertex]

    hrf_oversampled = mri_utils.create_hrf(hrf_params = hrf_params, TR = TR, osf = osf)
else:
    hrf_oversampled = mri_utils.create_hrf(TR=TR, osf=osf)
    hrf_params = np.ones((3, 1))
    hrf_params[2] = 0   

# make visual DM for each GLM regressor, and obtain prediction using pRF model
for reg in all_regressors['reg_name'].values:
    
    # filename for regressor dm
    DM_reg_filename = op.join(figures_pth,'DM_regressor-{reg}.npy'.format(reg=reg))
    
    # make array with spatial position of bar of interest 
    DM_cond = mri_utils.get_FA_bar_stim(DM_reg_filename, 
                        params, bar_pos, trial_info, attend_cond = all_regressors[all_regressors['reg_name']==reg].to_dict('r')[0], 
                        save_imgs = False, downsample = 0.1, oversampling_time = osf, 
                        stim_dur_seconds = params['feature']['bars_phase_dur'], 
                        crop = True, crop_unit = 'sec', 
                        crop_TR = params['feature']['crop_TR'],
                        shift_TRs = True, TR = TR, shift_TR_num = 1.5,
                        overwrite = True, save_DM = False)

    prediction = mri_utils.get_FA_regressor(DM_cond, params, pRF_pars, 
                                            pRFmodel = model_type, TR = TR, hrf_params = hrf_params, 
                                            oversampling_time = osf)
    
    ## append predictions in array, to use for FA GLM DM
    all_reg_predictions.append(prediction[np.newaxis,...][np.newaxis,...])

all_reg_predictions = np.vstack(all_reg_predictions)

### make cue regressors

# array with cue regressors
cue_regs = np.zeros((params['feature']['mini_blocks'],all_reg_predictions.shape[1],all_reg_predictions.shape[-1]))
# array with cue regressors - UPSAMPLED
cue_regs_upsampled = np.zeros((params['feature']['mini_blocks'], all_reg_predictions.shape[1], 
                               len(trial_info['trial_num'].values)*osf))

for blk in range(params['feature']['mini_blocks']): # for each miniblock

    cue_regressor = mri_utils.get_cue_regressor(params, trial_info, hrf_params = hrf_params, 
                                                cues = [blk], TR = TR, oversampling_time = osf, 
                                                baseline = pRF_pars['pRF_baseline'].value,
                                                crop_unit = 'sec', crop = True, crop_TR = params['feature']['crop_TR'], 
                                                shift_TRs = True, shift_TR_num = 1.5)

    cue_regs[blk] = cue_regressor.copy()[np.newaxis,...]
    
    ## also update regressors info to know name and order of regressors
    # basically including cues
    all_regressors = all_regressors.append(pd.DataFrame({'reg_name': 'cue_mblk-{blk}_run-{run}'.format(blk=blk,
                                                                                  run=run),
                                                             'color': np.nan,
                                                             'orientation': np.nan,
                                                             'condition_name': trial_info.loc[trial_info['trial_type']=='cue_%i'%blk]['attend_condition'].values[0],
                                                             'miniblock': blk,
                                                             'run': int(run)
                                                            }, index=[0]),ignore_index=True)
    

## Make actual DM to be used in GLM fit (4 regressors + intercept)

# number of regressors
num_regs = all_reg_predictions.shape[0] + cue_regs.shape[0] + 1 # conditions, cues, and intercept

DM_FA = np.zeros((all_reg_predictions.shape[1], all_reg_predictions.shape[-1], num_regs)) # shape of DM is (vox,time,reg)

# iterate over vertex/voxel
for i in range(all_reg_predictions.shape[1]):

    DM_FA[i,:,0] = np.repeat(1,all_reg_predictions.shape[-1]) # add intercept
    
    for w in range(all_reg_predictions.shape[0]): # add regressor (which will be the model from pRF estimates)
        DM_FA[i,:,w+1] = all_reg_predictions[w,i,:] 

    for k in range(cue_regs.shape[0]): # add cues
        DM_FA[i,:,w+2+k] = cue_regs[k,i,:]

### plot vertex DM for sanity check
mri_utils.plot_DM(DM_FA, 0, op.join(figures_pth,'DM_FA_vertex_%i.png'%vertex), names=['intercept']+list(all_regressors['reg_name'].values))

if fit_now:
    
    glm_outcome = mri_utils.fit_glm(timeseries[0], DM_FA[0])
    
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
#axis.set_ylim(-3,3)  

# times where bar is on screen [1st on, last on, 1st on, last on, etc] 
bar_onset = np.array([27,98,126,197,225,296,324,395])#/TR

if params['feature']['crop']:
    bar_onset = bar_onset - params['feature']['crop_TR']*TR - TR

bar_directions = [val for _,val in enumerate(params['feature']['bar_pass_direction']) if 'empty' not in val and 'cue' not in val]
# plot axis vertical bar on background to indicate stimulus display time
ax_count = 0
for h in range(len(bar_directions)):
    
    plt.axvspan(bar_onset[ax_count], bar_onset[ax_count+1]+TR, facecolor='#0040ff', alpha=0.1)
    
    ax_count += 2
    
fig.savefig(op.join(figures_pth,fig_name))

### PLOT BETAS for each regressor ####

fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)

sns.barplot(y="reg_name", x="betas", 
            data=pd.DataFrame({'reg_name': all_regressors['reg_name'].values,
                                'betas': glm_outcome[1][1:]}))

fig.savefig(op.join(figures_pth,fig_name.replace('.png','_betas.png')))

### PLOT REGRESSORS TO CHECK ###

for key in np.concatenate((list(params['mri']['fitting']['FA']['condition_keys'].keys()),['cue'])):
    
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
    
    if key != 'cue':
        # plot regressors
        plt.plot(time_sec, all_reg_predictions[ind_predict[0]][0], 
                c='#0040ff',lw=3,label='prediction %s'%all_regressors['reg_name'][ind_predict[0]],zorder=1)
        plt.plot(time_sec, all_reg_predictions[ind_predict[1]][0], 
                c='#4272ff',lw=3,label='prediction %s'%all_regressors['reg_name'][ind_predict[1]],zorder=1)
        plt.plot(time_sec, all_reg_predictions[ind_predict[2]][0], 
                c='#8ca9ff',lw=3,label='prediction %s'%all_regressors['reg_name'][ind_predict[2]],zorder=1)
        plt.plot(time_sec, all_reg_predictions[ind_predict[3]][0], 
                c='#c2d1ff',lw=3,label='prediction %s'%all_regressors['reg_name'][ind_predict[3]],zorder=1)
    else:
        # plot regressors
        plt.plot(time_sec, cue_regs[0][0], 
                c='#0040ff',lw=3,label='prediction %s'%all_regressors['reg_name'][ind_predict[0]],zorder=1)
        plt.plot(time_sec, cue_regs[1][0], 
                c='#4272ff',lw=3,label='prediction %s'%all_regressors['reg_name'][ind_predict[1]],zorder=1)
        plt.plot(time_sec, cue_regs[2][0], 
                c='#8ca9ff',lw=3,label='prediction %s'%all_regressors['reg_name'][ind_predict[2]],zorder=1)
        plt.plot(time_sec, cue_regs[3][0], 
                c='#c2d1ff',lw=3,label='prediction %s'%all_regressors['reg_name'][ind_predict[3]],zorder=1)

    # plot data and bars
    plt.plot(time_sec, timeseries[0],'k--',label='FA data')
    axis.set_xlabel('Time (s)',fontsize=20, labelpad=20)
    axis.set_ylabel('BOLD signal change (%)',fontsize=20, labelpad=10)
    axis.set_xlim(0,len(prediction)*TR)
    axis.legend(loc='upper left',fontsize=10)  # doing this to guarantee that legend is how I want it
    #axis.set_ylim(-3,3)  

    bar_directions = [val for _,val in enumerate(params['feature']['bar_pass_direction']) if 'empty' not in val and 'cue' not in val]
    # plot axis vertical bar on background to indicate stimulus display time
    ax_count = 0
    for h in range(len(bar_directions)):

        plt.axvspan(bar_onset[ax_count], bar_onset[ax_count+1]+TR, facecolor='#0040ff', alpha=0.1)

        ax_count += 2
        
    fig.savefig(op.join(figures_pth,fig_name))