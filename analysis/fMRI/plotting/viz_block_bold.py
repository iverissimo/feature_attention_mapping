
import numpy as np
import os, sys
import os.path as op
import yaml
from pathlib import Path
import glob

from FAM_utils import mri as mri_utils

import pandas as pd
import numpy as np


import cortex

import matplotlib.pyplot as plt

# inserting path to fitting, to get feature model objects
# should reorganize folder in future, to avoid this
sys.path.insert(1, op.join(str(Path(os.getcwd()).parents[0]), 'fitting'))
from feature_model import FA_GainModel

from lmfit import Parameters, minimize

from joblib import Parallel, delayed
from tqdm import tqdm

# load settings from yaml
with open(op.join(str(Path(os.getcwd()).parents[1]),'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')
else:
    # fill subject number 
    sj = str(sys.argv[1]).zfill(3)

## some settings
base_dir = params['general']['current_dir'] # which machine we run the data
acq = params['mri']['acq'] # if using standard files or nordic files
space = params['mri']['space'] # subject space

mask_prf = True # if we're masking pRFs
TR = params['mri']['TR']

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

## set paths
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
postfmriprep_dir = op.join(derivatives_dir, 'post_fmriprep',
                           'sub-{sj}'.format(sj=sj), space,'processed')

source_dir = glob.glob(op.join(params['mri']['paths'][base_dir]['root'], 'sourcedata', 
                               'sub-{sj}'.format(sj=sj), 'ses-*', 'func'))[0] 

# output dir to save plots etc
figures_pth = op.join(derivatives_dir, 'block_nuisance','sub-{sj}'.format(sj=sj), space)
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

## list with absolute file name 
proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-FA' in h and
                 'acq-{acq}'.format(acq=acq) in h and h.endswith(file_ext)]

## load data from all runs
all_data = np.stack((np.load(file,allow_pickle=True) for file in proc_files), axis = 0)

# average accross runs
all_data = np.mean(all_data, axis = 0)

## make movie with average activity over time

# first get events at each timepoint

all_evs = np.array([])
for ev in params['feature']['bar_pass_direction']:
    
    if 'empty' in ev:
        all_evs = np.concatenate((all_evs, np.tile(ev, params['feature']['empty_TR'])))
    elif 'cue' in ev:
        all_evs = np.concatenate((all_evs, np.tile(ev, params['feature']['cue_TR'])))
    elif 'mini_block' in ev:
        all_evs = np.concatenate((all_evs, np.tile(ev, np.prod(params['feature']['num_bar_position'])*2)))
        
# times where bar is on screen [1st on per miniblock]
bar_onset = np.array([i for i, name in enumerate(all_evs) if 'mini_block' in name and all_evs[i-1]=='empty'])
# times where cue is on screen [1st time point]
cue_onset = np.array([i for i, name in enumerate(all_evs) if 'cue' in name and all_evs[i-1]=='empty'])

# combined - 0 is nothing on screen, 1 is something there
stim_on_bool = np.array([1 if 'cue' in name or 'mini_block' in name else 0 for _, name in enumerate(all_evs) ])

osf = 10
if params['feature']['crop']:
    bar_onset = bar_onset - params['feature']['crop_TR']*TR - TR*1.5
    cue_onset = cue_onset - params['feature']['crop_TR']*TR - TR*1.5
    
    ## resample stim_on array
    tmp_arr = np.repeat(stim_on_bool, osf)#[int(params['feature']['crop_TR']*TR*osf):]
    tmp_arr[:-int(TR*1.5*osf)] = np.repeat(stim_on_bool, osf)[int(TR*1.5*osf):]
    stim_on_bool = tmp_arr.copy()[int(params['feature']['crop_TR']*TR*osf):]
    
    
    stim_on_bool = mri_utils.resample_arr(stim_on_bool, osf = osf, final_sf = TR)

# new figure path
movie_pth = op.join(figures_pth, 'movie')
if not os.path.exists(movie_pth):
    os.makedirs(movie_pth) 
    
# get pycortex sub
pysub = params['plotting']['pycortex_sub'] 

# make movie
movie_name = op.join(movie_pth,'flatmap_space-{space}_type-BOLD_avg_runs_movie.mp4'.format(space=pysub))

if not op.isfile(movie_name):
    
    ## loop over TRs
    for t in range(stim_on_bool.shape[0]):

        # set figure grid 
        full_fig = plt.figure(constrained_layout = True, figsize = (12,8))
        gs = full_fig.add_gridspec(3, 4)

        ## set axis
        dm_ax = full_fig.add_subplot(gs[:1,1:3])
        flatmap_ax = full_fig.add_subplot(gs[1:,:])

        # set flatmap
        flatmap = cortex.Vertex(all_data[...,t], 
                                        pysub,
                                        vmin = -4, vmax = 4,
                                        cmap='BuBkRd')
        cortex.quickshow(flatmap, 
                         with_colorbar = True, with_curvature = True, with_sulci = True,
                         with_labels = False, fig = flatmap_ax)

        flatmap_ax.set_xticks([])
        flatmap_ax.set_yticks([])

        # set dm timecourse
        dm_ax.plot(stim_on_bool)
        dm_ax.axvline(t, color='red', linestyle='solid', lw=1)
        dm_ax.set_yticks([])

        filename = op.join(movie_pth, 'flatmap_space-{space}_type-BOLD_avg_runs_visual_TR-{time}.png'.format(space=pysub,
                                                                                                                time = str(t).zfill(3)))
        print('saving %s' %filename)
        full_fig.savefig(filename)


    ## save as video
    img_name = filename.replace('_TR-%s.png'%str(t).zfill(3),'_TR-%3d.png')
    os.system("ffmpeg -r 6 -start_number 0 -i %s -vcodec mpeg4 -y %s"%(img_name, movie_name)) 
        
else:
    print('movie already exists as %s'%movie_name)

## get indices where miniblock starts and ends,
# in TR!!
stim_ind = np.where(stim_on_bool>=.5)[0]
miniblk_start_ind = []
miniblk_end_ind = []

for i, val in enumerate(stim_ind):
    if i>0: 
        if val - stim_ind[i-1] > 1:
            miniblk_start_ind.append(val)
            
            if stim_on_bool[stim_ind[i-1]+1]<1:
                miniblk_end_ind.append(stim_ind[i-1])

# remove cue start indices
miniblk_start_ind = np.array(miniblk_start_ind[::2])   
miniblk_end_ind = np.concatenate((miniblk_end_ind[1::2], np.array([stim_ind[-1]])))

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

# load masked pRF rsq, to select vertices that have 
# decent rsq

pRF_rsq = np.load(op.join(derivatives_dir, 'FA_gain', 'sub-{sj}'.format(sj=sj), space, 
                          params['mri']['fitting']['pRF']['fit_model'], 'masked_pRF_rsq.npy'))

# average timecourse accross ROI
avg_roi = {} #empty dictionary  
rsq_threshold = 0.12

for _,val in enumerate(ROIs):
    
    ind = np.array([vert for vert in roi_verts[val] if not np.isnan(pRF_rsq[vert]) and pRF_rsq[vert]>rsq_threshold])
    
    avg_roi[val] = np.mean(all_data[ind], axis=0)

# plot data with model
fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)
 
time_sec = np.linspace(0,len(all_data[0])*TR,num=len(all_data[0])) # array with timepoints, in seconds

for _,key in enumerate(ROIs):
    plt.plot(time_sec, avg_roi[key], linewidth = 2.5, label = '%s'%key, color = color_codes[key])


axis.set_xlabel('Time (s)',fontsize=20, labelpad=20)
axis.set_ylabel('BOLD signal change (%)',fontsize=20, labelpad=10)
axis.legend(loc='upper left',fontsize=10)  # doing this to guarantee that legend is how I want it 

# plot axis vertical bar on background to indicate stimulus display time
# and cue - in seconds!
for ax_count in range(params['feature']['mini_blocks']):
    
    plt.axvspan(bar_onset[ax_count], bar_onset[ax_count]+np.prod(params['feature']['num_bar_position'])*2, facecolor='#0040ff', alpha=0.1)
    
    plt.axvspan(cue_onset[ax_count], cue_onset[ax_count]+params['feature']['cue_TR'], facecolor='grey', alpha=0.1)
    
fig.savefig(op.join(figures_pth, 'average_bold_accross_runs_rois.png'))

## now get average timecourse for miniblock

avg_miniblk =  {} #empty dictionary 
interv = 3 # average from begining of miniblk to end, +/- 5 sec

for _,val in enumerate(ROIs):
    avg_miniblk[val] = np.mean(np.stack((avg_roi[val][miniblk_start_ind[i]-interv:miniblk_end_ind[i]+interv] for i in range(len(miniblk_start_ind))), axis = 0), axis = 0)

## plot timecourse of each miniblock
# aligned to same timepoint
fig, axis = plt.subplots(len(ROIs), 1, figsize=(6,24), dpi=100)

## plot for each ROI

for i,key in enumerate(ROIs):
    
    axis[i].plot(avg_roi[key][miniblk_start_ind[0]-interv:miniblk_end_ind[0]+interv])
    axis[i].plot(avg_roi[key][miniblk_start_ind[1]-interv:miniblk_end_ind[1]+interv])
    axis[i].plot(avg_roi[key][miniblk_start_ind[2]-interv:miniblk_end_ind[2]+interv])
    axis[i].plot(avg_roi[key][miniblk_start_ind[3]-interv:miniblk_end_ind[3]+interv])

    axis[i].plot(avg_miniblk[key], linewidth = 4, c='black')

    axis[i].axvspan(interv, miniblk_end_ind[0]-miniblk_start_ind[0]+interv, facecolor='#0040ff', alpha=0.1)
    axis[i].set_title(key, fontweight = "bold")
    
    if i<len(ROIs)-1:
        axis[i].set_xticks([])
axis[i].set_xlabel('Time (TR)',fontsize=20, labelpad=20)

fig.savefig(op.join(figures_pth, 'average_bold_accross_miniblocks_TR.png'))


## average across rois and miniblocks
## and model nuisance

avg_minblk_tc =  np.mean(np.stack((avg_miniblk[val] for val in ROIs), axis = 0), axis = 0)

## make nuisance regressor
# for the miniblock

pars = Parameters()
pars.add('duration', value = 0, min = 0, max = 5, vary = True, brute_step = .1) # duration in TR

## minimize residuals
out = minimize(mri_utils.make_nuisance_regressor, pars, #args = [avg_minblk_tc],
               kws={'timecourse': avg_minblk_tc, 'onsets': [interv], 
                    'hrf': mri_utils.create_hrf(hrf_params=[1.0, 1.0, 0.0], TR = TR, osf = 10)[0]}, 
               method = 'brute')

# save nuisance regressor duration
# (in TR!)
reg_dur_TR = out.params.valuesdict()['duration']
pars['duration'].value = reg_dur_TR

reg = mri_utils.make_nuisance_regressor(pars, timecourse = avg_minblk_tc, onsets = [interv], 
                    hrf = mri_utils.create_hrf(hrf_params=[1.0, 1.0, 0.0], TR = TR, osf = 10)[0],
                       fit = False)

## make plot to check
fig, axis = plt.subplots(1, figsize=(12,6), dpi=100)
plt.plot(avg_minblk_tc)
plt.plot(reg, linewidth = 4, c='black')
plt.axvspan(interv, miniblk_end_ind[0]-miniblk_start_ind[0]+interv, facecolor='#0040ff', alpha=0.1)

plt.xlabel('Time (TR)',fontsize=20, labelpad=20)

fig.savefig(op.join(figures_pth, 'nuisance_regressor_miniblk.png'))

# make nuisance regressor for whole brain
# and save in output dir

# path to pRF fits 
prf_fits_pth =  op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), space, 
                        'iterative_{model}'.format(model = params['mri']['fitting']['pRF']['fit_model']),
                        'run-{run}'.format(run = params['mri']['fitting']['pRF']['run']))

# load them into numpy dict
### define model
fa_model = FA_GainModel(params)
pRF_estimates = fa_model.get_pRF_estimates(prf_fits_pth, params['mri']['fitting']['pRF']['total_chunks'][space])

# create upsampled hrf
hrf_params = np.ones((3, fa_model.pRF_estimates['r2'].shape[0]))

if fa_model.fit_hrf: # use fitted hrf params
    hrf_params[1] = fa_model.pRF_estimates['hrf_derivative']
    hrf_params[2] = fa_model.pRF_estimates['hrf_dispersion']
else:
    hrf_params[2] = 0

# get indices that are relevant to create regressor
mask_ind = np.array([ind for ind,val in enumerate(pRF_rsq) if val > rsq_threshold])


all_regs = np.array(Parallel(n_jobs=16)(delayed(mri_utils.make_nuisance_regressor)(pars, 
                                                                    timecourse = all_data[vert], 
                                                                    onsets = miniblk_start_ind, 
                                                                    hrf = mri_utils.create_hrf(hrf_params = hrf_params[..., vert], TR = TR, osf = 10)[0],
                                                                    fit = False)
                                        for _,vert in enumerate(tqdm(mask_ind)))) 


## save in the same shape of data 
nuisance_regressor_surf = np.zeros(all_data.shape)
nuisance_regressor_surf[mask_ind] = all_regs 

## save regressor
filename = op.join(figures_pth, 'nuisance_regressor.npy')
print('saving %s'%filename)
np.save(filename, nuisance_regressor_surf)

# plot example vertex for sanity check
fig, axis = plt.subplots(1, figsize=(12,6), dpi=100)
plt.plot(nuisance_regressor_surf[102874])
for ax_count in range(params['feature']['mini_blocks']):
    
    plt.axvspan(miniblk_start_ind[ax_count], miniblk_end_ind[ax_count], facecolor='#0040ff', alpha=0.1)
    
fig.savefig(op.join(figures_pth, 'example_nuisance_regressor.png'))