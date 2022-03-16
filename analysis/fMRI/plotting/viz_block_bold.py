
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
from scipy.interpolate import UnivariateSpline


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

plt.plot(time_sec, avg_roi['V1'], linewidth = 2.5, label='V1 data')
plt.plot(time_sec, avg_roi['V2'], linewidth = 2.5, label='V2 data')
plt.plot(time_sec, avg_roi['V3'], linewidth = 2.5, label='V3 data')

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
fig, axis = plt.subplots(3, 1, figsize=(6,12), dpi=100)

## plot for V1
axis[0].plot(avg_roi['V1'][miniblk_start_ind[0]-interv:miniblk_end_ind[0]+interv])
axis[0].plot(avg_roi['V1'][miniblk_start_ind[1]-interv:miniblk_end_ind[1]+interv])
axis[0].plot(avg_roi['V1'][miniblk_start_ind[2]-interv:miniblk_end_ind[2]+interv])
axis[0].plot(avg_roi['V1'][miniblk_start_ind[3]-interv:miniblk_end_ind[3]+interv])

axis[0].plot(avg_miniblk['V1'], linewidth = 4, c='black')

axis[0].set_xticks([])
axis[0].axvspan(interv, miniblk_end_ind[0]-miniblk_start_ind[0]+interv, facecolor='#0040ff', alpha=0.1)
axis[0].set_title('V1',fontweight="bold")

## plot for V2
axis[1].plot(avg_roi['V2'][miniblk_start_ind[0]-interv:miniblk_end_ind[0]+interv])
axis[1].plot(avg_roi['V2'][miniblk_start_ind[1]-interv:miniblk_end_ind[1]+interv])
axis[1].plot(avg_roi['V2'][miniblk_start_ind[2]-interv:miniblk_end_ind[2]+interv])
axis[1].plot(avg_roi['V2'][miniblk_start_ind[3]-interv:miniblk_end_ind[3]+interv])

axis[1].plot(avg_miniblk['V2'], linewidth = 4, c='black')

axis[1].set_xticks([])
axis[1].axvspan(interv, miniblk_end_ind[0]-miniblk_start_ind[0]+interv, facecolor='#0040ff', alpha=0.1)
axis[1].set_title('V2',fontweight="bold")

## plot for V3
axis[2].plot(avg_roi['V3'][miniblk_start_ind[0]-interv:miniblk_end_ind[0]+interv])
axis[2].plot(avg_roi['V3'][miniblk_start_ind[1]-interv:miniblk_end_ind[1]+interv])
axis[2].plot(avg_roi['V3'][miniblk_start_ind[2]-interv:miniblk_end_ind[2]+interv])
axis[2].plot(avg_roi['V3'][miniblk_start_ind[3]-interv:miniblk_end_ind[3]+interv])

axis[2].plot(avg_miniblk['V3'], linewidth = 4, c='black')

axis[2].set_xlabel('Time (TR)',fontsize=20, labelpad=20)
axis[2].axvspan(interv, miniblk_end_ind[0]-miniblk_start_ind[0]+interv, facecolor='#0040ff', alpha=0.1)
axis[2].set_title('V3',fontweight="bold")

fig.savefig(op.join(figures_pth, 'average_bold_accross_miniblocks_TR.png'))


## average across rois and miniblocks

avg_minblk_tc =  np.mean(np.stack((avg_miniblk[val] for val in ROIs), axis = 0), axis = 0)

## model nuisance