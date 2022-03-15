
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
TR = 1.6

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

rsq_threshold = 0.12

avg_roi = {} #empty dictionary  

for _,val in enumerate(ROIs):
    
    ind = np.array([vert for vert in roi_verts[val] if not np.isnan(pRF_rsq[vert]) and pRF_rsq[vert]>rsq_threshold])
    print(len(ind))
    
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

# times where bar is on screen [1st on, last on, 1st on, last on, etc] 
TR = 1.6
bar_onset = np.array([27,98,126,197,225,296,324,395])#/TR

if params['feature']['crop']:
    bar_onset = bar_onset - params['feature']['crop_TR']*TR - TR*1.5


bar_directions = [val for _,val in enumerate(params['feature']['bar_pass_direction']) if 'empty' not in val and 'cue' not in val]
# plot axis vertical bar on background to indicate stimulus display time
ax_count = 0
for h in range(len(bar_directions)):
    
    plt.axvspan(bar_onset[ax_count], bar_onset[ax_count+1]+TR, facecolor='#0040ff', alpha=0.1)
    
    ax_count += 2

fig.savefig(op.join(figures_pth, 'average_bold_accross_runs_sub-003_cropped_confound_psc.png'))

## plot timecourse of each miniblock
## aligned to same timepoint

# first, interpolate to seconds instead of TR
old_indices = np.arange(0, all_data.shape[-1])
new_length = round(all_data.shape[-1]*TR)

new_indices = np.linspace(0, all_data.shape[-1]-1, new_length)

avg_roi_sec =  {} #empty dictionary  

for _,val in enumerate(ROIs):
    
    spl = UnivariateSpline(old_indices, avg_roi[val], k=3, s=0)
    avg_roi_sec[val] = spl(new_indices)

# now get average timecourse for miniblock

avg_miniblk =  {} #empty dictionary 
interv = 5 # average from begining of miniblk to end, +/- 5 sec

for _,val in enumerate(ROIs):
    
    avg_miniblk[val] = np.mean(np.stack((avg_roi_sec[val][round(onset-interv):round(onset+72+interv)] for onset in bar_onset[::2]), axis = 0), axis = 0)

## plot timecourse of each miniblock
# aligned to same timepoint
fig, axis = plt.subplots(3, 1, figsize=(6,12), dpi=100)

## plot for V1
axis[0].plot(avg_roi_sec['V1'][round(bar_onset[::2][0]-interv):round(bar_onset[::2][0]+72+interv)])
axis[0].plot(avg_roi_sec['V1'][round(bar_onset[::2][1]-interv):round(bar_onset[::2][1]+72+interv)])
axis[0].plot(avg_roi_sec['V1'][round(bar_onset[::2][2]-interv):round(bar_onset[::2][2]+72+interv)])
axis[0].plot(avg_roi_sec['V1'][round(bar_onset[::2][3]-interv):round(bar_onset[::2][3]+72+interv)])

axis[0].plot(avg_miniblk['V1'], linewidth = 4, c='black')

axis[0].set_xticks([])
axis[0].axvspan(interv, 72+interv, facecolor='#0040ff', alpha=0.1)
axis[0].set_title('V1',fontweight="bold")

## plot for V1
axis[1].plot(avg_roi_sec['V2'][round(bar_onset[::2][0]-interv):round(bar_onset[::2][0]+72+interv)])
axis[1].plot(avg_roi_sec['V2'][round(bar_onset[::2][1]-interv):round(bar_onset[::2][1]+72+interv)])
axis[1].plot(avg_roi_sec['V2'][round(bar_onset[::2][2]-interv):round(bar_onset[::2][2]+72+interv)])
axis[1].plot(avg_roi_sec['V2'][round(bar_onset[::2][3]-interv):round(bar_onset[::2][3]+72+interv)])

axis[1].plot(avg_miniblk['V2'], linewidth = 4, c='black')

axis[1].set_xticks([])
axis[1].axvspan(interv, 72+interv, facecolor='#0040ff', alpha=0.1)
axis[1].set_title('V2',fontweight="bold")

## plot for V3
axis[2].plot(avg_roi_sec['V3'][round(bar_onset[::2][0]-interv):round(bar_onset[::2][0]+72+interv)])
axis[2].plot(avg_roi_sec['V3'][round(bar_onset[::2][1]-interv):round(bar_onset[::2][1]+72+interv)])
axis[2].plot(avg_roi_sec['V3'][round(bar_onset[::2][2]-interv):round(bar_onset[::2][2]+72+interv)])
axis[2].plot(avg_roi_sec['V3'][round(bar_onset[::2][3]-interv):round(bar_onset[::2][3]+72+interv)])

axis[2].plot(avg_miniblk['V3'], linewidth = 4, c='black')

axis[2].set_xlabel('Time (s)',fontsize=20, labelpad=20)
axis[2].axvspan(interv, 72+interv, facecolor='#0040ff', alpha=0.1)
axis[2].set_title('V3',fontweight="bold")

fig.savefig(op.join(figures_pth, 'average_bold_accross_miniblocks_sub-003_cropped_confound_psc.png'))

## check same for raw signal?

## list with absolute file name 
raw_files = [op.join(op.split(postfmriprep_dir)[0], h) for h in os.listdir(op.split(postfmriprep_dir)[0]) if 'task-FA' in h and
                 'acq-{acq}'.format(acq=acq) in h and h.endswith('cropped.npy')]

## load each, psc and average
psc_data = []
for input_file in raw_files:
    data = np.load(input_file,allow_pickle=True)

    mean_signal = data.mean(axis = -1)[..., np.newaxis]
    data_psc = (data - mean_signal)/np.absolute(mean_signal)
    data_psc *= 100
    
    psc_data.append(data_psc)
    
all_psc_data = np.mean(psc_data, axis = 0)

avg_raw_roi = {} #empty dictionary  

for _,val in enumerate(ROIs):
    
    ind = np.array([vert for vert in roi_verts[val] if not np.isnan(pRF_rsq[vert]) and pRF_rsq[vert]>rsq_threshold])
    print(len(ind))
    
    avg_raw_roi[val] = np.mean(all_psc_data[ind], axis=0)

# plot data with model
fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)
 
time_sec = np.linspace(0,len(all_data[0])*TR,num=len(all_data[0])) # array with timepoints, in seconds

plt.plot(time_sec, avg_raw_roi['V1'], linewidth = 2.5, label='V1 data')
plt.plot(time_sec, avg_raw_roi['V2'], linewidth = 2.5, label='V2 data')
plt.plot(time_sec, avg_raw_roi['V3'], linewidth = 2.5, label='V3 data')

axis.set_xlabel('Time (s)',fontsize=20, labelpad=20)
axis.set_ylabel('BOLD signal change (%)',fontsize=20, labelpad=10)
axis.legend(loc='upper left',fontsize=10)  # doing this to guarantee that legend is how I want it 

# times where bar is on screen [1st on, last on, 1st on, last on, etc] 
TR = 1.6
bar_onset = np.array([27,98,126,197,225,296,324,395])#/TR

if params['feature']['crop']:
    bar_onset = bar_onset - params['feature']['crop_TR']*TR - TR*1.5


bar_directions = [val for _,val in enumerate(params['feature']['bar_pass_direction']) if 'empty' not in val and 'cue' not in val]
# plot axis vertical bar on background to indicate stimulus display time
ax_count = 0
for h in range(len(bar_directions)):
    
    plt.axvspan(bar_onset[ax_count], bar_onset[ax_count+1]+TR, facecolor='#0040ff', alpha=0.1)
    
    ax_count += 2

fig.savefig(op.join(figures_pth, 'average_bold_accross_runs_sub-003_cropped_psc.png'))

## also align like before 
avg_raw_roi_sec =  {} #empty dictionary  

for _,val in enumerate(ROIs):
    
    spl = UnivariateSpline(old_indices, avg_raw_roi[val], k=3, s=0)
    avg_raw_roi_sec[val] = spl(new_indices)

# now get average timecourse for miniblock
avg_raw_miniblk =  {} #empty dictionary 
interv = 5 # average from begining of miniblk to end, +/- 5 sec

for _,val in enumerate(ROIs):
    
    avg_raw_miniblk[val] = np.mean(np.stack((avg_raw_roi_sec[val][round(onset-interv):round(onset+72+interv)] for onset in bar_onset[::2]), axis = 0), axis = 0)

## plot timecourse of each miniblock
# aligned to same timepoint
fig, axis = plt.subplots(3, 1, figsize=(6,12), dpi=100)

## plot for V1
axis[0].plot(avg_raw_roi_sec['V1'][round(bar_onset[::2][0]-interv):round(bar_onset[::2][0]+72+interv)])
axis[0].plot(avg_raw_roi_sec['V1'][round(bar_onset[::2][1]-interv):round(bar_onset[::2][1]+72+interv)])
axis[0].plot(avg_raw_roi_sec['V1'][round(bar_onset[::2][2]-interv):round(bar_onset[::2][2]+72+interv)])
axis[0].plot(avg_raw_roi_sec['V1'][round(bar_onset[::2][3]-interv):round(bar_onset[::2][3]+72+interv)])

axis[0].plot(avg_raw_miniblk['V1'], linewidth = 4, c='black')

axis[0].set_xticks([])
axis[0].axvspan(interv, 72+interv, facecolor='#0040ff', alpha=0.1)
axis[0].set_title('V1',fontweight="bold")

## plot for V1
axis[1].plot(avg_raw_roi_sec['V2'][round(bar_onset[::2][0]-interv):round(bar_onset[::2][0]+72+interv)])
axis[1].plot(avg_raw_roi_sec['V2'][round(bar_onset[::2][1]-interv):round(bar_onset[::2][1]+72+interv)])
axis[1].plot(avg_raw_roi_sec['V2'][round(bar_onset[::2][2]-interv):round(bar_onset[::2][2]+72+interv)])
axis[1].plot(avg_raw_roi_sec['V2'][round(bar_onset[::2][3]-interv):round(bar_onset[::2][3]+72+interv)])

axis[1].plot(avg_raw_miniblk['V2'], linewidth = 4, c='black')

axis[1].set_xticks([])
axis[1].axvspan(interv, 72+interv, facecolor='#0040ff', alpha=0.1)
axis[1].set_title('V2',fontweight="bold")

## plot for V3
axis[2].plot(avg_raw_roi_sec['V3'][round(bar_onset[::2][0]-interv):round(bar_onset[::2][0]+72+interv)])
axis[2].plot(avg_raw_roi_sec['V3'][round(bar_onset[::2][1]-interv):round(bar_onset[::2][1]+72+interv)])
axis[2].plot(avg_raw_roi_sec['V3'][round(bar_onset[::2][2]-interv):round(bar_onset[::2][2]+72+interv)])
axis[2].plot(avg_raw_roi_sec['V3'][round(bar_onset[::2][3]-interv):round(bar_onset[::2][3]+72+interv)])

axis[2].plot(avg_raw_miniblk['V3'], linewidth = 4, c='black')

axis[2].set_xlabel('Time (s)',fontsize=20, labelpad=20)
axis[2].axvspan(interv, 72+interv, facecolor='#0040ff', alpha=0.1)
axis[2].set_title('V3',fontweight="bold")

fig.savefig(op.join(figures_pth, 'average_bold_accross_miniblocks_sub-003_cropped_psc.png'))