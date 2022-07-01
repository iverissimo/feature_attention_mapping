################################################
#      Plot bold func data, 
#    (after postfmriprep, so HP and psc)
#    and make video of bold change by TR
#    to check for visual stimuli
################################################

import numpy as np
import os, sys
import os.path as op
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import cortex
from FAM_utils import mri as mri_utils

# load settings from yaml
with open(op.join(str(Path(os.getcwd()).parents[1]),'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')
  
elif len(sys.argv) < 3:
    raise NameError('Please add type of run to be fitted (ex: leave_01_out vs median) '
                    'as 2nd argument in the command line!')

elif len(sys.argv) < 4:
    raise NameError('Please add task (ex: FA vs pRF) '
                    'as 3rd argument in the command line!')
    
else:
    # fill subject number and chunk number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(3)
    run_type = str(sys.argv[2])
    task = str(sys.argv[3])

# set font type for plots globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# some settings
base_dir = params['general']['current_dir'] # which machine we run the data
acq = params['mri']['acq'] # if using standard files or nordic files
space = params['mri']['space'] # subject space

TR = params['mri']['TR']

## define file extension that we want to use, 
# should include processing key words
task_key = 'feature' if task == 'FA' else 'prf'
file_ext = ''

# if cropped first
if params[task_key]['crop']:
    file_ext += '_{name}'.format(name = 'cropped')
# type of filtering/denoising
if params[task_key]['regress_confounds']:
    file_ext += '_{name}'.format(name = 'confound')
else:
    file_ext += '_{name}'.format(name = params['mri']['filtering']['type'])
# type of standardization 
file_ext += '_{name}'.format(name = params[task_key]['standardize'])
# don't forget its a numpy array
file_ext += '.npy'

## set paths
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
postfmriprep_dir = op.join(derivatives_dir,'post_fmriprep','sub-{sj}'.format(sj=sj),space,'processed')

# output dir to save fit and plot
figures_pth = op.join(derivatives_dir,'plots', 'bold', task, 'sub-{sj}'.format(sj=sj), space,'run-{run}'.format(run=run_type)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

## load functional data

# list with absolute file names to be fitted (iff gii, then 2 hemispheres)
proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-{task}'.format(task=task) in h and
                 'acq-{acq}'.format(acq=acq) in h and h.endswith(file_ext)]


# if we want mean of all feature runs, then we want to check for artifacts so average all
if (task == 'FA') and (run_type in ['mean', 'median']):
    
    # load data from all runs
    all_data = np.stack((np.load(file,allow_pickle=True) for file in proc_files), axis = 0)

    # average accross runs
    if run_type == 'median':
        data = np.median(all_data, axis = 0)
    else:
        data = np.mean(all_data, axis = 0)
else:
    file = [f for f in proc_files if 'run-{r}'.format(r=run_type) in f][0]
    data = np.load(file,allow_pickle=True) # will be (vertex, TR)

## get pycortex sub
pysub = params['plotting']['pycortex_sub']+'_sub-{sj}'.format(sj=sj) # because subject specific borders 

if pysub not in os.listdir(cortex.options.config.get('basic', 'filestore')): # if we haven-t drawn borders, then use standard overlay
    pysub = params['plotting']['pycortex_sub']

## get vertices for relevant ROIs and color codes
ROIs, roi_verts, color_codes = mri_utils.get_rois4plotting(params, pysub = pysub, 
                                            use_atlas = False, atlas_pth = op.join(derivatives_dir,'glasser_atlas','59k_mesh'))

## set moments when bar on screen, to plot simultaneously with bold signal
bar_pass_direction = params[task_key]['bar_pass_direction']
stim_on_screen = np.ones(data.shape[-1])

if task == 'FA':
    # crop beggining
    if params[task_key]['crop']:
        curr_tr = params[task_key]['empty_TR']-1-params[task_key]['crop_TR']
    else:
        curr_tr = params[task_key]['empty_TR']-1 # -1 because of trigger shift
    stim_on_screen[:curr_tr] = 0
    
    # fill end
    stim_on_screen[-int(params[task_key]['empty_TR']+1):] = 0
    
elif task == 'pRF':
    tr_counter = 0
    for ind, phase in enumerate(bar_pass_direction):

        if phase in ['empty', 'empty_long']:
            if ind == 0:
                if params[task_key]['crop']:
                    curr_tr = params[task_key]['num_TRs'][phase]-1-params[task_key]['crop_TR']
                else:
                    curr_tr = params[task_key]['num_TRs'][phase]-1
            else:
                curr_tr = params[task_key]['num_TRs'][phase]
                    
            stim_on_screen[tr_counter:tr_counter+curr_tr] = 0
            
        else:
            curr_tr = params[task_key]['num_TRs'][phase]
        
        tr_counter += curr_tr
        
    # fill end, due to shift
    stim_on_screen[-1] = 0

## get average timecourse across ROI
# to check if something off is happening
if (task == 'FA') and (run_type in ['mean', 'median']):
    avg_bold_roi = {} #empty dictionary 

    for _,val in enumerate(ROIs):    
        avg_bold_roi[val] = np.mean(data[roi_verts[val]], axis=0)
        
    # plot data with model
    fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)

    time_sec = np.linspace(0,len(data[0])*TR,num=len(data[0])) # array with timepoints, in seconds
    
    plt.plot(time_sec, stim_on_screen, linewidth = 5, alpha = 1, linestyle = 'solid', color = 'gray')

    for _,key in enumerate(ROIs):
        plt.plot(time_sec, avg_bold_roi[key], linewidth = 1.5, label = '%s'%key, color = color_codes[key], alpha = .6)

    # also plot average of all time courses
    plt.plot(time_sec, np.mean(np.stack((avg_bold_roi[val] for val in ROIs), axis = 0), axis = 0),
             linewidth = 2.5, label = 'average', linestyle = 'solid', color = 'k')

    axis.set_xlabel('Time (s)',fontsize=20, labelpad=20)
    axis.set_ylabel('BOLD signal change (%)',fontsize=20, labelpad=10)
    axis.legend(loc='upper left',fontsize=7)  # doing this to guarantee that legend is how I want it 
    #axis.set_xlim([0, time_sec[-1]])

    fig.savefig(op.join(figures_pth, 'average_bold_across_runs_rois.png'))
###

## make movie
movie_name = op.join(figures_pth,'flatmap_space-{space}_type-BOLD_visual_movie.mp4'.format(space=pysub))

if not op.isfile(movie_name):

    for num_tr in range(data.shape[-1]):

        # set figure grid 
        full_fig = plt.figure(constrained_layout = True, figsize = (15,8))
        gs = full_fig.add_gridspec(5, 6)

        ## set axis
        dm_ax = full_fig.add_subplot(gs[:1,2:4])
        flatmap_ax = full_fig.add_subplot(gs[1:,:])

        # set flatmap
        flatmap = cortex.Vertex(data[...,num_tr], 
                                        pysub,
                                        vmin = -5, vmax = 5,
                                        cmap='BuBkRd')
        cortex.quickshow(flatmap, 
                        with_colorbar = True, with_curvature = True, with_sulci = True,
                        with_labels = False, fig = flatmap_ax)

        flatmap_ax.set_xticks([])
        flatmap_ax.set_yticks([])

        # set dm timecourse
        dm_ax.plot(stim_on_screen)
        dm_ax.axvline(num_tr, color='red', linestyle='solid', lw=1)
        dm_ax.set_yticks([])
        
        filename = op.join(figures_pth,'flatmap_space-{space}_type-BOLD_visual_TR-{time}.png'.format(space=pysub,
                                                                                            time=str(num_tr).zfill(3)))
        print('saving %s' %filename)
        full_fig.savefig(filename)


    ## save as video
    img_name = filename.replace('_TR-%s.png'%str(num_tr).zfill(3),'_TR-%3d.png')
    os.system("ffmpeg -r 6 -start_number 0 -i %s -vcodec mpeg4 -y %s"%(img_name,movie_name)) 

else:
    print('movie already exists as %s'%movie_name)