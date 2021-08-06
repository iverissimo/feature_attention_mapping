
import numpy as np
import os, sys
import os.path as op
import yaml
import pandas as pd
import hedfpy

import matplotlib.pyplot as plt

from utils import * #import script to use relevante functions

# load settings from yaml
with open(os.path.join(os.path.split(os.getcwd())[0],'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

# define participant number
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex: 100)'
                    'as 1st argument in the command line!')

else:
    sj = str(sys.argv[1]).zfill(3) #fill subject number with 00 in case user forgets

base_dir = 'local'
ses = params['general']['session']

# select task to analyse
for _,task in enumerate(params['eyetracking']['tasks']):
    
    # path where I'm storing pilot behavioral data
    #data_dir = '/Users/verissimo/Documents/Projects/Feature_based_attention_mapping/behavioral_pilot/data'
    #data_dir = os.path.join(data_dir,'{task}'.format(task=task),'sub-{sj}'.format(sj=sj))
    data_dir = op.join(params['mri']['paths'][base_dir], 'sourcedata','sub-{sj}'.format(sj=sj),
                        'ses-{ses}'.format(ses=ses),'func')

    # path to output for processed eye tracking files
    #eye_dir = '/Users/verissimo/Documents/Projects/Feature_based_attention_mapping/behavioral_pilot/outputs'
    #eye_dir = os.path.join(eye_dir,'{task}'.format(task=task),'eyetracking','preprocessing','sub-{sj}'.format(sj=sj))
    eye_dir = op.join(params['mri']['paths'][base_dir],'derivatives','eyetracking','{task}'.format(task=task),
                        'sub-{sj}'.format(sj=sj),'ses-{ses}'.format(ses=ses))

    # if output path doesn't exist, create it
    if not os.path.isdir(eye_dir): 
        os.makedirs(eye_dir)
    print('saving eyetracking files in %s'%eye_dir)
    
    
    # select edf files for all runs
    edf_files = [os.path.join(data_dir,run) for _,run in enumerate(os.listdir(data_dir)) if run.endswith('.edf')
                and 'task-{task}'.format(task=task) in run]
    edf_files.sort()
    
    
    #single hdf5 file that contains all eye data for the runs of that task
    hdf_file = os.path.join(eye_dir, 'sub-{sj}_task-{task}_eyetrack.h5'.format(sj=sj, task=task))  

    # convert
    alias_list = edf2h5(edf_files, hdf_file)
    
    # save all data in csvs
    # and append filenames to use later

    gaze_filenames = []
    timestamps_filenames = []

    ho = hedfpy.HDFEyeOperator(hdf_file)

    for _,al in enumerate(alias_list):
        # for each run

        gaze_filenames.append(os.path.join(eye_dir,'gaze_%s.csv'%al))
        timestamps_filenames.append(os.path.join(eye_dir,'timestamps_%s.csv'%al))

        if not os.path.isfile(os.path.join(eye_dir,'gaze_%s.csv'%al)):

            with pd.HDFStore(ho.input_object) as h5_file:

                # load table with whole block data (run is one block)
                block_data = h5_file['%s/block_0'%al]

                # load block info (timestamp etc)
                block_info = h5_file['%s/blocks'%al]


            # save timestampts 
            block_info.to_csv(os.path.join(eye_dir,'timestamps_%s.csv'%al), sep="\t")

            # save data
            block_data.to_csv(os.path.join(eye_dir,'gaze_%s.csv'%al), sep="\t")
            
    
    # load things and plot gaze per miniblock, for sanity check
    for ind in range(len(gaze_filenames)):

        # load timsetamps
        timestamps_pd = pd.read_csv(timestamps_filenames[ind],sep = '\t')
        # load gaze
        gaze_pd = pd.read_csv(gaze_filenames[ind],sep = '\t')

        # start and end time 
        run_start = timestamps_pd['block_start_timestamp'][0]
        run_end = timestamps_pd['block_end_timestamp'][0]

        # select run within start and end time
        # should not be necessary, but for sanity
        run_gaze = gaze_pd.loc[(gaze_pd['time']>=run_start)&(gaze_pd['time']<=run_end)]

        # get x and y positions, within run time
        # for interpolated data 
        #x_pos = run_gaze['L_gaze_x_int'].values
        #y_pos = run_gaze['L_gaze_y_int'].values

        # order of events in trial
        bar_pass_direction = np.array(params['feature']['bar_pass_direction'])

        # set timings of events
        TR = params['mri']['TR']
        empty_TR = params['feature']['empty_TR']
        cue_TR = params['feature']['cue_TR']
        mini_blk_TR = params['feature']['num_bar_position'][0]*params['feature']['num_bar_position'][1]*2

        # sample rate eyetracking
        sample_rate = timestamps_pd['sample_rate'].values[0]

        # make plots for each miniblock
        # so cue, empty, miniblock, empty

        # duration of miniblock, in seconds
        mini_blk_dur = (cue_TR + 2*empty_TR + mini_blk_TR)*TR


        fig, axs = plt.subplots(4,1, figsize=(15, 20), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.001)

        axs = axs.ravel()

        for i in range(params['feature']['mini_blocks']): # 4 miniblocks

            mini_blk_gaze = run_gaze.loc[(run_gaze['time']>=(run_start+(mini_blk_dur*sample_rate*i)))
                     &(run_gaze['time']<=(run_start+(mini_blk_dur*sample_rate*(i+1))))]

            mini_blk_x_pos = mini_blk_gaze['L_gaze_x_int'].values
            mini_blk_y_pos = mini_blk_gaze['L_gaze_y_int'].values

            axs[i].plot(mini_blk_x_pos,c='k')
            axs[i].plot(mini_blk_y_pos,c='orange')

            axs[i].set_xlabel('Samples',fontsize=18)
            axs[i].set_ylabel('Position',fontsize=18)
            axs[i].legend(['xgaze','ygaze'], fontsize=10)
            axs[i].set_title('Gaze run-%s miniblock-%s' %(str(ind).zfill(2),str(i)), fontsize=20)
            axs[i].axvline(x = ((cue_TR + empty_TR)*TR*sample_rate),c='green',linestyle='--',alpha=0.5) #trial start
            axs[i].axvline(x = ((cue_TR + empty_TR + mini_blk_TR)*TR*sample_rate),c='green',linestyle='--',alpha=0.5) #trial end


        fig.savefig(os.path.join(eye_dir,'gaze_xydata_run-%s.png' %str(ind).zfill(2)))


