
import numpy as np
import os, sys
import os.path as op
import yaml
import pandas as pd
import hedfpy

import glob

import matplotlib.pyplot as plt

from FAM_utils import eye as eye_utils

# load settings from yaml
with open(op.join(op.split(os.getcwd())[0],'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

# define participant number
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex: 100)'
                    'as 1st argument in the command line!')

elif len(sys.argv)<3:  
    raise NameError('Please add task to process (ex: FA or pRF)'
                        'as 2nd argument in the command line!')
else:
    sj = str(sys.argv[1]).zfill(3) #fill subject number with 00 in case user forgets
    task = str(sys.argv[2]) 

base_dir = params['general']['current_dir']
ses_type = ['beh','func'] if base_dir == 'local' else ['beh']

out_dir = op.join(params['mri']['paths'][base_dir]['root'],'derivatives',
                  'eyetracking', 'preprocessing', '{task}'.format(task=task),
                        'sub-{sj}'.format(sj=sj))

# if output path doesn't exist, create it
if not os.path.isdir(out_dir): 
    os.makedirs(out_dir)
print('saving output files in %s'%out_dir)

# check results for behavioral session, and scanner session
for _,ses in enumerate(ses_type):
    
    # set data dir
    data_dir = op.join(params['mri']['paths'][base_dir]['root'], 'sourcedata', 'sub-{sj}'.format(sj=sj))

    if base_dir == 'local': 
        data_dir = glob.glob(op.join(data_dir, 'ses-*', ses))[0]
    
    # if session type doesn't exist
    if not op.exists(data_dir) or not os.listdir(data_dir):
        
        print('no files in %s'%data_dir)
        
    else:
        print('loading files from %s'%data_dir)

        # select edf files for all runs
        edf_files = [op.join(data_dir,x) for _,x in enumerate(os.listdir(data_dir)) if task in x
                        and x.endswith('.edf')]; edf_files.sort()
        
        #single hdf5 file that contains all eye data for the runs of that task
        hdf_file = op.join(out_dir, 'sub-{sj}_task-{task}_{ses_type}_eyetrack.h5'.format(sj=sj, task=task, ses_type = ses))  

        # convert
        alias_list = eye_utils.edf2h5(edf_files, hdf_file) 

        # save all data in csvs
        # and append filenames to use later

        gaze_filenames = []
        timestamps_filenames = []

        ho = hedfpy.HDFEyeOperator(hdf_file)

        for _,al in enumerate(alias_list):
            # for each run

            gaze_filenames.append(op.join(out_dir,'{ses_type}_gaze_%s.csv'.format(ses_type = ses)%al))
            timestamps_filenames.append(op.join(out_dir,'{ses_type}_timestamps_%s.csv'.format(ses_type = ses)%al))

            if not op.isfile(op.join(out_dir,'{ses_type}_gaze_%s.csv'.format(ses_type = ses)%al)):

                with pd.HDFStore(ho.input_object) as h5_file:

                    # load table with whole block data (run is one block)
                    block_data = h5_file['%s/block_0'%al]

                    # load block info (timestamp etc)
                    block_info = h5_file['%s/blocks'%al]

                # save timestampts 
                block_info.to_csv(op.join(out_dir,'{ses_type}_timestamps_%s.csv'.format(ses_type = ses)%al), sep="\t")

                # save data
                block_data.to_csv(op.join(out_dir,'{ses_type}_gaze_%s.csv'.format(ses_type = ses)%al), sep="\t")

        if task == 'FA':         
            # load gaze and plot gaze per run, for sanity check
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
                # so empty, cue, empty, miniblock

                # duration of miniblock, in seconds
                mini_blk_dur = (cue_TR + 2*empty_TR + mini_blk_TR)*TR

                # plot gaze
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
                    axs[i].set_title('Gaze run-%s miniblock-%s' %(str(ind+1).zfill(2),str(i)), fontsize=20)
                    axs[i].axvline(x = ((empty_TR)*TR*sample_rate),c='blue',linestyle='--',alpha=0.5) #cue start
                    axs[i].axvline(x = ((cue_TR + 2*empty_TR)*TR*sample_rate),c='green',linestyle='--',alpha=0.5) #trial start
                    axs[i].axvline(x = ((cue_TR + 2*empty_TR + mini_blk_TR)*TR*sample_rate),c='green',linestyle='--',alpha=0.5) #trial end

                fig.savefig(op.join(out_dir, 'sub-{sj}_task-{task}_run-{run}_gaze-timecourse_{ses_type}.png'.format(sj=sj, task=task, run=ind+1, ses_type = ses)))

                # same figure but with raw data, 
                # for comparison
                fig, axs = plt.subplots(4,1, figsize=(15, 20), facecolor='w', edgecolor='k')
                fig.subplots_adjust(hspace = .5, wspace=.001)

                axs = axs.ravel()

                for i in range(params['feature']['mini_blocks']): # 4 miniblocks

                    mini_blk_gaze = run_gaze.loc[(run_gaze['time']>=(run_start+(mini_blk_dur*sample_rate*i)))
                            &(run_gaze['time']<=(run_start+(mini_blk_dur*sample_rate*(i+1))))]

                    mini_blk_x_pos = mini_blk_gaze['L_gaze_x'].values
                    mini_blk_y_pos = mini_blk_gaze['L_gaze_y'].values

                    axs[i].plot(mini_blk_x_pos,c='k')
                    axs[i].plot(mini_blk_y_pos,c='orange')

                    axs[i].set_xlabel('Samples',fontsize=18)
                    axs[i].set_ylabel('Position',fontsize=18)
                    axs[i].legend(['xgaze','ygaze'], fontsize=10)
                    axs[i].set_title('Gaze raw run-%s miniblock-%s' %(str(ind+1).zfill(2),str(i)), fontsize=20)
                    axs[i].axvline(x = ((empty_TR)*TR*sample_rate),c='blue',linestyle='--',alpha=0.5) #cue start
                    axs[i].axvline(x = ((cue_TR + 2*empty_TR)*TR*sample_rate),c='green',linestyle='--',alpha=0.5) #trial start
                    axs[i].axvline(x = ((cue_TR + 2*empty_TR + mini_blk_TR)*TR*sample_rate),c='green',linestyle='--',alpha=0.5) #trial end

                fig.savefig(op.join(out_dir, 'sub-{sj}_task-{task}_run-{run}_gaze-timecourse_raw_{ses_type}.png'.format(sj=sj, task=task, run=ind+1, ses_type = ses)))

        elif task == 'pRF':

            # load gaze and plot gaze per run, for sanity check
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

                # set timings of events
                TR = params['mri']['TR']
                # sample rate eyetracking
                sample_rate = timestamps_pd['sample_rate'].values[0]

                # set timings for bar, to use for plotting
                timings_bar = {'D-U': [], 'L-R': [], 'R-L': [], 
                            'U-D': [], 'empty': [], 'empty_long': []}

                time = 0
                for bar_pass_type in params['prf']['bar_pass_direction']:
                    
                    timings_bar[bar_pass_type].append(time) 
                    
                    time += params['prf']['num_TRs'][bar_pass_type]*TR
                
                # plot gaze
                fig, axs = plt.subplots(1,1, figsize=(15, 20), facecolor='w', edgecolor='k')

                x_pos = run_gaze['L_gaze_x_int'].values
                y_pos = run_gaze['L_gaze_y_int'].values

                axs.plot(x_pos,c='k')
                axs.plot(y_pos,c='orange')

                axs.set_xlabel('Samples',fontsize=18)
                axs.set_ylabel('Position',fontsize=18)
                axs.legend(['xgaze','ygaze'], fontsize=10)
                axs.set_title('Gaze run-%s' %(str(ind+1).zfill(2)), fontsize=20)

                # lines for start of empty intervals
                [axs.axvline(x = point*sample_rate, c='blue',linestyle='--',alpha=0.5) 
                for _,point in enumerate(np.concatenate((timings_bar['empty'],timings_bar['empty_long'])))]

                # lines for start of horizontal bar pass
                [axs.axvline(x = point*sample_rate, c='green',linestyle='--',alpha=0.5) 
                for _,point in enumerate(np.concatenate((timings_bar['L-R'],timings_bar['R-L'])))]

                # lines for start of vertical bar pass
                [axs.axvline(x = point*sample_rate, c='red',linestyle='--',alpha=0.5) 
                for _,point in enumerate(np.concatenate((timings_bar['D-U'],timings_bar['U-D'])))]

                fig.savefig(op.join(out_dir, 'sub-{sj}_task-{task}_run-{run}_gaze-timecourse_{ses_type}.png'.format(sj=sj, task=task, run=ind+1, ses_type = ses)))


                # same figure but with raw data, 
                # for comparison
                fig, axs = plt.subplots(1,1, figsize=(15, 20), facecolor='w', edgecolor='k')

                x_pos = run_gaze['L_gaze_x'].values
                y_pos = run_gaze['L_gaze_y'].values

                axs.plot(x_pos,c='k')
                axs.plot(y_pos,c='orange')

                axs.set_xlabel('Samples',fontsize=18)
                axs.set_ylabel('Position',fontsize=18)
                axs.legend(['xgaze','ygaze'], fontsize=10)
                axs.set_title('Gaze run-%s' %(str(ind+1).zfill(2)), fontsize=20)

                # lines for start of empty intervals
                [axs.axvline(x = point*sample_rate, c='blue',linestyle='--',alpha=0.5) 
                for _,point in enumerate(np.concatenate((timings_bar['empty'],timings_bar['empty_long'])))]

                # lines for start of horizontal bar pass
                [axs.axvline(x = point*sample_rate, c='green',linestyle='--',alpha=0.5) 
                for _,point in enumerate(np.concatenate((timings_bar['L-R'],timings_bar['R-L'])))]

                # lines for start of vertical bar pass
                [axs.axvline(x = point*sample_rate, c='red',linestyle='--',alpha=0.5) 
                for _,point in enumerate(np.concatenate((timings_bar['D-U'],timings_bar['U-D'])))]

                fig.savefig(op.join(out_dir, 'sub-{sj}_task-{task}_run-{run}_gaze-timecourse_raw_{ses_type}.png'.format(sj=sj, task=task, run=ind+1, ses_type = ses)))

