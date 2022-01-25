
import numpy as np
import os, sys
import yaml
import pandas as pd
import os.path as op

from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import seaborn as sns

from hedfpy.EyeSignalOperator import detect_saccade_from_data

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
                  'eyetracking', 'visualization', '{task}'.format(task=task),
                        'sub-{sj}'.format(sj=sj))

# if output path doesn't exist, create it
if not os.path.isdir(out_dir): 
    os.makedirs(out_dir)
print('saving output files in %s'%out_dir)

# path to get processed eyetracking files 
eye_dir = op.join(params['mri']['paths'][base_dir]['root'],'derivatives',
                  'eyetracking', 'preprocessing', '{task}'.format(task=task),
                        'sub-{sj}'.format(sj=sj))

# path to get summary behavioral files
behav_dir =  op.join(params['mri']['paths'][base_dir]['root'],'derivatives',
                  'behavioral','{task}'.format(task=task),
                        'sub-{sj}'.format(sj=sj))

# general params
# set timings of events
TR = params['mri']['TR']
empty_TR = params['feature']['empty_TR']
cue_TR = params['feature']['cue_TR']
mini_blk_TR = params['feature']['num_bar_position'][0]*params['feature']['num_bar_position'][1]*2

# sample rate eyetracking
sample_rate = params['eyetracking']['sampling_freq']

# make plots for each miniblock
# so empty, cue, empty, miniblock

# duration of miniblock, in seconds
mini_blk_dur = (cue_TR + 2*empty_TR + mini_blk_TR)*TR

# check results for behavioral session, and scanner session
for _,ses in enumerate(ses_type):
    
    print('loading files from %s'%eye_dir)

    # list gaze filenames
    gaze_filenames = [op.join(eye_dir,x) for _,x in enumerate(os.listdir(eye_dir)) if x.startswith('{ses_type}_gaze'.format(ses_type = ses))
                 and x.endswith('.csv')]; gaze_filenames.sort()
    # list time stamps
    timestamps_filenames = [op.join(eye_dir,x) for _,x in enumerate(os.listdir(eye_dir)) if x.startswith('{ses_type}_timestamps'.format(ses_type = ses))
                                and x.endswith('.csv')]; timestamps_filenames.sort()

    # load behavioral dataframe
    behav_pd = pd.read_csv(op.join(behav_dir,'sub-{sj}_task-{task}_acc_RT_{ses_type}.csv'.format(sj=sj, task=task, ses_type = ses)))

    if task == 'FA':  
        # make
        # summary data frame to save relevant run info
        df_gaze = pd.DataFrame(columns=['run','mini_block','attend_condition','condition', 'gaze_x','gaze_y'])
        df_sacc = pd.DataFrame(columns=['run','mini_block','attend_condition','condition', 'expanded_start_time','expanded_end_time',
                           'expanded_vector','expanded_amplitude'])

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

            for i in range(params['feature']['mini_blocks']): # 4 miniblocks

                mini_blk_gaze = run_gaze.loc[(run_gaze['time']>=(run_start+(mini_blk_dur*sample_rate*i)))
                        &(run_gaze['time']<=(run_start+(mini_blk_dur*sample_rate*(i+1))))]

                gaze_x_mini_blk = mini_blk_gaze['L_gaze_x_int'].values
                gaze_y_mini_blk = mini_blk_gaze['L_gaze_y_int'].values

                # get which condition was being attended at miniblock
                attend_ind = np.where(behav_pd.loc[(behav_pd['run']==ind+1)&(behav_pd['mini_block']==i)]['attend_condition'].values)[0][0]

                df_gaze = df_gaze.append({'run': ind+1,
                                'mini_block': i, # miniblock number
                                'attend_condition': behav_pd.loc[(behav_pd['run']==ind+1)&(behav_pd['mini_block']==i)]['attend_condition'].values[0], # attended condition 
                                'condition': behav_pd.loc[(behav_pd['run']==ind+1)&(behav_pd['mini_block']==i)]['condition'].values[attend_ind],
                                'gaze_x': list(gaze_x_mini_blk), # gaze x
                                'gaze_y': list(gaze_y_mini_blk) # gaze y
                                }, ignore_index=True) 

                # detect saccades and save in dataframe

                data_x = list(gaze_x_mini_blk)
                data_y = list(gaze_y_mini_blk)

                # detect saccades
                saccades = detect_saccade_from_data(xy_data = np.array([np.array(data_x).squeeze(),np.array(data_y).squeeze()]).T, 
                                            l = params['eyetracking']['sacc_thresh'], 
                                            sample_rate = params['eyetracking']['sampling_freq'], 
                                            minimum_saccade_duration = params['eyetracking']['minimum_saccade_duration'])
                
                df_sacc = df_sacc.append({'run': ind+1,
                                'mini_block': i, # miniblock number
                                'attend_condition': behav_pd.loc[(behav_pd['run']==ind+1)&(behav_pd['mini_block']==i)]['attend_condition'].values[0], # attended condition  
                                'condition': behav_pd.loc[(behav_pd['run']==ind+1)&(behav_pd['mini_block']==i)]['condition'].values[attend_ind],
                                'expanded_start_time': [saccades[x]['expanded_start_time'] if saccades[x]['expanded_start_time']!=0
                                                        else np.nan for x,_ in enumerate(saccades)], # start time of saccades
                                'expanded_end_time': [saccades[x]['expanded_end_time'] if saccades[x]['expanded_end_time']!=0
                                                        else np.nan for x,_ in enumerate(saccades)], # end time of saccades
                                'expanded_vector': [saccades[x]['expanded_vector'] if np.array(saccades[x]['expanded_vector']).any () != 0
                                                        else [np.nan,np.nan] for x,_ in enumerate(saccades)], # position relative to center (0,0)
                                'expanded_amplitude': [saccades[x]['expanded_amplitude'] if saccades[x]['expanded_end_time']!=0
                                                        else np.nan for x,_ in enumerate(saccades)], # amplitude (in degrees?)
                                }, ignore_index=True) 


    elif task == 'pRF':   
        # make
        # summary data frame to save relevant run info
        df_gaze = pd.DataFrame(columns=['run', 'gaze_x','gaze_y'])
        df_sacc = pd.DataFrame(columns=['run', 'expanded_start_time','expanded_end_time','expanded_vector','expanded_amplitude'])

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

            data_x = list(run_gaze['L_gaze_x_int'].values)
            data_y = list(run_gaze['L_gaze_y_int'].values)

            df_gaze = df_gaze.append({'run': ind+1,
                                    'gaze_x': data_x, # gaze x
                                    'gaze_y': data_y # gaze y
                                    }, ignore_index=True) 

            # detect saccades and save in dataframe
            saccades = detect_saccade_from_data(xy_data = np.array([np.array(data_x).squeeze(),np.array(data_y).squeeze()]).T, 
                                        l = params['eyetracking']['sacc_thresh'], 
                                        sample_rate = params['eyetracking']['sampling_freq'], 
                                        minimum_saccade_duration = params['eyetracking']['minimum_saccade_duration'])

            df_sacc = df_sacc.append({'run': ind+1,
                            'expanded_start_time': [saccades[x]['expanded_start_time'] if saccades[x]['expanded_start_time']!=0
                                                    else np.nan for x,_ in enumerate(saccades)], # start time of saccades
                            'expanded_end_time': [saccades[x]['expanded_end_time'] if saccades[x]['expanded_end_time']!=0
                                                    else np.nan for x,_ in enumerate(saccades)], # end time of saccades
                            'expanded_vector': [saccades[x]['expanded_vector'] if np.array(saccades[x]['expanded_vector']).any () != 0
                                                    else [np.nan,np.nan] for x,_ in enumerate(saccades)], # position relative to center (0,0)
                            'expanded_amplitude': [saccades[x]['expanded_amplitude'] if saccades[x]['expanded_end_time']!=0
                                                    else np.nan for x,_ in enumerate(saccades)], # amplitude (in degrees?)
                            }, ignore_index=True) 

    
    ## actually make the plots

    for run in range(len(gaze_filenames)):

        # plot gaze density for run

        filename = op.join(out_dir, 'sub-{sj}_task-{task}_gaze_KDE_run-{run}_{ses_type}.png'.format(sj=sj, run=run+1, task=task, ses_type = ses))
        
        eye_utils.plot_gaze_kde(df_gaze, filename, task = task, run = run+1, conditions = ['green_horizontal','green_vertical','red_horizontal','red_vertical'], screen = params['window']['size'], downsample = 10)
    

        # plot saccade histogram for run

        filename = op.join(out_dir, 'sub-{sj}_task-{task}_sacc_hist_run-{run}_{ses_type}.png'.format(sj=sj, run=run+1, task=task, ses_type = ses))
        
        eye_utils.plot_sacc_hist(df_sacc, filename, task = task, run = run+1, 
                    conditions = ['green_horizontal','green_vertical','red_horizontal','red_vertical'])
        

    # # plot saccade polar angle histogram for run

    # # iterate over dataframe to save angles
    # angles = []
    # for i in range(len(df_sacc)):
        
    #     angles.append(get_saccade_angle(df_sacc.iloc[i]['expanded_vector'], angle_unit='radians'))
        
    # df_sacc['angle'] = angles


    # for run in range(len(gaze_filenames)):

    #     # Visualise with polar histogram
    #     fig, ax = plt.subplots(2, 2, subplot_kw=dict(projection='polar'), figsize=(30,15))

    #     plt_counter = 0
    #     conditions = ['green_horizontal','green_vertical','red_horizontal','red_vertical']
    #     color = {'green_horizontal':(0,1,0),'green_vertical':(0,1,0),'red_horizontal':(1,0,0),'red_vertical': (1,0,0)}

    #     for i in range(2):
    #         for w in range(2):

    #             ang = df_sacc.loc[(df_sacc['run'] == run+1) &
    #                                     (df_sacc['attend_condition'] == conditions[plt_counter])]['angle'].values[0]

    #             rose_plot(ax[i][w], np.array(ang),color=color[conditions[plt_counter]])
    #             ax[i][w].set_title(conditions[plt_counter],fontsize=18)
    #             ax[i][w].text(0.7, 0.9,'total %i saccades'%(sum(~np.isnan(ang))), 
    #                             ha='center', va='center', transform=ax[i][w].transAxes,
    #                             fontsize = 15)
    #             plt_counter += 1

    #     fig.tight_layout()
    #     fig.savefig(os.path.join(out_dir,'sacc_polar_hist_run-%s.png' %str(run+1).zfill(2)))





