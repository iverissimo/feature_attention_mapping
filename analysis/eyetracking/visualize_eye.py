
import numpy as np
import os, sys
import yaml
import pandas as pd
import hedfpy

from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import seaborn as sns

from hedfpy.EyeSignalOperator import detect_saccade_from_data

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


task = 'PRFfeature'

# path to get processed eyetracking files
eye_dir = '/Users/verissimo/Documents/Projects/Feature_based_attention_mapping/behavioral_pilot/outputs'
eye_dir = os.path.join(eye_dir,'{task}'.format(task=task),'eyetracking','preprocessing','sub-{sj}'.format(sj=sj))

out_dir = os.path.join(os.path.split(os.path.split(eye_dir)[0])[0],'visualization','sub-{sj}'.format(sj=sj))
# if output path doesn't exist, create it
if not os.path.isdir(out_dir): 
    os.makedirs(out_dir)
print('saving plots files in %s'%out_dir)

# set timings of events
TR = params['mri']['TR']
empty_TR = params['feature']['empty_TR']
cue_TR = params['feature']['cue_TR']
mini_blk_TR = 64*2

# sample rate eyetracking
sample_rate = params['eyetracking']['sampling_freq']

# duration of miniblock, in seconds (cue, empty, miniblock, empty)
mini_blk_dur = (cue_TR + 2*empty_TR + mini_blk_TR)*TR


# list gaze filenames
gaze_filenames = [os.path.join(eye_dir,run) for _,run in enumerate(os.listdir(eye_dir)) if run.startswith('gaze_')
                 and run.endswith('.csv')]
gaze_filenames.sort()

# list time stamps
timestamps_filenames = [os.path.join(eye_dir,run) for _,run in enumerate(os.listdir(eye_dir)) if run.startswith('timestamps_')
                             and run.endswith('.csv')]
timestamps_filenames.sort()


# get behavioral info, to check which conditions attended
behav_dir = '/Users/verissimo/Documents/Projects/Feature_based_attention_mapping/behavioral_pilot/outputs'
behav_dir = os.path.join(behav_dir,'{task}'.format(task=task),'behavioral','sub-{sj}'.format(sj=sj))

# list behavioral data filenames
behav_filenames = [os.path.join(behav_dir,run) for _,run in enumerate(os.listdir(behav_dir)) if run.endswith('_acc_RT.csv')]
# load behav
behav_pd = pd.read_csv(behav_filenames[0])


# check if gaze dataframe already in dir

gazeDFname = os.path.join(out_dir,'sub-{sj}_task-{task}_gaze_summary.csv'.format(sj=sj, task=task))

if os.path.exists(gazeDFname):
    # load
    df_gaze = pd.read_csv(gazeDFname)
    
else:
    # make
    # summary data frame to save relevant run info
    df_gaze = pd.DataFrame(columns=['run','mini_block','attend_condition', 'gaze_x','gaze_y'])

    # for each run
    for run in range(len(gaze_filenames)):

        # load timsetamps
        timestamps_pd = pd.read_csv(timestamps_filenames[run],sep = '\t')
        # load gaze
        gaze_pd = pd.read_csv(gaze_filenames[run],sep = '\t')

        # start and end time 
        run_start = timestamps_pd['block_start_timestamp'][0]
        run_end = timestamps_pd['block_end_timestamp'][0]

        # select run within start and end time
        # should not be necessary, but for sanity
        run_gaze = gaze_pd.loc[(gaze_pd['time']>=run_start)&(gaze_pd['time']<=run_end)]

        for ind in range(params['feature']['mini_blocks']): # 4 miniblocks

            mini_blk_gaze = run_gaze.loc[(run_gaze['time']>=(run_start+(mini_blk_dur*sample_rate*ind)))
                         &(run_gaze['time']<=(run_start+(mini_blk_dur*sample_rate*(ind+1))))]

            gaze_x_mini_blk = mini_blk_gaze['L_gaze_x_int'].values
            gaze_y_mini_blk = mini_blk_gaze['L_gaze_y_int'].values

            df_gaze = df_gaze.append({'run': run,
                              'mini_block': ind, # miniblock number
                              'attend_condition': behav_pd.loc[(behav_pd['run']==run)&(behav_pd['mini_block']==ind)]['attend_condition'].values[0], # attended condition 
                              'gaze_x': list(gaze_x_mini_blk), # gaze x
                              'gaze_y': list(gaze_y_mini_blk) # gaze y
                             }, ignore_index=True) 

    # save dataframe
    df_gaze.to_csv(gazeDFname, index = False, header=True)


# plot gaze density for run

for run in range(len(gaze_filenames)):
    
    plot_gaze_kde(df_gaze, out_dir, run = run, conditions = ['green_horizontal','green_vertical','red_horizontal','red_vertical'],
                 screen = params['window']['size'], downsample = 10)
    


# detect saccades and save in dataframe

# make
# summary data frame to save relevant run info
df_sacc = pd.DataFrame(columns=['run','mini_block','attend_condition', 'expanded_start_time','expanded_end_time',
                               'expanded_vector','expanded_amplitude'])

# for each run
for run in range(len(gaze_filenames)):

    # for each attended condition
    for ind in range(params['feature']['mini_blocks']): # 4 miniblocks
        
        # attended condition
        cond = behav_pd.loc[(behav_pd['run']==run)&(behav_pd['mini_block']==ind)]['attend_condition'].values[0]
        
        # get gaze for the miniblock
        data_x = df_gaze.loc[(df_gaze['run'] == run) &
                  (df_gaze['attend_condition'] == cond)]['gaze_x'].values[0]
        data_y = df_gaze.loc[(df_gaze['run'] == run) &
                  (df_gaze['attend_condition'] == cond)]['gaze_y'].values[0]
        
        # workaround for pandas convertion of list to string
        if type(data_x) != list:

            data_x = literal_eval(data_x)
            data_y = literal_eval(data_y)

        # detect saccades
        saccades = detect_saccade_from_data(xy_data = np.array([np.array(data_x).squeeze(),np.array(data_y).squeeze()]).T, 
                                    l = params['eyetracking']['sacc_thresh'], 
                                    sample_rate = params['eyetracking']['sampling_freq'], 
                                    minimum_saccade_duration = params['eyetracking']['minimum_saccade_duration'])
        
        df_sacc = df_sacc.append({'run': run,
                          'mini_block': ind, # miniblock number
                          'attend_condition': cond, # attended condition 
                          'expanded_start_time': [saccades[x]['expanded_start_time'] if saccades[x]['expanded_start_time']!=0
                                                  else np.nan for x,_ in enumerate(saccades)], # start time of saccades
                          'expanded_end_time': [saccades[x]['expanded_end_time'] if saccades[x]['expanded_end_time']!=0
                                                  else np.nan for x,_ in enumerate(saccades)], # end time of saccades
                          'expanded_vector': [saccades[x]['expanded_vector'] if np.array(saccades[x]['expanded_vector']).any () != 0
                                                  else [np.nan,np.nan] for x,_ in enumerate(saccades)], # position relative to center (0,0)
                          'expanded_amplitude': [saccades[x]['expanded_amplitude'] if saccades[x]['expanded_end_time']!=0
                                                  else np.nan for x,_ in enumerate(saccades)], # amplitude (in degrees?)
                         }, ignore_index=True) 


# plot saccade histogram for run

for run in range(len(gaze_filenames)):
    
    plot_sacc_hist(df_sacc, out_dir, run = run, 
                   conditions = ['green_horizontal','green_vertical','red_horizontal','red_vertical'])
    

# plot saccade polar angle histogram for run

# iterate over dataframe to save angles
angles = []
for i in range(len(df_sacc)):
    
    angles.append(get_saccade_angle(df_sacc.iloc[i]['expanded_vector'], angle_unit='radians'))
    
df_sacc['angle'] = angles


for run in range(len(gaze_filenames)):

    # Visualise with polar histogram
    fig, ax = plt.subplots(2, 2, subplot_kw=dict(projection='polar'), figsize=(30,15))

    plt_counter = 0
    conditions = ['green_horizontal','green_vertical','red_horizontal','red_vertical']
    color = {'green_horizontal':(0,1,0),'green_vertical':(0,1,0),'red_horizontal':(1,0,0),'red_vertical': (1,0,0)}

    for i in range(2):
        for w in range(2):

            ang = df_sacc.loc[(df_sacc['run'] == run) &
                                    (df_sacc['attend_condition'] == conditions[plt_counter])]['angle'].values[0]

            rose_plot(ax[i][w], np.array(ang),color=color[conditions[plt_counter]])
            ax[i][w].set_title(conditions[plt_counter],fontsize=18)
            ax[i][w].text(0.7, 0.9,'total %i saccades'%(sum(~np.isnan(ang))), 
                               ha='center', va='center', transform=ax[i][w].transAxes,
                              fontsize = 15)
            plt_counter += 1

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir,'sacc_polar_hist_run-%s.png' %str(run).zfill(2)))







