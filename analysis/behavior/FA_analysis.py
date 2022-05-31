import numpy as np
import os, sys
import os.path as op
import yaml
import pandas as pd

import glob

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from FAM_utils import beh as beh_utils

# load settings from yaml
with open(op.join(op.split(os.getcwd())[0],'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

# define participant number
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex: 100)'
                    'as 1st argument in the command line!')

else:
    sj = str(sys.argv[1]).zfill(3) #fill subject number with 00 in case user forgets

task = 'FA'
base_dir = params['general']['current_dir']
ses_type = ['beh','func'] if base_dir == 'local' else ['beh']

out_dir = op.join(params['mri']['paths'][base_dir]['root'],'derivatives',
                  'behavioral','{task}'.format(task=task),
                        'sub-{sj}'.format(sj=sj))

# if output path doesn't exist, create it
if not os.path.isdir(out_dir): 
    os.makedirs(out_dir)
print('saving output files in %s'%out_dir)

# conditions (colors)
color_categories = params['general']['color_categories']
task_colors = params['general']['task_colors']
all_colors = [element for sublist in task_colors.values() for element in sublist] 

TR = params['mri']['TR']

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
        
        # get list of tsv files with FA events for run
        FA_files = [op.join(data_dir,x) for _,x in enumerate(os.listdir(data_dir)) if task in x
                         and x.endswith('_events.tsv')]; FA_files.sort()

        print('%i files found'%len(FA_files))
        
        # get absolute path to csv with general infos for each run
        trial_info_list = [op.join(data_dir,x) for _,x in enumerate(os.listdir(data_dir)) if task in x
                    if x.endswith('_trial_info.csv')]; trial_info_list.sort()

        # summarize results, for later plotting
        df_summary = pd.DataFrame(columns=['run', 'att_color', 'att_bar_color', 'att_accuracy', 'RT', 
                                        'unatt_color', 'unatt_bar_color', 'unatt_accuracy'])

        # for all runs
        for i,run in enumerate(FA_files):

            # dataframe with events for run
            df_run = pd.read_csv(run, sep='\t')

            # load trial info dataframe
            trial_info = pd.read_csv(trial_info_list[i])

            # trial numbers where participant responsed
            sub_response_trials = np.unique(df_run[df_run['event_type']=='response']['trial_nr'].values)

            # number of trials (= total #TRs)
            total_trials = df_run.iloc[-1]['trial_nr']+1

            # get name of attended color category
            attend_condition = trial_info['attend_color'].unique()[0]
            # and unattended one
            unattend_condition = trial_info['unattend_color'].unique()[0]

            # to save number of participant responses, differentiating by color to check
            corr_responses = {}
            total_responses = {}
            rt_responses = {}

            for cc in color_categories:
                corr_responses[cc] = 0
                total_responses[cc] = 0
                rt_responses[cc] = []
            for ac in all_colors:
                corr_responses[ac] = 0
                total_responses[ac] = 0
                rt_responses[ac] = []


            for t in range(total_trials):

                if 'task' in trial_info['trial_type'][t]:

                    # find att bar color in that trial
                    att_bar_color = trial_info['attend_task_color'][t] 
                    # and unatended one
                    unatt_bar_color = trial_info['unattend_task_color'][t] 

                    # update total number of (potential) responses 
                    total_responses[att_bar_color]+=1
                    total_responses[unatt_bar_color]+=1

                    total_responses[attend_condition]+=1

                    # save actual participant response
                    response_df = df_run[(df_run['trial_nr']==t)&(df_run['event_type']=='response')]

                    if len(response_df.values)>0: 

                        # participant response key
                        sub_response = response_df['response'].values[0]

                        if sub_response in params['keys']['left_index']:

                            if att_bar_color in ['pink', 'blue']:
                                corr_responses[att_bar_color]+=1
                                corr_responses[attend_condition]+=1

                                rt_responses[att_bar_color].append(response_df['onset'].values[0] - t*TR)
                                rt_responses[attend_condition].append(response_df['onset'].values[0] - t*TR)

                            if unatt_bar_color in ['pink', 'blue']:
                                corr_responses[unatt_bar_color]+=1
                                rt_responses[unatt_bar_color].append(response_df['onset'].values[0] - t*TR)


                        elif sub_response in params['keys']['right_index']:

                            if att_bar_color in ['orange', 'yellow']:
                                corr_responses[att_bar_color]+=1
                                corr_responses[attend_condition]+=1

                                rt_responses[att_bar_color].append(response_df['onset'].values[0] - t*TR)
                                rt_responses[attend_condition].append(response_df['onset'].values[0] - t*TR)

                            if unatt_bar_color in ['orange', 'yellow']:
                                corr_responses[unatt_bar_color]+=1
                                rt_responses[unatt_bar_color].append(response_df['onset'].values[0] - t*TR)


            df_summary = df_summary.append(pd.DataFrame({'run': np.repeat(run[-16:-11], 2),
                                                        'att_color': np.repeat(attend_condition, 2),
                                                        'att_bar_color': np.array(task_colors[attend_condition]),
                                                        'att_accuracy': [corr_responses[x]/total_responses[x] for x in task_colors[attend_condition]],
                                                        'RT': [np.mean(rt_responses[x]) for x in task_colors[attend_condition]],
                                                        'unatt_color': np.repeat(unattend_condition, 2), 
                                                        'unatt_bar_color': np.array(task_colors[unattend_condition]),
                                                        'unatt_accuracy': [corr_responses[x]/total_responses[x] for x in task_colors[unattend_condition]]
                                                        }))   
            
        # save accuracy and RT values
        df_summary.to_csv(op.join(out_dir,'sub-{sj}_task-{task}_acc_RT_{ses_type}.csv'.format(sj=sj, task=task, ses_type = ses)), index = False, header=True)

        # plot ACCURACY barplot and save
        fig, axs = plt.subplots(1, 3, figsize=(22.5,7.5))

        a = sns.barplot(x='att_color', y='att_accuracy', palette = params['plotting']['cond_colors'],
                    data = df_summary, capsize=.2, ax = axs[0])
        axs[0].tick_params(labelsize=15)
        axs[0].set_xlabel('Color Category',fontsize=15, labelpad = 20)
        axs[0].set_ylabel('Accuracy',fontsize=15, labelpad = 15)
        axs[0].set_title('FA Accuracy (across runs)',fontsize=18)

        b = sns.barplot(x='att_bar_color', y='att_accuracy', palette = params['plotting']['cond_colors'],
                    data=df_summary, capsize=.2, ax = axs[1])
        axs[1].tick_params(labelsize=15)
        axs[1].set_xlabel('Bar color',fontsize=15, labelpad = 20)
        axs[1].set_ylabel('Accuracy',fontsize=15, labelpad = 15)
        axs[1].set_title('FA Attended Accuracy',fontsize=18)

        c = sns.barplot(x='unatt_bar_color', y='unatt_accuracy', palette = params['plotting']['cond_colors'],
                    data=df_summary, capsize=.2, ax = axs[2])
        axs[2].tick_params(labelsize=15)
        axs[2].set_xlabel('Bar color',fontsize=15, labelpad = 20)
        axs[2].set_ylabel('Accuracy',fontsize=15, labelpad = 15)
        axs[2].set_title('FA Unattended Accuracy',fontsize=18)

        fig.savefig(op.join(out_dir,'sub-{sj}_task-{task}_barplot-accuracy_across-runs_{ses_type}.png'.format(sj = sj, 
                                                                                                            task = task,
                                                                                                            ses_type = ses)))

        # plot RT barplot and save
        fig, axs = plt.subplots(1, 2, figsize=(15,7.5))

        a = sns.barplot(x='att_color', y='RT', palette = params['plotting']['cond_colors'],
                    data = df_summary, capsize=.2, ax = axs[0])
        axs[0].tick_params(labelsize=15)
        axs[0].set_xlabel('Color Category',fontsize=15, labelpad = 20)
        axs[0].set_ylabel('RT (s)',fontsize=15, labelpad = 15)
        axs[0].set_title('FA RT (across runs)',fontsize=18)

        b = sns.barplot(x='att_bar_color', y='RT', palette = params['plotting']['cond_colors'],
                    data=df_summary, capsize=.2, ax = axs[1])
        axs[1].tick_params(labelsize=15)
        axs[1].set_xlabel('Bar color',fontsize=15, labelpad = 20)
        axs[1].set_ylabel('RT (s)',fontsize=15, labelpad = 15)
        axs[1].set_title('FA RT (per attended bar color)',fontsize=18)

        fig.savefig(op.join(out_dir,'sub-{sj}_task-{task}_barplot-RT_across-runs_{ses_type}.png'.format(sj = sj, 
                                                                                                        task = task,
                                                                                                        ses_type = ses)))


                                                                                                            
                                                                                                                     