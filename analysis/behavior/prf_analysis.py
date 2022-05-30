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

task = 'pRF'
base_dir = params['general']['current_dir']
ses_type = ['beh','func'] if base_dir == 'local' else ['beh']

out_dir = op.join(params['mri']['paths'][base_dir]['root'],'derivatives',
                  'behavioral','{task}'.format(task=task),
                        'sub-{sj}'.format(sj=sj))

# if output path doesn't exist, create it
if not os.path.isdir(out_dir): 
    os.makedirs(out_dir)
print('saving output files in %s'%out_dir)

# set type of bar pass per TR
bar_pass_all = []
for bar_pass_type in params['prf']['bar_pass_direction']:

    bar_pass_all = np.hstack((bar_pass_all,[bar_pass_type]*params['prf']['num_TRs'][bar_pass_type]))

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
        
        # get list of tsv files with prf events for run
        prf_files = [op.join(data_dir,x) for _,x in enumerate(os.listdir(data_dir)) if task in x
                         and x.endswith('_events.tsv')]; prf_files.sort()

        print('%i files found'%len(prf_files))

        # summarize results, for later plotting
        df_summary = pd.DataFrame(columns=['run', 'color_category', 'bar_color', 'accuracy', 'RT'])

        # for all runs
        for i,run in enumerate(prf_files):
            
            df_run = pd.read_csv(run, sep='\t')
            
            # trial numbers where participant responsed
            sub_response_trials = np.unique(df_run[df_run['event_type']=='response']['trial_nr'].values)
            
            # number of trials (= total #TRs)
            total_trials = df_run.iloc[-1]['trial_nr']+1
            
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
                
                if 'empty' not in (bar_pass_all[t]):
        
                    # find bar color in that trial
                    bar_color = [x for x in df_run.loc[df_run['trial_nr']==t]['event_type'].unique() if x in all_colors][0]
                    
                    if bar_color == 'color_green' or bar_color in task_colors['color_green']:
                        main_color = 'color_green'
                    elif bar_color == 'color_red' or bar_color in task_colors['color_red']:
                        main_color = 'color_red' 

                    # update total number of (potential) responses 
                    total_responses[main_color]+=1
                    total_responses[bar_color]+=1
                    
                    # save actual participant response
                    response_df = df_run[(df_run['trial_nr']==t)&(df_run['event_type']=='response')]

                    if len(response_df.values)>0: 
                        
                        # participant response key
                        sub_response = response_df['response'].values[0]

                        if sub_response in params['keys']['left_index'] and main_color == 'color_red':
                            
                            corr_responses[main_color]+=1
                            corr_responses[bar_color]+=1
                            
                            rt_responses[main_color].append(response_df['onset'].values[0] - t*TR)
                            rt_responses[bar_color].append(response_df['onset'].values[0] - t*TR)

                        elif sub_response in params['keys']['right_index'] and main_color == 'color_green':

                            corr_responses[main_color]+=1
                            corr_responses[bar_color]+=1
                            
                            rt_responses[main_color].append(response_df['onset'].values[0] - t*TR)
                            rt_responses[bar_color].append(response_df['onset'].values[0] - t*TR)
                

            # set this again, guarantee correct order
            all_colors = task_colors[color_categories[0]]+task_colors[color_categories[1]]

            df_summary = df_summary.append(pd.DataFrame({'run': np.repeat(run[-16:-11],len(all_colors)),
                                                        'color_category': np.repeat(color_categories,2),
                                                        'bar_color': all_colors,
                                                        'accuracy': [corr_responses[x]/total_responses[x] for x in all_colors],
                                                        'RT': [np.mean(rt_responses[x]) for x in all_colors]
                                                        }))
                                             
     
        # save accuracy and RT values
        df_summary.to_csv(op.join(out_dir,'sub-{sj}_task-{task}_acc_RT_{ses_type}.csv'.format(sj=sj, task=task, ses_type = ses)), index = False, header=True)

        # plot ACCURACY barplot and save
        fig, axs = plt.subplots(1, 2, figsize=(15,7.5))

        a = sns.barplot(x='color_category', y='accuracy', palette = params['plotting']['cond_colors'],
                    data=df_summary, capsize=.2, ax = axs[0])
        axs[0].tick_params(labelsize=15)
        axs[0].set_xlabel('Color Category',fontsize=15, labelpad = 20)
        axs[0].set_ylabel('Accuracy',fontsize=15, labelpad = 15)
        axs[0].set_title('pRF task',fontsize=18)

        b = sns.barplot(x='bar_color', y='accuracy', palette = params['plotting']['cond_colors'],
                    data=df_summary, capsize=.2, ax = axs[1])
        axs[1].tick_params(labelsize=15)
        axs[1].set_xlabel('Bar color',fontsize=15, labelpad = 20)
        axs[1].set_ylabel('Accuracy',fontsize=15, labelpad = 15)
        axs[1].set_title('pRF task',fontsize=18)

        fig.savefig(op.join(out_dir,'sub-{sj}_task-{task}_barplot-accuracy_across-runs_{ses_type}.png'.format(sj = sj, 
                                                                                                      task = task,
                                                                                                      ses_type = ses)))

        # plot ACCURACY barplot and save
        fig, axs = plt.subplots(1, 2, figsize=(15,7.5))

        a = sns.barplot(x='color_category', y='RT', palette = params['plotting']['cond_colors'],
                    data=df_summary, capsize=.2, ax = axs[0])
        axs[0].tick_params(labelsize=15)
        axs[0].set_xlabel('Color Category',fontsize=15, labelpad = 20)
        axs[0].set_ylabel('RT (s)',fontsize=15, labelpad = 15)
        axs[0].set_title('pRF task',fontsize=18)

        b = sns.barplot(x='bar_color', y='RT', palette = params['plotting']['cond_colors'],
                    data=df_summary, capsize=.2, ax = axs[1])
        axs[1].tick_params(labelsize=15)
        axs[1].set_xlabel('Bar color',fontsize=15, labelpad = 20)
        axs[1].set_ylabel('RT (s)',fontsize=15, labelpad = 15)
        axs[1].set_title('pRF task',fontsize=18)

        fig.savefig(op.join(out_dir,'sub-{sj}_task-{task}_barplot-RT_across-runs_{ses_type}.png'.format(sj = sj, 
                                                                                                            task = task,
                                                                                                            ses_type = ses)))
