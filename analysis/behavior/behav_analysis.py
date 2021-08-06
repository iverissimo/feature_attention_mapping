
import numpy as np
import os, sys
import os.path as op
import yaml
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
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


task = 'FA'
base_dir = 'local'
ses = params['general']['session']

# path where I'm storing pilot behavioral data
#data_dir = '/Users/verissimo/Documents/Projects/Feature_based_attention_mapping/behavioral_pilot/data'
#data_dir = os.path.join(data_dir,'{task}'.format(task=task),'sub-{sj}'.format(sj=sj))
data_dir = op.join(params['mri']['paths'][base_dir], 'sourcedata','sub-{sj}'.format(sj=sj),
                        'ses-{ses}'.format(ses=ses),'func') 

# path to output for processed eye tracking files
#out_dir = '/Users/verissimo/Documents/Projects/Feature_based_attention_mapping/behavioral_pilot/outputs'
#out_dir = os.path.join(out_dir,'{task}'.format(task=task),'behavioral','sub-{sj}'.format(sj=sj))
out_dir = op.join(params['mri']['paths'][base_dir],'derivatives','behavioral','{task}'.format(task=task),
                        'sub-{sj}'.format(sj=sj),'ses-{ses}'.format(ses=ses))

# if output path doesn't exist, create it
if not os.path.isdir(out_dir): 
    os.makedirs(out_dir)
print('saving output files in %s'%out_dir)


# get absolute path to events tsv 
events_list = [os.path.join(data_dir,path) for _,path in enumerate(os.listdir(data_dir)) 
               if path.endswith('_events.tsv') and 'task-{task}'.format(task=task) in path]; events_list.sort()

# get absolute path to csv with bar positions for each run
bar_pos_list = [os.path.join(data_dir,path) for _,path in enumerate(os.listdir(data_dir)) 
               if path.endswith('_bar_positions.pkl') and 'task-{task}'.format(task=task) in path]; bar_pos_list.sort()

# get absolute path to csv with general infos for each run
trial_info_list = [os.path.join(data_dir,path) for _,path in enumerate(os.listdir(data_dir)) 
               if path.endswith('_trial_info.csv') and 'task-{task}'.format(task=task) in path]; trial_info_list.sort()


for run in range(len(trial_info_list)):

    # load trial info dataframe
    trial_info = pd.read_csv(trial_info_list[run]) 

    # get name of attended conditions per miniblock
    attend_condition = trial_info['attend_condition'].unique()

    # summary data frame to save relevant run info
    df_summary = pd.DataFrame(columns=['mini_block','attend_condition', 'hemifield','trial_ID','true_response'])

    # for each miniblock 
    for num in range(params['feature']['mini_blocks']):

        df_summary = df_summary.append({'mini_block': num, # miniblock number
                      'attend_condition': attend_condition[num], # attended condition 
                      'hemifield': trial_info.loc[trial_info['trial_type']=='mini_block_%s'%str(num)]['hemifield'].to_list(), # hemifield of attended bar
                      'trial_ID': trial_info.loc[trial_info['trial_type']=='mini_block_%s'%str(num)]['trial_num'].to_list(), # trial number for miniblocks
                      'true_response': get_true_response(trial_info.loc[trial_info['trial_type']=='mini_block_%s'%str(num)]['hemifield'].to_list())
                     }, ignore_index=True) 

    
    # load events csv, with behaviour data
    events = pd.read_csv(events_list[run],sep = '\t') 
    
    # summary data frame to save relevant events info
    df_ev_summary = pd.DataFrame(columns=['mini_block','attend_condition', 'response','response_onset','stim_onset'])

    # for each miniblock 
    for num in range(params['feature']['mini_blocks']):

        trial_ID_run = df_summary.loc[df_summary['mini_block']==num]['trial_ID'].values[0]

        responses, responses_onset = get_pp_responses(trial_ID_run,events,params)

        df_ev_summary = df_ev_summary.append({'mini_block': num, # miniblock number
                      'attend_condition': attend_condition[num], # attended condition 
                      'stim_onset': events.loc[(events['trial_nr'].isin(trial_ID_run))&(events['event_type']=='stim')]['onset'].to_list(), # onset of stimulus on screen                  
                      'response': responses, # participant responses
                      'response_onset': responses_onset # participant responses onset
                     }, ignore_index=True) 

        
    # calculate accuracy and RTs
    # for each miniblock

    # save in data frame 
    if run==0: # initialize if first run
        df_acc_rt = pd.DataFrame(columns=['run','mini_block','attend_condition', 'accuracy','RT'])

    for num in range(params['feature']['mini_blocks']):

        # correct trial indices, for that miniblock
        corr_trial_ind = np.where(df_ev_summary['response'][num] == df_summary['true_response'][num])[0]

        df_acc_rt = df_acc_rt.append({'run': run, # run number
                        'mini_block': num, # miniblock number
                        'attend_condition': attend_condition[num], # attended condition 
                        'accuracy': len(corr_trial_ind)/len(df_summary['true_response'][num]), 
                        'RT': np.take(np.array(df_ev_summary['response_onset'][num])-np.array(df_ev_summary['stim_onset'][num][1:]), 
                                    corr_trial_ind),
                        'RT_mean': np.mean(np.take(np.array(df_ev_summary['response_onset'][num])-np.array(df_ev_summary['stim_onset'][num][1:]), 
                                    corr_trial_ind))
                     }, ignore_index=True) 

    # save dataframe
    df_acc_rt.to_csv(os.path.join(out_dir,'sub-{sj}_task-{task}_acc_RT.csv'.format(sj=sj, task=task)), index = False, header=True)


# plot accuracy and reaction times
# per condition
# error bars for runs

fig, axs = plt.subplots(2,1, figsize=(10, 15), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .25, wspace=.001)

# sort conditions, to have bars in same order
attend_condition.sort()

# define costum palette
cond_colors = {'green_horizontal':(0,1,0), #(0,0.5412,0),
              'green_vertical':(0,1,0), #(0,0.5412,0),
              'red_horizontal':(1,0,0), #(0.6588,0,0),
              'red_vertical': (1,0,0)} #(0.6588,0,0)}

# Define some hatches
hatches = ['//']#['-', '|', '-', '|']

a = sns.barplot(ax = axs[0], x='attend_condition', y='accuracy', palette = cond_colors,
            data=df_acc_rt, capsize=.2, order = list(attend_condition))
#axs[0].set_title('Accuracy for all runs, sub-%s'%sj)
axs[0].set_xlabel('Attended condition',fontsize=18, labelpad = 15)
axs[0].set_ylabel('Accuracy',fontsize=18, labelpad = 15)
a.tick_params(labelsize=15)
a.set_ylim(0, 1)

# Loop over the bars
for i,thisbar in enumerate(a.patches):
    if 'vertical' in attend_condition[i]:
        # Set a different hatch for each bar
        thisbar.set_hatch(hatches[0])

b = sns.barplot(ax = axs[1], x='attend_condition', y='RT_mean', palette = cond_colors,
            data=df_acc_rt, capsize=.2, order = list(attend_condition))
#axs[1].set_title('RT')
axs[1].set_xlabel('Attended condition',fontsize=18, labelpad = 15)
axs[1].set_ylabel('RT (s)',fontsize=18, labelpad = 15)
b.tick_params(labelsize=15)
b.set_ylim(0, 1.2)

# Loop over the bars
for i,thisbar in enumerate(b.patches):
    if 'vertical' in attend_condition[i]:
        # Set a different hatch for each bar
        thisbar.set_hatch(hatches[0])

fig.savefig(os.path.join(out_dir,'sub-{sj}_task-{task}_acc_RT_all_runs.png'.format(sj=sj, task=task)))


# now plot same but over time
# to check if there's learning

fig, axs = plt.subplots(2,1, figsize=(10, 15), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .25, wspace=.001)

# get stats put plot in title
slope,intercept, _,_,_ = stats.linregress(df_acc_rt.reset_index()['index'].to_numpy(),
                                     y = df_acc_rt.reset_index()['accuracy'].to_numpy())

a = sns.regplot(ax=axs[0], x='index', y='accuracy',
           data=df_acc_rt.reset_index())
#axs[0].set_title('Accuracy for all runs, sub-%s'%sj)
axs[0].set_xlabel('Blocks',fontsize=18, labelpad = 15)
axs[0].set_ylabel('Accuracy',fontsize=18, labelpad = 15)
a.tick_params(labelsize=15)
axs[0].set_title('slope %0.3f, intercept %0.3f ' %(slope,intercept), fontsize=15)
a.set_ylim(.5, 1)

# get stats put plot in title
slope,intercept, _,_,_ = stats.linregress(df_acc_rt.reset_index()['index'].to_numpy(),
                                     y = df_acc_rt.reset_index()['RT_mean'].to_numpy())

b = sns.regplot(ax=axs[1], x='index', y='RT_mean',
           data=df_acc_rt.reset_index())
#axs[1].set_title('RT')
axs[1].set_xlabel('Blocks',fontsize=18, labelpad = 15)
axs[1].set_ylabel('RT (s)',fontsize=18, labelpad = 15)
b.tick_params(labelsize=15)
axs[1].set_title('slope %0.3f, intercept %0.3f ' %(slope,intercept), fontsize=15)
b.set_ylim(0.4, 1.2)

fig.savefig(os.path.join(out_dir,'sub-{sj}_task-{task}_acc_RT_through_time.png'.format(sj=sj, task=task)))




