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
        
        
        # get absolute path to csv with bar positions for each run
        bar_pos_list = [op.join(data_dir,x) for _,x in enumerate(os.listdir(data_dir)) if task in x
                       if x.endswith('_bar_positions.pkl')]; bar_pos_list.sort()

        # get absolute path to csv with general infos for each run
        trial_info_list = [op.join(data_dir,x) for _,x in enumerate(os.listdir(data_dir)) if task in x
                       if x.endswith('_trial_info.csv')]; trial_info_list.sort()

        ## get hemifield for all bars in miniblock, for an extra analysis
        df_hemifield = pd.DataFrame(columns=['run','mini_block','attend_condition', 'condition','hemifield','true_response'])

        # for all runs
        for i,run in enumerate(FA_files):
            
            # dataframe with events for run
            df_run = pd.read_csv(run, sep='\t')
            
            # load trial info dataframe
            trial_info = pd.read_csv(trial_info_list[i])

            # load bar positions for run
            bar_pos = pd.read_pickle(bar_pos_list[i]) 
            
            # get name of attended conditions per miniblock
            attend_condition = trial_info['attend_condition'].unique()
            
            ## summary data frame to save relevant run info
            df_summary = pd.DataFrame(columns=['mini_block','attend_condition', 'hemifield','trial_ID','true_response'])
  
            # for each miniblock in run
            for num in range(params['feature']['mini_blocks']):

                df_summary = df_summary.append({'mini_block': num, # miniblock number
                              'attend_condition': attend_condition[num], # attended condition 
                              'hemifield': trial_info.loc[trial_info['trial_type']=='mini_block_%s'%str(num)]['hemifield'].to_list(), # hemifield of attended bar
                              'trial_ID': trial_info.loc[trial_info['trial_type']=='mini_block_%s'%str(num)]['trial_num'].to_list(), # trial number for miniblocks
                              'true_response': beh_utils.get_true_response(trial_info.loc[trial_info['trial_type']=='mini_block_%s'%str(num)]['hemifield'].to_list())
                             }, ignore_index=True) 

                # save hemifield info
                for c in params['feature']['conditions']:
                    # get info for miniblock
                    min_cond = bar_pos[(bar_pos['mini_block']==num)&(bar_pos['condition']==c)]  
                    
                    # get hemifield bar was in
                    hemi = []
                    for _, val in enumerate(min_cond['bar_midpoint_at_TR'].values[0]):

                        if 'horizontal' in c:
                            pos = 'up' if val[-1]>0 else 'down'
                        elif 'vertical' in c:
                            pos = 'right' if val[0]>0 else 'left'

                        hemi.append(pos)
                        
                    # get "true" responses for that condition
                    tr_condition = beh_utils.get_true_response(hemi)

                    df_hemifield = df_hemifield.append({'run': i+1, # run number
                                            'mini_block': int(num), # miniblock number
                                            'attend_condition': np.bool(min_cond['attend_condition'].values[0]), # was condition attended condition?
                                            'condition': c, # condition                 
                                            'hemifield': hemi, # hemifield of condition, per trial
                                            'true_response': tr_condition # true response, for condition
                                            }, ignore_index=True) 
                    

            ## summary data frame to save relevant events info
            
            df_ev_summary = pd.DataFrame(columns=['mini_block','attend_condition', 'response','response_onset','stim_onset'])

            # for each miniblock 
            for num in range(params['feature']['mini_blocks']):

                trial_ID_run = df_summary.loc[df_summary['mini_block']==num]['trial_ID'].values[0]

                responses, responses_onset = beh_utils.get_pp_responses(trial_ID_run, df_run, params)

                df_ev_summary = df_ev_summary.append({'mini_block': num, # miniblock number
                              'attend_condition': attend_condition[num], # attended condition 
                              'stim_onset': df_run.loc[(df_run['trial_nr'].isin(trial_ID_run))&(df_run['event_type']=='stim')]['onset'].to_list(), # onset of stimulus on screen                  
                              'response': responses, # participant responses
                              'response_onset': responses_onset # participant responses onset
                             }, ignore_index=True) 
                
            # calculate accuracy and RTs
            # for each miniblock

            # save in data frame 
            if i==0: # initialize if in first run
                df_acc_rt = pd.DataFrame(columns=['run','mini_block','attend_condition', 'condition', 'accuracy','RT'])

            for num in range(params['feature']['mini_blocks']):
                
                for c in params['feature']['conditions']:
                    
                    # temporary dataframe, just to simplify
                    tmp_df = df_hemifield[(df_hemifield['run']==i+1)&(df_hemifield['condition']==c)&(df_hemifield['mini_block']==num)]

                    # correct trial indices, for that miniblock and condition
                    corr_trial_ind = np.where(df_ev_summary['response'][num] == tmp_df['true_response'].values[0])[0]
                    
                    if tmp_df['attend_condition'].values[0] == True:
                        rt = np.take(np.array(df_ev_summary['response_onset'][num])-np.array(df_ev_summary['stim_onset'][num][1:]), 
                                                corr_trial_ind)
                        rt_mean = np.mean(np.take(np.array(df_ev_summary['response_onset'][num])-np.array(df_ev_summary['stim_onset'][num][1:]), 
                                                corr_trial_ind))
                    else:
                        rt = np.nan
                        rt_mean = np.nan

                    df_acc_rt = df_acc_rt.append({'run': i+1, # run number
                                    'mini_block': num, # miniblock number
                                    'attend_condition': tmp_df['attend_condition'].values[0], # attended condition 
                                    'condition':c, 
                                    'accuracy': len(corr_trial_ind)/len(df_summary['true_response'][num]), 
                                    'RT': rt,
                                    'RT_mean': rt_mean
                                 }, ignore_index=True)  

        # save accuracy and RT values
        df_acc_rt.to_csv(op.join(out_dir,'sub-{sj}_task-{task}_acc_RT_{ses_type}.csv'.format(sj=sj, task=task, ses_type = ses)), index = False, header=True)

        # plot accuracy and reaction times
        # per condition
        # error bars for runs

        df_attended = df_acc_rt[df_acc_rt['attend_condition']==True].copy()

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

        a = sns.barplot(ax = axs[0], x='condition', y='accuracy', palette = cond_colors,
                    data=df_attended, capsize=.2, order = list(attend_condition))
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

        b = sns.barplot(ax = axs[1], x='condition', y='RT_mean', palette = cond_colors,
                    data=df_attended, capsize=.2, order = list(attend_condition))
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

        fig.savefig(op.join(out_dir,'sub-{sj}_task-{task}_acc_RT_all_runs_{ses_type}.png'.format(sj=sj, task=task, ses_type = ses)))

        # now plot same but over time
        # to check if there's learning

        fig, axs = plt.subplots(2,1, figsize=(10, 15), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .25, wspace=.001)

        # get stats put plot in title
        slope,intercept, _,_,_ = stats.linregress(df_attended.reset_index()['index'].to_numpy(),
                                            y = df_attended.reset_index()['accuracy'].to_numpy())

        a = sns.regplot(ax=axs[0], x='index', y='accuracy',
                data=df_attended.reset_index())
        
        #axs[0].set_title('Accuracy for all runs, sub-%s'%sj)
        axs[0].set_xlabel('Blocks',fontsize=18, labelpad = 15)
        axs[0].set_ylabel('Accuracy',fontsize=18, labelpad = 15)
        a.tick_params(labelsize=15)
        axs[0].set_title('slope %0.3f, intercept %0.3f ' %(slope,intercept), fontsize=15)
        a.set_ylim(.5, 1)
        #axs[0].set_xticks(range(len(df_attended.reset_index()['index'].to_numpy()))) 
        #axs[0].set_xticklabels(range(0, len(df_attended.reset_index()['index'].to_numpy())))

        # get stats put plot in title
        slope,intercept, _,_,_ = stats.linregress(df_attended.reset_index()['index'].to_numpy(),
                                            y = df_attended.reset_index()['RT_mean'].to_numpy())

        b = sns.regplot(ax=axs[1], x='index', y='RT_mean',
                data=df_attended.reset_index())
        
        #b.set_xticks(range(len(df_attended.reset_index()['index'].to_numpy())))
        #b.set_xticklabels(range(0, len(df_attended.reset_index()['index'].to_numpy())))
        #axs[1].set_title('RT')
        axs[1].set_xlabel('Blocks',fontsize=18, labelpad = 15)
        axs[1].set_ylabel('RT (s)',fontsize=18, labelpad = 15)
        b.tick_params(labelsize=15)
        axs[1].set_title('slope %0.3f, intercept %0.3f ' %(slope,intercept), fontsize=15)
        b.set_ylim(0.4, 1.2)

        fig.savefig(op.join(out_dir,'sub-{sj}_task-{task}_acc_RT_through_time_{ses_type}.png'.format(sj=sj, task=task, ses_type = ses)))


        # over miniblocks, because why not
        # plot the dataframe
        df_attended['run'] = df_attended['run'].astype(int)
        df_attended['mini_block'] = df_attended['mini_block'].astype(int)

        a = sns.lmplot(data = df_attended.reset_index(), x = 'mini_block', y = 'accuracy', hue='run', ci=None)

        b = sns.lmplot(data = df_attended.reset_index(), x = 'mini_block', y = 'RT_mean', hue='run', ci=None)

        a.savefig(op.join(out_dir,'sub-{sj}_task-{task}_acc_through_miniblocks_{ses_type}.png'.format(sj=sj, task=task, ses_type = ses)))
        b.savefig(op.join(out_dir,'sub-{sj}_task-{task}_RT_through_miniblocks_{ses_type}.png'.format(sj=sj, task=task, ses_type = ses)))

        
        ### CHECK IF OTHER CONDITIONS HAVE LOWER ACCURACY, for a same miniblock
        # figure
        fig, axs = plt.subplots(4,1, figsize=(10, 40), facecolor='w', edgecolor='k')
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

        # for each condition
        for w,cond in enumerate(attend_condition):
            
            # choose minibloks where attended condition was green
            mn_blk = df_acc_rt[(df_acc_rt['attend_condition']==True)&(df_acc_rt['condition']==cond)]['mini_block'].values
            runs = df_acc_rt[(df_acc_rt['attend_condition']==True)&(df_acc_rt['condition']==cond)]['run'].values

            for indx,r in enumerate(runs):

                if indx == 0:
                    plt_df = df_acc_rt[(df_acc_rt['mini_block']==mn_blk[indx])&(df_acc_rt['run']==r)]
                else:
                    plt_df = plt_df.append(df_acc_rt[(df_acc_rt['mini_block']==mn_blk[indx])&(df_acc_rt['run']==r)])


            
            a = sns.barplot(ax = axs[w], x='condition', y='accuracy', palette = cond_colors,
                        data=plt_df, capsize=.2, order = list(attend_condition))

            axs[w].set_title('Attended condition %s'%cond, fontsize=18)
            axs[w].set_xlabel('Conditions',fontsize=18, labelpad = 15)
            axs[w].set_ylabel('Accuracy',fontsize=18, labelpad = 15)
            a.tick_params(labelsize=15)
            a.set_ylim(0, 1)

            # Loop over the bars
            for i,thisbar in enumerate(a.patches):
                if 'vertical' in attend_condition[i]:
                    # Set a different hatch for each bar
                    thisbar.set_hatch(hatches[0])

        fig.savefig(op.join(out_dir,'sub-{sj}_task-{task}_acc_all_conditions_{ses_type}.png'.format(sj=sj, task=task, ses_type = ses)))