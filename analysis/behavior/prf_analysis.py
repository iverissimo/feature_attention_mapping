import numpy as np
import os, sys
import os.path as op
import yaml
import pandas as pd

import glob

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

        # for all runs
        for i,run in enumerate(prf_files):
            
            df_run = pd.read_csv(run, sep='\t')
            
            # trial numbers where participant responsed
            sub_response_trials = np.unique(df_run[df_run['event_type']=='response']['trial_nr'].values)
            
            # number of trials (= total #TRs)
            total_trials = df_run.iloc[-1]['trial_nr']+1
            
            # to save number of participant responses, differentiating by color to check
            corr_responses = {'color_green': 0, 'color_red': 0}
            total_responses = {'color_green': 0, 'color_red': 0}
            
            counter = 0
            
            for t in range(total_trials):
                
                if 'empty' not in (bar_pass_all[t]):
                    
                    # find bar color in that trial
                    bar_color = [x for _,x in enumerate(df_run[df_run['trial_nr']==t]['event_type'].values) if x!='pulse' and x!='response'][0]

                    # update total number of responses
                    total_responses[bar_color]+=1

                    # if participant responded
                    if t<len(sub_response_trials) and t == sub_response_trials[counter]:

                        # participant response key
                        sub_response = df_run[(df_run['trial_nr']==t)&(df_run['event_type']=='response')]['response'].values[0]

                        if ((bar_color == 'color_red') and (sub_response in params['keys']['left_index'])) \
                           or ((bar_color == 'color_green') and (sub_response in params['keys']['right_index'])):
                            corr_responses[bar_color]+=1

                        counter+=1

            # summarize results, for later plotting
            if i==0:
                df_summary = pd.DataFrame({'run': np.repeat(run[-16:-11],3),
                                           'condition': np.array(['color_green', 'color_red', 'total']),
                                           'accuracy': [corr_responses['color_green']/total_responses['color_green'],
                                                       corr_responses['color_red']/total_responses['color_red'],
                                                       sum(corr_responses.values())/sum(total_responses.values())]})
            else:
                df_summary = df_summary.append(pd.DataFrame({'run': np.repeat(run[-16:-11],3),
                                           'condition': np.array(['color_green', 'color_red', 'total']),
                                           'accuracy': [corr_responses['color_green']/total_responses['color_green'],
                                                       corr_responses['color_red']/total_responses['color_red'],
                                                       sum(corr_responses.values())/sum(total_responses.values())]}),
                                              ignore_index=True)
     
        # plot barplot and save
        fig, axs = plt.subplots(1, 1, sharex=True, figsize=(10,7.5))

        a = sns.barplot(x='condition', y='accuracy', palette=['red','green','yellow'],
                    data=df_summary, capsize=.2, order = ['color_red','color_green','total'])
        a.tick_params(labelsize=15)
        a.set_xlabel('bar color',fontsize=15, labelpad = 20)
        a.set_ylabel('Accuracy',fontsize=15, labelpad = 15)
        a.set_title('pRF task',fontsize=18)


        fig.savefig(op.join(out_dir,'sub-{sj}_task-{task}_barplot-across-runs_{ses_type}.png'.format(sj = sj, 
                                                                                                      task = task,
                                                                                                      ses_type = ses)))

