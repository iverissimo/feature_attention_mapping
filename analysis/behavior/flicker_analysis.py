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

task = 'flicker'
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
        
        # get list of yml files with bar colors for each flicker trial
        flicker_files = [op.join(data_dir,x) for _,x in enumerate(os.listdir(data_dir)) if task in x and 'trial' in x 
                         and x.endswith('_updated_settings.yml')]
        
        color = []
        for f in flicker_files:
            # load flicker settings from yaml
            with open(f, 'r') as f_in:
                flicker_params = yaml.safe_load(f_in)
            
            # append modulated color values
            color.append(flicker_params[params['flicker']['modulated_condition'][0]]['element_color'])
        
        # color used in task
        color_used = np.mean(color, axis=0)
        
        # plot histogram and save
        fig, axs = plt.subplots(1, 1, sharex=True, figsize=(10,7.5))

        a = sns.histplot(x = [i for i in np.array(color).ravel() if i != 0],
                        color = tuple(color_used/255))
        a.tick_params(labelsize=15)
        a.set_xlabel(params['flicker']['modulated_condition'][0],fontsize=15, labelpad = 20)
        a.set_ylabel('Count',fontsize=15, labelpad = 15)
        a.set_title('Flicker task',fontsize=18)
        a.text(0.8, .9,r'RGB %s'%('[%.1f, %.1f, %.1f]'%(color_used[0],color_used[1],color_used[2])), 
                       ha='center', va='center', transform=axs.transAxes,
                      fontsize = 13)

        fig.savefig(op.join(out_dir,'sub-{sj}_task-{task}_hist-{color}_{ses_type}.png'.format(sj = sj, 
                                                                                              task = task,
                                                                                              color = params['flicker']['modulated_condition'][0],
                                                                                              ses_type = ses)))