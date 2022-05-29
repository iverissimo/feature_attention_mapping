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

# reference color used
ref_color = params['flicker']['ref_color']

# updated color names
if ref_color in params['general']['color_categories']: # if comparing red and green
    updated_color_names = [c for c in params['general']['color_categories'] if c != ref_color]  
else:
    updated_color_names = []
    for key in params['general']['color_categories']:
        for name in params['general']['task_colors'][key]:
            if name != ref_color:
                updated_color_names.append(name) # update all task colors that are not reference color

# eccentricity indices we ran
ecc_ind = params['flicker']['bar_ecc_index'] 

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

        # get list of yml settings files for all trials of flicker task
        flicker_files = [op.join(data_dir,x) for _,x in enumerate(os.listdir(data_dir)) if 'trial' in x and x.endswith('_updated_settings.yml')]
        all_trials = []

        # go through colors we updated
        for col in updated_color_names:
        
            new_color = []

            # loop over eccentricities
            for e in ecc_ind:
                # filenames for that color and ecc
                c_files = [file for file in flicker_files if col in file and 'ecc-%i'%e in file]

                if len(c_files) == 0:
                    print('No files found for color %s and ecc %i, keeping initial settings'%(col, e))
                else:
                    ecc_color = []
                    for file in c_files:
                        # load updated settings for each trial 
                        with open(file, 'r', encoding='utf8') as f_in:
                            updated_settings = yaml.safe_load(f_in)

                        if col in params['general']['color_categories']: # if general color category (red, green)
                            ecc_color.append(updated_settings[col]['element_color'])
                        
                        elif col in ['pink','orange']: # if color variant from red
                            ecc_color.append(updated_settings['color_red']['task_color'][col]['element_color'])
                        
                        elif col in ['yellow','blue']: # if color variant from red
                            ecc_color.append(updated_settings['color_green']['task_color'][col]['element_color'])
                    
                    # new color is average across eccs
                    new_color.append(list(np.mean(ecc_color, axis=0)))
                
                    # also save all trials to check for differences
                    all_trials.append(ecc_color)

        ## save all this in dataframe, for easier plotting

        # tile the keys, to make it easier to make dataframe
        color_names = np.repeat(updated_color_names, len(ecc_ind))
        
        all_ecc_dict = {'color': [], 'ecc': [], 'R': [], 'G': [], 'B': [], 'luminance': []}
        for i, name in enumerate(color_names):
            if name == 'orange':
                ecc = i
            elif name == 'yellow':
                ecc = i-3
            elif name == 'blue':
                ecc = i-3*2
            
            for t in range(np.array(all_trials[i]).shape[0]):

                all_ecc_dict['color'].append(name)
                all_ecc_dict['ecc'].append(int(ecc))
                all_ecc_dict['R'].append(np.array(all_trials[i])[...,0][t])
                all_ecc_dict['G'].append(np.array(all_trials[i])[...,1][t])
                all_ecc_dict['B'].append(np.array(all_trials[i])[...,2][t])
                all_ecc_dict['luminance'].append(beh_utils.rgb255_2_hsv(np.array(all_trials[i])[t])[-1])
            
        # add also reference color, for comparison - SHOULD GENERALIZE; THIS WILL BREAK WITH DIFFERENT REF COLOR
        for e in ecc_ind:
            all_ecc_dict['color'].append(ref_color)
            all_ecc_dict['ecc'].append(e) # doesnt matter, its the same for all ecc
            all_ecc_dict['R'].append(updated_settings['color_red']['task_color'][ref_color]['element_color'][0])
            all_ecc_dict['G'].append(updated_settings['color_red']['task_color'][ref_color]['element_color'][1])
            all_ecc_dict['B'].append(updated_settings['color_red']['task_color'][ref_color]['element_color'][2])
            all_ecc_dict['luminance'].append(beh_utils.rgb255_2_hsv(np.array(updated_settings['color_red']['task_color'][ref_color]['element_color']))[-1])
                
        # convert to dataframe
        df_colors = pd.DataFrame(all_ecc_dict)

        

        
        ## NOW PLOT

        # Bar plot with luminance values, per ecc and color
        fig, axs = plt.subplots(1, 1, sharex=True, figsize=(10,7.5))

        a = sns.barplot(x = 'color', y = 'luminance', data = df_colors, hue = 'ecc')
        #sns.swarmplot(x="color", y="luminance", hue='ecc', data=df_colors)
        a.tick_params(labelsize=15)
        a.set_xlabel('Color',fontsize=15, labelpad = 20)
        a.set_ylabel('Luminance Value (HSV)',fontsize=15, labelpad = 15)
        a.set_title('Flicker task - reference color %s'%ref_color, fontsize=18)
        fig.savefig(op.join(out_dir,"luminance_across_ecc.png"))

        fig.savefig(op.join(out_dir,'sub-{sj}_task-{task}_ref2-{color}_barplot-luminance_{ses_type}.png'.format(sj = sj, 
                                                                                                    task = task,
                                                                                                    color = ref_color,
                                                                                                    ses_type = ses)))
        # plot RGB values per color
        fig, axs = plt.subplots(1, 3, figsize=(15,5))
        
        sns.boxplot(x = 'color', y = 'R', data = df_colors, hue = 'ecc', ax=axs[0])
        sns.swarmplot(x = 'color', y = 'R', data = df_colors, color = 'k', alpha = .3, ax=axs[0])

        sns.boxplot(x = 'color', y = 'G', data = df_colors, hue = 'ecc', ax=axs[1])
        sns.swarmplot(x = 'color', y = 'G', data = df_colors, color = 'k', alpha = .3, ax=axs[1])

        sns.boxplot(x = 'color', y = 'B', data = df_colors, hue = 'ecc', ax=axs[2])
        sns.swarmplot(x = 'color', y = 'B', data = df_colors, color = 'k', alpha = .3, ax=axs[2])

        #axs.tick_params(labelsize=15)

        fig.savefig(op.join(out_dir,'sub-{sj}_task-{task}_ref2-{color}_barplot-RGB_{ses_type}.png'.format(sj = sj, 
                                                                                                    task = task,
                                                                                                    color = ref_color,
                                                                                                    ses_type = ses)))
     