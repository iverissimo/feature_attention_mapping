
# useful functions to use in other scripts

import os
import numpy as np
import colorsys


def get_pRF_cond_per_TR(cond_TR_dict, bar_pass_direction):
    
    """"
    helper function to get number of TRs per condition
    specifically for pRF task
    
    return list 
    """
    
     # list of bar orientation at all TRs
    condition_per_TR = []
    
    for _,bartype in enumerate(bar_pass_direction):
        
        condition_per_TR = np.concatenate((condition_per_TR, np.tile(bartype, cond_TR_dict[bartype])))
        
    return condition_per_TR


def get_pRF_trials_bar_color(run_df, total_trials = None,
                             color_dict = {'color_red': ['orange', 'pink'], 'color_green': ['yellow', 'blue']}):

    """"
    helper function to get bar colors 
    and color category for all trials of pRF run
    
    return list 
    """
    
    ## all possible colors
    all_colors = [element for sublist in color_dict.values() for element in sublist]
    
    # if no list of trials provided, then get values for each trial
    if total_trials is None:
        total_trials = run_df['trial_nr'].unique()
        
    bar_color = []
    category_color = []
    
    # find bar color in that trial
    for t in total_trials:

        trial_color = [x for x in run_df.loc[run_df['trial_nr']==t]['event_type'].unique() if x in all_colors]

        if len(trial_color)>0:
            bar_color.append(trial_color[0])
            category_color.append([k for k, v in color_dict.items() if trial_color[0] in v][0])
        else:
            bar_color.append(np.nan)
            category_color.append(np.nan)

    return category_color, bar_color


def get_FA_trials_bar_color(run_info_df, total_trials = None):

    """"
    helper function to get bar colors 
    and color category for all trials of FA run
    
    return dict for attended and unattended bar 
    """
    
    # if no list of trials provided, then get values for each trial
    if total_trials is None:
        total_trials = run_info_df['trial_num'].unique()
        
    bar_color = {'attend_bar': [], 'unattend_bar': []}
    category_color = {'attend_bar': [], 'unattend_bar': []}
    
    # find bar color in that trial
    for t in total_trials:
        
        # get colors for attended/UNattended bar
        for cond in bar_color.keys():
            
            if cond == 'attend_bar':
                bar_color[cond].append(run_info_df[run_info_df['trial_num']==t]['attend_task_color'].values[0]) 
                category_color[cond].append(run_info_df[run_info_df['trial_num']==t]['attend_color'].values[0])
            else:
                bar_color[cond].append(run_info_df[run_info_df['trial_num']==t]['unattend_task_color'].values[0]) 
                category_color[cond].append(run_info_df[run_info_df['trial_num']==t]['unattend_color'].values[0])
                
    return category_color, bar_color


def get_pp_response_bool(trial_df, trial_bar_color, task = 'pRF',
                        task_key_name = {'pRF':{'left_index': ['color_red'], 'right_index': ['color_green']},
                                        'FA':{'left_index': ['pink', 'blue'], 
                                              'right_index': ['orange', 'yellow']}},
                        keys = {'right_index': ['right','b', 2, '2','num_2'],
                                'left_index': ['left','e', 1, '1','num_1']}):
    
    """"
    helper function to get response bool for trial
    
    return True or False 
    """
        
    ## get value of key pressed
    key_val = trial_df[trial_df['event_type'] == 'response']['response'].values[0]

    ## get name of key pressed
    key_name = [k for k, v in keys.items() if key_val in v][0]

    response_bool = True if trial_bar_color in task_key_name[task][key_name] else False   

    
    return response_bool


def get_pp_response_rt(trial_df, task = 'pRF', TR = 1.6):
    
    """"
    helper function to get response RT for trial
    
    return reaction time value
    """
    
    if task == 'pRF':
        
        trial_nr = trial_df[trial_df['event_type'] == 'response']['trial_nr'].values[0]
        
        RT = trial_df[trial_df['event_type'] == 'response']['onset'].values[0] - trial_nr * TR
        
    elif task == 'FA':
        
        RT = trial_df[trial_df['event_type'] == 'response']['onset'].values[0] - trial_df[trial_df['event_type'] == 'stim']['onset'].values[0]
    
    return RT


def rgb255_2_hsv(arr):
    
    """ convert RGB 255 to HSV
    
    Parameters
    ----------
    arr: list/array
        1D list of rgb values
        
    """
    
    rgb_norm = np.array(arr)/255
    
    hsv_color = np.array(colorsys.rgb_to_hsv(rgb_norm[0],rgb_norm[1],rgb_norm[2]))
    hsv_color[0] = hsv_color[0] * 360
    
    return hsv_color