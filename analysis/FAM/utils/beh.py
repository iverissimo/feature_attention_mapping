
# useful functions to use in other scripts
import numpy as np
import re
import os
import os.path as op
import pandas as pd

from FAM.utils.general import Utils

class BehUtils(Utils):

    def __init__(self):
        
        """__init__
        constructor for utilities behavioral class 
            
        """

    def get_pRF_cond_per_TR(self, cond_TR_dict = {}, bar_pass_direction = []):
        
        """"
        get number of TRs per condition (bar pass), for pRF task

        Parameters
        ----------
        cond_TR_dict : dict
            dict with number of TRs for each design conditions (bar pass L-R, R-L, empty, etc)
        bar_pass_direction : list/array
            order of events during run

        """
        
        # list of bar orientation at all TRs
        condition_per_TR = []
        
        for _,bartype in enumerate(bar_pass_direction):
            
            condition_per_TR = np.concatenate((condition_per_TR, np.tile(bartype, cond_TR_dict[bartype])))
            
        return condition_per_TR


    def get_pRF_trials_bar_color(self, run_df, total_trials = None,
                                color_dict = {'color_red': ['orange', 'pink'], 'color_green': ['yellow', 'blue']}):

        """"
        get bar colors and color category for all trials of pRF run

        Parameters
        ----------
        run_df : DataFrame
            dataframe with run events and relevant information
        total_trials: list
            list with trial ids to get bar info from
        color_dict: dict
            dict with category color and bar possible colors

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


    def get_FA_trials_bar_color(self, run_df, total_trials = None):

        """"
        get bar colors and color category for all trials of FA run
        
        returns dict for attended and unattended bar 

        Parameters
        ----------
        run_df : DataFrame
            dataframe with run events and relevant information
        total_trials: list
            list with trial ids to get bar info from

        """
        
        # if no list of trials provided, then get values for each trial
        if total_trials is None:
            total_trials = run_df['trial_num'].unique()
            
        bar_color = {'attend_bar': [], 'unattend_bar': []}
        category_color = {'attend_bar': [], 'unattend_bar': []}
        
        # find bar color in that trial
        for t in total_trials:
            
            # get colors for attended/UNattended bar
            for cond in bar_color.keys():
                
                if cond == 'attend_bar':
                    bar_color[cond].append(run_df[run_df['trial_num']==t]['attend_task_color'].values[0]) 
                    category_color[cond].append(run_df[run_df['trial_num']==t]['attend_color'].values[0])
                else:
                    bar_color[cond].append(run_df[run_df['trial_num']==t]['unattend_task_color'].values[0]) 
                    category_color[cond].append(run_df[run_df['trial_num']==t]['unattend_color'].values[0])
                    
        return category_color, bar_color


    def get_pp_response_bool(self, trial_df, trial_bar_color = '', task = 'pRF',
                            task_key_name = {'pRF':{'left_index': ['color_red'], 'right_index': ['color_green']},
                                            'FA':{'left_index': ['pink', 'blue'], 'right_index': ['orange', 'yellow']}},
                            keys = {'right_index': ['right','b', 2, '2','num_2'],
                                    'left_index': ['left','e', 1, '1','num_1']}):
        
        """"
        get response bool for trial

        Parameters
        ----------
        trial_df : DataFrame
            dataframe with trial events and relevant information
        trial_bar_color: str
            bar color name/category
        task: str
            task name (pRF vs FA)
        task_key_name: dict
            dict with bar color name/category for each response key
        keys: dict
            possible keys used for response (for keyboard, button box etc)

        """
            
        ## get value of key pressed
        key_val = trial_df[trial_df['event_type'] == 'response']['response'].values[-1]

        ## get name of key pressed
        key_name = [k for k, v in keys.items() if key_val in v][0]

        response_bool = True if trial_bar_color in task_key_name[task][key_name] else False   

        return response_bool


    def get_pp_response_rt(self, trial_df, task = 'pRF', TR = 1.6):
        
        """"
        get response RT for trial

        Parameters
        ----------
        trial_df : DataFrame
            dataframe with trial events and relevant information
        task: str
            task name (pRF vs FA)
        TR: float
            TR

        """
        
        if task == 'pRF':
            trial_nr = trial_df[trial_df['event_type'] == 'response']['trial_nr'].values[-1]
            RT = trial_df[trial_df['event_type'] == 'response']['onset'].values[-1] - trial_nr * TR
            
        elif task == 'FA':
            RT = trial_df[trial_df['event_type'] == 'response']['onset'].values[-1] - trial_df[trial_df['event_type'] == 'stim']['onset'].values[0]
        
        return RT


    def get_FA_run_struct(self, bar_pass_direction, num_bar_pos = [6,6], empty_TR = 20, task_trial_TR = 2):
        
        """ get FA run general structure
        This is which trials where task trials etc
        
        Parameters
        ----------
        bar_pass_direction: array/list
            list with order of "type of stimuli" throughout experiment (empty vs stim)
        num_bar_pos: list
            number of potential bar positions per axis
        empty_TR: int
            number of TRs for empty intervals of experiment
        task_trial_TR: int
            number of TRs for task trials of experiment (bar presentation + ITI)

        """
        
        ## 
        # define number of possible bar positions per axis
        num_bar_pos = np.array(num_bar_pos)
        
        ## total number of bar on screen trials (task trials)
        #6 AC vertical * 6 UC horizontal (36) +
        #6 AC vertical * 5 UC vertical (30) +
        #6 AC horizontal * 6 UC vertical (36)
        #6 AC horizontal * 5 UC horizontal (30) = 132 trials
        #
        num_task_trials = 2 * (num_bar_pos[0] * num_bar_pos[1] + num_bar_pos[0] * (num_bar_pos[1] - 1))
        
        # list of str with trial type at all TRs
        trial_type_all = [] 
        
        for _,bartype in enumerate(bar_pass_direction):
            
            if bartype in np.array(['empty']): # empty screen
                
                trial_type_all = trial_type_all + np.repeat(bartype,empty_TR).tolist()
            
            elif 'task' in bartype: # bars on screen
                
                # NOTE one feature trial is 1TR of bar display + 1TR of empty screen
                trial_type_all = trial_type_all + list([bartype]+list(np.tile('empty', task_trial_TR-1))) * num_task_trials

        ## then get task trial indices
        bar_pass_trials = np.where((np.array(trial_type_all) == 'task'))[0]
        
        return np.array(trial_type_all), bar_pass_trials
    
    
    def get_pp_task_keys(self, participant):

        """
        Get participant task keys, 
        because some participants swapped them

        Parameters
        ----------
        participant: str
            participant ID (just the number)
        
        """ 
        
        if str(participant).zfill(3) == '010':

             keys = {'right_index': ['left','e', 1, '1','num_1'],
                    'left_index': ['right','b', 2, '2','num_2']}

        else:
            keys = {'right_index': ['right','b', 2, '2','num_2', 'y'],
                    'left_index': ['left','e', 1, '1','num_1', 'w']}
    
        return keys


