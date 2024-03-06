import numpy as np
import os, sys
import os.path as op
import pandas as pd

import re

class PreprocBeh:

    def __init__(self, MRIObj):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
            
        """

        # set data object to use later on
        self.MRIObj = MRIObj

        ## general stuff
        #
        ## for pRF ##
        ## set type of bar pass per TR
        self.pRF_bar_pass_all = self.MRIObj.beh_utils.get_pRF_cond_per_TR(cond_TR_dict = self.MRIObj.pRF_nr_TRs, 
                                                                    bar_pass_direction = self.MRIObj.pRF_bar_pass)
        ## number of trials (= total #TRs)
        self.pRF_total_trials = len(self.pRF_bar_pass_all)
        
        ## actual bar pass trials indexes (not accounting for empty TRs)
        self.pRF_bar_pass_trials = np.array([ind for ind,val in enumerate(self.pRF_bar_pass_all) if 'empty' not in val])
        #
        
        ## same for FA ##
        #
        self.FA_bar_pass_all, self.FA_bar_pass_trials = self.MRIObj.beh_utils.get_FA_run_struct(self.MRIObj.FA_bar_pass, 
                                                                                    num_bar_pos = self.MRIObj.FA_num_bar_position, 
                                                                                    empty_TR = self.MRIObj.FA_nr_TRs['empty_TR'], 
                                                                                    task_trial_TR = self.MRIObj.FA_nr_TRs['task_trial_TR'])
        ## number of trials (= total #TRs)
        self.FA_total_trials = len(self.FA_bar_pass_all)
        
    def load_events(self, participant, ses = 'ses-1', ses_type = 'func', tasks = ['pRF', 'FA']):
        
        """
        Load behavioral events files for participant

        Parameters
        ----------
        participant : str
            participant number
        ses : str
            session number (default ses-1)
        ses_type: str
            type of session (default func)
        tasks: list
            list of tasks to load info from

        """ 
        
        # input path will be in sourcedata
        input_pth = op.join(self.MRIObj.sourcedata_pth, 'sub-{sj}'.format(sj=participant), ses, ses_type)
        
        # if session type doesn't exist
        if not op.exists(input_pth) or not os.listdir(input_pth):
            print('no files in %s'%input_pth)
        else:
            print('loading files from %s'%input_pth)
            
            # save in dict, because we want to store both tasks
            events_df = {}
            
            # loop over tasks
            for tsk in tasks:
                
                # if looking at func session
                if ses_type == 'func':
                    # first get bold file list 
                    bold_files = [x for x in os.listdir(input_pth) if 'task-{tname}'.format(tname = tsk) in x and x.endswith('_bold.nii.gz')]
                    # get run and ses_num identifier
                    ses_run_ids = ['{sn}_task-{tname}_run-{rn}'.format(sn = ses, 
                                                                    tname = tsk,
                                                                    rn = self.MRIObj.mri_utils.get_run_ses_from_str(file)[0]) for file in bold_files]
                    # get unique identifiers
                    ses_run_ids = np.unique(ses_run_ids) 
                    
                    # get bar position file names
                    events_files = [op.join(input_pth, 'sub-{sj}_{srid}{fext}'.format(sj = participant,
                                                                                srid = sn_rn_id,
                                                                                fext = self.MRIObj.events_ext)
                                        ) for sn_rn_id in ses_run_ids]          
                else:
                    events_files = [op.join(input_pth,x) for x in os.listdir(input_pth) if 'task-{tsk}'.format(tsk=tsk) in x \
                                    and x.endswith(self.MRIObj.events_ext)]
                
                print('{nr} events files found for task-{tsk}'.format(nr=len(events_files),
                                                                     tsk=tsk))
                
                # loop over runs
                events_df[tsk] = {}
                
                # for each run
                for run_filename in events_files:
                    if op.isfile(run_filename):
                        print('Loading {f}'.format(f=op.split(run_filename)[-1]))
                        events_df[tsk]['run-{r}'.format(r=self.MRIObj.mri_utils.get_run_ses_from_str(run_filename)[0])] = pd.read_csv(run_filename, sep='\t')
                    else:
                        print('No events file for run-{r}'.format(r=self.MRIObj.mri_utils.get_run_ses_from_str(run_filename)[0]))
        
        return events_df
    
    def load_trial_info(self, participant, ses = 'ses-1', ses_type = 'func', tasks = ['pRF', 'FA']):
        
        """
        Load Trial info files

        Parameters
        ----------
        participant : str
            participant number
        ses : str
            session number (default ses-1)
        ses_type: str
            type of session (default func)
        tasks: list
            list of tasks to load info from

        """ 
        
        # input path will be in sourcedata
        input_pth = op.join(self.MRIObj.sourcedata_pth, 'sub-{sj}'.format(sj=participant), ses, ses_type)
        
        # if session type doesn't exist
        if not op.exists(input_pth) or not os.listdir(input_pth):
            print('no files in %s'%input_pth)
        else:
            print('loading files from %s'%input_pth)
            
            # save in dict, because we want to store both tasks
            trial_info_df = {}
            
            # loop over tasks
            for tsk in tasks:
                
                # if looking at func session
                if ses_type == 'func':
                    # first get bold file list 
                    bold_files = [x for x in os.listdir(input_pth) if 'task-{tname}'.format(tname = tsk) in x and x.endswith('_bold.nii.gz')]
                    # get run and ses_num identifier
                    ses_run_ids = ['{sn}_task-{tname}_run-{rn}'.format(sn = ses, 
                                                                    tname = tsk,
                                                                    rn = self.MRIObj.mri_utils.get_run_ses_from_str(file)[0]) for file in bold_files]
                    # get unique identifiers
                    ses_run_ids = np.unique(ses_run_ids) 
                    
                    # get bar position file names
                    tf_files = [op.join(input_pth, 'sub-{sj}_{srid}{fext}'.format(sj = participant,
                                                                                srid = sn_rn_id,
                                                                                fext = self.MRIObj.trial_info_ext)
                                        ) for sn_rn_id in ses_run_ids]          
                else:
                    tf_files = [op.join(input_pth,x) for x in os.listdir(input_pth) if 'task-{tsk}'.format(tsk=tsk) in x \
                                    and x.endswith(self.MRIObj.trial_info_ext)]
                    
                print('{nr} trial info files found for task-{tsk}'.format(nr=len(tf_files),
                                                                     tsk=tsk))
                
                # loop over runs
                trial_info_df[tsk] = {}
                
                # for each run
                for run_filename in tf_files:
                    if op.isfile(run_filename):
                        print('Loading {f}'.format(f=op.split(run_filename)[-1]))
                        trial_info_df[tsk]['run-{r}'.format(r=self.MRIObj.mri_utils.get_run_ses_from_str(run_filename)[0])] = pd.read_csv(run_filename)
                    else:
                        print('No trial info file for run-{r}'.format(r=self.MRIObj.mri_utils.get_run_ses_from_str(run_filename)[0]))
        
        return trial_info_df

    def load_FA_bar_position(self, participant, ses_num = None, ses_type = 'func', run_num = None):
        
        """
        Load bar position from pickle files

        Parameters
        ----------
        participant : str
            participant number
        ses : str
            session number (default ses-1)
        ses_type: str
            type of session (default func)
        run_num: 
        """ 

        # if we provided a specific session number, only load that
        if ses_num:
            sessions = ['ses-{s}'.format(s = ses_num)]
        else:
            sessions = self.MRIObj.session['sub-{sj}'.format(sj=participant)]
            
        # save in dict
        bar_pos_df = {}

        for ses in sessions:

            bar_pos_df[ses] = {}

            # input path will be in sourcedata
            input_pth = op.join(self.MRIObj.sourcedata_pth, 'sub-{sj}'.format(sj=participant), ses, ses_type)
            
            # if session type doesn't exist
            if not op.exists(input_pth) or not os.listdir(input_pth):
                print('no files in %s'%input_pth)
            else:
                print('loading files from %s'%input_pth)
                
                # if looking at func session
                if ses_type == 'func':
                    # first get bold file list 
                    bold_files = [x for x in os.listdir(input_pth) if 'task-FA' in x and x.endswith('_bold.nii.gz')]
                    # get run and ses_num identifier
                    ses_run_ids = ['{sn}_task-FA_run-{rn}'.format(sn = ses, 
                                                                rn = self.MRIObj.mri_utils.get_run_ses_from_str(file)[0]) for file in bold_files]
                    # get unique identifiers
                    ses_run_ids = np.unique(ses_run_ids) 
                    
                    # get bar position file names
                    bp_files = [op.join(input_pth, 'sub-{sj}_{srid}{fext}'.format(sj = participant,
                                                                                srid = sn_rn_id,
                                                                                fext = self.MRIObj.bar_pos_ext)
                                        ) for sn_rn_id in ses_run_ids]          
                else:
                    bp_files = [op.join(input_pth,x) for x in os.listdir(input_pth) if 'task-FA' in x \
                                    and x.endswith(self.MRIObj.bar_pos_ext)]
                    
                print('{nr} bar position files found for task-FA'.format(nr=len(bp_files)))

                # if we provided a specific run number, only load that
                if run_num:
                    run_filename = op.join(input_pth, 'sub-{sj}_{sn}_task-FA_run-{rn}{fext}'.format(sj = participant,
                                                                                                    sn = ses,
                                                                                                    rn = run_num,
                                                                                                    fext = self.MRIObj.bar_pos_ext)
                                           )
                    if op.isfile(run_filename):
                        print('Loading {f}'.format(f=op.split(run_filename)[-1]))
                        bar_pos_df[ses]['run-{r}'.format(r=run_num)] = pd.read_pickle(run_filename)
                    else:
                        print('No trial info file for run-{r}'.format(r=run_num))
                else:
                    # for each run
                    for run_filename in bp_files:
                        if op.isfile(run_filename):
                            print('Loading {f}'.format(f=op.split(run_filename)[-1]))
                            bar_pos_df[ses]['run-{r}'.format(r=self.MRIObj.mri_utils.get_run_ses_from_str(run_filename)[0])] = pd.read_pickle(run_filename)
                        else:
                            print('No trial info file for run-{r}'.format(r=self.MRIObj.mri_utils.get_run_ses_from_str(run_filename)[0]))
        return bar_pos_df
    
    def get_group_FA_bar_position_dict(self, participant_list = [], ses_num = None, ses_type = 'func', run_num = None):
        
        """
        Load bar position from pickle files
        for all participants

        Parameters
        ----------
        participant_list : list
            list of strings with participant number
        ses : str
            session number (default ses-1)
        ses_type: str
            type of session (default func)
        run_num: int/str
            run number
        """ 
        
        group_bar_pos_df = {}
        
        for participant in participant_list:
            # get participant bar positions for FA task
            group_bar_pos_df['sub-{sj}'.format(sj = participant)] = self.load_FA_bar_position(participant, 
                                                                                            ses_num = ses_num, 
                                                                                            ses_type = ses_type, 
                                                                                            run_num = run_num)     
              
        return   group_bar_pos_df
    
    def get_run_ses_by_color(self, participant, ses_num = None, ses_type = 'func', run_num = None):

        """
        Get session and run number for same attended color
        """

        pp_bar_pos_df = self.load_FA_bar_position(participant, ses_num = ses_num, ses_type = ses_type, run_num = run_num)

        att_color_ses_run = {'color_green': {'ses': [], 'run': []}, 
                            'color_red': {'ses': [], 'run': []}}

        for ses_key in pp_bar_pos_df.keys():
            for run_key in pp_bar_pos_df[ses_key]:
                
                att_color = pp_bar_pos_df[ses_key][run_key][pp_bar_pos_df[ses_key][run_key]['attend_condition'] == 1].color[0]
                        
                att_color_ses_run[att_color]['ses'] += [int(re.findall(r'ses-\d{1,3}', ses_key)[0][4:])]
                att_color_ses_run[att_color]['run'] += [int(re.findall(r'run-\d{1,3}', run_key)[0][4:])]

        return att_color_ses_run
        
    def get_pRF_behavioral_results(self, ses_type = 'func'):
        
        """
        Get overview of behavioral results for pRF task

        Parameters
        ----------
        ses_type: str
            type of session (default func)
        
        """ 
        
        # summarize results in dataframe
        df_summary = pd.DataFrame({'sj': [], 'ses': [], 'run': [], 
                                   'color_category': [], 'accuracy': [], 'RT': []})
        
        # loop over participants
        for pp in self.MRIObj.sj_num:
            
            # and over sessions (if more than one)
            for ses in self.MRIObj.session['sub-{sj}'.format(sj=pp)]:
                
                ## load events files for that session
                events_df = self.load_events(pp, ses = ses, ses_type = ses_type, tasks=['pRF'])
                
                ## loop over runs
                for run in events_df['pRF'].keys():
                    
                    # get run event dataframe
                    run_ev_df = events_df['pRF'][run]

                    ## trial numbers where participant responsed
                    sub_response_trials = np.unique(run_ev_df[run_ev_df['event_type']=='response']['trial_nr'].values)
                    
                    ## get bar color and 
                    # bar color category for all trials
                    category_color, bar_color = self.MRIObj.beh_utils.get_pRF_trials_bar_color(run_ev_df)     
        
                    ## initialize a response array filled with nans for all trials in run
                    all_responses_bool = np.zeros(self.pRF_total_trials); all_responses_bool[:] = np.nan
                    all_responses_RT = np.zeros(self.pRF_total_trials); all_responses_RT[:] = np.nan

                    ## get boolean array showing if participant response was correct or not
                    # for trials where they responded

                    # some participants swapped the buttons, so make exceptions
                    pp_task_keys = self.MRIObj.beh_utils.get_pp_task_keys(pp)

                    sub_response_bool = np.array([self.MRIObj.beh_utils.get_pp_response_bool(run_ev_df[run_ev_df['trial_nr'] == t], trial_bar_color = category_color[t], 
                                                                                      task = 'pRF', keys = pp_task_keys) for t in sub_response_trials])

                    all_responses_bool[sub_response_trials] = sub_response_bool
                    
                    ## get reaction times for the same 
                    # trials
                    sub_response_RT = np.array([self.MRIObj.beh_utils.get_pp_response_rt(run_ev_df[run_ev_df['trial_nr'] == t], 
                                                                                  task = 'pRF', TR = self.MRIObj.TR) for t in sub_response_trials])
                    
                    all_responses_RT[sub_response_trials] = sub_response_RT

                    ## now slice array for ONLY bar passing trials
                    #
                    RUN_category_color = np.array(category_color)[self.pRF_bar_pass_trials]
                    RUN_bar_color = np.array(bar_color)[self.pRF_bar_pass_trials]
                    
                    RUN_responses_bool = all_responses_bool[self.pRF_bar_pass_trials]
                    
                    RUN_response_RT = all_responses_RT[self.pRF_bar_pass_trials]; 
                    RUN_response_RT[RUN_responses_bool!=1] = np.nan

                    ## Fill results DF for each color category 
                    # separately
                    for cc in self.MRIObj.color_categories_dict.keys():

                        acc_by_cc = (np.nansum(RUN_responses_bool[np.where(RUN_category_color == cc)[0]]))/len(np.where(RUN_category_color == cc)[0])

                        df_summary = pd.concat((df_summary,
                                                pd.DataFrame({'sj': ['sub-{sj}'.format(sj=pp)], 
                                                            'ses': [ses],
                                                            'run': [run],
                                                            'color_category': [cc],
                                                            'accuracy': [acc_by_cc],
                                                            'RT': [np.nanmean(RUN_response_RT[np.where(RUN_category_color == cc)[0]])]
                                                            })))
                                                                    
        return df_summary

    def get_pRF_mask_bool(self, ses_type = 'func', crop_nr = 0, shift = 0):
        
        """
        Make boolean mask based on subject responses, 
        to use in design matrix for pRF task

        Parameters
        ----------
        ses_type: str
            type of session (default func)
        
        """ 
        
        ## save mask in a dataframe for each participant
        df_mask_bool = pd.DataFrame({'sj': [], 'ses': [], 'mask_bool': []})
        
         # loop over participants
        for pp in self.MRIObj.sj_num:
            
            # and over sessions (if more than one)
            for ses in self.MRIObj.session['sub-{sj}'.format(sj=pp)]:
                
                ## load events files for that session
                events_df = self.load_events(pp, ses = ses, ses_type = ses_type, tasks=['pRF'])
                
                ## loop over runs
                run_bool = []
                for run in events_df['pRF'].keys():
                    
                    # get run event dataframe
                    run_ev_df = events_df['pRF'][run]

                    ## trial numbers where participant responsed
                    sub_response_trials = np.unique(run_ev_df[run_ev_df['event_type']=='response']['trial_nr'].values)
                    
                    ## get bar color and 
                    # bar color category for all trials
                    category_color, _ = self.MRIObj.beh_utils.get_pRF_trials_bar_color(run_ev_df)     
        
                     ## initialize a response array filled with 0 for all trials in run
                    all_responses_bool = np.zeros(self.pRF_total_trials)

                    ## get boolean array showing if participant response was correct or not
                    # for trials where they responded

                    # some participants swapped the buttons, so make exceptions
                    pp_task_keys = self.MRIObj.beh_utils.get_pp_task_keys(pp)

                    sub_response_bool = np.array([self.MRIObj.beh_utils.get_pp_response_bool(run_ev_df[run_ev_df['trial_nr'] == t], trial_bar_color = category_color[t], 
                                                                                      task = 'pRF', keys = pp_task_keys) for t in sub_response_trials])

                    all_responses_bool[sub_response_trials] = sub_response_bool

                    # append responses for that run
                    run_bool.append(all_responses_bool)
                    
                ## sums responses across runs
                # mask trials where wrong answer for more than 25% of runs 
                mask_bool = self.MRIObj.beh_utils.normalize(np.sum(np.array(run_bool), axis = 0))
                mask_bool[mask_bool>=.75] = 1
                mask_bool[mask_bool!=1] = 0

                mask_bool = self.MRIObj.mri_utils.crop_shift_arr(mask_bool, crop_nr = crop_nr, shift = shift)

                ## append in df
                df_mask_bool = pd.concat((df_mask_bool,
                                            pd.DataFrame({'sj': ['sub-{sj}'.format(sj=pp)], 
                                                        'ses': [ses],
                                                        'mask_bool': [mask_bool]})
                                        ))
        
        return df_mask_bool
    
    def get_stim_on_screen(self, task = 'pRF', crop_nr = 0, shift = 0):

        """
        Get boolean array indicating on which TRs stimuli is on screen
        (useful for plotting/bookeeping)

        Parameters
        ----------
        task: str
            task name
        crop_nr : None or int
            if not none, expects int with number of FIRST time points to crop
        shift : int
            positive or negative int, of number of time points to shift (if neg, will shift leftwards)
        """ 

        # make stim on screen arr
        if task == 'pRF':
            stim_on_screen = np.zeros(self.pRF_total_trials)
            stim_on_screen[self.pRF_bar_pass_trials] = 1
        elif task == 'FA':
            stim_on_screen = np.zeros(self.FA_total_trials)
            stim_on_screen[self.FA_bar_pass_trials] = 1
            
        return self.MRIObj.mri_utils.crop_shift_arr(stim_on_screen, crop_nr = crop_nr, shift = shift)

    def get_FA_behavioral_results(self, ses_type = 'func'):
        
        """
        Get overview of behavioral results for FA task

        Parameters
        ----------
        ses_type: str
            type of session (default func)
        
        """ 
        
        # summarize results in dataframe
        df_summary = pd.DataFrame({'sj': [], 'ses': [], 'run': [], 
                           'attended_color': [],'color_category': [],'bar_color': [],
                            'accuracy': [], 'RT': []})
        
        # loop over participants
        for pp in self.MRIObj.sj_num:
            
            # and over sessions (if more than one)
            for ses in self.MRIObj.session['sub-{sj}'.format(sj=pp)]:
                
                ## load events files for that session
                events_df = self.load_events(pp, ses = ses, ses_type = ses_type, tasks=['FA'])
                ## load trial info for that session
                trial_info_df = self.load_trial_info(pp, ses = ses, ses_type = ses_type, tasks=['FA'])
                
                ## loop over runs
                for run in events_df['FA'].keys():
                    
                    # get run event dataframe
                    run_ev_df = events_df['FA'][run]
                    # and trial info
                    run_trl_info_df = trial_info_df['FA'][run]

                    ## trial numbers where participant responded
                    # will include bar stim on screen + TR after
                    sub_response_trials = [[trl,trl+1] for trl in self.FA_bar_pass_trials if 'response' in run_ev_df[run_ev_df['trial_nr'].isin([trl,trl+1])]['event_type'].values]
                                        
                    ## get bar color and bar color category 
                    # for attended and unattended bars
                    # for all trials
                    category_color, bar_color = self.MRIObj.beh_utils.get_FA_trials_bar_color(run_trl_info_df)    
        
                    ## loop over attended and unattended conditions
                    # (we might want to compare)
                    for cond in category_color.keys():
                        
                        ## initialize a response array filled with nans for all trials in run
                        all_responses_bool = np.zeros(self.FA_total_trials); all_responses_bool[:] = np.nan
                        all_responses_RT = np.zeros(self.FA_total_trials); all_responses_RT[:] = np.nan
                        
                        ## get boolean array showing if participant response was correct or not
                        # for trials where they responded
                        
                        # some participants swapped the buttons, so make exceptions
                        pp_task_keys = self.MRIObj.beh_utils.get_pp_task_keys(pp)

                        sub_response_bool = np.array([self.MRIObj.beh_utils.get_pp_response_bool(run_ev_df[run_ev_df['trial_nr'].isin(t)], trial_bar_color = bar_color[cond][t[0]], 
                                                                                          task = 'FA', keys = pp_task_keys) for t in sub_response_trials])
                        
                        all_responses_bool[np.ravel(sub_response_trials)[::2]] = sub_response_bool
                        
                        ## get reaction times for the same 
                        # trials
                        sub_response_RT = np.array([self.MRIObj.beh_utils.get_pp_response_rt(run_ev_df[run_ev_df['trial_nr'].isin(t)],
                                                                                      task = 'FA', TR = self.MRIObj.TR) for t in sub_response_trials])

                        all_responses_RT[np.ravel(sub_response_trials)[::2]] = sub_response_RT
                        
                        ## now slice array for ONLY bar passing trials
                        #
                        RUN_category_color = np.array(category_color[cond])[self.FA_bar_pass_trials]
                        RUN_bar_color = np.array(bar_color[cond])[self.FA_bar_pass_trials]

                        RUN_responses_bool = all_responses_bool[self.FA_bar_pass_trials]

                        RUN_response_RT = all_responses_RT[self.FA_bar_pass_trials]; 
                        RUN_response_RT[RUN_responses_bool!=1] = np.nan
                        
                        
                        ## Fill results DF for each specifc bar color  
                        # separately
                        for bc in self.MRIObj.color_categories_dict[category_color[cond][0]]:
                            
                            at = True if cond == 'attend_bar' else False

                            acc_by_bc = (np.nansum(RUN_responses_bool[np.where(RUN_bar_color == bc)[0]]))/len(np.where(RUN_bar_color == bc)[0])

                            df_summary = pd.concat((df_summary,
                                                    pd.DataFrame({'sj': ['sub-{sj}'.format(sj=pp)], 
                                                                'ses': [ses],
                                                                'run': [run],
                                                                'attended_color': [at],
                                                                'color_category': [category_color[cond][0]],
                                                                'bar_color': [bc],
                                                                'accuracy': [acc_by_bc],
                                                                'RT': [np.nanmean(RUN_response_RT[np.where(RUN_bar_color == bc)[0]])]
                                                                })))
                         
        return df_summary

    def get_FA_RT(self, ses_type = 'func'):
        
        """
        Get RT for FA task

        Parameters
        ----------
        ses_type: str
            type of session (default func)
        
        """ 
        
        # number of task trials
        n_trials = len(self.FA_bar_pass_trials)
        
        # summarize results in dataframe
        df_RT = []
        
        # loop over participants
        for pp in self.MRIObj.sj_num:
            
            # and over sessions (if more than one)
            for ses in self.MRIObj.session['sub-{sj}'.format(sj=pp)]:
                
                ## load events files for that session
                events_df = self.load_events(pp, ses = ses, ses_type = ses_type, tasks=['FA'])
                ## load trial info for that session
                trial_info_df = self.load_trial_info(pp, ses = ses, ses_type = ses_type, tasks=['FA'])
                
                ## loop over runs
                for run in events_df['FA'].keys():
                    
                    # get run event dataframe
                    run_ev_df = events_df['FA'][run]
                    # and trial info
                    run_trl_info_df = trial_info_df['FA'][run]

                    ## trial numbers where participant responded
                    # will include bar stim on screen + TR after
                    sub_response_trials = [[trl,trl+1] for trl in self.FA_bar_pass_trials if 'response' in run_ev_df[run_ev_df['trial_nr'].isin([trl,trl+1])]['event_type'].values]
                                        
                    ## get bar color and bar color category 
                    # for attended and unattended bars
                    # for all trials
                    category_color, bar_color = self.MRIObj.beh_utils.get_FA_trials_bar_color(run_trl_info_df)    
        
                    ## select only attended conditions (to label for color)
                    att_category_color = category_color['attend_bar']
                    att_bar_color = bar_color['attend_bar']
                     
                    ## initialize a response array filled with nans for all trials in run
                    all_responses_bool = np.zeros(self.FA_total_trials); all_responses_bool[:] = np.nan
                    all_responses_RT = np.zeros(self.FA_total_trials); all_responses_RT[:] = np.nan
                    
                    ## get boolean array showing if participant response was correct or not
                    # for trials where they responded
                    
                    # some participants swapped the buttons, so make exceptions
                    pp_task_keys = self.MRIObj.beh_utils.get_pp_task_keys(pp)

                    sub_response_bool = np.array([self.MRIObj.beh_utils.get_pp_response_bool(run_ev_df[run_ev_df['trial_nr'].isin(t)], trial_bar_color = att_bar_color[t[0]], 
                                                                                        task = 'FA', keys = pp_task_keys) for t in sub_response_trials])
                    
                    all_responses_bool[np.ravel(sub_response_trials)[::2]] = sub_response_bool
                    
                    ## get reaction times for the same 
                    # trials
                    sub_response_RT = np.array([self.MRIObj.beh_utils.get_pp_response_rt(run_ev_df[run_ev_df['trial_nr'].isin(t)],
                                                                                    task = 'FA', TR = self.MRIObj.TR) for t in sub_response_trials])

                    all_responses_RT[np.ravel(sub_response_trials)[::2]] = sub_response_RT
                    
                    ## now slice array for ONLY bar passing trials
                    #
                    RUN_category_color = np.array(att_category_color)[self.FA_bar_pass_trials]
                    RUN_bar_color = np.array(att_bar_color)[self.FA_bar_pass_trials]

                    RUN_responses_bool = all_responses_bool[self.FA_bar_pass_trials]

                    RUN_response_RT = all_responses_RT[self.FA_bar_pass_trials]; 
                    #RUN_response_RT[RUN_responses_bool!=1] = np.nan
                    
                    ## append df
                    df_RT.append(pd.DataFrame({'sj': np.repeat('sub-{sj}'.format(sj=pp), n_trials), 
                                            'ses': np.repeat(ses, n_trials),
                                            'run': np.repeat(run, n_trials),
                                            'trial_ind': np.arange(n_trials),
                                            'color_category': RUN_category_color,
                                            'bar_color': RUN_bar_color,
                                            'correct': RUN_responses_bool,
                                            'RT': RUN_response_RT
                                            })
                    )
                    
        df_RT = pd.concat(df_RT, ignore_index=True)
                         
        return df_RT
    
    def get_pRF_bar_coords_per_TR(self, bar_direction = 'horizontal'):

        """
        Get array with pRF bar center coordinates per TR.
        Note - if horizontal bar pass (bar is vertically oriented), will return y coordinates; 
        if vertical bar passes (bar is horizontally oriented)) will return x coordinates

        """

        screen_res = self.MRIObj.screen_res[0] if bar_direction == 'horizontal' else self.MRIObj.screen_res[1]
        conditions = ['L-R', 'R-L'] if bar_direction == 'horizontal' else ['D-U','U-D'] 

        bar_coord_per_TR = []

        for _,bartype in enumerate(self.MRIObj.pRF_bar_pass):

            if bartype == conditions[0]:
                bar_coord_per_TR += list(screen_res*np.linspace(-.5,.5, self.MRIObj.pRF_nr_TRs[bartype]))
            elif bartype == conditions[1]:
                bar_coord_per_TR += list(screen_res*np.linspace(.5,-.5, self.MRIObj.pRF_nr_TRs[bartype]))
            else:
                # want to keep the irrelevant dimensions as nan, to avoid confusion
                bar_coord_per_TR += list(np.repeat(np.nan, self.MRIObj.pRF_nr_TRs[bartype]))

        # crop and shift array, if such was the case
        bar_coord_per_TR = self.MRIObj.beh_utils.crop_shift_arr(np.array(bar_coord_per_TR), 
                                                                crop_nr = self.MRIObj.task_nr_cropTR['pRF'], 
                                                                shift = self.MRIObj.shift_TRs_num)

        return bar_coord_per_TR
    
    def get_pRF_masked_bar_coords(self, participant_list = [], ses = 'mean', mask_bool_df = None, bar_direction = None):

        """
        Get dict with unique pRF bar coordinates (for horizontal and vertical bar passes)
        masked for participant visibility

        Parameters
        ----------
        participant_list: list
            list with participant ID
        ses : str
            session number (default mean)
        mask_bool_df: dataframe
            will be used to mask design matrix given behavioral performance
        bar_direction: str
            if given, will only return horizontal/vertical bar pass coordinates 
        """

        if isinstance(ses, str) and 'ses' in ses: # to account for differences in input
            ses = re.search(r'(?<=ses-).*', ses)[0]

        if mask_bool_df is None:
            mask_bool_df = self.get_pRF_mask_bool(ses_type = 'func',
                                                    crop_nr = self.MRIObj.task_nr_cropTR['pRF'], 
                                                    shift = self.MRIObj.shift_TRs_num)
            
        bar_coords_dict = {}
            
        # loop over participant
        for participant in participant_list:

            # if we set a specific session, then select that one, else combine
            if ses == 'mean':
                mask_bool = mask_bool_df[mask_bool_df['sj'] == 'sub-{sj}'.format(sj = participant)]['mask_bool'].values
            else:
                mask_bool = mask_bool_df[(mask_bool_df['ses'] == 'ses-{s}'.format(s = ses)) & \
                                    (mask_bool_df['sj'] == 'sub-{sj}'.format(sj = participant))]['mask_bool'].values 
            mask_bool = np.prod(mask_bool, axis = 0)
                
            # iterate over bar directions
            if bar_direction is None:
                bar_direction_list = ['vertical', 'horizontal']
            elif isinstance(bar_direction, str):
                bar_direction_list = [bar_direction]

            bar_coords_dict['sub-{sj}'.format(sj = participant)] = {}
            
            for bd_key in bar_direction_list:

                ## get pRF bar center coordinates per TR for bar pass direction
                bar_coords_masked = self.get_pRF_bar_coords_per_TR(bar_direction = bd_key).copy()
                bar_coords_masked[np.where((mask_bool == 0))[0]] = np.nan

                # get unique coordinates
                uniq_bar_coords = np.unique(bar_coords_masked)
                uniq_bar_coords = np.array([val for val in uniq_bar_coords if ~np.isnan(val)])

                bar_coords_dict['sub-{sj}'.format(sj = participant)][bd_key] = uniq_bar_coords

        return bar_coords_dict
    
    def make_FA_df_run_bar_pos(self, run_df = None, prf_bar_coords_dict = None):
        
        """make data frame with bar positions and indices for each trial
        for a given run (in a summarized way)
        """

        # define bar width in pixel
        bar_width_pix = self.MRIObj.screen_res * self.MRIObj.bar_width['FA']

        # define number of bars per direction
        num_bars = np.array(self.MRIObj.FA_num_bar_position) 

        # all possible positions in pixels [x,y] for midpoint of
        # horizontal bar passes 
        bar_x_coords_pix = np.sort(np.concatenate((-np.arange(bar_width_pix[0]/2,self.MRIObj.screen_res[0]/2,bar_width_pix[0])[0:int(num_bars[0]/2)],
                                        np.arange(bar_width_pix[0]/2,self.MRIObj.screen_res[0]/2,bar_width_pix[0])[0:int(num_bars[0]/2)])))
        
        ## get run bar midpoint and direction values
        # for each bar type (arrays will have len == total number of trial types)
        AttBar_bar_midpoint, AttBar_bar_pass_direction = run_df.loc[(run_df['attend_condition'] == 1), 
                                                                            ['bar_midpoint_at_TR', 'bar_pass_direction_at_TR']].to_numpy()[0]
        UnattBar_bar_midpoint, UnattBar_bar_pass_direction = run_df.loc[(run_df['attend_condition'] == 0), 
                                                                            ['bar_midpoint_at_TR', 'bar_pass_direction_at_TR']].to_numpy()[0]

        ## find parallel + crossed bar trial indices
        parallel_bar_ind = np.where((AttBar_bar_pass_direction == UnattBar_bar_pass_direction))[0]
        crossed_bar_ind = np.where((AttBar_bar_pass_direction != UnattBar_bar_pass_direction))[0]

        # check for masked trials
        if prf_bar_coords_dict is not None:
            print('checking for trials to mask')
            t_mask = self.MRIObj.beh_utils.get_trial_ind_mask(AttBar_bar_midpoint = AttBar_bar_midpoint, 
                                                            AttBar_bar_pass_direction = AttBar_bar_pass_direction,
                                                            UnattBar_bar_midpoint = UnattBar_bar_midpoint, 
                                                            UnattBar_bar_pass_direction = UnattBar_bar_pass_direction,
                                                            prf_bar_coords_dict = prf_bar_coords_dict)
            # if trials to mask
            if t_mask is not None:
                print('removing %i trials'%len(t_mask))
                parallel_bar_ind = np.array([val for val in parallel_bar_ind if val not in t_mask])
                crossed_bar_ind = np.array([val for val in crossed_bar_ind if val not in t_mask])

        ## make summary dataframe 
        position_df = []

        for keys, ind_arr in {'parallel': parallel_bar_ind, 'crossed': crossed_bar_ind}.items():
            for att_bool in [0,1]:
                tmp_df = pd.DataFrame({'x_pos': run_df[run_df['attend_condition'] == att_bool].bar_midpoint_at_TR.values[0][ind_arr][:,0],
                                    'y_pos': run_df[run_df['attend_condition'] == att_bool].bar_midpoint_at_TR.values[0][ind_arr][:,1],
                                    'trial_ind': ind_arr})
                tmp_df['attend_condition'] = bool(att_bool)
                tmp_df['bars_pos'] = keys
                position_df.append(tmp_df)
        position_df = pd.concat(position_df, ignore_index = True)
        
        ## add interbar distance (only for parallel bars)
        # for x
        inds_uatt = position_df[((position_df['attend_condition'] == 0) &\
                                (position_df['bars_pos'] == 'parallel') &\
                    ((position_df['x_pos'] != 0)))].sort_values('trial_ind').index
        inds_att = position_df[((position_df['attend_condition'] == 1) &\
                                (position_df['bars_pos'] == 'parallel') &\
                    ((position_df['x_pos'] != 0)))].sort_values('trial_ind').index
        inter_bar_dist = (position_df.iloc[inds_uatt].x_pos.values - position_df.iloc[inds_att].x_pos.values)/bar_width_pix[0]

        position_df.loc[inds_uatt,'inter_bar_dist'] = inter_bar_dist
        position_df.loc[inds_att,'inter_bar_dist'] = inter_bar_dist

        # for y
        inds_uatt = position_df[((position_df['attend_condition'] == 0) &\
                                (position_df['bars_pos'] == 'parallel') &\
                    ((position_df['y_pos'] != 0)))].sort_values('trial_ind').index
        inds_att = position_df[((position_df['attend_condition'] == 1) &\
                                (position_df['bars_pos'] == 'parallel') &\
                    ((position_df['y_pos'] != 0)))].sort_values('trial_ind').index
        inter_bar_dist = (position_df.iloc[inds_uatt].y_pos.values - position_df.iloc[inds_att].y_pos.values)/bar_width_pix[0]

        position_df.loc[inds_uatt,'inter_bar_dist'] = inter_bar_dist
        position_df.loc[inds_att,'inter_bar_dist'] = inter_bar_dist
        
        ## add bar eccentricity
        ecc_dict = {'far': bar_x_coords_pix[0::5], 'middle': bar_x_coords_pix[1::3], 'near': bar_x_coords_pix[2:4]}

        for ecc_key in ecc_dict.keys():
            inds = position_df[((position_df['x_pos'].isin(ecc_dict[ecc_key])) |\
                        (position_df['y_pos'].isin(ecc_dict[ecc_key])))].sort_values('trial_ind').index
            position_df.loc[inds,'bar_ecc'] = ecc_key
            
        ## also add absolute distance
        position_df.loc[:,'abs_inter_bar_dist'] = np.absolute(position_df.inter_bar_dist.values)
        
        ## add bar eccentricity in deg
        
        # get near absolute ecc value in deg
        min_ecc = bar_width_pix[0]/2 * self.MRIObj.beh_utils.dva_per_pix(height_cm = self.MRIObj.params['monitor']['height'], 
                                                            distance_cm = self.MRIObj.params['monitor']['distance'], 
                                                            vert_res_pix = self.MRIObj.screen_res[1])

        # replace ecc with numeric value
        ecc_deg = position_df.bar_ecc.values.copy()
        ecc_deg[ecc_deg == 'near'] = min_ecc
        ecc_deg[ecc_deg == 'middle'] = min_ecc + min_ecc*2
        ecc_deg[ecc_deg == 'far'] = min_ecc + min_ecc*4
        position_df.loc[:, 'bar_ecc_deg'] = ecc_deg
        
        ## also add label indicating if competing bar is closer to fovea (for given trial)
        position_df.loc[:, 'compbar_closer2fix'] = False
        # for attended bars
        ind_list = position_df[(position_df.sort_values(['attend_condition'],ascending=False).groupby('trial_ind')['bar_ecc_deg'].transform(lambda x: x.values[0] > x.values[1])) &\
                                (position_df['attend_condition'] == True)].index.values
        position_df.loc[ind_list, 'compbar_closer2fix'] = True
        # and unattended bars
        ind_list = position_df[(position_df.sort_values(['attend_condition'],ascending=False).groupby('trial_ind')['bar_ecc_deg'].transform(lambda x: x.values[0] < x.values[1])) &\
                                (position_df['attend_condition'] == False)].index.values
        position_df.loc[ind_list, 'compbar_closer2fix'] = True
   
        return position_df
    
    def get_FA_pp_run_position_df(self, pp_bar_pos_dict = None,  data_keys = ['ses-1_run-1'], pp_prf_bar_coords_dict = None):
        
        """get data frame with bar positions and indices for all runs
        of a given participant
        """
   
        run_position_df = []

        for ind, df_key in enumerate(data_keys):
            
            print('making df with bar position info for %s'%df_key)
            # get run number and session, to avoid mistakes 
            file_rn, file_sn = self.MRIObj.beh_utils.get_run_ses_from_str(df_key)

            tmp_df = self.make_FA_df_run_bar_pos(run_df = pp_bar_pos_dict['ses-{s}'.format(s = file_sn)]['run-{r}'.format(r=file_rn)],
                                                prf_bar_coords_dict = pp_prf_bar_coords_dict)
            tmp_df.loc[:, 'ses'] = 'ses-{s}'.format(s = file_sn)
            tmp_df.loc[:, 'run'] = 'run-{r}'.format(r = file_rn)
            
            run_position_df.append(tmp_df) 
        
        return pd.concat(run_position_df, ignore_index=True)
    
    def get_FA_group_run_position_df(self, participant_list = [], group_bar_pos_dict = None, data_keys_dict = None,
                                            prf_bar_coords_dict = None, ses_type = 'func', mask_df = True):
        
        """get data frame with bar positions and indices for all participants
        """

        # get all participant bar positions for FA task
        if group_bar_pos_dict is None: 
            group_bar_pos_dict = self.get_group_FA_bar_position_dict(participant_list = participant_list, 
                                                                    ses_num = None, 
                                                                    ses_type = ses_type, 
                                                                    run_num = None)
        
        if data_keys_dict is None:
            data_keys_dict = self.MRIObj.beh_utils.get_data_keys_dict(participant_list = participant_list, 
                                                                    group_bar_pos_dict = group_bar_pos_dict)
        
        ## get prf bar position dict
        # to mask out FA trials that were not fully visible
        if prf_bar_coords_dict is None and mask_df: 
            prf_bar_coords_dict = self.get_pRF_masked_bar_coords(participant_list = participant_list,
                                                                ses = 'mean')
        elif mask_df == False:
            prf_bar_coords_dict = {'sub-{sj}'.format(sj = pp): None for pp in participant_list}
        
        group_run_pos_df = []

        for participant in participant_list:
            
            print('Getting bar positions for sub-{sj}'.format(sj = participant)) 
            
            ## get FA bar position dict, across runs
            tmp_df =  self.get_FA_pp_run_position_df(pp_bar_pos_dict = group_bar_pos_dict['sub-{sj}'.format(sj = participant)],  
                                                    data_keys = data_keys_dict['sub-{sj}'.format(sj = participant)],
                                                    pp_prf_bar_coords_dict = prf_bar_coords_dict['sub-{sj}'.format(sj = participant)])
            tmp_df.loc[:, 'sj'] = 'sub-{sj}'.format(sj = participant)
            ## add label for attended condition
            tmp_df.loc[tmp_df.query('attend_condition').index.values, 'bar_type'] = 'att_bar'
            tmp_df.loc[tmp_df.query('~attend_condition').index.values, 'bar_type'] = 'unatt_bar'
            
            group_run_pos_df.append(tmp_df) 
               
        return pd.concat(group_run_pos_df, ignore_index=True)  

    def squeeze_FA_bar_pos_df(self, FA_run_position_df = None):

        """
        Helper function to reduce bar position df,
        keeping only minimum variables necessary to identify bar positions at a given trial
        """     

        ## create bar pos data frame, unstacked
        # select attended trials
        att_position_df = FA_run_position_df.query('attend_condition').loc[:, ['sj', 'run', 'ses', 'trial_ind', 'x_pos', 'y_pos', 'bars_pos']]
        att_position_df.rename(columns={'x_pos': 'att_x_pos', 'y_pos': 'att_y_pos'}, inplace=True)
        # select unattended trials
        unatt_position_df = FA_run_position_df.query('~attend_condition').loc[:, ['sj', 'run', 'ses', 'trial_ind', 'x_pos', 'y_pos', 'bars_pos']]
        unatt_position_df.rename(columns={'x_pos': 'unatt_x_pos', 'y_pos': 'unatt_y_pos'}, inplace=True)
        unatt_position_df

        # merge both
        new_position_df = att_position_df.merge(unatt_position_df, on = ['sj', 'run', 'ses', 'trial_ind'])

        return new_position_df