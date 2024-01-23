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


    
