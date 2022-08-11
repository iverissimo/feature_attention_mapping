import numpy as np
import os, sys
import os.path as op
import pandas as pd

from FAM.utils import beh as beh_utils
from FAM.utils import mri as mri_utils

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
        ## set type of bar pass per TR
        self.pRF_bar_pass_all = beh_utils.get_pRF_cond_per_TR(self.MRIObj.pRF_nr_TRs, 
                                                            self.MRIObj.pRF_bar_pass)
        ## number of trials (= total #TRs)
        self.pRF_total_trials = len(self.pRF_bar_pass_all)
        
        ## actual bar pass trials indexes (not accounting for empty TRs)
        self.pRF_bar_pass_trials = np.array([ind for ind,val in enumerate(self.pRF_bar_pass_all) if 'empty' not in val])
        #
        
    
    def load_events(self, participant, ses = 'ses-1', ses_type = 'func'):
        
        """
        Load behavioral events files

        Parameters
        ----------
        participant : str
            participant number
        ses : str
            session number (default ses-1)
        ses_type: str
            type of session (default func)

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
            for tsk in self.MRIObj.tasks:
                
                events_files = [op.join(input_pth,x) for x in os.listdir(input_pth) if 'task-{tsk}'.format(tsk=tsk) in x \
                                and x.endswith(self.MRIObj.events_ext)]
                
                print('{nr} events files found for task-{tsk}'.format(nr=len(events_files),
                                                                     tsk=tsk))
                
                # loop over runs
                events_df[tsk] = {}
                
                # for each run
                for r in np.arange(self.MRIObj.params['mri']['nr_runs']):

                    run_filename = [val for val in events_files if 'run-{r}'.format(r=(r+1)) in val]
                    if len(run_filename) == 0:
                        print('No events file for run-{r}'.format(r=(r+1)))
                    else:
                        print('Loading {f}'.format(f=op.split(run_filename[0])[-1]))
                        df_run = pd.read_csv(run_filename[0], sep='\t')
                        events_df[tsk]['run-{r}'.format(r=(r+1))] = df_run
        
        return events_df
    
    
    def load_trial_info(self, participant, ses = 'ses-1', ses_type = 'func'):
        
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
            for tsk in self.MRIObj.tasks:
                
                tf_files = [op.join(input_pth,x) for x in os.listdir(input_pth) if 'task-{tsk}'.format(tsk=tsk) in x \
                                and x.endswith(self.MRIObj.trial_info_ext)]
                
                print('{nr} trial info files found for task-{tsk}'.format(nr=len(tf_files),
                                                                     tsk=tsk))
                
                # loop over runs
                trial_info_df[tsk] = {}
                
                # for each run
                for r in np.arange(self.MRIObj.params['mri']['nr_runs']):

                    run_filename = [val for val in tf_files if 'run-{r}'.format(r=(r+1)) in val]
                    if len(run_filename) == 0:
                        print('No trial info file for run-{r}'.format(r=(r+1)))
                    else:
                        print('Loading {f}'.format(f=op.split(run_filename[0])[-1]))
                        df_run = pd.read_csv(run_filename[0])
                        trial_info_df[tsk]['run-{r}'.format(r=(r+1))] = df_run
        
        return trial_info_df
    
    
    def get_pRF_behavioral_results(self, ses_type = 'func'):
        
        """
        Get overview of behavioral results for pRF task
        
        """ 
        
        # summarize results in dataframe
        df_summary = pd.DataFrame({'sj': [], 'ses': [], 'run': [], 
                                   'color_category': [], 'accuracy': [], 'RT': []})
        
        # loop over participants
        for pp in self.MRIObj.sj_num:
            
            # and over sessions (if more than one)
            for ses in self.MRIObj.session['sub-{sj}'.format(sj=pp)]:
                
                ## load events files for that session
                events_df = self.load_events(pp, ses = ses, ses_type = ses_type)
                
                ## loop over runs
                for run in events_df['pRF'].keys():
                    
                    # get run event dataframe
                    run_ev_df = events_df['pRF'][run]

                    ## trial numbers where participant responsed
                    sub_response_trials = np.unique(run_ev_df[run_ev_df['event_type']=='response']['trial_nr'].values)
                    
                    ## get bar color and 
                    # bar color category for all trials
                    category_color, bar_color = beh_utils.get_pRF_trials_bar_color(run_ev_df)     
        
                    ## initialize a response array filled with nans for all trials in run
                    all_responses_bool = np.zeros(self.pRF_total_trials); all_responses_bool[:] = np.nan
                    all_responses_RT = np.zeros(self.pRF_total_trials); all_responses_RT[:] = np.nan

                    ## get boolean array showing if participant response was correct or not
                    # for trials where they responded
                    sub_response_bool = np.array([beh_utils.get_pp_response_bool(run_ev_df[run_ev_df['trial_nr'] == t], 
                                                                    category_color[t]) for t in sub_response_trials])

                    all_responses_bool[sub_response_trials] = sub_response_bool
                    
                    ## get reaction times for the same 
                    # trials
                    sub_response_RT = np.array([beh_utils.get_pp_response_rt(run_ev_df[run_ev_df['trial_nr'] == t]) for t in sub_response_trials])
                    
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


    def get_pRF_mask_bool(self, ses_type = 'func'):
        
        """
        Make boolean mask 
        to use in design matrix for pRF task
        based on subject responses
        
        """ 
        
        ## save mask in a dataframe for each participant
        df_mask_bool = pd.DataFrame({'sj': [], 'ses': [], 'mask_bool': []})
        
         # loop over participants
        for pp in self.MRIObj.sj_num:
            
            # and over sessions (if more than one)
            for ses in self.MRIObj.session['sub-{sj}'.format(sj=pp)]:
                
                ## load events files for that session
                events_df = self.load_events(pp, ses = ses, ses_type = ses_type)
                
                ## loop over runs
                run_bool = []
                for run in events_df['pRF'].keys():
                    
                    # get run event dataframe
                    run_ev_df = events_df['pRF'][run]

                    ## trial numbers where participant responsed
                    sub_response_trials = np.unique(run_ev_df[run_ev_df['event_type']=='response']['trial_nr'].values)
                    
                    ## get bar color and 
                    # bar color category for all trials
                    category_color, bar_color = beh_utils.get_pRF_trials_bar_color(run_ev_df)     
        
                     ## initialize a response array filled with 0 for all trials in run
                    all_responses_bool = np.zeros(self.pRF_total_trials)

                    ## get boolean array showing if participant response was correct or not
                    # for trials where they responded
                    sub_response_bool = np.array([beh_utils.get_pp_response_bool(run_ev_df[run_ev_df['trial_nr'] == t], 
                                                                    category_color[t]) for t in sub_response_trials])

                    all_responses_bool[sub_response_trials] = sub_response_bool

                    # append responses for that run
                    run_bool.append(all_responses_bool)
                    
                ## sums responses across runs
                # mask trials where wrong answer for more than 25% of runs 
                mask_bool = mri_utils.normalize(np.sum(np.array(run_bool), axis = 0))
                mask_bool[mask_bool>=.75] = 1
                mask_bool[mask_bool!=1] = 0

                ## append in df
                df_mask_bool = pd.concat((df_mask_bool,
                                            pd.DataFrame({'sj': ['sub-{sj}'.format(sj=pp)], 
                                                        'ses': [ses],
                                                        'mask_bool': [mask_bool]})
                                        ))
        
        return df_mask_bool