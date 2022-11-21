import numpy as np
import re
import os
import os.path as op
import pandas as pd
import yaml
import glob

import itertools

from PIL import Image, ImageDraw

from FAM.utils import mri as mri_utils
from FAM.processing import preproc_behdata
from FAM.fitting.model import Model

from scipy.optimize import minimize

from joblib import Parallel, delayed
from tqdm import tqdm

from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel


class GLMsingle_Model(Model):

    def __init__(self, MRIObj, outputdir = None, tasks = ['pRF', 'FA']):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
        outputdir: str or None
            path to general output directory
        tasks: list
            list of strings with task names (mostly needed for Model object input)
            
        """

        # need to initialize parent class (Model), indicating output infos
        super().__init__(MRIObj = MRIObj, outputdir = outputdir, tasks = tasks)

        ## prf rsq threshold, to select visual voxels
        # worth fitting
        self.prf_rsq_threshold = self.MRIObj.params['mri']['fitting']['FA']['prf_rsq_threshold']

        # prf estimate bounds
        self.prf_bounds = None


    def get_single_trial_combinations(self):

        """
        Helper function to get all possible trial combinations
        (useful to keep track of single trial DM later)

        will return a DataFrame 
        where the columns indicate the attended and unattended bar midpoint position (x,y) and bar pass direction (vertical vs horizontal)
        and each row is a unique trial type
            
        """


        # define bar width in pixel
        bar_width_pix = self.screen_res * self.bar_width['FA']

        # define number of bars per direction
        num_bars = np.array(self.MRIObj.params['FA']['num_bar_position']) 

        # all possible positions in pixels [x,y] for midpoint of
        # vertical bar passes, 
        ver_y = np.sort(np.concatenate((-np.arange(bar_width_pix[1]/2,self.screen_res[1]/2,bar_width_pix[1])[0:int(num_bars[1]/2)],
                                        np.arange(bar_width_pix[1]/2,self.screen_res[1]/2,bar_width_pix[1])[0:int(num_bars[1]/2)])))

        ver_bar_pos_pix = np.array([np.array([0,y]) for _,y in enumerate(ver_y)])

        # horizontal bar passes 
        hor_x = np.sort(np.concatenate((-np.arange(bar_width_pix[0]/2,self.screen_res[0]/2,bar_width_pix[0])[0:int(num_bars[0]/2)],
                                        np.arange(bar_width_pix[0]/2,self.screen_res[0]/2,bar_width_pix[0])[0:int(num_bars[0]/2)])))

        hor_bar_pos_pix = np.array([np.array([x,0]) for _,x in enumerate(hor_x)])

        ## make all possible combinations
        pos_dict = {'horizontal': hor_bar_pos_pix, 'vertical': ver_bar_pos_pix}
        attend_orientation = ['vertical','horizontal']
        unattend_orientation = ['vertical','horizontal']

        # total number of trials
        num_trials = len(attend_orientation)*(pos_dict['horizontal'].shape[0] * pos_dict['vertical'].shape[0] + \
                                            pos_dict['horizontal'].shape[0] * (pos_dict['horizontal'].shape[0]-1))

        print('number of bar trials is %i'%num_trials)

        # define dictionary to save positions and directions
        # of all bars
        trial_combinations_dict = {'AttBar_bar_midpoint': [], 'AttBar_bar_pass_direction': [],
                                        'UnattBar_bar_midpoint': [], 'UnattBar_bar_pass_direction': []}

        # append all postions in dict 
        for att_ori in attend_orientation:

            for unatt_ori in unattend_orientation:

                if att_ori != unatt_ori: # if bar orientations orthogonal

                    indice_pairs = list((x,y) for x in np.arange(pos_dict[att_ori].shape[0]) for y in np.arange(pos_dict[unatt_ori].shape[0]))

                else: # if bar orientations the same

                    indice_pairs = list(itertools.permutations(np.arange(pos_dict[att_ori].shape[0]), 2))

                # fill attended dict
                trial_combinations_dict['AttBar_bar_midpoint'].append(np.array([pos_dict[att_ori][i] for i in np.array(indice_pairs)[...,0]]))
                trial_combinations_dict['AttBar_bar_pass_direction'].append(np.tile(att_ori, np.array(indice_pairs).shape[0]))

                # fill unattended dict
                trial_combinations_dict['UnattBar_bar_midpoint'].append(np.array([pos_dict[unatt_ori][i] for i in np.array(indice_pairs)[...,-1]]))
                trial_combinations_dict['UnattBar_bar_pass_direction'].append(np.tile(unatt_ori, np.array(indice_pairs).shape[0]))

        ## turn into dataframe
        self.trial_combinations_df = pd.DataFrame.from_dict(trial_combinations_dict).apply(pd.Series.explode).reset_index().drop(columns=['index'])


    def make_singletrial_dm(self, participant, run_num_arr =[], ses_num_arr = []):

        """
        Make single trial design matrix
        for one or more runs

        Parameters
        ----------
        participant: str
            participant ID
        run_num_arr: list
            list of ints with each run number to add to the design matrix 
            (DM will be in same order of runs IDs in this list)
        ses_num_arr: list
            list of ints with each ses number of the abovementioned run
            (DM will be in same order of ses IDs in this list)
            
        """

        ## get all possible trial combinations
        # to use for bookkeeping of single trial DM
        try:
            self.trial_combinations_df
        except AttributeError:
            self.get_single_trial_combinations()


        # set number of TRs to crop
        crop_nr = self.crop_TRs_num['FA'] if self.crop_TRs['FA'] == True else None

        ## get conditions per TR
        ## crop and shift if such was the case
        condition_per_TR = mri_utils.crop_shift_arr(self.mri_beh.FA_bar_pass_all,
                                                crop_nr = crop_nr, 
                                                shift = self.shift_TRs_num)

        ## make single trial DM
        # with shape [runs, TRs, conditions]
        single_trl_DM = np.zeros((len(run_num_arr), len(condition_per_TR), len(self.trial_combinations_df)))

        ## loop over runs
        for file_ind in range(len(run_num_arr)):

            ses_num = self.ses_num_arr[file_ind]
            run_num = self.run_num_arr[file_ind]

            ## get bar position df for run
            run_bar_pos_df = self.mri_beh.load_FA_bar_position(participant, ses = 'ses-{s}'.format(s = ses_num), 
                                                    ses_type = 'func', run_num = run_num)

            ## get run bar midpoint and direction values
            # for each bar type
            AttBar_bar_midpoint = run_bar_pos_df[run_bar_pos_df['attend_condition'] == 1].bar_midpoint_at_TR.values[0]
            AttBar_bar_pass_direction = run_bar_pos_df[run_bar_pos_df['attend_condition'] == 1].bar_pass_direction_at_TR.values[0]

            UnattBar_bar_midpoint = run_bar_pos_df[run_bar_pos_df['attend_condition'] == 0].bar_midpoint_at_TR.values[0]
            UnattBar_bar_pass_direction = run_bar_pos_df[run_bar_pos_df['attend_condition'] == 0].bar_pass_direction_at_TR.values[0]

            # set trial index counter
            trl_ind = 0

            ## fill DM for all TRs
            for i_TR, cond in enumerate(condition_per_TR):
                
                if cond == 'task':
                    
                    ## get condition index 
                    # where midpoint and direction for both bars is the same as the one of this trial
                    cond_index = self.trial_combinations_df[(self.trial_combinations_df['AttBar_bar_midpoint'].apply(lambda x: str(AttBar_bar_midpoint[trl_ind]) == str(x))) &\
                                (self.trial_combinations_df['AttBar_bar_pass_direction'].apply(lambda x: str(AttBar_bar_pass_direction[trl_ind]) == str(x))) &\
                                (self.trial_combinations_df['UnattBar_bar_midpoint'].apply(lambda x: str(UnattBar_bar_midpoint[trl_ind]) == str(x))) &\
                                (self.trial_combinations_df['UnattBar_bar_pass_direction'].apply(lambda x: str(UnattBar_bar_pass_direction[trl_ind]) == str(x)))].index[0]

                    # set which condition had its onset at that TR
                    single_trl_DM[file_ind, i_TR, cond_index] = 1
                    
                    # increment trial counter
                    trl_ind += 1

        return single_trl_DM



