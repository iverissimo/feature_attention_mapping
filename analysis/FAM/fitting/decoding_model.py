import numpy as np
import re
import os
import os.path as op
import pandas as pd
import yaml
import glob

import itertools
from scipy.interpolate import pchip
import scipy

from tqdm import tqdm

from PIL import Image, ImageDraw

from FAM.fitting.model import Model
from FAM.fitting.glm_single_model import GLMsingle_Model

from glmsingle.glmsingle import GLM_single
from glmsingle.glmsingle import getcanonicalhrf

import time
import cortex
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import matplotlib.patches as patches
import seaborn as sns

import nibabel as nib
import neuropythy

import braincoder
from braincoder.models import GaussianPRF2DWithHRF, GaussianPRF2D, DifferenceOfGaussiansPRF2DWithHRF, DifferenceOfGaussiansPRF2D
from braincoder.hrf import SPMHRFModel
from braincoder.optimize import ParameterFitter, ResidualFitter, StimulusFitter


class Decoding_Model(GLMsingle_Model):

    def __init__(self, MRIObj, pRFModelObj = None, outputdir = None, pysub = 'hcp_999999', use_atlas = None):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
        outputdir: str or None
            path to general output directory
        use_atlas: str
            If we want to use atlas ROIs (ex: glasser, wang) or not [default].
            
        """

        # need to initialize parent class (Model), indicating output infos
        super().__init__(MRIObj = MRIObj, outputdir = outputdir, pysub = pysub, use_atlas = use_atlas)
        
        # prf sj space to define ROIs will be surface space
        self.prf_sj_space = 'fsnative'
        self.decoder_dir = op.join(self.MRIObj.derivatives_pth, 'decoder')
        self.pRFModelObj = pRFModelObj

        self.prf_condition_per_TR = self.pRFModelObj.condition_per_TR
        
        
    def load_prf_estimates(self, participant_list = [], ses = 'mean', run_type = 'mean', model_name = None, 
                                fit_hrf = False, rsq_threshold = .1, positive_rf = True, size_std = 2.5,
                                mask_bool_df = None, stim_on_screen = [], mask_arr = True):
        
        """
        Load prf estimates, obtained from fitting fsnative surface with prfpy.
        Returns dataframe with estimates for all participants in participant list
        """
        
        ## load pRF estimates and models for all participants 
        print('Loading iterative estimates')
        group_estimates, group_prf_models = self.pRFModelObj.load_pRF_model_estimates(participant_list = participant_list,
                                                                    ses = ses, run_type = run_type, 
                                                                    model_name = model_name, 
                                                                    iterative = True,
                                                                    sj_space = self.prf_sj_space,
                                                                    mask_bool_df = mask_bool_df, 
                                                                    stim_on_screen = stim_on_screen,
                                                                    fit_hrf = fit_hrf)
        
        # convert estimates to dataframe, for ease of handling
        group_estimates_df = []
        for pp in participant_list:
            tmp_df = pd.DataFrame(group_estimates['sub-{sj}'.format(sj = pp)])
            tmp_df['sj'] = 'sub-{sj}'.format(sj = pp)
            tmp_df['vertex'] = tmp_df.index
            group_estimates_df.append(tmp_df)
        group_estimates_df = pd.concat(group_estimates_df, ignore_index=True)

        return group_estimates_df, group_prf_models
    
    def get_prf_vertex_index(self, participant_list = [], ses = 'mean', run_type = 'mean', model_name = None, 
                                    fit_hrf = False, rsq_threshold = .1, positive_rf = True, size_std = 2.5,
                                    mask_bool_df = None, stim_on_screen = [], mask_arr = True, num_vert = None):
        
        """get pRF vertex indices that are above certain rsq threshold
        to later use to subselect best fitting vertices within ROI
        (makes decoding faster) 
        """
        
        # first get prf estimates for participant list
        group_estimates_df, _ = self.load_prf_estimates(participant_list = participant_list, 
                                                        ses = ses, run_type = run_type, model_name = model_name, 
                                                        fit_hrf = fit_hrf, positive_rf = positive_rf, size_std = size_std,
                                                        mask_bool_df = mask_bool_df, stim_on_screen = stim_on_screen, 
                                                        mask_arr = mask_arr)
        
        # get index dataframe with values for all participants
        group_index_df = []
        for pp in participant_list:
            if num_vert is not None:
                # get top X vertex
                best_vertex = group_estimates_df[group_estimates_df['sj'] == 'sub-{sj}'.format(sj = pp)].sort_values(by=['r2'], ascending=False).iloc[:num_vert].vertex.values
            else:
                # select based on r2 threshold
                best_vertex = group_estimates_df[((group_estimates_df['sj'] == 'sub-{sj}'.format(sj = pp)) &\
                                                (group_estimates_df['r2'] >= rsq_threshold))].vertex.values 
            
            tmp_df = pd.DataFrame({'vertex': best_vertex})
            tmp_df['sj'] = 'sub-{sj}'.format(sj = pp)
            group_index_df.append(tmp_df)
        group_index_df = pd.concat(group_index_df, ignore_index=True)
        
        return group_index_df
        
    def get_prf_ROI_data(self, participant = None, roi_name = 'V1', index_arr = [], overwrite_T1 = False, 
                                overwrite_func = False, file_ext = None):
        
        """Get pRF data for the ROI of a participant, averaged across runs,
        and return dataframe in a format compatible with braindecoder 
        """
        
        ## load pRF bold files

        ## get list of files to load
        prf_bold_filelist = self.MRIObj.mri_utils.get_bold_file_list(participant, 
                                                                    task = 'pRF', ses = 'all', 
                                                                    file_ext = file_ext,
                                                                    postfmriprep_pth = self.MRIObj.postfmriprep_pth, 
                                                                    acq_name = self.MRIObj.acq, 
                                                                    hemisphere = 'BH')
        
        ## get masked ROI data
        # averaged across runs
        masked_data_df = self.get_ROImask_data(participant, 
                                            file_list = prf_bold_filelist, 
                                            task = 'pRF', 
                                            run_type = 'mean', ses = 'mean', 
                                            roi_name = roi_name, 
                                            index_arr = index_arr,
                                            overwrite_T1 = overwrite_T1,
                                            overwrite_func = overwrite_func)
        
        return masked_data_df
    
    def get_prf_stim_grid(self, participant = None, ses = 'mean', mask_bool_df = None, stim_on_screen = [], 
                                prf_condition_per_TR = []):
        
        """Get prf stimulus array and grid coordinates for participant
        """
        
        ## get stimulus array (time, x, y)
        prf_stimulus_dm = self.make_prf_DM(participant, 
                                        ses = ses, 
                                        mask_bool_df = mask_bool_df, 
                                        stim_on_screen = stim_on_screen,
                                        prf_condition_per_TR = prf_condition_per_TR,
                                        filename = None, 
                                        transpose_dm = False)
        
        ## get grid coordinates
        size = prf_stimulus_dm.shape[-1]

        coord_x = np.linspace(-self.convert_pix2dva(self.MRIObj.screen_res[0]/2), 
                      self.convert_pix2dva(self.MRIObj.screen_res[0]/2), size+1, endpoint=True)
        coord_x = np.hstack((coord_x[:int(size/2)], coord_x[int(size/2 + 1):]))
        
        coord_y = np.linspace(-self.convert_pix2dva(self.MRIObj.screen_res[1]/2), 
                      self.convert_pix2dva(self.MRIObj.screen_res[1]/2), size+1, endpoint=True)
        coord_y = np.hstack((coord_y[:int(size/2)], coord_y[int(size/2 + 1):]))

        y, x = np.meshgrid(np.flip(coord_y), 
                            coord_x)
        x_deg = x.ravel().astype(np.float32)
        y_deg = y.ravel().astype(np.float32)

        prf_grid_coordinates = pd.DataFrame({'x':x_deg, 'y': y_deg}).astype(np.float32)
        
        return prf_stimulus_dm, prf_grid_coordinates
    
    def make_prf_DM(self, participant = None, ses = 'mean', mask_bool_df = None, stim_on_screen = [], filename = None,
                        prf_condition_per_TR = [], transpose_dm = False):

        """
        Get prf stimulus array for participant
        """

        visual_dm = None
        save_dm = False

        if isinstance(ses, str) and 'ses' in ses: # to account for differences in input
            ses = re.search(r'(?<=ses-).*', ses)[0]

        if filename:
            if op.exists(filename):
                print('Loading {file}'.format(file = filename))
                visual_dm = np.load(filename)
            else:
                save_dm = True

        # make design matrix
        if visual_dm is None:

            print('Making DM for sub-{pp}'.format(pp = participant))
            
            ## if we want to mask DM, then load behav mask
            if mask_bool_df is not None:

                # if we set a specific session, then select that one, else combine
                if ses == 'mean':
                    mask_bool = mask_bool_df[mask_bool_df['sj'] == 'sub-{sj}'.format(sj = participant)]['mask_bool'].values
                else:
                    mask_bool = mask_bool_df[(mask_bool_df['ses'] == 'ses-{s}'.format(s = ses)) & \
                                        (mask_bool_df['sj'] == 'sub-{sj}'.format(sj = participant))]['mask_bool'].values
                dm_mask = np.prod(mask_bool, axis = 0)
            else:
                dm_mask = np.ones(np.array(stim_on_screen).shape[0])

            # multiply boolean array with mask
            stim_on_screen = stim_on_screen * dm_mask

            # set DM size 
            dm_size = 80
            
            # get prf bar pass dict, for reference
            prf_bar_pass_dict = self.create_prf_barpass_arr(dm_size = dm_size)

            # save screen display for each TR
            visual_dm_array = np.zeros((len(prf_condition_per_TR), dm_size, dm_size))
            i = 0

            # loop over bar pass directions
            for trl, bartype in enumerate(prf_condition_per_TR): 
                
                if bartype not in np.array(['empty','empty_long']): # if not empty screen

                    visual_dm_array[trl] = prf_bar_pass_dict[bartype][i] * stim_on_screen[trl]

                    # increment counter
                    if trl < (len(prf_condition_per_TR) - 1):
                        i = i+1 if prf_condition_per_TR[trl] == prf_condition_per_TR[trl+1] else 0  

            if transpose_dm:
                # swap axis to have time in last axis [x,y,t]
                visual_dm = visual_dm_array.transpose([1,2,0])
            else:
                visual_dm = visual_dm_array.copy() # t, x, y

            if save_dm:
                # save design matrix
                print('Making and saving {file}'.format(file = filename))
                np.save(filename, visual_dm)  
                    
        return self.MRIObj.mri_utils.normalize(visual_dm).astype(np.float32)
    
    def create_prf_barpass_arr(self, dm_size = 80):

        """
        Get dict with prf bar pass array (for all bar pass directions)
        """

        # set bar width
        dm_bar_width = int(dm_size * self.MRIObj.bar_width['FA'])

        ## first create L-R bar pass array
        barpass_arr = []
        coord_ind_arr = (np.arange(-1, dm_size, dm_bar_width/2)+1).astype(int)

        for b_ind in range(self.MRIObj.pRF_nr_TRs['L-R']):

            trl_arr = np.zeros((dm_size, dm_size))
            
            if b_ind == 0:
                trl_arr[:int(coord_ind_arr[int(b_ind)] + (dm_bar_width/2))] = 1
            elif b_ind <= self.MRIObj.pRF_nr_TRs['L-R']:
                trl_arr[coord_ind_arr[int(b_ind - 1)]:int(coord_ind_arr[int(b_ind)] + (dm_bar_width/2))] = 1

            barpass_arr.append(trl_arr)
        barpass_arr = np.stack(barpass_arr) # time, x, y

        ## create dict of barpass arrays, for each condition
        prf_bar_pass_dict = {'L-R': barpass_arr,
                             'R-L': np.flip(barpass_arr, axis=0),
                             'U-D': np.rot90(barpass_arr, axes=(1, 2)),
                             'D-U': np.rot90(barpass_arr, axes=(2, 1)),
                             }
        
        return prf_bar_pass_dict

    def create_barpass_arr(self, dm_size = 80, num_b = 8):

        """
        Get dict with FA bar pass array (for both horizontal and vertical bar positions)
        """

        # set bar width
        dm_bar_width = int(dm_size/num_b)

        ## first create horizontal bar pass array
        barpass_arr = []
        coord_ind_arr = np.linspace(dm_bar_width/2, dm_size + dm_bar_width/2, 9, endpoint=True)[:-1].astype(int)

        for b_ind in range(num_b):

            trl_arr = np.zeros((dm_size, dm_size))
            
            if b_ind == 0:
                trl_arr[:int(coord_ind_arr[int(b_ind)] + (dm_bar_width/2))] = 1
            else:
                trl_arr[int(coord_ind_arr[int(b_ind)] - (dm_bar_width/2)):int(coord_ind_arr[int(b_ind)] + (dm_bar_width/2))] = 1

            barpass_arr.append(trl_arr)
        barpass_arr = np.stack(barpass_arr) # time, x, y

        ## create dict of barpass arrays, for each condition
        bar_pass_dict = {'horizontal': barpass_arr,
                        'vertical': np.rot90(barpass_arr, axes=(2, 1))
                        }
        
        return bar_pass_dict

    def make_run_visual_dm(self, midpoint_bar = None, direction_bar = None, trial_mask = None, dm_size = 80):

        """
        make DM array for given run 
        """

        # number of possible bar positions
        num_b = int(self.MRIObj.screen_res[0]/self.bar_width_pix[0])
        # reference bar midpoint coordinates in pixels
        ref_midpoints = np.linspace(-self.MRIObj.screen_res[0]/2 + self.bar_width_pix[0]/2, 
                                    self.MRIObj.screen_res[0]/2 - self.bar_width_pix[0]/2, 
                                    num_b)
        
        # get dict with FA bar pass array (for both horizontal and vertical bar positions)
        bar_pass_dict = self.create_barpass_arr(dm_size = dm_size, num_b = num_b)

        # save screen display for each TR
        visual_dm_array = np.zeros((len(direction_bar), dm_size, dm_size))
        
        for trl, bar_center in enumerate(midpoint_bar): # loop over trials

            if direction_bar[trl] == 'vertical':

                bar_coord = bar_center[-1]

            elif direction_bar[trl] == 'horizontal':  

                bar_coord = bar_center[0]

            # get index of bar poisition
            bar_ind = np.where((ref_midpoints == bar_coord))[0][0]

            # store array
            visual_dm_array[trl] = bar_pass_dict[direction_bar[trl]][bar_ind]

        # if we provided a trial mask, then set those trials to 0
        if trial_mask is not None:
            visual_dm_array[trial_mask] = 0

        return self.MRIObj.mri_utils.normalize(visual_dm_array).astype(np.float32)

    def make_DM_dict(self, pp_bar_pos_df = None, pp_prf_bar_coords_dict = None, dm_size = 80):
    
        """
        return dict for each participant run,
        with visual DM for each type of regressor (attended bar, unattended bar, overlap, both bars, etc)
        
        ex:
        out_dict['ses-1_run-1'] = {'att_bar': [trial, x,y], 'unatt_bar': [trial,x,y], ...}

        Parameters
        ----------
        pp_bar_pos_df: df/dict
            from behavioral object, with bar position info of participant runs
        pp_prf_bar_coords_dict: df/dict
            dict with prf bar positions for participant, vertical/horizontal runs
            when given will be used to mask out not visible bar locations
        """ 
        
        # set empty dicts
        out_dict = {}
        
        for ses_key in pp_bar_pos_df.keys():
            for run_key in pp_bar_pos_df[ses_key].keys():
                
                # get run identifier
                ses_run_id = '{sn}_{rn}'.format(sn = ses_key, rn = run_key)
                print('loading bar positions for {srid}'. format(srid = ses_run_id))
                
                # initialize dict
                out_dict[ses_run_id] = {}
                
                ## get bar position df for run
                run_bar_pos_df = pp_bar_pos_df[ses_key][run_key]

                ## get run bar midpoint and direction values
                # for each bar type (arrays will have len == total number of trial types)
                AttBar_bar_midpoint, AttBar_bar_pass_direction = run_bar_pos_df.loc[(run_bar_pos_df['attend_condition'] == 1), 
                                                                                    ['bar_midpoint_at_TR', 'bar_pass_direction_at_TR']].to_numpy()[0]
                UnattBar_bar_midpoint, UnattBar_bar_pass_direction = run_bar_pos_df.loc[(run_bar_pos_df['attend_condition'] == 0), 
                                                                                    ['bar_midpoint_at_TR', 'bar_pass_direction_at_TR']].to_numpy()[0]

                # if possible, mask out not visible bar positions
                if pp_prf_bar_coords_dict is not None:
                    # get trial indices to mask out
                    t_mask = self.MRIObj.beh_utils.get_trial_ind_mask(AttBar_bar_midpoint = AttBar_bar_midpoint, 
                                                                    AttBar_bar_pass_direction = AttBar_bar_pass_direction,
                                                                    UnattBar_bar_midpoint = UnattBar_bar_midpoint, 
                                                                    UnattBar_bar_pass_direction = UnattBar_bar_pass_direction,
                                                                    prf_bar_coords_dict = pp_prf_bar_coords_dict)
                else:
                    t_mask = None
                    
                ## GET DM FOR ATTENDED BAR
                out_dict[ses_run_id]['att_bar'] = self.make_run_visual_dm(midpoint_bar = AttBar_bar_midpoint, 
                                                                        direction_bar = AttBar_bar_pass_direction, 
                                                                        dm_size = dm_size,
                                                                        trial_mask = t_mask)
                ## GET DM FOR UNATTENDED BAR
                out_dict[ses_run_id]['unatt_bar'] = self.make_run_visual_dm(midpoint_bar = UnattBar_bar_midpoint, 
                                                                        direction_bar = UnattBar_bar_pass_direction, 
                                                                        dm_size = dm_size,
                                                                        trial_mask = t_mask)

                ## GET DM FOR OVERLAP OF BARS
                out_dict[ses_run_id]['overlap'] = self.MRIObj.mri_utils.get_bar_overlap_dm(np.stack((out_dict[ses_run_id]['att_bar'],
                                                                                                    out_dict[ses_run_id]['unatt_bar'])))

                ## GET DM FOR BOTH BARS COMBINED (FULL STIM THAT WAS ON SCREEN)
                stimulus_dm = np.sum(np.stack((out_dict[ses_run_id]['att_bar'],
                                                out_dict[ses_run_id]['unatt_bar'])), axis = 0)
                stimulus_dm[stimulus_dm >=1] = 1
                
                out_dict[ses_run_id]['full_stim'] = stimulus_dm

        # get keys for each condition
        condition_dm_keys = list(out_dict[ses_run_id].keys())
                
        return out_dict, condition_dm_keys

    def load_encoding_model_pars(self, participant_list = [], ROI_list = ['V1'], model_type = 'gauss_hrf',
                                    prf_file_ext = '_cropped_dc_psc.nii.gz', ses = 'mean', 
                                    mask_bool_df = None, stim_on_screen = [], pars_as_df = True):
        
        """
        Load encoding model fit params, for all participants and ROIs
        returns info in dict
        """

        # store pars for all ROIs and participants
        prf_pars_gd_dict = {}
        prf_best_voxels_dict = {}
        
        for roi_name in ROI_list:
    
            print('Loading enconding model parameters for %s'%roi_name)
            
            prf_pars_gd_dict[roi_name] = {}
            prf_best_voxels_dict[roi_name] = {}
            
            for participant in tqdm(participant_list):

                ## load prf parameters (decoder)
                pars_gd = self.load_ROI_encoding_model_pars(participant = participant, 
                                                            task = 'pRF', 
                                                            roi_name = roi_name, 
                                                            model_type = model_type)

                # get masked prf ROI data, averaged across runs
                prf_masked_data_df = self.get_prf_ROI_data(participant = participant, 
                                                            roi_name = roi_name, 
                                                            index_arr = [], 
                                                            overwrite_T1 = False, 
                                                            overwrite_func = False,
                                                            file_ext = prf_file_ext)

                # get prf stimulus DM and grid coordinates
                prf_stimulus_dm, prf_grid_coordinates = self.get_prf_stim_grid(participant = participant, 
                                                                                ses = ses, 
                                                                                mask_bool_df = mask_bool_df, 
                                                                                stim_on_screen = stim_on_screen,
                                                                                prf_condition_per_TR = self.prf_condition_per_TR)

                ## need to select best voxels 
                # get rsq
                r2_gd = self.get_encoding_r2(data = prf_masked_data_df, 
                                            grid_coordinates = prf_grid_coordinates, 
                                            model_type = model_type,
                                            paradigm = prf_stimulus_dm,  
                                            pars_gd = pars_gd)
                
                # get array with best voxel indices 
                best_voxels = self.get_best_voxels(pars_gd = pars_gd, r2_gd = r2_gd,  
                                                        x_lim = [-6,6], y_lim = [-6,6], 
                                                        size_min = .3, size_max = 'std', std_val = 3, 
                                                        n_vox = 300)
                
                # store in dict
                prf_pars_gd_dict[roi_name]['sub-{sj}'.format(sj = participant)] = pars_gd.join(r2_gd.to_frame('r2'))
                prf_best_voxels_dict[roi_name]['sub-{sj}'.format(sj = participant)] = best_voxels
                
                # also load r2 for gauss, for comparison
                if 'gauss' not in model_type:
                    pars_gauss_gd = self.load_ROI_encoding_model_pars(participant = participant, 
                                                                task = 'pRF', 
                                                                roi_name = roi_name, 
                                                                model_type = 'gauss_hrf')
                    r2_gauss_gd = self.get_encoding_r2(data = prf_masked_data_df, 
                                                        grid_coordinates = prf_grid_coordinates, 
                                                        model_type = 'gauss_hrf',
                                                        paradigm = prf_stimulus_dm,  
                                                        pars_gd = pars_gauss_gd)
                    
                    prf_pars_gd_dict[roi_name]['sub-{sj}'.format(sj = participant)] = prf_pars_gd_dict[roi_name]['sub-{sj}'.format(sj = participant)].join(r2_gauss_gd.to_frame('r2_gauss'))

        # if we want to convert to dataframe instead
        if pars_as_df:

            pars_df = []

            for roi_key in prf_pars_gd_dict.keys():
                for pp_key in prf_pars_gd_dict[roi_key].keys():

                    tmp_df = prf_pars_gd_dict[roi_key][pp_key].copy()
                    # set roi specific index as columns
                    tmp_df.loc[:, ['ROI_index']] = prf_pars_gd_dict[roi_key][pp_key].index.values
                    # create new column, selecting if voxel was used for decoding
                    tmp_df.loc[:, ['best_voxel']] = prf_pars_gd_dict[roi_key][pp_key].index.isin(prf_best_voxels_dict[roi_key][pp_key])

                    # now add ROI info and sub info
                    tmp_df.loc[:, ['ROI']] = roi_key
                    tmp_df.loc[:, ['sj']] = pp_key

                    # append
                    pars_df.append(tmp_df)

            pars_df = pd.concat(pars_df, ignore_index=True)

            return pars_df
        else:
            return prf_pars_gd_dict, prf_best_voxels_dict


    def get_trl_ecc_dist_df(self, position_df = None, bars_pos = 'parallel', bar_ecc = 'far', abs_inter_bar_dist = 5):
        
        """Given a data frame with bar positions and indices for each trial for a given run,
        return dataframe with trial indices that all have a specific attended ecc,
        inter bar distance, and bar position (ex: parallel vs crossed)
        
        this can later be used to transform 'collapsable' trials and average across stim
        """
        
        if bars_pos == 'parallel':
            
            t_ind_list = []
            for index, row in position_df[(position_df['attend_condition'] == True) &\
                                        (position_df['bar_ecc'] == bar_ecc) &\
                                        (position_df['abs_inter_bar_dist'] == abs_inter_bar_dist)].iterrows():
                
                trl_df = position_df[(position_df['trial_ind'] == row['trial_ind'])]
                
                if (trl_df[trl_df['attend_condition'] == True].x_pos.values[0] < 0) &\
                (trl_df[trl_df['attend_condition'] == True].x_pos.values[0] < trl_df[trl_df['attend_condition'] == False].x_pos.values[0]):
                    
                    t_ind_list.append(row['trial_ind'])
                
                elif (trl_df[trl_df['attend_condition'] == True].x_pos.values[0] > 0) &\
                (trl_df[trl_df['attend_condition'] == True].x_pos.values[0] > trl_df[trl_df['attend_condition'] == False].x_pos.values[0]):
                    
                    t_ind_list.append(row['trial_ind'])
                elif (trl_df[trl_df['attend_condition'] == True].y_pos.values[0] > 0) &\
                (trl_df[trl_df['attend_condition'] == True].y_pos.values[0] > trl_df[trl_df['attend_condition'] == False].y_pos.values[0]):
                    
                    t_ind_list.append(row['trial_ind'])
                elif (trl_df[trl_df['attend_condition'] == True].y_pos.values[0] < 0) &\
                (trl_df[trl_df['attend_condition'] == True].y_pos.values[0] < trl_df[trl_df['attend_condition'] == False].y_pos.values[0]):
                    
                    t_ind_list.append(row['trial_ind'])
    
            output_df = position_df[(position_df['attend_condition'] == True) &\
                                    (position_df['trial_ind'].isin(t_ind_list))]
        else:
            output_df = None ## not implemented yet
            
        return output_df
          
    def plot_decoded_stim(self,reconstructed_stimulus = None, frame = 0, vmin = 0, vmax = None, cmap = 'viridis',
                                xticklabels = True, yticklabels = True, square = True):
        
        """Quick func to plot reconstructed stimulus (for easy checking of outputs)
        """
        
        if vmax is None:
            vmax = np.quantile(reconstructed_stimulus.values.ravel(), 0.97)
        
        return sns.heatmap(reconstructed_stimulus.stack('y', dropna = False).loc[frame].iloc[::-1, :], vmin = vmin, vmax = vmax, cmap = cmap,
                           xticklabels = xticklabels, yticklabels = yticklabels, square = square)
        
    def get_run_trial_pairs(self, DM_arr = None):
        
        """find trials where attended bar and unattended bar in same position
        to correlate with each other
        """
        
        ## get trial indices that were NOT masked 
        trials_arr = np.where((np.sum(np.sum(DM_arr, axis = -1), axis = -1) != 0))[0]
        
        same_bar_pos_ind = []

        for i in trials_arr:

            ind_list = np.where((DM_arr.reshape(132, -1) == DM_arr[i].ravel()).all(-1))[0]
            same_bar_pos_ind.append(ind_list)
        same_bar_pos_ind = np.vstack(same_bar_pos_ind)
        
        # get unique
        same_bar_pos_ind = np.unique(same_bar_pos_ind, axis=0)
        
        return same_bar_pos_ind
    
    def get_flipped_trial_ind(self, trl_ind = None, DM_arr = None):
        
        """ find the flipped trial (same positions, different attended bar)
        """
    
        ## find trials where attended bar and unattended bar in same position
        # to correlate with each other
        same_bar_pos_ind = self.get_run_trial_pairs(DM_arr = DM_arr)
        
        # find which other trial bars are in same location
        pairs_arr = [np.where(same_bar_pos_ind == trl_ind)[0][0], np.where(same_bar_pos_ind == trl_ind)[-1][0]]

        if pairs_arr[-1] == 1:
            flip_trl_ind = same_bar_pos_ind[pairs_arr[0], 0]
        else:
            flip_trl_ind = same_bar_pos_ind[pairs_arr[0], 1]
            
        return flip_trl_ind
    
    def get_parallel_average_stim(self, reconstructed_stimulus = None, position_df = None, bar_ecc = 'far', abs_inter_bar_dist = 5, 
                                        flipped_stim = False, DM_arr = None):
        
        """Get average (reconstructed) stimulus
        for parallel bar trials of specific ecc and bar distance
        can also return flipped case (same bar pos, different attended bar)
        """
        
        # flag if symmetrical trial
        if (bar_ecc == 'far' and abs_inter_bar_dist == 5) | (bar_ecc == 'middle' and abs_inter_bar_dist == 3) | (bar_ecc == 'near' and abs_inter_bar_dist == 1):
            sym_trial = True
        else:
            sym_trial = False
                    
        # get collapsable trials
        df_ecc_dist = self.get_trl_ecc_dist_df(position_df = position_df, bars_pos = 'parallel', 
                                               bar_ecc = bar_ecc, abs_inter_bar_dist = abs_inter_bar_dist)
           
        ## iterate over trials, rotate stim as needed and stack
        # reference will be vertical bars with attended bar on the left visual field
        average_stim = []

        for ind in df_ecc_dist.trial_ind.unique():
                
            trl_df = position_df[position_df['trial_ind'] == ind]
            trl_ind = ind
            
            # reference trial (stays the same)
            if not trl_df.query('x_pos < 0 & attend_condition').empty:
                # if we want the flipped case, and conditions are not symmetrical
                if flipped_stim == True and sym_trial == False:
                    trl_ind = self.get_flipped_trial_ind(trl_ind = ind, DM_arr = DM_arr)
                trl_stim = reconstructed_stimulus.stack('y', dropna = False).loc[trl_ind].iloc[::-1, :].to_numpy()
            # trial to be flipped horizontally (this is mirrored left and right)
            elif not trl_df.query('x_pos > 0 & attend_condition').empty:
                # if we want the flipped case, and conditions are not symmetrical
                if flipped_stim == True and sym_trial == False:
                    trl_ind = self.get_flipped_trial_ind(trl_ind = ind, DM_arr = DM_arr)
                trl_stim = self.flip_arr(reconstructed_stimulus.stack('y', dropna = False).loc[trl_ind].iloc[::-1, :].to_numpy(),
                                         flip_type='lr')
            # trial to be rotated 90deg CCW
            elif not trl_df.query('y_pos > 0 & attend_condition').empty:
                # if we want the flipped case, and conditions are not symmetrical
                if flipped_stim == True and sym_trial == False:
                    trl_ind = self.get_flipped_trial_ind(trl_ind = ind, DM_arr = DM_arr)
                trl_stim = np.rot90(reconstructed_stimulus.stack('y', dropna = False).loc[trl_ind].iloc[::-1, :].to_numpy(),
                                    axes=(0, 1))
            # trial to be rotated 90deg CW
            elif not trl_df.query('y_pos < 0 & attend_condition').empty:
                # if we want the flipped case, and conditions are not symmetrical
                if flipped_stim == True and sym_trial == False:
                    trl_ind = self.get_flipped_trial_ind(trl_ind = ind, DM_arr = DM_arr)
                trl_stim = np.rot90(reconstructed_stimulus.stack('y', dropna = False).loc[trl_ind].iloc[::-1, :].to_numpy(),
                                    axes=(1, 0))
                
            average_stim.append(trl_stim)
        average_stim = np.stack(average_stim)
                        
        # and average
        average_stim = np.nanmean(average_stim, axis = 0)

        # if we want the flipped trials for symmetrical cases
        if flipped_stim == True and sym_trial == True:
            average_stim = np.flip(average_stim, axis = 1)
            
        return average_stim
    
    def get_crossed_average_stim(self, reconstructed_stimulus = None, position_df = None, bar_ecc = 'far', same_ecc = True, 
                                        flipped_stim = False, DM_arr = None):
        
        """Get average (reconstructed) stimulus
        for crossed bar trials of specific ecc. 
        Note, for each attended ecc there are 2 cases:
            a) both bars at the same ecc
            b) Att and Unatt bars at different eccs [far, near] or [middle, far] or [near, middle]
            
        can also return flipped case (same bar pos, different attended bar)
        """
        
        # make reference dict with unique conditions of attended and unattend
        uniq_cond_dict = {'far': 'near', 'near': 'middle', 'middle': 'far'}
        
        flip_att_bool = 0
        
        ### a) stack trials where both bars at the same ecc
        if same_ecc: 
            # select trials with crossed bars at that ecc
            bar_ecc_df = position_df[(position_df['bar_ecc'] == bar_ecc) &\
                                    (position_df['bars_pos'] == 'crossed')]
        
            # first find the indices where both attended bar and unattended bar are at the same ecc
            trial_ind_same_ecc = np.array([val for val in bar_ecc_df[(bar_ecc_df['attend_condition'] == True)].trial_ind.values if val in bar_ecc_df[(bar_ecc_df['attend_condition'] == False)].trial_ind.values])

            # filter df for those trials (keep attended and unattended to collapse)
            df_cross_ecc = bar_ecc_df[(bar_ecc_df['trial_ind'].isin(trial_ind_same_ecc))]
        
        else:
            # get relevant trials indices
            trial_ind_diff_ecc = np.array([val for val in position_df[(position_df['attend_condition'] == True) &\
                                                (position_df['bars_pos'] == 'crossed') &\
                                                (position_df['bar_ecc'] == bar_ecc)].trial_ind.values if val in position_df[(position_df['attend_condition'] == False) &\
                                                                                                                            (position_df['bars_pos'] == 'crossed') &\
                                                                                                                        (position_df['bar_ecc'] == uniq_cond_dict[bar_ecc])].trial_ind.values])
            # filter df for those trials
            df_cross_ecc = position_df[(position_df['trial_ind'].isin(trial_ind_diff_ecc))]
                
        ## iterate over trials, rotate stim as needed and stack
        average_stim = []

        for ind in df_cross_ecc.trial_ind.unique():
                        
            if flipped_stim == True and same_ecc == False:
                ind = self.get_flipped_trial_ind(trl_ind = ind, DM_arr = DM_arr)
                flip_att_bool = 1
                
            trl_df = position_df[position_df['trial_ind'] == ind]
            #print(ind)
            
            # reference trial
            # attended bar vertical, on the left. unattended bar horizontal, upper meridian
            if not trl_df[(trl_df['x_pos'] < 0) & (trl_df['attend_condition'] == bool(np.abs(1 - flip_att_bool)))].empty: 
                
                trl_stim = reconstructed_stimulus.stack('y', dropna = False).loc[ind].iloc[::-1, :].to_numpy()
                
                # trial to also be flipped vertically (this is mirrored up and down) 
                if not trl_df[(trl_df['y_pos'] < 0) & (trl_df['attend_condition'] == bool(np.abs(0 - flip_att_bool)))].empty:
                    trl_stim = self.flip_arr(trl_stim, flip_type='ud')
                
            # trial to be flipped horizontally (this is mirrored left and right) 
            elif not trl_df[(trl_df['x_pos'] > 0) & (trl_df['attend_condition'] == bool(np.abs(1 - flip_att_bool)))].empty:
                
                trl_stim = self.flip_arr(reconstructed_stimulus.stack('y', dropna = False).loc[ind].iloc[::-1, :].to_numpy(),
                                                flip_type='lr')
                
                # trial to also be flipped vertically (this is mirrored up and down) 
                if not trl_df[(trl_df['y_pos'] < 0) & (trl_df['attend_condition'] == bool(np.abs(0 - flip_att_bool)))].empty:
                    trl_stim = self.flip_arr(trl_stim, flip_type='ud')
                
            # trial to be rotated 90deg CW
            elif not trl_df[(trl_df['x_pos'] < 0) & (trl_df['attend_condition'] == bool(np.abs(0 - flip_att_bool)))].empty: 
                
                trl_stim = np.rot90(reconstructed_stimulus.stack('y', dropna = False).loc[ind].iloc[::-1, :].to_numpy(),
                                                        axes=(1, 0))
                
                if not trl_df[(trl_df['y_pos'] > 0) & (trl_df['attend_condition'] == bool(np.abs(1 - flip_att_bool)))].empty:
                    trl_stim = self.flip_arr(trl_stim, flip_type='lr')
            
            # trial to be rotated 90deg CCW + flip horizontally
            elif not trl_df[(trl_df['x_pos'] > 0) & (trl_df['attend_condition'] == bool(np.abs(0 - flip_att_bool)))].empty: 
                
                trl_stim = self.get_diag_mirror_arr(reconstructed_stimulus.stack('y', dropna = False).loc[ind].iloc[::-1, :].to_numpy(),
                                                        diag_type = 'minor')
                
                if not trl_df[(trl_df['y_pos'] > 0) & (trl_df['attend_condition'] == bool(np.abs(1 - flip_att_bool)))].empty:
                    trl_stim = np.rot90(reconstructed_stimulus.stack('y', dropna = False).loc[ind].iloc[::-1, :].to_numpy(),
                                                        axes=(0, 1))
            
            average_stim.append(trl_stim)
        average_stim = np.stack(average_stim)
                        
        # and average 
        average_stim = np.nanmean(average_stim, axis = 0)

        # if we want the flipped trials for symmetrical cases
        if flipped_stim == True and same_ecc == True:
            average_stim = self.get_diag_mirror_arr(average_stim, diag_type = 'major')
            
        return average_stim
    
    def get_decoder_grid_coords(self, y_coords_deg = None, x_coords_deg = None):
        
        """Get grid coordinates for FA task, to use in decoder 
        default will be (8x8 grid)
        """

        if y_coords_deg is None:
            y_coords_deg = self.y_coords_deg
        if x_coords_deg is None:
            x_coords_deg = self.x_coords_deg

        new_y, new_x = np.meshgrid(np.flip(np.sort(y_coords_deg)), 
                                    np.sort(x_coords_deg))
        fa_grid_coordinates = pd.DataFrame({'x':new_x.ravel().astype(np.float32), 'y': new_y.ravel().astype(np.float32)}).astype(np.float32)
        
        return fa_grid_coordinates
    
    def get_diag_mirror_arr(self, og_arr = None, diag_type = 'major'):
        
        """
        return diagonally mirrored version of array (relative to major \ or minor / diagonal)
        """  
        if diag_type == 'major':
            return np.flip(np.rot90(og_arr, axes =(0,1)), axis = 0)
        elif diag_type == 'minor':
            return np.flip(np.rot90(og_arr, axes =(0,1)), axis = 1)
        
    def flip_arr(self, og_arr = None, flip_type = 'lr'):
        
        """
        return flipped version of array (left-right vs up-down)
        """ 
        if flip_type == 'lr':
            return np.flip(og_arr, axis = 1)
        elif flip_type == 'ud':
            return np.flip(og_arr, axis = 0)
    
    def downsample_DM(self, DM_arr = None, dm_size = 80, new_size = 8):
        
        """downsample FA FAM to 8x8 grid
        to correlate with reconstructed image
        """
        
        ## reduce dimensionality 
        downsample_FA_DM = scipy.ndimage.zoom(DM_arr, (1, new_size/dm_size, new_size/dm_size))
        # binarize again
        downsample_FA_DM[downsample_FA_DM > .5] = 1
        downsample_FA_DM[downsample_FA_DM<.5] = 0
        
        return downsample_FA_DM
    
    def get_min_dist_competing_bar(self, trial_num = 0, coord_y_bar = None, coord_x_bar = None, 
                                        dm_comp_bar = None, fa_grid_coordinates = None):
        
        """Get minimum distance to competing bar
        note that provided coordinates expected to be in deg
        """
        
        # first get position index for pixels of competing bar
        competing_bar_pix_ind = np.where(dm_comp_bar.reshape(dm_comp_bar.shape[0], -1)[trial_num])[0]
        
        # from fa coordinates df, subselect pixels and respective coordinates
        competing_bar_coords_df = fa_grid_coordinates.iloc[competing_bar_pix_ind]

        # then get distances
        competing_dist_list = [np.sqrt((coord_x_bar - new_coords.x) ** 2 + (coord_y_bar - new_coords.y) ** 2) for _, new_coords in competing_bar_coords_df.iterrows()]
        
        # return smallest dist
        return np.min(competing_dist_list)
        
    def get_pix_intensity(self, stim_df = None, att_dm_arr = None, unatt_dm_arr = None, fa_grid_coordinates = None,
                                overlap_only = False):
        
        """
        per reconstructed stim, get intensity values per pixel
        for when there was an attended/unattended bar at coordinate
        returns intensity values for all pixel locations
        """
        
        ## make empty df to store intesity values
        # for all pixels
        pixel_df = []

        ## iterate over pixels
        for index, pix_coord_deg in fa_grid_coordinates.iterrows():
            
            ## stim intensity values for pixel across all trials
            stim_pix_trials = stim_df.iloc[...,index].values
            
            # get trial numbers 
            # where there was an attended or unattended bar at pixel coordinate
            trials_att = np.where(att_dm_arr.reshape(len(stim_pix_trials), -1)[...,index] == 1)[0]
            trials_unatt = np.where(unatt_dm_arr.reshape(len(stim_pix_trials), -1)[...,index] == 1)[0]
            
            # check if pixel has any overlap trials            
            trials_overlap = np.array([val for val in trials_att if val in trials_unatt])
            
            # if we only want to look at overlap
            if overlap_only:
                trials_att = trials_overlap
                trials_unatt = trials_overlap
            else:
                # remove overlap trials
                trials_att = np.array([val for val in trials_att if val not in trials_overlap])
                trials_unatt = np.array([val for val in trials_unatt if val not in trials_overlap])

            # if there are any trials where bars fall in this coordinate
            if len(trials_att) > 0 and len(trials_unatt) > 0:

                ## get intensity values for pixel
                inten_att = stim_pix_trials[trials_att]
                inten_unatt = stim_pix_trials[trials_unatt]

                ## get minimum distance to competing bar
                # when attended, distance to unattended
                min_dist_att = np.array([self.get_min_dist_competing_bar(trial_num = t, 
                                                                    coord_y_bar = pix_coord_deg.y, 
                                                                    coord_x_bar = pix_coord_deg.x, 
                                                                    dm_comp_bar = unatt_dm_arr, 
                                                                    fa_grid_coordinates = fa_grid_coordinates) for t in trials_att])

                # when unattended, distance to attended bar
                min_dist_unatt =  np.array([self.get_min_dist_competing_bar(trial_num = t, 
                                                                    coord_y_bar = pix_coord_deg.y, 
                                                                    coord_x_bar = pix_coord_deg.x, 
                                                                    dm_comp_bar = att_dm_arr, 
                                                                    fa_grid_coordinates = fa_grid_coordinates) for t in trials_unatt])

                ## and eccentricity for the point
                # (in deg)
                pix_ecc = np.abs(pix_coord_deg.x + pix_coord_deg.y * 1j)

                # store in df
                # att
                c_att_df = pd.DataFrame({'intensity': inten_att,'min_dist': min_dist_att, 'trial_ind': trials_att})
                c_att_df.loc[:,'bar_type'] = 'att_bar'
                c_att_df.loc[:,'ecc'] = pix_ecc
                c_att_df.loc[:,'coord_ind'] = index
                
                # unatt
                c_unatt_df = pd.DataFrame({'intensity': inten_unatt,'min_dist': min_dist_unatt, 'trial_ind': trials_unatt})
                c_unatt_df.loc[:,'bar_type'] = 'unatt_bar'
                c_unatt_df.loc[:,'ecc'] = pix_ecc
                c_unatt_df.loc[:,'coord_ind'] = index

                # and append 
                pixel_df.append(pd.concat((c_att_df, c_unatt_df), ignore_index=True))
            else:
                print('no trials for screen position (%.2f,%.2f)'%(pix_coord_deg.x,pix_coord_deg.y))

        pixel_df = pd.concat(pixel_df, ignore_index=True)
    
        return pixel_df
    
    def get_group_pix_intensity(self, participant_list = [], ROI_list = ['V1'], overlap_only = False,
                                    reconstructed_stim_dict = None, lowres_DM_dict = None, data_keys_dict = []):
        
        """
        per ROI reconstructed stim, get value of pixel for when there was an attended/unattended bar there
        """
        
        ## get FA grid coordinates (8x8)
        # to use in pixel quantification
        fa_grid_coordinates = self.get_decoder_grid_coords()
        
        ROI_pixel_df = []
                
        for roi_name in ROI_list:
            print(roi_name)
            for participant in tqdm(participant_list):
                
                # for all runs, get pixel intensity df          
                pixel_df = []
                for dfkey in data_keys_dict['sub-{sj}'.format(sj = participant)]:
                    tmp_df = self.get_pix_intensity(stim_df = reconstructed_stim_dict[roi_name]['sub-{sj}'.format(sj = participant)][dfkey], 
                                                        att_dm_arr = lowres_DM_dict['sub-{sj}'.format(sj = participant)]['att_bar'][dfkey], 
                                                        unatt_dm_arr = lowres_DM_dict['sub-{sj}'.format(sj = participant)]['unatt_bar'][dfkey],
                                                        fa_grid_coordinates = fa_grid_coordinates,
                                                        overlap_only = overlap_only)
                    tmp_df.loc[:, 'run'] = 'run-{rn}'.format(rn = self.MRIObj.mri_utils.get_run_ses_from_str(dfkey)[0])
                    tmp_df.loc[:, 'ses'] = 'ses-{sn}'.format(sn = self.MRIObj.mri_utils.get_run_ses_from_str(dfkey)[1])
                    pixel_df.append(tmp_df)
                pixel_df = pd.concat(pixel_df, ignore_index=True)
                ## average over relevant dimensions
                #pixel_df = pixel_df.groupby(['coord_ind', 'bar_type', 'min_dist', 'ecc']).mean().reset_index()
                # add relevant info
                pixel_df.loc[:, 'ROI'] = roi_name
                pixel_df.loc[:, 'sj'] = 'sub-{sj}'.format(sj = participant)
                # and append
                ROI_pixel_df.append(pixel_df)
        ROI_pixel_df = pd.concat(ROI_pixel_df, ignore_index=True)
        
        return ROI_pixel_df
    
    def plot_avg_parallel_stim(self, average_stim = None, flip_average_stim = None, DM_trl_ind = None,
                                    bar_ecc = None, bar_dist = None, downsample_FA_DM = None, 
                                    vmin = 0, vmax = .4, cmap = 'magma', filename = None, annot = True):
        
        """Make 4x4 plot of recontruscted stim, averaged across conditions
        (and also reverse case)
        followed by the corresponding downsampled DM for inspection
        """
        
        ## correlate each reconstructed average condition with stim position

        # note - here we transpose the DM array when correlating because the average_stim we calculated
        # has a different format than the reconstructed stim outputted by brain decoder
        avg_corr, avg_pval = scipy.stats.pointbiserialr(average_stim.ravel(), 
                                                  downsample_FA_DM[DM_trl_ind].T.ravel())
        flip_avg_corr, flip_avg_pval = scipy.stats.pointbiserialr(flip_average_stim.ravel(), 
                                                            downsample_FA_DM[DM_trl_ind].T.ravel())
                
        bar_ecc_ind = {'far': 1, 'middle': 2, 'near': 3}
        
        # plot figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (10,10))

        ## attended leftmost
        sns.heatmap(average_stim, cmap = cmap, ax = axes[0][0], square = True, cbar = False,
                    annot=annot, annot_kws={"size": 7},
                    vmin = vmin, vmax = vmax)
        axes[0][0].set_title('%s ecc, dist = %i (attended left bar)'%(bar_ecc, bar_dist))

        ## DM
        axes[1][0].imshow(downsample_FA_DM[DM_trl_ind].T, cmap = 'binary_r', vmax = 1.5)
        # Add the patch to the Axes
        axes[1][0].add_patch(patches.Rectangle((bar_ecc_ind[bar_ecc] - .5, -.5), 1, 8, 
                                            linewidth = 2, edgecolor='purple', 
                                            facecolor='purple', hatch = '///'))
        # annotate correlation between reconstructed stim and DM
        axes[1][0].set_title(r"$\rho$ = {r}".format(r = '%.2f'%(avg_corr))+\
                                '   pval = {p}'.format(p = "{:.2e}".format(avg_pval)))

        ## attended rightmost
        sns.heatmap(flip_average_stim, cmap = 'magma', ax = axes[0][1], square = True, cbar = False,
                annot=annot, annot_kws={"size": 7},
                vmin = vmin, vmax = vmax)
        axes[0][1].set_title('flipped case (attended right bar)')

        ## DM
        axes[1][1].imshow(downsample_FA_DM[DM_trl_ind].T, cmap = 'binary_r', vmax = 1.5)
        # Add the patch to the Axes
        axes[1][1].add_patch(patches.Rectangle((bar_ecc_ind[bar_ecc] - .5 + bar_dist, -.5), 1, 8, 
                                            linewidth = 2, edgecolor='green', 
                                            facecolor='green', hatch = '///'))
        # annotate correlation between reconstructed stim and DM
        axes[1][1].set_title(r"$\rho$ = {r}".format(r = '%.2f'%(flip_avg_corr))+\
                                '   pval = {p}'.format(p = "{:.2e}".format(flip_avg_pval)))

        # save figure
        if filename is not None:
            fig.savefig(filename, dpi= 200)
            
    def plot_avg_crossed_stim(self, average_stim = None, flip_average_stim = None, DM_trl_ind = None,
                                    bar_ecc = None, same_ecc = None, downsample_FA_DM = None, 
                                    vmin = 0, vmax = .4, cmap = 'magma', filename = None, annot = True):
        
        """Make 4x4 plot of recontruscted stim, averaged across conditions
        (and also reverse case)
        followed by the corresponding downsampled DM for inspection
        """
        
        ## correlate each reconstructed average condition with stim position

        # note - here we transpose the DM array when correlating because the average_stim we calculated
        # has a different format than the reconstructed stim outputted by brain decoder
                
        bar_ecc_ind = {'far': 1, 'middle': 2, 'near': 3}
        
        # make reference dict with unique conditions of attended and unattend
        uniq_cond_dict = {'far': 'near', 'near': 'middle', 'middle': 'far'}

        # bar ecc list of attended and unattended bar
        bar_ecc_list = [bar_ecc, bar_ecc] if same_ecc == True else [bar_ecc, uniq_cond_dict[bar_ecc]]
        
        # in ref trial is masked
        if DM_trl_ind is None:
            dm2plot = np.zeros((8,8))
            dm2plot[bar_ecc_ind[bar_ecc_list[0]]] = 1
            dm2plot[..., bar_ecc_ind[bar_ecc_list[1]]] = 1
        else:
            dm2plot = downsample_FA_DM[DM_trl_ind]
        
        avg_corr, avg_pval = scipy.stats.pointbiserialr(average_stim.ravel(), 
                                                  dm2plot.T.ravel())
        flip_avg_corr, flip_avg_pval = scipy.stats.pointbiserialr(flip_average_stim.ravel(), 
                                                            dm2plot.T.ravel())
        
        # plot figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (10,10))

        ## attended leftmost, vertical
        sns.heatmap(average_stim, cmap = cmap, ax = axes[0][0], square = True, cbar = False,
                    annot=annot, annot_kws={"size": 7},
                    vmin = vmin, vmax = vmax)
        axes[0][0].set_title('attend %s ecc, unattend %s ecc'%(bar_ecc_list[0], bar_ecc_list[1]))

        ## DM
        axes[1][0].imshow(dm2plot.T, cmap = 'binary_r', vmax = 1.5)
        # Add the patch to the Axes
        axes[1][0].add_patch(patches.Rectangle((bar_ecc_ind[bar_ecc_list[0]] - .5, -.5), 1, 8, 
                                            linewidth = 2, edgecolor='purple', 
                                            facecolor='purple', hatch = '///'))
        # annotate correlation between reconstructed stim and DM
        axes[1][0].set_title(r"$\rho$ = {r}".format(r = '%.2f'%(avg_corr))+\
                                '   pval = {p}'.format(p = "{:.2e}".format(avg_pval)))

        ## attended rightmost
        sns.heatmap(flip_average_stim, cmap = 'magma', ax = axes[0][1], square = True, cbar = False,
                annot=annot, annot_kws={"size": 7},
                vmin = vmin, vmax = vmax)
        axes[0][1].set_title('flipped case')

        ## DM
        axes[1][1].imshow(dm2plot.T, cmap = 'binary_r', vmax = 1.5)
        # Add the patch to the Axes
        axes[1][1].add_patch(patches.Rectangle((-.5, bar_ecc_ind[bar_ecc_list[1]] - .5), 8, 1, 
                                            angle=0.0,
                                            linewidth = 2, edgecolor='green', 
                                            facecolor='green', hatch = '///'))
        # annotate correlation between reconstructed stim and DM
        axes[1][1].set_title(r"$\rho$ = {r}".format(r = '%.2f'%(flip_avg_corr))+\
                                '   pval = {p}'.format(p = "{:.2e}".format(flip_avg_pval)))

        # save figure
        if filename is not None:
            fig.savefig(filename, dpi= 200)
                 
    def get_uniq_cond_trl_ind(self, position_df = None, bar_ecc = 'far', bars_pos = 'crossed', same_ecc = False,
                                    bar_dist = 5):
    
        """Get trial indices of unique conditions
        for bar trials of specific ecc. 
        *crossed bar trials*
        -> for each attended ecc there are 2 cases:
            a) both bars at the same ecc
            b) Att and Unatt bars at different eccs [far, near] or [middle, far] or [near, middle]

        can also return flipped case (same bar pos, different attended bar)
        """
        
        if bars_pos == 'crossed':
        
            # make reference dict with unique conditions of attended and unattend
            uniq_cond_dict = {'far': 'near', 'near': 'middle', 'middle': 'far'}
            
            # bar ecc list of attended and unattended bar
            bar_ecc_list = [bar_ecc, bar_ecc] if same_ecc == True else [bar_ecc, uniq_cond_dict[bar_ecc]]
            
            # filter bar position df
            masked_df = position_df[(position_df['bars_pos'] == 'crossed') &\
                            (((position_df['bar_ecc'] == bar_ecc_list[0]) & (position_df['x_pos'] < 0) &\
                                ((position_df['attend_condition'] == True)))|\
                                ((position_df['bar_ecc'] == bar_ecc_list[1]) & (position_df['y_pos'] > 0) &\
                                ((position_df['attend_condition'] == False))))]
            
            # get trial ind
            trl_list = masked_df[masked_df.duplicated(subset=['trial_ind'])].trial_ind.values
            if len(trl_list) > 0:
                trl_ind = trl_list[0]
            else:
                trl_ind = None
            
        elif bars_pos == 'parallel':
            
            # filter bar position df
            df_ecc_dist = self.get_trl_ecc_dist_df(position_df = position_df, bars_pos = bars_pos, 
                                                bar_ecc = bar_ecc, abs_inter_bar_dist = bar_dist)
            
            # get trial ind
            trl_ind = df_ecc_dist.query('x_pos < 0').trial_ind.values[0]
            
        return trl_ind
    
    def fit_decoder(self, participant_list = [], ROI_list = ['V1'], overwrite_T1 = False, overwrite_func = False, model_type = 'gauss_hrf',
                        prf_file_ext = '_cropped_dc_psc.nii.gz', ses = 'mean', fa_file_ext = '_cropped.nii.gz',
                        mask_bool_df = None, stim_on_screen = [], group_bar_pos_df = [], prf_bar_coords_dict = None,
                        mask_barpos = False):
        
        """
        Fit decoder across participants
        and ROIs
        """

        print('Model type --> %s'%model_type)
        
        # iterate over participants
        for pp in participant_list:
            # iterate over ROIs
            for roi_name in ROI_list:
                self.decode_ROI(participant = pp, roi_name = roi_name, overwrite_T1 = overwrite_T1, overwrite_func = overwrite_func, 
                                prf_file_ext = prf_file_ext, 
                                ses = ses, mask_bool_df = mask_bool_df, stim_on_screen = stim_on_screen,
                                model_type = model_type, fa_file_ext = fa_file_ext, 
                                pp_bar_pos_df = group_bar_pos_df['sub-{sj}'.format(sj = pp)],
                                prf_bar_coords_dict = prf_bar_coords_dict, mask_barpos = mask_barpos)
                       
    def decode_ROI(self, participant = None, roi_name = 'V1', overwrite_T1 = False, overwrite_func = False, model_type = 'gauss_hrf',
                        prf_file_ext = '_cropped_dc_psc.nii.gz', fa_file_ext = '_cropped.nii.gz', ses = 'mean',
                        mask_bool_df = None, stim_on_screen = [], save_estimates = True, pp_bar_pos_df = None,
                        prf_bar_coords_dict = None, mask_barpos = False):
        
        """For a given participant and ROI,
        run decoder analysis
        """
        
        # make dir to save estimates
        pp_outdir = op.join(self.decoder_dir, 'sub-{sj}'.format(sj = participant))
        os.makedirs(pp_outdir, exist_ok = True)
        print('saving files in %s'%pp_outdir)
        
        # save parameters as HDF5 file, to later load
        pars_filename = op.join(pp_outdir, 
                            'sub-{sj}_task-pRF_ROI-{rname}_model-{dmod}_pars.h5'.format(sj = participant,
                                                                                        rname = roi_name,
                                                                                        dmod = model_type))
                        
        # get masked prf ROI data, averaged across runs
        prf_masked_data_df = self.get_prf_ROI_data(participant = participant, 
                                                    roi_name = roi_name, 
                                                    index_arr = [], 
                                                    overwrite_T1 = overwrite_T1, 
                                                    overwrite_func = overwrite_func,
                                                    file_ext = prf_file_ext)
    
        # get prf stimulus DM and grid coordinates
        prf_stimulus_dm, prf_grid_coordinates = self.get_prf_stim_grid(participant = participant, 
                                                                        ses = ses, 
                                                                        mask_bool_df = mask_bool_df, 
                                                                        stim_on_screen = stim_on_screen,
                                                                        prf_condition_per_TR = self.prf_condition_per_TR)
        
        ## fit prf data with decoder model
        prf_decoder_model, pars_gd = self.fit_encoding_model(model_type = model_type, 
                                                            data = prf_masked_data_df,
                                                            grid_coordinates = prf_grid_coordinates, 
                                                            paradigm = prf_stimulus_dm,
                                                            filename = pars_filename)
            
        ## need to select best voxels 
        # get rsq
        prf_decoder_fitter = ParameterFitter(model = prf_decoder_model, 
                                    data = prf_masked_data_df, 
                                    paradigm = prf_stimulus_dm)
        r2_gd = prf_decoder_fitter.get_rsq(pars_gd)
        
        # get array with best voxel indices 
        best_voxels = self.get_best_voxels(pars_gd = pars_gd, r2_gd = r2_gd,  
                                           x_lim = [-6,6], y_lim = [-6,6], 
                                           size_min = .3, size_max = 'std', std_val = 3, 
                                           n_vox = 300)
        
        ## fit residuals on prf data
        omega, dof  = self.fit_residuals(model = prf_decoder_model, 
                                        data = prf_masked_data_df, 
                                        paradigm = prf_stimulus_dm, 
                                        parameters = pars_gd, 
                                        fit_method = 't', 
                                        best_vox = best_voxels,
                                        max_n_iterations = 10000, 
                                        learning_rate = 0.02, 
                                        min_n_iterations = 100,
                                        filename = pars_filename.replace('_pars.h5', '_resid.npz'))
        
        ## now get masked FA ROI data, all runs
        masked_FAdata_dict = self.get_FA_ROI_data(participant = participant, 
                                                roi_name = roi_name, 
                                                index_arr = [], 
                                                overwrite_T1 = overwrite_T1, 
                                                overwrite_func = overwrite_func, 
                                                file_ext = fa_file_ext,
                                                glmsingle_model = 'D', 
                                                trial_num = 132)
        ## get FA DM and grid coordinates (8x8)
        FA_DM_dict, fa_grid_coordinates = self.get_FA_stim_grid(participant = participant, 
                                                                group_bar_pos_df = {'sub-{sj}'.format(sj = participant): pp_bar_pos_df},
                                                                prf_bar_coords_dict = prf_bar_coords_dict,
                                                                dm_size = prf_stimulus_dm.shape[-1])
        
        ## decode over runs
        # save reconstructed stim as HDF5 file, to later load
        decoded_stim_filename = op.join(pp_outdir, 
                            'sub-{sj}_task-FA_ROI-{rname}_model-{dmod}_reconstructed_stim.h5'.format(sj = participant,
                                                                                                    rname = roi_name,
                                                                                                    dmod = model_type))
        # make filename generic to save per run
        decoded_stim_filename = decoded_stim_filename.replace('_task', '_{snrnkey}_task') 
        
        # rename is masking bar position
        if mask_barpos:
            print('decoding data with constrained (masked) bar positions')
            decoded_stim_filename = decoded_stim_filename.replace('_stim', '_stim_masked')
        
        reconstructed_stim_dict = {}
        lowres_DM_dict = {}
        for ind, df_key in enumerate(masked_FAdata_dict):
            
            ## downsample DM 
            file_rn, file_sn = self.MRIObj.mri_utils.get_run_ses_from_str(df_key)
            print('downsampling FA DM for ses-{sn}, run-{rn}'.format(sn = file_sn, rn=file_rn))
            
            lowres_DM = self.downsample_DM(DM_arr = FA_DM_dict[df_key]['full_stim'],
                                        dm_size = FA_DM_dict[df_key]['full_stim'].shape[-1],
                                        new_size = 8)
            
            ## and set downsampled DM as mask, if such is the case
            mask_fa = lowres_DM.astype(bool) if mask_barpos else None
                
            print('decoding data from %s'%df_key)
            
            # get run ROI dataframe
            masked_data_df = masked_FAdata_dict[df_key]

            # get reconstructed stim as df
            reconstructed_stimulus = self.decode_FA_stim(model_type = model_type,
                                                        data = masked_data_df, 
                                                        grid_coordinates = fa_grid_coordinates,  
                                                        parameters = pars_gd,
                                                        omega = omega, 
                                                        dof = dof,
                                                        mask = mask_fa,
                                                        best_voxels = best_voxels, 
                                                        filename = decoded_stim_filename.format(snrnkey = df_key))
            # append to dict
            reconstructed_stim_dict[df_key] = reconstructed_stimulus
            
            # correlate reconstructed stim with downsampled DM
            corr, pval = scipy.stats.pointbiserialr(reconstructed_stimulus.values.ravel(), 
                                            lowres_DM.ravel())
            print('correlation between reconstructed stim and DM is %.2f, %.2f'%(corr, pval))
            
            # append to dict
            lowres_DM_dict[df_key] = lowres_DM
        
        return lowres_DM_dict, reconstructed_stim_dict
               
    def decode_FA_stim(self, model_type = 'gauss_hrf', data = None, grid_coordinates = None,  parameters = None, 
                            omega = None, dof = None, best_voxels = None, filename = None, mask = None):
        
        """Decode FA betas for a given run
        and save reconstructed stim as hdf5 file
        """
        
        print('using params from %i best voxels'%(len(best_voxels)))

        # if there is a pars file already, just load it
        if filename is not None and op.isfile(filename):
            print('Loading reconstructed stim from %s'%filename)
            reconstructed_stimulus = pd.read_hdf(filename).astype(np.float32)  
        else:
            if model_type == 'gauss_hrf':
                model_single_trial = GaussianPRF2D(grid_coordinates = grid_coordinates,
                                                paradigm = None,
                                                data = data.loc[:, best_voxels],
                                                parameters = parameters.loc[best_voxels],
                                                weights = None,
                                                omega = omega,
                                                dof = dof)
            elif model_type == 'dog_hrf':
                model_single_trial = DifferenceOfGaussiansPRF2D(grid_coordinates = grid_coordinates,
                                                                paradigm = None,
                                                                data = data.loc[:, best_voxels],
                                                                parameters = parameters.loc[best_voxels],
                                                                weights = None,
                                                                omega = omega,
                                                                dof = dof)
            
            stim_fitter = StimulusFitter(data = data.loc[:, best_voxels], 
                                    model = model_single_trial, 
                                    omega = omega,
                                    dof = dof)

            # Legacy Adam is a bit faster than the default Adam optimizer on M1
            # Learning rate of 1.0 is a bit high, but works well here
            reconstructed_stimulus = stim_fitter.fit(legacy_adam=False, 
                                                    min_n_iterations=1000, 
                                                    max_n_iterations=5000, 
                                                    learning_rate = .01,
                                                    l2_norm = .01,
                                                    mask = mask)
            
            if filename is not None:
                print('Saving reconstructed stim in %s'%filename)
                reconstructed_stimulus.to_hdf(filename, key='df_stim', mode='w', index = False)  
        
        return reconstructed_stimulus
         
    def get_FA_stim_grid(self, participant = None, group_bar_pos_df = None, prf_bar_coords_dict = None, dm_size = 80, 
                            y_coords_deg = None, x_coords_deg = None):
        
        """Get participant FA DM + grid coordinates that will be used in decoder
        """
        
        # if we didnt provide a prf bar position mask, set to None
        if prf_bar_coords_dict is None:
            prf_bar_coords_dict = {}
            prf_bar_coords_dict['sub-{sj}'.format(sj = participant)] = None
        
        ## get FA DM
        FA_DM_dict, _ = self.make_DM_dict(pp_bar_pos_df = group_bar_pos_df['sub-{sj}'.format(sj = participant)],
                                        pp_prf_bar_coords_dict = prf_bar_coords_dict['sub-{sj}'.format(sj = participant)],
                                        dm_size = dm_size)
        
        ## get grid coordinates (8x8)
        fa_grid_coordinates = self.get_decoder_grid_coords(y_coords_deg = y_coords_deg, x_coords_deg = x_coords_deg)
        
        return FA_DM_dict, fa_grid_coordinates
        
    def get_FA_ROI_data(self, participant = None, roi_name = 'V1', index_arr = [], overwrite_T1 = False, overwrite_func = False,
                            glmsingle_model = 'D', file_ext = '_cropped.nii.gz', trial_num = 132,
                            return_data = True):
        
        """Get FA data (beta values) for the ROI of a participant, for all runs,
        and return dataframe in a format compatible with braindecoder 
        """
        
        ## convert betas estimates into volume images (if not done so already)
        # and get filenames
        betas_filelist = self.convert_betas_volume(participant, model_type = glmsingle_model, 
                                                    file_ext = file_ext, trial_num = trial_num)

        # if we want the loaded data file
        if return_data:
            ## get masked ROI data file list
            masked_FAdata_df_filelist = self.get_ROImask_data(participant, 
                                                        file_list = betas_filelist, 
                                                        task = 'FA', 
                                                        run_type = 'all', ses = 'all', 
                                                        roi_name = roi_name, 
                                                        index_arr = index_arr,
                                                        overwrite_T1 = overwrite_T1,
                                                        overwrite_func = overwrite_func)
        
            ## return as dict of dataframes
            output_dict = {}
            for filename in masked_FAdata_df_filelist:
                
                # get file run and ses number
                file_rn, file_sn = self.MRIObj.mri_utils.get_run_ses_from_str(filename)
                
                # load df
                masked_df = pd.read_csv(filename, sep='\t', index_col=['time'], 
                                                compression='gzip').astype(np.float32)
                # and format it accordingly (should clean up in the future - simplest is saving data as hdf5 files)
                out_df = pd.DataFrame(masked_df.to_numpy(), 
                                    index=pd.Index(np.arange(len(masked_df)), 
                                                name='time'),
                                    columns = pd.Index(range(masked_df.to_numpy().shape[1]), 
                                                    name='source')).astype(np.float32)
                
                output_dict['ses-{sn}_run-{rn}'.format(sn = file_sn, rn = file_rn)] = out_df
                
            return output_dict
        
        else:
            ## get masked ROI data file list
            masked_FAdata_df_filelist = self.get_ROImask_filenames(participant, 
                                                                    file_list = betas_filelist, 
                                                                    task = 'FA', 
                                                                    run_type = 'all', ses = 'all', 
                                                                    roi_name = roi_name, 
                                                                    index_arr = index_arr)
            
            # return file name list
            return masked_FAdata_df_filelist
          
    def fit_residuals(self, model = None, data = None, paradigm = None, parameters = None, fit_method = 't', 
                            best_vox = None, filename = None, max_n_iterations = 1000, learning_rate = 0.02, min_n_iterations = 100):
        
        """Fit noise model on residuals
        """
        # if there is a pars file already, just load it
        if filename is not None and op.isfile(filename):
            print('Loading omega and dof from %s'%filename)
            residuals_npz = np.load(filename)
            omega = residuals_npz['omega']
            dof = residuals_npz['dof'][0]
        else:
            if best_vox is None:
                train_data =  data
                train_pars = parameters.astype(np.float32)
            else:
                train_data =  data.loc[:, best_vox]
                train_pars = parameters.loc[best_vox].astype(np.float32)
                
            paradigm = paradigm.astype(np.float32)
            
            resid_fitter = ResidualFitter(model = model,
                                        data = train_data, 
                                        paradigm = paradigm, 
                                        parameters = train_pars)
            omega, dof = resid_fitter.fit(method=fit_method, 
                                          max_n_iterations=max_n_iterations, 
                                          learning_rate=learning_rate, 
                                          min_n_iterations=min_n_iterations)
            
            if filename is not None:
                print('Storing omega, dof and best voxel indices in %s'%filename)
                np.savez(filename, omega=omega, dof=[dof], best_voxels = best_vox)
        
        return omega, dof 
                
    def get_best_voxels(self, pars_gd = None, r2_gd = None, x_lim = [-6,6], y_lim = [-6,6], 
                                size_min = .3, size_max = 'std', std_val = 3, n_vox = 300):
        
        """
        Get best voxels to then use in fitter
        """   
        
        # sort indices according to r2 values
        sort_ind = r2_gd.sort_values(ascending=False).index
        
        # mask for pRFs
        # for x and y lim (to be within screen dimensions)
        masked_pars = pars_gd.query('x >= {xmin} & x <= {xmax} & y >= {ymin} & y <= {ymax}'.format(xmin = x_lim[0], xmax = x_lim[1],
                                                                                                ymin = y_lim[0], ymax = y_lim[1]))
        # for size (not too small or big)
        if isinstance(size_max, str):
            # use X standard deviation from mean as size max threshold
            masked_pars = masked_pars.query('sd >= {sdmin} & sd <= {sdmax}'.format(sdmin = size_min,
                                                                                sdmax = np.std(masked_pars.sd.values) * std_val))
        else:
            masked_pars = masked_pars.query('sd >= {sdmin} & sd <= {sdmax}'.format(sdmin = size_min, sdmax = size_max))
            
        # get index for masked parameters
        masked_voxels = masked_pars.index
        
        # of those, get best voxels
        best_voxels = np.array([val for val in sort_ind if val in masked_voxels])
        
        # select a subset
        if isinstance(n_vox, int):
            best_voxels = best_voxels[:n_vox]
        
        return best_voxels
                   
    def fit_encoding_model(self, data = None, grid_coordinates = None, model_type = 'gauss_hrf',
                                paradigm = None, filename = None):
        
        """
        Fit PRF parameters (encoding model)
        """
        
        # if we want to fit the hrf
        fit_hrf = True if 'hrf' in model_type else False
                
        # if there is a pars file already, just load it
        if filename is not None and op.isfile(filename):
            print('Loading pRF fit parameters stored in %s'%filename)
            output_pars = pd.read_hdf(filename).astype(np.float32)  
            
            # reload prf model accordingly            
            prf_decoder_model = self.setup_prf_model(data = data, grid_coordinates = grid_coordinates, 
                                                    model_type = model_type,
                                                    paradigm = paradigm, 
                                                    fit_hrf = fit_hrf) 
        else:
            ## set gauss prf model 
            # first without hrf
            prf_decoder_model = self.setup_prf_model(data = data, grid_coordinates = grid_coordinates, 
                                                    model_type = 'gauss_hrf',
                                                    paradigm = paradigm, 
                                                    fit_hrf = False)  
        
            # We set up a parameter fitter
            par_fitter = ParameterFitter(model = prf_decoder_model, 
                                        data = data, 
                                        paradigm = paradigm)

            # We set up a grid of parameters to search over
            x = np.linspace(-6, 6, 20) 
            y = np.linspace(-6, 6, 20) 
            sd = np.linspace(.25, 5, 20) 

            # For now, we only use one amplitude and baseline, because we
            # use a correlation cost function, which is indifferent to
            # the overall scaling of the model
            # We can easily estimate these later using OLS
            amplitudes = [1.0]
            baseline = [0.0]

            # Note that the grids should be given in the correct order (can be found back in
            # model.parameter_labels)
            grid_pars = par_fitter.fit_grid(x, y, sd, baseline, amplitudes, use_correlation_cost=True)

            # Once we have the best parameters from the grid, we can optimize the baseline
            # and amplitude
            refined_grid_pars = par_fitter.refine_baseline_and_amplitude(grid_pars)

            # Now we use gradient descent to further optimize the parameters
            pars_gd = par_fitter.fit(init_pars=refined_grid_pars, learning_rate=1e-2, max_n_iterations=5000,
                                            min_n_iterations=100,
                                            r2_atol=0.0001)
            
            ## now fit the hrf, if such is the case
            if 'hrf' in model_type:
                # redefine model
                prf_decoder_model = self.setup_prf_model(data = data, grid_coordinates = grid_coordinates, 
                                                model_type = 'gauss_hrf',
                                                paradigm = paradigm, 
                                                fit_hrf = True)  
                # and fitter
                par_fitter = ParameterFitter(model = prf_decoder_model, 
                                            data = data, 
                                            paradigm = paradigm)

                # We set hrf_delay and hrf_dispersion to standard values
                pars_gd['hrf_delay'] = 6
                pars_gd['hrf_dispersion'] = 1

                output_pars = par_fitter.fit(init_pars=pars_gd, learning_rate=1e-2, max_n_iterations=5000,
                                                min_n_iterations=100,
                                                r2_atol=0.0001) 
            else:
                output_pars = pars_gd
                
            # save gauss estimates
            if filename is not None:
                print('Saving Gauss pRF fit parameters in %s'%filename.replace(model_type, 'gauss_hrf'))
                output_pars.to_hdf(filename.replace(model_type, 'gauss_hrf'), 
                                   key='df_pars', mode='w', index = False) 
                
            ## now run other models, 
            # if such is the case
            if model_type == 'dog_hrf':
                
                prf_decoder_model = self.setup_prf_model(data = data, grid_coordinates = grid_coordinates, 
                                                        model_type = model_type,
                                                        paradigm = paradigm, 
                                                        fit_hrf = fit_hrf) 
                
                # initialize DoG model with gauss fit pars
                pars_dog_init = output_pars.copy()
                
                # This is the relative amplitude of the inhibitory receptive field
                # compared to the excitatory one.
                pars_dog_init['srf_amplitude'] = 0.1

                # This is the relative size of the inhibitory receptive field
                # compared to the excitatory one.
                pars_dog_init['srf_size'] = 2.

                # Let's set up a new parameterfitter
                par_fitter_dog = ParameterFitter(model = prf_decoder_model, 
                                                 data = data, 
                                                 paradigm = paradigm)

                # Note how, for now, we are not optimizing the HRF parameters.
                pars_dog = par_fitter_dog.fit(init_pars = pars_dog_init, 
                                            learning_rate = 1e-2, max_n_iterations = 5000,
                                            min_n_iterations = 100,
                                            r2_atol=0.0001,
                                            fixed_pars = ['hrf_delay', 'hrf_dispersion'])
                
                # Now we optimize _with_ the HRF parameters
                if fit_hrf:
                    output_pars = par_fitter_dog.fit(init_pars = pars_dog, learning_rate=1e-2, max_n_iterations=5000,
                                                    min_n_iterations=100,
                                                    r2_atol=0.0001)
                else:
                    output_pars = pars_dog

                # save estimates
                if filename is not None:
                    print('Saving pRF fit parameters in %s'%filename)
                    output_pars.to_hdf(filename, key='df_pars', mode='w', index = False)  
        
        return prf_decoder_model, output_pars
    
    def load_ROI_encoding_model_pars(self, participant = None, task = 'pRF', roi_name = 'V1', model_type = 'gauss_hrf'):
        
        """Load previously fitted parameters
        """
        
        # dir where estimates where saved 
        pp_outdir = op.join(self.decoder_dir, 'sub-{sj}'.format(sj = participant))

        # save parameters as HDF5 file, to later load
        pars_filename = op.join(pp_outdir, 
                            'sub-{sj}_task-{tsk}_ROI-{rname}_model-{dmod}_pars.h5'.format(sj = participant,
                                                                                        tsk = task,
                                                                                        rname = roi_name,
                                                                                        dmod = model_type))

        # if there is no pars file
        if not op.isfile(pars_filename):
           print('Could not find %s'%pars_filename)
           pars_gd = None
        else:
            pars_gd = pd.read_hdf(pars_filename).astype(np.float32)  
        
        return pars_gd
    
    def load_reconstructed_stim_dict(self, participant = None, task = 'FA', roi_name = 'V1', model_type = 'gauss_hrf',
                                    data_keys = ['ses-1_run-1'], masked_stim = False):
        
        """Load previously save reconstructed stim
        """
        
        # dir where estimates where saved 
        pp_outdir = op.join(self.decoder_dir, 'sub-{sj}'.format(sj = participant))
        
        # get reconstructed stim file name
        decoded_stim_filename = op.join(pp_outdir, 
                            'sub-{sj}_task-{tsk}_ROI-{rname}_model-{dmod}_reconstructed_stim.h5'.format(sj = participant,
                                                                                                    tsk = task,
                                                                                                    rname = roi_name,
                                                                                                    dmod = model_type))
        
        # if we want to load masked reconstructed stim
        if masked_stim:
            decoded_stim_filename = decoded_stim_filename.replace('_stim.h5', '_stim_masked.h5')
            
        # make filename generic to load per run
        decoded_stim_filename = decoded_stim_filename.replace('_task', '_{snrnkey}_task') 
        
        ## iterate over runs
        reconstructed_stim_dict = {}

        for df_key in data_keys:
            
            # load stim and append to dict
            reconstructed_stim_dict[df_key] = pd.read_hdf(decoded_stim_filename.format(snrnkey = df_key))
            
        return reconstructed_stim_dict
    
    def get_lowresDM_dict(self, DM_dict = None, data_keys = ['ses-1_run-1']):
        
        """downsample FA DM and append to dicts, for later plotting
        """
    
        ## iterate over runs
        lowres_DM_dict = {key: {} for key in list(DM_dict[data_keys[0]].keys())}

        for ind, df_key in enumerate(data_keys):
            
            for stim_key in lowres_DM_dict.keys():
                # downsample DM and append to dicts
                lowres_DM_dict[stim_key][df_key] = self.downsample_DM(DM_arr = DM_dict[df_key][stim_key])
                    
        return lowres_DM_dict
    
    def get_same_bar_pos_ind_dict(self, lowresDM_dict = None, data_keys = ['ses-1_run-1']):
        
        """find trials where attended bar and unattended bar in same position
        """
        
        same_bar_pos_ind_dict = {}
        
        for ind, df_key in enumerate(data_keys):
            
            same_bar_pos_ind_dict[df_key] = self.get_run_trial_pairs(DM_arr = lowresDM_dict['full_stim'][df_key])
            
        return same_bar_pos_ind_dict
    
    def get_group_average_stim_dict(self, participant_list = [], reconstructed_stim_dict = None, run_position_df_dict = None, 
                                            lowres_DM_dict = None, bar_type = 'parallel'):
        
        """Get average stim dict 
        averaged across group of participants
        """
        
        ## get average stim (across unique bar positions) for all participants 
        average_stim_dict, flip_average_stim_dict = self.get_average_stim_dict(participant_list = participant_list,
                                                                            reconstructed_stim_dict = reconstructed_stim_dict, 
                                                                            run_position_df_dict = run_position_df_dict, 
                                                                            lowres_DM_dict = lowres_DM_dict, 
                                                                            bar_type = bar_type)

        group_average_stim_dict = {}
        group_flip_average_stim_dict = {}

        # average across participants
        for keynames1 in average_stim_dict.keys():
            
            group_average_stim_dict[keynames1] = {}
            group_flip_average_stim_dict[keynames1] = {}
            
            for keynames2 in average_stim_dict[keynames1].keys():
                
                group_average_stim_dict[keynames1][keynames2] = np.nanmean(average_stim_dict[keynames1][keynames2], axis = 0)
                group_flip_average_stim_dict[keynames1][keynames2] = np.nanmean(flip_average_stim_dict[keynames1][keynames2], axis = 0)

        return group_average_stim_dict, group_flip_average_stim_dict
    
    def get_average_stim_dict(self, participant_list = [], reconstructed_stim_dict = None, run_position_df_dict = None, 
                                    lowres_DM_dict = None, bar_type = 'parallel'):
        
        """For all participants in list
        create dict with average reconstructed stim
        collapsed across conditions (where that's possible)
        """
        
        # save results in dict
        average_stim_dict = {}
        flip_average_stim_dict = {}
        
        if bar_type == 'parallel':

            bar_dist_dict = {'far': np.arange(5)+1, 'middle': np.arange(3)+1, 'near': np.arange(1)+1}
            
            # iterate over bar distances and ecc
            for bar_ecc, bar_dist_list in bar_dist_dict.items():
                
                average_stim_dict[bar_ecc] = {}
                flip_average_stim_dict[bar_ecc] = {}
            
                for bar_dist in bar_dist_list:
                    
                    average_stim = []
                    flip_average_stim = []
                    
                    print('Averaging data for parallel bars, for attended bar ecc %s, inter-bar distance %s'%(bar_ecc,str(bar_dist)))
                    
                    # iterate over participants
                    for participant in tqdm(participant_list):
                    
                        ## average over runs
                        pp_average_stim = []
                        pp_flip_average_stim = []
                        
                        ## stack all runs
                        for snrn_key, snrn_stim in reconstructed_stim_dict['sub-{sj}'.format(sj = participant)].items():
                            
                            pp_average_stim.append(self.get_parallel_average_stim(reconstructed_stimulus = snrn_stim, 
                                                                                position_df = run_position_df_dict['sub-{sj}'.format(sj = participant)][snrn_key], 
                                                                                bar_ecc = bar_ecc, 
                                                                                abs_inter_bar_dist = bar_dist, 
                                                                                flipped_stim = False, 
                                                                                DM_arr = lowres_DM_dict['sub-{sj}'.format(sj = participant)]['full_stim'][snrn_key]))

                            # also get average flipped case
                            pp_flip_average_stim.append(self.get_parallel_average_stim(reconstructed_stimulus = snrn_stim, 
                                                                                position_df = run_position_df_dict['sub-{sj}'.format(sj = participant)][snrn_key], 
                                                                                bar_ecc = bar_ecc, 
                                                                                abs_inter_bar_dist = bar_dist, 
                                                                                flipped_stim = True, 
                                                                                DM_arr = lowres_DM_dict['sub-{sj}'.format(sj = participant)]['full_stim'][snrn_key]))
                        pp_average_stim = np.stack(pp_average_stim)
                        pp_flip_average_stim = np.stack(pp_flip_average_stim)
                        
                        ## average over participant runs 
                        # and append
                        average_stim.append(np.nanmean(pp_average_stim, axis = 0)) 
                        flip_average_stim.append(np.nanmean(pp_flip_average_stim, axis = 0))
                        
                    # save in dict
                    average_stim_dict[bar_ecc][bar_dist] = np.stack(average_stim)
                    flip_average_stim_dict[bar_ecc][bar_dist] = np.stack(flip_average_stim)
                    
        elif bar_type == 'crossed':
            
            uniq_cond_dict = {'far': 'near', 'middle': 'far', 'near': 'middle'}
            
            # iterate over ecc
            for bar_ecc in uniq_cond_dict.keys():
                
                average_stim_dict[bar_ecc] = {}
                flip_average_stim_dict[bar_ecc] = {}
                
                # and if crossed bars where equidistant to center or not
                for same_ecc in [True, False]:
                    
                    average_stim = []
                    flip_average_stim = []
                    
                    print('Averaging data for crossed bars, for attended bar ecc %s, bars at same distance = %s'%(bar_ecc, str(same_ecc)))
                    
                    # iterate over participants
                    for participant in tqdm(participant_list):
                    
                        ## average over runs
                        pp_average_stim = []
                        pp_flip_average_stim = []
                    
                        ## stack all runs
                        for snrn_key, snrn_stim in reconstructed_stim_dict['sub-{sj}'.format(sj = participant)].items():
                            
                            pp_average_stim.append(self.get_crossed_average_stim(reconstructed_stimulus = snrn_stim, 
                                                                                position_df = run_position_df_dict['sub-{sj}'.format(sj = participant)][snrn_key], 
                                                                                bar_ecc = bar_ecc, 
                                                                                same_ecc = same_ecc,  
                                                                                flipped_stim = False, 
                                                                                DM_arr = lowres_DM_dict['sub-{sj}'.format(sj = participant)]['full_stim'][snrn_key]))

                            # also get average flipped case
                            pp_flip_average_stim.append(self.get_crossed_average_stim(reconstructed_stimulus = snrn_stim, 
                                                                                position_df = run_position_df_dict['sub-{sj}'.format(sj = participant)][snrn_key],
                                                                                bar_ecc = bar_ecc, 
                                                                                same_ecc = same_ecc, 
                                                                                flipped_stim = True, 
                                                                                DM_arr = lowres_DM_dict['sub-{sj}'.format(sj = participant)]['full_stim'][snrn_key]))

                        pp_average_stim = np.stack(pp_average_stim)
                        pp_flip_average_stim = np.stack(pp_flip_average_stim)
                        
                        ## average over participant runs 
                        # and append
                        average_stim.append(np.nanmean(pp_average_stim, axis = 0)) 
                        flip_average_stim.append(np.nanmean(pp_flip_average_stim, axis = 0))
                        
                    # save in dict
                    average_stim_dict[bar_ecc][int(same_ecc)] = np.stack(average_stim)
                    flip_average_stim_dict[bar_ecc][int(same_ecc)] = np.stack(flip_average_stim)
                    
        return average_stim_dict, flip_average_stim_dict
        
    def get_encoding_fitter(self, data = None, grid_coordinates = None, model_type = 'gauss_hrf',
                                paradigm = None):
        
        """Get encoding model fitter object
        """
        
        if 'hrf' in model_type:
            fit_hrf = True
            
        ## get prf model 
        prf_decoder_model = self.setup_prf_model(data = data, 
                                                grid_coordinates = grid_coordinates, 
                                                model_type = model_type,
                                                paradigm = paradigm, 
                                                fit_hrf = fit_hrf)  
            
        # set fitter
        prf_decoder_fitter = ParameterFitter(model = prf_decoder_model, 
                                            data = data, 
                                            paradigm = paradigm)
        
        return prf_decoder_fitter
        
    def get_encoding_r2(self, data = None, grid_coordinates = None, model_type = 'gauss_hrf',
                                paradigm = None, pars_gd = None):
        
        """Get encoding model rsq
        """
        
        # set fitter
        prf_decoder_fitter = self.get_encoding_fitter(data = data, 
                                                    grid_coordinates = grid_coordinates, 
                                                    model_type = model_type,
                                                    paradigm = paradigm)
        
        r2_gd = prf_decoder_fitter.get_rsq(pars_gd)

        return r2_gd
         
    def setup_prf_model(self, data = None, grid_coordinates = None, model_type = 'gauss_hrf',
                            paradigm = None, fit_hrf = False):
        
        """
        set up appropriate prf decoder model
        """
        
        # set hrf model
        hrf_model = SPMHRFModel(tr = self.MRIObj.TR, onset = -self.MRIObj.TR/2)
        
        ## set prf model 
        if model_type == 'gauss_hrf':
            # (gauss with HRF)
            prf_decoder_model = GaussianPRF2DWithHRF(data = data,
                                                    grid_coordinates = grid_coordinates, 
                                                    paradigm = paradigm,
                                                    hrf_model = hrf_model,
                                                    flexible_hrf_parameters = fit_hrf)
            
        elif model_type == 'dog_hrf':
            # (DoG with HRF)
            prf_decoder_model = DifferenceOfGaussiansPRF2DWithHRF(data = data,
                                                                grid_coordinates = grid_coordinates, 
                                                                paradigm = paradigm,
                                                                hrf_model = hrf_model,
                                                                flexible_hrf_parameters = fit_hrf)
        
        return prf_decoder_model
    
    def load_group_DM_dict(self, participant_list = [], group_bar_pos_df = None, data_keys_dict = {}, prf_bar_coords_dict = None):
        
        """Load FA downsampled DM for all participants in participant list
        returns dict of DMs 
        """
        
        group_lowres_DM_dict = {}

        for participant in participant_list:
            
            print('Getting downsampled DM for sub-{sj}'.format(sj = participant)) 
            
            ## get FA DM and grid coordinates (8x8)
            FA_DM_dict, _ = self.get_FA_stim_grid(participant = participant, 
                                                  group_bar_pos_df = group_bar_pos_df,
                                                  prf_bar_coords_dict = prf_bar_coords_dict)
            
            ## get downsampled FA DM
            group_lowres_DM_dict['sub-{sj}'.format(sj = participant)] = self.get_lowresDM_dict(DM_dict = FA_DM_dict, 
                                                                                            data_keys = data_keys_dict['sub-{sj}'.format(sj = participant)])
            
        return group_lowres_DM_dict
    
    def load_group_decoded_stim_dict(self, participant_list = [], roi_name = 'V1', task = 'FA',
                                model_type = 'gauss_hrf', data_keys_dict = {}, masked_stim = False):
        
        """Load FA downsampled DM for all participants in participant list
        returns dict of DMs 
        """
        
        group_reconstructed_stim_dict = {}

        for participant in participant_list:
            
            ## load reconstructed stim
            reconstructed_stim_dict = self.load_reconstructed_stim_dict(participant = participant, 
                                                                        task = task, 
                                                                        roi_name = roi_name, 
                                                                        model_type = model_type,
                                                                        data_keys = data_keys_dict['sub-{sj}'.format(sj = participant)],
                                                                        masked_stim = masked_stim)
            
            group_reconstructed_stim_dict['sub-{sj}'.format(sj = participant)] = reconstructed_stim_dict
            
        return group_reconstructed_stim_dict
               
                             
    def animate_parallel_stim(self, participant_list = [], average_stim_dict = None, flip_average_stim_dict = None, 
                                    run_position_df_dict = None, lowres_DM_dict = None,
                                    bar_ecc = 'far',  vmin = 0, vmax = .4, cmap = 'plasma', annot = False, 
                                    roi_name = 'V1', interval=600, filename = None):
        
        """Create animation with average reconstructed stim for parallel bar positions
        for attended bar at specific eccentricity
        """
        
        ## turn dict into average array, to plot in movie
        parallel_avg_arr = [np.nanmean(items1, axis = 0) for keynames1, items1 in average_stim_dict['parallel'][bar_ecc].items()]
        parallel_avg_arr = np.stack(parallel_avg_arr)

        parallel_flip_avg_arr = [np.nanmean(items1, axis = 0) for keynames1, items1 in flip_average_stim_dict['parallel'][bar_ecc].items()]
        parallel_flip_avg_arr = np.stack(parallel_flip_avg_arr)
        
        ## get frames of DM with corresponding bar position
        
        # get example DM for participant in list
        pp = participant_list[0]
        df_key = list(run_position_df_dict['sub-{sj}'.format(sj = pp)].keys())[0]
        
        downsample_FA_DM_list = []
        
        for i in average_stim_dict['parallel'][bar_ecc].keys():

            DM_trl_ind = self.get_uniq_cond_trl_ind(position_df = run_position_df_dict['sub-{sj}'.format(sj = pp)][df_key], 
                                                        bar_ecc = bar_ecc, 
                                                        bars_pos = 'parallel', 
                                                        bar_dist = i)
            downsample_FA_DM_list.append(lowres_DM_dict['sub-{sj}'.format(sj = pp)]['full_stim'][df_key][DM_trl_ind])
                
        ## initialize base figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (10,10))

        fig.suptitle('Reconstructed stim (parallel bars), ROI - %s'%roi_name, fontsize=16)
    
        ## set function to update frames
        def update_parallel_stim(frame, stim_arr = [], flip_stim_arr = [], 
                                bar_ecc = 'far', dm_list = [],
                                vmin = 0, vmax = .4, cmap = 'plasma', annot = False):
            
            bar_ecc_ind = {'far': 1, 'middle': 2, 'near': 3}
            bar_dist = 1+frame
            
            # clear axis of fig
            axes[0][0].clear() 
            axes[1][0].clear() 
            axes[0][1].clear() 
            axes[1][1].clear() 
            
            ## attended leftmost
            sns.heatmap(stim_arr[frame], cmap = cmap, ax = axes[0][0], 
                        square = True, cbar = False,
                        annot=annot, annot_kws={"size": 7},
                        vmin = vmin, vmax = vmax)
            axes[0][0].set_title('Inter-bar distance = %i'%(bar_dist))
            
            ## reversed case attend rightmost
            sns.heatmap(flip_stim_arr[frame], cmap = cmap, ax = axes[1][0], 
                        square = True, cbar = False,
                        annot=annot, annot_kws={"size": 7},
                        vmin = vmin, vmax = vmax)


            ## DMs
            # attend left
            axes[0][1].imshow(dm_list[frame].T, cmap = 'binary_r', vmax = 1.5)
            # Add the patch to the Axes
            axes[0][1].add_patch(patches.Rectangle((bar_ecc_ind[bar_ecc] - .5, -.5), 1, 8, 
                                                linewidth = 2, edgecolor='purple', 
                                                facecolor='purple', hatch = '///'))
            
            # annotate correlation value between stim and DM
            corr, pval = scipy.stats.pointbiserialr(stim_arr[frame].ravel(), 
                                            dm_list[frame].T.ravel())
            axes[0][1].set_title(r"$\rho$ = {r}".format(r = '%.2f'%(corr))+\
                                '   pval = {p}'.format(p = "{:.2e}".format(pval)))

            # attend right
            axes[1][1].imshow(dm_list[frame].T, cmap = 'binary_r', vmax = 1.5)
            # Add the patch to the Axes
            axes[1][1].add_patch(patches.Rectangle(((bar_ecc_ind[bar_ecc] - .5 + bar_dist), -.5), 1, 8, 
                                                linewidth = 2, edgecolor='green', 
                                                facecolor='green', hatch = '///'))
            
            # annotate correlation value between stim and DM
            corr, pval = scipy.stats.pointbiserialr(flip_stim_arr[frame].ravel(), 
                                            dm_list[frame].T.ravel())
            axes[1][1].set_title(r"$\rho$ = {r}".format(r = '%.2f'%(corr))+\
                                '   pval = {p}'.format(p = "{:.2e}".format(pval))) 
          
        ## create animation      
        ani = FuncAnimation(fig, update_parallel_stim, 
                            frames=range(parallel_avg_arr.shape[0]), 
                            fargs = (parallel_avg_arr, parallel_flip_avg_arr,
                                    bar_ecc, downsample_FA_DM_list,
                                    vmin, vmax, cmap, annot),
                            interval=interval)
        
        if filename is None:
            return ani
        else:
            ani.save(filename=filename, writer="ffmpeg") # save mp4 file
    
    def animate_crossed_stim(self, participant_list = [], average_stim_dict = None, flip_average_stim_dict = None, 
                                    run_position_df_dict = None, lowres_DM_dict = None,
                                    bar_ecc = 'far',  vmin = 0, vmax = .4, cmap = 'plasma', annot = False, 
                                    roi_name = 'V1', interval=600, filename = None):
        
        """Create animation with average reconstructed stim for crossed bar positions
        for attended bar at specific eccentricity
        """
        
        ## turn dict into average array, to plot in movie
        crossed_avg_arr = [np.nanmean(average_stim_dict['crossed'][bar_ecc][ecc_bool], axis = 0) for ecc_bool in [0, 1]]
        crossed_avg_arr = np.stack(crossed_avg_arr)

        crossed_flip_avg_arr = [np.nanmean(flip_average_stim_dict['crossed'][bar_ecc][ecc_bool], axis = 0) for ecc_bool in [0, 1]]
        crossed_flip_avg_arr = np.stack(crossed_flip_avg_arr)
        
        ## get frames of DM with corresponding bar position
        
        # get example DM for participant in list
        pp = participant_list[0]
        df_key = list(run_position_df_dict['sub-{sj}'.format(sj = pp)].keys())[0]
        
        downsample_FA_DM_list = []
        
        for i in [0, 1]:

            DM_trl_ind = self.get_uniq_cond_trl_ind(position_df = run_position_df_dict['sub-{sj}'.format(sj = pp)][df_key], 
                                                    bar_ecc = bar_ecc, 
                                                    bars_pos = 'crossed', 
                                                    same_ecc = i)
            if DM_trl_ind is None:
                downsample_FA_DM_list.append(None)
            else:
                downsample_FA_DM_list.append(lowres_DM_dict['sub-{sj}'.format(sj = pp)]['full_stim'][df_key][DM_trl_ind])
                
        ## initialize base figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (10,10))

        fig.suptitle('Reconstructed stim (crossed bars), ROI - %s'%roi_name, fontsize=16)
    
        ## set function to update frames
        def update_crossed_stim(frame, stim_arr = [], flip_stim_arr = [], 
                                bar_ecc = 'far', dm_list = [],
                                vmin = 0, vmax = .4, cmap = 'plasma', annot = False):
            
            bar_ecc_ind = {'far': 1, 'middle': 2, 'near': 3}
            # make reference dict with unique conditions of attended and unattend
            uniq_cond_dict = {'far': 'near', 'near': 'middle', 'middle': 'far'}

            # bar ecc list of attended and unattended bar
            bar_ecc_list = [bar_ecc, bar_ecc] if frame == 1 else [bar_ecc, uniq_cond_dict[bar_ecc]]
            
            # in case we didnt provide dm
            if len(dm_list) == 0 or dm_list[frame] is None:
                dm2plot = np.zeros((8,8))
                dm2plot[bar_ecc_ind[bar_ecc_list[0]]] = 1
                dm2plot[..., bar_ecc_ind[bar_ecc_list[1]]] = 1
            else:
                dm2plot = dm_list[frame]
            
            # clear axis of fig
            axes[0][0].clear() 
            axes[1][0].clear() 
            axes[0][1].clear() 
            axes[1][1].clear() 
            
            ## attended leftmost
            sns.heatmap(stim_arr[frame], cmap = cmap, ax = axes[0][0], 
                        square = True, cbar = False,
                        annot=annot, annot_kws={"size": 7},
                        vmin = vmin, vmax = vmax)
            axes[0][0].set_title('Same eccentricity = %s'%(str(bool(frame))))
            
            ## reversed case attend upper
            sns.heatmap(flip_stim_arr[frame], cmap = cmap, ax = axes[1][0], 
                        square = True, cbar = False,
                        annot=annot, annot_kws={"size": 7},
                        vmin = vmin, vmax = vmax)

            ## DMs
            # attend left
            axes[0][1].imshow(dm2plot.T, cmap = 'binary_r', vmax = 1.5)
            # Add the patch to the Axes
            axes[0][1].add_patch(patches.Rectangle((bar_ecc_ind[bar_ecc_list[0]] - .5, -.5), 1, 8, 
                                                    linewidth = 2, edgecolor='purple', 
                                                    facecolor='purple', hatch = '///'))
            
            # annotate correlation value between stim and DM
            corr, pval = scipy.stats.pointbiserialr(stim_arr[frame].ravel(), 
                                            dm2plot.T.ravel())
            axes[0][1].set_title(r"$\rho$ = {r}".format(r = '%.2f'%(corr))+\
                                '   pval = {p}'.format(p = "{:.2e}".format(pval)))

            # attend up
            axes[1][1].imshow(dm2plot.T, cmap = 'binary_r', vmax = 1.5)
            # Add the patch to the Axes
            axes[1][1].add_patch(patches.Rectangle((-.5, bar_ecc_ind[bar_ecc_list[1]] - .5), 8, 1, 
                                                    angle=0.0,
                                                    linewidth = 2, edgecolor='green', 
                                                    facecolor='green', hatch = '///'))
            
            # annotate correlation value between stim and DM
            corr, pval = scipy.stats.pointbiserialr(flip_stim_arr[frame].ravel(), 
                                            dm2plot.T.ravel())
            axes[1][1].set_title(r"$\rho$ = {r}".format(r = '%.2f'%(corr))+\
                                '   pval = {p}'.format(p = "{:.2e}".format(pval))) 
          
        ## create animation      
        ani = FuncAnimation(fig, update_crossed_stim, 
                            frames=range(crossed_avg_arr.shape[0]), 
                            fargs = (crossed_avg_arr, crossed_flip_avg_arr,
                                    bar_ecc, downsample_FA_DM_list,
                                    vmin, vmax, cmap, annot),
                            interval=interval)
        
        if filename is None:
            return ani
        else:
            ani.save(filename=filename, writer="ffmpeg") # save mp4 file

    def plot_decoder_results(self, participant_list = [], ROI_list = ['V1'], model_type = 'gauss_hrf',
                        prf_file_ext = '_cropped_dc_psc.nii.gz', ses = 'mean', fa_file_ext = '_cropped.nii.gz',
                        mask_bool_df = None, stim_on_screen = [], group_bar_pos_df = [], prf_bar_coords_dict = None):
        
        """plot reconstructed stim averaged over unique conditions
        for all participants and ROIs
        """
        
        # make dir to save estimates
        fig_dir = op.join(self.MRIObj.derivatives_pth, 'plots', 'reconstructed_stim')
        fig_id = 'sub-GROUP_task-FA_model-{modname}_reconstructed_stim'.format(modname = model_type)
        
        if len(participant_list) == 1:
            pp = participant_list[0]
            fig_dir = op.join(fig_dir, 'sub-{sj}'.format(sj = pp))
            fig_id = fig_id.replace('sub-GROUP', 'sub-{sj}'.format(sj = pp)) 
        
        os.makedirs(fig_dir, exist_ok = True)
        print('saving figures in %s'%fig_dir)
        
        ## load participant data keys
        # for reference later on
        data_keys_dict = self.MRIObj.beh_utils.get_data_keys_dict(participant_list = participant_list, 
                                                                group_bar_pos_dict = group_bar_pos_df)
        
        ## get downsampled FA DM for group
        lowres_DM_dict = self.load_group_DM_dict(participant_list = participant_list, 
                                                group_bar_pos_df = group_bar_pos_df, 
                                                data_keys_dict = data_keys_dict,
                                                prf_bar_coords_dict = prf_bar_coords_dict)
        
        ## get FA bar position dict, across runs, for group
        run_position_df_dict = self.load_group_run_position_df_dict(participant_list = participant_list, 
                                                                    group_bar_pos_df = group_bar_pos_df,  
                                                                    data_keys_dict = data_keys_dict,
                                                                    prf_bar_coords_dict = prf_bar_coords_dict)
        
        # iterate over ROIs             
        for roi_name in ROI_list:
            
            base_filename = op.join(fig_dir, fig_id+'_ROI-{rname}'.format(rname = roi_name))
            
            print('Plotting decoder results for ROI %s'%roi_name)
            
            ## load reconstructed stim
            reconstructed_stim_dict = self.load_group_decoded_stim_dict(participant_list = participant_list, 
                                                                        roi_name = roi_name,
                                                                        model_type = model_type, 
                                                                        data_keys_dict = data_keys_dict)
            
            ## average reconstructed stim across runs
            reconstructed_stim_dict = self.get_run_stim_average(participant_list = participant_list, 
                                                                reconstructed_stim_dict = reconstructed_stim_dict,
                                                                lowres_DM_dict = lowres_DM_dict, 
                                                                data_keys_dict = data_keys_dict)
            
            ## get reference data keys for all participants
            reference_data_keys_dict = {key: item[:1] for key, item in data_keys_dict.items()}
            
            # set vmax for plots, at 90% of value distribution
            all_pix_values = np.array([items2.values.ravel() for keynames1, items1 in reconstructed_stim_dict.items() for keynames2, items2 in items1.items()])
            all_pix_values = all_pix_values.ravel()
            vmax = np.quantile(all_pix_values, .85)
            print('setting vmax as %.2f'%vmax)
               
            # print correlation values, just to check         
            if len(participant_list) == 1:
                
                ## correlate reconstructed stim with downsampled DM across runs
                for ind, df_key in enumerate(reference_data_keys_dict['sub-{sj}'.format(sj = pp)]):
                    
                    corr, pval = scipy.stats.pointbiserialr(reconstructed_stim_dict['sub-{sj}'.format(sj = pp)][df_key].values.ravel(), 
                                                    lowres_DM_dict['sub-{sj}'.format(sj = pp)]['full_stim'][df_key].ravel())
                    print('correlation between reconstructed stim and DM is %.2f, %.2f'%(corr, pval))

                ## find trials where attended bar and unattended bar in same position
                same_bar_pos_ind_dict = self.get_same_bar_pos_ind_dict(lowresDM_dict = lowres_DM_dict['sub-{sj}'.format(sj = pp)], 
                                                                        data_keys = reference_data_keys_dict['sub-{sj}'.format(sj = pp)])

                for ind, df_key in enumerate(reference_data_keys_dict['sub-{sj}'.format(sj = pp)]):
                    
                    corr, pval = scipy.stats.pearsonr(reconstructed_stim_dict['sub-{sj}'.format(sj = pp)][df_key].values[same_bar_pos_ind_dict[df_key][:,0]].ravel(), 
                                                    reconstructed_stim_dict['sub-{sj}'.format(sj = pp)][df_key].values[same_bar_pos_ind_dict[df_key][:,1]].ravel())
                    print('correlation between trials where\nattended bar and unattended bar in same position is %.2f, %.2f'%(corr, pval))
                    
            ## get average stim (across unique bar positions) for all participant's ROI
            # and bar type (parallel or crossed)

            average_stim_dict = {'parallel': {}, 'crossed': {}}
            flip_average_stim_dict = {'parallel': {}, 'crossed': {}}

            average_stim_dict['parallel'], flip_average_stim_dict['parallel'] = self.get_average_stim_dict(participant_list = participant_list,
                                                                                                        reconstructed_stim_dict = reconstructed_stim_dict, 
                                                                                                        run_position_df_dict = run_position_df_dict, 
                                                                                                        lowres_DM_dict = lowres_DM_dict, 
                                                                                                        bar_type = 'parallel')
            
            average_stim_dict['crossed'], flip_average_stim_dict['crossed'] = self.get_average_stim_dict(participant_list = participant_list,
                                                                                                        reconstructed_stim_dict = reconstructed_stim_dict, 
                                                                                                        run_position_df_dict = run_position_df_dict, 
                                                                                                        lowres_DM_dict = lowres_DM_dict, 
                                                                                                        bar_type = 'crossed')

            ## create animation 
            # for the different ecc
            for bar_ecc in ['far', 'middle', 'near']:
                
                # for parallel bars
                self.animate_parallel_stim(participant_list = participant_list, 
                                        average_stim_dict = average_stim_dict, 
                                        flip_average_stim_dict = flip_average_stim_dict, 
                                        run_position_df_dict = run_position_df_dict, 
                                        lowres_DM_dict = lowres_DM_dict,
                                        bar_ecc = bar_ecc,  
                                        vmin = 0, 
                                        vmax = vmax, 
                                        cmap = 'magma', 
                                        annot = False, 
                                        roi_name = roi_name, 
                                        interval = 800, 
                                        filename = base_filename+'_barPOS-parallel_barECC-{be}.mp4'.format(be = bar_ecc))
                
                # for crossed bars
                self.animate_crossed_stim(participant_list = participant_list, 
                                        average_stim_dict = average_stim_dict, 
                                        flip_average_stim_dict = flip_average_stim_dict, 
                                        run_position_df_dict = run_position_df_dict, 
                                        lowres_DM_dict = lowres_DM_dict,
                                        bar_ecc = bar_ecc,  
                                        vmin = 0, 
                                        vmax = vmax, 
                                        cmap = 'magma', 
                                        annot = False, 
                                        roi_name = roi_name, 
                                        interval = 800, 
                                        filename = base_filename+'_barPOS-crossed_barECC-{be}.mp4'.format(be = bar_ecc))
                
            ## plot png of panels too, with annotation of values
            ## PARALLEL BARS ##
            pp = participant_list[0]
            df_key = list(run_position_df_dict['sub-{sj}'.format(sj = pp)].keys())[0]

            for bar_ecc in average_stim_dict['parallel'].keys():
                for bar_dist in average_stim_dict['parallel'][bar_ecc].keys():
                    
                    # get frame of DM with corresponding bar position
                    DM_trl_ind = self.get_uniq_cond_trl_ind(position_df = run_position_df_dict['sub-{sj}'.format(sj = pp)][df_key], 
                                                            bar_ecc = bar_ecc, 
                                                            bars_pos = 'parallel', 
                                                            bar_dist = bar_dist)

                    ## plot figure
                    self.plot_avg_parallel_stim(average_stim = np.nanmean(average_stim_dict['parallel'][bar_ecc][bar_dist], axis = 0), 
                                                flip_average_stim = np.nanmean(flip_average_stim_dict['parallel'][bar_ecc][bar_dist], axis = 0), 
                                                DM_trl_ind = DM_trl_ind,
                                                bar_ecc = bar_ecc, 
                                                bar_dist = bar_dist, 
                                                downsample_FA_DM = lowres_DM_dict['sub-{sj}'.format(sj = pp)]['full_stim'][df_key], 
                                                vmin = 0, 
                                                vmax = vmax, 
                                                cmap = 'magma',
                                                annot = True,
                                                filename = base_filename+'_barPOS-parallel_barECC-{be}_barDIST-{bd}.png'.format(be = bar_ecc,
                                                                                                                               bd = bar_dist))
                
            ## CROSSED BARS ##
            for bar_ecc in average_stim_dict['crossed'].keys():
                for same_ecc in average_stim_dict['crossed'][bar_ecc].keys():
                
                    # get frame of DM with corresponding bar position
                    DM_trl_ind = self.get_uniq_cond_trl_ind(position_df = run_position_df_dict['sub-{sj}'.format(sj = pp)][df_key], 
                                                            bar_ecc = bar_ecc, 
                                                            bars_pos = 'crossed', 
                                                            same_ecc = same_ecc)

                    ## plot figure
                    self.plot_avg_crossed_stim(average_stim = np.nanmean(average_stim_dict['crossed'][bar_ecc][same_ecc], axis = 0), 
                                                flip_average_stim = np.nanmean(flip_average_stim_dict['crossed'][bar_ecc][same_ecc], axis = 0), 
                                                DM_trl_ind = DM_trl_ind,
                                                bar_ecc = bar_ecc, 
                                                same_ecc = same_ecc, 
                                                downsample_FA_DM = lowres_DM_dict['sub-{sj}'.format(sj = pp)]['full_stim'][df_key], 
                                                vmin = 0, 
                                                vmax = vmax, 
                                                cmap = 'magma', 
                                                annot = True,
                                    filename = base_filename+'_barPOS-crossed_barECC-{be}_barDIST-{bd}.png'.format(be = bar_ecc,
                                                                                                                   bd = str(bool(same_ecc))))
                    
            ## get drive (median value within bar location) ##
            # PARALLEL BARS #
            drive_df_parallel = self.get_bar_drive(participant_list = participant_list, 
                                                    average_stim_dict = average_stim_dict, 
                                                    flip_average_stim_dict = flip_average_stim_dict, 
                                                    run_position_df_dict = run_position_df_dict, 
                                                    lowres_DM_dict = lowres_DM_dict,
                                                    bar_type = 'parallel')

            # CROSSED BARS #
            drive_df_crossed = self.get_bar_drive(participant_list = participant_list, 
                                                average_stim_dict = average_stim_dict, 
                                                flip_average_stim_dict = flip_average_stim_dict, 
                                                run_position_df_dict = run_position_df_dict, 
                                                lowres_DM_dict = lowres_DM_dict,
                                                bar_type = 'crossed')

            # COMBINED #
            drive_df = pd.concat((drive_df_parallel, drive_df_crossed), ignore_index = True)
            
            ## and make some bar plots to check
            # overall 
            self.plot_barplot_avg_drive(drive_df = drive_df, 
                                        hue_order = ['att_bar', 'unatt_bar'], 
                                        bar_key_list = ['parallel', 'crossed', None], 
                                        roi_name = roi_name, 
                                        filename = base_filename+'_avg_drive_barplot.png')
            # split by ecc 
            self.plot_barplot_ecc_drive(drive_df = drive_df, 
                                        hue_order = ['att_bar', 'unatt_bar'], 
                                        bar_key_list = ['parallel', 'crossed', None], 
                                        roi_name = roi_name, 
                                        filename = base_filename+'_avg_drive_ECC_barplot.png')
            # parallel - split by inter-bar distance 
            self.plot_barplot_parallel_dist_drive(drive_df_parallel = drive_df_parallel, 
                                                hue_order = ['att_bar', 'unatt_bar'], 
                                                roi_name = roi_name, 
                                                filename = base_filename+'_avg_drive_barPOS-parallel_barDIST_barplot.png')
            # crossed - split by inter-bar "distance" 
            self.plot_barplot_crossed_dist_drive(drive_df_crossed = drive_df_crossed, 
                                                hue_order = ['att_bar', 'unatt_bar'], 
                                                roi_name = roi_name, 
                                                filename = base_filename+'_avg_drive_barPOS-crossed_barDIST_barplot.png')
            
    def get_bar_drive(self, participant_list = [], average_stim_dict = None, flip_average_stim_dict = None, 
                            run_position_df_dict = None, lowres_DM_dict = None,
                            bar_type = 'parallel'):
        
        """Get average drive value within bar
        returns dataframe with drive values
        """
        
        ## reference dict with bar ecc for cond and flipped case
        bar_ecc_ind = {'far': 1, 'middle': 2, 'near': 3}
        flip_bar_ecc = np.array(['far', 'middle', 'near', 'near', 'middle', 'far'])
        
        output_df = []
        
        if bar_type == 'parallel':

            pp = participant_list[0]
            df_key = list(run_position_df_dict['sub-{sj}'.format(sj = pp)].keys())[0]

            for bar_ecc in average_stim_dict['parallel'].keys():
                
                for bar_dist in average_stim_dict['parallel'][bar_ecc].keys():
                    
                    # get frame of DM with corresponding bar position
                    DM_trl_ind = self.get_uniq_cond_trl_ind(position_df = run_position_df_dict['sub-{sj}'.format(sj = pp)][df_key], 
                                                            bar_ecc = bar_ecc, 
                                                            bars_pos = 'parallel', 
                                                            bar_dist = bar_dist)

                    # get dm array, for masking
                    dm_arr = lowres_DM_dict['sub-{sj}'.format(sj = pp)]['full_stim'][df_key][DM_trl_ind]
                    att_dm_arr = lowres_DM_dict['sub-{sj}'.format(sj = pp)]['att_bar'][df_key][DM_trl_ind]
                    unatt_dm_arr = lowres_DM_dict['sub-{sj}'.format(sj = pp)]['unatt_bar'][df_key][DM_trl_ind]

                    # attended bar drives
                    att_drive1 = [np.median(average_stim_dict['parallel'][bar_ecc][bar_dist][i][np.where(att_dm_arr.T)], axis = 0) for i in range(len(participant_list))]
                    att_drive2 = [np.median(flip_average_stim_dict['parallel'][bar_ecc][bar_dist][i][np.where(unatt_dm_arr.T)], axis = 0) for i in range(len(participant_list))]
                    att_drive = np.stack((att_drive1, att_drive2))

                    # unattended bar drives
                    unatt_drive1 = [np.median(average_stim_dict['parallel'][bar_ecc][bar_dist][i][np.where(unatt_dm_arr.T)], axis = 0) for i in range(len(participant_list))]
                    unatt_drive2 = [np.median(flip_average_stim_dict['parallel'][bar_ecc][bar_dist][i][np.where(att_dm_arr.T)], axis = 0) for i in range(len(participant_list))]
                    unatt_drive = np.stack((unatt_drive1, unatt_drive2))
                    
                    # first store values as dict
                    drive_dict = {'sj': np.stack((participant_list for i in range(4))),
                                'bar_type': np.repeat(['att_bar', 'unatt_bar'], 2)[...,np.newaxis],
                                'bar_ecc': np.tile([bar_ecc, flip_bar_ecc[bar_ecc_ind[bar_ecc] + bar_dist - 1]], 2)[...,np.newaxis],
                                'drive': np.vstack((att_drive, unatt_drive))}

                    # then convert to data frame
                    drive_df = pd.DataFrame({k:list(v) for k,v in drive_dict.items()}).explode(['sj', 'drive'], ignore_index = True)
                    drive_df.loc[:,'bar_type'] = drive_df['bar_type'].map(lambda x: x[0])
                    drive_df.loc[:,'bar_ecc'] = drive_df['bar_ecc'].map(lambda x: x[0])
                    drive_df.loc[:,'bar_dist'] = bar_dist
                    drive_df.loc[:,'bar_orientation'] = 'parallel'
                    
                    ## finally append in outdict
                    output_df.append(drive_df)
                    
        elif bar_type == 'crossed':
            
            uniq_cond_dict = {'far': 'near', 'near': 'middle', 'middle': 'far'}

            pp = participant_list[0]
            df_key = list(run_position_df_dict['sub-{sj}'.format(sj = pp)].keys())[0]

            for bar_ecc in average_stim_dict['crossed'].keys():
                
                for same_ecc in average_stim_dict['crossed'][bar_ecc].keys():
                    
                    # bar ecc list of attended and unattended bar
                    bar_ecc_list = [bar_ecc, bar_ecc] if same_ecc == True else [bar_ecc, uniq_cond_dict[bar_ecc]]

                    # get frame of DM with corresponding bar position
                    DM_trl_ind = self.get_uniq_cond_trl_ind(position_df = run_position_df_dict['sub-{sj}'.format(sj = pp)][df_key], 
                                                            bar_ecc = bar_ecc, 
                                                            bars_pos = 'crossed', 
                                                            same_ecc = same_ecc)
                    
                    # in case ref trial was masked
                    if DM_trl_ind is None:
                        dm_arr = np.zeros((8,8))
                        dm_arr[bar_ecc_ind[bar_ecc_list[0]]] = 1
                        dm_arr[..., bar_ecc_ind[bar_ecc_list[1]]] = 1
                        att_dm_arr = np.zeros((8,8))
                        att_dm_arr[bar_ecc_ind[bar_ecc_list[0]]] = 1
                        unatt_dm_arr = np.zeros((8,8))
                        unatt_dm_arr[..., bar_ecc_ind[bar_ecc_list[1]]] = 1
                    else:
                        # get dm array, for masking
                        dm_arr = lowres_DM_dict['sub-{sj}'.format(sj = pp)]['full_stim'][df_key][DM_trl_ind]
                        att_dm_arr = lowres_DM_dict['sub-{sj}'.format(sj = pp)]['att_bar'][df_key][DM_trl_ind]
                        unatt_dm_arr = lowres_DM_dict['sub-{sj}'.format(sj = pp)]['unatt_bar'][df_key][DM_trl_ind]

                    # attended bar drives
                    att_drive1 = [np.median(average_stim_dict['crossed'][bar_ecc][same_ecc][i][np.where(att_dm_arr.T)], axis = 0) for i in range(len(participant_list))]
                    att_drive2 = [np.median(flip_average_stim_dict['crossed'][bar_ecc][same_ecc][i][np.where(unatt_dm_arr.T)], axis = 0) for i in range(len(participant_list))]
                    att_drive = np.stack((att_drive1, att_drive2))

                    # unattended bar drives
                    unatt_drive1 = [np.median(average_stim_dict['crossed'][bar_ecc][same_ecc][i][np.where(unatt_dm_arr.T)], axis = 0) for i in range(len(participant_list))]
                    unatt_drive2 = [np.median(flip_average_stim_dict['crossed'][bar_ecc][same_ecc][i][np.where(att_dm_arr.T)], axis = 0) for i in range(len(participant_list))]
                    unatt_drive = np.stack((unatt_drive1, unatt_drive2))

                    # first store values as dict
                    drive_dict = {'sj': np.stack((participant_list for i in range(4))),
                                'bar_type': np.repeat(['att_bar', 'unatt_bar'], 2)[...,np.newaxis],
                                'bar_ecc': np.tile(bar_ecc_list, 2)[...,np.newaxis],
                                'drive': np.vstack((att_drive, unatt_drive))}

                    # then convert to data frame
                    drive_df = pd.DataFrame({k:list(v) for k,v in drive_dict.items()}).explode(['sj', 'drive'], ignore_index = True)
                    drive_df.loc[:,'bar_type'] = drive_df['bar_type'].map(lambda x: x[0])
                    drive_df.loc[:,'bar_ecc'] = drive_df['bar_ecc'].map(lambda x: x[0])
                    drive_df.loc[:,'bar_dist'] = bool(same_ecc)
                    drive_df.loc[:,'bar_orientation'] = 'crossed'

                    ## finally append in outdict
                    output_df.append(drive_df)
              
        # turn into full df      
        output_df = pd.concat(output_df, ignore_index = True)
        # turn drive values to float
        output_df.loc[:,'drive'] = output_df.drive.astype(float)
            
        return output_df
                            
    def plot_barplot_crossed_dist_drive(self, drive_df_crossed = None, hue_order = ['att_bar', 'unatt_bar'], roi_name = 'V1',
                                            filename = None):
        
        """ drive for crossed bars at same/different inter-bar eccs
        """
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (10,5), sharey = True)

        for ind, same_ecc in enumerate([0,1]):
            
            df2plot = drive_df_crossed[drive_df_crossed['bar_dist'] == same_ecc]

            sns.barplot(df2plot, 
                        y = 'drive', x = 'bar_type', estimator = np.mean, errorbar=('se'), 
                        capsize = .3 ,linewidth = 3, palette = ['#229c8d', '#cc742d'],
                    order = hue_order, ax = axes[ind])
            
            axes[ind].set_title('Bars at same ecc = %s'%(str(bool(same_ecc)).upper()))

            # overlay participant lines
            for pp in df2plot.sj.unique():
                sns.pointplot(df2plot[df2plot['sj'] == pp], 
                            y = 'drive', x = 'bar_type',
                            estimator = np.mean, errorbar=('se'),
                            order = hue_order,
                            linestyles = '--', alpha = .3, 
                            legend = False, ax = axes[ind])

        axes[ind].set_ylim([np.quantile(drive_df_crossed.drive.values, .01), 
                            np.quantile(drive_df_crossed.drive.values, .98)])
        fig.suptitle('Average drive within CROSSED bars - {rk}'.format(rk = roi_name), fontsize=16, y=.99)
        
        # save figure
        if filename is not None:
            fig.savefig(filename, dpi= 200)
          
    def plot_barplot_parallel_dist_drive(self, drive_df_parallel = None, hue_order = ['att_bar', 'unatt_bar'], roi_name = 'V1',
                                                filename = None):
        
        """ drive for parallel bars at different inter-bar distances
        """
                
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize = (15,5), sharey = True)

        for ind, dist in enumerate(np.arange(5)):
            
            df2plot = drive_df_parallel[drive_df_parallel['bar_dist'] == dist + 1]

            sns.barplot(df2plot, 
                        y = 'drive', x = 'bar_type', estimator = np.mean, errorbar=('se'), 
                        capsize = .3 ,linewidth = 3, palette = ['#229c8d', '#cc742d'],
                    order = hue_order, ax = axes[ind])
            
            axes[ind].set_title('Bar Distance %i'%(dist + 1))

            # overlay participant lines
            for pp in df2plot.sj.unique():
                sns.pointplot(df2plot[df2plot['sj'] == pp], 
                            y = 'drive', x = 'bar_type',
                            estimator = np.mean, errorbar=('se'),
                            order = hue_order,
                            linestyles = '--', alpha = .3, 
                            legend = False, ax = axes[ind])

        axes[ind].set_ylim([np.quantile(drive_df_parallel.drive.values, .01), 
                            np.quantile(drive_df_parallel.drive.values, .99)])
        fig.suptitle('Average drive within PARALELL bars - {rk}'.format(rk = roi_name), fontsize=16, y=.99)
        
        # save figure
        if filename is not None:
            fig.savefig(filename, dpi= 200)
        
    def plot_barplot_ecc_drive(self, drive_df = None, hue_order = ['att_bar', 'unatt_bar'], roi_name = 'V1',
                                bar_key_list = ['parallel', 'crossed', None], filename = None):
        
        """ drive for each eccentricity
        """
                
        ## plot overall drive of attended vs unattended bar  ##
        fig, axes = plt.subplots(nrows=3, ncols=len(bar_key_list), figsize = (15,12), sharey = True)

        # iterate over bar types
        for ind_r, bar_key in enumerate(bar_key_list):
            
            if bar_key is None:
                df2plot = drive_df
                bk = 'parallel + crossed'
            else:
                df2plot = drive_df[drive_df['bar_orientation'] == bar_key]
                bk = bar_key
                
            for ind_c, ecc in enumerate(['near', 'middle', 'far']):

                df2plot_ecc = df2plot[df2plot['bar_ecc'] == ecc]

                sns.barplot(df2plot_ecc, 
                            y = 'drive', x = 'bar_type', estimator = np.mean, errorbar=('se'), 
                            capsize = .3 ,linewidth = 3, palette = ['#229c8d', '#cc742d'],
                        order = hue_order, ax = axes[ind_r][ind_c])
                
                # overlay participant lines
                for pp in df2plot.sj.unique():
                    sns.pointplot(df2plot_ecc[df2plot_ecc['sj'] == pp], 
                                y = 'drive', x = 'bar_type',
                                estimator = np.mean, errorbar=('se'),
                                order = hue_order,
                                linestyles = '--', alpha = .3, 
                                legend = False, ax = axes[ind_r][ind_c])
                    
                if ind_r < 2:
                    axes[ind_r][ind_c].set_xticks([])
                    axes[ind_r][ind_c].set_xlabel('')
                    
                axes[ind_r][ind_c].set_title('{ek} ecc, {bk} bars'.format(bk = bk.upper(), ek = ecc.upper()))

        axes[ind_r][ind_c].set_ylim([np.quantile(df2plot_ecc.drive.values, .01), 
                                    np.quantile(df2plot_ecc.drive.values, .98)])
        fig.suptitle('Average drive within bar - {rk}'.format(rk = roi_name), fontsize=16, y=.95)
            
        # save figure
        if filename is not None:
            fig.savefig(filename, dpi= 200)
               
    def plot_barplot_avg_drive(self, drive_df = None, hue_order = ['att_bar', 'unatt_bar'], roi_name = 'V1',
                                bar_key_list = ['parallel', 'crossed', None], filename = None):
        
        """Overall drive
        """
                
        ## plot overall drive of attended vs unattended bar  ##
        fig, axes = plt.subplots(nrows=1, ncols=len(bar_key_list), figsize = (15,5), sharey = True)

        # iterate over bar types
        for ind, bar_key in enumerate(bar_key_list):
            
            if bar_key is None:
                df2plot = drive_df
                bk = 'parallel + crossed'
            else:
                df2plot = drive_df[drive_df['bar_orientation'] == bar_key]
                bk = bar_key
                
            sns.barplot(df2plot, y = 'drive', x = 'bar_type', estimator = np.mean, errorbar=('se'), 
                        capsize = .3 ,linewidth = 3, palette = ['#229c8d', '#cc742d'], 
                        hue_order = hue_order, ax = axes[ind])

            # overlay participant lines
            for pp in df2plot.sj.unique():

                sns.pointplot(df2plot[df2plot['sj'] == pp], 
                            y = 'drive', x = 'bar_type', estimator = np.mean, errorbar=('se'),
                            hue_order = hue_order,
                            linestyles = '--', alpha = .3, #color = 'k', 
                            legend = False, ax = axes[ind])
            
            axes[ind].set_title('{bk} bars'.format(bk = bk.upper()))

        axes[ind].set_ylim([np.quantile(df2plot.drive.values, .01), 
                            np.quantile(df2plot.drive.values, .98)])
        fig.suptitle('Average drive within bar - {rk}'.format(rk = roi_name), fontsize=16, y=1)
        
        # save figure
        if filename is not None:
            fig.savefig(filename, dpi= 200)
        
    def get_pp_stim_visual_dm_correlation(self, reconstructed_stim_dict = None, lowres_DM_dict = None, data_keys_dict = [],
                                                mask_nan = False):
        
        """
        correlate reconstructed stim with downsampled DM
        across runs for a given participant
        """ 
        
        ## if single trial correlation, flatten 2D arrays
        stim_runs, dm_runs = self.flatten_stim_dm(reconstructed_stim_dict = reconstructed_stim_dict, 
                                                  lowres_DM_dict = lowres_DM_dict, 
                                                  data_keys_dict = data_keys_dict, 
                                                  mask_nan = mask_nan)
        
        # correlate reconstructed stim with downsampled DM
        corr, pval = scipy.stats.pointbiserialr(stim_runs, dm_runs)
        
        return corr, pval
    
    def flatten_stim_dm(self, reconstructed_stim_dict = None, lowres_DM_dict = None, data_keys_dict = [], mask_nan = False):

        """
        convert reconstructed stim and downsampled DM matrix to 1D array
        and stack horizontally all participant runs
        """

        # check for masked trials -> make index array for each run
        bar_present_dict = {df_key: np.where((np.sum(np.sum(lowres_DM_dict[df_key], 
                                         axis = -1), 
                                        axis = -1) != 0))[0] for df_key in data_keys_dict}
        
        # concatenate arrays across runs
        stim_runs = np.hstack((reconstructed_stim_dict[df_key].values[bar_present_dict[df_key]].ravel() for df_key in data_keys_dict))
        dm_runs = np.hstack((lowres_DM_dict[df_key][bar_present_dict[df_key]].ravel() for df_key in data_keys_dict))

        # if we want to mask nan 
        if mask_nan:
            mask_ind = np.where((~np.isnan(dm_runs)))[0]
            stim_runs = stim_runs[mask_ind]
            dm_runs = dm_runs[mask_ind]

        return stim_runs, dm_runs

    def get_stim_visual_dm_correlation(self, participant_list = [], reconstructed_stim_dict = None, 
                                            lowres_DM_dict = None, data_keys_dict = [], ROI_list = [], mask_nan = False):
        
        """
        For each ROI, correlate reconstructed stim with downsampled DM
        across runs for all participants
        and return df with values
        """ 
        
        stim_corr_df = []

        for roi_name in  ROI_list:

            print('Correlation %s'%roi_name)
        
            for participant in participant_list:
                
                pp_corr, pp_pval = self.get_pp_stim_visual_dm_correlation(reconstructed_stim_dict = reconstructed_stim_dict[roi_name]['sub-{sj}'.format(sj = participant)], 
                                                                        lowres_DM_dict = lowres_DM_dict['sub-{sj}'.format(sj = participant)]['full_stim'], 
                                                                        data_keys_dict = data_keys_dict['sub-{sj}'.format(sj = participant)],
                                                                        mask_nan = mask_nan)
            
                stim_corr_df.append(pd.DataFrame({'sj': ['sub-{sj}'.format(sj = participant)],
                                                'corr': [pp_corr],
                                                'pval': [pp_pval],
                                                'ROI': [roi_name]}
                                                ))

                print('sub-{sj} correlation between reconstructed stim and DM is {c}, {p}'.format(sj = participant,
                                                                                                c = '%.2f'%pp_corr,
                                                                                                p = '%.2f'%pp_pval))
        
        stim_corr_df = pd.concat(stim_corr_df, ignore_index=True)  
        
        return stim_corr_df
    
    def get_run_avg_stim_dm_correlation(self, participant_list = [], ROI_list = [], group_stim_dict = None, 
                                            group_refDM_dict = None):
        
        """
        For each ROI, correlate reconstructed stim (average over runs) 
        with downsampled DM (in order of GLMsingle DM)
        for all participants
        and return df with values
        """ 
        
        avg_stim_corr_df = []

        for roi_name in  ROI_list:

            print('Correlation %s'%roi_name)
        
            for participant in participant_list:
                
                # first get reference dm and set values as binary
                dm4corr = group_refDM_dict['sub-{sj}'.format(sj = participant)].copy()
                dm4corr[dm4corr>0] = 1
                # mask out nan conditions
                mask_ind = np.where((~np.isnan(dm4corr)[...,0,0]))[0]

                pp_corr, pp_pval = scipy.stats.pointbiserialr(group_stim_dict[roi_name]['sub-{sj}'.format(sj = participant)].values[mask_ind].ravel(), 
                                                            dm4corr[mask_ind].ravel())
            
                avg_stim_corr_df.append(pd.DataFrame({'sj': ['sub-{sj}'.format(sj = participant)],
                                                'corr': [pp_corr],
                                                'pval': [pp_pval],
                                                'ROI': [roi_name]}
                                                ))

                print('sub-{sj} correlation between reconstructed stim and DM is {c}, {p}'.format(sj = participant,
                                                                                                c = '%.2f'%pp_corr,
                                                                                                p = '%.2f'%pp_pval))
        
        avg_stim_corr_df = pd.concat(avg_stim_corr_df, ignore_index=True)  
        
        return avg_stim_corr_df

    def get_run_avg_stim_glmsing_trials(self, participant_list = [], ROI_list = [], FA_run_position_df = None, 
                                            trial_combinations_df = None, reconstructed_stim_dict = None, data_keys_dict = None,
                                            lowres_DM_dict = None):

        """
        Average recontructed stim, for each ROI and participant,
        across runs.
        Using same order as DM of GLMsingle (for ease averaging of group and plotting)
        """

        ## store info for all participants
        group_stim_dict = {roi_name: {} for roi_name in ROI_list}
        group_refDM_dict = {}

        ## iterate over ROIs
        for ind, roi_name in enumerate(ROI_list):

            ## for a given ROI, get all participants reconstructed stim
            for participant in participant_list:
                
                pp_avg_stim_df, pp_reference_dm = self.average_pp_stim_glmsing_trials(participant, 
                                                                                    pp_bar_pos_df = FA_run_position_df[FA_run_position_df['sj'] == 'sub-{sj}'.format(sj = participant)], 
                                                                                    trial_combinations_df = trial_combinations_df, 
                                                                                    reconstructed_stim_dict = reconstructed_stim_dict[roi_name], 
                                                                                    data_keys_dict = data_keys_dict,
                                                                                    lowres_DM_dict = lowres_DM_dict)
            
                group_stim_dict[roi_name]['sub-{sj}'.format(sj = participant)] = pp_avg_stim_df
                if ind == 0:
                    group_refDM_dict['sub-{sj}'.format(sj = participant)] = pp_reference_dm

        return group_stim_dict, group_refDM_dict
    
    def average_group_stim_glmsing_trials(self, participant_list = [], ROI_list = [], group_stim_dict = None, group_refDM_dict = None):

        """
        Take participant run-average reconstructed stim, and create group average df
        Using same order as DM of GLMsingle (for ease averaging of group and plotting)
        """

        ## store group average
        avg_stim_dict = {roi_name: [] for roi_name in ROI_list}

        ## iterate over ROIs
        for roi_name in ROI_list:

            # copy structure of first participant
            avg_stim2plot = group_stim_dict[roi_name]['sub-{sj}'.format(sj = participant_list[1])].copy()
            ## average over participants
            avg_stim2plot.loc[:] = np.nanmean(np.stack((group_stim_dict[roi_name]['sub-{sj}'.format(sj = pp)] for pp in participant_list)), axis = 0)
            ## add to dict
            avg_stim_dict[roi_name] = avg_stim2plot

        # also return DM, for consistency
        avg_refDM_dict = group_refDM_dict['sub-{sj}'.format(sj = participant_list[1])].copy()

        return avg_stim_dict, avg_refDM_dict
    
    def get_pp_stim_same_bar_pos_correlation(self, reconstructed_stim_dict = None, lowres_DM_dict = None, data_keys_dict = []):
        
        """
        correlate reconstructed stim for
        trials where attended bar and unattended bar in same position
        across runs for a given participant
        """ 
        
        ## find trials where attended bar and unattended bar in same position
        same_bar_pos_ind_dict = self.get_same_bar_pos_ind_dict(lowresDM_dict = lowres_DM_dict, data_keys = data_keys_dict)
        
        # concatenate arrays across runs
        stimA_runs = np.hstack((reconstructed_stim_dict[df_key].values[same_bar_pos_ind_dict[df_key][:,0]].ravel() for df_key in data_keys_dict))
        stimB_runs = np.hstack((reconstructed_stim_dict[df_key].values[same_bar_pos_ind_dict[df_key][:,1]].ravel() for df_key in data_keys_dict))
        
        # correlate with each other
        corr, pval = scipy.stats.pearsonr(stimA_runs, stimB_runs)
        
        return corr, pval
    
    def get_stim_same_bar_pos_correlation(self, participant_list = [], reconstructed_stim_dict = None, 
                                                lowres_DM_dict = None, data_keys_dict = []):
        
        """
        correlate reconstructed stim for
        trials where attended bar and unattended bar in same position
        across runs for all participants
        and return df with values
        """ 
        
        stim_corr_df = []
        
        for participant in participant_list:
            
            pp_corr, pp_pval = self.get_pp_stim_same_bar_pos_correlation(reconstructed_stim_dict = reconstructed_stim_dict['sub-{sj}'.format(sj = participant)], 
                                                                      lowres_DM_dict = lowres_DM_dict['sub-{sj}'.format(sj = participant)], 
                                                                      data_keys_dict = data_keys_dict['sub-{sj}'.format(sj = participant)])
        
            stim_corr_df.append(pd.DataFrame({'sj': ['sub-{sj}'.format(sj = participant)],
                                              'corr': [pp_corr],
                                              'pval': [pp_pval]}
                                             ))

            print('sub-{sj} correlation between trials where\nattended bar and unattended bar in same position is {c}, {p}'.format(sj = participant,
                                                                                                                                c = '%.2f'%pp_corr,
                                                                                                                                p = '%.2f'%pp_pval))
        
        stim_corr_df = pd.concat(stim_corr_df, ignore_index=True)  
        
        return stim_corr_df    
   
    def get_run_stim_average(self, participant_list = [], reconstructed_stim_dict = None, lowres_DM_dict = None, data_keys_dict = []):
        
        """average reconstructed stim across runs
        for all participants
        """
        
        avg_stim_df_dict = {}
        
        for participant in participant_list:
            
            pp_avg_stim_df = self.get_pp_run_stim_average(reconstructed_stim_dict = reconstructed_stim_dict['sub-{sj}'.format(sj = participant)], 
                                                        lowres_DM_dict = lowres_DM_dict['sub-{sj}'.format(sj = participant)], 
                                                        data_keys_dict = data_keys_dict['sub-{sj}'.format(sj = participant)])
            
            # reference bar positions is first run configuraiton of data df keys, so store as such
            avg_stim_df_dict['sub-{sj}'.format(sj = participant)] = {data_keys_dict['sub-{sj}'.format(sj = participant)][0]: pp_avg_stim_df}
            
        return avg_stim_df_dict
        
    def get_pp_run_stim_average(self, reconstructed_stim_dict = None, lowres_DM_dict = None, data_keys_dict = []):
        
        """average reconstructed stim across runs
        for a given participant
        """
        
        # use first run of list as reference key
        ref_dfkey = data_keys_dict[0]

        # make reference trial index array
        ref_trial_ind = np.arange(len(lowres_DM_dict['full_stim'][ref_dfkey]))
        
        # check for masked trials
        t_mask = np.where((np.sum(np.sum(lowres_DM_dict['full_stim'][ref_dfkey], 
                                         axis = -1), 
                                  axis = -1) == 0))[0]

        # store reconstructed_stim, averaged across runs,
        pp_avg_stim_df = reconstructed_stim_dict[ref_dfkey].copy()

        # for each reference trial
        for ref_t in ref_trial_ind:
            
            # if not masked trial
            if ref_t not in t_mask:
            
                # first array stacked is reference run trial
                avg_values = [pp_avg_stim_df.loc[ref_t].values]
                    
                # iterate over rest of runs
                for dfkey in data_keys_dict[1:]:
                    
                    # get array with trial indices where 
                    # bar position == bar position from reference-run trial
                    trials_ref_barpos = np.where((lowres_DM_dict['full_stim'][dfkey].reshape(len(ref_trial_ind), -1) == lowres_DM_dict['full_stim'][ref_dfkey][ref_t].ravel()).all(-1))[0]

                    # then from those, check which one is the same condition
                    # (attended bar position == attended bar position from reference-run trial)
                    new_t = np.where((lowres_DM_dict['att_bar'][dfkey][trials_ref_barpos,...].reshape(2, -1) == lowres_DM_dict['att_bar'][ref_dfkey][ref_t].ravel()).all(-1))[0][0]
                    dfkey_trial = trials_ref_barpos[new_t]
                    
                    # append trial stim values
                    avg_values.append(reconstructed_stim_dict[dfkey].loc[dfkey_trial].values)
                    
                # replace with average values across runs
                pp_avg_stim_df.loc[ref_t, :] = np.nanmean(np.stack(avg_values), axis = 0)
            else:
                pp_avg_stim_df.loc[ref_t, :] = 0
                
        return pp_avg_stim_df
        
    def mask_reconstructed_stim(self, stim_df = None, y_lim = [], x_lim = []):
        
        """
        mask reconstructed stim df 
        filling positions outside x/y bounds with nans
        """
        
        masked_stim_df = stim_df.copy()
        
        ## insert nans in irrelevant locations
        if not masked_stim_df.iloc[:, masked_stim_df.columns.get_level_values(0) < x_lim[0]].empty:
            masked_stim_df.iloc[:, masked_stim_df.columns.get_level_values(0) < x_lim[0]] = np.nan
            
        if not masked_stim_df.iloc[:, masked_stim_df.columns.get_level_values(1) < y_lim[0]].empty:
            masked_stim_df.iloc[:, masked_stim_df.columns.get_level_values(1) < y_lim[0]] = np.nan
            
        if not masked_stim_df.iloc[:, masked_stim_df.columns.get_level_values(0) > x_lim[-1]].empty:
            masked_stim_df.iloc[:, masked_stim_df.columns.get_level_values(0) > x_lim[-1]] = np.nan
            
        if not masked_stim_df.iloc[:, masked_stim_df.columns.get_level_values(1) > y_lim[-1]].empty:
            masked_stim_df.iloc[:, masked_stim_df.columns.get_level_values(1) > y_lim[-1]] = np.nan
            
        return masked_stim_df
    
    def get_pp_all_trial_drive(self, reconstructed_stim_dict = None, lowres_DM_dict = None, reference_data_keys = [],
                                    run_position_df_dict = None, prf_bar_coords_dict = None):
        
        """
        Get bar drive for all trials (averaged across runs)
        of a given participant
        """
        
        # if we provided prf coordinates as reference, 
        # will get limits in deg and mask stim before getting drive values
        if prf_bar_coords_dict is not None:
            pp_lim_y = self.convert_pix2dva(prf_bar_coords_dict['vertical'])
            pp_lim_y.sort()
            pp_lim_x = self.convert_pix2dva(prf_bar_coords_dict['horizontal'])
            pp_lim_x.sort()
        
        df_drive_pp = []
        
        # loop over reference data key
        for ref_dfkeys in reference_data_keys:
        
            ## get reference stim + dms + bar positions
            pp_ref_att_dm = lowres_DM_dict['att_bar'][ref_dfkeys]
            pp_ref_unatt_dm = lowres_DM_dict['unatt_bar'][ref_dfkeys]
            pp_ref_run_pos_df = run_position_df_dict[ref_dfkeys]
            # mask stim df
            if prf_bar_coords_dict is not None:
                pp_ref_stim = self.mask_reconstructed_stim(stim_df = reconstructed_stim_dict[ref_dfkeys], 
                                                           y_lim = [pp_lim_y[0], pp_lim_y[-1]], 
                                                           x_lim = [pp_lim_x[0], pp_lim_x[-1]])
            else:
                pp_ref_stim = reconstructed_stim_dict[ref_dfkeys]
            
            # trial index aray
            trials_arr = pp_ref_run_pos_df.sort_values('trial_ind').trial_ind.unique()
            
            ## get attended bar drive for all trials
            att_drive_all = np.array([np.nanmean(pp_ref_stim.stack('y', dropna = False).loc[i].iloc[::-1, :].to_numpy()[np.where(pp_ref_att_dm[i].T)], axis = 0) for i in trials_arr])
            ## and same for unattended bar
            unatt_drive_all = np.array([np.nanmean(pp_ref_stim.stack('y', dropna = False).loc[i].iloc[::-1, :].to_numpy()[np.where(pp_ref_unatt_dm[i].T)], axis = 0) for i in trials_arr])

            ## make drive df all trials

            ## first for attended bar
            att_df = pp_ref_run_pos_df.query('attend_condition').sort_values('trial_ind')[['trial_ind', 'bars_pos', 'bar_ecc', 'abs_inter_bar_dist']]
            att_df.rename(columns={'bars_pos': 'bar_orientation', 
                                'abs_inter_bar_dist': 'bar_dist'}, inplace=True)
            # add relevant info
            att_df.loc[:, 'bar_type'] = 'att_bar'
            att_df.loc[:, 'drive'] = att_drive_all

            ## same for unattended bar
            unatt_df = pp_ref_run_pos_df.query('~attend_condition').sort_values('trial_ind')[['trial_ind', 'bars_pos', 'bar_ecc', 'abs_inter_bar_dist']]
            unatt_df.rename(columns={'bars_pos': 'bar_orientation', 
                                'abs_inter_bar_dist': 'bar_dist'}, inplace=True)
            # add relevant info
            unatt_df.loc[:, 'bar_type'] = 'unatt_bar'
            unatt_df.loc[:, 'drive'] = unatt_drive_all

            ## combine and add run/ses id
            df_comb = pd.concat((att_df, unatt_df), ignore_index=True)
            df_comb.loc[:, 'run'] = 'run-{rn}'.format(rn = self.MRIObj.mri_utils.get_run_ses_from_str(ref_dfkeys)[0])
            df_comb.loc[:, 'ses'] = 'ses-{sn}'.format(sn = self.MRIObj.mri_utils.get_run_ses_from_str(ref_dfkeys)[1])
            
            # and append
            df_drive_pp.append(df_comb)
        
        df_drive_pp = pd.concat(df_drive_pp, ignore_index = True)
        
        return df_drive_pp
    
    def get_all_trial_drive(self, participant_list = [], ROI_list = ['V1'],    
                                reconstructed_stim_dict = None, lowres_DM_dict = None, reference_data_keys_dict = None,
                                run_position_df_dict = None, df_FA_beh_RT = None, prf_bar_coords_dict = None):
        
        """
        Get bar drive for all trials (averaged across runs)
        of all participants and ROIs
        """
        
        df_drive = []
                
        for roi_name in ROI_list:
            for participant in tqdm(participant_list):
                
                # if we give prf bar coordinates, use those to mask drive values
                if prf_bar_coords_dict is not None:
                    pp_prf_bar_coords_dict = prf_bar_coords_dict['sub-{sj}'.format(sj = participant)]
                else:
                    pp_prf_bar_coords_dict = None
                                
                df_drive_pp = self.get_pp_all_trial_drive(reconstructed_stim_dict = reconstructed_stim_dict[roi_name]['sub-{sj}'.format(sj = participant)], 
                                                        lowres_DM_dict = lowres_DM_dict['sub-{sj}'.format(sj = participant)], 
                                                        reference_data_keys = reference_data_keys_dict['sub-{sj}'.format(sj = participant)],
                                                        run_position_df_dict = run_position_df_dict['sub-{sj}'.format(sj = participant)],
                                                        prf_bar_coords_dict = pp_prf_bar_coords_dict)
                # add relevant info
                df_drive_pp.loc[:, 'ROI'] = roi_name
                df_drive_pp.loc[:, 'sj'] = 'sub-{sj}'.format(sj = participant)
                
                # if we provided participant behavioral data
                if df_FA_beh_RT is not None:
                    
                    pp_beh_df = df_FA_beh_RT[df_FA_beh_RT['sj'] == 'sub-{sj}'.format(sj = participant)]
                    
                    ## merge info into drive df
                    # and drop incorrect trials
                    df_drive_pp = df_drive_pp.merge(pp_beh_df, on=['sj', 'run', 'ses', 'trial_ind'])
                    df_drive_pp = df_drive_pp[df_drive_pp['correct'] == 1]
                
                # and append
                df_drive.append(df_drive_pp)
        df_drive = pd.concat(df_drive, ignore_index=True)
        
        return df_drive
    
    def average_pp_stim_glmsing_trials(self, pp, pp_bar_pos_df = None, trial_combinations_df = None, 
                                            reconstructed_stim_dict = None, data_keys_dict = None,
                                            lowres_DM_dict = None):
        
        """
        Average ROI recontructed stim for a given participant, across runs,
        in same order of trial combinations (GLM single DM)
        """

        ## get ses and run numbers 
        ses_num_arr = [self.MRIObj.mri_utils.get_run_ses_from_str(dat_id)[1] for dat_id in data_keys_dict['sub-{sj}'.format(sj = pp)]]
        run_num_arr = [self.MRIObj.mri_utils.get_run_ses_from_str(dat_id)[0] for dat_id in data_keys_dict['sub-{sj}'.format(sj = pp)]]

        if trial_combinations_df is None:
            ## get all possible trial combinations
            # to use for bookkeeping of single trial DM
            trial_combinations_df = self.get_single_trial_combinations()

        ## copy reconstructed stim of random run, to keep structure
        pp_avg_stim_df = reconstructed_stim_dict['sub-{sj}'.format(sj = pp)][data_keys_dict['sub-{sj}'.format(sj = pp)][0]].copy()

        ## get participant DM dict
        pp_dm = lowres_DM_dict['sub-{sj}'.format(sj = pp)]

        ## create reference dm to fill
        pp_reference_dm = np.zeros(pp_dm['full_stim'][data_keys_dict['sub-{sj}'.format(sj = pp)][0]].shape)

        ## per unique condition
        for cond_ind in range(len(trial_combinations_df)):

            # filter bar position df for condition
            cond_df = pp_bar_pos_df[(pp_bar_pos_df['att_x_pos'] == trial_combinations_df.iloc[cond_ind].AttBar_bar_midpoint[0]) &\
                                    (pp_bar_pos_df['att_y_pos'] == trial_combinations_df.iloc[cond_ind].AttBar_bar_midpoint[1]) &\
                                    (pp_bar_pos_df['unatt_x_pos'] == trial_combinations_df.iloc[cond_ind].UnattBar_bar_midpoint[0]) &\
                                    (pp_bar_pos_df['unatt_y_pos'] == trial_combinations_df.iloc[cond_ind].UnattBar_bar_midpoint[1])
                                    ]

            # if condition wasn't masked
            if len(cond_df) == len(run_num_arr):
            
                avg_arr = []
                
                # iterate over runs to stack
                for _, row in cond_df.loc[:, ['run', 'ses', 'trial_ind']].iterrows():
                    
                    r_stim = reconstructed_stim_dict['sub-{sj}'.format(sj = pp)]['{sn}_{rn}'.format(sn = row['ses'],
                                                                                                    rn = row['run'])]
                    avg_arr.append(r_stim.loc[row['trial_ind']].values) 
                
                # replace with average values across runs
                pp_avg_stim_df.loc[cond_ind, :] = np.nanmean(np.stack(avg_arr), axis = 0)

                # replace DM with bar pos
                pp_reference_dm[cond_ind] = np.sum(np.stack((pp_dm['full_stim']['{sn}_{rn}'.format(sn = row['ses'],
                                                                                                rn = row['run'])][row['trial_ind']],
                                                            pp_dm['att_bar']['{sn}_{rn}'.format(sn = row['ses'],
                                                                                                rn = row['run'])][row['trial_ind']]), 
                                                            axis = 0), 
                                                axis = 0)
            else:
                pp_avg_stim_df.loc[cond_ind, :] = np.nan #0
                pp_reference_dm[cond_ind] = np.nan #0

        return pp_avg_stim_df, pp_reference_dm
    
    
        





