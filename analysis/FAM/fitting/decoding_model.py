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

from PIL import Image, ImageDraw

from FAM.fitting.model import Model
from FAM.fitting.glm_single_model import GLMsingle_Model

from glmsingle.glmsingle import GLM_single
from glmsingle.glmsingle import getcanonicalhrf

import time
import cortex
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

import nibabel as nib
import neuropythy

import braincoder
from braincoder.models import GaussianPRF2DWithHRF, GaussianPRF2D
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
        
    def get_prf_ROI_data(self, participant = None, roi_name = 'V1', index_arr = [], overwrite = False, file_ext = None):
        
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
                                            overwrite = overwrite)
        
        return masked_data_df
    
    def get_prf_stim_grid(self, participant = None, ses = 'mean', mask_bool_df = None, stim_on_screen = [], 
                            osf = 1, res_scaling = .1):
        
        """Get prf stimulus array and grid coordinates for participant
        """
        
        ## get stimulus array (time, y, x)
        prfpy_dm = self.pRFModelObj.get_DM(participant, 
                                    ses = ses, 
                                    mask_bool_df = mask_bool_df, 
                                    stim_on_screen = stim_on_screen,
                                    filename = None, 
                                    osf = osf, 
                                    res_scaling = res_scaling,
                                    transpose_dm = False)

        ## and swap positions to get (time, x, y)
        prf_stimulus_dm = np.rollaxis(prfpy_dm, 2, 1)
        
        ## get grid coordinates
        size = prf_stimulus_dm.shape[-1]

        y, x = np.meshgrid(np.linspace(-1, 1, size)[::-1] *(self.MRIObj.screen_res[0]/2), 
                            np.linspace(-1, 1, size) * (self.MRIObj.screen_res[0]/2))
        x_deg = self.convert_pix2dva(x.ravel())
        y_deg = self.convert_pix2dva(y.ravel())

        prf_grid_coordinates = pd.DataFrame({'x':x_deg, 'y': y_deg}).astype(np.float32)
        
        return prf_stimulus_dm, prf_grid_coordinates
    
    def make_df_run_bar_pos(self, run_df = None):
        
        """make data frame with bar positions and indices for each trial
        for a given run (in a summarized way)
        """
        
        ## find parallel + crossed bar trial indices
        parallel_bar_ind = np.where((run_df[run_df['attend_condition'] == 1].bar_pass_direction_at_TR.values[0] == run_df[run_df['attend_condition'] == 0].bar_pass_direction_at_TR.values[0]))[0]
        crossed_bar_ind = np.where((run_df[run_df['attend_condition'] == 1].bar_pass_direction_at_TR.values[0] != run_df[run_df['attend_condition'] == 0].bar_pass_direction_at_TR.values[0]))[0]

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
        inter_bar_dist = (position_df.iloc[inds_uatt].x_pos.values - position_df.iloc[inds_att].x_pos.values)/self.bar_width_pix[0]

        position_df.loc[inds_uatt,'inter_bar_dist'] = inter_bar_dist
        position_df.loc[inds_att,'inter_bar_dist'] = inter_bar_dist

        # for y
        inds_uatt = position_df[((position_df['attend_condition'] == 0) &\
                                (position_df['bars_pos'] == 'parallel') &\
                    ((position_df['y_pos'] != 0)))].sort_values('trial_ind').index
        inds_att = position_df[((position_df['attend_condition'] == 1) &\
                                (position_df['bars_pos'] == 'parallel') &\
                    ((position_df['y_pos'] != 0)))].sort_values('trial_ind').index
        inter_bar_dist = (position_df.iloc[inds_uatt].y_pos.values - position_df.iloc[inds_att].y_pos.values)/self.bar_width_pix[0]

        position_df.loc[inds_uatt,'inter_bar_dist'] = inter_bar_dist
        position_df.loc[inds_att,'inter_bar_dist'] = inter_bar_dist
        
        ## add bar eccentricity
        ecc_dict = {'far': self.bar_x_coords_pix[0::5], 'middle': self.bar_x_coords_pix[1::3], 'near': self.bar_x_coords_pix[2:4]}

        for ecc_key in ecc_dict.keys():
            inds = position_df[((position_df['x_pos'].isin(ecc_dict[ecc_key])) |\
                        (position_df['y_pos'].isin(ecc_dict[ecc_key])))].sort_values('trial_ind').index
            position_df.loc[inds,'bar_ecc'] = ecc_key
            
        ## also add absolute distance
        position_df.loc[:,'abs_inter_bar_dist'] = np.absolute(position_df.inter_bar_dist.values)
   
        return position_df
    
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
        
        return sns.heatmap(reconstructed_stimulus.stack('y').loc[frame].iloc[::-1, :], vmin = vmin, vmax = vmax, cmap = cmap,
                           xticklabels = xticklabels, yticklabels = yticklabels, square = square)
        
    def get_run_trial_pairs(self, DM_arr = None):
        
        """find trials where attended bar and unattended bar in same position
        to correlate with each other
        """
        n_trials = DM_arr.shape[0]
        same_bar_pos_ind = []

        for i in range(n_trials):

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
                trl_stim = reconstructed_stimulus.stack('y').loc[trl_ind].iloc[::-1, :].to_numpy()
            # trial to be flipped horizontally (this is mirrored left and right)
            elif not trl_df.query('x_pos > 0 & attend_condition').empty:
                # if we want the flipped case, and conditions are not symmetrical
                if flipped_stim == True and sym_trial == False:
                    trl_ind = self.get_flipped_trial_ind(trl_ind = ind, DM_arr = DM_arr)
                trl_stim = self.flip_arr(reconstructed_stimulus.stack('y').loc[trl_ind].iloc[::-1, :].to_numpy(),
                                         flip_type='lr')
            # trial to be rotated 90deg CCW
            elif not trl_df.query('y_pos > 0 & attend_condition').empty:
                # if we want the flipped case, and conditions are not symmetrical
                if flipped_stim == True and sym_trial == False:
                    trl_ind = self.get_flipped_trial_ind(trl_ind = ind, DM_arr = DM_arr)
                trl_stim = np.rot90(reconstructed_stimulus.stack('y').loc[trl_ind].iloc[::-1, :].to_numpy(),
                                    axes=(0, 1))
            # trial to be rotated 90deg CW
            elif not trl_df.query('y_pos < 0 & attend_condition').empty:
                # if we want the flipped case, and conditions are not symmetrical
                if flipped_stim == True and sym_trial == False:
                    trl_ind = self.get_flipped_trial_ind(trl_ind = ind, DM_arr = DM_arr)
                trl_stim = np.rot90(reconstructed_stimulus.stack('y').loc[trl_ind].iloc[::-1, :].to_numpy(),
                                    axes=(1, 0))
                
            average_stim.append(trl_stim)
        average_stim = np.stack(average_stim)
                        
        # and average (median)
        average_stim = np.median(average_stim, axis = 0)

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
                
                trl_stim = reconstructed_stimulus.stack('y').loc[ind].iloc[::-1, :].to_numpy()
                
                # trial to also be flipped vertically (this is mirrored up and down) 
                if not trl_df[(trl_df['y_pos'] < 0) & (trl_df['attend_condition'] == bool(np.abs(0 - flip_att_bool)))].empty:
                    trl_stim = self.flip_arr(trl_stim, flip_type='ud')
                
            # trial to be flipped horizontally (this is mirrored left and right) 
            elif not trl_df[(trl_df['x_pos'] > 0) & (trl_df['attend_condition'] == bool(np.abs(1 - flip_att_bool)))].empty:
                
                trl_stim = self.flip_arr(reconstructed_stimulus.stack('y').loc[ind].iloc[::-1, :].to_numpy(),
                                                flip_type='lr')
                
                # trial to also be flipped vertically (this is mirrored up and down) 
                if not trl_df[(trl_df['y_pos'] < 0) & (trl_df['attend_condition'] == bool(np.abs(0 - flip_att_bool)))].empty:
                    trl_stim = self.flip_arr(trl_stim, flip_type='ud')
                
            # trial to be rotated 90deg CW
            elif not trl_df[(trl_df['x_pos'] < 0) & (trl_df['attend_condition'] == bool(np.abs(0 - flip_att_bool)))].empty: 
                
                trl_stim = np.rot90(reconstructed_stimulus.stack('y').loc[ind].iloc[::-1, :].to_numpy(),
                                                        axes=(1, 0))
                
                if not trl_df[(trl_df['y_pos'] > 0) & (trl_df['attend_condition'] == bool(np.abs(1 - flip_att_bool)))].empty:
                    trl_stim = self.flip_arr(trl_stim, flip_type='lr')
            
            # trial to be rotated 90deg CCW + flip horizontally
            elif not trl_df[(trl_df['x_pos'] > 0) & (trl_df['attend_condition'] == bool(np.abs(0 - flip_att_bool)))].empty: 
                
                trl_stim = self.get_diag_mirror_arr(reconstructed_stimulus.stack('y').loc[ind].iloc[::-1, :].to_numpy(),
                                                        diag_type = 'minor')
                
                if not trl_df[(trl_df['y_pos'] > 0) & (trl_df['attend_condition'] == bool(np.abs(1 - flip_att_bool)))].empty:
                    trl_stim = np.rot90(reconstructed_stimulus.stack('y').loc[ind].iloc[::-1, :].to_numpy(),
                                                        axes=(0, 1))
            
            average_stim.append(trl_stim)
        average_stim = np.stack(average_stim)
                        
        # and average (median)
        average_stim = np.median(average_stim, axis = 0)

        # if we want the flipped trials for symmetrical cases
        if flipped_stim == True and same_ecc == True:
            average_stim = self.get_diag_mirror_arr(average_stim, diag_type = 'major')
            
        return average_stim
    
    def get_decoder_grid_coords(self):
        
        """Get grid coordinates for FA task, to use in decoder (8x8 grid)
        """
        new_y, new_x = np.meshgrid(np.flip(self.y_coords_deg), 
                                    self.x_coords_deg)
        fa_grid_coordinates = pd.DataFrame({'x':new_x.ravel(), 'y': new_y.ravel()}).astype(np.float32)
        
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
    
    def downsample_DM(self, DM_arr = None):
        
        """downsample FA FAM to 8x8 grid
        to correlate with reconstructed image
        """
        
        ## reduce dimensionality 
        downsample_FA_DM = scipy.ndimage.zoom(DM_arr, (1, 8/108., 8/108.))
        # binarize again
        downsample_FA_DM[downsample_FA_DM > .5] = 1
        downsample_FA_DM[downsample_FA_DM<.5] = 0
        
        return downsample_FA_DM
    
    def plot_avg_parallel_stim(self, average_stim = None, flip_average_stim = None, DM_trl_ind = None,
                                    bar_ecc = None, bar_dist = None, downsample_FA_DM = None, 
                                    vmin = 0, vmax = .4, cmap = 'magma', filename = None):
        
        """Make 4x4 plot of recontruscted stim, averaged across conditions
        (and also reverse case)
        followed by the corresponding downsampled DM for inspection
        """
        
        ## correlate each reconstructed average condition with stim position

        # note - here we transpose the DM array when correlating because the average_stim we calculated
        # has a different format than the reconstructed stim outputted by brain decoder
        avg_corr, avg_pval = scipy.stats.pearsonr(average_stim.ravel(), 
                                                  downsample_FA_DM[DM_trl_ind].T.ravel())
        flip_avg_corr, flip_avg_pval = scipy.stats.pearsonr(flip_average_stim.ravel(), 
                                                            downsample_FA_DM[DM_trl_ind].T.ravel())
                
        bar_ecc_ind = {'far': 1, 'middle': 2, 'near': 3}
        
        # plot figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (10,10))

        ## attended leftmost
        sns.heatmap(average_stim, cmap = cmap, ax = axes[0][0], square = True, cbar = False,
                    annot=True, annot_kws={"size": 7},
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
                annot=True, annot_kws={"size": 7},
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
                                    vmin = 0, vmax = .4, cmap = 'magma', filename = None):
        
        """Make 4x4 plot of recontruscted stim, averaged across conditions
        (and also reverse case)
        followed by the corresponding downsampled DM for inspection
        """
        
        ## correlate each reconstructed average condition with stim position

        # note - here we transpose the DM array when correlating because the average_stim we calculated
        # has a different format than the reconstructed stim outputted by brain decoder
        avg_corr, avg_pval = scipy.stats.pearsonr(average_stim.ravel(), 
                                                  downsample_FA_DM[DM_trl_ind].T.ravel())
        flip_avg_corr, flip_avg_pval = scipy.stats.pearsonr(flip_average_stim.ravel(), 
                                                            downsample_FA_DM[DM_trl_ind].T.ravel())
                
        bar_ecc_ind = {'far': 1, 'middle': 2, 'near': 3}
        
        # make reference dict with unique conditions of attended and unattend
        uniq_cond_dict = {'far': 'near', 'near': 'middle', 'middle': 'far'}

        # bar ecc list of attended and unattended bar
        bar_ecc_list = [bar_ecc, bar_ecc] if same_ecc == True else [bar_ecc, uniq_cond_dict[bar_ecc]]
        
        # plot figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (10,10))

        ## attended leftmost, vertical
        sns.heatmap(average_stim, cmap = cmap, ax = axes[0][0], square = True, cbar = False,
                    annot=True, annot_kws={"size": 7},
                    vmin = vmin, vmax = vmax)
        axes[0][0].set_title('attend %s ecc, unattend %s ecc'%(bar_ecc_list[0], bar_ecc_list[1]))

        ## DM
        axes[1][0].imshow(downsample_FA_DM[DM_trl_ind].T, cmap = 'binary_r', vmax = 1.5)
        # Add the patch to the Axes
        axes[1][0].add_patch(patches.Rectangle((bar_ecc_ind[bar_ecc_list[0]] - .5, -.5), 1, 8, 
                                            linewidth = 2, edgecolor='purple', 
                                            facecolor='purple', hatch = '///'))
        # annotate correlation between reconstructed stim and DM
        axes[1][0].set_title(r"$\rho$ = {r}".format(r = '%.2f'%(avg_corr))+\
                                '   pval = {p}'.format(p = "{:.2e}".format(avg_pval)))

        ## attended rightmost
        sns.heatmap(flip_average_stim, cmap = 'magma', ax = axes[0][1], square = True, cbar = False,
                annot=True, annot_kws={"size": 7},
                vmin = vmin, vmax = vmax)
        axes[0][1].set_title('flipped case')

        ## DM
        axes[1][1].imshow(downsample_FA_DM[DM_trl_ind].T, cmap = 'binary_r', vmax = 1.5)
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
            trl_ind = masked_df[masked_df.duplicated(subset=['trial_ind'])].trial_ind.values[0]
            
        elif bars_pos == 'parallel':
            
            # filter bar position df
            df_ecc_dist = self.get_trl_ecc_dist_df(position_df = position_df, bars_pos = bars_pos, 
                                                bar_ecc = bar_ecc, abs_inter_bar_dist = bar_dist)
            
            # get trial ind
            trl_ind = df_ecc_dist.query('x_pos < 0').trial_ind.values[0]
            
        return trl_ind
    
    def fit_decoder(self, participant_list = [], ROI_list = ['V1'], overwrite = False, model_type = 'gauss_hrf',
                        prf_file_ext = '_cropped_dc_psc.nii.gz', ses = 'mean', fa_file_ext = '_cropped.nii.gz',
                        mask_bool_df = None, stim_on_screen = [], group_bar_pos_df = []):
        
        """
        Fit decoder across participants
        and ROIs
        """
        
        # iterate over participants
        for pp in participant_list:
            # iterate over ROIs
            for roi_name in ROI_list:
                self.decode_ROI(participant = pp, roi_name = roi_name, overwrite = overwrite, prf_file_ext = prf_file_ext, 
                                ses = ses, mask_bool_df = mask_bool_df, stim_on_screen = stim_on_screen,
                                model_type = model_type, fa_file_ext = fa_file_ext, 
                                pp_bar_pos_df = group_bar_pos_df['sub-{sj}'.format(sj = pp)])
                       
    def decode_ROI(self, participant = None, roi_name = 'V1', overwrite = False, model_type = 'gauss_hrf',
                        prf_file_ext = '_cropped_dc_psc.nii.gz', fa_file_ext = '_cropped.nii.gz', ses = 'mean',
                        mask_bool_df = None, stim_on_screen = [], save_estimates = True, pp_bar_pos_df = None):
        
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
                                                    overwrite = overwrite, 
                                                    file_ext = prf_file_ext)
    
        # get prf stimulus DM and grid coordinates
        prf_stimulus_dm, prf_grid_coordinates = self.get_prf_stim_grid(participant = participant, 
                                                                        ses = ses, 
                                                                        mask_bool_df = mask_bool_df, 
                                                                        stim_on_screen = stim_on_screen)
        
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
                                           sd_lim = [0.3, 8], n_vox = 300)
        
        ## fit residuals on prf data
        omega, dof  = self.fit_residuals(model = prf_decoder_model, 
                                        data = prf_masked_data_df, 
                                        paradigm = prf_stimulus_dm, 
                                        parameters = pars_gd, 
                                        fit_method = 't', 
                                        best_vox = best_voxels,
                                        filename = pars_filename.replace('_pars.h5', '_resid.npz'))
        
        ## now get masked FA ROI data, all runs
        masked_FAdata_dict = self.get_FA_ROI_data(participant = participant, 
                                                roi_name = roi_name, 
                                                index_arr = [], 
                                                overwrite = overwrite, 
                                                file_ext = fa_file_ext,
                                                glmsingle_model = 'D', 
                                                trial_num = 132)
        ## get FA DM and grid coordinates (8x8)
        FA_DM_dict, fa_grid_coordinates = self.get_FA_stim_grid(participant = participant, 
                                                                ses = ses, 
                                                                pp_bar_pos_df = pp_bar_pos_df,
                                                                file_ext = fa_file_ext,
                                                                glmsingle_model = 'D', 
                                                                trial_num = 132)
        
        ## decode over runs
        # save reconstructed stim as HDF5 file, to later load
        decoded_stim_filename = op.join(pp_outdir, 
                            'sub-{sj}_task-FA_ROI-{rname}_model-{dmod}_reconstructed_stim.h5'.format(sj = participant,
                                                                                                    rname = roi_name,
                                                                                                    dmod = model_type))
        # make filename generic to save per run
        decoded_stim_filename = decoded_stim_filename.replace('_task', '_{snrnkey}_task') 
        
        reconstructed_stim_dict = {}
        lowres_DM_dict = {}
        for ind, df_key in enumerate(masked_FAdata_dict):
    
            print('decoding data from %s'%df_key)
            
            # get run ROI dataframe
            masked_data_df = masked_FAdata_dict[df_key]

            # get reconstructed stim as df
            reconstructed_stimulus = self.decode_FA_stim(data = masked_data_df, 
                                                        grid_coordinates = fa_grid_coordinates,  
                                                        parameters = pars_gd,
                                                        omega = omega, 
                                                        dof = dof,
                                                        best_voxels = best_voxels, 
                                                        filename = decoded_stim_filename.format(snrnkey = df_key))
            # append to dict
            reconstructed_stim_dict[df_key] = reconstructed_stimulus
            
            ## downsample DM to check correlation
            file_rn, file_sn = self.MRIObj.mri_utils.get_run_ses_from_str(df_key)
            
            print('downsampling FA DM for ses-{sn}, run-{rn}'.format(sn = file_sn, rn=file_rn))
            lowres_DM = self.downsample_DM(DM_arr = FA_DM_dict['r{rn}s{sn}'.format(sn = file_sn, 
                                                                                   rn = file_rn)]['full_stim'])
            
            # correlate reconstructed stim with downsampled DM
            corr, pval = scipy.stats.pearsonr(reconstructed_stimulus.values.ravel(), 
                                            lowres_DM.ravel())
            print('correlation between reconstructed stim and DM is %.2f, %.2f'%(corr, pval))
            
            # append to dict
            lowres_DM_dict[df_key] = lowres_DM
            
            ## save run position df (for convenience)
            #print('making df with bar position info for ses-{s}, run-{r}'. format(s = file_sn, r=file_rn)) 
            #run_position_df = self.make_df_run_bar_pos(run_df = pp_bar_pos_df['ses-{s}'.format(s = file_sn)]['run-{r}'.format(r=file_rn)])
        
        return lowres_DM_dict, reconstructed_stim_dict
               
    def decode_FA_stim(self, data = None, grid_coordinates = None,  parameters = None, omega = None, dof = None,
                            best_voxels = None, filename = None):
        
        """Decode FA betas for a given run
        and save reconstructed stim as hdf5 file
        """
        
        print('using params from %i best voxels'%(len(best_voxels)))

        # if there is a pars file already, just load it
        if filename is not None and op.isfile(filename):
            print('Loading reconstructed stim from %s'%filename)
            reconstructed_stimulus = pd.read_hdf(filename).astype(np.float32)  
        else:
            model_single_trial = GaussianPRF2D(grid_coordinates = grid_coordinates,
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
                                                    min_n_iterations=200, 
                                                    max_n_iterations=5000, 
                                                    learning_rate = .01,
                                                    l2_norm = .01)
            
            if filename is not None:
                print('Saving reconstructed stim in %s'%filename)
                reconstructed_stimulus.to_hdf(filename, key='df_stim', mode='w', index = False)  
        
        return reconstructed_stimulus
         
    def get_FA_stim_grid(self, participant = None, ses = 'mean', pp_bar_pos_df = None,
                                glmsingle_model = 'D', file_ext = '_cropped.nii.gz', trial_num = 132):
        
        """Get participant FA DM + grid coordinates that will be used in decoder
        """
        
        ## convert betas estimates into volume images (if not done so already)
        # and get filenames
        betas_filelist = self.convert_betas_volume(participant, model_type = glmsingle_model, 
                                                    file_ext = file_ext, trial_num = trial_num)

        ## get FA DM
        FA_DM_dict, _ = self.get_visual_DM_dict(participant, 
                                            filelist = betas_filelist, 
                                            pp_bar_pos_df = pp_bar_pos_df)
        
        # need to downsample stimulus space
        # to actually be able to fit on CPU

        ## get grid coordinates (8x8)
        fa_grid_coordinates = self.get_decoder_grid_coords()
        
        return FA_DM_dict, fa_grid_coordinates
        
    def get_FA_ROI_data(self, participant = None, roi_name = 'V1', index_arr = [], overwrite = False,
                            glmsingle_model = 'D', file_ext = '_cropped.nii.gz', trial_num = 132):
        
        """Get FA data (beta values) for the ROI of a participant, for all runs,
        and return dataframe in a format compatible with braindecoder 
        """
        
        ## convert betas estimates into volume images (if not done so already)
        # and get filenames
        betas_filelist = self.convert_betas_volume(participant, model_type = glmsingle_model, 
                                                    file_ext = file_ext, trial_num = trial_num)

        ## get masked ROI data file list
        masked_FAdata_df_filelist = self.get_ROImask_data(participant, 
                                                        file_list = betas_filelist, 
                                                        task = 'FA', 
                                                        run_type = 'all', ses = 'all', 
                                                        roi_name = roi_name, 
                                                        index_arr = index_arr,
                                                        overwrite = overwrite)
        
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
          
    def fit_residuals(self, model = None, data = None, paradigm = None, parameters = None, fit_method = 't', 
                            best_vox = None, filename = None):
        
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
            omega, dof = resid_fitter.fit(method=fit_method)
            
            if filename is not None:
                print('Storing omega, dof and best voxel indices in %s'%filename)
                np.savez(filename, omega=omega, dof=[dof], best_voxels = best_vox)
        
        return omega, dof 
                
    def get_best_voxels(self, pars_gd = None, r2_gd = None,  sd_lim = [0.3, 8], n_vox = 300):
        
        """
        Get best voxels to then use in fitter
        """   
        
        # sort indices according to r2 values
        sort_ind = r2_gd.sort_values(ascending=False).index
        
        # mask for pRF SD (not too small or big)
        masked_voxels = pars_gd[(pars_gd['sd'] > .3) & (pars_gd['sd'] < 6.5)].index
        
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
        
        ## set prf model 
        # first without hrf
        prf_decoder_model = self.setup_prf_model(data = data, grid_coordinates = grid_coordinates, 
                                                model_type = model_type,
                                                paradigm = paradigm, 
                                                fit_hrf = False)    
        
        # if there is a pars file already, just load it
        if filename is not None and op.isfile(filename):
            print('Loading pRF fit parameters stored in %s'%filename)
            output_pars = pd.read_hdf(filename).astype(np.float32)  
            
            # if we fitted the hrf, then reload prf model accordingly
            if 'hrf' in model_type:
                prf_decoder_model = self.setup_prf_model(data = data, grid_coordinates = grid_coordinates, 
                                                model_type = model_type,
                                                paradigm = paradigm, 
                                                fit_hrf = True) 
        else:
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
            # ## plot diagnostic figure 
            # self.plot_prf_diagnostics(pars_grid = grid_pars, pars_gd = pars_gd, pars_fitter = par_fitter,
            #                             par_keys=['ols', 'gd'], figurename = None)
            
            ## now fit the hrf, if such is the case
            if 'hrf' in model_type:
                # redefine model
                prf_decoder_model = self.setup_prf_model(data = data, grid_coordinates = grid_coordinates, 
                                                model_type = model_type,
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
        
            if filename is not None:
                print('Saving pRF fit parameters in %s'%filename)
                output_pars.to_hdf(filename, key='df_pars', mode='w', index = False)  
        
        return prf_decoder_model, output_pars
    
    def load_encoding_model_pars(self, participant = None, task = 'pRF', roi_name = 'V1', model_type = 'gauss_hrf'):
        
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
                                    data_keys = ['ses-1_run-1']):
        
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
        # make filename generic to load per run
        decoded_stim_filename = decoded_stim_filename.replace('_task', '_{snrnkey}_task') 
        
        ## iterate over runs
        reconstructed_stim_dict = {}

        for ind, df_key in enumerate(data_keys):
            
            # get run number and session, to avoid mistakes 
            file_rn, file_sn = self.MRIObj.mri_utils.get_run_ses_from_str(df_key)
            
            # load stim
            reconstructed_stimulus = pd.read_hdf(decoded_stim_filename.format(snrnkey = df_key))
            
            # append to dict
            reconstructed_stim_dict[df_key] = reconstructed_stimulus
            
        return reconstructed_stim_dict
    
    def get_lowresDM_dict(self, DM_dict = None, data_keys = ['ses-1_run-1']):
        
        """downsample FA DM and append to dicts, for later plotting
        """
    
        ## iterate over runs
        lowres_DM_dict = {'full_stim': {}, 'att_bar': {}, 'unatt_bar': {}}

        for ind, df_key in enumerate(data_keys):
            
            # get run number and session, to avoid mistakes 
            file_rn, file_sn = self.MRIObj.mri_utils.get_run_ses_from_str(df_key)
            
            # downsample DM and append to dicts, for later plotting
            lowres_DM_dict['full_stim'][df_key] = self.downsample_DM(DM_arr = DM_dict['r{rn}s{sn}'.format(sn = file_sn, 
                                                                                    rn = file_rn)]['full_stim'])
            lowres_DM_dict['att_bar'][df_key] = self.downsample_DM(DM_arr = DM_dict['r{rn}s{sn}'.format(sn = file_sn, 
                                                                                    rn = file_rn)]['att_bar'])
            lowres_DM_dict['unatt_bar'][df_key] = self.downsample_DM(DM_arr = DM_dict['r{rn}s{sn}'.format(sn = file_sn, 
                                                                                    rn = file_rn)]['unatt_bar'])
                    
        return lowres_DM_dict
    
    def get_same_bar_pos_ind_dict(self, lowresDM_dict = None, data_keys = ['ses-1_run-1']):
        
        """find trials where attended bar and unattended bar in same position
        """
        
        same_bar_pos_ind_dict = {}
        
        for ind, df_key in enumerate(data_keys):
            
            same_bar_pos_ind_dict[df_key] = self.get_run_trial_pairs(DM_arr = lowresDM_dict['full_stim'][df_key])
            
        return same_bar_pos_ind_dict
    
    def get_run_position_df_dict(self, pp_bar_pos_df = None,  data_keys = ['ses-1_run-1']):
        
        """make data frame with bar positions and indices in trial
        """
        
        run_position_df_dict = {}

        for ind, df_key in enumerate(data_keys):
            
            print('making df with bar position info for %s'%df_key)
            # get run number and session, to avoid mistakes 
            file_rn, file_sn = self.MRIObj.mri_utils.get_run_ses_from_str(df_key)
            
            run_position_df_dict[df_key] = self.make_df_run_bar_pos(run_df = pp_bar_pos_df['ses-{s}'.format(s = file_sn)]['run-{r}'.format(r=file_rn)])

        return run_position_df_dict
        
    def get_pp_average_stim_dict(self, reconstructed_stim_dict = None, run_position_df_dict = None, lowres_DM_dict = None, bar_type = 'parallel'):
        
        """Get average reconstructed stim dict for a participant
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
                    
                    ## average over runs
                    average_stim_all = []
                    flip_average_stim_all = []
                    
                    ## stack all runs
                    for snrn_key, snrn_stim in reconstructed_stim_dict.items():
                        
                        average_stim_all.append(self.get_parallel_average_stim(reconstructed_stimulus = snrn_stim, 
                                                                            position_df = run_position_df_dict[snrn_key], 
                                                                            bar_ecc = bar_ecc, 
                                                                            abs_inter_bar_dist = bar_dist, 
                                                                            flipped_stim = False, 
                                                                            DM_arr = lowres_DM_dict['full_stim'][snrn_key]))

                        # also get average flipped case
                        flip_average_stim_all.append(self.get_parallel_average_stim(reconstructed_stimulus = snrn_stim, 
                                                                            position_df = run_position_df_dict[snrn_key], 
                                                                            bar_ecc = bar_ecc, 
                                                                            abs_inter_bar_dist = bar_dist, 
                                                                            flipped_stim = True, 
                                                                            DM_arr = lowres_DM_dict['full_stim'][snrn_key]))
                    average_stim_all = np.stack(average_stim_all)
                    flip_average_stim_all = np.stack(flip_average_stim_all)
                    
                    ## average over runs
                    average_stim = np.mean(average_stim_all, axis = 0)
                    flip_average_stim = np.mean(flip_average_stim_all, axis = 0)
                    
                    # save in dict
                    average_stim_dict[bar_ecc][bar_dist] = average_stim
                    flip_average_stim_dict[bar_ecc][bar_dist] = flip_average_stim
                    
        elif bar_type == 'crossed':
            
            uniq_cond_dict = {'far': 'near', 'middle': 'far', 'near': 'middle'}
            
            # iterate over ecc
            for bar_ecc in uniq_cond_dict.keys():
                
                average_stim_dict[bar_ecc] = {}
                flip_average_stim_dict[bar_ecc] = {}
                
                # and if crossed bars where equidistant to center or not
                for same_ecc in [True, False]:
                    
                    ## average over runs
                    average_stim_all = []
                    flip_average_stim_all = []
                    
                    ## stack all runs
                    for snrn_key, snrn_stim in reconstructed_stim_dict.items():
                        
                        average_stim_all.append(self.get_crossed_average_stim(reconstructed_stimulus = snrn_stim, 
                                                                            position_df = run_position_df_dict[snrn_key], 
                                                                            bar_ecc = bar_ecc, 
                                                                            same_ecc = same_ecc,  
                                                                            flipped_stim = False, 
                                                                            DM_arr = lowres_DM_dict['full_stim'][snrn_key]))

                        # also get average flipped case
                        flip_average_stim_all.append(self.get_crossed_average_stim(reconstructed_stimulus = snrn_stim, 
                                                                            position_df = run_position_df_dict[snrn_key], 
                                                                            bar_ecc = bar_ecc, 
                                                                            same_ecc = same_ecc, 
                                                                            flipped_stim = True, 
                                                                            DM_arr = lowres_DM_dict['full_stim'][snrn_key]))
                    average_stim_all = np.stack(average_stim_all)
                    flip_average_stim_all = np.stack(flip_average_stim_all)
                    
                    ## average over runs
                    average_stim = np.mean(average_stim_all, axis = 0)
                    flip_average_stim = np.mean(flip_average_stim_all, axis = 0)
                
                    # save in dict
                    average_stim_dict[bar_ecc][int(same_ecc)] = average_stim
                    flip_average_stim_dict[bar_ecc][int(same_ecc)] = flip_average_stim
                    
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
        
        return prf_decoder_model
        
                      
    def plot_prf_diagnostics(self, pars_grid = None, pars_gd = None, pars_fitter = None, 
                                prf_decoder_model = None, data = None, 
                                par_keys=['ols', 'gd'], figurename = None):
        
        """
        plot some diagnostic figures
        """
        
        ## compare r2 of grid and iterative (or iterative and hrf) 
        r2_ols = pars_fitter.get_rsq(pars_grid)
        r2_gd = pars_fitter.get_rsq(pars_gd)
        r2_both = pd.concat((r2_ols, r2_gd), keys=['r2_%s'%par_keys[0], 
                                                   'r2_%s'%par_keys[1]], axis=1)

        sns.relplot(x='r2_%s'%par_keys[0], y='r2_%s'%par_keys[1], 
                    data=r2_both.reset_index(), kind='scatter')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.show()
        
        ## Let's plot the location of gradient descent PRFs
        sns.relplot(x='x', y='y', hue='r2', data=pars_gd.join(r2_gd.to_frame('r2')), 
                    size='sd', sizes=(10, 100), palette='viridis')
        plt.title('PRF locations')
        plt.show()
        
        ## plot a few voxels with highest r2 improvement
        improvement = r2_gd - r2_ols
        largest_improvements = improvement.sort_values(ascending=False).index[:9]
        print(improvement.sort_values(ascending=False).loc[largest_improvements])

        pred_grid = prf_decoder_model.predict(parameters=pars_grid)
        pred_gd = prf_decoder_model.predict(parameters=pars_gd)

        pred = pd.concat((data.loc[:, largest_improvements], 
                        pred_grid.loc[:, largest_improvements], 
                        pred_gd.loc[:, largest_improvements]), axis=1, 
                        keys=['data', par_keys[0], par_keys[1]], names=['model'])

        #
        tmp = pred.stack(['model', 'source']).to_frame('value')
        sns.relplot(x='level_0', y='value', hue='model', col='source', data=tmp.reset_index(), kind='line', col_wrap=3,
                palette = sns.color_palette(['black', 'orange', 'green'], 3))
        plt.show()      
                
        
        