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


class Decoding_Model(GLMsingle_Model):

    def __init__(self, MRIObj, outputdir = None, pysub = 'hcp_999999', use_atlas = None):
        
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
        
        
    def load_prf_estimates(self, pRFModelObj = None, participant_list = [], ses = 'mean', run_type = 'mean', model_name = None, 
                                fit_hrf = False, rsq_threshold = .1, positive_rf = True, size_std = 2.5,
                                mask_bool_df = None, stim_on_screen = [], mask_arr = True):
        
        """
        Load prf estimates, obtained from fitting fsnative surface with prfpy.
        Returns dataframe with estimates for all participants in participant list
        """
        
        prf_sj_space = 'fsnative'
        
        ## load pRF estimates and models for all participants 
        print('Loading iterative estimates')
        group_estimates, group_prf_models = pRFModelObj.load_pRF_model_estimates(participant_list = participant_list,
                                                                    ses = ses, run_type = run_type, 
                                                                    model_name = model_name, 
                                                                    iterative = True,
                                                                    sj_space = prf_sj_space,
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
    
    def get_prf_vertex_index(self, pRFModelObj = None, participant_list = [], ses = 'mean', run_type = 'mean', model_name = None, 
                                    fit_hrf = False, rsq_threshold = .1, positive_rf = True, size_std = 2.5,
                                    mask_bool_df = None, stim_on_screen = [], mask_arr = True, num_vert = None):
        
        """get pRF vertex indices that are above certain rsq threshold
        to later use to subselect best fitting vertices within ROI
        (makes decoding faster) 
        """
        
        # first get prf estimates for participant list
        group_estimates_df, _ = self.load_prf_estimates(pRFModelObj = pRFModelObj, participant_list = participant_list, 
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
    
    def get_prf_stim_grid(self, pRFModelObj = None, participant = None, ses = 'mean', mask_bool_df = None, stim_on_screen = [], 
                            osf = 1, res_scaling = .1):
        
        """Get prf stimulus array and grid coordinates for participant
        """
        
        ## get stimulus array (time, y, x)
        prfpy_dm = pRFModelObj.get_DM(participant, 
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
        
        ## rotate all stim in such a way that we can average across
        # reference will be vertical bars with attended bar on the left visual field

        ## get trial indices
        ref_trl_ind = df_ecc_dist.query('x_pos < 0').trial_ind.values[0] # reference trial (stays the same)
        flip_hor_trl_ind = df_ecc_dist.query('x_pos > 0').trial_ind.values[0] # trial to be flipped horizontally (this is mirrored left and right)
        rot90_CCW_trl_ind = df_ecc_dist.query('y_pos > 0').trial_ind.values[0] # trial to be rotated 90deg CCW
        rot90_CW_trl_ind = df_ecc_dist.query('y_pos < 0').trial_ind.values[0] # trial to be rotated 90deg CW
        
        # if we want the flipped case, and conditions are not symmetrical
        if flipped_stim == True and sym_trial == False:
            ref_trl_ind = self.get_flipped_trial_ind(trl_ind = ref_trl_ind, DM_arr = DM_arr)
            flip_hor_trl_ind = self.get_flipped_trial_ind(trl_ind = flip_hor_trl_ind, DM_arr = DM_arr)
            rot90_CCW_trl_ind = self.get_flipped_trial_ind(trl_ind = rot90_CCW_trl_ind, DM_arr = DM_arr)
            rot90_CW_trl_ind = self.get_flipped_trial_ind(trl_ind = rot90_CW_trl_ind, DM_arr = DM_arr)

        ## stack stim
        average_stim = reconstructed_stimulus.stack('y').loc[ref_trl_ind].iloc[::-1, :].to_numpy()
        average_stim = np.stack((average_stim,
                                np.flip(reconstructed_stimulus.stack('y').loc[flip_hor_trl_ind].iloc[::-1, :].to_numpy(),
                                axis = 1)))
        average_stim = np.vstack((average_stim,
                                np.rot90(reconstructed_stimulus.stack('y').loc[rot90_CCW_trl_ind].iloc[::-1, :].to_numpy(),
                                axes=(0, 1))[np.newaxis, ...]))                    
        average_stim = np.vstack((average_stim,
                                np.rot90(reconstructed_stimulus.stack('y').loc[rot90_CW_trl_ind].iloc[::-1, :].to_numpy(),
                                axes=(1, 0))[np.newaxis, ...]))  
        
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
        
        # select trials with crossed bars at that ecc
        bar_ecc_df = position_df[(position_df['bar_ecc'] == bar_ecc) &\
                                (position_df['bars_pos'] == 'crossed')]
        
        ### a) stack trials where both bars at the same ecc
        if same_ecc:
        
            # first find the indices where both attended bar and unattended bar are at the same ecc
            trial_ind_same_ecc = np.array([val for val in bar_ecc_df[(bar_ecc_df['attend_condition'] == True)].trial_ind.values if val in bar_ecc_df[(bar_ecc_df['attend_condition'] == False)].trial_ind.values])

            # filter df for those trials (keep attended and unattended to collapse)
            df_cross_same_ecc = bar_ecc_df[(bar_ecc_df['trial_ind'].isin(trial_ind_same_ecc))]
            
            ## rotate all stim in such a way that we can average across
            # get dfs with main trial types
            
            cond_ind = []
            # reference trial and it's flipped case 
            # attended bar vertical, on the left. unattended bar horizontal, upper meridian
            ref_trl_df = df_cross_same_ecc[(df_cross_same_ecc['trial_ind'].isin(df_cross_same_ecc.query('x_pos < 0').trial_ind.values)) &\
                                            (df_cross_same_ecc['y_pos'] > 0)]
            cond_ind.append([ref_trl_df.query('~ attend_condition').trial_ind.values[0],
                            ref_trl_df.query('attend_condition').trial_ind.values[0]])

            # trial to be flipped horizontally (this is mirrored left and right) and its flipped case
            flip_hor_trl_df = df_cross_same_ecc[(df_cross_same_ecc['trial_ind'].isin(df_cross_same_ecc.query('x_pos > 0').trial_ind.values)) &\
                                                (df_cross_same_ecc['y_pos'] > 0)]
            cond_ind.append([flip_hor_trl_df.query('~ attend_condition').trial_ind.values[0],
                            flip_hor_trl_df.query('attend_condition').trial_ind.values[0]])
            
            # trial to be rotated 90deg CW and its flipped case
            rot90_CW_trl_df = df_cross_same_ecc[(df_cross_same_ecc['trial_ind'].isin(df_cross_same_ecc.query('y_pos < 0').trial_ind.values)) &\
                                                    (df_cross_same_ecc['x_pos'] < 0)]
            cond_ind.append([rot90_CW_trl_df.query('~ attend_condition').trial_ind.values[0],
                             rot90_CW_trl_df.query('attend_condition').trial_ind.values[0]])
            
            # trial to be diagonally mirrored (this is rotated 90deg CCW + flipped horizontally) and its flipped case
            rot90_CCW_flip_hor_trl_df = df_cross_same_ecc[(df_cross_same_ecc['trial_ind'].isin(df_cross_same_ecc.query('y_pos < 0').trial_ind.values)) &\
                                                    (df_cross_same_ecc['x_pos'] > 0)]
            cond_ind.append([rot90_CCW_flip_hor_trl_df.query('~ attend_condition').trial_ind.values[0],
                            rot90_CCW_flip_hor_trl_df.query('attend_condition').trial_ind.values[0]])
            
            cond_ind = np.vstack(cond_ind)
            
            ## stack stim
            # relative to reference stim
            average_stim = reconstructed_stimulus.stack('y').loc[cond_ind[:,0][0]].iloc[::-1, :].to_numpy()
            average_stim = np.stack((average_stim,
                                    self.get_diag_mirror_arr(reconstructed_stimulus.stack('y').loc[cond_ind[:,1][0]].iloc[::-1, :].to_numpy(), 
                                                            diag_type = 'major')
                                    ))
            # relative to flip-horizontal stim
            average_stim = np.vstack((average_stim,
                                    self.flip_arr(reconstructed_stimulus.stack('y').loc[cond_ind[:,0][1]].iloc[::-1, :].to_numpy(),
                                                flip_type='lr')[np.newaxis, ...]
                                    ))     
            average_stim = np.vstack((average_stim,
                                    self.get_diag_mirror_arr(self.flip_arr(reconstructed_stimulus.stack('y').loc[cond_ind[:,1][1]].iloc[::-1, :].to_numpy(),
                                                                                    flip_type='lr'), 
                                                            diag_type = 'major')[np.newaxis, ...]
                                    )) 
            # relative to 90deg CW stim
            average_stim = np.vstack((average_stim,
                                    np.rot90(reconstructed_stimulus.stack('y').loc[cond_ind[:,0][2]].iloc[::-1, :].to_numpy(),
                                            axes=(1, 0))[np.newaxis, ...]
                                    ))    
            average_stim = np.vstack((average_stim,
                                    self.get_diag_mirror_arr(np.rot90(reconstructed_stimulus.stack('y').loc[cond_ind[:,1][2]].iloc[::-1, :].to_numpy(),
                                                                                axes=(1, 0)), 
                                                            diag_type = 'major')[np.newaxis, ...]
                                    )) 
            
            # relative to 90deg CCW + flip horizontal stim
            average_stim = np.vstack((average_stim,
                                    self.get_diag_mirror_arr(reconstructed_stimulus.stack('y').loc[cond_ind[:,0][3]].iloc[::-1, :].to_numpy(), 
                                                            diag_type = 'minor')[np.newaxis, ...]
                                    )) 
            average_stim = np.vstack((average_stim,
                                    self.get_diag_mirror_arr(self.get_diag_mirror_arr(reconstructed_stimulus.stack('y').loc[cond_ind[:,1][3]].iloc[::-1, :].to_numpy(), 
                                                                                    diag_type = 'minor'), 
                                                            diag_type = 'major')[np.newaxis, ...]
                                    ))    
            
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
    
    
        
            
        
        
        

        
        
        
        
        
        