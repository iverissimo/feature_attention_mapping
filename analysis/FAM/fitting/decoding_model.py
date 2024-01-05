import numpy as np
import re
import os
import os.path as op
import pandas as pd
import yaml
import glob

import itertools
from scipy.interpolate import pchip

from PIL import Image, ImageDraw

from FAM.fitting.model import Model
from FAM.fitting.glm_single_model import GLMsingle_Model

from glmsingle.glmsingle import GLM_single
from glmsingle.glmsingle import getcanonicalhrf

import time
import cortex
import matplotlib.pyplot as plt

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
    
    
        
        
        
        

        
        
        
        
        
        