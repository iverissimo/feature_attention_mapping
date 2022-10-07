import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
import utils
import glob

import ptitprince as pt # raincloud plots
import matplotlib.patches as mpatches
from  matplotlib.ticker import FuncFormatter

from PIL import Image, ImageDraw

import cortex

import subprocess

from scipy.optimize import LinearConstraint, NonlinearConstraint

from FAM.utils import mri as mri_utils
from FAM.processing import preproc_behdata

from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter, Norm_Iso2DGaussianFitter, DoG_Iso2DGaussianFitter

from FAM.fitting.prf_model import pRF_model

class FA_model:

    def __init__(self, MRIObj, outputdir = None):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
            
        """

        ## set data object to use later on
        # has relevant paths etc
        self.MRIObj = MRIObj

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.MRIObj.derivatives_pth,'pRF_fit')
        else:
            self.outputdir = outputdir
            
        ### some relevant params ###

        ## bar width ratio
        self.bar_width = self.MRIObj.params['FA']['bar_width_ratio'] 

        ## screen resolution in pix
        screen_res = self.MRIObj.params['window']['size']
        if self.MRIObj.params['window']['display'] == 'square': # if square display
            screen_res = np.array([screen_res[1], screen_res[1]])
        self.screen_res = screen_res
        
        ## type of model to fit
        self.model_type = self.MRIObj.params['mri']['fitting']['FA']['fit_model']
        # type of prf model estimates to use
        self.prf_model_type = self.MRIObj.params['mri']['fitting']['pRF']['fit_model']

        ## type of optimizer to use
        self.optimizer = self.MRIObj.params['mri']['fitting']['FA']['optimizer']

        # if we are fitting HRF params
        self.fit_hrf = self.MRIObj.params['mri']['fitting']['pRF']['fit_hrf']
        
        ## if we're shifting TRs to account for dummy scans
        self.shift_TRs_num =  self.MRIObj.params['mri']['shift_DM_TRs']

        ## if we're cropping TRs
        self.crop_TRs = self.MRIObj.params['FA']['crop'] 
        self.crop_TRs_num =  self.MRIObj.params['FA']['crop_TR']

        ## if we did slicetime correction
        self.stc = self.MRIObj.params['mri']['slicetimecorrection']

        # if we did stc, then we need to hrf onset
        if self.stc:
            self.hrf_onset = -self.MRIObj.TR/2
        else:
            self.hrf_onset = 0

        ## if we want to oversample when fitting
        self.osf = 10 # oversampling factor

        ## scaling factor for DM spatial resolution
        self.res_scaling = 0.1

        # task sampling rate
        self.FA_sampling_rate = self.MRIObj.TR

        ## if we want to keep the model baseline fixed a 0
        self.fix_bold_baseline = self.MRIObj.params['mri']['fitting']['FA']['fix_bold_baseline'] 

        ## if we want to correct bold baseline of data
        self.correct_baseline = self.MRIObj.params['mri']['fitting']['FA']['correct_baseline'] 
        # number of TRs to use for correction
        self.corr_base_TRs = self.MRIObj.params['mri']['fitting']['FA']['num_baseline_TRs'] 

        ## total number of chunks we divide data when fitting
        self.total_chunks = self.MRIObj.params['mri']['fitting']['FA']['total_chunks'][self.MRIObj.sj_space]


    def fit_data(self, participant, pp_models, ses = None,
                    run_type = None, chunk_num = None, vertex = None, ROI = None,
                    model2fit = 'gauss', file_ext = '_cropped_confound_psc.npy', 
                    outdir = None, save_estimates = False,
                    xtol = 1e-3, ftol = 1e-4, n_jobs = 16):

        """
        fit inputted FA models to each participant in participant list
                
        Parameters
        ----------
        participant: str
            participant ID
        run_type: string or int
            type of run to fit, mean (default), or if int will do single run fit
        file_ext: dict
            file extension, to select appropriate files
        """  

        ## get list of files to load
        bold_filelist = pRF_model.get_bold_file_list(participant, input_list = None, task = 'FA', 
                                                    ses = ses, file_ext = file_ext, MRIObj = self.MRIObj)

        ## Load data array
        data = self.get_data4fitting(bold_filelist, run_type = run_type, chunk_num = chunk_num, vertex = vertex, ses = ses)

        return data



    def get_data4fitting(self, file_list, run_type = 'loo_r1s1', chunk_num = None, vertex = None, ses = 1):

        """
        load data from file list
                
        Parameters
        ----------
        file_list: list
            list with files names convert into data array
        run_type: string or int
            type of run to fit (str) or if int will do single run fit
        chunk_num: int or None
            if we want to fit specific chunk of data, then will return chunk array
        vertex: int, or list of indices or None
            if we want to fit specific vertex of data, or list of vertices (from an ROI for example) then will return vertex array
        """  
        
        if isinstance(run_type, str) and 'loo_' in run_type:
            _, train_file_list = mri_utils.get_loo_filename(file_list, loo_key=run_type)

            data2fit = np.array([])

            for file in train_file_list:
                print('Loading %s'%file)
                run_num, ses_num = mri_utils.get_run_ses_from_str(file)

                ## Load data array for run and session
                data = pRF_model.get_data4fitting(file_list, run_type = run_num, chunk_num = chunk_num, vertex = vertex, 
                                            total_chunks = self.total_chunks, num_baseline_TRs = self.corr_base_TRs,
                                            MRIObj = self.MRIObj, shift_TRs_num = self.shift_TRs_num, crop_TRs_num = self.crop_TRs_num,
                                            correct_baseline = self.correct_baseline, baseline_interval = 'empty', ses = ses_num)[np.newaxis,...]

                ## STACK
                data2fit = np.vstack([data2fit, data]) if data2fit.size else data

        else:
            data2fit = pRF_model.get_data4fitting(file_list, run_type = run_type, chunk_num = chunk_num, vertex = vertex, 
                                            total_chunks = self.total_chunks, num_baseline_TRs = self.corr_base_TRs,
                                            MRIObj = self.MRIObj, shift_TRs_num = self.shift_TRs_num, crop_TRs_num = self.crop_TRs_num,
                                            correct_baseline = self.correct_baseline, baseline_interval = 'empty', ses = ses)[np.newaxis,...]

        return data2fit

        ## need to make list of runs to load
        # and loop or parallel stack by calling prf model get data func, that will load each run individually
        # by also performing shifting and baseline correction
        # should be careful when loading a vertex or chunk, want to be sure that loading same for all runs
        # also check loo code for leave-one-out -> there are several sessions per participant, need to take that into account if combining them
