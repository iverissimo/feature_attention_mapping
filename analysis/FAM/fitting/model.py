import numpy as np
import re
import os
import os.path as op
import pandas as pd

import glob

from FAM.utils import mri as mri_utils
from FAM.processing import preproc_behdata


class Model:

    def __init__(self, MRIObj, outputdir = None, pysub = 'hcp_999999'):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
        outputdir: str
            absolute path to save fits
        tasks: list
            list with task ID names (prf and feature)
            
        """

        ## set data object to use later on
        # has relevant paths etc
        self.MRIObj = MRIObj

        self.pysub = pysub

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.MRIObj.derivatives_pth,'plots')
        else:
            self.outputdir = outputdir
            
        ### some relevant params ###

        # set fit folder name
        self.fitfolder = {self.MRIObj.params['mri']['fitting'][key]['fit_folder'] for key in self.MRIObj.tasks}
        
        ## type of model to fit
        self.model_type = {key:self.MRIObj.params['mri']['fitting'][key]['fit_model'] for key in self.MRIObj.tasks}
        
        ## type of optimizer to use
        self.optimizer = {key:self.MRIObj.params['mri']['fitting'][key]['optimizer'] for key in self.MRIObj.tasks}

        # if we are fitting HRF params
        self.fit_hrf = self.MRIObj.params['mri']['fitting']['pRF']['fit_hrf']

        # if we did stc, then we need to hrf onset
        if self.MRIObj.stc:
            self.hrf_onset = -self.MRIObj.TR/2
        else:
            self.hrf_onset = 0

        ## if we want to oversample when fitting
        self.osf = 10

        ## scaling factor for DM spatial resolution
        self.res_scaling = 0.1

        ## if we want to keep the model baseline fixed a 0
        self.fix_bold_baseline = {key:self.MRIObj.params['mri']['fitting'][key]['fix_bold_baseline'] for key in self.MRIObj.tasks}

        ## if we want to correct bold baseline of data
        self.correct_baseline = {key:self.MRIObj.params['mri']['fitting'][key]['correct_baseline'] for key in self.MRIObj.tasks}
        # number of TRs to use for correction
        self.corr_base_TRs = {key:self.MRIObj.params['mri']['fitting'][key]['num_baseline_TRs'] for key in self.MRIObj.tasks}

        ## total number of chunks we divide data when fitting
        self.total_chunks = {key:self.MRIObj.params['mri']['fitting'][key]['total_chunks'][self.MRIObj.sj_space] for key in self.MRIObj.tasks}

    def get_data4fitting(self, file_list, task = 'pRF', run_type = 'mean',
                            chunk_num = None, vertex = None,
                            baseline_interval = 'empty_long', ses = 1, return_filenames = False):

        """
        load data from file list
        will subselect files from filelist depending on run/session/task at hand
                
        Parameters
        ----------
        file_list: list
            list with files to combine into unique data array
        task: str
            task name for files in list (default pRF)
        run_type: string or int
            type of run to fit - mean (default), median, loo_rXsY (leaving out specific run and session) 
            or if int/'run-X' will do single run fit
        chunk_num: int or None
            if we want to fit specific chunk of data, then will return chunk array
        vertex: int, or list of indices or None
            if we want to fit specific vertex of data, or list of vertices (from an ROI for example) then will return vertex array
        baseline_interval: str
            name of baseline interval to use (default 'empty_long')
        ses: int
            session number, only relevant when loading one specific run number (associated to a session number)
        return_filenames: bool
            If we want to also return filenames in the same order of data array rows, or not (default)

        """  

        # if loading specific run, select it
        if isinstance(run_type, int) or (isinstance(run_type, str) and 'loo_' not in run_type and len(re.findall(r'\d{1,10}', run_type))>0):

            run = re.findall(r'\d{1,10}', str(run_type))[0]
            print('Loading run-{r} from ses-{s}'.format(r = run, s = ses))

            file_list = [file for file in file_list if 'run-{r}'.format(r = run) in file and 'ses-{s}'.format(s = ses) in file]
            
        # if leaving one run out
        elif 'loo_' in run_type:
            
            print('Leave-one out runs ({r})'.format(r = run_type))
            _, file_list = mri_utils.get_loo_filename(file_list, loo_key=run_type)

        ## now actually load data
        print('Loading {x} files of task {t}'.format(x = len(file_list), t = task))
        data_arr = np.stack((np.load(arr,allow_pickle=True) for arr in file_list)) # will be (run, vertex, TR)

        # for pRF task, we always average runs (if several) because same design
        if task == 'pRF':
            if run_type == 'median':
                print('getting median of pRF runs')
                data_arr = np.median(data_arr, axis = 0)[np.newaxis, ...] # to keep 3D, for later
            else:
                print('averaging pRF runs')
                data_arr = np.mean(data_arr, axis = 0)[np.newaxis, ...]

        # loop over runs
        data2fit = np.array([])
        self.run_num_arr = [] 
        self.ses_num_arr = []

        for r in range(data_arr.shape[0]):
            
            # data for that run
            data = data_arr[r]

            # if we want to chunk it
            if isinstance(chunk_num, int):
                # number of vertices of chunk
                num_vox_chunk = int(data.shape[0]/self.total_chunks[task])
                print('Slicing data into chunk {ch} of {ch_total}'.format(ch = chunk_num, 
                                            ch_total = self.total_chunks[task]))
        
                # chunk it
                data_out = data[num_vox_chunk * int(chunk_num):num_vox_chunk * int(chunk_num + 1), :]
            
            # if we want specific vertex
            elif isinstance(vertex, int) or isinstance(vertex, list) or isinstance(vertex, np.ndarray):
                print('Slicing data into vertex {ver}'.format(ver = vertex))
                data_out = data[vertex]
                
                if isinstance(vertex, int):
                    data_out = data_out[np.newaxis,...]
            
            # return whole array
            else:
                print('Returning whole data array')
                data_out = data

            ## if we want to keep baseline fix, we need to correct it!
            if self.correct_baseline[task]:
                print('Correcting baseline to be 0 centered')

                # crop and shift if such was the case
                bar_pass = self.mri_beh.FA_bar_pass_all if task == 'FA' else self.mri_beh.pRF_bar_pass_all
                crop_nr = self.crop_TRs_num[task] if self.crop_TRs[task] == True else None
                only_edges = True if task == 'FA' else False

                condition_per_TR = mri_utils.crop_shift_arr(bar_pass, 
                                                            crop_nr = crop_nr, 
                                                            shift = self.shift_TRs_num)

                data_out = mri_utils.baseline_correction(data_out, condition_per_TR, 
                                                        num_baseline_TRs = self.corr_base_TRs[task], 
                                                        baseline_interval = baseline_interval, 
                                                        avg_type = 'median', only_edges = only_edges)

            ## STACK
            if data_arr.shape[0]>1:
                data2fit = np.vstack([data2fit, data_out[np.newaxis, ...]]) if data2fit.size else data_out[np.newaxis, ...]
            else:
                if task == 'FA':
                    data2fit = data_out[np.newaxis, ...]
                else:
                    data2fit = data_out

            ## append run number, and ses number in list of ints
            # useful for when fitting several runs at same time
            file_rn, file_sn = mri_utils.get_run_ses_from_str(file_list[r])
            self.run_num_arr.append(file_rn)
            self.ses_num_arr.append(file_sn)
            

        # return filelist if that matters for fitting (mainly for FA task)
        if return_filenames:
            return data2fit, file_list
        else:
            return data2fit

    def subselect_array(self, input_arr, task = 'pRF', chunk_num = None, vertex = None):
        
        """
        Helper function to subselect array (with estimate values for example)
        depending on task, chunk number or vertex

        Parameters
        ----------
        input_arr: np.array
            input array, of size of data
        task: str
            task name for files in list (default pRF)
        chunk_num: int or None
            if we want to select specific chunk of initial array, then will return chunk of array
        vertex: int, or list of indices or None
            if we want to select specific vertex of arr, or list of vertices (from an ROI for example) then will return vertex array

        """

        # if we want to chunk it
        if isinstance(chunk_num, int):
            # number of vertices of chunk
            num_vox_chunk = int(input_arr.shape[0]/self.total_chunks[task])
            print('Slicing array into chunk {ch} of {ch_total}'.format(ch = chunk_num, 
                                        ch_total = self.total_chunks[task]))
    
            # chunk it
            arr_out = input_arr[num_vox_chunk * int(chunk_num):num_vox_chunk * int(chunk_num + 1), :]
        
        # if we want specific vertex
        elif isinstance(vertex, int) or isinstance(vertex, list) or isinstance(vertex, np.ndarray):
            print('Slicing array into vertex {ver}'.format(ver = vertex))
            arr_out = input_arr[vertex]
            
            if isinstance(vertex, int):
                arr_out = arr_out[np.newaxis,...]
        
        # return whole array
        else:
            print('Returning whole data array')
            arr_out = input_arr

        return arr_out
    
    def calc_rsq(self, data_arr, prediction):

        """"
        Calculate rsq of fit
        """

        return np.nan_to_num(1 - (np.nansum((data_arr - prediction)**2, axis=0)/ np.nansum(((data_arr - np.mean(data_arr))**2), axis=0)))
    
    def error_resid(self, timecourse, prediction, mean_err = False, return_array = False):

        """
        calculate residual error between a timecourse and a model prediction

        Parameters
        ----------
        timecourse : ndarray
            The actual, measured time-series against which the model is fit
        prediction : ndarray
            model prediction timecourse
        mean_err : bool
            if True, will calculate the mean of squared residulas instead of sum
        return_array: bool
            if True, then function returns residual array instead of scalar
        """

        if return_array:
            return (timecourse - prediction).flatten() # return error "timecourse", that will be used by minimize
        else:
            if mean_err:
                return np.mean((timecourse - prediction) ** 2)  # calculate mean of squared residuals

            else:
                return np.sum((timecourse - prediction) ** 2) # calculate residual sum of squared errors
            


                    


