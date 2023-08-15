import numpy as np
import re
import os
import os.path as op
import pandas as pd

import glob

class Model:

    def __init__(self, MRIObj, outputdir = None, pysub = 'hcp_999999', use_atlas = None):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
        outputdir: str
            absolute path to save fits
        use_atlas: str
            If we want to use atlas ROIs (ex: glasser, wang) or not [default].
        """

        ## set data object to use later on
        # has relevant paths etc
        self.MRIObj = MRIObj

        self.pysub = pysub

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = self.MRIObj.derivatives_pth
        else:
            self.outputdir = outputdir

        ## set variables useful when loading ROIs
        if use_atlas is None:
            self.plot_key = self.MRIObj.sj_space 
            self.annot_filename = ''
        else:
            self.plot_key = use_atlas
            self.annot_filename = self.MRIObj.atlas_annot[self.plot_key ]
        
        self.use_atlas = use_atlas
            
        ### some relevant params ###

        # set fit folder name
        self.fitfolder = {key:self.MRIObj.params['mri']['fitting'][key]['fit_folder'] for key in self.MRIObj.tasks}
        
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

    def get_run_ses_pp(self, participant, task = 'pRF', run_type = 'mean', ses = 'mean', file_ext = '_cropped.npy'):

        """
        get data file list
        and output run and ses number only
                
        Parameters
        ----------
        participant: str
            participant ID
        task: str
            task name for files in list (default pRF)
        run_type: string or int
            type of run to fit - mean (default), median, loo_rXsY (leaving out specific run and session) 
            or if int/'run-X' will do single run fit
        ses: int
            session number, only relevant when loading one specific run number (associated to a session number)
        """  

        ## get list of files to load
        file_list = self.MRIObj.mri_utils.get_bold_file_list(participant, task = task, ses = ses, file_ext = file_ext,
                                                            postfmriprep_pth = self.MRIObj.postfmriprep_pth, 
                                                            acq_name = self.MRIObj.acq)
        
        # if loading specific run
        if isinstance(run_type, int) or (isinstance(run_type, str) and 'loo_' not in run_type and len(re.findall(r'\d{1,10}', run_type))>0):

            if not isinstance(ses, int) or not (isinstance(ses, str) and len(re.findall(r'\d{1,10}', ses))>0):
                raise ValueError('Want to run specific run but did not provide session number!')
            else:
                run = re.findall(r'\d{1,10}', str(run_type))[0]
                print('Found run-{r} from ses-{s}'.format(r = run, s = ses))

                file_list = [file for file in file_list if 'run-{r}'.format(r = run) in file and 'ses-{s}'.format(s = ses) in file]

        # if leaving one run out
        elif 'loo_' in run_type:
            
            print('Leave-one out runs ({r})'.format(r = run_type))
            _, file_list = self.MRIObj.mri_utils.get_loo_filename(file_list, loo_key=run_type)

        ## now actually load data
        print('Found {x} files of task {t}'.format(x = len(file_list), t = task))

        # loop over files
        run_num_arr = [] 
        ses_num_arr = []

        for file in file_list:
    
            ## append run number, and ses number in list of ints
            # useful for when fitting several runs at same time
            file_rn, file_sn = self.MRIObj.mri_utils.get_run_ses_from_str(file)
            run_num_arr.append(file_rn)
            ses_num_arr.append(file_sn)

        return run_num_arr, ses_num_arr

    def get_data4fitting(self, file_list, task = 'pRF', run_type = 'mean',
                            chunk_num = None, vertex = None,
                            baseline_interval = 'empty_long', ses = 'mean', return_filenames = False, correct_baseline = None):

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

        if correct_baseline is None:
            correct_baseline = self.correct_baseline[task]
        
        # if loading specific run
        if isinstance(run_type, int) or (isinstance(run_type, str) and 'loo_' not in run_type and len(re.findall(r'\d{1,10}', run_type))>0):

            if not isinstance(ses, int) or not (isinstance(ses, str) and len(re.findall(r'\d{1,10}', ses))>0):
                raise ValueError('Want to run specific run but did not provide session number!')
            else:
                run = re.findall(r'\d{1,10}', str(run_type))[0]
                print('Loading run-{r} from ses-{s}'.format(r = run, s = ses))

                file_list = [file for file in file_list if 'run-{r}'.format(r = run) in file and 'ses-{s}'.format(s = ses) in file]

        # if leaving one run out
        elif 'loo_' in run_type:
            
            print('Leave-one out runs ({r})'.format(r = run_type))
            _, file_list = self.MRIObj.mri_utils.get_loo_filename(file_list, loo_key=run_type)

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
        data2fit = []
        self.run_num_arr = [] 
        self.ses_num_arr = []

        for r in range(data_arr.shape[0]):
            
            # subselect data for that run
            data_out = self.subselect_array(data_arr[r], task = task, chunk_num = chunk_num, vertex = vertex)

            ## if we want to keep baseline fix, we need to correct it!
            if correct_baseline:
                print('Correcting baseline to be 0 centered')

                # number of TRs per condition (bar pass)
                if task == 'pRF':
                    bar_pass = self.MRIObj.beh_utils.get_pRF_cond_per_TR(cond_TR_dict = self.MRIObj.pRF_nr_TRs, 
                                                                                bar_pass_direction = self.MRIObj.pRF_bar_pass)
                    only_edges = False
                elif task == 'FA':
                    bar_pass, _ = self.MRIObj.beh_utils.get_FA_run_struct(self.MRIObj.FA_bar_pass, 
                                                                                num_bar_pos = self.MRIObj.FA_num_bar_position, 
                                                                                empty_TR = self.MRIObj.FA_nr_TRs['empty_TR'], 
                                                                                task_trial_TR = self.MRIObj.FA_nr_TRs['task_trial_TR'])
                    only_edges = True
                    
                # crop and shift if such was the case
                condition_per_TR = self.MRIObj.mri_utils.crop_shift_arr(bar_pass, 
                                                                    crop_nr = self.MRIObj.task_nr_cropTR[task], 
                                                                    shift = self.MRIObj.shift_TRs_num)

                data_out = self.baseline_correction(data_out, condition_per_TR, 
                                                        num_baseline_TRs = self.corr_base_TRs[task], 
                                                        baseline_interval = baseline_interval, 
                                                        avg_type = 'median', only_edges = only_edges)

            ## append
            data2fit.append(np.vstack(data_out[np.newaxis, ...]))

            ## append run number, and ses number in list of ints
            # useful for when fitting several runs at same time
            file_rn, file_sn = self.MRIObj.mri_utils.get_run_ses_from_str(file_list[r])
            self.run_num_arr.append(file_rn)
            self.ses_num_arr.append(file_sn)

        if task == 'pRF':
            data2fit = data2fit[0]
        else:
            data2fit = np.array(data2fit)
            
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
            
    def baseline_correction(self, data, condition_per_TR = [], num_baseline_TRs = 6, baseline_interval = 'empty_long', 
                                    avg_type = 'median', only_edges = False, TR2task = 3):
        
        """Do baseline correction to timecourse
        Useful when we want a fix baseline during fitting

        Parameters
        ----------
        data : array
        2D array with data timecourses
        num_baseline_TRs: int
            number of baseline TRs to consider (will always be the last X TRs of the interval)
        baseline_interval : str
        name of the condition to get baseline values
        avg_type: str
            type of averaging done
        only_edges: bool
            if we want to only use the edges of the array to correct (this is start and end)
        TR2task: int
            number of TRs between end of baseline period and start of bar pass (i.e., shifts baseline period in time by X TRs) 
        """

        baseline_index = [i for i, val in enumerate(condition_per_TR) if str(val) == baseline_interval]
        interval_ind = []
        for i, ind in enumerate(baseline_index):
            if i > 0:
                if (ind - baseline_index[i-1]) > 1: ## transition point
                    interval_ind.append([baseline_index[i-1] - TR2task - num_baseline_TRs, baseline_index[i-1] - TR2task]) 

        if condition_per_TR[-1] == baseline_interval:
            interval_ind.append([baseline_index[-1] - TR2task - num_baseline_TRs, baseline_index[-1] - TR2task]) 
        
        if only_edges:
            interval_ind = np.array((interval_ind[0], interval_ind[-1]))

        # get baseline values
        baseline_arr = np.hstack([data[..., ind[0]:ind[1]] for ind in interval_ind])

        # average
        if avg_type == 'median':
            avg_baseline = np.median(baseline_arr, axis = -1)
        else:
            avg_baseline = np.mean(baseline_arr, axis = -1)

        return data - avg_baseline[...,np.newaxis]
            
    def convert_pix2dva(self, val_pix):

        """
        Convert pixel value to dva
        """
        return val_pix * self.MRIObj.mri_utils.dva_per_pix(height_cm = self.MRIObj.params['monitor']['height'], 
                                                        distance_cm = self.MRIObj.params['monitor']['distance'], 
                                                        vert_res_pix = self.MRIObj.screen_res[1])

                    


