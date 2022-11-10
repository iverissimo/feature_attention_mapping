import numpy as np
import re
import os
import os.path as op
import pandas as pd
import yaml
import glob

from PIL import Image, ImageDraw

from FAM.utils import mri as mri_utils
from FAM.processing import preproc_behdata
from FAM.fitting.model import Model

from scipy.optimize import minimize

from joblib import Parallel, delayed
from tqdm import tqdm

from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel


class FA_model(Model):

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

    
    def get_bar_dm(self, run_bar_pos_df, attend_bar = True, osf = 10, res_scaling = .1, crop_nr = None,
                stim_dur_seconds = 0.5, FA_bar_pass_all = [], oversample_sec = True):
        
        """
        save an array with the FA (un)attended
        bar position for the run

        Parameters
        ----------
        run_bar_pos_df : pandas DataFrame
            bar position dataframe for the run, can be obtained by preproc_behdata.load_FA_bar_position()
        attend_bar : bool
            if we want position for attended bar or not
        osf: int
            oversampling factor, if we want to oversample in time
        res_scaling: float
            spatial rescaling factor for x and y axis
        stim_dur_seconds: float
            duration of stim (bar presentation) in seconds
        FA_bar_pass_all: list
            list with condition per TR/trial
        crop_nr: int
            if we are cropping TRs, how many (if None, it will not crop TRs)
        oversample_sec: bool
            if we want to oversample in seconds (relevant when stim dur is fraction of seconds and not ex: 1.6s = 1TR)

        """ 
        # if we want to oversample in seconds, then we need to multiply #TRs by TR duration in seconds
        if oversample_sec:
            oversampling_time = self.MRIObj.TR * osf
            stim_dur = stim_dur_seconds * osf
        else:
            oversampling_time = osf
            stim_dur = (stim_dur_seconds / self.MRIObj.TR) * osf # then we need to transform stim dur in seconds to TR

        ## crop and shift if such was the case
        condition_per_TR = mri_utils.crop_shift_arr(FA_bar_pass_all,
                                                crop_nr = crop_nr, 
                                                shift = self.shift_TRs_num)
        
        ## bar midpoint coordinates
        midpoint_bar = run_bar_pos_df[run_bar_pos_df['attend_condition'] == attend_bar].bar_midpoint_at_TR.values[0]
        ## bar direction (vertical vs horizontal)
        direction_bar = run_bar_pos_df[run_bar_pos_df['attend_condition'] == attend_bar].bar_pass_direction_at_TR.values[0]

        # save screen display for time stamp (#TRs * TR * osf)
        visual_dm_array = np.zeros((int(len(condition_per_TR) * oversampling_time), 
                                    round(self.screen_res[0] * res_scaling), 
                                    round(self.screen_res[1] * res_scaling)))
        i = 0

        for trl, bartype in enumerate(condition_per_TR): # loop over bar pass directions

            img = Image.new('RGB', tuple(self.screen_res)) # background image

            if bartype not in np.array(['empty','empty_long']): # if not empty screen

                if direction_bar[i] == 'vertical':
                    coordenates_bars = {'upLx': 0, 
                                        'upLy': self.screen_res[1]/2-midpoint_bar[i][-1]+0.5*self.bar_width['FA']*self.screen_res[1],
                                        'lowRx': self.screen_res[0], 
                                        'lowRy': self.screen_res[1]/2-midpoint_bar[i][-1]-0.5*self.bar_width['FA']*self.screen_res[1]}


                elif direction_bar[i] == 'horizontal':

                    coordenates_bars = {'upLx': self.screen_res[0]/2+midpoint_bar[i][0]-0.5*self.bar_width['FA']*self.screen_res[0], 
                                        'upLy': self.screen_res[1],
                                        'lowRx': self.screen_res[0]/2+midpoint_bar[i][0]+0.5*self.bar_width['FA']*self.screen_res[0], 
                                        'lowRy': 0}

                # set draw method for image
                draw = ImageDraw.Draw(img)
                # add bar, coordinates (upLx, upLy, lowRx, lowRy)
                draw.rectangle(tuple([coordenates_bars['upLx'],coordenates_bars['upLy'],
                                    coordenates_bars['lowRx'],coordenates_bars['lowRy']]), 
                            fill = (255,255,255),
                            outline = (255,255,255))

                # increment counter
                i = i+1
                
                ## save in array - takes into account stim dur in seconds
                visual_dm_array[int(trl*oversampling_time):int(trl*oversampling_time + stim_dur), ...] = np.array(img)[::round(1/res_scaling),::round(1/res_scaling),0][np.newaxis,...]
                
            else:
                ## save in array
                visual_dm_array[int(trl*oversampling_time):int(trl*oversampling_time + oversampling_time), ...] = np.array(img)[::round(1/res_scaling),::round(1/res_scaling),0][np.newaxis,...]

        # swap axis to have time in last axis [x,y,t]
        visual_dm = visual_dm_array.transpose([1,2,0])
        
        return mri_utils.normalize(visual_dm)
            
    
    def get_visual_DM_dict(self, participant, filelist, save_overlap = True, save_full_stim = True):
    
        """
        Given participant ID and filelist of runs to fit,
        will return dict for each run in list,
        with visual DM for each type of regressor (attended bar, unattended bar, overlap, both bars, etc)
        
        ex:
        out_dict['r1s1'] = {'att_bar': [x,y,t], 'unatt_bar': [x,y,t], ...}

        Parameters
        ----------
        participant : str
            participant ID
        filelist : list
            list with absolute filenames to fit
        save_overlap: bool
            if we also want to save the overlap of bars (spatial region where both bars overlap)
        save_full_stim: bool
            if we want to save both bars combined (full stimulus that was on screen)

        """ 
        
        # set number of TRs to crop
        crop_nr = self.crop_TRs_num['FA'] if self.crop_TRs['FA'] == True else None
        
        # set empty dicts
        out_dict = {}
        
        ## loop over files
        for ind, file in enumerate(filelist):
            
            ## get run and ses from file
            run_num, ses_num = mri_utils.get_run_ses_from_str(file) 
            
            out_dict['r{r}s{s}'.format(r = run_num, s = ses_num)] = {}
            
            ## get bar position df for run
            run_bar_pos_df = self.mri_beh.load_FA_bar_position(participant, ses = 'ses-{s}'.format(s = ses_num), 
                                                    ses_type = 'func', run_num = run_num)
            
            ## GET DM FOR ATTENDED BAR
            out_dict['r{r}s{s}'.format(r = run_num, 
                                        s = ses_num)]['att_bar'] = self.get_bar_dm(run_bar_pos_df,
                                                                                        attend_bar = True,
                                                                                        osf = self.osf, res_scaling = self.res_scaling,
                                                                                        crop_nr = crop_nr,
                                                                                        stim_dur_seconds = self.MRIObj.FA_bars_phase_dur,
                                                                                        FA_bar_pass_all = self.mri_beh.FA_bar_pass_all,
                                                                                        oversample_sec = True)
            ## GET DM FOR UNATTENDED BAR
            out_dict['r{r}s{s}'.format(r = run_num, 
                                        s = ses_num)]['unatt_bar'] = self.get_bar_dm(run_bar_pos_df,
                                                                                        attend_bar = False,
                                                                                        osf = self.osf, res_scaling = self.res_scaling,
                                                                                        crop_nr = crop_nr,
                                                                                        stim_dur_seconds = self.MRIObj.FA_bars_phase_dur,
                                                                                        FA_bar_pass_all = self.mri_beh.FA_bar_pass_all,
                                                                                        oversample_sec = True)

            if save_overlap:
                ## GET DM FOR OVERLAP OF BARS
                out_dict['r{r}s{s}'.format(r = run_num, 
                                            s = ses_num)]['overlap'] = mri_utils.get_bar_overlap_dm(np.stack((out_dict['r{r}s{s}'.format(r = run_num, s = ses_num)]['att_bar'],
                                                                                                             out_dict['r{r}s{s}'.format(r = run_num, s = ses_num)]['unatt_bar'])))

            if save_full_stim:
                ## GET DM FOR BOTH BARS COMBINED (FULL STIM THAT WAS ON SCREEN)
                stimulus_dm = np.sum(np.stack((out_dict['r{r}s{s}'.format(r = run_num, s = ses_num)]['att_bar'],
                                                out_dict['r{r}s{s}'.format(r = run_num, s = ses_num)]['unatt_bar'])), axis = 0)
                stimulus_dm[stimulus_dm >=1] = 1
                
                out_dict['r{r}s{s}'.format(r = run_num, 
                                            s = ses_num)]['full_stim'] = stimulus_dm

            if ind == 0:
                # get keys for each condition
                condition_dm_keys = [name for name in out_dict['r{r}s{s}'.format(r = run_num, s = ses_num)].keys()]
                
        return out_dict, condition_dm_keys


    def setup_vars4fitting(self, participant, pp_prf_estimates, ses = 1,
                    run_type = 'loo_r1s1', chunk_num = None, vertex = None, ROI = None,
                    prf_model_name = None, file_ext = '_cropped_confound_psc.npy', 
                    outdir = None, fit_overlap = True, fit_full_stim = False):

        """
        set up variables necessary for fitting
        will load data, fa visual design matrix and parameters
        for specified participant

        Parameters
        ----------
        participant : str
            participant ID
        pp_prf_estimates : dict
            dict with participant prf estimates
        ses: int/str
            session number of data we are fitting
        run_type: string or int
            type of run to fit - mean (default), median, loo_rXsY (leaving out specific run and session) 
            or if int/'run-X'/'rXsY' will do single run fit
        chunk_num: int or None
            if we want to fit specific chunk of data, then will return chunk array
        vertex: int, or list of indices or None
            if we want to fit specific vertex of data, or list of vertices (from an ROI for example) then will return vertex array
        ROI: str or None
            roi name 
        prf_model_name: str
            prf model name from which estimates were derived
        file_ext: str 
            ending of bold file, to know which one to load
        outdir: str
            path to save outputted files
        fit_overlap: bool
            if we want to fit overlap area as a separate regressor
        fit_full_stim: bool
            if we want to fit the full stimulus (both bars combined) or not
        """ 

        # if we provided session as str
        if isinstance(ses, str) and len(re.findall(r'\d{1,10}', ses))>0:
            ses = int(re.findall(r'\d{1,10}', ses)[0]) # make int

        ## get list of files to load
        bold_filelist = self.get_bold_file_list(participant, task = 'FA', ses = ses, file_ext = file_ext)

        ## Load data array and file list names
        data, train_file_list = self.get_data4fitting(bold_filelist, task = 'FA', run_type = run_type, chunk_num = chunk_num, vertex = vertex, ses = ses,
                                            baseline_interval = 'empty', return_filenames = True)
        print('Fitting files %s'%str(train_file_list))
        self.train_file_list = train_file_list # save to access later

        ## Set nan voxels to 0, to avoid issues when fitting
        masked_data = np.nan_to_num(data)

        ## set prf model name
        if prf_model_name is None:
            self.prf_model_name = self.model_type['pRF']
        else: 
            self.prf_model_name = prf_model_name

        ## set output dir to save estimates
        if outdir is None:
            if 'loo_' in run_type:
                outdir = op.join(self.outputdir, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant), run_type)
            else:
                outdir = op.join(self.outputdir, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant), 'ses-{s}'.format(s = ses))
            
        # add model identifier to it
        self.outdir = op.join(outdir, 'it_{pm}'.format(pm = self.prf_model_name))

        os.makedirs(self.outdir, exist_ok = True)
        print('saving files in %s'%self.outdir)

        ## set base filename that will be used for estimates
        basefilename = 'sub-{sj}_task-FA_acq-{acq}_runtype-{rt}'.format(sj = participant,
                                                                            acq = self.MRIObj.acq,
                                                                            rt = run_type)
        if chunk_num is not None:
            basefilename += '_chunk-{ch}'.format(ch = str(chunk_num).zfill(3))
        elif vertex is not None:
            basefilename += '_vertex-{ver}'.format(ver = str(vertex))
        elif ROI:
            basefilename += '_ROI-{roi}'.format(roi = str(ROI))
        
        basefilename += file_ext.replace('.npy', '.npz')
        self.basefilename = basefilename

        ## Get visual dm for different bars per run
        # and also condition dm keys (att bar, unatt bar, overlap, full_stim etc)
        self.visual_dm_dict, self.condition_dm_keys = self.get_visual_DM_dict(participant, train_file_list, save_overlap = fit_overlap, save_full_stim = fit_full_stim)

        ## get pRF model estimate keys
        self.prf_est_keys = [val for val in list(pp_prf_estimates.keys()) if val!='r2']
        print('pRF {m} model estimates found {l}'.format(m = self.prf_model_name, l = str(self.prf_est_keys)))

        # subselect pRF estimates similar to data
        # to avoid index issues
        if (chunk_num is not None) or (vertex is not None):
            masked_prf_estimates = {}
            for key in pp_prf_estimates.keys():
                print('Masking pRF estimates, to have same # vertices of FA data')
                masked_prf_estimates[key] = self.subselect_array(pp_prf_estimates[key], task = 'pRF', chunk_num = chunk_num, vertex = vertex)
        else:
            masked_prf_estimates = pp_prf_estimates

        return masked_data, masked_prf_estimates


    def initialize_parameters(self, prf_estimates, prf_pars2vary = [], prf_bounds = None, rsq_threshold = None):

        """
        initialize parameters for fitting
        for specified participant

        Parameters
        ----------
        prf_estimates : dict
            dict with participant prf estimates
        rsq_threshold: float or None
            fit vertices where prf fit above certain rsq threshold 
        prf_pars2vary: list
            list with name of prf estimates to vary during fitting
        prf_bounds: list
            list with tuples of prf estimate bounds for fitting
        """ 


        ## get relevant indexes to fit
        # set threshold
        if rsq_threshold is None:
            rsq_threshold = self.prf_rsq_threshold

        # find indexes worth fitting
        # this is, where pRF rsq > than predetermined threshold
        self.ind2fit = np.where(((prf_estimates['r2'] > rsq_threshold)))[0]

        ## set up parameters object for all vertices to be fitted
        # with pRF estimates values
        print('Initializing FA parameters with pRF estimates...')
        self.pars_arr = np.array([{key: prf_estimates[key][ind] for key in self.prf_est_keys} for ind in self.ind2fit])

        ## set up prf bounds 
        print('Setting FA parameter bounds...')
        pars_bounds_arr = np.array([{key: (prf_estimates[key][ind],prf_estimates[key][ind]) for key in self.prf_est_keys} for ind in self.ind2fit])

        if len(prf_pars2vary) > 0:
            print('fitting %i pRF parameters - %s'%(len(prf_pars2vary), str(prf_pars2vary)))

            if prf_bounds is None:
                if self.prf_bounds is None:
                    print('No prf bounds defined, will let all varying variables fit from -inf to +inf')
                    prf_bounds = [(-np.inf,+np.inf) for ind in range(len(self.prf_est_keys))]
                else:
                    prf_bounds = self.prf_bounds

            for key in prf_pars2vary:
                # depending on which parameters we want to vary, set bounds accordingly for all vertices
                pars_bounds_arr = [{**d, **{key: (prf_bounds[self.prf_est_keys.index(key)][0],
                                                prf_bounds[self.prf_est_keys.index(key)][-1])}} for d in pars_bounds_arr]

        self.pars_bounds_arr = pars_bounds_arr


    def iterative_fit(self, data, starting_params = None, model_function = None,
                                    ind2fit = None, method = None, kws_dict = {},
                                    xtol = 1e-3, ftol = 1e-3, n_jobs = 16, params_bounds = None):

        """
        perform iterative fit of params on all vertices of data

        Parameters
        ----------
        data: arr
            3D data array of [runs, vertices, time]
        starting_params: list/arr
            array with parameters dict of relevant estimates per vertex
        model_function: callable
            The objective function that takes `parameters` and `args` and
            produces a model time-series.
        ind2fit: list/arr
            array with indices of vertices of data array to fit
        method:
            optimizer method to use in minimize
        kws_dict: dict
            dictionary with extra inputs parameters (var name: val) that will be used by model function
        params_bounds: arr/list
            array with paramter bounds for each vertex of data to fit
        """ 

        # define optimizer method to use
        if method is None:
            method = self.optimizer['FA']

        # parameters to fit
        if starting_params is None:
            starting_params = self.pars_arr
        
        # bounds for parameters
        if params_bounds is None:
            params_bounds = self.pars_bounds_arr
        
        # list of indices to fit
        if ind2fit is None:
            ind2fit = self.ind2fit

        ## actually fit vertices
        # and output relevant params + rsq of model fit in dataframe
        results = np.array(Parallel(n_jobs=n_jobs)(delayed(self.iterative_search)(data[:,vert,:],
                                                                                starting_params[ind],
                                                                                model_function,
                                                                                kws_dict = kws_dict,
                                                                                xtol = xtol, ftol = ftol,
                                                                                method = method,
                                                                                bounds_dict = params_bounds[ind])
                                                                            for ind, vert in enumerate(tqdm(ind2fit))))

        return results


    def iterative_search(self, train_timecourse, tc_dict, model_function, kws_dict={}, 
                                                    xtol = 1e-3, ftol = 1e-3, method = None, bounds_dict = {}):

        """
        iterative search func for a single vertex 

        Parameters
        ----------
        train_timecourse: arr
            2D data array of [runs, time]
        tc_dict: dict
            parameters dict of relevant estimates for vertex
        model_function: callable
            The objective function that takes `parameters` and `args` and
            produces a model time-series.
        kws_dict: dict
            dictionary with extra inputs parameters (var name: val) that will be used by model function
        method:
            optimizer method to use in minimize
        bounds_dict: dict
            dict with paramter bounds for vertex
        
        """ 

        ## turn parameters and bounds into arrays because of scipy minimize
        # but save dict keys to guarantee order is correct
        parameters_keys = list(tc_dict.keys())

        # update kws
        kws_dict['parameters_keys'] = parameters_keys

        # set parameters and bounds into list, to conform to scipy minimze format
        tc_pars = [tc_dict[key] for key in parameters_keys]
        bounds = [bounds_dict[key] for key in parameters_keys]

        ## minimize residuals
        if method in ['lbfgsb', 'L-BFGS-B']: 
            
            out = minimize(self.get_fit_residuals, tc_pars, bounds=bounds, args = (train_timecourse, model_function, kws_dict),
                        method = method, options = dict(ftol = ftol, maxls = 40, disp = True))

        elif method == 'trust-constr':
            out = minimize(self.get_fit_residuals, tc_pars, bounds=bounds, args = (train_timecourse, model_function, kws_dict),
                        method = method, tol = ftol, options = dict(xtol = xtol, disp = True))

        # set output params as dict
        out_dict = {key: out['x'][ind] for ind, key in enumerate(parameters_keys)}
        # add rsq value
        out_dict['r2'] = 1 - out['fun']/(train_timecourse.ravel().shape[0] * train_timecourse.ravel().var())
        
        return out_dict

    
    def get_fit_residuals(self, tc_pars, timecourse, model_function, kws_dict):

        """
        given data timecourse and parameters, returns 
        residual sum of squared errors between the prediction and data

        Parameters
        ----------
        tc_pars: arr/list
            list with parameter values for vertex
        timecourse: arr
            2D data array of [runs, time]
        model_function: callable
            The objective function that takes `parameters` and `args` and
            produces a model time-series.
        kws_dict: dict
            dictionary with extra inputs parameters (var name: val) that will be used by model function
        """ 

        ## get prediction timecourse for that visual design matrix
        # and parameters
        model_arr = model_function(tc_pars, **kws_dict)

        return mri_utils.error_resid(timecourse, model_arr, mean_err = False, return_array = False)

    
    def get_fit_timecourse(self, pars, reg_name = 'full_stim', bar_keys = ['att_bar', 'unatt_bar'], parameters_keys = []):
    
        """
        given parameters for that vertex 
        use pRF model to produce a timecourse for the design matrix 

        Note: will return 2D model array [runs, time]

        Parameters
        ----------
        pars: array/list
            array with relevant estimates
        reg_name: str
            "regressor" name as described in visual dm dict. if None will assume we want to weight each bar and stack them
        bar_keys: list
            list with bar names
        parameters_keys: list
            list with parameter names, in same order of pars, for bookeeping
        
        """ 
        
        model_arr = np.array([])
        
        ## loop over runs
        for run_id in self.visual_dm_dict.keys():
            
            #print('making timecourse for run %s'%run_id)

            ## if we want a specifc regressor (example full stim, att bar)
            if reg_name is not None:
                run_visual_dm = self.visual_dm_dict[run_id][reg_name]

            else:
                # will weight and sum bars
                if 'overlap' in list(self.visual_dm_dict[run_id].keys()):
                    
                    run_visual_dm = mri_utils.sum_bar_dms(np.stack((self.visual_dm_dict[run_id][bar_keys[0]] * pars[parameters_keys.index('gain_{v}'.format(v = bar_keys[0]))],
                                                                    self.visual_dm_dict[run_id][bar_keys[1]] * pars[parameters_keys.index('gain_{v}'.format(v = bar_keys[1]))])), 
                                                        overlap_dm = self.visual_dm_dict[run_id]['overlap'], 
                                                        overlap_weight = pars[parameters_keys.index('gain_overlap')])
                else:
                    run_visual_dm = mri_utils.sum_bar_dms(np.stack((self.visual_dm_dict[run_id][bar_keys[0]] * pars[parameters_keys.index('gain_{v}'.format(v = bar_keys[0]))],
                                                                self.visual_dm_dict[run_id][bar_keys[1]] * pars[parameters_keys.index('gain_{v}'.format(v = bar_keys[1]))])), 
                                                                overlap_dm = None)
            
            
            ## make stimulus object, which takes an input design matrix and sets up its real-world dimensions
            fa_stim = PRFStimulus2D(screen_size_cm = self.MRIObj.params['monitor']['height'],
                                    screen_distance_cm = self.MRIObj.params['monitor']['distance'],
                                    design_matrix = run_visual_dm,
                                    TR = self.MRIObj.TR)

            # set hrf params
            if self.fit_hrf and 'hrf_derivative' in parameters_keys:
                hrf_params = [1, pars[parameters_keys.index('hrf_derivative')], pars[parameters_keys.index('hrf_dispersion')]]
            else:
                hrf_params = [1, 1, 0]

            ## set prf model to use
            if self.prf_model_name == 'gauss':

                model_obj = Iso2DGaussianModel(stimulus = fa_stim,
                                                filter_predictions = False,
                                                filter_type = self.MRIObj.params['mri']['filtering']['type']['pRF'],
                                                filter_params = {'highpass': self.MRIObj.params['mri']['filtering']['highpass'],
                                                                'add_mean': self.MRIObj.params['mri']['filtering']['add_mean'],
                                                                'window_length': self.MRIObj.params['mri']['filtering']['window_length'],
                                                                'polyorder': self.MRIObj.params['mri']['filtering']['polyorder']},
                                                osf = self.osf * self.MRIObj.TR,
                                                hrf_onset = self.hrf_onset,
                                                hrf = hrf_params,
                                                pad_length = int(20 * self.osf * self.MRIObj.TR)
                                                )
            elif self.prf_model_name == 'css':

                model_obj = CSS_Iso2DGaussianModel(stimulus = fa_stim,
                                                filter_predictions = False,
                                                filter_type = self.MRIObj.params['mri']['filtering']['type']['pRF'],
                                                filter_params = {'highpass': self.MRIObj.params['mri']['filtering']['highpass'],
                                                                'add_mean': self.MRIObj.params['mri']['filtering']['add_mean'],
                                                                'window_length': self.MRIObj.params['mri']['filtering']['window_length'],
                                                                'polyorder': self.MRIObj.params['mri']['filtering']['polyorder']},
                                                osf = self.osf * self.MRIObj.TR,
                                                hrf_onset = self.hrf_onset,
                                                hrf = hrf_params,
                                                pad_length = int(20 * self.osf * self.MRIObj.TR)
                                                )

            elif self.prf_model_name == 'dog':

                model_obj = DoG_Iso2DGaussianModel(stimulus = fa_stim,
                                                filter_predictions = False,
                                                filter_type = self.MRIObj.params['mri']['filtering']['type']['pRF'],
                                                filter_params = {'highpass': self.MRIObj.params['mri']['filtering']['highpass'],
                                                                'add_mean': self.MRIObj.params['mri']['filtering']['add_mean'],
                                                                'window_length': self.MRIObj.params['mri']['filtering']['window_length'],
                                                                'polyorder': self.MRIObj.params['mri']['filtering']['polyorder']},
                                                osf = self.osf * self.MRIObj.TR,
                                                hrf_onset = self.hrf_onset,
                                                hrf = hrf_params,
                                                pad_length = int(20 * self.osf * self.MRIObj.TR)
                                                )

            elif self.prf_model_name == 'dn':

                model_obj = Norm_Iso2DGaussianModel(stimulus = fa_stim,
                                                filter_predictions = False,
                                                filter_type = self.MRIObj.params['mri']['filtering']['type']['pRF'],
                                                filter_params = {'highpass': self.MRIObj.params['mri']['filtering']['highpass'],
                                                                'add_mean': self.MRIObj.params['mri']['filtering']['add_mean'],
                                                                'window_length': self.MRIObj.params['mri']['filtering']['window_length'],
                                                                'polyorder': self.MRIObj.params['mri']['filtering']['polyorder']},
                                                osf = self.osf * self.MRIObj.TR,
                                                hrf_onset = self.hrf_onset,
                                                hrf = hrf_params,
                                                pad_length = int(20 * self.osf * self.MRIObj.TR)
                                                )
            
            # get run timecourse
            run_timecourse = model_obj.return_prediction(*list([pars[parameters_keys.index(name)] for name in self.prf_est_keys if 'hrf' not in name]))
            
            ## resample to TR and stack
            model_arr = np.vstack([model_arr, mri_utils.resample_arr(run_timecourse, osf = self.osf, final_sf = self.MRIObj.TR)]) if model_arr.size else mri_utils.resample_arr(run_timecourse, osf = self.osf, final_sf = self.MRIObj.TR)

        # temporary, for testing purposes
        self.model_obj = model_obj
        self.run_timecourse = run_timecourse

        return model_arr


    def load_FA_model_estimates(self, participant, ses = 1, run = 'r1s1', run_type = 'loo_r1s1', model_name = None, 
                            prf_model_name = 'gauss', fit_hrf = False, outdir = None):

        """
        Helper function to load FA model estimates
        when they already where fitted and save in out folder

        Parameters
        ----------
        participant: str
            participant ID
        ses: int/str
            session number of data we are fitting
        run_type: string or int
            type of run to fit - 1, loo_rXsY (leaving out specific run and session) 
            or if int/'run-X' will do single run fit
        model_name: str or None
            model name to be loaded (if None defaults to class model)
        iterative: bool
            if we want to load iterative fitting results [default] or grid results

        """
        
        if outdir is not None:
            # if model name to load not given, use the one set in the class
            if model_name:
                model_name = model_name
            else:
                model_name = self.model_type['FA']

            # set folder name
            model_folder = self.MRIObj.params['mri']['fitting']['FA']['fit_folder'][model_name]

            # path to fa estimates 
            if 'loo_' in run_type:
                FAdir = op.join(self.MRIObj.derivatives_pth, model_folder, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant), run_type)
            else:
                FAdir = op.join(self.MRIObj.derivatives_pth, model_folder, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant), 'ses-{s}'.format(s = ses))

            outdir = op.join(FAdir, 'it_{mn}'.format(mn = prf_model_name))            
        
        
        # basefilename
        basefilename = 'sub-{sj}_task-FA_acq-{acq}_run-{r}_runtype-{rt}'.format(sj = participant,
                                                                            acq = self.MRIObj.acq,
                                                                            r = run,
                                                                            rt = run_type)
        
        ## now actually load 
        filelist = glob.glob(op.join(outdir, '*'))

        if fit_hrf:
            filename = [file for file in filelist if basefilename in file and 'HRF' in file and file.endswith('.csv')]
        else:
            filename = [file for file in filelist if basefilename in file and 'HRF' not in file and file.endswith('.csv')]
        
        if len(filename)>1:
            print('%s'%str(filename))
            raise ValueError('Too many files found, conflict')
        elif len(filename) == 0:
            raise ValueError('No file found')
        
        return pd.read_csv(filename[0],index_col=[0])



class FullStim_model(FA_model):

    ## Model that fits model on full stimulus
    # 
    # Goal is to check if there is already some variance explained by the 
    # pRF estimates alone (without attentional modulation)

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

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.MRIObj.derivatives_pth, self.MRIObj.params['mri']['fitting']['FA']['fit_folder']['full_stim'])
        else:
            self.outputdir = outputdir


    def fit_data(self, participant, pp_prf_estimates, ses = 1,
                    run_type = 'loo_r1s1', chunk_num = None, vertex = None, ROI = None,
                    prf_model_name = None, rsq_threshold = None, file_ext = '_cropped_LinDetrend_psc.npy', 
                    outdir = None, save_estimates = False, prf_bounds = None,
                    xtol = 1e-3, ftol = 1e-4, n_jobs = 16, prf_pars2vary = ['betas'], reg_name = 'full_stim', bar_keys = ['att_bar', 'unatt_bar']):

        """
                
        Parameters
        ----------
        participant: str
            participant ID
        run_type: string or int
            type of run to fit, mean (default), or if int will do single run fit
        file_ext: dict
            file extension, to select appropriate files
        """  

        ## set up fitting by loading data and visual dm for data
        # also loads prf parameters
        masked_data, masked_prf_estimates = self.setup_vars4fitting(participant, pp_prf_estimates, ses = ses,
                                                            run_type = run_type, chunk_num = chunk_num, vertex = vertex, ROI = ROI,
                                                            prf_model_name = prf_model_name, file_ext = file_ext, 
                                                            outdir = outdir, fit_overlap = False, fit_full_stim = True)

        ## initialize params for fitting
        self.initialize_parameters(masked_prf_estimates, prf_pars2vary = prf_pars2vary, prf_bounds = prf_bounds, rsq_threshold = rsq_threshold)

        ## actually fit data
        # call iterative fit function, giving it masked data and pars
        results = self.iterative_fit(masked_data, self.pars_arr, self.get_fit_timecourse, kws_dict = {'reg_name': reg_name, 'bar_keys': bar_keys}, 
                                                    xtol = xtol, ftol = ftol, method = None, n_jobs = n_jobs) 

        ## saves results as list of dataframes, with length = #runs
        results_list = []
        
        for r, name in enumerate(self.visual_dm_dict.keys()):
            
            # convert fitted params list of dicts as Dataframe
            fitted_params_df = pd.DataFrame(d for d in results)
            # and add vertex number for bookeeping
            if vertex is not None:
                fitted_params_df['vertex'] = vertex
            else:
                fitted_params_df['vertex'] = self.ind2fit

            results_list.append(fitted_params_df.copy())

            # if we want to save estimates, do so as csv
            if save_estimates:
                filename = self.basefilename.replace('runtype','run-{n}_runtype'.format(n = name))
                filename = filename.replace('.npz', '_it_{pm}_estimates.csv'.format(pm = prf_model_name))

                # if we are fitting hrf, include that in name
                if self.fit_hrf:
                    filename = filename.replace('estimates.csv', 'HRF_estimates.csv')

                fitted_params_df.to_csv(op.join(self.outdir, filename))
                
        return results_list


    
class Gain_model(FA_model):

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

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.MRIObj.derivatives_pth, self.MRIObj.params['mri']['fitting']['FA']['fit_folder']['gain'])
        else:
            self.outputdir = outputdir
    

    def fit_data(self, participant, pp_prf_estimates, ses = 1,
                    run_type = 'loo_r1s1', chunk_num = None, vertex = None, ROI = None,
                    prf_model_name = None, rsq_threshold = None, file_ext = '_cropped_confound_psc.npy', 
                    outdir = None, save_estimates = False, fit_overlap = True, fit_full_stim = False,
                    prf_pars2vary = ['betas'], prf_bounds = None, bar_keys = ['att_bar', 'unatt_bar'],
                    gain_cond = None, xtol = 1e-3, ftol = 1e-4, n_jobs = 16):

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

        ## set up fitting by loading data and visual dm for data
        # also loads prf parameters
        masked_data, masked_prf_estimates = self.setup_vars4fitting(participant, pp_prf_estimates, ses = ses,
                                                            run_type = run_type, chunk_num = chunk_num, vertex = vertex, ROI = ROI,
                                                            prf_model_name = prf_model_name, file_ext = file_ext, 
                                                            outdir = outdir, fit_overlap = fit_overlap, fit_full_stim = fit_full_stim)

        ## initialize params for fitting
        self.initialize_parameters(masked_prf_estimates, prf_pars2vary = prf_pars2vary, prf_bounds = prf_bounds, rsq_threshold = rsq_threshold)

        ## now get FA gain estimate keys
        if gain_cond is None:
            self.gain_keys = ['gain_{cond}'.format(cond = ckey) for ckey in self.condition_dm_keys]
        else:
            self.gain_keys = ['gain_{cond}'.format(cond = ckey) for ckey in gain_cond]

        ## and add to parameter array
        print('Updating FA parameters and bounds with Gain estimates %s'%str(self.gain_keys))
        for key in self.gain_keys:
            self.pars_arr = np.array([{**d, **{key: 1}} for d in self.pars_arr]) # starting value is 1 by default, consider making input
            self.pars_bounds_arr = [{**d, **{key: (-np.inf, +np.inf)}} for d in self.pars_bounds_arr] # unbounded, should also set as input/in yml

        ## actually fit data
        # call iterative fit function, giving it masked data and pars
        results = self.iterative_fit(masked_data, self.pars_arr, self.get_fit_timecourse, kws_dict = {'reg_name': None, 'bar_keys': bar_keys},
                                                xtol = xtol, ftol = ftol, method = None, n_jobs = n_jobs) 

        ## saves results as list of dataframes, with length = #runs
        results_list = []
        
        for r, name in enumerate(self.visual_dm_dict.keys()):
            
            # convert fitted params list of dicts as Dataframe
            fitted_params_df = pd.DataFrame(d for d in results)
            # and add vertex number for bookeeping
            if vertex is not None:
                fitted_params_df['vertex'] = vertex
            else:
                fitted_params_df['vertex'] = self.ind2fit

            results_list.append(fitted_params_df.copy())

            # if we want to save estimates, do so as csv
            if save_estimates:
                filename = self.basefilename.replace('runtype','run-{n}_runtype'.format(n = name))
                filename = filename.replace('.npz', '_it_{pm}_estimates.csv'.format(pm = prf_model_name))

                # if we are fitting hrf, include that in name
                if self.fit_hrf:
                    filename = filename.replace('estimates.csv', 'HRF_estimates.csv')

                fitted_params_df.to_csv(op.join(self.outdir, filename))

        return results_list


class GLM_model(FA_model):

    def __init__(self, MRIObj, outputdir = None, tasks = ['pRF', 'FA']):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
            
        """

        # need to initialize parent class (Model), indicating output infos
        super().__init__(MRIObj = MRIObj, outputdir = outputdir, tasks = tasks)

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.MRIObj.derivatives_pth, self.MRIObj.params['mri']['fitting']['FA']['fit_folder']['glm'])
        else:
            self.outputdir = outputdir

        # task regressors of interest to be used
        self.task_reg_names = self.MRIObj.params['mri']['fitting']['FA']['glm_task_regs']

        # if provided, nuisance regressor names (if empty, then NO nuisance regressors will be used)
        self.add_nuisance_reg = self.MRIObj.params['mri']['fitting']['FA']['glm_nuisance_regs']
        self.nuisance_reg_names = None


    def fit_data(self, participant, pp_prf_estimates, ses = 1,
                    run_type = 'loo_r1s1', chunk_num = None, vertex = None, ROI = None,
                    prf_model_name = None, rsq_threshold = None, file_ext = '_cropped_confound_psc.npy', 
                    outdir = None, save_estimates = False, fit_overlap = False, fit_full_stim = True, n_jobs = 8):

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
        
        ## set up fitting by loading data and visual dm for data
        masked_data, masked_prf_estimates = self.setup_vars4fitting(participant, pp_prf_estimates, ses = ses,
                                                            run_type = run_type, chunk_num = chunk_num, vertex = vertex, ROI = ROI,
                                                            prf_model_name = prf_model_name, file_ext = file_ext, 
                                                            outdir = outdir, fit_overlap = fit_overlap, fit_full_stim = fit_full_stim)

        ## initialize prf params for fitting
        self.initialize_parameters(masked_prf_estimates, prf_pars2vary = [], prf_bounds = None, rsq_threshold = rsq_threshold)

        ## if provided, get nuisance regressors
        if self.add_nuisance_reg:

            # get ses and run number 
            run_num, ses_num = mri_utils.get_run_ses_from_str(self.train_file_list[0]) ## assumes we are fitting one run, will need to change later if such is the case

            confounds_df = self.load_nuisance_df(participant, run_num = run_num, ses_num = ses_num)
        else:
            confounds_df = []


        ## actually fit vertices
        # and output relevant params + rsq of model fit in dataframe
        results = np.array(Parallel(n_jobs=n_jobs)(delayed(self.fit_glm)(masked_data[0,vert,:], ## hardcoded to use first run (which is only one), should change later
                                                                                self.pars_arr[ind],
                                                                                nuisances_df = confounds_df,
                                                                                bar_keys = ['att_bar', 'unatt_bar'],
                                                                                return_model_tc = False)
                                                                            for ind, vert in enumerate(tqdm(self.ind2fit))))

        ## saves results as list of dataframes, with length = #runs
        results_list = []
        
        for r, name in enumerate(self.visual_dm_dict.keys()):
            
            # convert fitted params list of dicts as Dataframe
            fitted_params_df = pd.DataFrame(d for d in results)
            # and add vertex number for bookeeping
            if vertex is not None:
                fitted_params_df['vertex'] = vertex
            else:
                fitted_params_df['vertex'] = self.ind2fit

            results_list.append(fitted_params_df.copy())

            # if we want to save estimates, do so as csv
            if save_estimates:
                filename = self.basefilename.replace('runtype','run-{n}_runtype'.format(n = name))
                filename = filename.replace('.npz', '_it_{pm}_estimates.csv'.format(pm = prf_model_name))

                # if we are fitting hrf, include that in name
                if self.fit_hrf:
                    filename = filename.replace('estimates.csv', 'HRF_estimates.csv')

                fitted_params_df.to_csv(op.join(self.outdir, filename))
                
        return results_list

    
    def load_nuisance_df(self, participant, run_num = 1, ses_num = 1):

        ## first select confounds for run
        # path to post fmriprep files
        conf_path = op.join(self.MRIObj.derivatives_pth, 'post_fmriprep', self.MRIObj.sj_space, 
                                'sub-{sj}'.format(sj = participant), 'ses-{s}'.format(s = ses_num))

        # get post fmriprep confound ext (will alway have select_cropped in name)
        confound_ext = self.MRIObj.confound_ext.replace('.{ext}'.format(ext=self.MRIObj.confound_ext.split('.')[-1]), 
                                                        '_select_cropped.{ext}'.format(ext=self.MRIObj.confound_ext.split('.')[-1]))

        confound_files = [op.join(conf_path, file) for file in os.listdir(conf_path) if 'acq-{a}'.format(a=self.MRIObj.acq) in file \
                    and 'task-FA' in file and file.endswith(confound_ext) and 'run-{r}'.format(r = run_num) in file]

        print('Loading counfound file %s'%confound_files[0])

        ## load confound DataFrame
        if len(confound_files)>1:
            raise ValueError('More than 1 confound file found, conflict %s'%str(confound_files)) # again, assumes only one run dangerous
        else:
            confounds_df = pd.read_csv(confound_files[0], sep="\t") 
        
        # name of nuisance regressors
        self.nuisance_reg_names = list(confounds_df.keys())

        return confounds_df

    
    def get_fa_glm_dm(self, tc_dict, nuisances_df = [], bar_keys = ['att_bar', 'unatt_bar']):

        ## turn parameters into arrays 
        # but save dict keys to guarantee order is correct
        parameters_keys = list(tc_dict.keys())

        # set parameters and bounds into list, to conform to scipy minimze format
        tc_pars = [tc_dict[key] for key in parameters_keys]

        ## Make DM        
        design_matrix = []
        all_regressor_names = []

        ## get task related regressor timecourse
        for reg in self.task_reg_names:

            design_matrix.append(self.get_fit_timecourse(tc_pars, reg_name = reg, bar_keys = bar_keys, parameters_keys = parameters_keys)[0])
            all_regressor_names.append(reg)

        ## add confounds if they exist
        if self.nuisance_reg_names is not None:

            for reg in self.nuisance_reg_names:

                design_matrix.append(nuisances_df[reg].values)
                all_regressor_names.append(reg)

        ## finally add intercept
        design_matrix.append(np.ones(np.array(design_matrix).shape[-1]))
        all_regressor_names.append('intercept')

        return np.array(design_matrix), all_regressor_names


    def fit_glm(self, train_timecourse, tc_dict, nuisances_df = [], bar_keys = ['att_bar', 'unatt_bar'], return_model_tc = False):

        design_matrix, all_regressor_names = self.get_fa_glm_dm(tc_dict, nuisances_df = nuisances_df, bar_keys = bar_keys)

        ## Fit the GLM
        prediction, betas, r2, _ = mri_utils.fit_glm(train_timecourse, design_matrix.T, error='mse')

        # set output params as dict
        out_dict = tc_dict.copy()
        for ind, key in enumerate(all_regressor_names):
            out_dict[key] = betas[ind]

        # add rsq value
        out_dict['r2'] = r2

        if return_model_tc:
            return out_dict, prediction
        else:
            return out_dict



