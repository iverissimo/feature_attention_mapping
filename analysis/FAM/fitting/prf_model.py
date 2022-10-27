import numpy as np
import os
import os.path as op
import pandas as pd
import yaml

from PIL import Image, ImageDraw

import cortex

from scipy.optimize import LinearConstraint, NonlinearConstraint

from FAM.utils import mri as mri_utils
from FAM.processing import preproc_behdata
from FAM.fitting.model import Model

from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter, Norm_Iso2DGaussianFitter, DoG_Iso2DGaussianFitter


class pRF_model(Model):

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
            self.outputdir = op.join(self.MRIObj.derivatives_pth,'pRF_fit')
        else:
            self.outputdir = outputdir
        
        # reset osf value, because model assumes 10 (for FA)
        self.osf = 1
            
    
    def get_DM(self, participant, ses = 'ses-mean', ses_type = 'func', mask_DM = True, filename = None, 
                                    osf = 1, res_scaling = .1):

        """
        Get pRF Design matrix

        Parameters
        ----------
        participant : str
            participant number
        ses : str
            session number (default ses-mean)
        ses_type: str
            type of session (default func)
        mask_DM:
            if we want to mask design matrix based on behavior
        filename: str
            absolute path to np file where we stored design matrix, if none it will make one anew
        osf: int
            oversampling factor, if bigger than one it will return DM of timepoints * osf
        res_scaling: float
            spatial rescaling factor

        """ 

        visual_dm = None
        save_dm = False

        if filename:
            if op.exists(filename):
                print('Loading {file}'.format(file = filename))
                visual_dm = np.load(filename)
            else:
                save_dm = True
        
        # make design matrix
        if visual_dm is None:

            print('Making DM for sub-{pp}'.format(pp = participant))
            
            ## get behavioral info 
            mri_beh = preproc_behdata.PreprocBeh(self.MRIObj)

            ## get boolean array of moments where bar was on screen
            stim_on_screen = np.zeros(mri_beh.pRF_total_trials)
            stim_on_screen[mri_beh.pRF_bar_pass_trials] = 1

            ## if we want to mask DM, then load behav mask
            if mask_DM:
                mask_bool_df = mri_beh.get_pRF_mask_bool(ses_type = ses_type)
                # if we set a specific session, then select that one
                if ses == 'ses-mean':
                    mask_bool = mask_bool_df[mask_bool_df['sj'] == 'sub-{sj}'.format(sj = participant)]['mask_bool'].values
                else:
                    mask_bool = mask_bool_df[(mask_bool_df['ses'] == ses) & \
                                        (mask_bool_df['sj'] == 'sub-{sj}'.format(sj = participant))]['mask_bool'].values
                dm_mask = np.prod(mask_bool, axis = 0)
            else:
                dm_mask = np.ones(mri_beh.pRF_total_trials)

            # multiply boolean array with mask
            stim_on_screen = stim_on_screen * dm_mask
                
            ## crop and shift if such was the case
            stim_on_screen = mri_utils.crop_shift_arr(stim_on_screen, 
                                                        crop_nr = self.crop_TRs_num['pRF'], 
                                                        shift = self.shift_TRs_num)
            # do same to bar pass direction str array
            condition_per_TR = mri_utils.crop_shift_arr(mri_beh.pRF_bar_pass_all, 
                                                        crop_nr = self.crop_TRs_num['pRF'], 
                                                        shift = self.shift_TRs_num)

            # all possible positions in pixels for for midpoint of
            # y position for vertical bar passes, 
            ver_y = self.screen_res[1]*np.linspace(0,1, self.MRIObj.pRF_nr_TRs['U-D'])
            # x position for horizontal bar passes 
            hor_x = self.screen_res[0]*np.linspace(0,1, self.MRIObj.pRF_nr_TRs['L-R'])

            # coordenates for bar pass, for PIL Image
            coordenates_bars = {'L-R': {'upLx': hor_x - 0.5 * self.bar_width['pRF'] * self.screen_res[0], 'upLy': np.repeat(self.screen_res[1], self.MRIObj.pRF_nr_TRs['L-R']),
                                        'lowRx': hor_x + 0.5 * self.bar_width['pRF'] * self.screen_res[0], 'lowRy': np.repeat(0, self.MRIObj.pRF_nr_TRs['L-R'])},
                                'R-L': {'upLx': np.array(list(reversed(hor_x - 0.5 * self.bar_width['pRF'] * self.screen_res[0]))), 'upLy': np.repeat(self.screen_res[1], self.MRIObj.pRF_nr_TRs['R-L']),
                                        'lowRx': np.array(list(reversed(hor_x+ 0.5 * self.bar_width['pRF'] * self.screen_res[0]))), 'lowRy': np.repeat(0, self.MRIObj.pRF_nr_TRs['R-L'])},
                                'U-D': {'upLx': np.repeat(0, self.MRIObj.pRF_nr_TRs['U-D']), 'upLy': ver_y+0.5 * self.bar_width['pRF'] * self.screen_res[1],
                                        'lowRx': np.repeat(self.screen_res[0], self.MRIObj.pRF_nr_TRs['U-D']), 'lowRy': ver_y - 0.5 * self.bar_width['pRF'] * self.screen_res[1]},
                                'D-U': {'upLx': np.repeat(0, self.MRIObj.pRF_nr_TRs['D-U']), 'upLy': np.array(list(reversed(ver_y + 0.5 * self.bar_width['pRF'] * self.screen_res[1]))),
                                        'lowRx': np.repeat(self.screen_res[0], self.MRIObj.pRF_nr_TRs['D-U']), 'lowRy': np.array(list(reversed(ver_y - 0.5 * self.bar_width['pRF'] * self.screen_res[1])))}
                                }

            # save screen display for each TR (or if osf > 1 then for #TRs * osf)
            visual_dm_array = np.zeros((len(condition_per_TR) * osf, round(self.screen_res[0] * res_scaling), round(self.screen_res[1] * res_scaling)))
            i = 0

            for trl, bartype in enumerate(condition_per_TR): # loop over bar pass directions

                img = Image.new('RGB', tuple(self.screen_res)) # background image

                if bartype not in np.array(['empty','empty_long']): # if not empty screen

                    #print(bartype)

                    # set draw method for image
                    draw = ImageDraw.Draw(img)
                    # add bar, coordinates (upLx, upLy, lowRx, lowRy)
                    draw.rectangle(tuple([coordenates_bars[bartype]['upLx'][i],coordenates_bars[bartype]['upLy'][i],
                                        coordenates_bars[bartype]['lowRx'][i],coordenates_bars[bartype]['lowRy'][i]]), 
                                fill = (255,255,255),
                                outline = (255,255,255))

                    # increment counter
                    if trl < (len(condition_per_TR) - 1):
                        i = i+1 if condition_per_TR[trl] == condition_per_TR[trl+1] else 0    
                
                ## save in array, and apply mask
                visual_dm_array[int(trl*osf):int(trl*osf + osf), ...] = np.array(img)[::round(1/res_scaling),::round(1/res_scaling),0][np.newaxis,...] * stim_on_screen[trl]

            # swap axis to have time in last axis [x,y,t]
            visual_dm = visual_dm_array.transpose([1,2,0])

            if save_dm:
                # save design matrix
                print('Making and saving {file}'.format(file = filename))
                np.save(filename, visual_dm)  
                    
        return mri_utils.normalize(visual_dm)


    def set_models(self, participant_list = [], mask_DM = True, combine_ses = True):

        """
        define pRF models to be used for each participant in participant list
                
        Parameters
        ----------
        participant_list: list
            list with participant ID
        mask_DM: bool
            if we want to mask design matrix given behavioral performance
        combine_ses: bool
            if we want to combine runs from different sessions (relevant for fitting of average across runs)
        """                 

        ## loop over participants

        ## if no participant list set, then run all
        if len(participant_list) == 0:
            participant_list = self.MRIObj.sj_num
        
        # empty dict where we'll store all participant models
        pp_models = {}
        
        for pp in participant_list:

            pp_models['sub-{sj}'.format(sj=pp)] = {}

            # if we're combining sessions
            if combine_ses:
                sessions = ['ses-mean']
            else:
                sessions = self.MRIObj.session['sub-{sj}'.format(sj=pp)]

            ## go over sessions (if its the case)
            # and save DM and models
            for ses in sessions:

                pp_models['sub-{sj}'.format(sj=pp)][ses] = {}

                visual_dm = self.get_DM(pp, ses = ses, ses_type = 'func', mask_DM = mask_DM, 
                                        filename = None, osf = self.osf, res_scaling = self.res_scaling)

                # make stimulus object, which takes an input design matrix and sets up its real-world dimensions
                prf_stim = PRFStimulus2D(screen_size_cm = self.MRIObj.params['monitor']['height'],
                                        screen_distance_cm = self.MRIObj.params['monitor']['distance'],
                                        design_matrix = visual_dm,
                                        TR = self.MRIObj.TR)

                pp_models['sub-{sj}'.format(sj=pp)][ses]['prf_stim'] = prf_stim
                                
                ## define models ##
                # GAUSS
                gauss_model = Iso2DGaussianModel(stimulus = prf_stim,
                                                    filter_predictions = True,
                                                    filter_type = self.MRIObj.params['mri']['filtering']['type']['pRF'],
                                                    filter_params = {'highpass': self.MRIObj.params['mri']['filtering']['highpass'],
                                                                    'add_mean': self.MRIObj.params['mri']['filtering']['add_mean'],
                                                                    'window_length': self.MRIObj.params['mri']['filtering']['window_length'],
                                                                    'polyorder': self.MRIObj.params['mri']['filtering']['polyorder']},
                                                    osf = self.osf,
                                                    hrf_onset = self.hrf_onset,
                                                )

                pp_models['sub-{sj}'.format(sj=pp)][ses]['gauss_model'] = gauss_model

                # CSS
                css_model = CSS_Iso2DGaussianModel(stimulus = prf_stim,
                                                    filter_predictions = True,
                                                    filter_type = self.MRIObj.params['mri']['filtering']['type']['pRF'],
                                                    filter_params = {'highpass': self.MRIObj.params['mri']['filtering']['highpass'],
                                                                    'add_mean': self.MRIObj.params['mri']['filtering']['add_mean'],
                                                                    'window_length': self.MRIObj.params['mri']['filtering']['window_length'],
                                                                    'polyorder': self.MRIObj.params['mri']['filtering']['polyorder']},
                                                    osf = self.osf,
                                                    hrf_onset = self.hrf_onset,
                                                )

                pp_models['sub-{sj}'.format(sj=pp)][ses]['css_model'] = css_model

                # DN 
                dn_model =  Norm_Iso2DGaussianModel(stimulus = prf_stim,
                                                    filter_predictions = True,
                                                    filter_type = self.MRIObj.params['mri']['filtering']['type']['pRF'],
                                                    filter_params = {'highpass': self.MRIObj.params['mri']['filtering']['highpass'],
                                                                    'add_mean': self.MRIObj.params['mri']['filtering']['add_mean'],
                                                                    'window_length': self.MRIObj.params['mri']['filtering']['window_length'],
                                                                    'polyorder': self.MRIObj.params['mri']['filtering']['polyorder']},
                                                    osf = self.osf,
                                                    hrf_onset = self.hrf_onset,
                                                )

                pp_models['sub-{sj}'.format(sj=pp)][ses]['dn_model'] = dn_model

                # DOG
                dog_model = DoG_Iso2DGaussianModel(stimulus = prf_stim,
                                                    filter_predictions = True,
                                                    filter_type = self.MRIObj.params['mri']['filtering']['type']['pRF'],
                                                    filter_params = {'highpass': self.MRIObj.params['mri']['filtering']['highpass'],
                                                                    'add_mean': self.MRIObj.params['mri']['filtering']['add_mean'],
                                                                    'window_length': self.MRIObj.params['mri']['filtering']['window_length'],
                                                                    'polyorder': self.MRIObj.params['mri']['filtering']['polyorder']},
                                                    osf = self.osf,
                                                    hrf_onset = self.hrf_onset,
                                                )
                
                pp_models['sub-{sj}'.format(sj=pp)][ses]['dog_model'] = dog_model


        return pp_models


    def fit_data(self, participant, pp_models, ses = 'ses-mean',
                    run_type = 'mean', chunk_num = None, vertex = None, ROI = None,
                    model2fit = 'gauss', file_ext = '_cropped_dc_psc.npy', 
                    outdir = None, save_estimates = False,
                    xtol = 1e-3, ftol = 1e-4, n_jobs = 16):

        """
        fit inputted pRF models to each participant in participant list
                
        Parameters
        ----------
        participant_list: list
            list with participant ID
        input_pth: str or None
            path to look for files, if None then will get them from derivatives/postfmriprep/<space>/sub-X folder
        run_type: string or int
            type of run to fit, mean (default), or if int will do single run fit
        file_ext: dict
            file extension, to select appropriate files
        mask_DM: bool
            if we want to mask design matrix given behavioral performance
        """  

        ## get list of files to load
        bold_filelist = self.get_bold_file_list(participant, task = 'pRF', ses = ses, file_ext = file_ext)
        
        ## Load data array
        data = self.get_data4fitting(bold_filelist, task = 'pRF', run_type = run_type, chunk_num = chunk_num, vertex = vertex, 
                                    baseline_interval = 'empty_long', ses = ses, return_filenames = False)

        ## Set nan voxels to 0, to avoid issues when fitting
        masked_data = data.copy()
        masked_data[np.where(np.isnan(data[...,0]))[0]] = 0

        ## set output dir to save estimates
        if outdir is None:
            if 'loo_' in run_type:
                outdir = op.join(self.MRIObj.derivatives_pth, 'pRF_fit', self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant), run_type)
            else:
                outdir = op.join(self.MRIObj.derivatives_pth, 'pRF_fit', self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant), ses)
            
        os.makedirs(outdir, exist_ok = True)
        print('saving files in %s'%outdir)

        ## set base filename that will be used for estimates
        basefilename = 'sub-{sj}_task-pRF_acq-{acq}_runtype-{rt}'.format(sj = participant,
                                                                            acq = self.MRIObj.acq,
                                                                            rt = run_type)
        if chunk_num is not None:
            basefilename += '_chunk-{ch}'.format(ch = str(chunk_num).zfill(3))
        elif vertex is not None:
            basefilename += '_vertex-{ver}'.format(ver = str(vertex))
        elif ROI:
            basefilename += '_ROI-{roi}'.format(roi = str(ROI))
        
        basefilename += file_ext.replace('.npy', '.npz')

        ## set model parameters 
        # relevant for grid and iterative fitting
        fit_params = self.get_fit_startparams(max_ecc_size = pp_models['sub-{sj}'.format(sj = participant)][ses]['prf_stim'].screen_size_degrees/2.0)

        ## set constraints
        # for now only changes minimizer used, but can also be useful to put contraints on dog and dn
        constraints = self.get_fit_constraints(method = self.optimizer['pRF'], ss_larger_than_centre = True, 
                                                positive_centre_only = True, normalize_RFs = False)

        ## ACTUALLY FIT 

        # always start with gauss of course
        grid_gauss_filename = op.join(outdir, 'grid_gauss', basefilename.replace('.npz', '_grid_gauss_estimates.npz'))
        it_gauss_filename = op.join(outdir, 'it_gauss', basefilename.replace('.npz', '_it_gauss_estimates.npz'))

        # if we want to fit hrf, change output name
        if self.fit_hrf:
            grid_gauss_filename = grid_gauss_filename.replace('_estimates.npz', '_HRF_estimates.npz')
            it_gauss_filename = it_gauss_filename.replace('_estimates.npz', '_HRF_estimates.npz')

        # already set other model name
        grid_model_filename = grid_gauss_filename.replace('gauss', model2fit)
        it_model_filename = it_gauss_filename.replace('gauss', model2fit)

        if not op.isfile(it_model_filename):

            print("Gauss model GRID fit")
            gauss_fitter = Iso2DGaussianFitter(data = masked_data, 
                                                model = pp_models['sub-{sj}'.format(sj = participant)][ses]['gauss_model'], 
                                                n_jobs = n_jobs,
                                                fit_hrf = self.fit_hrf)

            gauss_fitter.grid_fit(ecc_grid = fit_params['gauss']['eccs'], 
                                    polar_grid = fit_params['gauss']['polars'], 
                                    size_grid = fit_params['gauss']['sizes'], 
                                    fixed_grid_baseline = fit_params['gauss']['fixed_grid_baseline'],
                                    grid_bounds = fit_params['gauss']['grid_bounds'])

            # iterative fit
            print("Gauss model ITERATIVE fit")

            # if self.fit_hrf:
            #     # set hrf here, for now - grid gauss doesnt fit hrf params (need to raise issue in prfpy)
            #     pp_models['sub-{sj}'.format(sj = participant)][ses]['gauss_model'].hrf = np.array([1,1,0])

            gauss_fitter.iterative_fit(rsq_threshold = 0.05, 
                                        verbose = True,
                                        bounds = fit_params['gauss']['bounds'],
                                        constraints = constraints['gauss'],
                                        #starting_params = gauss_fitter.gridsearch_params,
                                        xtol = xtol,
                                        ftol = ftol)

            # if we want to save estimates
            if save_estimates and not op.isfile(it_gauss_filename):
                # for grid
                print('saving %s'%grid_gauss_filename)
                self.save_pRF_model_estimates(grid_gauss_filename, gauss_fitter.gridsearch_params, 
                                                model_type = 'gauss', grid = True) 
                # for it
                print('saving %s'%it_gauss_filename)
                self.save_pRF_model_estimates(it_gauss_filename, gauss_fitter.iterative_search_params, 
                                                model_type = 'gauss')

            if model2fit != 'gauss':

                print("{key} model GRID fit".format(key = model2fit))
                
                if model2fit == 'css':

                    fitter = CSS_Iso2DGaussianFitter(data = masked_data, 
                                                    model = pp_models['sub-{sj}'.format(sj = participant)][ses]['{key}_model'.format(key = model2fit)], 
                                                    n_jobs = n_jobs,
                                                    fit_hrf = self.fit_hrf,
                                                    previous_gaussian_fitter = gauss_fitter)

                    fitter.grid_fit(fit_params['css']['n_grid'],
                                fixed_grid_baseline = fit_params['css']['fixed_grid_baseline'],
                                grid_bounds = fit_params['css']['grid_bounds'],
                                rsq_threshold = 0.05)
                
                elif model2fit == 'dn':

                    fitter = Norm_Iso2DGaussianFitter(data = masked_data, 
                                                    model = pp_models['sub-{sj}'.format(sj = participant)][ses]['{key}_model'.format(key = model2fit)], 
                                                    n_jobs = n_jobs,
                                                    fit_hrf = self.fit_hrf,
                                                    previous_gaussian_fitter = gauss_fitter)

                    fitter.grid_fit(fit_params['dn']['surround_amplitude_grid'],
                                    fit_params['dn']['surround_size_grid'],
                                    fit_params['dn']['neural_baseline_grid'],
                                    fit_params['dn']['surround_baseline_grid'],
                                fixed_grid_baseline = fit_params['dn']['fixed_grid_baseline'],
                                grid_bounds = fit_params['dn']['grid_bounds'],
                                rsq_threshold = 0.05)

                
                elif model2fit == 'dog':

                    fitter = DoG_Iso2DGaussianFitter(data = masked_data, 
                                                    model = pp_models['sub-{sj}'.format(sj = participant)][ses]['{key}_model'.format(key = model2fit)], 
                                                    n_jobs = n_jobs,
                                                    fit_hrf = self.fit_hrf,
                                                    previous_gaussian_fitter = gauss_fitter)

                    fitter.grid_fit(fit_params['dog']['surround_amplitude_grid'],
                                    fit_params['dog']['surround_size_grid'],
                                fixed_grid_baseline = fit_params['dog']['fixed_grid_baseline'],
                                grid_bounds = fit_params['dog']['grid_bounds'],
                                rsq_threshold = 0.05)


                # iterative fit
                print("{key} model ITERATIVE fit".format(key = model2fit))

                fitter.iterative_fit(rsq_threshold = 0.05, 
                                    verbose = True,
                                    bounds = fit_params[model2fit]['bounds'],
                                    constraints = constraints[model2fit],
                                    #starting_params = fitter.gridsearch_params, 
                                    xtol = xtol,
                                    ftol = ftol)

                # if we want to save estimates
                if save_estimates:
                    # for grid
                    print('saving %s'%grid_model_filename)
                    self.save_pRF_model_estimates(grid_model_filename, fitter.gridsearch_params, 
                                                    model_type = model2fit, grid = True)
                    # for it
                    print('saving %s'%it_model_filename)
                    self.save_pRF_model_estimates(it_model_filename, fitter.iterative_search_params, 
                                                    model_type = model2fit)

        if not save_estimates:
            # if we're not saving them, assume we are running on the spot
            # and want to get back the estimates
            estimates = {}
            estimates['grid_gauss'] = gauss_fitter.gridsearch_params
            estimates['it_gauss'] = gauss_fitter.iterative_search_params
            if model2fit != 'gauss':
                estimates['grid_{key}'.format(key = model2fit)] = fitter.gridsearch_params
                estimates['it_{key}'.format(key = model2fit)] = fitter.iterative_search_params

            return estimates, masked_data


    def get_fit_startparams(self, max_ecc_size = 6):

        """
        Helper function that loads all fitting starting params
        and bounds into a dictionary

        Parameters
        ----------
        max_ecc_size: int/float
            max eccentricity (and size) to set grid array
        """

        eps = 1e-1

        fitpar_dict = {'gauss': {}, 'css': {}, 'dn': {}, 'dog': {}}

        ######################### GAUSS #########################

        ## number of grid points 
        fitpar_dict['gauss']['grid_nr'] = self.MRIObj.params['mri']['fitting']['pRF']['grid_nr']

        # size, ecc, polar angle
        fitpar_dict['gauss']['sizes'] = max_ecc_size * np.linspace(0.25, 1, fitpar_dict['gauss']['grid_nr'])**2
        fitpar_dict['gauss']['eccs'] = max_ecc_size * np.linspace(0.1, 1, fitpar_dict['gauss']['grid_nr'])**2
        fitpar_dict['gauss']['polars'] = np.linspace(0, 2*np.pi, fitpar_dict['gauss']['grid_nr'])

        ## bounds
        fitpar_dict['gauss']['bounds'] = [(-1.5 * (max_ecc_size * 2), 1.5 * (max_ecc_size * 2)),  # x
                                        (-1.5 * (max_ecc_size * 2), 1.5 * (max_ecc_size * 2)),  # y
                                        (eps, 1.5 * (max_ecc_size * 2)),  # prf size
                                        (0, 1000),  # prf amplitude
                                        (-500, 1000)] # bold baseline
        
        fitpar_dict['gauss']['fixed_grid_baseline'] = None

        # latest change in prfpy requires separate grid bound array
        fitpar_dict['gauss']['grid_bounds'] = [(0,1000)] #only prf amplitudes between 0 and 1000

        ######################### CSS #########################

        ## grid exponent parameter
        fitpar_dict['css']['n_grid'] = np.linspace(self.MRIObj.params['mri']['fitting']['pRF']['min_n'], 
                                                    self.MRIObj.params['mri']['fitting']['pRF']['max_n'], 
                                                    self.MRIObj.params['mri']['fitting']['pRF']['n_nr'], dtype='float32')

        ## bounds
        fitpar_dict['css']['bounds'] = [(-1.5 * (max_ecc_size * 2), 1.5 * (max_ecc_size * 2)),  # x
                                        (-1.5 * (max_ecc_size * 2), 1.5 * (max_ecc_size * 2)),  # y
                                        (eps, 1.5 * (max_ecc_size * 2)),  # prf size
                                        (0, 1000),  # prf amplitude
                                        (-500, 1000), # bold baseline
                                        (0.01, 1.5)]  # CSS exponent

        fitpar_dict['css']['fixed_grid_baseline'] = None

        # latest change in prfpy requires separate grid bound array
        fitpar_dict['css']['grid_bounds'] = [(0,1000)] #only prf amplitudes between 0 and 1000

        ######################### DN #########################

        # Surround amplitude (Normalization parameter C)
        fitpar_dict['dn']['surround_amplitude_grid'] = np.array([0.1,0.2,0.4,0.7,1,3], dtype='float32') 
        
        # Surround size (gauss sigma_2)
        fitpar_dict['dn']['surround_size_grid'] = np.array([3,5,8,12,18], dtype='float32')
        
        # Neural baseline (Normalization parameter B)
        fitpar_dict['dn']['neural_baseline_grid'] = np.array([0,1,10,100], dtype='float32')

        # Surround baseline (Normalization parameter D)
        fitpar_dict['dn']['surround_baseline_grid'] = np.array([0.1,1.0,10.0,100.0], dtype='float32')

        ## bounds
        fitpar_dict['dn']['bounds'] = [(-1.5 * (max_ecc_size * 2), 1.5 * (max_ecc_size * 2)),  # x
                                        (-1.5 * (max_ecc_size * 2), 1.5 * (max_ecc_size * 2)),  # y
                                        (eps, 1.5 * (max_ecc_size * 2)),  # prf size
                                        (0, 1000),  # prf amplitude
                                        (-500, 1000), # bold baseline
                                        (0, 1000),  # surround amplitude
                                        (eps, 3 * (max_ecc_size * 2)),  # surround size
                                        (0, 1000),  # neural baseline
                                        (1e-6, 1000)]  # surround baseline

        fitpar_dict['dn']['fixed_grid_baseline'] = None

        # latest change in prfpy requires separate grid bound array
        fitpar_dict['dn']['grid_bounds'] = [(0,1000),(0,1000)] #only prf amplitudes between 0 and 1000

        ######################### DOG #########################
        
        # amplitude for surround 
        fitpar_dict['dog']['surround_amplitude_grid'] = np.array([0.05,0.1,0.25,0.5,0.75,1,2], dtype='float32')

        # size for surround
        fitpar_dict['dog']['surround_size_grid'] = np.array([3,5,8,11,14,17,20,23,26], dtype='float32')

        ## bounds
        fitpar_dict['dog']['bounds'] = [(-1.5 * (max_ecc_size * 2), 1.5 * (max_ecc_size * 2)),  # x
                                        (-1.5 * (max_ecc_size * 2), 1.5 * (max_ecc_size * 2)),  # y
                                        (eps, 1.5 * (max_ecc_size * 2)),  # prf size
                                        (0, 1000),  # prf amplitude
                                        (-500, 1000), # bold baseline
                                        (0, 1000),  # surround amplitude
                                        (eps, 3 * (max_ecc_size * 2))]  # surround size

        fitpar_dict['dog']['fixed_grid_baseline'] = None

        # latest change in prfpy requires separate grid bound array
        fitpar_dict['dog']['grid_bounds'] = [(0,1000),(0,1000)] #only prf amplitudes between 0 and 1000

        ### EXTRA ###

        # if we want to also fit hrf
        if self.fit_hrf:
            fitpar_dict['gauss']['bounds'] += [(0,10),(0,0)]
            fitpar_dict['css']['bounds'] += [(0,10),(0,0)]
            fitpar_dict['dn']['bounds'] += [(0,10),(0,0)]
            fitpar_dict['dog']['bounds'] += [(0,10),(0,0)]
        
        # if we want to keep the baseline fixed at 0
        if self.fix_bold_baseline['pRF']:
            fitpar_dict['gauss']['bounds'][4] = (0,0)
            fitpar_dict['gauss']['fixed_grid_baseline'] = 0 
            
            fitpar_dict['css']['bounds'][4] = (0,0)
            fitpar_dict['css']['fixed_grid_baseline'] = 0 

            fitpar_dict['dn']['bounds'][4] = (0,0)
            fitpar_dict['dn']['fixed_grid_baseline'] = 0 

            fitpar_dict['dog']['bounds'][4] = (0,0)
            fitpar_dict['dog']['fixed_grid_baseline'] = 0 

                                        
        return fitpar_dict

    
    def get_fit_constraints(self, method = 'L-BFGS-B', ss_larger_than_centre = True, 
                        positive_centre_only = False, normalize_RFs = False):

        """
        Helper function sets constraints - which depend on minimizer used -
        for all model types and saves in dictionary

        Parameters
        ----------
        method: str
            minimizer that we want to use, ex: 'L-BFGS-B', 'trust-constr'

        """

        constraints = {'gauss': {}, 'css': {}, 'dn': {}, 'dog': {}}

        for key in constraints.keys():

            if method == 'L-BFGS-B':
                
                constraints[key] = None
            
            elif method == 'trust-constr':
                
                constraints[key] = []
                
                if 'dn' in key:
                    if ss_larger_than_centre:
                        #enforcing surround size larger than prf size
                        if self.fit_hrf:
                            A_ssc_norm = np.array([[0,0,-1,0,0,0,1,0,0,0,0]])
                        else:
                            A_ssc_norm = np.array([[0,0,-1,0,0,0,1,0,0]])
                            
                        constraints[key].append(LinearConstraint(A_ssc_norm,
                                                lb=0,
                                                ub=np.inf))
                        
                    if positive_centre_only:
                        #enforcing positive central amplitude in norm
                        def positive_centre_prf_norm(x):
                            if normalize_RFs:
                                return (x[3]/(2*np.pi*x[2]**2)+x[7])/(x[5]/(2*np.pi*x[6]**2)+x[8]) - x[7]/x[8]
                            else:
                                return (x[3]+x[7])/(x[5]+x[8]) - x[7]/x[8]

                        constraints[key].append(NonlinearConstraint(positive_centre_prf_norm,
                                                                    lb=0,
                                                                    ub=np.inf))
                elif 'dog' in key:
                    if ss_larger_than_centre:
                        #enforcing surround size larger than prf size
                        if self.fit_hrf:
                             A_ssc_dog = np.array([[0,0,-1,0,0,0,1,0,0]])
                        else:
                            A_ssc_dog = np.array([[0,0,-1,0,0,0,1]])
                            
                        constraints[key].append(LinearConstraint(A_ssc_dog,
                                                lb=0,
                                                ub=np.inf))
                        
                    if positive_centre_only:
                        #enforcing positive central amplitude in DoG
                        def positive_centre_prf_dog(x):
                            if normalize_RFs:
                                return x[3]/(2*np.pi*x[2]**2)-x[5]/(2*np.pi*x[6]**2)
                            else:
                                return x[3] - x[5]

                        constraints[key].append(NonlinearConstraint(positive_centre_prf_dog,
                                                                    lb=0,
                                                                    ub=np.inf))

        return constraints

    
    def load_pRF_model_estimates(self, participant, ses = 'ses-mean', run_type = 'mean', model_name = None, iterative = True, fit_hrf = False):

        """
        Helper function to load pRF model estimates
        when they already where fitted and save in out folder

        Parameters
        ----------
        participant: str
            participant ID
        ses: str
            session we are looking at
        run_type: str
            run type we fitted
        model_name: str or None
            model name to be loaded (if None defaults to class model)
        iterative: bool
            if we want to load iterative fitting results [default] or grid results

        """

        # if model name to load not given, use the one set in the class
        if model_name:
            model_name = model_name
        else:
            model_name = self.model_type['pRF']

        # if we want to load iterative results, or grid (iterative = False)
        if iterative:
            est_folder = 'it_{model_name}'.format(model_name = model_name)
        else:
            est_folder = 'grid_{model_name}'.format(model_name = model_name)
        
        # if not doing mean across session, then set combine ses to false
        combine_ses = True if ses == 'ses-mean' else False 

        # get participant models, which also will load 
        # DM and mask it according to participants behavior
        pp_prf_models = self.set_models(participant_list = [participant], 
                                                    mask_DM = True, combine_ses = combine_ses)

        ## load estimates to make it easier to load later
        if 'loo_' in run_type:
            pRFdir = op.join(self.MRIObj.derivatives_pth, 'pRF_fit', self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant), run_type, est_folder)
        else:
            pRFdir = op.join(self.MRIObj.derivatives_pth, 'pRF_fit', self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant), ses, est_folder)


        pp_prf_est_dict = self.load_pRF_model_chunks(pRFdir, 
                                                    fit_model = model_name,
                                                    basefilename = 'sub-{sj}_task-pRF_acq-{acq}_runtype-{rt}'.format(sj = participant,
                                                                                                                acq = self.MRIObj.acq,
                                                                                                                rt = run_type),
                                                    fit_hrf = fit_hrf,
                                                    iterative = iterative)

        return pp_prf_est_dict, pp_prf_models


    def mask_pRF_model_estimates(self, estimates, ROI = None, x_ecc_lim = [-6,6], y_ecc_lim = [-6,6],
                                rsq_threshold = .1, pysub = 'hcp_999999', estimate_keys = ['x','y','size','betas','baseline','r2']):
    
        """ 
        mask estimates, to be positive RF, within screen limits
        and for a certain ROI (if the case)

        Parameters
        ----------
        estimates : dict
            dict with estimates key-value pairs
        ROI : str
            roi to mask estimates (eg. 'V1', default None)
        estimate_keys: list/arr
            list or array of strings with keys of estimates to mask
        
        Outputs
        -------
        masked_estimates : npz 
            numpy array of masked estimates
        
        """
        
        # make new variables that are masked 
        masked_dict = {}
        
        for k in estimate_keys: 
            masked_dict[k] = np.zeros(estimates[k].shape)
            masked_dict[k][:] = np.nan

        
        # set limits for xx and yy, forcing it to be within the screen boundaries
        # also for positive pRFs

        indices = np.where((~np.isnan(estimates['r2']))& \
                            (estimates['r2']>= rsq_threshold)& \
                        (estimates['x'] <= np.max(x_ecc_lim))& \
                        (estimates['x'] >= np.min(x_ecc_lim))& \
                        (estimates['y'] <= np.max(y_ecc_lim))& \
                        (estimates['y'] >= np.min(y_ecc_lim))& \
                        (estimates['betas']>=0)
                        )[0]
                            
        # save values
        for k in estimate_keys:
            masked_dict[k][indices] = estimates[k][indices]

        # if we want to subselect for an ROI
        if ROI:
            roi_ind = cortex.get_roi_verts(pysub, ROI) # get indices for that ROI
            
            # mask for roi
            for k in estimate_keys:
                masked_dict[k] = masked_dict[k][roi_ind[ROI]]
        
        return masked_dict


    def save_pRF_model_estimates(self, filename, final_estimates, model_type = 'gauss', grid = False):
    
        """
        re-arrange estimates that were masked
        and save all in numpy file
        
        (only works for gii files, should generalize for nii and cifti also)
        
        Parameters
        ----------
        filename : str
            absolute filename of estimates to be saved
        final_estimates : arr
            2d estimates (datapoints,estimates)
        model_type: str
            model type used for fitting
        
        """ 

        # make dir if it doesnt exist already
        os.makedirs(op.split(filename)[0], exist_ok = True)
                
        if model_type == 'gauss':

            if self.fit_hrf and not grid:
                np.savez(filename,
                        x = final_estimates[..., 0],
                        y = final_estimates[..., 1],
                        size = final_estimates[..., 2],
                        betas = final_estimates[...,3],
                        baseline = final_estimates[..., 4],
                        hrf_derivative = final_estimates[..., 5],
                        hrf_dispersion = final_estimates[..., 6], 
                        r2 = final_estimates[..., 7])
            
            else:
                np.savez(filename,
                        x = final_estimates[..., 0],
                        y = final_estimates[..., 1],
                        size = final_estimates[..., 2],
                        betas = final_estimates[...,3],
                        baseline = final_estimates[..., 4],
                        r2 = final_estimates[..., 5])
        
        elif model_type == 'css':

            if self.fit_hrf and not grid:
                np.savez(filename,
                        x = final_estimates[..., 0],
                        y = final_estimates[..., 1],
                        size = final_estimates[..., 2],
                        betas = final_estimates[...,3],
                        baseline = final_estimates[..., 4],
                        ns = final_estimates[..., 5],
                        hrf_derivative = final_estimates[..., 6],
                        hrf_dispersion = final_estimates[..., 7], 
                        r2 = final_estimates[..., 8])
            
            else:
                np.savez(filename,
                        x = final_estimates[..., 0],
                        y = final_estimates[..., 1],
                        size = final_estimates[..., 2],
                        betas = final_estimates[...,3],
                        baseline = final_estimates[..., 4],
                        ns = final_estimates[..., 5],
                        r2 = final_estimates[..., 6])

        elif model_type == 'dn':

            if self.fit_hrf and not grid:
                np.savez(filename,
                        x = final_estimates[..., 0],
                        y = final_estimates[..., 1],
                        size = final_estimates[..., 2],
                        betas = final_estimates[...,3],
                        baseline = final_estimates[..., 4],
                        sa = final_estimates[..., 5],
                        ss = final_estimates[..., 6], 
                        nb = final_estimates[..., 7], 
                        sb = final_estimates[..., 8], 
                        hrf_derivative = final_estimates[..., 9],
                        hrf_dispersion = final_estimates[..., 10], 
                        r2 = final_estimates[..., 11])
            
            else:
                np.savez(filename,
                        x = final_estimates[..., 0],
                        y = final_estimates[..., 1],
                        size = final_estimates[..., 2],
                        betas = final_estimates[...,3],
                        baseline = final_estimates[..., 4],
                        sa = final_estimates[..., 5],
                        ss = final_estimates[..., 6], 
                        nb = final_estimates[..., 7], 
                        sb = final_estimates[..., 8], 
                        r2 = final_estimates[..., 9])

        elif model_type == 'dog':

            if self.fit_hrf and not grid:
                np.savez(filename,
                        x = final_estimates[..., 0],
                        y = final_estimates[..., 1],
                        size = final_estimates[..., 2],
                        betas = final_estimates[...,3],
                        baseline = final_estimates[..., 4],
                        sa = final_estimates[..., 5],
                        ss = final_estimates[..., 6], 
                        hrf_derivative = final_estimates[..., 7],
                        hrf_dispersion = final_estimates[..., 8], 
                        r2 = final_estimates[..., 9])
            
            else:
                np.savez(filename,
                        x = final_estimates[..., 0],
                        y = final_estimates[..., 1],
                        size = final_estimates[..., 2],
                        betas = final_estimates[...,3],
                        baseline = final_estimates[..., 4],
                        sa = final_estimates[..., 5],
                        ss = final_estimates[..., 6], 
                        r2 = final_estimates[..., 7])


    def load_pRF_model_chunks(self, fit_path, fit_model = 'css', fit_hrf = False, basefilename = None, overwrite = False, iterative = True):

        """ 
        combine all chunks 
        into one single estimate numpy array
        assumes input is whole brain ("vertex", time)

        Parameters
        ----------
        fit_path : str
            absolute path to files
        fit_model: str
            fit model of estimates
        fit_hrf: bool
            if we fitted hrf or not
        
        Outputs
        -------
        estimates : npz 
            numpy array of estimates
        
        """

        # if we are fitting HRF, then we want to look for those files
        if fit_hrf:
            filename_list = [op.join(fit_path, x) for x in os.listdir(fit_path) if fit_model in x and 'chunk-000' in x and 'HRF' in x]
        else:
            filename_list = [op.join(fit_path, x) for x in os.listdir(fit_path) if fit_model in x and 'chunk-000' in x]
        
        ## if we defined a base filename that should be used to fish out right estimates
        if basefilename:
            filename = [file for file in filename_list if basefilename in file][0]
        else:
            filename = filename_list[0]
        
        filename = filename.replace('_chunk-000', '')

        if not op.exists(filename) or overwrite:
        
            for ch in np.arange(self.total_chunks['pRF']):
                
                # if we are fitting HRF, then we want to look for those files
                if fit_hrf:
                    chunk_name_list = [op.join(fit_path, x) for x in os.listdir(fit_path) if fit_model in x and 'chunk-%s'%str(ch).zfill(3) in x and 'HRF' in x]
                else:
                    chunk_name_list = [op.join(fit_path, x) for x in os.listdir(fit_path) if fit_model in x and 'chunk-%s'%str(ch).zfill(3) in x]
                
                ## if we defined a base filename that should be used to fish out right estimates
                if basefilename:
                    chunk_name = [file for file in chunk_name_list if basefilename in file][0]
                else:
                    chunk_name = chunk_name_list[0]

                print('loading chunk %s'%chunk_name)
                chunk = np.load(chunk_name) # load chunk
                
                if ch == 0:
                    xx = chunk['x']
                    yy = chunk['y']

                    size = chunk['size']

                    beta = chunk['betas']
                    baseline = chunk['baseline']

                    if 'css' in fit_model: 
                        ns = chunk['ns']
                    elif fit_model in ['dn', 'dog']:
                        sa = chunk['sa']
                        ss = chunk['ss']
                    
                    if 'dn' in fit_model:
                        nb = chunk['nb']
                        sb = chunk['sb']

                    rsq = chunk['r2']

                    if fit_hrf and iterative:
                        hrf_derivative = chunk['hrf_derivative']
                        hrf_dispersion = chunk['hrf_dispersion']
                    else: # assumes standard spm params
                        hrf_derivative = np.ones(xx.shape)
                        hrf_dispersion = np.zeros(xx.shape) 

                else:
                    xx = np.concatenate((xx, chunk['x']))
                    yy = np.concatenate((yy, chunk['y']))

                    size = np.concatenate((size, chunk['size']))

                    beta = np.concatenate((beta, chunk['betas']))
                    baseline = np.concatenate((baseline, chunk['baseline']))

                    if 'css' in fit_model:
                        ns = np.concatenate((ns, chunk['ns']))
                    elif fit_model in ['dn', 'dog']:
                        sa = np.concatenate((sa, chunk['sa']))
                        ss = np.concatenate((ss, chunk['ss']))

                    if 'dn' in fit_model:
                        nb = np.concatenate((nb, chunk['nb']))
                        sb = np.concatenate((sb, chunk['sb']))

                    rsq = np.concatenate((rsq, chunk['r2']))
                    
                    if fit_hrf and iterative:
                        hrf_derivative = np.concatenate((hrf_derivative, chunk['hrf_derivative']))
                        hrf_dispersion = np.concatenate((hrf_dispersion, chunk['hrf_dispersion']))
                    else: # assumes standard spm params
                        hrf_derivative = np.concatenate((hrf_derivative, np.ones(xx.shape)))
                        hrf_dispersion = np.concatenate((hrf_dispersion, np.zeros(xx.shape))) 
            
            print('shape of estimates is %s'%(str(xx.shape)))

            # save file
            print('saving %s'%filename)

            if 'gauss' in fit_model:
                np.savez(filename,
                        x = xx,
                        y = yy,
                        size = size,
                        betas = beta,
                        baseline = baseline,
                        hrf_derivative = hrf_derivative,
                        hrf_dispersion = hrf_dispersion,
                        r2 = rsq)

            elif 'css' in fit_model:
                np.savez(filename,
                        x = xx,
                        y = yy,
                        size = size,
                        betas = beta,
                        baseline = baseline,
                        ns = ns,
                        hrf_derivative = hrf_derivative,
                        hrf_dispersion = hrf_dispersion,
                        r2 = rsq)

            elif 'dn' in fit_model:
                np.savez(filename,
                        x = xx,
                        y = yy,
                        size = size,
                        betas = beta,
                        baseline = baseline,
                        sa = sa,
                        ss = ss,
                        nb = nb,
                        sb = sb,
                        hrf_derivative = hrf_derivative,
                        hrf_dispersion = hrf_dispersion,
                        r2 = rsq)

            elif 'dog' in fit_model:
                np.savez(filename,
                        x = xx,
                        y = yy,
                        size = size,
                        betas = beta,
                        baseline = baseline,
                        sa = sa,
                        ss = ss,
                        hrf_derivative = hrf_derivative,
                        hrf_dispersion = hrf_dispersion,
                        r2 = rsq)
            
        else:
            print('file already exists, loading %s'%filename)
        
        return np.load(filename)
