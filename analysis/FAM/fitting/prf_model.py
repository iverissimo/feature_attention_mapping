import numpy as np
import os
import os.path as op
import pandas as pd
import yaml

from PIL import Image, ImageDraw

import cortex

from scipy.optimize import LinearConstraint, NonlinearConstraint

from joblib import Parallel, delayed
from tqdm import tqdm

from FAM.fitting.model import Model

from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter, Norm_Iso2DGaussianFitter, DoG_Iso2DGaussianFitter


class pRF_model(Model):

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

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.MRIObj.derivatives_pth, self.fitfolder['pRF'])
        else:
            self.outputdir = outputdir
        
        # reset osf value, because model assumes 10 (for FA)
        self.osf = 1

        # number of TRs per condition (bar pass)
        pRF_bar_pass_all = self.MRIObj.beh_utils.get_pRF_cond_per_TR(cond_TR_dict = self.MRIObj.pRF_nr_TRs, 
                                                                    bar_pass_direction = self.MRIObj.pRF_bar_pass)
        
        # do same to bar pass direction str array
        self.condition_per_TR = self.MRIObj.mri_utils.crop_shift_arr(pRF_bar_pass_all, 
                                                                crop_nr = self.MRIObj.task_nr_cropTR['pRF'], 
                                                                shift = self.MRIObj.shift_TRs_num)

                
    def get_DM(self, participant, ses = 'mean', mask_bool_df = None, filename = None, 
                    osf = 1, res_scaling = .1, stim_on_screen = []):

        """
        Get pRF Design matrix

        Parameters
        ----------
        participant : str
            participant number
        ses : str
            session number (default mean)
        filename: str
            absolute path to np file where we stored design matrix, if none it will make one anew
        osf: int
            oversampling factor, if bigger than one it will return DM of timepoints * osf
        res_scaling: float
            spatial rescaling factor
        stim_on_screen: arr
            boolean array with moments where stim was on screen
        mask_bool_df: dataframe
            if dataframe given, will be used to mask design matrix given behavioral performance
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

            # all possible positions in pixels for for midpoint of
            # y position for vertical bar passes, 
            ver_y = self.MRIObj.screen_res[1]*np.linspace(0,1, self.MRIObj.pRF_nr_TRs['U-D'])
            # x position for horizontal bar passes 
            hor_x = self.MRIObj.screen_res[0]*np.linspace(0,1, self.MRIObj.pRF_nr_TRs['L-R'])

            # coordenates for bar pass, for PIL Image
            coordenates_bars = {'L-R': {'upLx': hor_x - 0.5 * self.MRIObj.bar_width['pRF'] * self.MRIObj.screen_res[0], 'upLy': np.repeat(self.MRIObj.screen_res[1], self.MRIObj.pRF_nr_TRs['L-R']),
                                        'lowRx': hor_x + 0.5 * self.MRIObj.bar_width['pRF'] * self.MRIObj.screen_res[0], 'lowRy': np.repeat(0, self.MRIObj.pRF_nr_TRs['L-R'])},
                                'R-L': {'upLx': np.array(list(reversed(hor_x - 0.5 * self.MRIObj.bar_width['pRF'] * self.MRIObj.screen_res[0]))), 'upLy': np.repeat(self.MRIObj.screen_res[1], self.MRIObj.pRF_nr_TRs['R-L']),
                                        'lowRx': np.array(list(reversed(hor_x+ 0.5 * self.MRIObj.bar_width['pRF'] * self.MRIObj.screen_res[0]))), 'lowRy': np.repeat(0, self.MRIObj.pRF_nr_TRs['R-L'])},
                                'U-D': {'upLx': np.repeat(0, self.MRIObj.pRF_nr_TRs['U-D']), 'upLy': ver_y+0.5 * self.MRIObj.bar_width['pRF'] * self.MRIObj.screen_res[1],
                                        'lowRx': np.repeat(self.MRIObj.screen_res[0], self.MRIObj.pRF_nr_TRs['U-D']), 'lowRy': ver_y - 0.5 * self.MRIObj.bar_width['pRF'] * self.MRIObj.screen_res[1]},
                                'D-U': {'upLx': np.repeat(0, self.MRIObj.pRF_nr_TRs['D-U']), 'upLy': np.array(list(reversed(ver_y + 0.5 * self.MRIObj.bar_width['pRF'] * self.MRIObj.screen_res[1]))),
                                        'lowRx': np.repeat(self.MRIObj.screen_res[0], self.MRIObj.pRF_nr_TRs['D-U']), 'lowRy': np.array(list(reversed(ver_y - 0.5 * self.MRIObj.bar_width['pRF'] * self.MRIObj.screen_res[1])))}
                                }

            # save screen display for each TR (or if osf > 1 then for #TRs * osf)
            visual_dm_array = np.zeros((len(self.condition_per_TR) * osf, round(self.MRIObj.screen_res[0] * res_scaling), round(self.MRIObj.screen_res[1] * res_scaling)))
            i = 0

            for trl, bartype in enumerate(self.condition_per_TR): # loop over bar pass directions

                img = Image.new('RGB', tuple(self.MRIObj.screen_res)) # background image

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
                    if trl < (len(self.condition_per_TR) - 1):
                        i = i+1 if self.condition_per_TR[trl] == self.condition_per_TR[trl+1] else 0    
                
                ## save in array, and apply mask
                visual_dm_array[int(trl*osf):int(trl*osf + osf), ...] = np.array(img)[::round(1/res_scaling),::round(1/res_scaling),0][np.newaxis,...] * stim_on_screen[trl]

            # swap axis to have time in last axis [x,y,t]
            visual_dm = visual_dm_array.transpose([1,2,0])

            if save_dm:
                # save design matrix
                print('Making and saving {file}'.format(file = filename))
                np.save(filename, visual_dm)  
                    
        return self.MRIObj.mri_utils.normalize(visual_dm)

    def set_models(self, participant_list = [], mask_bool_df = None, ses2model = 'mean', stim_on_screen = [],
                        hrf=[1.0, 1.0, 0.0]):

        """
        define pRF models to be used for each participant in participant list
                
        Parameters
        ----------
        participant_list: list
            list with participant ID
        mask_bool_df: dataframe
            if dataframe given, will be used to mask design matrix given behavioral performance
        ses2model: str
            which sessions are we modeling: 1, all, mean [default]. Note --> mean indicates average across runs
        stim_on_screen: arr
            boolean array with moments where stim was on screen
        """                 

        ## if no participant list set, then run all
        if len(participant_list) == 0:
            participant_list = self.MRIObj.sj_num

        ## set filter params
        filter_params = {'highpass': self.MRIObj.params['mri']['filtering']['highpass'],
                        'add_mean': self.MRIObj.params['mri']['filtering']['add_mean'],
                        'first_modes_to_remove': self.MRIObj.params['mri']['filtering']['first_modes_to_remove'],
                        'last_modes_to_remove_percent': self.MRIObj.params['mri']['filtering']['last_modes_to_remove_percent'],
                        'window_length': self.MRIObj.params['mri']['filtering']['window_length'],
                        'polyorder': self.MRIObj.params['mri']['filtering']['polyorder']}
                
        # empty dict where we'll store all participant models
        pp_models = {}
        
        ## loop over participants
        for pp in participant_list:

            pp_models['sub-{sj}'.format(sj=pp)] = {}

            # if we're combining sessions
            if ses2model == 'mean':
                sessions = ['ses-mean']
            elif ses2model == 'all':
                sessions = self.MRIObj.session['sub-{sj}'.format(sj=pp)]
            else:
                sessions = ['ses-{s}'.format(s = ses2model)]

            ## go over sessions (if its the case)
            # and save DM and models
            for ses in sessions:

                pp_models['sub-{sj}'.format(sj=pp)][ses] = {}

                visual_dm = self.get_DM(pp, ses = ses, mask_bool_df = mask_bool_df, stim_on_screen = stim_on_screen,
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
                                                hrf=hrf,
                                                filter_predictions = True,
                                                filter_type = self.MRIObj.params['mri']['filtering']['type']['pRF'],
                                                filter_params = filter_params,
                                                osf = self.osf,
                                                hrf_onset = self.hrf_onset,
                                                )

                pp_models['sub-{sj}'.format(sj=pp)][ses]['gauss_model'] = gauss_model

                # CSS
                css_model = CSS_Iso2DGaussianModel(stimulus = prf_stim,
                                                    hrf=hrf,
                                                    filter_predictions = True,
                                                    filter_type = self.MRIObj.params['mri']['filtering']['type']['pRF'],
                                                    filter_params = filter_params,
                                                    osf = self.osf,
                                                    hrf_onset = self.hrf_onset,
                                                )

                pp_models['sub-{sj}'.format(sj=pp)][ses]['css_model'] = css_model

                # DN 
                dn_model =  Norm_Iso2DGaussianModel(stimulus = prf_stim,
                                                    hrf=hrf,
                                                    filter_predictions = True,
                                                    filter_type = self.MRIObj.params['mri']['filtering']['type']['pRF'],
                                                    filter_params = filter_params,
                                                    osf = self.osf,
                                                    hrf_onset = self.hrf_onset,
                                                )

                pp_models['sub-{sj}'.format(sj=pp)][ses]['dn_model'] = dn_model

                # DOG
                dog_model = DoG_Iso2DGaussianModel(stimulus = prf_stim,
                                                hrf=hrf,
                                                filter_predictions = True,
                                                filter_type = self.MRIObj.params['mri']['filtering']['type']['pRF'],
                                                filter_params = filter_params,
                                                osf = self.osf,
                                                hrf_onset = self.hrf_onset,
                                                )
                
                pp_models['sub-{sj}'.format(sj=pp)][ses]['dog_model'] = dog_model

        return pp_models

    def fit_data(self, participant, pp_models, ses = 'mean',
                    run_type = 'mean', chunk_num = None, vertex = [], ROI = None,
                    model2fit = 'gauss', file_ext = '_cropped_dc_psc.npy', 
                    outdir = None, save_estimates = False, hemisphere = 'BH',
                    xtol = 1e-3, ftol = 1e-4, n_jobs = 16, n_batches = 16,
                    rsq_threshold = 0.05, verbose = True):

        """
        fit inputted pRF models to each participant in participant list
                
        Parameters
        ----------
        participant: str
            participant ID
        input_pth: str or None
            path to look for files, if None then will get them from derivatives/postfmriprep/<space>/sub-X folder
        run_type: string or int
            type of run to fit, mean (default), or if int will do single run fit
        file_ext: dict
            file extension, to select appropriate files
        mask_DM: bool
            if we want to mask design matrix given behavioral performance
        hemisphere: str
            if we want to fit both hemis [default] or only one (LH or RH)
        """  

        ## get list of files to load
        bold_filelist = self.MRIObj.mri_utils.get_bold_file_list(participant, task = 'pRF', ses = ses, file_ext = file_ext,
                                                                postfmriprep_pth = self.MRIObj.postfmriprep_pth, 
                                                                acq_name = self.MRIObj.acq, hemisphere = hemisphere)
                        
        ## if we want an ROI, then get vertices
        if ROI is not None:
            ## get vertices for each relevant ROI
            ROIs_dict = self.MRIObj.mri_utils.get_ROIs_dict(sub_id = participant, pysub = self.pysub, use_atlas = self.use_atlas, 
                                                            annot_filename = self.annot_filename, hemisphere = hemisphere,
                                                            ROI_labels = self.MRIObj.params['plotting']['ROIs'][self.plot_key],
                                                            freesurfer_pth = self.MRIObj.freesurfer_pth, 
                                                            use_fs_label = self.use_fs_label)
            if len(vertex) == 0:
                vertex = ROIs_dict[ROI]
            else:
                vertex = ROIs_dict[ROI][vertex]

        print(bold_filelist)
        ## Load data array
        data = self.get_data4fitting(bold_filelist, task = 'pRF', run_type = run_type, chunk_num = chunk_num, vertex = vertex,
                                    baseline_interval = 'empty_long', ses = ses, return_filenames = False)

        ## Set nan voxels to 0, to avoid issues when fitting
        masked_data = data.copy()
        masked_data[np.where(np.isnan(data[...,0]))[0]] = 0

        print(masked_data.shape)

        # if leaving one run out, then load test data to also calculate cv-r2
        if 'loo_' in run_type:
            test_file_list, _ = self.MRIObj.mri_utils.get_loo_filename(bold_filelist, loo_key=run_type)
            test_run_num, test_ses_num = self.MRIObj.mri_utils.get_run_ses_from_str(test_file_list[0])
            test_data = self.get_data4fitting(test_file_list, task = 'pRF', run_type = test_run_num, chunk_num = chunk_num, vertex = vertex,
                                                            baseline_interval = 'empty_long', ses = test_ses_num, return_filenames = False)
            test_data[np.where(np.isnan(test_data[...,0]))[0]] = 0
            
        ## set output dir to save estimates
        if outdir is None:
            if 'loo_' in run_type:
                outdir = op.join(self.outputdir, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant), run_type)
            else:
                outdir = op.join(self.outputdir, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant), 'ses-{s}'.format(s = ses))
            
        os.makedirs(outdir, exist_ok = True)
        print('saving files in %s'%outdir)

        ## set base filename that will be used for estimates
        basefilename = self.get_estimates_basefilename(participant=participant, run_type=run_type, hemisphere=hemisphere, 
                                                       chunk_num=chunk_num, vertex=vertex, ROI=ROI, file_ext=file_ext)

        ## set model parameters 
        # relevant for grid and iterative fitting
        fit_params = self.get_fit_startparams(max_ecc_size = pp_models['sub-{sj}'.format(sj = participant)]['ses-{s}'.format(s = ses)]['prf_stim'].screen_size_degrees/2.0)

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
                                                model = pp_models['sub-{sj}'.format(sj = participant)]['ses-{s}'.format(s = ses)]['gauss_model'], 
                                                n_jobs = n_jobs)

            gauss_fitter.grid_fit(ecc_grid = fit_params['gauss']['eccs'], 
                                    polar_grid = fit_params['gauss']['polars'], 
                                    size_grid = fit_params['gauss']['sizes'], 
                                    fixed_grid_baseline = fit_params['gauss']['fixed_grid_baseline'],
                                    grid_bounds = fit_params['gauss']['grid_bounds'],
                                    n_batches = n_batches,
                                    verbose = verbose,
                                    hrf_1_grid = fit_params['gauss']['hrf_1_grid'],
                                    hrf_2_grid = fit_params['gauss']['hrf_2_grid'])

            # iterative fit
            print("Gauss model ITERATIVE fit")

            gauss_fitter.iterative_fit(rsq_threshold = rsq_threshold, verbose = verbose,
                                       starting_params = gauss_fitter.gridsearch_params,
                                        bounds = fit_params['gauss']['bounds'],
                                        constraints = constraints['gauss'],
                                        xtol = xtol, ftol = ftol)
            
            # if leaving one run out, then calculate cv-r2 of model
            if 'loo_' in run_type:
                print('Calculate CV-rsq for left out data: run-{r}, ses-{s}'.format(r = test_run_num, s = test_ses_num))
                cv_r2 = self.crossvalidate(test_data,
                                            model_object = pp_models['sub-{sj}'.format(sj = participant)]['ses-{s}'.format(s = ses)]['gauss_model'], 
                                            estimates = gauss_fitter.iterative_search_params, avg_hrf = False, n_jobs = n_jobs)
            else:
                cv_r2 = None

            # if we want to save estimates
            if save_estimates and not op.isfile(it_gauss_filename):
                # for grid
                print('saving %s'%grid_gauss_filename)
                self.save_pRF_model_estimates(grid_gauss_filename, gauss_fitter.gridsearch_params, 
                                                model_type = 'gauss', grid = True) 
                # for it
                print('saving %s'%it_gauss_filename)
                self.save_pRF_model_estimates(it_gauss_filename, gauss_fitter.iterative_search_params, 
                                                model_type = 'gauss', cv_r2 = cv_r2)

            if model2fit != 'gauss':

                print("{key} model GRID fit".format(key = model2fit))
                
                if model2fit == 'css':

                    fitter = CSS_Iso2DGaussianFitter(data = masked_data, 
                                                    model = pp_models['sub-{sj}'.format(sj = participant)]['ses-{s}'.format(s = ses)]['{key}_model'.format(key = model2fit)], 
                                                    n_jobs = n_jobs,
                                                    previous_gaussian_fitter = gauss_fitter,
                                                    use_previous_gaussian_fitter_hrf=self.fit_hrf)

                    fitter.grid_fit(fit_params['css']['n_grid'],
                                fixed_grid_baseline = fit_params['css']['fixed_grid_baseline'],
                                grid_bounds = fit_params['css']['grid_bounds'],
                                rsq_threshold = rsq_threshold,
                                n_batches = n_batches,
                                verbose = verbose,
                                hrf_1_grid = fit_params['css']['hrf_1_grid'],
                                hrf_2_grid = fit_params['css']['hrf_2_grid'])
                
                elif model2fit == 'dn':

                    fitter = Norm_Iso2DGaussianFitter(data = masked_data, 
                                                    model = pp_models['sub-{sj}'.format(sj = participant)]['ses-{s}'.format(s = ses)]['{key}_model'.format(key = model2fit)], 
                                                    n_jobs = n_jobs,
                                                    previous_gaussian_fitter = gauss_fitter,
                                                    use_previous_gaussian_fitter_hrf=self.fit_hrf)

                    fitter.grid_fit(surround_amplitude_grid = fit_params['dn']['surround_amplitude_grid'],
                                    surround_size_grid = fit_params['dn']['surround_size_grid'],
                                    neural_baseline_grid = fit_params['dn']['neural_baseline_grid'],
                                    surround_baseline_grid = fit_params['dn']['surround_baseline_grid'],
                                fixed_grid_baseline = fit_params['dn']['fixed_grid_baseline'],
                                grid_bounds = fit_params['dn']['grid_bounds'],
                                rsq_threshold = rsq_threshold,
                                n_batches = n_batches,
                                verbose = verbose,
                                hrf_1_grid = fit_params['dn']['hrf_1_grid'],
                                hrf_2_grid = fit_params['dn']['hrf_2_grid'],
                                ecc_grid = fit_params['dn']['eccs'], 
                                polar_grid = fit_params['dn']['polars'], 
                                size_grid = fit_params['dn']['sizes'])

                elif model2fit == 'dog':

                    fitter = DoG_Iso2DGaussianFitter(data = masked_data, 
                                                    model = pp_models['sub-{sj}'.format(sj = participant)]['ses-{s}'.format(s = ses)]['{key}_model'.format(key = model2fit)], 
                                                    n_jobs = n_jobs,
                                                    previous_gaussian_fitter = gauss_fitter,
                                                    use_previous_gaussian_fitter_hrf=self.fit_hrf)

                    fitter.grid_fit(fit_params['dog']['surround_amplitude_grid'],
                                    fit_params['dog']['surround_size_grid'],
                                fixed_grid_baseline = fit_params['dog']['fixed_grid_baseline'],
                                grid_bounds = fit_params['dog']['grid_bounds'],
                                rsq_threshold = rsq_threshold,
                                n_batches = n_batches,
                                verbose = verbose,
                                hrf_1_grid = fit_params['dog']['hrf_1_grid'],
                                hrf_2_grid = fit_params['dog']['hrf_2_grid'])

                # iterative fit
                print("{key} model ITERATIVE fit".format(key = model2fit))

                fitter.iterative_fit(rsq_threshold = rsq_threshold, 
                                    verbose = verbose,
                                    bounds = fit_params[model2fit]['bounds'],
                                    constraints = constraints[model2fit],
                                    xtol = xtol, ftol = ftol)
                
                # if leaving one run out, then calculate cv-r2 of model
                if 'loo_' in run_type:
                    print('Calculate CV-rsq for left out data: run-{r}, ses-{s}'.format(r = test_run_num, s = test_ses_num))
                    cv_r2 = self.crossvalidate(test_data,
                                                model_object = pp_models['sub-{sj}'.format(sj = participant)]['ses-{s}'.format(s = ses)]['{key}_model'.format(key = model2fit)], 
                                                estimates = fitter.iterative_search_params, avg_hrf = False, n_jobs = n_jobs)
                else:
                    cv_r2 = None

                # if we want to save estimates
                if save_estimates:
                    # for grid
                    print('saving %s'%grid_model_filename)
                    self.save_pRF_model_estimates(grid_model_filename, fitter.gridsearch_params, 
                                                    model_type = model2fit, grid = True)
                    # for it
                    print('saving %s'%it_model_filename)
                    self.save_pRF_model_estimates(it_model_filename, fitter.iterative_search_params, 
                                                    model_type = model2fit, cv_r2 = cv_r2)

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
        
    def get_estimates_basefilename(self, participant, run_type = 'mean', hemisphere = 'BH', 
                                   chunk_num = None, vertex = [], ROI = None, file_ext = '_cropped_dc_psc.npy'):
        
        """
        create base str for estimates filename 
        (might want to later adapt and include model-specific filename)
        """

        ## set base filename that will be used for estimates
        basefilename = 'sub-{sj}_task-pRF_acq-{acq}_runtype-{rt}'.format(sj = participant,
                                                                        acq = self.MRIObj.acq,
                                                                        rt = run_type)
        # if we want specific hemisphere
        if hemisphere in ['LH', 'hemi-L', 'left']:
            basefilename += '_hemi-L'
        elif hemisphere in ['RH', 'hemi-R', 'right']:
            basefilename += '_hemi-R'

        if chunk_num is not None:
            basefilename += '_chunk-{ch}'.format(ch = str(chunk_num).zfill(3))
        elif ROI is not None:
            basefilename += '_ROI-{roi}'.format(roi = str(ROI))
        elif len(vertex)> 0:
            basefilename += '_vertex-{ver}'.format(ver = str(vertex))
        
        basefilename += file_ext.replace('.npy', '.npz')

        return basefilename

    def get_fit_startparams(self, max_ecc_size = 6):

        """
        Load all fitting starting params and bounds into a dictionary

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

        ## set hrf grid
        fitpar_dict['gauss']['hrf_1_grid'] = np.linspace(0,1,
                                                        self.MRIObj.params['mri']['fitting']['pRF']['hrf_grid_nr']) # will iterate over hrf derivative grid values
        fitpar_dict['gauss']['hrf_2_grid'] = np.linspace(0,0,1) # we want to keep dispersion at 0

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

        ## set hrf grid
        fitpar_dict['css']['hrf_1_grid'] = np.linspace(0,1,
                                                        self.MRIObj.params['mri']['fitting']['pRF']['hrf_grid_nr']) # will iterate over hrf derivative grid values
        fitpar_dict['css']['hrf_2_grid'] = np.linspace(0,0,1) # we want to keep dispersion at 0

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
        fitpar_dict['dn']['surround_amplitude_grid'] = np.linspace(0.1,3, self.MRIObj.params['mri']['fitting']['pRF']['dn_grid_nr']) 
        
        # Surround size (gauss sigma_2)
        fitpar_dict['dn']['surround_size_grid'] = np.linspace(3,18, self.MRIObj.params['mri']['fitting']['pRF']['dn_grid_nr']) 
        
        # Neural baseline (Normalization parameter B)
        fitpar_dict['dn']['neural_baseline_grid'] = np.linspace(0.1,10, self.MRIObj.params['mri']['fitting']['pRF']['dn_grid_nr']) 

        # Surround baseline (Normalization parameter D)
        fitpar_dict['dn']['surround_baseline_grid'] = np.linspace(0.1,10, self.MRIObj.params['mri']['fitting']['pRF']['dn_grid_nr']) 

        ## set hrf grid
        fitpar_dict['dn']['hrf_1_grid'] = np.linspace(0,1,
                                                        self.MRIObj.params['mri']['fitting']['pRF']['dn_grid_nr']) # will iterate over hrf derivative grid values
        fitpar_dict['dn']['hrf_2_grid'] = np.linspace(0,0,1) # we want to keep dispersion at 0

        # if we want to re-fit grid prf positions
        if self.MRIObj.params['mri']['fitting']['pRF']['re_grid_dn']:
            # size, ecc, polar angle
            fitpar_dict['dn']['sizes'] = fitpar_dict['gauss']['sizes'][:self.MRIObj.params['mri']['fitting']['pRF']['dn_grid_nr']]
            fitpar_dict['dn']['eccs'] = fitpar_dict['gauss']['eccs'][:self.MRIObj.params['mri']['fitting']['pRF']['dn_grid_nr']]
            fitpar_dict['dn']['polars'] = fitpar_dict['gauss']['polars'][:self.MRIObj.params['mri']['fitting']['pRF']['dn_grid_nr']]
        else:
            fitpar_dict['dn']['sizes'] = None
            fitpar_dict['dn']['eccs'] = None
            fitpar_dict['dn']['polars'] = None

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

        ## set hrf grid
        fitpar_dict['dog']['hrf_1_grid'] = np.linspace(0,1,
                                                        self.MRIObj.params['mri']['fitting']['pRF']['hrf_grid_nr']) # will iterate over hrf derivative grid values
        fitpar_dict['dog']['hrf_2_grid'] = np.linspace(0,0,1) # we want to keep dispersion at 0

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

        # add hrf bounds (regardless of fitting hrf or not)
        fitpar_dict['gauss']['bounds'] += [(0,10),(0,0)]
        fitpar_dict['css']['bounds'] += [(0,10),(0,0)]
        fitpar_dict['dn']['bounds'] += [(0,10),(0,0)]
        fitpar_dict['dog']['bounds'] += [(0,10),(0,0)]

        # if not fitting hrf, then set grids to none
        if not self.fit_hrf:
            fitpar_dict['gauss']['hrf_1_grid'] = None
            fitpar_dict['gauss']['hrf_2_grid'] = None
            fitpar_dict['css']['hrf_1_grid'] = None
            fitpar_dict['css']['hrf_2_grid'] = None
            fitpar_dict['dn']['hrf_1_grid'] = None
            fitpar_dict['dn']['hrf_2_grid'] = None
            fitpar_dict['dog']['hrf_1_grid'] = None
            fitpar_dict['dog']['hrf_2_grid'] = None
        
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
                else:
                    constraints[key] = None

        return constraints

    def load_pRF_model_estimates(self, participant_list = [], ses = 'mean', run_type = 'mean', model_name = None, 
                                    iterative = True, fit_hrf = False, mask_bool_df = None, stim_on_screen = []):

        """
        Load pRF model estimates
        when they already where fitted and saved in out folder

        Parameters
        ----------
        participant_list: list
            list with participant ID
        ses: str
            session we are looking at
        run_type: str
            run type we fitted
        model_name: str or None
            model name to be loaded (if None defaults to class model)
        iterative: bool
            if we want to load iterative fitting results [default] or grid results
        mask_bool_df: dataframe
            if dataframe given, will be used to mask design matrix given behavioral performance
        stim_on_screen: arr
            boolean array with moments where stim was on screen
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

        # get participant models, which also will load 
        # DM and mask it according to participants behavior
        pp_prf_models = self.set_models(participant_list = participant_list, 
                                        ses2model = ses, mask_bool_df = mask_bool_df,
                                        stim_on_screen = stim_on_screen)
        
        # empty dict where we'll store all participant estimates
        pp_prf_est_dict = {}

        ## loop over participant list
        for participant in participant_list:

            ## path to estimates
            if 'loo_' in run_type:
                pRFdir = op.join(self.outputdir, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant), run_type, est_folder)
                crossval = True
            else:
                pRFdir = op.join(self.outputdir, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant), 'ses-{s}'.format(s = ses), est_folder)
                crossval = False

            # append
            if self.MRIObj.sj_space in ['fsnative']:
                pp_prf_est_dict['sub-{sj}'.format(sj=participant)] = self.load_pRF_model_hemis(pRFdir, fit_model = model_name,
                                                        basefilename = 'sub-{sj}_task-pRF_acq-{acq}_runtype-{rt}'.format(sj = participant,
                                                                                                                    acq = self.MRIObj.acq,
                                                                                                                    rt = run_type),
                                                        fit_hrf = fit_hrf, iterative = iterative, crossval = crossval)
            else:
                pp_prf_est_dict['sub-{sj}'.format(sj=participant)] = self.load_pRF_model_chunks(pRFdir, fit_model = model_name,
                                                        basefilename = 'sub-{sj}_task-pRF_acq-{acq}_runtype-{rt}'.format(sj = participant,
                                                                                                                    acq = self.MRIObj.acq,
                                                                                                                    rt = run_type),
                                                        fit_hrf = fit_hrf, iterative = iterative, crossval = crossval)
                
        return pp_prf_est_dict, pp_prf_models
    
    def load_pRF_model_hemis(self, fit_path, fit_model = 'css', fit_hrf = False, basefilename = None, 
                                    overwrite = False, iterative = True, crossval = False):
        
        """ 
        combine all both hemispheres from fsnative fit
        into one single estimate numpy array
        assumes input is whole brain (vertex, time)

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

        if not self.MRIObj.sj_space in ['fsnative']:
            raise NameError('Not using fsnative surface, used other function')

        # if we are fitting HRF, then we want to look for those files
        if fit_hrf:
            filename_list = [op.join(fit_path, x) for x in os.listdir(fit_path) if fit_model in x and 'hemi-L' in x and 'HRF' in x and 'chunk' not in x]
        else:
            filename_list = [op.join(fit_path, x) for x in os.listdir(fit_path) if fit_model in x and 'hemi-L' in x and 'HRF' not in x and 'chunk' not in x]
        
        ## if we defined a base filename that should be used to fish out right estimates
        if basefilename:
            filename = [file for file in filename_list if basefilename in file][0]
        else:
            filename = filename_list[0]
        
        filename = filename.replace('_hemi-L', '_hemi-B')

        if not op.exists(filename) or overwrite:

            for h_ind, hemi in enumerate(self.MRIObj.hemispheres):

                # if we are fitting HRF, then we want to look for those files
                if fit_hrf:
                    hemi_name_list = [op.join(fit_path, x) for x in os.listdir(fit_path) if fit_model in x and hemi in x and 'HRF' in x and 'chunk' not in x]
                else:
                    hemi_name_list = [op.join(fit_path, x) for x in os.listdir(fit_path) if fit_model in x and hemi in x and 'HRF' not in x and 'chunk' not in x]
                
                ## if we defined a base filename that should be used to fish out right estimates
                if basefilename:
                    hemi_name = [file for file in hemi_name_list if basefilename in file][0]
                else:
                    hemi_name = hemi_name_list[0]

                print('loading hemi %s'%hemi_name)
                chunk = np.load(hemi_name) # load chunk
                
                if h_ind == 0:
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
                    # if we cross validated
                    if crossval:
                        cv_r2 = chunk['cv_r2']
                    else:
                        cv_r2 = np.zeros(xx.shape) 

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

                    # if we cross validated
                    if crossval:
                        cv_r2 = np.concatenate((cv_r2, chunk['cv_r2'])) 
                    else:
                        cv_r2 = np.concatenate((cv_r2, np.zeros(chunk['r2'].shape))) 
                    
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
                        r2 = rsq,
                        cv_r2 = cv_r2)

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
                        r2 = rsq,
                        cv_r2 = cv_r2)

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
                        r2 = rsq,
                        cv_r2 = cv_r2)

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
                        r2 = rsq,
                        cv_r2 = cv_r2)
            
        else:
            print('file already exists, loading %s'%filename)
        
        return np.load(filename)

    def load_pRF_model_chunks(self, fit_path, fit_model = 'css', fit_hrf = False, basefilename = None, 
                                    overwrite = False, iterative = True, crossval = False):

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
            filename_list = [op.join(fit_path, x) for x in os.listdir(fit_path) if fit_model in x and 'chunk-000' in x and 'HRF' not in x]
        
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
                    chunk_name_list = [op.join(fit_path, x) for x in os.listdir(fit_path) if fit_model in x and 'chunk-%s'%str(ch).zfill(3) in x and 'HRF' not in x]
                
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
                    # if we cross validated
                    if crossval:
                        cv_r2 = chunk['cv_r2']
                    else:
                        cv_r2 = np.zeros(xx.shape) 

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

                    # if we cross validated
                    if crossval:
                        cv_r2 = np.concatenate((cv_r2, chunk['cv_r2'])) 
                    else:
                        cv_r2 = np.concatenate((cv_r2, np.zeros(chunk['r2'].shape))) 
                    
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
                        r2 = rsq,
                        cv_r2 = cv_r2)

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
                        r2 = rsq,
                        cv_r2 = cv_r2)

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
                        r2 = rsq,
                        cv_r2 = cv_r2)

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
                        r2 = rsq,
                        cv_r2 = cv_r2)
            
        else:
            print('file already exists, loading %s'%filename)
        
        return np.load(filename)

    def save_pRF_model_estimates(self, filename, final_estimates, model_type = 'gauss', grid = False, cv_r2 = None):
    
        """
        save estimates (key,value) in npz dict
                
        Parameters
        ----------
        filename : str
            absolute filename of estimates to be saved
        final_estimates : arr
            2d estimates (datapoints,estimates)
        model_type: str
            model type used for fitting
        cv_r2: arr
            crossvalidation r2
        
        """ 

        # make dir if it doesnt exist already
        os.makedirs(op.split(filename)[0], exist_ok = True)

        if cv_r2 is None:
            cv_r2 = np.zeros(final_estimates.shape[0]); cv_r2[:] = np.nan
                
        if model_type == 'gauss':
                
            np.savez(filename,
                    x = final_estimates[..., 0],
                    y = final_estimates[..., 1],
                    size = final_estimates[..., 2],
                    betas = final_estimates[...,3],
                    baseline = final_estimates[..., 4],
                    hrf_derivative = final_estimates[..., 5],
                    hrf_dispersion = final_estimates[..., 6], 
                    r2 = final_estimates[..., 7],
                    cv_r2 = cv_r2)
        
        elif model_type == 'css':

            np.savez(filename,
                    x = final_estimates[..., 0],
                    y = final_estimates[..., 1],
                    size = final_estimates[..., 2],
                    betas = final_estimates[...,3],
                    baseline = final_estimates[..., 4],
                    ns = final_estimates[..., 5],
                    hrf_derivative = final_estimates[..., 6],
                    hrf_dispersion = final_estimates[..., 7], 
                    r2 = final_estimates[..., 8],
                    cv_r2 = cv_r2)

        elif model_type == 'dn':

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
                    r2 = final_estimates[..., 11],
                    cv_r2 = cv_r2)

        elif model_type == 'dog':

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
                    r2 = final_estimates[..., 9],
                    cv_r2 = cv_r2)
            
    def crossvalidate(self, test_data, model_object = None, estimates = [], avg_hrf = False, n_jobs = 8):

        """
        Use previously fit parameters to obtain cross-validated Rsq
        on test data

        Parameters
        ----------
        test_data: arr
            Test data for crossvalidation
        model_object: pRF model object
            fitter object, from prfpy, to call apropriate return prediction function
        estimates: arr
            2D array of iterative estimates (vertex, estimates), in same order of fitter
        avg_hrf: bool
            if we want to average hrf estimates across vertices or not (only used if HRF was fit)

        """

        # if we fit HRF, use median HRF estimates for crossvalidation
        if (self.fit_hrf) and (avg_hrf) and (estimates.shape[0] > 1): 
            median_hrf_params = np.nanmedian(estimates[:,-3:-1], axis = 0)
            estimates[:,-3:-1] = median_hrf_params

        # get model prediciton for all vertices
        prediction = Parallel(n_jobs=n_jobs, verbose=10)(delayed(model_object.return_prediction)(*list(estimates[vert, :-1])) for vert in tqdm(range(test_data.shape[0])))
        prediction = np.squeeze(prediction, axis=1)

        #calculate CV-rsq        
        CV_rsq = np.nan_to_num(1-np.sum((test_data-prediction)**2, axis=-1)/(test_data.shape[-1]*test_data.var(-1)))
        
        return CV_rsq

    def mask_pRF_model_estimates(self, estimates, vertex = [], x_ecc_lim = [-6,6], y_ecc_lim = [-6,6],
                                rsq_threshold = .1, estimate_keys = ['x','y','size','betas','baseline','r2']):
    
        """ 
        mask estimates, to be positive RF, within screen limits
        and for a certain vertex list (if the case)

        Parameters
        ----------
        estimates : dict
            dict with estimates key-value pairs
        vertex : list
            list of vertices to mask estimates 
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

        # if we want to subselect for specifc vertices
        if len(vertex) > 0:
            for k in estimate_keys:
                masked_dict[k] = masked_dict[k][vertex]
        
        return masked_dict
    
    def get_prf_estimate_keys(self, prf_model_name = 'gauss'):

        """ 
        Helper function to get prf estimate keys
        
        Parameters
        ----------
        prf_model_name : str
            pRF model name (defaults to gauss)
            
        """

        # get estimate key names, which vary per model used
        keys = self.MRIObj.params['mri']['fitting']['pRF']['estimate_keys'][prf_model_name]
        
        if self.fit_hrf:
            keys = keys[:-1]+self.MRIObj.params['mri']['fitting']['pRF']['estimate_keys']['hrf']+['r2']

        return keys
    
    def get_eccentricity(self, xx = [], yy = [], rsq = []):

        """
        Helper function that calculates eccentricity and returns array 

        Parameters
        ----------
        xx : arr
            array with x position values
        yy : arr
            array with y position values
        rsq: arr
            rsq values, to be used as alpha level/threshold
        """

        ## calculate eccentricity
        eccentricity = np.abs(xx + yy * 1j)

        # mask nans
        eccentricity[np.where((np.isnan(rsq)))[0]] = np.nan

        return eccentricity
    
    def get_polar_angle(self, xx = [], yy = [], rsq = [], pa_transform = 'mirror', angle_thresh = 3*np.pi/4):

        """
        Helper function that calculates PA and returns array of
        PA values (which can also be transformed in some way)

        Parameters
        ----------
        xx : arr
            array with x position values
        yy : arr
            array with y position values
        rsq: arr
            rsq values, to be used as alpha level/threshold
        pa_transform: str
            if None, no transform, else will be:
            mirror - hemipsheres are mirrored, and meridians are over represented
            norm - normalized PA between 0 and 1
        """

        ## calculate polar angle
        polar_angle = np.angle(xx + yy * 1j)

        if pa_transform is not None:

            ## get mid vertex index (diving hemispheres)
            left_index = cortex.db.get_surfinfo(self.pysub).left.shape[0] 

            polar_angle_out = polar_angle.copy()

            if angle_thresh is not None:
                # mask out angles within threh interval
                polar_angle_out[:left_index][np.where((polar_angle[:left_index] > angle_thresh) | (polar_angle[:left_index] < -angle_thresh))[0]] = np.nan
                polar_angle_out[left_index:][np.where((polar_angle[left_index:] > (-np.pi + angle_thresh)) & (polar_angle[left_index:] < (np.pi - angle_thresh)))[0]] = np.nan
                polar_angle = polar_angle_out.copy()

            if pa_transform == 'mirror':
                ## get pa values transformed like in figure 8 of Larsson and Heeger 2006
                # --> Horizontal meridian = 0
                # --> upper VF goes from 0 to pi/2
                # --> lower VF goes from 0 to -pi/2

                # angles from pi/2 to pi (upper left quadrant)
                ind_ang = np.where((polar_angle > np.pi/2))[0]
                polar_angle_out[ind_ang] = (polar_angle_out[ind_ang] - np.pi)* -1 # minus pi, then taking absolute (positive) value

                # angles from -pi/2 to -pi (lower left quadrant)
                ind_ang = np.where((polar_angle < -np.pi/2))[0]
                polar_angle_out[ind_ang] = (polar_angle_out[ind_ang] + np.pi) * -1 

            if pa_transform == 'flip':
                # non-uniform representation. we flip vertically to make sure
                # order of colors same for both hemispheres
                ind_nan = np.where((~np.isnan(polar_angle_out[left_index:])))[0]
                polar_angle_out[left_index:][ind_nan] = np.angle(-1*xx + yy * 1j)[left_index:][ind_nan] 

            elif pa_transform == 'norm':
                polar_angle_out = ((polar_angle + np.pi) / (np.pi * 2.0)) # normalize PA between 0 and 1
                
        else:
            polar_angle_out = polar_angle

        polar_angle_out[np.where((np.isnan(rsq)))[0]] = np.nan

        return polar_angle_out
    
    def fwhmax_fwatmin(self, model, estimates, normalize_RFs=False, return_profiles=False):
    
        """
        taken from marco aqil's code, all credits go to him
        """
        
        model = model.lower()
        x=np.linspace(-50,50,1000).astype('float32')

        prf = estimates['betas'] * np.exp(-0.5*x[...,np.newaxis]**2 / estimates['size']**2)
        vol_prf =  2*np.pi*estimates['size']**2

        if 'dog' in model or 'dn' in model:
            srf = estimates['sa'] * np.exp(-0.5*x[...,np.newaxis]**2 / estimates['ss']**2)
            vol_srf = 2*np.pi*estimates['ss']*2

        if normalize_RFs==True:

            if model == 'gauss':
                profile =  prf / vol_prf
            elif model == 'css':
                #amplitude is outside exponent in CSS
                profile = (prf / vol_prf)**estimates['ns'] * estimates['betas']**(1 - estimates['ns'])
            elif model =='dog':
                profile = prf / vol_prf - \
                        srf / vol_srf
            elif 'dn' in model:
                profile = (prf / vol_prf + estimates['nb']) /\
                        (srf / vol_srf + estimates['sb']) - estimates['nb']/estimates['sb']
        else:
            if model == 'gauss':
                profile = prf
            elif model == 'css':
                #amplitude is outside exponent in CSS
                profile = prf**estimates['ns'] * estimates['betas']**(1 - estimates['ns'])
            elif model =='dog':
                profile = prf - srf
            elif 'dn' in model:
                profile = (prf + estimates['nb'])/(srf + estimates['sb']) - estimates['nb']/estimates['sb']


        half_max = np.max(profile, axis=0)/2
        fwhmax = np.abs(2*x[np.argmin(np.abs(profile-half_max), axis=0)])

        if 'dog' in model or 'dn' in model:

            min_profile = np.min(profile, axis=0)
            fwatmin = np.abs(2*x[np.argmin(np.abs(profile-min_profile), axis=0)])

            result = fwhmax, fwatmin
        else:
            result = fwhmax, np.nan

        if return_profiles:
            return result, profile.T
        else:
            return result
    
