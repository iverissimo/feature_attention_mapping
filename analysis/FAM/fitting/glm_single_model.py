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

from glmsingle.glmsingle import GLM_single
from glmsingle.glmsingle import getcanonicalhrf

import time
import cortex
import matplotlib.pyplot as plt


class GLMsingle_Model(Model):

    def __init__(self, MRIObj, outputdir = None, pysub = 'hcp_999999', use_atlas = None):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
        outputdir: str or None
            path to general output directory
            
        """

        # need to initialize parent class (Model), indicating output infos
        super().__init__(MRIObj = MRIObj, outputdir = outputdir, pysub = pysub, use_atlas = use_atlas)

        ## prf rsq threshold, to select visual voxels
        # worth fitting
        self.prf_rsq_threshold = self.MRIObj.params['mri']['fitting']['FA']['prf_rsq_threshold']

        # prf estimate bounds
        self.prf_bounds = None

        ## Not correcting baseline
        self.correct_baseline['FA'] = False

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.MRIObj.derivatives_pth, self.fitfolder['FA'], 'glmsingle')
        else:
            self.outputdir = outputdir

        ## get conditions per TR
        FA_bar_pass_all, _ = self.MRIObj.beh_utils.get_FA_run_struct(self.MRIObj.FA_bar_pass, 
                                                                num_bar_pos = self.MRIObj.FA_num_bar_position, 
                                                                empty_TR = self.MRIObj.FA_nr_TRs['empty_TR'], 
                                                                task_trial_TR = self.MRIObj.FA_nr_TRs['task_trial_TR'])

        # crop and shift if such was the case
        self.condition_per_TR = self.MRIObj.mri_utils.crop_shift_arr(FA_bar_pass_all, 
                                                            crop_nr = self.MRIObj.task_nr_cropTR['FA'], 
                                                            shift = self.MRIObj.shift_TRs_num)

    def get_correlation_mask(self, participant, task = 'pRF', ses = 'mean', file_ext = '_cropped_dc_psc.npy',
                                n_jobs = 8, seed_num = 2023, perc_thresh_nm = 95, smooth = True,
                                kernel=3, nr_iter=3, normalize = False, figure_name = None):

        """
        Split half correlate all runs in a task
        and create binary mask of task relevant vertices
        to be used in noise pool

        Parameters
        ----------
        participant: str
            participant ID
        prf_estimates : dict
            dict with participant prf estimates
        smooth_nm: bool
            if we want to smooth noise mask
        perc_thresh_nm: int
            noise mask percentile threshold
        file_extent_nm: dict
            dict with file extension for task files to be used for noise mask
        perc_thresh_nm: int
            noise mask percentile threshold
        smooth: bool
            if we want to return smoothed mask
        """

        # get bold filenames
        task_bold_files = self.MRIObj.mri_utils.get_bold_file_list(participant, task = task, 
                                                                ses = ses, file_ext = file_ext,
                                                                postfmriprep_pth = self.MRIObj.postfmriprep_pth, 
                                                                acq_name = self.MRIObj.acq)
        
        ## find unique session number
        task_ses_num = np.unique([self.MRIObj.mri_utils.get_run_ses_from_str(f)[-1] for f in task_bold_files])

        ## for each session, get split half correlation values
        corr_arr = []
        random_corr_arr = []
        ind_seed = int(participant) # seed index, to vary across participants

        for sn in task_ses_num:

            ses_files = [f for f in task_bold_files if 'ses-{s}'.format(s = sn) in f]

            ## split runs in half and get unique combinations
            run_sh_lists = self.MRIObj.mri_utils.split_half_comb(ses_files)

            # get correlation value for each combination
            for r in run_sh_lists:
                ## correlate the two halfs
                corr_arr.append(self.MRIObj.mri_utils.correlate_arrs(list(r[0]), list(r[-1]), 
                                                                     n_jobs = n_jobs, shuffle_axis = None, seed=None))
                ## correlate with randomized half
                random_corr_arr.append(self.MRIObj.mri_utils.correlate_arrs(list(r[0]), list(r[-1]), 
                                                                            n_jobs = n_jobs, shuffle_axis = -1, 
                                                                            seed = int(seed_num * ind_seed)))
                ind_seed += 1 # and update ind just as extra precaution

        # average values 
        task_avg_sh_corr = np.nanmean(corr_arr, axis = 0)
        task_avg_sh_rand_corr = np.nanmean(random_corr_arr, axis = 0)

        threshold = np.nanpercentile(task_avg_sh_rand_corr, perc_thresh_nm)

        print('X percentile for {tsk} correlation mask is {val}'.format(tsk = task, val = '%.3f'%threshold))

        ## make final mask
        if smooth:
            final_corr_arr = self.MRIObj.mri_utils.smooth_surface(task_avg_sh_corr, pysub = self.pysub, 
                                                            kernel = kernel, nr_iter = nr_iter, normalize = normalize)
        else:
            final_corr_arr = task_avg_sh_corr

        # we want to exclude vertices above threshold
        binary_mask = np.ones(final_corr_arr.shape)
        binary_mask[final_corr_arr >= threshold] = 0

        # if we want to plot --> will change to just save correlation file somewhere
        if figure_name:
            ## plot pRF correlation flatmap used for mask
            flatmap = cortex.Vertex(final_corr_arr, 
                            self.pysub,
                            vmin = 0, vmax = 1, #.7,
                            cmap='hot')
            cortex.quickshow(flatmap, with_curvature=True,with_sulci=True, with_labels=False)

            print('saving %s' %figure_name)
            _ = cortex.quickflat.make_png(figure_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False) 

        return binary_mask

    def get_single_trial_combinations(self):

        """
        Get all possible trial combinations
        (useful to keep track of single trial DM later)

        Returns DataFrame where each row is a unique trial type
        Columns: 
            - attended and unattended bar midpoint position (x,y)
            - bar pass direction (vertical vs horizontal)
        """

        # define bar width in pixel
        bar_width_pix = self.MRIObj.screen_res * self.MRIObj.bar_width['FA']

        # define number of bars per direction
        num_bars = np.array(self.MRIObj.FA_num_bar_position) 

        # all possible positions in pixels [x,y] for midpoint of
        # vertical bar passes, 
        ver_y = np.sort(np.concatenate((-np.arange(bar_width_pix[1]/2,self.MRIObj.screen_res[1]/2,bar_width_pix[1])[0:int(num_bars[1]/2)],
                                        np.arange(bar_width_pix[1]/2,self.MRIObj.screen_res[1]/2,bar_width_pix[1])[0:int(num_bars[1]/2)])))

        ver_bar_pos_pix = np.array([np.array([0,y]) for _,y in enumerate(ver_y)])

        # horizontal bar passes 
        hor_x = np.sort(np.concatenate((-np.arange(bar_width_pix[0]/2,self.MRIObj.screen_res[0]/2,bar_width_pix[0])[0:int(num_bars[0]/2)],
                                        np.arange(bar_width_pix[0]/2,self.MRIObj.screen_res[0]/2,bar_width_pix[0])[0:int(num_bars[0]/2)])))

        hor_bar_pos_pix = np.array([np.array([x,0]) for _,x in enumerate(hor_x)])

        ## make all possible combinations
        pos_dict = {'horizontal': hor_bar_pos_pix, 'vertical': ver_bar_pos_pix}
        attend_orientation = ['vertical','horizontal']
        unattend_orientation = ['vertical','horizontal']

        # total number of trials
        num_trials = len(attend_orientation)*(pos_dict['horizontal'].shape[0] * pos_dict['vertical'].shape[0] + \
                                            pos_dict['horizontal'].shape[0] * (pos_dict['horizontal'].shape[0]-1))

        print('number of bar trials is %i'%num_trials)

        # define dictionary to save positions and directions
        # of all bars
        trial_combinations_dict = {'AttBar_bar_midpoint': [], 'AttBar_bar_pass_direction': [],
                                    'UnattBar_bar_midpoint': [], 'UnattBar_bar_pass_direction': []}

        # append all postions in dict 
        for att_ori in attend_orientation:

            for unatt_ori in unattend_orientation:

                if att_ori != unatt_ori: # if bar orientations orthogonal

                    indice_pairs = list((x,y) for x in np.arange(pos_dict[att_ori].shape[0]) for y in np.arange(pos_dict[unatt_ori].shape[0]))

                else: # if bar orientations the same

                    indice_pairs = list(itertools.permutations(np.arange(pos_dict[att_ori].shape[0]), 2))

                # fill attended dict
                trial_combinations_dict['AttBar_bar_midpoint'].append(np.array([pos_dict[att_ori][i] for i in np.array(indice_pairs)[...,0]]))
                trial_combinations_dict['AttBar_bar_pass_direction'].append(np.tile(att_ori, np.array(indice_pairs).shape[0]))

                # fill unattended dict
                trial_combinations_dict['UnattBar_bar_midpoint'].append(np.array([pos_dict[unatt_ori][i] for i in np.array(indice_pairs)[...,-1]]))
                trial_combinations_dict['UnattBar_bar_pass_direction'].append(np.tile(unatt_ori, np.array(indice_pairs).shape[0]))

        ## turn into dataframe
        self.trial_combinations_df = pd.DataFrame.from_dict(trial_combinations_dict).apply(pd.Series.explode).reset_index().drop(columns=['index'])

    def make_singletrial_dm(self, run_num_arr =[], ses_num_arr = [], pp_bar_pos_df = {}):

        """
        Make single trial design matrix
        for one or more runs

        Parameters
        ----------
        participant: str
            participant ID
        run_num_arr: list
            list of ints with each run number to add to the design matrix 
            (DM will be in same order of runs IDs in this list)
        ses_num_arr: list
            list of ints with each ses number of the abovementioned run
            (DM will be in same order of ses IDs in this list)
        pp_bar_pos_df: dataframe
            participant bar position data frame, for all runs of task (from preproc_beh class)
        """

        ## get all possible trial combinations
        # to use for bookkeeping of single trial DM
        try:
            self.trial_combinations_df
        except AttributeError:
            self.get_single_trial_combinations()

        ## make single trial DM
        # with shape [runs, TRs, conditions]
        single_trl_DM = np.zeros((len(run_num_arr), len(self.condition_per_TR), len(self.trial_combinations_df)))

        ## loop over runs
        for file_ind in range(len(run_num_arr)):

            ses_num = ses_num_arr[file_ind]
            run_num = run_num_arr[file_ind]

            ## get bar position df for run
            run_bar_pos_df = pp_bar_pos_df['ses-{s}'.format(s = ses_num)]['run-{r}'.format(r=run_num)]

            ## get run bar midpoint and direction values
            # for each bar type
            AttBar_bar_midpoint = run_bar_pos_df[run_bar_pos_df['attend_condition'] == 1].bar_midpoint_at_TR.values[0]
            AttBar_bar_pass_direction = run_bar_pos_df[run_bar_pos_df['attend_condition'] == 1].bar_pass_direction_at_TR.values[0]

            UnattBar_bar_midpoint = run_bar_pos_df[run_bar_pos_df['attend_condition'] == 0].bar_midpoint_at_TR.values[0]
            UnattBar_bar_pass_direction = run_bar_pos_df[run_bar_pos_df['attend_condition'] == 0].bar_pass_direction_at_TR.values[0]

            # set trial index counter
            trl_ind = 0

            ## fill DM for all TRs
            for i_TR, cond in enumerate(self.condition_per_TR):
                
                if cond == 'task':
                    
                    ## get condition index 
                    # where midpoint and direction for both bars is the same as the one of this trial
                    cond_index = self.trial_combinations_df[(self.trial_combinations_df['AttBar_bar_midpoint'].apply(lambda x: str(AttBar_bar_midpoint[trl_ind]) == str(x))) &\
                                (self.trial_combinations_df['AttBar_bar_pass_direction'].apply(lambda x: str(AttBar_bar_pass_direction[trl_ind]) == str(x))) &\
                                (self.trial_combinations_df['UnattBar_bar_midpoint'].apply(lambda x: str(UnattBar_bar_midpoint[trl_ind]) == str(x))) &\
                                (self.trial_combinations_df['UnattBar_bar_pass_direction'].apply(lambda x: str(UnattBar_bar_pass_direction[trl_ind]) == str(x)))].index[0]

                    # set which condition had its onset at that TR
                    single_trl_DM[file_ind, i_TR, cond_index] = 1
                    
                    # increment trial counter
                    trl_ind += 1

        return single_trl_DM

    def get_average_hrf(self, pp_prf_estimates, prf_modelobj, rsq_threshold = None):

        """
        Make average pRF to give as input to glm single model
        Requires previously obtained HRF params from pRF fitting and 
        Defined pRF models

        Parameters
        ----------
        pp_prf_estimates : dict
            dict with participant prf estimates
        prf_modelobj: object
            pRF model object from prfpy, to use to create HRF
        rsq_threshold: float or None
            fit vertices where prf fit above certain rsq threshold 
            
        """

        ## find indices where pRF rsq high
        rsq_threshold = self.prf_rsq_threshold if rsq_threshold is None else rsq_threshold

        ind2use = np.where((pp_prf_estimates['r2'] > rsq_threshold))[0]
        print('selecting %i HRFs to average'%len(ind2use))

        ## make hrfs for all high rsq visual voxels
        # shifted by onset (stc) and upsampled
        hrf_ind2use = [prf_modelobj.create_hrf(hrf_params = [1, 
                                                            pp_prf_estimates['hrf_derivative'][vert],
                                                            pp_prf_estimates['hrf_dispersion'][vert]], 
                                                            osf = self.osf * self.MRIObj.TR, 
                                                            onset = self.hrf_onset) for vert in ind2use]
        hrf_ind2use = np.vstack(hrf_ind2use)

        ## average HRF, weighted by the pRF RSQ
        avg_hrf = np.average(hrf_ind2use, axis=0, weights=self.MRIObj.mri_utils.normalize(pp_prf_estimates['r2'][ind2use]))

        ## convolve to get the predicted response 
        # to the desired stimulus duration
        stim_dur = self.MRIObj.FA_bars_phase_dur # duration of bar presentation in seconds
        res_step = self.MRIObj.TR/(self.MRIObj.TR * self.osf) # resolution of upsampled HRF

        hrf_stim_convolved = np.convolve(avg_hrf, np.ones(int(np.max([1, np.round(stim_dur/res_step)]))))

        ## now resample again to the TR
        hrf_final = pchip(np.asarray(range(hrf_stim_convolved.shape[0])) * res_step,
                        hrf_stim_convolved)(np.asarray(np.arange(0, int((hrf_stim_convolved.shape[0]-1) * res_step), self.MRIObj.TR)))
        
        return hrf_final/np.max(hrf_final)

    def subselect_trial_combinations(self, att_bar_xy = [], unatt_bar_xy = [], orientation_bars = None):

        """
        subselect trial combinations according to (un)attended bar position
        or bar orientations, and obtain dataframe with possible trials
        """

        try:
            self.single_trial_reference_df
        except AttributeError:
            self.get_single_trial_reference_df()

        # get reference df
        ref_df = self.single_trial_reference_df.copy()

        if len(att_bar_xy) > 1: # if provided specific coordinates for attended bar
            ref_df = ref_df[(ref_df['AttBar_coord_x'].isin([att_bar_xy[0]])) &\
                             (ref_df['AttBar_coord_y'].isin([att_bar_xy[-1]]))]
        
        if len(unatt_bar_xy) > 1: # if provided specific coordinates for unattended bar
            ref_df = ref_df[(ref_df['UnattBar_coord_x'].isin([unatt_bar_xy[0]])) &\
                            (ref_df['UnattBar_coord_y'].isin([unatt_bar_xy[-1]]))]
            
        if orientation_bars is not None:
            ref_df = ref_df[ref_df['orientation_bars'] == orientation_bars]

        return ref_df

    def get_singletrial_avg_estimates(self, estimate_arr = [], single_trl_DM = [], return_std = True,
                                            att_bar_xy = [], unatt_bar_xy = [], average_betas = True):

        """
        Helper function that takes in an estimate array from glmsingle
        [vertex, singletrials] (note: single trial number is multiplied by nr of runs)
        
        and outputs an average value for each single trial type (our condition)
        [vertex, average_estimate4trialtype]

        """

        # get reference df
        ref_df = self.subselect_trial_combinations(att_bar_xy = att_bar_xy, unatt_bar_xy = unatt_bar_xy)
            
        # get single trial indices to extract betas from
        single_trial_ind = ref_df.ind.values.astype(int)

        ## now append the estimate for that vertex for the same trial type (and std if we also want that)
        avg_all = []
        std_all = []

        for i in single_trial_ind:

            ## indices select for task on TRs (trials)
            cond_ind = np.where(np.hstack(single_trl_DM[:, self.condition_per_TR == 'task', i] == 1))[0]
            
            if average_betas:
                avg_all.append(np.mean(estimate_arr[...,cond_ind], axis = -1))
            else:
                avg_all.append(estimate_arr[...,cond_ind])
            if return_std:
                std_all.append(np.std(estimate_arr[...,cond_ind], axis = -1))
            
        if return_std:
            return np.stack(avg_all), np.stack(std_all)
        else:
            return np.stack(avg_all)
        
    def get_single_trial_reference_df(self):

        """
        make reference dataframe of bar positions
        to facilitate getting single-trials estimates from DM
        """

        ## get all possible trial combinations
        try:
            self.trial_combinations_df
        except AttributeError:
            self.get_single_trial_combinations()

        single_trial_reference_df = pd.DataFrame({'ind': [], 'AttBar_coord_x': [], 'AttBar_coord_y': [], 
                                                'UnattBar_coord_x': [], 'UnattBar_coord_y': [], 'orientation_bars': []})

        for ind, row in self.trial_combinations_df.iterrows():
            
            # save relevant coordinates
            att_bar_xy = row['AttBar_bar_midpoint']
            unatt_bar_xy = row['UnattBar_bar_midpoint']
                
            # save bar orientations just for ease later on
            if (row['AttBar_bar_pass_direction'] == 'vertical' and row['UnattBar_bar_pass_direction'] == 'vertical'):
                
                orientation_bars = 'parallel_horizontal'
                
            elif (row['AttBar_bar_pass_direction'] == 'horizontal' and row['UnattBar_bar_pass_direction'] == 'horizontal'):
                
                orientation_bars = 'parallel_vertical'
                
            else:
                orientation_bars = 'crossed'
                
            # and replace not relevant coord with nan, for bookeeping
            if row['AttBar_bar_pass_direction'] == 'horizontal':
                att_bar_xy[-1] = np.nan 
            else:
                att_bar_xy[0] = np.nan 
                
            if row['UnattBar_bar_pass_direction'] == 'horizontal':
                unatt_bar_xy[-1] = np.nan 
            else:
                unatt_bar_xy[0] = np.nan 
                
            # append in df
            single_trial_reference_df = pd.concat((single_trial_reference_df,
                                                pd.DataFrame({'ind': [ind], 
                                                                'AttBar_coord_x': [att_bar_xy[0]], 
                                                                'AttBar_coord_y': [att_bar_xy[-1]], 
                                                                'UnattBar_coord_x': [unatt_bar_xy[0]], 
                                                                'UnattBar_coord_y': [unatt_bar_xy[-1]], 
                                                                'orientation_bars': [orientation_bars]})))
            
        self.single_trial_reference_df = single_trial_reference_df

    def fit_data(self, participant, pp_prf_estimates, prf_modelobj,  file_ext = '_cropped.npy', 
                        smooth_nm = True, perc_thresh_nm = 95, n_jobs = 8,
                        seed_num = 2023, kernel = 3, nr_iter = 3, normalize = False,
                        pp_bar_pos_df = {},
                        file_extent_nm = {'pRF': '_cropped_dc_psc.npy', 'FA': '_cropped_LinDetrend_psc.npy'}):

        """
        fit GLM single on participant data

        Parameters
        ----------
        participant: str
            participant ID
        prf_estimates : dict
            dict with participant prf estimates
        prf_modelobj: object
            pRF model object from prfpy, to use to create HRF
        smooth_nm: bool
            if we want to smooth noise mask
        perc_thresh_nm: int
            noise mask percentile threshold
        file_extent_nm: dict
            dict with file extension for task files to be used for noise mask
        pp_bar_pos_df: dataframe
            participant bar position data frame, for all runs of task (from preproc_beh class)
        """ 

        ## set output dir to save estimates
        outdir = op.join(self.outputdir, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant))

        os.makedirs(outdir, exist_ok = True)
        print('saving files in %s'%outdir)

        ## get list of files to load
        bold_filelist = self.MRIObj.mri_utils.get_bold_file_list(participant, task = 'FA', ses = 'all', file_ext = file_ext,
                                                                postfmriprep_pth = self.MRIObj.postfmriprep_pth, 
                                                                acq_name = self.MRIObj.acq)

        ## Load data array and file list names
        data, train_file_list = self.get_data4fitting(bold_filelist, task = 'FA', run_type = 'all', 
                                                chunk_num = None, vertex = None, ses = 'all',
                                                baseline_interval = 'empty', correct_baseline = None, return_filenames = True)
    
        ## Make single trial DM for all runs
        single_trl_DM = self.make_singletrial_dm(run_num_arr = self.run_num_arr, 
                                                ses_num_arr = self.ses_num_arr,
                                                pp_bar_pos_df = pp_bar_pos_df)

        print('Fitting {n} files: {f}'.format(n = len(train_file_list), f = str(train_file_list)))

        ## get average hrf
        hrf_final = self.get_average_hrf(pp_prf_estimates, prf_modelobj, rsq_threshold = self.prf_rsq_threshold)
        
        ### make mask array of pRF high fitting voxels,
        # to give as input to glmsingle (excluding them from noise pool)
        binary_prf_mask = self.get_correlation_mask(participant, task = 'pRF', ses = 'mean', 
                                                    file_ext = file_extent_nm['pRF'], n_jobs = n_jobs, 
                                                    seed_num = seed_num, perc_thresh_nm = perc_thresh_nm, 
                                                    smooth = smooth_nm, kernel = kernel, nr_iter = nr_iter, 
                                                    normalize = normalize, 
                                                    figure_name = op.join(outdir, 'modeltypeD_pRFcorrelation.png'))

        ## now do the same correlation mask for the FA runs
        binary_fa_mask = self.get_correlation_mask(participant, task = 'FA', ses = 'mean', 
                                                    file_ext = file_extent_nm['FA'], n_jobs = n_jobs, 
                                                    seed_num = int(seed_num * 2), perc_thresh_nm = perc_thresh_nm, 
                                                    smooth = smooth_nm, kernel = kernel, nr_iter = nr_iter, 
                                                    normalize = normalize,
                                                    figure_name = op.join(outdir, 'modeltypeD_FAcorrelation.png'))

        ### final mask is multiplication of the two
        final_mask = binary_fa_mask * binary_prf_mask

        # create a directory for saving GLMsingle outputs
        opt = dict()

        # set important fields for completeness (but these would be enabled by default)
        opt['wantlibrary'] = 0
        opt['wantglmdenoise'] = 1
        opt['wantfracridge'] = 1
        opt['hrfonset'] = 0 # already setting onset in hrf
        opt['hrftoassume'] = hrf_final
        opt['brainexclude'] = final_mask.astype(int)
        opt['sessionindicator'] = self.ses_num_arr 
        opt['brainthresh'] = [99, 0] # which allows all voxels to pass the intensity threshold --> we use surface data
        opt['brainR2'] = 100 # not using on-off model for noise pool

        # define polynomials to project out from data (we only want to use intercept and slope)
        opt['maxpolydeg'] = [[0, 1] for _ in range(data.shape[0])]

        # keep relevant outputs in memory and also save them to the disk
        opt['wantfileoutputs'] = [1,1,1,1]
        opt['wantmemoryoutputs'] = [1,1,1,1]

        # running python GLMsingle involves creating a GLM_single object
        # and then running the procedure using the .fit() routine
        glmsingle_obj = GLM_single(opt)

        # visualize all the hyperparameters
        print(glmsingle_obj.params)

        ## seems that data and design needs to be list of arrays
        #
        data_list = []
        dm_list = []
        for r_ind in range(data.shape[0]):
            
            data_list.append(data[r_ind])
            dm_list.append(single_trl_DM[r_ind])

        ## actually run it
        start_time = time.time()

        print(f'running GLMsingle...')

        # run GLMsingle
        results_glmsingle = glmsingle_obj.fit(dm_list,
                                            data_list,
                                            self.MRIObj.FA_bars_phase_dur,
                                            self.MRIObj.TR,
                                            outputdir = outdir)

        elapsed_time = time.time() - start_time

        print(
            '\telapsed time: ',
            f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
        )

        ## for now plot and save a few inspection figures too,
        # to see get a sense on quality of fit

        ## plot ON OFF R2
        flatmap = cortex.Vertex(results_glmsingle['typea']['onoffR2'], 
                  self.pysub,
                   vmin = 0, vmax = 15, #.7,
                   cmap='hot')

        fig_name = op.join(outdir, 'modeltypeA_ONOFF_rsq.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        ## plot ON OFF betas
        flatmap = cortex.Vertex(results_glmsingle['typea']['betasmd'][...,0], 
                  self.pysub,
                   vmin = -2, vmax = 2, #.7,
                   cmap='RdBu_r')

        fig_name = op.join(outdir, 'modeltypeA_ONOFF_betas.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)
        
        ## plot Full Model noise pool 
        flatmap = cortex.Vertex(results_glmsingle['typed']['noisepool'], 
                        self.pysub,
                        vmin = 0, vmax = 1, #.7,
                        cmap='hot')

        fig_name = op.join(outdir, 'modeltypeD_noisepool.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        ## plot Full Model RSQ
        flatmap = cortex.Vertex(results_glmsingle['typed']['R2'], 
                  self.pysub,
                   vmin = 0, vmax = 50, #.7,
                   cmap='hot')
        
        fig_name = op.join(outdir, 'modeltypeD_rsq.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        ## plot Full Model betas
        flatmap = cortex.Vertex(np.mean(results_glmsingle['typed']['betasmd'], axis = -1), 
                  self.pysub,
                   vmin = -2, vmax = 2, #.7,
                   cmap='RdBu_r')

        fig_name = op.join(outdir, 'modeltypeD_betas.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        ## plot Full Model betas for only high fitting RSQ
        avg_betas = np.mean(results_glmsingle['typed']['betasmd'], axis = -1)
        avg_betas[pp_prf_estimates['r2']< self.prf_rsq_threshold] = np.nan

        flatmap = cortex.Vertex(avg_betas, 
                        self.pysub,
                        vmin = -2, vmax = 2, #.7,
                        cmap='RdBu_r')

        fig_name = op.join(outdir, 'modeltypeD_betas_ROIpRF.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        ## plot Full Model FracRidge
        flatmap = cortex.Vertex(results_glmsingle['typed']['FRACvalue'], 
                  self.pysub,
                   vmin = 0, vmax = 1, #.7,
                   cmap='copper')

        fig_name = op.join(outdir, 'modeltypeD_fracridge.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        ## plot average HRF
        fig, axis = plt.subplots(1,figsize=(8,5),dpi=100)
        axis.plot(getcanonicalhrf(self.MRIObj.FA_bars_phase_dur, self.MRIObj.TR, onset = self.hrf_onset), label = 'canonical_hrf')
        axis.plot(hrf_final, label = 'average_hrf')
        axis.set_xlabel('Time (TR)')
        axis.legend()

        plt.savefig(op.join(outdir, 'hrf_avg.png'))

        ## plot pRF binary mask
        flatmap = cortex.Vertex(binary_prf_mask, 
                        self.pysub,
                        vmin = 0, vmax = 1, #.7,
                        cmap='hot')
        cortex.quickshow(flatmap, with_curvature=True,with_sulci=True, with_labels=False)

        fig_name = op.join(outdir, 'modeltypeD_pRFmask.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        ## plot FA binary mask
        flatmap = cortex.Vertex(binary_fa_mask, 
                        self.pysub,
                        vmin = 0, vmax = 1, #.7,
                        cmap='hot')
        cortex.quickshow(flatmap, with_curvature=True,with_sulci=True, with_labels=False)

        fig_name = op.join(outdir, 'modeltypeD_FAmask.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        ## plot beta standard deviation, to see how much they vary
        _, std_surf = self.get_singletrial_avg_estimates(estimate_arr = results_glmsingle['typed']['betasmd'], 
                                                        single_trl_DM = single_trl_DM, return_std = True)

        flatmap = cortex.Vertex(np.mean(std_surf, axis = 0), 
                        self.pysub,
                        vmin = 0, vmax = 2, #.7,
                        cmap='gnuplot')

        fig_name = op.join(outdir, 'modeltypeD_std_betas.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

