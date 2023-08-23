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
        use_atlas: str
            If we want to use atlas ROIs (ex: glasser, wang) or not [default].
            
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
        
        # define bar width in pixel
        self.bar_width_pix = self.MRIObj.screen_res * self.MRIObj.bar_width['FA']

        # define number of bars per direction
        num_bars = np.array(self.MRIObj.FA_num_bar_position) 

        # all possible positions in pixels [x,y] for midpoint of
        # vertical bar passes, 
        self.bar_y_coords_pix = np.sort(np.concatenate((-np.arange(self.bar_width_pix[1]/2,self.MRIObj.screen_res[1]/2,self.bar_width_pix[1])[0:int(num_bars[1]/2)],
                                        np.arange(self.bar_width_pix[1]/2,self.MRIObj.screen_res[1]/2,self.bar_width_pix[1])[0:int(num_bars[1]/2)])))

        self.ver_bar_pos_pix = np.array([np.array([0,y]) for _,y in enumerate(self.bar_y_coords_pix)])

        # horizontal bar passes 
        self.bar_x_coords_pix = np.sort(np.concatenate((-np.arange(self.bar_width_pix[0]/2,self.MRIObj.screen_res[0]/2,self.bar_width_pix[0])[0:int(num_bars[0]/2)],
                                        np.arange(self.bar_width_pix[0]/2,self.MRIObj.screen_res[0]/2,self.bar_width_pix[0])[0:int(num_bars[0]/2)])))

        self.hor_bar_pos_pix = np.array([np.array([x,0]) for _,x in enumerate(self.bar_x_coords_pix)])


    def get_correlation_mask(self, participant, task = 'pRF', ses = 'mean', file_ext = '_cropped_dc_psc.npy',
                                n_jobs = 8, seed_num = 2023, perc_thresh_nm = 95, smooth = True,
                                kernel=3, nr_iter=3, normalize = False, filename = None):

        """
        Split half correlate all runs in a task
        and create binary mask of task relevant vertices
        to be used in noise pool

        Parameters
        ----------
        participant: str
            participant ID
        task: str
            name of task [pRF vs FA]
        ses: int or str
            which session to get files from
        prf_estimates : dict
            dict with participant prf estimates
        file_ext: str
            bold file extension
        n_jobs: int
            number of jobs for parallel
        seed_num: int
            seed to initialize random (for reproducibility)
        smooth: bool
            if we want to smooth noise mask
        perc_thresh_nm: int
            noise mask percentile threshold
        filename: str
            if given, will save correlation file with absolute filename
        kernel : int
            size of "kernel" to use for smoothing (factor)
        nr_iter: int
            number of iterations to repeat smoothing, larger values smooths more
        normalize: bool
            if we want to max normalize smoothed data (default = False)
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

        # save correlation file
        if filename:
            np.save(filename, final_corr_arr)

        return binary_mask

    def get_single_trial_combinations(self):

        """
        Get all possible trial combinations (useful to keep track of single trial DM later)

        Returns DataFrame where each row is a unique trial type.
        Columns indicate attended and unattended bar midpoint position (x,y) and bar pass direction (vertical vs horizontal)
        """

        ## make all possible combinations
        pos_dict = {'horizontal': self.hor_bar_pos_pix, 'vertical': self.ver_bar_pos_pix}
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

    def make_singletrial_dm(self, run_num_arr = [], ses_num_arr = [], pp_bar_pos_df = {}):

        """
        Make single trial design matrix for one or more runs 

        Parameters
        ----------
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
        Make average HRF to give as input to glm single model
        (requires previously obtained HRF params from pRF fitting)

        Parameters
        ----------
        pp_prf_estimates : dict
            dict with participant prf estimates
        prf_modelobj: object
            pRF model object from prfpy, used to create HRF
        rsq_threshold: float or None
            will only fit vertices where prf fit above certain rsq threshold 
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
        subselect trial combinations according to (un)attended bar position or bar orientations, 
        and obtain dataframe with possible trials

        Parameters
        ----------
        att_bar_xy : list
            list with [x,y] bar coordinates for attended bar 
        unatt_bar_xy : list
            list with [x,y] bar coordinates for unattended bar
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
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

    def get_singletrial_estimates(self, estimate_arr = [], single_trl_DM = [], return_std = True,
                                        att_bar_xy = [], unatt_bar_xy = [], average_betas = True, att_color_ses_run = None,
                                        participant = None, file_ext = '_cropped.npy'):

        """
        Get beta values for each single trial type (our condition)
        and return an array of [conditions, vertex, runs] --> or without runs if we average

        Parameters
        ----------
        estimate_arr: array
            estimate array from glmsingle [vertex, singletrials] (note: single trial number is multiplied by nr of runs)
        single_trl_DM: arr
            single trial design matrix for one or more runs 
        att_bar_xy : list
            list with [x,y] bar coordinates for attended bar 
        unatt_bar_xy : list
            list with [x,y] bar coordinates for unattended bar
        return_std: bool
            if we want to obtain standard deviation across runs
        average_betas: bool
            if we want to average betas across runs
        """
        
        # if we provided dict with attended color run/ses keys, then average by color
        if att_color_ses_run:
            # get run and ses number in order used in DM 
            run_num_arr,ses_num_arr = self.get_run_ses_pp(participant, task = 'FA', run_type = 'all', ses = 'all', file_ext = file_ext)
            print(run_num_arr)
            print(ses_num_arr)

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
            
            if average_betas and att_color_ses_run is None: # average across all runs
               avg_all.append(np.nanmean(estimate_arr[...,cond_ind], axis = -1))
            else:
                avg_all.append(estimate_arr[...,cond_ind])
            
            if return_std:
                std_all.append(np.std(estimate_arr[...,cond_ind], axis = -1))

        # assumes we want to average across attended condition (this is, only average runs where same color bar was attended)
        if att_color_ses_run and average_betas:
            out_avg = []
            # get indices for runs of same attended color
            for col_name in att_color_ses_run.keys():
                col_indices = np.hstack((np.where(((np.array(run_num_arr) == rn) & (np.array(ses_num_arr) == att_color_ses_run[col_name]['ses'][i])
                            ))[0] for i, rn in enumerate(att_color_ses_run[col_name]['run'])))
                out_avg.append(np.nanmean(np.stack(avg_all)[...,col_indices], axis = -1))
            out_avg = np.swapaxes(np.swapaxes(np.stack(out_avg),0,1),1,2)
        else:
            out_avg = np.stack(avg_all)

        if return_std:
            return out_avg, np.stack(std_all)
        else:
            return out_avg
        
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

    def load_estimates(self, participant, model_type = 'D'):

        """
        Load glm single estimates dict

        Parameters
        ----------
        participant: str
            participant ID
        model_type: str
            model to load estimates from (A-D)
        """

        ## path to files
        fitpath = op.join(self.outputdir, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant))

        return np.load(op.join(fitpath, self.MRIObj.params['mri']['fitting']['FA']['glmsingle_models'][model_type]),
                       allow_pickle=True).item()
    
    def load_single_trl_DM(self, participant):

        """
        Load glm single design matrix

        Parameters
        ----------
        participant: str
            participant ID
        """

        ## path to files
        fitpath = op.join(self.outputdir, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant))

        return np.load(op.join(fitpath, 'single_trl_DM.npy'), allow_pickle=True)
    
    def fit_data(self, participant, pp_prf_estimates, prf_modelobj,  file_ext = '_cropped.npy', 
                        smooth_nm = True, perc_thresh_nm = 95, n_jobs = 8,
                        seed_num = 2023, kernel = 3, nr_iter = 3, normalize = False,
                        pp_bar_pos_df = {}, fit_hrf = False,
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
        file_ext: str
            file extent for FA trials to be fitted
        smooth_nm: bool
            if we want to smooth noise mask
        perc_thresh_nm: int
            noise mask percentile threshold
        file_extent_nm: dict
            dict with file extension for task files to be used for noise mask
        pp_bar_pos_df: dataframe
            participant bar position data frame, for all runs of task (from preproc_beh class)
        n_jobs: int
            number of jobs for parallel
        seed_num: int
            seed to initialize random (for reproducibility)
        kernel : int
            size of "kernel" to use for smoothing (factor)
        nr_iter: int
            number of iterations to repeat smoothing, larger values smooths more
        normalize: bool
            if we want to max normalize smoothed data (default = False)
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
                                                    filename = op.join(outdir, 'spcorrelation_task-pRF.npy'))
        # load correlation array - glmsingle deletes files in folder, should fix later
        corr_pRF = np.load(op.join(outdir, 'spcorrelation_task-pRF.npy'), allow_pickle=True)

        ## now do the same correlation mask for the FA runs
        binary_fa_mask = self.get_correlation_mask(participant, task = 'FA', ses = 'mean', 
                                                    file_ext = file_extent_nm['FA'], n_jobs = n_jobs, 
                                                    seed_num = int(seed_num * 2), perc_thresh_nm = perc_thresh_nm, 
                                                    smooth = smooth_nm, kernel = kernel, nr_iter = nr_iter, 
                                                    normalize = normalize,
                                                    filename = op.join(outdir, 'spcorrelation_task-FA.npy'))
        # load correlation array - glmsingle deletes files in folder, should fix later
        corr_FA = np.load(op.join(outdir, 'spcorrelation_task-FA.npy'), allow_pickle=True)

        ### final mask is multiplication of the two
        final_mask = binary_fa_mask * binary_prf_mask

        # create a directory for saving GLMsingle outputs
        opt = dict()

        # set important fields for completeness (but these would be enabled by default)
        opt['wantglmdenoise'] = 1
        opt['wantfracridge'] = 1

        if fit_hrf:
            opt['wantlibrary'] = 1
        else:
            opt['wantlibrary'] = 0
            opt['hrftoassume'] = hrf_final
        opt['hrfonset'] = 0 # already setting onset in hrf

        opt['brainexclude'] = final_mask.astype(int)
        opt['sessionindicator'] = self.ses_num_arr 
        opt['brainthresh'] = [99, 0] # which allows all voxels to pass the intensity threshold --> we use surface data
        opt['brainR2'] = 100 # not using on-off model for noise pool

        # define polynomials to project out from data (we only want to use intercept and slope)
        opt['maxpolydeg'] = [[0, 1] for _ in range(data.shape[0])]

        # keep relevant outputs in memory and also save them to the disk
        opt['wantfileoutputs'] = [1,1,1,1]
        opt['wantmemoryoutputs'] = [0,0,0,0] #[1,1,1,1]

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

        ## plot average HRF
        fig, axis = plt.subplots(1,figsize=(8,5),dpi=100)
        axis.plot(getcanonicalhrf(self.MRIObj.FA_bars_phase_dur, self.MRIObj.TR, onset = self.hrf_onset), label = 'canonical_hrf')
        axis.plot(hrf_final, label = 'average_hrf')
        axis.set_xlabel('Time (TR)')
        axis.legend()
        plt.savefig(op.join(outdir, 'hrf_avg.png'))

        # save DM for ease of use later
        np.save(op.join(outdir, 'single_trl_DM.npy'), single_trl_DM)

        # also save binary mask, to later check
        np.save(op.join(outdir, 'binary_mask_spcorrelation_task-pRF.npy'), binary_prf_mask)
        np.save(op.join(outdir, 'binary_mask_spcorrelation_task-FA.npy'), binary_fa_mask)

        # save correlations again
        np.save(op.join(outdir, 'spcorrelation_task-pRF.npy'), corr_pRF)
        np.save(op.join(outdir, 'spcorrelation_task-FA.npy'), corr_FA)

    def get_betas_coord_df(self, participant, betas_arr = [], single_trl_DM = [], att_color_ses_run = {}, 
                                            file_ext = '_cropped.npy', ROIs_dict = {}, prf_estimates = {}, orientation_bars = 'parallel_vertical'):

        """
        make dataframe with beta values, with info on prf location, attended color, ROI vertices belong to,
        for trials with a specific bar oriantation (parallel - vert/hor -, or crossed)

        Parameters
        ----------
        participant: str
            participant ID
        betas_arr : list/arr
            array with beta values for 
        single_trl_DM: arr
            glm single design matrix
        att_color_ses_run_dict: dict
            dict with info for each participant, indicating session and run number for same attended color
        file_ext: str
            file extent of FA data
        ROIs_dict: dict/df
            vertex index and ROI labels, to use when parsing data array
        prf_estimates : dict
            dict with participant prf estimates
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        """

        DF_betas_bar_coord = pd.DataFrame({'sj': [], 'ROI': [], 'betas': [], 'prf_rsq_coord': [],
                                            'prf_x_coord': [], 'prf_y_coord': [], 'attend_color': []})

        ## for bars going left to right (vertical orientation)
        if orientation_bars == 'parallel_vertical':
            coord_list = self.bar_x_coords_pix

        elif orientation_bars == 'parallel_horizontal':
            coord_list = self.bar_y_coords_pix

        else:
            raise ValueError('Cross sections not implemented yet')

        for UAtt_bar_coord in coord_list: # let's start with unattended bar vertical leftmost, attended bar left to right (5 positions)

            for Att_bar_coord in coord_list:

                if Att_bar_coord != UAtt_bar_coord: ## bars cannot fully overlap

                    att_bar_xy = [Att_bar_coord, np.nan] if orientation_bars == 'parallel_vertical' else [np.nan, Att_bar_coord]
                    unatt_bar_xy = [UAtt_bar_coord, np.nan] if orientation_bars == 'parallel_vertical' else [np.nan, UAtt_bar_coord]

                    ## get average beta for each trial type
                    avg_betas = self.get_singletrial_estimates(estimate_arr = betas_arr, 
                                                                single_trl_DM = single_trl_DM, return_std = False,
                                                                average_betas = True,
                                                                att_color_ses_run = att_color_ses_run,
                                                                participant = participant, file_ext = file_ext,
                                                                att_bar_xy = att_bar_xy, 
                                                                unatt_bar_xy = unatt_bar_xy)
                    # differentiate per color
                    for c, color_name in enumerate(att_color_ses_run.keys()):
                    
                        # iterate over ROIs
                        for rname in ROIs_dict.keys():

                            DF_betas_bar_coord = pd.concat((DF_betas_bar_coord,
                                                        pd.DataFrame({'sj': np.tile('sub-{sj}'.format(sj = participant), len(ROIs_dict[rname])), 
                                                            'ROI': np.tile(rname, len(ROIs_dict[rname])), 
                                                            'betas': avg_betas[0,:,c][ROIs_dict[rname]], 
                                                            'attend_color': np.tile(color_name, len(ROIs_dict[rname])),
                                                            'Att_bar_coord': np.tile(Att_bar_coord, len(ROIs_dict[rname])),
                                                            'UAtt_bar_coord': np.tile(UAtt_bar_coord, len(ROIs_dict[rname])),
                                                            'prf_rsq_coord': prf_estimates['sub-{sj}'.format(sj = participant)]['r2'][ROIs_dict[rname]],
                                                            'prf_x_coord': prf_estimates['sub-{sj}'.format(sj = participant)]['x'][ROIs_dict[rname]], 
                                                            'prf_y_coord': prf_estimates['sub-{sj}'.format(sj = participant)]['y'][ROIs_dict[rname]]})
                                                            )) 
                            
        return DF_betas_bar_coord

    def get_betas_binned1D_df(self, DF_betas_bar_coord = {}, ROI_list = [], orientation_bars = 'parallel_vertical', 
                                    max_ecc_ext = 5.5, bin_size = .5, bar_color2bin = None):

        """
        Transform model beta values (according to pRF x,y coordinates) into 1D binned average
        for different ROIs

        Parameters
        ----------
        DF_betas_bar_coord: dataframe
            FA beta values dataframe for a participant, with relevant prf estimates (x,y,r2)
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        ROI_list: list/arr
            list with ROI names 
        max_ecc_ext: float
            eccentricity limit (screen)
        bin_size: float
            size of bin (in dva)
        bar_color2bin: str
            attended bar color. if given, will bin betas for that bar color, else will average across colors
        """

        ## bins array (1/3 of bar width, equally spaced across x/y coordinates of screen)
        bins_arr = np.arange(0, max_ecc_ext + bin_size, bin_size)
        bins_arr = np.concatenate((bins_arr[1:]*-1, bins_arr))
        bins_arr.sort()

        # if no ROI specified, then plot all
        if len(ROI_list) == 0:
            ROI_list = DF_betas_bar_coord.ROI.unique()

        ## for bars going left to right (vertical orientation)
        if orientation_bars == 'parallel_vertical':
            coord_list = self.bar_x_coords_pix

        elif orientation_bars == 'parallel_horizontal':
            coord_list = self.bar_y_coords_pix

        else:
            raise ValueError('Cross sections not implemented yet')
        
        DF_betas_bar_coord1D = pd.DataFrame({'sj': [], 'ROI': [], 'betas': [], 'std': [], 'prf_rsq_coord': [], 'prf_x_coord': [], 'prf_y_coord': [],
                                            'attend_color': [], 'Att_bar_coord': [], 'UAtt_bar_coord': []})

        ## iterate over ROIs
        for roi_name in ROI_list:
            
            for UAtt_bar_coord in coord_list: 
                for Att_bar_coord in coord_list:
                    
                    if Att_bar_coord != UAtt_bar_coord: ## bars cannot fully overlap

                        trial_df = DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                            (DF_betas_bar_coord['Att_bar_coord'] == Att_bar_coord) &\
                                            (DF_betas_bar_coord['UAtt_bar_coord'] == UAtt_bar_coord)].dropna()
                        
                        if not trial_df.empty: # if df not empty (which might happend when we averaged across trial types etc)

                            ## if we want to bin estimates for specific bar color
                            if bar_color2bin:
                                trial_df = trial_df[trial_df['attend_color'] == bar_color2bin]
                            else:
                                # average them, if we dont care
                                trial_df = trial_df.groupby(['prf_x_coord', 'prf_y_coord', 'prf_rsq_coord', 'Att_bar_coord', 'UAtt_bar_coord',
                                                                                                'ROI', 'sj'])['betas'].mean().reset_index()

                            for b in range(len(bins_arr)-1):

                                bin_df = trial_df[(trial_df['prf_x_coord'] >= bins_arr[b]) &\
                                                (trial_df['prf_x_coord'] <= bins_arr[b+1])]

                                if not bin_df.empty:
                                    DF_betas_bar_coord1D = pd.concat((DF_betas_bar_coord1D, 
                                                                    pd.DataFrame({'sj': bin_df.sj.values[:1], 
                                                                                'ROI': [roi_name], 
                                                                                'betas': [self.MRIObj.mri_utils.weighted_mean(bin_df.betas.values, 
                                                                                                                            weights = bin_df.prf_rsq_coord.values, 
                                                                                                                            norm = True)], 
                                                                                'std': [self.MRIObj.mri_utils.weighted_mean_std_sem(bin_df.betas.values, 
                                                                                                                                        weights = bin_df.prf_rsq_coord.values, 
                                                                                                                                        norm = True)[0]],
                                                                                'sem': [self.MRIObj.mri_utils.weighted_mean_std_sem(bin_df.betas.values, 
                                                                                                                                        weights = bin_df.prf_rsq_coord.values, 
                                                                                                                                        norm = True)[-1]],
                                                                                'prf_rsq_coord': [np.nanmean(bin_df.prf_rsq_coord.values)],
                                                                                'prf_x_coord': [np.nanmean(bins_arr[b:b+2])], 
                                                                                'prf_y_coord': [np.nanmean(bins_arr[b:b+2])],
                                                                                'Att_bar_coord': [Att_bar_coord],
                                                                                'UAtt_bar_coord':[UAtt_bar_coord]})))
        if bar_color2bin:
            DF_betas_bar_coord1D['attend_color'] = bar_color2bin

        return DF_betas_bar_coord1D
    
    def get_betas_1D_df(self, DF_betas_bar_coord = {}, ROI_list = [], orientation_bars = 'parallel_vertical', bar_color2bin = None):

        """
        Transform model beta values (according to pRF x,y coordinates) into 1D average
        for different ROIs

        Parameters
        ----------
        DF_betas_bar_coord: dataframe
            FA beta values dataframe for a participant, with relevant prf estimates (x,y,r2)
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        ROI_list: list/arr
            list with ROI names 
        bar_color2bin: str
            attended bar color. if given, will bin betas for that bar color, else will average across colors
        """

        # if no ROI specified, then plot all
        if len(ROI_list) == 0:
            ROI_list = DF_betas_bar_coord.ROI.unique()

        ## for bars going left to right (vertical orientation)
        if orientation_bars == 'parallel_vertical':
            coord_list = self.bar_x_coords_pix

        elif orientation_bars == 'parallel_horizontal':
            coord_list = self.bar_y_coords_pix

        else:
            raise ValueError('Cross sections not implemented yet')
        
        DF_betas_bar_coord1D = pd.DataFrame({'sj': [], 'ROI': [], 'betas': [], 'std': [], 'prf_rsq_coord': [], 'prf_x_coord': [], 'prf_y_coord': [],
                                            'attend_color': [], 'Att_bar_coord': [], 'UAtt_bar_coord': []})

        ## iterate over ROIs
        for roi_name in ROI_list:
            
            for UAtt_bar_coord in coord_list: 
                for Att_bar_coord in coord_list:
                    
                    if Att_bar_coord != UAtt_bar_coord: ## bars cannot fully overlap

                        trial_df = DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                            (DF_betas_bar_coord['Att_bar_coord'] == Att_bar_coord) &\
                                            (DF_betas_bar_coord['UAtt_bar_coord'] == UAtt_bar_coord)].dropna()
                        
                        if not trial_df.empty: # if df not empty (which might happend when we averaged across trial types etc)

                            ## if we want to bin estimates for specific bar color
                            if bar_color2bin:
                                trial_df = trial_df[trial_df['attend_color'] == bar_color2bin]
                            else:
                                # average them, if we dont care
                                trial_df = trial_df.groupby(['prf_x_coord', 'prf_y_coord', 'prf_rsq_coord', 'Att_bar_coord', 'UAtt_bar_coord',
                                                                                                'ROI', 'sj'])['betas'].mean().reset_index()

                            # group by x-coordinates, and average
                            out_df = trial_df.groupby(['sj', 'ROI', 'Att_bar_coord', 'UAtt_bar_coord', 'prf_x_coord']).mean().reset_index()

                            DF_betas_bar_coord1D = pd.concat((DF_betas_bar_coord1D, out_df))

        if bar_color2bin:
            DF_betas_bar_coord1D['attend_color'] = bar_color2bin

        return DF_betas_bar_coord1D

    def get_betas_subtract_reverse_df(self, DF_betas_bar_coord = {}, ROI_list = [], orientation_bars = 'parallel_vertical', 
                                    color_name = ['color_red', 'color_green'], average = True):

        """
        Make "attention modulation" df, calculated from GLMsingle beta estimates, relative to pRF coordinates.
        Involves subtracting flipped trials (where target and distactor bars are in reversed postion), to isolate attentional effect

        Parameters
        ----------
        DF_betas_bar_coord: dataframe
            FA beta values dataframe for a participant, with relevant prf estimates (x,y,r2)
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        ROI_list: list/arr
            list with ROI names
        color_name: list
            list with color names
        average: bool
            if we want to average across attended colors or not
        """

        # if no ROI specified, then plot all
        if len(ROI_list) == 0:
            ROI_list = DF_betas_bar_coord.ROI.unique()

        ## for bars going left to right (vertical orientation)
        if orientation_bars == 'parallel_vertical':
            coord_list = self.bar_x_coords_pix
        elif orientation_bars == 'parallel_horizontal':
            coord_list = self.bar_y_coords_pix
        else:
            raise ValueError('Cross sections not implemented yet')
        
        attention_coord_df = pd.DataFrame()

        # per bar color
        for col_name in color_name: 

            ## iterate over ROIs
            for roi_name in ROI_list:

                for Att_bar_coord in coord_list:
                    for UAtt_bar_coord in coord_list: 
                        if Att_bar_coord < UAtt_bar_coord: ## we want to subtract over diagonal, so we want to select the lower half (where target always on the left of distractor)
                            
                            numerator =  DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                                            (DF_betas_bar_coord['Att_bar_coord'] == Att_bar_coord) &\
                                                            (DF_betas_bar_coord['UAtt_bar_coord'] == UAtt_bar_coord) &\
                                                            (DF_betas_bar_coord['attend_color'] == col_name)].dropna()

                            ## swap target and distractor bar location, for subtraction
                            denominator =  DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                                            (DF_betas_bar_coord['Att_bar_coord'] == UAtt_bar_coord) &\
                                                            (DF_betas_bar_coord['UAtt_bar_coord'] == Att_bar_coord) &\
                                                            (DF_betas_bar_coord['attend_color'] == col_name)].dropna()
                            
                            ## actually subtract
                            subtracted_df = numerator.set_index(['sj', 'ROI', 'attend_color',
                                                                'prf_x_coord', 'prf_y_coord',
                                                                'prf_rsq_coord'])['betas'].sub(denominator.set_index(['sj', 'ROI', 'attend_color',
                                                                                                                    'prf_x_coord', 'prf_y_coord',
                                                                                                                    'prf_rsq_coord'])['betas']).reset_index()
                            subtracted_df['Att_bar_coord'] = Att_bar_coord
                            subtracted_df['UAtt_bar_coord'] = UAtt_bar_coord
                            
                            ## append
                            attention_coord_df = pd.concat((attention_coord_df, subtracted_df))

        # if we want to average
        if average:
            attention_coord_df = attention_coord_df.dropna().groupby(['prf_x_coord', 'prf_y_coord', 'prf_rsq_coord', 'Att_bar_coord', 'UAtt_bar_coord',
                                                                                'ROI', 'sj'])['betas'].mean().reset_index()

        return attention_coord_df

    def get_attention_coord_flipped_df(self, DF_betas_bar_coord = {}, ROI_list = [], orientation_bars = 'parallel_vertical', 
                                    colA = ['color_red', 'color_green'], colB = ['color_green', 'color_red'], average = True):

        """
        Make attention modulation df, calculated from GLMsingle beta estimates, relative to pRF coordinates.
        Involves subtracting flipped trials, to isolate attentional effect

        Parameters
        ----------
        DF_betas_bar_coord: dataframe
            FA beta values dataframe for a participant, with relevant prf estimates (x,y,r2)
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        ROI_list: list/arr
            list with ROI names
        colA: list
            list with color names
        colB: list
            list with color names, reversed of A
        average: bool
            if we want to average across attended colors or not
        """

        # if no ROI specified, then plot all
        if len(ROI_list) == 0:
            ROI_list = DF_betas_bar_coord.ROI.unique()

        ## for bars going left to right (vertical orientation)
        if orientation_bars == 'parallel_vertical':
            coord_list = self.bar_x_coords_pix

        elif orientation_bars == 'parallel_horizontal':
            coord_list = self.bar_y_coords_pix

        else:
            raise ValueError('Cross sections not implemented yet')
        
        attention_coord_df = pd.DataFrame()

        for col_ind in range(len(colA)):
            ## iterate over ROIs
            for roi_name in ROI_list:
                
                for UAtt_bar_coord in coord_list: 
                    for Att_bar_coord in coord_list:
                        
                        if Att_bar_coord != UAtt_bar_coord: ## bars cannot fully overlap

                            coords_colA = DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                                (DF_betas_bar_coord['Att_bar_coord'] == Att_bar_coord) &\
                                                (DF_betas_bar_coord['UAtt_bar_coord'] == UAtt_bar_coord) &\
                                                (DF_betas_bar_coord['attend_color'] == colA[col_ind])]
                            coords_colA = coords_colA.dropna()

                            ## swap attended color and bar location,
                            # for subtraction
                            coords_colB = DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                                (DF_betas_bar_coord['Att_bar_coord'] == UAtt_bar_coord) &\
                                                (DF_betas_bar_coord['UAtt_bar_coord'] == Att_bar_coord) &\
                                                (DF_betas_bar_coord['attend_color'] == colB[col_ind])]
                            coords_colB = coords_colB.dropna()

                            ## actually subtract
                            subtracted_df = coords_colA.set_index(['prf_x_coord', 'prf_y_coord', 'prf_rsq_coord',
                                                'ROI', 'sj'])['betas'].sub(coords_colB.set_index(['prf_x_coord', 'prf_y_coord', 'prf_rsq_coord',
                                                                                                        'ROI', 'sj'])['betas']).reset_index()
                            subtracted_df['Att_bar_coord'] = Att_bar_coord
                            subtracted_df['UAtt_bar_coord'] = UAtt_bar_coord
                            subtracted_df['attend_color'] = colA[col_ind]
                            
                            ## append
                            attention_coord_df = pd.concat((attention_coord_df, subtracted_df))
        
        # if we want to average
        if average:
            attention_coord_df = attention_coord_df.dropna().groupby(['prf_x_coord', 'prf_y_coord', 'prf_rsq_coord', 'Att_bar_coord', 'UAtt_bar_coord',
                                                                                'ROI', 'sj'])['betas'].mean().reset_index()

        return attention_coord_df
    
    def get_attention_coord_df(self, DF_betas_bar_coord = {}, ROI_list = [], orientation_bars = 'parallel_vertical', 
                                    color_name = ['color_red', 'color_green'], average = True):

        """
        Make attention modulation df, calculated from GLMsingle beta estimates, relative to pRF coordinates.
        Involves subtracting average bar position from each trial type, to isolate attentional effect

        Parameters
        ----------
        DF_betas_bar_coord: dataframe
            FA beta values dataframe for a participant, with relevant prf estimates (x,y,r2)
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        ROI_list: list/arr
            list with ROI names
        color_name: list
            list with color names
        average: bool
            if we want to average across attended colors or not
        """

        # if no ROI specified, then plot all
        if len(ROI_list) == 0:
            ROI_list = DF_betas_bar_coord.ROI.unique()

        ## for bars going left to right (vertical orientation)
        if orientation_bars == 'parallel_vertical':
            coord_list = self.bar_x_coords_pix
        elif orientation_bars == 'parallel_horizontal':
            coord_list = self.bar_y_coords_pix
        else:
            raise ValueError('Cross sections not implemented yet')
        
        attention_coord_df = pd.DataFrame()

        # per bar color
        for col_name in color_name: 

            ## iterate over ROIs
            for roi_name in ROI_list:
                
                # for each unattended bar position
                for UAtt_bar_coord in coord_list: 
                    
                    # filter df for trials where unattended bar in the same position
                    coords_trials = DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                        (DF_betas_bar_coord['Att_bar_coord'] != UAtt_bar_coord) &\
                                        (DF_betas_bar_coord['UAtt_bar_coord'] == UAtt_bar_coord) &\
                                        (DF_betas_bar_coord['attend_color'] == col_name)].dropna()

                    ## and average them
                    average_bar_loc_df = coords_trials.groupby(['sj', 'ROI', 'attend_color',
                                                                'prf_x_coord', 'prf_y_coord','prf_rsq_coord'])['betas'].mean().reset_index()

                    ## finally subtract average from each trial
                    for Att_bar_coord in coord_list:
                        
                        if Att_bar_coord != UAtt_bar_coord: ## bars cannot fully overlap
                            
                            ## actually subtract
                            trial_df =  DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                                            (DF_betas_bar_coord['Att_bar_coord'] == Att_bar_coord) &\
                                                            (DF_betas_bar_coord['UAtt_bar_coord'] == UAtt_bar_coord) &\
                                                            (DF_betas_bar_coord['attend_color'] == col_name)].dropna()
                            subtracted_df = trial_df.set_index(['sj', 'ROI', 'attend_color',
                                                                'prf_x_coord', 'prf_y_coord','prf_rsq_coord'])['betas'].sub(average_bar_loc_df.set_index(['sj', 'ROI', 'attend_color',
                                                                                                                                                'prf_x_coord', 'prf_y_coord',
                                                                                                                                                'prf_rsq_coord'])['betas']).reset_index()
                            subtracted_df['Att_bar_coord'] = Att_bar_coord
                            subtracted_df['UAtt_bar_coord'] = UAtt_bar_coord
                            
                            ## append
                            attention_coord_df = pd.concat((attention_coord_df, subtracted_df))

        # if we want to average
        if average:
            attention_coord_df = attention_coord_df.dropna().groupby(['prf_x_coord', 'prf_y_coord', 'prf_rsq_coord', 'Att_bar_coord', 'UAtt_bar_coord',
                                                                                'ROI', 'sj'])['betas'].mean().reset_index()

        return attention_coord_df
    
    def get_distractor_coord_df(self, DF_betas_bar_coord = {}, ROI_list = [], orientation_bars = 'parallel_vertical', 
                                    color_name = ['color_red', 'color_green'], average = True):

        """
        Make distractor bar modulation df, calculated from GLMsingle beta estimates, relative to pRF coordinates.
        Involves subtracting average bar position from each trial type, to isolate attentional effect

        Parameters
        ----------
        DF_betas_bar_coord: dataframe
            FA beta values dataframe for a participant, with relevant prf estimates (x,y,r2)
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        ROI_list: list/arr
            list with ROI names
        color_name: list
            list with color names
        average: bool
            if we want to average across attended colors or not
        """

        # if no ROI specified, then plot all
        if len(ROI_list) == 0:
            ROI_list = DF_betas_bar_coord.ROI.unique()

        ## for bars going left to right (vertical orientation)
        if orientation_bars == 'parallel_vertical':
            coord_list = self.bar_x_coords_pix
        elif orientation_bars == 'parallel_horizontal':
            coord_list = self.bar_y_coords_pix
        else:
            raise ValueError('Cross sections not implemented yet')
        
        distractor_coord_df = pd.DataFrame()

        # per bar color
        for col_name in color_name: 

            ## iterate over ROIs
            for roi_name in ROI_list:
                
                # for each attended bar position
                for Att_bar_coord in coord_list: 
                    
                    # filter df for trials where unattended bar in the same position
                    coords_trials = DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                        (DF_betas_bar_coord['Att_bar_coord'] == Att_bar_coord) &\
                                        (DF_betas_bar_coord['UAtt_bar_coord'] != Att_bar_coord) &\
                                        (DF_betas_bar_coord['attend_color'] == col_name)].dropna()

                    ## and average them
                    average_bar_loc_df = coords_trials.groupby(['sj', 'ROI', 'attend_color',
                                                                'prf_x_coord', 'prf_y_coord','prf_rsq_coord'])['betas'].mean().reset_index()

                    ## finally subtract average from each trial
                    for UAtt_bar_coord in coord_list:
                        
                        if UAtt_bar_coord != Att_bar_coord: ## bars cannot fully overlap
                            
                            ## actually subtract
                            trial_df =  DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                                            (DF_betas_bar_coord['Att_bar_coord'] == Att_bar_coord) &\
                                                            (DF_betas_bar_coord['UAtt_bar_coord'] == UAtt_bar_coord) &\
                                                            (DF_betas_bar_coord['attend_color'] == col_name)].dropna()
                            subtracted_df = trial_df.set_index(['sj', 'ROI', 'attend_color',
                                                                'prf_x_coord', 'prf_y_coord','prf_rsq_coord'])['betas'].sub(average_bar_loc_df.set_index(['sj', 'ROI', 'attend_color',
                                                                                                                                                'prf_x_coord', 'prf_y_coord',
                                                                                                                                                'prf_rsq_coord'])['betas']).reset_index()
                            subtracted_df['Att_bar_coord'] = Att_bar_coord
                            subtracted_df['UAtt_bar_coord'] = UAtt_bar_coord
                            
                            ## append
                            distractor_coord_df = pd.concat((distractor_coord_df, subtracted_df))

        # if we want to average
        if average:
            distractor_coord_df = distractor_coord_df.dropna().groupby(['prf_x_coord', 'prf_y_coord', 'prf_rsq_coord', 'Att_bar_coord', 'UAtt_bar_coord',
                                                                                'ROI', 'sj'])['betas'].mean().reset_index()

        return distractor_coord_df
    
    def get_betas_bar_1D_df(self, DF_betas_bar_coord = {}, ROI_list = [], orientation_bars = 'parallel_vertical', bar_color2bin = None, avg_bool = True):

        """
        Average model beta values (according to pRF x,y coordinates) within each bar
        for different ROIs

        Parameters
        ----------
        DF_betas_bar_coord: dataframe
            FA beta values dataframe for a participant, with relevant prf estimates (x,y,r2)
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        ROI_list: list/arr
            list with ROI names 
        bar_color2bin: str
            attended bar color. if given, will bin betas for that bar color, else will average across colors
        """

        # if no ROI specified, then plot all
        if len(ROI_list) == 0:
            ROI_list = DF_betas_bar_coord.ROI.unique()

        ## for bars going left to right (vertical orientation)
        if orientation_bars == 'parallel_vertical':
            coord_list = self.bar_x_coords_pix
        elif orientation_bars == 'parallel_horizontal':
            coord_list = self.bar_y_coords_pix
        else:
            raise ValueError('Cross sections not implemented yet')
        
        DF_betas_bar_avg1D = pd.DataFrame({'sj': [], 'ROI': [], 'bar_type': [], 'betas': [], 'std': [], 'sem': [], 'attend_color': [], 'Att_bar_coord': [], 'UAtt_bar_coord': [],
                                            'dist_bars': []})

        ## iterate over ROIs
        for roi_name in ROI_list:
            
            for UAtt_bar_coord in coord_list: 
                for Att_bar_coord in coord_list:
                    
                    if Att_bar_coord != UAtt_bar_coord: ## bars cannot fully overlap

                        trial_df = DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                            (DF_betas_bar_coord['Att_bar_coord'] == Att_bar_coord) &\
                                            (DF_betas_bar_coord['UAtt_bar_coord'] == UAtt_bar_coord)]
                        trial_df = trial_df.dropna()

                        ## if we want to bin estimates for specific bar color
                        if bar_color2bin:
                            trial_df = trial_df[trial_df['attend_color'] == bar_color2bin]
                        else:
                            # average them, if we dont care
                            trial_df = trial_df.groupby(['prf_x_coord', 'prf_y_coord', 'prf_rsq_coord', 'Att_bar_coord', 'UAtt_bar_coord',
                                                                                            'ROI', 'sj'])['betas'].mean().reset_index()

                        ## filter df for each bar position
                        UAtt_bar_bin_df = trial_df[(trial_df['prf_x_coord'] >= self.convert_pix2dva(UAtt_bar_coord - self.bar_width_pix[0]/2)) &\
                                                    (trial_df['prf_x_coord'] <= self.convert_pix2dva(UAtt_bar_coord + self.bar_width_pix[0]/2))]

                        Att_bar_bin_df = trial_df[(trial_df['prf_x_coord'] >= self.convert_pix2dva(Att_bar_coord - self.bar_width_pix[0]/2)) &\
                                                 (trial_df['prf_x_coord'] <= self.convert_pix2dva(Att_bar_coord + self.bar_width_pix[0]/2))]
                        
                        ## and calculate distance between bars
                        # (of distractor bar, relative to attended bar)
                        dist_bars = (UAtt_bar_coord - Att_bar_coord)/ self.bar_width_pix[0]

                        ## append df
                        if avg_bool:
                            DF_betas_bar_avg1D = pd.concat((DF_betas_bar_avg1D, 
                                                        pd.DataFrame({'sj': Att_bar_bin_df.sj.values[:2], 
                                                                    'ROI': np.tile(roi_name, 2), 
                                                                    'bar_type': ['distractor', 'target'],
                                                                    'betas': [self.MRIObj.mri_utils.weighted_mean(UAtt_bar_bin_df.betas.values, 
                                                                                                                weights = UAtt_bar_bin_df.prf_rsq_coord.values, 
                                                                                                                norm = True),
                                                                            self.MRIObj.mri_utils.weighted_mean(Att_bar_bin_df.betas.values, 
                                                                                                                weights = Att_bar_bin_df.prf_rsq_coord.values, 
                                                                                                                norm = True)], 
                                                                    'std': [self.MRIObj.mri_utils.weighted_mean_std_sem(UAtt_bar_bin_df.betas.values, 
                                                                                                                weights = UAtt_bar_bin_df.prf_rsq_coord.values, 
                                                                                                                norm = True)[0],
                                                                            self.MRIObj.mri_utils.weighted_mean_std_sem(Att_bar_bin_df.betas.values, 
                                                                                                                weights = Att_bar_bin_df.prf_rsq_coord.values, 
                                                                                                                norm = True)[0]],
                                                                    'sem': [self.MRIObj.mri_utils.weighted_mean_std_sem(UAtt_bar_bin_df.betas.values, 
                                                                                                                weights = UAtt_bar_bin_df.prf_rsq_coord.values, 
                                                                                                                norm = True)[-1],
                                                                            self.MRIObj.mri_utils.weighted_mean_std_sem(Att_bar_bin_df.betas.values, 
                                                                                                                weights = Att_bar_bin_df.prf_rsq_coord.values, 
                                                                                                                norm = True)[-1]],
                                                                    'Att_bar_coord': np.tile(Att_bar_coord, 2),
                                                                    'UAtt_bar_coord':np.tile(UAtt_bar_coord, 2),
                                                                    'dist_bars': np.tile(dist_bars, 2)})))
                        else:
                            DF_betas_bar_avg1D = pd.concat((DF_betas_bar_avg1D, 
                                                        pd.DataFrame({'sj': np.tile(Att_bar_bin_df.sj.values[:1], 
                                                                                    int(len(Att_bar_bin_df.betas.values) + len(UAtt_bar_bin_df.betas.values))),
                                                                    'ROI': np.tile(roi_name, 
                                                                                    int(len(Att_bar_bin_df.betas.values) + len(UAtt_bar_bin_df.betas.values))),
                                                                    'bar_type': np.concatenate((np.tile('distractor', len(UAtt_bar_bin_df.betas.values)), 
                                                                                                np.tile('target', len(Att_bar_bin_df.betas.values))), axis=None),                                                                     
                                                                    'betas': np.concatenate((UAtt_bar_bin_df.betas.values, 
                                                                                            Att_bar_bin_df.betas.values), axis=None),
                                                                    'Att_bar_coord': np.tile(Att_bar_coord, 
                                                                                            int(len(Att_bar_bin_df.betas.values) + len(UAtt_bar_bin_df.betas.values))),
                                                                    'UAtt_bar_coord': np.tile(UAtt_bar_coord, 
                                                                                            int(len(Att_bar_bin_df.betas.values) + len(UAtt_bar_bin_df.betas.values))),
                                                                    'dist_bars': np.tile(dist_bars, 
                                                                                        int(len(Att_bar_bin_df.betas.values) + len(UAtt_bar_bin_df.betas.values)))})))
        if bar_color2bin:
            DF_betas_bar_avg1D['attend_color'] = bar_color2bin

        return DF_betas_bar_avg1D