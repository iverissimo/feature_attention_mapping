import numpy as np
import re
import os
import os.path as op
import pandas as pd
import yaml
import glob

import itertools


from PIL import Image, ImageDraw

from FAM.fitting.model import Model

from glmsingle.glmsingle import GLM_single
from glmsingle.glmsingle import getcanonicalhrf

import time
import cortex
import matplotlib.pyplot as plt

import nibabel as nib
import neuropythy


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
        # and in deg
        self.bar_width_deg = self.convert_pix2dva(self.bar_width_pix)

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

        # get screen coordinates in deg
        # in a range of "bar position" center coordinates
        x_coords_deg = np.concatenate(([self.bar_x_coords_pix[0] - self.bar_width_pix[0]], self.bar_x_coords_pix))
        x_coords_deg = np.concatenate((x_coords_deg, [self.bar_x_coords_pix[-1] + self.bar_width_pix[0]]))
        self.x_coords_deg = self.convert_pix2dva(x_coords_deg)

        y_coords_deg = np.concatenate(([self.bar_y_coords_pix[0] - self.bar_width_pix[1]], self.bar_y_coords_pix))
        y_coords_deg = np.concatenate((y_coords_deg, [self.bar_y_coords_pix[-1] + self.bar_width_pix[1]]))
        self.y_coords_deg = self.convert_pix2dva(y_coords_deg)
          
    def get_visual_DM_dict(self, pp_bar_pos_df = None, pp_prf_bar_coords_dict = None):
    
        """
        return dict for each participant run,
        with visual DM for each type of regressor (attended bar, unattended bar, overlap, both bars, etc)
        
        ex:
        out_dict['ses-1_run-1'] = {'att_bar': [trial, x,y], 'unatt_bar': [trial,x,y], ...}

        Parameters
        ----------
        pp_bar_pos_df: df/dict
            from behavioral object, with bar position info of participant runs
        pp_prf_bar_coords_dict: df/dict
            dict with prf bar positions for participant, vertical/horizontal runs
            when given will be used to mask out not visible bar locations
        """ 
        
        # set empty dicts
        out_dict = {}
        
        for ses_key in pp_bar_pos_df.keys():
            for run_key in pp_bar_pos_df[ses_key].keys():
                
                # get run identifier
                ses_run_id = '{sn}_{rn}'.format(sn = ses_key, rn = run_key)
                print('loading bar positions for {srid}'. format(srid = ses_run_id))
                
                # initialize dict
                out_dict[ses_run_id] = {}
                
                ## get bar position df for run
                run_bar_pos_df = pp_bar_pos_df[ses_key][run_key]

                ## get run bar midpoint and direction values
                # for each bar type (arrays will have len == total number of trial types)
                AttBar_bar_midpoint, AttBar_bar_pass_direction = run_bar_pos_df.loc[(run_bar_pos_df['attend_condition'] == 1), 
                                                                                    ['bar_midpoint_at_TR', 'bar_pass_direction_at_TR']].to_numpy()[0]
                UnattBar_bar_midpoint, UnattBar_bar_pass_direction = run_bar_pos_df.loc[(run_bar_pos_df['attend_condition'] == 0), 
                                                                                    ['bar_midpoint_at_TR', 'bar_pass_direction_at_TR']].to_numpy()[0]

                # if possible, mask out not visible bar positions
                if pp_prf_bar_coords_dict is not None:
                    # get trial indices to mask out
                    t_mask = self.MRIObj.beh_utils.get_trial_ind_mask(AttBar_bar_midpoint = AttBar_bar_midpoint, 
                                                                    AttBar_bar_pass_direction = AttBar_bar_pass_direction,
                                                                    UnattBar_bar_midpoint = UnattBar_bar_midpoint, 
                                                                    UnattBar_bar_pass_direction = UnattBar_bar_pass_direction,
                                                                    prf_bar_coords_dict = pp_prf_bar_coords_dict)
                else:
                    t_mask = None
                    
                ## GET DM FOR ATTENDED BAR
                out_dict[ses_run_id]['att_bar'] = self.get_bar_visual_dm(midpoint_bar = AttBar_bar_midpoint, 
                                                                        direction_bar = AttBar_bar_pass_direction, 
                                                                        res_scaling = .1,
                                                                        trial_mask = t_mask)
                ## GET DM FOR UNATTENDED BAR
                out_dict[ses_run_id]['unatt_bar'] = self.get_bar_visual_dm(midpoint_bar = UnattBar_bar_midpoint, 
                                                                        direction_bar = UnattBar_bar_pass_direction, 
                                                                        res_scaling = .1,
                                                                        trial_mask = t_mask)

                ## GET DM FOR OVERLAP OF BARS
                out_dict[ses_run_id]['overlap'] = self.MRIObj.mri_utils.get_bar_overlap_dm(np.stack((out_dict[ses_run_id]['att_bar'],
                                                                                                    out_dict[ses_run_id]['unatt_bar'])))

                ## GET DM FOR BOTH BARS COMBINED (FULL STIM THAT WAS ON SCREEN)
                stimulus_dm = np.sum(np.stack((out_dict[ses_run_id]['att_bar'],
                                                out_dict[ses_run_id]['unatt_bar'])), axis = 0)
                stimulus_dm[stimulus_dm >=1] = 1
                
                out_dict[ses_run_id]['full_stim'] = stimulus_dm

        # get keys for each condition
        condition_dm_keys = list(out_dict[ses_run_id].keys())
                
        return out_dict, condition_dm_keys
        
    def get_bar_visual_dm(self, midpoint_bar = None, direction_bar = None, res_scaling = .1, trial_mask = None):
        
        """Make visual design matrix, for a specific bar
        returns array of (trials, x, y)
        """
        
        # save screen display for each trial
        visual_dm_array = np.zeros((len(direction_bar), 
                            round(self.MRIObj.screen_res[0] * res_scaling), 
                            round(self.MRIObj.screen_res[1] * res_scaling)))
        
        for trl, bar_center in enumerate(midpoint_bar): # loop over trials

            img = Image.new('RGB', tuple(self.MRIObj.screen_res)) # background image

            if direction_bar[trl] == 'vertical':
                coordenates_bars = {'upLx': 0, 
                                    'upLy': self.MRIObj.screen_res[1]/2-bar_center[-1]+0.5*self.MRIObj.bar_width['FA']*self.MRIObj.screen_res[1],
                                    'lowRx': self.MRIObj.screen_res[0], 
                                    'lowRy': self.MRIObj.screen_res[1]/2-bar_center[-1]-0.5*self.MRIObj.bar_width['FA']*self.MRIObj.screen_res[1]}


            elif direction_bar[trl] == 'horizontal':

                coordenates_bars = {'upLx': self.MRIObj.screen_res[0]/2+bar_center[0]-0.5*self.MRIObj.bar_width['FA']*self.MRIObj.screen_res[0], 
                                    'upLy': self.MRIObj.screen_res[1],
                                    'lowRx': self.MRIObj.screen_res[0]/2+bar_center[0]+0.5*self.MRIObj.bar_width['FA']*self.MRIObj.screen_res[0], 
                                    'lowRy': 0}

            # set draw method for image
            draw = ImageDraw.Draw(img)
            # add bar, coordinates (upLx, upLy, lowRx, lowRy)
            draw.rectangle(tuple([coordenates_bars['upLx'],coordenates_bars['upLy'],
                                coordenates_bars['lowRx'],coordenates_bars['lowRy']]), 
                        fill = (255,255,255),
                        outline = (255,255,255))

            ## save in array - takes into account stim dur in seconds
            visual_dm_array[int(trl), ...] = np.array(img)[::round(1/res_scaling),::round(1/res_scaling),0][np.newaxis,...]
        
        ## and swap positions to get (time, x, y)
        visual_dm_array = np.rollaxis(visual_dm_array, 2, 1)
        
        # if we provided a trial mask, then set those trials to 0
        if trial_mask is not None:
            visual_dm_array[trial_mask] = 0

        return self.MRIObj.mri_utils.normalize(visual_dm_array)
        
    def convert_betas_volume(self, participant, model_type = 'D', file_ext = '_cropped.nii.gz', trial_num = 132, save_r2 = True):
        
        """Convert beta estimates from GLMsingle into nii files
        to use later in volume analysis
        
        return list with filenames
        """
        
        ## load FA bold files

        ## get list of files to load
        bold_filelist = self.MRIObj.mri_utils.get_bold_file_list(participant, task = 'FA', ses = 'all', 
                                                                file_ext = file_ext,
                                                                postfmriprep_pth = self.MRIObj.postfmriprep_pth, 
                                                                acq_name = self.MRIObj.acq, 
                                                                hemisphere = 'BH')
        
        # load GLMsingle estimates dict
        GLMsing_estimates_dict = self.load_estimates(participant, model_type = model_type)
        
        # get GLMsingle fit path, where we will store niftis
        fitpath = op.join(self.outputdir, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant))
        
        all_files = []
        
        for ind, file in enumerate(bold_filelist):
            
            # nifti filename
            betas_filename = op.join(fitpath,
                                     op.split(file)[-1].replace('.nii.gz', 
                                                                '_model-{mod_name}_betas.nii.gz'.format(mod_name = model_type))
                                     )
            
            if not op.isfile(betas_filename):
                
                # select betas relative to run
                betas_run = GLMsing_estimates_dict['betasmd'][...,int(ind*trial_num):int((ind+1)*trial_num)]
                
                # load func image to get affine
                func_img = neuropythy.io.load(file)
                
                # create betas image
                betas_img = nib.Nifti1Image(betas_run, func_img.affine)
                
                print('saving %s'%betas_filename)
                nib.save(betas_img, betas_filename)

                # also save rsq
                if save_r2:
                    r2_run = GLMsing_estimates_dict['R2run'][..., ind]

                    # create r2 image
                    r2_img = nib.Nifti1Image(r2_run, func_img.affine)
                    
                    print('saving %s'%betas_filename.replace('betas', 'r2'))
                    nib.save(r2_img, betas_filename.replace('betas', 'r2'))

            # append to use later
            all_files.append(betas_filename)
        
        return all_files   

    def convert_coord_pRF2screen_grid(self, prf_x_coord = np.array([]), prf_y_coord = np.array([]), grid_num = None):

        """
        convert prf x and y arrays, to "bar position" center coordinates
        --> will return array of labels for bins of screen coordinates (in deg)
        useful when plotting/analysis grid heatmap
        """

        ## label prf coordinates to
        # screen binned coords
        x_deg_arr = np.zeros(prf_x_coord.shape)
        y_deg_arr = np.zeros(prf_y_coord.shape)

        if grid_num is None: # if we didnt provide a grid number, will split into 8x8 grid (of bar positions)

            for (x_val, y_val) in zip(self.x_coords_deg, self.y_coords_deg):
                x_deg_arr[(prf_x_coord <= (x_val + self.bar_width_deg[0])) & (prf_x_coord > (x_val - self.bar_width_deg[0]))] = x_val
                y_deg_arr[(prf_y_coord <= (y_val + self.bar_width_deg[1])) & (prf_y_coord > (y_val - self.bar_width_deg[1]))] = y_val

        else: # will split into given number of point, making square grid between screen limits
            x_coords_deg = self.convert_pix2dva(np.linspace(self.bar_x_coords_pix[0] - (3/2) * self.bar_width_pix[0],
                                                            self.bar_x_coords_pix[-1] + (3/2) * self.bar_width_pix[0],
                                                            grid_num))
            y_coords_deg = self.convert_pix2dva(np.linspace(self.bar_y_coords_pix[0] - (3/2) * self.bar_width_pix[1],
                                                            self.bar_y_coords_pix[-1] + (3/2) * self.bar_width_pix[1],
                                                            grid_num))

            for ind, (x_val, y_val) in enumerate(zip(x_coords_deg[:-1], y_coords_deg[:-1])):
                x_deg_arr[(prf_x_coord <= x_coords_deg[ind+1]) & (prf_x_coord > x_val)] = np.mean([x_val, x_coords_deg[ind+1]])
                y_deg_arr[(prf_y_coord <= y_coords_deg[ind+1]) & (prf_y_coord > y_val)] = np.mean([y_val, y_coords_deg[ind+1]])

        return x_deg_arr, y_deg_arr

    def get_correlation_mask(self, participant, task = 'pRF', ses = 'mean', file_ext = '_cropped_dc_psc.npy',
                                n_jobs = 8, seed_num = 2023, perc_thresh_nm = 95, smooth = True, hemisphere = 'BH',
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

        if self.MRIObj.sj_space in ['fsnative']:
            pysub = 'sub-{pp}_{ps}'.format(pp = participant, ps = self.MRIObj.sj_space)
        else:
            pysub = self.pysub

        # get bold filenames
        task_bold_files = self.MRIObj.mri_utils.get_bold_file_list(participant, task = task, 
                                                                ses = ses, file_ext = file_ext,
                                                                postfmriprep_pth = self.MRIObj.postfmriprep_pth, 
                                                                acq_name = self.MRIObj.acq, hemisphere = hemisphere)
        
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
            final_corr_arr = self.MRIObj.mri_utils.smooth_surface(task_avg_sh_corr, pysub = pysub, hemisphere = hemisphere,
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
        trial_combinations_df = pd.DataFrame.from_dict(trial_combinations_dict).apply(pd.Series.explode).reset_index().drop(columns=['index'])

        return trial_combinations_df

    def make_singletrial_dm(self, run_num_arr = [], ses_num_arr = [], pp_bar_pos_df = None, trial_combinations_df = None):

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
        if trial_combinations_df is None:
            trial_combinations_df = self.get_single_trial_combinations()

        ## make single trial DM
        # with shape [runs, TRs, conditions]
        single_trl_DM = np.zeros((len(run_num_arr), len(self.condition_per_TR), len(trial_combinations_df)))

        ## per unique condition
        for cond_ind in range(len(trial_combinations_df)):

            # filter bar position df for condition
            cond_df = pp_bar_pos_df[(pp_bar_pos_df['att_x_pos'] == trial_combinations_df.iloc[cond_ind].AttBar_bar_midpoint[0]) &\
                                    (pp_bar_pos_df['att_y_pos'] == trial_combinations_df.iloc[cond_ind].AttBar_bar_midpoint[1]) &\
                                    (pp_bar_pos_df['unatt_x_pos'] == trial_combinations_df.iloc[cond_ind].UnattBar_bar_midpoint[0]) &\
                                    (pp_bar_pos_df['unatt_y_pos'] == trial_combinations_df.iloc[cond_ind].UnattBar_bar_midpoint[1])
                                    ]
            
            ## get trial indices when bar on screen 
            # (making sure that run and ses order are correct)
            if len(cond_df) == len(run_num_arr):
                i_trials = [cond_df[(cond_df['ses'] == 'ses-{sn}'.format(sn = ses_num_arr[i])) &\
                                (cond_df['run'] == 'run-{rn}'.format(rn = run_num_arr[i]))].trial_ind.values[0] for i in range(len(run_num_arr))]

                ## fill DM 
                # fill DM with which condition had its onset at what TR
                for i, t in enumerate(i_trials):
                    single_trl_DM[i, np.where((self.condition_per_TR == 'task'))[0][t], cond_ind] = 1

        return single_trl_DM, trial_combinations_df

    

    def subselect_trial_combinations(self, att_bar_xy = [], unatt_bar_xy = [], orientation_bars = None, single_trial_reference_df = None, 
                                            participant = None, hemisphere = None):

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

        # get reference df
        if single_trial_reference_df is None:
            ref_df = self.get_single_trial_reference_df(participant = participant, hemisphere = hemisphere)
        else:
            ref_df = single_trial_reference_df

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
                                        participant = None, file_ext = '_cropped.npy', hemisphere = None):

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

        if self.MRIObj.sj_space in ['fsnative'] and hemisphere is None:
            hemisphere = 'hemi-L'
        
        # if we provided dict with attended color run/ses keys, then average by color
        if att_color_ses_run:
            # get run and ses number in order used in DM 
            run_num_arr,ses_num_arr = self.get_run_ses_pp(participant, task = 'FA', run_type = 'all', ses = 'all', file_ext = file_ext, hemisphere = hemisphere)
            print(run_num_arr)
            print(ses_num_arr)

        # get reference df
        ref_df = self.subselect_trial_combinations(att_bar_xy = att_bar_xy, unatt_bar_xy = unatt_bar_xy, participant = participant, hemisphere = hemisphere)
            
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
                std_all.append(np.nanstd(estimate_arr[...,cond_ind], axis = -1))

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
            if att_color_ses_run is None:
                out_avg = out_avg[..., np.newaxis]

        if return_std:
            return out_avg, np.stack(std_all)
        else:
            return out_avg
        
    def get_single_trial_reference_df(self, participant = None, hemisphere = None):

        """
        make reference dataframe of bar positions
        to facilitate getting single-trials estimates from DM
        """
        
        ## get all possible trial combinations

        ## path to files
        fitpath = op.join(self.outputdir, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant))

        if self.MRIObj.sj_space in ['fsnative']:
            fitpath = op.join(fitpath, 'hemi-L')

            if isinstance(hemisphere, str) and hemisphere in ['right', 'RH', 'hemi-R']: # if we want to load right hemisphere specifically (example, to compare DM and check if same)
                fitpath = fitpath.replace('hemi-L', 'hemi-R')

        filename = op.join(fitpath, 'trial_combinations_df.csv')

        # if file was stored, load
        if op.exists(filename):
            trial_combinations_df = pd.read_csv(filename, sep='\t') 
        else:
            trial_combinations_df = self.get_single_trial_combinations()

        single_trial_reference_df = pd.DataFrame({'ind': [], 'AttBar_coord_x': [], 'AttBar_coord_y': [], 
                                                'UnattBar_coord_x': [], 'UnattBar_coord_y': [], 'orientation_bars': []})

        for ind, row in trial_combinations_df.iterrows():
            
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
        return single_trial_reference_df

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

        if self.MRIObj.sj_space in ['fsnative']:
            # load hemis
            LH_arr = np.load(op.join(fitpath, 'hemi-L', self.MRIObj.params['mri']['fitting']['FA']['glmsingle_models'][model_type]),
                                            allow_pickle=True).item()
            RH_arr = np.load(op.join(fitpath, 'hemi-R', self.MRIObj.params['mri']['fitting']['FA']['glmsingle_models'][model_type]),
                                            allow_pickle=True).item()
            out_arr = {}
            for key in LH_arr.keys():
                if isinstance(LH_arr[key], int):
                    out_arr[key] = LH_arr[key]
                else:
                    out_arr[key] = np.concatenate((LH_arr[key], RH_arr[key]))
        else:
            out_arr = np.load(op.join(fitpath, self.MRIObj.params['mri']['fitting']['FA']['glmsingle_models'][model_type]),
                       allow_pickle=True).item()
        return out_arr
    
    def load_single_trl_DM(self, participant, hemisphere = None):

        """
        Load glm single design matrix

        Parameters
        ----------
        participant: str
            participant ID
        """

        ## path to files
        fitpath = op.join(self.outputdir, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant))

        if self.MRIObj.sj_space in ['fsnative']:
            fitpath = op.join(fitpath, 'hemi-L')

            if isinstance(hemisphere, str) and hemisphere in ['right', 'RH', 'hemi-R']: # if we want to load right hemisphere specifically (example, to compare DM and check if same)
                fitpath = fitpath.replace('hemi-L', 'hemi-R')

        return np.load(op.join(fitpath, 'single_trl_DM.npy'), allow_pickle=True)
    
    def fit_data(self, participant, file_ext = '_cropped.npy', hrf_arr = None,
                        smooth_nm = True, perc_thresh_nm = 95, n_jobs = 8,
                        seed_num = 2023, kernel = 3, nr_iter = 3, normalize = False,
                        pp_bar_pos_df = None, fit_hrf = False, hemisphere = 'BH', use_corr_mask = True,
                        file_extent_nm = {'pRF': '_cropped_dc_psc.npy', 'FA': '_cropped_LinDetrend_psc.npy'}):

        """
        fit GLM single on participant data

        Parameters
        ----------
        participant: str
            participant ID
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
        if hemisphere != 'BH':
            outdir = op.join(outdir, 'hemi-L') if hemisphere in ['LH', 'hemi-L', 'left'] else op.join(outdir, 'hemi-R')
        os.makedirs(outdir, exist_ok = True)
        print('saving files in %s'%outdir)
        
        ## make figure dir
        figuredir = op.join(self.outputdir, self.MRIObj.sj_space, 'figures', 'sub-{sj}'.format(sj = participant))
        if hemisphere != 'BH':
            outdir = op.join(figuredir, 'hemi-L') if hemisphere in ['LH', 'hemi-L', 'left'] else op.join(figuredir, 'hemi-R')
        os.makedirs(figuredir, exist_ok = True)
        print('GLMsingle fit figures in %s'%figuredir)
        
        # if fitting niftis, need to make some changes
        if self.MRIObj.sj_space == 'T1w':
            use_corr_mask = False
            file_ext = file_ext.replace('.npy', '.nii.gz')

        ## get list of files to load
        bold_filelist = self.MRIObj.mri_utils.get_bold_file_list(participant, task = 'FA', ses = 'all', file_ext = file_ext,
                                                                postfmriprep_pth = self.MRIObj.postfmriprep_pth, 
                                                                acq_name = self.MRIObj.acq, hemisphere = hemisphere)

        ## Load data array and file list names
        data, train_file_list = self.get_data4fitting(bold_filelist, task = 'FA', run_type = 'all', 
                                                chunk_num = None, vertex = None, ses = 'all',
                                                baseline_interval = 'empty', correct_baseline = None, return_filenames = True)
    
        ## Make single trial DM for all runs
        print('Run ID : {ri}  Ses ID: {si}'.format(ri = self.run_num_arr, si = self.ses_num_arr))
        single_trl_DM, trial_combinations_df = self.make_singletrial_dm(run_num_arr = self.run_num_arr, 
                                                                        ses_num_arr = self.ses_num_arr,
                                                                        pp_bar_pos_df = pp_bar_pos_df)
                                
        print('Fitting {n} files: {f}'.format(n = len(train_file_list), f = str(train_file_list)))
        
        if use_corr_mask:
            ### make mask array of pRF high fitting voxels,
            # to give as input to glmsingle (excluding them from noise pool)
            pRFcorr_filename = op.join(outdir, 'spcorrelation_task-pRF.npy')

            binary_prf_mask = self.get_correlation_mask(participant, task = 'pRF', ses = 'mean', 
                                                        file_ext = file_extent_nm['pRF'], n_jobs = n_jobs, 
                                                        seed_num = seed_num, perc_thresh_nm = perc_thresh_nm, 
                                                        smooth = smooth_nm, kernel = kernel, nr_iter = nr_iter, 
                                                        normalize = normalize, hemisphere = hemisphere,
                                                        filename = pRFcorr_filename)
            # load correlation array - glmsingle deletes files in folder, should fix later
            corr_pRF = np.load(pRFcorr_filename, allow_pickle=True)

            ## now do the same correlation mask for the FA runs
            FAcorr_filename = pRFcorr_filename.replace('-pRF', '-FA')

            binary_fa_mask = self.get_correlation_mask(participant, task = 'FA', ses = 'mean', 
                                                        file_ext = file_extent_nm['FA'], n_jobs = n_jobs, 
                                                        seed_num = int(seed_num * 2), perc_thresh_nm = perc_thresh_nm, 
                                                        smooth = smooth_nm, kernel = kernel, nr_iter = nr_iter, 
                                                        normalize = normalize, hemisphere = hemisphere,
                                                        filename = FAcorr_filename)
            # load correlation array - glmsingle deletes files in folder, should fix later
            corr_FA = np.load(FAcorr_filename, allow_pickle=True)

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
            opt['hrftoassume'] = hrf_arr
        opt['hrfonset'] = 0 # already setting onset in hrf

        if use_corr_mask:
            opt['brainexclude'] = final_mask.astype(int)
            opt['brainthresh'] = [99, 0] # which allows all voxels to pass the intensity threshold --> we use surface data
            opt['brainR2'] = 100 # not using on-off model for noise pool
        
        # session indicator
        opt['sessionindicator'] = self.ses_num_arr 
        
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
                                            outputdir = outdir,
                                            figuredir = figuredir)

        elapsed_time = time.time() - start_time

        print(
            '\telapsed time: ',
            f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
        )

        ## plot average HRF
        fig, axis = plt.subplots(1,figsize=(8,5),dpi=100)
        axis.plot(getcanonicalhrf(self.MRIObj.FA_bars_phase_dur, self.MRIObj.TR, onset = self.hrf_onset), label = 'canonical_hrf')
        axis.plot(hrf_arr, label = 'average_hrf')
        axis.set_xlabel('Time (TR)')
        axis.legend()
        plt.savefig(op.join(outdir, 'hrf_avg.png'))

        # save DM for ease of use later
        np.save(op.join(outdir, 'single_trl_DM.npy'), single_trl_DM)
        # as well as single trial combinations
        trial_combinations_df.to_csv(op.join(outdir, 'trial_combinations_df.csv'), sep='\t', index=False)#, quoting=csv.QUOTE_ALL)
        
        # also save binary mask, to later check
        if use_corr_mask:
            np.save(pRFcorr_filename.replace('spcorrelation', 'binary_mask_spcorrelation'), binary_prf_mask)
            np.save(FAcorr_filename.replace('spcorrelation', 'binary_mask_spcorrelation'), binary_fa_mask)

            # save correlations again
            np.save(pRFcorr_filename, corr_pRF)
            np.save(FAcorr_filename, corr_FA)
            
        # if fitting niftis, convert betas to nii for inspection
        # only of final model
        if self.MRIObj.sj_space == 'T1w':
            betas_filelist = self.convert_betas_volume(participant, model_type = 'D', 
                                             file_ext = file_ext.replace('.npy', '.nii.gz'), 
                                             trial_num = 132)
            
            print('saved %i beta volume files'%len(betas_filelist))

    def load_betas_coord(self, participant_list = [], ROIs_dict = {}, model_type = 'D', prf_bar_coords_dict = {},
                                att_color_ses_run_dict = {}, betas_per_color = False, file_ext = '_cropped.npy', 
                                orientation_bars = 'parallel_vertical', demean = False, rotate_bars = True, prf_estimates = {}):
        
        """
        Load beta estimates dataframe for all participants in participant list 
        
        Parameters
        ----------
        participant_list: list
            list with participant ID
        ROIs_dict: dict
            dictionary with participant ROI vertices
        model_type: str
            name of GLM single model to load betas from
        prf_bar_coords_dict: dict
            if given, will use pRF bar coordinates as reference to mask FA trials where bar not visible
        att_color_ses_run_dict: dict
            dict with info for each participant, indicating session and run number for same attended color
        betas_per_color: bool
            if we want to load betas separately per attended color
        demean: bool
            if we want to demean betas in 2D space
        rotate_bars: bool
            if we want to rotate bars (when orientation calls for that)
        prf_estimates: dict
            dict with prf estimates for all participants in participant list
        file_ext: str
            name of data file extent used for fitting
        """
        
        DF_betas_bar_coord_GROUP = []
        
        # iterate over participant list
        for pp in participant_list:

            # select ROI dict for participant
            pp_ROI_dict = {} if 'sub-{sj}'.format(sj = pp) not in list(ROIs_dict.keys()) else ROIs_dict['sub-{sj}'.format(sj = pp)]
            
            # load GLMsingle estimates dict
            GLMsing_estimates_dict = self.load_estimates(pp, model_type = model_type)

            # load single trial DM
            single_trl_DM = self.load_single_trl_DM(pp)
            
            # if we want to load separately per attended color
            if ('sub-{sj}'.format(sj = pp) in list(att_color_ses_run_dict.keys())) and betas_per_color:
                att_color_ses_run = att_color_ses_run_dict['sub-{sj}'.format(sj = pp)]
            else:
                att_color_ses_run = None
                
            pp_prf_bar_coord = None if 'sub-{sj}'.format(sj = pp) not in list(prf_bar_coords_dict.keys()) else prf_bar_coords_dict['sub-{sj}'.format(sj = pp)]
                
            DF_betas_bar_coord_GROUP.append(self.get_betas_coord_df(pp, betas_arr = GLMsing_estimates_dict['betasmd'], 
                                                                single_trl_DM = single_trl_DM, 
                                                                att_color_ses_run = att_color_ses_run, 
                                                                file_ext = file_ext, 
                                                                demean = demean,
                                                                prf_bar_coords_dict = pp_prf_bar_coord,
                                                                rotate_bars = rotate_bars,
                                                                ROIs_dict = pp_ROI_dict, 
                                                                prf_estimates = prf_estimates, 
                                                                orientation_bars = orientation_bars)
            )
            
        ## concatenate to convert into data frame
        return pd.concat(DF_betas_bar_coord_GROUP, ignore_index = True)
                
    def get_betas_coord_df(self, participant, betas_arr = [], single_trl_DM = [], att_color_ses_run = {}, 
                                demean = False, rotate_bars = True, prf_bar_coords_dict = None,
                                file_ext = '_cropped.npy', ROIs_dict = {}, prf_estimates = {}, orientation_bars = 'parallel_vertical'):

        """
        Get dataframe with beta values, with info on prf location, attended color, ROI vertices belong to,
        for trials with a specific bar orientation (parallel - vert/hor -, or crossed) or for the combination of different 
        orientations

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
        orientation_bars: str/list
            if string with descriptor for bar orientations (crossed, parallel_vertical, parallel_horizontal or parallel (combination of previous ones))
            if list then gets and concatenates all orientations in list 
        prf_bar_coords_dict: dict
            if given, will use pRF bar coordinates as reference to mask FA trials where bar not visible
        """

        output_df = []
        
        # check which bar orientations to load
        if isinstance(orientation_bars, str):
            if orientation_bars == 'parallel':
                orientation_bars_list = ['parallel_vertical', 'parallel_horizontal']
            else:
                orientation_bars_list = [orientation_bars]
        else:
            orientation_bars_list = orientation_bars

        for ori_bars in orientation_bars_list:

            # make betas dataframe
            ori_betas_df = self.get_orientation_betas_coord_df(participant, betas_arr = betas_arr, single_trl_DM = single_trl_DM, 
                                                               att_color_ses_run = att_color_ses_run, 
                                                               demean = demean, prf_bar_coords_dict = prf_bar_coords_dict,
                                                                file_ext = file_ext, ROIs_dict = ROIs_dict, 
                                                                prf_estimates = prf_estimates, orientation_bars = ori_bars)
            
            # if we want to rotate horizontal bars
            if ori_bars == 'parallel_horizontal' and rotate_bars:
                ori_betas_df = self.rotate_prf_coordinates(DF_betas_bar_coord = ori_betas_df, 
                                                            og_orientation_bars = ori_bars)
            
            output_df.append(ori_betas_df)
            
        return pd.concat(output_df, ignore_index = True)
    
    def get_orientation_betas_coord_df(self, participant, betas_arr = [], single_trl_DM = [], att_color_ses_run = {}, 
                                            demean = False, prf_bar_coords_dict = None,
                                            file_ext = '_cropped.npy', ROIs_dict = {}, prf_estimates = {}, orientation_bars = 'parallel_vertical'):

        """
        make dataframe with beta values, with info on prf location, attended color, ROI vertices belong to,
        for trials with a specific bar orientation (parallel - vert/hor -, or crossed)

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
        prf_bar_coords_dict: dict
            if given, will use pRF bar coordinates as reference to mask FA trials where bar not visible
        """

        # if we dont want to load betas per color
        if att_color_ses_run is None or len(att_color_ses_run) == 0:
            att_color_keys = [None]
        else:
            att_color_keys = list(att_color_ses_run.keys())

        DF_betas_bar_coord = []

        ## for bars going left to right (vertical orientation)
        if orientation_bars == 'parallel_vertical':
            coord_list = self.bar_x_coords_pix
            # if we provided prf bar coordinates
            if prf_bar_coords_dict is not None:
                coord_list = [val for val in coord_list if val in prf_bar_coords_dict['horizontal']]  # horizontal bar pass, bars are vertically oriented
        elif orientation_bars == 'parallel_horizontal':
            coord_list = self.bar_y_coords_pix
            # if we provided prf bar coordinates
            if prf_bar_coords_dict is not None:
                coord_list = [val for val in coord_list if val in prf_bar_coords_dict['vertical']]  # vertical bar pass, bars are horizontally oriented
        else:
            raise ValueError('Cross sections not implemented yet')

        for UAtt_bar_coord in coord_list: # let's start with unattended bar vertical leftmost, attended bar left to right (5 positions)
            for Att_bar_coord in coord_list:

                if Att_bar_coord != UAtt_bar_coord: ## bars cannot fully overlap

                    # get eccentricity label of attended bar
                    ecc_label = [key for key, val in self.MRIObj.params['plotting']['bar_ecc_label'].items() if val == np.absolute(Att_bar_coord)]
                    # and distractor
                    distr_ecc_label = [key for key, val in self.MRIObj.params['plotting']['bar_ecc_label'].items() if val == np.absolute(UAtt_bar_coord)]

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
                    
                    ## get inter-bar distance 
                    if orientation_bars == 'crossed':
                        inter_bar_dist = np.nan
                        contralateral_bars_bool = np.nan
                    else:
                        inter_bar_dist = (UAtt_bar_coord - Att_bar_coord)/ self.bar_width_pix[0]
                        contralateral_bars_bool = np.sign(Att_bar_coord) != np.sign(UAtt_bar_coord) # if bars in contralateral hemifields
                        if orientation_bars == 'parallel_vertical':
                            Att_bar_contralateral_RF_bool = np.sign(prf_estimates['sub-{sj}'.format(sj = participant)]['x']) != np.sign(Att_bar_coord) # if attended bar in the same hemifield of pRF
                        elif orientation_bars == 'parallel_horizontal':
                            Att_bar_contralateral_RF_bool = np.sign(prf_estimates['sub-{sj}'.format(sj = participant)]['y']) != np.sign(Att_bar_coord) # if attended bar in the same hemifield of pRF

                    # differentiate per color
                    for c, color_name in enumerate(att_color_keys):
                    
                        # iterate over ROIs
                        for rname in ROIs_dict.keys():

                            DF_betas_bar_coord.append(pd.DataFrame({'sj': np.tile('sub-{sj}'.format(sj = participant), len(ROIs_dict[rname])), 
                                                                    'ROI': np.tile(rname, len(ROIs_dict[rname])), 
                                                                    'betas': avg_betas[0,:,c][ROIs_dict[rname]], 
                                                                    'attend_color': np.tile(color_name, len(ROIs_dict[rname])),
                                                                    'Att_bar_coord': np.tile(Att_bar_coord, len(ROIs_dict[rname])),
                                                                    'UAtt_bar_coord': np.tile(UAtt_bar_coord, len(ROIs_dict[rname])),
                                                                    'inter_bar_dist': np.tile(inter_bar_dist, len(ROIs_dict[rname])),
                                                                    'abs_inter_bar_dist': np.tile(np.absolute(inter_bar_dist), len(ROIs_dict[rname])),
                                                                    'contralateral_bars': np.tile(contralateral_bars_bool, len(ROIs_dict[rname])),
                                                                    'contralateral_Attbar_pRF': Att_bar_contralateral_RF_bool[ROIs_dict[rname]],
                                                                    'Att_ecc_label': np.tile(ecc_label, len(ROIs_dict[rname])),
                                                                    'UAtt_ecc_label': np.tile(distr_ecc_label, len(ROIs_dict[rname])),
                                                                    'prf_index_coord': ROIs_dict[rname],
                                                                    'prf_rsq_coord': prf_estimates['sub-{sj}'.format(sj = participant)]['r2'][ROIs_dict[rname]],
                                                                    'prf_x_coord': prf_estimates['sub-{sj}'.format(sj = participant)]['x'][ROIs_dict[rname]], 
                                                                    'prf_y_coord': prf_estimates['sub-{sj}'.format(sj = participant)]['y'][ROIs_dict[rname]]})
                                                                    )
                            
        DF_betas_bar_coord = pd.concat(DF_betas_bar_coord, ignore_index = True)
        DF_betas_bar_coord = DF_betas_bar_coord.dropna(subset=['prf_rsq_coord', 'prf_x_coord', 'prf_y_coord'])               
        
        # if we want to demean betas, per trial type and ROI
        if demean:
            DF_betas_bar_coord = self.demean_betas_df(DF_betas_bar_coord = DF_betas_bar_coord, 
                                            ROI_list = list(ROIs_dict.keys()), 
                                            orientation_bars = orientation_bars, 
                                            bar_color = None) ## HARDCODED, change later to go through different colors if wanted
            
        return DF_betas_bar_coord
    
    def get_betas_1D_df(self, DF_betas_bar_coord = {}, ROI_list = [], orientation_bars = 'parallel_vertical', bar_color2bin = None,
                                max_ecc_ext = 5.5, bin_size = .5, center_bin = False, avg_bool = True, bin_bool = False):

        """
        Transform participant model beta values (according to pRF x,y coordinates) into 1D average/binned average
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
        bin_size: float
            size of bin (in dva)
        bin_bool: bool
            if we want to bin over x or y coordinate
        avg_bool: bool
            if we bin, then we might also want to average
        center_bin: bool
            if we want to center the bin distribution around 0
        """

        # if we want to bin data
        if bin_bool:
            ## bins array (1/3 of bar width, equally spaced across x/y coordinates of screen)
            # if we want to have the bins centered around 0 
            if center_bin:
                bins_arr = np.arange(bin_size/2, max_ecc_ext + bin_size, bin_size)
                bins_arr = np.concatenate((bins_arr*-1, bins_arr))
            else:
                bins_arr = np.arange(0, max_ecc_ext + bin_size, bin_size)
                bins_arr = np.concatenate((bins_arr[1:]*-1, bins_arr))
            bins_arr.sort()

        # if no ROI specified, then plot all
        if len(ROI_list) == 0:
            ROI_list = DF_betas_bar_coord.ROI.unique()

        ## for bars going left to right (vertical orientation)
        if orientation_bars == 'parallel_vertical':
            coord_list = self.bar_x_coords_pix
            key2bin = 'prf_x_coord' # column name to bin values
        elif orientation_bars == 'parallel_horizontal':
            coord_list = self.bar_y_coords_pix
            key2bin = 'prf_y_coord'
        else:
            raise ValueError('Cross sections not implemented yet')
        
        DF_betas_bar_coord1D = pd.DataFrame()

        ## iterate over ROIs
        for roi_name in ROI_list:
            
            for UAtt_bar_coord in coord_list: 
                for Att_bar_coord in coord_list:
                    
                    trial_df = DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                        (DF_betas_bar_coord['Att_bar_coord'] == Att_bar_coord) &\
                                        (DF_betas_bar_coord['UAtt_bar_coord'] == UAtt_bar_coord)].dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas'])
                    
                    if not trial_df.empty: # if df not empty (which might happend when we averaged across trial types etc)

                        ## if we want to bin estimates for specific bar color
                        if bar_color2bin:
                            trial_df = trial_df[trial_df['attend_color'] == bar_color2bin]
                        else:
                            # average them, if we dont care
                            df_column_names = [str(name) for name in list(trial_df.columns) if name not in ['attend_color', 'betas']]
                            trial_df = trial_df.groupby(df_column_names).mean().reset_index()

                        if bin_bool:
                            for b in range(len(bins_arr)-1):

                                bin_df = trial_df[(trial_df[key2bin] >= bins_arr[b]) &\
                                                (trial_df[key2bin] <= bins_arr[b+1])]

                                if not bin_df.empty:
                                    if avg_bool:
                                        avg_bin_df = pd.DataFrame(np.hstack((bin_df.select_dtypes(exclude=np.number)[:1].values,
                                                                            bin_df.select_dtypes(include=np.number).mean().to_frame().T.values)),
                                                            columns = np.hstack((bin_df.select_dtypes(exclude=np.number)[:1].columns,
                                                                                bin_df.select_dtypes(include=np.number).mean().to_frame().T.columns)))
                                        avg_bin_df['betas'] = self.MRIObj.mri_utils.weighted_mean(bin_df.betas.values, 
                                                                                                weights = bin_df.prf_rsq_coord.values, 
                                                                                                norm = True)
                                        avg_bin_df['std'] = self.MRIObj.mri_utils.weighted_mean_std_sem(bin_df.betas.values, 
                                                                                                weights = bin_df.prf_rsq_coord.values, 
                                                                                                norm = True)[0]
                                        avg_bin_df['sem'] = self.MRIObj.mri_utils.weighted_mean_std_sem(bin_df.betas.values, 
                                                                                                weights = bin_df.prf_rsq_coord.values, 
                                                                                                norm = True)[-1]
                                        avg_bin_df['prf_x_coord'] = np.nanmean(bins_arr[b:b+2])
                                        avg_bin_df['prf_y_coord'] = np.nanmean(bins_arr[b:b+2])

                                        DF_betas_bar_coord1D = pd.concat((DF_betas_bar_coord1D, avg_bin_df))
                                    else:
                                        # if we dont want to average withing bins
                                        bin_df['prf_x_coord'] = np.nanmean(bins_arr[b:b+2])
                                        bin_df['prf_y_coord'] = np.nanmean(bins_arr[b:b+2])

                                        DF_betas_bar_coord1D = pd.concat((DF_betas_bar_coord1D, bin_df))

                        else:    
                            ## group by x-coordinates, and average
                            #out_df = trial_df.groupby(['sj', 'ROI', 'Att_bar_coord', 'UAtt_bar_coord', key2bin]).mean().reset_index()

                            DF_betas_bar_coord1D = pd.concat((DF_betas_bar_coord1D, trial_df))

        if bar_color2bin:
            DF_betas_bar_coord1D['attend_color'] = bar_color2bin
        else:
            DF_betas_bar_coord1D['attend_color'] = np.nan

        return DF_betas_bar_coord1D

    def get_betas_subtract_reverse_df(self, DF_betas_bar_coord = {}, ROI_list = [], orientation_bars = 'parallel_vertical', 
                                    bar_color = None):

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
        bar_color: str
            attended bar color name to filter for. if None will average across colors
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

        ## iterate over ROIs
        for roi_name in ROI_list:

            for Att_bar_coord in coord_list:
                for UAtt_bar_coord in coord_list: 
                    if Att_bar_coord < UAtt_bar_coord: ## we want to subtract over diagonal, so we want to select the lower half (where target always on the left of distractor)

                        numerator =  DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                                        (DF_betas_bar_coord['Att_bar_coord'] == Att_bar_coord) &\
                                                        (DF_betas_bar_coord['UAtt_bar_coord'] == UAtt_bar_coord)].dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas'])

                        ## swap target and distractor bar location, for subtraction
                        denominator =  DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                                        (DF_betas_bar_coord['Att_bar_coord'] == UAtt_bar_coord) &\
                                                        (DF_betas_bar_coord['UAtt_bar_coord'] == Att_bar_coord)].dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas'])
                    
                        if not numerator.empty and not denominator.empty: # if df not empty (which might happend when we averaged across trial types etc)

                            ## if we want to focus on specific bar color
                            if bar_color:
                                numerator = numerator[numerator['attend_color'] == bar_color]
                                denominator = denominator[denominator['attend_color'] == bar_color]
                            else:
                                # average them, if we dont care
                                df_column_names = [str(name) for name in list(numerator.columns) if name not in ['attend_color', 'betas']]
                                numerator = numerator.groupby(df_column_names).mean().reset_index()
                                denominator = denominator.groupby(df_column_names).mean().reset_index()
                        
                            ## actually subtract
                            subtracted_df = numerator.set_index(['sj', 'ROI',
                                                                'prf_x_coord', 'prf_y_coord',
                                                                'prf_rsq_coord'])['betas'].sub(denominator.set_index(['sj', 'ROI',
                                                                                                                    'prf_x_coord', 'prf_y_coord',
                                                                                                                    'prf_rsq_coord'])['betas']).reset_index()
                            subtracted_df['Att_bar_coord'] = Att_bar_coord
                            subtracted_df['UAtt_bar_coord'] = UAtt_bar_coord
                            
                            ## append
                            attention_coord_df = pd.concat((attention_coord_df, subtracted_df))

        if bar_color:
            attention_coord_df['attend_color'] = bar_color
        else:
            attention_coord_df['attend_color'] = np.nan

        return attention_coord_df

    def get_attention_coord_flipped_df(self, DF_A = {}, DF_B = {}, ROI_list = [], orientation_bars = 'parallel_vertical', 
                                    color_name = ['color_red', 'color_green'], average = True):

        """
        Make attention modulation df, calculated from GLMsingle beta estimates, relative to pRF coordinates.
        Involves subtracting two dataframes, where for the 2nd we flip trials (where target and distactor bars are in reversed postion), to isolate attentional effect

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
            ROI_list = DF_A.ROI.unique()

        ## for bars going left to right (vertical orientation)
        if orientation_bars == 'parallel_vertical':
            coord_list = self.bar_x_coords_pix
        elif orientation_bars == 'parallel_horizontal':
            coord_list = self.bar_y_coords_pix
        else:
            raise ValueError('Cross sections not implemented yet')
        
        ## reverse distractor dataframe
        # flipped over diagonal
        reverse_DF_B = DF_B.rename(columns={'Att_bar_coord': 'UAtt_bar_coord', 
                                            'UAtt_bar_coord': 'Att_bar_coord'})
        
        attention_coord_df = pd.DataFrame()

        # per bar color
        for col_name in color_name: 

            ## iterate over ROIs
            for roi_name in ROI_list:

                for Att_bar_coord in coord_list:
                    for UAtt_bar_coord in coord_list: 
                            
                        trial_DF_A =  DF_A[(DF_A['ROI'] == roi_name) &\
                                            (DF_A['Att_bar_coord'] == Att_bar_coord) &\
                                            (DF_A['UAtt_bar_coord'] == UAtt_bar_coord) &\
                                            (DF_A['attend_color'] == col_name)].dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas'])

                        ## swapped target and distractor bar location, for subtraction
                        trial_DF_B =  reverse_DF_B[(reverse_DF_B['ROI'] == roi_name) &\
                                                    (reverse_DF_B['Att_bar_coord'] == Att_bar_coord) &\
                                                    (reverse_DF_B['UAtt_bar_coord'] == UAtt_bar_coord) &\
                                                    (reverse_DF_B['attend_color'] == col_name)].dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas'])

                        if not trial_DF_A.empty:   
                            ## actually subtract
                            subtracted_df = trial_DF_A.set_index(['sj', 'ROI', 'attend_color',
                                                                'prf_x_coord', 'prf_y_coord',
                                                                'prf_rsq_coord'])['betas'].sub(trial_DF_B.set_index(['sj', 'ROI', 'attend_color',
                                                                                                                    'prf_x_coord', 'prf_y_coord',
                                                                                                                    'prf_rsq_coord'])['betas']).reset_index()
                            subtracted_df['Att_bar_coord'] = Att_bar_coord
                            subtracted_df['UAtt_bar_coord'] = UAtt_bar_coord
                            
                            ## append
                            attention_coord_df = pd.concat((attention_coord_df, subtracted_df))

        # if we want to average
        if average:
            attention_coord_df = attention_coord_df.dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas']).groupby(['prf_x_coord', 'prf_y_coord', 'prf_rsq_coord', 'Att_bar_coord', 'UAtt_bar_coord',
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
                                        (DF_betas_bar_coord['attend_color'] == col_name)].dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas'])

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
                                                            (DF_betas_bar_coord['attend_color'] == col_name)].dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas'])
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
            attention_coord_df = attention_coord_df.dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas']).groupby(['prf_x_coord', 'prf_y_coord', 'prf_rsq_coord', 'Att_bar_coord', 'UAtt_bar_coord',
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
                                        (DF_betas_bar_coord['attend_color'] == col_name)].dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas'])

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
                                                            (DF_betas_bar_coord['attend_color'] == col_name)].dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas'])
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
            distractor_coord_df = distractor_coord_df.dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas']).groupby(['prf_x_coord', 'prf_y_coord', 'prf_rsq_coord', 'Att_bar_coord', 'UAtt_bar_coord',
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
            key2bin = 'prf_x_coord' # column name to bin values
        elif orientation_bars == 'parallel_horizontal':
            coord_list = self.bar_y_coords_pix
            key2bin = 'prf_y_coord'
        else:
            raise ValueError('Cross sections not implemented yet')
        
        DF_betas_bar_avg1D = pd.DataFrame()

        ## iterate over ROIs
        for roi_name in ROI_list:
            
            for UAtt_bar_coord in coord_list: 
                for Att_bar_coord in coord_list:
                    
                    trial_df = DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                        (DF_betas_bar_coord['Att_bar_coord'] == Att_bar_coord) &\
                                        (DF_betas_bar_coord['UAtt_bar_coord'] == UAtt_bar_coord)].dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas'])

                    if not trial_df.empty: # if df not empty (which might happend when we averaged across trial types etc)

                        ## if we want to bin estimates for specific bar color
                        if bar_color2bin:
                            trial_df = trial_df[trial_df['attend_color'] == bar_color2bin]
                        else:
                            # average them, if we dont care
                            df_column_names = [str(name) for name in list(trial_df.columns) if name not in ['attend_color', 'betas']]
                            trial_df = trial_df.groupby(df_column_names).mean().reset_index()

                        ## filter df for each bar position
                        UAtt_bar_bin_df = trial_df[(trial_df[key2bin] >= self.convert_pix2dva(UAtt_bar_coord - self.bar_width_pix[0]/2)) &\
                                                    (trial_df[key2bin] <= self.convert_pix2dva(UAtt_bar_coord + self.bar_width_pix[0]/2))]

                        Att_bar_bin_df = trial_df[(trial_df[key2bin] >= self.convert_pix2dva(Att_bar_coord - self.bar_width_pix[0]/2)) &\
                                                    (trial_df[key2bin] <= self.convert_pix2dva(Att_bar_coord + self.bar_width_pix[0]/2))]
                        
                        ## combine both binned dfs
                        trial_binned_df = pd.concat((UAtt_bar_bin_df, Att_bar_bin_df))
                        trial_binned_df['bar_type'] = np.concatenate((np.tile('distractor', len(UAtt_bar_bin_df.betas.values)), 
                                                                      np.tile('target', len(Att_bar_bin_df.betas.values))), axis=None)
                        
                        if avg_bool:
                            avg_Att_df = pd.DataFrame(np.hstack((trial_binned_df[trial_binned_df['bar_type'] == 'target'].select_dtypes(exclude=np.number)[:1].values,
                                                                trial_binned_df[trial_binned_df['bar_type'] == 'target'].select_dtypes(include=np.number).mean().to_frame().T.values)),
                                                    columns = np.hstack((trial_binned_df[trial_binned_df['bar_type'] == 'target'].select_dtypes(exclude=np.number)[:1].columns,
                                                                trial_binned_df[trial_binned_df['bar_type'] == 'target'].select_dtypes(include=np.number).mean().to_frame().T.columns)))
                            
                            avg_Att_df['betas'] = self.MRIObj.mri_utils.weighted_mean(trial_binned_df[trial_binned_df['bar_type'] == 'target'].betas.values, 
                                                                                    weights = trial_binned_df[trial_binned_df['bar_type'] == 'target'].prf_rsq_coord.values, 
                                                                                    norm = True)
                            avg_Att_df['std'] = self.MRIObj.mri_utils.weighted_mean_std_sem(trial_binned_df[trial_binned_df['bar_type'] == 'target'].betas.values, 
                                                                                    weights = trial_binned_df[trial_binned_df['bar_type'] == 'target'].prf_rsq_coord.values, 
                                                                                    norm = True)[0]
                            avg_Att_df['sem'] = self.MRIObj.mri_utils.weighted_mean_std_sem(trial_binned_df[trial_binned_df['bar_type'] == 'target'].betas.values, 
                                                                                    weights = trial_binned_df[trial_binned_df['bar_type'] == 'target'].prf_rsq_coord.values, 
                                                                                    norm = True)[-1]
                            
                            avg_UAtt_df = pd.DataFrame(np.hstack((trial_binned_df[trial_binned_df['bar_type'] == 'distractor'].select_dtypes(exclude=np.number)[:1].values,
                                                                trial_binned_df[trial_binned_df['bar_type'] == 'distractor'].select_dtypes(include=np.number).mean().to_frame().T.values)),
                                                    columns = np.hstack((trial_binned_df[trial_binned_df['bar_type'] == 'distractor'].select_dtypes(exclude=np.number)[:1].columns,
                                                                trial_binned_df[trial_binned_df['bar_type'] == 'distractor'].select_dtypes(include=np.number).mean().to_frame().T.columns)))
                            
                            avg_UAtt_df['betas'] = self.MRIObj.mri_utils.weighted_mean(trial_binned_df[trial_binned_df['bar_type'] == 'distractor'].betas.values, 
                                                                                    weights = trial_binned_df[trial_binned_df['bar_type'] == 'distractor'].prf_rsq_coord.values, 
                                                                                    norm = True)
                            avg_UAtt_df['std'] = self.MRIObj.mri_utils.weighted_mean_std_sem(trial_binned_df[trial_binned_df['bar_type'] == 'distractor'].betas.values, 
                                                                                    weights = trial_binned_df[trial_binned_df['bar_type'] == 'distractor'].prf_rsq_coord.values, 
                                                                                    norm = True)[0]
                            avg_UAtt_df['sem'] = self.MRIObj.mri_utils.weighted_mean_std_sem(trial_binned_df[trial_binned_df['bar_type'] == 'distractor'].betas.values, 
                                                                                    weights = trial_binned_df[trial_binned_df['bar_type'] == 'distractor'].prf_rsq_coord.values, 
                                                                                    norm = True)[-1]

                            DF_betas_bar_avg1D = pd.concat((DF_betas_bar_avg1D, pd.concat((avg_UAtt_df, avg_Att_df))))

                        else:
                            DF_betas_bar_avg1D = pd.concat((DF_betas_bar_avg1D, trial_binned_df))

                        
        if bar_color2bin:
            DF_betas_bar_avg1D['attend_color'] = bar_color2bin

        return DF_betas_bar_avg1D
    
    def get_delta_betas_bar_df(self, DF_betas_bar_coord = {}, ROI_list = [], orientation_bars = 'parallel_vertical', bar_color2bin = None, avg_bool = True):

        """
        Calculate delta beta values (according to pRF x,y coordinates) -> beta target bar - beta distractor bar
        for different ROIs. Essentially subtracting the flipped trial

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
            key2bin = 'prf_x_coord' # column name to bin values
        elif orientation_bars == 'parallel_horizontal':
            coord_list = self.bar_y_coords_pix
            key2bin = 'prf_y_coord'
        else:
            raise ValueError('Cross sections not implemented yet')
        
        DF_betas_bar_avg1D = pd.DataFrame()

        ## iterate over ROIs
        for roi_name in ROI_list:
            
            for UAtt_bar_coord in coord_list: 
                for Att_bar_coord in coord_list:
                    
                    trial_df = DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                        (DF_betas_bar_coord['Att_bar_coord'] == Att_bar_coord) &\
                                        (DF_betas_bar_coord['UAtt_bar_coord'] == UAtt_bar_coord)].dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas'])
                    
                    flipped_trial_df = DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                        (DF_betas_bar_coord['Att_bar_coord'] == UAtt_bar_coord) &\
                                        (DF_betas_bar_coord['UAtt_bar_coord'] == Att_bar_coord)].dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas'])

                    if not trial_df.empty and not flipped_trial_df.empty: # if dfs not empty (which might happend when we averaged across trial types etc)

                        ## if we want to bin estimates for specific bar color
                        if bar_color2bin:
                            trial_df = trial_df[trial_df['attend_color'] == bar_color2bin]
                            flipped_trial_df = flipped_trial_df[flipped_trial_df['attend_color'] == bar_color2bin]
                        else:
                            # average them, if we dont care
                            df_column_names = [str(name) for name in list(trial_df.columns) if name not in ['attend_color', 'betas']]
                            trial_df = trial_df.groupby(df_column_names).mean().reset_index()
                            flipped_trial_df = flipped_trial_df.groupby(df_column_names).mean().reset_index()

                        ## filter df for each bar position
                        Att_bar_bin_df = trial_df[(trial_df[key2bin] >= self.convert_pix2dva(Att_bar_coord - self.bar_width_pix[0]/2)) &\
                                                    (trial_df[key2bin] <= self.convert_pix2dva(Att_bar_coord + self.bar_width_pix[0]/2))]
                        
                        flipped_bar_bin_df = flipped_trial_df[(flipped_trial_df[key2bin] >= self.convert_pix2dva(Att_bar_coord - self.bar_width_pix[0]/2)) &\
                                                            (flipped_trial_df[key2bin] <= self.convert_pix2dva(Att_bar_coord + self.bar_width_pix[0]/2))]
                        
                        ## subtract reverse trial type
                        Att_bar_bin_df.sort_values('prf_index_coord', inplace=True)
                        flipped_bar_bin_df.sort_values('prf_index_coord', inplace=True)
                        
                        delta_betas_df = Att_bar_bin_df.copy()
                        delta_betas_df['betas'] = Att_bar_bin_df['betas'] - flipped_bar_bin_df['betas']
                        
                        if avg_bool:
                            avg_Delta_df = pd.DataFrame(np.hstack((delta_betas_df.select_dtypes(exclude=np.number)[:1].values,
                                                                delta_betas_df.select_dtypes(include=np.number).mean().to_frame().T.values)),
                                                    columns = np.hstack((delta_betas_df.select_dtypes(exclude=np.number)[:1].columns,
                                                                        delta_betas_df.select_dtypes(include=np.number).mean().to_frame().T.columns)))
                            
                            avg_Delta_df['betas'] = self.MRIObj.mri_utils.weighted_mean(delta_betas_df.betas.values, 
                                                                                    weights = delta_betas_df.prf_rsq_coord.values, 
                                                                                    norm = True)
                            avg_Delta_df['std'] = self.MRIObj.mri_utils.weighted_mean_std_sem(delta_betas_df.betas.values, 
                                                                                    weights = delta_betas_df.prf_rsq_coord.values, 
                                                                                    norm = True)[0]
                            avg_Delta_df['sem'] = self.MRIObj.mri_utils.weighted_mean_std_sem(delta_betas_df.betas.values, 
                                                                                    weights = delta_betas_df.prf_rsq_coord.values, 
                                                                                    norm = True)[-1]

                            DF_betas_bar_avg1D = pd.concat((DF_betas_bar_avg1D, avg_Delta_df))

                        else:
                            DF_betas_bar_avg1D = pd.concat((DF_betas_bar_avg1D, delta_betas_df))
    
        if bar_color2bin:
            DF_betas_bar_avg1D['attend_color'] = bar_color2bin

        return DF_betas_bar_avg1D
    
    def demean_betas_df(self, DF_betas_bar_coord = {}, ROI_list = [], orientation_bars = 'parallel_vertical', 
                                    bar_color = None):

        """
        Quick func to shift beta values by mean of trial type.
        Might change location/content later
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
        
        new_df = pd.DataFrame()

         ## iterate over ROIs
        for roi_name in ROI_list:
            
            for UAtt_bar_coord in coord_list: 
                for Att_bar_coord in coord_list:
                    
                    trial_df = DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                        (DF_betas_bar_coord['Att_bar_coord'] == Att_bar_coord) &\
                                        (DF_betas_bar_coord['UAtt_bar_coord'] == UAtt_bar_coord)].dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas'])
                    
                    if not trial_df.empty: # if df not empty (which might happend when we averaged across trial types etc)

                        ## if we want to focus on specific bar color
                        if bar_color:
                            trial_df = trial_df[trial_df['attend_color'] == bar_color]
                        else:
                            # average them, if we dont care
                            df_column_names = [str(name) for name in list(trial_df.columns) if name not in ['attend_color', 'betas']]
                            trial_df = trial_df.groupby(df_column_names).mean().reset_index()

                        ## get average across betas and subtract
                        trial_df['betas'] -= np.mean(trial_df['betas'])

                        ## append
                        new_df = pd.concat((new_df, trial_df))

        if bar_color:
            new_df['attend_color'] = bar_color
        else:
            new_df['attend_color'] = np.nan

        return new_df
    
    def get_betas_grid_coord_df(self, DF_betas_bar_coord = {}, collapse_ecc = False, orientation_bars = 'parallel_vertical', grid_num = None,
                                    mirror_symetric = True):

        """
        convert dataframe of betas coordinates, into grid betas coord 
        --> to then use for heatmap representation <---

        for trials with a specific bar orientation (parallel - vert/hor -, or crossed) 
        or for the combination of different orientations

        Parameters
        ----------
        DF_betas_bar_coord: dataframe
            FA beta values dataframe for a/all participant, with relevant prf estimates (x,y,r2)
        collapse_ecc: bool
            if we want to collapse over eccentricity (will still keep reversed condition)
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        grid_num: int
            if provided, will bin screen coordinates into given number 
        """

        ## when parallel assumes that we have already combined parallel vertical with rotated parallel horizontal in betas dataframe
        # so heatmap like coordinates will be applied to "vertical" reference frame
        grid_orientation_bars = 'parallel_vertical' if orientation_bars == 'parallel' else orientation_bars

        return self.get_orientation_betas_grid_coord_df(DF_betas_bar_coord = DF_betas_bar_coord, 
                                                        collapse_ecc = collapse_ecc, 
                                                        orientation_bars = grid_orientation_bars,
                                                        grid_num = grid_num,
                                                        mirror_symetric = mirror_symetric)

    def get_orientation_betas_grid_coord_df(self, DF_betas_bar_coord = {}, collapse_ecc = False, orientation_bars = 'parallel_vertical', 
                                                grid_num = None, mirror_symetric = True):

        """
        convert dataframe of betas coordinates, into grid betas coord 
        --> to then use for heatmap representation <---

        NOTE: provide DF_betas_bar_coord for SPECIFIC bar orientation (ex: parallel_vertical), 
        because mixing will lead to errors when labelling

        Parameters
        ----------
        DF_betas_bar_coord: dataframe
            FA beta values dataframe for a participant, with relevant prf estimates (x,y,r2)
        collapse_ecc: bool
            if we want to collapse over eccentricity (will still keep reversed condition)
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        """
        
        # dict with absolute inter bar distance per eccentricity --> hardcoded, so improve later
        abs_dist_dict = {'far': np.arange(5)+1, 'middle': np.arange(3)+1, 'near': np.arange(1)+1}
        #ecc_order_dict = {'far': ['middle', 'near'], 'middle': ['near'], 'near': []}

        ## for bars going left to right (vertical orientation) or bottom-up (horizontal orientation)
        if orientation_bars == 'parallel_vertical':
            collapse_coord_key = 'screen_x_coord'
        elif orientation_bars == 'parallel_horizontal':
            collapse_coord_key = 'screen_y_coord'
        else:
            raise ValueError('Cross sections not implemented yet')

        DF_betas_GRID_coord = []
        
        for (cols, tdf) in DF_betas_bar_coord.groupby(['sj', 'ROI', 'Att_bar_coord', 'UAtt_bar_coord']):
            
            trial_df = tdf.copy()
            
            ## label prf coordinates to
            # screen binned coords
            x_deg_arr, y_deg_arr = self.convert_coord_pRF2screen_grid(prf_x_coord = trial_df.prf_x_coord.values, 
                                                                        prf_y_coord = trial_df.prf_y_coord.values,
                                                                        grid_num = grid_num)
            
            ## keep some information on the attended bar, for bookeeping --> need to multiindex
            df_betas_heatmap = trial_df.loc[:, ['sj', 'ROI', 'betas', 'prf_index_coord', 'Att_bar_coord', 'UAtt_bar_coord', 
                                                'inter_bar_dist', 'abs_inter_bar_dist', 'Att_ecc_label', 'UAtt_ecc_label']]
            df_betas_heatmap[['screen_x_coord', 'screen_y_coord']] = np.vstack((x_deg_arr, y_deg_arr)).T
            
            DF_betas_GRID_coord.append(df_betas_heatmap)
        output_df = pd.concat(DF_betas_GRID_coord, ignore_index = True)
        
        # collapsing same eccentricities of attended and unattended bar
        # while keeping flipped trial types
        if collapse_ecc:
            
            ecc_df = []
            
            for (cols, rdf) in output_df.groupby(['sj', 'ROI']): # per participant and ROI
                
                ROI_betas_GRID_coord = rdf.copy()

                ecc_df.append(self.collapse_conditions_ecc(DF_betas_bar_coord = ROI_betas_GRID_coord,
                                                                    orientation_bars = orientation_bars,
                                                                    collapse_coord_key = collapse_coord_key,
                                                                    mirror_symetric = mirror_symetric))
            ecc_df = pd.concat(ecc_df, ignore_index = True)
            output_df = ecc_df.copy()
            
        return output_df

    def collapse_conditions_ecc(self, DF_betas_bar_coord = None, orientation_bars = 'parallel_vertical', collapse_coord_key = None,
                                    mirror_symetric = True):

        """
        quick function to collapse conditions over eccentricity (will still keep reversed condition)
        by flipping prf x/y coordinates (or other coordinates - example for bins/grid - if given)
        """

        # if we didnt specify the column key over which to collapse
        # will change prf coordinates directly
        if collapse_coord_key is None:
            ## for bars going left to right (vertical orientation) or bottom-up (horizontal orientation)
            if orientation_bars == 'parallel_vertical':
                collapse_coord_key = 'prf_x_coord'
            elif orientation_bars == 'parallel_horizontal':
                collapse_coord_key = 'prf_y_coord'
            else:
                raise ValueError('Cross sections not implemented yet')

        # dict with absolute inter bar distance per eccentricity --> hardcoded, so improve later
        abs_dist_dict = {'far': np.arange(5)+1, 'middle': np.arange(3)+1, 'near': np.arange(1)+1}

        # collapsing same eccentricities of attended and unattended bar
        # while keeping flipped trial types
        ECC_betas_GRID_coord = []
                
        for Att_ecc_label in abs_dist_dict.keys():
            for abs_dist in abs_dist_dict[Att_ecc_label]:

                # filter dataframe for specific attended bar eccentricity and inter-bar distance
                distance_ecc_df = DF_betas_bar_coord.loc[(DF_betas_bar_coord['Att_ecc_label'] == Att_ecc_label) &\
                                                        (DF_betas_bar_coord['abs_inter_bar_dist'] == abs_dist)]
            
                if not distance_ecc_df.empty:

                    # get all attended bar coordinates that belong to this eccentricity and distance
                    Att_bar_coord_list = distance_ecc_df.Att_bar_coord.unique()
                    Att_bar_coord_list = np.sort(Att_bar_coord_list) # and sort

                    # if simetrical case (bars equidistant to fixation) 
                    # then there's nothing to collapse across ecc () unless we want to
                    if mirror_symetric == False and not distance_ecc_df[distance_ecc_df['Att_ecc_label'] == distance_ecc_df['UAtt_ecc_label']].empty:
                        Att_bar_coord_list = Att_bar_coord_list[:1]

                     # actually collapse
                    for Att_bar_coord in Att_bar_coord_list:
                        print('Target coordinate: %s \nAbsolute dist: %i \nnum collapsed trials %i'%(str(Att_bar_coord), abs_dist, len(Att_bar_coord_list)))

                        # overlay both sides of hemifield
                        if Att_bar_coord < 0: 
                            ATT_df = distance_ecc_df[(distance_ecc_df['Att_bar_coord'] == Att_bar_coord) &\
                                                    (distance_ecc_df['UAtt_bar_coord'] > Att_bar_coord)]
                        else:
                            ATT_df = distance_ecc_df[(distance_ecc_df['Att_bar_coord'] == Att_bar_coord) &\
                                                    (distance_ecc_df['UAtt_bar_coord'] < Att_bar_coord)] 
                            
                        # easiest way to split conditions into "unique" trial types and their reverse
                        UAtt_bar_coord = ATT_df.UAtt_bar_coord.unique()[0]
                        UATT_df = DF_betas_bar_coord[(DF_betas_bar_coord['Att_bar_coord'] == UAtt_bar_coord) &\
                                                    (DF_betas_bar_coord['UAtt_bar_coord'] == Att_bar_coord) &\
                                                    (DF_betas_bar_coord['abs_inter_bar_dist'] == abs_dist)]
                        
                        # rotate relevant axis coordinate, because we want to collapse over ecc 
                        if Att_bar_coord > 0: 
                            ATT_df[collapse_coord_key] = ATT_df[collapse_coord_key].values * -1
                            UATT_df[collapse_coord_key] = UATT_df[collapse_coord_key].values * -1

                        # label if condition or flipped, for bookeeping
                        ATT_df[['flipped_condition']] = np.zeros(len(ATT_df)).reshape(-1,1)
                        UATT_df[['flipped_condition']] = np.ones(len(UATT_df)).reshape(-1,1)

                        ECC_betas_GRID_coord.append(ATT_df)
                        ECC_betas_GRID_coord.append(UATT_df)
                        
        ECC_betas_GRID_coord = pd.concat(ECC_betas_GRID_coord, ignore_index = True)
        
        return ECC_betas_GRID_coord

    def rotate_prf_coordinates(self, DF_betas_bar_coord = None, og_orientation_bars = 'parallel_horizontal'):

        """
        quick function to rotate betas coordinates
        """
        NEW_DF_betas_bar_coord = DF_betas_bar_coord.copy()

        ## if given parallel horizontal case, we want to rotate 90 deg clockwise
        # which implies transformation of y --> new x and x --> -new y
        if og_orientation_bars == 'parallel_horizontal':

            ## rotate 90 deg to align to parallel vertical
            new_y = DF_betas_bar_coord.prf_x_coord.values * -1
            new_x = DF_betas_bar_coord.prf_y_coord.values 
            
            NEW_DF_betas_bar_coord['prf_x_coord'] = new_x
            NEW_DF_betas_bar_coord['prf_y_coord'] = new_y

        return NEW_DF_betas_bar_coord
    
    def get_mean_betas_grid(self, DF_betas_GRID_coord = None, roi_name = 'V1'):
        
        """
        Get mean betas across y coordinate (collapse over bar ecc)
        given a df with beta values binned (grid) in x,y screen coordinates
        will return df with mean betas and sem for each condition - and its reverse case 
        for the participant/group
        """
        
        ## get y coordinates for plotting
        y_coord_grid = DF_betas_GRID_coord.screen_y_coord.unique()
        y_coord_grid = np.sort(y_coord_grid)
        
        # dict of bar ecc position --> to label conditions
        abs_dist_dict = {'far': np.arange(5)+1, 'middle': np.arange(3)+1, 'near': np.arange(1)+1}
        pos_ecc_rect_dict = {'far': 1, 'middle': 2, 'near': 3}

        flipped_condition_dict = {0: 'Att_ecc_label', 1:'UAtt_ecc_label'}

        ## make a df with mean betas across y
        # per trial and flipped condition
        mean_betas_df = pd.DataFrame()

        for flipped_trials in flipped_condition_dict.keys():
            for Att_bar_ecc in abs_dist_dict.keys(): # for each attended bar distance     
                for abs_dist in abs_dist_dict[Att_bar_ecc]: # for each inter bar distance
                    
                    GRIDdf2plot = DF_betas_GRID_coord[(DF_betas_GRID_coord['ROI'] == roi_name) &\
                                        (DF_betas_GRID_coord['flipped_condition'] == int(flipped_trials)) &\
                                        (DF_betas_GRID_coord[flipped_condition_dict[int(flipped_trials)]] == Att_bar_ecc) &\
                                        (DF_betas_GRID_coord['abs_inter_bar_dist'] == abs_dist)]

                    if not GRIDdf2plot.empty: # if dataframe not empty

                        # if more than 1 participant (GROUP)
                        if len(GRIDdf2plot.sj.unique()) > 1:
                            
                            ## average for same grid screen x coordinates
                            mean_GRIDdf2plot = GRIDdf2plot.groupby(['sj', 'ROI', 'abs_inter_bar_dist', 'screen_x_coord', 'flipped_condition']).mean().reset_index()
                            
                            # Divide each sj response by norm
                            norm_data = []
                            for (cols, g) in mean_GRIDdf2plot.sort_values(by=['screen_x_coord']).groupby(['sj']):
                                sd = g.copy()
                                sd.loc[:, 'norm'] = np.linalg.norm(sd['betas'])
                                sd.loc[:, 'betas_norm'] = sd['betas'] / sd['norm']
                                norm_data.append(sd)
                            norm_data = pd.concat(norm_data)
                            
                            # Average across subjects and multiply by average norm to get units back
                            mean_norm_data = norm_data.groupby(['screen_x_coord']).mean().reset_index()
                            mean_norm_data.loc[:, 'sem_betas_norm'] = norm_data.groupby(['screen_x_coord']).sem().betas_norm.values
                            mean_norm_data['betas_adj'] = mean_norm_data['betas_norm'] * mean_norm_data['norm']
                            mean_norm_data['sem_adj'] = mean_norm_data['sem_betas_norm'] * mean_norm_data['norm'] # rescaling sem by the same constant as mean
                            #mean_norm_data = mean_norm_data.drop(columns=['norm', 'betas_norm'])
                            
                            # store mean
                            mean_betas = mean_norm_data.betas_adj.values
                            x_pos = mean_norm_data.screen_x_coord.values
                            # store sem
                            sem_betas = mean_norm_data.sem_adj.values
                        else:
                            # get mean over across y coordinates to plot average line + SEM
                            mean_betas = GRIDdf2plot.sort_values(by=['screen_x_coord']).groupby(['screen_x_coord']).mean().betas.values
                            sem_betas = GRIDdf2plot.sort_values(by=['screen_x_coord']).groupby(['screen_x_coord']).sem().betas.values
                            x_pos = GRIDdf2plot.sort_values(by=['screen_x_coord']).groupby(['screen_x_coord']).mean().index.values

                        # also store bar position for plotting
                        if int(flipped_trials) == 0:   
                            bar_pos = y_coord_grid[pos_ecc_rect_dict[Att_bar_ecc]]
                            distractor_pos = y_coord_grid[pos_ecc_rect_dict[Att_bar_ecc] + abs_dist]
                        else:
                            bar_pos = y_coord_grid[pos_ecc_rect_dict[Att_bar_ecc] + abs_dist]
                            distractor_pos = y_coord_grid[pos_ecc_rect_dict[Att_bar_ecc]]
                            
                        ## concat
                        mean_betas_df = pd.concat((mean_betas_df, 
                                                pd.DataFrame({'ROI': np.repeat(roi_name, len(x_pos)),
                                                            'flipped_condition': np.repeat(int(flipped_trials), len(x_pos)),
                                                            'bar_pos': np.repeat(bar_pos, len(x_pos)),
                                                            'distractor_pos': np.repeat(distractor_pos, len(x_pos)),
                                                            'Att_bar_ecc': np.repeat(Att_bar_ecc, len(x_pos)),
                                                            'abs_dist': np.repeat(abs_dist, len(x_pos)),
                                                            'x_pos': x_pos,
                                                            'mean_betas': mean_betas,
                                                            'sem_betas': sem_betas})), 
                                                ignore_index = True)
                        
        return mean_betas_df

                        
        
        