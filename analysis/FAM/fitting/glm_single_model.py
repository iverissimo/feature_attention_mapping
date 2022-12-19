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

from FAM.utils import mri as mri_utils
from FAM.processing import preproc_behdata
from FAM.fitting.model import Model

from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter

from joblib import Parallel, delayed
from tqdm import tqdm

from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel
from glmsingle.glmsingle import GLM_single
from glmsingle.glmsingle import getcanonicalhrf

import time
import cortex
import matplotlib.pyplot as plt


class GLMsingle_Model(Model):

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

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.MRIObj.derivatives_pth, self.MRIObj.params['mri']['fitting']['FA']['fit_folder']['glmsingle'])
        else:
            self.outputdir = outputdir


    def get_single_trial_combinations(self):

        """
        Helper function to get all possible trial combinations
        (useful to keep track of single trial DM later)

        will return a DataFrame 
        where the columns indicate the attended and unattended bar midpoint position (x,y) and bar pass direction (vertical vs horizontal)
        and each row is a unique trial type
            
        """


        # define bar width in pixel
        bar_width_pix = self.screen_res * self.bar_width['FA']

        # define number of bars per direction
        num_bars = np.array(self.MRIObj.params['FA']['num_bar_position']) 

        # all possible positions in pixels [x,y] for midpoint of
        # vertical bar passes, 
        ver_y = np.sort(np.concatenate((-np.arange(bar_width_pix[1]/2,self.screen_res[1]/2,bar_width_pix[1])[0:int(num_bars[1]/2)],
                                        np.arange(bar_width_pix[1]/2,self.screen_res[1]/2,bar_width_pix[1])[0:int(num_bars[1]/2)])))

        ver_bar_pos_pix = np.array([np.array([0,y]) for _,y in enumerate(ver_y)])

        # horizontal bar passes 
        hor_x = np.sort(np.concatenate((-np.arange(bar_width_pix[0]/2,self.screen_res[0]/2,bar_width_pix[0])[0:int(num_bars[0]/2)],
                                        np.arange(bar_width_pix[0]/2,self.screen_res[0]/2,bar_width_pix[0])[0:int(num_bars[0]/2)])))

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


    def make_singletrial_dm(self, participant, run_num_arr =[], ses_num_arr = []):

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
            
        """

        ## get all possible trial combinations
        # to use for bookkeeping of single trial DM
        try:
            self.trial_combinations_df
        except AttributeError:
            self.get_single_trial_combinations()


        # set number of TRs to crop
        crop_nr = self.crop_TRs_num['FA'] if self.crop_TRs['FA'] == True else None

        ## get conditions per TR
        ## crop and shift if such was the case
        condition_per_TR = mri_utils.crop_shift_arr(self.mri_beh.FA_bar_pass_all,
                                                crop_nr = crop_nr, 
                                                shift = self.shift_TRs_num)

        ## make single trial DM
        # with shape [runs, TRs, conditions]
        single_trl_DM = np.zeros((len(run_num_arr), len(condition_per_TR), len(self.trial_combinations_df)))

        ## loop over runs
        for file_ind in range(len(run_num_arr)):

            ses_num = ses_num_arr[file_ind]
            run_num = run_num_arr[file_ind]

            ## get bar position df for run
            run_bar_pos_df = self.mri_beh.load_FA_bar_position(participant, ses = 'ses-{s}'.format(s = ses_num), 
                                                    ses_type = 'func', run_num = run_num)

            ## get run bar midpoint and direction values
            # for each bar type
            AttBar_bar_midpoint = run_bar_pos_df[run_bar_pos_df['attend_condition'] == 1].bar_midpoint_at_TR.values[0]
            AttBar_bar_pass_direction = run_bar_pos_df[run_bar_pos_df['attend_condition'] == 1].bar_pass_direction_at_TR.values[0]

            UnattBar_bar_midpoint = run_bar_pos_df[run_bar_pos_df['attend_condition'] == 0].bar_midpoint_at_TR.values[0]
            UnattBar_bar_pass_direction = run_bar_pos_df[run_bar_pos_df['attend_condition'] == 0].bar_pass_direction_at_TR.values[0]

            # set trial index counter
            trl_ind = 0

            ## fill DM for all TRs
            for i_TR, cond in enumerate(condition_per_TR):
                
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
        # shifted by onset (stc)
        # upsampled
        hrf_ind2use = [prf_modelobj.create_hrf(hrf_params = [1, 
                                                            pp_prf_estimates['hrf_derivative'][vert],
                                                            pp_prf_estimates['hrf_dispersion'][vert]], 
                                                            osf = self.osf * self.MRIObj.TR, 
                                                            onset = self.hrf_onset) for vert in ind2use]
        hrf_ind2use = np.vstack(hrf_ind2use)

        ## average HRF, weighted by the pRF RSQ
        avg_hrf = np.average(hrf_ind2use, axis=0, weights=pp_prf_estimates['r2'][ind2use])

        ## convolve to get the predicted response 
        # to the desired stimulus duration
        stim_dur = self.MRIObj.FA_bars_phase_dur # duration of bar presentation in seconds
        res_step = self.MRIObj.TR/(self.MRIObj.TR * self.osf) # resolution of upsampled HRF

        hrf_stim_convolved = np.convolve(avg_hrf, np.ones(int(np.max([1, np.round(stim_dur/res_step)]))))

        ## now resample again to the TR
        hrf_final = pchip(np.asarray(range(hrf_stim_convolved.shape[0])) * res_step,
                        hrf_stim_convolved)(np.asarray(np.arange(0, int((hrf_stim_convolved.shape[0]-1) * res_step), self.MRIObj.TR)))
        

        return hrf_final/np.max(hrf_final)


    def get_singletrial_avg_estimates(self, estimate_arr = [], single_trl_DM = [], return_std = True):

        """
        Helper function that takes in an estimate array from glmsingle
        [vertex, singletrials] (note: single trial number is multiplied by nr of runs)
        
        and outputs an average value for each single trial type (our condition)
        [vertex, average_estimate4trialtype]

        """

        # set number of TRs to crop
        crop_nr = self.crop_TRs_num['FA'] if self.crop_TRs['FA'] == True else None

        ## get conditions per TR
        ## crop and shift if such was the case
        condition_per_TR = mri_utils.crop_shift_arr(self.mri_beh.FA_bar_pass_all,
                                                crop_nr = crop_nr, 
                                                shift = self.shift_TRs_num)
        
        ## subselect task TRs indices
        task_indices = np.where((condition_per_TR) == 'task')[0]
        
        ## now append the estimate for that vertex for the same trial type (and std if we also want that)
        avg_all = []
        std_all = []

        for i in range(len(task_indices)):

            ## indices select for task on TRs (trials)
            cond_ind = np.where((np.hstack(single_trl_DM[:,task_indices, i])) == 1)[0]
            
            avg_all.append(np.mean(estimate_arr[...,cond_ind], axis = -1))
            if return_std:
                std_all.append(np.std(estimate_arr[...,cond_ind], axis = -1))
            
        if return_std:
            return np.stack(avg_all), np.stack(std_all)
        else:
            return np.stack(avg_all)


    def fit_data(self, participant, pp_prf_estimates, prf_modelobj,  file_ext = '_cropped.npy'):

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

        """ 

        ## get list of files to load
        bold_filelist = self.get_bold_file_list(participant, task = 'FA', ses = 'ses-combined', file_ext = file_ext)

        ## Not correcting baseline
        self.correct_baseline['FA'] = False

        ## Load data array and file list names
        data, train_file_list = self.get_data4fitting(bold_filelist, task = 'FA', run_type = 'all', 
                                                chunk_num = None, vertex = None, ses = 'ses-combined',
                                                baseline_interval = 'empty', return_filenames = True)

        ## Make single trial DM for all runs
        single_trl_DM = self.make_singletrial_dm(participant, 
                                                run_num_arr = self.run_num_arr, 
                                                ses_num_arr = self.ses_num_arr)

        print('Fitting files %s'%str(train_file_list))

        ## set output dir to save estimates
        outdir = op.join(self.outputdir, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant))

        os.makedirs(outdir, exist_ok = True)
        print('saving files in %s'%outdir)

        ## get average hrf
        hrf_final = self.get_average_hrf(pp_prf_estimates, prf_modelobj, rsq_threshold = self.prf_rsq_threshold)

        ### make mask array of pRF high fitting voxels,
        # to give as input to glmsingle
        # excluding them from noise pool

        # # first make a smooth mask of rsq
        # fwhm = 2
        # voxelsize = 1.6
        # rsq_arr = pp_prf_estimates['r2'].copy()
        # smoothed_vol = gaussian_filter(rsq_arr, sigma = fwhm / (np.sqrt(8 * np.log(2)) * voxelsize))

        # ## documentation is misleading, we want to set to 0 the ones that are not in the noise pool 
        # #prf_mask = np.ones(pp_prf_estimates['r2'].shape)
        # #prf_mask[np.where((pp_prf_estimates['r2'] > self.prf_rsq_threshold))[0]] = 0
        # prf_mask = np.ones(rsq_arr.shape)
        # prf_mask[smoothed_vol > 0] = 0

        # get prf bold filenames
        prf_bold_files = self.get_bold_file_list(participant, task = 'pRF', ses = 'ses-mean', file_ext = '_cropped_dc_psc.npy')

        ## find unique session number
        prf_ses_num = np.unique([mri_utils.get_run_ses_from_str(f)[-1] for f in prf_bold_files])

        ## for each session, get split half correlation values
        corr_arr = []
        random_corr_arr = []
        for sn in prf_ses_num:

            ses_files = [f for f in prf_bold_files if 'ses-{s}'.format(s = sn) in f]

            ## split runs in half and get unique combinations
            run_sh_lists = mri_utils.split_half_comb(ses_files)

            # get correlation value for each combination
            for r in run_sh_lists:
                ## correlate the two halfs
                corr_arr.append(mri_utils.correlate_arrs(list(r[0]), list(r[-1]), n_jobs = 8))
                ## correlate with randomized half
                random_corr_arr.append(mri_utils.correlate_arrs(list(r[0]), list(r[-1]), n_jobs = 8, shuffle_axis = -1))

        # average values 
        avg_sh_corr = np.nanmean(corr_arr, axis = 0)
        avg_sh_rand_corr = np.nanmean(random_corr_arr, axis = 0)

        print('95 percentile for pRF runs at %.3f'%np.nanpercentile(avg_sh_rand_corr, 95))

        ## make final mask
        # we want to exclude vertices above threshold
        binary_prf_mask = np.ones(avg_sh_corr.shape)
        binary_prf_mask[avg_sh_corr >= np.nanpercentile(avg_sh_rand_corr, 95)] = 0

        ## now do the same correlation mask for the FA runs ###################

        # get prf bold filenames
        fa_bold_files = self.get_bold_file_list(participant, task = 'FA', ses = 'ses-mean', file_ext = file_ext)

        ## find unique session number
        fa_ses_num = np.unique([mri_utils.get_run_ses_from_str(f)[-1] for f in fa_bold_files])

        ## for each session, get split half correlation values
        corr_arr = []
        random_corr_arr = []
        for sn in fa_ses_num:

            ses_files = [f for f in fa_bold_files if 'ses-{s}'.format(s = sn) in f]

            ## split runs in half and get unique combinations
            run_sh_lists = mri_utils.split_half_comb(ses_files)

            # get correlation value for each combination
            for r in run_sh_lists:
                ## correlate the two halfs
                corr_arr.append(mri_utils.correlate_arrs(list(r[0]), list(r[-1]), n_jobs = 8))
                ## correlate with randomized half
                random_corr_arr.append(mri_utils.correlate_arrs(list(r[0]), list(r[-1]), n_jobs = 8, shuffle_axis = -1))

        # average values 
        fa_avg_sh_corr = np.nanmean(corr_arr, axis = 0)
        fa_avg_sh_rand_corr = np.nanmean(random_corr_arr, axis = 0)

        print('95 percentile for FA runs at %.3f'%np.nanpercentile(fa_avg_sh_rand_corr, 95))

        ## make final mask
        # we want to exclude vertices above threshold
        binary_fa_mask = np.ones(fa_avg_sh_corr.shape)
        binary_fa_mask[fa_avg_sh_corr >= np.nanpercentile(fa_avg_sh_rand_corr, 95)] = 0

        ### final mask is multiplication of the two
        final_mask = binary_fa_mask * binary_prf_mask

        # create a directory for saving GLMsingle outputs
        opt = dict()

        # set important fields for completeness (but these would be enabled by default)
        opt['wantlibrary'] = 0
        opt['wantglmdenoise'] = 1
        opt['wantfracridge'] = 1
        opt['hrfonset'] = 0 #FAM_FA.hrf_onset
        opt['hrftoassume'] = hrf_final
        opt['brainexclude'] = final_mask.astype(int)
        opt['sessionindicator'] = self.ses_num_arr 
        opt['brainthresh'] = [99, 0] # which allows all voxels to pass the intensity threshold --> we use surface data
        opt['brainR2'] = 100 # not using on-off model for noise pool

        # define polynomials to project out from data (we only want to use intercept and slope)
        opt['maxpolydeg'] = [[0, 1] for _ in range(data.shape[0])]
        

        # for the purpose of this example we will keep the relevant outputs in memory
        # and also save them to the disk
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
                                            self.MRIObj.params['FA']['bars_phase_dur'],
                                            self.MRIObj.TR,
                                            outputdir = outdir)
                                            #figuredir=outdir)

        elapsed_time = time.time() - start_time

        print(
            '\telapsed time: ',
            f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
        )


        ## for now plot and save a few inspection figures too,
        # to see get a sense on quality of fit

        ## plot ON OFF R2
        flatmap = cortex.Vertex(results_glmsingle['typea']['onoffR2'], 
                  'hcp_999999',
                   vmin = 0, vmax = 15, #.7,
                   cmap='hot')

        fig_name = op.join(outdir, 'modeltypeA_ONOFF_rsq.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        ## plot ON OFF betas
        flatmap = cortex.Vertex(results_glmsingle['typea']['betasmd'][...,0], 
                  'hcp_999999',
                   vmin = -2, vmax = 2, #.7,
                   cmap='RdBu_r')

        fig_name = op.join(outdir, 'modeltypeA_ONOFF_betas.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)


        ## plot Full Model noise pool 
        flatmap = cortex.Vertex(results_glmsingle['typed']['noisepool'], 
                        'hcp_999999',
                        vmin = 0, vmax = 1, #.7,
                        cmap='hot')

        fig_name = op.join(outdir, 'modeltypeD_noisepool.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        
        ## plot Full Model RSQ
        flatmap = cortex.Vertex(results_glmsingle['typed']['R2'], 
                  'hcp_999999',
                   vmin = 0, vmax = 50, #.7,
                   cmap='hot')
        
        fig_name = op.join(outdir, 'modeltypeD_rsq.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        ## plot Full Model betas
        flatmap = cortex.Vertex(np.mean(results_glmsingle['typed']['betasmd'], axis = -1), 
                  'hcp_999999',
                   vmin = -2, vmax = 2, #.7,
                   cmap='RdBu_r')

        fig_name = op.join(outdir, 'modeltypeD_betas.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)


        ## plot Full Model betas for only high fitting RSQ
        avg_betas = np.mean(results_glmsingle['typed']['betasmd'], axis = -1)
        avg_betas[pp_prf_estimates['r2']< self.prf_rsq_threshold] = np.nan

        flatmap = cortex.Vertex(avg_betas, 
                        'hcp_999999',
                        vmin = -2, vmax = 2, #.7,
                        cmap='RdBu_r')

        fig_name = op.join(outdir, 'modeltypeD_betas_ROIpRF.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)


        ## plot Full Model FracRidge

        flatmap = cortex.Vertex(results_glmsingle['typed']['FRACvalue'], 
                  'hcp_999999',
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
                        'hcp_999999',
                        vmin = 0, vmax = 1, #.7,
                        cmap='hot')
        cortex.quickshow(flatmap, with_curvature=True,with_sulci=True, with_labels=False)

        fig_name = op.join(outdir, 'modeltypeD_pRFmask.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        ## plot FA binary mask
        flatmap = cortex.Vertex(binary_fa_mask, 
                        'hcp_999999',
                        vmin = 0, vmax = 1, #.7,
                        cmap='hot')
        cortex.quickshow(flatmap, with_curvature=True,with_sulci=True, with_labels=False)

        fig_name = op.join(outdir, 'modeltypeD_FAmask.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        ## plot binary mask used for noise pool
        flatmap = cortex.Vertex(final_mask, 
                        'hcp_999999',
                        vmin = 0, vmax = 1, #.7,
                        cmap='hot')
        cortex.quickshow(flatmap, with_curvature=True,with_sulci=True, with_labels=False)

        fig_name = op.join(outdir, 'modeltypeD_multiplicationmask.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

        ## plot beta standard deviation, to see how much they vary
        _, std_surf = self.get_singletrial_avg_estimates(estimate_arr = results_glmsingle['typed']['betasmd'], 
                                                        single_trl_DM = single_trl_DM, return_std = True)

        flatmap = cortex.Vertex(np.mean(std_surf, axis = 0), 
                        'hcp_999999',
                        vmin = 0, vmax = 2, #.7,
                        cmap='gnuplot')

        fig_name = op.join(outdir, 'modeltypeD_std_betas.png')
        print('saving %s' %fig_name)
        _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

