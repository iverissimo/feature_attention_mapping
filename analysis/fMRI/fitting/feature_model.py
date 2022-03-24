import numpy as np
import os
import os.path as op

from FAM_utils import mri as mri_utils

import pandas as pd
import numpy as np


# requires pfpy to be installed - preferably with python setup.py develop
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D

from joblib import Parallel, delayed

from tqdm import tqdm

from lmfit import minimize, Parameters


class FA_model:
    def __init__(self, exp_params):
        
        self.exp_params = exp_params
        
        ## general settings
        
        # screen resolution
        self.screen_res = exp_params['window']['size']
        if exp_params['window']['display'] == 'square': # if square display
            self.screen_res = np.array([self.screen_res[1], self.screen_res[1]])
            
        # screen info
        self.screen_size_cm = exp_params['monitor']['height']
        self.screen_distance_cm = exp_params['monitor']['distance']
            
        # mri TR
        self.TR = exp_params['mri']['TR']
        # scaling factor for DM spatial resolution
        self.res_scaling = 0.1
        # if we want to shift TRs
        self.shift_TRs = True


        ## pRF analysis settings
        
        self.prf_model_type = exp_params['mri']['fitting']['pRF']['fit_model']
        self.prf_run_type = exp_params['mri']['fitting']['pRF']['run']
        
        self.fit_hrf = exp_params['mri']['fitting']['pRF']['fit_hrf'] 
        self.prf_crop = exp_params['prf']['crop']
        self.prf_crop_TRs = exp_params['prf']['crop_TR']
        self.prf_shift_TR_num = 1
        
        ## FA settings
        
        # task sampling rate
        self.FA_sampling_rate = self.TR if exp_params['feature']['task_rate']=='TR' else exp_params['feature']['task_rate']
        # oversampling factor
        self.osf = 10 
        # bar condition keys
        self.unique_cond = exp_params['mri']['fitting']['FA']['condition_keys']
        # number miniblock
        self.num_mblk = exp_params['feature']['mini_blocks']
        # duration of bars on screen
        self.stim_dur_seconds = exp_params['feature']['bars_phase_dur']
        
        
        self.fa_crop = exp_params['feature']['crop']
        self.fa_crop_TRs = exp_params['feature']['crop_TR']
        self.fa_shift_TR_num = 1.5
        self.fa_shift_TRs = True

        # if we include nuisance reg
        self.use_nuisance_reg = True
    
        

    def get_pRF_estimates(self, prf_path, nr_chunks):
        
        # path to combined estimates
        prf_estimates_pth = op.join(prf_path,'combined')

        # combined estimates filename
        est_name = [x for _,x in enumerate(os.listdir(prf_path)) if 'chunk-001' in x]
        if len(est_name)>1:
            raise ValueError('%s files found as pRF estimates of same chuck, unsure of which to use'%len(est_name))
        else:
            est_name = est_name[0].replace('chunk-001_of_{ch}'.format(ch=str(nr_chunks).zfill(3)),'chunk-combined')
        
        # total path to estimates path
        prf_estimates_combi = op.join(prf_estimates_pth, est_name)

        if op.isfile(prf_estimates_combi): # if combined estimates exists

            print('loading %s'%prf_estimates_combi)
            self.pRF_estimates = np.load(prf_estimates_combi) # load it

        else: # if not join chunks and save file
            if not op.exists(prf_estimates_pth):
                os.makedirs(prf_estimates_pth) 

            self.pRF_estimates = mri_utils.join_chunks(prf_path, prf_estimates_combi, 
                                                  fit_hrf = self.fit_hrf, chunk_num = nr_chunks, 
                                                  fit_model = 'it{model}'.format(model = self.prf_model_type)) #'{model}'.format(model=model_type)))#
            
        return self.pRF_estimates
            
        
    def mask_pRF_estimates(self, prf_root_path, prf_dm_mask):
                
        
        # define design matrix for pRF task
        self.prf_dm = mri_utils.make_pRF_DM(op.join(prf_root_path, 'DMprf.npy'), self.exp_params, 
                            save_imgs = False, res_scaling = self.res_scaling, 
                            crop =  self.prf_crop, crop_TR = self.prf_crop_TRs,
                            shift_TRs = self.shift_TRs, shift_TR_num = self.prf_shift_TR_num, 
                            overwrite = True, mask = prf_dm_mask)

        # make stimulus object, which takes an input design matrix and sets up its real-world dimensions
        prf_stim = PRFStimulus2D(screen_size_cm = self.screen_size_cm,
                                screen_distance_cm = self.screen_distance_cm,
                                design_matrix = self.prf_dm,
                                TR = self.TR)
        
        # get the ecc limits (in dva)
        # to mask estimates
        self.x_ecc_lim, self.y_ecc_lim = mri_utils.get_ecc_limits(self.prf_dm, self.exp_params,
                                                                  screen_size_deg = [prf_stim.screen_size_degrees,
                                                                                     prf_stim.screen_size_degrees])

        # also compute limit in pixels
        # to make spatial position mask for FA DM   
        self.xy_lim_pix = {'x_lim': self.x_ecc_lim * self.screen_res[0] / prf_stim.screen_size_degrees,
                          'y_lim': self.y_ecc_lim * self.screen_res[1] / prf_stim.screen_size_degrees}

        # mask estimates, to be within screen boundaries
        print('masking estimates')
        pRF_estimates_masked = mri_utils.mask_estimates(self.pRF_estimates, 
                                                             fit_model = self.prf_model_type,
                                                             x_ecc_lim = self.x_ecc_lim, 
                                                             y_ecc_lim = self.y_ecc_lim)            
        
        return pRF_estimates_masked 
    
    
    def get_conditions_dataframe(self, bar_pos, trial_info):
        
        if isinstance(self.bar_pos, list) or isinstance(self.trial_info, list):
            print("list of bar positions and/or trial info provided")
        else:
            bar_pos = [bar_pos]
            trial_info = [trial_info]

        self.bar_pos = bar_pos
        self.trial_info = trial_info
            
        # loop over list of runs
        all_conditions = []
        
        for i in range(len(bar_pos)):
            
            ## get info on conditions in run (4 conditions x 4 miniblocks = 16)
            df = pd.DataFrame(columns = ['reg_name', 'color','orientation','miniblock', 'condition_name'])

            for key in self.unique_cond.keys(): # for each condition

                for blk in range(self.num_mblk): # for each miniblock

                    # name of attended condition in miniblock
                    attended_cond = bar_pos[i].loc[(bar_pos[i]['mini_block']==blk)&(bar_pos[i]['attend_condition']==1)]['condition'].values[0]

                    # append "regressor" info in dataframe
                    df = df.append(pd.DataFrame({'reg_name': '{cond}_mblk-{blk}'.format(cond = key, blk = blk),
                                                                         'color': self.unique_cond[key]['color'],
                                                                         'orientation': self.unique_cond[key]['orientation'],
                                                                         'condition_name': mri_utils.get_cond_name(attended_cond, key),
                                                                         'miniblock': blk
                                                                        }, index=[0]),ignore_index=True)
            all_conditions.append(df)

        return all_conditions if len(bar_pos)>1 else all_conditions[0]
     
    
    def make_FA_visual_DM(self, regs, crop = False, shift_TRs = False,
                          crop_unit = 'sec', oversampling_time = None, **kwargs):
        
        
        ## get all condition dataframe, with relevant info
        all_conditions = self.get_conditions_dataframe(self.bar_pos, self.trial_info)
        
        if isinstance(all_conditions, list):
            print("list of condition dataframe provided")
        else:
            all_conditions = [all_conditions]
            
        # loop over list of runs (if such is the case)
        all_regs_dict = []
        tmp_visual_DM = []
        
        for i in range(len(all_conditions)):
            
            tmp_dict = {} 
        
            for reg in regs: # for each regressor we want to include in final design matrix

                # doign this because we might want to group "regressors"
                r_list = [val for _,val in enumerate(all_conditions[i]['reg_name'].values) if reg in val]

                all_r_dm = []

                for r in r_list:

                    # make array with spatial position of bar of interest 
                    r_dm =  mri_utils.get_FA_bar_stim(self.bar_pos[i], self.trial_info[i], 
                                                      TR = self.TR,
                                    attend_cond = all_conditions[i][all_conditions[i]['reg_name'] == r].to_dict('r')[0], 
                                    res_scaling = self.res_scaling, oversampling_time = oversampling_time, 
                                    stim_dur_seconds = self.stim_dur_seconds, 
                                    xy_lim_pix = self.xy_lim_pix,
                                    crop = crop, crop_unit = crop_unit, 
                                    crop_TR = self.fa_crop_TRs,
                                    shift_TRs = shift_TRs, shift_TR_num = self.fa_shift_TR_num 
                                    )

                    all_r_dm.append(r_dm)

                # collapse all
                # and append in regressor DM
                tmp_dict[reg] = np.amax(np.array(all_r_dm),axis=0)
                
            all_regs_dict.append(tmp_dict) # store in list of dict, to keep track
            
            ## stack DM in array         
            tmp_visual_DM.append(np.stack((tmp_dict[k].astype(np.float64) for k in regs),axis = 0))
        
        self.FA_visual_DM = np.array(tmp_visual_DM)
        
        return all_regs_dict # return dict
    
    
    def make_FA_DM(self, pars_dict,
               hrf_params = [1,1,0], 
               cue_regressors = {'cue_0': [], 'cue_1': [], 'cue_2': [], 'cue_3': []},
               weight_stim = False,
               nuisance_regressors = [],
               visual_dm = [],
               **kwargs):
    
        # if we want to weight bar stim
        if weight_stim:
            # get weighted visual FA dm   
            weights_arr = np.array([pars_dict['gain_{key}'.format(key = name)] for name in self.unique_cond.keys()]).astype(np.float64)

            # taking the max value of the spatial position at each time point (to account for overlaps)
            dm = mri_utils.weight_dm(visual_dm, weights_arr)
        else:
            dm = visual_dm

        if len(dm.shape) == 3:
            new_dm = dm[np.newaxis,...]
        else:
            new_dm = dm

        # get FA regressor
        bar_stim_regressors = {}

        for i, key in enumerate(self.bar_stim_regressors_keys):
            bar_stim_regressors[key] = mri_utils.get_FA_regressor(new_dm[i], self.exp_params, pars_dict, 
                                                                stim_ind = self.bar_on_screen_ind,
                                                                TR = self.TR, hrf_params = hrf_params, oversampling_time = self.osf,
                                                                crop = self.fa_crop, crop_TR = self.fa_crop_TRs, 
                                                                shift_TRs = self.fa_shift_TRs, shift_TR_num = self.fa_shift_TR_num
                                                                )
        # save len of regressor, to use later
        len_reg = bar_stim_regressors[key].shape[0]

        # set all DM regressor names
        all_regressor_keys = np.concatenate((np.array(['intercept']), 
                                              self.bar_stim_regressors_keys, 
                                              self.cue_regressors_keys))

        # if we're including nuisance regressor
        if len(nuisance_regressors)>0:
            all_regressor_keys = np.concatenate((all_regressor_keys, np.array(['nuisance'])))


        # fill DM array
        FA_design_matrix = np.zeros((len_reg, len(all_regressor_keys)))

        for i, val in enumerate(all_regressor_keys):

            if 'intercept' in val:
                FA_design_matrix[...,i] = 1

            elif val in self.bar_stim_regressors_keys:
                FA_design_matrix[...,i] = bar_stim_regressors[val]

            elif 'nuisance' in val:
                FA_design_matrix[...,i] = nuisance_regressors

            else:
                FA_design_matrix[...,i] = cue_regressors[val]

        return FA_design_matrix, all_regressor_keys
        

class FA_GLM(FA_model):
    
    def __init__(self, 
                 exp_params):
        
        super().__init__(exp_params)
        
    
    def fit(self, data,
        hrf_params = None, mask_ind = [], nr_cue_regs = 4, reg_cond_names = ['ACAO', 'ACUO', 'UCAO', 'UCUO']):

        ## set mask indices to all, if not specified
        if len(mask_ind) == 0:
            mask_ind = np.arange(data.shape[0])


        ## hrf params - should be (3, #vertices)
        self.hrf_params = hrf_params

        ## get cue regressor(s)
        if not hasattr(self, 'cue_regressors'):
            self.cue_regressors = np.stack((mri_utils.get_cue_regressor(self.trial_info, 
                                                        hrf_params = self.hrf_params, cues = [i],
                                                        TR = self.TR, oversampling_time = self.osf, 
                                                        baseline = self.pRF_estimates['baseline'],
                                                        crop_unit = 'sec', crop = self.fa_crop, 
                                                        crop_TR = self.fa_crop_TRs, 
                                                        shift_TRs = self.fa_shift_TRs, 
                                                        shift_TR_num = self.fa_shift_TR_num) for i in range(nr_cue_regs)), axis = 0)


        # save cue regressor names (for bookeeping)
        self.cue_regressors_keys = np.stack(('cue_{num}'.format(num = num) for num in range(nr_cue_regs)), axis = 0)

        # also save FA regressor name
        self.bar_stim_regressors_keys = reg_cond_names

        ## get indices when bar was on screen
        # (useful for upsampling)
        self.bar_on_screen_ind = np.where(np.sum(self.FA_visual_DM, axis=0).reshape(-1, self.FA_visual_DM.shape[-1]).sum(axis=0)>0)[0]


        ## actually fit vertices
        # and output relevant params + rsq of model fit in dataframe

        results = np.array(Parallel(n_jobs=16)(delayed(self.get_glm_params)(data[vertex],
                                                                                 hrf_params = self.hrf_params[...,vertex],
                                                                                 rf_params = {'pRF_x': self.pRF_estimates['x'][vertex], 
                                                                                               'pRF_y': self.pRF_estimates['y'][vertex], 
                                                                                               'pRF_beta': self.pRF_estimates['beta'][vertex], 
                                                                                               'pRF_size': self.pRF_estimates['size'][vertex], 
                                                                                               'pRF_baseline': self.pRF_estimates['baseline'][vertex], 
                                                                                               'pRF_n': self.pRF_estimates['ns'][vertex]},
                                                                                 cue_regressors = {'cue_0': self.cue_regressors[0][vertex], 
                                                                                                   'cue_1': self.cue_regressors[1][vertex], 
                                                                                                   'cue_2': self.cue_regressors[2][vertex], 
                                                                                                   'cue_3': self.cue_regressors[3][vertex]})
                                                                       for _,vertex in enumerate(tqdm(mask_ind))))

        return results

    
    def get_glm_params(self, timecourse, 
                           hrf_params = [1,1,0], 
                           rf_params = {'pRF_x': None, 'pRF_y': None, 'pRF_beta': None, 
                                           'pRF_size': None, 'pRF_baseline': None, 'pRF_n': None},
                           cue_regressors = {'cue_0': [], 'cue_1': [], 'cue_2': [], 'cue_3': []}):

        ## set up actual DM that goes into fitting

        ## set up actual DM that goes into fitting
        FA_design_matrix, all_regressor_keys  = self.make_FA_DM(rf_params,
                                                           hrf_params = hrf_params, 
                                                           cue_regressors = cue_regressors,
                                                           weight_stim = False)

        self.all_regressor_keys = all_regressor_keys

        ## Fit GLM on FA data
        prediction, betas , r2, mse = mri_utils.fit_glm(timecourse, FA_design_matrix)

        return prediction, betas , r2, mse

    
        
class FA_GainModel(FA_model):
    
    def __init__(self, 
                 exp_params):
        
        super().__init__(exp_params)
        
        self.all_regressor_keys = np.array(['intercept', 'bar_stim', 'cue'])
        
    
    def iterative_fit(self, data, starting_params, 
                      hrf_params = None, mask_ind = [], nr_cue_regs = 4, nuisance_regressors = [], 
                    prev_fit_params = []):
        
        ## set mask indices to all, if not specified
        if len(mask_ind) == 0:
            mask_ind = np.arange(data.shape[0])
        
        ## set starting params, also includes bounds and if are varied
        self.starting_params = starting_params

        if len(prev_fit_params) == 0: # if no previously fitted params
            prev_fit_params = np.full(data.shape[0], None) # make None array
        
        ## hrf params - should be (3, #vertices)
        self.hrf_params = hrf_params
        
        ## get cue regressor(s)
        if not hasattr(self, 'cue_regressors'):
            self.cue_regressors = np.stack((mri_utils.get_cue_regressor(self.trial_info, 
                                                        hrf_params = self.hrf_params, cues = [i],
                                                        TR = self.TR, oversampling_time = self.osf, 
                                                        baseline = self.pRF_estimates['baseline'],
                                                        crop_unit = 'sec', crop = self.fa_crop, 
                                                        crop_TR = self.fa_crop_TRs, 
                                                        shift_TRs = self.fa_shift_TRs, 
                                                        shift_TR_num = self.fa_shift_TR_num) for i in range(nr_cue_regs)), axis = 0)

        
        # save cue regressor names (for bookeeping)
        self.cue_regressors_keys = np.stack(('cue_{num}'.format(num = num) for num in range(nr_cue_regs)), axis = 0)
        
        # also save FA regressor name
        self.bar_stim_regressors_keys = np.array(['bar_stim'])
        
        ## get indices when bar was on screen
        # (useful for upsampling)
        self.bar_on_screen_ind = np.where(np.sum(self.FA_visual_DM, axis=0).reshape(-1, self.FA_visual_DM.shape[-1]).sum(axis=0)>0)[0]

        
        ## if using nuisance regressor
        if self.use_nuisance_reg:
            self.nuisance_regressors = nuisance_regressors

        ## actually fit vertices
        # and output relevant params + rsq of model fit in dataframe
        
        results = np.array(Parallel(n_jobs=16)(delayed(self.get_iterative_params)(data[vertex], self.starting_params, 
                                                                                 hrf_params = self.hrf_params[...,vertex],
                                                                                 set_params = {'pRF_x': self.pRF_estimates['x'][vertex], 
                                                                                               'pRF_y': self.pRF_estimates['y'][vertex], 
                                                                                               'pRF_beta': self.pRF_estimates['beta'][vertex], 
                                                                                               'pRF_size': self.pRF_estimates['size'][vertex], 
                                                                                               'pRF_baseline': self.pRF_estimates['baseline'][vertex], 
                                                                                               'pRF_n': self.pRF_estimates['ns'][vertex]},
                                                                                 cue_regressors = {'cue_0': self.cue_regressors[0][vertex], 
                                                                                                   'cue_1': self.cue_regressors[1][vertex], 
                                                                                                   'cue_2': self.cue_regressors[2][vertex], 
                                                                                                   'cue_3': self.cue_regressors[3][vertex]},
                                                                                 nuisance_regressors = self.nuisance_regressors[vertex],
                                                                                prev_fit_params = prev_fit_params[ind])
                                                                       for ind, vertex in enumerate(tqdm(mask_ind))))
            
        ## save fitted params list of dicts as Dataframe
        fitted_params_df = pd.DataFrame(d for d in results)
        
        # and add vertex number for bookeeping
        fitted_params_df['vertex'] = mask_ind

        
        return fitted_params_df
    
    
    def get_iterative_params(self, timecourse, starting_params, 
                             hrf_params = [1,1,0],
                             set_params = {'pRF_x': None, 'pRF_y': None, 'pRF_beta': None, 
                                           'pRF_size': None, 'pRF_baseline': None, 'pRF_n': None},
                             cue_regressors = {'cue_0': [], 'cue_1': [], 'cue_2': [], 'cue_3': []},
                             nuisance_regressors = [], prev_fit_params = None,
                             **kwargs):
        
        ## set parameters 
            
        # if previous params provided, override starting params
        if prev_fit_params is not None:
            for key in starting_params.keys():
                starting_params[key].set(prev_fit_params[key])

        else: # just set params that are vertex specific
            for key in set_params.keys():
                starting_params[key].set(set_params[key])
            
        
        ## minimize residuals
        out = minimize(self.get_gain_residuals, starting_params, args = [timecourse],
                       kws={'hrf_params': hrf_params, 'cue_regressors': cue_regressors, 'nuisance_regressors':nuisance_regressors}, 
                       method = 'lbfgsb')
        
        # return best fitting params
        return out.params.valuesdict()
        
        
        
    def get_gain_residuals(self, fit_pars, timecourse, 
                           hrf_params = [1,1,0], 
                           cue_regressors = {'cue_0': [], 'cue_1': [], 'cue_2': [], 'cue_3': []}, nuisance_regressors = []):
        
        
        if len(timecourse.shape) < 2: ## reshape data if needed, so everything is coherent 
            timecourse = timecourse[np.newaxis,...]

        # turn Parameters into dict (if not already), for simplification
        if type(fit_pars) is dict:
            fit_pars_dict = fit_pars
        else:
            fit_pars_dict = fit_pars.valuesdict()

        ## loop over concatenated data 
        ## (also fine for single timecourse, with shape (1, #TRs))
        resid = []

        for r in range(timecourse.shape[0]):

            # convert params key names
            run_keys = np.array([k for k in fit_pars_dict.keys() if k.endswith('_i{r}'.format(r = r))])
            if len(run_keys) == 0: # in case we use this out of fitting procedure
                run_keys = list(fit_pars_dict.keys()) 
            run_pars_dict = {key.replace('_i{r}'.format(r = r), ''): value  for key, value in fit_pars_dict.items() if key in run_keys}

            ## set up actual DM that goes into fitting
            FA_design_matrix, all_regressor_keys  = self.make_FA_DM(run_pars_dict,
                                                               hrf_params = hrf_params, 
                                                               cue_regressors = cue_regressors,
                                                               nuisance_regressors = nuisance_regressors,
                                                               visual_dm = self.FA_visual_DM[r],
                                                               weight_stim = True)

            ## Fit GLM on FA data
            prediction, betas , r2, _ = mri_utils.fit_glm(timecourse[r], FA_design_matrix)

            # update values obtained by GLM
            if type(fit_pars) is not dict and type(fit_pars) is not pd.DataFrame: # if input params was Parameters object

                for i, val in enumerate(all_regressor_keys):

                    if 'intercept' in val:
                        fit_pars[[s for s in run_keys if 'intercept' in s][0]].set(value = betas[i])
                    else:
                        fit_pars[[s for s in run_keys if 'beta_{k}'.format(k = val) in s][0]].set(value = betas[i])

                # also save rsq for quick check
                fit_pars[[s for s in run_keys if 'rsq' in s][0]].set(value = r2)
        
            ## append residuals
            resid.append(timecourse[r] - prediction)
            print(r2)
        
        # return error "timecourse", that will be used by minimize
        return np.array(resid).flatten()
    
    
    def get_grid_params(self, timecourse, starting_params, 
                         hrf_params = [1,1,0],
                         set_params = {'pRF_x': None, 'pRF_y': None, 'pRF_beta': None, 
                                       'pRF_size': None, 'pRF_baseline': None, 'pRF_n': None},
                         cue_regressors = {'cue_0': [], 'cue_1': [], 'cue_2': [], 'cue_3': []},
                         nuisance_regressors = [], constrained_vars = ['gain_ACAO', 'gain_ACUO', 'gain_UCAO', 'gain_UCUO']):
        
        ## reshape data if needed, so everything is coherent 
        if len(timecourse.shape) < 2:
            timecourse = timecourse[np.newaxis,...]
        
        ## set parameters 
        fit_params = Parameters() # initialize to make substitution easier

        for r in range(timecourse.shape[0]):
            
            # dict with values for the run
            run_dict = starting_params[r].valuesdict().copy()
            
            # original keys used
            orig_keys = list(run_dict.keys())
            
            # new keys for multiple data fitting (as lmfit needs)
            new_keys = np.array([key if '{key}_i{r}'.format(key = key, r = r) in key else '{key}_i{r}'.format(key = key, r = r) for key in orig_keys])
            
            # fill new params with parameter objects
            for i, key in enumerate(new_keys):
                
                fit_params.add(key,
                                value = starting_params[r][orig_keys[i]].value,
                                vary = starting_params[r][orig_keys[i]].vary, 
                                min = starting_params[r][orig_keys[i]].min, 
                                max = starting_params[r][orig_keys[i]].max, 
                                brute_step = starting_params[r][orig_keys[i]].brute_step)

            # set those that are vertex specific)
            for rep_key in set_params.keys():
                fit_params['{key}_i{r}'.format(key = rep_key, r = r)].set(value = set_params[rep_key]) 

        ## constrain gain values to be the same for all runs, if more than 1 run provided 
        if  timecourse.shape[0]>1: 
            for r in range(timecourse.shape[0]-1):
                for c in constrained_vars:
                    fit_params['{c}_i{r}'.format(c = c, r = r+1)].expr = '{c}_i0'.format(c = c)

        
        ## minimize residuals
        out = minimize(self.get_gain_residuals, fit_params, args = [timecourse],
                       kws={'hrf_params': hrf_params, 'cue_regressors': cue_regressors, 'nuisance_regressors': nuisance_regressors}, 
                       method = 'brute', workers = 16) #, Ns = 8)
        
        # return best fitting params
        return out#.params.valuesdict()
    
    
    def grid_fit(self, data, starting_params, 
                      hrf_params = None, mask_ind = [], nr_cue_regs = 4, nuisance_regressors = []):
        
        ## reshape data if needed, so everything is coherent 
        if len(data.shape) < 3:
            data = data[np.newaxis,...]
        
        ## set mask indices to all, if not specified
        if len(mask_ind) == 0:
            mask_ind = np.arange(data.shape[1])
        
        ## set starting params, also includes bounds and if are varied
        if isinstance(starting_params, list):
            print("list of parameters provided")
            self.starting_params = starting_params 
        else:
            self.starting_params = [starting_params]
        
        ## hrf params - should be (3, #vertices)
        self.hrf_params = hrf_params
        
        ## get cue regressor(s)
        if not hasattr(self, 'cue_regressors'):
            cue_regressors = np.stack((mri_utils.get_cue_regressor(self.trial_info[0], 
                                                        hrf_params = self.hrf_params[..., mask_ind], cues = [i],
                                                        TR = self.TR, oversampling_time = self.osf, 
                                                        baseline = self.pRF_estimates['baseline'][mask_ind],
                                                        crop_unit = 'sec', crop = self.fa_crop, 
                                                        crop_TR = self.fa_crop_TRs, 
                                                        shift_TRs = self.fa_shift_TRs, 
                                                        shift_TR_num = self.fa_shift_TR_num) for i in range(nr_cue_regs)), axis = 0)
            
            # to account for whole surface
            self.cue_regressors = np.zeros((nr_cue_regs, self.hrf_params.shape[-1], cue_regressors.shape[-1]))
            self.cue_regressors[:, mask_ind, :] = cue_regressors
        
        # save cue regressor names (for bookeeping)
        self.cue_regressors_keys = np.stack(('cue_{num}'.format(num = num) for num in range(nr_cue_regs)), axis = 0)
        
        # also save FA regressor name
        self.bar_stim_regressors_keys = np.array(['bar_stim'])
        
        ## get indices when bar was on screen
        # (useful for upsampling)
        self.bar_on_screen_ind = np.where(np.sum(self.FA_visual_DM[0], axis=0).reshape(-1, self.FA_visual_DM[0].shape[-1]).sum(axis=0)>0)[0]

        ## if using nuisance regressor
        if self.use_nuisance_reg:
            self.nuisance_regressors = nuisance_regressors
 
        ## actually fit vertices
        # and output relevant params + rsq of model fit in dataframe
        results = np.array(Parallel(n_jobs=16)(delayed(self.get_grid_params)(data[:,vertex,:], self.starting_params, 
                                                                                 hrf_params = self.hrf_params[...,vertex],
                                                                                 set_params = {'pRF_x': self.pRF_estimates['x'][vertex], 
                                                                                               'pRF_y': self.pRF_estimates['y'][vertex], 
                                                                                               'pRF_beta': self.pRF_estimates['beta'][vertex], 
                                                                                               'pRF_size': self.pRF_estimates['size'][vertex], 
                                                                                               'pRF_baseline': self.pRF_estimates['baseline'][vertex], 
                                                                                               'pRF_n': self.pRF_estimates['ns'][vertex]},
                                                                                 cue_regressors = {'cue_0': self.cue_regressors[0][vertex], 
                                                                                                   'cue_1': self.cue_regressors[1][vertex], 
                                                                                                   'cue_2': self.cue_regressors[2][vertex], 
                                                                                                   'cue_3': self.cue_regressors[3][vertex]},
                                                                                 nuisance_regressors = self.nuisance_regressors[vertex])
                                                                       for _,vertex in enumerate(tqdm(mask_ind))))
        
        # # output list of fitted params dataframe
        # fitted_params_list = []
        
        # for r in range(len(self.starting_params)):
        #     ## save fitted params list of dicts as Dataframe
        #     fitted_params_df = pd.DataFrame(d for d in results[r])

        #     # and add vertex number for bookeeping
        #     fitted_params_df['vertex'] = mask_ind
            
        #     fitted_params_list.append(fitted_params_df)

        return results #fitted_params_list
        