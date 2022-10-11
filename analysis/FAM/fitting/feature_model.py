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


class FA_model(Model):

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

        # # if output dir not defined, then make it in derivatives
        # if outputdir is None:
        #     self.outputdir = op.join(self.MRIObj.derivatives_pth,'FA_fit')
        # else:
        #     self.outputdir = outputdir

    
    def get_bar_dm(self, run_bar_pos_df, attend_bar = True, osf = 10, res_scaling = .1, 
                stim_dur_seconds = 0.5, FA_bar_pass_all = []):
        
        """
        save an array with the FA (un)attended
        bar position for the run

        Parameters
        ----------
        run_bar_pos_df : dataframe
            bar position dataframe for the run
        attend_bar : bool
            if we want position for attended bar or not
        osf: int
            oversampling factor, if we want to oversample in time
        res_scaling: float
            spatial rescaling factor
        stim_dur_seconds: float
            duration of stim (bar presentation) in seconds
        FA_bar_pass_all: list
            list with condition per TR/trial

        """ 
        ## crop and shift if such was the case
        condition_per_TR = mri_utils.crop_shift_arr(FA_bar_pass_all,
                                                crop_nr = self.crop_TRs_num['FA'], 
                                                shift = self.shift_TRs_num)
        
        ## bar midpoint coordinates
        midpoint_bar = run_bar_pos_df[run_bar_pos_df['attend_condition'] == attend_bar].bar_midpoint_at_TR.values[0]
        ## bar direction (vertical vs horizontal)
        direction_bar = run_bar_pos_df[run_bar_pos_df['attend_condition'] == attend_bar].bar_pass_direction_at_TR.values[0]

        # save screen display for each TR (or if osf > 1 then for #TRs * osf)
        visual_dm_array = np.zeros((len(condition_per_TR) * osf, 
                                    round(self.screen_res[0] * res_scaling), 
                                    round(self.screen_res[1] * res_scaling)))
        i = 0

        for trl, bartype in enumerate(condition_per_TR): # loop over bar pass directions

            img = Image.new('RGB', tuple(self.screen_res)) # background image

            if bartype not in np.array(['empty','empty_long']): # if not empty screen

                if direction_bar[i] == 'vertical':
                    coordenates_bars = {'upLx': 0, 
                                        'upLy': self.screen_res[1]/2+midpoint_bar[i][-1]+0.5*self.bar_width['FA']*self.screen_res[1],
                                        'lowRx': self.screen_res[0], 
                                        'lowRy': self.screen_res[1]/2+midpoint_bar[i][-1]-0.5*self.bar_width['FA']*self.screen_res[1]}


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
                visual_dm_array[int(trl*osf):int(trl*osf + osf*stim_dur_seconds), ...] = np.array(img)[::round(1/res_scaling),::round(1/res_scaling),0][np.newaxis,...]
                
            else:
                ## save in array
                visual_dm_array[int(trl*osf):int(trl*osf + osf), ...] = np.array(img)[::round(1/res_scaling),::round(1/res_scaling),0][np.newaxis,...]

        # swap axis to have time in last axis [x,y,t]
        visual_dm = visual_dm_array.transpose([1,2,0])
        
        return mri_utils.normalize(visual_dm)
            
    
    def get_visual_DM_dict(self, participant, filelist, save_overlap = True):
    
        """
        Given participant and filelist of runs to fit,
        will return dict for each run in list,
        with visual DM for each type of regressor (attended bar, unattended bar, overlap etc)
        
        ex:
        out_dict['run-1_ses-1'] = {'att_bar': [x,y,t], 'unatt_bar': [x,y,t], ...}

        Parameters
        ----------
        participant : str
            participant ID
        filelist : list
            list with filenames to fit

        """ 
        
        ## get behavioral info 
        mri_beh = preproc_behdata.PreprocBeh(self.MRIObj)
        
        # set empty dict
        out_dict = {}
        
        ## loop over files
        for file in filelist:
            
            ## get run and ses from file
            run_num, ses_num = mri_utils.get_run_ses_from_str(file) 
            
            out_dict['run-{r}_ses-{s}'.format(r = run_num, s = ses_num)] = {}
            
            ## get bar position df for run
            bar_pos_df = mri_beh.load_FA_bar_position(participant, ses = 'ses-{s}'.format(s = ses_num), 
                                                    ses_type = 'func')
            run_bar_pos_df = bar_pos_df['run-{r}'.format(r = run_num)]
            
            ## GET DM FOR ATTENDED BAR
            out_dict['run-{r}_ses-{s}'.format(r = run_num, 
                                            s = ses_num)]['att_bar'] = self.get_bar_dm(run_bar_pos_df,
                                                                                        attend_bar = True,
                                                                                        osf = self.osf, res_scaling = self.res_scaling,
                                                                                        stim_dur_seconds = self.MRIObj.FA_bars_phase_dur,
                                                                                        FA_bar_pass_all = mri_beh.FA_bar_pass_all)
            ## GET DM FOR UNATTENDED BAR
            out_dict['run-{r}_ses-{s}'.format(r = run_num, 
                                            s = ses_num)]['unatt_bar'] = self.get_bar_dm(run_bar_pos_df,
                                                                                        attend_bar = False,
                                                                                        osf = self.osf, res_scaling = self.res_scaling,
                                                                                        stim_dur_seconds = self.MRIObj.FA_bars_phase_dur,
                                                                                        FA_bar_pass_all = mri_beh.FA_bar_pass_all)

            if save_overlap:
                ## GET DM FOR OVERLAP OF BARS
                out_dict['run-{r}_ses-{s}'.format(r = run_num, 
                                            s = ses_num)]['overlap'] = mri_utils.get_bar_overlap_dm(np.stack((out_dict['run-{r}_ses-{s}'.format(r = run_num, s = ses_num)]['att_bar'],
                                                                                                             out_dict['run-{r}_ses-{s}'.format(r = run_num, s = ses_num)]['unatt_bar'])))
                
        return out_dict


    def fit_data(self, participant, pp_models, ses = 1,
                    run_type = 'loo_r1s1', chunk_num = None, vertex = None, ROI = None,
                    model2fit = 'gauss', file_ext = '_cropped_confound_psc.npy', 
                    outdir = None, save_estimates = False,
                    xtol = 1e-3, ftol = 1e-4, n_jobs = 16):

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

        ## get list of files to load
        bold_filelist = self.get_bold_file_list(participant, task = 'FA', ses = ses, file_ext = file_ext)

        ## Load data array and file list names
        data, train_file_list = self.get_data4fitting(bold_filelist, task = 'FA', run_type = run_type, chunk_num = chunk_num, vertex = vertex, ses = ses,
                                            baseline_interval = 'empty', return_filenames = True)

        #print('Loading %s'%file)
        #run_num, ses_num = mri_utils.get_run_ses_from_str(file)
        
        
        return data, train_file_list

        




