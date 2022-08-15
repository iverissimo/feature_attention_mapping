import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
import utils

import ptitprince as pt # raincloud plots
import matplotlib.patches as mpatches
from  matplotlib.ticker import FuncFormatter

from PIL import Image, ImageDraw

import cortex

import subprocess

from FAM.utils import mri as mri_utils
from FAM.processing import preproc_behdata


class pRF_model:

    def __init__(self, MRIObj, outputdir = None):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
            
        """

        ## set data object to use later on
        # has relevant paths etc
        self.MRIObj = MRIObj

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.MRIObj.derivatives_pth,'pRF_fit')
        else:
            self.outputdir = outputdir
            
        ### some relevant params ###

        ## bar width ratio
        self.bar_width = self.MRIObj.params['pRF']['bar_width_ratio'] 

        ## screen resolution in pix
        screen_res = self.MRIObj.params['window']['size']
        if self.MRIObj.params['window']['display'] == 'square': # if square display
            screen_res = np.array([screen_res[1], screen_res[1]])
        self.screen_res = screen_res
        
        ## type of model to fit
        self.model_type = self.MRIObj.params['mri']['fitting']['pRF']['fit_model']

        # if we are fitting HRF params
        self.fit_hrf = self.MRIObj.params['mri']['fitting']['pRF']['fit_hrf']
        
        ## if we're shifting TRs to account for dummy scans
        self.shift_TRs_num =  self.MRIObj.params['mri']['shift_DM_TRs']

        ## if we're cropping TRs
        self.crop_TRs = self.MRIObj.params['pRF']['crop'] 
        self.crop_TRs_num =  self.MRIObj.params['pRF']['crop_TR']

        ## if we did slicetime correction
        self.stc = self.MRIObj.params['mri']['slicetimecorrection']

        ## if we want to keep the model baseline fixed a 0
        self.fix_bold_baseline = self.MRIObj.params['mri']['fitting']['pRF']['fix_bold_baseline'] 

        ## if we want to correct bold baseline of data
        self.correct_baseline = self.MRIObj.params['mri']['fitting']['pRF']['correct_baseline'] 
        # number of TRs to use for correction
        self.corr_base_TRs = self.MRIObj.params['mri']['fitting']['pRF']['num_baseline_TRs'] 


    # def get_data4fitting(self, input_pth = None, run_type = 'mean',
    #                         chunk_num = None, vertex = None, ROI = None):


    def get_DM(self, participant, ses = None, ses_type = 'func', mask_DM = True, filename = None, 
                                    osf = 1, res_scaling = .1):

        """
        Get pRF Design matrix

        Parameters
        ----------
        participant : str
            participant number
        ses : str
            session number (default ses-1)
        ses_type: str
            type of session (default func)

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
                if ses:
                    mask_bool = mask_bool_df[(mask_bool_df['ses'] == ses) & \
                                        (mask_bool_df['sj'] == 'sub-{sj}'.format(sj = participant))]['mask_bool'].values
                else:
                    mask_bool = mask_bool_df[mask_bool_df['sj'] == 'sub-{sj}'.format(sj = participant)]['mask_bool'].values
                dm_mask = np.prod(mask_bool, axis = 0)
            else:
                dm_mask = np.ones(mri_beh.pRF_total_trials)

            # multiply boolean array with mask
            stim_on_screen = stim_on_screen * dm_mask
                
            ## crop and shift if such was the case
            stim_on_screen = mri_utils.crop_shift_arr(stim_on_screen, 
                                                        crop_nr = self.crop_TRs_num, 
                                                        shift = self.shift_TRs_num)
            # do same to bar pass direction str array
            condition_per_TR = mri_utils.crop_shift_arr(mri_beh.pRF_bar_pass_all, 
                                                        crop_nr = self.crop_TRs_num, 
                                                        shift = self.shift_TRs_num)

            # all possible positions in pixels for for midpoint of
            # y position for vertical bar passes, 
            ver_y = self.screen_res[1]*np.linspace(0,1, self.MRIObj.pRF_nr_TRs['U-D'])
            # x position for horizontal bar passes 
            hor_x = self.screen_res[0]*np.linspace(0,1, self.MRIObj.pRF_nr_TRs['L-R'])

            # coordenates for bar pass, for PIL Image
            coordenates_bars = {'L-R': {'upLx': hor_x - 0.5 * self.bar_width * self.screen_res[0], 'upLy': np.repeat(self.screen_res[1], self.MRIObj.pRF_nr_TRs['L-R']),
                                        'lowRx': hor_x + 0.5 * self.bar_width * self.screen_res[0], 'lowRy': np.repeat(0, self.MRIObj.pRF_nr_TRs['L-R'])},
                                'R-L': {'upLx': np.array(list(reversed(hor_x - 0.5 * self.bar_width * self.screen_res[0]))), 'upLy': np.repeat(self.screen_res[1], self.MRIObj.pRF_nr_TRs['R-L']),
                                        'lowRx': np.array(list(reversed(hor_x+ 0.5 * self.bar_width * self.screen_res[0]))), 'lowRy': np.repeat(0, self.MRIObj.pRF_nr_TRs['R-L'])},
                                'U-D': {'upLx': np.repeat(0, self.MRIObj.pRF_nr_TRs['U-D']), 'upLy': ver_y+0.5 * self.bar_width * self.screen_res[1],
                                        'lowRx': np.repeat(self.screen_res[0], self.MRIObj.pRF_nr_TRs['U-D']), 'lowRy': ver_y - 0.5 * self.bar_width * self.screen_res[1]},
                                'D-U': {'upLx': np.repeat(0, self.MRIObj.pRF_nr_TRs['D-U']), 'upLy': np.array(list(reversed(ver_y + 0.5 * self.bar_width * self.screen_res[1]))),
                                        'lowRx': np.repeat(self.screen_res[0], self.MRIObj.pRF_nr_TRs['D-U']), 'lowRy': np.array(list(reversed(ver_y - 0.5 * self.bar_width * self.screen_res[1])))}
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



    def set_models(self,  participant_list = [], input_pth = None,
                            run_type = 'mean', file_ext = '_cropped_dc_psc.npy', mask_DM = True, combine_ses = True):

        """
        define pRF models to be used for each participant in participant list
                
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
        combine_ses: bool
            if we want to combine runs from different sessions (relevant for fitting of average across runs)
        """                 

        ## input path, if not defined get's it from post-fmriprep dir
        if input_pth is None:
            input_pth = op.join(self.MRIObj.derivatives_pth, 'post_fmriprep', self.MRIObj.sj_space)

        ## loop over participants

        ## if no participant list set, then run all
        if len(participant_list) == 0:
            participant_list = self.MRIObj.sj_num
        
        for pp in participant_list:

            ## and over sessions (if more than one)
            for ses in self.MRIObj.session['sub-{sj}'.format(sj=pp)]:


                # path to post fmriprep dir
                postfmriprep_pth = op.join(input_pth, 'sub-{sj}'.format(sj=pp), ses)

                ## bold filenames
                bold_files = [op.join(postfmriprep_pth, run) for run in os.listdir(postfmriprep_pth) if 'space-{sp}'.format(sp=self.MRIObj.sj_space) in run \
                                    and 'acq-{a}'.format(a=self.MRIObj.acq) in run and \
                              'task-{tsk}'.format(tsk=task) in run and run.endswith(file_ext[task])]