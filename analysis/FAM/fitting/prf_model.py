import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
import utils
import glob

import ptitprince as pt # raincloud plots
import matplotlib.patches as mpatches
from  matplotlib.ticker import FuncFormatter

from PIL import Image, ImageDraw

import cortex

import subprocess

from FAM.utils import mri as mri_utils
from FAM.processing import preproc_behdata

from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel, Norm_Iso2DGaussianModel

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



    def get_DM(self, participant, ses = 'ses-mean', ses_type = 'func', mask_DM = True, filename = None, 
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
                                        filename = None, osf = 1, res_scaling = .1)

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
                                                    filter_type = self.MRIObj.params['mri']['filtering']['type'],
                                                    filter_params = {'highpass': self.MRIObj.params['mri']['filtering']['highpass'],
                                                                    'add_mean': self.MRIObj.params['mri']['filtering']['add_mean'],
                                                                    'window_length': self.MRIObj.params['mri']['filtering']['window_length'],
                                                                    'polyorder': self.MRIObj.params['mri']['filtering']['polyorder']}
                                                )

                pp_models['sub-{sj}'.format(sj=pp)][ses]['gauss_model'] = gauss_model

                # CSS
                css_model = CSS_Iso2DGaussianModel(stimulus = prf_stim,
                                                    filter_predictions = True,
                                                    filter_type = self.MRIObj.params['mri']['filtering']['type'],
                                                    filter_params = {'highpass': self.MRIObj.params['mri']['filtering']['highpass'],
                                                                    'add_mean': self.MRIObj.params['mri']['filtering']['add_mean'],
                                                                    'window_length': self.MRIObj.params['mri']['filtering']['window_length'],
                                                                    'polyorder': self.MRIObj.params['mri']['filtering']['polyorder']}
                                                )

                pp_models['sub-{sj}'.format(sj=pp)][ses]['css_model'] = css_model

                # DN 
                dn_model =  Norm_Iso2DGaussianModel(stimulus = prf_stim,
                                                    filter_predictions = True,
                                                    filter_type = self.MRIObj.params['mri']['filtering']['type'],
                                                    filter_params = {'highpass': self.MRIObj.params['mri']['filtering']['highpass'],
                                                                    'add_mean': self.MRIObj.params['mri']['filtering']['add_mean'],
                                                                    'window_length': self.MRIObj.params['mri']['filtering']['window_length'],
                                                                    'polyorder': self.MRIObj.params['mri']['filtering']['polyorder']}
                                                )

                pp_models['sub-{sj}'.format(sj=pp)][ses]['dn_model'] = dn_model


        return pp_models


    def fit_data(self, participant, pp_models, ses = 'ses-mean',
                            run_type = 'mean', chunk_num = None, vertex = None, ROI = None,
                            model2fit = None, file_ext = '_cropped_dc_psc.npy', outdir = None, save_estimates = False):

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
        combine_ses: bool
            if we want to combine runs from different sessions (relevant for fitting of average across runs)
        """  

        ## get list of possible input paths
        # (sessions)
        input_list = glob.glob(op.join(self.MRIObj.derivatives_pth, 'post_fmriprep', self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant), 'ses-*'))

        # list with absolute file names to be fitted
        bold_filelist = [op.join(file_path, file) for file_path in input_list for file in os.listdir(file_path) if 'task-pRF' in file and \
                        'acq-{acq}'.format(acq = self.MRIObj.acq) in file and file.endswith(file_ext)]
        
        # if we're not combining sessions
        if ses != 'ses-mean':
            bold_filelist = [file for file in bold_filelist if ses in file]
        
        ## Load data array
        data = self.get_data4fitting(bold_filelist, run_type = run_type, chunk_num = chunk_num, vertex = vertex)

        ## set output dir to save estimates
        if outdir is None:
            outdir = op.join(self.MRIObj.derivatives_pth, 'pRF_fit', self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant), ses)
            
        if not op.exists(outdir):
            os.makedirs(outdir)
        print('saving files in %s'%outdir)

        ## set base filename that will be used for estimates
        basefilename = op.join(outdir, 'sub-{sj}_task-pRF_acq-{acq}_runtype-{rt}'.format(sj = participant,
                                                                            acq = self.MRIObj.acq,
                                                                            rt = run_type))
        if chunk_num:
            basefilename += '_chunk-{ch}'.format(ch = str(chunk_num).zfill(3))
        elif vertex:
            basefilename += '_vertex-{ver}'.format(ver = str(vertex))
        elif ROI:
            basefilename += '_ROI-{roi}'.format(roi = str(ROI))
        
        basefilename += file_ext.replace('.npy', '.npz')

        ## set fitters


        ## now need to mask array for nans
        # set fitters 
        # actually fit
        # save? -- for that need to define filenames somewhere else
        # this func will be called from other one (that will submit batch jobs or just run functions depending on system)



    def get_data4fitting(self, file_list, run_type = 'mean',
                            chunk_num = None, vertex = None):

        """
        load data from file list
                
        Parameters
        ----------
        file_list: list
            list with files to combine into unique data array
        run_type: string or int
            type of run to fit, mean (default), or if int will do single run fit
        chunk_num: int or None
            if we want to fit specific chunk of data, then will return chunk array
        vertex: int, or list of indices or None
            if we want to fit specific vertex of data, or list of vertices (from an ROI for example) then will return vertex array

        """  

        ## Load data array
        # average runs (or loo or get single run)
        if run_type == 'mean':
            print('averaging runs')
            data_arr = np.stack((np.load(arr,allow_pickle=True) for arr in file_list)) # will be (vertex, TR)
            data_arr = np.mean(data_arr, axis = 0)
        elif run_type == 'median':
            print('getting median of runs')
            data_arr = np.stack((np.load(arr,allow_pickle=True) for arr in file_list)) # will be (vertex, TR)
            data_arr = np.median(data_arr, axis = 0)
        elif 'loo_' in run_type:
            print('Leave-one out averaging runs ({r})'.format(r = run_type))
            file_list = [file for file in file_list if 'run-{r}'.format(r = run_type.split('_')[1]) not in file]
            data_arr = np.stack((np.load(arr,allow_pickle=True) for arr in file_list)) # will be (vertex, TR)
            data_arr = np.mean(data_arr, axis = 0)
        elif isinstance(run_type, int):
            print('Loading run-{r}'.format(r = run_type))
            file_list = [file for file in file_list if 'run-{r}'.format(r = run_type) in file]
            data_arr = np.stack((np.load(arr,allow_pickle=True) for arr in file_list)) # will be (vertex, TR)
            data_arr = np.mean(data_arr, axis = 0)
        
        # if we want to chunk it
        if isinstance(chunk_num, int):
            # number of vertices of chunk
            num_vox_chunk = int(data_arr.shape[0]/self.MRIObj.params['mri']['fitting']['pRF']['total_chunks'][self.MRIObj.sj_space])
            print('Slicing data into chunk {ch} of {ch_total}'.format(ch = chunk_num, 
                                        ch_total = self.MRIObj.params['mri']['fitting']['pRF']['total_chunks'][self.MRIObj.sj_space]))
    
            # chunk it
            data_out = data_arr[num_vox_chunk * int(chunk_num):num_vox_chunk * int(chunk_num + 1), :]
        
        # if we want specific vertex
        elif isinstance(vertex, int) or isinstance(vertex, list) or isinstance(vertex, np.ndarray):
            print('Slicing data into vertex {ver}'.format(ver = vertex))
            data_out = data_arr[vertex]
            
            if isinstance(vertex, int):
                data_out = data_out[np.newaxis,...]
        
        # return whole array
        else:
            print('Returning whole data array')
            data_out = data_arr

        ## if we want to keep baseline fix, we need to correct it!
        if self.correct_baseline:
            print('Correcting baseline to be 0 centered')

            ## get behavioral info 
            mri_beh = preproc_behdata.PreprocBeh(self.MRIObj)
            # do same to bar pass direction str array
            condition_per_TR = mri_utils.crop_shift_arr(mri_beh.pRF_bar_pass_all, 
                                                        crop_nr = self.crop_TRs_num, 
                                                        shift = self.shift_TRs_num)

            data_out = mri_utils.baseline_correction(data_out, condition_per_TR, 
                                                    num_baseline_TRs = 6, 
                                                    baseline_interval = 'empty_long', 
                                                    avg_type = 'median')

        return data_out


        