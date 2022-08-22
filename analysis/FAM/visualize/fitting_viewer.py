

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

import cortex

import subprocess

from FAM.utils import mri as mri_utils
from FAM.processing import preproc_behdata
from PIL import Image, ImageDraw

class pRFViewer:

    def __init__(self, MRIObj, outputdir = None, pRFModelObj = None):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
            
        """

        # set data object to use later on
        self.MRIObj = MRIObj

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.MRIObj.derivatives_pth,'plots')
        else:
            self.outputdir = outputdir

        # Load pRF model object
        self.pRFModelObj = pRFModelObj
            
        # number of participants to plot
        self.nr_pp = len(self.MRIObj.sj_num)


    def plot_pRF_DM(self, dm_array, filename):

        """
        Function to plot design matrix frame by frame 
        and save movie in folder

        """

        # if output path doesn't exist, create it

        outfolder = op.split(filename)[0]

        if not op.isdir(outfolder): 
            os.makedirs(outfolder)
        print('saving files in %s'%filename)

        dm_array = (dm_array * 255).astype(np.uint8)

        for w in range(dm_array.shape[-1]):
            im = Image.fromarray(dm_array[...,w])
            im.save(op.join(outfolder,"DM_TR-%s.png"%str(w).zfill(4)))  

        ## save as video
        img_name = op.join(outfolder,'DM_TR-%4d.png')
        os.system("ffmpeg -r 6 -start_number 0 -i %s -vcodec mpeg4 -y %s"%(img_name, filename))    

   
    def plot_singlevert(self, participant, task = 'pRF',
                    ses = 'ses-mean', run_type = 'mean', vertex = None, ROI = None,
                    prf_model_name = 'gauss', file_ext = '_cropped_dc_psc.npy', 
                    fit_now = False, figures_pth = None):

        """

        Function to plot single vertex timecourse

        """
        
        if task == 'pRF':
            
            # make output folder for figures
            if figures_pth is None:
                figures_pth = op.join(self.outputdir, 'single_vertex', 'pRF_fit', 'sub-{sj}'.format(sj = participant), ses)
            
            if not op.exists(figures_pth):
                os.makedirs(figures_pth)

            # get participant models, which also will load 
            # DM and mask it according to participants behavior
            pp_prf_models = self.pRFModelObj.set_models(participant_list = [participant], 
                                                        mask_DM = True, combine_ses = True)


            # if we want to fit it now
            if fit_now:
                estimates_dict, data_arr = self.pRFModelObj.fit_data(participant, pp_prf_models, 
                                                                        vertex = vertex, 
                                                                        run_type = run_type, ses = ses,
                                                                        model2fit = prf_model_name, xtol = 1e-2,
                                                                        file_ext = file_ext)

            else:
                print('Loading estimates')
                raise NameError('Not implemented')


            # if we fitted hrf, need to also get that from params
            # and set model array
            
            # define spm hrf
            spm_hrf = pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].create_hrf(hrf_params = [1, 1, 0],
                                                                                                                     onset=self.pRFModelObj.hrf_onset)

            if self.pRFModelObj.fit_hrf:
                hrf = pp_prf_models[ 'sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].create_hrf(hrf_params = [1.0,
                                                                                                                                 estimates_dict['it_{name}'.format(name = prf_model_name)][0][-3],
                                                                                                                                 estimates_dict['it_{name}'.format(name = prf_model_name)][0][-2]],
                                                                                                                     onset=self.pRFModelObj.hrf_onset)
            
                pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].hrf = hrf

                model_arr = pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].return_prediction(*list(estimates_dict['it_{name}'.format(name = prf_model_name)][0, :-3]))
            
            else:
                pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].hrf = spm_hrf

                model_arr = pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].return_prediction(*list(estimates_dict['it_{name}'.format(name = prf_model_name)][0, :-1]))
            
            
            # get array with name of condition per TR, to plot in background
            ## get behavioral info 
            mri_beh = preproc_behdata.PreprocBeh(self.MRIObj)

            condition_per_TR = mri_utils.crop_shift_arr(mri_beh.pRF_bar_pass_all, 
                                            crop_nr = self.pRFModelObj.crop_TRs_num, 
                                            shift = self.pRFModelObj.shift_TRs_num)

            ## actually plot

            # set figure name
            fig_name = 'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_roi-{roi}_vertex-{vert}.png'.format(sj = participant,
                                                                                                    acq = self.MRIObj.acq,
                                                                                                    space = self.MRIObj.sj_space,
                                                                                                    run = run_type,
                                                                                                    model = prf_model_name,
                                                                                                    roi = str(ROI),
                                                                                                    vert = str(vertex))
            if self.pRFModelObj.fit_hrf:

                fig_name = fig_name.replace('.png','_withHRF.png') 

                ## also plot hrf shapes for comparison
                fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)

                axis.plot(spm_hrf[0],'grey',label='spm hrf')
                axis.plot(hrf[0],'red',label='fitted hrf')
                axis.set_xlim(self.pRFModelObj.hrf_onset, 25)
                axis.legend(loc='upper right',fontsize=10) 
                axis.set_xlabel('Time (s)',fontsize=10, labelpad=10)
                #plt.show()
                fig.savefig(op.join(figures_pth, 'HRF_model-{model}_roi-{roi}_vertex-{vert}.png'.format(model = prf_model_name,
                                                                                                        roi = str(ROI), 
                                                                                                        vert = str(vertex)))) 

            # plot data with model
            fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)

            # plot data with model
            time_sec = np.linspace(0,len(model_arr[0,...]) * self.MRIObj.TR, num = len(model_arr[0,...])) # array in seconds
                
            axis.plot(time_sec, model_arr[0,...], c = 'red', lw = 3, 
                                                label = 'model R$^2$ = %.2f'%estimates_dict['it_{name}'.format(name = prf_model_name)][0][-1], 
                                                zorder = 1)
            #axis.scatter(time_sec, data_reshape[ind_max_rsq,:], marker='v',s=15,c='k',label='data')
            axis.plot(time_sec, data_arr[0,...],'k--',label='data')
            
            axis.set_xlabel('Time (s)',fontsize = 20, labelpad = 5)
            axis.set_ylabel('BOLD signal change (%)',fontsize = 20, labelpad = 5)
            axis.set_xlim(0, len(model_arr[0,...]) * self.MRIObj.TR)
            
            axis.legend(loc='upper left',fontsize = 10) 

            # plot axis vertical bar on background to indicate stimulus display time
            for i,cond in enumerate(condition_per_TR):
    
                if cond in ['L-R','R-L']: # horizontal bar passes will be darker 
                    plt.axvspan(i * self.MRIObj.TR, i * self.MRIObj.TR + self.MRIObj.TR, facecolor = '#8f0000', alpha=0.1)
                    
                elif cond in ['U-D','D-U']: # vertical bar passes will be lighter 
                    plt.axvspan(i * self.MRIObj.TR, i * self.MRIObj.TR + self.MRIObj.TR, facecolor = '#ff0000', alpha=0.1)
                    
            #plt.show()
            fig.savefig(op.join(figures_pth, fig_name))


            
                





            


                


