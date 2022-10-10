import numpy as np
import re
import os
import os.path as op
import pandas as pd
import yaml
import glob


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
                                            baseline_interval = 'empty', ses = ses, return_filenames = True)

        #print('Loading %s'%file)
        #run_num, ses_num = mri_utils.get_run_ses_from_str(file)
        
        
        return data

        




