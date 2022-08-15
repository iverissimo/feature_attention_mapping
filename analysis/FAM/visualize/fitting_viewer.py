

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

    def __init__(self, MRIObj, outputdir = None):
        
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
            


