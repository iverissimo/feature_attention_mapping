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

import subprocess

class MRIViewer:

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


    def check_fs_seg(self, participant_list=[], input_pth = None, check_type = 'view', output_pth = None, use_T2=False):

        """
        Check FREESURFER segmentations
        
        NOTE - needs to be run in local system!!
        NOTE2 - of course we need to already have run freesurfer, successfully 
        
        Parameters
        ----------
        input_pth: str
            path to look for files, if None then will get them from derivatives/freesurfer/sub-X folder
        output_pth: str
            path to save original files, if None then will save them in derivatives/check_segmentations/sub-X folder
        check_type : str
            if viewing or making movie (view, movie) 
        """ 

        ## set input path where fmaps are
        if input_pth is None:
            input_pth = self.MRIObj.freesurfer_pth

        print('freesurfer files in %s'%input_pth)

        ## if no participant list set, then run all
        if len(participant_list) == 0:
            participant_list = self.MRIObj.sj_num

        for pp in participant_list:

            if check_type == 'view':

                batch_string = """#!/bin/bash
conda activate i36

export SUBJECTS_DIR=$DATADIR

cd $DATADIR

freeview -v \
    $FILES \
    sub-$SJ_NR/mri/wm.mgz \
    sub-$SJ_NR/mri/brainmask.mgz \
    sub-$SJ_NR/mri/aseg.mgz:colormap=lut:opacity=0.2 \
    -f \
    sub-$SJ_NR/surf/lh.white:edgecolor=blue \
    sub-$SJ_NR/surf/lh.pial:edgecolor=red \
    sub-$SJ_NR/surf/rh.white:edgecolor=blue \
    sub-$SJ_NR/surf/rh.pial:edgecolor=red
"""
                if use_T2:
                    working_string = batch_string.replace('$FILES', "sub-$SJ_NR/mri/T1.mgz sub-$SJ_NR/mri/T2.mgz")
                else:
                    working_string = batch_string.replace('$FILES', "sub-$SJ_NR/mri/T1.mgz")

                working_string = working_string.replace('$SJ_NR', pp)
                working_string = working_string.replace('$DATADIR', input_pth)

                os.system(working_string)

            elif check_type == 'movie':

                ## set output path where we want to store movie
                if output_pth is None:
                    output_pth = op.join(self.MRIObj.derivatives_pth, 'check_segmentations', 'sub-{sj}'.format(sj=pp))

                if not op.isdir(output_pth):
                    os.makedirs(output_pth)

                batch_string = """#!/bin/bash
    conda activate i36
    export SUBJECTS_DIR=$DATADIR
    cd $DATADIR
    freeview -v \
        sub-$SJ_NR/mri/T1.mgz:grayscale=10,100 \
        -f \
        sub-$SJ_NR/surf/lh.white:edgecolor=blue \
        sub-$SJ_NR/surf/lh.pial:edgecolor=red \
        sub-$SJ_NR/surf/rh.white:edgecolor=blue \
        sub-$SJ_NR/surf/rh.pial:edgecolor=red \
        -viewport sagittal \
        -slice {$XPOS} 127 127 \
        -ss {$OPFN}
    """

                working_string = batch_string.replace('$SJ_NR', pp)
                working_string = working_string.replace('$DATADIR', input_pth)

                # number of slices for saggital view
                sag_slices = range(77,280) #268) #248)

                for slice in sag_slices:
                    if not op.exists(op.join(output_pth, str(slice).zfill(3) + '.png')): # if image already in dir, skip
                        plot_slice = working_string.replace('$XPOS', str(slice).zfill(3))
                        plot_slice = plot_slice.replace('$OPFN', op.join(output_pth, str(slice).zfill(3) + '.png'))

                        os.system(plot_slice)

                subject = 'sub-{sj}'.format(sj=pp)
                convert_command = f'ffmpeg -framerate 5 -pattern_type glob -i "{output_pth}/*.png" -b:v 2M -c:v mpeg4 {output_pth}/{subject}.mp4'
                subprocess.call(convert_command, shell=True)