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

from FAM.utils import mri as mri_utils


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


    def check_fs_seg(self, participant_list=[], input_pth = None, check_type = 'view', use_T2=False):

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


    def compare_nordic2standard(self, participant_list = [], input_pth = None, file_ext = {'pRF': '_cropped_dc_psc.npy', 'FA': '_cropped_confound_psc.npy'},
                                use_atlas_rois = True, acq_keys = ['standard', 'nordic']):

        """
        Make nordic vs standard comparison plots
        
        NOTE - expects that we already ran postfmriprep, for both 
        
        Parameters
        ----------
        input_pth: str
            path to look for files, if None then will get them from derivatives/freesurfer/sub-X folder
        output_pth: str
            path to save original files, if None then will save them in derivatives/check_segmentations/sub-X folder
        check_type : str
            if viewing or making movie (view, movie) 
        """ 

        ## output path to save plots
        output_pth = op.join(self.MRIObj.derivatives_pth, 'nordic_comparison')

        ## input path, if not defined get's it from post-fmriprep dir
        if input_pth is None:
            input_pth = op.join(self.MRIObj.derivatives_pth, 'post_fmriprep', self.MRIObj.sj_space)

        ## get vertices for each relevant ROI
        # from glasser atlas
        ROIs, roi_verts, color_codes = mri_utils.get_rois4plotting(self.MRIObj.params, 
                                                                pysub = self.MRIObj.params['plotting']['pycortex_sub'], 
                                                                use_atlas = use_atlas_rois, 
                                                                atlas_pth = op.join(self.MRIObj.derivatives_pth,
                                                                                    'glasser_atlas','59k_mesh'), 
                                                                space = self.MRIObj.sj_space)

        ## empty dataframe to save mean values per run
        corr_df = pd.DataFrame({'sj': [], 'ses': [], 'task': [], 'acq': [], 'ROI': [], 'mean_r': [], 'Wmean_r': []})
        tsnr_df = pd.DataFrame({'sj': [], 'ses': [], 'task': [], 'acq': [], 'ROI': [], 'mean_tsnr': []})

        ## loop over participants
        for pp in self.MRIObj.sj_num:

            # and over sessions (if more than one)
            for ses in self.MRIObj.session['sub-{sj}'.format(sj=pp)]:
                
                bold_files = {}
                
                # path to post fmriprep dir
                postfmriprep_pth = op.join(input_pth, 'sub-{sj}'.format(sj=pp), ses)

                outdir = op.join(output_pth,'sub-{sj}'.format(sj=pp), ses)
                # if output path doesn't exist, create it
                if not op.isdir(outdir): 
                    os.makedirs(outdir)
                print('saving files in %s'%outdir)
                
                # and acquisition types
                for acq in acq_keys:
                    
                    ## load data for both tasks
                    for tsk in self.MRIObj.tasks:

                        ## bold filenames
                        bold_files[tsk] = [op.join(postfmriprep_pth, run) for run in os.listdir(postfmriprep_pth) if 'space-{sp}'.format(sp=self.MRIObj.sj_space) in run \
                                            and 'acq-{a}'.format(a=acq) in run and 'task-{t}'.format(t=tsk) in run and run.endswith(file_ext[tsk])]

                        ## calculate tSNR for each run
                        for ind,r in enumerate(bold_files[tsk]):
                            
                            ## use non-PSC file to calculate tSNR
                            if 'cropped' in file_ext[tsk]:
                                r = r.replace(file_ext[tsk], '_cropped.npy')
                            else:
                                r = r.replace(file_ext[tsk], '.npy')
                            
                            ## stack whole brain tsnr - will be used to weight correlations
                            if ind == 0:
                                tsnr_arr = np.mean(np.load(r),axis=-1)/np.std(np.load(r),axis=-1)
                            else:
                                tsnr_arr = np.vstack((tsnr_arr, np.mean(np.load(r),axis=-1)/np.std(np.load(r),axis=-1)))

                            tsnr_df = pd.concat((tsnr_df, 
                                                pd.DataFrame({'sj': np.tile(pp, len(ROIs)), 
                                                            'ses': np.tile(ses, len(ROIs)), 
                                                            'task': np.tile(tsk, len(ROIs)), 
                                                            'acq': np.tile(acq, len(ROIs)), 
                                                            'ROI': ROIs, 
                                                            'mean_tsnr': [np.nanmean((np.mean(np.load(r),axis=-1)/np.std(np.load(r),axis=-1))[roi_verts[roi_name]]) for roi_name in ROIs]})
                                                ))

                        ## split runs in half and get unique combinations
                        run_sh_lists = mri_utils.split_half_comb(bold_files[tsk])
                        
                        # for each combination
                        for r in run_sh_lists:
                            ## correlate the two halfs
                            correlations = mri_utils.correlate_arrs(list(r[0]), list(r[-1]))
                            ## correlate weighting by tSNR (average across runs)
                            #Wcorrelations =  mri_utils.correlate_arrs(list(r[0]), list(r[-1]), weights=np.mean(tsnr_arr, axis = 0))

                            # ## save in dataframe
                            # corr_df = pd.concat((corr_df, 
                            #                     pd.DataFrame({'sj': np.tile(pp, len(ROIs)), 
                            #                                 'ses': np.tile(ses, len(ROIs)), 
                            #                                 'task': np.tile(tsk, len(ROIs)), 
                            #                                 'acq': np.tile(acq, len(ROIs)), 
                            #                                 'ROI': ROIs, 
                            #                                 'mean_r': [np.nanmean(correlations[roi_verts[roi_name]]) for roi_name in ROIs],
                            #                                 'Wmean_r': [np.nanmean(Wcorrelations[roi_verts[roi_name]]) for roi_name in ROIs]})
                            #                     ))


        return tsnr_df, corr_df, tsnr_arr, correlations
                            
                        
