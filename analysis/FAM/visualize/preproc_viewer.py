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
                                use_atlas_rois = True, acq_keys = ['standard', 'nordic'], plot_group=True):

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
        ## also save full corr arrays
        surf_avg_corr = pd.DataFrame({'sj': [], 'ses': [], 'task': [], 'acq': [], 'ROI': [], 'vertex': [], 'pearson_r': []})

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
                        tsnr_arr = []
                        for ind,r in enumerate(bold_files[tsk]):
                            
                            ## use non-PSC file to calculate tSNR
                            if 'cropped' in file_ext[tsk]:
                                r = r.replace(file_ext[tsk], '_cropped.npy')
                            else:
                                r = r.replace(file_ext[tsk], '.npy')
                            
                            ## stack whole brain tsnr - will be used to weight correlations
                            tsnr_arr.append(mri_utils.get_tsnr(np.load(r), return_mean = False))

                            tsnr_df = pd.concat((tsnr_df, 
                                                pd.DataFrame({'sj': np.tile(pp, len(ROIs)), 
                                                            'ses': np.tile(ses, len(ROIs)), 
                                                            'task': np.tile(tsk, len(ROIs)), 
                                                            'acq': np.tile(acq, len(ROIs)), 
                                                            'ROI': ROIs, 
                                                            'mean_tsnr': [np.nanmean(mri_utils.get_tsnr(np.load(r), return_mean = False)[roi_verts[roi_name]]) for roi_name in ROIs]})
                                                ))
                        # make it an array, for simplicity 
                        tsnr_arr = np.array(tsnr_arr)

                        ## split runs in half and get unique combinations
                        run_sh_lists = mri_utils.split_half_comb(bold_files[tsk])
                        
                        # get correlation value for each combination
                        corr_arr = []
                        for r in run_sh_lists:
                            ## correlate the two halfs
                            correlations = mri_utils.correlate_arrs(list(r[0]), list(r[-1]))
                            corr_arr.append(correlations)

                            ## save in dataframe
                            corr_df = pd.concat((corr_df, 
                                                pd.DataFrame({'sj': np.tile(pp, len(ROIs)), 
                                                            'ses': np.tile(ses, len(ROIs)), 
                                                            'task': np.tile(tsk, len(ROIs)), 
                                                            'acq': np.tile(acq, len(ROIs)), 
                                                            'ROI': ROIs, 
                                                            'mean_r': [np.nanmean(correlations[roi_verts[roi_name]]) for roi_name in ROIs],
                                                            'Wmean_r': [mri_utils.weighted_mean(correlations[roi_verts[roi_name]],
                                                                          weights=mri_utils.normalize(np.mean(tsnr_arr, axis = 0))[roi_verts[roi_name]]) for roi_name in ROIs]})
                                                ))

                        ## plot average correlation values on flatmap surface ##
                        corr_flatmap = cortex.Vertex(np.mean(corr_arr, axis=0), 
                                                    self.MRIObj.params['plotting']['pycortex_sub'],
                                                    vmin = 0, vmax = 1,
                                                    cmap='hot')
                        #cortex.quickshow(corr_flatmap, with_curvature=True, with_sulci=True)
                        _ = cortex.quickflat.make_png(op.join(outdir,
                                        'half_split_correlation_flatmap_sub-{sj}_{ses}_task-{tsk}_acq-{acq}.png'.format(sj=pp, ses=ses, tsk=tsk, acq=acq)), 
                                        corr_flatmap, 
                                        recache = False, with_colorbar = True,
                                        with_curvature = True, with_sulci = True,
                                        curvature_brightness = 0.4, curvature_contrast = 0.1)

                        ## save surface correlation for relevant ROIS
                        # to make distribution plots
                        for roi_name in ROIs:
                            surf_avg_corr = pd.concat((surf_avg_corr, 
                                                    pd.DataFrame({'sj': np.tile(pp, len(roi_verts[roi_name])), 
                                                                'ses': np.tile(ses, len(roi_verts[roi_name])), 
                                                                'task': np.tile(tsk, len(roi_verts[roi_name])), 
                                                                'acq': np.tile(acq, len(roi_verts[roi_name])), 
                                                                'ROI': np.tile(roi_name, len(roi_verts[roi_name])), 
                                                                'vertex': roi_verts[roi_name],
                                                                'pearson_r': np.mean(corr_arr, axis=0)[roi_verts[roi_name]]})
                                                    ))

                ### PLOTS ####
                
                ## split half correlation across runs for the participant and session ##

                fig, all_axis = plt.subplots(2, 2, figsize=(20,15), dpi=100, facecolor='w', edgecolor='k')
                sns.set_theme(style="darkgrid")
                sns.set(font_scale=1.5) 
                key = ['mean_r', 'Wmean_r']
                key2 = ['Mean', 'Weighted Mean']

                for ind,axs in enumerate(all_axis):

                    b1 = sns.barplot(x = 'ROI', y = key[ind], 
                            hue = 'acq', data = corr_df[corr_df['task'] == 'pRF'], 
                            capsize = .2 ,linewidth = 1.8, ax=axs[0])
                    b1.set(xlabel=None)
                    b1.set(ylabel=None)
                    axs[0].set_ylabel('{k} Pearson R'.format(k=key2[ind]),fontsize = 18,labelpad=12)
                    axs[0].set_ylim(-.1,.6)
                    axs[0].set_title('Half-split correlation pRF runs, sub-{sj}_{ses}'.format(sj=pp, ses=ses)) 

                    b2 = sns.barplot(x = 'ROI', y = key[ind], 
                                    hue = 'acq', data = corr_df[corr_df['task'] == 'FA'], 
                                    capsize = .2 ,linewidth = 1.8, ax=axs[1])
                    b2.set(xlabel=None)
                    b2.set(ylabel=None)
                    axs[1].set_ylabel('{k} Pearson R'.format(k=key2[ind]),fontsize = 18,labelpad=12)
                    axs[1].set_ylim(-.1,.6)
                    axs[1].set_title('Half-split correlation FA runs, sub-{sj}_{ses}'.format(sj=pp, ses=ses)) 

                fig.savefig(op.join(outdir,'half_split_correlation_ROIS_sub-{sj}_{ses}.png'.format(sj=pp, ses=ses)), dpi=100,bbox_inches = 'tight')

                ### tSNR across runs for the participant and session

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5), dpi=100, facecolor='w', edgecolor='k')

                b1 = sns.barplot(x = 'ROI', y = 'mean_tsnr', 
                                hue = 'acq', data = tsnr_df[tsnr_df['task'] == 'pRF'], 
                                capsize = .2 ,linewidth = 1.8, ax=ax1)
                b1.set(xlabel=None)
                b1.set(ylabel=None)
                ax1.set_ylabel('mean tSNR',fontsize = 20,labelpad=18)
                ax1.set_ylim(0,130)
                ax1.set_title('tSNR pRF runs, sub-{sj}_{ses}'.format(sj=pp, ses=ses)) 

                b2 = sns.barplot(x = 'ROI', y = 'mean_tsnr', 
                                hue = 'acq', data = tsnr_df[tsnr_df['task'] == 'FA'], 
                                capsize = .2 ,linewidth = 1.8, ax=ax2)
                ax2.set_ylabel('mean tSNR',fontsize = 20,labelpad=18)
                ax2.set_ylim(0,130)
                ax2.set_title('tSNR FA runs, sub-{sj}_{ses}'.format(sj=pp, ses=ses)) 

                fig.savefig(op.join(outdir,'tSNR_ROIS_sub-{sj}_{ses}.png'.format(sj=pp, ses=ses)), dpi=100,bbox_inches = 'tight')

                ### split half correlation distribution per ROI ##
                fig, ax1 = plt.subplots(1, len(ROIs), figsize=(50,5), dpi=100, facecolor='w', edgecolor='k')

                for i in np.arange(len(ROIs)):
                    A = surf_avg_corr[(surf_avg_corr['task']=='pRF')&\
                            (surf_avg_corr['ROI']==ROIs[i])&\
                            (surf_avg_corr['acq']=='standard')].sort_values(by=['vertex'])['pearson_r'].values

                    B = surf_avg_corr[(surf_avg_corr['task']=='pRF')&\
                                (surf_avg_corr['ROI']==ROIs[i])&\
                                (surf_avg_corr['acq']=='nordic')].sort_values(by=['vertex'])['pearson_r'].values

                    sns.scatterplot(A,B, ax=ax1[i])
                    sns.lineplot([-1,1],[-1,1], color='red', ax=ax1[i])

                    ax1[i].set_xlabel('STANDARD',fontsize = 12,labelpad=18)
                    ax1[i].set_ylabel('NORDIC',fontsize = 12,labelpad=18)
                    ax1[i].set_ylim(-.2,1)
                    ax1[i].set_xlim(-.2,1)
                    ax1[i].set_title(ROIs[i]) 

                fig.savefig(op.join(outdir,'half_split_correlation_ROIS_distribution_sub-{sj}_{ses}.png'.format(sj=pp, ses=ses)), dpi=100,bbox_inches = 'tight')

        ## if we want a group plot
        if plot_group:
            ## group df
            group_corr_df = corr_df.groupby(['sj', 'task', 'acq', 'ROI'])['mean_r', 'Wmean_r'].mean().reset_index()
            group_tsnr_df = tsnr_df.groupby(['sj', 'task', 'acq', 'ROI'])['mean_tsnr'].mean().reset_index()
            group_surf_avg_corr = surf_avg_corr.groupby(['sj', 'task', 'acq', 'ROI', 'vertex'])['pearson_r'].mean().reset_index()

            fig, all_axis = plt.subplots(2, 2, figsize=(20,15), dpi=100, facecolor='w', edgecolor='k')
            sns.set_theme(style="darkgrid")
            sns.set(font_scale=1.5) 
            key = ['mean_r', 'Wmean_r']
            key2 = ['Mean', 'Weighted Mean']

            for ind,axs in enumerate(all_axis):

                b1 = sns.boxplot(x = 'ROI', y = key[ind], 
                        hue = 'acq', data = group_corr_df[group_corr_df['task'] == 'pRF'], 
                        ax=axs[0])
                b1.set(xlabel=None)
                b1.set(ylabel=None)
                axs[0].set_ylabel('{k} Pearson R'.format(k=key2[ind]),fontsize = 18,labelpad=12)
                axs[0].set_ylim(-.1,.6)
                axs[0].set_title('Half-split correlation pRF runs') 

                b2 = sns.boxplot(x = 'ROI', y = key[ind], 
                                hue = 'acq', data = group_corr_df[group_corr_df['task'] == 'FA'], 
                                ax=axs[1])
                b2.set(xlabel=None)
                b2.set(ylabel=None)
                axs[1].set_ylabel('{k} Pearson R'.format(k=key2[ind]),fontsize = 18,labelpad=12)
                axs[1].set_ylim(-.1,.6)
                axs[1].set_title('Half-split correlation FA runs') 

            fig.savefig(op.join(output_pth,'half_split_correlation_ROIS_sub-GROUP.png'), dpi=100,bbox_inches = 'tight')

            ### tSNR across runs for the participant and session

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5), dpi=100, facecolor='w', edgecolor='k')

            b1 = sns.boxplot(x = 'ROI', y = 'mean_tsnr', 
                            hue = 'acq', data = group_tsnr_df[group_tsnr_df['task'] == 'pRF'], 
                            ax=ax1)
            b1.set(xlabel=None)
            b1.set(ylabel=None)
            ax1.set_ylabel('mean tSNR',fontsize = 20,labelpad=18)
            ax1.set_ylim(0,130)
            ax1.set_title('tSNR pRF runs') 

            b2 = sns.boxplot(x = 'ROI', y = 'mean_tsnr', 
                            hue = 'acq', data = group_tsnr_df[group_tsnr_df['task'] == 'FA'], 
                            ax=ax2)
            ax2.set_ylabel('mean tSNR',fontsize = 20,labelpad=18)
            ax2.set_ylim(0,130)
            ax2.set_title('tSNR FA runs') 

            fig.savefig(op.join(outdir,'tSNR_ROIS_sub-GROUP.png'), dpi=100,bbox_inches = 'tight')

            ### split half correlation distribution per ROI ##
            fig, ax1 = plt.subplots(1, len(ROIs), figsize=(50,5), dpi=100, facecolor='w', edgecolor='k')

            for i in np.arange(len(ROIs)):
                A = group_surf_avg_corr[(group_surf_avg_corr['task']=='pRF')&\
                        (group_surf_avg_corr['ROI']==ROIs[i])&\
                        (group_surf_avg_corr['acq']=='standard')].sort_values(by=['vertex'])['pearson_r'].values

                B = group_surf_avg_corr[(group_surf_avg_corr['task']=='pRF')&\
                            (group_surf_avg_corr['ROI']==ROIs[i])&\
                            (group_surf_avg_corr['acq']=='nordic')].sort_values(by=['vertex'])['pearson_r'].values

                sns.scatterplot(A,B, ax=ax1[i])
                sns.lineplot([-1,1],[-1,1], color='red', ax=ax1[i])

                ax1[i].set_xlabel('STANDARD',fontsize = 12,labelpad=18)
                ax1[i].set_ylabel('NORDIC',fontsize = 12,labelpad=18)
                ax1[i].set_ylim(-.2,1)
                ax1[i].set_xlim(-.2,1)
                ax1[i].set_title(ROIs[i]) 

            fig.savefig(op.join(outdir,'half_split_correlation_ROIS_distribution_sub-GROUP.png'), dpi=100,bbox_inches = 'tight')

        #return tsnr_df, corr_df, surf_avg_corr
                            
                        
    def plot_tsnr(self, participant_list = [], input_pth = None, use_atlas_rois = True,
              file_ext = {'pRF': '_cropped_dc_psc.npy', 'FA': '_cropped_confound_psc.npy'}):

        """
        Plot tSNR
        
        NOTE - expects that we already ran postfmriprep, for both 
        
        Parameters
        ----------
        input_pth: str
            path to look for files, if None then will get them from derivatives/freesurfer/sub-X folder
        """ 

        ## output path to save plots
        output_pth = op.join(self.outputdir, 'tSNR')

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
        tsnr_df = pd.DataFrame({'sj': [], 'ses': [], 'task': [], 'ROI': [], 'mean_tsnr': []})
        
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
                    
                ## load data for both tasks
                for tsk in self.MRIObj.tasks:

                    ## bold filenames
                    bold_files[tsk] = [op.join(postfmriprep_pth, run) for run in os.listdir(postfmriprep_pth) if 'space-{sp}'.format(sp=self.MRIObj.sj_space) in run \
                                        and 'acq-{a}'.format(a=self.MRIObj.acq) in run and 'task-{t}'.format(t=tsk) in run and run.endswith(file_ext[tsk])]

                    ## calculate tSNR for each run
                    tsnr_arr = []
                    for ind,r in enumerate(bold_files[tsk]):

                        ## use non-PSC file to calculate tSNR
                        if 'cropped' in file_ext[tsk]:
                            r = r.replace(file_ext[tsk], '_cropped.npy')
                        else:
                            r = r.replace(file_ext[tsk], '.npy')

                        ## stack whole brain tsnr - will be used to weight correlations
                        tsnr_arr.append(mri_utils.get_tsnr(np.load(r), return_mean = False))

                        tsnr_df = pd.concat((tsnr_df, 
                                            pd.DataFrame({'sj': np.tile(pp, len(ROIs)), 
                                                        'ses': np.tile(ses, len(ROIs)), 
                                                        'task': np.tile(tsk, len(ROIs)), 
                                                        'ROI': ROIs, 
                                                        'mean_tsnr': [np.nanmean(mri_utils.get_tsnr(np.load(r), return_mean = False)[roi_verts[roi_name]]) for roi_name in ROIs]})
                                            ))
                    
                    ## plot average tSNR values on flatmap surface ##
                    tSNR_flatmap = cortex.Vertex(np.mean(tsnr_arr, axis=0), 
                                                self.MRIObj.params['plotting']['pycortex_sub'],
                                                vmin = 0, vmax = 150,
                                                cmap='hot')
                    #cortex.quickshow(tSNR_flatmap, with_curvature=True, with_sulci=True)
                    _ = cortex.quickflat.make_png(op.join(outdir,
                                    'tSNR_flatmap_sub-{sj}_{ses}_task-{tsk}.png'.format(sj=pp, ses=ses, tsk=tsk)), 
                                    tSNR_flatmap, 
                                    recache = False, with_colorbar = True,
                                    with_curvature = True, with_sulci = True,
                                    curvature_brightness = 0.4, curvature_contrast = 0.1)

                ### plot tSNR across runs for the participant and session
                fig, ax1 = plt.subplots(1, 1, figsize=(15,5), dpi=100, facecolor='w', edgecolor='k')
                sns.set_theme(style="darkgrid")
                sns.set(font_scale=1.5) 
                b1 = sns.barplot(x = 'ROI', y = 'mean_tsnr', hue = 'task', data = tsnr_df,
                            capsize = .2 ,linewidth = 1.8, ax=ax1)
                b1.set(xlabel=None)
                b1.set(ylabel=None)
                ax1.set_ylabel('mean tSNR',fontsize = 20,labelpad=18)
                ax1.set_ylim(0,130)
                         
                fig.savefig(op.join(outdir,'tSNR_ROIS_sub-{sj}_{ses}.png'.format(sj=pp, ses=ses)), dpi=100,bbox_inches = 'tight')

        #return tsnr_df


    def plot_vasculature(self, participant_list = [], input_pth = None, 
              file_ext = {'pRF': '_cropped_dc_psc.npy', 'FA': '_cropped_confound_psc.npy'}):

        """
        Plot mean EPI across pRF runs as a proxy of vasculature
        
        NOTE - expects that we already ran postfmriprep, 
        
        Parameters
        ----------
        input_pth: str
            path to look for files, if None then will get them from derivatives/freesurfer/sub-X folder
        """ 

        ## output path to save plots
        output_pth = op.join(self.outputdir, 'vasculature')

        ## input path, if not defined get's it from post-fmriprep dir
        if input_pth is None:
            input_pth = op.join(self.MRIObj.derivatives_pth, 'post_fmriprep', self.MRIObj.sj_space)

        ## loop over participants
        for pp in self.MRIObj.sj_num:

            # and over sessions (if more than one)
            for ses in self.MRIObj.session['sub-{sj}'.format(sj=pp)]:
                
                # path to post fmriprep dir
                postfmriprep_pth = op.join(input_pth, 'sub-{sj}'.format(sj=pp), ses)

                outdir = op.join(output_pth,'sub-{sj}'.format(sj=pp), ses)
                # if output path doesn't exist, create it
                if not op.isdir(outdir): 
                    os.makedirs(outdir)
                print('saving files in %s'%outdir)

                ## bold filenames
                bold_files = [op.join(postfmriprep_pth, run) for run in os.listdir(postfmriprep_pth) if 'space-{sp}'.format(sp=self.MRIObj.sj_space) in run \
                                    and 'acq-{a}'.format(a=self.MRIObj.acq) in run and 'task-pRF' in run and run.endswith(file_ext['pRF'])]

                mean_epi = []

                for file in bold_files:
                    ## use non-PSC file
                    if 'cropped' in file:
                        file = file.replace(file_ext['pRF'], '_cropped.npy')
                    else:
                        file = file.replace(file_ext['pRF'], '.npy')
                    mean_epi.append(np.load(file,allow_pickle=True)) 

                # average the EPI time course
                mean_epi = np.nanmean(np.nanmean(mean_epi, axis=0), axis=-1)
                
                # normalize image by dividing the value of each vertex 
                # by the value of the vertex with the maximum intensity
                norm_data = mri_utils.normalize(mean_epi)

                ## plot average tSNR values on flatmap surface ##
                epi_flatmap = cortex.Vertex(norm_data, 
                                            self.MRIObj.params['plotting']['pycortex_sub'],
                                            vmin = 0, vmax = 1,
                                            cmap='hot')
                #cortex.quickshow(epi_flatmap, with_curvature=True, with_sulci=True)
                _ = cortex.quickflat.make_png(op.join(outdir,
                                'mean_epi_flatmap_sub-{sj}_{ses}_task-pRF.png'.format(sj=pp, ses=ses)), 
                                epi_flatmap, 
                                recache = False, with_colorbar = True,
                                with_curvature = True, with_sulci = True,
                                curvature_brightness = 0.4, curvature_contrast = 0.1)


