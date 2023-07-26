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

#from FAM.utils import mri as mri_utils
#from FAM.utils import plot as plot_utils
#from FAM.processing import preproc_behdata

from matplotlib.backend_bases import MouseButton

from scipy.signal import periodogram

from FAM.visualize.viewer import Viewer


class MRIViewer(Viewer):

    def __init__(self, MRIObj, outputdir = None, pysub = 'hcp_99999'):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
        outputdir: str
            path to save plots
            
        """

        # need to initialize parent class (Model), indicating output infos
        super().__init__(MRIObj, pysub = pysub, outputdir = outputdir)

        ## output path to save plots
        self.figures_pth = op.join(self.outputdir)
        os.makedirs(self.figures_pth, exist_ok=True)


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
                output_pth = op.join(self.figures_pth, 'check_segmentations', 'sub-{sj}'.format(sj=pp))
                os.makedirs(output_pth, exist_ok=True)

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
                sag_slices = range(77, 280) #268) #248)

                for slice in sag_slices:
                    if not op.exists(op.join(output_pth, str(slice).zfill(3) + '.png')): # if image already in dir, skip
                        plot_slice = working_string.replace('$XPOS', str(slice).zfill(3))
                        plot_slice = plot_slice.replace('$OPFN', op.join(output_pth, str(slice).zfill(3) + '.png'))

                        os.system(plot_slice)

                subject = 'sub-{sj}'.format(sj=pp)
                convert_command = f'ffmpeg -framerate 5 -pattern_type glob -i "{output_pth}/*.png" -b:v 2M -c:v mpeg4 {output_pth}/{subject}.mp4'
                subprocess.call(convert_command, shell=True)


    def compare_nordic2standard(self, participant_list = [], input_pth = None, file_ext = {'pRF': '_cropped_dc_psc.npy', 'FA': '_cropped_confound_psc.npy'},
                                use_atlas_rois = None, acq_keys = ['standard', 'nordic'], plot_group=True):

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

        ## empty dataframe to save mean values per run
        corr_df = pd.DataFrame({'sj': [], 'ses': [], 'task': [], 'acq': [], 'ROI': [], 'mean_r': [], 'Wmean_r': []})
        tsnr_df = pd.DataFrame({'sj': [], 'ses': [], 'task': [], 'acq': [], 'ROI': [], 'mean_tsnr': []})
        ## also save full corr arrays
        surf_avg_corr = pd.DataFrame({'sj': [], 'ses': [], 'task': [], 'acq': [], 'ROI': [], 'vertex': [], 'pearson_r': []})

        ## if no participant list set, then run all
        if len(participant_list) == 0:
            participant_list = self.MRIObj.sj_num

        ## get vertices for each relevant ROI
        # from glasser atlas
        ROIs, roi_verts, color_codes = plot_utils.get_rois4plotting(self.MRIObj.params, 
                                                                sub_id = participant_list,
                                                                pysub = self.MRIObj.params['plotting']['pycortex_sub'], 
                                                                use_atlas = use_atlas_rois, 
                                                                atlas_pth = op.join(self.MRIObj.derivatives_pth,
                                                                                    'glasser_atlas','59k_mesh'), 
                                                                space = self.MRIObj.sj_space)
        ## loop over participants
        for pp in participant_list:

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
                                                pd.DataFrame({'sj': np.tile(pp, len(ROIs['sub-{sj}'.format(sj=pp)])), 
                                                            'ses': np.tile(ses, len(ROIs['sub-{sj}'.format(sj=pp)])), 
                                                            'task': np.tile(tsk, len(ROIs['sub-{sj}'.format(sj=pp)])), 
                                                            'acq': np.tile(acq, len(ROIs['sub-{sj}'.format(sj=pp)])), 
                                                            'ROI': ROIs['sub-{sj}'.format(sj=pp)], 
                                                            'mean_tsnr': [np.nanmean(mri_utils.get_tsnr(np.load(r), return_mean = False)[roi_verts['sub-{sj}'.format(sj=pp)][roi_name]]) for roi_name in ROIs['sub-{sj}'.format(sj=pp)]]})
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
                                                pd.DataFrame({'sj': np.tile(pp, len(ROIs['sub-{sj}'.format(sj=pp)])), 
                                                            'ses': np.tile(ses, len(ROIs['sub-{sj}'.format(sj=pp)])), 
                                                            'task': np.tile(tsk, len(ROIs['sub-{sj}'.format(sj=pp)])), 
                                                            'acq': np.tile(acq, len(ROIs['sub-{sj}'.format(sj=pp)])), 
                                                            'ROI': ROIs['sub-{sj}'.format(sj=pp)], 
                                                            'mean_r': [np.nanmean(correlations[roi_verts['sub-{sj}'.format(sj=pp)][roi_name]]) for roi_name in ROIs['sub-{sj}'.format(sj=pp)]],
                                                            'Wmean_r': [mri_utils.weighted_mean(correlations[roi_verts['sub-{sj}'.format(sj=pp)][roi_name]],
                                                                          weights=mri_utils.normalize(np.mean(tsnr_arr, axis = 0))[roi_verts['sub-{sj}'.format(sj=pp)][roi_name]]) for roi_name in ROIs['sub-{sj}'.format(sj=pp)]]})
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
                        for roi_name in ROIs['sub-{sj}'.format(sj=pp)]:
                            surf_avg_corr = pd.concat((surf_avg_corr, 
                                                    pd.DataFrame({'sj': np.tile(pp, len(roi_verts['sub-{sj}'.format(sj=pp)][roi_name])), 
                                                                'ses': np.tile(ses, len(roi_verts['sub-{sj}'.format(sj=pp)][roi_name])), 
                                                                'task': np.tile(tsk, len(roi_verts['sub-{sj}'.format(sj=pp)][roi_name])), 
                                                                'acq': np.tile(acq, len(roi_verts['sub-{sj}'.format(sj=pp)][roi_name])), 
                                                                'ROI': np.tile(roi_name, len(roi_verts['sub-{sj}'.format(sj=pp)][roi_name])), 
                                                                'vertex': roi_verts['sub-{sj}'.format(sj=pp)][roi_name],
                                                                'pearson_r': np.mean(corr_arr, axis=0)[roi_verts['sub-{sj}'.format(sj=pp)][roi_name]]})
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
                fig, ax1 = plt.subplots(1, len(ROIs['sub-{sj}'.format(sj=pp)]), figsize=(50,5), dpi=100, facecolor='w', edgecolor='k')

                for i in np.arange(len(ROIs['sub-{sj}'.format(sj=pp)])):
                    A = surf_avg_corr[(surf_avg_corr['task']=='pRF')&\
                            (surf_avg_corr['ROI']==ROIs['sub-{sj}'.format(sj=pp)][i])&\
                            (surf_avg_corr['acq']=='standard')].sort_values(by=['vertex'])['pearson_r'].values

                    B = surf_avg_corr[(surf_avg_corr['task']=='pRF')&\
                                (surf_avg_corr['ROI']==ROIs['sub-{sj}'.format(sj=pp)][i])&\
                                (surf_avg_corr['acq']=='nordic')].sort_values(by=['vertex'])['pearson_r'].values

                    sns.scatterplot(A,B, ax=ax1[i])
                    sns.lineplot([-1,1],[-1,1], color='red', ax=ax1[i])

                    ax1[i].set_xlabel('STANDARD',fontsize = 12,labelpad=18)
                    ax1[i].set_ylabel('NORDIC',fontsize = 12,labelpad=18)
                    ax1[i].set_ylim(-.2,1)
                    ax1[i].set_xlim(-.2,1)
                    ax1[i].set_title(ROIs['sub-{sj}'.format(sj=pp)][i]) 

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

            fig.savefig(op.join(output_pth,'tSNR_ROIS_sub-GROUP.png'), dpi=100,bbox_inches = 'tight')

            ### split half correlation distribution per ROI ##
            fig, ax1 = plt.subplots(1, len(ROIs['sub-{sj}'.format(sj=pp)]), figsize=(50,5), dpi=100, facecolor='w', edgecolor='k')

            for i in np.arange(len(ROIs['sub-{sj}'.format(sj=pp)])):
                A = group_surf_avg_corr[(group_surf_avg_corr['task']=='pRF')&\
                        (group_surf_avg_corr['ROI']==ROIs['sub-{sj}'.format(sj=pp)][i])&\
                        (group_surf_avg_corr['acq']=='standard')].sort_values(by=['vertex'])['pearson_r'].values

                B = group_surf_avg_corr[(group_surf_avg_corr['task']=='pRF')&\
                            (group_surf_avg_corr['ROI']==ROIs['sub-{sj}'.format(sj=pp)][i])&\
                            (group_surf_avg_corr['acq']=='nordic')].sort_values(by=['vertex'])['pearson_r'].values

                sns.scatterplot(A,B, ax=ax1[i])
                sns.lineplot([-1,1],[-1,1], color='red', ax=ax1[i])

                ax1[i].set_xlabel('STANDARD',fontsize = 12,labelpad=18)
                ax1[i].set_ylabel('NORDIC',fontsize = 12,labelpad=18)
                ax1[i].set_ylim(-.2,1)
                ax1[i].set_xlim(-.2,1)
                ax1[i].set_title(ROIs['sub-{sj}'.format(sj=pp)][i]) 

            fig.savefig(op.join(output_pth,'half_split_correlation_ROIS_distribution_sub-GROUP.png'), dpi=100,bbox_inches = 'tight')

        #return tsnr_df, corr_df, surf_avg_corr
                            
                        
    def plot_tsnr(self, participant_list = [], input_pth = None, use_atlas_rois = None,
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

        ## empty dataframe to save mean values per run
        tsnr_df = pd.DataFrame({'sj': [], 'ses': [], 'task': [], 'ROI': [], 'mean_tsnr': []})
        
        ## if no participant list set, then run all
        if len(participant_list) == 0:
            participant_list = self.MRIObj.sj_num

        ## get vertices for each relevant ROI
        # from glasser atlas
        ROIs, roi_verts, color_codes = plot_utils.get_rois4plotting(self.MRIObj.params, 
                                                                sub_id = participant_list,
                                                                pysub = self.MRIObj.params['plotting']['pycortex_sub'], 
                                                                use_atlas = use_atlas_rois, 
                                                                atlas_pth = op.join(self.MRIObj.derivatives_pth,
                                                                                    'glasser_atlas','59k_mesh'), 
                                                                space = self.MRIObj.sj_space)
        
        ## loop over participants

        for pp in participant_list:

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
                                            pd.DataFrame({'sj': np.tile(pp, len(ROIs['sub-{sj}'.format(sj=pp)])), 
                                                        'ses': np.tile(ses, len(ROIs['sub-{sj}'.format(sj=pp)])), 
                                                        'task': np.tile(tsk, len(ROIs['sub-{sj}'.format(sj=pp)])), 
                                                        'ROI': ROIs['sub-{sj}'.format(sj=pp)], 
                                                        'mean_tsnr': [np.nanmean(mri_utils.get_tsnr(np.load(r), return_mean = False)[roi_verts['sub-{sj}'.format(sj=pp)][roi_name]]) for roi_name in ROIs['sub-{sj}'.format(sj=pp)]]})
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
                ax1.set_ylim(0,150)
                         
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

        ## if no participant list set, then run all
        if len(participant_list) == 0:
            participant_list = self.MRIObj.sj_num
        
        for pp in participant_list:

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


    def plot_bold_on_surface(self, participant_list = [], input_pth = None, run_type = 'mean', task = 'pRF',
                         stim_on_screen = None, use_atlas_rois = None,
                         file_ext = {'pRF': '_cropped_dc_psc.npy', 'FA': '_cropped_confound_psc.npy'}):

        """
        Plot bold func data, 
        and make video of bold change by TR
        to check for visual stimuli
        
        NOTE - expects that we already ran postfmriprep
        
        Parameters
        ----------
        participant_list: list
            list with participant ID
        input_pth: str or None
            path to look for files, if None then will get them from derivatives/postfmriprep/<space>/sub-X folder
        run_type: string or int
            if we want to average (mean vs median) or just plot a single run (1)
        stim_on_screen: arr
            boolean array with moments where stim was on screen
        file_ext: dict
            dictionary with file extension per task, to select appropriate files
        """ 
        
        ## if not array provided with instances where stim was on screen, make it
        if not stim_on_screen:
            
            # load preproc functions for object
            mri_beh = preproc_behdata.PreprocBeh(self.MRIObj)
            
            # make stim on screen arr
            if task == 'pRF':
                stim_on_screen = np.zeros(mri_beh.pRF_total_trials)
                stim_on_screen[mri_beh.pRF_bar_pass_trials] = 1
            elif task == 'FA':
                stim_on_screen = np.zeros(mri_beh.FA_total_trials)
                stim_on_screen[mri_beh.FA_bar_pass_trials] = 1
                
            # crop and shift if it's the case
            crop_nr = self.MRIObj.params[task]['crop_TR'] if self.MRIObj.params[task]['crop'] else None
    
            stim_on_screen = mri_utils.crop_shift_arr(stim_on_screen, 
                                                    crop_nr = crop_nr, 
                                                    shift = self.MRIObj.params['mri']['shift_DM_TRs'])
            
        ## output path to save plots
        output_pth = op.join(self.outputdir, 'BOLD', task)

        ## input path, if not defined get's it from post-fmriprep dir
        if input_pth is None:
            input_pth = op.join(self.MRIObj.derivatives_pth, 'post_fmriprep', self.MRIObj.sj_space)

    
        ## if no participant list set, then run all
        if len(participant_list) == 0:
            participant_list = self.MRIObj.sj_num

        ## get vertices for each relevant ROI
        # from glasser atlas
        ROIs, roi_verts, color_codes = plot_utils.get_rois4plotting(self.MRIObj.params, 
                                                                sub_id = participant_list,
                                                                pysub = self.MRIObj.params['plotting']['pycortex_sub'], 
                                                                use_atlas = use_atlas_rois, 
                                                                atlas_pth = op.join(self.MRIObj.derivatives_pth,
                                                                                    'glasser_atlas','59k_mesh'), 
                                                                space = self.MRIObj.sj_space)
        
        ## loop over participants

        for pp in participant_list:

            # and over sessions (if more than one)
            for ses in self.MRIObj.session['sub-{sj}'.format(sj=pp)]:
                
                # path to post fmriprep dir
                postfmriprep_pth = op.join(input_pth, 'sub-{sj}'.format(sj=pp), ses)

                outdir = op.join(output_pth,'sub-{sj}'.format(sj=pp), ses, 'run-{rt}'.format(rt=run_type))
                # if output path doesn't exist, create it
                if not op.isdir(outdir): 
                    os.makedirs(outdir)
                print('saving files in %s'%outdir)

                ## bold filenames
                bold_files = [op.join(postfmriprep_pth, run) for run in os.listdir(postfmriprep_pth) if 'space-{sp}'.format(sp=self.MRIObj.sj_space) in run \
                                    and 'acq-{a}'.format(a=self.MRIObj.acq) in run and \
                              'task-{tsk}'.format(tsk=task) in run and run.endswith(file_ext[task])]

                ## Load data we want to look at 
                # single run
                if isinstance(run_type, int):
                    bold_files = [val for val in bold_files if 'run-{rt}'.format(rt=run_type) in val]
                    if len(bold_files)>0:
                        data_arr = np.load(bold_files[0],allow_pickle=True)
                    else:
                        raise NameError('run-{rt} doesnt exist in {ip}'.format(rt=run_type,
                                                                                ip=input_pth))
                # average runs
                elif isinstance(run_type, str):
                    match run_type:
                        case 'mean':
                            data_arr = np.mean(np.stack((np.load(val,allow_pickle=True) for val in bold_files)), axis = 0)
                        case 'median':
                            data_arr = np.median(np.stack((np.load(val,allow_pickle=True) for val in bold_files)), axis = 0)
                        case TypeError:
                            print('run-{rt} not implemented/exists'.format(rt=run_type))
                            
                            
                ### if FA then we also want to get average timecourse across ROI ####
                # to check if something is off (some arousal artifact or so) 
                if (task == 'FA') and (run_type in ['mean', 'median']):

                    avg_bold_roi = {} #empty dictionary 

                    for _,val in enumerate(ROIs['sub-{sj}'.format(sj=pp)]):    
                        avg_bold_roi[val] = np.nanmean(data_arr[roi_verts['sub-{sj}'.format(sj=pp)][val]], axis=0)
                        
                    # plot data with model
                    fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)

                    time_sec = np.linspace(0,len(data_arr[0]) * self.MRIObj.params['mri']['TR'], num=len(data_arr[0])) # array with timepoints, in seconds
                    
                    plt.plot(time_sec, stim_on_screen, linewidth = 5, alpha = 1, linestyle = 'solid', color = 'gray')

                    for _,key in enumerate(ROIs['sub-{sj}'.format(sj=pp)]):
                        plt.plot(time_sec, avg_bold_roi[key], linewidth = 1.5, label = '%s'%key, color = color_codes['sub-{sj}'.format(sj=pp)][key], alpha = .6)

                    # also plot average of all time courses
                    plt.plot(time_sec, np.mean(np.stack((avg_bold_roi[val] for val in ROIs['sub-{sj}'.format(sj=pp)]), axis = 0), axis = 0),
                            linewidth = 2.5, label = 'average', linestyle = 'solid', color = 'k')

                    axis.set_xlabel('Time (s)',fontsize=20, labelpad=20)
                    axis.set_ylabel('BOLD signal change (%)',fontsize=20, labelpad=10)
                    axis.legend(loc='upper left',fontsize=7)  # doing this to guarantee that legend is how I want it 
                    #axis.set_xlim([0, time_sec[-1]])

                    fig.savefig(op.join(outdir, 'average_BOLD_across_runs_rois.png'))
                ###


                ######## make movie #########
                movie_name = op.join(outdir,
                                     'flatmap_space-{space}_type-BOLD_visual_movie.mp4'.format(space=self.MRIObj.sj_space))

                if not op.isfile(movie_name):
                    for num_tr in range(data_arr.shape[-1]):

                        filename = op.join(outdir,'flatmap_space-{space}_type-BOLD_visual_TR-{time}.png'.format(space = self.MRIObj.sj_space,
                                                                                                                time = str(num_tr).zfill(3)))
                        if not op.isfile(filename): # if image already in dir, skip
                        
                            # set figure grid 
                            full_fig = plt.figure(constrained_layout = True, figsize = (15,8))
                            gs = full_fig.add_gridspec(5, 6)
                            ## set axis
                            dm_ax = full_fig.add_subplot(gs[:1,2:4])
                            flatmap_ax = full_fig.add_subplot(gs[1:,:])
                            # set flatmap
                            flatmap = cortex.Vertex(data_arr[...,num_tr], 
                                                    self.MRIObj.params['plotting']['pycortex_sub'],
                                                    vmin = -5, vmax = 5,
                                                    cmap='BuBkRd')
                            cortex.quickshow(flatmap, 
                                            with_colorbar = True, with_curvature = True, with_sulci = True,
                                            with_labels = False, fig = flatmap_ax)

                            flatmap_ax.set_xticks([])
                            flatmap_ax.set_yticks([])

                            # set dm timecourse
                            dm_ax.plot(stim_on_screen)
                            dm_ax.axvline(num_tr, color='red', linestyle='solid', lw=1)
                            dm_ax.set_yticks([])

                            print('saving %s' %filename)
                            full_fig.savefig(filename)

                    ## save as video
                    img_name = filename.replace('_TR-%s.png'%str(num_tr).zfill(3),'_TR-%3d.png')
                    os.system("ffmpeg -r 6 -start_number 0 -i %s -vcodec mpeg4 -y %s"%(img_name,movie_name)) 

                else:
                    print('movie already exists as %s'%movie_name)
                    
                #########


    def check_click_bold(self, participant, run, ses, task = 'FA', input_pth = None, file_ext = '_cropped_LinDetrend_psc.npy',
                                flatmap_height = 2048, full_figsize = (12, 8), pysub = 'hcp_999999'):

        """
        Check BOLD timecourse on flatmap, by clicking around
        will show raw (cropped) tc, filtered, and PSC to compare
        
        NOTE - needs to be run in local system!!
        NOTE2 - requires post_fmriprep to have been run already
        
        Parameters
        ----------
        participant: str
            participant ID to look at
        run: str/int
            run number
        ses: str/int
            session number
        task: str
            task identifier ('FA' vs 'pRF')
        input_pth: str
            path to look for files, if None then will get them from derivatives/freesurfer/sub-X folder
        
        """ 

        ## set input path where processed files are
        if input_pth is None:
            input_pth = op.join(self.MRIObj.derivatives_pth, 'post_fmriprep', self.MRIObj.sj_space, 'sub-{sj}'.format(sj=participant), 'ses-{s}'.format(s=ses))

        print('post_fmriprep files in %s'%input_pth)

        ## get bold data that is PSC
        psc_bold_list = [op.join(input_pth, file) for file in os.listdir(input_pth) if 'run-{r}'.format(r=run) in file \
                        and 'acq-{a}'.format(a=self.MRIObj.acq) in file and 'task-{t}'.format(t=task) in file and file.endswith(file_ext)]
        if len(psc_bold_list) > 1:
            raise ValueError('Too many runs found, dont know which to choose %s'%str(psc_bold_list))
        else:
            print('Loading PSC data %s'%psc_bold_list[0])
            psc_bold_arr = np.load(psc_bold_list[0])

        ## get bold data that is filtered
        filt_ext = file_ext.replace('_psc.npy', '.npy') ## should be careful with this, might raise issues if I change file format in the future

        filt_bold_list = [op.join(input_pth, file) for file in os.listdir(input_pth) if 'run-{r}'.format(r=run) in file \
                        and 'acq-{a}'.format(a=self.MRIObj.acq) in file and 'task-{t}'.format(t=task) in file and file.endswith(filt_ext)]
        if len(filt_bold_list) > 1:
            raise ValueError('Too many runs found, dont know which to choose %s'%str(filt_bold_list))
        else:
            print('Loading Filtered data %s'%filt_bold_list[0])
            filt_bold_arr = np.load(filt_bold_list[0])

        ## get raw data (usually cropped)
        raw_ext = '_cropped.npy' if self.MRIObj.params[task]['crop'] == True else self.MRIObj.file_ext.replace('.','_').replace('_nii', '.npy')

        raw_bold_list = [op.join(input_pth, file) for file in os.listdir(input_pth) if 'run-{r}'.format(r=run) in file \
                        and 'acq-{a}'.format(a=self.MRIObj.acq) in file and 'task-{t}'.format(t=task) in file and file.endswith(raw_ext)]
        if len(raw_bold_list) > 1:
            raise ValueError('Too many runs found, dont know which to choose %s'%str(raw_bold_list))
        else:
            print('Loading Raw data %s'%raw_bold_list[0])
            raw_bold_arr = np.load(raw_bold_list[0])

        ## save data in variable (to access later)
        self.bold_data = {'psc': psc_bold_arr, 'filt': filt_bold_arr, 'raw': raw_bold_arr}

        ## calculate tSNR 
        # from raw bold data
        tsnr_arr = mri_utils.get_tsnr(raw_bold_arr, return_mean=False, affine=[], hdr=[], filename=None)

        ### now set click viewer
        
        ## figure settings
        self.flatmap_height = flatmap_height
        self.full_figsize = full_figsize
        self.images = {}

        ## create pycortex vars
        self.mask, extents = cortex.quickflat.utils.get_flatmask(pysub, height = self.flatmap_height)
        self.vc = cortex.quickflat.utils._make_vertex_cache(pysub, height = self.flatmap_height)

        self.mask_index = np.zeros(self.mask.shape)
        self.mask_index[self.mask] = np.arange(self.mask.sum())

        ## set figure grid 
        self.full_fig = plt.figure(constrained_layout = True, figsize = self.full_figsize)

        gs = self.full_fig.add_gridspec(5, 3)

        self.flatmap_ax = self.full_fig.add_subplot(gs[:2, :])

        self.raw_timecourse_ax = self.full_fig.add_subplot(gs[2, :2])
        self.filt_timecourse_ax = self.full_fig.add_subplot(gs[3, :2])
        self.psc_timecourse_ax = self.full_fig.add_subplot(gs[4, :])

        self.raw_freq_ax = self.full_fig.add_subplot(gs[2, 2:])
        self.filt_freq_ax = self.full_fig.add_subplot(gs[3, 2:])

        self.flatmap_ax.set_title('flatmap')
        self.raw_timecourse_ax.set_title('Raw timecourse')
        self.filt_timecourse_ax.set_title('Filtered timecourse')
        self.psc_timecourse_ax.set_title('PSC timecourse')

        self.raw_freq_ax.set_title('Raw frequency-domain')
        self.filt_freq_ax.set_title('Filtered frequency-domain')

        ## set tSNR flatmap to show
        self.images['tSNR'] = plot_utils.get_flatmaps(tsnr_arr, 
                                                        vmin1 = 0, vmax1 = 150,
                                                        pysub = pysub, 
                                                        cmap = 'hot')

        cortex.quickshow(self.images['tSNR'], fig = self.flatmap_ax,
                        with_rois = False, with_curvature = True, with_colorbar=False, 
                        with_sulci = True, with_labels = False)

        ## set clicker func and show
        self.full_fig.canvas.mpl_connect('button_press_event', self.onclick)

        plt.show()


    def onclick(self, event):

        """
        Helper function for click plotting
        """

        print('you pressed', event.button, event.xdata, event.ydata)

        if  event.button is MouseButton.RIGHT:
            refresh_fig = True
        else:
            refresh_fig = False
        
        if event.inaxes == self.flatmap_ax:
            xmin, xmax = self.flatmap_ax.get_xbound()
            ax_xrange = xmax-xmin
            ymin, ymax = self.flatmap_ax.get_ybound()
            ax_yrange = ymax-ymin

            rel_x = int(self.mask.shape[0] * (event.xdata-xmin)/ax_xrange)
            rel_y = int(self.mask.shape[1] * (event.ydata-ymin)/ax_yrange)
            clicked_pixel = (rel_x, rel_y)

            clicked_vertex = self.vc[int(
                self.mask_index[clicked_pixel[0], clicked_pixel[1]])]

            print(clicked_vertex)
            self.redraw_vertex_plots(clicked_vertex.indices[0], refresh_fig)
            plt.draw()

    
    def redraw_vertex_plots(self, vertex, refresh):

        """
        Helper function for click redrawing of vertex
            
        """
        
        self.vertex = vertex

        print(refresh)

        if refresh: # if we want to clean up timecourses
            self.raw_timecourse_ax.clear()
            self.filt_timecourse_ax.clear()
            self.psc_timecourse_ax.clear()
            self.raw_freq_ax.clear()
            self.filt_freq_ax.clear()

        # plot timecourse data
        self.raw_timecourse_ax = self.plot_bold_tc(self.raw_timecourse_ax, timecourse = self.bold_data['raw'][vertex])
        
        self.filt_timecourse_ax = self.plot_bold_tc(self.filt_timecourse_ax, timecourse = self.bold_data['filt'][vertex],
                                                            timecourse2 = self.bold_data['raw'][vertex])
        
        self.psc_timecourse_ax = self.plot_bold_tc(self.psc_timecourse_ax, timecourse = self.bold_data['psc'][vertex], units = 'psc')

        # plot frequency power plots
        self.raw_freq_ax = self.plot_periodogram(self.raw_freq_ax, timecourse = self.bold_data['raw'][vertex], TR = self.MRIObj.TR)
        self.filt_freq_ax = self.plot_periodogram(self.filt_freq_ax, timecourse = self.bold_data['filt'][vertex], TR = self.MRIObj.TR)

    
    def plot_bold_tc(self, axis, timecourse = None, timecourse2 = None, units = '', start_task_ind = 14, end_task_ind = 279):

        """
        Helper function that actually plots timecourse
            
        """

        if timecourse2 is not None:
            axis.plot(timecourse, 'k-', alpha = .8, label='data')
            axis.plot(timecourse2, 'k-', alpha = .25, label='data2')
        else:
            axis.plot(timecourse, 'k-', alpha = .8, label='data')

        axis.set_xlabel('Time (TR)',fontsize = 15, labelpad = 10)
        if units == 'psc':
            axis.set_ylabel('BOLD (% sig change)',fontsize = 15, labelpad = 10)
        else:
            axis.set_ylabel('BOLD (a.u.)',fontsize = 15, labelpad = 10)

        axis.vlines(start_task_ind, np.min(timecourse),np.max(timecourse), color='red', alpha = .5)
        axis.vlines(end_task_ind, np.min(timecourse),np.max(timecourse), color='red', alpha = .5)

        axis.set_xlim(0, len(timecourse))
        
        return axis

    
    def plot_periodogram(self, axis, timecourse = None, TR = 1.6):

        """
        Helper function that actually plots power spectral density
            
        """

        sampling_frequency = 1 / TR  
        freq, power = periodogram(timecourse, fs = sampling_frequency)#, detrend = False)
        
        axis.plot(freq, power, 'g-', alpha = .8, label='data')

        axis.set_xlabel('Frequency (Hz)',fontsize = 15, labelpad = 10)
        axis.set_ylabel('Power (dB)',fontsize = 15, labelpad = 10)

        axis.axvline(x=0.01,color='r',ls='dashed', lw=2)
        
        return axis






        