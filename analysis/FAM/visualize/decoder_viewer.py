

import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

#import ptitprince as pt # raincloud plots
import matplotlib.patches as mpatches
from  matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
from matplotlib.collections import PolyCollection

import cortex

import subprocess
from tqdm import tqdm

from FAM.visualize.viewer import Viewer

from PIL import Image, ImageDraw

from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display


class DecoderViewer(Viewer):


    def __init__(self, MRIObj, outputdir = None, DecoderObj = None, pysub = 'hcp_999999', use_atlas = None):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
        DecoderModelObj: Decoder object
            object from decoder class
        outputdir: str
            path to save plots
        pysub: str
            basename of pycortex subject folder, where we drew all ROIs. 
        """

        # need to initialize parent class (Model), indicating output infos
        super().__init__(MRIObj, pysub = pysub, outputdir = outputdir, use_atlas = use_atlas)

        ## output path to save plots
        self.figures_pth = op.join(self.outputdir)
        os.makedirs(self.figures_pth, exist_ok=True)

        # Load Decoder model objects
        self.DecoderObj = DecoderObj

        # set font type for plots globally
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Helvetica'

    def plot_prf_locations2D(self, pp = None, pars_df = None, ROI_list = ['V1'], filename = None, best_vox = False, polar_plot = False,
                                wspace=0.05, hspace=0.2):

        """
        Plot pRF locations on screen (2D visualization to check spatial distribution of fitted voxels)
        """

        if polar_plot:
            fig, axes = plt.subplots(nrows=1, ncols=len(ROI_list), figsize = (3.5*len(ROI_list),4), sharey=True, sharex=True,
                                    subplot_kw={'projection': 'polar'})
        else:
            fig, axes = plt.subplots(nrows=1, ncols=len(ROI_list), figsize = (3.5*len(ROI_list),4), sharey=True, sharex=True)

        ## if we want to plot best fitting voxels (actually used in encoding model)
        if best_vox:
            df2plot = pars_df[(pars_df['best_voxel'] == True) &\
                            (pars_df['sj'] == 'sub-{sj}'.format(sj = pp))]
            fig.suptitle('PRF locations (Best Voxels) - sub-{sj}'.format(sj = pp), fontsize=14, y = 1.05)
        else:
            df2plot = pars_df[(pars_df['sj'] == 'sub-{sj}'.format(sj = pp))]
            fig.suptitle('PRF locations - sub-{sj}'.format(sj = pp), fontsize=14, y = 1.05)

        ## loop over ROIs in list
        for ind, roi_name in enumerate(ROI_list):

            if polar_plot:
                # plot polar
                axes[ind].scatter(df2plot[(df2plot['ROI'] == roi_name)].polar_angle.values, 
                                df2plot[(df2plot['ROI'] == roi_name)].eccentricity.values, 
                                c=df2plot[(df2plot['ROI'] == roi_name)].polar_angle.values, 
                                cmap='hsv')
                
                axes[ind].set_title(roi_name, fontsize=14, y = 1.15)
                axes[ind].tick_params(axis='both', labelsize=10)

            else:
                sns.scatterplot(x='x', y='y', hue='r2', hue_norm = (.1, .9),
                                data=df2plot[(df2plot['ROI'] == roi_name)], 
                            size='sd', sizes=(15, 150), palette='viridis',
                            ax = axes[ind], legend = True)
                axes[ind].vlines(0, -6, 6, linestyles='dashed', color='k', alpha = .3)
                axes[ind].hlines(0, -6, 6, linestyles='dashed', color='k', alpha = .3)

                axes[ind].set_title(roi_name, fontsize=14,y = 1.05)

                axes[ind].set_box_aspect(1)

                axes[ind].set_ylim([-6,6])
                axes[ind].set_xlim([-6,6])
                
                axes[ind].tick_params(axis='both', labelsize=13)
                
                axes[ind].legend(loc='upper right', fontsize='x-small')
        
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.tight_layout()

        ## if we gave filename, then save
        if filename:
            fig.savefig(filename)

    def plot_rsq_distribution(self, pp = None, pars_df = None, filename = None, best_vox = False, figsize=(8,5), model_type = 'dog_hrf'):

        """
        Plot RSQ distribution of pRF fits for a given participant
        """

        fig, axes0 = plt.subplots(nrows=1, ncols=1, figsize = figsize, sharey=True, sharex=True)

        ## if we want to plot best fitting voxels (actually used in encoding model)
        if best_vox:
            df2plot = pars_df[(pars_df['best_voxel'] == True) &\
                            (pars_df['sj'] == 'sub-{sj}'.format(sj = pp))]
            fig.suptitle('{mname} model (Best Voxels) - sub-{sj}'.format(sj = pp,
                                                                        mname = model_type.upper()), fontsize=14, y = .98)
        else:
            df2plot = pars_df[(pars_df['sj'] == 'sub-{sj}'.format(sj = pp))]
            fig.suptitle('{mname} model - sub-{sj}'.format(sj = pp,
                                                                mname = model_type.upper()), fontsize=14, y = .98)
            
        sns.boxplot(y='r2', x = 'ROI', hue = 'ROI',
                    data=df2plot, 
                    palette= self.MRIObj.params['plotting']['ROI_pal'],
                    width=.5, linewidth = 2, linecolor= 'black', legend = False,
                    ax = axes0)
        sns.stripplot(data = df2plot, 
                    dodge=False, alpha = .35, s = 3, linewidth=0.75, 
                    y = 'r2', hue = 'ROI', x = 'ROI',
                    palette = self.MRIObj.params['plotting']['ROI_pal'],
                    legend=False, ax=axes0)

        axes0.tick_params(axis='both', labelsize=14)

        #axes0.legend(loc='upper right', fontsize='small')

        axes0.set_ylim(0,1)
        axes0.set_ylabel('RSQ',fontsize = 16, labelpad=15)
        axes0.set_xlabel('')
        plt.tight_layout()

        ## if we gave filename, then save
        if filename:
            fig.savefig(filename)

    def plot_rsq_group(self, pars_df = None, filename = None, best_vox = False, figsize=(8,5), model_type = 'dog_hrf', ROI_list = ['V1']):

        """
        Plot RSQ distribution of pRF fits at group level
        """

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize = figsize, sharey=True, sharex=True)

        ## if we want to plot best fitting voxels (actually used in encoding model)
        if best_vox:
            df2plot = pars_df[(pars_df['best_voxel'] == True)]
            fig.suptitle('{mname} model (Best Voxels) - sub-GROUP'.format(mname = model_type.upper()), fontsize=14, y = .98)
        else:
            df2plot = pars_df
            fig.suptitle('{mname} model - sub-GROUP'.format(mname = model_type.upper()), fontsize=14, y = .98)
            
        sns.barplot(y = 'r2', x = 'ROI',
                    data = df2plot.groupby(['ROI', 'sj']).mean(numeric_only=True).reset_index(), 
                    palette = self.MRIObj.params['plotting']['ROI_pal'],
                    order = ROI_list,
                    hue = 'ROI', hue_order = ROI_list,
                    width=.8, linewidth = 1.5,
                    errorbar = 'se',
                ax = ax1)
        sns.pointplot(data = df2plot, 
                    dodge=True, alpha = 1, linewidth=1, linestyle = 'none', markersize = 3,
                    y = 'r2', hue = 'sj', x = 'ROI', errorbar = ('se'), #capsize = .08,
                    #palette = self.MRIObj.params['plotting']['ROI_pal'],
                    legend=True,
                    ax = ax1)
        ax1.legend(title = None, fontsize = 'small', loc='center left', bbox_to_anchor=(1, 0.5))

        ax1.axhline(y=0, color='k', linestyle='--')

        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.set_ylim(.2,.9)
        ax1.set_ylabel('RSQ', fontsize = 16, labelpad = 15)
        ax1.set_xlabel('')
        plt.tight_layout()

        ## if we gave filename, then save
        if filename:
            fig.savefig(filename)

    def plot_ecc_size(self, pp = None, pars_df = None, filename = None, best_vox = False, figsize=(8,5), model_type = 'dog_hrf', fwhmax = True):

        """
        Plot ECC-size of pRF fits for participant 
        """

        #fig, axes0 = plt.subplots(nrows=1, ncols=1, figsize = figsize, sharey=True, sharex=True)

        ## if we want to plot best fitting voxels (actually used in encoding model)
        if best_vox:
            df2plot = pars_df[(pars_df['best_voxel'] == True) &\
                            (pars_df['sj'] == 'sub-{sj}'.format(sj = pp))]
            title_str = '{mname} model (Best Voxels) - sub-{sj}'.format(sj = pp,
                                                                        mname = model_type.upper())
        else:
            df2plot = pars_df[(pars_df['sj'] == 'sub-{sj}'.format(sj = pp))]
            title_str = '{mname} model - sub-{sj}'.format(sj = pp,
                                                            mname = model_type.upper())
            
        if fwhmax:
            g = sns.lmplot(df2plot, 
                            x = 'eccentricity', y = 'size_fwhmax', hue = 'ROI', markers="x", 
                        palette = self.MRIObj.params['plotting']['ROI_pal'],
                scatter_kws={'alpha':0.2}, height=5, legend = True)
            
            axes0 = plt.gca()
            axes0.set_ylabel('pRF size FWHMax [deg]', fontsize = 16, labelpad = 15)
        else:
            g = sns.lmplot(df2plot, 
                            x = 'eccentricity', y = 'sd', hue = 'ROI', markers="x", 
                        palette = self.MRIObj.params['plotting']['ROI_pal'],
                scatter_kws={'alpha':0.2}, height=5, legend = True)
            
            axes0 = plt.gca()
            axes0.set_ylabel('pRF size [deg]', fontsize = 16, labelpad = 15)

        axes0.set_title(title_str, fontsize=14, y = 1.05)
        axes0.tick_params(axis='both', labelsize=14)
        axes0.axes.set_xlim(0, 8)
        axes0.axes.set_ylim(0.3, 14)
        axes0.set_xlabel('pRF eccentricity [deg]', fontsize = 16, labelpad = 15)
        
        #axes0.legend(title = 'ROI', fontsize = 'small', loc='center left', bbox_to_anchor=(1, 0.5))
        sns.despine(offset=15)

        # to make legend full alpha
        for lh in g._legend.legendHandles: 
            lh.set_alpha(1)
        #fig2 = plt.gca()

        ## if we gave filename, then save
        if filename:
            g.savefig(filename)

    def plot_prf_diagnostics(self, participant_list = [], ROI_list = ['V1'], model_type = 'gauss_hrf',
                                    prf_file_ext = '_cropped_dc_psc.nii.gz', ses = 'mean', 
                                    mask_bool_df = None, stim_on_screen = [], fig_type = 'png'):
        
        """plot encoding model pRF estimates
        for all participants and ROIs
        """
        
        # make dir to save estimates
        fig_dir = op.join(self.figures_pth, 'prf_decoder')
        # and set base figurename 
        fig_id = 'sub-GROUP_task-pRF_model-{modname}_estimates'.format(modname = model_type)

        # base filename for figures 
        base_filename = op.join(fig_dir, fig_id)
               
        ## 
        # get pars for all ROIs and participants
        pars_df = self.DecoderObj.load_encoding_model_pars(participant_list = participant_list, 
                                                        ROI_list = ROI_list,
                                                        model_type = model_type,
                                                        prf_file_ext = prf_file_ext, 
                                                        ses = ses, 
                                                        mask_bool_df = mask_bool_df, 
                                                        stim_on_screen = stim_on_screen,
                                                        pars_as_df = True)
        ## calculate ecc and polar angle
        # and add to df
        pars_df.loc[:, ['eccentricity']] = np.abs(pars_df.x.values + pars_df.y.values * 1j)
        pars_df.loc[:, ['polar_angle']] =  np.angle(pars_df.x.values + pars_df.y.values * 1j)

        ## also get size as FWHM
        size_fwhmax, _ = self.MRIObj.mri_utils.fwhmax_fwatmin('dog', 
                                                    amplitude = pars_df.amplitude.values, 
                                                    size = pars_df.sd.values, 
                                                    sa = pars_df.srf_amplitude.values, 
                                                    ss = pars_df.srf_size.values,  
                                                    return_profiles=False)
        pars_df.loc[:, ['size_fwhmax']] = size_fwhmax

        ## now actually plot

        for pp in participant_list:
            
            # make dir to save estimates
            pp_fig_dir = op.join(fig_dir, 'sub-{sj}'.format(sj = pp))
            pp_fig_id = fig_id.replace('sub-GROUP', 'sub-{sj}'.format(sj = pp)) 
        
            os.makedirs(pp_fig_dir, exist_ok = True)
            print('saving figures in %s'%pp_fig_dir)
            
            # base filename for figures 
            pp_base_filename = op.join(pp_fig_dir, pp_fig_id)

            ## PRF LOCATIONS 2D
            self.plot_prf_locations2D(pp = pp, 
                                    pars_df = pars_df, 
                                    ROI_list = ROI_list,
                                    filename = pp_base_filename+'_pRF_locations.'+fig_type, 
                                    best_vox = False)
            
            ## PRF LOCATIONS 2D - voxel locations used in decoder
            self.plot_prf_locations2D(pp = pp, 
                                    pars_df = pars_df, 
                                    ROI_list = ROI_list,
                                    filename = pp_base_filename+'_pRF_locations_bestvox.'+fig_type, 
                                    best_vox = True)
            
            ## PRF LOCATIONS POLAR - voxel locations used in decoder
            self.plot_prf_locations2D(pp = pp, 
                                    pars_df = pars_df, 
                                    ROI_list = ROI_list,
                                    filename = pp_base_filename+'_pRF_PA_bestvox.'+fig_type, 
                                    best_vox = True,
                                    polar_plot = True,
                                    wspace = .3)
            
            ## RSQ distribution
            self.plot_rsq_distribution(pp = pp, 
                                    pars_df = pars_df,
                                    filename = pp_base_filename+'_pRF_RSQ_bestvox.'+fig_type,
                                    best_vox = True, 
                                    figsize=(8,5), 
                                    model_type = model_type)
            self.plot_rsq_distribution(pp = pp, 
                                    pars_df = pars_df,
                                    filename = pp_base_filename+'_pRF_RSQ.'+fig_type,
                                    best_vox = False, 
                                    figsize=(8,5), 
                                    model_type = model_type)

            # ## ECC
            self.plot_ecc_size(pp = pp, 
                                pars_df = pars_df,
                                filename = pp_base_filename+'_pRF_ECC_SIZE_FWHM_bestvox.'+fig_type,
                                best_vox = True, 
                                figsize=(8,5), 
                                model_type = model_type,
                                fwhmax = True)
            self.plot_ecc_size(pp = pp, 
                                pars_df = pars_df,
                                filename = pp_base_filename+'_pRF_ECC_SIZE_bestvox.'+fig_type,
                                best_vox = True, 
                                figsize=(8,5), 
                                model_type = model_type,
                                fwhmax = False)
            
        ## if more than one participant, plot for group
        if len(participant_list) > 1:

            ## RSQ distribution
            self.plot_rsq_group(pars_df = pars_df,
                                filename = base_filename+'_pRF_RSQ_bestvox.'+fig_type,
                                best_vox = True, 
                                figsize=(8,5), 
                                ROI_list = ROI_list,
                                model_type = model_type)
            
            self.plot_rsq_group(pars_df = pars_df,
                                filename = base_filename+'_pRF_RSQ.'+fig_type,
                                best_vox = False, 
                                figsize=(8,5), 
                                ROI_list = ROI_list,
                                model_type = model_type)

            # ## ECC
            # ## make binned df for group plot
            # avg_binned_df = [] 
            # n_bins = 10
            # for roi_name in ROI_list:
                
            #     cuts = pd.cut(ecc_size_df[ecc_size_df['ROI'] == roi_name]['ecc'], n_bins)

            #     # get binned average
            #     tmp_df = ecc_size_df[ecc_size_df['ROI'] == roi_name].groupby(['sj', cuts])['size'].mean().reset_index()

            #     # create average ecc range
            #     ecc_range = np.linspace(ecc_size_df[ecc_size_df['ROI'] == roi_name]['ecc'].min(),
            #                             ecc_size_df[ecc_size_df['ROI'] == roi_name]['ecc'].max(), n_bins)

            #     # get category codes
            #     ind_categ = tmp_df.ecc.cat.codes.values

            #     # replace ecc with average ecc of bin
            #     tmp_df.loc[:,'ecc'] = np.array([ecc_range[i] for i in ind_categ])
            #     tmp_df.loc[:, 'ROI'] = roi_name
    
            #     avg_binned_df.append(tmp_df)
            # avg_binned_df = pd.concat(avg_binned_df, ignore_index=True)
            
            # ## make binned eccentricity size plots
            # # for group
            # lm = sns.lmplot(avg_binned_df, x = 'ecc', y = 'size', hue = 'ROI', markers="x", 
            #                 palette = self.MRIObj.params['plotting']['ROI_pal'],
            #         scatter_kws={'alpha':0.2}, height=5, legend = False)

            # lm.fig.suptitle('encoding model %s\n(voxels used in decoding)'%model_type.upper(), fontsize=16)
            # lm.set_xlabels('Eccentricity', fontsize = 20, labelpad=18)
            # lm.set_ylabels('Size', fontsize = 20, labelpad=18)
            # lm.tick_params(axis='both', which='major', labelsize=18)

            # leg = lm.axes[0,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # for lh in leg.legendHandles: 
            #     lh.set_alpha(1)
            
            # lm.figure.savefig(base_filename+'_pRF_ECC-SIZE_bestvox.png')

    def plot_ground_truth_correlations(self, participant_list = [], reconstructed_stim_dict = None, lowres_DM_dict = None, 
                                            data_keys_dict = [],  ROI_list = ['V1'], mask_nan = True, return_df = False,
                                            figsize=(8,5), model_type = 'gauss_hrf', fig_type = 'png'):
        
        """
        Plot correlation of reconstructed stim with downsampled DM
        across runs for all participants and ROIs
        """

        # make dir to save estimates
        fig_dir = op.join(self.figures_pth, 'correlations_decoder')

        os.makedirs(fig_dir, exist_ok = True)
        print('saving figures in %s'%fig_dir)
        
        # and set base figurename 
        fig_id = 'sub-GROUP_task-FA_pRFmodel-{modname}_ground_truth_correlations.{fext}'.format(modname = model_type, 
                                                                                              fext = fig_type)
        # filename for figures 
        filename = op.join(fig_dir, fig_id)

        ## correlate average reconstructed stim (all trials) with downsampled DM
        ROIs_stim_dm_corr_df = self.DecoderObj.get_stim_visual_dm_correlation(participant_list = participant_list, 
                                                                            reconstructed_stim_dict = reconstructed_stim_dict, 
                                                                            lowres_DM_dict = lowres_DM_dict, 
                                                                            data_keys_dict = data_keys_dict,
                                                                            ROI_list = ROI_list, 
                                                                            mask_nan = mask_nan)
        
        ## plot correlation values per ROI
        fig, ax1 = plt.subplots(1,1, figsize=figsize)

        sns.barplot(y = 'corr', x = 'ROI',
                    data = ROIs_stim_dm_corr_df, 
                    palette = self.MRIObj.params['plotting']['ROI_pal'],
                    order = ROI_list,
                    hue = 'ROI', hue_order = ROI_list,
                    width=.8, linewidth = 1.5,
                    errorbar = 'se',
                    ax = ax1)

        sns.stripplot(data = ROIs_stim_dm_corr_df, 
                    x = 'ROI', y = 'corr', hue = 'sj',
                    order = ROI_list,
                    alpha = 1, zorder=100, jitter = .15,
                    palette = sns.color_palette("bright", len(participant_list)),
                    ax=ax1)
        ax1.legend(title = None, loc='upper right', fontsize = 'small')

        ax1.axhline(y=0, color='k', linestyle='--')

        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.set_ylim(-.03,.25)
        ax1.set_title('Ground truth correlation (single trial, across runs)',fontsize=14)
        ax1.set_ylabel(r'Point-biserial Correlation ($\it{r_{pb}}$)', fontsize = 16, labelpad = 15)
        ax1.set_xlabel('')
        fig.tight_layout()

        fig.savefig(filename)

        if return_df:
            return ROIs_stim_dm_corr_df
        
    def plot_runAVG_ground_truth_correlations(self, participant_list = [], group_stim_dict = None, group_refDM_dict = None, 
                                                    ROI_list = ['V1'], return_df = False,
                                                    figsize=(8,5), model_type = 'gauss_hrf', fig_type = 'png'):
                
        """
        Plot correlation of reconstructed stim with downsampled DM
        AVERAGED over runs for all participants and ROIs
        """

        # make dir to save estimates
        fig_dir = op.join(self.figures_pth, 'correlations_decoder')

        os.makedirs(fig_dir, exist_ok = True)
        print('saving figures in %s'%fig_dir)
        
        # and set base figurename 
        fig_id = 'sub-GROUP_task-FA_pRFmodel-{modname}_ground_truth_correlations_runAVG.{fext}'.format(modname = model_type, 
                                                                                                        fext = fig_type)
        # filename for figures 
        filename = op.join(fig_dir, fig_id)

        ## get correlation of across run average
        avg_stim_corr_df = self.DecoderObj.get_run_avg_stim_dm_correlation(participant_list = participant_list, 
                                                                            ROI_list = ROI_list, 
                                                                            group_stim_dict = group_stim_dict, 
                                                                            group_refDM_dict = group_refDM_dict)
        
        ## plot correlation values per ROI
        fig, ax1 = plt.subplots(1,1, figsize=figsize)

        sns.barplot(y = 'corr', x = 'ROI',
                    data = avg_stim_corr_df, 
                    palette = self.MRIObj.params['plotting']['ROI_pal'],
                    order = ROI_list,
                    hue = 'ROI', hue_order = ROI_list,
                    width=.8, linewidth = 1.5,
                    errorbar = 'se',
                    ax = ax1)

        sns.stripplot(data = avg_stim_corr_df, 
                    x = 'ROI', y = 'corr', hue = 'sj',
                    order = ROI_list,
                    alpha = 1, zorder=100, jitter = .15,
                    palette = sns.color_palette("bright", len(participant_list)),
                    ax=ax1)
        ax1.legend(title = None, loc='upper right', fontsize = 'medium')

        ax1.axhline(y=0, color='k', linestyle='--')

        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.set_ylim(-.03,.4)
        ax1.set_title('Ground truth correlation (single trial, average across runs)',fontsize=14)
        ax1.set_ylabel(r'Point-biserial Correlation ($\it{r_{pb}}$)', fontsize = 16, labelpad = 15)
        ax1.set_xlabel('')
        fig.tight_layout()

        fig.savefig(filename)

        if return_df:
            return avg_stim_corr_df
        

    def plot_trial_stim_movie(self, participant_list = [], ROI_list = ['V1'], model_type = 'gauss_hrf', 
                                    group_stim_dict = None, group_refDM_dict = None, avg_pp = False, fig_type = 'mp4',
                                    cmap = 'magma', annot = False, interval = 132, figsize = (8,5), fps = 6, dpi = 100):

        """
        Plot and save video animation of recontructed stim, 
        averaged across runs,
        for each ROI and participant
        """

        # make dir to save estimates
        fig_dir = op.join(self.figures_pth, 'movies_reconstructed_stim')
        # and set base figurename 
        fig_id = 'sub-GROUP_task-FA_pRFmodel-{modname}_decoded_stim'.format(modname = model_type)

        # base filename for figures 
        base_filename = op.join(fig_dir, fig_id)
    
        ## if we want to plot group average
        if len(participant_list) > 1 and avg_pp:

            os.makedirs(fig_dir, exist_ok = True)
            print('saving figures in %s'%fig_dir)

            # and average across participants (for plotting mainly)
            avg_stim_dict, avg_refDM_dict = self.DecoderObj.average_group_stim_glmsing_trials(participant_list = participant_list, 
                                                                                            ROI_list = ROI_list, 
                                                                                            group_stim_dict = group_stim_dict, 
                                                                                            group_refDM_dict = group_refDM_dict)
            
            ## for each ROI
            for roi_name in ROI_list:

                ani = self.make_trial_stim_movie(roi_name = roi_name, 
                                                            stim2plot = avg_stim_dict[roi_name], 
                                                            dm2plot = avg_refDM_dict, 
                                                            vmin = np.nanquantile(avg_stim_dict[roi_name].values.ravel(),
                                                                                .01), 
                                                            vmax = np.nanquantile(avg_stim_dict[roi_name].values.ravel(),
                                                                                .99), 
                                                            cmap = cmap,
                                                            annot = annot, 
                                                            interval = interval, 
                                                            figsize = figsize, 
                                                            name = 'GROUP', 
                                                            frame_inds = None, 
                                                            filename = base_filename+'_ROI-{rname}.{fext}'.format(rname = roi_name,
                                                                                                                fext = fig_type), 
                                                            fps = fps, 
                                                            dpi = dpi)
                
        else:
            ## now actually plot
            for pp in participant_list:
                
                # make dir to save estimates
                pp_fig_dir = op.join(fig_dir, 'sub-{sj}'.format(sj = pp))
                pp_fig_id = fig_id.replace('sub-GROUP', 'sub-{sj}'.format(sj = pp)) 
            
                os.makedirs(pp_fig_dir, exist_ok = True)
                print('saving figures in %s'%pp_fig_dir)
                
                # base filename for figures 
                pp_base_filename = op.join(pp_fig_dir, pp_fig_id)

                ## for each ROI
                for roi_name in ROI_list:

                    ani = self.make_trial_stim_movie(roi_name = roi_name, 
                                                                stim2plot = group_stim_dict[roi_name]['sub-{sj}'.format(sj = pp)], 
                                                                dm2plot = group_refDM_dict['sub-{sj}'.format(sj = pp)], 
                                                                vmin = np.nanquantile(group_stim_dict[roi_name]['sub-{sj}'.format(sj = pp)].values.ravel(),
                                                                                    .01), 
                                                                vmax = np.nanquantile(group_stim_dict[roi_name]['sub-{sj}'.format(sj = pp)].values.ravel(),
                                                                                    .99), 
                                                                cmap = cmap,
                                                                annot = annot, 
                                                                interval = interval, 
                                                                figsize = figsize, 
                                                                name = 'sub-{sj}'.format(sj = pp), 
                                                                frame_inds = None, 
                                                                filename = pp_base_filename+'_ROI-{rname}.{fext}'.format(rname = roi_name,
                                                                                                                        fext = fig_type), 
                                                                fps = fps, 
                                                                dpi = dpi)
                
    def make_trial_stim_movie(self, roi_name = 'V1', stim2plot = None, dm2plot = None, vmin = 0, vmax = .29, cmap = 'magma',
                                    annot = False, interval = 132, figsize = (8,5), name = 'Group', frame_inds = None, 
                                    filename = None, fps=24, dpi=100):

        """
        Create animation of recontructed stim,
        for a given ROI of participant/group
        """

        # if we didnt specify frame indices
        if frame_inds is None:
            frame_inds = range(dm2plot.shape[0])

        ## initialize base figure
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize = figsize)

        fig.suptitle('Reconstructed stimulus (%s), ROI - %s'%(name, roi_name), fontsize=14)

        ## create animation      
        ani = FuncAnimation(fig, self.update_movie_frame, 
                            frames = frame_inds, 
                            fargs = (stim2plot.stack('y', future_stack=True), 
                                    dm2plot,
                                    axes,
                                    vmin, 
                                    vmax, 
                                    cmap, 
                                    annot),
                            interval=interval)
        
        if filename is None:
            return ani
        else:
            ani.save(filename=filename, writer="ffmpeg", fps=fps, dpi=dpi) # save mp4 file+

    ## set function to update frames
    def update_movie_frame(self, frame, stim_arr = [], dm_list = [], axes = [],
                                vmin = 0, vmax = .4, cmap = 'plasma', annot = False,
                                line_color = 'green', alpha = .5, title = ''):
        
        # clear axis of fig
        axes[0].clear() 
        axes[1].clear() 

        # DMs
        # attend left
        axes[1].imshow(dm_list[frame].T, cmap = 'binary_r', vmax = 1.5)
        axes[1].vlines(3.5, -.5, 7.5, linestyles='dashed', color=line_color, alpha = alpha)
        axes[1].hlines(3.5, -.5, 7.5, linestyles='dashed', color=line_color, alpha = alpha)

        # plot stim
        sns.heatmap(stim_arr.loc[frame], cmap = cmap, ax = axes[0], 
                    square = True, cbar = False,
                    annot=annot, annot_kws={"size": 7},
                    vmin = vmin, vmax = vmax, fmt='.2f')
        axes[0].vlines(4, 0, 8, linestyles='dashed', color=line_color, alpha = alpha)
        axes[0].hlines(4, 0, 8, linestyles='dashed', color=line_color, alpha = alpha)

    def barplot_mean_pix_intensity(self, pixel_df = None, ROI_list = ['V1'], error_bars = 'within', figsize=(8,5), filename = None):

        """
        Make barplot with attend vs unattend pixel values
        across ROIs

        """

        ## first get mean values
        df2plot = pixel_df.groupby(['ROI', 'sj', 'bar_type']).mean(numeric_only=True).reset_index()

        if error_bars == 'within':
            ## calculate within sub error bars 
            df2plot = self.MRIObj.beh_utils.calc_within_sub_sem(df_data = df2plot, 
                                                                        main_var = 'intensity', 
                                                                        conditions = ['ROI', 'bar_type'], 
                                                                        pp_key = 'sj')
            error_key = None
        else:
            error_key = ('se')

        ## make bar plot
        fig, ax1 = plt.subplots(1,1, figsize = figsize)

        v1 = sns.barplot(data = df2plot[df2plot['bar_type'] == 'att_bar'], 
                    x = 'ROI', y = 'intensity', width = .35,
                    order = ROI_list, hue = 'ROI', hue_order = ROI_list,
                    palette = self.MRIObj.params['plotting']['ROI_pal'],
                    estimator = np.mean, errorbar=error_key, ax=ax1)

        v2 = sns.barplot(data = df2plot[df2plot['bar_type'] == 'unatt_bar'], 
                    x = 'ROI', y = 'intensity', width = .35,
                    order = ROI_list, hue = 'ROI', hue_order = ROI_list,
                    palette = self.MRIObj.params['plotting']['ROI_pal'],
                    estimator = np.mean, errorbar=error_key, ax=ax1)

        ## change position of bars to make then not overlap
        factor = np.repeat([-1,1], len(ROI_list))
        new_value = .2

        for ind, patch in enumerate(ax1.patches):
            # we move the bar
            patch.set_x(patch.get_x() + new_value * factor[ind])

            if factor[ind] > 0: # if other condition bar
                patch.set_hatch('//')

            if error_key is not None:
                # also change the error bar location
                ax1.lines[ind].set_xdata(ax1.lines[ind].get_xdata() + new_value * factor[ind])

        handleA = mpatches.Patch(facecolor='w',edgecolor = 'k',label='Attended bar')
        handleB = mpatches.Patch( facecolor='w',edgecolor = 'k',label='Unattended bar',hatch = '//')

        #leg = ax1.legend(handles = [handleA,handleB],loc='center left', bbox_to_anchor=(1, 0.5))
        leg = ax1.legend(handles = [handleA,handleB],loc='upper right')

        frame = leg.get_frame()
        frame.set_facecolor('w')
        frame.set_edgecolor('k')

        v1.set(xlabel=None)
        v1.set(ylabel=None)
        plt.margins(x=0.075)
        ax1.tick_params(axis='both', labelsize=13)

        ax1.set_xlabel('ROI',fontsize = 16, labelpad = 15)
        ax1.set_ylabel('Mean drive [a.u.]',fontsize = 16, labelpad = 15)
        ax1.set_ylim(0.08,.2)

        ## add within sub error bars
        if error_bars == 'within':

            att_intensity = [df2plot[(df2plot['bar_type'] == 'att_bar') &\
                                    (df2plot['ROI'] == rname)].intensity.values.mean() for rname in ROI_list]
            att_sem = [df2plot[(df2plot['bar_type'] == 'att_bar') &\
                                (df2plot['ROI'] == rname)].SEM_intensity.values.mean() for rname in ROI_list]

            unatt_intensity = [df2plot[(df2plot['bar_type'] == 'unatt_bar') &\
                                    (df2plot['ROI'] == rname)].intensity.values.mean() for rname in ROI_list]
            unatt_sem = [df2plot[(df2plot['bar_type'] == 'unatt_bar') &\
                                (df2plot['ROI'] == rname)].SEM_intensity.values.mean() for rname in ROI_list]
            ## target bar
            ax1.errorbar(x = np.array(ax1.get_xticks()) - new_value, 
                        y = att_intensity, 
                        yerr = att_sem,
                        elinewidth = 3, capsize = 5, capthick=2,
                        zorder = 100, c='#545759', alpha=1, fmt='none')

            ## distractor bar
            ax1.errorbar(x = np.array(ax1.get_xticks()) + new_value, 
                        y = unatt_intensity, 
                        yerr = unatt_sem,
                        elinewidth = 3, capsize = 5, capthick=2,
                        zorder = 100, c='#545759', alpha=1, fmt='none')

        plt.tight_layout()

        ## save figure
        if filename is not None:
            fig.savefig(filename)

    def pointplot_mean_bar_configuration(self, pixel_df = None, ROI_list = ['V1'], error_bars = 'within', figsize=(15,5), filename = None,
                                            bars_pos_colors = {'crossed': '#1cad98', 'parallel': '#de921f'}):

        """
        Make pointplot with attend vs unattend pixel values
        per bar configuration and
        across ROIs

        """

        ## first get mean values
        df2plot = pixel_df.groupby(['ROI', 'sj', 'bar_type', 'bars_pos']).mean(numeric_only=True).reset_index()

        if error_bars == 'within':
            ## calculate within sub error bars 
            df2plot = self.MRIObj.beh_utils.calc_within_sub_sem(df_data = df2plot, 
                                                                        main_var = 'intensity', 
                                                                        conditions = ['ROI', 'bar_type', 'bars_pos'], 
                                                                        pp_key = 'sj')
            error_key = None
        else:
            error_key = ('se')

        ## make pointplots
            
        fig, axes = plt.subplots(nrows=1, ncols=len(ROI_list), figsize = figsize, sharey=True, sharex=True)

        for ind, roi_name in enumerate(ROI_list):
            
            # plot conditions for given ROI
            sns.pointplot(data = df2plot[df2plot['ROI'] == roi_name], 
                        y = 'intensity', hue = 'bars_pos', x = 'bar_type',
                        errorbar = error_key, order=['att_bar', 'unatt_bar'],
                        markersize=8, dodge = False, palette = bars_pos_colors,
                        ax = axes[ind], legend=False)

            axes[ind].set_title(roi_name, fontsize=14)
            axes[ind].set_xlabel('Bar type',fontsize = 16, labelpad = 15)
            axes[ind].tick_params(axis='both', labelsize=14)
            axes[ind].set_xticks([0, 1])
            axes[ind].set_xticklabels(['Target', 'Distractor'])

            ## add error bars
            if error_key is None:

                for e_ind, bar_cond in enumerate(['att_bar', 'unatt_bar']):
                    
                    ## PARALLEL
                    error_parallel_df = df2plot[(df2plot['ROI'] == roi_name) &\
                                                (df2plot['bars_pos'] == 'parallel') &\
                                                (df2plot['bar_type'] == bar_cond)]
                    
                    axes[ind].errorbar(x = axes[ind].get_xticks()[e_ind], 
                                        y = [error_parallel_df.intensity.values.mean()], 
                                        yerr = [error_parallel_df.SEM_intensity.values.mean()],
                                elinewidth = 2, capsize = 5,
                                zorder = 0, c = bars_pos_colors['parallel'], 
                                alpha=1, fmt='none')
                    
                    ## CROSSED
                    error_crossed_df = df2plot[(df2plot['ROI'] == roi_name) &\
                                                (df2plot['bars_pos'] == 'crossed') &\
                                                (df2plot['bar_type'] == bar_cond)]
                    
                    axes[ind].errorbar(x = axes[ind].get_xticks()[e_ind], 
                                        y = [error_crossed_df.intensity.values.mean()], 
                                        yerr = [error_crossed_df.SEM_intensity.values.mean()],
                                elinewidth = 2, capsize = 5,
                                zorder = 0, c = bars_pos_colors['crossed'], 
                                alpha=1, fmt='none')
                    ##

        axes[0].set_ylabel('Mean Drive [a.u.]',fontsize = 16, labelpad = 15)
        axes[0].set_ylim([.115, .18])
        axes[0].set_xlim([-.5, 1.5])

        plt.margins(x=0.075)

        #axes[0].set_title('Attended Bar Drive Distribution',fontsize=14)
        plt.subplots_adjust(wspace=0.02, hspace=0.02)

        handleA = mpatches.Patch(facecolor = bars_pos_colors['crossed'], edgecolor = 'k',label='Crossed', fill=True, linewidth=1)
        handleB = mpatches.Patch(facecolor = bars_pos_colors['parallel'],edgecolor = 'k',label='Parallel', fill=True,linewidth=1)

        leg = axes[ind].legend(handles = [handleA,handleB],loc='upper right', fontsize = 'medium',
                            title= 'Bar configuration', title_fontsize = 'medium')

        frame = leg.get_frame()
        frame.set_facecolor('w') 
        frame.set_edgecolor('k')

        plt.tight_layout()

        ## save figure
        if filename is not None:
            fig.savefig(filename)


    def plot_group_pixel_results(self, pixel_df = None,  ROI_list = ['V1'], error_bars = 'within', figsize=(8,5), 
                                        model_type = 'gauss_hrf', fig_type = 'png'):

        """
        Create and save several group plots
        comparing average pixel intensity values
        across conditions and ROIs
        """

        # make dir to save estimates
        fig_dir = op.join(self.figures_pth, 'pixel_drive')
        # and set base figurename 
        fig_id = 'sub-GROUP_task-FA_pRFmodel-{modname}_pixel_intensity'.format(modname = model_type)

        # base filename for figures 
        base_filename = op.join(fig_dir, fig_id)

        os.makedirs(fig_dir, exist_ok = True)
        print('saving figures in %s'%fig_dir)

        ## first make plot with attend vs unattend pixel values
        # across ROIs
        self.barplot_mean_pix_intensity(pixel_df = pixel_df, 
                                        ROI_list = ROI_list, 
                                        error_bars = error_bars, 
                                        figsize=(8,5), 
                                        filename = base_filename+'_attention_diff.{fext}'.format(fext = fig_type))

        ## make plot with attend vs unattend pixel values
        # across ROIs and per bar configuration
        self.pointplot_mean_bar_configuration(pixel_df = pixel_df, 
                                        ROI_list = ROI_list, 
                                        error_bars = error_bars, 
                                        figsize=(15,5), 
                                        filename = base_filename+'_attention_bar_configuration.{fext}'.format(fext = fig_type))



