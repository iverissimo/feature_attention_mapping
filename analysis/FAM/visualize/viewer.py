import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

import ptitprince as pt # raincloud plots
import matplotlib.patches as mpatches
from  matplotlib.ticker import FuncFormatter

import cortex

import subprocess

from FAM.utils.plot import PlotUtils

class Viewer:

    def __init__(self, MRIObj, outputdir = None, pysub = 'hcp_999999', use_atlas = None):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
        outputdir: str
            path to save plots
        pysub: str
            basename of pycortex subject folder, where we drew all ROIs, sulci etc 
        use_atlas: str
            If we want to use atlas ROIs (ex: glasser, wang) or not [default].
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

        # pycortex subject
        self.pysub = pysub

        ## set variables useful when loading ROIs
        if use_atlas is None:
            self.plot_key = self.MRIObj.sj_space 
            self.annot_filename = ''
        else:
            self.plot_key = use_atlas
            self.annot_filename = self.MRIObj.atlas_annot[self.plot_key ]
        
        self.use_atlas = use_atlas

        # if we are using atlas ROIs, then can already load here and avoid further reloading
        if isinstance(self.use_atlas, str):
            ## get vertices for each relevant ROI
            self.ROIs_dict = self.MRIObj.mri_utils.get_ROIs_dict(sub_id = None, pysub = self.pysub, use_atlas = self.use_atlas, 
                                                            annot_filename = self.annot_filename, hemisphere = 'BH',
                                                            ROI_labels = self.MRIObj.params['plotting']['ROIs'][self.plot_key])

        # set some generic variables useful for plotting
        self.bar_cond_colors = self.MRIObj.params['plotting']['cond_colors']
        self.ROI_pallete = self.MRIObj.params['plotting']['ROI_pal']
        self.rsq_threshold_plot = self.MRIObj.params['plotting']['rsq_threshold']

        # initialize utilities class
        self.plot_utils = PlotUtils() 

    def plot_rsq(self, participant_list = [], group_estimates = {}, ses = 'mean',  run_type = 'mean',
                        model_name = 'gauss', task = 'pRF', figures_pth = None, vmin1 = 0, vmax1 = .8,
                        fit_hrf = True):
        
        # make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.outputdir, 'rsq')

        # save values per roi in dataframe
        avg_roi_df = pd.DataFrame()
        
        ## loop over participants in list
        for pp in participant_list:

            # make path to save sub-specific figures
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp))
            os.makedirs(sub_figures_pth, exist_ok=True)

            ## plot rsq values on flatmap surface ##
            fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_task-{tsk}_acq-{acq}_space-{space}_ses-{ses}_run-{run}_model-{model}_flatmap_RSQ.png'.format(sj=pp, tsk = task,
                                                                                                            acq = self.MRIObj.acq, space = self.MRIObj.sj_space,
                                                                                                            ses=ses, run = run_type, model = model_name))

            # if we fitted hrf, then add that to fig name
            if fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            self.plot_utils.plot_flatmap(group_estimates['sub-{sj}'.format(sj = pp)]['r2'], 
                                        pysub = self.pysub, cmap = 'hot', 
                                        vmin1 = vmin1, vmax1 = vmax1, 
                                        fig_abs_name = fig_name)

            ## get estimates per ROI
            pp_roi_df = self.MRIObj.mri_utils.get_estimates_roi_df(pp, estimates_pp = group_estimates['sub-{sj}'.format(sj = pp)], 
                                                ROIs_dict = self.ROIs_dict, 
                                                est_key = 'r2', model = model_name)

            #### plot distribution ###
            fig, ax1 = plt.subplots(1,1, figsize=(20,7.5), dpi=100, facecolor='w', edgecolor='k')

            v1 = pt.RainCloud(data = pp_roi_df, move = .2, alpha = .9,
                        x = 'ROI', y = 'value', pointplot = False, hue = 'ROI',
                        palette = self.ROI_pallete, ax = ax1)
            
            # quick fix for legen
            handles = [mpatches.Patch(color = self.ROI_pallete[k], label = k) for k in self.ROIs_dict.keys()]
            ax1.legend(loc = 'upper right',fontsize=8, handles = handles, title="ROIs")#, fancybox=True)

            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)

            plt.xlabel('ROI',fontsize = 20,labelpad=18)
            plt.ylabel('RSQ',fontsize = 20,labelpad=18)
            plt.ylim(0,1)
            fig.savefig(fig_name.replace('flatmap','violinplot'))

            ## concatenate average per participant, to make group plot
            avg_roi_df = pd.concat((avg_roi_df, pp_roi_df))

        # if we provided several participants, make group plot
        if len(participant_list) > 1:

            fig, ax1 = plt.subplots(1,1, figsize=(15,5), dpi=100, facecolor='w', edgecolor='k')

            v1 = sns.pointplot(data = avg_roi_df.groupby(['sj', 'ROI'])['value'].mean().reset_index(),
                                x = 'ROI', y = 'value', color = 'k', markers = 'D', #scale = 1, 
                                palette = self.ROI_pallete, order = self.ROIs_dict.keys(), 
                                dodge = False, join = False, ci=68, ax = ax1)
            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            sns.stripplot(data = avg_roi_df.groupby(['sj', 'ROI'])['value'].mean().reset_index(), 
                          x = 'ROI', y = 'value', #hue = 'sj', palette = sns.color_palette("husl", len(participant_list)),
                            order = self.ROIs_dict.keys(),
                            color="gray", alpha=0.5, ax=ax1)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)

            plt.xlabel('ROI',fontsize = 20,labelpad=18)
            plt.ylabel('RSQ',fontsize = 20,labelpad=18)
            plt.ylim(0,1)

            fig.savefig(op.join(figures_pth, op.split(fig_name)[-1].replace('flatmap','violinplot').replace('sub-{sj}'.format(sj = pp),'sub-GROUP')))



