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

import cortex

import subprocess

from FAM.visualize.viewer import Viewer

class BehViewer(Viewer):

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
        self.figures_pth = op.join(self.outputdir, 'behavioral')
        os.makedirs(self.figures_pth, exist_ok=True)

        # set font type for plots globally
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Helvetica'


    def plot_pRF_behavior(self, results_df = [], plot_group = True):

        """
        Plot behavioral results for pRF
        essentially acc and RT for the color categories
        
        """ 

        ## loop over participants in dataframe
        for pp in results_df['sj'].unique():

            pp_df = results_df[results_df['sj'] == pp]

            for ses in pp_df['ses'].unique():

                # plot ACCURACY and RT barplot and save
                fig, axs = plt.subplots(1, 2, figsize=(15,7.5))

                a = sns.barplot(x = 'color_category', y = 'accuracy', 
                                palette = self.bar_cond_colors,
                            data = pp_df, capsize=.2, ax = axs[0])
                #a.set(xlabel=None)
                a.set(ylabel=None)
                    
                axs[0].tick_params(labelsize=15)
                axs[0].set_xlabel('Color Category', fontsize=15, labelpad = 20)
                axs[0].set_ylabel('Accuracy',fontsize=15, labelpad = 15)
                axs[0].set_ylim(0,1)
                axs[0].set_title('pRF task accuraccy, {sj}_{ses}'.format(sj=pp, ses=ses),fontsize=18)

                b = sns.barplot(x = 'color_category', y = 'RT', 
                                palette = self.bar_cond_colors,
                            data = pp_df, capsize=.2, ax = axs[1])
                #a.set(xlabel=None)
                b.set(ylabel=None)
                    
                axs[1].tick_params(labelsize=15)
                axs[1].set_xlabel('Color Category', fontsize=15, labelpad = 20)
                axs[1].set_ylabel('Reaction Times (s)',fontsize=15, labelpad = 15)
                axs[1].set_ylim(0,1)
                axs[1].set_title('pRF task RT, {sj}_{ses}'.format(sj=pp, ses=ses),fontsize=18)

                fig.savefig(op.join(self.figures_pth,'{sj}_{ses}_task-pRF_RT_accuracy.png'.format(sj=pp, ses=ses)), dpi=100,bbox_inches = 'tight')

        ## Plot group results
        if plot_group:
            ## group df
            group_df = results_df.groupby(['sj', 'color_category'])['accuracy', 'RT'].mean().reset_index()

            # plot ACCURACY and RT barplot and save
            fig, axs = plt.subplots(1, 2, figsize=(15,7.5))

            a = sns.boxplot(x = 'color_category', y = 'accuracy', 
                            palette = self.bar_cond_colors,
                            data = group_df, ax = axs[0])
            #a.set(xlabel=None)
            a.set(ylabel=None)
                
            axs[0].tick_params(labelsize=15)
            axs[0].set_xlabel('Color Category', fontsize=15, labelpad = 20)
            axs[0].set_ylabel('Accuracy',fontsize=15, labelpad = 15)
            axs[0].set_ylim(0,1)
            axs[0].set_title('pRF task accuraccy', fontsize=18)

            b = sns.boxplot(x = 'color_category', y = 'RT', 
                            palette = self.bar_cond_colors,
                        data = group_df, ax = axs[1])
            #a.set(xlabel=None)
            b.set(ylabel=None)
                
            axs[1].tick_params(labelsize=15)
            axs[1].set_xlabel('Color Category', fontsize=15, labelpad = 20)
            axs[1].set_ylabel('Reaction Times (s)',fontsize=15, labelpad = 15)
            axs[1].set_ylim(0,1)
            axs[1].set_title('pRF task RT', fontsize=18)

            fig.savefig(op.join(self.figures_pth,'sub-GROUP_task-pRF_RT_accuracy.png'), dpi=100,bbox_inches = 'tight')


    def plot_FA_behavior(self, att_RT_df = None, acc_df = None, participant_list = []):

        """
        Plot behavioral results for FA
        (accuracy and RT for different ecc/distances)
        
        """ 

        # set generic filename
        filename = op.join(self.figures_pth, 'sub-{sj}_task-FA_{data_type}.png')

        ## loop over participants in dataframe
        for pp in participant_list:

            ## RT over ecc
            self.plot_FA_RTecc(att_RT_df = att_RT_df, 
                            sub_id = pp,
                            filename = filename.format(sj = pp, 
                                                       data_type = 'RT_ECC'), 
                            figsize = (8,5), 
                            ecc_colors=['#006e7f', '#f8cb2e', '#ee5007'])
            
            ## RT over accuracy
            self.plot_FA_RTdist(att_RT_df = att_RT_df, 
                            sub_id = pp,
                            filename = filename.format(sj = pp, 
                                                       data_type = 'RT_DIST'), 
                            figsize = (8,5), 
                            cmap = 'magma')
            # repeat but by swapping x-axis and hue around
            self.plot_FA_RTdist(att_RT_df = att_RT_df, 
                            sub_id = pp,
                            filename = filename.format(sj = pp, 
                                                       data_type = 'RT_DIST2'), 
                            figsize = (8,5), 
                            ecc_colors=['#006e7f', '#f8cb2e', '#ee5007'])
            
        ## also plot group

        self.plot_FA_RTecc(att_RT_df = att_RT_df, 
                        sub_id = None,
                        filename = filename.format(sj = 'GROUP', 
                                                    data_type = 'RT_ECC'), 
                        figsize = (8,5), 
                        ecc_colors=['#006e7f', '#f8cb2e', '#ee5007'])
        
        self.plot_FA_RTdist(att_RT_df = att_RT_df, 
                        sub_id = None,
                        filename = filename.format(sj = 'GROUP',  
                                                    data_type = 'RT_DIST'), 
                        figsize = (8,5), 
                        cmap = 'magma')
        # repeat but by swapping x-axis and hue around
        self.plot_FA_RTdist(att_RT_df = att_RT_df, 
                        sub_id = None,
                        filename = filename.format(sj = 'GROUP',  
                                                    data_type = 'RT_DIST2'), 
                        figsize = (8,5), 
                        ecc_colors=['#006e7f', '#f8cb2e', '#ee5007'])
        
        # plot accuracy
        self.plot_FA_ACCecc(acc_df = acc_df, 
                        filename = filename.format(sj = 'GROUP', 
                                                    data_type = 'Accuracy_ECC_pp'), 
                        figsize = (8,5), 
                        ecc_colors=['#006e7f', '#f8cb2e', '#ee5007'],
                        per_pp = True)
        self.plot_FA_ACCecc(acc_df = acc_df, 
                        filename = filename.format(sj = 'GROUP', 
                                                    data_type = 'Accuracy_ECC_group'), 
                        figsize = (8,5), 
                        ecc_colors=['#006e7f', '#f8cb2e', '#ee5007'],
                        per_pp = False)

    def plot_FA_RTecc(self, att_RT_df = None, filename = None, figsize = (8,5), ecc_colors=['#006e7f', '#f8cb2e', '#ee5007'], 
                            sub_id = None):

        """
        For each attended ecc, 
        split data into unattended bar ecc,
        and plot mean RT

        Boxplot + swarmplot

        Parameters
        ----------
        att_RT_df: df
            behavioral dataframe from preproc class
        """

        # filter for correct trials only

        # make group plot
        if sub_id is None: 
            df2plot = att_RT_df[att_RT_df['correct'] == 1].groupby(['sj', 'bar_ecc_deg', 'unatt_bar_ecc_deg']).mean(numeric_only=True).reset_index()
        else:
            df2plot = att_RT_df[(att_RT_df['correct'] == 1) & (att_RT_df['sj'] == 'sub-{sj}'.format(sj = sub_id))]

        ## create figure
        fig, axes0 = plt.subplots(nrows=1, ncols=1, figsize = figsize)

        ## boxplot, to show distribution for all participants
        box_p = sns.boxplot(data = df2plot,
                y = 'RT', x = 'unatt_bar_ecc_deg', hue = 'bar_ecc_deg', palette = ecc_colors,
                linewidth=1.5, ax=axes0)
        for patch in box_p.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .7))
            patch.set_edgecolor((1, 1, 1, 1))
                                
        sns.swarmplot(data = df2plot, dodge=True,
                y = 'RT', x = 'unatt_bar_ecc_deg', hue = 'bar_ecc_deg', palette = ecc_colors,
                linewidth=1.5, legend=False, ax=axes0)
        axes0.set_xticks(ticks = np.arange(len(np.sort(df2plot.unatt_bar_ecc_deg.unique()))), 
                        labels = np.round(np.sort(df2plot.unatt_bar_ecc_deg.unique()), 2)
                        )
        handles, labels = box_p.get_legend_handles_labels()
        box_p.legend(handles, ['{eval} deg'.format(eval = label[:4]) for label in labels], title = 'Target bar ecc', 
                    loc='upper right', fontsize = 'small', title_fontsize= 'medium')

        axes0.set_ylabel('RT [s]', fontsize = 16, labelpad = 15)
        if sub_id is None:
            axes0.set_ylim([.5, 1.2])
        axes0.set_xlabel('Distractor bar ecc [deg]', fontsize = 16, labelpad = 15)
        axes0.set_title('Average RT Distribution',fontsize=14)
        axes0.tick_params(axis='both', labelsize=14)

        # if we provided filename, save
        if filename:
            fig.savefig(filename, bbox_inches='tight')
        else:
            return fig
        
    def plot_FA_ACCecc(self, acc_df = None, filename = None, figsize = (8,5), ecc_colors=['#006e7f', '#f8cb2e', '#ee5007'], per_pp = False):

        """
        For each attended ecc, 
        split data into unattended bar ecc,
        and plot accuracy

        Boxplot + swarmplot

        Parameters
        ----------
        acc_df: df
            behavioral dataframe from preproc class
        """

        df2plot = acc_df[acc_df['correct'] == 1]

        ## create figure

        fig, axes0 = plt.subplots(nrows=1, ncols=1, figsize = figsize)

        ## if we want to plot accurcy per participant
        if per_pp:
            sns.lineplot(data = df2plot, x = 'bar_ecc_deg', y = 'accuracy',
                        hue = 'sj', err_style='bars', errorbar='se', marker='o', ms=7, err_kws = {'capsize': 5},
                        linewidth=3, ax = axes0)
            axes0.set_xlabel('Target bar ecc [deg]', fontsize = 16, labelpad = 15)

        else:
            ## plot group distribution

            box_p = sns.boxplot(data = df2plot,
                    y = 'accuracy', x = 'unatt_bar_ecc_deg', hue = 'bar_ecc_deg', palette = ecc_colors,
                    linewidth=1.5, ax = axes0)
            for patch in box_p.patches:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, .7))
                patch.set_edgecolor((1, 1, 1, 1))
                                    
            sns.swarmplot(data = df2plot, dodge=True,
                    y = 'accuracy', x = 'unatt_bar_ecc_deg', hue = 'bar_ecc_deg', palette = ecc_colors,
                    linewidth=1.5, legend=False, ax = axes0)
            plt.xticks(ticks=np.arange(len(np.sort(df2plot.unatt_bar_ecc_deg.unique()))), 
                    labels = np.round(np.sort(df2plot.unatt_bar_ecc_deg.unique()), 2)
                    )

            handles, labels = box_p.get_legend_handles_labels()
            box_p.legend(handles, ['{eval} deg'.format(eval = label[:4]) for label in labels], title = 'Target bar ecc', 
                        loc='lower left', fontsize = 'small', title_fontsize= 'medium')

            axes0.set_xlabel('Distractor bar ecc [deg]', fontsize = 16, labelpad = 15)
            axes0.set_title('Accuracy Distribution',fontsize=14)

        axes0.set_ylabel('Accuracy [%]', fontsize = 16, labelpad = 15)
        axes0.set_ylim([75, 101])

        axes0.tick_params(axis='both', labelsize=14)

        # if we provided filename, save
        if filename:
            fig.savefig(filename, bbox_inches='tight')
        else:
            return fig

    def plot_FA_RTdist(self, att_RT_df = None, filename = None, figsize = (8,5), cmap = 'magma', sub_id = None, ecc_colors = None):

        """
        For each attended ecc (of parallel trials)
        and distance between bars,
        plot mean RT

        Lineplot

        Parameters
        ----------
        att_RT_df: df
            behavioral dataframe from preproc class
        """

        # filter for correct trials only
        # make group plot
        if sub_id is None: 
            df2plot_dist = att_RT_df[(att_RT_df['correct'] == 1) &\
                                    (att_RT_df['bars_pos'] == 'parallel')].groupby(['sj', 'bar_ecc_deg', 
                                                                                    'interbar_dist_deg']).mean(numeric_only=True).reset_index()
        else:
            df2plot_dist = att_RT_df[(att_RT_df['correct'] == 1)  &\
                                (att_RT_df['bars_pos'] == 'parallel') &\
                                (att_RT_df['sj'] == 'sub-{sj}'.format(sj = sub_id))]
        
        ## create figure
        fig, axes0 = plt.subplots(nrows=1, ncols=1, figsize = figsize)

        if ecc_colors is not None:
            ## lineplots to show the linear trends,    
            line_p = sns.lineplot(data = df2plot_dist,
                                y = 'RT', hue = 'bar_ecc_deg', x = 'interbar_dist_deg', palette = ecc_colors,
                                err_style='bars', errorbar='se', marker='o', ms=10, err_kws = {'capsize': 5},
                                linewidth=5, ax=axes0, legend=True)

            handles, labels = line_p.get_legend_handles_labels()
            line_p.legend(handles, ['{eval} deg'.format(eval = label[:4]) for label in labels], title = 'Target bar ecc', 
                        loc='upper left', fontsize = 'small', title_fontsize= 'medium')

            axes0.set_ylabel('RT [s]', fontsize = 16, labelpad = 15)
            if sub_id is None:
                axes0.set_ylim([.6, .9])
            axes0.set_xlabel('Inter-bar distance [deg]', fontsize = 16, labelpad = 15)
            axes0.set_title('Average RT per inter-bar distance',fontsize=14)
            axes0.tick_params(axis='both', labelsize=14)

        else:
            # create color palette
            dist_colors = self.MRIObj.beh_utils.create_palette(key_list = np.sort(df2plot_dist.interbar_dist_deg.unique()), 
                                                            cmap = cmap, 
                                                            num_col = None)

            ## lineplots to show the linear trends,    
            line_p = sns.lineplot(data = df2plot_dist,
                                y = 'RT', x = 'bar_ecc_deg', hue = 'interbar_dist_deg', palette = dist_colors,
                                err_style='bars', errorbar='se', marker='o', ms=10, err_kws = {'capsize': 5},
                                linewidth=5, ax=axes0, legend=True)

            handles, labels = line_p.get_legend_handles_labels()

            line_p.legend(handles, ['{eval} deg'.format(eval = label[:4]) for label in labels], title = 'Inter-bar distance', 
                        loc='upper left', fontsize = 'small', title_fontsize= 'medium')

            axes0.set_ylabel('RT [s]', fontsize = 16, labelpad = 15)
            if sub_id is None:
                axes0.set_ylim([.6, .9])
            axes0.set_xlabel('Target bar ecc [deg]', fontsize = 16, labelpad = 15)
            axes0.set_title('Average RT per inter-bar distance',fontsize=14)
            axes0.tick_params(axis='both', labelsize=14)

        # if we provided filename, save
        if filename:
            fig.savefig(filename, bbox_inches='tight')
        else:
            return fig

                        

