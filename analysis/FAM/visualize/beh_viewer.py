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

from FAM.utils import beh as beh_utils


class BehViewer:

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


    def plot_pRF_behavior(self, results_df = [], plot_group = True):

        """
        Plot behavioral results for pRF
        essentially acc and RT for the color categories
        
        """ 

        ## output path to save plots
        output_pth = op.join(self.outputdir, 'behavioral')

        # if output path doesn't exist, create it
        if not op.isdir(output_pth): 
            os.makedirs(output_pth)
        print('saving files in %s'%output_pth)

        ## loop over participants in dataframe
        for pp in results_df['sj'].unique():

            pp_df = results_df[results_df['sj'] == pp]

            for ses in pp_df['ses'].unique():

                # plot ACCURACY and RT barplot and save
                fig, axs = plt.subplots(1, 2, figsize=(15,7.5))

                a = sns.barplot(x = 'color_category', y = 'accuracy', 
                                palette = self.MRIObj.params['plotting']['cond_colors'],
                            data = pp_df, capsize=.2, ax = axs[0])
                #a.set(xlabel=None)
                a.set(ylabel=None)
                    
                axs[0].tick_params(labelsize=15)
                axs[0].set_xlabel('Color Category', fontsize=15, labelpad = 20)
                axs[0].set_ylabel('Accuracy',fontsize=15, labelpad = 15)
                axs[0].set_ylim(0,1)
                axs[0].set_title('pRF task accuraccy, {sj}_{ses}'.format(sj=pp, ses=ses),fontsize=18)

                b = sns.barplot(x = 'color_category', y = 'RT', 
                                palette = self.MRIObj.params['plotting']['cond_colors'],
                            data = pp_df, capsize=.2, ax = axs[1])
                #a.set(xlabel=None)
                b.set(ylabel=None)
                    
                axs[1].tick_params(labelsize=15)
                axs[1].set_xlabel('Color Category', fontsize=15, labelpad = 20)
                axs[1].set_ylabel('Reaction Times (s)',fontsize=15, labelpad = 15)
                axs[1].set_ylim(0,1)
                axs[1].set_title('pRF task RT, {sj}_{ses}'.format(sj=pp, ses=ses),fontsize=18)

                fig.savefig(op.join(output_pth,'{sj}_{ses}_task-pRF_RT_accuracy.png'.format(sj=pp, ses=ses)), dpi=100,bbox_inches = 'tight')

        ## Plot group results

        if plot_group:
            ## group df
            group_df = results_df.groupby(['sj', 'color_category'])['accuracy', 'RT'].mean().reset_index()

            # plot ACCURACY and RT barplot and save
            fig, axs = plt.subplots(1, 2, figsize=(15,7.5))

            a = sns.boxplot(x = 'color_category', y = 'accuracy', 
                            palette = self.MRIObj.params['plotting']['cond_colors'],
                            data = group_df, ax = axs[0])
            #a.set(xlabel=None)
            a.set(ylabel=None)
                
            axs[0].tick_params(labelsize=15)
            axs[0].set_xlabel('Color Category', fontsize=15, labelpad = 20)
            axs[0].set_ylabel('Accuracy',fontsize=15, labelpad = 15)
            axs[0].set_ylim(0,1)
            axs[0].set_title('pRF task accuraccy', fontsize=18)

            b = sns.boxplot(x = 'color_category', y = 'RT', 
                            palette = self.MRIObj.params['plotting']['cond_colors'],
                        data = group_df, ax = axs[1])
            #a.set(xlabel=None)
            b.set(ylabel=None)
                
            axs[1].tick_params(labelsize=15)
            axs[1].set_xlabel('Color Category', fontsize=15, labelpad = 20)
            axs[1].set_ylabel('Reaction Times (s)',fontsize=15, labelpad = 15)
            axs[1].set_ylim(0,1)
            axs[1].set_title('pRF task RT', fontsize=18)

            fig.savefig(op.join(output_pth,'sub-GROUP_task-pRF_RT_accuracy.png'), dpi=100,bbox_inches = 'tight')








            
