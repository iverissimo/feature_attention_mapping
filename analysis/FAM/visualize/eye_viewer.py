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

#from FAM.utils import beh as beh_utils


class EyeViewer:

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
            self.outputdir = op.join(self.MRIObj.derivatives_pth, 'plots', 'eyetracking')
        else:
            self.outputdir = outputdir
            
        # number of participants to plot
        self.nr_pp = len(self.MRIObj.sj_num)


    def plot_gaze_kde(self, df_gaze, filename, run = 1, task = 'FA',
                    conditions = ['green_horizontal','green_vertical','red_horizontal','red_vertical'],
                    screen = [1920,1080], downsample = 10, color = {'green_horizontal':(0,1,0),'green_vertical':(0,1,0),
                                                                'red_horizontal':(1,0,0),'red_vertical': (1,0,0)}):
        
        """ 
        NOT FUNCTIONING - HARD COPY FROM PREVIOUS ITERATION OF DESIGN
        TO BE CHANGED 

        plot kde per run
        
        Parameters
        ----------
        df_gaze : pd dataframe
            with gaze data
        filename : str
            absolute path to save plot
        run : int
            run to plot
        downsample : int
            value to downsample gaze data, to make it faster
        
        """
        # plot gaze density

        if task=='FA':

            fig, axs = plt.subplots(2, 2, sharex=True, figsize=(30,15))
            #fig.subplots_adjust(hspace = .25, wspace=.001)

            plt_counter = 0

            for i in range(2):

                for w in range(2): 

                    data_x = df_gaze.loc[(df_gaze['run'] == run) &
                                                (df_gaze['condition'] == conditions[plt_counter])]['gaze_x'].values[0]
                    data_y = df_gaze.loc[(df_gaze['run'] == run) &
                                                (df_gaze['condition'] == conditions[plt_counter])]['gaze_y'].values[0]

                    # turn string list to actual list (workaround for pandas)
                    if type(data_x) != list:

                        data_x = literal_eval(data_x)[::downsample] 
                        data_y = literal_eval(data_y)[::downsample]
                    
                    else:
                        data_x = data_x[::downsample]
                        data_y = data_y[::downsample]

                    # get mean gaze and std
                    mean_gaze, mean_std = mean_dist_deg(data_x, data_y)
                    
                    # downsample data to make kde faster
                    a = sns.kdeplot(ax = axs[i,w], x = data_x, y = data_y, fill = True, color = color[conditions[plt_counter]])
                    a.tick_params(labelsize=15)

                    axs[i][w].set_title(conditions[plt_counter],fontsize=18)
                    axs[i][w].text(10, 10, 'mean gaze distance from center = %.2f +/- %.2f dva'%(mean_gaze, mean_std),
                                fontsize = 15)

                    axs[i][w].set_ylim(0, screen[1])
                    axs[i][w].set_xlim(0, screen[0])

                    axs[i][w].axvline(screen[0]/2, lw=0.5, color='k',alpha=0.5)
                    axs[i][w].axhline(screen[1]/2, lw=0.5, color='k',alpha=0.5)

                    axs[i][w].add_artist(plt.Circle((screen[0]/2, screen[1]/2), radius=102, color='grey',alpha=0.5 , fill=False)) # add circle of 1dva radius, for reference 

                    plt_counter += 1
                    
        elif task=='pRF':

            fig, axs = plt.subplots(1, 1, sharex=True, figsize=(30,15))

            data_x = df_gaze.loc[(df_gaze['run'] == run)]['gaze_x'].values[0]
            data_y = df_gaze.loc[(df_gaze['run'] == run)]['gaze_y'].values[0]

            # turn string list to actual list (workaround for pandas)
            if type(data_x) != list:

                data_x = literal_eval(data_x)[::downsample] 
                data_y = literal_eval(data_y)[::downsample]
            
            else:
                data_x = data_x[::downsample]
                data_y = data_y[::downsample]

            # get mean gaze and std
            mean_gaze, mean_std = mean_dist_deg(data_x, data_y)
            
            # downsample data to make kde faster
            a = sns.kdeplot(ax = axs, x = data_x, y = data_y, fill = True, color = 'blue')
            a.tick_params(labelsize=15)

            axs.set_title('run-%s'%str(run),fontsize=18)
            axs.text(10, 10, 'mean gaze distance from center = %.2f +/- %.2f dva'%(mean_gaze, mean_std),
                            fontsize = 15)

            axs.set_ylim(0, screen[1])
            axs.set_xlim(0, screen[0])

            axs.axvline(screen[0]/2, lw=0.5, color='k',alpha=0.5)
            axs.axhline(screen[1]/2, lw=0.5, color='k',alpha=0.5)

            axs.add_artist(plt.Circle((screen[0]/2, screen[1]/2), radius=102, color='grey',alpha=0.5 , fill=False)) # add circle of 1dva radius, for reference 


        fig.savefig(filename)



    def plot_sacc_hist(self, df_sacc, filename, run = 1, task = 'FA', conditions = ['green_horizontal','green_vertical','red_horizontal','red_vertical'],
                    color = {'green_horizontal':(0,1,0),'green_vertical':(0,1,0),'red_horizontal':(1,0,0),'red_vertical': (1,0,0)}):
        
        
        """ NOT FUNCTIONING - HARD COPY FROM PREVIOUS ITERATION OF DESIGN
        TO BE CHANGED 
        
        
        plot saccade histogram
        
        Parameters
        ----------
        df_sacc : pd dataframe
            with saccade data
        filename : str
            absolute path to save plot
        run : int
            run to plot
        
        """
        # plot gaze density

        if task=='FA':

            fig, axs = plt.subplots(2, 2, sharex=True, figsize=(30,15))
            #fig.subplots_adjust(hspace = .25, wspace=.001)

            plt_counter = 0

            for i in range(2):

                for w in range(2):

                    amp = df_sacc.loc[(df_sacc['run'] == run) &
                                        (df_sacc['condition'] == conditions[plt_counter])]['expanded_amplitude'].values[0]

                    if amp == [0]: # if 0, then no saccade

                        amp = [np.nan]

                    a = sns.histplot(ax = axs[i,w], 
                                    x = amp,
                                    color = color[conditions[plt_counter]])
                    a.tick_params(labelsize=15)
                    a.set_xlabel('Amplitude (degrees)',fontsize=15, labelpad = 15)

                    axs[i][w].set_title(conditions[plt_counter],fontsize=18)
                    axs[i][w].axvline(0.5, lw=0.5, color='k',alpha=0.5,linestyle='--')
                    
                    # count number of saccades with amplitude bigger than 0.5 deg
                    sac_count = len(np.where(np.array(amp) >= 0.5)[0])
                    axs[i][w].text(0.7, 0.9,'%i saccades > 0.5deg'%(sac_count), 
                                ha='center', va='center', transform=axs[i][w].transAxes,
                                fontsize = 15)

                    plt_counter += 1
                    
                
        elif task=='pRF':

            fig, axs = plt.subplots(1, 1, sharex=True, figsize=(30,15))

            amp = df_sacc.loc[(df_sacc['run'] == run)]['expanded_amplitude'].values[0]

            if amp == [0]: # if 0, then no saccade

                amp = [np.nan]

            a = sns.histplot(ax = axs, 
                            x = amp,
                            color = 'blue')
            a.tick_params(labelsize=15)
            a.set_xlabel('Amplitude (degrees)',fontsize=15, labelpad = 15)

            axs.set_title('run-%s'%str(run),fontsize=18)
            axs.axvline(0.5, lw=0.5, color='k',alpha=0.5,linestyle='--')
            
            # count number of saccades with amplitude bigger than 0.5 deg
            sac_count = len(np.where(np.array(amp) >= 0.5)[0])
            axs.text(0.7, 0.9,'%i saccades > 0.5deg'%(sac_count), 
                            ha='center', va='center', transform=axs.transAxes,
                            fontsize = 15)


        fig.savefig(filename)


    