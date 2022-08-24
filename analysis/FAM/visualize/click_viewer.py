import numpy as np
import os
import os.path as op

import cortex
import matplotlib.pyplot as plt

from prfpy.rf import gauss2D_iso_cart

from FAM.utils import mri as mri_utils

from matplotlib.backend_bases import MouseButton


class visualize_on_click:

    def __init__(self, MRIObj, pRFModelObj = None, FAModelObj = None,
                        pRF_data = [], FA_data = [],
                        prf_dm = [], max_ecc_ext = 5.5,
                        pysub = 'hcp_999999', flatmap_height = 2048, full_figsize = (12, 8)):

        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
        pRFModelObj: pRF Model object
            object from one of the classes defined in prf_model.pRF_model
            
        """

        # set data object to use later on
        self.MRIObj = MRIObj

        # Load pRF and model object
        self.pRFModelObj = pRFModelObj
        self.FAModelObj = FAModelObj

        ## data to be plotted 
        self.pRF_data = pRF_data
        self.FA_data = FA_data

        ## figure settings
        self.flatmap_height = flatmap_height
        self.full_figsize = full_figsize
        self.images = {}
        
        ## create pycortex vars
        self.mask, extents = cortex.quickflat.utils.get_flatmask(pysub, height = self.flatmap_height)
        self.vc = cortex.quickflat.utils._make_vertex_cache(pysub, height = self.flatmap_height)

        self.mask_index = np.zeros(self.mask.shape)
        self.mask_index[self.mask] = np.arange(self.mask.sum())

        # set prf dm
        self.prf_dm = prf_dm

        ## set grid of possible points in downsampled space
        self.point_grid_2D = np.array(np.meshgrid(np.linspace(-1, 1, prf_dm.shape[0]) * max_ecc_ext,
                                         np.linspace(1, -1, prf_dm.shape[0]) * max_ecc_ext))

    def set_figure(self, participant, 
                    task2viz = 'both', ses = 'ses-mean', run_type = 'mean', pRFmodel_name = None):

        """
        Set base figure with placeholders 
        for relevant plots

        Parameters
        ----------
        task2viz : str
            task to visualize, can be 'prf', 'FA' or 'both'
            
        """

        ## set task of interest
        self.task2viz = task2viz

        ## set participant ID
        self.participant = participant
        self.session = ses
        self.pRFmodel_name = pRFmodel_name

        ## load model and prf estimates for that participant
        self.pp_prf_est_dict, self.pp_prf_models = self.pRFModelObj.load_pRF_model_estimates(participant, ses = self.session, run_type = run_type, model_name = pRFmodel_name, iterative = True)

        # when loading, dict has key-value pairs stored,
        # need to convert it to make it in same format as when fitting on the spot
        self.pRF_keys = self.MRIObj.params['mri']['fitting']['pRF']['estimate_keys'][pRFmodel_name]
        
        if self.pRFModelObj.fit_hrf:
            self.pRF_keys = self.pRF_keys[:-1]+self.MRIObj.params['mri']['fitting']['pRF']['estimate_keys']['hrf']+['r2']

        ## set figure grid 
        self.full_fig = plt.figure(constrained_layout = True, figsize = self.full_figsize)

        if task2viz in ['both', 'FA', 'feature']:

            gs = self.full_fig.add_gridspec(4, 3)

            self.flatmap_ax = self.full_fig.add_subplot(gs[:2, :])

            self.prf_timecourse_ax = self.full_fig.add_subplot(gs[2, :2])
            self.fa_timecourse_ax = self.full_fig.add_subplot(gs[3, :2])

            self.prf_ax = self.full_fig.add_subplot(gs[2, 2])

            self.flatmap_ax.set_title('flatmap')
            self.fa_timecourse_ax.set_title('FA timecourse')
            self.prf_timecourse_ax.set_title('pRF timecourse')
            self.prf_ax.set_title('prf')
        
        elif task2viz in ['prf', 'pRF']:

            gs = self.full_fig.add_gridspec(4, 2)

            self.flatmap_ax = self.full_fig.add_subplot(gs[:3, :])

            self.prf_timecourse_ax = self.full_fig.add_subplot(gs[3, :1])

            self.prf_ax = self.full_fig.add_subplot(gs[3, 1])

            self.flatmap_ax.set_title('flatmap')
            self.prf_timecourse_ax.set_title('pRF timecourse')
            self.prf_ax.set_title('prf')


    def get_vertex_model_tc(self, vertex):

        """
        Get model estimates for that vertex

        Parameters
        ----------
        vertex : int
            vertex index
            
        """

        # if we fitted hrf, need to also get that from params
        # and set model array

        estimates_arr = np.stack((self.pp_prf_est_dict[val][vertex] for val in self.pRF_keys))
        
        # define spm hrf
        spm_hrf = self.pp_prf_models['sub-{sj}'.format(sj = self.participant)][self.session]['{name}_model'.format(name = self.pRFModelObj.model_type)].create_hrf(hrf_params = [1, 1, 0],
                                                                                                                    onset=self.pRFModelObj.hrf_onset)

        if self.pRFModelObj.fit_hrf:
            hrf = self.pp_prf_models[ 'sub-{sj}'.format(sj = self.participant)][self.session]['{name}_model'.format(name = self.pRFModelObj.model_type)].create_hrf(hrf_params = [1.0,
                                                                                                                                estimates_arr[-3],
                                                                                                                                estimates_arr[-2]],
                                                                                                                    onset=self.pRFModelObj.hrf_onset)
        
            self.pp_prf_models['sub-{sj}'.format(sj = self.participant)][self.session]['{name}_model'.format(name = self.pRFModelObj.model_type)].hrf = hrf

            model_arr = self.pp_prf_models['sub-{sj}'.format(sj = self.participant)][self.session]['{name}_model'.format(name = self.pRFModelObj.model_type)].return_prediction(*list(estimates_arr[:-3]))
        
        else:
            self.pp_prf_models['sub-{sj}'.format(sj = self.participant)][self.session]['{name}_model'.format(name = self.pRFModelObj.model_type)].hrf = spm_hrf

            model_arr = self.pp_prf_models['sub-{sj}'.format(sj = self.participant)][self.session]['{name}_model'.format(name = self.pRFModelObj.model_type)].return_prediction(*list(estimates_arr[:-1]))

            
        return model_arr[0], estimates_arr[-1]



    def plot_prf_tc(self, axis, timecourse = None, plot_model = True):

        """
        plot pRF timecourse for model and data

        Parameters
        ----------
        timecourse : arr
            data time course
            
        """
        
        # plotting will be in seconds
        time_sec = np.linspace(0, len(timecourse) * self.MRIObj.TR,
                               num = len(timecourse)) 
        
        axis.plot(time_sec, timecourse,'k--', label = 'data')
        
        if plot_model:
            prediction, r2 = self.get_vertex_model_tc(self.vertex)
            axis.plot(time_sec, prediction, c = 'red',lw=3,label='model R$^2$ = %.2f'%r2,zorder=1)
            print('pRF model R$^2$ = %.2f'%r2)
            
        axis.set_xlabel('Time (s)')#,fontsize=20, labelpad=20)
        axis.set_ylabel('BOLD signal change (%)')#,fontsize=20, labelpad=10)
        axis.set_xlim(0, len(timecourse)*self.MRIObj.TR)
        #axis.legend(loc='upper left',fontsize=10)  # doing this to guarantee that legend is how I want it
        
        return axis

    def plot_fa_tc(self, axis, timecourse = None, plot_model = True):

        """
        plot FA timecourse for model and data ---> NOT IMPLEMENTED

        Parameters
        ----------
        timecourse : arr
            data time course
            
        """
        
        # plotting will be in seconds
        time_sec = np.linspace(0, len(timecourse)*self.MRIObj.TR,
                               num = len(timecourse)) 
        
        axis.plot(time_sec, timecourse,'k--', label = 'data')
        
        if plot_model:
            axis.plot(time_sec, self.FA_model[self.vertex], c = 'blue',lw=3)
            #print('FA model R$^2$ = %.2f'%r2)
            

        axis.set_xlabel('Time (s)')#,fontsize=20, labelpad=20)
        axis.set_ylabel('BOLD signal change (%)')#,fontsize=20, labelpad=10)
        axis.set_xlim(0, len(timecourse)*self.MRIObj.TR)
        #axis.legend(loc='upper left',fontsize=10)  # doing this to guarantee that legend is how I want it
        
        return axis
        
    
    def redraw_vertex_plots(self, vertex, refresh):

        """
        redraw vertex
            
        """
        
        self.vertex = vertex

        print(refresh)

        if refresh: # if we want to clean up timecourses
            self.prf_timecourse_ax.clear()
            if self.task2viz in ['both', 'FA', 'feature']:
                self.fa_timecourse_ax.clear()
            
        self.prf_timecourse_ax = self.plot_prf_tc(self.prf_timecourse_ax, timecourse = self.pRF_data[vertex])

        # plot fa data (and model if provided) 
        if self.FAModelObj:
            self.fa_timecourse_ax = self.plot_fa_tc(self.fa_timecourse_ax, timecourse = self.FA_data[vertex])
        elif self.FA_data:
            self.fa_timecourse_ax = self.plot_fa_tc(self.fa_timecourse_ax, timecourse = self.FA_data[vertex], plot_model = False) 

        prf = gauss2D_iso_cart(self.point_grid_2D[0],
                               self.point_grid_2D[1],
                               mu = (self.pp_prf_est_dict['x'][vertex], 
                                     self.pp_prf_est_dict['y'][vertex]),
                               sigma = self.pp_prf_est_dict['size'][vertex]) #, alpha=0.6)

        self.prf_ax.clear()
        self.prf_ax.imshow(prf, cmap='cubehelix')
        self.prf_ax.axvline(self.prf_dm.shape[0]/2, color='white', linestyle='dashed', lw=0.5)
        self.prf_ax.axhline(self.prf_dm.shape[1]/2, color='white', linestyle='dashed', lw=0.5)
        #prf_ax.set_title(f"x: {self.pp_prf_est_dict['x'][vertex]}, y: {self.pp_prf_est_dict['y'][vertex]}")

        # just to check if exponent values make sense
        if self.pRFModelObj.model_type == 'css':
            print('pRF exponent = %.2f'%self.pp_prf_est_dict['ns'][vertex])
        
        
    def onclick(self, event):
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

    def onkey(self, event):
        
        # clear flatmap axis
        self.flatmap_ax.clear()

        if event.key == '1':  # pRF rsq
            cortex.quickshow(self.images['pRF_rsq'], with_rois = False, with_curvature = True,
                        fig = self.flatmap_ax, with_colorbar = False)
            self.flatmap_ax.set_title('pRF rsq')
        elif (event.key == '2') & (self.task2viz in ['both', 'FA', 'feature']):  # FA rsq
            cortex.quickshow(self.images['FA_rsq'], with_rois = False, with_curvature = True,
                        fig = self.flatmap_ax, with_colorbar = False)
            self.flatmap_ax.set_title('FA rsq')
        elif event.key == '3':  # pRF eccentricity
            cortex.quickshow(self.images['ecc'], with_rois = False, with_curvature = True,
                        fig = self.flatmap_ax, with_colorbar = False)
            self.flatmap_ax.set_title('pRF eccentricity')
        elif event.key == '4':  # pRF Size
            cortex.quickshow(self.images['size_fwhmax'], with_rois = False, with_curvature = True,
                        fig = self.flatmap_ax, with_colorbar = False)
            self.flatmap_ax.set_title('pRF size (FWHMax)')
        elif event.key == '5':  # pRF PA
            cortex.quickshow(self.images['PA'], with_rois = False, with_curvature = True,
                        fig = self.flatmap_ax, with_colorbar = False)
            self.flatmap_ax.set_title('pRF PA')
        elif (event.key == '6') & (self.pRFmodel_name == 'css'):  # pRF exponent
            cortex.quickshow(self.images['ns'], with_rois = False, with_curvature = True,
                        fig = self.flatmap_ax, with_colorbar = False)
            self.flatmap_ax.set_title('pRF exponent')      
        plt.draw()


            
