import numpy as np

import cortex
import matplotlib.pyplot as plt

from prfpy.rf import gauss2D_iso_cart
from prfpy.stimulus import PRFStimulus2D

from FAM_utils import mri as mri_utils
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel, Norm_Iso2DGaussianModel

from matplotlib.backend_bases import MouseButton

class visualize_on_click:
    
    def __init__(self, exp_params, prf_pars_dict, 
                         prf_dm = [], max_ecc_ext = 5.5, pRF_data = [], 
                         pysub = 'hcp_999999', flatmap_height = 2048, full_figsize = (12, 8)):
        
        ## general experiment settings
        self.exp_params = exp_params
        
        ## prf estimates
        self.prf_pars_dict = prf_pars_dict
        
        ## data to be plotted 
        self.pRF_data = pRF_data
        
        ## figure settings
        self.flatmap_height = flatmap_height
        self.full_figsize = full_figsize
        
        ## create pycortex vars
        self.mask, extents = cortex.quickflat.utils.get_flatmask(pysub, height = self.flatmap_height)
        self.vc = cortex.quickflat.utils._make_vertex_cache(pysub, height = self.flatmap_height)

        self.mask_index = np.zeros(self.mask.shape)
        self.mask_index[self.mask] = np.arange(self.mask.sum())
        
        # other relevant settings
        self.TR = exp_params['mri']['TR']
        self.prf_model_type = exp_params['mri']['fitting']['pRF']['fit_model']

        # if fitting hrf
        self.fit_hrf = exp_params['mri']['fitting']['pRF']['fit_hrf'] 
        
        # set prf dm
        self.prf_dm = prf_dm
        ## set grid of possible points in downsampled space
        self.point_grid_2D = np.array(np.meshgrid(np.linspace(-1, 1, prf_dm.shape[0])*max_ecc_ext,
                                         np.linspace(1, -1, prf_dm.shape[0])*max_ecc_ext))

        
    
    def set_figure(self):

        # set figure grid 
        self.full_fig = plt.figure(constrained_layout = True, figsize = self.full_figsize)
        gs = self.full_fig.add_gridspec(4, 3)

        self.flatmap_ax = self.full_fig.add_subplot(gs[:2, :])

        self.prf_timecourse_ax = self.full_fig.add_subplot(gs[2, :2])
        self.fa_timecourse_ax = self.full_fig.add_subplot(gs[3, :2])

        self.prf_ax = self.full_fig.add_subplot(gs[2, 2])

        self.flatmap_ax.set_title('flatmap')
        self.fa_timecourse_ax.set_title('FA timecourse')
        self.prf_timecourse_ax.set_title('pRF timecourse')
        self.prf_ax.set_title('prf')
        
    
    def get_prf_model(self, vertex, model_type = 'css', fit_hrf = False):
        
        ## set stim
        # make stimulus object, which takes an input design matrix and sets up its real-world dimensions
        prf_stim = PRFStimulus2D(screen_size_cm = self.exp_params['monitor']['height'],
                                 screen_distance_cm = self.exp_params['monitor']['distance'],
                                 design_matrix = self.prf_dm,
                                 TR = self.TR)
        
        ## define different possible models
        # Gaussian model 
        gauss_model = Iso2DGaussianModel(stimulus = prf_stim,
                                         filter_predictions = True,
                                         filter_type = self.exp_params['mri']['filtering']['type'],
                                         filter_params = {'highpass': self.exp_params['mri']['filtering']['highpass'],
                                                         'add_mean': self.exp_params['mri']['filtering']['add_mean'],
                                                         'window_length': self.exp_params['mri']['filtering']['window_length'],
                                                         'polyorder': self.exp_params['mri']['filtering']['polyorder']}
                                        )
        
        # CSS model 
        css_model = CSS_Iso2DGaussianModel(stimulus = prf_stim,
                                         filter_predictions = True,
                                         filter_type = self.exp_params['mri']['filtering']['type'],
                                         filter_params = {'highpass': self.exp_params['mri']['filtering']['highpass'],
                                                         'add_mean': self.exp_params['mri']['filtering']['add_mean'],
                                                         'window_length': self.exp_params['mri']['filtering']['window_length'],
                                                         'polyorder': self.exp_params['mri']['filtering']['polyorder']}
                                        )
        
        # DN model
        dn_model =  Norm_Iso2DGaussianModel(stimulus = prf_stim,
                                            filter_predictions = True,
                                            filter_type = self.exp_params['mri']['filtering']['type'],
                                            filter_params = {'highpass': self.exp_params['mri']['filtering']['highpass'],
                                                            'add_mean': self.exp_params['mri']['filtering']['add_mean'],
                                                            'window_length': self.exp_params['mri']['filtering']['window_length'],
                                                            'polyorder': self.exp_params['mri']['filtering']['polyorder']}
                                        )

        ## if we fitted the hrf
        if fit_hrf:
            
            hrf_deriv = self.prf_pars_dict['hrf_derivative'][vertex] 
            hrf_disp = self.prf_pars_dict['hrf_dispersion'][vertex]
            
            # create it and add to model
            hrf = mri_utils.create_hrf(hrf_params=[1.0, hrf_deriv, hrf_disp], TR = self.TR)
            gauss_model.hrf = hrf
            css_model.hrf = hrf
            dn_model.hrf = hrf
            
        
        ### get prediction
        if model_type == 'css':
            model_fit = css_model.return_prediction(self.prf_pars_dict['x'][vertex], self.prf_pars_dict['y'][vertex],
                                                    self.prf_pars_dict['size'][vertex], 
                                                    self.prf_pars_dict['betas'][vertex],
                                                    self.prf_pars_dict['baseline'][vertex], 
                                                    self.prf_pars_dict['ns'][vertex])
            
        elif model_type == 'dn':
            model_fit = dn_model.return_prediction(self.prf_pars_dict['x'][vertex], self.prf_pars_dict['y'][vertex],
                                                    self.prf_pars_dict['size'][vertex], 
                                                    self.prf_pars_dict['betas'][vertex],
                                                    self.prf_pars_dict['baseline'][vertex], 
                                                    self.prf_pars_dict['sa'][vertex],
                                                    self.prf_pars_dict['ss'][vertex],
                                                    self.prf_pars_dict['nb'][vertex],
                                                    self.prf_pars_dict['sb'][vertex]) 


        else:
            model_fit = gauss_model.return_prediction(self.prf_pars_dict['x'][vertex], self.prf_pars_dict['y'][vertex],
                                                    self.prf_pars_dict['size'][vertex], 
                                                    self.prf_pars_dict['betas'][vertex],
                                                    self.prf_pars_dict['baseline'][vertex])  
            
        return model_fit[0], self.prf_pars_dict['r2'][vertex]

        
        
    def plot_prf_tc(self, axis, timecourse = None, plot_model = True):
        
        # plotting will be in seconds
        time_sec = np.linspace(0, len(timecourse)*self.TR,
                               num = len(timecourse)) 
        
        axis.plot(time_sec, timecourse,'k--', label = 'data')
        
        if plot_model:
            prediction, r2 = self.get_prf_model(self.vertex, model_type = self.prf_model_type, fit_hrf = self.fit_hrf)
            axis.plot(time_sec, prediction, c = 'red',lw=3,label='model R$^2$ = %.2f'%r2,zorder=1)
            print('model R$^2$ = %.2f'%r2)
            

        axis.set_xlabel('Time (s)')#,fontsize=20, labelpad=20)
        axis.set_ylabel('BOLD signal change (%)')#,fontsize=20, labelpad=10)
        axis.set_xlim(0, len(timecourse)*self.TR)
        #axis.legend(loc='upper left',fontsize=10)  # doing this to guarantee that legend is how I want it
        
        return axis
        
    
    def redraw_vertex_plots(self, vertex, refresh):
        
        self.vertex = vertex

        print(refresh)

        if refresh: # if we want to clean up timecourses
            self.fa_timecourse_ax.clear()
            self.prf_timecourse_ax.clear()
            
        self.prf_timecourse_ax = self.plot_prf_tc(self.prf_timecourse_ax, timecourse = self.pRF_data[vertex])


        prf = gauss2D_iso_cart(self.point_grid_2D[0],
                               self.point_grid_2D[1],
                               mu = (self.prf_pars_dict['x'][vertex], 
                                     self.prf_pars_dict['y'][vertex]),
                               sigma = self.prf_pars_dict['size'][vertex]) #, alpha=0.6)

        self.prf_ax.clear()
        self.prf_ax.imshow(prf, cmap='cubehelix')
        self.prf_ax.axvline(self.prf_dm.shape[0]/2, color='white', linestyle='dashed', lw=0.5)
        self.prf_ax.axhline(self.prf_dm.shape[1]/2, color='white', linestyle='dashed', lw=0.5)
        #prf_ax.set_title(f"x: {self.prf_pars_dict['x'][vertex]}, y: {self.prf_pars_dict['y'][vertex]}")

    #def polar_angle():
        
        
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
 