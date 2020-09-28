
import os
import numpy as np
import math
from psychopy import visual, tools

from utils import *


class PRFStim(object):
    def __init__(self, session, bar_width_ratio, grid_pos):
        
        """ Initializes a PRFStim object. 

        Parameters
        ----------
        session : exptools Session object
            A Session object (needed for metadata)
        bar_width_ratio : float
            Ratio of screen dim to use as bar width
        grid_pos : array
            List/array of grid positions within display, to select a subset that will be the bar
            
        """

        
        # general parameters
        self.session = session
        self.screen = self.session.win.size # screen res [hRes,vRes]
        
        self.bar_width_ratio = bar_width_ratio
        
        self.grid_pos = grid_pos

        self.condition_settings = self.session.settings['stimuli']['conditions']



    def draw(self, bar_midpoint_at_TR, bar_direction_at_TR, this_phase):
        
        """ Draw stimuli - pRF bar - for each trial 
        
        Parameters
        ----------
        bar_midpoint_at_TR : array
            List/array of bar midpoint positions [x,y] at that TR (trial)
        bar_direction_at_TR : str
            Direction of bar at that TR (trial)
            
        """
        
        ## define elements for bar
        
        # first define bar width in pixels (depends if vertical or horizontal bar pass)
        # and bounds for x and y positions
        
        if bar_direction_at_TR in np.array(['L-R','R-L']): # if horizontal bar pass
            
            bar_width_pix = self.screen[0]*self.bar_width_ratio 
            
            x_bounds = np.array([bar_midpoint_at_TR[0]-bar_width_pix/2,bar_midpoint_at_TR[0]+bar_width_pix/2])
            y_bounds = np.array([-self.screen[1]/2,self.screen[1]/2])

        elif bar_direction_at_TR in np.array(['U-D','D-U']): # if vertical bar pass
            
            bar_width_pix = self.screen[1]*self.bar_width_ratio

            x_bounds = np.array([-self.screen[0]/2,self.screen[0]/2])
            y_bounds = np.array([bar_midpoint_at_TR[1]-bar_width_pix/2, bar_midpoint_at_TR[1]+bar_width_pix/2])
            

            
        # check which grid positions are within bounds
        bar_ind = np.where(((self.grid_pos[...,0]>=min(x_bounds))&
                    (self.grid_pos[...,0]<=max(x_bounds))&
                    (self.grid_pos[...,1]>=min(y_bounds))&
                    (self.grid_pos[...,1]<=max(y_bounds))
                ))[0]
        
        # element positions (#elements,(x,y))
        self.element_positions = self.grid_pos[bar_ind]

       
        # number of bar elements
        self.num_elements = self.element_positions.shape[0]
        
        # element sizes
        element_sizes_px = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['element_size'], self.session.monitor) # in pix
        self.element_sizes = np.ones((self.num_elements)) * element_sizes_px 
        
        # element spatial frequency
        element_sfs_pix = tools.monitorunittools.deg2pix(self.condition_settings[this_phase]['element_sf'], self.session.monitor) # (transform cycles/degree to cycles/pixel)
        self.element_sfs = np.ones((self.num_elements)) * element_sfs_pix
        
        # element orientation (half ori1, half ori2)
        ori_arr = np.concatenate((np.ones((math.floor(self.num_elements * .5))) * self.condition_settings[this_phase]['element_ori'][0], 
                                  np.ones((math.ceil(self.num_elements * .5))) * self.condition_settings[this_phase]['element_ori'][1])) 
        
        #self.element_orientations = ori_arr
        if this_phase in ('ori_left','ori_right'):
            self.element_orientations = jitter(ori_arr,max_val=10,min_val=5) # add some jitter to the orientations, needs to be bigger to allow for orientation differentiation
        else:
            self.element_orientations = jitter(ori_arr) # add some jitter to the orientations
        
        np.random.shuffle(self.element_orientations) # shuffle the orientations
        
        # element colors
        self.colors = np.ones((int(np.round(self.num_elements)),3)) * np.array(self.condition_settings[this_phase]['element_color'])
        
        # define bar array element
        if this_phase in ('color_green','color_red'):
            
            # to make colored gabor, need to do it a bit differently (psychopy forces colors to be opposite)
            grating = visual.filters.makeGrating(res=math.ceil(element_sizes_px))

            # initialise a 'black' texture
            colored_grating = np.ones((math.ceil(element_sizes_px), math.ceil(element_sizes_px), 3)) * -1.0
            
            # replace the red/green channel with the grating
            if this_phase=='color_red': 
                colored_grating[..., 0] = grating 
            else:
                colored_grating[..., 1] = grating 

            self.session.element_array = visual.ElementArrayStim(win=self.session.win, nElements = self.num_elements,
                                                                    units='pix', elementTex=colored_grating, elementMask='gauss',
                                                                    sizes = self.element_sizes, sfs = self.element_sfs, 
                                                                    xys = self.element_positions, oris=self.element_orientations, 
                                                                    #colors = self.colors, 
                                                                    colorSpace = 'rgb') 


        else:
            self.session.element_array = visual.ElementArrayStim(win=self.session.win, nElements = self.num_elements,
                                                                    units='pix', elementTex='sin', elementMask='gauss',
                                                                    sizes = self.element_sizes, sfs = self.element_sfs, 
                                                                    xys = self.element_positions, oris=self.element_orientations, 
                                                                    colors = self.colors, 
                                                                    colorSpace = 'rgb') 
        self.session.element_array.draw()





