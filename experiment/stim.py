
import os
import numpy as np
import math
from psychopy import visual, tools

from utils import *


class Stim(object):

    def __init__(self, session, bar_width_ratio, grid_pos):
        
        """ Initializes a Stim object. 

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
        
        self.bar_width_pix = self.session.screen*bar_width_ratio
        
        self.grid_pos = grid_pos

        self.condition_settings = self.session.settings['stimuli']['conditions']

        
        ## define elements array, with all grid positions
        # by using background characteristics and contrast = 0 (so it will be hidden)

        # element positions (#elements,(x,y))
        self.element_positions = self.grid_pos 

        # total number of elements (all grid points)
        self.num_elements = self.element_positions.shape[0]

        # set element contrasts (initially will be 0 because we don't want to see elements)
        self.element_contrast = np.ones((self.num_elements)) * self.condition_settings['background']['element_contrast']

        # element sizes
        element_sizes_px = tools.monitorunittools.deg2pix(self.condition_settings['background']['element_size'], self.session.monitor) # in pix
        self.element_sizes = np.ones((self.num_elements)) * element_sizes_px 
        
        # element background spatial frequency 
        element_sfs_pix = tools.monitorunittools.deg2pix(self.condition_settings['background']['element_sf'], self.session.monitor) # (transform cycles/degree to cycles/pixel)
        self.element_sfs = np.ones((self.num_elements)) * element_sfs_pix
        
        # element background orientation (half ori1, half ori2)
        ori_arr = np.concatenate((np.ones((math.floor(self.num_elements * .5))) * self.condition_settings['background']['element_ori'][0], 
                                  np.ones((math.ceil(self.num_elements * .5))) * self.condition_settings['background']['element_ori'][1])) 

        self.element_orientations = jitter(ori_arr,
                                            max_val = self.condition_settings['background']['ori_jitter_max'],
                                            min_val = self.condition_settings['background']['ori_jitter_min']) 

        np.random.shuffle(self.element_orientations) # shuffle the orientations

        # element background colors
        self.colors = np.ones((int(np.round(self.num_elements)),3)) * np.array(self.condition_settings['background']['element_color'])

        # element background texture
        self.elementTex = 'sin'

        ### save these initial settings in dictionary, to be used later when display is updated
        self.background_dict = {'nElements': self.num_elements,
                               'elementTex': self.elementTex,
                               'sfs': self.element_sfs,
                               'xys': self.element_positions,
                               'oris': self.element_orientations,
                               'contrs': self.element_contrast,
                               'colors': self.colors,
                               'sizes': self.element_sizes
                               }

        # define background element array, for all possible display positions
        self.session.background_array = visual.ElementArrayStim(win = self.session.win, nElements = self.background_dict['nElements'],
                                                            units = 'pix', elementTex = self.background_dict['elementTex'], elementMask = 'gauss',
                                                            sizes = self.background_dict['sizes'], sfs = self.background_dict['sfs'], 
                                                            xys = self.background_dict['xys'], oris = self.background_dict['oris'],
                                                            contrs = self.background_dict['contrs'], 
                                                            colors = self.background_dict['colors'], 
                                                            colorSpace = 'rgb') 

        

class PRFStim(Stim):

    def __init__(self, session, bar_width_ratio, grid_pos):

        # need to initialize parent class (Stim)
        super().__init__(session=session, bar_width_ratio=bar_width_ratio, grid_pos=grid_pos)


        # define a bar element array, for now same settings as background,
        # will be updated later
        self.session.bar0_array = visual.ElementArrayStim(win = self.session.win)
        # self.session.bar0_array = visual.ElementArrayStim(win = self.session.win, nElements = self.background_dict['nElements'],
        #                                                     units = 'pix', elementTex = self.background_dict['elementTex'], elementMask = 'gauss',
        #                                                     sizes = self.background_dict['sizes'], sfs = self.background_dict['sfs'], 
        #                                                     xys = self.background_dict['xys'], oris = self.background_dict['oris'],
        #                                                     contrs = self.background_dict['contrs'], 
        #                                                     colors = self.background_dict['colors'], 
        #                                                     colorSpace = 'rgb')



    def draw(self, bar_midpoint_at_TR, bar_direction_at_TR, this_phase):
        
        """ Draw stimuli - pRF bar - for each trial 
        
        Parameters
        ----------
        bar_midpoint_at_TR : array
            List/array of bar midpoint positions [x,y] at that TR (trial)
        bar_direction_at_TR : str
            Direction of bar at that TR (trial)
            
        """
        
        # get bar and background positions
        position_dictionary = get_bar_positions(self.grid_pos,bar_midpoint_at_TR, bar_direction_at_TR,
                                                self.bar_width_pix, screen = self.session.screen, num_bar = 1)


        # update background elements
        # self.session.background_array =  update_elements(ElementArrayStim = self.session.background_array, 
        #                                                 condition_settings = self.condition_settings, 
        #                                                 this_phase = 'background', 
        #                                                 elem_positions = position_dictionary['background']['xys'], 
        #                                                 nElements = position_dictionary['background']['nElements'],
        #                                                 monitor = self.session.monitor, 
        #                                                 screen = self.session.screen)

        # update bar elements
        self.session.bar0_array =  update_elements(ElementArrayStim = self.session.background_array, 
                                                    condition_settings = self.condition_settings, 
                                                    this_phase = this_phase, 
                                                    elem_positions = position_dictionary['bar0']['xys'], 
                                                    nElements = position_dictionary['bar0']['nElements'],
                                                    monitor = self.session.monitor, 
                                                    screen = self.session.screen)


        # actually draw
        # self.session.background_array.draw()
        self.session.bar0_array.draw()


        






