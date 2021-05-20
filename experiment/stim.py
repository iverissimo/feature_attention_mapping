
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


        # define element arrays here, with settings of background
        # will be updated later when drawing

        # number of elements
        self.nElements = self.grid_pos.shape[0]

        # element positions
        self.element_positions = self.grid_pos

         # element sizes
        element_sizes_px = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['element_size'], self.session.monitor) 
        self.element_sizes = np.ones((self.nElements)) * element_sizes_px 

        # elements spatial frequency
        self.element_sfs = np.ones((self.nElements)) * self.condition_settings['background']['element_sf'] # in cycles/gabor width

        # element orientation (half ori1, half ori2)
        ori_arr = np.concatenate((np.ones((math.floor(self.nElements * .5))) * self.condition_settings['background']['element_ori'][0], 
                                  np.ones((math.ceil(self.nElements * .5))) * self.condition_settings['background']['element_ori'][1]))

        # add some jitter to the orientations
        self.element_ori = jitter(ori_arr,
                         max_val = self.condition_settings['background']['ori_jitter_max'],
                         min_val = self.condition_settings['background']['ori_jitter_min']) 

        np.random.shuffle(self.element_ori) # shuffle the orientations

        # element contrasts
        self.element_contrast =  np.ones((self.nElements)) * self.condition_settings['background']['element_contrast']

        # element colors 
        self.element_color = np.ones((int(np.round(self.nElements)),3)) * np.array(self.condition_settings['background']['element_color'])
    

        self.session.background_array = visual.ElementArrayStim(win = self.session.win, 
                                                                nElements = self.nElements,
                                                                units = 'pix', 
                                                                elementTex = 'sin', 
                                                                elementMask = 'gauss',
                                                                sizes = self.element_sizes, 
                                                                sfs = self.element_sfs, 
                                                                xys = self.element_positions, 
                                                                oris = self.element_ori,
                                                                contrs = self.element_contrast, 
                                                                colors = self.element_color, 
                                                                colorSpace = self.session.settings['stimuli']['colorSpace']) 

        self.session.bar0_array = visual.ElementArrayStim(win = self.session.win, 
                                                                nElements = self.nElements,
                                                                units = 'pix', 
                                                                elementTex = 'sin', 
                                                                elementMask = 'gauss',
                                                                sizes = self.element_sizes, 
                                                                sfs = self.element_sfs, 
                                                                xys = self.element_positions, 
                                                                oris = self.element_ori,
                                                                contrs = self.element_contrast, 
                                                                colors = self.element_color, 
                                                                colorSpace = self.session.settings['stimuli']['colorSpace']) 


class PRFStim(Stim):

    def __init__(self, session, bar_width_ratio, grid_pos):

        # need to initialize parent class (Stim)
        super().__init__(session=session, bar_width_ratio=bar_width_ratio, grid_pos=grid_pos)


    def draw(self, bar_midpoint_at_TR, bar_pass_direction_at_TR, this_phase, position_dictionary, orientation = True, background_contrast = 0):
        
        """ Draw stimuli - pRF bar - for each trial 
        
        Parameters
        ----------
        bar_midpoint_at_TR : array
            List/array of bar midpoint positions [x,y] at that TR (trial)
        bar_pass_direction_at_TR : str
            Direction of bar at that TR (trial)
        this_phase: str
            strings with name of condition to draw
        """
        

        # update background elements
        self.session.background_array = update_elements(ElementArrayStim = self.session.background_array,
                                                        condition_settings = self.condition_settings,
                                                        position_jitter = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['pos_jitter'], self.session.monitor),
                                                        orientation = orientation,
                                                        background_contrast = background_contrast,
                                                        this_phase = 'background', 
                                                        elem_positions = position_dictionary['background']['xys'], 
                                                        grid_pos = self.grid_pos,
                                                        monitor = self.session.monitor, 
                                                        screen = self.session.screen)

        if this_phase != 'background':
            # update bar elements
            self.session.bar0_array = update_elements(ElementArrayStim = self.session.bar0_array,
                                                        condition_settings = self.condition_settings, 
                                                        position_jitter = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['pos_jitter'], self.session.monitor), 
                                                        orientation = orientation,
                                                        this_phase = this_phase, 
                                                        elem_positions = position_dictionary['bar0']['xys'], 
                                                        grid_pos = self.grid_pos,
                                                        monitor = self.session.monitor, 
                                                        screen = self.session.screen)


        # actually draw
        self.session.background_array.draw()
        if this_phase != 'background':
            self.session.bar0_array.draw() 


        
class FeatureStim(Stim):

    def __init__(self, session, bar_width_ratio, grid_pos):

        # need to initialize parent class (Stim)
        super().__init__(session=session, bar_width_ratio=bar_width_ratio, grid_pos=grid_pos)

        self.session.bar1_array = visual.ElementArrayStim(win = self.session.win, 
                                                                nElements = self.nElements,
                                                                units = 'pix', 
                                                                elementTex = 'sin', 
                                                                elementMask = 'gauss',
                                                                sizes = self.element_sizes, 
                                                                sfs = self.element_sfs, 
                                                                xys = self.element_positions, 
                                                                oris = self.element_ori,
                                                                contrs = self.element_contrast, 
                                                                colors = self.element_color, 
                                                                colorSpace = self.session.settings['stimuli']['colorSpace']) 
        
        self.session.bar2_array = visual.ElementArrayStim(win = self.session.win, 
                                                                nElements = self.nElements,
                                                                units = 'pix', 
                                                                elementTex = 'sin', 
                                                                elementMask = 'gauss',
                                                                sizes = self.element_sizes, 
                                                                sfs = self.element_sfs, 
                                                                xys = self.element_positions, 
                                                                oris = self.element_ori,
                                                                contrs = self.element_contrast, 
                                                                colors = self.element_color, 
                                                                colorSpace = self.session.settings['stimuli']['colorSpace']) 
        
        self.session.bar3_array = visual.ElementArrayStim(win = self.session.win, 
                                                                nElements = self.nElements,
                                                                units = 'pix', 
                                                                elementTex = 'sin', 
                                                                elementMask = 'gauss',
                                                                sizes = self.element_sizes, 
                                                                sfs = self.element_sfs, 
                                                                xys = self.element_positions, 
                                                                oris = self.element_ori,
                                                                contrs = self.element_contrast, 
                                                                colors = self.element_color, 
                                                                colorSpace = self.session.settings['stimuli']['colorSpace']) 



    def draw(self, bar_midpoint_at_TR, bar_pass_direction_at_TR, this_phase, position_dictionary, orientation = True, drawing_ind = [0,1,2,3]):
        
        """ Draw stimuli - pRF bars - for each trial 
        
        Parameters
        ----------
        bar_midpoint_at_TR : array
            List/array of bar midpoint positions [x,y] at that TR (trial)
        bar_pass_direction_at_TR : str
            Direction of bar at that TR (trial)
        this_phase: arr
            List/arr of strings with condition names to draw
            
        """


        # update background elements
        self.session.background_array =  update_elements(ElementArrayStim = self.session.background_array,
                                                        condition_settings = self.condition_settings, 
                                                        position_jitter = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['pos_jitter'], self.session.monitor),
                                                        orientation = orientation,
                                                        this_phase = 'background', 
                                                        elem_positions = position_dictionary['background']['xys'], 
                                                        grid_pos = self.grid_pos,
                                                        monitor = self.session.monitor, 
                                                        screen = self.session.screen)
        if this_phase != 'background':

            # update bar elements
            self.session.bar0_array =  update_elements(ElementArrayStim = self.session.bar0_array,
                                                        condition_settings = self.condition_settings, 
                                                        position_jitter = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['pos_jitter'], self.session.monitor),
                                                        orientation = orientation,
                                                        this_phase = this_phase[0], 
                                                        elem_positions = position_dictionary['bar0']['xys'], 
                                                        grid_pos = self.grid_pos,
                                                        monitor = self.session.monitor, 
                                                        screen = self.session.screen)

            self.session.bar1_array =  update_elements(ElementArrayStim = self.session.bar1_array,
                                                        condition_settings = self.condition_settings, 
                                                        position_jitter = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['pos_jitter'], self.session.monitor),
                                                        orientation = orientation,
                                                        this_phase = this_phase[1], 
                                                        elem_positions = position_dictionary['bar1']['xys'], 
                                                        grid_pos = self.grid_pos,
                                                        monitor = self.session.monitor, 
                                                        screen = self.session.screen)

            self.session.bar2_array =  update_elements(ElementArrayStim = self.session.bar2_array,
                                                        condition_settings = self.condition_settings, 
                                                        position_jitter = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['pos_jitter'], self.session.monitor),
                                                        orientation = orientation,
                                                        this_phase = this_phase[2], 
                                                        elem_positions = position_dictionary['bar2']['xys'], 
                                                        grid_pos = self.grid_pos,
                                                        monitor = self.session.monitor, 
                                                        screen = self.session.screen)

            self.session.bar3_array =  update_elements(ElementArrayStim = self.session.bar3_array,
                                                        condition_settings = self.condition_settings, 
                                                        position_jitter = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['pos_jitter'], self.session.monitor),
                                                        orientation = orientation,
                                                        this_phase = this_phase[3], 
                                                        elem_positions = position_dictionary['bar3']['xys'], 
                                                        grid_pos = self.grid_pos,
                                                        monitor = self.session.monitor, 
                                                        screen = self.session.screen)


        # actually draw
        self.session.background_array.draw()
        if this_phase != 'background':
            
            bars2plot = [self.session.bar0_array,self.session.bar1_array,self.session.bar2_array,self.session.bar3_array]

            bars2plot[drawing_ind[0]].draw()
            bars2plot[drawing_ind[1]].draw()
            bars2plot[drawing_ind[2]].draw()
            bars2plot[drawing_ind[3]].draw()
            

class FlickerStim(Stim):

    def __init__(self, session, bar_width_ratio, grid_pos):

        # need to initialize parent class (Stim)
        super().__init__(session=session, bar_width_ratio=bar_width_ratio, grid_pos=grid_pos)


    def draw(self, ecc_midpoint_at_trial, this_phase, position_dictionary, orientation = True):
        
        """ Draw stimuli - pRF bar - for each trial 
        
        Parameters
        ----------
        ecc_midpoint_at_trial : float
            eccentricity (in pixels) of bar position for trial (if empty, then nan) 
        this_phase: str
            strings with name of condition to draw
        """
        

        # update background elements
        self.session.background_array = update_elements(ElementArrayStim = self.session.background_array,
                                                        condition_settings = self.condition_settings,
                                                        position_jitter = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['pos_jitter'], self.session.monitor),
                                                        orientation = orientation,
                                                        this_phase = 'background', 
                                                        elem_positions = position_dictionary['background']['xys'], 
                                                        grid_pos = self.grid_pos,
                                                        background_contrast = self.session.settings['stimuli']['flicker']['backgr_contrast'],
                                                        monitor = self.session.monitor, 
                                                        screen = self.session.screen)


        self.session.bar0_array = update_elements(ElementArrayStim = self.session.bar0_array,
                                                    condition_settings = self.condition_settings, 
                                                    position_jitter = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['pos_jitter'], self.session.monitor), 
                                                    orientation = orientation,
                                                    this_phase = this_phase, 
                                                    elem_positions = position_dictionary['bar0']['xys'], 
                                                    grid_pos = self.grid_pos,
                                                    monitor = self.session.monitor, 
                                                    screen = self.session.screen)


        # actually draw
        self.session.background_array.draw()
        self.session.bar0_array.draw()             



