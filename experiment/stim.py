
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

              

class PRFStim(Stim):

    def __init__(self, session, bar_width_ratio, grid_pos):

        # need to initialize parent class (Stim)
        super().__init__(session=session, bar_width_ratio=bar_width_ratio, grid_pos=grid_pos)



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
        position_dictionary = get_object_positions(self.grid_pos,bar_midpoint_at_TR, bar_direction_at_TR,
                                                    self.bar_width_pix, screen = self.session.screen, num_bar = 1)


        # update background elements
        self.session.background_array =  update_elements(win = self.session.win,
                                                        condition_settings = self.condition_settings, 
                                                        this_phase = 'background', 
                                                        elem_positions = position_dictionary['background']['xys'], 
                                                        nElements = position_dictionary['background']['nElements'],
                                                        monitor = self.session.monitor, 
                                                        screen = self.session.screen)

        # update bar elements
        self.session.bar0_array =  update_elements(win = self.session.win,
                                                    condition_settings = self.condition_settings, 
                                                    this_phase = this_phase, 
                                                    elem_positions = position_dictionary['bar0']['xys'], 
                                                    nElements = position_dictionary['bar0']['nElements'],
                                                    monitor = self.session.monitor, 
                                                    screen = self.session.screen)


        # actually draw
        self.session.background_array.draw()
        self.session.bar0_array.draw()


        






