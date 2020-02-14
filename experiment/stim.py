
import os
import numpy as np
from psychopy import visual
from psychopy import tools 



class PRFStim(object):
    def __init__(self, session, bar_width_deg):

        
        # parameters
        self.session = session
        self.bar_width_deg = bar_width_deg

        # bar width in pixels
        self.bar_width_pix = tools.monitorunittools.deg2pix(self.bar_width_deg, self.session.monitor)

        self.num_elements = self.session.settings['stimuli']['num_elements'] # number of elements to put in bar
        self.element_sizes = np.ones((self.num_elements)) * self.session.settings['stimuli']['element_size'] # sizes of elements in pix

        self.screen = self.session.win.size # screen res [hRes,vRes]

        # color ratios for elements
        red_ratio = 0.5
        blue_ratio = 0.5

        # Now set the actual stimulus parameters
        self.colors = np.concatenate((np.ones((int(np.round(self.num_elements*red_ratio)),3)) * np.array([1, -1, -1]),  # red elements
                                np.ones((int(np.round(self.num_elements*blue_ratio)),3)) * np.array([-1,1,-1]))) # green elements                
        # shuffle the colors
        np.random.shuffle(self.colors)

        # spatial frequency
        element_sfs_pix = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['element_spatial_frequency'], self.session.monitor)
        self.element_sfs = np.ones((self.num_elements)) * element_sfs_pix

        # element size
        element_sizes_px = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['element_size'], self.session.monitor)
        self.element_sizes = np.ones((self.num_elements)) * element_sizes_px

        # element orientation
        self.element_orientations = np.random.rand(self.num_elements) * 720.0 - 360.0



    def draw(self, time, bar_pos_midpoint, orientation):

        # dictionary with all angles 
        dict_orientation_deg = self.session.settings['stimuli']['direction_angle']

        self.orientation_deg = dict_orientation_deg[orientation] # get orientation for bar pos in deg

        self.rotation_matrix = np.matrix([[np.cos(self.orientation_deg), -np.sin(self.orientation_deg)],
                                [np.sin(self.orientation_deg), np.cos(self.orientation_deg)]])
        

        if orientation in np.array(['L-R','R-L']): # horizontal bar pass
            self.bar_heigth = self.screen[1]
            self.bar_length = self.bar_width_pix
        else:
            self.bar_heigth = self.bar_width_pix  
            self.bar_length = self.screen[0]		

        # bar element position
        self.element_positions = np.random.rand(self.num_elements, 2) * np.array([self.bar_length, self.bar_heigth]) - np.array([self.bar_length/2.0, self.bar_heigth/2.0])

        ####### NEED TO CORRECT THIS
        # I think better way to do it is defining boundaries for x and y and make random combinations for elements
        # easier and straightforward
        self.session.element_array.setSfs(self.element_sfs)
        self.session.element_array.setSizes(self.element_sizes)
        self.session.element_array.setColors(self.colors)
        self.session.element_array.setOris(self.element_orientations)

        self.session.element_array.setXYs(np.array(np.matrix(self.element_positions + bar_pos_midpoint) * self.rotation_matrix)) 

        self.session.element_array = visual.ElementArrayStim(self.screen, nElements = self.session.num_elements, sizes = self.element_sizes, sfs = self.element_sfs, 
            xys = self.element_positions, colors = self.colors, colorSpace = 'rgb') 

        self.session.element_array.draw()





