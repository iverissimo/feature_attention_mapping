
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


        # set boundaries for element positions, depending on bar position at TR and direction
        if orientation in np.array(['L-R','R-L']): # horizontal bar pass
            x_bound = np.array([bar_pos_midpoint[0]-self.bar_width_pix/2, bar_pos_midpoint[0]+self.bar_width_pix/2])
            y_bound = np.array([-self.session.win.size[1]/2,self.session.win.size[1]/2])

        elif orientation in np.array(['U-D','D-U']): # vertical bar pass
            x_bound = np.array([-self.session.win.size[0]/2,self.session.win.size[0]/2])
            y_bound = np.array([bar_pos_midpoint[1]-self.bar_width_pix/2, bar_pos_midpoint[1]+self.bar_width_pix/2])


        # x and y positions for all elements, within set boundaries
        x_pos = np.random.uniform(x_bound[0],x_bound[1],self.num_elements)
        y_pos = np.random.uniform(y_bound[0],y_bound[1],self.num_elements)

        # bar element position in pairs (x,y)
        self.element_positions = np.array([np.array([x_pos[i],y_pos[i]]) for i in range(self.num_elements)])

        self.session.element_array = visual.ElementArrayStim(win=self.session.win, nElements = self.num_elements,
                                                                units='pix', elementTex='sin', elementMask='gauss',
                                                                sizes = self.element_sizes, sfs = self.element_sfs, 
                                                                xys = self.element_positions, oris=self.element_orientations, 
                                                                colors = self.colors, 
                                                                colorSpace = 'rgb') 

        self.session.element_array.draw()






