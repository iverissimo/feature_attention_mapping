
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
        self.element_contrast = np.zeros((self.num_elements))

        # element sizes
        element_sizes_px = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['element_size'], self.session.monitor) # in pix
        self.element_sizes = np.ones((self.num_elements)) * element_sizes_px 
        
        # element background spatial frequency 
        element_sfs_pix = tools.monitorunittools.deg2pix(self.condition_settings['background']['element_sf'], self.session.monitor) # (transform cycles/degree to cycles/pixel)
        self.element_sfs = np.ones((self.num_elements)) * element_sfs_pix
        
        # element background orientation (half ori1, half ori2)
        ori_arr = np.concatenate((np.ones((math.floor(self.num_elements * .5))) * self.condition_settings['background']['element_ori'][0], 
                                  np.ones((math.ceil(self.num_elements * .5))) * self.condition_settings['background']['element_ori'][1])) 

        self.element_orientations = jitter(ori_arr) # add some jitter to the orientations

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
                               'colors': self.colors
                               }

        # define element array, for all possible display positions
        self.session.element_array = visual.ElementArrayStim(win = self.session.win, nElements = self.background_dict['nElements'],
                                                            units = 'pix', elementTex = self.background_dict['elementTex'], elementMask = 'gauss',
                                                            sizes = self.element_sizes, sfs = self.background_dict['sfs'], 
                                                            xys = self.background_dict['xys'], oris = self.background_dict['oris'],
                                                            contrs = self.background_dict['contrs'], 
                                                            colors = self.background_dict['colors'], 
                                                            colorSpace = 'rgb') 

        

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
        
        ## define elements for bar
        
        # first define bar width in pixels (depends if vertical or horizontal bar pass)
        # and bounds for x and y positions
        
        if bar_direction_at_TR in np.array(['L-R','R-L']): # if horizontal bar pass
                        
            x_bounds = np.array([bar_midpoint_at_TR[0]-self.bar_width_pix[0]/2,bar_midpoint_at_TR[0]+self.bar_width_pix[0]/2])
            y_bounds = np.array([-self.session.screen[1]/2,self.session.screen[1]/2])

        elif bar_direction_at_TR in np.array(['U-D','D-U']): # if vertical bar pass
            
            x_bounds = np.array([-self.session.screen[1]/2,self.session.screen[1]/2])
            y_bounds = np.array([bar_midpoint_at_TR[1]-self.bar_width_pix[1]/2, bar_midpoint_at_TR[1]+self.bar_width_pix[1]/2])
            

            
        # check which grid positions are within bounds
        bar_ind = np.where(((self.grid_pos[...,0]>=min(x_bounds))&
                    (self.grid_pos[...,0]<=max(x_bounds))&
                    (self.grid_pos[...,1]>=min(y_bounds))&
                    (self.grid_pos[...,1]<=max(y_bounds))
                ))[0]

        # boolean mask of grid position
        # to be updated throughout experiment, 
        # False keeps background settings, True changes 

        bool_mask = np.full(self.grid_pos.shape,False)
        bool_mask[bar_ind] = True


        # set element contrasts
        self.element_contrast = np.array([1 if bool_mask[ind][0]==True else 0 for ind,val in enumerate(self.element_contrast)])
                
        # element spatial frequency
        element_sfs_pix = tools.monitorunittools.deg2pix(self.condition_settings[this_phase]['element_sf'], self.session.monitor) # (transform cycles/degree to cycles/pixel)
        
        self.element_sfs = np.array([element_sfs_pix if bool_mask[ind][0]==True else val for ind,val in enumerate(self.element_sfs)])
        
        # element orientation (half ori1, half ori2)
        ori_arr = np.concatenate((np.ones((math.floor(sum(bool_mask[...,0]) * .5))) * self.condition_settings[this_phase]['element_ori'][0], 
                                  np.ones((math.ceil(sum(bool_mask[...,0]) * .5))) * self.condition_settings[this_phase]['element_ori'][1])) 

        # add some jitter to the orientations
        # needs to be bigger to allow for orientation differentiation
        ori_arr = jitter(ori_arr,max_val=10,min_val=5) if this_phase in ('ori_left','ori_right') else jitter(ori_arr)

        np.random.shuffle(ori_arr) # shuffle the orientations

        updated_ori = []
        ori_counter = 0
        
        for ind,val in enumerate(self.element_sfs):
            if bool_mask[ind][0]==True:
                updated_ori.append(ori_arr[ori_counter])
                ori_counter +=1
            else:
                updated_ori.append(val)

        self.element_orientations = np.array(updated_ori)
        
        
        # # element colors
        self.colors = np.ones((int(np.round(self.num_elements)),3)) * np.array(self.condition_settings[this_phase]['element_color'])
        
        # define bar array element
        if this_phase in ('color_green','color_red'):
            
            # to make colored gabor, need to do it a bit differently (psychopy forces colors to be opposite)
            grat_res = near_power_of_2(self.element_sizes[0],near='previous') # use power of 2 as grating res, to avoid error
            grating = visual.filters.makeGrating(res=grat_res)

            # initialise a 'black' texture 
            colored_grating = np.ones((grat_res, grat_res, 3)) #* -1.0
            
            # replace the red/green channel with the grating
            if this_phase=='color_red': 
                colored_grating[..., 0] = grating 
            else:
                colored_grating[..., 1] = grating 

            self.elementTex = colored_grating

        else:
            self.elementTex = 'sin'

        
        # actually set the settings
        self.session.element_array.setContrs(self.element_contrast)#, log=False)
        self.session.element_array.setSfs(self.element_sfs)
        self.session.element_array.setOris(self.element_orientations)
        self.session.element_array.setColors(self.colors)
        self.session.element_array.setTex(self.elementTex)#, log=False)

        self.session.element_array.draw()


        






