
import os
import numpy as np
from psychopy import visual, tools


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
        

    def draw(self, bar_midpoint_at_TR, bar_direction_at_TR):
        
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
        element_sfs_pix = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['element_high_sf'], self.session.monitor) # (transform cycles/degree to cycles/pixel)
        self.element_sfs = np.ones((self.num_elements)) * element_sfs_pix
        
        # element orientation
        self.element_orientations = np.concatenate((np.ones((int(self.num_elements * .5))) * 180,  # vertical elements
                                                   np.ones((int(self.num_elements * .5))) * 90)) # horizontal elements
        
        # add some jitter to the orientations 
        jit = np.concatenate((np.random.uniform(-1,-0.5,int(self.num_elements * 0.5)),
                              np.random.uniform(0.5,1,int(self.num_elements * 0.5))))
        np.random.shuffle(jit)

        self.element_orientations += jit
        np.random.shuffle(self.element_orientations) # shuffle the orientations
        
        # element colors
        self.colors = np.ones((int(np.round(self.num_elements)),3)) * np.array([-1, -1, -1])
        np.random.shuffle(self.colors) # shuffle the colors
        
        
        # define bar array element
        self.session.element_array = visual.ElementArrayStim(win=self.session.win, nElements = self.num_elements,
                                                                units='pix', elementTex='sin', elementMask='gauss',
                                                                sizes = self.element_sizes, sfs = self.element_sfs, 
                                                                xys = self.element_positions, oris=self.element_orientations, 
                                                                colors = self.colors, 
                                                                colorSpace = 'rgb') 
        self.session.element_array.draw()






#####


class PRFStim2(object): # So then I still save older version here
    def __init__(self, session, bar_width_deg):

        
        # general parameters
        self.session = session
        self.screen = self.session.win.size # screen res [hRes,vRes]

        # bar width
        self.bar_width_deg = bar_width_deg
        self.bar_width_pix = tools.monitorunittools.deg2pix(self.bar_width_deg, self.session.monitor) # in pix

        # bar number of elements
        self.num_elements = self.session.settings['stimuli']['num_elements']
        # background number of elements
        #self.background_num_elements = self.session.settings['background']['num_elements']

        # bar element sizes
        element_sizes_px = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['element_size'], self.session.monitor) # in pix
        self.element_sizes = np.ones((self.num_elements)) * element_sizes_px 
        # background element sizes
        #background_element_sizes_px = tools.monitorunittools.deg2pix(self.session.settings['background']['element_size'], self.session.monitor) # in pix
        #self.background_element_sizes = np.ones((self.background_num_elements)) *  background_element_sizes_px

        # bar element spatial frequency
        # (transform cycles/degree to cycles/pixel)
        element_sfs_pix = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['element_high_sf'], self.session.monitor)
        self.element_sfs = np.ones((self.num_elements)) * element_sfs_pix
        # background element spatial frequency
        #background_element_sfs_pix = tools.monitorunittools.deg2pix(self.session.settings['background']['element_sf'], self.session.monitor)
        #self.background_element_sf = np.ones((self.background_num_elements)) * background_element_sfs_pix

        # bar element orientation
        self.element_orientations = np.concatenate((np.ones((int(self.num_elements * .5))) * 180,  # vertical elements
                                                   np.ones((int(self.num_elements * .5))) * 90)) # horizontal elements

        #self.element_orientations = np.ones(int(self.num_elements)) * 180 + self.session.settings['stimuli']['element_right_ori']  # right orientation
                                                    

        # add some jitter to the orientations 
        jit = np.concatenate((np.random.uniform(-1,-0.5,int(self.num_elements * 0.5)),
                              np.random.uniform(0.5,1,int(self.num_elements * 0.5))))
        np.random.shuffle(jit)

        self.element_orientations += jit
        #
        # background element orientation
        # self.background_element_orientations = np.concatenate((np.ones((int(self.background_num_elements * .5))) * 180,  # vertical elements
        #                                             np.ones((int(self.background_num_elements * .5))) * 90)) # horizontal elements
        # # add some jitter to the background orientations 
        # background_jit = np.concatenate((np.random.uniform(-1,-0.5,int(self.background_num_elements * 0.5)),
        #                      np.random.uniform(0.5,1,int(self.background_num_elements * 0.5))))
        # np.random.shuffle(background_jit)

        # self.background_element_orientations += background_jit


        # color ratios for elements
        #red_ratio = 1#0.5
        #green_ratio = 0#0.5

        # Now set the actual stimulus parameters
        #self.colors = np.concatenate((np.ones((int(np.round(self.num_elements*red_ratio)),3)) * np.array([1, -1, 0]),  # red elements
        #                        np.ones((int(np.round(self.num_elements*green_ratio)),3)) * np.array([-1,1,0]))) # green elements  
        self.colors = np.ones((int(np.round(self.num_elements)),3)) * np.array([-1, -1, -1]) # red elements
        # shuffle the colors
        np.random.shuffle(self.colors)
        


    def draw(self, bar_midpoint, bar_direction):

        # dictionary with all angles 
        dict_orientation_deg = self.session.settings['stimuli']['direction_angle']

        self.orientation_rad = np.radians(dict_orientation_deg[bar_direction]) # get orientation for bar pos in radians

        # to rotate bar for diagonal bar passes
        self.rotation_matrix = np.matrix([[np.cos(self.orientation_rad), -np.sin(self.orientation_rad)],
                                [np.sin(self.orientation_rad), np.cos(self.orientation_rad)]])

        # calculate screen diagonal length in pixels
        diag_pix = np.sqrt(self.session.win.size[0]**2 + self.session.win.size[1]**2)


        # set boundaries for element positions, depending on bar position at TR and direction
        if bar_direction in np.array(['L-R','R-L']): # horizontal bar pass
            x_bound = np.array([bar_midpoint[0]-self.bar_width_pix/2, bar_midpoint[0]+self.bar_width_pix/2])
            y_bound = np.array([-self.session.win.size[1]/2,self.session.win.size[1]/2])

        elif bar_direction in np.array(['U-D','D-U']): # vertical bar pass
            x_bound = np.array([-self.session.win.size[0]/2,self.session.win.size[0]/2])
            y_bound = np.array([bar_midpoint[1]-self.bar_width_pix/2, bar_midpoint[1]+self.bar_width_pix/2])

        else: # diagonals
            y_bound = np.array([-diag_pix/2,diag_pix/2])
            x_bound = np.array([bar_midpoint[0]-self.bar_width_pix/2, bar_midpoint[0]+self.bar_width_pix/2])


        # x and y positions for all elements, within set boundaries
        x_pos = np.random.uniform(x_bound[0],x_bound[1],self.num_elements)
        y_pos = np.random.uniform(y_bound[0],y_bound[1],self.num_elements)

        if bar_direction in np.array(['UR-DL','DR-UL','DL-UR','UL-DR']): # give diagonal correct orientation
        #    # bar element position in pairs (x,y)
            self.element_positions = np.array([np.array([x_pos[i],y_pos[i]]) for i in range(self.num_elements)]) * self.rotation_matrix
        else:
            # bar element position in pairs (x,y)
            self.element_positions = np.array([np.array([x_pos[i],y_pos[i]]) for i in range(self.num_elements)])


        # # background element position
        # background_xpos = np.random.uniform(-self.screen[0]/2,self.screen[0]/2,self.background_num_elements) 
        # background_ypos = np.random.uniform(-self.screen[1]/2,self.screen[1]/2,self.background_num_elements)
        # self.background_element_positions = np.array([np.array([background_xpos[i],background_ypos[i]]) for i in range(self.background_num_elements)])


        # define bar array element
        self.session.element_array = visual.ElementArrayStim(win=self.session.win, nElements = self.num_elements,
                                                                units='pix', elementTex='sin', elementMask='gauss',
                                                                sizes = self.element_sizes, sfs = self.element_sfs, 
                                                                xys = self.element_positions, oris=self.element_orientations, 
                                                                colors = self.colors, 
                                                                colorSpace = 'rgb') 

        # define background array element
        #self.session.background_array = visual.ElementArrayStim(win=self.session.win, nElements = self.background_num_elements,
        #                                                        units='pix', elementTex='sin', elementMask='gauss',
        #                                                        sizes = self.background_element_sizes, sfs = self.background_element_sf, 
        #                                                        xys = self.background_element_positions, oris=self.background_element_orientations, 
                                                                #colors = self.colors, 
        #                                                          colorSpace = 'rgb') 

        #self.session.background_array.draw()

        self.session.element_array.draw()







