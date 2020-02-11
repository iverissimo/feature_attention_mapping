
import os
import numpy as np 



class PRFStim(object):
    def __init__(self,session,trial_type,prev_trial_type,switch):

        
        # parameters
        self.session = session

        self.num_elements = self.session.settings['stimuli']['num_elements'] # number of elements to put in bar
        self.element_sizes = np.ones((self.num_elements)) * self.session.settings['stimuli']['element_size'] # sizes of elements in pix

        self.screen = self.session.settings['window']['size'] # screen res [hRes,vRes]
        self.bar_width_ratio = self.session.settings['stimuli']['bar_width_ratio'] # ratio of bar width, relative to length of direction of movement

        midpoint = 0 # midpoint of bar

        if trial_type == 'LR' or trial_type == 'RL': #horizontal bar passes
            self.bar_width_pix = self.screen[0]*self.bar_width_ratio # in pix
            self.bar_length_pix = self.screen[1] # in pix
            print('horizontal bar')
            
        elif trial_type == 'TB' or trial_type == 'BT': #vertical bar passes:
            self.bar_width_pix = self.screen[1]*self.bar_width_ratio # in pix
            self.bar_length_pix = self.screen[0] # in pix
            print('vertical bar')

        elif trial_type == 'blank':
        	print('empty')



        # PRECISO DE PENSAR NAS POSIÇÕES E COMO FAZER UPDATE
        # FAZER DRAW FUNCTION
        # 
            

            # # # # 

        # # #



        #self.element_array = visual.ElementArrayStim(screen, nElements = self.num_elements, sizes = self.element_sizes, sfs = self.element_sfs, 
        #    xys = self.element_positions, colors = self.colors, colorSpace = 'rgb')


