
import os
import numpy as np

from exptools2.core import Session

from trial import PRFTrial, FeatureTrial
from stim import PRFStim

from psychopy import visual, tools

import itertools


class ExpSession(Session):

    def __init__(self, output_str, output_dir, settings_file,macbook_bool):  # initialize child class

            """ Initializes ExpSession object. 
          
            Parameters
            ----------
            output_str : str
                Basename for all output-files (like logs), e.g., "sub-01_task-PRFstandard_run-1"
            output_dir : str
                Path to desired output-directory (default: None, which results in $pwd/logs)
            settings_file : str
                Path to yaml-file with settings (default: None, which results in the package's
                default settings file (in data/default_settings.yml)
            """

            # need to initialize parent class (Session), indicating output infos
            super().__init__(output_str=output_str,output_dir=output_dir,settings_file=settings_file)

            # set size of display
            if self.settings['window']['display'] == 'square':
                self.screen = np.array([self.win.size[1], self.win.size[1]])
                rect_contrast = 1
            
            elif self.settings['window']['display'] == 'rectangle':
                self.screen = np.array([self.win.size[0], self.win.size[1]])
                rect_contrast = 0 # then rectangles will be hidden

            if macbook_bool: # to compensate for macbook retina display
                self.screen = self.screen/2

            # some MRI params
            self.bar_step = self.settings['mri']['TR'] # in seconds
            self.mri_trigger = self.settings['mri']['sync'] #'t'

            ## make grid of possible positions for gabors 
            # (grid spans whole display, bar will alter specific part of grid)

            # first set the number of elements that fit each dimension
            gabor_diameter_pix = tools.monitorunittools.deg2pix(self.settings['stimuli']['element_size'], self.monitor) # diameter of each element (pix)
            elem_num = np.round(np.array(self.screen)/(gabor_diameter_pix * 0.6)) # [horiz #elements, vert #elements], also made it so that the elements will overlap a bit, to avoid emptyness 

            # then set equally spaced x and y coordinates for grid
            x_grid_pos = np.linspace(-self.screen[0]/2 + gabor_diameter_pix/2, # to make sure gabors within display
                                     self.screen[0]/2 - gabor_diameter_pix/2,
                                     int(elem_num[0]))

            y_grid_pos = np.linspace(-self.screen[1]/2 + gabor_diameter_pix/2, # to make sure gabors within display
                                     self.screen[1]/2 - gabor_diameter_pix/2,
                                     int(elem_num[1]))
            

            self.grid_pos = np.array(list(itertools.product(x_grid_pos, y_grid_pos))) # list of lists [[x0,y0],[x0,y1],...]


            ## create some elements that will be common to both tasks ##
            
            #create black bars on the side, for cases where we want square display
            rect_width = (self.win.size[0]-self.win.size[1])/2 # half of the difference between horizontal and vertical resolution
            rect_left_pos = [-self.screen[1]/2 - rect_width/2, 0] #[-self.screen[1]/2 ,0] # left rectangle position
            rect_right_pos = [self.screen[1]/2 + rect_width/2, 0] # [self.screen[1]/2 ,0] # right rectangle position
            
            self.rect_left = visual.Rect(win = self.win,
                                        units = "pix",
                                        width = rect_width,
                                        height = self.screen[1],
                                        fillColor = self.settings['stimuli']['rect_fill_color'],
                                        lineColor = self.settings['stimuli']['rect_line_color'],
                                        contrast = rect_contrast,
                                        pos = rect_left_pos
                                        )

            self.rect_right = visual.Rect(win = self.win,
                                        units = "pix",
                                        width = rect_width,
                                        height = self.screen[1],
                                        fillColor = self.settings['stimuli']['rect_fill_color'],
                                        lineColor = self.settings['stimuli']['rect_line_color'],
                                        contrast = rect_contrast,
                                        pos = rect_right_pos
                                        )

            
            # create fixation lines
            self.line1 = visual.Line(win = self.win,
                                    units = "pix",
                                    lineColor = self.settings['stimuli']['fix_line_color'],
                                    lineWidth = self.settings['stimuli']['fix_line_width'],
                                    contrast = self.settings['stimuli']['fix_line_contrast'],
                                    start = [-self.screen[0]/2, self.screen[1]/2],
                                    end = [self.screen[0]/2, -self.screen[1]/2]
                                    )

            self.line2 = visual.Line(win = self.win,
                                    units = "pix",
                                    lineColor = self.settings['stimuli']['fix_line_color'],
                                    lineWidth = self.settings['stimuli']['fix_line_width'],
                                    contrast = self.settings['stimuli']['fix_line_contrast'],
                                    start = [-self.screen[0]/2, -self.screen[1]/2],
                                    end = [self.screen[0]/2, self.screen[1]/2]
                                    )



class PRFSession(ExpSession):
   
    def __init__(self, output_str, output_dir, settings_file,macbook_bool):  # initialize child class

        """ Initializes PRFSession object. 
      
        Parameters
        ----------
        output_str : str
            Basename for all output-files (like logs), e.g., "sub-01_task-PRFstandard_run-1"
        output_dir : str
            Path to desired output-directory (default: None, which results in $pwd/logs)
        settings_file : str
            Path to yaml-file with settings (default: None, which results in the package's
            default settings file (in data/default_settings.yml)
        """

        # need to initialize parent class (Session), indicating output infos
        super().__init__(output_str=output_str,output_dir=output_dir,settings_file=settings_file,macbook_bool=macbook_bool)

        

    def create_stimuli(self):

        """ Create Stimuli - pRF bar and fixation dot """
        
        #generate PRF stimulus
        self.prf_stim = PRFStim(session=self, 
                                bar_width_ratio = self.settings['stimuli']['bar_width_ratio'], 
                                grid_pos = self.grid_pos
                                )
        
        # Convert fixation dot radius in degrees to pixels for a given Monitor object
        fixation_rad_pix = tools.monitorunittools.deg2pix(self.settings['stimuli']['fix_dot_size_deg'], 
                                                        self.monitor)/2 
        
        # create black fixation circle
        # note - fixation dot will change color during task
        self.fixation = visual.Circle(self.win, units = 'pix', radius = fixation_rad_pix, 
                                            fillColor = self.settings['stimuli']['fix_dot_color'], 
                                            lineColor = self.settings['stimuli']['fix_line_color'])  



    def create_trials(self):

        """ Creates trials (before running the session) """

        #
        # counter for responses
        self.total_responses = 0
        self.correct_responses = 0

        # number of TRs per "condition"
        bar_pass_hor_TR = self.settings['stimuli']['bar_pass_hor_TR']
        bar_pass_ver_TR = self.settings['stimuli']['bar_pass_ver_TR']
        empty_TR = self.settings['stimuli']['empty_TR']

        # list with order of bar orientations throught experiment
        bar_direction = self.settings['stimuli']['bar_direction'] 

        # all positions in pixels [x,y] for for midpoint of
        # vertical bar passes, 
        ver_y = self.screen[1]*np.linspace(-0.5,0.5, bar_pass_ver_TR)
        ver_bar_pos_pix = np.array([np.array([0,y]) for _,y in enumerate(ver_y)])

        # horizontal bar passes 
        hor_x = self.screen[0]*np.linspace(-0.5,0.5, bar_pass_hor_TR)
        hor_bar_pos_pix = np.array([np.array([x,0]) for _,x in enumerate(hor_x)])

        #create as many trials as TRs
        trial_number = 0
        bar_direction_all = [] # list of bar orientation at all TRs

        bar_pos_array = [] # list with bar midpoint (x,y) for all TRs (if nan, then empty screen)

        for _,bartype in enumerate(bar_direction):
            if bartype in np.array(['empty']): # empty screen
                trial_number += empty_TR
                bar_direction_all = bar_direction_all + np.repeat(bartype,empty_TR).tolist()
                bar_pos_array.append([np.array([np.nan,np.nan]) for i in range(empty_TR)])

            elif bartype in np.array(['U-D','D-U']): # vertical bar pass
                trial_number += bar_pass_ver_TR
                bar_direction_all =  bar_direction_all + np.repeat(bartype,bar_pass_ver_TR).tolist()
                
                # order depending on starting point for bar pass, and append to list
                position_list = np.sort(ver_bar_pos_pix,axis=0) if bartype=='D-U' else np.sort(ver_bar_pos_pix,axis=0)[::-1]
                bar_pos_array.append(position_list)

            elif bartype in np.array(['L-R','R-L']): # horizontal bar pass
                trial_number += bar_pass_hor_TR
                bar_direction_all =  bar_direction_all + np.repeat(bartype,bar_pass_hor_TR).tolist()
                
                # order depending on starting point for bar pass, and append to list
                position_list = np.sort(hor_bar_pos_pix,axis=0) if bartype=='L-R' else np.sort(hor_bar_pos_pix,axis=0)[::-1]
                bar_pos_array.append(position_list)

        self.trial_number = trial_number # total number of trials 
        print("Total number of (expected) TRs: %d"%self.trial_number)
        self.bar_direction_all = bar_direction_all # list of strings with bar orientation/empty

        # list of midpoint position (x,y) of bar for all TRs (if empty, then nan)
        self.bar_midpoint_all = np.array([val for sublist in bar_pos_array for val in sublist])

        
        # get condition names and randomize them for each trial 
        key_list = []
        for key in self.settings['stimuli']['conditions']:
            if key != 'background': # we don't want to show background gabors in background
                key_list.append(key)

        np.random.shuffle(key_list)
        phase_conditions = np.array(key_list)

        for r in range(trial_number-1):
            np.random.shuffle(key_list)
            
            phase_conditions = np.vstack((phase_conditions,key_list))


        # define list with number of phases and their duration (duration of each must be the same)
        self.phase_durations = np.repeat(self.bar_step/phase_conditions.shape[-1],
                                        phase_conditions.shape[-1])

        # append all trials
        self.all_trials = []
        for i in range(self.trial_number):

            self.all_trials.append(PRFTrial(session=self,
                                            trial_nr=i,  
                                            phase_durations = self.phase_durations,
                                            phase_names = phase_conditions[i],
                                            bar_direction_at_TR=self.bar_direction_all[i],
                                            bar_midpoint_at_TR=self.bar_midpoint_all[i]
                                            ))

        ## define timepoints for fixation dot to change color
        # total experiment time (in seconds)
        self.total_time = self.trial_number*self.bar_step 
        # switch time points (around 4 s between switches + jitter to avoid expectation effects)
        self.fixation_switch_times = np.arange(1,self.total_time,4)
        self.fixation_switch_times += 2*np.random.random_sample((len(self.fixation_switch_times),)) 
        # counter for fixation dot switches
        self.fix_counter = 0

        # print window size just to check, not actually needed
        print(self.screen)

    
    def run(self):
        """ Loops over trials and runs them """
        
        # draw instructions wait a few seconds
        this_instruction_string = 'Please fixate at the center, \ndo not move your eyes'
        self.display_text(this_instruction_string, duration=3,
                                    color=(1, 1, 1), font = 'Helvetica Neue', pos = (0, 0), 
                                    italic = True, alignHoriz = 'center')

        # draw instructions wait for scanner t trigger
        this_instruction_string = 'Index finger - Black\n Middle finger - White\nwaiting for scanner'
        self.display_text(this_instruction_string, keys=self.settings['mri'].get('sync', 't'),
                                color=(1, 1, 1), font = 'Helvetica Neue', pos = (0, 0), 
                                italic = True, alignHoriz = 'center')

        # create trials before running!
        self.create_stimuli()
        self.create_trials() 

        self.start_experiment()
        
        # cycle through trials
        for trl in self.all_trials: 
            trl.run() # run forrest run


        print('Expected number of responses: %d'%len(self.fixation_switch_times))
        print('Total subject responses: %d'%self.total_responses)
        print('Correct responses (within 0.8s of dot color change): %d'%self.correct_responses)
          

        self.close() # close session
        


class FeatureSession(ExpSession):
    
    def __init__(self, output_str, output_dir, settings_file,macbook_bool): # initialize child class

        """ Initializes PRFSession object. 
      
        Parameters
        ----------
        output_str : str
            Basename for all output-files (like logs), e.g., "sub-01_task-PRFfeature_run-1"
        output_dir : str
            Path to desired output-directory (default: None, which results in $pwd/logs)
        settings_file : str
            Path to yaml-file with settings (default: None, which results in the package's
            default settings file (in data/default_settings.yml)
        """


        # need to initialize parent class (Session), indicating output infos
        #super().__init__(output_str=output_str,output_dir=output_dir,settings_file=settings_file)

        pass # needs to be filled in

    def run(self):
        """ Loops over trials and runs them """

        pass





