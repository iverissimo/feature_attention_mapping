
import os
import numpy as np

from exptools2.core import Session, PylinkEyetrackerSession

from trial import PRFTrial, FeatureTrial, FlickerTrial
from stim import PRFStim, FeatureStim, FlickerStim

from psychopy import visual, tools

import itertools

from utils import *


class ExpSession(PylinkEyetrackerSession):

    def __init__(self, output_str, output_dir, settings_file, macbook_bool, eyetracker_on = True):  # initialize child class

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
            macbook_bool: bool
                variable to know if using macbook for running experiment or not
            """

            # need to initialize parent class (Session), indicating output infos
            super().__init__(output_str = output_str, output_dir = output_dir, settings_file = settings_file, eyetracker_on = eyetracker_on)

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
            self.gabor_diameter_pix = tools.monitorunittools.deg2pix(self.settings['stimuli']['element_size'], self.monitor) # diameter of each element (pix)
            print('gabor diameter in pix %s'%str(self.gabor_diameter_pix))
            
            elem_num = np.round(np.array(self.screen)/(self.gabor_diameter_pix * self.settings['stimuli']['gab_ratio'])) # [horiz #elements, vert #elements], also made it so that the elements will overlap a bit, to avoid emptyness 

            # then set equally spaced x and y coordinates for grid
            x_grid_pos = np.linspace(-self.screen[0]/2,
                                     self.screen[0]/2,
                                     int(elem_num[0]))

            y_grid_pos = np.linspace(-self.screen[1]/2,
                                     self.screen[1]/2,
                                     int(elem_num[1]))
            

            self.grid_pos = np.array(list(itertools.product(x_grid_pos, y_grid_pos))) # list of lists [[x0,y0],[x0,y1],...]
            print(self.grid_pos.shape)

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
                                        pos = rect_left_pos,
                                        fillColorSpace = self.settings['stimuli']['colorSpace'],
                                        lineColorSpace = self.settings['stimuli']['colorSpace']
                                        )

            self.rect_right = visual.Rect(win = self.win,
                                        units = "pix",
                                        width = rect_width,
                                        height = self.screen[1],
                                        fillColor = self.settings['stimuli']['rect_fill_color'],
                                        lineColor = self.settings['stimuli']['rect_line_color'],
                                        contrast = rect_contrast,
                                        pos = rect_right_pos,
                                        fillColorSpace = self.settings['stimuli']['colorSpace'],
                                        lineColorSpace = self.settings['stimuli']['colorSpace']
                                        )

            
            # create fixation lines
            self.line1 = visual.Line(win = self.win,
                                    units = "pix",
                                    lineColor = self.settings['stimuli']['fix_line_color'],
                                    lineWidth = self.settings['stimuli']['fix_line_width'],
                                    contrast = self.settings['stimuli']['fix_line_contrast'],
                                    start = [-self.screen[0]/2, self.screen[1]/2],
                                    end = [self.screen[0]/2, -self.screen[1]/2],
                                    lineColorSpace = self.settings['stimuli']['colorSpace']
                                    )

            self.line2 = visual.Line(win = self.win,
                                    units = "pix",
                                    lineColor = self.settings['stimuli']['fix_line_color'],
                                    lineWidth = self.settings['stimuli']['fix_line_width'],
                                    contrast = self.settings['stimuli']['fix_line_contrast'],
                                    start = [-self.screen[0]/2, -self.screen[1]/2],
                                    end = [self.screen[0]/2, self.screen[1]/2],
                                    lineColorSpace = self.settings['stimuli']['colorSpace']
                                    )


class PRFSession(ExpSession):
   
    def __init__(self, output_str, output_dir, settings_file, macbook_bool, eyetracker_on):  # initialize child class

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
        macbook_bool: bool
            variable to know if using macbook for running experiment or not
        """

        # need to initialize parent class (ExpSession), indicating output infos
        super().__init__(output_str = output_str, output_dir = output_dir, settings_file = settings_file, 
                        macbook_bool = macbook_bool, eyetracker_on = eyetracker_on)

        

    def create_stimuli(self):

        """ Create Stimuli - pRF bar and fixation dot """
        
        #generate PRF stimulus
        self.prf_stim = PRFStim(session = self, 
                                bar_width_ratio = self.settings['stimuli']['prf']['bar_width_ratio'], 
                                grid_pos = self.grid_pos
                                )
        
        # Convert fixation dot radius in degrees to pixels for a given Monitor object
        fixation_rad_pix = tools.monitorunittools.deg2pix(self.settings['stimuli']['fix_dot_size_deg'], 
                                                        self.monitor)/2 
        
        # create black fixation circle
        # note - fixation dot will change color during task
        self.fixation = visual.Circle(self.win, units = 'pix', radius = fixation_rad_pix, 
                                            fillColor = self.settings['stimuli']['fix_dot_color'], 
                                            lineColor = self.settings['stimuli']['fix_line_color'],
                                            fillColorSpace = self.settings['stimuli']['colorSpace'],
                                            lineColorSpace = self.settings['stimuli']['colorSpace'])  



    def create_trials(self):

        """ Creates trials (before running the session) """

        #
        # counter for responses
        self.total_responses = 0
        self.correct_responses = 0

        # define bar width 
        bar_width_ratio = self.settings['stimuli']['prf']['bar_width_ratio']
        self.bar_width_pix = self.screen*bar_width_ratio

        # number of TRs per "type of stimuli"
        bar_pass_hor_TR = self.settings['stimuli']['prf']['bar_pass_hor_TR'] 
        bar_pass_ver_TR = self.settings['stimuli']['prf']['bar_pass_ver_TR']
        empty_TR = self.settings['stimuli']['prf']['empty_TR']
        switch_bckg_TR = self.settings['stimuli']['prf']['switch_bckg_TR']

        # list with order of bar orientations throught experiment
        bar_pass_direction = self.settings['stimuli']['prf']['bar_pass_direction'] 

        # all possible positions in pixels [x,y] for for midpoint of
        # vertical bar passes, 
        ver_y = self.screen[1]*np.linspace(-0.5,0.5, bar_pass_ver_TR)

        ver_bar_pos_pix = np.array([np.array([0,y]) for _,y in enumerate(ver_y)])

        # horizontal bar passes 
        hor_x = self.screen[0]*np.linspace(-0.5,0.5, bar_pass_hor_TR)

        hor_bar_pos_pix = np.array([np.array([x,0]) for _,x in enumerate(hor_x)])

        #create as many trials as TRs
        trial_number = 0
        bar_pass_direction_all = [] # list of bar orientation at all TRs

        bar_pos_array = [] # list with bar midpoint (x,y) for all TRs (if nan, then empty screen)

        for _,bartype in enumerate(bar_pass_direction):
            if bartype in np.array(['empty']): # empty screen
                trial_number += empty_TR
                bar_pass_direction_all = bar_pass_direction_all + np.repeat(bartype,empty_TR).tolist()
                bar_pos_array.append([np.array([np.nan,np.nan]) for i in range(empty_TR)])

            elif bartype in np.array(['U-D','D-U']): # vertical bar pass
                trial_number += bar_pass_ver_TR
                bar_pass_direction_all =  bar_pass_direction_all + np.repeat(bartype,bar_pass_ver_TR).tolist()
                
                # order depending on starting point for bar pass, and append to list
                position_list = np.sort(ver_bar_pos_pix,axis=0) if bartype=='D-U' else np.sort(ver_bar_pos_pix,axis=0)[::-1]
                bar_pos_array.append(position_list)

            elif bartype in np.array(['L-R','R-L']): # horizontal bar pass
                trial_number += bar_pass_hor_TR
                bar_pass_direction_all =  bar_pass_direction_all + np.repeat(bartype,bar_pass_hor_TR).tolist()
                
                # order depending on starting point for bar pass, and append to list
                position_list = np.sort(hor_bar_pos_pix,axis=0) if bartype=='L-R' else np.sort(hor_bar_pos_pix,axis=0)[::-1]
                bar_pos_array.append(position_list)

        self.trial_number = trial_number # total number of trials 
        print("Total number of (expected) TRs: %d"%self.trial_number)
        self.bar_pass_direction_all = bar_pass_direction_all # list of strings with bar orientation/empty

        # list of midpoint position (x,y) of bar for all TRs (if empty, then nan)
        self.bar_midpoint_all = np.array([val for sublist in bar_pos_array for val in sublist])

        
        # get condition names and randomize them for each trial 
        key_list = []
        for key in self.settings['stimuli']['conditions']:
            if key != 'background': # we don't want to show background gabors in background
                key_list.append(key)

        # if in scanner, we want it to be synced to trigger, so lets increase trial time (in seconds, like TR)
        max_trial_time = 5 if self.settings['mri']['scanner']==True else self.settings['mri']['TR']


        # define how many times bar features switch during TR, according to flick rate defined 
        if self.settings['stimuli']['prf']['flick_rate'] == 'TR':

            phase_conditions = np.repeat(key_list, self.trial_number/len(key_list))
            np.random.shuffle(phase_conditions) # randomized conditions, for attention to bar task

            phase_conditions = np.array([[val] for _,val in enumerate(phase_conditions)])

            # define list with number of phases and their duration (duration of each must be the same)
            # in this case we want one color per trial
            self.phase_durations = np.repeat(max_trial_time,phase_conditions.shape[-1])
            
        else:

            # flicker frequency
            flick_rate = self.settings['stimuli']['prf']['flick_rate']
            
            # number of samples in trial
            n_samples = max_trial_time * flick_rate

            # repeat keys, so for each bar pass it shows each condition X times
            key_list = np.array(key_list*round(n_samples/len(key_list))) 
            phase_conditions = key_list

            # stach them in trial
            for r in range(self.trial_number-1):            
                phase_conditions = np.vstack((phase_conditions,key_list))
                
            # define list with number of phases and their duration (duration of each must be the same)
            self.phase_durations = np.repeat(1/flick_rate,phase_conditions.shape[-1])


        # total experiment time (in seconds)
        self.total_time = self.trial_number * max_trial_time 
        
        # append all trials
        self.all_trials = []
        for i in range(self.trial_number):

            self.all_trials.append(PRFTrial(session =self ,
                                            trial_nr = i,  
                                            phase_durations = self.phase_durations,
                                            phase_names = phase_conditions[i],
                                            bar_pass_direction_at_TR = self.bar_pass_direction_all[i],
                                            bar_midpoint_at_TR = self.bar_midpoint_all[i]
                                            ))


        ## define timepoints for fixation dot to change color
        # switch time points (around 4 s between switches + jitter to avoid expectation effects)
        self.fixation_switch_times = np.arange(1,self.total_time,1/self.settings['stimuli']['fix_dot_shift_rate'])
        self.fixation_switch_times += 2*np.random.random_sample((len(self.fixation_switch_times),)) 
        # counter for fixation dot switches
        self.fix_counter = 0

        # define time points for element orientation to change
        # switch orientation time points
        if self.settings['stimuli']['ori_shift_rate'] == 'TR':
            ori_shift_rate = 1/self.bar_step 
        else:
            ori_shift_rate = self.settings['stimuli']['ori_shift_rate']
        self.ori_switch_times = np.arange(0,self.total_time,1/ori_shift_rate)
        # counter for orientation switches
        self.ori_counter = 0
        # index for orientation
        self.ori_ind = 0 


        # print window size just to check, not actually needed
        print(self.screen)
        print(tools.monitorunittools.pix2deg(self.screen[0], self.monitor))

    
    def run(self):
        """ Loops over trials and runs them """

        # update color of settings
        self.settings = get_average_color(self.output_dir, self.settings, task = 'standard')
        

        # create trials before running!
        self.create_stimuli()
        self.create_trials() 

        # if eyetracking then calibrate
        if self.eyetracker_on:
            self.calibrate_eyetracker()

        # draw instructions wait a few seconds
        this_instruction_string = ('During the experiment\n'
                                    'you will see a flickering bar pass\n'
                                    'in different directions\n'
                                    'throughout the screen\n\n\n'
                                    '[Press y/middle finger to continue]\n'
                                    '[Press b/index finger to skip]')

        key_pressed = draw_instructions(self.win, this_instruction_string, keys = ['b','y'], visual_obj = [self.rect_left,self.rect_right])

        if key_pressed[0] != 'b': #if instructions not skipped

            # draw instructions wait a few seconds
            this_instruction_string = ('Your task is to fixate\n'
                                        'at the center of the screen,\n'
                                        'and indicate when\n'
                                        'the central dot changes color\n\n\n'
                                        '[Press y/middle finger to continue]')
            
            draw_instructions(self.win, this_instruction_string, keys = ['y'], visual_obj = [self.rect_left,self.rect_right])

            
            # draw instructions wait a few seconds
            this_instruction_string = ('Do NOT look at the bars!\n'
                                        'Please fixate at the center,\n'
                                        'and do not move your eyes\n\n\n'
                                        '[Press y/middle finger to continue]')
            
            draw_instructions(self.win, this_instruction_string, keys = ['y'], visual_obj = [self.rect_left,self.rect_right])

        # draw instructions wait for scanner t trigger
        this_instruction_string = ('Index finger/b key - Black dot\n'
                                    'Middle finger/y key - White dot\n\n\n'
                                    '          [waiting for scanner]')
        
        draw_instructions(self.win, this_instruction_string, keys = [self.settings['mri'].get('sync', 't')], visual_obj = [self.rect_left,self.rect_right])


        # start recording gaze
        if self.eyetracker_on:
            self.start_recording_eyetracker()

        self.start_experiment()
        
        # cycle through trials
        for trl in self.all_trials: 
            trl.run() # run forrest run


        print('Expected number of responses: %d'%(len(self.fixation_switch_times)+1))
        print('Total subject responses: %d'%self.total_responses)
        print('Correct responses (within 0.8s of dot color change): %d'%self.correct_responses)
          

        self.close() # close session
        


class FeatureSession(ExpSession):
    
    def __init__(self, output_str, output_dir, settings_file, eyetracker_on, macbook_bool): # initialize child class

        """ Initializes FeatureSession object. 
      
        Parameters
        ----------
        output_str : str
            Basename for all output-files (like logs), e.g., "sub-01_task-PRFfeature_run-1"
        output_dir : str
            Path to desired output-directory (default: None, which results in $pwd/logs)
        settings_file : str
            Path to yaml-file with settings (default: None, which results in the package's
            default settings file (in data/default_settings.yml)
        macbook_bool: bool
                variable to know if using macbook for running experiment or not
        """


        # need to initialize parent class (ExpSession), indicating output infos
        super().__init__(output_str = output_str, output_dir = output_dir, settings_file = settings_file, 
                        macbook_bool = macbook_bool, eyetracker_on = eyetracker_on)
        

    
    def create_stimuli(self):

        """ Create Stimuli - pRF bars and fixation dot """
        
        #generate PRF stimulus
        self.feature_stim = FeatureStim(session = self, 
                                        bar_width_ratio = self.settings['stimuli']['feature']['bar_width_ratio'], 
                                        grid_pos = self.grid_pos
                                        )
        
        # Convert fixation dot radius in degrees to pixels for a given Monitor object
        fixation_rad_pix = tools.monitorunittools.deg2pix(self.settings['stimuli']['fix_dot_size_deg'], 
                                                        self.monitor)/2 
        
        # create black fixation circle
        self.fixation = visual.Circle(self.win, units = 'pix', radius = fixation_rad_pix, 
                                            fillColor = self.settings['stimuli']['fix_dot_color'], 
                                            lineColor = self.settings['stimuli']['fix_line_color'],
                                            fillColorSpace = self.settings['stimuli']['colorSpace'],
                                            lineColorSpace = self.settings['stimuli']['colorSpace'])  



    def create_trials(self):

        """ Creates trials (before running the session) """

        #
        # counter for responses
        self.total_responses = 0
        #self.correct_responses = 0

        ## get condition names and randomize them 
        ## setting order for what condition to attend per mini block 
        self.attend_block_conditions = randomize_conditions(self.settings['stimuli']['feature']['conditions'])
        print('Conditions to attend throughout blocks will be %s'%str(self.attend_block_conditions))


        ## get all possible bar positions

        # define bar width 
        bar_width_ratio = self.settings['stimuli']['feature']['bar_width_ratio']
        self.bar_width_pix = self.screen * bar_width_ratio

        # define number of bars per direction
        num_bars = np.array(self.screen)/self.bar_width_pix; num_bars = np.array(num_bars,dtype=int)

        # all possible positions in pixels [x,y] for for midpoint of
        # vertical bar passes, 
        ver_y = np.linspace((-self.screen[1]/2 + self.bar_width_pix[1]/2),
                            (self.screen[1]/2 - self.bar_width_pix[1]/2),
                            num_bars[1])

        ver_bar_pos_pix = np.array([np.array([0,y]) for _,y in enumerate(ver_y)])

        # horizontal bar passes 
        hor_x = np.linspace((-self.screen[0]/2 + self.bar_width_pix[0]/2),
                            (self.screen[0]/2 - self.bar_width_pix[0]/2),
                            num_bars[0])

        hor_bar_pos_pix = np.array([np.array([x,0]) for _,x in enumerate(hor_x)])


        # set bar midpoint position and direction for each condition
        # for all mini blocks
        self.all_bar_pos = set_bar_positions(attend_block_conditions = self.attend_block_conditions,
                                            horizontal_pos = hor_bar_pos_pix,
                                            vertical_pos = ver_bar_pos_pix,
                                            mini_blocks = self.settings['stimuli']['feature']['mini_blocks'], 
                                            num_bars = len(self.attend_block_conditions), 
                                            num_ver_bars = 2, 
                                            num_hor_bars = 2)

        # save bar positions for run in output folder
        save_bar_position(self.all_bar_pos, self.settings['stimuli']['feature']['mini_blocks'], 
                          os.path.join(self.output_dir, self.output_str+'_bar_positions.pkl'))


        # number of TRs per "type of stimuli"
        cue_TR = self.settings['stimuli']['feature']['cue_TR']
        empty_TR = self.settings['stimuli']['feature']['empty_TR']
        # get info from first block, to know how many trials in a mini block (all miniblocks have same length)
        dict_blk0 = self.all_bar_pos['mini_block_0'][list(self.all_bar_pos['mini_block_0'].keys())[0]]
        mini_block_TR = np.array(dict_blk0[list(dict_blk0.keys())[0]]).shape[0]

        # list with order of "type of stimuli" throught experiment (called bar direction to make analogous with other class)
        bar_pass_direction = self.settings['stimuli']['feature']['bar_pass_direction'] 

        # set number of trials,
        # list of type of trial (cue, empty, miniblock) for all TRs,
        # list of strings/lists with bar direction/orientation or 'empty',
        # list of midpoint position (x,y) of bars for all TRs (if empty, then nan)
        self.trial_number, self.trial_type_all, self.bar_pass_direction_all, self.bar_midpoint_all = define_feature_trials(bar_pass_direction, 
                                                                                                                      self.all_bar_pos, 
                                                                                                                      empty_TR = empty_TR, 
                                                                                                                      cue_TR = cue_TR, 
                                                                                                                      mini_block_TR = mini_block_TR)
                
        print("Total number of (expected) TRs: %d"%self.trial_number)

        # set plotting order index, to randomize which bars appear on top, for all trials in all miniblocks
        # and get hemifield position of attended bar
        self.drawing_ind = []
        self.hemifield = []

        for indx, val in enumerate(self.trial_type_all):
            
            if 'mini_block' in val:
                ind_list = np.arange(self.settings['stimuli']['feature']['num_bars'])
                np.random.shuffle(ind_list)
                
                self.drawing_ind.append(ind_list)

                if self.bar_pass_direction_all[indx][0] == 'horizontal': # if attended bar vertical (horizontal bar pass)

                    if self.bar_midpoint_all[indx][0][0] < 0: # append hemifield
                        
                        self.hemifield.append('left')
                    
                    else:

                        self.hemifield.append('right')

                elif self.bar_pass_direction_all[indx][0] == 'vertical': # if attended bar horizontal (vertical bar pass)

                    if self.bar_midpoint_all[indx][0][-1] < 0: # append hemifield
                        
                        self.hemifield.append('down')
                    
                    else:
                        
                        self.hemifield.append('up')
                    
                
            else: # if not in miniblock, these are nan
                self.drawing_ind.append([np.nan])
                self.hemifield.append(np.nan)


        # save relevant trial info in df (for later analysis)
        save_all_TR_info(self.all_bar_pos, self.trial_type_all, self.attend_block_conditions, 
                        self.hemifield, self.drawing_ind, os.path.join(self.output_dir, self.output_str+'_trial_info.csv'))
                         
        # if in scanner, we want it to be synced to trigger, so lets increase trial time (in seconds, like TR)
        max_trial_time = 5 if self.settings['mri']['scanner']==True else self.settings['mri']['TR']

        # append all trials
        self.all_trials = []
        for i in range(self.trial_number):

            self.all_trials.append(FeatureTrial(session = self,
                                                trial_nr = i, 
                                                phase_durations = np.array([max_trial_time]),
                                                attend_block_conditions = self.attend_block_conditions, 
                                                bar_pass_direction_at_TR = self.bar_pass_direction_all[i],
                                                bar_midpoint_at_TR = self.bar_midpoint_all[i],
                                                trial_type_at_TR = self.trial_type_all[i],
                                                ))


        # total experiment time (in seconds)
        self.total_time = self.trial_number * max_trial_time 

        # define time points for element orientation to change
        # switch orientation time points
        if self.settings['stimuli']['ori_shift_rate'] == 'TR':
            ori_shift_rate = 1/self.bar_step 
        else:
            ori_shift_rate = self.settings['stimuli']['ori_shift_rate']
        self.ori_switch_times = np.arange(0,self.total_time,1/ori_shift_rate)
        # counter for orientation switches
        self.ori_counter = 0
        # index for orientation
        self.ori_ind = 0

        # print window size just to check, not actually needed
        print(self.screen)


    def run(self):
        """ Loops over trials and runs them """

        # update color of settings
        self.settings = get_average_color(self.output_dir, self.settings, task = 'feature')

        # create trials before running!
        self.create_stimuli()
        self.create_trials() 

        # if eyetracking then calibrate
        if self.eyetracker_on:
            self.calibrate_eyetracker()

        # draw instructions wait a few seconds
        this_instruction_string = ('During the experiment\nyou will see green and red bars\n'
                                'oriented vertically or horizontally\n'
                                'throughout the screen\n\n\n'
                                '[Press right arrow key or\n'
                                'middle finger to continue]\n\n'
                                '[Press left arrow key or\n'
                                'index finger to skip]\n\n')

        key_pressed = draw_instructions(self.win, this_instruction_string, keys = ['b','left','y','right'], visual_obj = [self.rect_left,self.rect_right])

        if key_pressed[0] not in ['left','b']: #if instructions not skipped

            # draw instructions wait a few seconds
            this_instruction_string = ('These bars can be\n'
                                        'on the right/left side\n'
                                        'or above/below the\n'
                                        'central fixation dot\n\n\n'
                                        '[Press right arrow key or\n'
                                        'middle finger to continue]\n\n')
            
            draw_instructions(self.win, this_instruction_string, keys = ['y','right'], visual_obj = [self.rect_left,self.rect_right])

            this_instruction_string = ('Your task is to fixate\n'
                                        'at the center of the screen,\n'
                                        'and indicate if one of the bars\n'
                                        'is on the SAME side of the dot\n'
                                        'relative to the PREVIOUS trial\n\n\n'
                                        '[Press right arrow key or\n'
                                        'middle finger to continue]\n\n')
            
            draw_instructions(self.win, this_instruction_string, keys = ['y','right'], visual_obj = [self.rect_left,self.rect_right])

            this_instruction_string = ('The experiment is divided\n'
                                        'into different mini-blocks.\n\n'
                                        'At the beggining of each\n'
                                        'you will see a single bar,\n'
                                        'at the center of the screen.\n\n\n'
                                        '[Press right arrow key or\n'
                                        'middle finger to continue]\n\n')
            
            draw_instructions(self.win, this_instruction_string, keys = ['y','right'], visual_obj = [self.rect_left,self.rect_right])

            this_instruction_string = ('This bar will be\n'
                                        'vertical/horizontal and\n'
                                        'green/red\n\n'
                                        'That will be the bar\n'
                                        'that you have to search for.\n\n\n'
                                        '[Press right arrow key or\n'
                                        'middle finger to continue]\n\n')
            
            draw_instructions(self.win, this_instruction_string, keys = ['y','right'], visual_obj = [self.rect_left,self.rect_right])


            # draw instructions wait a few seconds
            this_instruction_string = ('Do NOT look at the bars!\n'
                                        'Please fixate at the center,\n'
                                        'and do not move your eyes\n\n\n'
                                        '[Press right arrow key or\n'
                                        'middle finger to continue]\n\n')
            
            draw_instructions(self.win, this_instruction_string, keys = ['y','right'], visual_obj = [self.rect_left,self.rect_right])

        # draw instructions wait for scanner t trigger
        this_instruction_string = ('Index finger or\n'
                                    'left arrow key - same side\n\n'
                                    'Middle finger or\n'
                                    'right arrow key - different side\n\n\n'
                                    '          [waiting for scanner]')
        
        draw_instructions(self.win, this_instruction_string, keys = [self.settings['mri'].get('sync', 't')], visual_obj = [self.rect_left,self.rect_right])

        # start recording gaze
        if self.eyetracker_on:
            self.start_recording_eyetracker()

        self.start_experiment()
        
        # cycle through trials
        for trl in self.all_trials: 
            trl.run() # run forrest run


        print('Total subject responses: %d'%self.total_responses)
          

        self.close() # close session



class FlickerSession(ExpSession):
    
    def __init__(self, output_str, output_dir, settings_file, eyetracker_on, macbook_bool): # initialize child class

        """ Initializes FlickerSession object. 
      
        Parameters
        ----------
        output_str : str
            Basename for all output-files (like logs), e.g., "sub-01_task-PRFflicker_run-1"
        output_dir : str
            Path to desired output-directory (default: None, which results in $pwd/logs)
        settings_file : str
            Path to yaml-file with settings (default: None, which results in the package's
            default settings file (in data/default_settings.yml)
        macbook_bool: bool
                variable to know if using macbook for running experiment or not
        """


        # need to initialize parent class (ExpSession), indicating output infos
        super().__init__(output_str = output_str, output_dir = output_dir, settings_file = settings_file, 
                        macbook_bool = macbook_bool, eyetracker_on = eyetracker_on)
        

    
    def create_stimuli(self):

        """ Create Stimuli - pRF bars and fixation dot """
        
        #generate PRF stimulus
        self.flicker_stim = FlickerStim(session = self, 
                                        bar_width_ratio = self.settings['stimuli']['flicker']['bar_width_ratio'], 
                                        grid_pos = self.grid_pos
                                        )
        
        # Convert fixation dot radius in degrees to pixels for a given Monitor object
        fixation_rad_pix = tools.monitorunittools.deg2pix(self.settings['stimuli']['fix_dot_size_deg'], 
                                                        self.monitor)/2 
        
        # create black fixation circle
        self.fixation = visual.Circle(self.win, units = 'pix', radius = fixation_rad_pix, 
                                            fillColor = self.settings['stimuli']['fix_dot_color'], 
                                            lineColor = self.settings['stimuli']['fix_line_color'],
                                            fillColorSpace = self.settings['stimuli']['colorSpace'],
                                            lineColorSpace = self.settings['stimuli']['colorSpace'])  



    def create_trials(self):

        """ Creates trials (before running the session) """

        #
        # start luminance, will increase or decrease given responses
        self.lum_responses = 1

        self.updated_settings = self.settings['stimuli']['conditions']

        ## get all possible bar positions

        # define bar width 
        bar_width_ratio = self.settings['stimuli']['flicker']['bar_width_ratio']
        self.bar_width_pix = self.screen * bar_width_ratio

        # define number of bars per direction
        num_bars = np.array(self.screen)/self.bar_width_pix; num_bars = np.array(num_bars,dtype=int)

        # all possible positions in pixels [x,y] for for midpoint of
        # vertical bar passes
        ver_y = np.linspace((-self.screen[1]/2 + self.bar_width_pix[1]/2),
                            (self.screen[1]/2 - self.bar_width_pix[1]/2),
                            num_bars[1])

        ver_bar_pos_pix = np.array([np.array([0,y]) for _,y in enumerate(ver_y)])


        # positions to put bars of square, per trial
        # eccentricity index for bar positions: 0 - furthest ecc; 3 - closest ecc
        self.bar_ecc_index_all = self.settings['stimuli']['flicker']['bar_ecc_index']

        # eccentricity (in pixels) of bar position for trial (if empty, then nan) 
        self.ecc_midpoint_all = ver_bar_pos_pix[self.bar_ecc_index_all][...,1]

        # one eccentricity for trial
        self.trial_number = len(self.bar_ecc_index_all)

        print("Total number of trials: %d"%self.trial_number)

        # get condition names and randomize them for each trial 
        key_list = []
        for key in self.settings['stimuli']['conditions']:
            if key != 'background': # we don't want to show background gabors in background
                key_list.append(key)

        # max trial time
        max_trial_time = self.settings['stimuli']['flicker']['max_trial_time']*60
        
        # define how many times square colors switch, according to flick rate defined 
        flick_rate = self.settings['stimuli']['flicker']['flick_rate']

        # number of samples in trial
        n_samples = max_trial_time * flick_rate

        # repeat keys, so for each trial it shows each condition X times
        key_list = np.array(key_list*round(n_samples/len(key_list)))
        phase_conditions = key_list
        
        for r in range(self.trial_number-1):            
            phase_conditions = np.vstack((phase_conditions,key_list))

        
        # define list with number of phases and their duration (duration of each must be the same)
        self.phase_durations = np.repeat(1/flick_rate,phase_conditions.shape[-1]) 


        # append all trials
        self.all_trials = []
        for i in range(self.trial_number):

            self.all_trials.append(FlickerTrial(session = self,
                                                trial_nr = i, 
                                                phase_durations = self.phase_durations,
                                                phase_names = phase_conditions[i],
                                                bar_ecc_index_at_trial = self.bar_ecc_index_all[i],
                                                ecc_midpoint_at_trial = self.ecc_midpoint_all[i]
                                                ))


        # total experiment time (in seconds)
        self.total_time = self.trial_number * max_trial_time

        # define time points for element orientation to change
        # switch orientation time points
        if self.settings['stimuli']['ori_shift_rate'] == 'TR':
            ori_shift_rate = 1/self.settings['mri']['TR'] # in seconds
        else:
            ori_shift_rate = self.settings['stimuli']['ori_shift_rate']
        self.ori_switch_times = np.arange(0,self.total_time,1/ori_shift_rate)
        # counter for orientation switches
        self.ori_counter = 0
        # index for orientation
        self.ori_ind = 0

        # print window size just to check, not actually needed
        print(self.screen)


    def run(self):
        """ Loops over trials and runs them """

        # create trials before running!
        self.create_stimuli()
        self.create_trials() 

        # if eyetracking then calibrate
        if self.eyetracker_on:
            self.calibrate_eyetracker()

        # draw instructions wait a few seconds
        this_instruction_string = ('Welcome to this experiment!\n\n'
                                'In the first task, you will see a\n'
                                'flickering red/green square\n\n\n'
                                '[Press right arrow key or\n'
                                'middle finger to continue]\n\n'
                                '[Press left arrow key or\n'
                                'index finger to skip]\n\n')

        key_pressed = draw_instructions(self.win, this_instruction_string, keys = ['right','y','left','b'], visual_obj = [self.rect_left,self.rect_right])

        if key_pressed[0] not in ['left','b']: #if instructions not skipped

            this_instruction_string = ('If you press the\n'
                                        'right arrow key/middle finger OR\n'
                                        'left arrow key/index finger,\n'
                                        'you will realize that\n'
                                        'the flickering changes\n\n\n'
                                        '[Press right arrow key or\n'
                                        'middle finger to continue]\n\n')
            
            draw_instructions(self.win, this_instruction_string, keys = ['right','y'], visual_obj = [self.rect_left,self.rect_right])


            this_instruction_string = ('Your task is to fixate\n'
                                        'at the center of the screen,\n'
                                        'and press the\n'
                                        'right arrow key/middle finger OR\n'
                                        'left arrow key/index finger,\n'
                                        'until the square does not\n'
                                        'flicker anymore\n\n\n'
                                        '[Press right arrow key or\n'
                                        'middle finger to continue]\n\n')
            
            draw_instructions(self.win, this_instruction_string, keys = ['right','y'], visual_obj = [self.rect_left,self.rect_right])


            # draw instructions wait a few seconds
            this_instruction_string = ('Do NOT look at the square!\n\n'
                                        'Please fixate at the center,\n'
                                        'and do not move your eyes\n\n\n'
                                        '[Press right arrow key or\n'
                                        'middle finger to continue]\n\n')
            
            draw_instructions(self.win, this_instruction_string, keys = ['right','y'], visual_obj = [self.rect_left,self.rect_right])

        # draw instructions wait for scanner t trigger
        this_instruction_string = ('When you are certain the square\n'
                                    'does not flicker anymore,\n'
                                    'press the space bar\n'
                                    '(or pinky finger if in the scanner)\n\n'
                                    'Ready when you are!\n\n\n'
                                    '[Press left arrow key or\n'
                                    'index finger to start]\n\n')
        
        draw_instructions(self.win, this_instruction_string, keys = ['left','b'], visual_obj = [self.rect_left,self.rect_right])

        # start recording gaze
        if self.eyetracker_on:
            self.start_recording_eyetracker()

        self.start_experiment()
        
        # cycle through trials
        for trl in self.all_trials: 
            trl.run() # run forrest run
          

        self.close() # close session





