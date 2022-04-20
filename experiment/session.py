
import os
import os.path as op
import numpy as np

from exptools2.core import Session, PylinkEyetrackerSession

from trial import PRFTrial, FeatureTrial, FlickerTrial
from stim import PRFStim, FeatureStim, FlickerStim

from psychopy import visual, tools
from psychopy.data import QuestHandler, StairHandler

import itertools
import pickle

from utils import *


class ExpSession(PylinkEyetrackerSession):

    def __init__(self, output_str, output_dir, settings_file, eyetracker_on = True):  # initialize child class

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
            super().__init__(output_str = output_str, output_dir = output_dir, settings_file = settings_file, eyetracker_on = eyetracker_on)

            # set size of display
            if self.settings['window_extra']['display'] == 'square':
                self.screen = np.array([self.win.size[1], self.win.size[1]])
                rect_contrast = 1
            
            elif self.settings['window_extra']['display'] == 'rectangle':
                self.screen = np.array([self.win.size[0], self.win.size[1]])
                rect_contrast = 0 # then rectangles will be hidden

            if self.settings['window_extra']['mac_bool']: # to compensate for macbook retina display
                self.screen = self.screen/2
                print('Running experiment on macbook, defining display accordingly')

            # some MRI params
            self.bar_step = self.settings['mri']['TR'] # in seconds
            self.mri_trigger = self.settings['mri']['sync'] #'t'

            ## make grid of possible positions for gabors 
            # (grid spans whole display, bar will alter specific part of grid)

            ## first set the number of elements that fit each dimension
            elem_num = np.array(self.settings['stimuli']['num_elem'])

            gabor_diameter_pix = np.array(self.screen)/(elem_num * self.settings['stimuli']['gab_ratio'])
            self.gabor_diameter_pix = gabor_diameter_pix[0]

            print('gabor diameter in pix %s'%str(self.gabor_diameter_pix))
            print('gabor diameter in deg %s'%str(tools.monitorunittools.pix2deg(self.gabor_diameter_pix, self.monitor)))
            
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
   
    def __init__(self, output_str, output_dir, settings_file, eyetracker_on):  # initialize child class

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

        # need to initialize parent class (ExpSession), indicating output infos
        super().__init__(output_str = output_str, output_dir = output_dir, settings_file = settings_file, 
                        eyetracker_on = eyetracker_on)

        

    def create_stimuli(self):

        """ Create Stimuli - pRF bar """
        
        #generate PRF stimulus
        self.prf_stim = PRFStim(session = self, 
                                bar_width_ratio = self.settings['stimuli']['prf']['bar_width_ratio'], 
                                grid_pos = self.grid_pos
                                )


    def create_trials(self):

        """ Creates trials (before running the session) """

        #
        # counter for responses
        self.total_responses = 0
        self.expected_responses = 0
        self.correct_responses = 0
        self.bar_counter = 0

        # define bar width 
        bar_width_ratio = self.settings['stimuli']['prf']['bar_width_ratio']
        self.bar_width_pix = self.screen*bar_width_ratio

        # number of TRs per "type of stimuli"
        bar_pass_hor_TR = self.settings['stimuli']['prf']['bar_pass_hor_TR'] 
        bar_pass_ver_TR = self.settings['stimuli']['prf']['bar_pass_ver_TR']

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
            if 'empty' in bartype: # empty screen
                trial_number += self.settings['stimuli']['prf'][bartype+'_TR']
                bar_pass_direction_all = bar_pass_direction_all + np.repeat('empty',self.settings['stimuli']['prf'][bartype+'_TR']).tolist()
                bar_pos_array.append([np.array([np.nan,np.nan]) for i in range(self.settings['stimuli']['prf'][bartype+'_TR'])])

            elif bartype in np.array(['U-D','D-U']): # vertical bar pass
                trial_number += bar_pass_ver_TR
                self.expected_responses += bar_pass_ver_TR
                bar_pass_direction_all =  bar_pass_direction_all + np.repeat(bartype,bar_pass_ver_TR).tolist()
                
                # order depending on starting point for bar pass, and append to list
                position_list = np.sort(ver_bar_pos_pix,axis=0) if bartype=='D-U' else np.sort(ver_bar_pos_pix,axis=0)[::-1]
                bar_pos_array.append(position_list)

            elif bartype in np.array(['L-R','R-L']): # horizontal bar pass
                trial_number += bar_pass_hor_TR
                self.expected_responses += bar_pass_hor_TR
                bar_pass_direction_all =  bar_pass_direction_all + np.repeat(bartype,bar_pass_hor_TR).tolist()
                
                # order depending on starting point for bar pass, and append to list
                position_list = np.sort(hor_bar_pos_pix,axis=0) if bartype=='L-R' else np.sort(hor_bar_pos_pix,axis=0)[::-1]
                bar_pos_array.append(position_list)

        self.trial_number = trial_number # total number of trials 
        print("Total number of (expected) TRs: %d"%self.trial_number)
        self.bar_pass_direction_all = bar_pass_direction_all # list of strings with bar orientation/empty

        # list of midpoint position (x,y) of bar for all TRs (if empty, then nan)
        self.bar_midpoint_all = np.array([val for sublist in bar_pos_array for val in sublist])

        # make boolean array to see which trials are stim trials
        self.bar_bool = [False if x == 'empty' else True for _,x in enumerate(self.bar_pass_direction_all)]

        # if in scanner, we want it to be synced to trigger, so lets increase trial time (in seconds, like TR)
        max_trial_time = 5 if self.settings['stimuli']['prf']['sync_scanner']==True else self.settings['mri']['TR']

        # flicker frequency
        flick_rate = self.settings['stimuli']['prf']['flick_rate']

        # number of samples in trial
        n_samples = max_trial_time * flick_rate

        ## get condition names and randomize them for each trial 
        
        key_list = []
        for key in self.settings['stimuli']['prf']['color_categories']:  
            # if showing red and green bars
            if self.settings['stimuli']['prf']['task_on_main_categories']: 
                key_list.append(key)
            
            else: # other color variants
                for name in self.settings['stimuli']['prf']['task_colors'][key]:
                    key_list.append(name)

        # define how many times bar features switch during TR, according to flick rate defined 
        if self.settings['stimuli']['prf']['flick_stim_rate'] == 'TR': # if changing features at every TR

            phase_conditions = np.repeat(key_list, sum(self.bar_bool)/len(key_list)) # only do this for bar showing trials
            np.random.shuffle(phase_conditions) # randomized conditions, for attention to bar task

            if self.settings['stimuli']['prf']['flick_on_off'] == True: # interleave with background if we want and on-off bar

                self.phase_conditions = np.array([list(np.tile([val,'background'],int(np.round(n_samples)))) for _,val in enumerate(phase_conditions)])

            else:
                self.phase_conditions = np.array([list(np.tile(val,int(np.round(n_samples)))) for _,val in enumerate(phase_conditions)])

            # define list with number of phases and their duration (duration of each must be the same)
            self.phase_durations = np.repeat(max_trial_time/self.phase_conditions.shape[-1], self.phase_conditions.shape[-1])   

        else: # if changing features randomly at flick rate

            # repeat keys, so for each bar pass it shows each condition X times
            key_list = np.array(key_list*round(n_samples/len(key_list))) 

            if self.settings['stimuli']['prf']['flick_on_off'] == True: # interleave with background if we want and on-off bar

                on_list = list(key_list)
                off_list = list(np.tile('background',len(on_list)))

                key_list = on_list + off_list
                key_list[::2] = on_list
                key_list[1::2] = off_list
            else:
                key_list = list(key_list) + list(key_list) 

            phase_conditions = key_list

            # stack them in trial
            for r in range(sum(self.bar_bool)-1):            
                phase_conditions = np.vstack((phase_conditions,key_list))

            self.phase_conditions = phase_conditions

            # define list with number of phases and their duration (duration of each must be the same)
            self.phase_durations = np.repeat(max_trial_time/self.phase_conditions.shape[-1], self.phase_conditions.shape[-1])

        ## now make phase condition array the same size as task, 
        # i.e. fill non bar trials with background
        all_phase_conditions = (np.stack(np.tile('background', self.phase_conditions.shape[-1]) for i in range(self.trial_number)))
        all_phase_conditions[self.bar_bool] = self.phase_conditions.copy()

        self.phase_conditions = all_phase_conditions.copy() 

        # total experiment time (in seconds)
        self.total_time = self.trial_number * max_trial_time  

        # append all trials
        self.all_trials = []
        for i in range(self.trial_number):

            self.all_trials.append(PRFTrial(session =self ,
                                            trial_nr = i,  
                                            phase_durations = self.phase_durations,
                                            phase_names = self.phase_conditions[i],
                                            bar_pass_direction_at_TR = self.bar_pass_direction_all[i],
                                            bar_midpoint_at_TR = self.bar_midpoint_all[i]
                                            ))

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

        # for counting bars and checking responses in real time
        self.bar_timing = [i*self.settings['mri']['TR'] for i,x in enumerate(self.bar_pass_direction_all) if x!='empty']


        # print window size just to check, not actually needed
        print(self.screen)
        print(tools.monitorunittools.pix2deg(self.screen[0], self.monitor))

    
    def run(self):
        """ Loops over trials and runs them """

        # update color of settings
        self.settings = get_average_color(self.output_dir, self.settings,
                    updated_color_names = ['pink', 'orange', 'yellow', 'blue'])
        

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
                                    '[Press right index finger\nto continue]\n'
                                    '[Press left index finger\nto skip]')

        key_pressed = draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['left_index']+self.settings['keys']['right_index'], 
            visual_obj = [self.rect_left,self.rect_right])

        if key_pressed[0] not in self.settings['keys']['left_index']: #if instructions not skipped

            # draw instructions wait a few seconds
            this_instruction_string = ('Your task is to fixate\n'
                                        'at the center of the screen,\n'
                                        'and indicate the\n'
                                        'bar color category\n'
                                        'every time the bar moves\n\n\n'
                                        '[Press right index finger\nto continue]')
            
            draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['right_index'], visual_obj = [self.rect_left,self.rect_right])

            # draw instructions wait a few seconds
            this_instruction_string = ('The RED category includes\n'
                                        'pink and orange tones.\n\n'
                                        'The GREEN category includes\n'
                                        'blue and yellow tones\n\n\n'
                                        '[Press right index finger\nto continue]')
            
            draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['right_index'], visual_obj = [self.rect_left,self.rect_right])
            
            # draw instructions wait a few seconds
            this_instruction_string = ('Do NOT look at the bars!\n'
                                        'Please fixate at the center,\n'
                                        'and do not move your eyes\n\n\n'
                                        '[Press right index finger\nto continue]')
            
            draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['right_index'], visual_obj = [self.rect_left,self.rect_right])

        # draw instructions wait for scanner t trigger
        this_instruction_string = ('Left index finger - RED color category\n'
                                    'Right index finger - GREEN color category\n\n\n'
                                    '[waiting for scanner]')
        
        draw_instructions(self.win, this_instruction_string, keys = [self.settings['mri'].get('sync', 't')], visual_obj = [self.rect_left,self.rect_right])


        # start recording gaze
        if self.eyetracker_on:
            self.start_recording_eyetracker()

        self.start_experiment()
        
        # cycle through trials
        for trl in self.all_trials: 
            trl.run() # run forrest run


        print('Expected number of responses: %d'%(self.expected_responses))
        print('Total subject responses: %d'%self.total_responses)
        print('Correct responses: %d'%self.correct_responses)
        print('Accuracy %.2f %%'%(self.correct_responses/self.expected_responses*100))
          

        self.close() # close session
        


class FeatureSession(ExpSession):
    
    def __init__(self, output_str, output_dir, settings_file, eyetracker_on, att_color = 'color_green'): # initialize child class

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
        """


        # need to initialize parent class (ExpSession), indicating output infos
        super().__init__(output_str = output_str, output_dir = output_dir, settings_file = settings_file, 
                        eyetracker_on = eyetracker_on)
        
        self.att_color = att_color

        ## set task colors
        self.task_colors = self.settings['stimuli']['feature']['task_colors']
    
    def create_stimuli(self):

        """ Create Stimuli - pRF bars and fixation dot """
        
        #generate PRF stimulus
        self.feature_stim = FeatureStim(session = self, 
                                        bar_width_ratio = self.settings['stimuli']['feature']['bar_width_ratio'], 
                                        grid_pos = self.grid_pos
                                        )


    def create_trials(self):

        """ Creates trials (before running the session) """

        # some counters for internal bookeeping
        self.total_responses = 0
        self.correct_responses = 0
        self.bar_counter = 0
        self.thisResp = []

        ## set attended bar color 
        self.att_condition = [val for val in self.settings['stimuli']['feature']['conditions'] if self.att_color in val][0]
        self.unatt_condition = [val for val in self.settings['stimuli']['feature']['conditions'] if self.att_color not in val][0]

        ## get all possible bar positions

        # define bar width 
        bar_width_ratio = self.settings['stimuli']['feature']['bar_width_ratio']
        self.bar_width_pix = self.screen * bar_width_ratio

        # define number of bars per direction
        num_bars = np.array(self.settings['stimuli']['feature']['num_bar_position']) 

        # all possible positions in pixels [x,y] for midpoint of
        # vertical bar passes, 
        ver_y = np.sort(np.concatenate((-np.arange(self.bar_width_pix[1]/2,self.screen[1]/2,self.bar_width_pix[1])[0:int(num_bars[1]/2)],
                                        np.arange(self.bar_width_pix[1]/2,self.screen[1]/2,self.bar_width_pix[1])[0:int(num_bars[1]/2)])))

        ver_bar_pos_pix = np.array([np.array([0,y]) for _,y in enumerate(ver_y)])

        # horizontal bar passes 
        hor_x = np.sort(np.concatenate((-np.arange(self.bar_width_pix[0]/2,self.screen[0]/2,self.bar_width_pix[0])[0:int(num_bars[0]/2)],
                                        np.arange(self.bar_width_pix[0]/2,self.screen[0]/2,self.bar_width_pix[0])[0:int(num_bars[0]/2)])))

        hor_bar_pos_pix = np.array([np.array([x,0]) for _,x in enumerate(hor_x)])


        # set bar midpoint position and direction for each condition
        self.all_bar_pos = set_bar_positions(pos_dict = {'horizontal': hor_bar_pos_pix, 'vertical': ver_bar_pos_pix},
                                                attend_condition = self.att_condition, 
                                                unattend_condition = self.unatt_condition,
                                                attend_orientation = ['vertical','horizontal'],
                                                unattend_orientation = ['vertical','horizontal'])

        # save bar positions for run in output folder
        save_bar_position(self.all_bar_pos, 
                          op.join(self.output_dir, self.output_str+'_bar_positions.pkl'))

        # list with order of "type of stimuli" throughout experiment (called bar direction to make analogous with other class)
        bar_pass_direction = self.settings['stimuli']['feature']['bar_pass_direction'] 

        # number of TRs per "type of stimuli"
        empty_TR = self.settings['stimuli']['feature']['empty_TR']
        task_trial_TR = self.settings['stimuli']['feature']['task_trial_TR'] 

        # set number of trials,
        # list of type of trial (empty, task) for all TRs,
        # list of strings/lists with bar direction/orientation or 'empty',
        # list of midpoint position (x,y) of bars for all TRs (if empty, then nan)
        self.trial_number, self.trial_type_all, self.bar_pass_direction_all, self.bar_midpoint_all = define_feature_trials(bar_pass_direction, 
                                                                                                                            self.all_bar_pos, 
                                                                                                                            empty_TR = empty_TR, 
                                                                                                                            task_trial_TR = task_trial_TR)
                
        print("Total number of (expected) TRs: %d"%self.trial_number)

        ## get eccentricity indice for all trials
        # of attended and UNattended bar
        self.ecc_ind_all = {}
        self.ecc_ind_all[self.att_condition] = get_bar_eccentricity(self.all_bar_pos, 
                                                        hor_bar_pos_pix = hor_bar_pos_pix, 
                                                        ver_bar_pos_pix = ver_bar_pos_pix, 
                                                        bar_key = 'attended_bar')
        self.ecc_ind_all[self.unatt_condition] = get_bar_eccentricity(self.all_bar_pos, 
                                                        hor_bar_pos_pix = hor_bar_pos_pix, 
                                                        ver_bar_pos_pix = ver_bar_pos_pix, 
                                                        bar_key = 'unattended_bar')

        ## randomly assign which color the bar will have, 
        # for target bar (attended color)
        self.ctask_ind_all = {}
        self.ctask_ind_all[self.att_condition] = np.random.randint(2, size = len(self.all_bar_pos['attended_bar']['bar_pass_direction_at_TR']))
        # for distractor bar (unattended color)
        self.ctask_ind_all[self.unatt_condition] = np.random.randint(2, size = len(self.all_bar_pos['unattended_bar']['bar_pass_direction_at_TR']))

        # set plotting order index, to randomize which bars appear on top, for all trials in all miniblocks
        self.drawing_ind = []

        for _, val in enumerate(self.trial_type_all):
            
            if 'task' in val:
                ind_list = np.arange(self.settings['stimuli']['feature']['num_bars'])
                np.random.shuffle(ind_list)
                
                self.drawing_ind.append(ind_list)

            else: # if not in miniblock, these are nan
                self.drawing_ind.append([np.nan])

        # save relevant trial info in df (for later analysis)
        save_all_TR_info(bar_dict = self.all_bar_pos, trial_type = self.trial_type_all, 
                        task_colors = self.task_colors, task_color_ind = self.ctask_ind_all,
                        crossing_ind = self.drawing_ind,
                        ecc_ind = self.ecc_ind_all,
                        output_path = op.join(self.output_dir, self.output_str+'_trial_info.csv'))
                         
        # if in scanner, we want it to be synced to trigger, so lets increase trial time (in seconds, like TR)
        max_trial_time = 5 if self.settings['stimuli']['feature']['sync_scanner']==True else self.settings['mri']['TR']

        # append all trials
        self.all_trials = []

        for i in range(self.trial_number):

            # set phase conditions (for logging) and durations
            if 'task' in self.trial_type_all[i]:
                phase_cond = tuple(['stim','background'])
                phase_dur = tuple([self.settings['stimuli']['feature']['bars_phase_dur'],
                                    max_trial_time-self.settings['stimuli']['feature']['bars_phase_dur']])
                            
            else:
                phase_cond = tuple([self.trial_type_all[i]])
                phase_dur = tuple([max_trial_time])

            self.all_trials.append(FeatureTrial(session = self,
                                                trial_nr = i, 
                                                phase_durations = phase_dur,
                                                phase_names = phase_cond, 
                                                bar_pass_direction_at_TR = self.bar_pass_direction_all[i],
                                                bar_midpoint_at_TR = self.bar_midpoint_all[i],
                                                trial_type_at_TR = self.trial_type_all[i],
                                                num_bars_on_screen = self.settings['stimuli']['feature']['num_bars'],
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

        # make boolean array to see which trials are stim trials
        self.bar_bool = [True if x == 'task' else False for _,x in enumerate(self.trial_type_all)]
        # time in seconds for when bar trial on screen
        self.bar_timing = [x * self.settings['mri']['TR'] for x in np.where(self.bar_bool)[0]]

        # print window size just to check, not actually needed
        print(self.screen)


    def run(self):
        """ Loops over trials and runs them """

        # update color of settings
        self.settings = get_average_color(self.output_dir, self.settings,
                    updated_color_names = ['pink', 'orange', 'yellow', 'blue'])

        # create trials before running!
        self.create_stimuli()
        self.create_trials()

        # if eyetracking then calibrate
        if self.eyetracker_on:
            self.calibrate_eyetracker()

        # draw instructions wait a few seconds
        this_instruction_string = ('During the experiment\nyou will see 2 bars\n'
                                'oriented vertically or horizontally\n'
                                'throughout the screen\n\n\n'
                                '[Press right index finger\nto continue]\n\n'
                                '[Press left index finger\nto skip]\n\n')

        key_pressed = draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['left_index']+self.settings['keys']['right_index'], visual_obj = [self.rect_left,self.rect_right])

        if key_pressed[0] not in self.settings['keys']['left_index']: #if instructions not skipped

            # draw instructions wait a few seconds
            this_instruction_string = ('At the beggining of the experiment\n'
                                        'you will be told which\n'
                                        'bar color category (red/green)\n\n'
                                        'you have to search for.\n\n\n'
                                        '[Press right index finger\nto continue]\n\n')
            

            draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['right_index'], visual_obj = [self.rect_left,self.rect_right])

            this_instruction_string = ('Your task is to fixate\n'
                                        'at the center of the screen,\n'
                                        'and indicate if the attended bar\n'
                                        'is orange/pink (red category)\n'
                                        'or blue/yellow (green category)\n\n\n'
                                        '[Press right index finger\nto continue]\n\n')
            

            draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['right_index'], visual_obj = [self.rect_left,self.rect_right])

            # draw instructions wait a few seconds
            this_instruction_string = ('Do NOT look at the bars!\n'
                                        'Please fixate at the center,\n'
                                        'and do not move your eyes\n\n\n'
                                        '[Press right index finger\nto continue]\n\n')
            

            draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['right_index'], visual_obj = [self.rect_left,self.rect_right])

        # draw instructions wait for scanner t trigger
        if self.att_color == 'color_red':
            this_instruction_string = ('Pay attention to the RED bar\n\n'
                                    'Left index finger - pink\n\n'
                                    'Right index finger - orange\n\n\n'
                                    '[waiting for scanner]')
        elif self.att_color == 'color_green':
            this_instruction_string = ('Pay attention to the GREEN bar\n\n'
                                    'Left index finger - blue\n\n'
                                    'Right index finger - yellow\n\n\n'
                                    '[waiting for scanner]')
        
        draw_instructions(self.win, this_instruction_string, keys = [self.settings['mri'].get('sync', 't')], visual_obj = [self.rect_left,self.rect_right])

        # start recording gaze
        if self.eyetracker_on:
            self.start_recording_eyetracker()

        self.start_experiment()
        
        # cycle through trials
        for trl in self.all_trials: 
            trl.run() # run forrest run


        print('Expected number of responses: %d'%(sum(self.bar_bool)))
        print('Total subject responses: %d'%self.total_responses)
        print('Correct responses: %d'%self.correct_responses)
        print('Overall accuracy %.2f %%'%(self.correct_responses/sum(self.bar_bool)*100))
          

        self.close() # close session



class FlickerSession(ExpSession):
    
    def __init__(self, output_str, output_dir, settings_file, eyetracker_on): # initialize child class

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
        """


        # need to initialize parent class (ExpSession), indicating output infos
        super().__init__(output_str = output_str, output_dir = output_dir, settings_file = settings_file, 
                        eyetracker_on = eyetracker_on)
        

    
    def create_stimuli(self):

        """ Create Stimuli - pRF bars and fixation dot """
        
        #generate PRF stimulus
        self.flicker_stim = FlickerStim(session = self, 
                                        bar_width_ratio = self.settings['stimuli']['flicker']['bar_width_ratio'], 
                                        grid_pos = self.grid_pos
                                        )
    


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

        # bar eccentricities we want to go through
        self.bar_ecc_ind = self.settings['stimuli']['flicker']['bar_ecc_index']
        # number of repetitions per ecc
        self.num_rep_ecc = self.settings['stimuli']['flicker']['num_rep_ecc']

        ## positions to put bars of square, per trial
        # we swap so that 0 - furthest ecc; 3 - closest ecc (different from yml)
        # to save coding hassle for now
        bar_ecc_index_arr = np.tile(np.abs(np.array(self.bar_ecc_ind) - 3), self.num_rep_ecc)

        ## make it a dict, for bookeeping
        # set color comparisons first
        self.ref_color = self.settings['stimuli']['flicker']['ref_color']

        if self.ref_color in list(self.settings['stimuli']['conditions'].keys()): # if comparing red and green
            updat_colors_keys = [c for c in list(self.settings['stimuli']['conditions'].keys()) if c not in ['background', self.ref_color] ]
            
        else:
            updat_colors_keys = []
            for key in list(self.settings['stimuli']['feature']['task_colors'].keys()):
                for name in self.settings['stimuli']['feature']['task_colors'][key]:
                    if name != self.ref_color:
                        updat_colors_keys.append(name) # update all task colors that are not reference color

        # color names to be updated
        self.updat_colors_keys = np.array(updat_colors_keys)
                        
        # now make bar ecc index dictionary
        bar_ecc_index_dict = {}
        # and get eccentricity (in pixels) of bar position for trial (if empty, then nan) 
        ecc_midpoint_dict = {}

        for col in self.updat_colors_keys:
            
            bar_ecc_index_dict[col] = bar_ecc_index_arr.copy()
            # shuffle it to make it random
            np.random.shuffle(bar_ecc_index_dict[col]) 

            ecc_midpoint_dict[col] = ver_bar_pos_pix[bar_ecc_index_dict[col]][...,1]

        # save here so accessible throughout
        self.bar_ecc_index_dict = bar_ecc_index_dict
        self.ecc_midpoint_dict = ecc_midpoint_dict

        ## save eccentricities in array, to use in file naming
        all_ecc_array = (np.hstack(self.bar_ecc_index_dict[x] for x in self.bar_ecc_index_dict.keys()))
        self.all_ecc_array = all_ecc_array*-1 +3

        # save total number of trials (one eccentricity per trial) 
        self.trial_number = len(bar_ecc_index_arr)*len(updat_colors_keys)
        print("Total number of trials: %d"%self.trial_number)

        # max trial time
        max_trial_time = self.settings['stimuli']['flicker']['max_trial_time']*60

        # define how many times square colors switch, according to flick rate defined 
        flick_rate = self.settings['stimuli']['flicker']['flick_rate']

        # number of samples in trial
        n_samples = max_trial_time * flick_rate

        # get condition names for each trial
        for i, cname in enumerate(updat_colors_keys):
            
            key_list = []

            if self.ref_color in list(self.settings['stimuli']['conditions'].keys()): # if comparing red and green
                
                for key in self.settings['stimuli']['conditions']:
                    if key != 'background': 
                        key_list.append(key)
                        
            else: # other color variants
                key_list = [self.ref_color, cname]
                
            # repeat keys, so for each trial it shows each condition X times
            key_list = np.array(key_list*round(n_samples/len(key_list)))
            key_list = list(key_list) + list(key_list)
            phase_conditions = key_list
            
            for r in range(len(bar_ecc_index_arr)-1):            
                phase_conditions = np.vstack((phase_conditions,key_list))
            
            if i == 0:  
                all_phase_conditions = phase_conditions.copy()
            else:
                all_phase_conditions = np.vstack((all_phase_conditions,phase_conditions))
            
        # define list with number of phases and their duration (duration of each must be the same)
        phase_durations = np.repeat(max_trial_time/all_phase_conditions.shape[-1], all_phase_conditions.shape[-1])

        ## again make so that we can access phase throughout
        self.phase_conditions = all_phase_conditions
        self.phase_durations = phase_durations

        # append all trials
        self.all_trials = []
        c_counter = 0
        for i in range(self.trial_number):
            if i in [int(len(bar_ecc_index_arr)*d) for d in np.arange(len(updat_colors_keys)) if d]:
                c_counter += 1 # cheap trick to not worry about colors and trial numbers

            self.all_trials.append(FlickerTrial(session = self,
                                                trial_nr = i, 
                                                phase_durations = self.phase_durations,
                                                phase_names = self.phase_conditions[i],
                                                bar_ecc_index_at_trial = self.bar_ecc_index_dict[updat_colors_keys[c_counter]][int(i-len(bar_ecc_index_arr)*c_counter)],
                                                ecc_midpoint_at_trial = self.ecc_midpoint_dict[updat_colors_keys[c_counter]][int(i-len(bar_ecc_index_arr)*c_counter)]
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
                                'flickering square\n\n\n'
                                '[Press right index finger\nto continue]\n\n'
                                '[Press left index finger\nto skip]\n\n')

        key_pressed = draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['left_index']+self.settings['keys']['right_index'], visual_obj = [self.rect_left,self.rect_right])
        print(self.settings['keys']['left_index']+self.settings['keys']['right_index'])
        if key_pressed[0] not in self.settings['keys']['left_index']: #if instructions not skipped

            this_instruction_string = ('If you press the buttons\n'
                                        'with your left/right index finger\n'
                                        'you will realize that\n'
                                        'the flickering changes\n\n\n'
                                        '[Press right index finger\nto continue]\n\n')
        
            draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['right_index'], visual_obj = [self.rect_left,self.rect_right])


            this_instruction_string = ('Your task is to fixate\n'
                                        'at the center of the screen,\n'
                                        'and press the buttons\n'
                                        'until the square does not\n'
                                        'flicker anymore\n\n\n'
                                        '[Press right index finger\nto continue]\n\n')
            
            draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['right_index'], visual_obj = [self.rect_left,self.rect_right])


            # draw instructions wait a few seconds
            this_instruction_string = ('Do NOT look at the square!\n\n'
                                        'Please fixate at the center,\n'
                                        'and do not move your eyes\n\n\n'
                                        '[Press right index finger\nto continue]\n\n')
            
            draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['right_index'], visual_obj = [self.rect_left,self.rect_right])


        # draw instructions wait for scanner t trigger
        this_instruction_string = ('When you are certain the square\n'
                                    'does not flicker anymore,\n'
                                    'press the space button\n'
                                    '(or right pinky finger\nif in the scanner)\n\n'
                                    'Ready when you are!\n\n\n'
                                    '[Press left index finger\nto start]\n\n')
        

        draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['left_index'], visual_obj = [self.rect_left,self.rect_right])

        # start recording gaze
        if self.eyetracker_on:
            self.start_recording_eyetracker()

        self.start_experiment()
        
        # cycle through trials
        for trl in self.all_trials: 
            trl.run() # run forrest run

        ## make plots (need to improve this)
        #all_ecc_colors = get_average_color(self.output_dir, self.settings, updated_color_names = ['orange','yellow','blue'],
        #                          average_ecc = False)
        #make_lum_plots(all_ecc_colors, out_dir = self.output_dir)
          
        self.close() # close session





