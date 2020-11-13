
import os
import numpy as np
from exptools2.core import Trial

from psychopy import event, tools, colors, visual
from psychopy.visual import TextStim

from utils import *


class PRFTrial(Trial):

    def __init__(self, session, trial_nr, bar_direction_at_TR, bar_midpoint_at_TR, phase_durations,
                phase_names, timing='seconds', *args, **kwargs):

        """ Initializes a PRFTrial object. 

        Parameters
        ----------
        session : exptools Session object
            A Session object (needed for metadata)
        trial_nr: int
            Trial nr of trial
        phase_durations : array-like
            List/tuple/array with phase durations
        phase_names : array-like
            List/tuple/array with names for phases (only for logging),
            optional (if None, all are named 'stim')
        timing : str
            The "units" of the phase durations. Default is 'seconds', where we
            assume the phase-durations are in seconds. The other option is
            'frames', where the phase-"duration" refers to the number of frames.
        bar_direction_at_TR : list
            List/array with the bar direction at each TR. Total length = total # TRs
        bar_midpoint_at_TR : array
            Numpy array with the pairs of positions [x,y] of the midpoint of the bar
            per TR. Shape (#TRs, 2)
            
        """
        
        self.ID = trial_nr # trial identifier, not sure if needed
        self.bar_direction_at_TR = bar_direction_at_TR
        self.bar_midpoint_at_TR = bar_midpoint_at_TR
        self.session = session

        #dummy value: if scanning or simulating a scanner, everything is synced to the output 't' of the scanner
        #phase_durations = [100]

        # phase durations for each condition 
        self.phase_durations = phase_durations
        # name of each condition
        self.phase_names = phase_names 

        super().__init__(session, trial_nr, phase_durations, phase_names, verbose=False, *args, **kwargs)

        # get bar and background positions

        self.position_dictionary = get_object_positions(self.session.grid_pos, self.bar_midpoint_at_TR, self.bar_direction_at_TR,
                                                    self.session.bar_width_pix, screen = self.session.screen, num_bar = 1)
       

    def draw(self): 

        """ Draw stimuli - pRF bar and fixation dot - for each trial """

        current_time = self.session.clock.getTime() # get time

        # background switch time
        if self.session.bckg_counter<len(self.session.bckg_switch_times): # if counter within number of switch moments
            if current_time >= (self.session.bckg_switch_times[self.session.bckg_counter] + self.session.switch_start_time): # when switch time reached, update background contrast and increment counter

                _ , self.session.background_contrast = gradual_shift(curr_point = [self.session.bckg_switch_times[self.session.bckg_counter], self.session.background_contrast],
                                                                  end_point = self.session.bckg_switch_end_point,
                                                                  x_step = self.session.settings['stimuli']['prf']['switch_step'], 
                                                                  slope = self.session.bckg_switch_slope, 
                                                                  L = self.session.settings['stimuli']['conditions']['background']['element_contrast'], 
                                                                  function = 'logistic')
                self.session.bckg_counter += 1


        ## orientation switch times
        if self.session.ori_counter<len(self.session.ori_switch_times): # if counter within number of switch moments
            if current_time >= self.session.ori_switch_times[self.session.ori_counter]: # when switch time reached, switch ori and increment counter
                
                self.session.ori_ind = 0 if (self.session.ori_counter % 2) == 0 else 1
                self.session.ori_counter += 1

        ## draw stim
        if (self.bar_direction_at_TR == 'empty') or (self.bar_direction_at_TR == 'switch_interval'): # if empty trial, show background

            self.session.prf_stim.draw(bar_midpoint_at_TR = np.nan, 
                                       bar_direction_at_TR = np.nan,
                                       this_phase = 'background',
                                       position_dictionary = self.position_dictionary,
                                       orientation_ind = self.session.ori_ind,
                                       background_contrast = self.session.background_contrast) 
            print('background')

        else: # if bar pass at TR, then draw bar

            self.session.prf_stim.draw(bar_midpoint_at_TR = self.bar_midpoint_at_TR, 
                                       bar_direction_at_TR = self.bar_direction_at_TR,
                                       this_phase = self.phase_names[int(self.phase)],
                                       position_dictionary = self.position_dictionary,
                                       orientation_ind = self.session.ori_ind,
                                       background_contrast = self.session.background_contrast) 

            print(self.phase_names[int(self.phase)]) #'ori_left')

        ## draw delimitating black bars, to make display square
        self.session.rect_left.draw()
        self.session.rect_right.draw()

        ## fixation lines
        self.session.line1.draw() 
        self.session.line2.draw() 
            
        ## fixation dot
        if self.session.fix_counter<len(self.session.fixation_switch_times): # if counter within number of switch moments
            if current_time<self.session.fixation_switch_times[self.session.fix_counter]: # if current time under switch time
                self.session.fixation.draw() # just draw

            else: # when switch time reached, switch color and increment counter
                self.session.fixation.fillColor *= -1
                self.session.fixation.lineColor *= -1
                self.session.fixation.draw()
                self.session.fix_counter += 1


    def get_events(self):
        """ Logs responses/triggers """
        for ev, t in event.getKeys(timeStamped=self.session.clock): # list of of (keyname, time) relative to Clock’s last reset
            if len(ev) > 0:
                if ev in ['q']:
                    print('trial canceled by user')  
                    self.session.close()
                    self.session.quit()

                elif ev == self.session.mri_trigger: # TR pulse
                    event_type = 'pulse'
                    self.exit_phase=True # what for?

                else: # any other key pressed will be response to color change
                    event_type = 'response'
                    self.session.total_responses += 1

                     #track percentage of correct responses per session (only correct if reply within 0.8s of color switch)
                    if t < self.session.fixation_switch_times[-1]: # avoid crash when running, need to optimize this later
                        if t < (self.session.fixation_switch_times[self.session.fix_counter]+0.8):
                            self.session.correct_responses += 1

                # log everything into session data frame
                idx = self.session.global_log.shape[0]
                self.session.global_log.loc[idx, 'trial_nr'] = self.ID
                self.session.global_log.loc[idx, 'onset'] = t
                self.session.global_log.loc[idx, 'event_type'] = event_type
                self.session.global_log.loc[idx, 'phase'] = self.phase
                self.session.global_log.loc[idx, 'response'] = ev                

                for param, val in self.parameters.items():
                    self.session.global_log.loc[idx, param] = val



class FeatureTrial(Trial):

    def __init__(self, session, trial_nr, phase_durations, 
        attend_block_conditions, bar_direction_at_TR, bar_midpoint_at_TR, trial_type_at_TR, timing='seconds', *args, **kwargs):


        """ Initializes a FeatureTrial object. 

        Parameters
        ----------
        session : exptools Session object
            A Session object (needed for metadata)
        trial_nr: int
            Trial nr of trial
        timing : str
            The "units" of the phase durations. Default is 'seconds', where we
            assume the phase-durations are in seconds. The other option is
            'frames', where the phase-"duration" refers to the number of frames.
        attend_block_conditions: arr
        	list/array with name of attended condition on each miniblock. Total length = total # miniblocks
        bar_direction_at_TR : list
            List/array with the bar direction at each TR. In same cases it can have 
            a list of direction (when several bars on screen). Total length = total # TRs
        bar_midpoint_at_TR : arr
            List/array with the pairs of positions [x,y] of the midpoint of the bar
            per TR. In same cases it can have a list of pairs (when several bars on screen). Total length = total # TRs
        trial_type_at_TR: arr
            List/array with trial identifier ("trial type"). To know if cue, empty or miniblock

            
        """
        
        self.ID = trial_nr # trial identifier, not sure if needed
        self.bar_direction_at_TR = bar_direction_at_TR
        self.bar_midpoint_at_TR = bar_midpoint_at_TR
        self.trial_type_at_TR = trial_type_at_TR
        self.attend_block_conditions = attend_block_conditions
        self.session = session

        # phase durations for each condition 
        self.phase_durations = phase_durations


        super().__init__(session, trial_nr, phase_durations, verbose=False, *args, **kwargs)

        # get bar and background positions for this trial
        self.position_dictionary = get_object_positions(self.session.grid_pos, self.bar_midpoint_at_TR, self.bar_direction_at_TR,
                                                    self.session.bar_width_pix, screen = self.session.screen, 
                                                    num_bar = len(self.session.attend_block_conditions))


    def draw(self): 

        """ Draw stimuli - pRF bars and fixation dot - for each trial """
        
        current_time = self.session.clock.getTime() # get time

        ## orientation switch times
        if self.session.ori_counter<len(self.session.ori_switch_times): # if counter within number of switch moments
            if current_time >= self.session.ori_switch_times[self.session.ori_counter]: # when switch time reached, switch ori and increment counter
                
                self.session.ori_ind = 0 if (self.session.ori_counter % 2) == 0 else 1
                self.session.ori_counter += 1

        ## draw stim
        if 'cue' in self.trial_type_at_TR: # if cue at TR, draw background and word cue

            # define appropriate cue string for the upcoming mini block
            attend_cond = self.attend_block_conditions[int(self.trial_type_at_TR[-1])]

            # define cue direction
            if 'vertical' in attend_cond:
                cue_width = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['feature']['cue_width'], self.session.monitor)
                cue_height = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['feature']['cue_height'], self.session.monitor)

            else:
                cue_width = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['feature']['cue_height'], self.session.monitor)
                cue_height = tools.monitorunittools.deg2pix(self.session.settings['stimuli']['feature']['cue_width'], self.session.monitor)

            # get cue condition name, to use in local elements
            cue_condition = 'color_red' if 'red' in attend_cond else 'color_green'


            self.session.feature_stim.draw(bar_midpoint_at_TR = np.nan, 
                                       bar_direction_at_TR = np.nan,
                                       this_phase = 'background',
                                       position_dictionary = self.position_dictionary,
                                       orientation_ind = self.session.ori_ind)

            self.cue_stim = visual.Rect(win = self.session.win,
                                        units = "pix",
                                        width = cue_width,
                                        height = cue_height,
                                        fillColor = self.session.settings['stimuli']['conditions'][cue_condition]['element_color'],
                                        lineColor = self.session.settings['stimuli']['conditions'][cue_condition]['element_color'],
                                        contrast = self.session.settings['stimuli']['feature']['cue_contrast'],
                                        pos = [0, 0],
                                        fillColorSpace = 'hsv',
                                        lineColorSpace = 'hsv'
                                        )
            self.cue_stim.draw()

            print('cue '+attend_cond)
                    
        elif self.bar_direction_at_TR == 'empty': # if empty trial, show background

            self.session.feature_stim.draw(bar_midpoint_at_TR = np.nan, 
                                       bar_direction_at_TR = np.nan,
                                       this_phase = 'background',
                                       position_dictionary = self.position_dictionary,
                                       orientation_ind = self.session.ori_ind) 
            print('background')

        # bar pass

        else: # if bar pass at TR, then draw bar

            # get list of condition names (as defined in yml) for this phase
            this_phase = list(self.session.all_bar_pos[self.trial_type_at_TR].keys())
            this_phase = ['color_red' if 'red' in p else 'color_green' for _,p in enumerate(this_phase)]

            self.session.feature_stim.draw(bar_midpoint_at_TR = self.bar_midpoint_at_TR, 
                                           bar_direction_at_TR = self.bar_direction_at_TR,
                                           this_phase = this_phase,
                                           position_dictionary = self.position_dictionary,
                                           orientation_ind = [self.session.ori_ind] + self.session.local_ori[self.ID],
                                           drawing_ind = self.session.drawing_ind[self.ID]) 
            
            print('bar stim') 

        # draw delimitating black bars, to make display square
        self.session.rect_left.draw()
        self.session.rect_right.draw()

        # fixation lines
        self.session.line1.draw() 
        self.session.line2.draw() 
            
        # fixation dot
        self.session.fixation.draw() # just draw



    def get_events(self):
        """ Logs responses/triggers """
        for ev, t in event.getKeys(timeStamped=self.session.clock): # list of of (keyname, time) relative to Clock’s last reset
            if len(ev) > 0:
                if ev in ['q']:
                    print('trial canceled by user')  
                    self.session.close()
                    self.session.quit()

                elif ev == self.session.mri_trigger: # TR pulse
                    event_type = 'pulse'
                    self.exit_phase=True # what for?

                else: # any other key pressed will be response to color change
                    event_type = 'response'
                    self.session.total_responses += 1


                # log everything into session data frame
                idx = self.session.global_log.shape[0]
                self.session.global_log.loc[idx, 'trial_nr'] = self.ID
                self.session.global_log.loc[idx, 'onset'] = t
                self.session.global_log.loc[idx, 'event_type'] = event_type
                self.session.global_log.loc[idx, 'phase'] = self.phase
                self.session.global_log.loc[idx, 'response'] = ev                

                for param, val in self.parameters.items():
                    self.session.global_log.loc[idx, param] = val







