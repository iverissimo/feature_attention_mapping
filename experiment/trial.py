
import os
import numpy as np
from exptools2.core import Trial

from psychopy import event 
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

        if self.bar_direction_at_TR == 'empty': # if empty trial, show background

            self.session.prf_stim.draw(bar_midpoint_at_TR = np.nan, 
                                       bar_direction_at_TR = np.nan,
                                       this_phase = 'background',
                                       position_dictionary = self.position_dictionary) 
            print('background')

        else: # if bar pass at TR, then draw bar

            self.session.prf_stim.draw(bar_midpoint_at_TR = self.bar_midpoint_at_TR, 
                                       bar_direction_at_TR = self.bar_direction_at_TR,
                                       this_phase = self.phase_names[int(self.phase)],
                                       position_dictionary = self.position_dictionary) 

            print(self.phase_names[int(self.phase)]) #'ori_left')

        # draw delimitating black bars, to make display square
        self.session.rect_left.draw()
        self.session.rect_right.draw()

        # fixation lines
        self.session.line1.draw() 
        self.session.line2.draw() 
            
        # fixation dot
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
                    if t<self.session.fixation_switch_times[-1]: # avoid crash when running, need to optimize this later
                        if t>self.session.fixation_switch_times[self.session.fix_counter] and t<self.session.fixation_switch_times[self.session.fix_counter]+0.8:
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

    def __init__(self, session, trial_nr, phase_durations, phase_names, 
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
        # name of each condition
        self.phase_names = phase_names 


        super().__init__(session, trial_nr, phase_durations, phase_names, verbose=False, *args, **kwargs)

        # get bar and background positions for this trial
        self.position_dictionary = get_object_positions(self.session.grid_pos, self.bar_midpoint_at_TR, self.bar_direction_at_TR,
                                                    self.session.bar_width_pix, screen = self.session.screen, 
                                                    num_bar = len(self.session.attend_block_conditions))


    def draw(self): 

        """ Draw stimuli - pRF bars and fixation dot - for each trial """
        

        if 'cue' in self.trial_type_at_TR: # if cue at TR, draw background and word cue

            self.session.feature_stim.draw(bar_midpoint_at_TR = np.nan, 
                                       bar_direction_at_TR = np.nan,
                                       this_phase = 'background',
                                       position_dictionary = self.position_dictionary)

            # define appropriate cue string for the upcoming mini block
            attend_cond = self.attend_block_conditions[int(self.trial_type_at_TR[-1])]

            if attend_cond == 'ori_left':
                cue_str = 'left'
            elif attend_cond == 'ori_right':
                cue_str = 'right'
            elif attend_cond == 'color_red':
                cue_str = 'red'
            elif attend_cond == 'color_green':  
                cue_str = 'green'

            cue_stim = TextStim(self.session.win, text=cue_str,
                                color=(1, 1, 1), font = 'Helvetica Neue', pos = (0, 0), 
                                italic = False, alignHoriz = 'center', alignVert = 'center')
            cue_stim.draw()
            
            print(cue_str)
        
        elif self.bar_direction_at_TR == 'empty': # if empty trial, show background

            self.session.feature_stim.draw(bar_midpoint_at_TR = np.nan, 
                                       bar_direction_at_TR = np.nan,
                                       this_phase = 'background',
                                       position_dictionary = self.position_dictionary) 
            print('background')

        # bar pass

        else: # if bar pass at TR, then draw bar

            if self.phase == 0: # if bar phase, draw stim

                self.session.feature_stim.draw(bar_midpoint_at_TR = self.bar_midpoint_at_TR, 
                                               bar_direction_at_TR = self.bar_direction_at_TR,
                                               this_phase = list(self.session.all_bar_pos[self.trial_type_at_TR].keys()),
                                               position_dictionary = self.position_dictionary) 
            else:
                
                all_positions_dict = {'background': {'xys': self.session.grid_pos}} # draw background in all positions

                self.session.feature_stim.draw(bar_midpoint_at_TR = np.nan, 
                                               bar_direction_at_TR = np.nan,
                                               this_phase = 'background',
                                               position_dictionary = all_positions_dict)

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







