
import os
import numpy as np
from exptools2.core import Trial

from psychopy import event 


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
       

    def draw(self): 

        """ Draw stimuli - pRF bar and fixation dot - for each trial """
        
        current_time = self.session.clock.getTime() # get time

        # bar pass
        if self.bar_direction_at_TR != 'empty': # if bar pass at TR, then draw bar

            self.session.prf_stim.draw(bar_midpoint_at_TR=self.bar_midpoint_at_TR, 
                                       bar_direction_at_TR=self.bar_direction_at_TR,
                                       this_phase=self.phase_names[int(self.phase)]) #'ori_left')

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

    def __init__(self, session, trial_nr, bar_direction_at_TR, bar_midpoint_at_TR, timing='seconds', *args, **kwargs):

        """ Initializes a FeatureTrial object. 

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


        super().__init__(session, trial_nr, phase_durations, phase_names, verbose=False, *args, **kwargs)
       

    def draw(self): 

        """ Draw stimuli - pRF bar and fixation dot - for each trial """
        
        current_time = self.session.clock.getTime() # get time

        # bar pass
        if self.bar_direction_at_TR != 'empty': # if bar pass at TR, then draw bar

            self.session.prf_stim.draw(bar_midpoint_at_TR=self.bar_midpoint_at_TR, 
                                       bar_direction_at_TR=self.bar_direction_at_TR,
                                       this_phase=self.phase_names[int(self.phase)]) #'ori_left')

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







