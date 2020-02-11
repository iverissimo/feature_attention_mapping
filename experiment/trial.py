import os
import numpy as np
from exptools2.core import Trial
from psychopy.visual import TextStim, Circle
from psychopy import core


from stim import PRFStim

class PRFTrial(Trial):
        
        def __init__(self, session, trial_nr, parameters = {}, phase_durations = [] ,timing='seconds', phase_names=None):

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
            parameters : dict
                Dict of parameters that needs to be added to the log of this trial
            timing : str
                The "units" of the phase durations. Default is 'seconds', where we
                assume the phase-durations are in seconds. The other option is
                'frames', where the phase-"duration" refers to the number of frames.
            load_next_during_phase : int (or None)
                If not None, the next trial will be loaded during this phase
            verbose : bool
                Whether to print extra output (mostly timing info)
            """
            
            super().__init__(session = session, trial_nr=trial_nr, parameters = parameters, phase_durations = phase_durations)

            #self.stim = PRFStim(self, self.session, orientation = self.parameters['orientation'])
        
            this_instruction_string = '\t\t\t  Index\t\t/\tMiddle:\n\nColor\t\t-\tB\t\t/\t\tW'
            self.instruction = TextStim(self.session.win, text = this_instruction_string, 
                                                font = 'Helvetica Neue', pos = (0, 0), 
                                                italic = True, height = 30, alignHoriz = 'center')
            self.instruction.setSize((1200,50))

            self.run_time = 0.0
            self.instruct_time = self.t_time = self.fix_time = self.stimulus_time = self.post_stimulus_time = 0.0
            self.instruct_sound_played = False
            self.ID = trial_nr

                
            def draw(self):
                """docstring for draw"""
                old_color = self.session.fixation.color[0]

                self.session.fixation.color = [self.session.fix_task_frame_values[self.session.frame_nr], self.session.fix_task_frame_values[self.session.frame_nr], self.session.fix_task_frame_values[self.session.frame_nr]]
                if (old_color != self.session.fixation.color[0]) and hasattr(self.session, 'scanner_start_time'):
                    self.session.transition_list.append([self.session.clock.getTime() - self.session.scanner_start_time, self.session.fixation.color[0]])

                if self.phase == 0:
                    if self.ID == 0:
                        self.instruction.draw()
                #elif self.phase == 2:
                    #self.stim.draw(phase = np.max([(self.phase_times[self.phase] - self.phase_times[self.phase-1]) / self.stim.period,0]))

                self.session.fixation_0.draw()
                self.session.fixation_1.draw()




            def event(self):
                for ev in event.getKeys():
                    if len(ev) > 0:
                        if ev in ['esc', 'escape']:
                            self.events.append([-99,self.session.clock.getTime()-self.start_time])
                            self.stopped = True
                            self.session.stopped = True
                            print('run canceled by user')
                        # it handles both numeric and lettering modes 
                        elif ev == ' ':
                            self.events.append([0,self.session.clock.getTime()-self.start_time])
                            if self.phase == 0:
                                self.phase_forward()
                            else:
                                self.events.append([-99,self.session.clock.getTime()-self.start_time])
                                self.stopped = True
                                print('trial canceled by user')
                        elif ev == 't': # TR pulse
                            self.events.append([99,self.session.clock.getTime()-self.start_time])
                            if (self.phase == 0) and (self.ID == 0): 
                                # first trial, first phase receives the first 't' of the experiment
                                self.session.scanner_start_time = self.session.clock.getTime()
                            if (self.phase == 0) + (self.phase==1):
                                self.phase_forward()

                        event_msg = 'trial ' + str(self.ID) + ' key: ' + str(ev) + ' at time: ' + str(self.session.clock.getTime())
                        self.events.append(event_msg)
                        print(event_msg + ' ' + str(self.phase))
                
                    super(PRFTrial, self).key_event( ev )

            
       





