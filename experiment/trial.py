import os
import numpy as np
from exptools2.core import Trial
from psychopy.visual import TextStim, Circle
from psychopy import core


from stim import PRFStim

class PRFTrial(Trial):
        
        def __init__(self, session, trial_nr, phase_durations, timing, trial_type, phase_names):
            """ Initializes a StroopTrial object. 

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
            
            super().__init__(session, trial_nr, phase_durations, timing, phase_names)

            # get trial type, to know what to actually present
            self.trial_type = trial_type
            self.prev_trial_type = []

            # keep track of when trial types switch
            if trial_nr == 0:
                self.prev_trial_type = trial_type
                self.switch = False
            else:
                if trial_type != self.prev_trial_type: # if current trial different from previous trial
                    self.switch = True
                    self.prev_trial_type = trial_type # update previous trial type
                else:
                    self.switch = False

            iti_s = self.session.settings['stimuli']['ITI_TR'] * self.session.settings['mri']['TR'] # iti in seconds

            # define fixation dot
            fix_rad = self.session.settings['stimuli']['fixation_size'] # get value from yaml
            self.fixation_dot = Circle(self.session.win, radius=fix_rad, edges=100,fillColor=(1, 1, 1))


            self.prf_stim = PRFStim(session=self.session,
                                    trial_type=self.trial_type,
                                    prev_trial_type=self.prev_trial_type,
                                    switch=self.switch) 

            
        # draw stimuli depending on phase of trial
        def draw(self):
            
            #if self.switch == True: # switching task, introduce an ITI

            #    self.fixation_dot.draw()
            #    self.session.win.flip()
            #    core.wait(iti_s)
             
            if self.phase==0:   
                # draw the stimulus 
                self.prf_stim.draw()





