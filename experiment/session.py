
import os
import numpy as np
from exptools2.core import Session

from trial import PRFTrial
from psychopy.visual import Circle
from psychopy import core



class PRFSession(Session):
    def __init__(self, output_str, output_dir,scanner=False,tracker_on=False,settings_file='experiment_settings.yml'):


        # need to initialize parent class, indicating output infos
        super().__init__(output_str=output_str,output_dir=output_dir,settings_file=settings_file)
        
        # define type and order of trials
        self.trial_type = np.concatenate((np.repeat('blank',self.settings['stimuli']['blank_TR']),
                                          np.repeat('LR',self.settings['stimuli']['bar_pass_hor_TR']),
                                          np.repeat('TB',self.settings['stimuli']['bar_pass_ver_TR']),
                                          np.repeat('blank',self.settings['stimuli']['blank_TR']),
                                          np.repeat('BT',self.settings['stimuli']['bar_pass_ver_TR']),
                                          np.repeat('LR',self.settings['stimuli']['bar_pass_hor_TR']),
                                          np.repeat('blank',self.settings['stimuli']['blank_TR'])
                                          ))

        self.n_trials = len(self.trial_type)  # number of trials per session
        self.trials = []  # will be filled with Trials later

        # create trials before running!
        self.create_trials() 


        
    def create_trials(self):
        """ Creates trials (before running the session) """

        TR = self.settings['mri']['TR'] 
        
        for i in range(self.n_trials): # for all trials
            # set trial
            trial = PRFTrial(session=self, 
                             trial_nr=i,
                             phase_durations=[TR], # duration of each phase of the trial (iti,stim)
                             timing='seconds', # in seconds
                             trial_type = self.trial_type[i],
                             phase_names=[self.trial_type]) # names for each phase (stored in log)
            # append in list
            self.trials.append(trial)

    
    def run(self):
        """ Loops over trials and runs them """
        
        self.start_experiment()

        # draw instructions
        this_instruction_string = 'Please fixate at the center, \ndo not move your eyes'
        self.display_text(this_instruction_string, keys=['q'],
                                    color=(1, 1, 1), font = 'Helvetica Neue', pos = (0, 0), 
                                    italic = True, alignHoriz = 'center')

        #self.fixation_dot.draw()

        # ### SHOULD HAVE A WAIT FOR TRIGGER HERE #####

        for trl in self.trials: # run all
            trl.run()
            
        self.close() # close session
        
