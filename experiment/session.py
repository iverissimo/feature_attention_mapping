
import os
import numpy as np
from exptools2.core import Session

from trial import PRFTrial

from psychopy.visual import filters
from psychopy import core, visual, tools


class PRFSession(Session):
    def __init__(self, output_str, output_dir,scanner=False,tracker_on=False,settings_file='experiment_settings.yml'):


        # need to initialize parent class, indicating output infos
        super().__init__(output_str=output_str,output_dir=output_dir,settings_file=settings_file)

        # some MRI params
        self.bar_step_length = self.settings['mri']['TR'] # in seconds
        self.mri_trigger='t'

        # create trials before running!
        self.create_trials() 

        
    def create_trials(self):
        """ Creates trials (before running the session) """

        self.TR = self.settings['mri']['TR'] 
        self.bar_present_booleans = np.array(self.settings['stimuli']['bar_present_booleans'])
        self.bar_orientation = np.array(self.settings['stimuli']['bar_orientation'])

        # save duration in seconds for all bar passes + blank screen in list
        self.bar_pass_durations = [] 

        for i,val in enumerate(self.bar_present_booleans):
            
            if val == 0: # no bar pass
                self.bar_pass_durations.append(self.settings['stimuli']['empty_TR'] * self.TR) # in seconds
            else:
                if self.bar_orientation[i] in ['LR','RL']: # horizontal bar passes
                    self.bar_pass_durations.append(self.settings['stimuli']['bar_pass_hor_TR'] * self.TR) # in seconds

                elif self.bar_orientation[i] in ['DU','UD']: # vertical bar passes
                    self.bar_pass_durations.append(self.settings['stimuli']['bar_pass_ver_TR'] * self.TR) # in seconds

        # set list of trial arrays (orientation, bool)
        self.trial_array = np.array([[self.bar_orientation[i], val] for i,val in enumerate(self.bar_present_booleans)])

        # set phase durations for all trials (instruct time,wait for scan pulse,bar pass duration,iti)
        self.phase_durations = np.array([[-0.001, 
                                            180.0,  
                                            self.bar_pass_durations[i],
                                            self.settings['stimuli']['ITI_TR'] * self.TR] for i in range(len(self.bar_present_booleans))])  

        self.total_duration = np.sum(np.array(self.phase_durations)) # max experiment duration
        self.phase_durations[0,0] = 1800 # we guarantee that beginning of experiment has enough time to wait for start

        
        # fixation dot radius in pixels
        fixation_rad_pix = tools.monitorunittools.deg2pix(self.settings['stimuli']['fixation_size'], self.monitor)/2 #Convert size in degrees to size in pixels for a given Monitor object
        
        # fixation dot changes color during task
        self.fixation_0 = visual.Circle(self.win, 
            units='pix', radius=fixation_rad_pix, 
            fillColor=[1,-1,-1], lineColor=[1,-1,-1])
        
        self.fixation_1 = visual.Circle(self.win, 
            units='pix', radius=fixation_rad_pix, 
            fillColor=[-1,1,-1], lineColor=[-1,1,-1])


        # fixation task timing
        self.fix_task_frame_values = self._get_frame_values(framerate=self.settings['window']['framerate'], 
                                                            trial_duration=self.total_duration, 
                                                            safety_margin=3000.0)
        
        # append all trials
        self.all_trials = []
        for i,trial in enumerate(self.trial_array):


            self.all_trials.append(PRFTrial(session=self,
                                            trial_nr=i,  
                                            stim_duration = self.phase_durations[i, -2], # total phase duration for bar pass of trial
                                            orientation = trial[0], # orientation of bar pass
                                            stim_bool = trial[1], # if bar pass in trial or not
                                            phase_durations = self.phase_durations[i] # array with all phase timings for trial
                                            #,tracker=self.tracker_on
                                            ))
    
    def run(self):
        """ Loops over trials and runs them """
        
        # draw instructions
        this_instruction_string = 'Please fixate at the center, \ndo not move your eyes \npress q to start'
        self.display_text(this_instruction_string, keys=['q'],
                                    color=(1, 1, 1), font = 'Helvetica Neue', pos = (0, 0), 
                                    italic = True, alignHoriz = 'center')

        # cycle through trials
        self.display_text('Waiting for scanner', keys=self.settings['mri'].get('sync', 't'))

        self.start_experiment()
            
        for trl in self.all_trials: # run all
            trl.run()
            
        self.close() # close session
        



    def _get_frame_values(self,
                        framerate=60,
                        trial_duration=3000,
                        min_value=1,
                        exp_scale=1,
                        values=[-1, 1],
                        safety_margin=None):

      if safety_margin is None:
          safety_margin = 5

      n_values = len(values)

      total_duration = trial_duration + safety_margin
      total_n_frames = total_duration * framerate

      result = np.zeros(int(total_n_frames))

      n_samples = np.ceil(total_duration * 2 / (exp_scale + min_value)).astype(int) # up-rounded number of samples
      durations = np.random.exponential(exp_scale, n_samples) + min_value #  random duration for that value (exponential distribution)

      frame_times = np.linspace(0, total_duration, total_n_frames, endpoint=False)

      first_index = np.random.randint(n_values) # randomly pick index to start with (0 = -1, 1 = 1)

      result[frame_times < durations[0]] = values[first_index] 

      for ix, c in enumerate(np.cumsum(durations)): # cumulative sum of durations
          result[frame_times > c] = values[(first_index + ix) % n_values]

      return result



# need to make a feature session
# and add that as an argument in main, and according to user input will call and run specific session

