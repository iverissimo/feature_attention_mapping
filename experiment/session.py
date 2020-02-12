
import os
import numpy as np

from exptools2.core import Session

from trial import PRFTrial
from stim import PRFStim

from psychopy import visual, tools


class PRFSession(Session):
    # initialize child class
    def __init__(self, output_str, output_dir, settings_file):

        # need to initialize parent class, indicating output infos
        super().__init__(output_str=output_str,output_dir=output_dir,settings_file=settings_file)

        # some MRI params
        self.bar_step_length = self.settings['mri']['TR'] # in seconds
        self.mri_trigger='t'

        # create trials before running!
        self.create_stimuli()
        self.create_trials() 


    
    # create stimuli - pRF bar and fixation dot
    def create_stimuli(self):
        
        #generate PRF stimulus
        self.prf_stim = PRFStim(session=self, 
                        bar_width_deg=self.settings['stimuli']['bar_width_deg']
                        )
        
        # fixation dot radius in pixels
        fixation_rad_pix = tools.monitorunittools.deg2pix(self.settings['stimuli']['fixation_size_deg'], self.monitor)/2 #Convert size in degrees to size in pixels for a given Monitor object
        
        # fixation dot changes color during task
        self.fixation = visual.Circle(self.win, units='pix', radius=fixation_rad_pix, 
                                            fillColor=[-1,-1,-1], lineColor=[-1,-1,-1])


    def create_trials(self):
        """ Creates trials (before running the session) """

        bar_pass_hor_TR = self.settings['stimuli']['bar_pass_hor_TR']
        bar_pass_ver_TR = self.settings['stimuli']['bar_pass_ver_TR']
        bar_pass_diag_TR = self.settings['stimuli']['bar_pass_diag_TR']
        empty_TR = self.settings['stimuli']['empty_TR']

        bar_orientation = self.settings['stimuli']['bar_orientation'] # order of bar orientations throught experiment

        # all positions in pixels [x,y] for
        # horizontal bar passes, 
        hor_x = window_size[0]*np.linspace(-0.5,0.5, bar_pass_hor_TR)
        hor_bar_pos_pix = np.array([np.array([x,0]) for _,x in enumerate(hor_x)])

        # vertical bar passes, 
        ver_y = window_size[1]*np.linspace(-0.5,0.5, bar_pass_ver_TR)
        ver_bar_pos_pix = np.array([np.array([0,y]) for _,y in enumerate(ver_y)])

        # and diagonal bar passes
        diag_x = window_size[0]*np.linspace(-0.5,0.5, bar_pass_diag_TR)
        diag_y = window_size[1]*np.linspace(-0.5,0.5, bar_pass_diag_TR)

        diag_pos_pix = np.array([np.array([diag_x[i],diag_y[i]]) for i in range(len(diag_x))  ])


        #create as many trials as TRs
        trial_number = 0
        bar_orientation_at_TR = [] # list of bar orientation at all TRs

        bar_pos_array = [] # list with bar midpoint (x,y) for all TRs (if nan, then empty screen)

        for _,bartype in enumerate(bar_orientation):
            if bartype in np.array(['empty']): # empty screen
                trial_number += empty_TR
                bar_orientation_at_TR = bar_orientation_at_TR + np.repeat(bartype,empty_TR).tolist()
                bar_pos_array.append([np.array([np.nan,np.nan]) for i in range(empty_TR)])
                
            elif bartype in np.array(['U-D','D-U']): # vertical bar pass
                trial_number += bar_pass_ver_TR
                bar_orientation_at_TR =  bar_orientation_at_TR + np.repeat(bartype,bar_pass_ver_TR).tolist()
                bar_pos_array.append(ver_bar_pos_pix)
                
            elif bartype in np.array(['L-R','R-L']): # horizontal bar pass
                trial_number += bar_pass_hor_TR
                bar_orientation_at_TR =  bar_orientation_at_TR + np.repeat(bartype,bar_pass_hor_TR).tolist()
                bar_pos_array.append(hor_bar_pos_pix)
                
            elif bartype in np.array(['UR-DL','DL-UR','UL-DR','DR-UL']): # diagonal bar pass
                trial_number += bar_pass_diag_TR
                bar_orientation_at_TR =  bar_orientation_at_TR + np.repeat(bartype,bar_pass_diag_TR).tolist()
                bar_pos_array.append(diag_pos_pix)

        self.trial_number = trial_number # total number of trials 
        print("Total number of (expected) TRs: %d"%self.trial_number)
        self.bar_orientation_at_TR = bar_orientation_at_TR # list of strings with bar orientation/empty

        # positions (x,y) of bar for all TRs
        self.bar_pos_midpoint = np.array([val for sublist in bar_pos_array for val in sublist])


        # append all trials
        self.all_trials = []
        for i in range(self.trial_number):

            self.all_trials.append(PRFTrial(session=self,
                                            trial_nr=i,  
                                            bar_orientation_at_TR=self.bar_orientation_at_TR[i],
                                            bar_pos_midpoint=self.bar_pos_midpoint[i]
                                            ))

# continue from here
# just need to add moment of dot switch (figure out smart way of randomizing that) in create trials
# then do the draw and run functions here, then can pass to trial and stim . py 


#######################################

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
        self.fixation = visual.Circle(self.win, units='pix', radius=fixation_rad_pix, 
                                            fillColor=[-1,-1,-1], lineColor=[-1,-1,-1])


        # fixation task timing
        self.fix_task_frame_values = self._get_frame_values(framerate=self.settings['window']['framerate'], 
                                                            trial_duration=self.total_duration, 
                                                            safety_margin=3000.0)

        # to append fixation transitions
        self.transition_list = []
        self.frame_nr = 0
        
        # append all trials
        self.all_trials = []
        for i,trial in enumerate(self.trial_array):

            this_trial_parameters = {}
            this_trial_parameters['stim_duration'] = self.phase_durations[i, -2] # total phase duration for bar pass of trial
            this_trial_parameters['orientation'] = trial[0] # orientation of bar pass
            this_trial_parameters['stim_bool'] = trial[1] # if bar pass in trial or not

            self.all_trials.append(PRFTrial(session=self,
                                            trial_nr=i,  
                                            parameters = this_trial_parameters, # Dict of parameters that needs to be added to the log of this trial
                                            phase_durations = self.phase_durations[i] # array with all phase timings for trial
                                            #,tracker=self.tracker_on
                                            ))
    
    def run(self):
        """ Loops over trials and runs them """
        
        # draw instructions
        this_instruction_string = 'Please fixate at the center, \ndo not move your eyes'
        self.display_text(this_instruction_string, duration=3,
                                    color=(1, 1, 1), font = 'Helvetica Neue', pos = (0, 0), 
                                    italic = True, alignHoriz = 'center')

        # draw instructions
        this_instruction_string = 'Index finger - Black // Middle finger - White\nwaiting for scanner'
        self.display_text(this_instruction_string, keys=self.settings['mri'].get('sync', 't'),
                                color=(1, 1, 1), font = 'Helvetica Neue', pos = (0, 0), 
                                italic = True, alignHoriz = 'center')

        self.start_experiment()
        
        # cycle through trials
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




