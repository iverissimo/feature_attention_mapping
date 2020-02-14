
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
        self.bar_step = self.settings['mri']['TR'] # in seconds
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

        # counter for responses
        self.total_responses = 0
        self.correct_responses = 0

        bar_pass_hor_TR = self.settings['stimuli']['bar_pass_hor_TR']
        bar_pass_ver_TR = self.settings['stimuli']['bar_pass_ver_TR']
        bar_pass_diag_TR = self.settings['stimuli']['bar_pass_diag_TR']
        empty_TR = self.settings['stimuli']['empty_TR']

        bar_orientation = self.settings['stimuli']['bar_orientation'] # order of bar orientations throught experiment

        # all positions in pixels [x,y] for
        # horizontal bar passes, 
        hor_x = self.win.size[0]*np.linspace(-0.5,0.5, bar_pass_hor_TR)
        hor_bar_pos_pix = np.array([np.array([x,0]) for _,x in enumerate(hor_x)])

        # vertical bar passes, 
        ver_y = self.win.size[1]*np.linspace(-0.5,0.5, bar_pass_ver_TR)
        ver_bar_pos_pix = np.array([np.array([0,y]) for _,y in enumerate(ver_y)])

        # and diagonal bar passes
        diag_x = self.win.size[0]*np.linspace(-0.5,0.5, bar_pass_diag_TR)
        diag_y = self.win.size[1]*np.linspace(-0.5,0.5, bar_pass_diag_TR)

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

        # define moments for fixation dot to change color
        self.total_time = self.trial_number*self.bar_step # in seconds
        self.fixation_switch_times = np.arange(1,self.total_time,4)
        self.fixation_switch_times += 2*np.random.random_sample((len(self.fixation_switch_times),)) # use these time points to switch colors

        # counter for fixation dot switches
        self.fix_counter = 0

        # print window size just to check, not actually needed
        print(self.win.size)


    def draw_stimulus(self):
 
        # draw the actual stimuli

        # get time
        current_time = self.clock.getTime()

        # bar pass
        if self.this_trial.bar_orientation_at_TR != 'empty': # if orientation not empty, draw bar
            
            self.prf_stim.draw(bar_pos_midpoint=self.this_trial.bar_pos_midpoint, 
                               orientation=self.this_trial.bar_orientation_at_TR)
            
        # fixation dot
        if self.fix_counter<len(self.fixation_switch_times):
            if current_time<self.fixation_switch_times[self.fix_counter]:
                self.fixation.draw()

            else: # when switch time reached, switch color and increment counter
                self.fixation.fillColor *= -1
                self.fixation.lineColor *= -1
                self.fixation.draw()
                self.fix_counter += 1

    
    def run(self):
        """ Loops over trials and runs them """
        
        # draw instructions wait a few seconds
        this_instruction_string = 'Please fixate at the center, \ndo not move your eyes'
        self.display_text(this_instruction_string, duration=3,
                                    color=(1, 1, 1), font = 'Helvetica Neue', pos = (0, 0), 
                                    italic = True, alignHoriz = 'center')

        # draw instructions wait for scanner t trigger
        this_instruction_string = 'Index finger - Black\n Middle finger - White\nwaiting for scanner'
        self.display_text(this_instruction_string, keys=self.settings['mri'].get('sync', 't'),
                                color=(1, 1, 1), font = 'Helvetica Neue', pos = (0, 0), 
                                italic = True, alignHoriz = 'center')

        self.start_experiment()
        
        # cycle through trials
        for trl in self.all_trials: 
            self.this_trial = trl
            self.this_trial.run() # run forrest run


        print('Expected number of responses: %d'%len(self.fixation_switch_times))
        print('Total subject responses: %d'%self.total_responses)
        print('Correct responses (within 0.8s of dot color change): %d'%self.correct_responses)
          

        self.close() # close session
        


# need to make a feature session
# and add that as an argument in main, and according to user input will call and run specific session




