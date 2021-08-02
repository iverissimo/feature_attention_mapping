
import os
import numpy as np
import yaml, re

from exptools2.core import Trial

from psychopy import event, tools, colors, visual
from psychopy.visual import TextStim

from utils import *

import pickle


class PRFTrial(Trial):

    def __init__(self, session, trial_nr, bar_pass_direction_at_TR, bar_midpoint_at_TR, phase_durations,
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
        bar_pass_direction_at_TR : list
            List/array with the bar direction at each TR. Total length = total # TRs
        bar_midpoint_at_TR : array
            Numpy array with the pairs of positions [x,y] of the midpoint of the bar
            per TR. Shape (#TRs, 2)
            
        """
        
        self.ID = trial_nr # trial identifier, not sure if needed
        self.bar_pass_direction_at_TR = bar_pass_direction_at_TR
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

        self.position_dictionary = get_object_positions(self.session.grid_pos, self.bar_midpoint_at_TR, self.bar_pass_direction_at_TR,
                                                    self.session.bar_width_pix, screen = self.session.screen, num_bar = 1)
       

    def draw(self): 

        """ Draw stimuli - pRF bar - for each trial """

        current_time = self.session.clock.getTime() # get time


        ## orientation switch times
        if self.session.ori_counter<len(self.session.ori_switch_times): # if counter within number of switch moments
            if current_time >= self.session.ori_switch_times[self.session.ori_counter]: # when switch time reached, switch ori and increment counter
                
                self.session.ori_bool = True
                self.session.ori_counter += 1

        ## draw stim
        if (self.bar_pass_direction_at_TR == 'empty'): # if empty trial, show background

            print('background')

        else: # if bar pass at TR, then draw bar


            self.session.prf_stim.draw(bar_midpoint_at_TR = self.bar_midpoint_at_TR, 
                                       bar_pass_direction_at_TR = self.bar_pass_direction_at_TR,
                                       this_phase = self.phase_names[int(self.phase)],
                                       position_dictionary = self.position_dictionary,
                                       orientation = self.session.ori_bool) 

            print(self.phase_names[int(self.phase)]) #'ori_left')

        # set orientation bool counter to false
        self.session.ori_bool = False

        ## draw delimitating black bars, to make display square
        self.session.rect_left.draw()
        self.session.rect_right.draw()

        ## fixation lines
        self.session.line1.draw() 
        self.session.line2.draw() 
            


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
                    self.stop_phase()
                    self.stop_trial()

                else: # any other key pressed will be response to color change
                    event_type = 'response'
                    self.session.total_responses += 1

                    if t >= self.session.bar_timing[self.session.bar_counter]:

                        if (ev in self.session.settings['keys']['right_index']) and (self.phase_names == 'color_green'):
                            self.session.correct_responses += 1

                        elif (ev in self.session.settings['keys']['left_index']) and (self.phase_names == 'color_red'):
                            self.session.correct_responses += 1
                        
                        if self.session.bar_counter<len(self.session.bar_timing)-1:
                            self.session.bar_counter +=1

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
        attend_block_conditions, bar_pass_direction_at_TR, bar_midpoint_at_TR, trial_type_at_TR, timing='seconds', *args, **kwargs):


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
        bar_pass_direction_at_TR : list
            List/array with the bar direction at each TR. In same cases it can have 
            a list of direction (when several bars on screen). Total length = total # TRs
        bar_midpoint_at_TR : arr
            List/array with the pairs of positions [x,y] of the midpoint of the bar
            per TR. In same cases it can have a list of pairs (when several bars on screen). Total length = total # TRs
        trial_type_at_TR: arr
            List/array with trial identifier ("trial type"). To know if cue, empty or miniblock

            
        """
        
        self.ID = trial_nr # trial identifier, not sure if needed
        self.bar_pass_direction_at_TR = bar_pass_direction_at_TR
        self.bar_midpoint_at_TR = bar_midpoint_at_TR
        self.trial_type_at_TR = trial_type_at_TR
        self.attend_block_conditions = attend_block_conditions
        self.session = session

        # phase durations for each condition 
        self.phase_durations = phase_durations
        self.phase_names = phase_names


        super().__init__(session, trial_nr, phase_durations, phase_names, verbose=False, *args, **kwargs)

        # get bar and background positions for this trial
        self.position_dictionary = get_object_positions(self.session.grid_pos, self.bar_midpoint_at_TR, self.bar_pass_direction_at_TR,
                                                    self.session.bar_width_pix, screen = self.session.screen, 
                                                    num_bar = len(self.session.attend_block_conditions))


    def draw(self): 

        """ Draw stimuli - pRF bars - for each trial """
        
        current_time = self.session.clock.getTime() # get time

        ## orientation switch times
        if self.session.ori_counter<len(self.session.ori_switch_times): # if counter within number of switch moments
            if current_time >= self.session.ori_switch_times[self.session.ori_counter]: # when switch time reached, switch ori and increment counter
                
                self.session.ori_bool = True
                self.session.ori_counter += 1

        ## bar counter, for responses sanity check
        if self.session.bar_counter<len(self.session.true_responses):
            if current_time >= (self.session.bar_timing[self.session.bar_counter] + self.session.settings['mri']['TR']): # if no valid reply in this window, increment
                self.session.bar_counter += 1 


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


            self.cue_stim = visual.Rect(win = self.session.win,
                                        units = "pix",
                                        width = cue_width,
                                        height = cue_height,
                                        fillColor = self.session.settings['stimuli']['conditions'][cue_condition]['element_color'],
                                        lineColor = self.session.settings['stimuli']['conditions'][cue_condition]['element_color'],
                                        contrast = self.session.settings['stimuli']['feature']['cue_contrast'],
                                        opacity = 1.0,
                                        pos = [0, 0],
                                        fillColorSpace = 'rgb255',
                                        lineColorSpace = 'rgb255'
                                        )
            self.cue_stim.draw()

            print('cue '+attend_cond)
                    

        elif 'mini_block' in self.trial_type_at_TR: # if cue at TR, draw background and word cue: # if bar pass at TR, then draw bar

            if self.phase_names[int(self.phase)] == 'stim': 

                # get list of condition names (as defined in yml) for this phase
                this_phase = list(self.session.all_bar_pos[self.trial_type_at_TR].keys())
                this_phase = ['color_red' if 'red' in p else 'color_green' for _,p in enumerate(this_phase)]

                self.session.feature_stim.draw(bar_midpoint_at_TR = self.bar_midpoint_at_TR, 
                                               bar_pass_direction_at_TR = self.bar_pass_direction_at_TR,
                                               this_phase = this_phase,
                                               position_dictionary = self.position_dictionary,
                                               orientation = self.session.ori_bool,
                                               drawing_ind = self.session.drawing_ind[self.ID]) 


        print(self.phase_names[int(self.phase)]) #print(self.phase_names[int(self.phase)])
                

        # set orientation bool counter to false
        self.session.ori_bool = False

        # draw delimitating black bars, to make display square
        self.session.rect_left.draw()
        self.session.rect_right.draw()

        # fixation lines
        self.session.line1.draw() 
        self.session.line2.draw() 
            


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
                    self.stop_phase()

                else: # any other key pressed will be response to color change
                    event_type = 'response'
                    self.session.total_responses += 1

                    if t >= self.session.bar_timing[self.session.bar_counter]:

                        if (ev in self.session.settings['keys']['left_index']) and (self.session.true_responses[self.session.bar_counter] == 'same'):
                            self.session.correct_responses += 1
                            if self.session.bar_counter<len(self.session.true_responses):
                                self.session.bar_counter += 1 
                        elif (ev in self.session.settings['keys']['right_index']) and (self.session.true_responses[self.session.bar_counter] == 'different'): 
                            self.session.correct_responses += 1
                            if self.session.bar_counter<len(self.session.true_responses):
                                self.session.bar_counter += 1 



                # log everything into session data frame
                idx = self.session.global_log.shape[0]
                self.session.global_log.loc[idx, 'trial_nr'] = self.ID
                self.session.global_log.loc[idx, 'onset'] = t
                self.session.global_log.loc[idx, 'event_type'] = event_type
                self.session.global_log.loc[idx, 'phase'] = self.phase
                self.session.global_log.loc[idx, 'response'] = ev                

                for param, val in self.parameters.items():
                    self.session.global_log.loc[idx, param] = val




class FlickerTrial(Trial):

    def __init__(self, session, trial_nr, phase_durations, phase_names,
        bar_ecc_index_at_trial, ecc_midpoint_at_trial, timing='seconds', *args, **kwargs):


        """ Initializes a FlickerTrial object. 

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
        bar_ecc_index_at_trial : int
            eccentricity index for bar positions: 0 - furthest ecc; 3 - closest ecc
        ecc_midpoint_at_trial : float
            eccentricity (in pixels) of bar position for trial (if empty, then nan) 

            
        """
        
        self.ID = trial_nr # trial identifier, not sure if needed
        self.bar_ecc_index_at_trial = bar_ecc_index_at_trial
        self.ecc_midpoint_at_trial = ecc_midpoint_at_trial
        self.session = session

        # phase durations for each condition 
        self.phase_durations = phase_durations
        # name of each condition
        self.phase_names = phase_names 


        super().__init__(session, trial_nr, phase_durations, phase_names, verbose=False, *args, **kwargs)

        # get bar and background positions for this trial
        self.position_dictionary = get_square_positions(self.session.grid_pos, self.ecc_midpoint_at_trial, 
                                                    self.session.bar_width_pix, screen = self.session.screen)


    def draw(self): 

        """ Draw stimuli - pRF bars - for each trial """
        
        ## draw stim

        self.session.flicker_stim.draw(ecc_midpoint_at_trial = self.ecc_midpoint_at_trial, 
                                       this_phase = self.phase_names[int(self.phase)],
                                       position_dictionary = self.position_dictionary,
                                       orientation = False) 
        

        # set orientation bool counter to false
        self.session.ori_bool = False

        # draw delimitating black bars, to make display square
        self.session.rect_left.draw()
        self.session.rect_right.draw()

        # fixation lines
        self.session.line1.draw() 
        self.session.line2.draw() 



    def get_events(self):
        """ Logs responses/triggers """
        for ev, t in event.getKeys(timeStamped=self.session.clock): # list of of (keyname, time) relative to Clock’s last reset
            if len(ev) > 0:
                if ev in ['q']:
                    print('experiment canceled by user')  
                    self.session.close()
                    self.session.quit()

                elif ev in ['space',3]: # end trial
                    print('trial ended by user')  
                    event_type = 'end_trial'

                    # save updated condition settings per trial
                    # so color is used for other tasks
                    settings_out = os.path.join(self.session.output_dir, self.session.output_str + '_updated_settings.yml')
                    settings_out = re.sub(r'run-.+?,?(\_|$)', "trial-{ID}_".format(ID = str(self.ID).zfill(2)), settings_out)
                    

                    with open(settings_out, 'w') as f_out:  # write settings to disk
                        yaml.dump(self.session.updated_settings, f_out, indent=4, default_flow_style=False)


                    if self.ID == (len(self.session.settings['stimuli']['flicker']['bar_ecc_index'])-1): # if last trial                        
                        self.session.close()
                        self.session.quit()
                    else:
                        self.session.lum_responses = 1 # restart luminance counter for next trial
                        self.stop_phase()
                        self.stop_trial() 


                else: # any other key pressed will be response to color change
                    event_type = 'response'
                    
                    if ev in self.session.settings['keys']['right_index']:
                        self.session.lum_responses += self.session.settings['stimuli']['flicker']['increment']
                    elif ev in self.session.settings['keys']['left_index']:
                        self.session.lum_responses -= self.session.settings['stimuli']['flicker']['increment']

                    # clip it so participants don't endup with humongous values
                    self.session.lum_responses = np.clip(self.session.lum_responses,0,1) 


                # log everything into session data frame
                idx = self.session.global_log.shape[0]
                self.session.global_log.loc[idx, 'trial_nr'] = self.ID
                self.session.global_log.loc[idx, 'onset'] = t
                self.session.global_log.loc[idx, 'event_type'] = event_type
                self.session.global_log.loc[idx, 'phase'] = self.phase
                self.session.global_log.loc[idx, 'response'] = ev                

                for param, val in self.parameters.items():
                    self.session.global_log.loc[idx, param] = val









