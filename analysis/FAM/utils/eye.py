import numpy as np
import re
import os
import os.path as op
import pandas as pd
import hedfpy


from FAM.utils import Utils

class EyeUtils(Utils):

    def __init__(self):
        
        """__init__
        constructor for utilities eyetracking class 
            
        """

    def edf2h5(self, edf_files = [], hdf_file = '', pupil_hp = 0.01, pupil_lp = 6.0):
    
        """ convert edf file(s) into one hdf5 file, for later analysis
        
        Parameters
        ----------
        edf_files : List/arr
            list of absolute filenames for edf file(s)
        hdf_file : str
            absolute filename of output hdf5 file

        Outputs
        -------
        all_alias: List
            list of strings with alias for each run
        
        """
        
        # first check if hdf5 already exists
        if op.isfile(hdf_file):
            print('The file %s already exists, skipping'%hdf_file)
            
            all_alias = [op.split(ef)[-1].replace('.edf','') for _,ef in enumerate(edf_files)]
            
        else:
            ho = hedfpy.HDFEyeOperator(input_object=hdf_file)

            all_alias = []

            for ef in edf_files:
                alias = op.splitext(os.path.split(ef)[1])[0] #name of data for that run
                ho.add_edf_file(ef)
                ho.edf_message_data_to_hdf(alias = alias) #write messages ex_events to hdf5
                ho.edf_gaze_data_to_hdf(alias = alias, pupil_hp = pupil_hp, pupil_lp = pupil_lp) #add raw and preprocessed data to hdf5   

                all_alias.append(alias)
        
        return all_alias
    
    def mean_dist_deg(self, xx = [], yy = [], origin = None, screen_res = [1920,1080], height_cm = 39.3, distance_cm = 194):
    
        """ calculate mean distance in deg (and std)
        from an origin point 
        given list of positions in pix
        
        Parameters
        ----------
        xx : list/array
            list of x gaze positions
        yy : list/array
            list of y gaze positions
        origin : list
            origin point [x,y]
        screen_res : list
            resolution of screen
        height_cm : float
            screen height
        distance_cm: float
            screen distance (same unit as height)

        Outputs
        -------
        mean_dist_deg : float
            in degree (dva)
        std_dist_deg : float
            in degree (dva)
        
        """
        
        # calculate distance of gaze from origin
        if origin is None:
            origin = np.array(screen_res)/2 # if not given, defaults to center of screen
        
        dist_pix = np.sqrt((np.array(xx - origin[0])**2) + (np.array(yy - origin[1])**2))
        
        # convert from pixel to dva
        dist_deg = dist_pix * self.dva_per_pix(height_cm = height_cm, distance_cm = distance_cm, vert_res_pix = screen_res[-1])
        
        # calculate mean and std
        mean_dist_deg = np.nanmean(dist_deg)
        std_dist_deg = np.nanstd(dist_deg)
        
        return mean_dist_deg, std_dist_deg
    
    def get_saccade_angle(self, sacc_arr, use_deg = True):
    
        """
        convert vector position of saccade to angle

        Parameters
        ----------
        sacc_arr : array
            array of saccade coordinates [N x 2] - where N is number of saccades
        use_deg : bool
            if we want angles in deg or radians
        """
        
        # compute complex location
        complex_list = [sac[0] + sac[1] * 1j for _,sac in enumerate(sacc_arr)]
        
        # actually calculate angle
        angles = np.angle(complex_list, deg = use_deg)
        
        return list(angles)