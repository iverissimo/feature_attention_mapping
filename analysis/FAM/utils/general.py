import numpy as np
import re
import os
import os.path as op
import pandas as pd

import colorsys

class Utils:

    def __init__(self):

        """__init__
        constructor for utilities general class 
        for functions that are useful throughout data types and analysis pipelines
        but dont really have dependencies
            
        """

    def dva_per_pix(self, height_cm = 39.3, distance_cm = 194, vert_res_pix = 1080):

        """ calculate degrees of visual angle per pixel, 
        to use for screen boundaries when plotting/masking

        Parameters
        ----------
        height_cm : int
            screen height
        distance_cm: float
            screen distance (same unit as height)
        vert_res_pix : int
            vertical resolution of screen
        
        Outputs
        -------
        deg_per_px : float
            degree (dva) per pixel
        
        """

        # screen size in degrees / vertical resolution
        deg_per_px = (2.0 * np.degrees(np.arctan(height_cm /(2.0*distance_cm))))/vert_res_pix

        return deg_per_px 
    
    def rgb255_2_hsv(self, arr):
        
        """ convert RGB 255 to HSV
        
        Parameters
        ----------
        arr: list/array
            1D list of rgb values
            
        """
        
        rgb_norm = np.array(arr)/255
        
        hsv_color = np.array(colorsys.rgb_to_hsv(rgb_norm[0],rgb_norm[1],rgb_norm[2]))
        hsv_color[0] = hsv_color[0] * 360
        
        return hsv_color
    
    def normalize(self, M):

        """
        normalize data array
        """
        return (M-np.nanmin(M))/(np.nanmax(M)-np.nanmin(M))
    

    
    
