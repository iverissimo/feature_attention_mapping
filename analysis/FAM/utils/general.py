import numpy as np
import re
import os
import os.path as op
import pandas as pd

import colorsys
import itertools

from joblib import Parallel, delayed
from statsmodels.stats import weightstats

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
    
    def split_half_comb(self, input_list):

        """ make list of lists, by spliting half
        and getting all unique combinations
        
        Parameters
        ----------
        input_list : list/arr
            list of items
        Outputs
        -------
        unique_pairs : list/arr
            list of tuples
        
        """

        A = list(itertools.combinations(input_list, int(len(input_list)/2)))
        
        combined_pairs = []
        for pair in A:
            combined_pairs.append(tuple([pair, tuple([r for r in input_list if r not in pair])]))

        # get unique pairs
        seen = set()
        unique_pairs = [t for t in combined_pairs if tuple(sorted(t)) not in seen and not seen.add(tuple(sorted(t)))]

        return unique_pairs
    
    def correlate_arrs(self, data1, data2, n_jobs = 4, weights=[], shuffle_axis = None, seed=None):
        
        """
        Compute Pearson correlation between two numpy arrays
        
        Parameters
        ----------
        data1 : str/list/array
            numpy array OR absolute filename of array OR list filenames
        data2 : str/list/array
            same as data1
        n_jobs : int
            number of jobs for parallel
        seed: int
            if provided, will initialize random with specific seed
        
        """ 

        # if we want to use specific seed
        if seed is not None:
            np.random.seed(seed)
        
        data1_arr = []
        data2_arr = []
        
        ## if list was provided, then load and average
        if isinstance(data1, list) and np.array(data1).shape[0]>1:
            data1_arr = np.mean(np.stack(np.load(v) for v in list(data1)), axis = 0)
        elif isinstance(data1, str):
            data1_arr = np.load(data1)
        else: # isinstance(data1, np.ndarray):
            data1_arr = np.array(data1)
            
        if isinstance(data2, list) and np.array(data2).shape[0]>1:
            data2_arr = np.mean(np.stack(np.load(v) for v in list(data2)), axis = 0)
        elif isinstance(data2, str):
            data2_arr = np.load(data2)
        else: # isinstance(data2, np.ndarray):
            data2_arr = np.array(data2)

        # if we indicate an axis to shuffle, then do so
        if shuffle_axis is not None:

            if shuffle_axis == -1:
                data_shuf1 = data1_arr.T.copy()
                np.random.shuffle(data_shuf1)
                data1_arr = data_shuf1.T.copy()

                data_shuf2 = data2_arr.T.copy()
                np.random.shuffle(data_shuf2)
                data2_arr = data_shuf2.T.copy()

            elif shuffle_axis == 0:
                np.random.shuffle(data1_arr)
                np.random.shuffle(data2_arr)
            
        ## actually correlate
        correlations = np.array(Parallel(n_jobs=n_jobs)(delayed(np.corrcoef)(data1_arr[i], data2_arr[i]) for i in np.arange(data1_arr.shape[0])))[...,0,1]
                
        return correlations
    
    def weighted_mean(self, data1, weights=None, norm=False):
        
        """
        Compute (Weighted) mean 
        with statsmodel
        
        Parameters
        ----------
        data1 : arr
            numpy array 
        weights : arr
        
        """ 

        if norm:
            weights = self.normalize(weights)

        if weights is not None:
            weights[np.where((np.isinf(weights)) | (np.isnan(weights)) | (weights == 0))] = 0.000000001

        avg_data = weightstats.DescrStatsW(data1,weights=weights).mean

        return avg_data
    
