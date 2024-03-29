import numpy as np
import re
import os
import os.path as op
import pandas as pd
from tqdm import tqdm

import colorsys
import itertools

from joblib import Parallel, delayed
from statsmodels.stats import weightstats
from scipy import fft, interpolate

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

class Utils:

    def __init__(self):

        """__init__
        constructor for utilities general class 
        for functions that are useful throughout data types and analysis pipelines
        but dont really have dependencies
            
        """

    def save_str2file(self, txt = '', filename = ''):

        """
        Save string (example, specific command) as txt file

        Parameters
        ----------
        txt: str
            str to save
        filename: str
            absolute filename to save text
        """
        file2write = open(filename, 'w')
        file2write.write(txt)
        file2write.close()

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
        if isinstance(data1, list):
            if len(data1) == 1:
                data1_arr = np.load(data1[0])
            else:
                data1_arr = np.mean(np.stack(np.load(v) for v in list(data1)), axis = 0)
        elif isinstance(data1, str):
            data1_arr = np.load(data1)
        elif isinstance(data1, np.ndarray):
            data1_arr = data1
            
        if isinstance(data2, list):
            if len(data2) == 1:
                data2_arr = np.load(data2[0])
            else:
                data2_arr = np.mean(np.stack(np.load(v) for v in list(data2)), axis = 0)
        elif isinstance(data2, str):
            data2_arr = np.load(data2)
        elif isinstance(data2, np.ndarray):
            data2_arr = data2

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
        correlations = np.array(Parallel(n_jobs=n_jobs)(delayed(np.corrcoef)(data1_arr[i], data2_arr[i]) for i in tqdm(np.arange(data1_arr.shape[0]))))[...,0,1]
                
        return correlations
    
    def weighted_mean(self, data1, weights=None, norm=False):
        
        """
        Compute (Weighted) mean with statsmodel
        
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
    
    def weighted_mean_std_sem(self, data1, weights=None, norm=False):
        
        """
        Compute (Weighted) standard deviations of mean with statsmodel
        and weigthed standard error of the mean
        
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

        avg_data = weightstats.DescrStatsW(data1,weights=weights).std_mean

        return avg_data, avg_data/np.sqrt(len(data1)) # sem
    
    def weighted_corr(self, data1, data2, weights=None, norm=False):

        """
        Compute (Weighted) correlation between two numpy arrays
        with statsmodel
        
        Parameters
        ----------
        data1 : arr
            numpy array 
        data2 : arr
            same as data1
        weights : arr
        """ 

        if norm:
            weights = self.normalize(weights)

        if weights is not None:
            weights[np.where((np.isinf(weights)) | (np.isnan(weights)) | (weights == 0))] = 0.000000001

        corr = weightstats.DescrStatsW(np.vstack((data1,data2)), weights=weights).corrcoef

        return corr

    def resample_arr(self, upsample_data, osf = 10, final_sf = 1.6):

        """ resample array using cubic interpolation
        
        Parameters
        ----------
        upsample_data : arr
            1d array that is upsampled
        osf : int
            oversampling factor (that data was upsampled by)
        final_sf: float
            final sampling rate that we want to obtain
            
        """
        
        # original scale of data in seconds
        original_scale = np.arange(0, upsample_data.shape[-1]/osf, 1/osf)

        # cubic interpolation of predictor
        interp = interpolate.interp1d(original_scale, 
                                    upsample_data, 
                                    kind = "cubic", axis=-1)
        
        desired_scale = np.arange(0, upsample_data.shape[-1]/osf, final_sf) # we want the predictor to be sampled in TR

        out_arr = interp(desired_scale)
        
        return out_arr
    
    def leave_one_out(self, input_list):

        """ make list of lists, by leaving one out

        Parameters
        ----------
        input_list : list/arr
            list of items

        Outputs
        -------
        out_lists : list/arr
            list of lists, with each element
            of the input_list left out of the returned lists once, in order
        """

        out_lists = []
        for x in input_list:
            out_lists.append([y for y in input_list if y != x])

        return out_lists
    
    def crop_shift_arr(self, arr, crop_nr = None, shift = 0):
        
        """
        crop and shift array
        
        Parameters
        ----------
        arr : array
            original array
            assumes time dim is last one (arr.shape[-1])
        crop_nr : None or int
            if not none, expects int with number of FIRST time points to crop
        shift : int
            positive or negative int, of number of time points to shift (if neg, will shift leftwards)
        """
            
        # if cropping
        if crop_nr:
            new_arr = arr[...,crop_nr:]
        else:
            new_arr = arr
            
        # if shiftting
        out_arr = new_arr.copy()
        if shift > 0:
            out_arr[...,shift:] = new_arr[..., :-int(shift)]
        elif shift < 0:
            out_arr[...,:shift] = new_arr[..., np.abs(shift):]
            
        return out_arr
    
    def get_loo_filename(self, input_list, loo_key = 'loo_r1s1'):
        
        """ get filename for loo run, and return
        that filename plus list with rest of files
        
        Parameters
        ----------
        input_list : list/arr
            list of items
        loo_key: str
            key with info about run number and session number to leave out
            (requires both, will through error if not provided)
        
        """
        
        if 'loo_' in loo_key:
            
            # find run to use
            run_num = re.findall(r'r\d{1,3}', loo_key)[0][1:]
            
            # find ses number to use
            ses_num = re.findall(r's\d{1,3}', loo_key)[0][1:]
            
            if len(ses_num) == 0 or len(run_num) == 0:
                raise NameError('Run number or session number not provided')
            else:
                print('Leaving out run-{r} from ses-{s}'.format(r=run_num, s=ses_num))
                test_filename = [x for x in input_list if 'run-{r}'.format(r=run_num) in x and \
                                'ses-{s}'.format(s=ses_num) in x]
                
                train_filename = [x for x in input_list if x not in test_filename]
        
        if len(test_filename) == 0 or len(train_filename) == 0:
                raise NameError('Could not find test/train runs with loo key')
        else:
            return test_filename, train_filename
        
    def get_run_ses_from_str(self, input_name):
        
        """ 
        get run number and session number from string
        
        Parameters
        ----------
        input_name : str
            name of file
        
        """
        # find run number
        run_num = int(re.findall(r'run-\d{1,3}', input_name)[0][4:])
        
        # find ses number
        ses_num = int(re.findall(r'ses-\d{1,3}', input_name)[0][4:])
        
        return run_num, ses_num
    
    def get_bar_overlap_dm(self, bar_arr):
        
        """
        get DM of spatial positions where bars overlap

        Parameters
        ----------
        bar_arr: arr
            4D array with [bars,x,y,t]

        """ 
        
        if len(bar_arr.shape) != 4:
            raise ValueError('Input array must be 4D')
            
        # sum over bars, and set locations of overlap as 1, else 0
        overlap_dm = np.sum(bar_arr, axis = 0)
        overlap_dm[overlap_dm <= 1] = 0
        overlap_dm[overlap_dm > 1] = 1
        
        return overlap_dm

    def get_weighted_bins(self, data_df, x_key = 'ecc', y_key = 'size', weight_key = 'rsq', sort_key = 'ecc', n_bins = 10):

        """ 
        
        Get weighted bins from dataframe, sorted by one of the variables

        """
        
        # sort values by eccentricity
        data_df = data_df.sort_values(by=[sort_key])

        #divide in equally sized bins
        df_batches = np.array_split(data_df, n_bins)
        print('Bin size is %i'%int(len(data_df)/n_bins))
        
        mean_x = []
        mean_x_std = []
        mean_y = []
        mean_y_std = []
        
        # for each bin calculate rsq-weighted means and errors of binned ecc/gain 
        for j in np.arange(len(df_batches)):
            mean_x.append(weightstats.DescrStatsW(df_batches[j][x_key],
                                                weights = df_batches[j][weight_key]).mean)
            mean_x_std.append(weightstats.DescrStatsW(df_batches[j][x_key],
                                                    weights = df_batches[j][weight_key]).std_mean)

            mean_y.append(weightstats.DescrStatsW(df_batches[j][y_key],
                                                weights = df_batches[j][weight_key]).mean)
            mean_y_std.append(weightstats.DescrStatsW(df_batches[j][y_key],
                                                    weights = df_batches[j][weight_key]).std_mean)

        return mean_x, mean_x_std, mean_y, mean_y_std

    def sum_bar_dms(self, stacked_dms, overlap_dm = None, overlap_weight = 1):

        """
        sum visual dms of both bars
        and set value of overlap if given

        Parameters
        ----------
        stacked_dms: arr
            4D array with [bars,x,y,t]. Assumes dms we're already weighted (if such is the case)
        overlap_dm: arr
            if not None, excepts binary array of [x,y,t] with overlap positions in time
        overlap_weight: int/float
            weight to give overlap area

        """ 

        final_dm = np.sum(stacked_dms, axis=0) 
        
        if overlap_dm is not None:
            final_dm[overlap_dm == 1] = overlap_weight
        
        return final_dm
    
    def create_palette(self, key_list = [], cmap = 'magma', num_col = None):

        """
        create dict with color palette colors
        """
        
        if num_col is None:
            num_col = len(key_list)

        # get color map array
        col_arr = np.array(sns.color_palette(cmap, num_col))

        # actually create dict
        dict_colors = {key: col_arr[i] for i, key in enumerate(key_list)}

        return dict_colors
    
    def calc_within_sub_sem(self, df_data = None, main_var = 'RT', conditions = ['bar_ecc_deg', 'interbar_dist_deg'], 
                                pp_key = 'sj'):

        """
        remove between-subject variability
        and calculate SEM (for plotting purposes)
        """

        ## get subject average
        sub_avg = df_data.groupby([pp_key]).mean(numeric_only=True).reset_index()
        sub_avg = sub_avg.loc[:, [pp_key, main_var]]
        # rename to new column
        sub_avg.rename(columns={main_var: 'avg_%s'%main_var}, inplace=True)
        # add column with grand average
        sub_avg.loc[:, ['group_avg_%s'%main_var]] = df_data[main_var].mean()

        ## merge values to dataframe
        df_data = df_data.merge(sub_avg, on=[pp_key])

        ## remove the between-subject variability
        norm_value = df_data[main_var].values - df_data['avg_%s'%main_var].values + df_data['group_avg_%s'%main_var].values
        df_data.loc[:, ['norm_%s'%main_var]] = norm_value

        ## calculate the standard error of the mean
        sem_df = df_data.groupby(conditions)['norm_%s'%main_var].sem().reset_index()
        sem_df.rename(columns={'norm_%s'%main_var: 'SEM_%s'%main_var}, inplace=True)
        
        ## final merge
        df_data = df_data.merge(sem_df, on=conditions)

        return df_data







