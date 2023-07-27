## general packages
from filecmp import cmp
import numpy as np
import os
from os import path as op
import pandas as pd
import re, json
from shutil import copy2
import itertools

## imaging, processing, stats packages
import nibabel as nib

from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter, fftconvolve
from scipy import fft, interpolate
from scipy.stats import t, norm, linregress
from scipy.interpolate import interp1d

from nilearn import surface
from nilearn.signal import clean

from nilearn.glm.first_level.hemodynamic_models import spm_hrf, spm_time_derivative, spm_dispersion_derivative
from nilearn.glm.first_level import first_level
from nilearn.glm.regression import ARModel

from statsmodels.stats import weightstats

## plotting packages
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import cortex
from PIL import Image, ImageDraw

## fitting packages 
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel
from prfpy.rf import gauss2D_iso_cart
from prfpy.timecourse import stimulus_through_prf, filter_predictions

from joblib import Parallel, delayed
from lmfit import Parameters, minimize
from tqdm import tqdm

import functools
import operator

from nilearn.glm.first_level.design_matrix import _cosine_drift as discrete_cosine_transform

from FAM.utils.general import Utils

class MRIUtils(Utils):

    def __init__(self):
        
        """__init__
        constructor for utilities mri class 
            
        """

    def create_glasser_df(self, annot_filename = '', pysub = 'hcp_999999'):

        """ Function to create glasser dataframe
        with ROI names, colors (RGBA) and vertex indices

        Parameters
        ----------
        path2file : str 
            absolute name of the parcelation file (defined in MRIObj.atlas_annot)
        """
        
        # we read in the atlas data, which consists of 180 separate regions per hemisphere. 
        # These are labeled separately, so the labels go to 360.
        cifti = nib.load(annot_filename)

        # get index data array
        cifti_data = np.array(cifti.get_fdata(dtype=np.float32))[0]
        # from header get label dict with key + rgba
        cifti_hdr = cifti.header
        label_dict = cifti_hdr.get_axis(0)[0][1]

        # number of vertices in one hemisphere (for bookeeping) 
        hemi_vert_num = cortex.db.get_surfinfo('hcp_999999').left.shape[0] 
        
        ## make atlas data frame
        atlas_df = pd.DataFrame({'ROI': [], 'hemi_vertex': [], 'merge_vertex': [], 'hemisphere': [],
                                 'R': [],'G': [],'B': [],'A': []})

        for key in label_dict.keys():

            if label_dict[key][0] != '???': 
                
                # name of atlas roi
                roi_name = re.findall(r'_(.*?)_ROI', label_dict[key][0])[0]
                # hemisphere
                hemi = re.findall(r'(.*?)_', label_dict[key][0])[0]
                # get vertex indices for whole surface or each hemisphere separately
                merge_vert = np.where(cifti_data == key)[0] 
                hemi_vert = np.where(cifti_data == key)[0] - hemi_vert_num  if hemi == 'R' else np.where(cifti_data == key)[0]
                
                # fill df
                atlas_df = pd.concat((atlas_df,
                                    pd.DataFrame({'ROI': np.tile(roi_name, len(merge_vert)),
                                                'hemisphere': np.tile(hemi, len(merge_vert)),
                                                'hemi_vertex': hemi_vert, 
                                                'merge_vertex': merge_vert,
                                                'R': np.tile(label_dict[key][1][0], len(merge_vert)),
                                                'G': np.tile(label_dict[key][1][1], len(merge_vert)),
                                                'B': np.tile(label_dict[key][1][2], len(merge_vert)),
                                                'A': np.tile(label_dict[key][1][3], len(merge_vert))
                                                })), ignore_index=True
                                    )
                
        return atlas_df
    

    def get_vertex_rois(self, sub_id = None, pysub = 'hcp_999999', use_atlas = None, 
                        annot_filename = ''):

        """ 

        NOT DONE YET - BEFORE USE ATLAS WAS BOOL, NOW NEED TO DEFINE NONE IF USING HAND-DRAWN ROIS,
        OR STRING OF GLASSER VS WANG ATLAS


        helper function to get ROI names, vertice index and color palette
        to be used in plotting scripts
        
        Parameters
        ----------
        params : dict
            yaml dict with task related infos  
        sub_id: str, int or list
            subject ID to add as identifier in outputed dictionaries
        pysub: str/dict
            name of pycortex subject folder, where we drew all ROIs.
            if dict, assumes key is participant ID, and value is sub specific pycortex folder 
        use_atlas: bool
            if we want to use the glasser atlas ROIs instead (this is, from the keys conglomerate defined in the params yml)
        atlas_pth: str
            path to atlas file
        space: str
            pycortex subject space
        """ 

        if use_atlas == 'glasser':

            # Get Glasser atlas
            atlas_df, atlas_array = self.create_glasser_df(annot_filename = annot_filename)
        
        if sub_id:
            
            # if single id provided, put in list
            if isinstance(sub_id, str) or isinstance(sub_id, int):
                sub_id = [sub_id]

            sub_id_list = ['sub-{sj}'.format(sj = str(pp).zfill(3)) if 'sub-' not in str(pp) else str(pp) for pp in sub_id]

            ## start empty dictionaries  
            ROIs = {}
            color_codes = {}
            roi_verts = {}

            # loop over participant list
            for pp in sub_id_list:

                print('Getting ROIs for participants %s'%pp)

                if use_atlas:
                    print('Using Glasser ROIs')
                    # ROI names
                    ROIs[pp] = list(params['plotting']['ROIs']['glasser_atlas'].keys())

                    # colors
                    color_codes[pp] = {key: params['plotting']['ROIs']['glasser_atlas'][key]['color'] for key in ROIs[pp]}

                    # get vertices for ROI
                    roi_verts[pp] = {}
                    for _,key in enumerate(ROIs[pp]):
                        print(key)
                        roi_verts[pp][key] = np.hstack((np.where(atlas_array == ind)[0] for ind in atlas_df[atlas_df['ROI'].isin(params['plotting']['ROIs']['glasser_atlas'][key]['ROI'])]['index'].values))

                else:
                    ## check if dict or str
                    if isinstance(pysub, dict):
                        pysub_pp = pysub[pp]
                    else:
                        pysub_pp = pysub

                    # set ROI names
                    ROIs[pp] = params['plotting']['ROIs'][space]

                    # dictionary with one specific color per group - similar to fig3 colors
                    color_codes[pp] = {key: params['plotting']['ROI_pal'][key] for key in ROIs[pp]}

                    # get vertices for ROI
                    roi_verts[pp] = {}
                    for _,val in enumerate(ROIs[pp]):
                        print(val)
                        roi_verts[pp][val] = cortex.get_roi_verts(pysub_pp,val)[val]
                
        else:
            raise NameError('No subject ID provided')
        
        return ROIs, roi_verts, color_codes


    def get_tsnr(input_file, return_mean=True, affine=[], hdr=[], filename=None):
        
        """
        Compute the tSNR of NIFTI file
        and generate the equivalent NIFTI SNR 3Ds. 

        Parameters
        ----------
        input_file : str/array
            if str then absolute NIFTI filename to load, else it should be array with data
        return_mean : bool
            if we want to return mean tsnr value over volume, or array with tsnr value per voxel
        affine : array
            if we input data array, then we also need to input affine
        hdr : array
            if we input data array, then we also need to input header
        filename : str or None
            if str is provided, then it will save tSNR NIFTI with said filename
        
        Outputs
        -------
        mean_tsnr: float
            mean tsnr value over volume
        OR
        tsnr: arr
            array with tsnr value per voxel
        
        """
        
        if isinstance(input_file, np.ndarray): # if already data array, then just use it 
            data = input_file
        else:
            if isinstance(input_file, nib.Nifti1Image): # load nifti image
                img = input_file
            elif isinstance(input_file, str):
                img = nib.load(input_file)
        
            data = img.get_fdata()
            affine = img.affine
            hdr = img.header

        # calculate tSNR
        mean_d = np.mean(data,axis=-1)
        std_d = np.std(data,axis=-1)
        
        tsnr = mean_d/std_d
        tsnr[np.where(np.isinf(tsnr))] = np.nan

        mean_tsnr = np.nanmean(np.ravel(tsnr))

        # if we want to save image, need to provide an output filename
        if filename:
            tsnr_image = nib.Nifti1Image(tsnr, affine=affine, header=hdr).to_filename(filename)

        if return_mean:
            return mean_tsnr
        else:
            return tsnr


    def weighted_corr(data1, data2, weights=None):

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

        if weights is not None:
            weights[np.where((np.isinf(weights)) | (np.isnan(weights)) | (weights == 0))] = 0.000000001

        corr = weightstats.DescrStatsW(np.vstack((data1,data2)), weights=weights).corrcoef

        return corr


    def weighted_mean(data1, weights=None, norm=False):
        
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
            weights = normalize(weights)

        if weights is not None:
            weights[np.where((np.isinf(weights)) | (np.isnan(weights)) | (weights == 0))] = 0.000000001

        avg_data = weightstats.DescrStatsW(data1,weights=weights).mean

        return avg_data
        

    def correlate_arrs(data1, data2, n_jobs = 4, weights=[], shuffle_axis = None, seed=None):
        
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
            data1_arr = np.mean(np.stack(np.load(v) for v in list(data1)), axis = 0)
        elif isinstance(data1, str):
            data1_arr = np.load(data1)
        elif isinstance(data1, np.ndarray):
            data1_arr = data1
            
        if isinstance(data2, list):
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
        correlations = np.array(Parallel(n_jobs=n_jobs)(delayed(np.corrcoef)(data1_arr[i], data2_arr[i]) for i in np.arange(data1_arr.shape[0])))[...,0,1]
                
        return correlations

    def smooth_surface(data, pysub = 'hcp_999999', kernel=3, nr_iter=3, normalize = False):

        """
        smooth surface data, with a given kernel size 
        (not mm but factor, see cortex.polyutils.Surface.smooth)

        Parameters
        ----------
        data : array
            data array to be smoothed
        pysub : str
            basename of pycortex subject, to use coordinate information
        kernel : int
            size of "kernel" to use for smoothing (factor)
        nr_iter: int
            number of iterations to repeat smoothing, larger values smooths more
        normalize: bool
            if we want to max normalize smoothed data (default = False)
        
        """

        ## get surface data for both hemispheres
        lh_surf_data, rh_surf_data = cortex.db.get_surf(pysub, 'fiducial')
        lh_surf, rh_surf = cortex.polyutils.Surface(lh_surf_data[0], lh_surf_data[1]),cortex.polyutils.Surface(rh_surf_data[0], rh_surf_data[1])

        ## smooth data from each hemisphere, according to surface coordinates
        ## first remove nans (turn to 0)
        data[np.isnan(data)] = 0

        lh_data_smooth = lh_surf.smooth(data[:lh_surf_data[0].shape[0]], factor=kernel, iterations=nr_iter)
        rh_data_smooth = rh_surf.smooth(data[rh_surf_data[0].shape[0]:], factor=kernel, iterations=nr_iter)
        if normalize:
            lh_data_smooth /= lh_data_smooth.max()
            rh_data_smooth /= rh_data_smooth.max()

        return np.concatenate((lh_data_smooth,rh_data_smooth), axis=0)


    def crop_epi(file, outdir, num_TR_crop = 5):

        """ crop epi file (expects numpy file)
        and thus remove the first TRs
        
        Parameters
        ----------
        file : str/list/array
            absolute filename to be cropped (or list of filenames)
        outdir : str
            path to save new file(s)
        num_TR_crop : int
            number of TRs to remove from beginning of file
        
        Outputs
        -------
        out_file: str/list/arr
            absolute output filename (or list of filenames)
        
        """
        
        # check if single filename or list of filenames
        if isinstance(file, list) or isinstance(file, np.ndarray): 
            file_list = file  
        else:
            file_list = [file]
        
        # store output filename in list
        outfiles = []
        
        # for each file, do the same
        for input_file in file_list:
            
            # get file extension
            file_extension = '.{b}'.format(b = input_file.rsplit('.', 2)[-1])

            # set output filename
            output_file = op.join(outdir, 
                        op.split(input_file)[-1].replace(file_extension,'_{name}{ext}'.format(name = 'cropped',
                                                                                            ext = file_extension)))
            # if file already exists, skip
            if op.exists(output_file): 
                print('already exists, skipping %s'%output_file)
            
            else:
                print('making %s'%output_file)
                
                data = np.load(input_file,allow_pickle=True)
                
                crop_data = data[:,num_TR_crop:] 
                        
                print('new file with shape %s' %str(crop_data.shape))
                    
                ## save cropped file
                np.save(output_file,crop_data)

            # append out files
            outfiles.append(output_file)
            
        # if input file was not list, then return output that is also not list
        if not isinstance(file, list) and not isinstance(file, np.ndarray): 
            outfiles = outfiles[0] 

        return outfiles


    def filter_data(file, outdir, filter_type = 'HPgauss', plot_vert = False,
                    first_modes_to_remove = 5, baseline_inter1 = None, baseline_inter2 = None, TR = 1.6, **kwargs):
        
        """ 
        Generic filtering function, implemented different types of filters
        High pass filter NIFTI run with gaussian kernel
        
        Parameters
        ----------
        file : str/list/array
            absolute filename to be filtered (or list of filenames)
        outdir : str
            path to save new file
        filter_type : str
            type of filter to use, defaults to gaussian kernel high pass
        
        Outputs
        -------
        out_file: str
            absolute output filename (or list of filenames)
        
        """
        
        # check if single filename or list of filenames
        
        if isinstance(file, list) or isinstance(file, np.ndarray):
            file_list = file  
        else:
            file_list = [file]
        
        # store output filename in list
        outfiles = []
        
        # for each file, do the same
        for input_file in file_list:
            
            # get file extension
            file_extension = '.{b}'.format(b = input_file.rsplit('.', 2)[-1])

            # set output filename
            output_file = op.join(outdir, 
                        op.split(input_file)[-1].replace(file_extension,'_{filt}{ext}'.format(filt = filter_type,
                                                                                            ext = file_extension)))
            # if file already exists, skip
            if op.exists(output_file): 
                print('already exists, skipping %s'%output_file)
            
            else:
                print('making %s'%output_file)
                
                data = np.load(input_file,allow_pickle=True)
    
                ### implement filter types, by calling their specific functions

                if filter_type == 'HPgauss':

                    data_filt = gausskernel_data(data, **kwargs)
                    
                elif filter_type == 'sg':

                    data_filt = savgol_data(data, **kwargs)

                elif filter_type == 'dc': 

                    data_filt = dc_data(data, first_modes_to_remove = first_modes_to_remove, **kwargs) 

                elif filter_type == 'LinDetrend':

                    data_filt = detrend_data(data, detrend_type = 'linear', TR = TR, baseline_inter1 = baseline_inter1, baseline_inter2 = baseline_inter2, **kwargs)
                    
                else:
                    raise NameError('filter type not implemented')
                    
                # if plotting true, make figure of vertix with high tSNR,
                # to compare the difference
                if plot_vert:

                    tsnr = np.mean(data, axis = -1)/np.std(data, axis = -1)
                    tsnr[np.where(np.isinf(tsnr))] = np.nan
                    
                    ind2plot = np.where(tsnr == np.nanmax(tsnr))[0][0]
                    fig = plt.figure()
                    plt.plot(data[ind2plot,...], color='dimgray',label='Original data')
                    plt.plot(data_filt[ind2plot,...], color='mediumseagreen',label='Filtered data')

                    plt.xlabel('Time (TR)')
                    plt.ylabel('Signal amplitude (a.u.)')
                    plt.legend(loc = 'upper right')

                    fig.savefig(output_file.replace(file_extension,'_vertex_%i.png'%ind2plot))
                
                ## save filtered file
                np.save(output_file,data_filt)

            # append out files
            outfiles.append(output_file)
            
        # if input file was not list, then return output that is also not list
        if not isinstance(file, list) and not isinstance(file, np.ndarray): 
            outfiles = outfiles[0] 
        
        return outfiles


    def gausskernel_data(data, TR = 1.6, cut_off_hz = 0.01, **kwargs):
        
        """ 
        High pass filter array with gaussian kernel
        
        Parameters
        ----------
        data : arr
            data array
        TR : float
            TR for run
        cut_off_hz : float
            cut off frequency to filter
        
        Outputs
        -------
        data_filt: arr
            filtered array
        """ 
            
        # save shape, for file reshaping later
        arr_shape = data.shape
        
        sigma = (1/cut_off_hz) / (2 * TR) 

        # filter signal
        filtered_signal = np.array(Parallel(n_jobs=2)(delayed(gaussian_filter)(i, sigma=sigma) for _,i in enumerate(data.T))) 

        # add mean image back to avoid distribution around 0
        data_filt = data.T - filtered_signal + np.mean(filtered_signal, axis=0)
        
        return data_filt.T # to be again vertex, time


    def savgol_data(data, window_length=201, polyorder=3, **kwargs):
        
        """ 
        High pass savitzky golay filter array
        
        Parameters
        ----------
        data : arr
            data array
        TR : float
            TR for run
        window_length : int
            window length for SG filter (the default is 201, which is ok for prf experiments, and 
            a bit long for event-related experiments)
        polyorder: int
            polynomial order for SG filter (the default is 3, which performs well for fMRI signals
                when the window length is longer than 2 minutes)

        Outputs
        -------
        data_filt: arr
            filtered array
        """ 
            
        if window_length % 2 != 1:
            raise ValueError  # window_length should be odd

        # filter signal
        filtered_signal = savgol_filter(data.T, window_length, polyorder)
        
        # add mean image back to avoid distribution around 0
        data_filt = data.T - filtered_signal + np.mean(filtered_signal, axis=0)

        return data_filt.T # to be again vertex, time


    def dc_data(data, first_modes_to_remove=5, **kwargs):
        
        """ 
        High pass discrete cosine filter array
        
        Parameters
        ----------
        data : arr
            data array
        first_modes_to_remove: int
            Number of low-frequency eigenmodes to remove (highpass)

        Outputs
        -------
        data_filt: arr
            filtered array
        """ 

        # get Discrete Cosine Transform
        coeffs = fft.dct(data, norm='ortho', axis=-1)
        coeffs[...,:first_modes_to_remove] = 0

        # filter signal
        filtered_signal = fft.idct(coeffs, norm='ortho', axis=-1)
        # add mean image back to avoid distribution around 0
        data_filt = filtered_signal + np.mean(data, axis=-1)[..., np.newaxis]

        return data_filt # vertex, time


    def detrend_data(data, detrend_type = 'linear', baseline_inter1 = None, baseline_inter2 = None, 
                                                    dct_set_nr = 1, TR = 1.6, **kwargs):
        
        """ 
        Detrend data array
        
        Parameters
        ----------
        data : arr
            data array 2D [vert, time]
        detrend_type: str
            type of detrending (default 'linear'), can also use discrete cosine transform as low-frequency drift predictors
        dct_set_nr: int
            Number of discrete cosine transform to use
        baseline_inter1: int
            int with indice for end of baseline period
        baseline_inter2: int
            int with indice for start of baseline period

        Outputs
        -------
        data_filt: arr
            filtered array
        """ 

        ## if we want to do a linear detrend
        if detrend_type == 'linear':
            
            # if indices for baseline interval given, use those values for regression line
            if baseline_inter1 is not None and baseline_inter2 is not None:

                ## start baseline interval
                print('Taking TR indices of %s to get trend of start baseline'%str(baseline_inter1))
                
                # make regressor + intercept for initial baseline period
                if isinstance(baseline_inter1, int):
                    start_trend_func = [linregress(np.arange(baseline_inter1), vert[:baseline_inter1]) for _,vert in enumerate(tqdm(data))]
                    start_trend = np.stack((start_trend_func[ind].intercept + start_trend_func[ind].slope * np.arange(baseline_inter1) for ind,_ in enumerate(tqdm(data))))
                else:
                    raise ValueError('index not provided as int')

                ## end baseline interval
                print('Taking TR indices of %s to get trend of end baseline'%str(baseline_inter2))

                # make regressor + intercept for end baseline period
                if isinstance(baseline_inter2, int):
                    end_trend_func = [linregress(np.arange(np.abs(baseline_inter2)), vert[baseline_inter2:]) for _,vert in enumerate(tqdm(data))]
                    end_trend = np.stack((end_trend_func[ind].intercept + end_trend_func[ind].slope * np.arange(np.abs(baseline_inter2)) for ind,_ in enumerate(tqdm(data))))
                else:
                    raise ValueError('index not provided as int')  

                ## now interpolate slope for task period
                baseline_timepoints = np.concatenate((np.arange(data.shape[-1])[:baseline_inter1, ...],
                                                    np.arange(data.shape[-1])[baseline_inter2:, ...]))

                trend_func = [interp1d(baseline_timepoints, np.concatenate((start_trend[ind], end_trend[ind])), kind='linear') for ind,_ in enumerate(tqdm(data))]

                # and save trend line
                trend_line = np.stack((trend_func[ind](np.arange(data.shape[-1])) for ind,_ in enumerate(tqdm(data))))

            # just make line across time series
            else:
                raise ValueError('index not provided')  
            
            # add mean image back to avoid distribution around 0
            data_detr = data - trend_line + np.mean(data, axis=-1)[..., np.newaxis]

        ## if we want remove basis set
        elif detrend_type == 'dct':

            frame_times = np.linspace(0, data.shape[-1]*TR, data.shape[-1], endpoint=False)
            dc_set = discrete_cosine_transform(high_pass=0.01, frame_times=frame_times)

            raise NameError('Not implemented yet')


        return data_detr # vertex, time



    def psc_epi(file, outdir):

        """ percent signal change file
        
        Parameters
        ----------
        file : str/list/array
            absolute filename to be psc (or list of filenames)
        outdir : str
            path to save new file

        Outputs
        -------
        out_file: str
            absolute output filename (or list of filenames)
        
        """
        
        # check if single filename or list of filenames
        
        if isinstance(file, list) or isinstance(file, np.ndarray): 
            file_list = file  
        else:
            file_list = [file]
        
        # store output filename in list
        outfiles = []
        
        # for each file, do the same
        for input_file in file_list:
            
            # get file extension
            file_extension = '.{b}'.format(b = input_file.rsplit('.', 2)[-1])

            # set output filename
            output_file = op.join(outdir, 
                        op.split(input_file)[-1].replace(file_extension,'_{name}{ext}'.format(name = 'psc',
                                                                                            ext = file_extension)))
            # if file already exists, skip
            if op.exists(output_file): 
                print('already exists, skipping %s'%output_file)
            
            else:
                print('making %s'%output_file)
                
                data = np.load(input_file, allow_pickle=True)
                
                mean_signal = data.mean(axis = -1)[..., np.newaxis]
                data_psc = (data - mean_signal)/np.absolute(mean_signal)
                data_psc *= 100
                    
                ## save psc file
                np.save(output_file, data_psc)

            # append out files
            outfiles.append(output_file)
            
        # if input file was not list, then return output that is also not list
        if not isinstance(file, list) and not isinstance(file, np.ndarray): 
            outfiles = outfiles[0] 
        
        return outfiles


    def average_epi(file, outdir, method = 'mean'):

        """ average epi files
        
        Parameters
        ----------
        file : list/array
            list of absolute filename to be averaged
        outdir : str
            path to save new file
        method: str
            if mean or median
        Outputs
        -------
        output_file: str
            absolute output filename (or list of filenames)
        
        """
        
        # check if single filename or list of filenames
        if not isinstance(file, list) and not isinstance(file, np.ndarray): 
            raise NameError('List of files not provided')
            
        file_list = file

        # set output filename
        output_file = op.join(outdir, re.sub('run-\d{1}_','run-{mtd}_'.format(mtd = method), op.split(file_list[0])[-1]))

        # if file already exists, skip
        if op.exists(output_file): 
            print('already exists, skipping %s'%output_file)
        
        else:
            print('making %s'%output_file)

            # store all run data in list, to average later
            all_runs = []

            # for each file, do the same
            for i, input_file in enumerate(file_list):
                
                print('loading %s'%input_file)
                
                data = np.load(input_file,allow_pickle=True)

                all_runs.append(data)
            
            # average all
            if method == 'median':
                avg_data = np.nanmedian(all_runs, axis = 0)
                
            elif method == 'mean':
                avg_data = np.nanmean(all_runs, axis = 0)
                
            # actually save
            np.save(output_file, avg_data)


        return output_file


    def reorient_nii_2RAS(input_pth, output_pth):

        """
        Reorient niftis to RAS
        (useful for anat files after dcm2niix)

        Parameters
        ----------
        input_pth: str
            path to look for files, ex: root/anat_preprocessing/sub-X/ses-1/anat folder
        output_pth: str
            path to save original files, ex: root/orig_anat/sub-X folder

        """

        ## set output path where we want to store original files
        if not op.isdir(output_pth):
            os.makedirs(output_pth)

        # list of original niftis
        orig_nii_files = [op.join(input_pth, val) for val in os.listdir(input_pth) if val.endswith('.nii.gz')]

        # then for each file
        for file in orig_nii_files:

            # copy the original to the new folder
            ogfile = op.join(output_pth, op.split(file)[-1])

            if op.exists(ogfile):
                print('already exists %s'%ogfile)
            else:
                copy2(file, ogfile)
                print('file copied to %s'%ogfile)

            # reorient all files to RAS+ (used by nibabel & fMRIprep) 
            orig_img = nib.load(file)
            orig_img_hdr = orig_img.header

            qform = orig_img_hdr['qform_code'] # set qform code to original

            canonical_img = nib.as_closest_canonical(orig_img)

            if qform != 0:
                canonical_img.header['qform_code'] = np.array([qform], dtype=np.int16)
            else:
                # set to 1 if original qform code = 0
                canonical_img.header['qform_code'] = np.array([1], dtype=np.int16)

            nib.save(canonical_img, file)


    def convert64bit_to_16bit(input_file, output_file):

        """
        Convert niftis from 64-bit (float) to 16-bit (int)
        (useful for nifits obtained from parrec2nii)

        Parameters
        ----------
        input_file: str
            absolute filename of original file
        output_file: str
            absolute filename of new file
        """

        # load nifti image
        image_nii = nib.load(input_file)

        image_nii_hdr = image_nii.header
        qform = image_nii_hdr['qform_code'] # set qform code to original

        ## try saving and int16 but preserving scaling
        new_image = nib.Nifti1Image(image_nii.dataobj.get_unscaled(), image_nii.affine, image_nii.header)
        new_image.header['scl_inter'] = 0
        new_image.set_data_dtype(np.int16)

        if qform != 0:
            new_image.header['qform_code'] = np.array([qform], dtype=np.int16)
        else:
            # set to 1 if original qform code = 0
            new_image.header['qform_code'] = np.array([1], dtype=np.int16)

        # save in same dir
        nib.save(new_image, output_file)


    def convert_parrec2nii(input_dir, output_dir, cmd = None):

        """
        convert PAR/REC to nifti

        Parameters
        ----------
        input_dir: str
            absolute path to input folder (where we have the parrecs)
        output_dir: str
            absolute path to output folder (where we want to store the niftis)
        cmd: str or None
            if not specified, command will be dcm2niix version from conda environment. But we can also specify the specific dcm2niix version
            by giving the path where it's installed (in terminal write which dcm2niix and will output path [e.g: /Users/verissimo/anaconda3/bin/dcm2niix])
        """

        if cmd is None:
            cmd = 'dcm2niix'

        cmd_txt = "{cmd} -d 0 -b y -f %n_%p -o {out} -z y {in_folder}".format(cmd = cmd, 
                                                                            out = output_dir, 
                                                                            in_folder = input_dir)

        # convert files
        print("Converting files to nifti-format")
        os.system(cmd_txt)


    def crop_shift_arr(arr, crop_nr = None, shift = 0):
        
        """
        helper function to crop and shift array
        
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
        

    def surf_data_from_cifti(data, axis, surf_name, medial_struct=False):

        """
        load surface data from cifti, from one hemisphere
        taken from https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb
        """

        assert isinstance(axis, nib.cifti2.BrainModelAxis)
        for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
            if name == surf_name:                                 # Just looking for a surface
                data = data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
                vtx_indices = model.vertex                        # Generally 1-N, except medial wall vertices
                if medial_struct:
                    surf_data = np.zeros((len(vtx_indices), data.shape[-1]), dtype=data.dtype)
                else:
                    surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
                surf_data[vtx_indices] = data
                return surf_data


    def load_data_save_npz(file, outdir, save_subcortical=False):
        
        """ load data file, be it nifti, gifti or cifti
        and save as npz - (whole brain: ("vertex", TR))
        
        Parameters
        ----------
        file : str/list/array
            absolute filename of ciftis to be decomposed (or list of filenames)
        outdir : str
            path to save new files
        save_subcortical: bool
            if we also want to save subcortical structures
        
        Outputs
        -------
        out_file: str
            absolute output filename (or list of filenames)
            
        """
        
        # some params
        hemispheres = ['hemi-L','hemi-R']
        cifti_hemis = {'hemi-L': 'CIFTI_STRUCTURE_CORTEX_LEFT', 
                    'hemi-R': 'CIFTI_STRUCTURE_CORTEX_RIGHT'}
        subcortical_hemis = ['BRAIN_STEM', 'ACCUMBENS', 'AMYGDALA', 'CAUDATE', 'CEREBELLUM', 
                            'DIENCEPHALON_VENTRAL', 'HIPPOCAMPUS', 'PALLIDUM', 'PUTAMEN', 'THALAMUS']

        # make sub-folder to save other files
        if save_subcortical:
            subcort_dir = op.join(outdir, 'subcortical')
            if not op.isdir(subcort_dir): 
                os.makedirs(subcort_dir)
            
        # check if single filename or list of filenames
        if isinstance(file, list) or isinstance(file, np.ndarray): 
            file_list = file  
        else:
            file_list = [file]
            
        # store output filename in list
        outfiles = []
        
        # for each file, do the same
        for input_file in file_list:
            
            # get file extension
            file_extension = '.{a}.{b}'.format(a = input_file.rsplit('.', 2)[-2],
                                    b = input_file.rsplit('.', 2)[-1])
            
            # set output filename
            output_file = op.join(outdir, 
                        op.split(input_file)[-1].replace(file_extension,'_{name}{ext}'.format(name = input_file.rsplit('.', 2)[-2],
                                                                                            ext = '.npy')))

            # if file already exists, skip
            if op.exists(output_file): 
                print('already exists, skipping %s'%output_file)

            else:
                print('making %s'%output_file)
                        
                if file_extension == '.dtseries.nii': # load cifti file

                    cifti = nib.load(input_file)
                    cifti_data = cifti.get_fdata(dtype=np.float32) # volume array (time, "voxels") 
                    cifti_hdr = cifti.header
                    axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]

                    # load data, per hemisphere
                    # note that surface data is (time, "vertex")
                    data = np.vstack([surf_data_from_cifti(cifti_data, axes[1], cifti_hemis[hemi]) for hemi in hemispheres])

                    if save_subcortical:
                        print('also saving subcortical structures in separate folder')
                        subcort_dict = {}

                        #for name in subcortical_hemis:
                        for name,_,_ in axes[-1].iter_structures():
                            if 'CORTEX' not in name:
                                print('saving data for %s'%name)
                                subcort_dict[name] = surf_data_from_cifti(cifti_data, axes[1], name, medial_struct=True)
                        
                        # save dict in folder
                        np.savez(op.join(subcort_dir, op.split(output_file)[-1].replace('_bold_','_struct-subcortical_bold_').replace('.npy','.npz')), **subcort_dict)
                
                elif file_extension == '.func.gii': # load gifti file

                    bold_gii = nib.load(input_file)
                    # Each time point is an individual data array with intent NIFTI_INTENT_TIME_SERIES. agg_data() will aggregate these into a single array
                    data = bold_gii.agg_data('time series')
                
                elif file_extension == '.nii.gz': # load nifti file
                    img = nib.load(input_file)
                    #affine = img.affine
                    #header = img.header
                    data = img.get_fdata()
                    
                # actually save
                np.save(output_file, data)

            # append out files
            outfiles.append(output_file)

        # if input file was not list, then return output that is also not list
        if not isinstance(file, list): 
            outfiles = outfiles[0] 

        return outfiles



    def plot_DM(DM, vertex, output, names=['intercept','ACAO', 'ACUO', 'UCAO', 'UCUO'], save_fig = False):
        
        """ plot design matrix for a given vertex
        similar to nilearn dm plotting func
        
        Parameters
        ----------
        DM : array
            design matrix with shape (vertices, time, regressors)
        vertex : int
            vertex
        output: str
            absolute output filename
            
        """
        X = DM[vertex]
        
        max_len = np.max([len(str(name)) for name in names])
        
        fig_height = 1 + .1 * X.shape[0] + .04 * max_len
        if fig_height < 3:
            fig_height = 3
        elif fig_height > 10:
            fig_height = 10
        
        plt.figure(figsize=(1 + .5 * len(names), fig_height))
        ax = plt.subplot(1, 1, 1)

        ax.imshow(X, interpolation='nearest', aspect='auto')
        ax.set_label('conditions')
        ax.set_ylabel('scan number')

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha='left')
        # Set ticks above, to have a display more similar to the display of a
        # corresponding dataframe
        ax.xaxis.tick_top()

        plt.tight_layout()

        if save_fig: 
            print('saving %s'%output)
            plt.savefig(output)

    
    def fit_glm(voxel, dm, error='mse'):
        
        """ GLM fit on timeseries
        Regress a created design matrix on the input_data.

        Parameters
        ----------
        voxel : arr
            timeseries of a single voxel
        dm : arr
            DM array (#TR,#regressors)
        

        Outputs
        -------
        prediction : arr
            model fit for voxel
        betas : arr
            betas for model
        r2 : arr
            coefficient of determination
        mse/rss : arr
            mean of the squared residuals/residual sum of squares
        
        """

        if np.isnan(voxel).any() or np.isnan(dm).any():
            betas = np.repeat(np.nan, dm.shape[-1])
            prediction = np.repeat(np.nan, dm.shape[0])
            mse = np.nan
            r2 = np.nan

        else:   # if not nan (some vertices might have nan values)
            betas = np.linalg.lstsq(dm, voxel, rcond = None)[0]
            prediction = dm.dot(betas)

            mse = np.mean((voxel - prediction) ** 2) # calculate mean of squared residuals
            rss =  np.sum((voxel - prediction) ** 2) # calculate residual sum of squared errors

            r2 = 1 - (np.sum((voxel - prediction)**2)/ np.sum((voxel - np.mean(voxel))**2))  # and the rsq

        if error == 'mse': 
            return prediction, betas, r2, mse
        else:
            return prediction, betas, r2, rss 


    def set_contrast(dm_col,tasks,contrast_val=[1],num_cond=1):
        
        """ define contrast matrix

        Parameters
        ----------
        dm_col : list/arr
            design matrix columns (all possible task names in list)
        tasks : list/arr
            list with list of tasks to give contrast value
            if num_cond=1 : [tasks]
            if num_cond=2 : [tasks1,tasks2], contrast will be tasks1 - tasks2     
        contrast_val : list/arr 
            list with values for contrast
            if num_cond=1 : [value]
            if num_cond=2 : [value1,value2], contrast will be tasks1 - tasks2
        num_cond : int
            if one task vs the implicit baseline (1), or if comparing 2 conditions (2)

        Outputs
        -------
        contrast : list/arr
            contrast array

        """
        
        contrast = np.zeros(len(dm_col))

        if num_cond == 1: # if only one contrast value to give ("task vs implicit intercept")

            for j,name in enumerate(tasks[0]):
                for i in range(len(contrast)):
                    if dm_col[i] == name:
                        contrast[i] = contrast_val[0]

        elif num_cond == 2: # if comparing 2 conditions (task1 - task2)

            for k,lbl in enumerate(tasks):
                idx = []
                for i,val in enumerate(lbl):
                    idx.extend(np.where([1 if val == label else 0 for _,label in enumerate(dm_col)])[0])

                val = contrast_val[0] if k==0 else contrast_val[1] # value to give contrast

                for j in range(len(idx)):
                    for i in range(len(dm_col)):
                        if i==idx[j]:
                            contrast[i]=val

        print('contrast for %s is %s'%(tasks,contrast))
        return contrast


    def design_variance(X, which_predictor=1):
            
            ''' Returns the design variance of a predictor (or contrast) in X.
            
            Parameters
            ----------
            X : numpy array
                Array of shape (N, P)
            which_predictor : int or list/array
                The index of the predictor you want the design var from.
                Note that 0 refers to the intercept!
                Alternatively, "which_predictor" can be a contrast-vector
                
            Outputs
            -------
            des_var : float
                Design variance of the specified predictor/contrast from X.
            '''
        
            is_single = isinstance(which_predictor, int)
            if is_single:
                idx = which_predictor
            else:
                idx = np.array(which_predictor) != 0

            if np.isnan(X).any():
                des_var = np.nan
            else:
                c = np.zeros(X.shape[1])
                c[idx] = 1 if is_single == 1 else which_predictor[idx]
                des_var = c.dot(np.linalg.pinv(X.T.dot(X))).dot(c.T)
            
            return des_var

    def compute_stats(voxel, dm, contrast, betas, pvalue = 'oneside'):
        
        """ compute statistis for GLM

        Parameters
        ----------
        voxel : arr
            timeseries of a single voxel
        dm : arr
            DM array (#TR,#regressors)
        contrast: arr
            contrast vector
        betas : arr
            betas for model at that voxel
        pvalue : str
            type of tail for p-value - 'oneside'/'twoside'

        Outputs
        -------
        t_val : float
            t-statistic for that voxel relative to contrast
        p_val : float
            p-value for that voxel relative to contrast
        z_score : float
            z-score for that voxel relative to contrast
        
        """

        if np.isnan(voxel).any() or np.isnan(dm).any():
            t_val = np.nan
            p_val = np.nan
            z_score = np.nan

        else:   # if not nan (some vertices might have nan values)
            
            # calculate design variance
            design_var = design_variance(dm, contrast)
            
            # sum of squared errors
            sse = ((voxel - (dm.dot(betas))) ** 2).sum() 
            
            #degrees of freedom = N - P = timepoints - predictores
            df = (dm.shape[0] - dm.shape[1])
            
            # t statistic for vertex
            t_val = contrast.dot(betas) / np.sqrt((sse/df) * design_var)

            if pvalue == 'oneside': 
                # compute the p-value (right-tailed)
                p_val = t.sf(t_val, df) 

                # z-score corresponding to certain p-value
                z_score = norm.isf(np.clip(p_val, 1.e-300, 1. - 1.e-16)) # deal with inf values of scipy

            elif pvalue == 'twoside':
                # take the absolute by np.abs(t)
                p_val = t.sf(np.abs(t_val), df) * 2 # multiply by two to create a two-tailed p-value

                # z-score corresponding to certain p-value
                z_score = norm.isf(np.clip(p_val/2, 1.e-300, 1. - 1.e-16)) # deal with inf values of scipy

        return t_val,p_val,z_score


    def leave_one_out(input_list):

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


    def split_half_comb(input_list):

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


    def resample_arr(upsample_data, osf = 10, final_sf = 1.6):

        """ resample array
        using cubic interpolation
        
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


    def CV_FA(voxel,dm,betas):
        
        """ Use betas and DM to create prediction
        and get CV rsq of left out run
        
        Parameters
        ----------
        voxel : arr
            timeseries of a single voxel
        dm : arr
            DM array (#TR,#regressors)
        betas: arr
            betas array (#regressors)
        
        Outputs
        -------
        prediction : arr
            model fit for voxel
        cv_r2 : arr
            coefficient of determination
        
        """
        if np.isnan(voxel).any() or np.isnan(dm).any():
            
            prediction = np.repeat(np.nan, dm.shape[0])
            cv_r2 = np.nan
        
        else:
            prediction = dm.dot(betas)

            #calculate CV-rsq        
            cv_r2 = np.nan_to_num(1-np.sum((voxel-prediction)**2, axis=-1)/(voxel.shape[0]*voxel.var(0)))

        return prediction, cv_r2


    def select_confounds(file, outdir, reg_names = ['a_comp_cor','cosine','framewise_displacement'],
                        CumulativeVarianceExplained = 0.4, num_TR_crop = 5, 
                        select = 'num', num_components = 5):
        
        """ 
        function to subselect relevant nuisance regressors
        from fmriprep confounds output tsv
        and save them in new tsv in outdir
        
        Parameters
        ----------
        file : str/list/array
            absolute filename of original confounds.tsv (or list of filenames)
        outdir : str
            path to save new file
        reg_names : list
            list with (sub-)strings of the nuisance regressor names
        select: str
            selection factor ('num' - select x number of components, 'cve' - use CVE as threshold)
        CumulativeVarianceExplained: float
            value of CVE up to which a_comp_cor components are selected
        num_components: int
            number of components to select
        num_TR_crop: int
            if we are cropping bold file TRs, also need to do it with confounds, to match
            
        Outputs
        -------
        out_file: str
            absolute output filename (or list of filenames)
        
        """
        
        # check if single filename or list of filenames
        if isinstance(file, list) or isinstance(file, np.ndarray): 
            file_list = file  
        else:
            file_list = [file]
        
        # store output filename in list
        outfiles = []
        
        # for each file, do the same
        for input_file in file_list:
            
            # get file extension
            file_extension = '.{a}'.format(a = input_file.rsplit('.')[-1])
            
            # set output filename
            output_file = op.join(outdir, 
                        op.split(input_file)[-1].replace(file_extension,'_{name}{ext}'.format(name = 'select_cropped',
                                                                                            ext = file_extension)))
                
            # if file already exists, skip
            if op.exists(output_file): 
                print('already exists, skipping %s'%output_file)
            
            else:
                print('making %s'%output_file)
                
                # load confounds tsv
                data = pd.read_csv(input_file, sep="\t")
                
                # get names of nuisances regressors of interest
                nuisance_columns = []

                for reg in reg_names:

                    if reg == 'a_comp_cor': # in this case, subselect a few, according to jason file

                        # Opening JSON file
                        f = open(input_file.replace(file_extension,'.json'))
                        reg_info = json.load(f)
                        reg_info = pd.DataFrame.from_dict(reg_info)
                        f.close

                        reg_info = reg_info.filter(regex=reg)

                        if select == 'num': 
                            cut_off_ind = num_components
                        elif select == 'cve':  
                            # cut off index
                            cut_off_ind = np.where(reg_info.iloc[reg_info.index=='CumulativeVarianceExplained'].values[0] > CumulativeVarianceExplained)[0][0] 

                        nuisance_columns += list(reg_info.columns[0:cut_off_ind])

                    else:
                        nuisance_columns += [col for col in data.columns if reg in col]
                    
                # get final df
                filtered_data = data[nuisance_columns]
                # dont forget to crop first TRs
                crop_data = filtered_data.iloc[num_TR_crop:] 
                
                print('nuisance dataframe with %i TRs and %i columns: %s' %(len(crop_data),len(crop_data.columns),str(list(crop_data.columns))))

                # saving as tsv file
                crop_data.to_csv(output_file, sep="\t", index = False)
                
            # append out files
            outfiles.append(output_file)
            
        # if input file was not list, then return output that is also not list
        if not isinstance(file, list) and not isinstance(file, np.ndarray): 
            outfiles = outfiles[0] 

        return outfiles


    def regressOUT_confounds(file, counfounds, outdir, TR=1.6, plot_vert = False, 
                                            detrend = True, standardize = 'psc', standardize_confounds = True):
        
        """ 
        regress out confounds from data
        
        Parameters
        ----------
        file : str/list/array
            absolute filename to be filtered (or list of filenames)
        counfounds : str/list/array
            absolute filename of confounds tsv (or list of filenames)
        outdir : str
            path to save new file
        detrend: bool
            input for nilearn signal clean, Whether to detrend signals or not
        standardize: str or False
            input for nilearn signal clean, Strategy to standardize the signal (if False wont do it)
        standardize_confounds: bool
            input for nilearn signal clean, If set to True, the confounds are z-scored
        
        Outputs
        -------
        out_file: str
            absolute output filename (or list of filenames)
        """ 

        # check if single filename or list of filenames
        
        if isinstance(file, list) or isinstance(file, np.ndarray): 
            file_list = file  
        else:
            file_list = [file]
        
        # store output filename in list
        outfiles = []
        
        # for each file, do the same
        for input_file in file_list:
            
            # get file extension
            file_extension = '.{b}'.format(b = input_file.rsplit('.', 2)[-1])

            # if we are also standardizing, then add that to name
            if isinstance(standardize, str):
                stand_name = '_{s}'.format(s = standardize)
            else:
                stand_name = ''
                standardize = False

            # set output filename
            output_file = op.join(outdir, 
                        op.split(input_file)[-1].replace(file_extension,'_{c}{s}{ext}'.format(c = 'confound',
                                                                                            s = stand_name,
                                                                                            ext = file_extension)))
            # if file already exists, skip
            if op.exists(output_file): 
                print('already exists, skipping %s'%output_file)
            
            else:
                print('making %s'%output_file)
                
                data = np.load(input_file, allow_pickle = True)
                
                # get confound file for that run
                run_id = re.search(r'run-._', input_file).group(0)
                conf_file = [val for val in counfounds if run_id in val][0]

                print('using confounds from %s'%conf_file)

                # load dataframe
                conf_df = pd.read_csv(conf_file, sep="\t")
                
                # clean confounds from data
                filtered_signal = clean(signals = data.T, confounds = conf_df.values, detrend = detrend, 
                                standardize = standardize, standardize_confounds = standardize_confounds, filter = False, t_r = TR)
                
                data_filt = filtered_signal.T
                
                ## save filtered file
                np.save(output_file,data_filt)

                ## if we want to compare a high tSNR voxel before and after filtering
                if plot_vert:
                    tsnr = np.mean(data, axis = -1)/np.std(data, axis = -1)
                    tsnr[np.where(np.isinf(tsnr))] = np.nan

                    # psc original data, so they are in same scale
                    mean_signal = data.mean(axis = -1)[..., np.newaxis]
                    data_psc = (data - mean_signal)/np.absolute(mean_signal)
                    data_psc *= 100
                    
                    ind2plot = np.where(tsnr == np.nanmax(tsnr))[0][0]
                    fig = plt.figure()
                    plt.plot(data_psc[ind2plot,...], color='dimgray',label='Original data')
                    plt.plot(data_filt[ind2plot,...], color='mediumseagreen',label='Filtered data')

                    plt.xlabel('Time (TR)')
                    plt.ylabel('Signal amplitude (a.u.)')
                    plt.legend(loc = 'upper right')

                    fig.savefig(output_file.replace(file_extension,'_vertex_%i.png'%ind2plot))

            # append out files
            outfiles.append(output_file)
            
        # if input file was not list, then return output that is also not list
        if not isinstance(file, list) and not isinstance(file, np.ndarray): 
            outfiles = outfiles[0] 
        
        return outfiles


    def get_ecc_limits(visual_dm, params, screen_size_deg = [11,11]):
        
        """
        Given a DM and the pRF bar directions
        get ecc limits of visual stimulation
        (in degrees)
        
        Parameters
        ----------
        visual_dm : array
        design matrix for pRF task [x,y,t]
        params : yml dict
            with experiment params
        screen_size_deg : list/array
            size of screen (width, height) in degrees
        
        """
        
        # number TRs per condition
        TR_conditions = {'L-R': params['prf']['num_TRs']['L-R'],
                        'R-L': params['prf']['num_TRs']['R-L'],
                        'U-D': params['prf']['num_TRs']['U-D'],
                        'D-U': params['prf']['num_TRs']['D-U'],
                        'empty': params['prf']['num_TRs']['empty'],
                        'empty_long': params['prf']['num_TRs']['empty_long']}

        # order of conditions in run
        bar_pass_direction = params['prf']['bar_pass_direction']

        # list of bar orientation at all TRs
        condition_per_TR = []
        for _,bartype in enumerate(bar_pass_direction):
            condition_per_TR = np.concatenate((condition_per_TR,np.tile(bartype, TR_conditions[bartype])))

        if params['prf']['crop']:
            condition_per_TR = condition_per_TR[params['prf']['crop_TR']:]
            
        ## get ecc limits of mask 
        # not best aproach, only considers square/rectangular mask

        x_mask = np.zeros(visual_dm[...,0].shape)
        y_mask = np.zeros(visual_dm[...,0].shape)

        for i,cond in enumerate(condition_per_TR):

            if cond in ['L-R','R-L']:
                x_mask += visual_dm[...,i]
            elif cond in ['U-D','D-U']:
                y_mask += visual_dm[...,i]

        x_mask = np.clip(x_mask,0,1)
        y_mask = np.clip(y_mask,0,1)
        
        ## get y ecc limits
        y_ecc_limit = [np.clip(np.max(y_mask.nonzero()[0])+1,0,visual_dm.shape[1])*screen_size_deg[1]/visual_dm.shape[1],
                        np.clip(np.min(y_mask.nonzero()[0])-1,0,visual_dm.shape[1])*screen_size_deg[1]/visual_dm.shape[1]]

        y_ecc_limit = screen_size_deg[1]/2 - y_ecc_limit

        ## get x ecc limits
        x_ecc_limit = [np.clip(np.max(x_mask.nonzero()[1])+1,0,visual_dm.shape[0])*screen_size_deg[0]/visual_dm.shape[0],
                    np.clip(np.min(x_mask.nonzero()[1])-1,0,visual_dm.shape[0])*screen_size_deg[0]/visual_dm.shape[0]]

        x_ecc_limit = screen_size_deg[0]/2 - x_ecc_limit

        return x_ecc_limit, y_ecc_limit


    def weight_dm(bar_dm_list, weights_array):

        gain_dm = bar_dm_list * weights_array[:,None,None,None]
        gain_dm = np.max(gain_dm, axis=0) 
        
        return gain_dm





    def get_FA_regressor(fa_dm, params, pRFfit_pars, filter = True, stim_ind = [], 
                        TR = 1.6, hrf_params = [1,1,0], oversampling_time = 1, pad_length=20,
                        crop_unit = 'sec', crop = True, crop_TR = 3, shift_TRs = True, shift_TR_num = 1.5):
        
        """ Get timecourse for FA regressor, given a dm
        
        Parameters
        ----------
        fa_dm: array
            fa design matrix N x samples (N = (x,y))
        params: yml dict
            with experiment params
        pRFfit_pars: dict
            Dictionary with parameter key, value from lmfit
        oversampling_time: int
            value that FA dm is oversampled by, to then downsample predictor
        
        """
            
        ## set hrf
        hrf = create_hrf(hrf_params = hrf_params, TR = TR, osf = oversampling_time)

        ## make screen  grid
        screen_size_degrees = 2.0 * \
        np.degrees(np.arctan(params['monitor']['height'] /
                            (2.0*params['monitor']['distance'])))

        oneD_grid = np.linspace(-screen_size_degrees/2, screen_size_degrees/2, fa_dm.shape[0], endpoint=True)
        x_coordinates,y_coordinates = np.meshgrid(oneD_grid, oneD_grid)

        # create the single rf
        rf = np.rot90(gauss2D_iso_cart(x = x_coordinates[..., np.newaxis],
                            y = y_coordinates[..., np.newaxis],
                            mu = (pRFfit_pars['pRF_x'], pRFfit_pars['pRF_y']),
                            sigma = pRFfit_pars['pRF_size'],
                            normalize_RFs = False).T, axes=(1,2))

        if not 'pRF_n' in pRFfit_pars.keys(): # accounts for gauss or css model
            pRFfit_pars['pRF_n'] = 1

        tc = stimulus_through_prf(rf, fa_dm, 1)**pRFfit_pars['pRF_n']

        # get oversampled (if not done already) indices where bar on screen
        # and upsample dm
    
        if len(stim_ind)>0:
            osf_stim_ind = get_oversampled_ind(stim_ind, osf = oversampling_time)
            up_samp_tc = np.zeros((tc.shape[0], tc.shape[1]*oversampling_time))
            up_samp_tc[...,osf_stim_ind] = np.repeat(tc[...,stim_ind], oversampling_time, axis = -1)
            neural_tc = up_samp_tc 
        else:
            neural_tc = tc 

        # in case we want to crop the beginning of the DM
        if crop == True:
            if crop_unit == 'sec': # fix for the fact that I crop TRs, but task not synced to TR
                neural_tc = neural_tc[...,int(crop_TR*TR*oversampling_time)::] 
            else: # assumes unit is TR
                neural_tc = neural_tc[...,crop_TR*oversampling_time::]
        
        # shifting TRs to the left (quick fix)
        # to account for first trigger that was "dummy" - in future change experiment settings to skip 1st TR
        if shift_TRs == True:
            up_neural_tc = neural_tc.copy()
            if crop_unit == 'sec': # fix for the fact that I shift TRs, but task not synced to TR
                up_neural_tc[...,:-int(shift_TR_num*TR*oversampling_time)] = neural_tc[...,int(shift_TR_num*TR*oversampling_time):]
            else: # assumes unit is TR
                up_neural_tc[...,:-int(shift_TR_num*oversampling_time)] = neural_tc[...,int(shift_TR_num*oversampling_time):]

            neural_tc = up_neural_tc.copy()

        # convolve with hrf
        #scipy fftconvolve does not have padding options so doing it manually
        pad = np.tile(neural_tc[:,0], (pad_length*oversampling_time,1)).T
        padded_cue = np.hstack((pad,neural_tc))

        tc_convolved = fftconvolve(padded_cue, hrf, axes=(-1))[..., pad_length*oversampling_time:neural_tc.shape[-1]+pad_length*oversampling_time]  

        # save convolved upsampled cue in array
        tc_convolved = np.array(tc_convolved)

        if filter:
            model_fit = pRFfit_pars['pRF_baseline'] + pRFfit_pars['pRF_beta'] * filter_predictions(
                                                                                                tc_convolved,
                                                                                                filter_type = params['mri']['filtering']['type'],
                                                                                                filter_params = {'highpass': params['mri']['filtering']['highpass'],
                                                                                                            'add_mean': params['mri']['filtering']['add_mean'],
                                                                                                            'window_length': params['mri']['filtering']['window_length'],
                                                                                                            'polyorder': params['mri']['filtering']['polyorder']})
        else:
            model_fit = pRFfit_pars['pRF_baseline'] + pRFfit_pars['pRF_beta'] * tc_convolved

        # resample to data sampling rate
        FA_regressor = resample_arr(model_fit, osf = oversampling_time, final_sf = TR)
        
        # squeeze out single dimension
        FA_regressor = np.squeeze(FA_regressor)
        
        return FA_regressor
        


    def plot_FA_DM(output, bar_dm_dict, 
                bar_weights = {'ACAO': 1, 'ACUO': 1, 'UCAO': 1,'UCUO': 1}, oversampling_time = 10):
        
        """Plot FA DM
        
        Parameters
        ----------
        output : string
        absolute output name for numpy array
        bar_dm_dict: dict
            dictionary with the dm for each condition
        bar_weights: dict
            dictionary with weight values (gain) given to each bar
        oversampling_time: int
            value that cond dm is oversampled by, to then downsample when plotting
            (note that FA_DM saved will be oversampled, same as inputs)
        
        """
        
        ## set DM - all bars simultaneously on screen, multiplied by weights
        for i, cond in enumerate(bar_dm_dict.keys()):
            
            if i==0:
                weighted_dm = bar_dm_dict[cond][np.newaxis,...]*bar_weights[cond]
            else:
                weighted_dm = np.vstack((weighted_dm, 
                                    bar_dm_dict[cond][np.newaxis,...]*bar_weights[cond]))
        
        # taking the max value of the spatial position at each time point (to account for overlaps)
        weighted_dm = np.amax(weighted_dm, axis=0)
        
        # save array with DM
        np.save(output, weighted_dm)
        
        ## save frames as images
        #take into account oversampling
        if oversampling_time == 1:
            frames = np.arange(weighted_dm.shape[-1])
        else:
            frames = np.arange(0,weighted_dm.shape[-1],oversampling_time, dtype=int)  

        outfolder = op.split(output)[0]

        weighted_dm = weighted_dm.astype(np.uint8)

        for w in frames:
            im = Image.fromarray(weighted_dm[...,int(w)])
            im.save(op.join(outfolder,op.split(output)[-1].replace('.npy','_trial-{time}.png'.format(time=str(int(w/oversampling_time)).zfill(3))))) 

        print('saved dm in %s'%outfolder)



    def get_oversampled_ind(orig_ind, osf = 10):

        """Helper function to get oversampled indices
        for bookeeping
        """
        
        osf_ind = []
        
        for _,val in enumerate(orig_ind):
            osf_ind += list(np.arange(val*osf,val*osf+osf))
        
        return np.array(osf_ind)


    def get_fa_prediction_tc(dm, betas, 
                            timecourse = [], r2 = 0, viz_model = False, TR = 1.6, 
                            bar_onset = [27,98,126,197,225,296,324,395], crop_TR = 3, shift_TR_num = 1.5):
        
        
        prediction = dm.dot(betas)
        
        if viz_model:
            
            fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)
            # plot data with model
            time_sec = np.linspace(0,len(timecourse)*TR, num=len(timecourse)) # array with timepoints, in seconds

            plt.plot(time_sec, prediction, c='#0040ff',lw=3,label='model R$^2$ = %.2f'%r2,zorder=1)
            plt.plot(time_sec, timecourse,'k--',label='FA data')
            axis.set_xlabel('Time (s)',fontsize=20, labelpad=20)
            axis.set_ylabel('BOLD signal change (%)',fontsize=20, labelpad=10)
            axis.set_xlim(0,len(prediction)*TR)
            axis.legend(loc='upper left',fontsize=10) 
            #axis.set_ylim(-3,3)  

            # times where bar is on screen [1st on, last on, 1st on, last on, etc] 
            # ugly AF - change in future 
            bar_onset = np.array(bar_onset)#/TR
            bar_onset = bar_onset - crop_TR*TR - TR*shift_TR_num

            bar_directions = np.array(['mini_block_0', 'mini_block_1', 'mini_block_2', 'mini_block_3'])
            # plot axis vertical bar on background to indicate stimulus display time
            ax_count = 0
            for h in range(len(bar_directions)):
                plt.axvspan(bar_onset[ax_count], bar_onset[ax_count+1]+TR, facecolor='#0040ff', alpha=0.1)
                ax_count += 2

        
        return prediction



    def baseline_correction(data, condition_per_TR, num_baseline_TRs = 6, baseline_interval = 'empty_long', 
                            avg_type = 'median', only_edges = False, TR2task = 3):
        
        """Do baseline correction to timecourse
        Useful when we want a fix baseline during fitting

        Parameters
        ----------
        data : array
        2D array with data timecourses
        num_baseline_TRs: int
            number of baseline TRs to consider (will always be the last X TRs of the interval)
        baseline_interval : str
        name of the condition to get baseline values
        avg_type: str
            type of averaging done
        only_edges: bool
            if we want to only use the edges of the array to correct (this is start and end)
        TR2task: int
            number of TRs between end of baseline period and start of bar pass (i.e., shifts baseline period in time by X TRs) 
        """

        baseline_index = [i for i, val in enumerate(condition_per_TR) if str(val) == baseline_interval]
        interval_ind = []
        for i, ind in enumerate(baseline_index):
            if i > 0:
                if (ind - baseline_index[i-1]) > 1: ## transition point
                    interval_ind.append([baseline_index[i-1] - TR2task - num_baseline_TRs, baseline_index[i-1] - TR2task]) 

        if condition_per_TR[-1] == baseline_interval:
            interval_ind.append([baseline_index[-1] - TR2task - num_baseline_TRs, baseline_index[-1] - TR2task]) 
        
        if only_edges:
            interval_ind = np.array((interval_ind[0], interval_ind[-1]))

        # get baseline values
        baseline_arr = np.hstack([data[..., ind[0]:ind[1]] for ind in interval_ind])

        # average
        if avg_type == 'median':
            avg_baseline = np.median(baseline_arr, axis = -1)
        else:
            avg_baseline = np.mean(baseline_arr, axis = -1)

        return data - avg_baseline[...,np.newaxis]




    def get_event_onsets(behav_files, TR = 1.6, crop = True, crop_TR = 8):
        
        """ Get behavioral event onsets
        to use in design matrix for pRF task
        based on actual MRI pulses
        
        Parameters
        ----------
        behav_files : list/array
        list with absolute filenames for all pRF runs .tsv
        """
        
        avg_onset = []

        for r in range(len(behav_files)):

            # load df for run
            df_run = pd.read_csv(behav_files[r], sep='\t')
            # get onsets
            onset_run = df_run[df_run['event_type']=='pulse']['onset'].values

            # first pulse does not start at 0, correct for that
            onset = onset_run - onset_run[0]

            if r == 0:
                avg_onset = onset
            else:
                avg_onset = np.vstack((avg_onset, onset))

        # average accross runs
        avg_onset = np.mean(avg_onset, axis=0)
        
        # if we're cropping fMRI data, then also need to crop length of events array
        if crop: 
            avg_onset = avg_onset[crop_TR:] - avg_onset[crop_TR]
            
        return avg_onset


    def get_loo_filename(input_list, loo_key = 'loo_r1s1'):
        
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


    def get_run_ses_from_str(input_name):
        
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


    def get_bar_overlap_dm(bar_arr):
        
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


    def sum_bar_dms(stacked_dms, overlap_dm = None, overlap_weight = 1):

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


    def error_resid(timecourse, prediction, mean_err = False, return_array = False):

        """

        Helper function to get the residual error between a timecourse and a model prediction

        Parameters
        ----------
        timecourse : ndarray
            The actual, measured time-series against which the model is fit
        prediction : ndarray
            model prediction timecourse
        mean_err : bool
            if True, will calculate the mean of squared residulas instead of sum
        return_array: bool
            if True, then function returns residual array instead of scalar
        """

        if return_array:
            return (timecourse - prediction).flatten() # return error "timecourse", that will be used by minimize
        else:
            if mean_err:
                return np.mean((timecourse - prediction) ** 2)  # calculate mean of squared residuals

            else:
                return np.sum((timecourse - prediction) ** 2) # calculate residual sum of squared errors



    def calc_rsq(voxel, prediction):

        """"
        Calculate rsq of fit
        """

        return 1 - (np.sum((voxel - prediction)**2)/ np.sum((voxel - np.mean(voxel))**2))  