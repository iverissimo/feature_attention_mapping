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
from scipy.stats import t, norm

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



def import_fmriprep2pycortex(source_directory, sj, dataset=None, ses=None, acq=None):
    
    """Import a subject from fmriprep-output to pycortex
    
    Parameters
    ----------
    source_directory : string
       Local directory that contains both fmriprep and freesurfer subfolders 
    sj : string
        Fmriprep subject name (without "sub-")
    dataset : string
       If you have multiple fmriprep outputs from different datasets, use this attribute
       to add a prefix to every subject id ('ds01.01' rather than '01')
    ses : string, optional
       BIDS session that contains the anatomical data
    acq : string, optional
        If we intend to specific the acquisition of the T1w file (for naming purposes)
    """
    if dataset is not None:
        pycortex_sub = '{ds}.{sub}'.format(ds=dataset, sub=sj)
    else:
        pycortex_sub = '{sub}'.format(sub=sj)

    if pycortex_sub in cortex.database.db.subjects.keys():
        print('subject %s already in filestore, will not overwrite'%pycortex_sub)
    else:
        
        # import subject into pycortex database
        cortex.fmriprep.import_subj(subject = sj, source_dir = source_directory, 
                             session = ses, dataset = dataset, acq = acq)


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
    


def correlate_arrs(data1, data2, n_jobs = 4, weights=[]):
    
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
    
    """ 
    
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
                
    ## actually correlate
    correlations = np.array(Parallel(n_jobs=n_jobs)(delayed(np.corrcoef)(data1_arr[i], data2_arr[i]) for i in np.arange(data1_arr.shape[0])))[...,0,1]
            
    return correlations


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
                first_modes_to_remove = 5, **kwargs):
    
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
    image_nii = nib. load(input_file)

    ## try saving and int16 but preserving scaling
    new_image = nib.Nifti1Image(image_nii.dataobj.get_unscaled(), image_nii.affine, image_nii.header)
    new_image.header['scl_inter'] = 0
    new_image.set_data_dtype(np.int16)

    # save in same dir
    nib.save(new_image, output_file)


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
    


def load_and_mask_data(file, chunk_num = 1, total_chunks = 1):
    
    """ load data, split into chunk/slice and mask nan voxels
    used to create a 2D array for pRF fitting
    with only relevant voxels/vertices
    
    Parameters
    ----------
    file : str
        absolute filename of the data to be fitted
    chunk_num: int
        number of chunk for slicing
    total_chunks: int
        total amount of chunks, if 1 then returns orig data array size (no chunking)

    Outputs
    -------
    masked_data: arr
        (masked) data array
    not_nan_vox: list/arr
        voxel indices that were NOT masked out
    orig_shape: tuple
        shape of original data chunk/slice (for later reshaping)
    """
    
    # get file extension
    file_extension = '.{a}.{b}'.format(a = file.rsplit('.', 2)[-2],
                                       b = file.rsplit('.', 2)[-1])
    
    # load data array, if necessary convert to 2D (vertex, time)
    # and select only relevant chunk/slice

    if file_extension == '.func.gii':

        # load surface data
        data_all = np.array(surface.load_surf_data(file))

        # number of vertices of chunk
        num_vox_chunk = int(data_all.shape[0]/total_chunks) 

        # new data chunk to fit
        data = data_all[num_vox_chunk*(int(chunk_num)-1):num_vox_chunk*int(chunk_num),:]

        # store chunk shape, useful later
        orig_shape = data.shape

        print('fitting chunk %s/%d of data with shape %s'%(chunk_num,total_chunks,str(data.shape)))
        
    elif file_extension == '.nii.gz':
        
        print('not implemented')  

    # define non nan voxels for sanity check
    not_nan_vox = np.where(~np.isnan(data[...,0]))[0]
    print('masked data with shape %s'%(str(data[not_nan_vox].shape)))

    # mask data
    # to avoid errors in fitting (all nan batches) and make fitting faster
    masked_data = data[not_nan_vox]

    return masked_data, not_nan_vox, orig_shape





def save_estimates(filename, estimates, mask_indices, orig_shape = np.array([1974,220]), 
                    model_type = 'gauss', fit_hrf = False):
    
    """
    re-arrange estimates that were masked
    and save all in numpy file
    
    (only works for gii files, should generalize for nii and cifti also)
    
    Parameters
    ----------
    filename : str
        absolute filename of estimates to be saved
    estimates : arr
        2d estimates (datapoints,estimates)
    mask_indices : arr
        array with voxel indices that were NOT masked out
    orig_shape: tuple/arr
        original data shape 
    model_type: str
        model type used for fitting
        
    
    """ 
    final_estimates = np.zeros((orig_shape[0], estimates.shape[-1])); final_estimates[:] = np.nan

    counter = 0
    for _,ind in enumerate(mask_indices):
        final_estimates[ind] = estimates[counter]
        counter += 1
            
    if model_type == 'gauss':

        if fit_hrf:
            np.savez(filename,
                    x = final_estimates[..., 0],
                    y = final_estimates[..., 1],
                    size = final_estimates[..., 2],
                    betas = final_estimates[...,3],
                    baseline = final_estimates[..., 4],
                    hrf_derivative = final_estimates[..., 5],
                    hrf_dispersion = final_estimates[..., 6], 
                    r2 = final_estimates[..., 7])
        
        else:
            np.savez(filename,
                    x = final_estimates[..., 0],
                    y = final_estimates[..., 1],
                    size = final_estimates[..., 2],
                    betas = final_estimates[...,3],
                    baseline = final_estimates[..., 4],
                    r2 = final_estimates[..., 5])
    
    elif model_type == 'css':

        if fit_hrf:
            np.savez(filename,
                    x = final_estimates[..., 0],
                    y = final_estimates[..., 1],
                    size = final_estimates[..., 2],
                    betas = final_estimates[...,3],
                    baseline = final_estimates[..., 4],
                    ns = final_estimates[..., 5],
                    hrf_derivative = final_estimates[..., 6],
                    hrf_dispersion = final_estimates[..., 7], 
                    r2 = final_estimates[..., 8])
        
        else:
            np.savez(filename,
                    x = final_estimates[..., 0],
                    y = final_estimates[..., 1],
                    size = final_estimates[..., 2],
                    betas = final_estimates[...,3],
                    baseline = final_estimates[..., 4],
                    ns = final_estimates[..., 5],
                    r2 = final_estimates[..., 6])

    elif model_type == 'dn':

        if fit_hrf:
            np.savez(filename,
                    x = final_estimates[..., 0],
                    y = final_estimates[..., 1],
                    size = final_estimates[..., 2],
                    betas = final_estimates[...,3],
                    baseline = final_estimates[..., 4],
                    sa = final_estimates[..., 5],
                    ss = final_estimates[..., 6], 
                    nb = final_estimates[..., 7], 
                    sb = final_estimates[..., 8], 
                    hrf_derivative = final_estimates[..., 9],
                    hrf_dispersion = final_estimates[..., 10], 
                    r2 = final_estimates[..., 11])
        
        else:
            np.savez(filename,
                    x = final_estimates[..., 0],
                    y = final_estimates[..., 1],
                    size = final_estimates[..., 2],
                    betas = final_estimates[...,3],
                    baseline = final_estimates[..., 4],
                    sa = final_estimates[..., 5],
                    ss = final_estimates[..., 6], 
                    nb = final_estimates[..., 7], 
                    sb = final_estimates[..., 8], 
                    r2 = final_estimates[..., 9])
        

def combine_slices(file_list,outdir,num_slices=89, ax=2):
    
    """ High pass filter NIFTI run with gaussian kernel
    
    Parameters
    ----------
    file_list : list
        list of absolute filenames of all volumes to combine
    outdir : str
        path to save new file
    num_slices : int
        number of slices to combine
    ax: int
        which ax to stack slices
    
    Outputs
    -------
    out_file: str
        absolute output filename
    
    """
    

    for num in np.arange(num_slices):
        
        vol = [x for _,x in enumerate(file_list) if '_slice-{num}.nii.gz'.format(num = str(num).zfill(2)) in x]
        
        if len(vol)==0: # if empty
            raise NameError('Slice %s doesnt exist!'%str(num).zfill(2)) 
        
        else:
            nibber = nib.load(vol[0])
            data = np.array(nibber.dataobj)
            data = np.take(data, indices = num, axis=ax)
            
            if num == 0: # for first slice    
                outdata = data[np.newaxis,...] 
            else:
                outdata = np.vstack((outdata,data[np.newaxis,...] ))
                
    outdata = np.moveaxis(outdata,0,ax)
    
    out_file = op.split(vol[0])[-1].replace('_slice-{num}.nii.gz'.format(num = str(num).zfill(2)),'.nii.gz')
    out_file = op.join(outdir,out_file)
    
    # Save estimates data
    new_img = nib.Nifti1Image(dataobj = outdata, affine = nibber.affine, header = nibber.header)
    new_img.to_filename(out_file)
    
    return out_file

  
def make_colormap(colormap = 'rainbow_r', bins = 256, add_alpha = True, invert_alpha = False, cmap_name = 'costum',
                      discrete = False, return_cmap = False):

    """ make custom colormap
    can add alpha channel to colormap,
    and save to pycortex filestore
    Parameters
    ----------
    colormap : str or List/arr
        if string then has to be a matplolib existent colormap
        if list/array then contains strings with color names, to create linear segmented cmap
    bins : int
        number of bins for colormap
    invert_alpha : bool
        if we want to invert direction of alpha channel
        (y can be from 0 to 1 or 1 to 0)
    cmap_name : str
        new cmap filename, final one will have _alpha_#-bins added to it
    discrete : bool
        if we want a discrete colormap or not (then will be continuous)
    Outputs
    -------
    rgb_fn : str
        absolute path to new colormap
    """
    
    if isinstance(colormap, str): # if input is string (so existent colormap)

        # get colormap
        cmap = cm.get_cmap(colormap)

    else: # is list of strings
        cvals  = np.arange(len(colormap))
        norm = plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(norm,cvals), colormap))
        cmap = colors.LinearSegmentedColormap.from_list("", tuples)
        
        if discrete == True: # if we want a discrete colormap from list
            cmap = colors.ListedColormap(colormap)
            bins = int(len(colormap))

    # convert into array
    cmap_array = cmap(range(bins))

    # reshape array for map
    new_map = []
    for i in range(cmap_array.shape[-1]):
        new_map.append(np.tile(cmap_array[...,i],(bins,1)))

    new_map = np.moveaxis(np.array(new_map), 0, -1)
    
    if add_alpha: 
        # make alpha array
        if invert_alpha == True: # in case we want to invert alpha (y from 1 to 0 instead pf 0 to 1)
            _, alpha = np.meshgrid(np.linspace(0, 1, bins, endpoint=False), 1-np.linspace(0, 1, bins))
        else:
            _, alpha = np.meshgrid(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins, endpoint=False))

        # add alpha channel
        new_map[...,-1] = alpha
        cmap_ext = (0,1,0,1)
    else:
        new_map = new_map[:1,...].copy() 
        cmap_ext = (0,100,0,1)
    
    fig = plt.figure(figsize=(1,1))
    ax = fig.add_axes([0,0,1,1])
    # plot 
    plt.imshow(new_map,
    extent = cmap_ext,
    origin = 'lower')
    ax.axis('off')

    if add_alpha: 
        rgb_fn = op.join(op.split(cortex.database.default_filestore)[
                          0], 'colormaps', cmap_name+'_alpha_bins_%d.png'%bins)
    else:
        rgb_fn = op.join(op.split(cortex.database.default_filestore)[
                          0], 'colormaps', cmap_name+'_bins_%d.png'%bins)
    #misc.imsave(rgb_fn, new_map)
    plt.savefig(rgb_fn, dpi = 200,transparent=True)

    if return_cmap:
        return cmap
    else:
        return rgb_fn 

def join_chunks(path, out_name, chunk_num = 83, fit_model = 'css', fit_hrf = False):
    """ combine all chunks into one single estimate numpy array
        assumes input is whole brain ("vertex", time)
    Parameters
    ----------
    path : str
        absolute path to files
    out_name: str
        absolute output name of combined estimates
    hemi : str
        'hemi_L' or 'hemi_R' hemisphere
    chunk_num : int
        total number of chunks to combine (per hemi)
    fit_model: str
        fit model of estimates
    
    Outputs
    -------
    estimates : npz 
        numpy array of estimates
    
    """
    
    for ch in range(chunk_num):
        
        chunk_name = [x for _,x in enumerate(os.listdir(path)) if fit_model in x and 'chunk-%s'%str(ch+1).zfill(3) in x][0]
        print('loading chunk %s'%chunk_name)
        chunk = np.load(op.join(path, chunk_name)) # load chunk
        
        if ch == 0:
            xx = chunk['x']
            yy = chunk['y']

            size = chunk['size']

            beta = chunk['betas']
            baseline = chunk['baseline']

            if 'css' in fit_model: 
                ns = chunk['ns']
            elif 'dn' in fit_model:
                sa = chunk['sa']
                ss = chunk['ss']
                nb = chunk['nb']
                sb = chunk['sb']

            rsq = chunk['r2']

            if fit_hrf:
                hrf_derivative = chunk['hrf_derivative']
                hrf_dispersion = chunk['hrf_dispersion']
            else: # assumes standard spm params
                hrf_derivative = np.ones(xx.shape)
                hrf_dispersion = np.zeros(xx.shape) 

        else:
            xx = np.concatenate((xx, chunk['x']))
            yy = np.concatenate((yy, chunk['y']))

            size = np.concatenate((size, chunk['size']))

            beta = np.concatenate((beta, chunk['betas']))
            baseline = np.concatenate((baseline, chunk['baseline']))

            if 'css' in fit_model:
                ns = np.concatenate((ns, chunk['ns']))
            elif 'dn' in fit_model:
                sa = np.concatenate((sa, chunk['sa']))
                ss = np.concatenate((ss, chunk['ss']))
                nb = np.concatenate((nb, chunk['nb']))
                sb = np.concatenate((sb, chunk['sb']))

            rsq = np.concatenate((rsq, chunk['r2']))
            
            if fit_hrf:
                hrf_derivative = np.concatenate((hrf_derivative, chunk['hrf_derivative']))
                hrf_dispersion = np.concatenate((hrf_dispersion, chunk['hrf_dispersion']))
            else: # assumes standard spm params
                hrf_derivative = np.concatenate((hrf_derivative, np.ones(xx.shape)))
                hrf_dispersion = np.concatenate((hrf_dispersion, np.zeros(xx.shape))) 
    
    print('shape of estimates is %s'%(str(xx.shape)))

    # save file
    output = op.join(out_name)
    print('saving %s'%output)

    if 'css' in fit_model:
        np.savez(output,
                x = xx,
                y = yy,
                size = size,
                betas = beta,
                baseline = baseline,
                ns = ns,
                hrf_derivative = hrf_derivative,
                hrf_dispersion = hrf_dispersion,
                r2 = rsq)

    elif 'dn' in fit_model:
        np.savez(output,
                x = xx,
                y = yy,
                size = size,
                betas = beta,
                baseline = baseline,
                sa = sa,
                ss = ss,
                nb = nb,
                sb = sb,
                hrf_derivative = hrf_derivative,
                hrf_dispersion = hrf_dispersion,
                r2 = rsq)
    
    else: # assumes gauss
        np.savez(output,
                x = xx,
                y = yy,
                size = size,
                betas = beta,
                baseline = baseline,
                hrf_derivative = hrf_derivative,
                hrf_dispersion = hrf_dispersion,
                r2 = rsq)
     
    return np.load(output)

def dva_per_pix(height_cm,distance_cm,vert_res_pix):

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


def mask_estimates(estimates, ROI = 'None', x_ecc_lim = [-6,6], y_ecc_lim = [-6,6],
                    rsq_threshold = .1, space = 'fsaverage', estimate_keys = ['x','y','size','betas','baseline','r2']):
    
    """ mask estimates, to be positive RF, within screen limits
    and for a certain ROI (if the case)
    Parameters
    ----------
    estimates : List/arr
        list of estimates.npz for both hemispheres
    ROI : str
        roi to mask estimates (eg. 'V1', default 'None')
    estimate_keys: list/arr
        list or array of strings with keys of estimates to mask
    
    Outputs
    -------
    masked_estimates : npz 
        numpy array of masked estimates
    
    """
    
    # make new variables that are masked 
    masked_dict = {}
    
    for k in estimate_keys: 
        masked_dict[k] = np.zeros(estimates[k].shape)
        masked_dict[k][:] = np.nan

    
    # set limits for xx and yy, forcing it to be within the screen boundaries
    # also for positive pRFs

    indices = np.where((~np.isnan(estimates['r2']))& \
                        (estimates['r2']>= rsq_threshold)& \
                       (estimates['x'] <= np.max(x_ecc_lim))& \
                      (estimates['x'] >= np.min(x_ecc_lim))& \
                      (estimates['y'] <= np.max(y_ecc_lim))& \
                       (estimates['y'] >= np.min(y_ecc_lim))& \
                       (estimates['betas']>=0)
                      )[0]
                        
    # save values
    for k in estimate_keys:
        masked_dict[k][indices] = estimates[k][indices]


    if ROI != 'None':
        
        roi_ind = cortex.get_roi_verts(space, ROI) # get indices for that ROI
        
        # mask for roi
        for k in estimate_keys:
            masked_dict[k] = masked_dict[k][roi_ind[ROI]]
    
    return masked_dict


def normalize(M):
    """
    normalize data array
    """
    return (M-np.nanmin(M))/(np.nanmax(M)-np.nanmin(M))


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


def get_cond_name(attend_cond, cond_type='UCUO', C = ['red','green'], O = ['vertical','horizontal']):
    
    """
    Get condition name for a certain key 
    given attended condition for miniblock
    
    """ 
    
    if cond_type == 'ACAO':
        
        cond_name = attend_cond
        
    elif cond_type == 'ACUO':
        
        cond_name = '{col}_{ori}'.format(col = attend_cond.split('_')[0],
                                        ori = O[O.index(attend_cond.split('_')[-1])-1])
        
    elif cond_type == 'UCAO':
        
        cond_name = '{col}_{ori}'.format(col = C[C.index(attend_cond.split('_')[0])-1],
                                        ori = attend_cond.split('_')[-1])
    
    elif cond_type == 'UCUO':
        
        cond_name = '{col}_{ori}'.format(col = C[C.index(attend_cond.split('_')[0])-1],
                                        ori = O[O.index(attend_cond.split('_')[-1])-1])
        
    return cond_name


def get_FA_bar_stim(bar_pos, trial_info, 
                    attend_cond = {'reg_name': 'ACAO_mblk-0',
                                   'color': True, 'orientation': True,
                                   'condition_name': 'red_vertical',
                                   'miniblock': 0}, xy_lim_pix = {'x_lim': [-540,540],'y_lim': [-540,540]},
                                   bar_width = .125, screen_res = [1080, 1080],
                                   TR = 1.6, crop_unit = 'sec',
                    res_scaling = 1, oversampling_time = None, stim_dur_seconds = 0.5,
                    crop = False, crop_TR = 3, shift_TRs=True, shift_TR_num = 1):
    
    """Get visual stim for FA condition.
    Similar to make_pRF_DM, it will
    save an array with the FA (un)attended
    bar position for the run
    
    Parameters
    ----------
    output : string
       absolute output name for DM
    params : yml dict
        with experiment params
    bar_pos: pd
        pandas dataframe with bar positions for the whole run
    trial_info: pd
        pandas dataframe with useful run info
    save_imgs : bool
       if we want to save images in folder, for sanity check
    """
    
    
    # general infos
    #bar_width = params['feature']['bar_width_ratio'] 

    #screen_res = params['window']['size']
    #if params['window']['display'] == 'square': # if square display
    #    screen_res = np.array([screen_res[1], screen_res[1]])            
    
    # miniblock number
    mini_blk_num = int(attend_cond['miniblock'])

    # if oversampling is None, then we're working with the trials
    if oversampling_time is None:
        osf = 1
    else:
        osf = oversampling_time
        
    # total number of TRs and of trials
    # if oversampling then they will not match
    total_trials = len(trial_info)
    total_TR = total_trials*osf

    # stimulus duration (how long was each bar on screen)
    stim_dur_TR = int(np.ceil(stim_dur_seconds*osf))

    # save screen display for each TR
    visual_dm_array = np.zeros((total_TR,  round(screen_res[0]*res_scaling), round(screen_res[1]*res_scaling)))

    # some counters
    trl_blk_counter = 0
    trial_counter = 0
    trial = 0
    stim_counter = 0

    # for each "TR"
    for i in range(total_TR):

        img = Image.new('RGB', tuple(screen_res)) # background image

        if trial_info.iloc[trial]['trial_type'] == 'mini_block_%i'%mini_blk_num:

            # choose part of DF that corresponds to miniblock
            miniblk_df = bar_pos.loc[bar_pos['mini_block'] == mini_blk_num]

            # get name of attended condition for miniblock
            attended_condition = miniblk_df.loc[miniblk_df['attend_condition'] == 1]['condition'].values[0]

            # which bar do we want?
            if attend_cond['color'] and attend_cond['orientation']: # if we want fully attended bar
                chosen_condition = get_cond_name(attended_condition,'ACAO')

            elif not attend_cond['color'] and not attend_cond['orientation']: # if we want fully un-attended bar
                chosen_condition = get_cond_name(attended_condition,'UCUO')

            elif attend_cond['color'] and not attend_cond['orientation']: # if we want semi-attended bar (attend color not orientation)
                chosen_condition = get_cond_name(attended_condition,'ACUO')

            elif not attend_cond['color'] and attend_cond['orientation']: # if we want semi-attended bar (attend orientation not color)
                chosen_condition = get_cond_name(attended_condition,'UCAO')
            
            if trial_counter == 0:
                print('attended condition in miniblock %s, chosen condition is %s'%(attended_condition, chosen_condition))

            # bar positions for miniblock
            miniblk_positions = miniblk_df.loc[miniblk_df['condition'] == chosen_condition]['bar_midpoint_at_TR'].values[0]

            # coordenates for bar pass of trial, for PIL Image - DO NOT CONFUSE WITH CONDITION ORIENTATION
            # x position, y position 
            hor_x = miniblk_positions[trl_blk_counter][0] + screen_res[0]/2
            hor_y = miniblk_positions[trl_blk_counter][1] + screen_res[1]/2
            
            coordenates_bars = {'vertical': {'upLx': hor_x-0.5*bar_width*screen_res[0], 
                                                'upLy': screen_res[1],
                                                'lowRx': hor_x+0.5*bar_width*screen_res[0], 
                                                'lowRy': 0},
                                'horizontal': {'upLx': 0, 
                                                'upLy': hor_y+0.5*bar_width*screen_res[1],
                                                'lowRx': screen_res[0], 
                                                'lowRy': hor_y-0.5*bar_width*screen_res[1]}
                                }
            
            # if within time to display bar, then draw it
            if stim_counter < stim_dur_TR:
                # set draw method for image
                draw = ImageDraw.Draw(img)
                # add bar, coordinates (upLx, upLy, lowRx, lowRy)
                draw.rectangle(tuple([coordenates_bars[chosen_condition.split('_')[-1]]['upLx'],coordenates_bars[chosen_condition.split('_')[-1]]['upLy'],
                                    coordenates_bars[chosen_condition.split('_')[-1]]['lowRx'],coordenates_bars[chosen_condition.split('_')[-1]]['lowRy']]), 
                            fill = (255,255,255),
                            outline = (255,255,255))

                stim_counter += 1 # increment counter

            # update counter of trials within miniblok
            if trial_counter == osf-1:
                trl_blk_counter += 1
                # reset stim counter
                stim_counter = 0

            # if last trial of miniblock
            if trl_blk_counter == len(bar_pos.iloc[0]['bar_pass_direction_at_TR']):
                # update counters, so we can do same in next miniblock
                trl_blk_counter = 0

        ## mask the array - messy, fix later
        mask_array = np.array(img)[...,0].copy()
        x_bounds = [int(screen_res[0]/2 + xy_lim_pix['x_lim'][0]), int(screen_res[0]/2 + xy_lim_pix['x_lim'][1])]
        y_bounds = [int(screen_res[1]/2 + xy_lim_pix['y_lim'][0]), int(screen_res[1]/2 + xy_lim_pix['y_lim'][1])]

        mask_array[..., (screen_res[0] - x_bounds[0]):] = 0
        mask_array[..., 0:(screen_res[0] - x_bounds[1])] = 0
        mask_array[(screen_res[1] - y_bounds[0]):, ...] = 0
        mask_array[0:(screen_res[1] - y_bounds[1]), ...] = 0

        ## save and dpwnsample spatial resolution
        visual_dm_array[i, ...] = mask_array[::round(1/res_scaling),::round(1/res_scaling)][np.newaxis,...]
        
        # increment trial counter
        trial_counter += 1 
        
        # if sampling time reached,
        if trial_counter == osf: # then reset counter and update trial
            trial_counter = 0 
            trial += 1

    # swap axis to have time in last axis [x,y,t]
    visual_dm = visual_dm_array.transpose([1,2,0])
    
    # in case we want to crop the beginning of the DM
    if crop == True:
        if crop_unit == 'sec': # fix for the fact that I crop TRs, but task not synced to TR
            visual_dm = visual_dm[...,int(crop_TR*TR*osf)::] 
        else: # assumes unit is TR
            visual_dm = visual_dm[...,int(crop_TR*osf)::] 

    # shifting TRs to the left (quick fix)
    # to account for first trigger that was "dummy" - in future change experiment settings to skip 1st TR
    if shift_TRs == True:

        new_visual_dm = visual_dm.copy()

        if crop_unit == 'sec': # fix for the fact that I shift TRs, but task not synced to TR
            new_visual_dm[...,:-int(shift_TR_num*TR*osf)] = visual_dm[...,int(shift_TR_num*TR*osf):]
        else: # assumes unit is TR
            new_visual_dm[...,:-int(shift_TR_num*osf)] = visual_dm[...,int(shift_TR_num*osf):]
            
        visual_dm = new_visual_dm.copy()

            
    return visual_dm


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


def create_hrf(hrf_params=[1.0, 1.0, 0.0], TR = 1.6, osf = 1, onset = 0):
    """
    construct single or multiple HRFs 
    [taken from prfpy - all credits go to Marco]

    Parameters
    ----------
    hrf_params : TYPE, optional
        DESCRIPTION. The default is [1.0, 1.0, 0.0].
    Returns
    -------
    hrf : ndarray
        the hrf.
    """

    hrf = np.array(
        [
            np.ones_like(hrf_params[1])*hrf_params[0] *
            spm_hrf(
                tr = TR,
                oversampling = osf,
                onset=onset,
                time_length = 40)[...,np.newaxis],
            hrf_params[1] *
            spm_time_derivative(
                tr = TR,
                oversampling = osf,
                onset=onset,
                time_length = 40)[...,np.newaxis],
            hrf_params[2] *
            spm_dispersion_derivative(
                tr = TR,
                oversampling = osf,
                onset=onset,
                time_length = 40)[...,np.newaxis]]).sum(
        axis=0)                    

    return hrf.T


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


def regressOUT_confounds(file, counfounds, outdir, TR=1.6, plot_vert = False):
    
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
                    op.split(input_file)[-1].replace(file_extension,'_{name}{ext}'.format(name = 'confound_psc',
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
            filtered_signal = clean(signals = data.T, confounds = conf_df.values, detrend = True, 
                              standardize = 'psc', standardize_confounds = True, filter = False, t_r = TR)
            
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


def get_cue_regressor(trial_info, hrf_params = [1,1,0], cues = [0,1,2,3], TR = 1.6, oversampling_time = 1, 
                        baseline = None, first_modes_to_remove = 5,
                      crop_unit = 'sec', crop = False, crop_TR = 3, shift_TRs = True, shift_TR_num = 1, pad_length = 20):
    
    """Get timecourse for cue regressor
    
    Parameters
    ----------
    output : string
       absolute output name for numpy array
    params : yml dict
        with experiment params
    trial_info: pd
        pandas dataframe with useful run info
    hrf : array
       hrf to convolve with cue
    cues: list
        list with cue miniblock numbers (we might want to have all cues in reg or just one)
    
    """

    hrf = create_hrf(hrf_params = hrf_params, TR = TR, osf = oversampling_time)
        
    # initialized array of zeros for cue regressors - UPSAMPLED
    cue_regs_upsampled = np.zeros((hrf.shape[0], len(trial_info)*oversampling_time))
    
    # fill it given cue onsets
    for c in cues:
        for trl in trial_info.loc[trial_info['trial_type']=='cue_%i'%c]['trial_num'].values:
            
            cue_regs_upsampled[:,trl*oversampling_time:trl*oversampling_time+oversampling_time] = 1
            
    # in case we want to crop the beginning of the DM
    if crop == True:
        if crop_unit == 'sec': # fix for the fact that I crop TRs, but task not synced to TR
            cue_regs_upsampled = cue_regs_upsampled[...,int(crop_TR*TR*oversampling_time)::] 
        else: # assumes unit is TR
            cue_regs_upsampled = cue_regs_upsampled[...,crop_TR*oversampling_time::]
            
    # shifting TRs to the left (quick fix)
    # to account for first trigger that was "dummy" - in future change experiment settings to skip 1st TR
    if shift_TRs == True:
        new_cue_regs_upsampled = cue_regs_upsampled.copy()

        if crop_unit == 'sec': # fix for the fact that I shift TRs, but task not synced to TR
            new_cue_regs_upsampled[...,:-int(shift_TR_num*TR*oversampling_time)] = cue_regs_upsampled[...,int(shift_TR_num*TR*oversampling_time):]
        else: # assumes unit is TR
            new_cue_regs_upsampled[...,:-int(shift_TR_num*oversampling_time)] = cue_regs_upsampled[...,int(shift_TR_num*oversampling_time):]

        cue_regs_upsampled = new_cue_regs_upsampled.copy()
    
    ## convolve with hrf
    #scipy fftconvolve does not have padding options so doing it manually
    pad = np.tile(cue_regs_upsampled[:,0], (pad_length*oversampling_time,1)).T
    padded_cue = np.hstack((pad,cue_regs_upsampled))

    print('convolving cue regressor')
    cue_regs_upsampled = np.array(Parallel(n_jobs=16)(delayed(fftconvolve)(padded_cue[vertex], hrf[vertex], axes=(-1))
                                             for _,vertex in enumerate(tqdm(range(padded_cue.shape[0])))))[..., pad_length*oversampling_time:cue_regs_upsampled.shape[-1]+pad_length*oversampling_time] 
    
    ## resample to data sampling rate
    print('resampling to TR')
    cue_regs_RESAMPLED = np.array(Parallel(n_jobs=16)(delayed(resample_arr)(cue_regs_upsampled[vertex], osf = oversampling_time, final_sf = TR)
                                                                    for _,vertex in enumerate(tqdm(range(cue_regs_upsampled.shape[0])))))

    ## filter it, like we do to the data
    cue_regs =  dc_data(cue_regs_RESAMPLED,
                                first_modes_to_remove = first_modes_to_remove)
    
    # add baseline if baseline array/value specified
    if baseline is not None:
        if np.size(baseline) > 1:
            cue_regs = np.add(cue_regs, baseline[..., np.newaxis])
        else:
            cue_regs += baseline 
        
    return cue_regs


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


def create_glasser_df(path2file):
    
    # we read in the atlas data, which consists of 180 separate regions per hemisphere. 
    # These are labeled separately, so the labels go to 360.
    cifti = nib.load(op.join(path2file,
                         'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii'))

    # get index data array
    cifti_data = np.array(cifti.get_fdata(dtype=np.float32))[0]
    # from header get label dict with key + rgba
    cifti_hdr = cifti.header
    label_dict = cifti_hdr.get_axis(0)[0][1]
    
    ## make atlas data frame
    atlas_df = pd.DataFrame(columns = ['ROI', 'index','R','G','B','A'])

    for key in label_dict.keys():

        if label_dict[key][0] != '???': 
            atlas_df = atlas_df.append(pd.DataFrame({'ROI': label_dict[key][0].replace('_ROI',''),
                                                     'index': key,
                                                     'R': label_dict[key][1][0],
                                                     'G': label_dict[key][1][1],
                                                     'B': label_dict[key][1][2],
                                                     'A': label_dict[key][1][3]
                                                    }, index=[0]),ignore_index=True)
            
    return atlas_df, cifti_data


def make_nuisance_tc(pars, timecourse = [], onsets = [1], hrf = [], fit = True, 
                     osf = 10, blk_duration = 45):
    
    """ helper function to make task on nuisance regressor timecourse
    if fitting, returns residuals from model with block on + nuisance + intercept
    if not fitting retuns the nuisance timecourse (good for visualization)
    
    Parameters
    ----------
    pars : Parameters object from lmfit
        needs to include duration of nuisance regressor, and bounds
    timecourse : array
        data to be fitted
    onset: list
        list of onset times
    hrf: list
        hrf to use
    blk_duration: int/float
        duration of miniblock
    
    """
    
    # set nuisance regressor - that will be modeled
    nuisance_reg = np.zeros((timecourse.shape[0]*osf))
    # set regressor that accounts for task being on-screen
    blkON_reg = np.zeros((timecourse.shape[0]*osf))
    
    for t in onsets:
        nuisance_reg[t*osf:int((t+pars['duration'].value)*osf)] = 1
        blkON_reg[t*osf:int((t+blk_duration)*osf)] = 1

    ## convolve regressors
    pad_length = 20*osf
    pad = np.tile(0, pad_length).T
    
    # nuisance
    padded_tc = np.hstack((pad, nuisance_reg))
    conv_nuisance_reg = fftconvolve(padded_tc, hrf)[pad_length:timecourse.shape[0]*osf+pad_length]
    # on-off
    padded_tc = np.hstack((pad, blkON_reg))
    conv_blkON_reg = fftconvolve(padded_tc, hrf)[pad_length:timecourse.shape[0]*osf+pad_length]
    
    ## donwsample again
    conv_nuisance_reg = resample_arr(conv_nuisance_reg, osf = osf, final_sf = 1)
    conv_blkON_reg = resample_arr(conv_blkON_reg, osf = osf, final_sf = 1)
    
    if fit:
        # make dm
        dm = np.ones((3, conv_nuisance_reg.shape[0]))
        dm[-2] = conv_blkON_reg
        dm[-1] = conv_nuisance_reg

        prediction, _ , r2, _ = fit_glm(timecourse, dm.T)
        print(r2)
        
        return timecourse - prediction
    
    else:
        return conv_nuisance_reg



def make_raw_vertex_image(data1, cmap = 'hot', vmin = 0, vmax = 1, 
                          data2 = [], vmin2 = 0, vmax2 = 1, subject = 'fsaverage', data2D = False):  
    """ function to fix web browser bug in pycortex
        allows masking of data with nans
    
    Parameters
    ----------
    data1 : array
        data array
    cmap : str
        string with colormap name (not the alpha version)
    vmin: int/float
        minimum value
    vmax: int/float 
        maximum value
    subject: str
        overlay subject name to use
    
    Outputs
    -------
    vx_fin : VertexRGB
        vertex object to call in webgl
    
    """
    
    # Get curvature
    curv = cortex.db.get_surfinfo(subject, type = 'curvature', recache=False)#,smooth=1)
    # Adjust curvature contrast / color. Alternately, you could work
    # with curv.data, maybe threshold it, and apply a color map.     
    curv.data[curv.data>0] = .1
    curv.data[curv.data<=0] = -.1
    #curv.data = np.sign(curv.data.data) * .25
    
    curv.vmin = -1
    curv.vmax = 1
    curv.cmap = 'gray'
    
    # Create display data 
    vx = cortex.Vertex(data1, subject, cmap = cmap, vmin = vmin, vmax = vmax)
    
    # Pick an arbitrary region to mask out
    # (in your case you could use np.isnan on your data in similar fashion)
    if data2D:
        data2[np.isnan(data2)] = vmin2
        norm2 = colors.Normalize(vmin2, vmax2)  
        alpha = np.clip(norm2(data2), 0, 1)
    else:
        alpha = ~np.isnan(data1) #(data < 0.2) | (data > 0.4)
    alpha = alpha.astype(np.float)
    
    # Map to RGB
    vx_rgb = np.vstack([vx.raw.red.data, vx.raw.green.data, vx.raw.blue.data])
    vx_rgb[:,alpha>0] = vx_rgb[:,alpha>0] * alpha[alpha>0]
    
    curv_rgb = np.vstack([curv.raw.red.data, curv.raw.green.data, curv.raw.blue.data])
    # do this to avoid artifacts where curvature gets color of 0 valur of colormap
    curv_rgb[:,np.where((vx_rgb > 0))[-1]] = curv_rgb[:,np.where((vx_rgb > 0))[-1]] * (1-alpha)[np.where((vx_rgb > 0))[-1]]

    # Alpha mask
    display_data = curv_rgb + vx_rgb 

    # Create vertex RGB object out of R, G, B channels
    vx_fin = cortex.VertexRGB(*display_data, subject, curvature_brightness = 0.4, curvature_contrast = 0.1)

    return vx_fin


def get_weighted_bins(data_df, x_key = 'ecc', y_key = 'size', weight_key = 'rsq', n_bins = 10):
    
    # sort values by eccentricity
    data_df = data_df.sort_values(by=[x_key])

    #divide in equally sized bins
    bin_size = int(len(data_df)/n_bins) 
    
    mean_x = []
    mean_x_std = []
    mean_y = []
    mean_y_std = []
    
    # for each bin calculate rsq-weighted means and errors of binned ecc/gain 
    for j in range(n_bins): 
        
        mean_x.append(weightstats.DescrStatsW(data_df[bin_size * j:bin_size * (j+1)][x_key],
                                              weights = data_df[bin_size * j:bin_size * (j+1)][weight_key]).mean)
        mean_x_std.append(weightstats.DescrStatsW(data_df[bin_size * j:bin_size * (j+1)][x_key],
                                                  weights = data_df[bin_size * j:bin_size * (j+1)][weight_key]).std_mean)

        mean_y.append(weightstats.DescrStatsW(data_df[bin_size * j:bin_size * (j+1)][y_key],
                                              weights = data_df[bin_size * j:bin_size*(j+1)][weight_key]).mean)
        mean_y_std.append(weightstats.DescrStatsW(data_df[bin_size * j:bin_size * (j+1)][y_key],
                                                  weights = data_df[bin_size * j:bin_size * (j+1)][weight_key]).std_mean)

    return mean_x, mean_x_std, mean_y, mean_y_std


def baseline_correction(data, condition_per_TR, num_baseline_TRs = 6, baseline_interval = 'empty_long', 
                        avg_type = 'median'):
    
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
    """


    baseline_index = [i for i, val in enumerate(condition_per_TR) if str(val) == baseline_interval]
    interval_ind = []
    for i, ind in enumerate(baseline_index):
        if i > 0:
            if (ind - baseline_index[i-1]) > 1: ## transition point
                interval_ind.append([baseline_index[i-1] - num_baseline_TRs, baseline_index[i-1]]) 

    if condition_per_TR[-1] == 'empty_long':
        interval_ind.append([baseline_index[-1] - num_baseline_TRs, baseline_index[-1]]) 

    # get baseline values
    baseline_arr = np.hstack([data[..., ind[0]:ind[1]] for ind in interval_ind])

    # average
    if avg_type == 'median':
        avg_baseline = np.median(baseline_arr, axis = -1)
    else:
        avg_baseline = np.mean(baseline_arr, axis = -1)

    return data - avg_baseline[...,np.newaxis]


def make_blk_nuisance_regressor(data, params, pRF_rsq, 
                                TR = 1.6, osf = 10, hrf_estimates = {'hrf_derivative': [], 'hrf_dispersion': []},
                                pRF_rsq_threshold = .1, roi_verts = {'V1': [], 'V2': []}):
    
    """ function to make task on nuisance regressor 
    for whole surface!
    if fitting, returns residuals from model with block on + nuisance + intercept
    if not fitting retuns the model timecourse (good for visualization)
    
    Parameters
    ----------
    data: array
        2D array of data(vertices, TRs)
    params : dict
        yaml dict with task related infos
    pRF_rsq : array
        rsq of the prf estimates - needed to fit only visually responsive vertices
    roi_verts: dict
        dict with roi vertices (indices) to use
    
    """
    
   ## first get events at each timepoint
    all_evs = np.array([])
    for ev in params['feature']['bar_pass_direction']:

        if 'empty' in ev:
            all_evs = np.concatenate((all_evs, np.tile(ev, params['feature']['empty_TR'])))
        elif 'cue' in ev:
            all_evs = np.concatenate((all_evs, np.tile(ev, params['feature']['cue_TR'])))
        elif 'mini_block' in ev:
            all_evs = np.concatenate((all_evs, np.tile(ev, np.prod(params['feature']['num_bar_position'])*2)))

    # times where bar is on screen [1st on per miniblock]
    bar_onset = np.array([i for i, name in enumerate(all_evs) if 'mini_block' in name and all_evs[i-1]=='empty'])
    # times where cue is on screen [1st time point]
    cue_onset = np.array([i for i, name in enumerate(all_evs) if 'cue' in name and all_evs[i-1]=='empty'])

    # combined - 0 is nothing on screen, 1 is something there
    stim_on_bool = np.array([1 if 'cue' in name or 'mini_block' in name else 0 for _, name in enumerate(all_evs) ])

    ## if cropping
    if params['feature']['crop']:
        bar_onset = bar_onset - params['feature']['crop_TR']*TR - TR*1.5 ## NOTE - doing this subtraction because of shift+no slicetime correction, in future generalize
        cue_onset = cue_onset - params['feature']['crop_TR']*TR - TR*1.5

        ## resample stim_on array
        tmp_arr = np.repeat(stim_on_bool, osf)
        tmp_arr[:-int(TR*1.5*osf)] = np.repeat(stim_on_bool, osf)[int(TR*1.5*osf):]
        stim_on_bool = tmp_arr.copy()[int(params['feature']['crop_TR']*TR*osf):]

        stim_on_bool = resample_arr(stim_on_bool, osf = osf, final_sf = TR)

    ## get indices where miniblock starts and ends (in TR!!)
    stim_ind = np.where(stim_on_bool>=.5)[0]
    miniblk_start_ind = []
    miniblk_end_ind = []

    for i, val in enumerate(stim_ind):
        if i>0: 
            if val - stim_ind[i-1] > 1:
                miniblk_start_ind.append(val)

                if stim_on_bool[stim_ind[i-1]+1]<1:
                    miniblk_end_ind.append(stim_ind[i-1])

    # remove cue start indices
    miniblk_start_ind = np.array(miniblk_start_ind[::2])-1    
    miniblk_end_ind = np.concatenate((miniblk_end_ind[1::2], np.array([stim_ind[-1]])))+1
    
    ## average timecourse across ROI
    avg_roi = {} #empty dictionary  

    for _,val in enumerate(roi_verts.keys()):

        ind = np.array([vert for vert in roi_verts[val] if not np.isnan(pRF_rsq[vert]) and pRF_rsq[vert]>=pRF_rsq_threshold])
        avg_roi[val] = np.mean(data[ind], axis=0)

    ## now get average timecourse for miniblock
    avg_miniblk =  {} #empty dictionary 
    interv = 3 # average from begining of miniblk to end, +/- 5 sec

    for _,val in enumerate(roi_verts.keys()):
        avg_miniblk[val] = np.mean(np.stack((avg_roi[val][miniblk_start_ind[i]-interv:miniblk_end_ind[i]+interv] for i in range(len(miniblk_start_ind))), axis = 0), axis = 0)

    ## average across rois and miniblocks
    avg_minblk_tc =  np.mean(np.stack((avg_miniblk[val] for val in roi_verts.keys()), axis = 0), axis = 0)
    
    ## make nuisance regressor
    # for the miniblock
    pars = Parameters()
    pars.add('duration', value = 0, min = 0, max = 6, vary = True, brute_step = .1) # duration in TR

    ## minimize residuals
    out = minimize(make_nuisance_tc, pars, 
                   kws={'timecourse': avg_minblk_tc, 'onsets': [interv], 
                        'hrf': create_hrf(hrf_params = [1.0, 1.0, 0.0], TR = TR, osf = osf)[0],
                       'fit': True, 'osf': osf, 'blk_duration': miniblk_end_ind[0]-miniblk_start_ind[0]-1}, 
                   method = 'brute')

    # update nuisance regressor duration (in TR!)
    pars['duration'].value = out.params.valuesdict()['duration']
    print('modeled nuisance duration is %.2f TR'%out.params.valuesdict()['duration'])
    
    ## use subject specific hrf params (if provided)
    hrf_params = np.ones((3, pRF_rsq.shape[0]))

    if len(hrf_estimates['hrf_derivative'])==0: # if hrf not defined
        hrf_params[2] = 0
    
    else: # use fitted hrf params
        hrf_params[1] = hrf_estimates['hrf_derivative']
        hrf_params[2] = hrf_estimates['hrf_dispersion']
        
    ## get indices that are relevant to create regressor (saves time)
    mask_ind = np.array([ind for ind,val in enumerate(pRF_rsq) if val >= pRF_rsq_threshold])
    
    ## actually get regressors for surface
    all_regs = np.array(Parallel(n_jobs=16)(delayed(make_nuisance_tc)(pars, 
                                                                    timecourse = data[vert], 
                                                                    onsets = miniblk_start_ind, 
                                                                    hrf = create_hrf(hrf_params = hrf_params[..., vert], 
                                                                                     TR = TR, 
                                                                                     osf = osf)[0],
                                                                    fit = False)
                                        for _,vert in enumerate(tqdm(mask_ind)))) 


    ## save in the same shape of data 
    nuisance_regressor_surf = np.zeros(data.shape)
    nuisance_regressor_surf[mask_ind] = all_regs 
    
    return nuisance_regressor_surf


def get_rois4plotting(params, pysub = 'hcp_999999', use_atlas = True, atlas_pth = '', space = 'fsLR_den-170k'):

    """ helper function to get ROI names, vertice index and color palette
   to be used in plotting scripts
    
    Parameters
    ----------
    params : dict
        yaml dict with task related infos  
    """ 
    
    roi_verts = {} #empty dictionary  
    
    if use_atlas:
        # Get Glasser atlas
        atlas_df, atlas_array = create_glasser_df(atlas_pth)

        # ROI names
        ROIs = list(params['plotting']['ROIs']['glasser_atlas'].keys())
        # colors
        color_codes = {key: params['plotting']['ROIs']['glasser_atlas'][key]['color'] for key in ROIs}

        # get vertices for ROI
        for _,key in enumerate(ROIs):
            roi_verts[key] = np.hstack((np.where(atlas_array == ind)[0] for ind in atlas_df[atlas_df['ROI'].isin(params['plotting']['ROIs']['glasser_atlas'][key]['ROI'])]['index'].values))

    else:
        # set ROI names
        ROIs = params['plotting']['ROIs'][space]

        # dictionary with one specific color per group - similar to fig3 colors
        ROI_pal = params['plotting']['ROI_pal']
        color_codes = {key: ROI_pal[key] for key in ROIs}

        # get vertices for ROI
        for _,val in enumerate(ROIs):
            print(val)
            roi_verts[val] = cortex.get_roi_verts(pysub,val)[val]
            
    return ROIs, roi_verts, color_codes


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


def fwhmax_fwatmin(model, estimates, normalize_RFs=False, return_profiles=False):
    
    """
    taken from marco aqil's code, all credits go to him
    """
    
    model = model.lower()
    x=np.linspace(-50,50,1000).astype('float32')

    prf = estimates['betas'] * np.exp(-0.5*x[...,np.newaxis]**2 / estimates['size']**2)
    vol_prf =  2*np.pi*estimates['size']**2

    if 'dog' in model or 'dn' in model:
        srf = estimates['sa'] * np.exp(-0.5*x[...,np.newaxis]**2 / estimates['ss']**2)
        vol_srf = 2*np.pi*estimates['ss']*2

    if normalize_RFs==True:

        if model == 'gauss':
            profile =  prf / vol_prf
        elif model == 'css':
            #amplitude is outside exponent in CSS
            profile = (prf / vol_prf)**estimates['ns'] * estimates['betas']**(1 - estimates['ns'])
        elif model =='dog':
            profile = prf / vol_prf - \
                       srf / vol_srf
        elif 'dn' in model:
            profile = (prf / vol_prf + estimates['nb']) /\
                      (srf / vol_srf + estimates['sb']) - estimates['nb']/estimates['sb']
    else:
        if model == 'gauss':
            profile = prf
        elif model == 'css':
            #amplitude is outside exponent in CSS
            profile = prf**estimates['ns'] * estimates['betas']**(1 - estimates['ns'])
        elif model =='dog':
            profile = prf - srf
        elif 'dn' in model:
            profile = (prf + estimates['nb'])/(srf + estimates['sb']) - estimates['nb']/estimates['sb']


    half_max = np.max(profile, axis=0)/2
    fwhmax = np.abs(2*x[np.argmin(np.abs(profile-half_max), axis=0)])


    if 'dog' in model or 'dn' in model:

        min_profile = np.min(profile, axis=0)
        fwatmin = np.abs(2*x[np.argmin(np.abs(profile-min_profile), axis=0)])

        result = fwhmax, fwatmin
    else:
        result = fwhmax

    if return_profiles:
        return result, profile.T
    else:
        return result


def load_pRF_estimates(fits_pth, params, total_chunks = 54, model_type = 'gauss'):

    """ Load pRF estimates and combined them
    
    Parameters
    ----------
    fits_pth : str
        absolute path to fits locations
    params : dict
        yaml dict with task related infos 
    """
    
    # path to combined estimates
    estimates_pth = op.join(fits_pth,'combined')

    # combined estimates filename
    est_name = [x for _,x in enumerate(os.listdir(fits_pth)) if 'chunk-001' in x][0]
    est_name = est_name.replace('chunk-001_of_{ch}'.format(ch=str(total_chunks).zfill(3)),'chunk-combined')

    # total path to estimates path
    estimates_combi = op.join(estimates_pth, est_name)

    if op.isfile(estimates_combi): # if combined estimates exists

        print('loading %s'%estimates_combi)
        estimates = np.load(estimates_combi) # load it

    else: # if not join chunks and save file
        if not op.exists(estimates_pth):
            os.makedirs(estimates_pth) 

        estimates = join_chunks(fits_pth, estimates_combi, fit_hrf = params['mri']['fitting']['pRF']['fit_hrf'],
                                chunk_num = total_chunks, fit_model = 'it{model}'.format(model=model_type)) #'{model}'.format(model=model_type)))#
    
    return estimates