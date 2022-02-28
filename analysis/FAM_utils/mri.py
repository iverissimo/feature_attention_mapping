
# script to calculate tSNR
# for the different sequences piloted


import numpy as np
import os
from os import path as op
import nibabel as nib
import re

import pandas as pd
import json

from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from scipy import fft, interpolate

from nilearn import surface

from joblib import Parallel, delayed

import matplotlib.pyplot as plt

import cortex
from cortex import fmriprep

from PIL import Image, ImageDraw

from matplotlib import cm
import matplotlib.colors

from scipy.stats import t, norm

import scipy.signal as signal
from nilearn.glm.first_level.hemodynamic_models import spm_hrf, spm_time_derivative, spm_dispersion_derivative

from nilearn.signal import clean

from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel

from nilearn.glm.first_level import first_level
from nilearn.glm.regression import ARModel

from prfpy.rf import gauss2D_iso_cart
from prfpy.timecourse import stimulus_through_prf

from prfpy.timecourse import filter_predictions

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
        fmriprep.import_subj(subject = sj, source_dir = source_directory, 
                             session = ses, dataset = dataset, acq = acq)


def get_tsnr(data,affine,file_name):
    """
    Compute the tSNR of nifti file
    and generate the equivalent nifti SNR 3Ds. 
    """ 

    if not op.exists(file_name): 
        print('making %s'%file_name)
    
        mean_d = np.mean(data,axis=-1)
        std_d = np.std(data,axis=-1)
        
        tsnr = mean_d/std_d
        #tsnr[np.where(np.isinf(tsnr))] = np.nan
        
        tsnr_image = nib.nifti1.Nifti1Image(tsnr,affine)
        
        nib.save(tsnr_image,file_name)

    else:
        print('already exists, skipping %s'%file_name)
        tsnr = nib.load(file_name)
        tsnr = np.array(tsnr.dataobj)
    
    return tsnr


def correlate_vol(data1,data2,outfile):
    """
    Compute Pearson correlation between 2 of nifti files
    and generate the equivalent correlation nifti. 
    """ 

    if not op.exists(outfile): 
        print('making %s'%outfile)
    
        # get affine for one of the runs
        nibber = nib.load(data1)
        affine = nibber.affine
        
        # load data 
        data1 = np.array(nib.load(data1).dataobj)
        data2 = np.array(nib.load(data2).dataobj)
        
        #- Calculate the number of voxels (number of elements in one volume)
        n_voxels = np.prod(data1.shape[:-1])

        #- Reshape 4D array to 2D array n_voxels by n_volumes
        data1_2d = np.reshape(data1, (n_voxels, data1.shape[-1]))
        data2_2d = np.reshape(data2, (n_voxels, data2.shape[-1]))

        #- Make a 1D array of size (n_voxels,) to hold the correlation values
        correlations_1d = np.zeros((n_voxels,))

        #- Loop over voxels filling in correlation at this voxel
        for i in range(n_voxels):
            correlations_1d[i] = np.corrcoef(data1_2d[i, :], data2_2d[i, :])[0, 1]
            
        #- Reshape the correlations array back to 3D
        correlations = np.reshape(correlations_1d, data1.shape[:-1])
        
        corr_image = nib.nifti1.Nifti1Image(correlations,affine)
        
        nib.save(corr_image,outfile)

    else:
        print('already exists, skipping %s'%outfile)
        correlations = nib.load(outfile)
        correlations = np.array(correlations.dataobj)
    
    return correlations


def crop_epi(file, outdir, num_TR_task=220, num_TR_crop = 5):

    """ crop epi file (expects numpy file)
    and thus remove the first recorded "dummy" trials, if such was the case
    
    Parameters
    ----------
    file : str/list/array
        absolute filename to be filtered (or list of filenames)
    outdir : str
        path to save new file
    num_TR_task: int
        number of TRs of task, for safety check
    num_TR_crop : int
        number of TRs to remove from beginning of file
    
    Outputs
    -------
    out_file: str
        absolute output filename (or list of filenames)
    
    """
    
    # check if single filename or list of filenames
    
    if isinstance(file, list): 
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
    if not isinstance(file, list): 
        outfiles = outfiles[0] 

    return outfiles


def filter_data(file, outdir, filter_type = 'HPgauss', plot_vert=False,
                first_modes_to_remove=5, **kwargs):
    
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
    
    if isinstance(file, list): 
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
                raise NameError('Not implemented')
                
            # if plotting true, make figure of voxel with high variance,
            # to compare the difference
            if plot_vert == True:
                
                ind2plot = np.argwhere(np.std(data, axis=-1)==np.max(np.std(data, axis=-1)))[0][0]
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
    if not isinstance(file, list): 
        outfiles = outfiles[0] 
    
    return outfiles

def gausskernel_data(data, TR = 1.2, cut_off_hz = 0.01, **kwargs):
    
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
    
    if isinstance(file, list): 
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
            
            data = np.load(input_file,allow_pickle=True)
            
            mean_signal = data.mean(axis = -1)[..., np.newaxis]
            data_psc = (data - mean_signal)/np.absolute(mean_signal)
            data_psc *= 100
                
            ## save psc file
            np.save(output_file,data_psc)

        # append out files
        outfiles.append(output_file)
        
    # if input file was not list, then return output that is also not list
    if not isinstance(file, list): 
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
    if not isinstance(file, list): 
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


def make_pRF_DM(output, params, save_imgs = False, downsample = None, 
                    crop = False, crop_TR = 8, overwrite=False, shift_TRs = True, mask = []):
    
    """Make design matrix for pRF task
    
    Parameters
    ----------
    output : string
       absolute output name for DM
    params : yml dict
        with experiment params
    save_imgs : bool
       if we want to save images in folder, for sanity check
    """
    
    if not op.exists(output) or overwrite == True: 
        print('making %s'%output)

        if not op.exists(op.split(output)[0]): # make base dir to save files
            os.makedirs(op.split(output)[0])
        
        # general infos
        bar_width = params['prf']['bar_width_ratio'] 

        screen_res = params['window']['size']
        if params['window']['display'] == 'square': # if square display
            screen_res = np.array([screen_res[1], screen_res[1]])

        if downsample != None: # if we want to downsample screen res
            screen_res = (screen_res*downsample).astype(int)

        # number TRs per condition
        TR_conditions = {'L-R': params['prf']['num_TRs']['L-R'],
                         'R-L': params['prf']['num_TRs']['R-L'],
                         'U-D': params['prf']['num_TRs']['U-D'],
                         'D-U': params['prf']['num_TRs']['D-U'],
                         'empty': params['prf']['num_TRs']['empty'],
                         'empty_long': params['prf']['num_TRs']['empty_long']}

        # order of conditions in run
        bar_pass_direction = params['prf']['bar_pass_direction']
        
        # get total number of TRs in run
        # list of bar orientation at all TRs
        total_TR = 0
        for _,bartype in enumerate(bar_pass_direction):
            total_TR += TR_conditions[bartype]

        # set fake mask, in case we are not masking actually
        if len(mask) == 0:
            mask = np.tile(False,total_TR)

        # all possible positions in pixels for for midpoint of
        # y position for vertical bar passes, 
        ver_y = screen_res[1]*np.linspace(0,1, TR_conditions['U-D'])
        # x position for horizontal bar passes 
        hor_x = screen_res[0]*np.linspace(0,1, TR_conditions['L-R'])        

        # coordenates for bar pass, for PIL Image
        coordenates_bars = {'L-R': {'upLx': hor_x-0.5*bar_width*screen_res[0], 'upLy': np.repeat(screen_res[1],TR_conditions['L-R']),
                                     'lowRx': hor_x+0.5*bar_width*screen_res[0], 'lowRy': np.repeat(0,TR_conditions['L-R'])},
                            'R-L': {'upLx': np.array(list(reversed(hor_x-0.5*bar_width*screen_res[0]))), 'upLy': np.repeat(screen_res[1],TR_conditions['R-L']),
                                     'lowRx': np.array(list(reversed(hor_x+0.5*bar_width*screen_res[0]))), 'lowRy': np.repeat(0,TR_conditions['R-L'])},
                            'U-D': {'upLx': np.repeat(0,TR_conditions['U-D']), 'upLy': ver_y+0.5*bar_width*screen_res[1],
                                     'lowRx': np.repeat(screen_res[0],TR_conditions['U-D']), 'lowRy': ver_y-0.5*bar_width*screen_res[1]},
                            'D-U': {'upLx': np.repeat(0,TR_conditions['D-U']), 'upLy': np.array(list(reversed(ver_y+0.5*bar_width*screen_res[1]))),
                                     'lowRx': np.repeat(screen_res[0],TR_conditions['D-U']), 'lowRy': np.array(list(reversed(ver_y-0.5*bar_width*screen_res[1])))}
                             }

        # save screen display for each TR
        visual_dm_array = np.zeros((total_TR, screen_res[0],screen_res[1]))
        counter = 0
        for _,bartype in enumerate(bar_pass_direction): # loop over bar pass directions

            for i in range(TR_conditions[bartype]):

                img = Image.new('RGB', tuple(screen_res)) # background image

                if bartype not in np.array(['empty','empty_long']): # if not empty screen

                    if mask[counter]==False: # if not masked 

                        # set draw method for image
                        draw = ImageDraw.Draw(img)
                        # add bar, coordinates (upLx, upLy, lowRx, lowRy)
                        draw.rectangle(tuple([coordenates_bars[bartype]['upLx'][i],coordenates_bars[bartype]['upLy'][i],
                                            coordenates_bars[bartype]['lowRx'][i],coordenates_bars[bartype]['lowRy'][i]]), 
                                    fill = (255,255,255),
                                    outline = (255,255,255))

                visual_dm_array[counter, ...] = np.array(img)[:,:,0][np.newaxis,...]
                counter += 1

        # swap axis to have time in last axis [x,y,t]
        visual_dm = visual_dm_array.transpose([1,2,0])

        # in case we want to crop the beginning of the DM
        if crop == True:
            visual_dm = visual_dm[...,crop_TR::] 
        
        # save design matrix
        np.save(output, visual_dm)
        
    else:
        print('already exists, skipping %s'%output)
        
        # load
        visual_dm = np.load(output)


    # shifting TRs to the left (quick fix)
    # to account for first trigger that was "dummy" - in future change experiment settings to skip 1st TR
    if shift_TRs == True:
        new_visual_dm = visual_dm.copy()
        new_visual_dm[...,:-1] = visual_dm[...,1:]
        visual_dm = new_visual_dm.copy()
        
    #if we want to save the images
    if save_imgs == True:
        outfolder = op.split(output)[0]

        visual_dm = visual_dm.astype(np.uint8)

        for w in range(visual_dm.shape[-1]):
            im = Image.fromarray(visual_dm[...,w])
            im.save(op.join(outfolder,"DM_TR-%i.png"%w))      
            
    return visual_dm


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
        orginal data shape 
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

  
def add_alpha2colormap(colormap = 'rainbow_r', bins = 256, invert_alpha = False, cmap_name = 'costum',
                      discrete = False):

    """ add alpha channel to colormap,
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
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
        
        if discrete == True: # if we want a discrete colormap from list
            cmap = matplotlib.colors.ListedColormap(colormap)
            bins = int(len(colormap))

    # convert into array
    cmap_array = cmap(range(bins))
    
    # make alpha array
    if invert_alpha == True: # in case we want to invert alpha (y from 1 to 0 instead pf 0 to 1)
        _, alpha = np.meshgrid(np.linspace(0, 1, bins, endpoint=False), 1-np.linspace(0, 1, bins))
    else:
        _, alpha = np.meshgrid(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins, endpoint=False))
    
    # reshape array for map
    new_map = []
    for i in range(cmap_array.shape[-1]):
        new_map.append(np.tile(cmap_array[...,i],(bins,1)))

    new_map = np.moveaxis(np.array(new_map), 0, -1)

    # add alpha channel
    new_map[...,-1] = alpha
    
    fig = plt.figure(figsize=(1,1))
    ax = fig.add_axes([0,0,1,1])
    # plot 
    plt.imshow(new_map,
    extent = (0,1,0,1),
    origin = 'lower')
    ax.axis('off')

    rgb_fn = os.path.join(os.path.split(cortex.database.default_filestore)[
                          0], 'colormaps', cmap_name+'_alpha_bins_%d.png'%bins)

    #misc.imsave(rgb_fn, new_map)
    plt.savefig(rgb_fn, dpi = 200,transparent=True)
       
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

            rsq = chunk['r2']

            if fit_hrf:
                hrf_derivative = chunk['hrf_derivative']
                hrf_dispersion = chunk['hrf_dispersion']
        else:
            xx = np.concatenate((xx,chunk['x']))
            yy = np.concatenate((yy,chunk['y']))

            size = np.concatenate((size,chunk['size']))

            beta = np.concatenate((beta,chunk['betas']))
            baseline = np.concatenate((baseline,chunk['baseline']))

            if 'css' in fit_model:
                ns = np.concatenate((ns,chunk['ns']))

            rsq = np.concatenate((rsq,chunk['r2']))
            
            if fit_hrf:
                hrf_derivative = np.concatenate((hrf_derivative, chunk['hrf_derivative']))
                hrf_dispersion = np.concatenate((hrf_dispersion, chunk['hrf_dispersion']))
    
    print('shape of estimates is %s'%(str(xx.shape)))

    # save file
    output = op.join(out_name)
    print('saving %s'%output)
    
    if 'css' in fit_model:

        if fit_hrf:
            np.savez(output,
              x = xx,
              y = yy,
              size = size,
              betas = beta,
              baseline = baseline,
              hrf_derivative = hrf_derivative,
              hrf_dispersion = hrf_dispersion,
              ns = ns,
              r2 = rsq)

        else:
            np.savez(output,
                x = xx,
                y = yy,
                size = size,
                betas = beta,
                baseline = baseline,
                ns = ns,
                r2 = rsq)
    else:        
        if fit_hrf:
            np.savez(output,
              x = xx,
              y = yy,
              size = size,
              betas = beta,
              baseline = baseline,
              hrf_derivative = hrf_derivative,
              hrf_dispersion = hrf_dispersion,
              r2 = rsq)

        else:
            np.savez(output,
                x = xx,
                y = yy,
                size = size,
                betas = beta,
                baseline = baseline,
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


def mask_estimates(estimates, ROI = 'None', fit_model = 'gauss', x_ecc_lim = [-6,6], y_ecc_lim = [-6,6],
                    max_size = 20, space = 'fsaverage'):
    
    """ mask estimates, to be positive RF, within screen limits
    and for a certain ROI (if the case)
    Parameters
    ----------
    estimates : List/arr
        list of estimates.npz for both hemispheres
    ROI : str
        roi to mask estimates (eg. 'V1', default 'None')
    fit_model: str
        fit model of estimates
    
    Outputs
    -------
    masked_estimates : npz 
        numpy array of masked estimates
    
    """
    
    xx = estimates['x']
    yy = estimates['y']
       
    size = estimates['size']
    
    beta = estimates['betas']
    baseline = estimates['baseline']
    
    if 'css' in fit_model:
        ns = estimates['ns']
    else: #if gauss
        ns = np.ones(xx.shape)

    rsq = estimates['r2']
    
    # set limits for xx and yy, forcing it to be within the screen boundaries
    # also for max fitting size used and for positive pRFs

    # make new variables that are masked 
    masked_xx = np.zeros(xx.shape); masked_xx[:]=np.nan
    masked_yy = np.zeros(yy.shape); masked_yy[:]=np.nan
    masked_size = np.zeros(size.shape); masked_size[:]=np.nan
    masked_beta = np.zeros(beta.shape); masked_beta[:]=np.nan
    masked_baseline = np.zeros(baseline.shape); masked_baseline[:]=np.nan
    masked_rsq = np.zeros(rsq.shape); masked_rsq[:]=np.nan
    masked_ns = np.zeros(ns.shape); masked_ns[:]=np.nan

    for i in range(len(xx)): #for all vertices
        if xx[i] <= np.max(x_ecc_lim) and xx[i] >= np.min(x_ecc_lim): # if x within horizontal screen dim
            if yy[i] <= np.max(y_ecc_lim)and yy[i] >= np.min(y_ecc_lim): # if y within vertical screen dim
                if beta[i]>=0: # only account for positive RF
                    if size[i]<=max_size: # limit size to max size defined in fit

                        # save values
                        masked_xx[i] = xx[i]
                        masked_yy[i] = yy[i]
                        masked_size[i] = size[i]
                        masked_beta[i] = beta[i]
                        masked_baseline[i] = baseline[i]
                        masked_rsq[i] = rsq[i]
                        masked_ns[i]=ns[i]

    if ROI != 'None':
        
        roi_ind = cortex.get_roi_verts(space,ROI) # get indices for that ROI
        
        # mask for roi
        masked_xx = masked_xx[roi_ind[ROI]]
        masked_yy = masked_yy[roi_ind[ROI]]
        masked_size = masked_size[roi_ind[ROI]]
        masked_beta = masked_beta[roi_ind[ROI]]
        masked_baseline = masked_baseline[roi_ind[ROI]]
        masked_rsq = masked_rsq[roi_ind[ROI]]
        masked_ns = masked_ns[roi_ind[ROI]]

    masked_estimates = {'x':masked_xx,'y':masked_yy,'size':masked_size,
                        'beta':masked_beta,'baseline':masked_baseline,'ns':masked_ns,
                        'rsq':masked_rsq}
    
    return masked_estimates

def normalize(M):
    """
    normalize data array
    """
    return (M-np.nanmin(M))/(np.nanmax(M)-np.nanmin(M))


def surf_data_from_cifti(data, axis, surf_name):

    """
    load surface data from cifti, from one hemisphere
    taken from https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb
    """

    assert isinstance(axis, nib.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:                                 # Just looking for a surface
            data = data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex                        # Generally 1-N, except medial wall vertices
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data

def load_data_save_npz(file, outdir):
    
    """ load data file, be it nifti, gifti or cifti
    and save as npz - (whole brain: ("vertex", TR))
    
    Parameters
    ----------
    file : str/list/array
        absolute filename of ciftis to be decomposed (or list of filenames)
    outdir : str
        path to save new files
    
    Outputs
    -------
    out_file: str
        absolute output filename (or list of filenames)
        
    """
    
    # some params
    hemispheres = ['hemi-L','hemi-R']
    cifti_hemis = {'hemi-L': 'CIFTI_STRUCTURE_CORTEX_LEFT', 
                   'hemi-R': 'CIFTI_STRUCTURE_CORTEX_RIGHT'}
        
    # check if single filename or list of filenames
    
    if isinstance(file, list): 
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

                # save gii, per hemisphere
                # note that surface data is (time, "vertex")
                data = np.vstack([surf_data_from_cifti(cifti_data, axes[1], cifti_hemis[hemi]) for hemi in hemispheres])
            
            elif file_extension == '.func.gii': # load gifti file
                
                print('implement later')
            else:
                print('implement later')
                
            # actually save
            np.save(output_file,data)

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


def get_FA_bar_stim(output, params, bar_pos, trial_info, 
                    attend_cond = {'reg_name': 'ACAO_mblk-0_run-1',
                                   'color': True, 'orientation': True,
                                   'condition_name': 'red_vertical',
                                   'miniblock': 0,'run': 1}, TR = 1.6, crop_unit = 'sec',
                    save_imgs = False, downsample = None, oversampling_time = 1, stim_dur_seconds = 0.5,
                    crop = False, crop_TR = 8, overwrite=False, shift_TRs=True, shift_TR_num = 1, save_DM=True):
    
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
    
    if not op.exists(output) or overwrite == True: 
        print('making %s'%output)

        if not op.exists(op.split(output)[0]): # make base dir to save files
            os.makedirs(op.split(output)[0])
    
        # general infos
        bar_width = params['feature']['bar_width_ratio'] 

        screen_res = params['window']['size']
        if params['window']['display'] == 'square': # if square display
            screen_res = np.array([screen_res[1], screen_res[1]])
            
        if downsample != None: # if we want to downsample screen res
            screen_res = (screen_res*downsample).astype(int)
        
        # miniblock number
        mini_blk_num = int(attend_cond['miniblock'])
        
        # total number of TRs and of trials
        # if oversampling then they will not match
        total_trials = len(trial_info)
        total_TR = total_trials*oversampling_time

        # stimulus duration (how long was each bar on screen)
        stim_dur_TR = stim_dur_seconds*oversampling_time 
        if not stim_dur_TR.is_integer():
            raise ValueError('Stimulus duration is %s TRs, not accepted value'%str(stim_dur_TR)) 

        # save screen display for each TR
        visual_dm_array = np.zeros((total_TR, screen_res[0],screen_res[1]))

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
                hor_x = miniblk_positions[trl_blk_counter][0]; hor_y = miniblk_positions[trl_blk_counter][1]
                
                if downsample != None: # if we want to downsample screen res
                    hor_x = hor_x*downsample; hor_y = hor_y*downsample
                
                hor_x = hor_x + screen_res[0]/2; hor_y = hor_y + screen_res[1]/2
                
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
                if trial_counter == oversampling_time-1:
                    trl_blk_counter += 1
                    # reset stim counter
                    stim_counter = 0

                # if last trial of miniblock
                if trl_blk_counter == len(bar_pos.iloc[0]['bar_pass_direction_at_TR']):
                    # update counters, so we can do same in next miniblock
                    trl_blk_counter = 0

            # save in array
            visual_dm_array[i, ...] = np.array(img)[:,:,0][np.newaxis,...]
            
            # increment trial counter
            trial_counter += 1 
            
            # if sampling time reached,
            if trial_counter == oversampling_time: # then reset counter and update trial
                trial_counter = 0 
                trial += 1

        # swap axis to have time in last axis [x,y,t]
        visual_dm = visual_dm_array.transpose([1,2,0])
        
        # in case we want to crop the beginning of the DM
        if crop == True:
            if crop_unit == 'sec': # fix for the fact that I crop TRs, but task not synced to TR
                visual_dm = visual_dm[...,int(crop_TR*TR*oversampling_time)::] 
            else: # assumes unit is TR
               visual_dm = visual_dm[...,int(crop_TR*oversampling_time)::] 

        # shifting TRs to the left (quick fix)
        # to account for first trigger that was "dummy" - in future change experiment settings to skip 1st TR
        if shift_TRs == True:

            new_visual_dm = visual_dm.copy()

            if crop_unit == 'sec': # fix for the fact that I shift TRs, but task not synced to TR
                new_visual_dm[...,:-int(shift_TR_num*TR*oversampling_time)] = visual_dm[...,int(shift_TR_num*TR*oversampling_time):]
            else: # assumes unit is TR
                new_visual_dm[...,:-int(shift_TR_num*oversampling_time)] = visual_dm[...,int(shift_TR_num*oversampling_time):]
                
            visual_dm = new_visual_dm.copy()

        if save_DM == True: 
            # save design matrix
            np.save(output, visual_dm)
        
    else:
        print('already exists, skipping %s'%output)
        
        # load
        visual_dm = np.load(output)

    #if we want to save the images
    if save_imgs == True:

        #take into account oversampling
        if oversampling_time == 1:
            frames = np.arange(visual_dm.shape[-1])
        else:
            frames = np.arange(0,visual_dm.shape[-1],oversampling_time, dtype=int)  

        outfolder = op.split(output)[0]

        visual_dm = visual_dm.astype(np.uint8)

        for w in frames:
            im = Image.fromarray(visual_dm[...,int(w)])
            im.save(op.join(outfolder,op.split(output)[-1].replace('.npy','_trial-{time}.png'.format(time=str(int(w/oversampling_time)).zfill(3)))))      
            
    return visual_dm


def plot_DM(DM, vertex, output, names=['intercept','ACAO', 'ACUO', 'UCAO', 'UCUO']):
    
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


def create_hrf(hrf_params=[1.0, 1.0, 0.0], TR = 1.2, osf = 1):
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
                time_length = 40)[...,np.newaxis],
            hrf_params[1] *
            spm_time_derivative(
                tr = TR,
                oversampling = osf,
                time_length = 40)[...,np.newaxis],
            hrf_params[2] *
            spm_dispersion_derivative(
                tr = TR,
                oversampling = osf,
                time_length = 40)[...,np.newaxis]]).sum(
        axis=0)                    

    return hrf.T

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
                    select = 'num',num_components = 5):
    
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
        
    Outputs
    -------
    out_file: str
        absolute output filename (or list of filenames)
    
    """
    
    # check if single filename or list of filenames
    
    if isinstance(file, list): 
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
    if not isinstance(file, list): 
        outfiles = outfiles[0] 

    return outfiles


def regressOUT_confounds(file, counfounds, outdir, TR=1.2):
    
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
    
    if isinstance(file, list): 
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
            
            data = np.load(input_file,allow_pickle=True)
            
            # get confound file for that run
            run_id = re.search(r'run-._', input_file).group(0)
            conf_file = [val for val in counfounds if run_id in val][0]

            print('using confounds from %s'%conf_file)

            # load dataframe
            conf_df = pd.read_csv(conf_file, sep="\t")
            
            # clean confounds from data
            filtered_signal = clean(signals = data.T, confounds = conf_df.values, detrend = True, 
                              standardize='psc', standardize_confounds = True, filter = False, t_r = TR)
            
            data_filt = filtered_signal.T
            
            ## save filtered file
            np.save(output_file,data_filt)

        # append out files
        outfiles.append(output_file)
        
    # if input file was not list, then return output that is also not list
    if not isinstance(file, list): 
        outfiles = outfiles[0] 
    
    return outfiles


def get_beh_mask(files,params):
    
    """Make boolean mask 
    to use in design matrix for pRF task
    based on subject responses
    
    Parameters
    ----------
    files : list/array
       list with absolute filenames for all pRF runs
    params : yml dict
        with experiment params
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
      
    # make mask
    mask_bool = []
    for t in range(len(condition_per_TR)):

        if 'empty' in condition_per_TR[t]: # true for empty trials (nothing on screen)

            mask_bool += [True]

        else:
            trial_response = 0

            for _, file in enumerate(files): # iterate over run df, to check for responses
                
                df_run = pd.read_csv(file, sep='\t')

                response = df_run[(df_run['trial_nr']==t)&(df_run['event_type']=='response')]['response'].values

                if (len(response) and response[0] in params['keys']['right_index']+params['keys']['left_index'])>0: 

                    trial_response += 1 # if there was a response, increment counter

            if trial_response < len(files)*.75: # if no response for trial in the majority of runs

                mask_bool += [True]

            else:
                mask_bool += [False] # when something was on screen and they could see it

            
    return np.array(mask_bool)


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

    x_mask = np.clip(x_mask,0,255)
    y_mask = np.clip(y_mask,0,255)
    
    ## get y ecc limits
    y_ecc_limit = [np.clip(np.max(y_mask.nonzero()[0])+1,0,visual_dm.shape[1])*screen_size_deg[1]/visual_dm.shape[1],
                    np.clip(np.min(y_mask.nonzero()[0])-1,0,visual_dm.shape[1])*screen_size_deg[1]/visual_dm.shape[1]]

    y_ecc_limit = screen_size_deg[1]/2 - y_ecc_limit

    ## get x ecc limits
    x_ecc_limit = [np.clip(np.max(x_mask.nonzero()[1])+1,0,visual_dm.shape[0])*screen_size_deg[0]/visual_dm.shape[0],
                   np.clip(np.min(x_mask.nonzero()[1])-1,0,visual_dm.shape[0])*screen_size_deg[0]/visual_dm.shape[0]]

    x_ecc_limit = screen_size_deg[0]/2 - x_ecc_limit

    return x_ecc_limit, y_ecc_limit


def get_cue_regressor(params, trial_info, hrf_params = [1,1,0], cues = [0,1,2,3], TR = 1.6, oversampling_time = 1, baseline = None,
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
    cue_regs_upsampled = np.zeros((1, len(trial_info)*oversampling_time))
    
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

    cue_convolved = signal.fftconvolve(padded_cue, hrf, axes=(-1))[..., pad_length*oversampling_time:cue_regs_upsampled.shape[-1]+pad_length*oversampling_time]  
    
    # save convolved upsampled cue in array
    cue_regs_upsampled = np.array(cue_convolved)#[...,:cue_regs_upsampled.shape[-1]]
    
    ## resample to data sampling rate
    
    # original scale of data in seconds
    original_scale = np.arange(0, cue_regs_upsampled.shape[-1]/oversampling_time, 1/oversampling_time)

    # cubic interpolation of predictor
    interp = interpolate.interp1d(original_scale, 
                                cue_regs_upsampled, 
                                kind = "cubic", axis=-1)
    desired_scale = np.arange(0, cue_regs_upsampled.shape[-1]/oversampling_time, TR) # we want the predictor to be sampled in TR

    cue_regs_RESAMPLED = interp(desired_scale)

    ## filter it, like we do to the data
    cue_regs_RESAMPLED =  dc_data(cue_regs_RESAMPLED,
                                first_modes_to_remove = params['mri']['filtering']['first_modes_to_remove'])
    
    ## make nan vertices that are not to be fitted
    cue_regs = cue_regs_RESAMPLED
    
    # add baseline if baseline array/value specified
    if baseline is not None:
        cue_regs += baseline 
        
    return cue_regs


def get_FA_regressor(fa_dm, params, pRFfit_pars, filter = True, 
                     pRFmodel = 'css', TR = 1.6, hrf_params = [1,1,0], oversampling_time = 1, pad_length=20):
    
    """ Get timecourse for FA regressor, given a dm
    
    Parameters
    ----------
    fa_dm: array
        fa design matrix N x samples (N = (x,y))
    params: yml dict
        with experiment params
    pRFfit_pars: parameters object
        Parameters object from lmfit
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
                        mu = (pRFfit_pars['pRF_x'].value, pRFfit_pars['pRF_y'].value),
                        sigma = pRFfit_pars['pRF_size'].value,
                        normalize_RFs = False).T, axes=(1,2))

    if not 'pRF_n' in pRFfit_pars.keys(): # accounts for gauss or css model
        pRFfit_pars.add('pRF_n', value = 1, vary = False)

    neural_tc = stimulus_through_prf(rf, fa_dm, 1)**pRFfit_pars['pRF_n'].value

    # convolve with hrf
    #scipy fftconvolve does not have padding options so doing it manually
    pad = np.tile(neural_tc[:,0], (pad_length*oversampling_time,1)).T
    padded_cue = np.hstack((pad,neural_tc))

    tc_convolved = signal.fftconvolve(padded_cue, hrf, axes=(-1))[..., pad_length*oversampling_time:neural_tc.shape[-1]+pad_length*oversampling_time]  

    # save convolved upsampled cue in array
    tc_convolved = np.array(tc_convolved)#[...,:cue_regs_upsampled.shape[-1]]

    if filter:
        model_fit = pRFfit_pars['pRF_baseline'].value + pRFfit_pars['pRF_beta'].value * filter_predictions(
                                                                                            tc_convolved,
                                                                                            filter_type = params['mri']['filtering']['type'],
                                                                                            filter_params = {'highpass': params['mri']['filtering']['highpass'],
                                                                                                        'add_mean': params['mri']['filtering']['add_mean'],
                                                                                                        'window_length': params['mri']['filtering']['window_length'],
                                                                                                        'polyorder': params['mri']['filtering']['polyorder']})
    else:
        model_fit = pRFfit_pars['pRF_baseline'].value + pRFfit_pars['pRF_beta'].value * tc_convolved

    
    # original scale of data in seconds
    original_scale = np.arange(0, model_fit.shape[-1]/oversampling_time, 1/oversampling_time)

    # cubic interpolation of predictor
    interp = interpolate.interp1d(original_scale, 
                                model_fit, 
                                kind = "cubic", axis=-1)
    desired_scale = np.arange(0, model_fit.shape[-1]/oversampling_time, TR) # we want the predictor to be sampled in TR

    FA_regressor = interp(desired_scale)
    
    # squeeze out single dimension
    FA_regressor = np.squeeze(FA_regressor)
    
    return FA_regressor
    

def get_residuals_FA(fit_pars, data, bar_dm_dict, params, hrf_params = [1,1,0], 
                     cue_regressor = [], pRFmodel = 'css', ar_noise_model = 1, 
                     TR = 1.6, oversampling_time = 1, num_regs = 2):


    ## set DM - all bars simultaneously on screen, multiplied by weights
    for i, cond in enumerate(bar_dm_dict.keys()):
        
        if i==0:
            gain_dm = bar_dm_dict[cond][np.newaxis,...]*fit_pars['gain_{cond}'.format(cond=cond)].value
        else:
            gain_dm = np.vstack((gain_dm, 
                                bar_dm_dict[cond][np.newaxis,...]*fit_pars['gain_{cond}'.format(cond=cond)].value))
    
    # taking the max value of the spatial position at each time point (to account for overlaps)
    gain_dm = np.amax(gain_dm, axis=0)

    ## get FA regressor
    FA_regressor = get_FA_regressor(gain_dm, params, fit_pars, 
                                    pRFmodel = pRFmodel, TR = TR, hrf_params = hrf_params, oversampling_time = oversampling_time)
    
    ## calculate model
    if num_regs == 1: # if 1 regressor given, then comparing pRF FA regressor directly to data

        prediction = FA_regressor

        # calculate rsq of fit
        r2 = 1 - (np.sum((data - prediction)**2)/ np.sum((data - np.mean(data))**2))  

    else:
        ## Make actual DM to be used in GLM fit (intercept + FA regressor + cue regressor)
        
        DM_FA = np.zeros((FA_regressor.shape[0], num_regs+1))
        
        DM_FA[...,0] = 1 # add intercept in first position
        DM_FA[...,1] = FA_regressor # add FA regressor
        DM_FA[...,2] = cue_regressor # add cue regressor
        
        ## Fit GLM on FA data
        prediction, betas , r2, _ = fit_glm(data, DM_FA)
        
        # update values of pars, as outputed by glm 
        fit_pars['intercept'].set(betas[0])
        fit_pars['FA_beta'].set(betas[1])
        fit_pars['cue_beta'].set(betas[2])

    print('FA GLM fit R2 is %.2f'%r2)

    # OLS residuals
    residuals = data - prediction

    # return error "timecourse", that will be used by minimize
    return  residuals


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


def plot_FA_model(fit_pars, data, bar_dm_dict, params, hrf_params = [1,1,0], 
                     cue_regressor = [], pRFmodel = 'css', 
                     TR = 1.6, oversampling_time = 1, num_regs = 2):
    
    ### REDUNDANT FUNC - JUST FOR QUICK CHECK ###

    ## set DM - all bars simultaneously on screen, multiplied by weights
    for i, cond in enumerate(bar_dm_dict.keys()):
        
        if i==0:
            gain_dm = bar_dm_dict[cond][np.newaxis,...]*fit_pars['gain_{cond}'.format(cond=cond)].value
        else:
            gain_dm = np.vstack((gain_dm, 
                                bar_dm_dict[cond][np.newaxis,...]*fit_pars['gain_{cond}'.format(cond=cond)].value))
    
    # taking the max value of the spatial position at each time point (to account for overlaps)
    gain_dm = np.amax(gain_dm, axis=0)

    ## get FA regressor
    FA_regressor = get_FA_regressor(gain_dm, params, fit_pars, 
                                    pRFmodel = pRFmodel, TR = TR, hrf_params = hrf_params, oversampling_time = oversampling_time)
    
    ## calculate model
    if num_regs == 1: # if 1 regressor given, then comparing pRF FA regressor directly to data

        prediction = FA_regressor

        # calculate rsq of fit
        r2 = 1 - (np.sum((data - prediction)**2)/ np.sum((data - np.mean(data))**2))  

    else:
        ## Make actual DM to be used in GLM fit (intercept + FA regressor + cue regressor)
        
        DM_FA = np.zeros((FA_regressor.shape[0], num_regs+1))
        
        DM_FA[...,0] = 1 # add intercept in first position
        DM_FA[...,1] = FA_regressor # add FA regressor
        DM_FA[...,2] = cue_regressor # add cue regressor
        
        ## Fit GLM on FA data
        prediction, _ , r2, _ = fit_glm(data, DM_FA)
        
    
    ## now plot data with model

    fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)

    # plot data with model
    time_sec = np.linspace(0,len(data)*TR, num=len(data)) # array with timepoints, in seconds

    plt.plot(time_sec, prediction, c='#0040ff',lw=3,label='model R$^2$ = %.2f'%r2,zorder=1)
    plt.plot(time_sec, data,'k--',label='FA data')
    axis.set_xlabel('Time (s)',fontsize=20, labelpad=20)
    axis.set_ylabel('BOLD signal change (%)',fontsize=20, labelpad=10)
    axis.set_xlim(0,len(prediction)*TR)
    axis.legend(loc='upper left',fontsize=10)  # doing this to guarantee that legend is how I want it 
    #axis.set_ylim(-3,3)  

    # times where bar is on screen [1st on, last on, 1st on, last on, etc] 
    bar_onset = np.array([27,98,126,197,225,296,324,395])#/TR

    if params['feature']['crop']:
        bar_onset = bar_onset - params['feature']['crop_TR']*TR - TR

    bar_directions = [val for _,val in enumerate(params['feature']['bar_pass_direction']) if 'empty' not in val and 'cue' not in val]
    # plot axis vertical bar on background to indicate stimulus display time
    ax_count = 0
    for h in range(len(bar_directions)):

        plt.axvspan(bar_onset[ax_count], bar_onset[ax_count+1]+TR, facecolor='#0040ff', alpha=0.1)

        ax_count += 2


