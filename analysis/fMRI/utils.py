
# script to calculate tSNR
# for the different sequences piloted


import numpy as np
import os
from os import path as op
import nibabel as nib
import re

from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

from nilearn.glm.first_level.design_matrix import _cosine_drift as dct_set
from nilearn import signal, surface

from joblib import Parallel, delayed

import matplotlib.pyplot as plt

import cortex
from cortex import fmriprep

from PIL import Image, ImageDraw

from matplotlib import cm
import matplotlib.colors




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

    """ crop epi file
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
        file_extension = '.{a}.{b}'.format(a = input_file.rsplit('.', 2)[-2],
                                   b = input_file.rsplit('.', 2)[-1])
        # set output filename
        output_file = op.join(outdir, 
                    op.split(input_file)[-1].replace(file_extension,'_{name}{ext}'.format(name = 'cropped',
                                                                                           ext = file_extension)))
        # if file already exists, skip
        if op.exists(output_file): 
            print('already exists, skipping %s'%output_file)
        
        else:
            print('making %s'%output_file)
            
            # load file
            nibber = nib.load(input_file)

            # way depends on type of extension
            if file_extension == '.func.gii':
                data = np.array([nibber.darrays[i].data for i in range(len(nibber.darrays))]) #load surface data (time, vertex)
            else:
                affine = nibber.affine
                data = np.array(nibber.get_fdata())
            
            # check shape
            # (note: for nifti files, time in last dim)
            arr_shape = data.shape
            
            #
            if (len(arr_shape)>2 and data.shape[-1]==num_TR_task) or (len(arr_shape)==2 and data.shape[0]==num_TR_task):
                    raise NameError('Data already has shape of task! Sure we need to crop??')
            else:
                # crop initial TRs
                if len(arr_shape)>2:
                    crop_data = data[...,num_TR_crop:] 
                else:
                    crop_data = data[num_TR_crop:,:] 
                    
                print('new file with shape %s' %str(crop_data.shape))
                
            ## save cropped file
            # again, way depends on type of extension
            if file_extension == '.func.gii':
                darrays = [nib.gifti.gifti.GiftiDataArray(d) for d in crop_data]
                output_image = nib.gifti.gifti.GiftiImage(header = nibber.header, 
                                                                  extra = nibber.extra, 
                                                                  darrays = darrays)
            else:
                output_image = nib.nifti1.Nifti1Image(crop_data, affine, header = nibber.header)

            # actually save
            nib.save(output_image,output_file)

        # append out files
        outfiles.append(output_file)
        
    # if input file was not list, then return output that is also not list
    if not isinstance(file, list): 
        outfiles = outfiles[0] 

    return outfiles


def filter_data(file, outdir, filter_type = 'HPgauss', plot_vert=False, **kwargs):
    
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
        file_extension = '.{a}.{b}'.format(a = input_file.rsplit('.', 2)[-2],
                                   b = input_file.rsplit('.', 2)[-1])
        # set output filename
        output_file = op.join(outdir, 
                    op.split(input_file)[-1].replace(file_extension,'_{filt}{ext}'.format(filt = filter_type,
                                                                                           ext = file_extension)))
        # if file already exists, skip
        if op.exists(output_file): 
            print('already exists, skipping %s'%output_file)
        
        else:
            print('making %s'%output_file)
            
            # load file
            nibber = nib.load(input_file)

            # way depends on type of extension
            if file_extension == '.func.gii':
                data = np.array([nibber.darrays[i].data for i in range(len(nibber.darrays))]) #load surface data
            else:
                affine = nibber.affine
                data = np.array(nibber.get_fdata())
 
            ### implement filter types, by calling their specific functions

            if filter_type == 'HPgauss':

                data_filt = gausskernel_data(data, **kwargs)
                
            elif filter_type == 'sg':

                data_filt = savgol_data(data, **kwargs)

            elif filter_type == 'dc': 

                data_filt = dc_data(data, **kwargs) 
                
            else:
                raise NameError('Not implemented')
                
            # if plotting true, make figure of voxel with high variance,
            # to compare the difference
            if plot_vert == True:
                
                ind2plot = np.argwhere(np.std(data, axis=0)==np.max(np.std(data, axis=0)))[0][0]
                fig = plt.figure()
                plt.plot(data[...,ind2plot], color='dimgray',label='Original data')
                plt.plot(data_filt[...,ind2plot], color='mediumseagreen',label='Filtered data')

                plt.xlabel('Time (TR)')
                plt.ylabel('Signal amplitude (a.u.)')
                plt.legend(loc = 'upper right')

                fig.savefig(output_file.replace(file_extension,'_vertex_%i.png'%ind2plot))
            

            ## save filtered file
            # again, way depends on type of extension
            if file_extension == '.func.gii':
                darrays = [nib.gifti.gifti.GiftiDataArray(d) for d in data_filt]
                output_image = nib.gifti.gifti.GiftiImage(header = nibber.header, 
                                                                  extra = nibber.extra, 
                                                                  darrays = darrays)
            else:
                output_image = nib.nifti1.Nifti1Image(data_filt, affine, header = nibber.header)

            # actually save
            nib.save(output_image,output_file)

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

    # reshape to 2D if necessary, to have shape (time, vertex)
    if len(arr_shape)>2:
        data = np.reshape(data, (-1, data.shape[-1])) 
        data = data.T

    # filter signal
    filtered_signal = np.array(Parallel(n_jobs=2)(delayed(gaussian_filter)(i, sigma=sigma) for _,i in enumerate(data))) 

    # add mean image back to avoid distribution around 0
    data_filt = data - filtered_signal + np.mean(filtered_signal, axis=0)

    # put back in original shape
    if len(arr_shape)>2:
        data_filt = data_filt.T.reshape(*arr_shape) 
    
    return data_filt


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
            
    # save shape, for file reshaping later
    arr_shape = data.shape
    
    # reshape to 2D if necessary, to have shape (time, vertex)
    if len(arr_shape)>2:
        data = np.reshape(data, (-1, data.shape[-1])) 
        data = data.T

    # filter signal
    filtered_signal = savgol_filter(data, window_length, polyorder)
    
    # add mean image back to avoid distribution around 0
    data_filt = data - filtered_signal + np.mean(filtered_signal, axis=0)

   # put back in original shape
    if len(arr_shape)>2:
        data_filt = data_filt.T.reshape(*arr_shape) 

    return data_filt


def dc_data(data, cut_off_hz = 0.01, **kwargs):
    
    """ 
    High pass discrete cosine filter array
    
    Parameters
    ----------
    data : arr
        data array
    cut_off_hz: float
        cut off frequency to filter

    Outputs
    -------
    data_filt: arr
        filtered array
    """ 
            
    # save shape, for file reshaping later
    arr_shape = data.shape

    # reshape to 2D if necessary, to have shape (time, vertex)
    if len(arr_shape)>2:
        data = np.reshape(data, (-1, data.shape[-1])) 
        data = data.T

    # filter signal
    ft = np.linspace(0, data.shape[0], data.shape[0], endpoint=False) 
    hp_set = dct_set(cut_off_hz, ft)

    filtered_signal = signal.clean(data, detrend=False,
                            standardize=False, confounds=hp_set)

    # add mean image back to avoid distribution around 0
    data_filt = filtered_signal + np.mean(data, axis=0) 

    # put back in original shape
    if len(arr_shape)>2:
        data_filt = data_filt.T.reshape(*arr_shape)

    return data_filt


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
        file_extension = '.{a}.{b}'.format(a = input_file.rsplit('.', 2)[-2],
                                   b = input_file.rsplit('.', 2)[-1])
        # set output filename
        output_file = op.join(outdir, 
                    op.split(input_file)[-1].replace(file_extension,'_{name}{ext}'.format(name = 'psc',
                                                                                           ext = file_extension)))
        # if file already exists, skip
        if op.exists(output_file): 
            print('already exists, skipping %s'%output_file)
        
        else:
            print('making %s'%output_file)
            
            # load file
            nibber = nib.load(input_file)

            # way depends on type of extension
            if file_extension == '.func.gii':
                data = np.array([nibber.darrays[i].data for i in range(len(nibber.darrays))]) #load surface data (time, vertex)
            else:
                affine = nibber.affine
                data = np.array(nibber.get_fdata())
            
            # check shape
            # (note: for nifti files, time in last dim)
            arr_shape = data.shape
            
            # reshape nii to 2D
            if len(arr_shape)>2:
                data = np.reshape(data, (-1, data.shape[-1])) 
                
                mean_signal = data.mean(axis = -1)[..., np.newaxis] 
                data_psc = (data - mean_signal)/np.absolute(mean_signal)
                data_psc *= 100
                data_psc = data_psc.reshape(*arr_shape)
                
            else:
                data_psc = (data - np.mean(data,axis=0))/np.absolute(np.mean(data,axis=0))
                data_psc *= 100
                
            ## save psc file
            # again, way depends on type of extension
            if file_extension == '.func.gii':
                darrays = [nib.gifti.gifti.GiftiDataArray(d) for d in data_psc]
                output_image = nib.gifti.gifti.GiftiImage(header = nibber.header, 
                                                                  extra = nibber.extra, 
                                                                  darrays = darrays)
            else:
                output_image = nib.nifti1.Nifti1Image(data_psc, affine, header = nibber.header)

            # actually save
            nib.save(output_image,output_file)

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
    
    # get file extension
    file_extension = '.{a}.{b}'.format(a = file_list[0].rsplit('.', 2)[-2],
                                       b = file_list[0].rsplit('.', 2)[-1])
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
            
            # load file
            nibber = nib.load(input_file)

            # way depends on type of extension
            if file_extension == '.func.gii':
                data = np.array([nibber.darrays[i].data for i in range(len(nibber.darrays))]) #load surface data (time, vertex)
            else:
                affine = nibber.affine
                data = np.array(nibber.get_fdata())
            
            # check shape
            # (note: for nifti files, time in last dim)
            arr_shape = data.shape
            
            # reshape to 2D if necessary, to have shape (time, vertex)
            if len(arr_shape)>2:
                data = np.reshape(data, (-1, data.shape[-1])) 
                data = data.T

            all_runs.append(data)
          
        # average all
        if method == 'median':
            avg_data = np.median(all_runs, axis = 0)
            
        elif method == 'mean':
            avg_data = np.mean(all_runs, axis = 0)
            
        # put back in original shape
        if len(arr_shape)>2:
            avg_data = avg_data.T.reshape(*arr_shape) 

        ## save averaged file
        # again, way depends on type of extension
        if file_extension == '.func.gii':
            darrays = [nib.gifti.gifti.GiftiDataArray(d) for d in avg_data]
            output_image = nib.gifti.gifti.GiftiImage(header = nibber.header, 
                                                              extra = nibber.extra, 
                                                              darrays = darrays)
        else:
            output_image = nib.nifti1.Nifti1Image(avg_data, affine, header = nibber.header)

        # actually save
        nib.save(output_image,output_file)


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


def make_pRF_DM(output, params, save_imgs = False, downsample = None, crop = False, crop_TR = 8, overwrite=False):
    
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
        
    #if we want to save the images
    if save_imgs == True:
        outfolder = op.split(output)[0]

        visual_dm = visual_dm.astype(np.uint8)

        for w in range(visual_dm.shape[-1]):
            im = Image.fromarray(visual_dm[...,w])
            im.save(op.join(outfolder,"DM_TR-%i.png"%w))      
            
    return visual_dm


def save_estimates(filename, estimates, mask_indices, orig_shape = np.array([1974,220]), model_type = 'gauss'):
    
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
        
        np.savez(filename,
                 x = final_estimates[..., 0],
                 y = final_estimates[..., 1],
                 size = final_estimates[..., 2],
                 betas = final_estimates[...,3],
                 baseline = final_estimates[..., 4],
                 r2 = final_estimates[..., 5])
    
    elif model_type == 'css':
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

def join_chunks(path, out_name, hemi, chunk_num = 83, fit_model = 'css'):
    """ combine all chunks into one single estimate numpy array (per hemisphere)
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
    print(hemi)
    
    for ch in range(chunk_num):
        
        chunk_name = [x for _,x in enumerate(os.listdir(path)) if hemi in x and fit_model in x and 'chunk-%s'%str(ch+1).zfill(3) in x][0]
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
        else:
            xx = np.concatenate((xx,chunk['x']))
            yy = np.concatenate((yy,chunk['y']))

            size = np.concatenate((size,chunk['size']))

            beta = np.concatenate((beta,chunk['betas']))
            baseline = np.concatenate((baseline,chunk['baseline']))

            if 'css' in fit_model:
                ns = np.concatenate((ns,chunk['ns']))

            rsq = np.concatenate((rsq,chunk['r2']))
    
    print('shape of estimates for hemifield %s is %s'%(hemi,str(xx.shape)))

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


def mask_estimates(estimates, ROI = 'None', fit_model = 'gauss', screen_limit_deg = [6,6], max_size = 15, space = 'fsaverage'):
    
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
    
    xx = np.concatenate((estimates[0]['x'],estimates[1]['x']))
    yy = np.concatenate((estimates[0]['y'],estimates[1]['y']))
       
    size = np.concatenate((estimates[0]['size'],estimates[1]['size']))
    
    beta = np.concatenate((estimates[0]['betas'],estimates[1]['betas']))
    baseline = np.concatenate((estimates[0]['baseline'],estimates[1]['baseline']))
    
    if 'css' in fit_model:
        ns = np.concatenate((estimates[0]['ns'],estimates[1]['ns'])) # exponent of css
    else: #if gauss
        ns = np.ones(xx.shape)

    rsq = np.concatenate((estimates[0]['r2'],estimates[1]['r2']))
    
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
        if xx[i] <= screen_limit_deg[0] and xx[i] >= -screen_limit_deg[0]: # if x within horizontal screen dim
            if yy[i] <= screen_limit_deg[1] and yy[i] >= -screen_limit_deg[1]: # if y within vertical screen dim
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