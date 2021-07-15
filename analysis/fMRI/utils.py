
# script to calculate tSNR
# for the different sequences piloted


import numpy as np
import os, sys
from os import path as op

import cortex
from cortex import fmriprep

import nibabel as nib
import nilearn

from nilearn import plotting
import matplotlib.pyplot as plt

from nilearn.image import mean_img, math_img
from matplotlib import cm

from nipype.interfaces.freesurfer import BBRegister

import pandas as pd
import seaborn as sns

from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed


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


def filter_data(file, outdir, filter_type = 'HPgauss', TR = 1.2, cut_off_hz = 0.01, file_extension = '_HPgauss.nii.gz'):
    
    """ High pass filter NIFTI run with gaussian kernel
    
    Parameters
    ----------
    file : str
        absolute filename to be filtered
    outdir : str
        path to save new file
    filter_type : str
        type of filter to use, defaults to gaussian kernel high pass
    
    Outputs
    -------
    out_file: str
        absolute output filename
    
    """

    if filter_type == 'HPgauss':
    
        sigma = (1/cut_off_hz) / (2 * TR) 
        
        # output filename
        outfile = op.join(outdir,op.split(file)[-1].replace('.nii.gz',file_extension))
        
        if not op.exists(outfile): 
            print('making %s'%outfile)
        
            nibber = nib.load(file)
            affine = nibber.affine
            data = np.array(nibber.dataobj)
            
            # reshape to 2D
            data_reshap = np.reshape(data, (-1, data.shape[-1])) 
            
            # filter signal
            filtered_signal = np.array(Parallel(n_jobs=2)(delayed(gaussian_filter)(i, sigma=sigma) for _,i in enumerate(data_reshap))) 

            # add mean image back to avoid distribution around 0
            data_filt = data_reshap - filtered_signal + filtered_signal.mean(axis = -1)[..., np.newaxis] 
            data_filt = data_filt.reshape(*data.shape)
            
            output_image = nib.nifti1.Nifti1Image(data_filt,affine,header=nibber.header)
            nib.save(output_image,outfile)
            
        else:
            print('already exists, skipping %s'%outfile)

    else:
        raise NameError('Not implemented')

    return outfile


def psc(file, outpth, file_extension = '_psc.nii.gz'):

    """ percent signal change nii file
    Parameters
    ----------
    file : str
        absolute filename for nifti
    outpth: str
        path to save new files
    extension: str
        file extension
    Outputs
    -------
    output: str
        absolute filename for psc nifti
    
    """
    
    # output filename
    output = op.join(outpth,op.split(file)[-1].replace('.nii.gz',file_extension))
    
    if not op.exists(output): 
        print('making %s'%output)

        nibber = nib.load(file)
        affine = nibber.affine
        data = np.array(nibber.dataobj)
        
        # reshape to 2D
        data_reshap = np.reshape(data, (-1, data.shape[-1])) 
        
        # psc signal
        mean_signal = data_reshap.mean(axis = -1)[..., np.newaxis] 
        data_psc = (data_reshap - mean_signal)/np.absolute(mean_signal)
        data_psc *= 100
        data_psc = data_psc.reshape(*data.shape)

        output_image = nib.nifti1.Nifti1Image(data_psc,affine,header=nibber.header)
        nib.save(output_image,output)

    else:
        print('already exists, skipping %s'%output)


    return output


def avg_nii(files, outpth):

    """ percent signal change gii file
    Parameters
    ----------
    files : list
        list of strings with absolute filename for nifti
    out_pth: str
        path to save new files
    extension: str
        file extension
    Outputs
    -------
    output: str
        absolute filename for psc nifti
    """
    
    # sort files
    files.sort()
    # output filename
    output = op.join(outpth,op.split(files[0])[-1].replace('run-1','run-average'))
    
    if not op.exists(output): 
        print('making %s'%output)
        
        for ind, run in enumerate(files):

            nibber = nib.load(run)
            affine = nibber.affine
            data = np.array(nibber.dataobj)
            
            if ind == 0:
                data_avg = data.copy()[np.newaxis,...] 
            else:
                data_avg = np.vstack((data_avg,data.copy()[np.newaxis,...]))

        # average
        data_avg = np.mean(data_avg,axis=0)

        output_image = nib.nifti1.Nifti1Image(data_avg,affine,header=nibber.header)
        nib.save(output_image,output)

    else:
        print('already exists, skipping %s'%output)
        
    return output

    