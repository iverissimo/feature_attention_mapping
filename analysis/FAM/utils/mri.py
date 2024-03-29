## general packages
from filecmp import cmp
import numpy as np
import os
from os import path as op
import pandas as pd
import re, json
from shutil import copy2
import itertools
import glob
import struct

import subprocess

## imaging, processing, stats packages
import nibabel as nib

from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter, fftconvolve
from scipy import fft, interpolate
from scipy.stats import t, norm, linregress

import nilearn
from nilearn import surface, signal

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

from nilearn.glm.first_level.design_matrix import create_cosine_drift

from FAM.utils.general import Utils

import neuropythy

class MRIUtils(Utils):

    def __init__(self):
        
        """__init__
        constructor for utilities mri class 
            
        """
        
    def create_T1mask_from_label(self, sub_id = None, freesurfer_pth = None, sourcedata_pth = None,
                                 roi_name = 'V1', index_arr = [], filename = None, overwrite = False):
        
        """Convert T1w image mask object from custom ROI label files
        Can also mask further in we provide value array (ex: index of prf fit vertices)
        """
        
        if filename is not None and op.isfile(filename) and overwrite == False:
            print('Loading %s'%filename)
            mask_img = nilearn.image.load_img(filename) 
        else:
            print('Making T1w mask for ROI %s'%roi_name)
        
            ## load T1w image for reference
            # path for sourcedata anat files of that participant
            anat_pth = glob.glob(op.join(sourcedata_pth, 'sub-{sj}'.format(sj = sub_id), 
                                        'ses-*', 'anat'))[0]
            T1_filename = [op.join(anat_pth,val) for val in os.listdir(anat_pth) if val.endswith('.nii.gz') and not val.startswith('.') and 'T1w' in val][0]
            print(T1_filename)
            T1_img = nilearn.image.load_img(T1_filename)

            # clear image of data, because we will fill it with mask values
            T1_img_empty = neuropythy.mri.to_image(neuropythy.mri.image_clear(T1_img, fill=0.0), 
                                                                                dtype=np.int32)

            ## import freesurfer subject
            fs_sub = neuropythy.freesurfer.subject(op.join(freesurfer_pth, 
                                                        'sub-{sj}'.format(sj = sub_id)))
            
            # get path to FS labels 
            # and make generic base path str that can be update later
            sub_label_str = op.join(freesurfer_pth, 'sub-{sj}'.format(sj = sub_id), 
                                    'label', '{hemi}.custom.{roi}.label')
            
            # get mask of ROI indices
            mask_ix_l = nilearn.surface.load_surf_data(sub_label_str.format(hemi = 'lh', roi = roi_name))
            mask_ix_r = nilearn.surface.load_surf_data(sub_label_str.format(hemi = 'rh', roi = roi_name))
            
            # join hemi masks
            mask_l = np.zeros(fs_sub.LH.vertex_count)
            mask_l[mask_ix_l] = 1
            mask_r = np.zeros(fs_sub.RH.vertex_count)
            mask_r[mask_ix_r] = 1
            mask_data = [mask_l, mask_r]
            
            # if we provided an index array, mask out vertices that are not relevant
            if len(index_arr) > 0:
                
                mask_data = np.concatenate(mask_data)
                
                # create index mask
                index_mask = np.zeros(mask_data.shape)
                index_mask[list(index_arr)] = 1
                
                # actually mask
                mask_data[index_mask == 0] = 0 
                mask_tuple = tuple([mask_data[:fs_sub.LH.vertex_count],
                                    mask_data[fs_sub.LH.vertex_count:]])
            else:
                mask_tuple = tuple(mask_data)
                
            # make volume mask image
            print('Generating volume...')
            mask_img = fs_sub.cortex_to_image(mask_tuple,
                                            T1_img_empty,
                                            hemi = None,
                                            method = 'nearest',
                                            fill = 0.0)
            
            # save file
            if filename is not None:
                print('saving %s'%filename)
                os.makedirs(op.dirname(filename), exist_ok=True)
                neuropythy.io.save(filename, mask_img)
        
        return mask_img
    
    def resample_T1mask_to_func(self, mask_img = None, bold_filename = None, filename = None, overwrite = False):
        
        """Resample mask image to func data format
        """
        if filename is not None and op.isfile(filename) and overwrite == False:
            print('Loading %s'%filename)
            mask_img = nilearn.image.load_img(filename) 
        else:
            print('Resampling T1w mask to func dim')
            
            mask_img = nilearn.image.resample_to_img(mask_img, 
                                                    bold_filename, interpolation='nearest')
            
            # save file
            if filename is not None:
                print('saving %s'%filename)
                os.makedirs(op.dirname(filename), exist_ok=True)
                neuropythy.io.save(filename, mask_img)
                
        return mask_img
        
    def get_masked_timeseries(self, mask_img = None, bold_filename = None, resample_mask = True,
                                    filename = None, return_arr = True, overwrite = False):
        
        """Resample mask image to func data format
        and then apply mask to get 2D array
        (save data as pd dataframe)
        """
        
        if filename is not None and op.isfile(filename) and overwrite == False:
            print('Loading %s'%filename)
            masked_data_df = pd.read_csv(filename, sep='\t', index_col=['time'], compression='gzip').astype(np.float32)
        else:
            print('Masking data file')
        
            if resample_mask: 
                print('resampling mask image first')
                mask_img = nilearn.image.resample_to_img(mask_img, 
                                                        bold_filename, interpolation='nearest')
                
            masked_data = nilearn.masking.apply_mask(bold_filename, mask_img)
            
            # convert to dataframe
            masked_data_df = pd.DataFrame(masked_data, 
                                        index=pd.Index(np.arange(len(masked_data)), name='time'),
                                        columns = pd.Index(range(masked_data.shape[1]), name='source')).astype(np.float32)

            # save dataframe
            if filename is not None:
                print('saving %s'%filename)
                os.makedirs(op.dirname(filename), exist_ok=True)
                masked_data_df.to_csv(filename, sep='\t', header = True, index = True)
           
        # if we want data as array     
        if return_arr:
            return masked_data_df.to_numpy()
        else:
            return masked_data_df
        
    def combine_mask_imgs(self, mask_filenames_list = [], return_img = True):
        
        """from a list of mask images (binary nii files)
        sum them and normalize, and replace values with new mask
        """
        
        print('combining masks')
        
        # load images 
        mask_imgs_list = [nilearn.image.load_img(img_name).get_fdata().astype(np.float32) for img_name in mask_filenames_list]

        # sum data values
        sum_data = np.sum(mask_imgs_list, axis = 0)
        sum_data[sum_data > 0] = 1 # and make binary again
        
        new_mask_imgs_list = []
        
        # replace and save new file name
        for img_name in mask_filenames_list:
            
            old_mask = nilearn.image.load_img(img_name) 
            new_mask_img = nilearn.image.new_img_like(old_mask, sum_data)
            
            neuropythy.io.save(img_name, new_mask_img)
            
            # append new mask images
            new_mask_imgs_list.append(new_mask_img)
        
        # if we want list of images
        if return_img:
            return new_mask_imgs_list

    def create_atlas_df(self, annot_filename = '', pysub = 'hcp_999999', atlas_name = 'glasser'):

        """ Function to create glasser/wang dataframe
        with ROI names, colors (RGBA) and vertex indices

        Parameters
        ----------
        annot_filename : str 
            absolute name of the parcelation file (defined in MRIObj.atlas_annot)
        pysub: str
            pycortex subject
        atlas_name: str
            atlas name (glasser vs wang)
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
        hemi_vert_num = cortex.db.get_surfinfo(pysub).left.shape[0] 
        
        ## make atlas data frame
        atlas_df = pd.DataFrame({'ROI': [], 'hemi_vertex': [], 'merge_vertex': [], 'hemisphere': [],
                                 'R': [],'G': [],'B': [],'A': []})

        for key in label_dict.keys():

            if label_dict[key][0] != '???': 

                # get vertex indices for whole surface 
                merge_vert = np.where(cifti_data == key)[0] 

                if atlas_name == 'glasser':
                    # name of atlas roi
                    roi_name = re.findall(r'_(.*?)_ROI', label_dict[key][0])[0]
                    # hemisphere
                    hemi = re.findall(r'(.*?)_', label_dict[key][0])[0]
                    hemi_arr = np.tile(hemi, len(merge_vert))
                    # get vertex indices for each hemisphere separately
                    hemi_vert = np.where(cifti_data == key)[0] - hemi_vert_num  if hemi == 'R' else np.where(cifti_data == key)[0]
                    
                elif atlas_name == 'wang':
                    # name of atlas roi
                    roi_name = label_dict[key][0]
                    # hemisphere
                    hemi_arr = np.tile('R', len(merge_vert))
                    hemi_arr[np.where(merge_vert < hemi_vert_num)[0]] = 'L'
                    # get vertex indices for each hemisphere separately
                    hemi_vert = merge_vert.copy()
                    hemi_vert[np.where(merge_vert >= hemi_vert_num)[0]] -= hemi_vert_num

                # fill df
                atlas_df = pd.concat((atlas_df,
                                    pd.DataFrame({'ROI': np.tile(roi_name, len(merge_vert)),
                                                'hemisphere': hemi_arr,
                                                'hemi_vertex': hemi_vert, 
                                                'merge_vertex': merge_vert,
                                                'R': np.tile(label_dict[key][1][0], len(merge_vert)),
                                                'G': np.tile(label_dict[key][1][1], len(merge_vert)),
                                                'B': np.tile(label_dict[key][1][2], len(merge_vert)),
                                                'A': np.tile(label_dict[key][1][3], len(merge_vert))
                                                })), ignore_index=True
                                    )
                
        return atlas_df
    
    def create_sjROI_df(self, sub_id = None, pysub = 'hcp_999999', ROI_list = ['V1', 'V2', 'V3', 'hV4', 'LO', 'V3AB'], 
                            freesurfer_pth = None, use_fs_label = False):
        
        """ Function to create pycortex subject dataframe
        with ROI names and vertex indices

        Parameters
        ----------
        sub_id : str 
            subject ID
        pysub: str
            pycortex subject
        ROI_list: list/array
            relevant roi names
        """

        # if we want to FS labels to get ROIs
        if use_fs_label:
            if freesurfer_pth is None:
                raise NameError('path to freesurfer folder not provided!')
            else:
                # load surface vertices, for each hemi, as dict
                n_verts_dict = self.load_FS_nverts_nfaces(sub_id = sub_id, 
                                                        freesurfer_pth = freesurfer_pth, 
                                                        return_faces = False)
                # number of vertices for left hemisphere (for bookeeping) 
                hemi_vert_num = n_verts_dict['lh']

                # get path to FS labels 
                # and make generic base path str that can be update later
                sub_label_str = op.join(freesurfer_pth, 'sub-{sj}'.format(sj = sub_id), 
                                        'label', '{hemi}.custom.{roi}.label')

        else:
            # subject pycortex folder
            sub_pysub = 'sub-{pp}_{ps}'.format(ps = pysub, pp = sub_id)
            
            # number of vertices in one hemisphere (for bookeeping) 
            hemi_vert_num = cortex.db.get_surfinfo(sub_pysub).left.shape[0] 
            
        ## make subject ROI data frame
        sjROI_df = pd.DataFrame({'ROI': [], 'hemi_vertex': [], 'merge_vertex': [], 'hemisphere': []})

        for roi_name in ROI_list:

            if use_fs_label:
                
                # check if drawn label exists
                roi_label_str = {key: sub_label_str.format(hemi = key, roi = roi_name) for key in ['lh','rh']}

                if op.exists(roi_label_str['lh']) and op.exists(roi_label_str['rh']):

                    # get vertices for each hemisphere
                    hemi_vert_dict = {key: nib.freesurfer.io.read_label(roi_label_str[key]) for key in ['lh','rh']}
                    
                    # get vertex indices for each hemisphere separately 
                    hemi_vert = np.concatenate((hemi_vert_dict['lh'], 
                                                hemi_vert_dict['rh']))
                    # and combined (merge)
                    merge_vert = np.concatenate((hemi_vert_dict['lh'], 
                                                 hemi_vert_dict['rh']+hemi_vert_num))
                    # also add a hemisphere name label for bookeeping
                    hemi_arr = np.concatenate((np.tile('L', len(hemi_vert_dict['lh'])), 
                                               np.tile('R', len(hemi_vert_dict['rh']))))
                else:
                    # add empty lists and raise warning
                    print('WARNING: No label found for {roi} in freesurfer folder, skipping'.format(roi = roi_name))
                    hemi_vert = [None, None]
                    merge_vert = [None, None]
                    hemi_arr = ['L', 'R']
            else:
                # get vertex indices for whole surface 
                merge_vert = cortex.get_roi_verts(sub_pysub, roi = roi_name)[roi_name]

                # hemisphere
                hemi_arr = np.tile('R', len(merge_vert))
                hemi_arr[np.where(merge_vert < hemi_vert_num)[0]] = 'L'
                # get vertex indices for each hemisphere separately
                hemi_vert = merge_vert.copy()
                hemi_vert[np.where(merge_vert >= hemi_vert_num)[0]] -= hemi_vert_num

            # fill df
            sjROI_df = pd.concat((sjROI_df,
                                pd.DataFrame({'ROI': np.tile(roi_name, len(merge_vert)),
                                            'hemisphere': hemi_arr,
                                            'hemi_vertex': hemi_vert, 
                                            'merge_vertex': merge_vert
                                            })), ignore_index=True
                                )
            
        return sjROI_df

    def get_roi_vert(self, allROI_df, roi_list = [], hemi = 'BH'):

        """
        get vertex indices for an ROI (or several)
        as defined by the list of labels
        for a specific hemisphere (or both)
        
        Parameters
        ----------
        allROI_df: df
            dataframe with all ROIs
        roi_list: list
            list of strings with ROI labels to load
        hemi: str
            which hemisphere (LH, RH or BH - both)
        """

        roi_vert = []

        for roi2plot in roi_list:
            if hemi == 'BH':
                verts = allROI_df[allROI_df['ROI'] == roi2plot].merge_vertex.values
            else:
                verts = allROI_df[(allROI_df['ROI'] == roi2plot) & \
                                (allROI_df['hemisphere'] == hemi[0])].merge_vertex.values
            
            verts_list = [np.nan] if len(verts) <= 2 else list(verts.astype(int))
            roi_vert += verts_list

        return np.array(roi_vert)
    
    def get_ROIs_dict(self, sub_id = None, pysub = 'hcp_999999', use_atlas = None, 
                            annot_filename = '', hemisphere = 'BH',
                            ROI_labels = {'V1': ['V1v', 'V1d'], 'V2': ['V2v', 'V2d'],'V3': ['V3v', 'V3d'],
                                          'V3AB': ['V3A', 'V3B'], 'LO': ['LO1', 'LO2'], 'hV4': ['hV4'], 
                                          'IPS0': ['IPS0'], 'IPS1': ['IPS1'], 'IPS2': ['IPS2']},
                            freesurfer_pth = None, use_fs_label = False):

        """ 
        get straightforward dictionary of (ROI, vert) pairs

        Parameters
        ----------
        sub_id : str 
            subject ID
        pysub: str
            pycortex subject
        use_atlas: str
            atlas name (glasser vs wang)
        annot_filename : str 
            absolute name of the parcelation file (defined in MRIObj.atlas_annot)
        hemisphere: str
            which hemisphere (LH, RH or BH - both)
        ROI_labels: list/dict/array
            list of strings with ROI labels to load. can also be dict of list of strings
        """ 

        # if we gave a list, then make it dict, for consistency
        if isinstance(ROI_labels, list) or isinstance(ROI_labels, np.ndarray):
            rlabels_dict = {val: [val] for val in ROI_labels}
        else:
            rlabels_dict = ROI_labels

        if use_atlas is not None:
            # if we want to use atlas, then load it
            allROI_df = self.create_atlas_df(annot_filename = annot_filename, 
                                            pysub = pysub, atlas_name = use_atlas)
        else:
            # if not, load subject hand-drawn rois
            tmp_arr = sorted({x for v in rlabels_dict.values() for x in v})
            allROI_df = self.create_sjROI_df(sub_id = sub_id, pysub = pysub, ROI_list = tmp_arr, 
                                            freesurfer_pth = freesurfer_pth, use_fs_label = use_fs_label)

        # iterate over rois and get vertices
        output_dict = {}
        for rname in rlabels_dict.keys():

            roi_vert_arr = self.get_roi_vert(allROI_df, roi_list = rlabels_dict[rname], 
                                                   hemi = hemisphere)
            if len(roi_vert_arr) > 2:
                output_dict[rname] = roi_vert_arr
            else:
                output_dict[rname] = []
            
        return output_dict       

    def get_estimates_roi_df(self, participant, estimates_pp = {}, ROIs_dict = {}, est_key = 'r2', model = 'gauss', iterative = True):

        """
        Helper function to get estimates dataframe values for each ROI
        will select values based on est key param 
        """

        ## save rsq values in dataframe, for plotting
        df_est = pd.DataFrame({'sj': [], 'index': [], 'ROI': [], 'value': [], 'model': []})

        for roi_name in ROIs_dict.keys():

            # mask estimates
            print('masking sub-{s} estimates for ROI {r}'.format(s = participant, r = roi_name))

            if isinstance(estimates_pp, dict):
                roi_arr = estimates_pp[est_key][ROIs_dict[roi_name]]
            else:
                roi_arr = estimates_pp[ROIs_dict[roi_name]]

            df_est = pd.concat((df_est,
                                pd.DataFrame({'sj': np.tile('sub-{sj}'.format(sj = participant), len(roi_arr)), 
                                            'index': ROIs_dict[roi_name], 
                                            'ROI': np.tile(roi_name, len(roi_arr)), 
                                            'value': roi_arr,
                                            'model': np.tile(model, len(roi_arr)),
                                            'iterative': np.tile(iterative, len(roi_arr))})
                            ))
        return df_est     

    def get_tsnr(self, input_file, return_mean=True, affine=[], hdr=[], filename=None):
        
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
        tsnr = np.mean(data,axis=-1)/np.std(data,axis=-1)
        tsnr[np.where(np.isinf(tsnr))] = np.nan

        # if we want to save image, need to provide an output filename
        if filename:
            tsnr_image = nib.Nifti1Image(tsnr, affine=affine, header=hdr).to_filename(filename)

        if return_mean:
            return np.nanmean(np.ravel(tsnr))
        else:
            return tsnr

    def get_bold_file_list(self, participant, task = 'pRF', ses = 'mean', file_ext = '_cropped_dc_psc.npy',
                                postfmriprep_pth = '', acq_name = 'nordic', run_list = [], hemisphere = 'BH'):

        """
        Helper function to get list of bold file names
        to then be loaded and used

        Parameters
        ----------
        participant: str
            participant ID
        ses: str
            session we are looking at

        """

        ## get list of possible input paths
        # (sessions)
        input_list = glob.glob(op.join(postfmriprep_pth, 'sub-{sj}'.format(sj = participant), 'ses-*'))

        # list with absolute file names to be fitted
        bold_filelist = [op.join(file_path, file) for file_path in input_list for file in os.listdir(file_path) if 'task-{tsk}'.format(tsk = task) in file and \
                        'acq-{acq}'.format(acq = acq_name) in file and file.endswith(file_ext) and not file.startswith('.')]
        
        # if we're not combining sessions
        if isinstance(ses, int) or (isinstance(ses, str) and len(re.findall(r'\d{1,10}', ses))>0):

            ses_key = 'ses-{s}'.format(s = re.findall(r'\d{1,10}', str(ses))[0])
            bold_filelist = [file for file in bold_filelist if ses_key in file]

        # if we only want specific runs
        if len(run_list)>0:
            tmp_boldlist = []
            for rn in run_list:
                tmp_boldlist += [val for val in bold_filelist if 'run-{r}'.format(r=rn) in val]
            
            bold_filelist = tmp_boldlist

        # if we want specific hemisphere
        if hemisphere in ['LH', 'hemi-L', 'left']:
            bold_filelist = [file for file in bold_filelist if 'hemi-L' in file] 
        elif hemisphere in ['RH', 'hemi-R', 'right']:
            bold_filelist = [file for file in bold_filelist if 'hemi-R' in file] 

        ## sort by session and run keys, to avoid mismatchings later on
        bold_filelist.sort(key = lambda x:(re.search(r'ses-._', x).group(0), re.search(r'run-._', x).group(0)))

        return bold_filelist

    def smooth_surface(self, data, pysub = 'hcp_999999', kernel=3, nr_iter=3, normalize = False, hemisphere = 'BH'):

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

        if hemisphere == 'BH':
            lh_data_smooth = lh_surf.smooth(data[:lh_surf_data[0].shape[0]], factor=kernel, iterations=nr_iter)
            rh_data_smooth = rh_surf.smooth(data[lh_surf_data[0].shape[0]:], factor=kernel, iterations=nr_iter)

            if normalize:
                lh_data_smooth /= lh_data_smooth.max()
                rh_data_smooth /= rh_data_smooth.max()

            out_data = np.concatenate((lh_data_smooth,rh_data_smooth), axis=0)
        else:
            if data.shape[0] ==  (lh_surf_data[0].shape[0] + rh_surf_data[0].shape[0]): # if providing data from both hemispheres
                if hemisphere in ['hemi-L', 'LH', 'left']:
                    hemi_data_smooth = lh_surf.smooth(data[:lh_surf_data[0].shape[0]], factor=kernel, iterations=nr_iter)
                else:
                    hemi_data_smooth = rh_surf.smooth(data[lh_surf_data[0].shape[0]:], factor=kernel, iterations=nr_iter)
            else:
                if hemisphere in ['hemi-L', 'LH', 'left']:
                    hemi_data_smooth = lh_surf.smooth(data, factor=kernel, iterations=nr_iter)
                else:
                    hemi_data_smooth = rh_surf.smooth(data, factor=kernel, iterations=nr_iter)

            if normalize:
                hemi_data_smooth /= hemi_data_smooth.max()

            out_data = hemi_data_smooth

        return out_data

    def crop_epi(self, file, outdir = None, num_TR_crop = 5):

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
                if len(data.shape) == 4: # if working with niftis
                    crop_data = data[...,num_TR_crop:] 
                else:
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

    def filter_data(self, file, outdir = None, filter_type = 'HPgauss', plot_vert = False,
                    first_modes_to_remove = 5, baseline_inter1 = None, baseline_inter2 = None, TR = 1.6, 
                    cut_off_hz = 0.01, window_length=201, polyorder=3):
        
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
                
                if len(data.shape) == 4: # if working with niftis
                    
                    orig_shape = data.shape # keep original shape to use later
                    
                    n_voxels = np.prod(data[...,0].shape)
                    n_trs = data.shape[-1]
                    new_data = data.reshape(n_voxels, n_trs) # (voxels by time)
                else:
                    new_data = data.copy()
    
                ### implement filter types, by calling their specific functions
                if filter_type == 'HPgauss':

                    data_filt = self.gausskernel_data(new_data, TR = TR, cut_off_hz = cut_off_hz)
                    
                elif filter_type == 'sg':

                    data_filt = self.savgol_data(new_data, window_length = window_length, polyorder = polyorder)

                elif filter_type == 'dc': 

                    data_filt = self.dc_data(new_data, first_modes_to_remove = first_modes_to_remove) 

                elif filter_type == 'LinDetrend':

                    data_filt = self.detrend_data(new_data, detrend_type = 'linear', TR = TR, 
                                                        baseline_inter1 = baseline_inter1, baseline_inter2 = baseline_inter2)
                    
                else:
                    raise NameError('filter type not implemented')
                    
                # if plotting true, make figure of vertix with high tSNR,
                # to compare the difference
                if plot_vert:

                    tsnr = self.get_tsnr(new_data, return_mean=False)
                    
                    ind2plot = np.where(tsnr == np.nanmax(tsnr))[0][0]
                    fig = plt.figure()
                    plt.plot(new_data[ind2plot,...], color='dimgray',label='Original data')
                    plt.plot(data_filt[ind2plot,...], color='mediumseagreen',label='Filtered data')

                    plt.xlabel('Time (TR)')
                    plt.ylabel('Signal amplitude (a.u.)')
                    plt.legend(loc = 'upper right')

                    fig.savefig(output_file.replace(file_extension,'_vertex_%i.png'%ind2plot))
                
                ## save filtered file
                if len(data.shape) == 4: # if working with niftis
                    data_filt = data_filt.reshape(orig_shape)
                
                np.save(output_file,data_filt)

            # append out files
            outfiles.append(output_file)
            
        # if input file was not list, then return output that is also not list
        if not isinstance(file, list) and not isinstance(file, np.ndarray): 
            outfiles = outfiles[0] 
        
        return outfiles

    def gausskernel_data(self, data, TR = 1.6, cut_off_hz = 0.01):
        
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
        
        sigma = (1/cut_off_hz) / (2 * TR) 

        # filter signal
        filtered_signal = np.array(Parallel(n_jobs=2)(delayed(gaussian_filter)(i, sigma=sigma) for _,i in enumerate(tqdm(data.T)))) 

        # add mean image back to avoid distribution around 0
        data_filt = data.T - filtered_signal + np.mean(filtered_signal, axis=0)
        
        return data_filt.T # to be again vertex, time

    def savgol_data(self, data, window_length=201, polyorder=3):
        
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

    def dc_data(self, data, first_modes_to_remove=5):
        
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

    def detrend_data(self, data, detrend_type = 'linear', baseline_inter1 = None, baseline_inter2 = None, 
                                dct_set_nr = 1, TR = 1.6, cut_off_hz = 0.01):
        
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
                    start_trend = np.stack([start_trend_func[ind].intercept + start_trend_func[ind].slope * np.arange(baseline_inter1) for ind,_ in enumerate(tqdm(data))])
                else:
                    raise ValueError('index not provided as int')

                ## end baseline interval
                print('Taking TR indices of %s to get trend of end baseline'%str(baseline_inter2))

                # make regressor + intercept for end baseline period
                if isinstance(baseline_inter2, int):
                    end_trend_func = [linregress(np.arange(np.abs(baseline_inter2)), vert[baseline_inter2:]) for _,vert in enumerate(tqdm(data))]
                    end_trend = np.stack([end_trend_func[ind].intercept + end_trend_func[ind].slope * np.arange(np.abs(baseline_inter2)) for ind,_ in enumerate(tqdm(data))])
                else:
                    raise ValueError('index not provided as int')  

                ## now interpolate slope for task period
                baseline_timepoints = np.concatenate((np.arange(data.shape[-1])[:baseline_inter1, ...],
                                                    np.arange(data.shape[-1])[baseline_inter2:, ...]))

                trend_func = [interpolate.interp1d(baseline_timepoints, np.concatenate((start_trend[ind], end_trend[ind])), kind='linear') for ind,_ in enumerate(tqdm(data))]

                # and save trend line
                trend_line = np.stack([trend_func[ind](np.arange(data.shape[-1])) for ind,_ in enumerate(tqdm(data))])

            # just make line across time series
            else:
                raise ValueError('index not provided')  
            
            # add mean image back to avoid distribution around 0
            data_detr = data - trend_line + np.mean(data, axis=-1)[..., np.newaxis]

        ## if we want remove basis set
        elif detrend_type == 'dct':

            frame_times = np.linspace(0, data.shape[-1]*TR, data.shape[-1], endpoint=False)
            dc_set = create_cosine_drift(high_pass = cut_off_hz, frame_times=frame_times)

            raise NameError('Not implemented yet')


        return data_detr # vertex, time

    def psc_epi(self, file, outdir = None):

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

    def average_epi(self, file, outdir = None, method = 'mean'):

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

    def reorient_nii_2RAS(self, input_pth, output_pth):

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
        os.makedirs(output_pth, exist_ok=True)

        # list of original niftis
        orig_nii_files = [op.join(input_pth, val) for val in os.listdir(input_pth) if val.endswith('.nii.gz') and not val.startswith('.')]

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

    def convert64bit_to_16bit(self, input_file, output_file):

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

    def convert_parrec2nii(self, input_dir, output_dir, cmd = None):

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

    def surf_data_from_cifti(self, data, axis = None, surf_name = '', medial_struct=False):

        """
        load surface data from cifti, from one hemisphere
        taken from https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb

        Parameters
        ----------
        data : array
            cifti data, from cifti.get_fdata
        axis:
            relevant data axis, from cifti.get_axis
        surf_name: str
            surface name to load data from
        medial_struct: bool
            medial wall vertices

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

    def load_data_save_npz(self, file, outdir = None, save_subcortical=False, hemispheres = ['hemi-L','hemi-R'],
                           cifti_hemis = {'hemi-L': 'CIFTI_STRUCTURE_CORTEX_LEFT', 'hemi-R': 'CIFTI_STRUCTURE_CORTEX_RIGHT'},
                           subcortical_hemis = ['BRAIN_STEM', 'ACCUMBENS', 'AMYGDALA', 'CAUDATE', 'CEREBELLUM', 
                            'DIENCEPHALON_VENTRAL', 'HIPPOCAMPUS', 'PALLIDUM', 'PUTAMEN', 'THALAMUS']):
        
        """ load data file (nifti, gifti or cifti)
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

        # make sub-folder to save other files
        if save_subcortical:
            subcort_dir = op.join(outdir, 'subcortical')
            os.makedirs(subcort_dir, exist_ok=True)
            
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
                    data = np.vstack([self.surf_data_from_cifti(cifti_data, axis = axes[1], surf_name = cifti_hemis[hemi]) for hemi in hemispheres])

                    if save_subcortical:
                        print('also saving subcortical structures in separate folder')
                        subcort_dict = {}

                        #for name in subcortical_hemis:
                        for name,_,_ in axes[-1].iter_structures():
                            if 'CORTEX' not in name:
                                print('saving data for %s'%name)
                                subcort_dict[name] = self.surf_data_from_cifti(cifti_data, axis = axes[1], surf_name = name, medial_struct=True)
                        
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
    
    def convert_npz_nifti(self, file = None, nifti_path = None):
        
        """Load numpy file array (or list of) and save as nifit,
        given affine and shape of original nifti
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
            
            # set output filename
            output_file = input_file.replace('.npy', '.nii.gz')
            
            # if file already exists, skip
            if op.exists(output_file): 
                print('already exists, skipping %s'%output_file)
            else:
                print('making %s'%output_file)  
                
                # load data
                data = np.load(input_file, allow_pickle=True)
            
                # get file reference string, to find original nifti
                file_reference = re.findall("(.*?)bold_nii", op.split(input_file)[-1])[0]
                nifti_ref = op.join(nifti_path, file_reference+'bold.nii.gz')
                
                print('Loading affine from %s'%nifti_ref)
                img = nib.load(nifti_ref)
                
                # make new nifti and save
                new_img = nib.Nifti1Image(data.astype(np.float32) , img.affine, img.header.set_data_dtype(np.float32))
                nib.save(new_img, output_file)
                
            # append out files
            outfiles.append(output_file)
            
        # if input file was not list, then return output that is also not list
        if not isinstance(file, list): 
            outfiles = outfiles[0] 

        return outfiles
        
    def select_confounds(self, file, outdir = None, reg_names = ['a_comp_cor','cosine','framewise_displacement'],
                        CumulativeVarianceExplained = 0.4, num_TR_crop = 5, select = 'num', num_components = 5):
        
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

    def regressOUT_confounds(self, file, counfounds = None, outdir = None, TR=1.6, plot_vert = False, 
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
                filtered_signal = signal.clean(signals = data.T, confounds = conf_df.values, detrend = detrend, 
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

    def fit_glm_tc(self, voxel, dm):
    
        """ GLM fit on timeseries
        Regress a created design matrix on the input_data.

        Parameters
        ----------
        voxel : arr
            timeseries of a single voxel
        dm : arr
            DM array (#TR,#regressors)
        """

        if np.isnan(voxel).any():
            betas = np.repeat(np.nan, dm.shape[-1])
            prediction = np.repeat(np.nan, dm.shape[0])
            mse = np.nan
            r2 = np.nan

        else:   # if not nan (some vertices might have nan values)
            betas = np.linalg.lstsq(dm, voxel, rcond = -1)[0]
            prediction = dm.dot(betas)

            mse = np.mean((voxel - prediction) ** 2) # calculate mean of squared residuals
            r2 = np.nan_to_num(1 - (np.nansum((voxel - prediction)**2, axis=0)/ np.nansum(((voxel - np.mean(voxel))**2), axis=0)))# and the rsq
        
        return prediction, betas, r2, mse

    def set_contrast(self, dm_col, tasks, contrast_val = [1], num_cond = 1):
    
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

    def design_variance(self, X, which_predictor=[]):
        
        """Returns the design variance of a predictor (or contrast) in X.

        Parameters
        ----------
        X : numpy array
            Array of shape (N, P)
        which_predictor : list/array
            contrast-vector of the predictors you want the design var from.

        Outputs
        -------
        des_var : float
            Design variance of the specified predictor/contrast from X.
        """

        idx = np.array(which_predictor) != 0

        c = np.zeros(X.shape[1])
        c[idx] = which_predictor[idx]
        des_var = c.dot(np.linalg.pinv(X.T.dot(X))).dot(c.T)

        return des_var

    def calc_contrast_stats(self, betas = [], contrast = [], 
                                sse = None, df = None, design_var = None, pval_1sided = True):

        """Calculates stats for a given contrast and beta values
        
        Parameters
        ----------
        betas : arr
            array of beta values
        contrast: arr/list
            contrast vector       
        sse: float
            sum of squared errors between model prediction and data
        df: int
            degrees of freedom (timepoints - predictores)   
        design_var: float
            design variance 
        pval_1sided: bool
            if we want one or two sided p-value

        """
        # t statistic for vertex
        t_val = contrast.dot(betas) / np.sqrt((sse/df) * design_var)

        if pval_1sided == True:
            # compute the p-value (right-tailed)
            p_val = scipy.stats.t.sf(t_val, df) 

            # z-score corresponding to certain p-value
            z_score = scipy.stats.norm.isf(np.clip(p_val, 1.e-300, 1. - 1.e-16)) # deal with inf values of scipy
        else:
            # take the absolute by np.abs(t)
            p_val = scipy.stats.t.sf(np.abs(t_val), df) * 2 # multiply by two to create a two-tailed p-value

            # z-score corresponding to certain p-value
            z_score = scipy.stats.norm.isf(np.clip(p_val/2, 1.e-300, 1. - 1.e-16)) # deal with inf values of scipy

        return t_val,p_val,z_score

    def load_FS_nverts_nfaces(self, sub_id = None, freesurfer_pth = None, return_faces = False):
        
        """
        Load the number of vertices and faces in a given mesh
        (Adapted from https://github.com/gallantlab/pycortex)
        and return dict for both hemispheres

        Parameters
        ----------
        sub_id : str 
            subject ID
        freesurfer_pth: str
            absolute path to freesurfer files
        return_faces: bool
            if we also want to return faces, or only vertices 
        """    
        n_faces = []
        n_verts = []
        for i in ['lh', 'rh']:
            surf = op.join(freesurfer_pth, 'sub-{sj}'.format(sj = sub_id), 'surf', f'{i}.inflated')
            with open(surf, 'rb') as fp:
                #skip magic
                fp.seek(3)
                fp.readline()
                comment = fp.readline()            
                i_verts, i_faces = struct.unpack('>2I', fp.read(8))
                n_verts.append(i_verts)    
                n_faces.append(i_faces)    

        n_verts_dict = {'lh': n_verts[0], 'rh': n_verts[1]}
        n_faces_dict = {'lh': n_faces[0], 'rh': n_faces[1]}

        if return_faces:
            return n_verts_dict, n_faces_dict
        else:
            return n_verts_dict

    def FS_write_curv(self, fn = None, curv = None, fnum = None):
        
        """
        Writes a freesurfer .curv file 
        (based on Marcus Daghlian's implementation, credits go to him)

        Parameters
        ------------
        fn: str
            File name to be written
        curv: ndaray
            Data array to be written
        fnum: int
            Number of faces in the mesh
        """
        
        NEW_VERSION_MAGIC_NUMBER = 16777215
        vnum = len(curv)
        with open(fn, 'wb') as f:
            self.write_3byte_integer(f, NEW_VERSION_MAGIC_NUMBER)
            f.write(struct.pack(">i", int(vnum)))
            f.write(struct.pack('>i', int(fnum)))
            f.write(struct.pack('>i', 1))
            f.write(curv.astype('>f').tobytes())
            f.close()

    def write_3byte_integer(self, f, n):
        
        """"
        helper function for FS_write_curv 
        """
        b1 = struct.pack('B', (n >> 16) & 255)
        b2 = struct.pack('B', (n >> 8) & 255)
        b3 = struct.pack('B', (n & 255))
        f.write(b1)
        f.write(b2)
        f.write(b3)

    def get_register_dat(self, freesurfer_mri_pth = None, mov_file = 'rawavg.mgz', 
                                targ_file = 'orig.mgz', overwrite = False):
        
        """Create freesurfer register.dat file 
        for a given participant with tkregisterfv
        """
        
        ## get FS orig files (structural mgz files)
        mov = op.join(freesurfer_mri_pth, mov_file)
        targ = op.join(freesurfer_mri_pth, targ_file)

        # where to save register.dat file
        out_file = op.join(op.dirname(targ), 'register.dat')

        ## check if register.dat file already in dir
        if op.isfile(out_file) and overwrite == False:
            print('will not overwrite register.dat file')
        else:
            ## write command for tkregisterfv
            cmd = f"tkregisterfv --mov {mov} --targ {targ} --reg {out_file} --no-config --regheader".format(mov = mov,
                                                                                                    targ = targ,
                                                                                                    out_file = out_file)
            os.system(cmd)
            
        ## actually read file
        with open(out_file) as f:
            d = f.readlines()[4:-1]
        regist_arr = np.array([[float(s) for s in dd.split() if s] for dd in d])
        
        return regist_arr
        
    def get_vox2ras_tkr(self, mov = None):
        
        """Get transform vox2ras-tkr matrix from an image (Torig/Tmov on the FreeSurfer wiki)
        """
        
        mri_cmd = ('mri_info', '--vox2ras-tkr', mov)
        print(mri_cmd)
        
        # get torig matrix
        L = subprocess.check_output(mri_cmd).splitlines()
        torig = np.array([[np.float(s) for s in ll.split() if s] for ll in L])

        return torig
    
    def get_ras2vox(self, img = None):

        """
        fetch the ras2vox matrix from an image (Rorig on the FreeSurfer wiki)
        """

        cmd = ('mri_info', '--ras2vox', img)
        L = subprocess.check_output(cmd).splitlines()
        rorig = np.array([[np.float(s) for s in ll.split() if s] for ll in L])

        return rorig
    
    def get_vox2ras(self, img = None):

        """fetch the vox2ras matrix from an image (Norig on the FreeSurfer wiki)"""

        cmd = ('mri_info', '--vox2ras', img)
        L = subprocess.check_output(cmd).splitlines()
        norig = np.array([[np.float(s) for s in ll.split() if s] for ll in L])

        return norig
        

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
    
    def fwhmax_fwatmin(self, model, amplitude = None, size = None, sa = None, ss = None, ns = None, normalize_RFs=False, return_profiles=False):
    
        """
        taken from marco aqil's code, all credits go to him
        """
        
        model = model.lower()
        x=np.linspace(-50,50,1000).astype('float32')

        prf = amplitude * np.exp(-0.5*x[...,np.newaxis]**2 / size**2)
        vol_prf =  2*np.pi*size**2

        if 'dog' in model:
            srf = sa * np.exp(-0.5*x[...,np.newaxis]**2 / ss**2)
            vol_srf = 2*np.pi*ss*2

        if normalize_RFs==True:

            if model == 'gauss':
                profile =  prf / vol_prf
            elif model == 'css':
                #amplitude is outside exponent in CSS
                profile = (prf / vol_prf)**ns * amplitude**(1 - ns)
            elif model =='dog':
                profile = prf / vol_prf - \
                        srf / vol_srf
        else:
            if model == 'gauss':
                profile = prf
            elif model == 'css':
                #amplitude is outside exponent in CSS
                profile = prf**ns * amplitude**(1 - ns)
            elif model =='dog':
                profile = prf - srf 

        half_max = np.max(profile, axis=0)/2
        fwhmax = np.abs(2*x[np.argmin(np.abs(profile-half_max), axis=0)])

        if 'dog' in model:

            min_profile = np.min(profile, axis=0)
            fwatmin = np.abs(2*x[np.argmin(np.abs(profile-min_profile), axis=0)])

            result = fwhmax, fwatmin
        else:
            result = fwhmax

        if return_profiles:
            return result, profile.T
        else:
            return result





