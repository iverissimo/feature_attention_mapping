
import numpy as np
import os, sys
import os.path as op
import yaml

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import nibabel as nb

from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter

from utils import * #import script to use relevante functions

import datetime


# load settings from yaml
with open(os.path.join(os.path.split(os.getcwd())[0],'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)


# define participant number, and if looking at nordic or pre-nordic data
if len(sys.argv)<6: 
    raise NameError('Please add subject number (ex: 001) '
                    'as 1st argument in the command line!')
elif len(sys.argv)<5: 
    raise NameError('Please specify where running data (local vs lisa)'
                    'as 2nd argument in the command line!')
elif len(sys.argv)<4: 
    raise NameError('Please specify data we are looking at (nordic vs standard)'
                    'as 3rd argument in the command line!')
elif len(sys.argv)<3: 
    raise NameError('Please specify acquisition type (ex: ORIG)'
                    'as 4th argument in the command line!')
elif len(sys.argv)<2: 
    raise NameError('Please data slice number (ex: 00)'
                    'as 5th argument in the command line!')

else:
    sj = str(sys.argv[1]).zfill(3) #fill subject number with 00 in case user forgets
    base_dir = str(sys.argv[2]) # which machine we run the data
    preproc = str(sys.argv[3]) # if using standard files or nordic files
    acq_type = str(sys.argv[4]) # acquisition type to analyse
    slice_num = str(sys.argv[5]).zfill(2) #fill subject number with 00 in case user forgets


# print start time, for bookeeping
start_time = datetime.datetime.now()

# define paths
derivatives_pth = params['mri']['paths'][base_dir][preproc]
input_pth = op.join(derivatives_pth, params['mri']['fitting']['pRF']['input_file_dir']) # for input files to fit model
output_pth =  op.join(derivatives_pth,'pRF_fitting') # for estimates output

if not op.exists(output_pth):
    print('output dir does not existing, saving files in %s'%output_pth)
    os.makedirs(output_pth)

DM_pth = op.join(output_pth,'DM') # design matrix


# load data

run = params['mri']['fitting']['pRF']['runs']
space = params['mri']['fitting']['pRF']['space']
file_ext = params['mri']['fitting']['pRF']['input_file_ext']

input_file = op.join(input_pth,
                     'sub-{sub}_ses-01_task-PRF_acq-{acq}_{run}_space-{space}_desc-preproc_bold_{file_ext}'.format(
                    sub = sj, acq = acq_type, run = run, space = space, file_ext = file_ext))

# load whole brain data
data_img = nb.load(input_file).get_fdata()

# mask data for slice we are going to fit

# determine voxel indices, to associate to estimates and use for reconstruct later
vox_indices = [(xx,yy,int(slice_num)) for xx in range(data_img.shape[0]) for yy in range(data_img.shape[1])]
# number of voxels in slice
n_voxels = np.prod(data_img.shape[:-2])
# reshape slice data to 2D, because of fitting format
data_slice = data_img[:,:,int(slice_num),:]
data_2d = np.reshape(data_slice, (n_voxels, data_slice.shape[-1]))

# define non nan voxels for sanity check
not_nan_vox = np.where(~np.isnan(data_2d[...,0]))[0]

print("z-Slice {slice_num} containing {not_nan}/{n_voxels} non-nan voxels".format(slice_num = slice_num, 
                                                                                  n_voxels = n_voxels,
                                                                                 not_nan = len(not_nan_vox)))

# if there are nan voxels
if len(not_nan_vox)>0:
    # mask them out
    # to avoid errors in fitting (all nan batches) and make fitting faster
    data_2d = data_2d[not_nan_vox]
    vox_indices = list(map(tuple, np.array(vox_indices)[not_nan_vox]))


# define filenames for grid and search estimates

#filename the estimates of the grid fit
grid_estimates_filename = op.split(input_file)[-1].replace('.nii.gz','_estimates-gaussgrid_slice-{slice_num}.nii.gz'.format(slice_num=slice_num))
grid_estimates_filename = op.join(output_pth,grid_estimates_filename)

#filename the estimates of the iterative fit
it_estimates_filename = op.split(input_file)[-1].replace('.nii.gz','_estimates-gaussit_slice-{slice_num}.nii.gz'.format(slice_num=slice_num))
it_estimates_filename = op.join(output_pth,it_estimates_filename)

if op.exists(it_estimates_filename): # if file exists, skip
    print('already exists %s'%it_estimates_filename)

else:

    if len(not_nan_vox)==0: # if all voxels nan, skip fitting completely
        estimates_grid = np.zeros((n_voxels,6)); estimates_grid[:] = np.nan
        estimates_it = np.zeros((n_voxels,6)); estimates_it[:] = np.nan

    else:
        # define design matrix 
        visual_dm = make_pRF_DM(op.join(DM_pth, 'DMprf.npy'), params, save_imgs=False, downsample=0.1)

        prf_stim = PRFStimulus2D(screen_size_cm = params['monitor']['width'],
                                 screen_distance_cm = params['monitor']['distance'],
                                 design_matrix = visual_dm,
                                 TR = params['mri']['TR'])


        # define model 
        gauss_model = Iso2DGaussianModel(stimulus = prf_stim)
        # and parameters
        grid_nr = params['mri']['fitting']['pRF']['grid_nr']
        sizes = params['mri']['fitting']['pRF']['max_size'] * \
            np.linspace(np.sqrt(params['mri']['fitting']['pRF']['min_size']/params['mri']['fitting']['pRF']['max_size']),1,grid_nr)**2
        eccs = params['mri']['fitting']['pRF']['max_eccen'] * \
            np.linspace(np.sqrt(params['mri']['fitting']['pRF']['min_eccen']/params['mri']['fitting']['pRF']['max_eccen']),1,grid_nr)**2
        polars = np.linspace(0, 2*np.pi, grid_nr)


        ## GRID FIT
        print("Grid fit")
        gauss_fitter = Iso2DGaussianFitter(data = data_2d, 
                                           model = gauss_model, 
                                           n_jobs = 16)
                                           
        gauss_fitter.grid_fit(ecc_grid = eccs, 
                              polar_grid = polars, 
                              size_grid = sizes, 
                              pos_prfs_only = True)


        estimates_grid = gauss_fitter.gridsearch_params


        ## ITERATIVE FIT
        # to set up parameter bounds in iterfit
        inf = np.inf
        eps = 1e-1
        ss = prf_stim.screen_size_degrees
        xtol = 1e-7
        ftol = 1e-6

        # model parameter bounds
        gauss_bounds = [(-2*ss, 2*ss),  # x
                        (-2*ss, 2*ss),  # y
                        (eps, 2*ss),  # prf size
                        (0, +inf),  # prf amplitude
                        (-5, +inf)]  # bold baseline


        # iterative fit
        print("Iterative fit")
        gauss_fitter.iterative_fit(rsq_threshold = 0.05, 
                                   verbose = False,
                                   bounds=gauss_bounds,
                                   xtol = xtol,
                                   ftol = ftol)


        estimates_it = gauss_fitter.iterative_search_params

    # save estimates
    # for grid
    save_estimates(grid_estimates_filename, estimates_grid, vox_indices, input_file)
    # for it
    save_estimates(it_estimates_filename, estimates_it, vox_indices, input_file)


# Print duration
end_time = datetime.datetime.now()
print("\nStart time:\t{start_time}\nEnd time:\t{end_time}\nDuration:\t{dur}".format(
                start_time = start_time,
                end_time = end_time,
                dur  = end_time - start_time))

























