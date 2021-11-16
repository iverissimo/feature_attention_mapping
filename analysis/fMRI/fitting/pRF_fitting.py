
import numpy as np
import os, sys
import os.path as op
import yaml
from pathlib import Path

import nibabel as nb

# requires pfpy to be installed - preferably with python setup.py develop
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter

sys.path.insert(0,'..') # add parent folder to path
from utils import * #import script to use relevante functions

import datetime

# load settings from yaml
with open(op.join(str(Path(os.getcwd()).parents[1]),'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

# define participant number, run and which chunk of data to fitted

if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')
  
elif len(sys.argv) < 3:
    raise NameError('Please add type of run to be fitted (ex: leave_01_out vs median) '
                    'as 2nd argument in the command line!')
    
elif len(sys.argv) < 4:
    raise NameError('Please add data chunk number to be fitted '
                    'as 3rd argument in the command line!')

else:
    # fill subject number and chunk number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(2)
    run_type = str(sys.argv[2])
    chunk_num = str(sys.argv[3]).zfill(3)


# print start time, for bookeeping
start_time = datetime.datetime.now()

# some settings
base_dir = params['general']['current_dir'] # which machine we run the data
acq = params['mri']['acq'] # if using standard files or nordic files
space = params['mri']['space'] # subject space
total_chunks = params['mri']['fitting']['pRF']['total_chunks'][space] # number of chunks that data was split in
hemispheres = ['hemi-L','hemi-R'] # only used for gifti files

TR = params['mri']['TR']

# type of model to fit
model_type = params['mri']['fitting']['pRF']['fit_model']

# define file extension that we want to use, 
# should include processing key words
file_ext = '_cropped_{filt}_{stand}.{a}.{b}'.format(filt = params['mri']['filtering']['type'],
                                                    stand = 'psc',
                                                    a = params['mri']['file_ext'].rsplit('.', 2)[-2],
                                                    b = params['mri']['file_ext'].rsplit('.', 2)[-1])

# set paths
derivatives_dir = op.join(params['mri']['paths'][base_dir]['root'],'derivatives')
postfmriprep_dir = op.join(derivatives_dir,'post_fmriprep','sub-{sj}'.format(sj=sj),space,'processed')

output_dir =  op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), model_type,'run-{run}'.format(run=run_type))

# check if path to save processed files exist
if not op.exists(output_dir): 
    os.makedirs(output_dir) 
    #also make gauss dir, to save intermediate estimates
    if model_type!='gauss':
        os.makedirs(output_dir.replace(model_type,'gauss')) 

# send message to user
print('fitting functional files from %s'%postfmriprep_dir)

# list with absolute file names to be fitted (iff gii, then 2 hemispheres)
proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-pRF' in h and
                 'acq-{acq}'.format(acq=acq) in h and run_type in h and h.endswith(file_ext)]

# fit model
for w, file in enumerate(proc_files):

    ### define filenames for grid and search estimates

    # absolute filename for the estimates of the grid fit
    grid_estimates_filename = file.replace('.{a}.{b}'.format(a = params['mri']['file_ext'].rsplit('.', 2)[-2], b = params['mri']['file_ext'].rsplit('.', 2)[-1]),
                                           '_chunk-%s_of_%s_gauss_estimates.npz'%(str(chunk_num).zfill(3), str(total_chunks).zfill(3)))
    grid_estimates_filename = op.join(output_dir.replace(model_type,'gauss'), op.split(grid_estimates_filename)[-1])

    # absolute filename for the estimates of the iterative fit
    it_estimates_filename = grid_estimates_filename.replace('gauss_estimates.npz', 'itgauss_estimates.npz')
    it_estimates_filename = op.join(output_dir.replace('/'+model_type,'/iterative_gauss'), op.split(it_estimates_filename)[-1])
    
    if not op.exists(op.split(it_estimates_filename)[0]): # check if path to save iterative files exist
        os.makedirs(op.split(it_estimates_filename)[0]) 

    if model_type == 'css':
        #filename the estimates of the css fit
        css_grid_estimates_filename = op.join(output_dir,
                                        op.split(grid_estimates_filename)[-1].replace('gauss_estimates.npz', 'css_estimates.npz'))
        
        css_it_estimates_filename = it_estimates_filename.replace('itgauss_estimates.npz', 'itcss_estimates.npz')
        css_it_estimates_filename = op.join(output_dir.replace('/'+model_type,'/iterative_css'), op.split(css_it_estimates_filename)[-1])
      
    ### now actually fit the data, if it was not fit before
    
    if (op.exists(it_estimates_filename) and model_type != 'css'): # if iterative fit exists, then gaussian was run
        print('already exists %s'%it_estimates_filename)
    
    elif (model_type == 'css' and op.exists(css_it_estimates_filename)):
        print('already exists %s'%css_it_estimates_filename)
        
    else:
        # load data
        print('loading data from %s' % file)
        
        masked_data, not_nan_vox, orig_shape = load_and_mask_data(file, chunk_num = chunk_num, total_chunks = total_chunks)
        
        if len(masked_data)==0: # if all voxels nan, skip fitting completely 
            estimates_grid = np.zeros((orig_shape[0],6)); estimates_grid[:] = np.nan
            estimates_it = np.zeros((orig_shape[0],6)); estimates_it[:] = np.nan
            if model_type == 'css':
                estimates_css_grid = np.zeros((orig_shape[0],7)); estimates_css_grid[:] = np.nan
                estimates_css_it = np.zeros((orig_shape[0],7)); estimates_css_it[:] = np.nan
                
        else:
            # define design matrix 
            visual_dm = make_pRF_DM(op.join(derivatives_dir,'pRF_fit', 'DMprf.npy'), params, save_imgs=False, downsample=0.1)
        
            # make stimulus object, which takes an input design matrix and sets up its real-world dimensions
            prf_stim = PRFStimulus2D(screen_size_cm = params['monitor']['width'],
                                     screen_distance_cm = params['monitor']['distance'],
                                     design_matrix = visual_dm,
                                     TR = TR)
            
            # define model 
            gauss_model = Iso2DGaussianModel(stimulus = prf_stim,
                                             filter_predictions = True,
                                             filter_type = params['mri']['filtering']['type'],
                                             filter_params = {'highpass': params['mri']['filtering']['highpass'],
                                                             'add_mean': params['mri']['filtering']['add_mean'],
                                                             'window_length': params['mri']['filtering']['window_length'],
                                                             'polyorder': params['mri']['filtering']['polyorder']}
                                            )
            
            
            # and parameters
            grid_nr = params['mri']['fitting']['pRF']['grid_nr']
            sizes = params['mri']['fitting']['pRF']['max_size'] * \
                np.linspace(np.sqrt(params['mri']['fitting']['pRF']['min_size']/params['mri']['fitting']['pRF']['max_size']),1,grid_nr)**2
            eccs = params['mri']['fitting']['pRF']['max_eccen'] * \
                np.linspace(np.sqrt(params['mri']['fitting']['pRF']['min_eccen']/params['mri']['fitting']['pRF']['max_eccen']),1,grid_nr)**2
            polars = np.linspace(0, 2*np.pi, grid_nr)


            ## GRID FIT
            print("Grid fit")
            gauss_fitter = Iso2DGaussianFitter(data = masked_data, 
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
        save_estimates(grid_estimates_filename, estimates_grid, not_nan_vox, orig_shape, model_type = 'gauss')
        # for it
        save_estimates(it_estimates_filename, estimates_it, not_nan_vox, orig_shape, model_type = 'gauss')
        
        if model_type == 'css':
            
            if len(masked_data)>0:
            
                # grid exponent parameter
                css_n_grid = np.linspace(params['mri']['fitting']['pRF']['min_n'], 
                                         params['mri']['fitting']['pRF']['max_n'],12)

                # define model 
                css_model = CSS_Iso2DGaussianModel(stimulus = prf_stim,
                                                 filter_predictions = True,
                                                 filter_type = params['mri']['filtering']['type'],
                                                 filter_params = {'highpass': params['mri']['filtering']['highpass'],
                                                                 'add_mean': params['mri']['filtering']['add_mean'],
                                                                 'window_length': params['mri']['filtering']['window_length'],
                                                                 'polyorder': params['mri']['filtering']['polyorder']}
                                                )

                ## GRID FIT
                print("Grid fit")
                css_fitter = CSS_Iso2DGaussianFitter(data = masked_data, 
                                                   model = css_model, 
                                                   n_jobs = 16,
                                                    previous_gaussian_fitter = gauss_fitter)

                css_fitter.grid_fit(exponent_grid = css_n_grid, 
                                    pos_prfs_only = True)


                estimates_css_grid = css_fitter.gridsearch_params

                ## ITERATIVE FIT

                # model parameter bounds
                css_bounds = [(-2*ss, 2*ss),  # x
                              (-2*ss, 2*ss),  # y
                              (eps, 2*ss),  # prf size
                              (0, +inf),  # prf amplitude
                              (-5, +inf),  # bold baseline
                              (params['mri']['fitting']['pRF']['min_n'], 2*params['mri']['fitting']['pRF']['max_n'])]  # CSS exponent


                # iterative fit
                print("Iterative fit")
                css_fitter.iterative_fit(rsq_threshold = 0.05, 
                                           verbose = False,
                                           bounds = css_bounds,
                                           xtol = xtol,
                                           ftol = ftol)


                estimates_css_it = css_fitter.iterative_search_params

            # save estimates
            # for grid
            save_estimates(css_grid_estimates_filename, estimates_css_grid, not_nan_vox, orig_shape, model_type = 'css')
            # for it
            save_estimates(css_it_estimates_filename, estimates_css_it, not_nan_vox, orig_shape, model_type = 'css')


# Print duration
end_time = datetime.datetime.now()
print("\nStart time:\t{start_time}\nEnd time:\t{end_time}\nDuration:\t{dur}".format(
                start_time = start_time,
                end_time = end_time,
                dur  = end_time - start_time))
























