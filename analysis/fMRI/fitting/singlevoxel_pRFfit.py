################################################
#      Do pRF fit on single voxel, 
#    by loading estimates, getting fit OR
#    by fitting the timeseries
#    saving plot of fit on timeseries
################################################

import numpy as np
import os, sys
import os.path as op
import yaml
from pathlib import Path


# requires pfpy to be installed - preferably with python setup.py develop
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter

sys.path.insert(0,'..') # add parent folder to path
from utils import * #import script to use relevante functions

import datetime

from nilearn import surface

# load settings from yaml
with open(op.join(str(Path(os.getcwd()).parents[1]),'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

# define participant number, ROI (if the case) and vertex number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex: 01) '	
                    'as 1st argument in the command line!')	

elif len(sys.argv)<3:   
    raise NameError('Please add ROI name (ex: V1) or "None" if looking at vertex from no specific ROI  '	
                    'as 2nd argument in the command line!')	

elif len(sys.argv)<4:   
    raise NameError('Please vertex index number of that ROI (or from whole brain)'	
                    'as 3rd argument in the command line!'
                    '(can also be "max" or "min" to fit vertex of max or min RSQ)')	
elif len(sys.argv)<5:   
    raise NameError('fit vs load estimates'	
                    'as 4th argument in the command line!')	

else:
    sj = str(sys.argv[1]).zfill(3) #fill subject number with 0 in case user forgets	

    roi = str(sys.argv[2]) # ROI or 'None'

    if str(sys.argv[3]) != 'max' and str(sys.argv[3]) != 'min': # if we actually get a number for the vertex
    
        vertex = int(sys.argv[3]) # vertex number
        
    fit_now = True if str(sys.argv[4])=='fit' else False

# set font type for plots globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# some settings
base_dir = params['general']['current_dir'] # which machine we run the data
acq = params['mri']['acq'] # if using standard files or nordic files
space = params['mri']['space'] # subject space
total_chunks = params['mri']['fitting']['pRF']['total_chunks'][space] # number of chunks that data was split in
hemispheres = ['hemi-L','hemi-R'] # only used for gifti files
run_type = 'median'

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
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
postfmriprep_dir = op.join(derivatives_dir,'post_fmriprep','sub-{sj}'.format(sj=sj),space,'processed')

# path to pRF fits (if they exist)
fits_pth =  op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run_type))

# output dir to save fit and plot
figures_pth = op.join(derivatives_dir,'plots','single_vertex','pRFfit','sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run_type)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

# define design matrix 
visual_dm = make_pRF_DM(op.join(derivatives_dir,'pRF_fit', 'DMprf.npy'), params, save_imgs=False, downsample=0.1)

# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm = params['monitor']['width'],
                         screen_distance_cm = params['monitor']['distance'],
                         design_matrix = visual_dm,
                         TR = TR)

# define gaussian model 
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

# grid exponent parameter
css_n_grid = np.linspace(params['mri']['fitting']['pRF']['min_n'], 
                         params['mri']['fitting']['pRF']['max_n'],12)

# define CSS model 
css_model = CSS_Iso2DGaussianModel(stimulus = prf_stim,
                                 filter_predictions = True,
                                 filter_type = params['mri']['filtering']['type'],
                                 filter_params = {'highpass': params['mri']['filtering']['highpass'],
                                                 'add_mean': params['mri']['filtering']['add_mean'],
                                                 'window_length': params['mri']['filtering']['window_length'],
                                                 'polyorder': params['mri']['filtering']['polyorder']}
                                )

# model parameter bounds
css_bounds = [(-2*ss, 2*ss),  # x
              (-2*ss, 2*ss),  # y
              (eps, 2*ss),  # prf size
              (0, +inf),  # prf amplitude
              (-5, +inf),  # bold baseline
              (params['mri']['fitting']['pRF']['min_n'], 2*params['mri']['fitting']['pRF']['max_n'])]  # CSS exponent

# list with absolute file names to be fitted (iff gii, then 2 hemispheres)
proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-pRF' in h and
                 'acq-{acq}'.format(acq=acq) in h and run_type in h and h.endswith(file_ext)]
 
## load functional data
if '.func.gii' in file_ext: # if gifti
    
    data = []
    for _,h in enumerate(hemispheres):
        gii_file = [x for _,x in enumerate(proc_files) if h in x][0]
        print('loading %s' %gii_file)
        data.append(np.array(surface.load_surf_data(gii_file)))

    data = np.vstack(data) # will be (vertex, TR)
    
else:
    print('not implemented for other types of spaces')

if roi != 'None':
    print('masking data for ROI %s'%roi)
    roi_ind = cortex.get_roi_verts(params['processing']['space'],roi) # get indices for that ROI
    data = data[roi_ind[roi]]

# do we want to fit now, or load estimates?

if fit_now:
    
    timeseries = data[vertex][np.newaxis,...]
    
    ## GRID FIT
    print("Grid fit")
    gauss_fitter = Iso2DGaussianFitter(data = timeseries, 
                                       model = gauss_model, 
                                       n_jobs = 16)
    
    gauss_fitter.grid_fit(ecc_grid = eccs, 
                          polar_grid = polars, 
                          size_grid = sizes, 
                          pos_prfs_only = True)


    estimates_grid = gauss_fitter.gridsearch_params[0]

    # iterative fit
    print("Iterative fit")
    gauss_fitter.iterative_fit(rsq_threshold = 0.05, 
                               verbose = False,
                               bounds=gauss_bounds,
                               xtol = xtol,
                               ftol = ftol)


    estimates_it = gauss_fitter.iterative_search_params[0]
    
    if model_type == 'css':
        
        ## GRID FIT
        print("Grid fit")
        css_fitter = CSS_Iso2DGaussianFitter(data = timeseries, 
                                           model = css_model, 
                                           n_jobs = 16,
                                            previous_gaussian_fitter = gauss_fitter)

        css_fitter.grid_fit(exponent_grid = css_n_grid, 
                            pos_prfs_only = True)
        
        estimates_css_grid = css_fitter.gridsearch_params[0]
        
        # iterative fit
        print("Iterative fit")
        css_fitter.iterative_fit(rsq_threshold = 0.05, 
                                   verbose = False,
                                   bounds = css_bounds,
                                   xtol = xtol,
                                   ftol = ftol)


        estimates_css_it = css_fitter.iterative_search_params[0]
        
else:
    print('should load estimates, select the ones from the chosen vertex - NOT IMPLEMENTED YET')
    
# get prediction
if model_type == 'css':
    model_fit = css_model.return_prediction(estimates_css_it[0],estimates_css_it[1],
                                            estimates_css_it[2], estimates_css_it[3],
                                            estimates_css_it[4], estimates_css_it[5])
    rsquare = estimates_css_it[6]
else:
    model_fit = gauss_model.return_prediction(estimates_it[0],estimates_it[1],
                                            estimates_it[2], estimates_it[3],
                                            estimates_it[4])   
    rsquare = estimates_it[5]

# set figure name
fig_name = 'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_roi-{roi}_vertex-{vert}.png'.format(sj=sj,
                                                                                        acq=acq,
                                                                                        space=space,
                                                                                        run=run_type,
                                                                                        model=model_type,
                                                                                        roi=roi,
                                                                                        vert=vertex) 
# plot data with model
fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)

# plot data with model
time_sec = np.linspace(0,len(model_fit[0,...])*TR,num=len(model_fit[0,...])) # array with 90 timepoints, in seconds
    
axis.plot(time_sec, model_fit[0,...],c='red',lw=3,label='model R$^2$ = %.2f'%rsquare,zorder=1)
#axis.scatter(time_sec, data_reshape[ind_max_rsq,:], marker='v',s=15,c='k',label='data')
axis.plot(time_sec, timeseries[0,...],'k--',label='data')
axis.set_xlabel('Time (s)',fontsize=20, labelpad=20)
axis.set_ylabel('BOLD signal change (%)',fontsize=20, labelpad=10)
axis.set_xlim(0,len(model_fit[0,...])*TR)
axis.legend(loc='upper left',fontsize=10)  # doing this to guarantee that legend is how I want it  

# times where bar is on screen [1st on, last on, 1st on, last on, etc] 
bar_onset = np.array([20,36,37,53,66,82,83,99,120,136,137,153,166,182,183,199])*TR
bar_directions = [val for _,val in enumerate(params['prf']['bar_pass_direction']) if 'empty' not in val]
# plot axis vertical bar on background to indicate stimulus display time
ax_count = 0
for h in range(8):
    
    if bar_directions[h] in ['L-R','R-L']: # horizontal bar passes will be darker 
        plt.axvspan(bar_onset[ax_count], bar_onset[ax_count+1]+TR, facecolor='#8f0000', alpha=0.1)
        
    elif bar_directions[h] in ['U-D','D-U']: # vertical bar passes will be lighter 
        plt.axvspan(bar_onset[ax_count], bar_onset[ax_count+1]+TR, facecolor='#ff0000', alpha=0.1)
    
    ax_count += 2

fig.savefig(op.join(figures_pth,fig_name))
