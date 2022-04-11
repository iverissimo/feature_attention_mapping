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
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel, Norm_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter, Norm_Iso2DGaussianFitter

from FAM_utils import mri as mri_utils
import cortex
import matplotlib.pyplot as plt

import glob

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
    else:
        vertex = str(sys.argv[3]) 
        
    fit_now = True if str(sys.argv[4])=='fit' else False

if fit_now == True and vertex in ['min','max']:
    raise NameError('Cannot fit vertex, need to load pre-fitted estimates')

# set font type for plots globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# some settings
base_dir = params['general']['current_dir'] # which machine we run the data
acq = params['mri']['acq'] # if using standard files or nordic files
space = params['mri']['space'] # subject space
total_chunks = params['mri']['fitting']['pRF']['total_chunks'][space] # number of chunks that data was split in
hemispheres = ['hemi-L','hemi-R'] # only used for gifti files

run_type = 'mean'

TR = params['mri']['TR']

# type of model to fit
model_type = params['mri']['fitting']['pRF']['fit_model']
fit_hrf = params['mri']['fitting']['pRF']['fit_hrf']

# if we are keeping baseline fixed at 0
fix_bold_baseline = params['mri']['fitting']['pRF']['fix_bold_baseline']
# if we want to do bold baseline correction
correct_baseline = params['mri']['fitting']['pRF']['correct_baseline'] 

# define file extension that we want to use, 
# should include processing key words
file_ext = '_cropped_{filt}_{stand}.npy'.format(filt = params['mri']['filtering']['type'],
                                                    stand = 'psc')

# set paths
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
postfmriprep_dir = op.join(derivatives_dir,'post_fmriprep','sub-{sj}'.format(sj=sj),space,'processed')

# path to pRF fits (if they exist)
fits_pth =  op.join(derivatives_dir,'pRF_fit','sub-{sj}'.format(sj=sj), space, 'iterative_{model}'.format(model=model_type),'run-{run}'.format(run=run_type))

## make DM mask, according to sub responses
# sourcedata dir
source_dir = glob.glob(op.join(params['mri']['paths'][base_dir]['root'], 'sourcedata', 
                               'sub-{sj}'.format(sj=sj), 'ses-*', 'func'))[0] 
# list of behavior files
behav_files = [op.join(source_dir, h) for h in os.listdir(source_dir) if 'task-pRF' in h and
                 h.endswith('events.tsv')]
# behav boolean mask
DM_mask_beh = mri_utils.get_beh_mask(behav_files,params)

# output dir to save fit and plot
figures_pth = op.join(derivatives_dir,'plots','single_vertex','pRFfit','sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run_type)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

# define design matrix 
visual_dm = mri_utils.make_pRF_DM(op.join(derivatives_dir,'pRF_fit', 'sub-{sj}'.format(sj=sj), 'DMprf.npy'), params, 
                                save_imgs = False, res_scaling=0.1, crop = params['prf']['crop'] , crop_TR = params['prf']['crop_TR'], 
                                overwrite = False,  mask = DM_mask_beh)

# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm = params['monitor']['height'],
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
max_ecc_size = params['mri']['fitting']['pRF']['max_eccen'] #prf_stim.screen_size_degrees/2.0
sizes, eccs, polars = max_ecc_size * np.linspace(0.25, 1, grid_nr)**2, \
    max_ecc_size * np.linspace(0.1, 1, grid_nr)**2, \
    np.linspace(0, 2*np.pi, grid_nr)

# to set up parameter bounds in iterfit
inf = np.inf
eps = 1e-1
ss = prf_stim.screen_size_degrees
xtol = 1e-7
ftol = 1e-6

# model parameter bounds
gauss_bounds = [(-1.5*ss, 1.5*ss),  # x
                (-1.5*ss, 1.5*ss),  # y
                (eps, 1.5*ss),  # prf size
                (0, 1000),  # prf amplitude
                (-1000, 1000)]  # bold baseline

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
css_bounds = [(-1.5*ss, 1.5*ss),  # x
            (-1.5*ss, 1.5*ss),  # y
            (eps, 1.5*ss),  # prf size
            (0, 1000),  # prf amplitude
            (-1000, 1000),  # bold baseline
            (0.01, 1)]  # CSS exponent

# define DN model

# define model 
dn_model =  Norm_Iso2DGaussianModel(stimulus = prf_stim,
                                    filter_predictions = True,
                                    filter_type = params['mri']['filtering']['type'],
                                    filter_params = {'highpass': params['mri']['filtering']['highpass'],
                                                    'add_mean': params['mri']['filtering']['add_mean'],
                                                    'window_length': params['mri']['filtering']['window_length'],
                                                    'polyorder': params['mri']['filtering']['polyorder']}
                                )


if fit_hrf:
    gauss_bounds += [(0,10),(0,0)]
    css_bounds += [(0,10),(0,0)]

# list with absolute file name to be fitted
proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-pRF' in h and
                 'acq-{acq}'.format(acq=acq) in h and run_type in h and h.endswith(file_ext)]
 
## load functional data
data = np.load(proc_files[0],allow_pickle=True) # will be (vertex, TR)

# if we want to keep baseline fix, we need to correct it!
if correct_baseline:
    data = mri_utils.baseline_correction(data, params, num_baseline_TRs = 7, baseline_interval = 'empty_long', 
                            avg_type = 'median', crop = True, crop_TR = 8)
    
if roi != 'None' and vertex not in ['max','min']:
    print('masking data for ROI %s'%roi)
    roi_ind = cortex.get_roi_verts(params['plotting']['pycortex_sub']+'_sub-{sj}'.format(sj=sj),roi) # get indices for that ROI
    data = data[roi_ind[roi]]

if vertex not in ['max','min']:
    ## do we want to fit now, or load estimates?
    timeseries = data[vertex][np.newaxis,...]

if fit_now:

    timeseries = data[vertex][np.newaxis,...]

    ## GRID FIT
    print("Grid fit")
    gauss_fitter = Iso2DGaussianFitter(data = timeseries, 
                                       model = gauss_model, 
                                       n_jobs = 16,
                                       fit_hrf = fit_hrf)
    
    gauss_fitter.grid_fit(ecc_grid = eccs, 
                          polar_grid = polars, 
                          size_grid = sizes, 
                          pos_prfs_only = True)


    estimates_grid = gauss_fitter.gridsearch_params[0]

    # iterative fit
    print("Iterative fit")
    gauss_fitter.iterative_fit(rsq_threshold = 0.05, 
                               verbose = True,
                               bounds=gauss_bounds,
                               xtol = xtol,
                               ftol = ftol)


    estimates_it = gauss_fitter.iterative_search_params[0]

    # set outcomes
    xx = estimates_it[0]
    yy = estimates_it[1]
    size = estimates_it[2] 
    beta = estimates_it[3]
    baseline = estimates_it[4]
    rsq = estimates_it[-1]

    if fit_hrf:
        hrf_deriv = estimates_it[5]
        hrf_disp = estimates_it[6]  
    
    if model_type == 'css':
        
        ## GRID FIT
        print("Grid fit")
        css_fitter = CSS_Iso2DGaussianFitter(data = timeseries, 
                                           model = css_model, 
                                           n_jobs = 16,
                                            fit_hrf = fit_hrf,
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
        
        # set outcomes
        xx = estimates_css_it[0]
        yy = estimates_css_it[1]
        size = estimates_css_it[2] 
        beta = estimates_css_it[3]
        baseline = estimates_css_it[4]
        ns = estimates_css_it[5]
        rsq = estimates_css_it[-1] 

        if fit_hrf:
            hrf_deriv = estimates_css_it[6]
            hrf_disp = estimates_css_it[7] 
        
else:
    print('loading estimates')

    ## Load pRF estimates 
    
    # path to combined estimates
    estimates_pth = op.join(fits_pth,'combined')
        
    # combined estimates filename
    est_name = [x for _,x in enumerate(os.listdir(fits_pth)) if 'chunk-001' in x][0]
    est_name = est_name.replace('chunk-001_of_{ch}'.format(ch=str(total_chunks).zfill(3)),'chunk-combined')
    
    # total path to estimates path
    estimates_combi = op.join(estimates_pth,est_name)
    
    if op.isfile(estimates_combi): # if combined estimates exists
            
            print('loading %s'%estimates_combi)
            estimates = np.load(estimates_combi) # load it
    
    else: # if not join chunks and save file
        if not op.exists(estimates_pth):
            os.makedirs(estimates_pth) 

        estimates = mri_utils.join_chunks(fits_pth, estimates_combi,
                                chunk_num = total_chunks, fit_model = 'it{model}'.format(model=model_type)) #'{model}'.format(model=model_type)))#

    if roi != 'None':
        print('masking data for ROI %s'%roi)
        #roi_ind = cortex.get_roi_verts(params['plotting']['pycortex_sub'],roi) # get indices for that ROI
        #roi_ind = cortex.get_roi_verts(params['plotting']['pycortex_sub']+'_sub-{sj}'.format(sj=sj),roi) # get indices for that ROI
        #data = data[roi_ind[roi]]

        if vertex == 'max':
            vertex = np.where(estimates['r2'][roi_ind[roi]] == np.nanmax(estimates['r2'][roi_ind[roi]]))[0][0]
        elif vertex == 'min': 
            vertex = np.where(estimates['r2'][roi_ind[roi]] == np.nanmin(estimates['r2'][roi_ind[roi]]))[0][0]

        timeseries = data[vertex][np.newaxis,...]

        xx = estimates['x'][roi_ind[roi]][vertex]
        yy = estimates['y'][roi_ind[roi]][vertex]
        size = estimates['size'][roi_ind[roi]][vertex]
        beta = estimates['betas'][roi_ind[roi]][vertex]
        baseline = estimates['baseline'][roi_ind[roi]][vertex]
        if 'css' in model_type:
            ns = estimates['ns'][roi_ind[roi]][vertex]
        
        if 'dn' in model_type:
            sa = estimates['sa'][roi_ind[roi]][vertex]
            ss = estimates['ss'][roi_ind[roi]][vertex] 
            nb = estimates['nb'][roi_ind[roi]][vertex] 
            sb = estimates['sb'][roi_ind[roi]][vertex]

        rsq = estimates['r2'][roi_ind[roi]][vertex] 

        if fit_hrf:
            hrf_deriv = estimates['hrf_derivative'][roi_ind[roi]][vertex] 
            hrf_disp = estimates['hrf_dispersion'][roi_ind[roi]][vertex] 
    
    else:
        xx = estimates['x'][vertex]
        yy = estimates['y'][vertex]
        
        size = estimates['size'][vertex]
        
        beta = estimates['betas'][vertex]
        baseline = estimates['baseline'][vertex]
        
        if 'css' in model_type:
            ns = estimates['ns'][vertex]

        if 'dn' in model_type:
            sa = estimates['sa'][vertex]
            ss = estimates['ss'][vertex] 
            nb = estimates['nb'][vertex] 
            sb = estimates['sb'][vertex]

        rsq = estimates['r2'][vertex]

        if fit_hrf:
            hrf_deriv = estimates['hrf_derivative'][vertex] 
            hrf_disp = estimates['hrf_dispersion'][vertex]

if fit_hrf:
    hrf = mri_utils.create_hrf(hrf_params=[1.0,hrf_deriv,hrf_disp],TR=TR)
    gauss_model.hrf = hrf
    css_model.hrf = hrf
    dn_model.hrf = hrf

# get prediction
if model_type == 'css':
    model_fit = css_model.return_prediction(xx,yy,
                                            size, beta,
                                            baseline, ns)

    print('exponent for vertex is %.2f'%ns)

elif model_type == 'dn':
    model_fit = dn_model.return_prediction(xx,yy,
                                            size, beta,
                                            baseline,
                                            sa, ss, nb, sb) 

else:
    model_fit = gauss_model.return_prediction(xx,yy,
                                            size, beta,
                                            baseline)   

# set figure name
fig_name = 'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_roi-{roi}_vertex-{vert}.png'.format(sj=sj,
                                                                                        acq=acq,
                                                                                        space=space,
                                                                                        run=run_type,
                                                                                        model=model_type,
                                                                                        roi=roi,
                                                                                        vert=vertex) 

if fit_now == False:
    fig_name = fig_name.replace('.png','_loaded.png')

# plot data with model
fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)

# plot data with model
time_sec = np.linspace(0,len(model_fit[0,...])*TR,num=len(model_fit[0,...])) # array with 90 timepoints, in seconds
    
axis.plot(time_sec, model_fit[0,...],c='red',lw=3,label='model R$^2$ = %.2f'%rsq,zorder=1)
#axis.scatter(time_sec, data_reshape[ind_max_rsq,:], marker='v',s=15,c='k',label='data')
axis.plot(time_sec, timeseries[0,...],'k--',label='data')
axis.set_xlabel('Time (s)',fontsize=20, labelpad=20)
axis.set_ylabel('BOLD signal change (%)',fontsize=20, labelpad=10)
axis.set_xlim(0,len(model_fit[0,...])*TR)
axis.legend(loc='upper left',fontsize=10)  # doing this to guarantee that legend is how I want it  

# times where bar is on screen [1st on, last on, 1st on, last on, etc] 
bar_onset = np.array([20,36,37,53,66,82,83,99,120,136,137,153,166,182,183,199])*TR

if params['prf']['crop']:
    bar_onset = bar_onset - params['prf']['crop_TR']

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

# if fitting hrf, then plot shape in comparison with canonical for comparison
if fit_hrf:
    fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)

    axis.plot(mri_utils.create_hrf(TR=TR)[0],'grey',label='spm hrf')
    axis.plot(hrf[0],'red',label='fitted hrf')
    axis.set_xlim(0, 25)
    axis.legend(loc='upper right',fontsize=10) 
    axis.set_xlabel('Time (s)',fontsize=10, labelpad=10)
    fig.savefig(op.join(figures_pth,'HRF_roi-{roi}_vertex-{vert}.png'.format(roi=roi,vert=vertex))) 


