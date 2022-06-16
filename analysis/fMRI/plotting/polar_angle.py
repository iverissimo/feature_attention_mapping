import numpy as np
import os, sys
import os.path as op
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
import cortex
import pandas as pd
import seaborn as sns
from prfpy.stimulus import PRFStimulus2D

from FAM_utils import mri as mri_utils

# load settings from yaml
with open(op.join(str(Path(os.getcwd()).parents[1]),'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')
  
elif len(sys.argv) < 3:
    raise NameError('Please add type of run to be fitted (ex: leave_01_out vs median) '
                    'as 2nd argument in the command line!')
    
else:
    # fill subject number and chunk number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(3)
    run_type = str(sys.argv[2])

task = 'pRF'

# set font type for plots globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# some settings
base_dir = params['general']['current_dir'] # which machine we run the data
acq = params['mri']['acq'] # if using standard files or nordic files
space = params['mri']['space'] # subject space
total_chunks = params['mri']['fitting']['pRF']['total_chunks'][space] # number of chunks that data was split in

TR = params['mri']['TR']

# type of model to fit
model_type = params['mri']['fitting']['pRF']['fit_model']

# if we fitted hrf
fit_hrf = params['mri']['fitting']['pRF']['fit_hrf']

# set estimate key names
estimate_keys = params['mri']['fitting']['pRF']['estimate_keys'][model_type]

# set paths
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
postfmriprep_dir = op.join(derivatives_dir,'post_fmriprep','sub-{sj}'.format(sj=sj),space,'processed')

# path to pRF fits 
fits_pth =  op.join(derivatives_dir,'{task}_fit'.format(task=task),'sub-{sj}'.format(sj=sj), space, 
                    'iterative_{model}'.format(model=model_type),'run-{run}'.format(run=run_type))

if fit_hrf:
    fits_pth = op.join(fits_pth,'with_hrf')
    estimate_keys = estimate_keys+['hrf_derivative','hrf_dispersion']

# output dir to save fit and plot
figures_pth = op.join(derivatives_dir,'plots','polar_angle','{task}_fit'.format(task=task),
                      'sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run_type)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth)


## Load pRF estimates 

est_name = [x for _,x in enumerate(os.listdir(fits_pth)) if 'chunk-001' in x][0]
est_name = est_name.replace('chunk-001_of_{ch}'.format(ch=str(total_chunks).zfill(3)),'chunk-combined')

# total path to estimates path
estimates_combi = op.join(fits_pth,'combined', est_name)

if op.isfile(estimates_combi): # if combined estimates exists

        print('loading %s'%estimates_combi)
        estimates = np.load(estimates_combi) # load it

else: # if not join chunks and save file
    if not op.exists(op.join(fits_pth,'combined')):
        os.makedirs(op.join(fits_pth,'combined')) 

    # combine estimate chunks
    estimates = mri_utils.join_chunks(fits_pth, estimates_combi, fit_hrf = fit_hrf,
                            chunk_num = total_chunks, fit_model = 'it{model}'.format(model=model_type)) 

# define design matrix 
visual_dm = mri_utils.make_pRF_DM(op.join(derivatives_dir,'pRF_fit', 'sub-{sj}'.format(sj=sj), 'DMprf.npy'), params, 
                                     save_imgs = False, res_scaling = 0.1, crop = params['prf']['crop'] , crop_TR = params['prf']['crop_TR'], 
                                            shift_TRs = params['mri']['fitting']['pRF']['shift_DM'], 
                                            shift_TR_num = params['mri']['fitting']['pRF']['shift_DM_TRs'],
                                            overwrite = False)

# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm = params['monitor']['height'],
                        screen_distance_cm = params['monitor']['distance'],
                        design_matrix = visual_dm,
                        TR = TR)

# get the ecc limits (in dva)
# to mask estimates
x_ecc_lim, y_ecc_lim = mri_utils.get_ecc_limits(visual_dm,params,screen_size_deg = [prf_stim.screen_size_degrees,prf_stim.screen_size_degrees])

# mask estimates
print('masking estimates')
masked_est = mri_utils.mask_estimates(estimates, 
                                      estimate_keys = estimate_keys,
                                      x_ecc_lim = x_ecc_lim, y_ecc_lim = y_ecc_lim)

rsq = masked_est['r2']
x = masked_est['x']
y = masked_est['y']

complex_location = x + y * 1j # calculate eccentricity values
polar_angle = np.angle(complex_location)

### make alpha level based on pRF rsquared ###

alpha_level = mri_utils.normalize(np.clip(rsq, 0, .8))#mask, 0, .8)) # normalize 

# number of bins for colormaps
n_bins_colors = 256

# set threshold for plotting
rsq_threshold = params['plotting']['rsq_threshold']

# get pycortex sub
pysub = params['plotting']['pycortex_sub'] #+'_sub-{sj}'.format(sj=sj) # because subject specific borders 

ROIs, roi_verts, color_codes = mri_utils.get_rois4plotting(params, pysub = pysub, 
                                            use_atlas = True, atlas_pth = op.join(derivatives_dir,'glasser_atlas','59k_mesh'))


images = {}

## plot pRF polar angle ##

# only used voxels where pRF rsq bigger than 0
pa4plot = np.zeros(rsq.shape); pa4plot[:] = np.nan
pa4plot[rsq>0] = ((polar_angle + np.pi) / (np.pi * 2.0))[rsq>0]

# get matplotlib color map from segmented colors
PA_cmap = mri_utils.make_colormap(colormap = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb',
                              '#3d549f','#655099','#ad5a9b','#dd3933'], bins = n_bins_colors, 
                                    cmap_name = 'PA_mackey_costum',
                              discrete = False, add_alpha = False, return_cmap = True)

images['PA'] = mri_utils.make_raw_vertex_image(pa4plot, 
                                               cmap = PA_cmap, vmin = 0, vmax = 1, 
                                              data2 = alpha_level, vmin2 = 0, vmax2 = 1, 
                                               subject = pysub, data2D = True)

#cortex.quickshow(images['PA'],with_curvature=True,with_sulci=True,with_colorbar=True,
#                 curvature_brightness = 0.4, curvature_contrast = 0.1)#, recache = True)
filename = op.join(figures_pth,'flatmap_space-{space}_type-PA_mackey_colorwheel_withHRF-{hrf}.png'.format(space = pysub,
                                                                                    hrf = str(params['mri']['fitting']['pRF']['fit_hrf'])))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['PA'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

#### do same but without alpha level #########
images['PA_noalpha'] = mri_utils.make_raw_vertex_image(pa4plot, 
                                               cmap = PA_cmap, vmin = 0, vmax = 1, 
                                              data2 = alpha_level, vmin2 = 0, vmax2 = 1, 
                                               subject = pysub, data2D = False)

filename = op.join(figures_pth,'flatmap_space-{space}_type-PA_NOalpha_mackey_colorwheel_withHRF-{hrf}.png'.format(space = pysub,
                                                                                    hrf = str(params['mri']['fitting']['pRF']['fit_hrf'])))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['PA_noalpha'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)
######################


## plot polar angle in non uniform color wheel
## for better visualization of boundaries

hsv_angle = []
hsv_angle = np.ones((len(rsq), 3))
# set normalized polar angle (0-1), and make nan irrelevant vertices
hsv_angle[:, 0] = np.nan; hsv_angle[:, 0][rsq>0] = ((polar_angle + np.pi) / (np.pi * 2.0))[rsq>0]

# set angle threshold for overepresentation
angle_thresh = 3*np.pi/4 #value upon which to make it red for this hemifield (above it or below -angle will be red)
# normalize it
angle_thresh_norm = (angle_thresh + np.pi) / (np.pi * 2.0)


# get mid vertex index (diving hemispheres)
left_index = cortex.db.get_surfinfo(pysub).left.shape[0] 

# set angles within threh interval to 0
ind_thresh = np.where((hsv_angle[:left_index, 0] > angle_thresh_norm) | (hsv_angle[:left_index, 0] < 1-angle_thresh_norm))[0]
hsv_angle[:left_index, 0][ind_thresh] = 0

## now take angles from RH (thus LVF) 
# ATENÇÃO -> minus sign to flip angles vertically (then order of colors same for both hemispheres)
hsv_angle[left_index:, 0] = ((np.angle(-1*x + y * 1j ) + np.pi) / (np.pi * 2.0))[left_index:]

# set angles within threh interval to 0
ind_thresh = np.where((hsv_angle[left_index:, 0] > angle_thresh_norm) | (hsv_angle[left_index:, 0] < 1-angle_thresh_norm))[0]

hsv_angle[left_index:, 0][ind_thresh] = 0

rgb_angle = np.ones((len(rsq), 3)); rgb_angle[:] = np.nan;
rgb_angle[rsq>0] = colors.hsv_to_rgb(hsv_angle[rsq>0])

images['PA_half_hemi'] = cortex.VertexRGB(rgb_angle[:, 0], rgb_angle[:, 1], rgb_angle[:, 2],
                                           alpha = alpha_level,
                                           subject = pysub)

#cortex.quickshow(images['PA_half_hemi'],with_curvature=True,with_sulci=True,with_colorbar=True,
#                 curvature_brightness = 0.4, curvature_contrast = 0.1)

filename = op.join(figures_pth,'flatmap_space-{space}_type-PA_hsv_half_colorwheel_withHRF-{hrf}.png'.format(space = pysub,
                                                                                hrf = str(params['mri']['fitting']['pRF']['fit_hrf'])))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['PA_half_hemi'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

# # plot colorwheel and save in folder

resolution = 800
circle_x, circle_y = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
circle_radius = np.sqrt(circle_x**2 + circle_y**2)
circle_pa = np.arctan2(circle_y, circle_x) # all polar angles calculated from our mesh
circle_pa[circle_radius > 1] = np.nan # then we're excluding all parts of bitmap outside of circle

cmap = plt.get_cmap('hsv')
norm = colors.Normalize(-angle_thresh, angle_thresh) # normalize between the point where we defined our color threshold

# # for LH (RVF)
circle_pa_left = circle_pa.copy()
# # between thresh angle make it red
circle_pa_left[(circle_pa_left < -angle_thresh) | (circle_pa_left > angle_thresh)] = angle_thresh 
plt.imshow(circle_pa_left, cmap=cmap, norm=norm,origin='lower') # origin lower because imshow flips it vertically, now in right order for VF
plt.axis('off')
plt.savefig(op.join(figures_pth,'color_wheel_4LH-RVF.png'),dpi=100)

# # for RH (LVF)
circle_pa_right = circle_pa.copy()

circle_pa_right = np.fliplr(circle_pa_right)

circle_pa_right[(circle_pa_right < -.75 * np.pi) | (circle_pa_right > 0.75 * np.pi)] = .75*np.pi
plt.imshow(circle_pa_right, cmap=cmap, norm=norm,origin='lower')
plt.axis('off')
plt.savefig(op.join(figures_pth,'color_wheel_4RH-LVF.png'),dpi=100)


# # normal color wheel

# # make linear range of colors
colormap = colors.ListedColormap(['#ec9b3f','#f3eb53','#7cb956','#82cbdb','#3d549f','#655099','#ad5a9b','#dd3933'])
boundaries = np.linspace(0,1,8)
norm = colors.BoundaryNorm(boundaries, colormap.N, clip=True)

# # normalize between the point where we defined our color threshold
norm = colors.Normalize(-np.pi, np.pi) 

plt.imshow(circle_pa, cmap = colormap, norm=norm, origin='lower')
plt.axis('off')
plt.savefig(op.join(figures_pth,'color_wheel_mackey_discrete.png'),dpi=100)

## continuous colormap

color_codes = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb','#3d549f','#655099','#ad5a9b','#dd3933']

cvals  = np.arange(len(color_codes))
norm = plt.Normalize(min(cvals),max(cvals))
tuples = list(zip(map(norm,cvals), color_codes))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

plt.imshow(circle_pa, cmap = cmap, norm = colors.Normalize(-np.pi, np.pi) , origin='lower')
plt.axis('off')
plt.savefig(op.join(figures_pth,'color_wheel_mackey_continuous.png'),dpi=100)


### ADD TO OVERLAY, TO DRAW BORDERS
#cortex.utils.add_roi(images['PA_noalpha'], name = 'PA_noalpha', open_inkscape = False)
#cortex.utils.add_roi(images['PA_half_hemi'], name = 'PA_half_hemi', open_inkscape = False)