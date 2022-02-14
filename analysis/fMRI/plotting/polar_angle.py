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

# define file extension that we want to use, 
# should include processing key words
file_ext = '_cropped_{filt}_{stand}.npy'.format(filt = params['mri']['filtering']['type'],
                                                    stand = 'psc')

# set paths
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
postfmriprep_dir = op.join(derivatives_dir,'post_fmriprep','sub-{sj}'.format(sj=sj),space,'processed')

# path to pRF fits 
fits_pth =  op.join(derivatives_dir,'{task}_fit'.format(task=task),'sub-{sj}'.format(sj=sj), space, 'iterative_{model}'.format(model=model_type),'run-{run}'.format(run=run_type))
#fits_pth =  op.join(derivatives_dir,'{task}_fit'.format(task=task),'sub-{sj}'.format(sj=sj), space, '{model}'.format(model=model_type),'run-{run}'.format(run=run_type))

# output dir to save fit and plot
figures_pth = op.join(derivatives_dir,'plots','polar_angle','{task}fit'.format(task=task),
                      'sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run_type)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

if task == 'pRF':
    
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


# define design matrix 
    visual_dm = mri_utils.make_pRF_DM(op.join(derivatives_dir,'pRF_fit', 'sub-{sj}'.format(sj=sj), 'DMprf.npy'), params, save_imgs=False, downsample=0.1, 
                                            crop = params['prf']['crop'] , crop_TR = params['prf']['crop_TR'], overwrite=False)

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
masked_est = mri_utils.mask_estimates(estimates, fit_model = model_type,
                            x_ecc_lim = x_ecc_lim, y_ecc_lim = y_ecc_lim)

rsq = masked_est['rsq']
x = masked_est['x']
y = masked_est['y']

complex_location = x + y * 1j # calculate eccentricity values
polar_angle = np.angle(complex_location)

# set threshold for plotting
rsq_threshold = params['plotting']['rsq_threshold']

# get vertices for subject fsaverage
ROIs = params['plotting']['ROIs'][space]

# dictionary with one specific color per group - similar to fig3 colors
ROI_pal = params['plotting']['ROI_pal']
color_codes = {key: ROI_pal[key] for key in ROIs}

# get pycortex sub
pysub = params['plotting']['pycortex_sub'] 

# get vertices for ROI
roi_verts = {} #empty dictionary  
for _,val in enumerate(ROIs):
    roi_verts[val] = cortex.get_roi_verts(pysub,val)[val]


images = {}

# normalize polar angles to have values in circle between 0 and 1
polar_ang_norm = (polar_angle + np.pi) / (np.pi * 2.0)

# make alpha level based on rsquared
# normalize the distribution, for better visualization
alpha_level = mri_utils.normalize(np.clip(rsq,rsq_threshold,.3))#np.clip(rsq,rsq_threshold,.3) 

# make costum colormap, similar to curtis mackey paper
# orange to red, counter clockwise
n_bins = 256#8
PA_colors = mri_utils.add_alpha2colormap(colormap = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb',
                              '#3d549f','#655099','#ad5a9b','#dd3933'],bins = n_bins, cmap_name = 'PA_mackey_costum',
                              discrete = False)

# create costume colormp rainbow_r
col2D_name = op.splitext(op.split(PA_colors)[-1])[0]
print('created costum colormap %s'%col2D_name)

images['PA'] = cortex.Vertex2D(polar_ang_norm, alpha_level,
                                subject = pysub, 
                                vmin = 0, vmax = 1,
                                vmin2 = 0, vmax2 = np.nanmax(alpha_level),
                                cmap = col2D_name)

#cortex.quickshow(images['PA'],with_curvature=True,with_sulci=True,with_colorbar=True,
#                 curvature_brightness = 0.4, curvature_contrast = 0.1)

filename = op.join(figures_pth,'flatmap_space-{space}_type-PA_mackey_colorwheel.svg'.format(space=pysub))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['PA'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

## do the same but with a color wheel than spans just one hemifield,  
## (non uniform color wheel)
## for better visualization of boundaries

# shift radians in order to overrepresent red color
# useful to make NON-REPRESENTED retinotopic hemifield per hemisphere red
# then easier to define borders

# create HSV array, with PA values (-pi to pi) that were obtained from estimates
# saturation wieghted by a shifted distribution of RSQ (better visualization)
# value bolean (if I don't give it an rsq threshold then it's always 1)

hsv_angle = []
hsv_angle = np.ones((len(rsq), 3))
hsv_angle[:, 0] = polar_angle.copy()
#hsv_angle[:, 1] = np.clip(estimates_dict['rsq'] / np.nanmax(estimates_dict['rsq']) * 3, 0, 1)
hsv_angle[:, 2] = mri_utils.normalize(np.clip(rsq,rsq_threshold,.3)) #rsq > rsq_threshold 


# get mid vertex index (diving hemispheres)
left_index = cortex.db.get_surfinfo(pysub).left.shape[0] 

### take angles from LH (thus RVF)##
angle_ = hsv_angle[:left_index, 0].copy()
angle_thresh = 3*np.pi/4 #value upon which to make it red for this hemifield (above it or below -angle will be red)

#normalized angles, between 0 and 1
hsv_angle[:left_index, 0] = np.clip((angle_ + angle_thresh)/(2*angle_thresh), 0, 1)

### take angles from RH (thus LVF) ##
angle_ = -hsv_angle[left_index:, 0].copy() # ATENÇÃO -> minus sign to flip angles vertically (then order of colors same for both hemispheres)

# sum 2pi to originally positive angles (now our trig circle goes from pi to 2pi to pi again, all positive)
angle_[hsv_angle[left_index:, 0] > 0] += 2 * np.pi

#normalized angles, between 0 and 1
angle_ = np.clip((angle_ + (angle_thresh-np.pi))/(2*angle_thresh), 0, 1) # ATENÇÃO -> we subtract -pi to angle thresh because now we want to rotate the whole thing -180 degrees

hsv_angle[left_index:, 0] = angle_.copy()
rgb_angle = []
rgb_angle = colors.hsv_to_rgb(hsv_angle)

# make alpha same as saturation, reduces clutter
alpha_angle = hsv_angle[:, 2]

images['PA_half_hemi'] = cortex.VertexRGB(rgb_angle[:, 0], rgb_angle[:, 1], rgb_angle[:, 2],
                                           alpha = alpha_angle,
                                           subject = pysub)

#cortex.quickshow(images['PA_half_hemi'],with_curvature=True,with_sulci=True,with_colorbar=True,
#                 curvature_brightness = 0.4, curvature_contrast = 0.1)

filename = op.join(figures_pth,'flatmap_space-{space}_type-PA_hsv_half_colorwheel.svg'.format(space=pysub))
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
plt.savefig(op.join(figures_pth,'color_wheel_4LH-RVF.svg'),dpi=100)

# # for RH (LVF)
circle_pa_right = circle_pa.copy()

circle_pa_right = np.fliplr(circle_pa_right)

circle_pa_right[(circle_pa_right < -.75 * np.pi) | (circle_pa_right > 0.75 * np.pi)] = .75*np.pi
plt.imshow(circle_pa_right, cmap=cmap, norm=norm,origin='lower')
plt.axis('off')
plt.savefig(op.join(figures_pth,'color_wheel_4RH-LVF.svg'),dpi=100)


# # normal color wheel

# # make linear range of colors
colormap = colors.ListedColormap(['#ec9b3f','#f3eb53','#7cb956','#82cbdb','#3d549f','#655099','#ad5a9b','#dd3933'])
boundaries = np.linspace(0,1,n_bins)
norm = colors.BoundaryNorm(boundaries, colormap.N, clip=True)

# # normalize between the point where we defined our color threshold
norm = colors.Normalize(-np.pi, np.pi) 

plt.imshow(circle_pa, cmap = colormap, norm=norm, origin='lower')
plt.axis('off')
plt.savefig(op.join(figures_pth,'color_wheel_mackey_discrete.svg'),dpi=100)

## continuous colormap

color_codes = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb','#3d549f','#655099','#ad5a9b','#dd3933']

cvals  = np.arange(len(color_codes))
norm = plt.Normalize(min(cvals),max(cvals))
tuples = list(zip(map(norm,cvals), color_codes))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

plt.imshow(circle_pa, cmap = cmap, norm = colors.Normalize(-np.pi, np.pi) , origin='lower')
plt.axis('off')
plt.savefig(op.join(figures_pth,'color_wheel_mackey_continuous.svg'),dpi=100)