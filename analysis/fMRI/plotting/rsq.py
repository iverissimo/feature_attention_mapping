################################################
#      Make some rsq plots 
#    To see distribution of fits
#    in surface and per ROI
################################################

import numpy as np
import os, sys
import os.path as op
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import cortex
import pandas as pd
import seaborn as sns
from prfpy.stimulus import PRFStimulus2D

sys.path.insert(0,'..') # add parent folder to path
from utils import * #import script to use relevante functions

import datetime

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
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
postfmriprep_dir = op.join(derivatives_dir,'post_fmriprep','sub-{sj}'.format(sj=sj),space,'processed')

# path to pRF fits 
fits_pth =  op.join(derivatives_dir,'{task}_fit'.format(task=task),'sub-{sj}'.format(sj=sj), space, 'iterative_{model}'.format(model=model_type),'run-{run}'.format(run=run_type))
#fits_pth =  op.join(derivatives_dir,'{task}_fit'.format(task=task),'sub-{sj}'.format(sj=sj), space, '{model}'.format(model=model_type),'run-{run}'.format(run=run_type))

# output dir to save fit and plot
figures_pth = op.join(derivatives_dir,'plots','rsq','{task}fit'.format(task=task),
                      'sub-{sj}'.format(sj=sj), space, model_type,'run-{run}'.format(run=run_type)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

if task == 'pRF':
    
    # Load pRF estimates 
    estimates = []
    
    # path to combined estimates
    estimates_pth = op.join(fits_pth,'combined')

    for _,field in enumerate(hemispheres): # each hemi field
        
        # combined estimates filename
        est_name = [x for _,x in enumerate(os.listdir(fits_pth)) if 'chunk-001' in x and field in x][0]
        est_name = est_name.replace('chunk-001_of_{ch}'.format(ch=str(total_chunks).zfill(3)),'chunk-combined')
        
        # total path to estimates path
        estimates_combi = op.join(estimates_pth,est_name)
        
        if op.isfile(estimates_combi): # if combined estimates exists
                
                print('loading %s'%estimates_combi)
                estimates.append(np.load(estimates_combi)) #save both hemisphere estimates in same array
        
        else: # if not join chunks and save file
            if not op.exists(estimates_pth):
                os.makedirs(estimates_pth) 

            estimates.append(join_chunks(fits_pth, estimates_combi, field,
                                         chunk_num = total_chunks, fit_model = 'it{model}'.format(model=model_type))) #'{model}'.format(model=model_type)))#

    
# define design matrix 
visual_dm = make_pRF_DM(op.join(derivatives_dir,'pRF_fit', 'DMprf.npy'), params, save_imgs=False, downsample=0.1)

# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm = params['monitor']['height'],
                         screen_distance_cm = params['monitor']['distance'],
                         design_matrix = visual_dm,
                         TR = TR)

# mask estimates
print('masking estimates')
masked_est = mask_estimates(estimates, fit_model = model_type,
                            screen_limit_deg = [prf_stim.screen_size_degrees/2,prf_stim.screen_size_degrees/2])

rsq = masked_est['rsq']

## make violin plots with values per ROI

# get vertices for subject fsaverage
ROIs = ['V1','V2','V3','V3AB','hV4','LO','IPS0','IPS1','IPS2+','sPCS','iPCS']

# Make a dictionary with one specific color per group - similar to fig3 colors
ROI_pal = {'V1': (0.03137255, 0.11372549, 0.34509804), 'V2': (0.14136101, 0.25623991, 0.60530565),
           'V3': (0.12026144, 0.50196078, 0.72156863), 'V3AB': (0.25871588, 0.71514033, 0.76807382), 
           'hV4': (0.59215686, 0.84052288, 0.72418301), 'LO': (0.88207612, 0.9538639 , 0.69785467),
           'IPS0': (0.99764706, 0.88235294, 0.52862745), 'IPS1': (0.99529412, 0.66901961, 0.2854902), 
           'IPS2+': (0.83058824, 0.06117647, 0.1254902),
           'sPCS': (0.88221453, 0.83252595, 0.91109573), 'iPCS': (0.87320261, 0.13071895, 0.47320261)
         }

roi_verts = {} #empty dictionary  
for _,val in enumerate(ROIs):
    roi_verts[val] = cortex.get_roi_verts(space,val)[val]


for idx,rois_ks in enumerate(ROIs): 
    
    # mask estimates
    print('masking estimates for ROI %s'%rois_ks)
    
    roi_rsq = rsq[roi_verts[rois_ks]]

    if idx == 0:
        df_rsq = pd.DataFrame({'roi': np.tile(rois_ks,len(roi_rsq)),'rsq': roi_rsq})
    else:
        df_rsq = df_rsq.append(pd.DataFrame({'roi': np.tile(rois_ks,len(roi_rsq)),'rsq': roi_rsq}),
                                                   ignore_index = True)


# make figures
fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

v1 = sns.violinplot(data = df_rsq, x = 'roi', y = 'rsq', 
                    cut=0, inner='box', palette = ROI_pal, linewidth=1.8) # palette ='Set3',linewidth=1.8)

v1.set(xlabel=None)
v1.set(ylabel=None)
plt.margins(y=0.025)
#sns.swarmplot(x='ecc', y='cs', data=crwd_df4plot,color=".25",alpha=0.5)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)

plt.xlabel('ROI',fontsize = 20,labelpad=18)
plt.ylabel('RSQ',fontsize = 20,labelpad=18)
plt.ylim(0,1)

fig.savefig(op.join(figures_pth,'rsq_visual_violinplot.svg'), dpi=100)

images = {}

images['rsq'] = cortex.Vertex(rsq, 
                             space,
                               vmin = 0, vmax = 0.7,
                               cmap='Reds')
#cortex.quickshow(images['rsq'],with_curvature=True,with_sulci=True)

filename = op.join(figures_pth,'flatmap_space-fsaverage_type-rsq_visual.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

# now mask out nans, makes it look nicer

new_rsq = np.array([x if np.isnan(x)==False else 0 for _,x in enumerate(rsq)])

images['rsq_masked'] = cortex.Vertex(new_rsq, 
                             space,
                               vmin = 0, vmax = 0.7,
                               cmap='Reds')
#cortex.quickshow(images['rsq_masked'],with_curvature=True,with_sulci=True)

filename = op.join(figures_pth,'flatmap_space-fsaverage_type-rsq_masked_visual.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_masked'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)




