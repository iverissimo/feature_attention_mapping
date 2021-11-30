################################################
#      Plot bold func data, 
#    (after postfmriprep, so HP and psc)
#    and make video of bold change by TR
#    to check for visual stimuli
################################################

import numpy as np
import os, sys
import os.path as op
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import cortex

sys.path.insert(0,'..') # add parent folder to path
from utils import * #import script to use relevante functions


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

TR = params['mri']['TR']

# define file extension that we want to use, 
# should include processing key words
file_ext = '_cropped_{filt}_{stand}.npy'.format(filt = params['mri']['filtering']['type'],
                                                    stand = 'psc')

# set paths
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
postfmriprep_dir = op.join(derivatives_dir,'post_fmriprep','sub-{sj}'.format(sj=sj),space,'processed')

# output dir to save fit and plot
figures_pth = op.join(derivatives_dir,'plots', 'bold', task, 'sub-{sj}'.format(sj=sj), space,'run-{run}'.format(run=run_type)) # path to save plots
if not os.path.exists(figures_pth):
    os.makedirs(figures_pth) 

# list with absolute file names to be fitted (iff gii, then 2 hemispheres)
proc_files = [op.join(postfmriprep_dir, h) for h in os.listdir(postfmriprep_dir) if 'task-pRF' in h and
                 'acq-{acq}'.format(acq=acq) in h and run_type in h and h.endswith(file_ext)]

## load functional data
file = proc_files[0]
data = np.load(file,allow_pickle=True) # will be (vertex, TR)

# get pycortex sub
pysub = params['plotting']['pycortex_sub'] 

# make movie
movie_name = op.join(figures_pth,'flatmap_space-{space}_type-BOLD_visual_movie.mp4'.format(space=pysub))

if not op.isfile(movie_name):

    for num_tr in range(data.shape[-1]):

        flatmap = cortex.Vertex(data[...,num_tr], 
                                pysub,
                                vmin = -5, vmax = 5,
                                cmap='BuBkRd')
        #cortex.quickshow(flatmap,with_curvature=True,with_sulci=True)
        
        filename = op.join(figures_pth,'flatmap_space-{space}_type-BOLD_visual_TR-{time}.png'.format(space=pysub,
                                                                                            time=str(num_tr).zfill(3)))
        print('saving %s' %filename)
        _ = cortex.quickflat.make_png(filename, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)


    ## save as video
    img_name = filename.replace('_TR-%s.png'%str(num_tr).zfill(3),'_TR-%3d.png')
    os.system("ffmpeg -r 6 -start_number 0 -i %s -vcodec mpeg4 -y %s"%(img_name,movie_name)) 

else:
    print('movie already exists as %s'%movie_name)