
import numpy as np
import os, sys
import os.path as op

import cortex

from nilearn import image

import yaml
from utils import * #import script to use relevante functions


# define participant number, and if looking at nordic or pre-nordic data
if len(sys.argv)<4: 
    raise NameError('Please add subject number (ex: 001) '
                    'as 1st argument in the command line!')
elif len(sys.argv)<3: 
	raise NameError('Please specify where running data (local vs lisa)'
                    'as 2nd argument in the command line!')

elif len(sys.argv)<2: 
    raise NameError('Please specify data we are looking at (nordic vs standard)'
                    'as 3rd argument in the command line!')

else:
    sj = str(sys.argv[1]).zfill(3) #fill subject number with 00 in case user forgets
    base_dir = str(sys.argv[2]) # which machine we run the data
    preproc = str(sys.argv[3]) # if using standard files or nordic files


# load settings from yaml
with open(op.join(op.split(os.getcwd())[0],'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)


derivatives_pth = params['mri']['paths'][base_dir][preproc]
output_pth =  op.join(derivatives_pth,'post_fmriprep')


# get func files
func_fmriprep = op.join(derivatives_pth, 'fmriprep', 'sub-{sj}'.format(sj=sj), 
                        'ses-{ses}'.format(ses=params['general']['session']),'func')

# get list of func epis
vol_list = [op.join(func_fmriprep,run) for _,run in enumerate(os.listdir(func_fmriprep)) 
            if 'T1w' in run and run.endswith('desc-preproc_bold.nii.gz')]
    

### FILTER RUNS
# if we want to filter runs
if params['mri']['filtering']['type'] is not None:

    filter_type = params['mri']['filtering']['type']

    data_path = op.join(output_pth, filter_type) # save filtered files in folder
    if not op.exists(data_path):
        print('output dir does not existing, saving files in %s'%data_path)
        os.makedirs(data_path)

    filtered_vol = []
    for _,run in enumerate(vol_list):

        filtered_vol.append(filter_data(run, 
                                        data_path, 
                                        TR = params['mri']['TR'],
                                        filter_type = filter_type,
                                        cut_off_hz = params['mri']['filtering'][filter_type]['cut_off_hz'],
                                        file_extension = params['mri']['filtering'][filter_type]['file_extension']))
    # replace filenames with filtered filenames
    vol_list = filtered_vol.copy()


### PSC signal
psc_vol = []
for _,run in enumerate(vol_list):
    
    data_path = op.join(output_pth, 'PSC') # save filtered files in folder
    if not op.exists(data_path):
        print('output dir does not existing, saving files in %s'%data_path)
        os.makedirs(data_path)

    psc_vol.append(psc(run, 
                    data_path, 
                    file_extension = params['mri']['psc']['file_extension']))
# replace filenames with filtered filenames
vol_list = psc_vol.copy()


### AVERAGE RUNS

data_path = op.join(output_pth, 'average') # save filtered files in folder
if not op.exists(data_path):
    print('output dir does not existing, saving files in %s'%data_path)
    os.makedirs(data_path)


# iterate over runs
avg_list = []
for _,acq in enumerate(params['general']['acq_type']):

    vols = [x for _,x in enumerate(vol_list) if 'acq-{acq}_'.format(acq=acq) in x]
    
    avg_list.append(avg_nii(vols, data_path))



