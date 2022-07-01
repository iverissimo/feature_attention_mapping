# quick script to reorient niftis to RAS

import os, sys
import os.path as op
import glob
from shutil import copy2

import yaml
import nibabel as nb

# load settings from yaml
with open(op.join(op.split(op.split(os.getcwd())[0])[0],'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

# define participant number
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex: 001) '
                    'as 1st argument in the command line!')
elif len(sys.argv)<3:
    raise NameError('Please specify where running data (local vs lisa)'
                    'as 2nd argument in the command line!')

else:
    sj = str(sys.argv[1]).zfill(3) #fill subject number with 00 in case user forgets
    base_dir = str(sys.argv[2]) # which machine we run the data


# path to sourcedata fmaps
anat_processing_folder = op.join(params['mri']['paths'][base_dir]['root'], 'anat_preprocessing', 'sub-{sj}'.format(sj = sj), 'ses-1', 'anat')

# list of original niftis
orig_nii_files = [op.join(anat_processing_folder, val) for val in os.listdir(anat_processing_folder) if val.endswith('.nii.gz')]

# make folder to store original niftis (in case something goes wrong)
new_folder = op.join(params['mri']['paths'][base_dir]['root'], 'anat_preprocessing', 'sub-{sj}'.format(sj = sj), 'ses-1', 'orig_anat')
if not op.isdir(new_folder):
    os.makedirs(new_folder)

# then for each file
for file in orig_nii_files:

    # copy the original to the new folder
    copy2(file, op.join(new_folder, op.split(file)[-1]))

    # and crop the one in sourcedata
    orig_img = nb.load(file)

    canonical_img = nb.as_closest_canonical(orig_img)

    nb.save(canonical_img, file)
