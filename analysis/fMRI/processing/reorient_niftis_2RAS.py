# quick script to reorient niftis to RAS

import os, sys
import os.path as op
import numpy as np
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

    # for convenience, reorient all files to RAS+ (used by nibabel & fMRIprep) 
    orig_img = nb.load(file)
    orig_img_hdr = orig_img.header

    qform = orig_img_hdr['qform_code'] # set qform code to original

    canonical_img = nb.as_closest_canonical(orig_img)

    if qform != 0:
        canonical_img.header['qform_code'] = np.array([qform], dtype=np.int16)
    else:
        # set to 1 if original qform code = 0
        canonical_img.header['qform_code'] = np.array([1], dtype=np.int16)

    nb.save(canonical_img, file)

