
# quick script to crop first dummy scans from fmap files

import os, sys
import os.path as op
import glob
from shutil import copy2

import yaml

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

# number of dummy TRS
tr_n = 5

# path to sourcedata fmaps
fmap_folder = op.join(params['mri']['paths'][base_dir]['sourcedata'], 'sub-{sj}'.format(sj = sj), 'ses-1', 'fmap')

# list of original niftis
orig_nii_files = [op.join(fmap_folder, val) for val in os.listdir(fmap_folder) if val.endswith('.nii.gz')]

# make folder to store original niftis (in case something goes wrong)
new_folder = op.join(params['mri']['paths'][base_dir]['root'], 'orig_fmaps', 'sub-{sj}'.format(sj = sj))
if not op.isdir(new_folder):
    os.makedirs(new_folder)

# then for each file
for file in orig_nii_files:

    # copy the original to the new folder
    copy2(file, op.join(new_folder, op.split(file)[-1]))

    # and crop the one in sourcedata
    os.system('fslroi {old} {new} {tr_n} {tr_n2}'.format(old = file, new = file, 
                                                        tr_n = tr_n, tr_n2 = tr_n))
