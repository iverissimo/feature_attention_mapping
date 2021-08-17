## run FMRIPREP on T1w and (if present) T2w images ONLY##
## first step to check if segmentations are fine ##

import os, sys
import os.path as op
import numpy as np

import yaml

# load settings from yaml
with open(op.join(op.split(op.split(os.getcwd())[0])[0],'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

# define participant number
if len(sys.argv)<3: 
    raise NameError('Please add subject number (ex: 001) '
                    'as 1st argument in the command line!')
elif len(sys.argv)<2:
    raise NameError('Please specify where running data (local vs lisa)'
                    'as 2nd argument in the command line!')

else:
    sj = str(sys.argv[1]).zfill(3) #fill subject number with 00 in case user forgets
    base_dir = str(sys.argv[2]) # which machine we run the data

# path to singularity image
sing_img = '/home/inesv/my_images/fmriprep.20.2.1.simg'


batch_string = """#!/bin/bash
#SBATCH -t 96:00:00
#SBATCH -N 1 --mem=60G

# call the programs
echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

PYTHONPATH="" singularity run --cleanenv -B /project/k_lab \
$SINGIMG \
$ROOTFOLDER/sourcedata $ROOTFOLDER/derivatives/ participant \
--participant-label sub-$SJ_NR --output-space T1w  \
--nthreads 30 --omp-nthreads 30 --low-mem --fs-license-file $FREESURFER/license.txt \
--anat-only -w /scratch/FAM_wf

wait          # wait until programs are finished

echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"
"""

batch_dir = '/home/inesv/batch/'
os.chdir(batch_dir)

keys2replace = {'$SJ_NR': str(sj).zfill(3),
                '$SINGIMG': sing_img,
                '$ROOTFOLDER': params['mri']['paths'][base_dir]['root'] 
                 }

# replace all key-value pairs in batch string
for key, value in keys2replace.items():
    batch_string = batch_string.replace(key, value)
    
# run it
js_name = op.join(batch_dir, 'FMRIPREP-' + str(sj).zfill(3) + '_FAM_anat.sh')
of = open(js_name, 'w')
of.write(batch_string)
of.close()

print('submitting ' + js_name + ' to queue')
print(batch_string)
os.system('sh ' + js_name) if base_dir == 'local' else os.system('sbatch ' + js_name)

