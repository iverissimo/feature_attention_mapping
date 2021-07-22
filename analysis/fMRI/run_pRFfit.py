
## PRF FITTING ##

import re
import os
import glob
import yaml
import sys
import numpy as np

# load settings from yaml
with open(os.path.join(os.path.split(os.getcwd())[0],'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

subjects = ['007'] # subjects
base_dir = 'lisa' # where we are running the scripts
preproc = ['standard','nordic'] 

acq_type = params['mri']['fitting']['pRF']['acq_type'] # type of acquisition

# number of slices to split data in (makes fitting faster)
total_slices = 89 # slice in z direction


batch_string = """#!/bin/bash
#SBATCH -t 96:00:00
#SBATCH -N 1 --mem=60G
#SBATCH -v
#SBATCH --output=/home/inesv/batch/slurm_output_%A.out

# call the programs
echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

conda activate i36

rsync -chavzP $ORIGDIR/ $TMPDIR/$PREPROC

wait

python post_fmriprep.py $SJ_NR $BASE_DIR $PREPROC

wait          # wait until programs are finished

python pRF_fitting.py $SJ_NR $BASE_DIR $PREPROC $ACQ $SLICE

wait          # wait until programs are finished

cp $TMPDIR/$PREPROC/pRF_fitting/*.nii.gz $ORIGDIR/pRF_fitting/

wait          # wait until programs are finished

echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"
"""

batch_dir = '/home/inesv/batch/'


for subject in subjects:

        for proc in preproc:
 
                orig_dir = params['mri']['paths'][base_dir][proc+'_orig'] # original file dir, will be copied to scratch

                for acq in acq_type:

                        for slice_num in np.arange(total_slices):

                                working_string = batch_string.replace('$SJ_NR', str(subject).zfill(3))
                                working_string = working_string.replace('$BASE_DIR', base_dir)
                                working_string = working_string.replace('$PREPROC', proc)
                                working_string = working_string.replace('$ACQ', acq)
                                working_string = working_string.replace('$SLICE', str(slice_num).zfill(2))
                                working_string = working_string.replace('$ORIGDIR', orig_dir)

                                js_name = os.path.join(batch_dir, 'pRF_sub-' + str(subject).zfill(2) + '_pRFfit.sh')
                                of = open(js_name, 'w')
                                of.write(working_string)
                                of.close()

                                print('submitting ' + js_name + ' to queue')
                                print(working_string)
                                os.system('sbatch ' + js_name)

