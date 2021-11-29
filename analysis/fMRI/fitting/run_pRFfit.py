
################################################
#      run prf fit jobs through SLURM 
# (to use in cartesius or similar server)
################################################

import os, sys
import os.path as op
import yaml
from pathlib import Path

# load settings from yaml
with open(op.join(str(Path(os.getcwd()).parents[1]),'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

# define participant number, run and which chunk of data to fitted

if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')

elif len(sys.argv)<3:
    raise NameError('Please specify where running data (local vs lisa)'
                    'as 3rd argument in the command line!')

else:
    # fill subject number and chunk number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(2)
    base_dir = str(sys.argv[2]) # which machine we run the data

run_type = 'median'
acq = params['mri']['acq'] # if using standard files or nordic files
space = params['mri']['space'] # subject space

total_chunks = params['mri']['fitting']['pRF']['total_chunks'][space] # number of chunks that data was split in
# number of slices to split data in (makes fitting faster)
#total_slices = 89 # slice in z direction

code_base_dir = op.split(os.getcwd())[0] #base dir for python code on fMRI analysis

batch_string = """#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -N 1 --mem=65536
#SBATCH --cpus-per-task=16
#SBATCH -v
#SBATCH --output=/home/inesv/batch/slurm-pRFfit_%A.out

# call the programs
echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

conda activate i36

# make derivatives dir in node
mkdir $TMPDIR/derivatives
mkdir $TMPDIR/derivatives/fmriprep
mkdir $TMPDIR/derivatives/post_fmriprep

#cp -r $DERIV_DIR/fmriprep/sub-$SJ_NR $TMPDIR/derivatives/fmriprep
cp -r $DERIV_DIR/post_fmriprep/sub-$SJ_NR $TMPDIR/derivatives/post_fmriprep

wait

#python $CODE_DIR/processing/post_fmriprep.py $SJ_NR

wait          # wait until programs are finished

python pRF_model.py $SJ_NR $RUN_TYPE $CHUNK_NR

wait          # wait until programs are finished

rsync -chavzP $TMPDIR/derivatives/ $DERIV_DIR

wait          # wait until programs are finished

echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"
"""

batch_dir = '/home/inesv/batch/'

for _,chu in enumerate(range(total_chunks)): # submit job for each chunk 
#for slice_num in np.arange(total_slices):

        working_string = batch_string.replace('$SJ_NR', str(sj).zfill(3))
        working_string = working_string.replace('$RUN_TYPE', run_type)
        working_string = working_string.replace('$CHUNK_NR', str(chu+1).zfill(3))
        working_string = working_string.replace('$CODE_DIR', code_base_dir)
        working_string = working_string.replace('$DERIV_DIR', op.join(params['mri']['paths'][base_dir]['root'], 'derivatives'))

        # run it
        js_name = op.join(batch_dir, 'pRF_sub-' + str(sj).zfill(3) + '_FAM.sh')
        of = open(js_name, 'w')
        of.write(working_string)
        of.close()

        print('submitting ' + js_name + ' to queue')
        print(working_string)
        os.system('sbatch ' + js_name)

