################################################
#      run FA gain model fit jobs through SLURM 
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
    raise NameError('Please add run to be fitted (ex: 1) '
                    'as 2nd argument in the command line!')

elif len(sys.argv)<4:
    raise NameError('Please specify where running data (local vs lisa)'
                    'as 3rd argument in the command line!')

else:
    # fill subject number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(2)
    run = str(sys.argv[2]) # run number
    base_dir = str(sys.argv[3]) # which machine we run the data

acq = params['mri']['acq'] # if using standard files or nordic files
space = params['mri']['space'] # subject space

batch_string = """#!/bin/bash
#SBATCH -t 40:00:00
#SBATCH -N 1 --mem=90G
#SBATCH -v
#SBATCH --output=/home/inesv/batch/slurm-FA_Gainfit_%A.out

# call the programs
echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

conda init bash
conda activate i38

# make sourcedata and derivatives dir in node
mkdir -p $TMPDIR/{derivatives/{post_fmriprep,pRF_fit,FA_gain}/sub-$SJ_NR,sourcedata/sub-$SJ_NR}

wait
cp -r $DERIV_DIR/post_fmriprep/sub-$SJ_NR/$SPACE $TMPDIR/derivatives/post_fmriprep/sub-$SJ_NR
wait
cp -r $DERIV_DIR/pRF_fit/sub-$SJ_NR/$SPACE $TMPDIR/derivatives/pRF_fit/sub-$SJ_NR
wait
cp -r $SOURCEDATA_DIR/sub-$SJ_NR $TMPDIR/sourcedata
wait

if [ -d "$DERIV_DIR/FA_gain/sub-$SJ_NR/$SPACE" ] 
then
    cp -r $DERIV_DIR/FA_gain/sub-$SJ_NR/$SPACE $TMPDIR/derivatives/FA_gain/sub-$SJ_NR
fi
wait

python FA_gain.py $SJ_NR $RUN 

wait          # wait until programs are finished

rsync -chavzP $TMPDIR/derivatives/ $DERIV_DIR

wait          # wait until programs are finished

echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"
"""

batch_dir = '/home/inesv/batch/'

working_string = batch_string.replace('$SJ_NR', str(sj).zfill(3))
working_string = working_string.replace('$RUN', run)
working_string = working_string.replace('$SPACE', space)
working_string = working_string.replace('$DERIV_DIR', op.join(params['mri']['paths'][base_dir]['root'], 'derivatives'))
working_string = working_string.replace('$SOURCEDATA_DIR', op.join(params['mri']['paths'][base_dir]['root'], 'sourcedata'))

# run it
js_name = op.join(batch_dir, 'FA_GAIN_sub-' + str(sj).zfill(3) + '_FAM.sh')
of = open(js_name, 'w')
of.write(working_string)
of.close()

print('submitting ' + js_name + ' to queue')
print(working_string)
os.system('sbatch ' + js_name)