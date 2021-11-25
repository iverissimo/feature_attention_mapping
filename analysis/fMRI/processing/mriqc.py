
## run MRIQC on functional sourcedata ##

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


#Make mriqc directory and participant directory in derivatives folder
mriqc_dir = op.join(params['mri']['paths'][base_dir]['root'], 'derivatives','mriqc','sub-{sj}'.format(sj=sj))
if not op.exists(mriqc_dir):
    os.makedirs(mriqc_dir)
print('saving files in %s'%mriqc_dir)

# path to singularity image
sing_img = '/home/inesv/my_images/mriqc-0.15.1.simg'


if base_dir == 'local': # for local machine

    batch_string = """#!/bin/bash

conda activate i36

wait

docker run -it --rm \
-v $ROOTFOLDER/sourcedata:/data:ro \
-v $ROOTFOLDER/derivatives/mriqc/sub-$SJ_NR:/out \
poldracklab/mriqc:latest /data /out participant --participant_label $SJ_NR


"""

else: # assumes slurm systems

    if sj == 'group':

        batch_string = """#!/bin/bash
#SBATCH -t 96:00:00
#SBATCH -N 1 --mem=65536
#SBATCH --cpus-per-task=16
#SBATCH -v
#SBATCH --output=/home/inesv/batch/slurm_output_%A.out

# call the programs
echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

conda activate i36

wait

echo ""
echo "Running MRIQC on participant $SJ_NR"
echo ""

wait

# make working directory in node
mkdir $TMPDIR/FAM_wf

wait

singularity run --cleanenv -B /project/projects_verissimo -B $TMPDIR/FAM_wf \
$SINGIMG \
$ROOTFOLDER/sourcedata $ROOTFOLDER/derivatives/mriqc \
group --hmc-fsl --float32 \
-w $TMPDIR/FAM_wf

wait          # wait until programs are finished

echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"
"""

    # for single subject
    else:

        batch_string = """#!/bin/bash
#SBATCH -t 96:00:00
#SBATCH -N 1 --mem=65536
#SBATCH --cpus-per-task=16
#SBATCH -v
#SBATCH --output=/home/inesv/batch/slurm_output_%A.out

# call the programs
echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

conda activate i36

wait

echo ""
echo "Running MRIQC on participant $SJ_NR"
echo ""

wait

# make working directory in node
mkdir $TMPDIR/FAM_wf

wait

singularity run --cleanenv -B /project/projects_verissimo -B $TMPDIR/FAM_wf \
$SINGIMG \
$ROOTFOLDER/sourcedata $ROOTFOLDER/derivatives/mriqc/sub-$SJ_NR \
participant \
--participant-label $SJ_NR \
--hmc-fsl \
--float32 \
-w $TMPDIR/FAM_wf

wait          # wait until programs are finished

echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"
"""

    batch_dir = '/home/inesv/batch/'

os.chdir(batch_dir)

keys2replace = {'$SJ_NR': sj,
                '$SINGIMG': sing_img,
                '$ROOTFOLDER': params['mri']['paths'][base_dir]['root'] 
                 }

# replace all key-value pairs in batch string
for key, value in keys2replace.items():
    batch_string = batch_string.replace(key, value)
    
# run it
js_name = op.join(batch_dir, 'MRIQC-' + str(sj).zfill(3) + '_FAM.sh')
of = open(js_name, 'w')
of.write(batch_string)
of.close()

print('submitting ' + js_name + ' to queue')
print(batch_string)
os.system('sh ' + js_name) if base_dir == 'local' else os.system('sbatch ' + js_name)


