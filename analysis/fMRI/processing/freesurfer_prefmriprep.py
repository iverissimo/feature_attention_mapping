
## run FREESURFER 7.2 on outputs of first fmriprep try out (anat only)##
## because 7.2 includes T2 in calculations ##

import os, sys
import os.path as op
import glob

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

# path to store freesurfer outputs 
out_dir = op.join(params['mri']['paths'][base_dir]['root'],'pre_fmriprep','sub-{sj}'.format(sj=sj))

if not op.exists(out_dir):
    os.makedirs(out_dir)
print('saving files in %s'%out_dir)

# T1 and T2 filenames
anat_dir = glob.glob(op.join(params['mri']['paths'][base_dir]['root'], 'sourcedata','sub-{sj}'.format(sj=sj),'ses-*','anat'))[0]
t1_filename = [op.join(anat_dir,run) for _,run in enumerate(os.listdir(anat_dir)) if run.endswith('.nii.gz') and 'T1w' in run]
t2_filename = [op.join(anat_dir,run) for _,run in enumerate(os.listdir(anat_dir)) if run.endswith('.nii.gz') and 'T2w' in run]

batch_string = """#!/bin/bash
#SBATCH -t 96:00:00
#SBATCH -N 1 --mem=65536
#SBATCH --cpus-per-task=16
#SBATCH -v
#SBATCH --output=/home/inesv/batch/slurm_output_%A.out

# call the programs
echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

conda activate i36

cp -r $ANATDIR $TMPDIR
wait
cp -r $OUTDIR/$SJ_NR $TMPDIR

wait

export SUBJECTS_DIR=$TMPDIR/$SJ_NR

wait

cd $SUBJECTS_DIR

wait

recon-all -s $SJ_NR -hires -i $T1_file \
    -T2 $T2_file -T2pial -all

wait

rsync -chavzP $SUBJECTS_DIR/ $OUTDIR

wait          # wait until programs are finished

echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"
"""

batch_dir = '/home/inesv/batch/'
os.chdir(batch_dir)

keys2replace = {'$SJ_NR': 'sub-{sj}'.format(sj = str(sj).zfill(3)),
                '$ANATDIR': anat_dir,
                '$OUTDIR': op.split(out_dir)[0], 
                '$T1_file': t1_filename[0].replace(anat_dir,'/scratch/anat'),
                '$T2_file': t2_filename[0].replace(anat_dir,'/scratch/anat')
                 }

# replace all key-value pairs in batch string
for key, value in keys2replace.items():
    batch_string = batch_string.replace(key, value)
    
# run it
js_name = op.join(batch_dir, 'FREESURFER7-' + str(sj).zfill(3) + '_FAM_prefmriprep.sh')
of = open(js_name, 'w')
of.write(batch_string)
of.close()

print('submitting ' + js_name + ' to queue')
print(batch_string)
os.system('sh ' + js_name) if base_dir == 'local' else os.system('sbatch ' + js_name)






