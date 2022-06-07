
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
elif len(sys.argv)<4:
    raise NameError('Please specify type of freesurfer commad (all, pial)'
                    'as 3rd argument in the command line!') 

else:
    sj = str(sys.argv[1]).zfill(3) #fill subject number with 00 in case user forgets
    base_dir = str(sys.argv[2]) # which machine we run the data
    freesurfer_cmd = str(sys.argv[3]) # which freesurfer command

if base_dir in ['local']:
    raise NameError('Dont run freesurfer locally - only implemented in slurm systems')

list_cmd = ['all', 'pial']
if freesurfer_cmd not in list_cmd:
    raise NameError('Not implemented, use freesurfer commands from %s'%str(list_cmd))

# path to store freesurfer outputs, in derivatives
out_dir = op.join(params['mri']['paths'][base_dir]['root'],'derivatives','freesurfer','sub-{sj}'.format(sj=sj))
print('saving files in %s'%out_dir)

if not op.exists(out_dir):
    os.makedirs(out_dir)
elif len(os.listdir(out_dir)) > 0 and freesurfer_cmd == 'all':
    overwrite = ''
    while overwrite not in ('y','yes','n','no'):
        overwrite = input('dir already has files, continue with recon-all\n(y/yes/n/no)?: ')
    if overwrite in ['no','n']:
        raise NameError('directory already has files\nstopping analysis!')

# T1 and T2 filenames
anat_dir = glob.glob(op.join(params['mri']['paths'][base_dir]['root'], 'sourcedata','sub-{sj}'.format(sj=sj),'ses-*','anat'))[0]
t1_filename = [op.join(anat_dir,run) for _,run in enumerate(os.listdir(anat_dir)) if run.endswith('.nii.gz') and 'T1w' in run]
t2_filename = [op.join(anat_dir,run) for _,run in enumerate(os.listdir(anat_dir)) if run.endswith('.nii.gz') and 'T2w' in run]

## make T1w string, which accounts for several T1ws that will be averaged
t1_str = ''
for ind,t in enumerate(t1_filename):
    if ind==0:
        t1_str += t.replace(anat_dir,'/scratch/anat')
    else:
        t1_str += ' -i '+t.replace(anat_dir,'/scratch/anat')
        
## same but for T2w string
# although not sure if freesurfer actually averages them or takes last one
t2_str = ''
for ind,t in enumerate(t2_filename):
    if ind==0:
        t2_str += t.replace(anat_dir,'/scratch/anat')
    else:
        t2_str += ' -T2 '+t.replace(anat_dir,'/scratch/anat')

batch_string = """#!/bin/bash
#SBATCH -t 96:00:00
#SBATCH -N 1 --mem=65536
#SBATCH --cpus-per-task=16
#SBATCH -v
#SBATCH --output=/home/inesv/batch/slurm_FREESURFER_%A.out

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

if [ "$CMD" == all ]; then

    echo "running full pipeline (recon all)"
    
    recon-all -s $SJ_NR -hires -i $T1_file -T2 $T2_file -T2pial -all

elif [ "$CMD" == pial ]; then

    echo "running pial fixes"

    cd $TMPDIR

    recon-all -s $SJ_NR -hires -autorecon-pial

else
    echo "command not implemented, skipping"
    exit 1
fi

wait

rsync -chavzP $TMPDIR/$SJ_NR/ $OUTDIR

wait          # wait until programs are finished

echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"
"""

batch_dir = '/home/inesv/batch/'
os.chdir(batch_dir)

keys2replace = {'$SJ_NR': 'sub-{sj}'.format(sj = str(sj).zfill(3)),
                '$ANATDIR': anat_dir,
                '$OUTDIR': op.split(out_dir)[0], 
                '$T1_file': t1_str,
                '$T2_file': t2_str,
                '$CMD': freesurfer_cmd 
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






