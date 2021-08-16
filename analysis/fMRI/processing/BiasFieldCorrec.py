## run SPM 12 bias field correction on T1w and (if present) T2w images ##
## actual script taken from https://github.com/layerfMRI/repository/tree/master/bias_field_corr ##
## all credits go to Renzo Huber ##


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

# session number 
ses = '1'

# get current repo path
repo_pth = os.getcwd()

# matlab install location
matlab_pth = params['mri']['paths'][base_dir]['matlab']
  
# path to source data       
sourcedata_pth = op.join(params['mri']['paths'][base_dir]['root'], 'sourcedata','sub-{sj}'.format(sj=sj),
                        'ses-{ses}'.format(ses=ses),'anat')

# list original (uncorrected) files (can be T1 or T2)
orig_files = [op.join(sourcedata_pth,run) for _,run in enumerate(os.listdir(sourcedata_pth)) 
            if run.endswith('.nii.gz') and ('T1w' in run or 'T2w' in run)]; orig_files.sort()


# make output folder to store copy of original, tmp and output files
out_pth = op.join(params['mri']['paths'][base_dir]['root'], 'BiasFieldCorrection')

if not op.exists(out_pth):
    os.makedirs(out_pth)
print('saving files in %s'%out_pth)


if base_dir == 'local': # for local machine
    
    batch_string = """#!/bin/bash
    
    echo "moving $ORIG to new folder"
    mv $ORIG $OUTFOLDER # move original file to tmp folder
    
    cd $OUTFOLDER # go to the folder
    
    pigz -d $INPUT.gz # unzip the .nii.gz file
    
    cp $INPUT uncorr.nii
    
    echo "running SPM"
    cp $REPO/Bias_field_script_job.m ./Bias_field_script_job.m # copy matlab script to here
    
    $MATLAB -nodesktop -nosplash -r "Bias_field_script_job" # execute the SPM script in matlab
    
    mv muncorr.nii bico_$INPUT # rename output file

    rm uncorr.nii
    
    pigz bico_$INPUT
    
    echo "moving corrected $INPUT to original folder"
    mv bico_$INPUT.gz $ORIG # move to sourcedata again
    
    echo SUCCESS
    
    """
    
    batch_dir = op.join(params['mri']['paths'][base_dir]['root'],'batch')
    if not op.exists(batch_dir):
            os.makedirs(batch_dir)

else: # assumes slurm systems
    
    batch_string = """#!/bin/bash
    #SBATCH -t 96:00:00
    #SBATCH -N 1 --mem=65536
    #SBATCH --cpus-per-task=16
    #SBATCH -v
    #SBATCH --output=/home/inesv/batch/slurm_output_%A.out
    
    # call the programs
    echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

    conda activate i36
    
    mkdir $TMPDIR/PILOT
    
    # copy necessary folders to node tmp dir
    cp -r $ROOTFOLDER/sourcedata/ $TMPDIR
    cp -r $ROOTFOLDER/BiasFieldCorrection/ $TMPDIR

    wait
    
    echo "moving ${$ORIG/$ROOTFOLDER/$TMPDIR} to new folder"
    mv ${$ORIG/$ROOTFOLDER/$TMPDIR} ${$OUTFOLDER/$ROOTFOLDER/$TMPDIR} # move original file to tmp folder
    
    cd ${$OUTFOLDER/$ROOTFOLDER/$TMPDIR} # go to the folder
    
    pigz -d $INPUT.gz # unzip the .nii.gz file
    
    cp $INPUT uncorr.nii
    
    echo "running SPM"
    cp $REPO/Bias_field_script_job.m ./Bias_field_script_job.m # copy matlab script to here
    
    $MATLAB -nodesktop -nosplash -r "Bias_field_script_job" # execute the SPM script in matlab
    
    mv muncorr.nii bico_$INPUT # rename output file

    rm uncorr.nii
    
    pigz bico_$INPUT
    
    echo "moving corrected $INPUT to original folder"
    mv bico_$INPUT.gz ${$ORIG/$ROOTFOLDER/$TMPDIR} # move to sourcedata again
    
    echo SUCCESS
    
    wait
    
    rsync -chavzP $TMPDIR/sourcedata/ $ROOTFOLDER/sourcedata
    rsync -chavzP $TMPDIR/BiasFieldCorrection/ $ROOTFOLDER/BiasFieldCorrection

    wait          # wait until programs are finished

    echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"

    """

    batch_dir = '/home/inesv/batch/'
    
# loop over files we want to bias field correct
for orig in [orig_files[0]]:#orig_files:  
    
    # check if outfolder exists
    outfolder = op.join(out_pth,op.split(orig)[-1].replace('.nii.gz',''))
    
    # if exists and bias field corrected file there, skip
    if op.exists(outfolder) and op.exists(op.join(outfolder,'bico_'+op.split(orig)[-1])):
        
        print('BIAS FIELD CORRECTION ALREDY PERFORMED ON %s,\nSKIPPING'%orig)
    
    else:
        # proceed
        if not op.exists(outfolder):
            os.makedirs(outfolder)
        print('saving files in %s'%outfolder)

        keys2replace = {'$SJ_NR': str(sj).zfill(3),
                        '$ORIG': orig, 
                        '$OUTFOLDER': outfolder, 
                        '$INPUT': op.split(orig)[-1].replace('.nii.gz','.nii'),
                        '$REPO': repo_pth,
                        '$MATLAB': matlab_pth,
                        '$ROOTFOLDER': params['mri']['paths'][base_dir]['root'] 
                         }

        # replace all key-value pairs in batch string
        for key, value in keys2replace.items():
            batch_string = batch_string.replace(key, value)
            
    
        # run it
        js_name = os.path.join(batch_dir, 'BFC-' + op.split(orig)[-1].replace('.nii.gz','.nii') + '.sh')
        of = open(js_name, 'w')
        of.write(batch_string)
        of.close()

        print('submitting ' + js_name + ' to queue')
        print(batch_string)
        os.system('sbatch ' + js_name)




