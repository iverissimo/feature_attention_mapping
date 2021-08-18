## run NORDIC of functional files ##
## actual script from Luca's Vizioli, all credits go to him ##


import os, sys
import os.path as op
import numpy as np
from shutil import copy2

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

# tasks to apply NORDIC
tasks = ['pRF','FA']

# get current repo path
repo_pth = os.getcwd()

# matlab install location
matlab_pth = params['mri']['paths'][base_dir]['matlab']

# make output folder to store copy of original, tmp and output files
out_pth = op.join(params['mri']['paths'][base_dir]['root'], 'NORDIC')
if not op.exists(out_pth):
    os.makedirs(out_pth)
print('saving files in %s'%out_pth)


# path to input folder
input_folder = op.join(out_pth,'pre_nordic','sub-{sj}'.format(sj=sj),'ses-{ses}'.format(ses=ses))

# list original (uncorrected) files 
input_mag = [op.join(input_folder,run) for _,tsk in enumerate(tasks) 
             for _,run in enumerate(os.listdir(input_folder)) 
             if run.endswith('.nii.gz') and 'acq-standard' in run and 'phase' not in run
             and tsk in run]; input_mag.sort()


input_phase = [op.join(input_folder,run) for _,tsk in enumerate(tasks) 
               for _,run in enumerate(os.listdir(input_folder)) 
               if run.endswith('_phase.nii.gz') and 'acq-standard' in run
               and tsk in run]; input_phase.sort()

# path to source data 
sourcedata_pth = op.join(params['mri']['paths'][base_dir]['root'], 'sourcedata','sub-{sj}'.format(sj=sj),
                        'ses-{ses}'.format(ses=ses),'func')

# if mag files not in source data, copy them there (we still want to process the non nordic data)
for _,file in enumerate(input_mag):
    outfile = file.replace(input_folder,sourcedata_pth)
    
    if op.exists(outfile):
        print('already exists %s'%outfile)
    else:
        copy2(file,outfile)
        print('file copied to %s'%outfile)
        

# path to output folder
output_folder = op.join(out_pth,'post_nordic','sub-{sj}'.format(sj=sj),'ses-{ses}'.format(ses=ses))
if not op.exists(output_folder):
    os.makedirs(output_folder)
print('saving files in %s'%output_folder)


# loop over files, make sure using correct phase
# (this is, with same run and phase)

for _,tsk in enumerate(tasks):
    
    for run in range(len([file for _, file in enumerate(input_mag) 
                          if 'task-{task}'.format(task=tsk) in file])):

        mag = [val for _,val in enumerate(input_mag) 
               if 'run-{run}'.format(run=str(run+1)) in val and 
               'task-{task}'.format(task=tsk) in val][0]
        
        phase = [val for _,val in enumerate(input_phase) 
               if 'run-{run}'.format(run=str(run+1)) in val and 
               'task-{task}'.format(task=tsk) in val][0]
        
        nordic_nii = mag.replace(input_folder,output_folder).replace('acq-standard','acq-nordic')
        
        # if file aready exists, skip
        if op.exists(nordic_nii):
             print('NORDIC ALREDY PERFORMED ON %s,\nSKIPPING'%nordic_nii)
                
        else:

            if base_dir == 'local': # for local machine
    
                batch_string = """#!/bin/bash
                
                echo "applying nordic to $INMAG"

                cd $FILEPATH # go to the folder

                cp $REPO/NIFTI_NORDIC.m ./NIFTI_NORDIC.m # copy matlab script to here
                
                $MATLAB -nodesktop -nosplash -r "NIFTI_NORDIC('$INMAG', '$INPHASE', '$OUTFILE'); exit;" # execute the NORDIC script in matlab
                
                pigz $OUTFILE.nii # compress file

                mv $OUTFILE.nii.gz $OUTPATH # move to post nordic folder

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
                cp -r $ROOTFOLDER/NORDIC/ $TMPDIR

                wait
                

                $MATLAB -nodesktop -nosplash -r "Bias_field_script_job" # execute the SPM script in matlab
                
                
                
                echo SUCCESS
                
                wait
                
                rsync -chavzP $TMPDIR/sourcedata/ $ROOTFOLDER/sourcedata
                rsync -chavzP $TMPDIR/BiasFieldCorrection/ $ROOTFOLDER/BiasFieldCorrection

                wait          # wait until programs are finished

                echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"

                """

                batch_dir = '/home/inesv/batch/'


            keys2replace = {'$SJ_NR': str(sj).zfill(3),
                            '$FILEPATH': out_pth,
                            '$REPO': repo_pth,
                            '$INMAG': mag,
                            '$INPHASE': phase, 
                            '$OUTFILE': op.split(nordic_nii)[-1].replace('.nii.gz',''),
                            '$OUTPATH': nordic_nii,
                            '$MATLAB': matlab_pth,
                            '$ROOTFOLDER': params['mri']['paths'][base_dir]['root'] 
                             }

            # replace all key-value pairs in batch string
            for key, value in keys2replace.items():
                batch_string = batch_string.replace(key, value)
                    
            # run it
            js_name = op.join(batch_dir, 'NORDIC-' + op.split(nordic_nii)[-1].replace('.nii.gz','.sh'))
            of = open(js_name, 'w')
            of.write(batch_string)
            of.close()

            print('submitting ' + js_name + ' to queue')
            print(batch_string)
            os.system('sh ' + js_name) if base_dir == 'local' else os.system('sbatch ' + js_name)

            






