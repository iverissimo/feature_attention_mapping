## run NORDIC of functional files ##
## actual script from Luca's Vizioli, all credits go to him ##

## requires phase and mag data to be stored in a NORDIC folder:
# NORDIC/pre_nordic/sub-X/ses-Y/sub-X_ses-Y_..._bold_phase.nii.gz
# NORDIC/pre_nordic/sub-X/ses-Y/sub-X_ses-Y_..._bold.nii.gz

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

if base_dir in ['lisa','cartesius']:

    raise NameError('Cannot run BFC on slurm systems - needs MATLAB')

# tasks to apply NORDIC
tasks = ['pRF','FA']

# get current repo path
repo_pth = os.getcwd()

# matlab install location
matlab_pth = params['mri']['paths'][base_dir]['matlab']

# path to input folder
input_folder = glob.glob(op.join(params['mri']['paths'][base_dir]['root'],'NORDIC','pre_nordic','sub-{sj}'.format(sj=sj),'ses-*'))[0]

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
sourcedata_pth = glob.glob(op.join(params['mri']['paths'][base_dir]['root'], 'sourcedata','sub-{sj}'.format(sj=sj),'ses-*','func'))[0]

# if mag files not in source data, copy them there (we still want to process the non nordic data)
for _,file in enumerate(input_mag):
    outfile = file.replace(input_folder,sourcedata_pth)
    
    if op.exists(outfile):
        print('already exists %s'%outfile)
    else:
        copy2(file,outfile)
        print('file copied to %s'%outfile)
        
# path to output folder
output_folder = op.join(params['mri']['paths'][base_dir]['root'], 'NORDIC','post_nordic','sub-{sj}'.format(sj=sj),'ses-1')
if not op.exists(output_folder):
    os.makedirs(output_folder)
print('saving files in %s'%output_folder)

# loop over files, make sure using correct phase
# (this is, with same run and phase)

for _,tsk in enumerate(tasks):
    
    for _,filename in enumerate([file for _, file in enumerate(input_mag) 
                          if 'task-{task}'.format(task=tsk) in file]):
        
        # get run number, to ensure we use the phase and mag files for same run
        run = filename[-17:-12]

        mag = [val for _,val in enumerate(input_mag) 
               if run in val and 'task-{task}'.format(task=tsk) in val][0] 
    
        phase = [val for _,val in enumerate(input_phase) 
            if run in val and 'task-{task}'.format(task=tsk) in val][0]

        nordic_nii = op.join(output_folder,op.split(mag)[-1].replace('acq-standard','acq-nordic'))
        
        # if file aready exists, skip
        if op.exists(nordic_nii):
            print('NORDIC ALREADY PERFORMED ON %s,\nSKIPPING'%nordic_nii)
                
        else:

            if base_dir == 'local': # for local machine
    
                batch_string = """#!/bin/bash
                
                echo "applying nordic to $INMAG"

                cd $FILEPATH # go to the folder

                cp $REPO/NIFTI_NORDIC.m ./NIFTI_NORDIC.m # copy matlab script to here
                
                $MATLAB -nodesktop -nosplash -r "NIFTI_NORDIC('$INMAG', '$INPHASE', '$OUTFILE'); quit;" # execute the NORDIC script in matlab
                
                wait 

                pigz $OUTFILE.nii # compress file

                wait

                mv $OUTFILE.nii.gz $OUTPATH # move to post nordic folder

                echo SUCCESS
                
                """
                
                batch_dir = op.join(params['mri']['paths'][base_dir]['root'],'batch')
                if not op.exists(batch_dir):
                    os.makedirs(batch_dir)

            else: # assumes slurm systems
                
                print('NOT IMPLEMENTED ON %s'%base_dir)


            keys2replace = {'$SJ_NR': str(sj).zfill(3),
                            '$FILEPATH': op.join(params['mri']['paths'][base_dir]['root'], 'NORDIC'),
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

            # copy file to sourcedata
            copy2(nordic_nii,op.join(sourcedata_pth,op.split(nordic_nii)[-1]))
            print('file copied to %s'%op.join(sourcedata_pth,op.split(nordic_nii)[-1]))





