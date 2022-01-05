## run SPM 12 bias field correction on T1w and (if present) T2w images ##
## actual script taken from https://github.com/layerfMRI/repository/tree/master/bias_field_corr ##
## all credits go to Renzo Huber ##

import os, sys
import os.path as op
import glob

import yaml

from nilearn.plotting import plot_anat
import matplotlib.pyplot as plt

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

if base_dir in ['lisa','cartesius']:

    raise NameError('Cannot run BFC on slurm systems - needs MATLAB')

# get current repo path
repo_pth = os.getcwd()

# matlab install location
matlab_pth = params['mri']['paths'][base_dir]['matlab']
  
# path to source data       
sourcedata_pth = glob.glob(op.join(params['mri']['paths'][base_dir]['root'], 'sourcedata','sub-{sj}'.format(sj=sj),'ses-*','anat'))[0]

# list original (uncorrected) files (can be T1 or T2)
orig_files = [op.join(sourcedata_pth,run) for _,run in enumerate(os.listdir(sourcedata_pth)) 
            if run.endswith('.nii.gz') and ('T1w' in run or 'T2w' in run)]; orig_files.sort()


# make output folder to store copy of original, tmp and output files
out_pth = op.join(params['mri']['paths'][base_dir]['root'], 'BiasFieldCorrection')

if not op.exists(out_pth):
    os.makedirs(out_pth)
print('saving files in %s'%out_pth)

    
# loop over files we want to bias field correct
for orig in orig_files:  
    
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
            cp bico_$INPUT.gz new_$INPUT.gz
            mv new_$INPUT.gz $ORIG # move to sourcedata again
            
            echo SUCCESS
            
            """
            
            batch_dir = op.join(params['mri']['paths'][base_dir]['root'],'batch')
            if not op.exists(batch_dir):
                    os.makedirs(batch_dir)

        else: # assumes slurm systems

            raise NameError('NOT IMPLEMENTED ON %s'%base_dir)
            
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
        js_name = op.join(batch_dir, 'BFC-' + op.split(orig)[-1].replace('.nii.gz','.nii') + '.sh')
        of = open(js_name, 'w')
        of.write(batch_string)
        of.close()

        print('submitting ' + js_name + ' to queue')
        print(batch_string)
        os.system('sh ' + js_name) if base_dir == 'local' else os.system('sbatch ' + js_name)


    # make image with before vs after, for comparison
    # and save in dir

    uncorr_file = op.join(outfolder,op.split(orig)[-1])[:-3] # remove gz
    corr_file = op.join(outfolder,'bico_'+op.split(orig)[-1])

    vmax = 50000 if 'T1w' in corr_file else 200000

    # create a figure with multiple axes to plot each anatomical image
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    plot_anat(uncorr_file,vmax=vmax, title='uncorrected', axes=axes[0],cut_coords=(0,-4,-2))
    plot_anat(corr_file,vmax=vmax, title='corrected', axes=axes[1],cut_coords=(0,-4,-2))

    # save the output figure with all the anatomical images
    fig.savefig("%s_BFC.png"%op.join(outfolder,op.split(orig)[-1].split('.')[0]),dpi=100)



