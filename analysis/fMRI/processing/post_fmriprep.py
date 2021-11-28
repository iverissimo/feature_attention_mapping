import numpy as np
import os, sys
import os.path as op
from pathlib import Path
import glob
import shutil

import yaml

sys.path.insert(0,'..') # add parent folder to path
from utils import * #import script to use relevante functions


# define participant number, and if looking at nordic or pre-nordic data
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex: 001) '
                    'as 1st argument in the command line!')

else:
    sj = str(sys.argv[1]).zfill(3) #fill subject number with 00 in case user forgets
    

# load settings from yaml
with open(op.join(str(Path(os.getcwd()).parents[1]),'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

# some settings
base_dir = params['general']['current_dir'] # which machine we run the data
acq = params['mri']['acq'] # if using standard files or nordic files
space = params['mri']['space'] # subject space
file_ext = params['mri']['file_ext'] # file extension

hemispheres = ['hemi-L','hemi-R'] # only used for gifti files

# set paths
derivatives_dir = params['mri']['paths'][base_dir]['derivatives']
output_dir =  op.join(derivatives_dir,'post_fmriprep','sub-{sj}'.format(sj=sj),space)

# if output path doesn't exist, create it
if not op.isdir(output_dir): 
    os.makedirs(output_dir)
print('saving files in %s'%output_dir)

# get list of functional files to process, per task
fmriprep_dir = glob.glob(op.join(derivatives_dir, 'fmriprep', 'sub-{sj}'.format(sj=sj), 'ses-*', 'func'))[0]

epi_files = {'pRF':[op.join(fmriprep_dir,run) for _,run in enumerate(os.listdir(fmriprep_dir)) 
            if space in run and acq in run and 'pRF' in run and run.endswith(file_ext)],
             'FA': [op.join(fmriprep_dir,run) for _,run in enumerate(os.listdir(fmriprep_dir)) 
            if space in run and acq in run and 'FA' in run and run.endswith(file_ext)]}

# exception for this run that could not be nordiced
if sj == '004':
    epi_files['FA'].append(op.join(fmriprep_dir,'sub-{sj}_ses-1_task-FA_acq-standard_run-4_space-{space}_hemi-L{file_ext}'.format(sj=sj, space=space, file_ext=file_ext))) 
    epi_files['FA'].append(op.join(fmriprep_dir,'sub-{sj}_ses-1_task-FA_acq-standard_run-4_space-{space}_hemi-R{file_ext}'.format(sj=sj, space=space, file_ext=file_ext)))

# dict to store names of processed files, per task
proc_files = {'pRF': [],
             'FA': []}

# per task

for _,task in enumerate(['pRF','FA']):
    
    task_name = 'feature' if task == 'FA' else 'prf' # due to params yml notation, should change later

    ## crop files, due to "dummies" 
    proc_files[task] = crop_epi(epi_files[task], output_dir, num_TR_task = params[task_name]['total_number_TR'], 
                                           num_TR_crop = params[task_name]['dummy_TR'])

    ## filter files, to remove drifts
    proc_files[task] = filter_data(proc_files[task], output_dir, filter_type = params['mri']['filtering']['type'], 
                            cut_off_hz = params['mri']['filtering']['cut_off_hz'], plot_vert=False)

    ## percent signal change finals
    proc_files[task] = psc_epi(proc_files[task], output_dir)
        
        
    ## make new outdir, to save final files that will be used for further analysis
    # avoids mistakes later on
    final_output_dir =  op.join(output_dir, 'processed')
    # if output path doesn't exist, create it
    if not op.isdir(final_output_dir): 
        os.makedirs(final_output_dir)
    print('saving FINAL processed files in %s'%final_output_dir)

    ## average all runs for pRF task
    if task == 'pRF':
        
        if '.func.gii' in file_ext:
            
            hemi_files = []
            
            for hemi in hemispheres:
                hemi_files.append(average_epi([val for val in proc_files[task] if hemi in val], 
                                              final_output_dir, method = 'median'))
        
            proc_files[task] = hemi_files
            
        else:
            proc_files[task] = average_epi(proc_files[task], final_output_dir, method = 'median')
        
    
    else:
        # save FA files in final output folder too
        for f in proc_files[task]:
            shutil.copyfile(f, op.join(final_output_dir,op.split(f)[-1]))

    

