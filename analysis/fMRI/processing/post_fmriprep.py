import numpy as np
import os, sys
import os.path as op
from pathlib import Path
import glob
import shutil

import yaml

from FAM_utils import mri as mri_utils

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
file_ext = params['mri']['file_ext'][space] # file extension
confound_ext = params['mri']['confounds']['file_ext'] # file extension

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

confound_files = {'pRF':[op.join(fmriprep_dir,run) for _,run in enumerate(os.listdir(fmriprep_dir)) 
            if acq in run and 'pRF' in run and run.endswith(confound_ext)],
             'FA': [op.join(fmriprep_dir,run) for _,run in enumerate(os.listdir(fmriprep_dir)) 
            if acq in run and 'FA' in run and run.endswith(confound_ext)]}


# dict to store names of processed files, per task
proc_files = {'pRF': [],
             'FA': []}

# per task

for _,task in enumerate(['pRF','FA']):
    
    task_name = 'feature' if task == 'FA' else 'prf' # due to params yml notation, should change later

    # load and convert files in numpy arrays, to make format issue obsolete
    epi_files[task] = mri_utils.load_data_save_npz(epi_files[task], output_dir, save_subcortical=True)
    
    ## crop files, due to "dummies"
    crop_TR = params[task_name]['dummy_TR'] + params[task_name]['crop_TR'] if params[task_name]['crop'] == True else params[task_name]['dummy_TR'] 

    proc_files[task] = mri_utils.crop_epi(epi_files[task], output_dir, num_TR_crop = crop_TR)

    if params[task_name]['regress_confounds']: # if regressing confounds
    
        # first sub select confounds that we are using, and store in output dir
        confounds_list = mri_utils.select_confounds(confound_files['FA'], output_dir, reg_names = params['mri']['confounds']['regs'],
                                                    CumulativeVarianceExplained = params['mri']['confounds']['CumulativeVarianceExplained'],
                                                    select =  'num', num_components = 5,
                                                    num_TR_crop = crop_TR)
        
        # regress out confounds, and percent signal change
        proc_files[task] = mri_utils.regressOUT_confounds(proc_files[task], confounds_list, output_dir, TR = params['mri']['TR'])
        
    else: 
        ## filter files, to remove drifts
        proc_files[task] = mri_utils.filter_data(proc_files[task], output_dir, filter_type = params['mri']['filtering']['type'], 
                                first_modes_to_remove = params['mri']['filtering']['first_modes_to_remove'], plot_vert=True)
        
        ## percent signal change finals
        proc_files[task] = mri_utils.psc_epi(proc_files[task], output_dir)
            
        
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
                hemi_files.append(mri_utils.average_epi([val for val in proc_files[task] if hemi in val], 
                                              final_output_dir, method = 'mean'))
        
            proc_files[task] = hemi_files
            
        else:
            proc_files[task] = mri_utils.average_epi(proc_files[task], final_output_dir, method = 'mean')
        
    
    else:
        # save FA files in final output folder too
        for f in proc_files[task]:
            shutil.copyfile(f, op.join(final_output_dir,op.split(f)[-1]))

    

