import os, sys
import os.path as op
import argparse

import yaml
from FAM.processing import load_exp_settings, preproc_mridata

# load settings from yaml
with open('exp_params.yml', 'r') as f_in:
    params = yaml.safe_load(f_in)


## to get inputs 
parser = argparse.ArgumentParser()

parser.add_argument("--subject",
                    nargs = "*", # 0 or more values expected => creates a list
                    type = str,  # any type/callable can be used here
                    default = [],
                    required = True,
                    help = 'Subject number (ex:1). If "all" will run for all participants. If list of subs, will run only those (ex: 1 2 3 4)'
                    )
parser.add_argument("--step", 
                    type = str.lower, 
                    required = True,
                    help = "Step of processing pipeline we want to run: anat_preproc, fmriprep, nordic, etc..."
                    )
parser.add_argument("--dir", 
                    type = str.lower, 
                    default = 'local',
                    help = "System we are making plots in - local [default] vs slurm (snellius)"
                    )
parser.add_argument("--data_type", 
                    type = str.lower, 
                    default = 'func',
                    help = "Type of data to process (func [default] or anat)"
                    )
parser.add_argument("--exclude_sj", 
                    nargs = '*', # 0 or more values expected => creates a list
                    default = [],
                    type = int,
                    help = "List of subs to exclude (ex: 1 2 3 4). Default []"
                    )
parser.add_argument("--use_T2", 
                    action = 'store_true',
                    help = "if option called, will consider T2 file (only relevant for freeview command)")
parser.add_argument("--fs_cmd", 
                    type = str, 
                    default = 'all',
                    help = "Freesurfer command to run (all [default], t2, pial)"
                    )
parser.add_argument("--node_name", 
                    type = str, 
                    help = "Node name, to send job to [default None]"
                    )
parser.add_argument("--partition_name", 
                    type = str, 
                    help = "Partition name, to send job to [default None]"
                    )
parser.add_argument("--node_mem", 
                    type = int, 
                    default = 5000,
                    help = "fmriprep memory limit for processes [default 5000]"
                    )
parser.add_argument("--batch_mem_Gib", 
                    type = int, 
                    default = 90,
                    help = "Node memory limit [default 90]"
                    )
parser.add_argument("--low_mem", 
                    action = 'store_true',
                    help = "if option called, pass the --low-mem option in fmriprep")
parser.add_argument("--use_fmap", 
                    action = 'store_true',
                    help = "if option called, use/ignore fieldmaps (when present) during fmriprep processing")
parser.add_argument("--json_folder", 
                    type = str, 
                    default = 'fmap', 
                    help = "Update jason files from which folder (fmap [default] vs func)"
                    )

# parse the command line
args = parser.parse_args()

# access parser options
sj = args.subject[0] if len(args.subject) == 1 else args.subject # for situation where 1 sj vs list
step = args.step # what step of pipeline we want to run
exclude_sj = args.exclude_sj # list of excluded subjects
system_dir = args.dir
data_type = args.data_type
T2_file = args.use_T2
node_name = args.node_name # node name to submit slurm job 
partition_name = args.partition_name # partition name to submit slurm job
node_mem = args.node_mem 
batch_mem_Gib = args.batch_mem_Gib
low_mem = args.low_mem # to reduce memory usage, impacting disk space
use_fmap = args.use_fmap
fs_cmd = args.fs_cmd # freesurfer commands
json_folder = args.json_folder 

## Load data object --> as relevant paths, variables and utility functions
print("Preprocessing {data} data for subject {sj}!".format(data=data_type, sj=sj))

FAM_data = load_exp_settings.MRIData(params, sj, 
                                    repo_pth = op.split(load_exp_settings.__file__)[0], 
                                    base_dir = system_dir, exclude_sj = exclude_sj)

print('Subject list to vizualize is {l}'.format(l=str(FAM_data.sj_num)))

## Load preprocessing class for each data type ###
# get mri info
FAM_mri = preproc_mridata.PreprocMRI(FAM_data)


## run specific steps ##
match step:
    case 'anat_preproc':

        print('Running {step}\ Check if we ran preprocessing for anatomical data'.format(step=step))
        
        FAM_mri.check_anatpreproc(T2file = T2_file)

    case 'freesurfer':

        print('Running {step}\ Running Freesurfer 7.2 on T1w and T2w (if available)'.format(step=step))

        FAM_mri.call_freesurfer(cmd = fs_cmd, node_name = node_name, partition_name = partition_name,
                                            batch_mem_Gib = batch_mem_Gib)

    case 'fmriprep':

        print('Running {step}\ Running fmriprep singularity'.format(step=step))

        FAM_mri.call_fmriprep(data_type = data_type, node_name = node_name, partition_name = partition_name, use_fmap=use_fmap, 
                                    node_mem = node_mem, batch_mem_Gib = batch_mem_Gib, low_mem = low_mem)

    case 'nordic':

        print('Running {step}\ Run NORDIC on functional files'.format(step=step))

        FAM_mri.NORDIC(participant=sj)

    case 'func_preproc':

        print('Running {step}\ Check if we ran preprocessing for functional data'.format(step=step))

        FAM_mri.check_funcpreproc()

    case 'mriqc':

        print('Running {step}\ Run MRIQC on data'.format(step=step))

        if FAM_data.base_dir == 'local':
            FAM_mri.call_mriqc(batch_dir = op.join(FAM_data.proj_root_pth, 'batch'))
        else:
            FAM_mri.call_mriqc()

    case 'up_json':

        print('Running {step}\ Update json params given raw PAR/REC header info'.format(step=step))

        FAM_mri.update_jsons(participant = sj, json_folder = json_folder)

    case 'post_fmriprep':

        print('Running {step}\ Run final processing steps on functional data (after fmriprep)'.format(step=step))

        FAM_mri.post_fmriprep_proc(save_subcortical = False)
