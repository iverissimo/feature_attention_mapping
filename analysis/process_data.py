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
parser.add_argument("--subject", help="Subject number (ex: 001, or 'group'/'all')", required=True)
parser.add_argument("--step", type = str, help="Step of processing pipeline we want to run: anat_preproc, fmriprep, nordic, etc...", required=True)
# optional
parser.add_argument("--data_type", type = str, help="Type of data to process (func [default] or anat)")
parser.add_argument("--dir", type = str, help="System we are running analysis (lisa [default] vs local)")
parser.add_argument("--T2", type = int, help="Consider T2 file in analysis (0 [default] vs 1)")
parser.add_argument("--fs_cmd", type = str, help="Freesurfer command to run (all [default], t2, pial)")
#  only relevant if subject == group/all
parser.add_argument("--exclude_sj", nargs='+', help="List of subjects to exclude, define as --exclude_sj 0 1 ...", default=[])
# only relevant for LISA/slurm system and fMRIPREP calls
parser.add_argument("--node_name", type = str, help="Node name, to send job to [default None]")
parser.add_argument("--partition_name", type = str, help="Partition name, to send job to [default None]")
parser.add_argument("--node_mem", type = int, help="fmriprep memory limit for processes [default 5000]")
parser.add_argument("--batch_mem_Gib", type = int, help="Node memory limit [default 90]")
parser.add_argument("--low_mem", type = int, help="Set to True we are passing the --low-mem option in fmriprep [default True]")
# fmriprep fmap input
parser.add_argument("--use_fmap", type = int, help="Use/ignore fieldmaps if present during fmriprep processing, 1 [default] vs 0")
# update jsons
parser.add_argument("--json_folder", type = str, help="Update jason files from which folder (fmap [default] vs func)")

## set variables 
args = parser.parse_args()

# subject id and processing step of pipeline
sj = str(args.subject).zfill(3) 
step = args.step 
#
# if we want to exclude participants
exclude_sj = args.exclude_sj # list of excluded subjects
if len(exclude_sj)>0:
    exclude_sj = [val.zfill(3) for val in exclude_sj]
    print('Excluding participants {expp}'.format(expp = exclude_sj))
else:
    exclude_sj = []
#
# data type
data_type = args.data_type if args.data_type is not None else "func" 
#
# system location
system_dir = args.dir if args.dir is not None else "lisa" 
#
# if we want to consider T2 file 
T2_file = bool(args.T2) if args.T2 is not None else False # make it boolean
#
# SLURM and fmriprep options
# for LISA
node_name = args.node_name # node name to submit slurm job (or None)
partition_name = args.partition_name # partition name to submit slurm job (or None)
node_mem = args.node_mem if args.node_mem is not None else 5000
batch_mem_Gib = args.batch_mem_Gib if args.batch_mem_Gib is not None else 90
# for fmriprep
low_mem = bool(args.low_mem) if args.low_mem is not None else True # to reduce memory usage, impacting disk space
use_fmap = bool(args.use_fmap) if args.use_fmap is not None else True # make it boolean
#
# freesurfer commands
fs_cmd = args.fs_cmd if args.fs_cmd is not None else 'all' 
#
# jason folder location 
json_folder = args.json_folder if args.json_folder is not None else 'fmap' 

## Load data object
print("Preprocessing {data} data for subject {sj}!".format(data=data_type, sj=sj))
FAM_data = load_exp_settings.MRIData(params, sj, 
                                    repo_pth = op.split(load_exp_settings.__file__)[0], 
                                    base_dir=system_dir, 
                                    exclude_sj = exclude_sj)

print('Subject list is {l}'.format(l=str(FAM_data.sj_num)))

## Load preprocessing class for each data type
FAM_mri_preprocess = preproc_mridata.PreprocMRI(FAM_data)

## run specific steps ##
match step:
    case 'anat_preproc':

        print('Running {step}\ Check if we ran preprocessing for anatomical data'.format(step=step))
        
        FAM_mri_preprocess.check_anatpreproc(T2file = T2_file)

    case 'freesurfer':

        print('Running {step}\ Running Freesurfer 7.2 on T1w and T2w (if available)'.format(step=step))

        FAM_mri_preprocess.call_freesurfer(cmd = fs_cmd, node_name = node_name, partition_name = partition_name,
                                            batch_mem_Gib = batch_mem_Gib)

    case 'fmriprep':

        print('Running {step}\ Running fmriprep singularity'.format(step=step))

        FAM_mri_preprocess.call_fmriprep(data_type = data_type, node_name = node_name, partition_name = partition_name, use_fmap=use_fmap, 
                                    node_mem = node_mem, batch_mem_Gib = batch_mem_Gib, low_mem = low_mem)

    case 'nordic':

        print('Running {step}\ Run NORDIC on functional files'.format(step=step))

        FAM_mri_preprocess.NORDIC(participant=sj)

    case 'func_preproc':

        print('Running {step}\ Check if we ran preprocessing for functional data'.format(step=step))

        FAM_mri_preprocess.check_funcpreproc()

    case 'mriqc':

        print('Running {step}\ Run MRIQC on data'.format(step=step))

        if FAM_data.base_dir == 'local':
            FAM_mri_preprocess.call_mriqc(batch_dir = op.join(FAM_data.proj_root_pth, 'batch'))
        else:
            FAM_mri_preprocess.call_mriqc()

    case 'up_json':

        print('Running {step}\ Update json params given raw PAR/REC header info'.format(step=step))

        FAM_mri_preprocess.update_jsons(participant = sj, json_folder = json_folder)

    case 'post_fmriprep':

        print('Running {step}\ Run final processing steps on functional data (after fmriprep)'.format(step=step))

        FAM_mri_preprocess.post_fmriprep_proc(save_subcortical = False)
