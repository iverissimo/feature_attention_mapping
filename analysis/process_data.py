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
# only relevant for LISA/slurm system
parser.add_argument("--node_name", type = str, help="Node name, to send job to [default None]")
parser.add_argument("--partition_name", type = str, help="Partition name, to send job to [default None]")
# fmriprep fmap input
parser.add_argument("--use_fmap", type = int, help="Use/ignore fieldmaps if present during fmriprep processing, 1 [default] vs 0")
# update jsons
parser.add_argument("--json_folder", type = str, help="Update jason files from which folder (fmap [default] vs func)")

# set variables 
args = parser.parse_args()

sj = str(args.subject).zfill(3) # subject
step = args.step # what step of pipeline we want to run

data_type = args.data_type if args.data_type is not None else "func" # type of data 

system_dir = args.dir if args.dir is not None else "lisa" # system location

T2_file = bool(args.T2) if args.T2 is not None else False # make it boolean

node_name = args.node_name # node name to submit slurm job (or None)
partition_name = args.partition_name # partition name to submit slurm job (or None)

exclude_sj = args.exclude_sj # list of excluded subjects
if len(exclude_sj)>0:
    exclude_sj = [val.zfill(3) for val in exclude_sj]
    print('Excluding participants {expp}'.format(expp = exclude_sj))
else:
    exclude_sj = []

fs_cmd = args.fs_cmd if args.fs_cmd is not None else 'all' # freesurfer command to run

use_fmap = bool(args.use_fmap) if args.use_fmap is not None else True # make it boolean

json_folder = args.json_folder if args.json_folder is not None else 'fmap' # freeview command to run

## Load data object
print("Preprocessing {data} data for subject {sj}!".format(data=data_type, sj=sj))
FAM_data = load_exp_settings.MRIData(params, sj, repo_pth = op.split(load_exp_settings.__file__)[0], base_dir=system_dir, exclude_sj = exclude_sj)
#print(FAM_data.sj_num)

## Load preprocessing class for each data type
FAM_mri_preprocess = preproc_mridata.PreprocMRI(FAM_data)

## run specific steps
print('Running {step}'.format(step=step))

if step == 'anat_preproc':
    
    FAM_mri_preprocess.check_anatpreproc(T2file = T2_file)

elif step == 'freesurfer':

    FAM_mri_preprocess.call_freesurfer(cmd = fs_cmd)

elif step == 'fmriprep':
    
    FAM_mri_preprocess.call_fmriprep(data_type = data_type, node_name = node_name, partition_name = partition_name, use_fmap=use_fmap)

elif step == 'nordic':

    FAM_mri_preprocess.NORDIC(participant=sj)

elif step == 'func_preproc':

    FAM_mri_preprocess.check_funcpreproc()

elif step == 'mriqc':

    if FAM_data.base_dir == 'local':
        FAM_mri_preprocess.call_mriqc(batch_dir = op.join(FAM_data.proj_root_pth, 'batch'))
    else:
        FAM_mri_preprocess.call_mriqc()

elif step == 'up_json':

    FAM_mri_preprocess.update_jsons(participant = sj, json_folder = json_folder)

elif step == 'post_fmriprep':

    FAM_mri_preprocess.post_fmriprep_proc(save_subcortical = True)