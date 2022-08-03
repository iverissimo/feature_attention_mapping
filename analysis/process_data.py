import os, sys
import os.path as op
import argparse

import yaml
from FAM.processing import load_exp_data
from FAM.visualize.preproc_viewer import MRIViewer

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
# to view/plot FS segmentation:
parser.add_argument("--freeview", type = str, help="Check Freesurfer segmentations (view [default] vs movie)")

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
else:
    exclude_sj = []

fs_cmd = args.fs_cmd if args.fs_cmd is not None else 'all' # freesurfer command to run

freeview_cmd = args.freeview if args.freeview is not None else 'view' # freeview command to run

## Load data object
print("Preprocessing {data} data for subject {sj}!".format(data=data_type, sj=sj))
preproc_data = load_exp_data.MRIData(params, sj, repo_pth = op.split(load_exp_data.__file__)[0], base_dir=system_dir)

## run specific steps
print('Running {step}'.format(step=step))

if step == 'anat_preproc':
    
    preproc_data.check_anatpreproc(T2file = T2_file)

elif step == 'freesurfer':

    preproc_data.call_freesurfer(cmd = fs_cmd)

elif step == 'fmriprep':
    
    preproc_data.call_fmriprep(data_type = data_type, node_name = node_name, partition_name = partition_name)

elif step == 'nordic':

    preproc_data.NORDIC(participant=sj)

elif step == 'func_preproc':

    preproc_data.check_funcpreproc()

elif step == 'mriqc':

    if preproc_data.base_dir == 'local':
        preproc_data.call_mriqc(batch_dir = op.join(preproc_data.proj_root_pth, 'batch'))
    else:
        preproc_data.call_mriqc()

elif step == 'check_fs':

    plotter = MRIViewer(preproc_data)
    plotter.check_fs_seg(check_type = freeview_cmd, use_T2 = T2_file)