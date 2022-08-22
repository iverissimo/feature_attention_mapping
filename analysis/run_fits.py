import os, sys
import os.path as op
import numpy as np
import argparse

import yaml
from FAM.processing import load_exp_settings
from FAM.fitting import prf_model

# load settings from yaml
with open('exp_params.yml', 'r') as f_in:
    params = yaml.safe_load(f_in)

## to get inputs 
parser = argparse.ArgumentParser()
parser.add_argument("--subject", help="Subject number (ex: 001, or 'group'/'all')", required=True)
parser.add_argument("--task", type = str, help="On which task to fit model (pRF/FA)", required=True)
# optional
parser.add_argument("--dir", type = str, help="System we are running analysis (lisa [default] vs local)")

# only relevant for pRF fitting
parser.add_argument("--prf_model_name", type = str, help="Type of pRF model to fit: gauss [default], css, dn, etc...")
parser.add_argument("--fit_hrf", type = int, help="1/0 - if we want to fit hrf on the data or not [default]")

# data arguments
parser.add_argument("--ses2fit", type = str, help="Session to fit (if ses-mean [default] then will average both session when that's possible)")
parser.add_argument("--run_type", help="Type of run to fit (mean of runs [default], median, 1, loo_1, ...)")

# if we want to divide in batches (chunks)
parser.add_argument("--chunk_data", type = int, help="1/0 - if we want to divide the data into chunks [default] or not")

#  only relevant if subject == group/all
parser.add_argument("--exclude_sj", nargs='+', help="List of subjects to exclude, define as --exclude_sj 0 1 ...", default=[])

# only relevant for LISA/slurm system 
parser.add_argument("--node_name", type = str, help="Node name, to send job to [default None]")
parser.add_argument("--partition_name", type = str, help="Partition name, to send job to [default None]")
parser.add_argument("--batch_mem_Gib", type = int, help="Node memory limit [default 90]")

## set variables 
args = parser.parse_args()

# subject id and processing step of pipeline
sj = str(args.subject).zfill(3) 
task = args.task 
#
#
# type of session and run to use
ses2fit = args.ses2fit if args.ses2fit is not None else 'ses-mean'
run_type = args.run_type if args.run_type is not None else 'mean'
#
chunk_data = bool(args.chunk_data) if args.chunk_data is not None else True
#
#
# if we want to exclude participants
exclude_sj = args.exclude_sj # list of excluded subjects
if len(exclude_sj)>0:
    exclude_sj = [val.zfill(3) for val in exclude_sj]
    print('Excluding participants {expp}'.format(expp = exclude_sj))
else:
    exclude_sj = []
#
#
# system location
system_dir = args.dir if args.dir is not None else "lisa" 
#
#
# prf model name and options
prf_model_name = args.prf_model_name if args.prf_model_name is not None else "gauss" 
fit_hrf = bool(args.fit_hrf) if args.fit_hrf is not None else False 

# SLURM options
# for LISA
node_name = args.node_name # node name to submit slurm job (or None)
partition_name = args.partition_name # partition name to submit slurm job (or None)
batch_mem_Gib = args.batch_mem_Gib if args.batch_mem_Gib is not None else 90
run_time = '24:00:00' # should make input too

## Load data object
print("Fitting data for subject {sj}!".format(sj=sj))
FAM_data = load_exp_settings.MRIData(params, sj, 
                                    repo_pth = op.split(load_exp_settings.__file__)[0], 
                                    base_dir = system_dir, 
                                    exclude_sj = exclude_sj)

print('Subject list is {l}'.format(l=str(FAM_data.sj_num)))

match system_dir:

    case 'lisa':

        ## set start of slurm command

        slurm_cmd = """#!/bin/bash
#SBATCH -t {rtime}
#SBATCH -N 1
#SBATCH -v
#SBATCH --cpus-per-task=16
#SBATCH --output=$BD/slurm_pRFfit_%A.out\n""".format(rtime=run_time)
                    
        if partition_name is not None:
            slurm_cmd += '#SBATCH --partition {p}\n'.format(p=partition_name)
        if node_name is not None:
            slurm_cmd += '#SBATCH -w {n}\n'.format(n=node_name)

        # add memory for node
        slurm_cmd += '#SBATCH --mem={mem}G\n'.format(mem=batch_mem_Gib)

        # set fit folder name
        if task == 'pRF':
            fitfolder = 'pRF_fit'

        # batch dir to save .sh files
        batch_dir = '/home/inesv/batch/'

        # loop over participants
        for pp in FAM_data.sj_num:

            # if we're chunking the data, then need to submit each chunk at a time
            if chunk_data:
                # total number of chunks
                total_ch = FAM_data.params['mri']['fitting'][task]['total_chunks'][FAM_data.sj_space]
                ch_list = np.arange(total_ch)
            else:
                ch_list = [None]

            for ch in ch_list:

                # set fitting model command 

                fit_cmd = """python {pth}/fit_model.py --participant {pp} --task2model {task} --dir {dir} \
--ses {ses} --run_type {rt} --chunk_num {ch} \
--prf_model_name {prf_mod} --fit_hrf {fh} --wf_dir $TMPDIR\n\n""".format(pth = op.split(prf_model.__file__)[0],
                                                        pp = pp,
                                                        task = task,
                                                        dir = system_dir,
                                                        ses = ses2fit,
                                                        rt = run_type,
                                                        ch = ch,
                                                        prf_mod = prf_model_name,
                                                        fh = int(fit_hrf))

                # update slurm job script

                batch_string =  slurm_cmd + """# call the programs
echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

# make derivatives dir in node and sourcedata because we want to access behav files
mkdir -p $TMPDIR/derivatives/{post_fmriprep,$FITFOLDER}/$SPACE/sub-$SJ_NR
mkdir -p $TMPDIR/sourcedata/sub-$SJ_NR

wait

cp -r $DERIV_DIR/post_fmriprep/$SPACE/sub-$SJ_NR $TMPDIR/derivatives/post_fmriprep/$SPACE

wait

cp -r $SOURCE_DIR/sub-$SJ_NR $TMPDIR/sourcedata/

wait

if [ -d "$DERIV_DIR/$FITFOLDER/$SPACE/sub-$SJ_NR" ] 
then
    cp -r $DERIV_DIR/$FITFOLDER/$SPACE/sub-$SJ_NR $TMPDIR/derivatives/$FITFOLDER/$SPACE
fi

wait

$PY_CMD

wait          # wait until programs are finished

rsync -chavzP $TMPDIR/derivatives/ $DERIV_DIR

wait          # wait until programs are finished

echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"
"""

                working_string = batch_string.replace('$SJ_NR', str(pp).zfill(3))
                working_string = working_string.replace('$SPACE', FAM_data.sj_space)
                working_string = working_string.replace('$FITFOLDER', fitfolder)
                working_string = working_string.replace('$PY_CMD', fit_cmd)
                working_string = working_string.replace('$BD', batch_dir)
                working_string = working_string.replace('$DERIV_DIR', FAM_data.derivatives_pth)
                working_string = working_string.replace('$SOURCE_DIR', FAM_data.sourcedata_pth)

                print(working_string)

                # run it
                js_name = op.join(batch_dir, '{fname}_sub-{sj}_chunk-{ch}_FAM.sh'.format(fname=fitfolder,
                                                                                        ch=ch,
                                                                                        sj=pp))
                of = open(js_name, 'w')
                of.write(working_string)
                of.close()

                print('submitting ' + js_name + ' to queue')
                os.system('sbatch ' + js_name)




