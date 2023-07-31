import os, sys
import os.path as op
import numpy as np
import argparse

import yaml
from FAM.processing import load_exp_settings, preproc_mridata, preproc_behdata
from FAM.fitting.prf_model import pRF_model

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
parser.add_argument("--task", 
                    type = str, 
                    default = 'pRF',
                    required = True,
                    help = "On which task to run analysis (pRF [default] vs FA)"
                    )
parser.add_argument("--cmd", 
                    type = str, 
                    default = 'fitmodel',
                    required = True,
                    help = "What analysis to run (ex: fitmodel)"
                    )
parser.add_argument("--dir", 
                    type = str.lower, 
                    default = 'local',
                    help = "System we are running analysis in - local [default] vs slurm (snellius)"
                    )
parser.add_argument("--wf_dir", 
                    type = str, 
                    help="Path to workflow dir, if such if not standard root dirs (None [default] vs /scratch)"
                    )
parser.add_argument("--exclude_sj", 
                    nargs = '*', # 0 or more values expected => creates a list
                    default = [],
                    type = int,
                    help = "List of subs to exclude (ex: 1 2 3 4). Default []"
                    )
parser.add_argument("--chunk_num", 
                    type = int,
                    help = "Chunk number to fit or None [default]"
                    ) # if we want to divide in batches (chunks)
parser.add_argument("--prf_model_name", 
                    type = str, 
                    default = 'gauss',
                    help="Type of pRF model to fit: gauss [default], css, dn, etc..."
                    )
parser.add_argument("--fit_hrf", 
                    action = 'store_true',
                    help="if option called, fit hrf on the data"
                    )
parser.add_argument("--run_type", 
                    default = 'mean',
                    help="Type of run to fit (mean of runs [default], 1, loo_r1s1, ...)"
                    )
parser.add_argument("--ses2fit", 
                    default = 'mean',
                    help="Session to fit (if mean [default] then will average both session when that's possible)"
                    )
parser.add_argument("--fa_model_name", 
                    type = str, 
                    default = 'glmsingle',
                    help="Type of FA model to fit: glmsingle [default], gain, glm, etc...]"
                    )
parser.add_argument("--vertex",
                    nargs = '*', 
                    default = [],
                    type = int,
                    help="list of vertex indice(s) to fit or default []"
                    )
parser.add_argument("--ROI", 
                    type = str,
                    help="ROI name to fit or None [default]")

# parse the command line
args = parser.parse_args()

# access parser options
sj = args.subject[0] if len(args.subject) == 1 else args.subject # for situation where 1 sj vs list
exclude_sj = args.exclude_sj # list of excluded subjects
py_cmd = args.cmd # what step of pipeline we want to run
system_dir = args.dir
wf_dir = args.wf_dir
task = args.task
chunk_num = args.chunk_num
prf_model_name = args.prf_model_name
fit_hrf = args.fit_hrf
run_type = args.run_type
ses2fit = args.ses2fit # 'ses-mean'
fa_model_name = args.fa_model_name

# vertex list
if len(args.vertex)>0:
    vertex = [int(val) for val in args.vertex]
# ROI name
ROI = args.ROI

## Load data object --> as relevant paths, variables and utility functions
print("Fitting data for subject {sj}!".format(sj=sj))

FAM_data = load_exp_settings.MRIData(params, sj, 
                                    repo_pth = op.split(load_exp_settings.__file__)[0], 
                                    base_dir = system_dir, exclude_sj = exclude_sj)

print('Subject list is {l}'.format(l=str(FAM_data.sj_num)))

## Load preprocessing class for each data type ###

# get behavioral info 
FAM_beh = preproc_behdata.PreprocBeh(FAM_data)
# and mri info
FAM_mri = preproc_mridata.PreprocMRI(FAM_data)


## load pRF model class
FAM_pRF = pRF_model(FAM_data)

# set specific params
FAM_pRF.model_type['pRF'] = prf_model_name
FAM_pRF.fit_hrf = fit_hrf


## run specific steps ##
match task:

    case 'pRF':
        
        if py_cmd == 'fitmodel': # fit pRF model

            # get participant models, which also will load 
            # DM and mask it according to participants behavior
            pp_prf_models = FAM_pRF.set_models(participant_list = FAM_data.sj_num, 
                                               mask_DM = True, 
                                               ses2model = ses2fit)
            





                   






## where to define total chunks
# if not calling model here, then where?
# also want to be able to run fit locally, prf and glmsingle
# which should be thorugh here






## OLD ##

match system_dir:

    case 'lisa':

        ## set start of slurm command

        slurm_cmd = """#!/bin/bash
#SBATCH -t {rtime}
#SBATCH -N 1
#SBATCH -v
#SBATCH --cpus-per-task=16
#SBATCH --output=$BD/slurm_{task}_{model}_fit_%A.out\n""".format(rtime=run_time, task = task, model = model2fit)
                    
        if partition_name is not None:
            slurm_cmd += '#SBATCH --partition {p}\n'.format(p=partition_name)
        if node_name is not None:
            slurm_cmd += '#SBATCH -w {n}\n'.format(n=node_name)

        # add memory for node
        slurm_cmd += '#SBATCH --mem={mem}G\n'.format(mem=batch_mem_Gib)

        # set fit folder name
        if task == 'pRF':
            fitfolder = params['mri']['fitting']['pRF']['fit_folder'] 
        elif task == 'FA':
            fitfolder = params['mri']['fitting']['FA']['fit_folder'][model2fit]

        # batch dir to save .sh files
        batch_dir = '/home/inesv/batch'

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
--prf_model_name {prf_mod} --fa_model_name {fa_mod} --fit_hrf {fh} --wf_dir $TMPDIR\n\n""".format(pth = op.split(prf_model.__file__)[0],
                                                        pp = pp,
                                                        task = task,
                                                        dir = system_dir,
                                                        ses = ses2fit,
                                                        rt = run_type,
                                                        ch = ch,
                                                        prf_mod = prf_model_name,
                                                        fa_mod = fa_model_name,
                                                        fh = int(fit_hrf))

                if task == 'pRF':
                    slurm_cmd = slurm_cmd + """# call the programs
$START_EMAIL

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

"""
                else:
                    # if we are fitting FA, then also need to copy pRF estimates to scratch

                    slurm_cmd = slurm_cmd + """# call the programs
$START_EMAIL

# make derivatives dir in node and sourcedata because we want to access behav files
mkdir -p $TMPDIR/derivatives/{post_fmriprep,$FITFOLDER,$PRFFITFOLDER}/$SPACE/sub-$SJ_NR
mkdir -p $TMPDIR/sourcedata/sub-$SJ_NR

wait

cp -r $DERIV_DIR/post_fmriprep/$SPACE/sub-$SJ_NR $TMPDIR/derivatives/post_fmriprep/$SPACE

wait

cp -r $SOURCE_DIR/sub-$SJ_NR $TMPDIR/sourcedata/

wait

if [ -d "$DERIV_DIR/$PRFFITFOLDER/$SPACE/sub-$SJ_NR" ] 
then
    cp -r $DERIV_DIR/$PRFFITFOLDER/$SPACE/sub-$SJ_NR $TMPDIR/derivatives/$PRFFITFOLDER/$SPACE
fi

if [ -d "$DERIV_DIR/$FITFOLDER/$SPACE/sub-$SJ_NR" ] 
then
    cp -r $DERIV_DIR/$FITFOLDER/$SPACE/sub-$SJ_NR $TMPDIR/derivatives/$FITFOLDER/$SPACE
fi

wait

""".replace('$PRFFITFOLDER', params['mri']['fitting']['pRF']['fit_folder'])

                ### update slurm job script

                batch_string =  slurm_cmd + """$PY_CMD

wait          # wait until programs are finished

rsync -chavzP $TMPDIR/derivatives/ $DERIV_DIR

wait          # wait until programs are finished

$END_EMAIL
"""

                ### if we want to send email
                if send_email == True:
                    batch_string = batch_string.replace('$START_EMAIL', 'echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"')
                    batch_string = batch_string.replace('$END_EMAIL', 'echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"')
                
                ## replace other variables

                working_string = batch_string.replace('$SJ_NR', str(pp).zfill(3))
                working_string = working_string.replace('$SPACE', FAM_data.sj_space)
                working_string = working_string.replace('$FITFOLDER', fitfolder)
                working_string = working_string.replace('$PY_CMD', fit_cmd)
                working_string = working_string.replace('$BD', batch_dir)
                working_string = working_string.replace('$DERIV_DIR', FAM_data.derivatives_pth)
                working_string = working_string.replace('$SOURCE_DIR', FAM_data.sourcedata_pth)

                print(working_string)

                # run it
                js_name = op.join(batch_dir, '{fname}_sub-{sj}_chunk-{ch}_run-{r}_FAM.sh'.format(fname=fitfolder,
                                                                                        ch=ch,
                                                                                        sj=pp,
                                                                                        r=run_type))
                of = open(js_name, 'w')
                of.write(working_string)
                of.close()

                print('submitting ' + js_name + ' to queue')
                os.system('sbatch ' + js_name)




