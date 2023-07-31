import os, sys
import os.path as op
import numpy as np
import argparse

import yaml
from FAM.processing import load_exp_settings

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
parser.add_argument("--dir", 
                    type = str.lower, 
                    default = 'slurm',
                    help = "System we are running analysis in - local vs slurm [default] (snellius)"
                    )
parser.add_argument("--task", 
                    type = str, 
                    default = 'pRF',
                    required = True,
                    help = "On which task to fit model (pRF [default] vs FA)"
                    )
parser.add_argument("--exclude_sj", 
                    nargs = '*', # 0 or more values expected => creates a list
                    default = [],
                    type = int,
                    help = "List of subs to exclude (ex: 1 2 3 4). Default []"
                    )
parser.add_argument("--node_name", 
                    type = str, 
                    help = "Node name, to send job to [default None]"
                    )
parser.add_argument("--partition_name", 
                    type = str, 
                    help = "Partition name, to send job to [default None]"
                    )
parser.add_argument("--batch_mem_Gib", 
                    type = int, 
                    default = 90,
                    help = "Node memory limit [default 90]"
                    )
parser.add_argument("--email", 
                    action = 'store_true',
                    help = "if option called, send job email"
                    )
parser.add_argument("--hours", 
                    type = int, 
                    default = 10,
                    help="Number of hours to set as time limit for job [default 10h]"
                    )
parser.add_argument("--chunk_data", 
                    action = 'store_true',
                    help = "if option called, divide the data into chunks"
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
                    help="Type of run to fit (mean of runs [default], 1, loo_run, ...)"
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

# parse the command line
args = parser.parse_args()

# access parser options
sj = args.subject[0] if len(args.subject) == 1 else args.subject # for situation where 1 sj vs list
exclude_sj = args.exclude_sj # list of excluded subjects
system_dir = args.dir #'local' #
task = args.task
node_name = args.node_name # node name to submit slurm job 
partition_name = args.partition_name # partition name to submit slurm job
batch_mem_Gib = args.batch_mem_Gib
run_time = '{h}:00:00'.format(h = str(args.hours)) 
send_email = args.email
chunk_data = args.chunk_data
prf_model_name = args.prf_model_name
fit_hrf = args.fit_hrf
run_type = args.run_type
ses2fit = args.ses2fit # 'ses-mean'
fa_model_name = args.fa_model_name

## Load data object --> as relevant paths, variables and utility functions
print("Fitting data for subject {sj}!".format(sj=sj))

FAM_data = load_exp_settings.MRIData(params, sj, 
                                    repo_pth = op.split(load_exp_settings.__file__)[0], 
                                    base_dir = system_dir, exclude_sj = exclude_sj)

print('Subject list is {l}'.format(l=str(FAM_data.sj_num)))

## submit jobs
def main():

    model_name = prf_model_name if task == 'pRF' else fa_model_name

    submit_SLURMjobs(participant_list = FAM_data.sj_num, chunk_data = chunk_data, run_time = run_time, task = task,
                            model_name = model_name, partition_name = partition_name, node_name = node_name, batch_mem_Gib = batch_mem_Gib, 
                            batch_dir = FAM_data.batch_dir, send_email = send_email)


def submit_SLURMjobs(participant_list = [], chunk_data = True, run_time = '10:00:00', task = 'pRF',
                            model_name = 'gauss', partition_name = None, node_name = None, batch_mem_Gib = 90, 
                            batch_dir ='/home/inesv/batch', send_email = False):

        """
        Submit slurm jobs, to fit pRF model on data

        Parameters
        ----------
        participant_list: list
            list with participant ID
        chunk_data: bool
            if we want to split data into subsets
        model_name: str
            prf model name to fit
        run_time: str
            script max run time
        logfilename: str
            base string for outputted log filename
        partition_name: str
            slurm partition name
        node_name: str
            slurm node name
        batch_mem_Gib: int
            max mem to request in Gb
        batch_dir: str
            absolute path to dir where we save .sh files 
        send_email: bool
            if we want to send email when jobs starts/finishes

        """

        # if we're chunking the data, then need to submit each chunk at a time
        if chunk_data:
            # total number of chunks
            ch_list = np.arange(FAM_data.params['mri']['fitting'][task]['total_chunks'][FAM_data.sj_space])
        else:
            ch_list = [None]

        # get base format for bash script
        bash_basetxt = make_SLURM_script(run_time = run_time, logfilename = 'slurm_{tsk}_{md}_fit'.format(md = model_name, tsk = task), 
                                              partition_name = partition_name, node_name = node_name, batch_mem_Gib = batch_mem_Gib, 
                                              task = task, batch_dir = batch_dir, send_email = send_email)
           
        # loop over participants
        for pp in participant_list:

            for ch in ch_list:
                
                # set fitting model command 
                fit_cmd = """python run_analysis.py --subject {pp} --cmd fitmodel --task {task} --dir {dir} --ses2fit {ses} --run_type {rt} \
--prf_model_name {prf_mod} --fa_model_name {fa_mod} --wf_dir $TMPDIR """.format(pp = pp, task = task, dir = system_dir,
                                                        ses = ses2fit, rt = run_type, prf_mod = prf_model_name, fa_mod = fa_model_name)
                # if chunking data
                if ch is not None:
                     fit_cmd += '--chunk_num {ch} '.format(ch = ch)

                # if we want to fit hrf
                if fit_hrf:
                    fit_cmd += '--fit_hrf'

                fit_cmd += """\n\n"""

                ## replace command and subject number in base string
                working_string = bash_basetxt.replace('$SJ_NR', pp)
                working_string = working_string.replace('$PY_CMD', fit_cmd)

                print(working_string)

                # run it
                js_name = op.join(working_string, '{fname}_sub-{sj}_chunk-{ch}_run-{r}_FAM.sh'.format(fname=FAM_data.params['mri']['fitting'][task]['fit_folder'],
                                                                                        ch = ch, sj = pp, r = run_type))
                of = open(js_name, 'w')
                of.write(working_string)
                of.close()

                print('submitting ' + js_name + ' to queue')
                os.system('sbatch ' + js_name)


def make_SLURM_script(run_time = '10:00:00', logfilename = '', partition_name = None, node_name = None, batch_mem_Gib = 90, task = 'pRF', 
                          batch_dir = '/home/inesv/batch', send_email = False):

        """
        Set up bash script, with generic structure 
        to be used for fitting in SLURM systems

        Parameters
        ----------
        run_time: str
            script max run time
        logfilename: str
            base string for outputted log filename
        partition_name: str
            slurm partition name
        node_name: str
            slurm node name
        batch_mem_Gib: int
            max mem to request in Gb
        task: str
            task that we are fitting
        batch_dir: str
            absolute path to dir where we save .sh files 
        send_email: bool
            if we want to send email when jobs starts/finishes
        """

        slurm_cmd = """#!/bin/bash
#SBATCH -t {rtime}
#SBATCH -N 1
#SBATCH -v
#SBATCH --cpus-per-task=16
#SBATCH --output=$BD/{logfilename}_%A.out\n""".format(rtime = run_time, logfilename = logfilename)
        
        if partition_name is not None:
            slurm_cmd += '#SBATCH --partition {p}\n'.format(p=partition_name)
        if node_name is not None:
            slurm_cmd += '#SBATCH -w {n}\n'.format(n=node_name)

        # add memory for node
        slurm_cmd += '#SBATCH --mem={mem}G\n'.format(mem=batch_mem_Gib)

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
        elif task == 'FA':
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

""".replace('$PRFFITFOLDER', FAM_data.params['mri']['fitting']['pRF']['fit_folder'])
            
        ## add final part 
        bash_string =  slurm_cmd + """$PY_CMD

wait          # wait until programs are finished

rsync -chavzP $TMPDIR/derivatives/ $DERIV_DIR

wait          # wait until programs are finished

$END_EMAIL
"""

        ## if we want to send email
        if send_email:
            bash_string = bash_string.replace('$START_EMAIL', 'echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"')
            bash_string = bash_string.replace('$END_EMAIL', 'echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"')
        
        ## replace some variables
        bash_string = bash_string.replace('$SPACE', FAM_data.sj_space)
        bash_string = bash_string.replace('$FITFOLDER', FAM_data.params['mri']['fitting'][task]['fit_folder'])
        bash_string = bash_string.replace('$BD', batch_dir)
        bash_string = bash_string.replace('$DERIV_DIR', FAM_data.derivatives_pth)
        bash_string = bash_string.replace('$SOURCE_DIR', FAM_data.sourcedata_pth)

        return bash_string


## call it
main()




