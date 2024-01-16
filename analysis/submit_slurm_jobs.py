import os, sys
import os.path as op
import numpy as np
import argparse
import time

import yaml
from FAM.processing import load_exp_settings

# load settings from yaml
with open('exp_params.yml', 'r') as f_in:
    params = yaml.safe_load(f_in)

## to get inputs 
parser = argparse.ArgumentParser()

# general
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
parser.add_argument("--cmd", 
                    type = str.lower, 
                    required = True,
                    help = "Step of pipeline we want to run: post_fmriprep, fitmodel, fitdecoder, etc..."
                    )
parser.add_argument("--task", 
                    type = str, 
                    default = 'pRF',
                    #required = True,
                    help = "On which task to fit model (pRF [default] vs FA)"
                    )
parser.add_argument("--exclude_sj", 
                    nargs = '*', # 0 or more values expected => creates a list
                    default = [],
                    type = int,
                    help = "List of subs to exclude (ex: 1 2 3 4). Default []"
                    )

# system specific 
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
                    help = "Node memory limit, ex: 90 [default None]"
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
parser.add_argument("--n_cpus", 
                    type = int, 
                    default = 32,
                    help = "Number of CPUs per node [default 32]"
                    )
parser.add_argument("--n_nodes", 
                    type = int, 
                    default = 1,
                    help = "Number of nodes [default 1]"
                    )
parser.add_argument("--n_batches", 
                    type = int, 
                    default = 10,
                    help = "Number of batches to split data into when fitting [default 10]"
                    )
parser.add_argument("--n_cpus_task", 
                    type = int, 
                    default = 4,
                    help = "Number of CPUS to use per process, which should allow threading [default 4]"
                    )
parser.add_argument("--n_jobs", 
                    type = int, 
                    help = "If given, sets number of jobs for parallel"
                    )
parser.add_argument("--n_tasks", 
                    type = int, 
                    default = 16,
                    help = "If given, sets number of processes"
                    )
parser.add_argument("--concurrent_job", 
                    action = 'store_true',
                    help="if option called, run analysis concurrently for all participants/all chunks of data of 1 participant (ex: prf fitting)"
                    )
parser.add_argument("--use_rsync", 
                    action = 'store_true',
                    help="if option called, use rsync to copy to node (instead of cp)"
                    )
parser.add_argument("--dry_run", 
                    action = 'store_true',
                    help="if option called, make jib without exactly running it"
                    )

# analysis specific
parser.add_argument("--chunk_data", 
                    action = 'store_true',
                    help = "if option called, divide the data into chunks"
                    ) # if we want to divide in batches (chunks)
parser.add_argument("--n_chunks", 
                    type = int, 
                    default = 100,
                    help = "Number of chunks to split jobs into into when fitting [default 100]"
                    )
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
# general
sj = args.subject[0] if len(args.subject) == 1 else args.subject # for situation where 1 sj vs list
system_dir = args.dir #'local' #
pycmd = args.cmd
task = args.task
exclude_sj = args.exclude_sj # list of excluded subjects

# system specific 
node_name = args.node_name # node name to submit slurm job 
partition_name = args.partition_name # partition name to submit slurm job
batch_mem_Gib = args.batch_mem_Gib
send_email = args.email
run_time = '{h}:00:00'.format(h = str(args.hours)) 
n_cpus = args.n_cpus
n_nodes = args.n_nodes
n_batches = args.n_batches
n_cpus_task = args.n_cpus_task
n_jobs = args.n_jobs
n_tasks = args.n_tasks
concurrent_job = args.concurrent_job
use_rsync = args.use_rsync
dry_run = args.dry_run

# analysis specific
chunk_data = args.chunk_data
n_chunks = args.n_chunks
prf_model_name = args.prf_model_name
fit_hrf = args.fit_hrf
run_type = args.run_type
ses2fit = args.ses2fit # 'ses-mean'
fa_model_name = args.fa_model_name

## Load data object --> as relevant paths, variables and utility functions
print("Running data analysis for subject {sj}!".format(sj=sj))

FAM_data = load_exp_settings.MRIData(params, sj, 
                                    repo_pth = op.split(load_exp_settings.__file__)[0], 
                                    base_dir = system_dir, exclude_sj = exclude_sj)

print('Subject list is {l}'.format(l=str(FAM_data.sj_num)))

def main(concurrent_job = False, dry_run = False, **kwargs):
    
    """Main caller for job submission
    """
    if concurrent_job:
        make_concurrent_job(participant_list = FAM_data.sj_num, step_type = pycmd, task = task, run_time = run_time, 
                    partition_name = partition_name, node_name = node_name, batch_mem_Gib = batch_mem_Gib, 
                    n_tasks = n_tasks, n_nodes = n_nodes, n_cpus_task = n_cpus_task,
                    send_email = send_email, use_rsync = use_rsync, dry_run = dry_run)
    else:
        submit_jobs(participant_list = FAM_data.sj_num, step_type = pycmd, task = task, run_time = run_time, 
                partition_name = partition_name, node_name = node_name, batch_mem_Gib = batch_mem_Gib, 
                n_tasks = n_tasks, n_nodes = n_nodes, n_cpus_task = n_cpus_task,
                send_email = send_email)


def submit_jobs(participant_list = [], step_type = 'fitmodel', run_time = '10:00:00', partition_name = None, node_name = None, 
                batch_mem_Gib = None, task = 'pRF', n_tasks = 16, n_nodes = 1, n_cpus_task = 4,
                batch_dir = '/home/inesv/batch', send_email = False):
    
    """
    Submit jobs for specific analysis step
    """
    
    if step_type == 'post_fmriprep':
        
        # get base format for bash script
        bash_basetxt = make_SLURM_script(step_type = step_type, run_time = run_time, 
                                        logfilename = 'slurm_FAM_{st}_proc'.format(st = step_type), 
                                        partition_name = partition_name, node_name = node_name, batch_mem_Gib = batch_mem_Gib, 
                                        task = task, batch_dir = batch_dir, send_email = send_email, 
                                        n_tasks = n_tasks, n_nodes=n_nodes, n_cpus_task = n_cpus_task)
        
        # set general analysis command 
        fit_cmd = """python process_data.py --subject $SJ_NR --step post_fmriprep --dir slurm """
        fit_cmd += """\n"""
        
        # bash file name
        js_name = op.join(batch_dir, 'post_fmriprep_sub-$SJ_NR_FAM.sh')

    # loop over participants
    for pp in participant_list:
        
        ## replace command, subject number and scratch dir in base string
        working_string = bash_basetxt.replace('$PY_CMD', fit_cmd)
        working_string = working_string.replace('$SJ_NR', pp)
        working_string = working_string.replace('$TMPDIR', '/scratch-shared/$USER/FAM') 

        print(working_string)
        
        of = open(js_name.replace('$SJ_NR', pp), 'w')
        of.write(working_string)
        of.close()

        print('submitting ' + js_name.replace('$SJ_NR', pp) + ' to queue')
        os.system('sbatch ' + js_name.replace('$SJ_NR', pp))

        # wait a bit, to give stuff time to start running
        time.sleep(.2)

def call_fitmodel_jobs(participant_list = [], chunk_data = True, run_time = '10:00:00', task = 'pRF',
                model_name = 'gauss', partition_name = None, node_name = None, batch_mem_Gib = None, 
                batch_dir ='/home/inesv/batch', send_email = False, n_cpus = 128, n_nodes = 1, n_batches = 16, n_cpus_task = 4,
                n_tasks = None, n_jobs = None, n_chunks = None):

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
    if n_chunks is None:
        # total number of chunks
        n_chunks = FAM_data.params['mri']['fitting'][task]['total_chunks'][FAM_data.sj_space]

    if chunk_data:
        ch_list = np.arange(n_chunks)
    else:
        ch_list = [None]

    ## limit number of cpus in usage = nodes x cpus -2 (to avoid memory issues)
    #new_n_cpus = int((n_cpus * n_nodes - 2))
    #print('allocating %i CPUS'%new_n_cpus)
    
    ## allocate node resources efficiently
    # number of cpus that can be used per task = threads within a process
    # so then we can obtain max possible number of processes, for the number of cpus we allocate
    if n_tasks is None:
        n_tasks = int(n_cpus/n_cpus_task) # (processes that will run in paralell)
    else:
        n_cpus_task = int((n_cpus * n_nodes)/n_tasks) # if we specify number of tasks, divide cpus accordingly  

    ## number of jobs will be number of tasks -1 (to avoid memory issues)
    if n_jobs is None:
        n_jobs = int(n_tasks - 1)            

    # get base format for bash script
    bash_basetxt = make_SLURM_script(run_time = run_time, logfilename = 'slurm_FAM_{tsk}_{md}_fit'.format(md = model_name, tsk = task), 
                                            partition_name = partition_name, node_name = node_name, batch_mem_Gib = batch_mem_Gib, 
                                            task = task, batch_dir = batch_dir, send_email = send_email, 
                                            n_tasks = n_tasks, n_nodes=n_nodes, n_cpus_task = n_cpus_task)
        
    # loop over participants
    for pp in participant_list:

        for ch in ch_list:
            
            # set fitting model command 
            fit_cmd = """python run_analysis.py --subject {pp} --cmd fitmodel --task {task} --dir {dir} --ses2fit {ses} --run_type {rt} \
            --prf_model_name {prf_mod} --fa_model_name {fa_mod} --n_jobs {n_jobs} --n_batches {n_batches} --wf_dir $TMPDIR """.format(pp = pp, task = task, dir = system_dir,
                                                    ses = ses2fit, rt = run_type, prf_mod = prf_model_name, fa_mod = fa_model_name, n_jobs = n_jobs, n_batches = n_batches)
            # if chunking data
            if ch is not None:
                fit_cmd += '--chunk_num {ch} --total_chunks {tch} '.format(ch = ch, tch = n_chunks)

            # if we want to fit hrf
            if fit_hrf:
                fit_cmd += '--fit_hrf'

            fit_cmd += """\n\n"""

            ## replace command, subject number and scratch dir in base string
            working_string = bash_basetxt.replace('$SJ_NR', pp)
            working_string = working_string.replace('$PY_CMD', fit_cmd)
            working_string = working_string.replace('$TMPDIR', '/scratch-shared/$USER/FAM') 

            print(working_string)

            # run it
            js_name = op.join(batch_dir, '{fname}_model-{m}_sub-{sj}_chunk-{ch}_run-{r}_FAM.sh'.format(fname=FAM_data.params['mri']['fitting'][task]['fit_folder'],
                                                                                    ch = str(ch).zfill(3), sj = pp, r = run_type, m = prf_model_name))
            of = open(js_name, 'w')
            of.write(working_string)
            of.close()

            print('submitting ' + js_name + ' to queue')
            os.system('sbatch ' + js_name)

            # wait a bit, to give stuff time to start running
            time.sleep(.2)

def make_SLURM_script(step_type = 'fitmodel', run_time = '10:00:00', logfilename = '', partition_name = None, node_name = None, 
                    batch_mem_Gib = None, task = 'pRF', n_tasks = 16, n_nodes = 1, n_cpus_task = 4,
                    batch_dir = '/home/inesv/batch', send_email = False, group_bool = False, use_rsync = False):

    """
    Set up bash script, with generic structure 
    to be used for submitting in SLURM systems

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

    ## initialize batch script, with general node specs
    slurm_cmd = """#!/bin/bash\n#SBATCH -t {rtime}\n#SBATCH -N {n_nodes}\n"""+ \
    """#SBATCH -v\n#SBATCH --ntasks-per-node={ntasks}\n"""+ \
    """#SBATCH --cpus-per-task={n_cpus_task}\n"""+ \
    """#SBATCH --output=$BD/{logfilename}_%A.out\n\n"""
    slurm_cmd = slurm_cmd.format(rtime = run_time, logfilename = logfilename, 
                                n_nodes = n_nodes, n_cpus_task = n_cpus_task, ntasks = n_tasks)
    
    ## if we want a specific node/partition
    if partition_name is not None:
        slurm_cmd += '#SBATCH --partition={p}\n'.format(p=partition_name)
    if node_name is not None:
        slurm_cmd += '#SBATCH -w {n}\n'.format(n=node_name)

    ## add memory for node
    if batch_mem_Gib is not None:
        slurm_cmd += '#SBATCH --mem={mem}G\n'.format(mem=batch_mem_Gib)
        
    # rsync general folders that should be needed for 
    if use_rsync:
        slurm_cmd += rsync_deriv(group_bool = group_bool)
    else:
        slurm_cmd += cp_deriv(group_bool = group_bool) # or copy if more efficient
        
    # join command specific lines
    if step_type == 'fitmodel':
        
        if task == 'pRF':
            fit_cmd = """mkdir -p $TMPDIR/derivatives/$FITFOLDER/$SPACE/sub-$SJ_NR\n\nwait\n\n"""+ \
            """if [ -d "$DERIV_DIR/$FITFOLDER/$SPACE/sub-$SJ_NR" ]\nthen\n"""+ \
            """    rsync -chavP --exclude=".*" $DERIV_DIR/$FITFOLDER/$SPACE/sub-$SJ_NR/ $TMPDIR/derivatives/$FITFOLDER/$SPACE/sub-$SJ_NR --no-compress\nfi\n\nwait\n\n"""
            
        elif task == 'FA':
            # if we are fitting FA, then also need to copy pRF estimates to scratch
            fit_cmd = """mkdir -p $TMPDIR/derivatives/{$FITFOLDER,$PRFFITFOLDER}/$SPACE/sub-$SJ_NR\n\nwait\n\n"""+ \
            """if [ -d "$DERIV_DIR/$PRFFITFOLDER/$SPACE/sub-$SJ_NR" ]\nthen\n"""+ \
            """    rsync -chavP --exclude=".*" $DERIV_DIR/$PRFFITFOLDER/$SPACE/sub-$SJ_NR/ $TMPDIR/derivatives/$PRFFITFOLDER/$SPACE/sub-$SJ_NR --no-compress\nfi\n\n"""+ \
            """if [ -d "$DERIV_DIR/$FITFOLDER/$SPACE/sub-$SJ_NR" ]\nthen\n"""+ \
            """    rsync -chavP --exclude=".*" $DERIV_DIR/$FITFOLDER/$SPACE/sub-$SJ_NR/ $TMPDIR/derivatives/$FITFOLDER/$SPACE/sub-$SJ_NR --no-compress\nfi\n\n"""+ \
            """wait\n\n"""
            fit_cmd = fit_cmd.replace('$PRFFITFOLDER', FAM_data.params['mri']['fitting']['pRF']['fit_folder'])
        
        slurm_cmd += fit_cmd
        
    ## call final part (with actual command)
    bash_string = slurm_cmd + \
        """$PY_CMD\n\nwait # wait until programs are finished\n\n"""+ \
            """rsync -chavP --exclude=".*" $TMPDIR/derivatives/ $DERIV_DIR --no-compress\n\nwait\n\n$END_EMAIL\n"""
    
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
              
def rsync_deriv(group_bool = False):
    
    """General script for rsyncing postfmriprep derivatives
    """
    
    # if we want to run it for the group, then copy dir for all participants
    if group_bool:
        cmd = """# call the programs\n$START_EMAIL\n\n"""+\
        """# make derivatives dir in node and sourcedata because we want to access behav files\n"""+ \
        """mkdir -p $TMPDIR/derivatives/post_fmriprep/$SPACE\n"""+ \
        """mkdir -p $TMPDIR/sourcedata\n\nwait\n\n"""+\
        """rsync -chavP --exclude=".*" $DERIV_DIR/post_fmriprep/$SPACE/ $TMPDIR/derivatives/post_fmriprep/$SPACE --no-compress\n\nwait\n\n"""+\
        """rsync -chavP --exclude=".*" $SOURCE_DIR/ $TMPDIR/sourcedata --no-compress\n\nwait\n\n"""
    else:
        cmd = """# call the programs\n$START_EMAIL\n\n"""+\
        """# make derivatives dir in node and sourcedata because we want to access behav files\n"""+ \
        """mkdir -p $TMPDIR/derivatives/post_fmriprep/$SPACE/sub-$SJ_NR\n"""+ \
        """mkdir -p $TMPDIR/sourcedata/sub-$SJ_NR\n\nwait\n\n"""+\
        """rsync -chavP --exclude=".*" $DERIV_DIR/post_fmriprep/$SPACE/sub-$SJ_NR/ $TMPDIR/derivatives/post_fmriprep/$SPACE/sub-$SJ_NR --no-compress\n\nwait\n\n"""+\
        """rsync -chavP --exclude=".*" $SOURCE_DIR/sub-$SJ_NR/ $TMPDIR/sourcedata/sub-$SJ_NR --no-compress\n\nwait\n\n"""
        
    return cmd

def cp_deriv(group_bool = False):
    
    """General script for cp postfmriprep derivatives
    """
    
    # if we want to run it for the group, then copy dir for all participants
    if group_bool:
        cmd = """# call the programs\n$START_EMAIL\n\n"""+\
        """# make derivatives dir in node and sourcedata because we want to access behav files\n"""+ \
        """mkdir -p $TMPDIR/derivatives/post_fmriprep/$SPACE\n"""+ \
        """mkdir -p $TMPDIR/sourcedata\n\nwait\n\n"""+\
        """cp -r $DERIV_DIR/post_fmriprep/$SPACE/ $TMPDIR/derivatives/post_fmriprep/$SPACE\n\nwait\n\n"""+\
        """cp -r $SOURCE_DIR/ $TMPDIR/sourcedata\n\nwait\n\n"""
    else:
        cmd = """# call the programs\n$START_EMAIL\n\n"""+\
        """# make derivatives dir in node and sourcedata because we want to access behav files\n"""+ \
        """mkdir -p $TMPDIR/derivatives/post_fmriprep/$SPACE/sub-$SJ_NR\n"""+ \
        """mkdir -p $TMPDIR/sourcedata/sub-$SJ_NR\n\nwait\n\n"""+\
        """cp -r $DERIV_DIR/post_fmriprep/$SPACE/sub-$SJ_NR/ $TMPDIR/derivatives/post_fmriprep/$SPACE/sub-$SJ_NR\n\nwait\n\n"""+\
        """cp -r $SOURCE_DIR/sub-$SJ_NR/ $TMPDIR/sourcedata/sub-$SJ_NR\n\nwait\n\n"""
        
    return cmd

def make_concurrent_job(participant_list = [], step_type = 'fitmodel', run_time = '10:00:00', 
                        partition_name = None, node_name = None, 
                        batch_mem_Gib = None, task = 'pRF', n_tasks = 16, n_nodes = 1, n_cpus_task = 4,
                        batch_dir = '/home/inesv/batch', send_email = False, use_rsync = False, dry_run = False):
    
    """execute the same pipeline (or single program) on different samples of data.
    """
    
    #n_tasks = len(participant_list)
    
    if step_type == 'post_fmriprep':
        
        # get base format for bash script
        bash_basetxt = make_SLURM_script(step_type = step_type, run_time = run_time, 
                                        logfilename = 'slurm_FAM_{st}_proc'.format(st = step_type), 
                                        partition_name = partition_name, node_name = node_name, batch_mem_Gib = batch_mem_Gib, 
                                        task = task, batch_dir = batch_dir, send_email = send_email, 
                                        n_tasks = n_tasks, n_nodes=n_nodes, n_cpus_task = n_cpus_task,
                                        group_bool = True, use_rsync = use_rsync)
        
        # set general analysis command iterating over participant list
        fit_cmd = """declare -a pp_arr=($PP_LIST)\n"""+ \
            """for ((i = 0; i < ${#pp_arr[@]}; i++)); do\n"""+ \
            """(\n  python process_data.py --subject ${pp_arr[$i]} --step post_fmriprep --dir slurm\n) &\n"""+ \
            """done\n\nwait\n\n"""

    # bash file name
    js_name = op.join(batch_dir, '{skey}_sub-GROUP_FAM.sh'.format(skey = step_type))
        
    ## replace command and scratch dir in base string
    working_string = bash_basetxt.replace('$PY_CMD', fit_cmd)
    working_string = working_string.replace('$TMPDIR', '/scratch-shared/$USER/FAM') 
    
    # replace list of participants
    working_string = working_string.replace('$PP_LIST', ' '.join(participant_list))

    print(working_string)
    
    of = open(js_name, 'w')
    of.write(working_string)
    of.close()

    # if just want to see how it looks, then dont actually run script
    if dry_run:
        print('submitting ' + js_name + ' to queue')
        os.system('sbatch ' + js_name)
  
  
## actually submit jobs
main(concurrent_job = concurrent_job, dry_run = dry_run)      
        
        