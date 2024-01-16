import os, sys
import os.path as op
import numpy as np
import argparse
import time

import yaml
from FAM.processing import load_exp_settings
from FAM.batch import Batcher

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
                    default = 8,
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
                    help="if option called, make job without exactly running it"
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
n_hours = args.hours 
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

## Load BATCHER object 
FAM_SLURM = Batcher(FAM_data)

# set up some general parameters for job(s) nodes
FAM_SLURM.setup_slurm_pars(n_hours = n_hours, n_tasks = n_tasks, n_nodes = n_nodes, n_cpus_task = n_cpus_task)

## actually submit jobs
FAM_SLURM.submit_jobs(participant_list = FAM_data.sj_num, 
                step_type = pycmd, taskname = task, concurrent_job = concurrent_job,
                n_jobs = n_jobs, n_batches = n_batches, chunk_data = None, fit_hrf = fit_hrf, use_rsync = use_rsync, 
                prf_model_name = prf_model_name, fa_model_name = fa_model_name, run_type = run_type, ses2fit = ses2fit,
                dry_run = dry_run)

