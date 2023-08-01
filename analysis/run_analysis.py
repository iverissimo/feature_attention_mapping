import os, sys
import os.path as op
import numpy as np
import argparse

import time

import yaml
from FAM.processing import load_exp_settings, preproc_mridata, preproc_behdata
from FAM.fitting.prf_model import pRF_model
from FAM.fitting.glm_single_model import GLMsingle_Model

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
parser.add_argument("--n_jobs", 
                type = int, 
                default = 8,
                help="number of jobs for parallel"
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
parser.add_argument("--atlas", 
                    type = str, 
                    default = None,
                    help = "If we want to use atlas ROIs (ex: glasser, wang) or not [default]."
                    )

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
n_jobs = args.n_jobs
prf_model_name = args.prf_model_name
fit_hrf = args.fit_hrf
run_type = args.run_type
ses2fit = args.ses2fit 
fa_model_name = args.fa_model_name
use_atlas = args.atlas

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
FAM_pRF = pRF_model(FAM_data, use_atlas = use_atlas)

# set specific params
FAM_pRF.model_type['pRF'] = prf_model_name
FAM_pRF.fit_hrf = fit_hrf


## run specific steps ##
match task:

    case 'pRF':
        
        if py_cmd == 'fitmodel': # fit pRF model

            print('Fitting {mn} model on the data\n'.format(mn = prf_model_name))
            print('fit HRF params set to {op}'.format(op = fit_hrf))

            # get participant models, which also will load 
            # DM and mask it according to participants behavior
            pp_prf_models = FAM_pRF.set_models(participant_list = FAM_data.sj_num, 
                                               ses2model = ses2fit,
                                               mask_bool_df = FAM_beh.get_pRF_mask_bool(ses_type = 'func'), # Make DM boolean mask based on subject responses
                                               stim_on_screen = FAM_beh.get_stim_on_screen(task = task, 
                                                                                        crop_nr = 0, shift = 0, dummy = 0) # dont crop here, to do so within func
                                            )
            
            ## actually fit
            print('Fitting started!')
            # to time it
            start_time = time.time()

            for pp in FAM_data.sj_num:

                FAM_pRF.fit_data(pp, pp_prf_models, 
                                ses = ses2fit, run_type = run_type, file_ext = FAM_mri.get_mrifile_ext()['pRF'],
                                vertex = vertex, chunk_num = chunk_num, ROI = ROI,
                                model2fit = prf_model_name,
                                save_estimates = True,
                                xtol = 1e-3, ftol = 1e-4, n_jobs = n_jobs)

            print('Fitting finished, total time = {tempo}!'.format(tempo = time.time() - start_time))

    case 'FA':

        if py_cmd == 'fitmodel': # fit FA model

            print('Loading pRF {mn} model estimates\n'.format(mn = prf_model_name))
            print('fit HRF params set to {op}'.format(op = fit_hrf))

            ## load pRF estimates - implies pRF model was already fit (should change to fit pRF model on the spot if needed)
            pp_prf_estimates, pp_prf_models = FAM_pRF.load_pRF_model_estimates(participant_list = FAM_data.sj_num, 
                                                                            ses = 'mean', run_type = 'mean', 
                                                                            model_name = prf_model_name, iterative = True, 
                                                                            fit_hrf = fit_hrf,
                                                                            mask_bool_df = FAM_beh.get_pRF_mask_bool(ses_type = 'func'), # Make DM boolean mask based on subject responses
                                                                            stim_on_screen = FAM_beh.get_stim_on_screen(task = task, 
                                                                                                                        crop_nr = 0, 
                                                                                                                        shift = 0, 
                                                                                                                        dummy = 0) # dont crop here, to do so within func)
                                                                            )

            print('Fitting {mn} model on the data\n'.format(mn = fa_model_name))

            ## now fit appropriate feature model
            match fa_model_name:

                case 'glmsingle':

                    ## load FA model class
                    FAM_FA = GLMsingle_Model(FAM_data)

                    ## actually fit
                    print('Fitting started!')
                    # to time it
                    start_time = time.time()

                    for pp in FAM_data.sj_num:

                        _ = FAM_FA.fit_data(pp, pp_prf_estimates['sub-{sj}'.format(sj = pp)], 
                                            pp_prf_models['sub-{sj}'.format(sj = pp)]['ses-mean']['{mname}_model'.format(mname = prf_model_name)],  
                                            file_ext = '_cropped.npy', smooth_nm = True, perc_thresh_nm = 99, 
                                            file_extent_nm = FAM_mri.get_mrifile_ext()) 







                