import os, sys
import os.path as op
from pathlib import Path
import argparse
import ast

import time

import yaml
from FAM.processing import load_exp_settings, preproc_mridata
from FAM.fitting import prf_model, feature_model

# load settings from yaml
with open('exp_params.yml', 'r') as f_in:
    params = yaml.safe_load(f_in)

## to get inputs 
parser = argparse.ArgumentParser()
parser.add_argument("--participant", help="Subject number (ex: 001)", required=True)
parser.add_argument("--task2model", type = str, help="On which task to fit model (pRF/FA)", required=True)
# optional
parser.add_argument("--dir", type = str, help="System we are running analysis (lisa [default] vs local)")
parser.add_argument("--wf_dir", type = str, help="Path to workflow dir, if such if not standard root dirs(None [default] vs /scratch)")

# data arguments
parser.add_argument("--ses", type = str, help="Session to fit (if ses-mean [default for pRF task] then will average both session when that's possible)")
parser.add_argument("--run_type", help="Type of run to fit (mean [default for pRF task], median, 1, loo_r1s1, ...)")
parser.add_argument("--chunk_num", help="Chunk number to fit or None [default]")

#parser.add_argument("--vertex", nargs='+', type=int, help="Vertex index to fit, or list of indexes or None [default]", default =[])
parser.add_argument("--vertex", help="Vertex index to fit, or list of indexes or None [default]")
parser.add_argument("--ROI", help="ROI name to fit or None [default]")

# only relevant for pRF fitting
parser.add_argument("--prf_model_name", type = str, help="Type of model to fit: gauss [default], css, dn, etc...")
parser.add_argument("--fit_hrf", type = int, help="1/0 - if we want tp fit hrf on the data or not [default]")

# only relevant for FA fitting
parser.add_argument("--fa_model_name", type = str, help="Type of FA model to fit: gain [default], glm, etc...")

## set variables 
args = parser.parse_args()

# subject id and processing step of pipeline
participant = str(args.participant).zfill(3) 
task2model = args.task2model 
#
#
# type of session and run to use, depending on task
if task2model == 'pRF':
    ses = args.ses if args.ses is not None else 'ses-mean'
    combine_ses = True if ses == 'ses-mean' else False # if we want to combine sessions
    run_type = args.run_type if args.run_type is not None else 'mean'

elif task2model == 'FA':
    prf_ses = 'ses-mean'
    combine_ses = True
    prf_run_type = 'mean'
    ses = args.ses if args.ses is not None else 1
    run_type = args.run_type if args.run_type is not None else 'loo_r1s1'
    
#
#
# vertex, chunk_num, ROI
#vertex = str(args.vertex).strip('][').split(', ')
vertex = ast.literal_eval(str(args.vertex)) if args.vertex is not None else None
if args.chunk_num is None or str(args.chunk_num) == 'None':
    chunk_num = None
else:
    chunk_num = int(args.chunk_num)
ROI = str(args.ROI) if args.ROI is not None else None
#
#
# system location
system_dir = args.dir if args.dir is not None else "lisa" 
wf_dir = args.wf_dir
#
#
# prf model name and options
prf_model_name = args.prf_model_name if args.prf_model_name is not None else "gauss" 
fit_hrf = bool(args.fit_hrf) if args.fit_hrf is not None else False 
#
# FA model name
fa_model_name = args.fa_model_name if args.fa_model_name is not None else 'gain'

## Load data object
print("Fitting data for subject {sj}!".format(sj=participant))
FAM_data = load_exp_settings.MRIData(params, participant, 
                                    repo_pth = op.split(load_exp_settings.__file__)[0], 
                                    base_dir = system_dir, 
                                    wf_dir = wf_dir,
                                    exclude_sj = [])

## Load preprocessing class for each data type
FAM_mri_preprocess = preproc_mridata.PreprocMRI(FAM_data)

## run specific steps ##
match task2model:
    
    case 'pRF':

        print('Fitting {mn} model on the data\n'.format(mn = prf_model_name))
        print('fit HRF params set to {op}'.format(op = fit_hrf))

        ## load pRF model class
        FAM_pRF = prf_model.pRF_model(FAM_data)

        # set specific params
        FAM_pRF.model_type['pRF'] = prf_model_name
        FAM_pRF.fit_hrf = fit_hrf

        # get participant models, which also will load 
        # DM and mask it according to participants behavior
        pp_prf_models = FAM_pRF.set_models(participant_list = [participant], mask_DM = True, combine_ses = combine_ses)
        
        ## get file extension for post fmriprep
        # processed files
        file_ext = FAM_mri_preprocess.get_mrifile_ext()['pRF']

        ## actually fit
        print('Fitting started!')
        # to time it
        start_time = time.time()

        FAM_pRF.fit_data(participant, pp_prf_models, 
                        ses = ses, run_type = run_type, file_ext = file_ext,
                        vertex = vertex, chunk_num = chunk_num, ROI = ROI,
                        model2fit = prf_model_name,
                        save_estimates = True,
                        xtol = 1e-3, ftol = 1e-4, n_jobs = 16)

        print('Fitting finished, total time = {tempo}!'.format(tempo = time.time() - start_time))

    case 'FA':

        print('Fitting {mn} model on the data\n'.format(mn = fa_model_name))
        print('fit HRF params set to {op}'.format(op = fit_hrf))

        ## load pRF model class
        FAM_pRF = prf_model.pRF_model(FAM_data)

        # set specific params
        FAM_pRF.model_type['pRF'] = prf_model_name
        FAM_pRF.fit_hrf = fit_hrf

        ## load pRF estimates - implies pRF model was already fit (should change to fit pRF model on the spot if needed)
        pp_prf_estimates, pp_prf_models = FAM_pRF.load_pRF_model_estimates(participant,
                                                                    ses = prf_ses, run_type = prf_run_type, 
                                                                    model_name = prf_model_name, 
                                                                    iterative = True,
                                                                    fit_hrf = fit_hrf)

        ## get bounds used for prf estimates of specific model
        pp_prf_stim = pp_prf_models['sub-{sj}'.format(sj = participant)][prf_ses]['prf_stim']
        prf_bounds = FAM_pRF.get_fit_startparams(max_ecc_size = pp_prf_stim.screen_size_degrees/2.0)[FAM_pRF.model_type['pRF']]['bounds']

        ## get file extension for post-fmriprep
        # processed files
        file_ext = FAM_mri_preprocess.get_mrifile_ext()['FA']
        
        ## now fit appropriate feature model
        match fa_model_name:

            case 'gain':

                ## load FA model class
                FAM_FA = feature_model.Gain_model(FAM_data)

                ## actually fit
                print('Fitting started!')
                # to time it
                start_time = time.time()

                _ = FAM_FA.fit_data(participant, pp_prf_estimates, 
                                            ses = ses, run_type = run_type,
                                            chunk_num = chunk_num, vertex = vertex, ROI = ROI,
                                            prf_model_name = prf_model_name, rsq_threshold = None, file_ext = file_ext, 
                                            outdir = None, save_estimates = True,
                                            fit_overlap = True,
                                            xtol = 1e-3, ftol = 1e-4, n_jobs = 16) 

                print('Fitting finished, total time = {tempo}!'.format(tempo = time.time() - start_time))

            case 'glm':

                ## load FA model class
                FAM_FA = feature_model.GLM_model(FAM_data)

                # if we want to fit hrf
                FAM_FA.fit_hrf = FAM_pRF.fit_hrf

                ## actually fit
                print('Fitting started!')
                # to time it
                start_time = time.time()

                _ = FAM_FA.fit_data(participant, pp_prf_estimates, 
                                            ses = ses, run_type = run_type,
                                            chunk_num = chunk_num, vertex = vertex, ROI = ROI,
                                            prf_model_name = prf_model_name, rsq_threshold = None, file_ext = file_ext, 
                                            outdir = None, save_estimates = True,
                                            fit_overlap = False, fit_full_stim = True,
                                            n_jobs = 16) 

                print('Fitting finished, total time = {tempo}!'.format(tempo = time.time() - start_time))

            
            case 'full_stim':

                ## load FA model class
                FAM_FA = feature_model.FullStim_model(FAM_data)

                # if we want to fit hrf
                FAM_FA.fit_hrf = FAM_pRF.fit_hrf
                # set prf bounds
                FAM_FA.prf_bounds = prf_bounds

                # we're only scalling betas, keeping other pRF params the same
                prf_pars2vary = ['betas']

                ## actually fit
                print('Fitting started!')
                # to time it
                start_time = time.time()

                _ = FAM_FA.fit_data(participant, pp_prf_estimates, 
                                            ses = ses, run_type = run_type,
                                            chunk_num = chunk_num, vertex = vertex, ROI = ROI,
                                            prf_model_name = prf_model_name, rsq_threshold = None, file_ext = file_ext, 
                                            outdir = None, save_estimates = True,
                                            prf_pars2vary = prf_pars2vary, reg_name = 'full_stim', bar_keys = ['att_bar', 'unatt_bar'],
                                            xtol = 1e-3, ftol = 1e-4, n_jobs = 16, prf_bounds = None) 

                print('Fitting finished, total time = {tempo}!'.format(tempo = time.time() - start_time))



        


