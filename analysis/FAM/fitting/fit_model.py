import os, sys
import os.path as op
import argparse
import ast

import yaml
from FAM.processing import load_exp_settings, preproc_mridata
from FAM.fitting import prf_model

# load settings from yaml
with open('exp_params.yml', 'r') as f_in:
    params = yaml.safe_load(f_in)

## to get inputs 
parser = argparse.ArgumentParser()
parser.add_argument("--participant", help="Subject number (ex: 001)", required=True)
parser.add_argument("--task2model", type = str, help="On which task to fit model (pRF/FA)", required=True)
# optional
parser.add_argument("--dir", type = str, help="System we are running analysis (lisa [default] vs local)")

# data arguments
parser.add_argument("--ses", type = str, help="Session to fit (if ses-mean [default] then will average both session when that's possible)")
parser.add_argument("--run_type", help="Type of run to fit (mean [default], median, 1, loo_1, ...)")
parser.add_argument("--chunk_num", type = int, help="Chunk number to fit or None [default]")
#parser.add_argument("--vertex", nargs='+', type=int, help="Vertex index to fit, or list of indexes or None [default]")
parser.add_argument("--vertex", type = str, help="Vertex index to fit, or list of indexes or None [default]")
parser.add_argument("--ROI",type = str, help="ROI name to fit")

# only relevant for pRF fitting
parser.add_argument("--prf_model_name", type = str, help="Type of model to fit: gauss [default], css, dn, etc...")
parser.add_argument("--fit_hrf", type = int, help="1/0 - if we want tp fit hrf on the data or not [default]")


## set variables 
args = parser.parse_args()

# subject id and processing step of pipeline
participant = str(args.participant).zfill(3) 
task2model = args.task2model 
#
#
# type of session and run to use
ses = args.ses if args.ses is not None else 'ses-mean'
run_type = args.run_type if args.run_type is not None else 'mean'
#
#
# vertex, chunk_num, ROI
#vertex = str(args.vertex).strip('][').split(', ')
vertex = ast.literal_eval(str(args.vertex)) if args.vertex is not None else None
chunk_num = args.chunk_num 
ROI = args.ROI 
#
#
# system location
system_dir = args.dir if args.dir is not None else "lisa" 
#
#
# prf model name and options
prf_model_name = args.prf_model_name if args.prf_model_name is not None else "gauss" 
fit_hrf = bool(args.fit_hrf) if args.fit_hrf is not None else False 



## Load data object
print("Fitting data for subject {sj}!".format(sj=participant))
FAM_data = load_exp_settings.MRIData(params, participant, 
                                    repo_pth = op.split(load_exp_settings.__file__)[0], 
                                    base_dir = system_dir, 
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
        FAM_pRF.model_type = prf_model_name
        FAM_pRF.fit_hrf = fit_hrf

        # get participant models, which also will load 
        # DM and mask it according to participants behavior
        pp_prf_models = FAM_pRF.set_models(participant_list = [participant], mask_DM = True, combine_ses = True)
        
        ## get file extension for post fmriprep
        # processed files
        file_ext = FAM_mri_preprocess.get_mrifile_ext()['pRF']

        ## actually fit
        FAM_pRF.fit_data(participant, pp_prf_models, 
                        ses = ses, run_type = run_type, file_ext = file_ext,
                        vertex = vertex, chunk_num = chunk_num, ROI = ROI,
                        model2fit = prf_model_name,
                        save_estimates = True,
                        xtol = 1e-2, ftol = 1e-4, n_jobs = 16)

    case 'FA':

        raise NameError('Not implemented yet')


