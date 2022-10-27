import os, sys
import os.path as op
import argparse
import ast

import yaml
from FAM.processing import load_exp_settings, preproc_mridata, preproc_behdata
from FAM.visualize.preproc_viewer import MRIViewer
from FAM.visualize.beh_viewer import BehViewer

from FAM.fitting import prf_model, feature_model
from FAM.visualize.fitting_viewer import pRFViewer, FAViewer

# load settings from yaml
with open('exp_params.yml', 'r') as f_in:
    params = yaml.safe_load(f_in)

## to get inputs 
parser = argparse.ArgumentParser()
parser.add_argument("--subject", help="Subject number (ex: 001, or 'group'/'all')", required=True)
parser.add_argument("--task", type = str, help="On which task to fit model (pRF/FA)", required=True)
parser.add_argument("--viz", type = str.lower, help="What we want to vizualize: flatmaps, click, single_vert, etc...", required=True)

# optional
parser.add_argument("--dir", type = str.lower, help="System we are making plots in (local [default] vs lisa)")

# only relevant for pRF fitting
parser.add_argument("--prf_model_name", type = str, help="Type of pRF model to fit: gauss [default], css, dn, etc...")
parser.add_argument("--fit_hrf", type = int, help="1/0 - if we want to fit hrf on the data or not [default]")

# only relevant for FA fitting
parser.add_argument("--fa_model_name", type = str, help="Type of FA model to fit: gain [default], glm, etc...")

# data arguments
parser.add_argument("--ses2fit", type = str, help="Session to fit (if ses-mean [default] then will average both session when that's possible)")
parser.add_argument("--run_type", help="Type of run to fit (mean of runs [default], median, 1, loo_1, ...)")

#parser.add_argument("--vertex", nargs='+', type=int, help="Vertex index to fit, or list of indexes or None [default]", default =[])
## mostly relevant for single vertex plots 
parser.add_argument("--vertex", type = str, help="Vertex index to view, or list of indexes or None [default]")
parser.add_argument("--ROI",type = str, help="ROI name to fit")

# if we want to use atlas for ROI definition
parser.add_argument("--atlas_bool", type = int, help="1/0 - if we want to use atlas ROI [default] or sub specific roi")

# only relevant if single voxel viewer
parser.add_argument("--fit_now", type = int, help="1/0 - if we want to fit the data now [default] or load when possible")

#  only relevant if subject == group/all
parser.add_argument("--exclude_sj", nargs='+', help="List of subjects to exclude, define as --exclude_sj 0 1 ...", default=[])


# set variables 
args = parser.parse_args()

sj = str(args.subject).zfill(3) # subject
viz = args.viz # what step of pipeline we want to run
task = args.task # type of task 
#
#
system_dir = args.dir if args.dir is not None else "local" # system location
#
#
# type of session and run to use, depending on task
if task == 'pRF':
    ses = args.ses2fit if args.ses2fit is not None else 'ses-mean'
    combine_ses = True if ses == 'ses-mean' else False # if we want to combine sessions
    run_type = args.run_type if args.run_type is not None else 'mean'

elif task == 'FA':
    prf_ses = 'ses-mean'
    combine_ses = True
    prf_run_type = 'mean'
    ses = args.ses2fit if args.ses2fit is not None else 1
    run_type = args.run_type if args.run_type is not None else 'loo_r1s1'
#
#
# prf model name and options
prf_model_name = args.prf_model_name if args.prf_model_name is not None else "gauss" 
fit_hrf = bool(args.fit_hrf) if args.fit_hrf is not None else False 
fit_now = bool(args.fit_now) if args.fit_now is not None else True
#
# FA model name
fa_model_name = args.fa_model_name if args.fa_model_name is not None else 'gain'
#
#
# vertex, chunk_num, ROI
#vertex = str(args.vertex).strip('][').split(', ')
vertex = ast.literal_eval(str(args.vertex)) if args.vertex is not None else None
ROI = args.ROI 
#
atlas_bool = bool(args.atlas_bool) if args.atlas_bool is not None else True 

exclude_sj = args.exclude_sj # list of excluded subjects
if len(exclude_sj)>0:
    exclude_sj = [val.zfill(3) for val in exclude_sj]
    print('Excluding participants {expp}'.format(expp = exclude_sj))
else:
    exclude_sj = []

## Load data object
print("Loading data for subject {sj}!".format(sj=sj))

FAM_data = load_exp_settings.MRIData(params, sj, 
                                    repo_pth = op.split(load_exp_settings.__file__)[0], 
                                    base_dir=system_dir, exclude_sj = exclude_sj)

print('Subject list to vizualize is {l}'.format(l=str(FAM_data.sj_num)))

## Load preprocessing class for each data type
FAM_mri_preprocess = preproc_mridata.PreprocMRI(FAM_data)

print('Setting pRF {mn} model\n'.format(mn = prf_model_name))
print('fit HRF params set to {op}'.format(op = fit_hrf))

## load pRF model class
FAM_pRF = prf_model.pRF_model(FAM_data)

# set specific params
FAM_pRF.model_type['pRF'] = prf_model_name
FAM_pRF.fit_hrf = fit_hrf

## run specific steps ##
match task:

    case 'pRF':

        print('Visualizing pRF {mn} model outcomes\n'.format(mn = prf_model_name))

        ## get file extension for post fmriprep
        # processed files
        file_ext = FAM_mri_preprocess.get_mrifile_ext()['pRF']

        ## load plotter class
        plotter = pRFViewer(FAM_data, pRFModelObj = FAM_pRF, combine_ses = combine_ses, use_atlas_rois = atlas_bool,
                                pysub = FAM_data.params['plotting']['pycortex_sub'], use_sub_rois = FAM_data.params['plotting']['use_sub_rois'])

        ## run specific vizualizer
        match viz:

            case 'single_vertex':
                plotter.plot_singlevert_pRF(sj, vertex = vertex, file_ext = file_ext, 
                                        fit_now = fit_now, prf_model_name = prf_model_name)


            case 'click':
                plotter.open_click_viewer(sj, task2viz = 'pRF',
                                        ses = ses, run_type = run_type,
                                        prf_model_name = prf_model_name, file_ext = file_ext)

            case 'prf_estimates':
                plotter.plot_prf_results(participant_list = FAM_data.sj_num,
                                        prf_model_name = prf_model_name, 
                                        rsq_threshold = FAM_data.params['plotting']['rsq_threshold'])

            case 'draw_roi':

                # in case we want to average participants
                sub_name = FAM_data.sj_num if len(FAM_data.sj_num) > 1 else sj

                plotter.save_estimates4drawing(sub_name, task2draw = 'pRF',
                                                ses = ses, run_type = run_type,
                                                prf_model_name = prf_model_name, file_ext = file_ext)

            case 'prf_rsq':
                ## ask for user input on models to compare
                print('Comparing pRF model rsq...')
                model_list = []
                
                mod_1 = ''
                while mod_1 not in ('gauss','css','dog','dn'):
                    mod_1 = input("First model name to compare RSQ?: ")

                model_list.append(mod_1)

                mod_2 = ''
                while mod_2 not in ('gauss','css','dog','dn'):
                    mod_2 = input("Second model name to compare RSQ?: ")

                if mod_2 != mod_1:
                    model_list.append(mod_2)

                plotter.compare_pRF_model_rsq(participant_list = FAM_data.sj_num,
                                            ses = ses, run_type = run_type,
                                            prf_model_list = model_list,
                                            rsq_threshold = FAM_data.params['plotting']['rsq_threshold'])


    case 'FA':

        print('Visualizing FA {mn} model outcomes\n'.format(mn = fa_model_name))

        ## get file extension for post fmriprep
        # processed files
        file_ext = FAM_mri_preprocess.get_mrifile_ext()['FA']

        ## load FA model class
        FAM_FA = feature_model.FullStim_model(FAM_data)

        # if we want to fit hrf
        FAM_FA.fit_hrf = FAM_pRF.fit_hrf


        ## load plotter class
        plotter = FAViewer(FAM_data, pRFModelObj = FAM_pRF, FAModelObj = FAM_FA,
                                combine_ses = combine_ses, use_atlas_rois = atlas_bool,
                                pysub = FAM_data.params['plotting']['pycortex_sub'], use_sub_rois = FAM_data.params['plotting']['use_sub_rois'])

        ## run specific vizualizer
        match viz:

            case 'single_vertex':
                plotter.plot_singlevert_FA(sj, vertex = vertex, file_ext = file_ext, 
                                        ses = ses, run_type = run_type, prf_ses = prf_ses, prf_run_type = prf_run_type,
                                        fit_now = fit_now, prf_model_name = prf_model_name, fa_model_name = fa_model_name)








