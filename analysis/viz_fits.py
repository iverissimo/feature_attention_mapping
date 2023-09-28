import os, sys
import os.path as op
import argparse
import ast

import yaml
from FAM.processing import load_exp_settings, preproc_mridata, preproc_behdata

from FAM.fitting.prf_model import pRF_model
from FAM.fitting.glm_single_model import GLMsingle_Model
from FAM.fitting.feature_model import Gain_model, GLM_model, FullStim_model

from FAM.visualize.fitting_viewer import pRFViewer, FAViewer

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
                    help = "Task to look at (pRF [default] vs FA)"
                    )
parser.add_argument("--cmd", #"--viz",
                    type = str.lower, 
                    required = True,
                    help = "What we want to vizualize: flatmaps, click, single_vert, etc..."
                    )
parser.add_argument("--dir", 
                    type = str.lower, 
                    default = 'local',
                    help = "System we are making plots in - local [default] vs slurm (snellius)"
                    )
parser.add_argument("--exclude_sj", 
                    nargs = '*', # 0 or more values expected => creates a list
                    default = [],
                    type = int,
                    help = "List of subs to exclude (ex: 1 2 3 4). Default []"
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
parser.add_argument("--fa_model_name", 
                    type = str, 
                    default = 'glmsingle',
                    help="Type of FA model to fit: glmsingle [default], gain, glm, etc...]"
                    )
parser.add_argument("--ses2fit", 
                    default = 'mean',
                    help="Session to fit (if mean [default] then will average both session when that's possible)"
                    )
parser.add_argument("--run_type", 
                    default = 'mean',
                    help="Type of run to fit (mean of runs [default], 1, loo_r1s1, ...)"
                    )
parser.add_argument("--vertex",
                    nargs = '*', 
                    default = [],
                    type = int,
                    help="list of vertex indice(s) to view or default []"
                    )
parser.add_argument("--ROI", 
                    type = str,
                    help="ROI name to view or None [default]")
parser.add_argument("--atlas", 
                    type = str, 
                    default = None,
                    help = "If we want to use atlas ROIs (ex: glasser, wang) or not [default]."
                    )
parser.add_argument("--fit_now", 
                    action = 'store_true',
                    help="if option called, fit the data now"
                    )

# parse the command line
args = parser.parse_args()

# access parser options
sj = args.subject[0] if len(args.subject) == 1 else args.subject # for situation where 1 sj vs list
py_cmd = args.cmd # what step of pipeline we want to run
system_dir = args.dir
exclude_sj = args.exclude_sj # list of excluded subjects
task = args.task
use_atlas = args.atlas
prf_model_name = args.prf_model_name
fit_hrf = args.fit_hrf
fit_now = args.fit_now
run_type = args.run_type
ses2fit = args.ses2fit 
fa_model_name = args.fa_model_name

# vertex list
if len(args.vertex)>0:
    vertex = [int(val) for val in args.vertex]

# ROI name
ROI = args.ROI

## Load data object --> as relevant paths, variables and utility functions
print("\Loading data for subject {sj}!".format(sj=sj))

FAM_data = load_exp_settings.MRIData(params, sj, 
                                    repo_pth = op.split(load_exp_settings.__file__)[0], 
                                    base_dir = system_dir, exclude_sj = exclude_sj)

print('Subject list to vizualize is {l}\n'.format(l=str(FAM_data.sj_num)))

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

        print('Visualizing pRF {mn} model outcomes\n'.format(mn = prf_model_name))

        ## get file extension for post fmriprep
        # processed files
        file_ext = FAM_mri.get_mrifile_ext()['pRF']

        ## load plotter class
        plotter = pRFViewer(FAM_data, pRFModelObj = FAM_pRF, use_atlas = use_atlas)

        ## run specific vizualizer
        match py_cmd:

            case 'single_vertex': ## need to review
                plotter.plot_singlevert_pRF(sj, vertex = vertex, file_ext = file_ext, 
                                        fit_now = fit_now, prf_model_name = prf_model_name)


            case 'click': ## need to review
                plotter.open_click_viewer(sj, task2viz = 'pRF',
                                        ses = ses2fit, run_type = run_type,
                                        prf_model_name = prf_model_name, file_ext = file_ext)

            case 'prf_estimates':
                plotter.plot_prf_results(participant_list = FAM_data.sj_num,
                                        prf_model_name = prf_model_name, ses = ses2fit, run_type = run_type,
                                        mask_bool_df = FAM_beh.get_pRF_mask_bool(ses_type = 'func',
                                                                                crop_nr = FAM_data.task_nr_cropTR['pRF'], 
                                                                                shift = FAM_data.shift_TRs_num), # Make DM boolean mask based on subject responses
                                        stim_on_screen = FAM_beh.get_stim_on_screen(task = 'pRF', 
                                                                                    crop_nr = FAM_data.task_nr_cropTR['pRF'], 
                                                                                    shift = FAM_data.shift_TRs_num),
                                        angles2plot_list = ['back']
                                        )

            case 'draw_roi':

                if FAM_data.sj_space in ['fsnative']: # will draw custom label in freesurfer folder

                    plotter.view_pRF_surf_estimates(participant_list = FAM_data.sj_num, 
                                                    ses = ses2fit, run_type = run_type, prf_model_name = prf_model_name,
                                                    mask_bool_df = FAM_beh.get_pRF_mask_bool(ses_type = 'func',
                                                                                crop_nr = FAM_data.task_nr_cropTR['pRF'], 
                                                                                shift = FAM_data.shift_TRs_num), # Make DM boolean mask based on subject responses
                                                    stim_on_screen = FAM_beh.get_stim_on_screen(task = 'pRF', 
                                                                                    crop_nr = FAM_data.task_nr_cropTR['pRF'], 
                                                                                    shift = FAM_data.shift_TRs_num),
                                                    mask_arr = True, iterative = True, open_fs = True, surf_type = 'sphere')
                else:
                    plotter.save_estimates4drawing(participant_list = FAM_data.sj_num, 
                                                    ses = ses2fit, run_type = run_type,
                                                    prf_model_name = prf_model_name, 
                                                    rsq_threshold = 0.1,
                                                    mask_bool_df = FAM_beh.get_pRF_mask_bool(ses_type = 'func',
                                                                                    crop_nr = FAM_data.task_nr_cropTR['pRF'], 
                                                                                    shift = FAM_data.shift_TRs_num), # Make DM boolean mask based on subject responses
                                                    stim_on_screen = FAM_beh.get_stim_on_screen(task = 'pRF', 
                                                                                        crop_nr = FAM_data.task_nr_cropTR['pRF'], 
                                                                                        shift = FAM_data.shift_TRs_num)
                                                    )

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
                model_list.append(mod_2)

                plotter.compare_pRF_model_rsq(participant_list = FAM_data.sj_num,
                                            ses = ses2fit, run_type = run_type,
                                            prf_model_list = model_list,
                                            mask_bool_df = FAM_beh.get_pRF_mask_bool(ses_type = 'func',
                                                                                crop_nr = FAM_data.task_nr_cropTR['pRF'], 
                                                                                shift = FAM_data.shift_TRs_num), # Make DM boolean mask based on subject responses
                                            stim_on_screen = FAM_beh.get_stim_on_screen(task = 'pRF', 
                                                                                    crop_nr = FAM_data.task_nr_cropTR['pRF'], 
                                                                                    shift = FAM_data.shift_TRs_num)
                                            )

    case 'FA':

        print('Visualizing FA {mn} model outcomes\n'.format(mn = fa_model_name))

        ## get file extension for post fmriprep
        # processed files
        file_ext = FAM_mri.get_mrifile_ext()['FA']

        ## load FA model class
        match fa_model_name:

            case 'glmsingle':
                FAM_FA = GLMsingle_Model(FAM_data, use_atlas = use_atlas)

            case 'full_stim':   
                FAM_FA = FullStim_model(FAM_data)
            case 'gain':
                FAM_FA = Gain_model(FAM_data)
            case 'glm':
                FAM_FA = GLM_model(FAM_data)
            
        ## load plotter class
        plotter = FAViewer(FAM_data, pRFModelObj = FAM_pRF, FAModelObj = FAM_FA, use_atlas = use_atlas)

        ## load a few extra params, for specific commands
        match py_cmd:

            case 'betas_coord' | 'attention_mod' | 'bar_dist':

                orientation_bars = {0: 'parallel_vertical', 1: 'parallel_horizontal', 2: 'crossed'}
                
                choice = None
                while choice not in (0,1,2):
                    choice = int(input("Which trial type?\n0) parallel vertical\n1) parallel horizontal\n2) crossed\nNumber picked: "))

                # get participant run number and session number
                # by attended bar
                group_att_color_ses_run = {'sub-{sj}'.format(sj = pp): FAM_beh.get_run_ses_by_color(pp, ses_num = None, 
                                                                                                    ses_type = 'func', 
                                                                                                    run_num = None) for pp in FAM_data.sj_num}

        ## run specific vizualizer
        match py_cmd:

            case 'fa_estimates':
                plotter.plot_glmsingle_estimates(participant_list = FAM_data.sj_num, 
                                                 model_type = ['D'], #['A','D'],
                                                 mask_bool_df = FAM_beh.get_pRF_mask_bool(ses_type = 'func',
                                                                                crop_nr = FAM_data.task_nr_cropTR['pRF'], 
                                                                                shift = FAM_data.shift_TRs_num), # Make DM boolean mask based on subject responses
                                                 stim_on_screen = FAM_beh.get_stim_on_screen(task = 'pRF', 
                                                                                    crop_nr = FAM_data.task_nr_cropTR['pRF'], 
                                                                                    shift = FAM_data.shift_TRs_num)
                                                )
                
            case 'betas_coord':

                ## ask for user input on models to compare
                print('plotting GLMsingle beta estimates...')

                plotter.plot_betas_coord(participant_list = FAM_data.sj_num, 
                                                 model_type = 'D',
                                                 file_ext = '_cropped.npy', 
                                                 orientation_bars = orientation_bars[choice], 
                                                 ROI_list = ['V1'], #['V1', 'V2', 'V3'],
                                                 att_color_ses_run_dict = group_att_color_ses_run,
                                                 mask_bool_df = FAM_beh.get_pRF_mask_bool(ses_type = 'func',
                                                                                crop_nr = FAM_data.task_nr_cropTR['pRF'], 
                                                                                shift = FAM_data.shift_TRs_num), # Make DM boolean mask based on subject responses
                                                 stim_on_screen = FAM_beh.get_stim_on_screen(task = 'pRF', 
                                                                                    crop_nr = FAM_data.task_nr_cropTR['pRF'], 
                                                                                    shift = FAM_data.shift_TRs_num)
                                                )
            case 'attention_mod':

                ## ask for user input on models to compare
                print('plotting GLMsingle attention modulation...')

                plotter.plot_att_modulation(participant_list = FAM_data.sj_num, 
                                                 model_type = 'D',
                                                 file_ext = '_cropped.npy', 
                                                 orientation_bars = orientation_bars[choice], 
                                                 ROI_list = ['V1', 'V2', 'V3'],
                                                 att_color_ses_run_dict = group_att_color_ses_run,
                                                 mask_bool_df = FAM_beh.get_pRF_mask_bool(ses_type = 'func',
                                                                                crop_nr = FAM_data.task_nr_cropTR['pRF'], 
                                                                                shift = FAM_data.shift_TRs_num), # Make DM boolean mask based on subject responses
                                                 stim_on_screen = FAM_beh.get_stim_on_screen(task = 'pRF', 
                                                                                    crop_nr = FAM_data.task_nr_cropTR['pRF'], 
                                                                                    shift = FAM_data.shift_TRs_num)
                                                )

                
            case 'bar_dist':

                ## ask for user input on models to compare
                print('plotting GLMsingle mean beta vs bar distance...')

                plotter.plot_betas_bar_dist(participant_list = FAM_data.sj_num, 
                                            model_type = 'D',
                                            file_ext = '_cropped.npy', 
                                            orientation_bars = orientation_bars[choice], 
                                            ROI_list = ['V1'], #['V1', 'V2', 'V3'],
                                            att_color_ses_run_dict = group_att_color_ses_run,
                                            mask_bool_df = FAM_beh.get_pRF_mask_bool(ses_type = 'func',
                                                                        crop_nr = FAM_data.task_nr_cropTR['pRF'], 
                                                                        shift = FAM_data.shift_TRs_num), # Make DM boolean mask based on subject responses
                                            stim_on_screen = FAM_beh.get_stim_on_screen(task = 'pRF', 
                                                                            crop_nr = FAM_data.task_nr_cropTR['pRF'], 
                                                                            shift = FAM_data.shift_TRs_num)
                                        )

            case 'single_vertex':
                plotter.plot_singlevert_FA(sj, vertex = vertex, file_ext = file_ext, 
                                        ses = ses2fit, run_type = run_type, prf_ses = 'mean', prf_run_type = 'mean',
                                        fit_now = fit_now, prf_model_name = prf_model_name, fa_model_name = fa_model_name)

            case 'click':
                plotter.open_click_viewer(sj, task2viz = 'FA',
                                        prf_ses = 'mean', prf_run_type = 'mean', 
                                        fa_ses = ses2fit, fa_run_type = run_type,
                                        prf_model_name = prf_model_name, fa_model_name = fa_model_name,
                                        fa_file_ext = file_ext, prf_file_ext = FAM_mri.get_mrifile_ext()['pRF'])








