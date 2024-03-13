import os, sys
import os.path as op
import argparse

import numpy as np
import pandas as pd

import yaml
from FAM.processing import load_exp_settings, preproc_mridata, preproc_behdata
from FAM.visualize.preproc_viewer import MRIViewer
from FAM.visualize.beh_viewer import BehViewer

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
parser.add_argument("--cmd", 
                    type = str.lower, 
                    required = True,
                    help = "Vizualization step of processed data: freeview, nordic, tsnr, bold, etc..."
                    )
parser.add_argument("--dir", 
                    type = str.lower, 
                    default = 'local',
                    help = "System we are making plots in - local [default] vs slurm (snellius)"
                    )
parser.add_argument("--data_type", 
                    type = str.lower, 
                    default = "mri",
                    help = "Type of data to process (mri [default], beh or eye)"
                    )
parser.add_argument("--exclude_sj", 
                    nargs = '*', # 0 or more values expected => creates a list
                    default = [],
                    type = int,
                    help = "List of subs to exclude (ex: 1 2 3 4). Default []"
                    )
parser.add_argument("--task", 
                    type = str, 
                    default = 'pRF',
                    help = "Task to look at (pRF [default] vs FA)"
                    )
parser.add_argument("--atlas", 
                    type = str, 
                    default = None,
                    help = "If we want to use atlas ROIs (ex: glasser, wang) or not [default]."
                    )
parser.add_argument("--use_T2", 
                    action = 'store_true',
                    help = "if option called, will consider T2 file (only relevant for freeview command)")


# parse the command line
args = parser.parse_args()

# access parser options
sj = args.subject[0] if len(args.subject) == 1 else args.subject # for situation where 1 sj vs list
py_cmd = args.cmd # what step of pipeline we want to run
system_dir = args.dir
data_type = args.data_type
exclude_sj = args.exclude_sj # list of excluded subjects
task = args.task
use_atlas = args.atlas
T2_file = args.use_T2

## Load data object --> as relevant paths, variables and utility functions
print("Loading {data} data for subject {sj}!".format(data=data_type, sj=sj))

FAM_data = load_exp_settings.MRIData(params, sj, 
                                    repo_pth = op.split(load_exp_settings.__file__)[0], 
                                    base_dir = system_dir, exclude_sj = exclude_sj)

print('Subject list to vizualize is {l}'.format(l=str(FAM_data.sj_num)))

## Load preprocessing class for each data type ###

# get behavioral info 
FAM_beh = preproc_behdata.PreprocBeh(FAM_data)
# and mri info
FAM_mri = preproc_mridata.PreprocMRI(FAM_data)

match data_type:

    case 'mri':
        
        ## initialize plotter object
        plotter = MRIViewer(FAM_data, use_atlas=use_atlas)

        ## run specific vizualizer
        match py_cmd:

            case 'freeview':

                print('Opening Freeview...')

                freeview_cmd = ''
                while freeview_cmd not in ('movie','view'):
                    freeview_cmd = input("View segmentations (view) or make movie (movie)?: ")

                plotter.check_fs_seg(check_type = freeview_cmd, use_T2 = T2_file, 
                                     participant_list = FAM_data.sj_num)

            case 'nordic':

                print('Comparing NORDIC to standard runs')

                plotter.compare_nordic2standard(participant_list = FAM_data.sj_num, 
                                                file_ext = FAM_mri.get_mrifile_ext())

            case 'tsnr':

                print('Plotting tSNR')

                plotter.plot_tsnr(participant_list = FAM_data.sj_num, 
                                file_ext = FAM_mri.get_mrifile_ext())

            case 'vasculature':

                print('Plotting vasculature proxy for pRF task')

                plotter.plot_vasculature(participant_list = FAM_data.sj_num, 
                                        file_ext = FAM_mri.get_mrifile_ext())

            case 'bold':
                
                print('Plotting BOLD amplitude')

                use_mean_run = ''
                while use_mean_run not in ['y', 'n']:
                    use_mean_run = str(input("Plot mean run? [y/n]: "))

                if use_mean_run == 'y':
                    run_type = 'mean'
                    ses_num = 'mean'
                else:
                    run_type = ''
                    while not isinstance(run_type, int):
                        run_type = int(input("Which run number to choose? (Ex 1, 2, ..): "))
                    ses_num = ''
                    while not isinstance(ses_num, int):
                        ses_num = int(input("Which session number to choose? (Ex 1, 2): "))

                plotter.plot_bold_on_surface(participant_list = FAM_data.sj_num, 
                                            run_num = run_type, ses_num = ses_num, task = task, 
                                            stim_on_screen = FAM_beh.get_stim_on_screen(task = task, 
                                                                                        crop_nr = FAM_data.task_nr_cropTR[task], 
                                                                                        shift = FAM_data.shift_TRs_num),
                                            file_ext = FAM_mri.get_mrifile_ext())

            case 'click':
                
                print('Opening click viewer with raw, filtered and PSC data')

                task_name = 'pRF' if task.lower == 'prf' else 'FA' 

                run = ''
                while isinstance(run, int) == False:
                    run = int(input("Which run number to choose? (Ex 1, 2, ..): "))
                
                ses = ''
                while isinstance(ses, int) == False:
                    ses = int(input("Which session number to choose? (Ex 1, 2): "))

                plotter.check_click_bold(FAM_data.sj_num[0], 
                                        run, ses, 
                                        task = task_name, input_pth = None,
                                        file_ext = FAM_mri.get_mrifile_ext()[task_name])

            case TypeError:
                print('viz option NOT VALID')

    case 'beh':

        ## initialize plotter object
        plotter = BehViewer(FAM_data)
        
        match py_cmd:

            case 'behavior':

                if task == 'FA':
                    
                    # make dataframe with behavioral results
                    att_RT_df = FAM_beh.get_FA_behavioral_results(participant_list = FAM_data.sj_num,
                                                                ses_type = 'func')
                    
                    # get accuracy per ecc
                    acc_df = FAM_beh.get_FA_accuracy(att_RT_df = att_RT_df)

                    # actually plot
                    plotter.plot_FA_behavior(att_RT_df = att_RT_df, 
                                             acc_df = acc_df, 
                                             participant_list = FAM_data.sj_num)

        
                # print('Plotting behavior results for pRF and FA task') ## should do for both
                
                # # first get the dataframe with the mean results
                # df_pRF_beh_summary = FAM_beh.get_pRF_behavioral_results(ses_type = 'func')
                # df_FA_beh_summary = FAM_beh.get_FA_behavioral_results(ses_type = 'func')

                # # actually plot
                # plotter.plot_pRF_behavior(results_df = df_pRF_beh_summary, plot_group = True)
                # plotter.plot_FA_behavior(results_df = df_FA_beh_summary, plot_group = True)

            case TypeError: 
                print('viz option NOT VALID')

    case TypeError: 
        print('data type option NOT VALID')
