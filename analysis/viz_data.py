import os, sys
import os.path as op
import argparse

import yaml
from FAM.processing import load_exp_settings, preproc_mridata, preproc_behdata
from FAM.visualize.preproc_viewer import MRIViewer
from FAM.visualize.beh_viewer import BehViewer

# load settings from yaml
with open('exp_params.yml', 'r') as f_in:
    params = yaml.safe_load(f_in)

## to get inputs 
parser = argparse.ArgumentParser()
parser.add_argument("--subject", help="Subject number (ex: 001, or 'group'/'all')", required=True)
parser.add_argument("--viz", type = str.lower, help="Vizualization step of processed data: freeview, nordic, tsnr, bold, etc...", required=True)

# optional
parser.add_argument("--data_type", type = str.lower, help="Type of data to process (mri [default], beh or eye)")
parser.add_argument("--dir", type = str.lower, help="System we are making plots in (local [default] vs lisa)")
parser.add_argument("--T2", type = int, help="Consider T2 file - only for freeview command (0 [default] vs 1)")
#  only relevant if subject == group/all
parser.add_argument("--exclude_sj", nargs='+', help="List of subjects to exclude, define as --exclude_sj 0 1 ...", default=[])


# set variables 
args = parser.parse_args()

sj = str(args.subject).zfill(3) # subject
viz = args.viz # what step of pipeline we want to run

data_type = args.data_type if args.data_type is not None else "mri" # type of data 

system_dir = args.dir if args.dir is not None else "local" # system location

T2_file = bool(args.T2) if args.T2 is not None else False # make it boolean

exclude_sj = args.exclude_sj # list of excluded subjects
if len(exclude_sj)>0:
    exclude_sj = [val.zfill(3) for val in exclude_sj]
    print('Excluding participants {expp}'.format(expp = exclude_sj))
else:
    exclude_sj = []

## Load data object
print("Loading {data} data for subject {sj}!".format(data=data_type, sj=sj))

FAM_data = load_exp_settings.MRIData(params, sj, 
                                    repo_pth = op.split(load_exp_settings.__file__)[0], 
                                    base_dir=system_dir, exclude_sj = exclude_sj)

print('Subject list to vizualize is {l}'.format(l=str(FAM_data.sj_num)))

## Load preprocessing class for each data type ###

match data_type:

    case 'mri':

        FAM_preproc = preproc_mridata.PreprocMRI(FAM_data)

        ## run specific vizualizer
        match viz:
            case 'freeview':

                print('Opening Freeview...')

                freeview_cmd = ''
                while freeview_cmd not in ('movie','view'):
                    freeview_cmd = input("View segmentations (view) or make movie (movie)?: ")

                plotter = MRIViewer(FAM_data)
                plotter.check_fs_seg(check_type = freeview_cmd, use_T2 = T2_file, participant_list = FAM_preproc.MRIObj.sj_num)

            case 'nordic':

                print('Comparing NORDIC to standard runs')

                plotter = MRIViewer(FAM_data)
                plotter.compare_nordic2standard(participant_list = FAM_preproc.MRIObj.sj_num, 
                                                input_pth = None, 
                                                file_ext = FAM_preproc.get_mrifile_ext())

            case 'tsnr':

                print('Plotting tSNR')

                plotter = MRIViewer(FAM_data)
                plotter.plot_tsnr(participant_list = FAM_preproc.MRIObj.sj_num, 
                                                input_pth = None, 
                                                file_ext = FAM_preproc.get_mrifile_ext())

            case 'vasculature':

                print('Plotting vasculature proxy for pRF task')

                plotter = MRIViewer(FAM_data)
                plotter.plot_vasculature(participant_list = FAM_preproc.MRIObj.sj_num, 
                                                input_pth = None, 
                                                file_ext = FAM_preproc.get_mrifile_ext())

            case 'bold':
                
                print('Plotting BOLD amplitude')

                plotter = MRIViewer(FAM_data)
                plotter.plot_bold_on_surface(participant_list = FAM_preproc.MRIObj.sj_num, 
                                            input_pth = None, 
                                            run_type = 'mean', 
                                            task = 'pRF',
                                            stim_on_screen = None,
                                            file_ext = FAM_preproc.get_mrifile_ext())

            case TypeError:
                print('viz option NOT VALID')


    case 'beh':
        
        FAM_preproc = preproc_behdata.PreprocBeh(FAM_data)

        ## run specific vizualizer
        match viz:
            case 'behavior':
        
                print('Plotting behavior results for pRF task') ## should do for both
                
                # first get the dataframe with the mean results
                df_beh_summary = FAM_preproc.get_pRF_behavioral_results(ses_type = 'func')

                plotter = BehViewer(FAM_data)
                plotter.plot_pRF_behavior(results_df = df_beh_summary, plot_group = True)

            case TypeError: 
                print('viz option NOT VALID')

    case TypeError: 
        print('data type option NOT VALID')
