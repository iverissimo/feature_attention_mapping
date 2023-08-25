

import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

import ptitprince as pt # raincloud plots
import matplotlib.patches as mpatches
from  matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

import cortex

import subprocess

# from FAM.utils import mri as mri_utils
# from FAM.utils import plot as plot_utils
# from FAM.processing import preproc_behdata
# from FAM.visualize import click_viewer

from FAM.visualize.viewer import Viewer

from PIL import Image, ImageDraw

class pRFViewer(Viewer):

    def __init__(self, MRIObj, outputdir = None, pRFModelObj = None, pysub = 'hcp_999999', use_atlas = None):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
        pRFModelObj: pRF Model object
            object from one of the classes defined in prf_model.pRF_model
        outputdir: str
            path to save plots
        pysub: str
            basename of pycortex subject folder, where we drew all ROIs. 
        """

        # need to initialize parent class (Model), indicating output infos
        super().__init__(MRIObj, pysub = pysub, outputdir = outputdir, use_atlas = use_atlas)

        ## output path to save plots
        self.figures_pth = op.join(self.outputdir)
        os.makedirs(self.figures_pth, exist_ok=True)

        # Load pRF model object
        self.pRFModelObj = pRFModelObj
                
    def save_estimates4drawing(self, participant_list = [],
                                    ses = 'mean', run_type = 'mean',  mask_bool_df = None, stim_on_screen = [],
                                    prf_model_name = 'gauss', rsq_threshold = .1, mask_arr = True,
                                    colormap_list = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb', '#3d549f','#655099','#ad5a9b','#dd3933'],
                                    n_bins_colors = 256, angle_thresh = 3*np.pi/4):

        """
        Load pRF estimates into pycortex sub specific overlay, to draw ROIs
        Will load polar angle (with and without alpha level) + eccentricity +  size +  r2

        Parameters
        ----------
        participant_list : list
            list of subject ID
        ses: str
            session of input data
        run_type : str
            type of run of input data (ex: 1/mean)
        prf_model_name: str
            name of prf model that was fit
        rsq_threshold: float
            minimum RSQ threshold to use for figures
        """

        # if we provide a list, we want to save average estimates in overlay
        if len(participant_list) == 1:
            sub_name = 'sub-{sj}'.format(sj = participant_list[0])
        else:
            sub_name = 'sub-group'

        # check if subject pycortex folder exists
        pysub_folder = '{pp}_{ps}'.format(ps = self.pysub, pp = sub_name)

        if op.exists(op.join(cortex.options.config.get('basic', 'filestore'), pysub_folder)):
            print('Participant overlay %s in pycortex filestore, assumes we draw ROIs there'%pysub_folder)
        else:
            raise NameError('FOLDER %s DOESNT EXIST'%op.join(cortex.options.config.get('basic', 'filestore'), pysub_folder))

        ## load estimates for all participants 
        # store in dict, for ease of access
        print('Loading iterative estimates')
        group_estimates, group_prf_models = self.pRFModelObj.load_pRF_model_estimates(participant_list = participant_list,
                                                                    ses = ses, run_type = run_type, 
                                                                    model_name = prf_model_name, 
                                                                    iterative = True,
                                                                    mask_bool_df = mask_bool_df, stim_on_screen = stim_on_screen,
                                                                    fit_hrf = self.pRFModelObj.fit_hrf)

        ## mask the estimates, if such is the case
        if mask_arr:
            print('masking estimates')

            # get estimate keys
            keys = self.pRFModelObj.get_prf_estimate_keys(prf_model_name = prf_model_name)

            # get screen lim for all participants
            max_ecc_ext = {'sub-{sj}'.format(sj = pp): group_prf_models['sub-{sj}'.format(sj = pp)]['ses-{s}'.format(s = ses)]['prf_stim'].screen_size_degrees/2 for pp in participant_list}

            final_estimates = {'sub-{sj}'.format(sj = pp): self.pRFModelObj.mask_pRF_model_estimates(group_estimates['sub-{sj}'.format(sj = pp)], 
                                                                                estimate_keys = keys,
                                                                                x_ecc_lim = np.array([- 1, 1]) * max_ecc_ext['sub-{sj}'.format(sj = pp)],
                                                                                y_ecc_lim = np.array([- 1, 1]) * max_ecc_ext['sub-{sj}'.format(sj = pp)],
                                                                                rsq_threshold = rsq_threshold) for pp in participant_list}
        else:
            final_estimates = group_estimates
        
        
        ## get estimates for all participants, if applicable 
        r2_avg = np.stack((final_estimates['sub-{sj}'.format(sj = pp)]['r2'] for pp in participant_list))
        xx_avg = np.stack((final_estimates['sub-{sj}'.format(sj = pp)]['x'] for pp in participant_list))
        yy_avg = np.stack((final_estimates['sub-{sj}'.format(sj = pp)]['y'] for pp in participant_list))
        size_avg = np.stack((final_estimates['sub-{sj}'.format(sj = pp)]['size'] for pp in participant_list))

        ## TAKE MEDIAN
        r2_avg = np.nanmedian(r2_avg, axis = 0)
        xx_avg = np.nanmedian(xx_avg, axis = 0)
        yy_avg = np.nanmedian(yy_avg, axis = 0)
        size_avg = np.nanmedian(size_avg, axis = 0)

        ## use RSQ as alpha level for flatmaps
        alpha_level = np.ones(r2_avg.shape[0]) #self.MRIObj.mri_utils.normalize(np.clip(r2_avg, 0, .6)) # normalize 

        ## get ECCENTRICITY estimates
        eccentricity = self.pRFModelObj.get_eccentricity(xx = xx_avg,
                                                        yy = yy_avg,
                                                        rsq = r2_avg)
        
        ## calculate polar angle (normalize PA between 0 and 1)
        polar_angle_norm = self.pRFModelObj.get_polar_angle(xx = xx_avg, yy = yy_avg, rsq = r2_avg,
                                                        pa_transform = 'norm', angle_thresh = None)
            
        # get matplotlib color map from segmented colors
        PA_cmap = self.plot_utils.make_colormap(colormap_list, bins = n_bins_colors, 
                                                    cmap_name = 'PA_mackey_custom',
                                                    discrete = False, add_alpha = False, return_cmap = True)
        
         ## also plot non-uniform color wheel ##
        cmap_pa_sns = sns.hls_palette(as_cmap=True, h = 0.01, s=.9, l=.65)
        pa_transformed = self.pRFModelObj.get_polar_angle(xx = xx_avg, yy = yy_avg, rsq = r2_avg,
                                                        pa_transform = 'flip', angle_thresh = angle_thresh)


        ## set flatmaps ##
        images = {}
        
        ## pRF rsq
        images['pRF_rsq'] = self.plot_utils.plot_flatmap(r2_avg, 
                                                        pysub = pysub_folder, cmap = 'hot', 
                                                        vmin1 = 0, vmax1 = 1, 
                                                        fig_abs_name = None)
        
        images['ECC'] = self.plot_utils.plot_flatmap(eccentricity, 
                                                    pysub = pysub_folder, cmap = 'viridis', 
                                                    vmin1 = 0, vmax1 = 5.5,
                                                    est_arr2 = alpha_level,
                                                    vmin2 = 0, vmax2 = 1, 
                                                    fig_abs_name = None)
        
        images['Size'] = self.plot_utils.plot_flatmap(size_avg, 
                                                    pysub = pysub_folder, cmap = 'cubehelix', 
                                                    vmin1 = 0, vmax1 = 10,
                                                    est_arr2 = alpha_level,
                                                    vmin2 = 0, vmax2 = 1, 
                                                    fig_abs_name = None)

        images['PA'] = self.plot_utils.plot_flatmap(polar_angle_norm, 
                                                pysub = pysub_folder, cmap = PA_cmap, 
                                                vmin1 = 0, vmax1 = 1,
                                                est_arr2 = alpha_level,
                                                vmin2 = 0, vmax2 = 1, 
                                                with_colorbar = False,
                                                fig_abs_name = None)
        
        images['PA_nonUNI'] = self.plot_utils.plot_flatmap(pa_transformed, 
                                                    pysub = pysub_folder, cmap = cmap_pa_sns, 
                                                    vmin1 = -angle_thresh, vmax1 = angle_thresh, 
                                                    est_arr2 = alpha_level,
                                                    vmin2 = 0, vmax2 = 1, 
                                                    with_colorbar = False,
                                                    fig_abs_name = None)
        
        ### ADD TO OVERLAY, TO DRAW BORDERS
        self.plot_utils.add_data2overlay(flatmap = images['pRF_rsq'], name = 'RSQ_{sj}_task-pRF_run-{run}_model-{model}_withHRF-{hrf_bool}'.format(sj = sub_name,
                                                                                                                            run = run_type, model = prf_model_name,
                                                                                                                            hrf_bool = str(self.pRFModelObj.fit_hrf)))
        self.plot_utils.add_data2overlay(flatmap = images['ECC'], name = 'ECC_{sj}_task-pRF_run-{run}_model-{model}_withHRF-{hrf_bool}'.format(sj = sub_name,
                                                                                                                            run = run_type, model = prf_model_name,
                                                                                                                            hrf_bool = str(self.pRFModelObj.fit_hrf)))
        self.plot_utils.add_data2overlay(flatmap = images['Size'], name = 'Size_{sj}_task-pRF_run-{run}_model-{model}_withHRF-{hrf_bool}'.format(sj = sub_name,
                                                                                                                            run = run_type, model = prf_model_name,
                                                                                                                            hrf_bool = str(self.pRFModelObj.fit_hrf)))
        self.plot_utils.add_data2overlay(flatmap = images['PA'], name = 'PA_{sj}_task-pRF_run-{run}_model-{model}_withHRF-{hrf_bool}'.format(sj = sub_name,
                                                                                                                            run = run_type, model = prf_model_name,
                                                                                                                            hrf_bool = str(self.pRFModelObj.fit_hrf)))
        self.plot_utils.add_data2overlay(flatmap = images['PA_nonUNI'], name = 'PA_nonUNI_{sj}_task-pRF_run-{run}_model-{model}_withHRF-{hrf_bool}'.format(sj = sub_name,
                                                                                                                            run = run_type, model = prf_model_name,
                                                                                                                            hrf_bool = str(self.pRFModelObj.fit_hrf)))

        print('Done')

    def plot_prf_results(self, participant_list = [], mask_bool_df = None, stim_on_screen = [],
                                ses = 'mean', run_type = 'mean', prf_model_name = 'gauss',
                                mask_arr = True, iterative = True):


        ## load estimates for all participants 
        # store in dict, for ease of access
        print('Loading iterative estimates')
        group_estimates, group_prf_models = self.pRFModelObj.load_pRF_model_estimates(participant_list = participant_list,
                                                                    ses = ses, run_type = run_type, 
                                                                    model_name = prf_model_name, 
                                                                    iterative = iterative,
                                                                    mask_bool_df = mask_bool_df, stim_on_screen = stim_on_screen,
                                                                    fit_hrf = self.pRFModelObj.fit_hrf)

        ## mask the estimates, if such is the case
        if mask_arr:
            print('masking estimates')

            # get estimate keys
            keys = self.pRFModelObj.get_prf_estimate_keys(prf_model_name = prf_model_name)

            # get screen lim for all participants
            max_ecc_ext = {'sub-{sj}'.format(sj = pp): group_prf_models['sub-{sj}'.format(sj = pp)]['ses-{s}'.format(s = ses)]['prf_stim'].screen_size_degrees/2 for pp in participant_list}

            final_estimates = {'sub-{sj}'.format(sj = pp): self.pRFModelObj.mask_pRF_model_estimates(group_estimates['sub-{sj}'.format(sj = pp)], 
                                                                                estimate_keys = keys,
                                                                                x_ecc_lim = np.array([- 1, 1]) * max_ecc_ext['sub-{sj}'.format(sj = pp)],
                                                                                y_ecc_lim = np.array([- 1, 1]) * max_ecc_ext['sub-{sj}'.format(sj = pp)],
                                                                                rsq_threshold = self.rsq_threshold_plot) for pp in participant_list}
        else:
            final_estimates = group_estimates

        ## Now actually plot results
        # 
        ### RSQ ###
        self.plot_rsq(participant_list = participant_list, group_estimates = final_estimates, ses = ses, run_type = run_type,
                                            model_name = prf_model_name, fit_hrf = self.pRFModelObj.fit_hrf, vmin1 = 0, vmax1 = .8,
                                            figures_pth = op.join(self.figures_pth, 'rsq', self.pRFModelObj.fitfolder['pRF']))

        ### ECC and SIZE ###
        self.plot_ecc_size(participant_list = participant_list, group_estimates = final_estimates, ses = ses, run_type = run_type,
                                            model_name = prf_model_name, n_bins_dist = 8, 
                                            vmin1 = {'ecc': 0, 'size': 0}, vmax1 = {'ecc': 5.5, 'size': 15})

        ### EXPONENT ###
        if prf_model_name == 'css':
            self.plot_exponent(participant_list = participant_list, group_estimates = final_estimates, ses = ses, run_type = run_type,
                                            model_name = prf_model_name, vmin1 = 0, vmax1 = 1)

        ### POLAR ANGLE ####
        self.plot_pa(participant_list = participant_list, group_estimates = final_estimates, ses = ses, run_type = run_type,
                                        model_name = prf_model_name, 
                                        n_bins_colors = 256, max_x_lim = 5.5, angle_thresh = 3*np.pi/4)
        
        ### Visual Field coverage ###
        self.plot_VFcoverage(participant_list = participant_list, group_estimates = final_estimates, ses = ses, run_type = run_type,
                                        model_name = prf_model_name, max_ecc_ext_dict = max_ecc_ext)

    def compare_pRF_model_rsq(self, participant_list = [], ses = 'mean', run_type = 'mean', 
                                prf_model_list = ['gauss', 'css'], mask_bool_df = None, stim_on_screen = [],
                                mask_arr = True, figures_pth = None, vmin1 = -0.1, vmax1 = 0.1):
        
        # make general output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.figures_pth, 'rsq', self.pRFModelObj.fitfolder['pRF'])

        # if we only provided one model name, assumes we want to compare grid to iterative rsq
        it_bool = [True, False] if prf_model_list[0] == prf_model_list[-1] else [True, True]
        it_key = ['iterative', 'grid'] if prf_model_list[0] == prf_model_list[-1] else ['iterative', 'iterative']

        # for violin plots - to indicate which variable will be used for hue
        plot_hue = 'iterative' if prf_model_list[0] == prf_model_list[-1] else 'model'

        final_estimates = {k: {} for k in prf_model_list}

        ## iterate over models
        for ind, mod_name in enumerate(prf_model_list):

            ## load estimates for all participants 
            # store in dict, for ease of access
            print('Loading iterative estimates')
            group_estimates, group_prf_models = self.pRFModelObj.load_pRF_model_estimates(participant_list = participant_list,
                                                                        ses = ses, run_type = run_type, 
                                                                        model_name = mod_name, 
                                                                        iterative = it_bool[ind],
                                                                        mask_bool_df = mask_bool_df, stim_on_screen = stim_on_screen,
                                                                        fit_hrf = self.pRFModelObj.fit_hrf)
            
            ## mask the estimates, if such is the case
            if mask_arr:
                print('masking estimates')

                # get estimate keys
                keys = self.pRFModelObj.get_prf_estimate_keys(prf_model_name = mod_name)

                # get screen lim for all participants
                max_ecc_ext = {'sub-{sj}'.format(sj = pp): group_prf_models['sub-{sj}'.format(sj = pp)]['ses-{s}'.format(s = ses)]['prf_stim'].screen_size_degrees/2 for pp in participant_list}

                final_estimates[mod_name][it_key[ind]] = {'sub-{sj}'.format(sj = pp): self.pRFModelObj.mask_pRF_model_estimates(group_estimates['sub-{sj}'.format(sj = pp)], 
                                                                                    estimate_keys = keys,
                                                                                    x_ecc_lim = np.array([- 1, 1]) * max_ecc_ext['sub-{sj}'.format(sj = pp)],
                                                                                    y_ecc_lim = np.array([- 1, 1]) * max_ecc_ext['sub-{sj}'.format(sj = pp)],
                                                                                    rsq_threshold = self.rsq_threshold_plot) for pp in participant_list}
            else:
                final_estimates[mod_name][it_key[ind]] = group_estimates

        # iterate over participants, and plot diff in rsq over surface
        # save values per roi in dataframe
        avg_roi_df = pd.DataFrame()
        
        ## loop over participants in list
        for pp in participant_list:

            ## load ROI dict for participant
            pp_ROI_dict = self.load_ROIs_dict(sub_id = pp)

            # make path to save sub-specific figures
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp))
            os.makedirs(sub_figures_pth, exist_ok=True)

            ## plot rsq diff values on flatmap surface ##
            fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_task-{tsk}_acq-{acq}_space-{space}_ses-{ses}_run-{run}_model1-{model1}_{it1}_model2-{model2}_{it2}_diff_flatmap_RSQ.png'.format(sj=pp, tsk = 'pRF',
                                                                                                                            acq = self.MRIObj.acq, space = self.MRIObj.sj_space,
                                                                                                                            ses = ses, run = run_type, model1 = prf_model_list[0],
                                                                                                                            model2 = prf_model_list[1], it1 = it_key[0], it2 = it_key[1] ))

            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            self.plot_utils.plot_flatmap(final_estimates[prf_model_list[0]][it_key[0]]['sub-{sj}'.format(sj = pp)]['r2'] - final_estimates[prf_model_list[1]][it_key[1]]['sub-{sj}'.format(sj = pp)]['r2'], 
                                        pysub = self.get_pysub_name(sub_id = pp), cmap = 'BuBkRd', 
                                        vmin1 = vmin1, vmax1 = vmax1, 
                                        fig_abs_name = fig_name)
            
            ## iterate over models
            for ind, mod_name in enumerate(prf_model_list):

                ## concatenate estimates per ROI per participant, to make group plot
                avg_roi_df = pd.concat((avg_roi_df,
                                        self.MRIObj.mri_utils.get_estimates_roi_df(pp, estimates_pp = final_estimates[mod_name][it_key[ind]]['sub-{sj}'.format(sj = pp)], 
                                                                            ROIs_dict = pp_ROI_dict, 
                                                                            est_key = 'r2', model = mod_name,
                                                                            iterative = it_bool[ind])
                                        ))
                                    
            fig, ax1 = plt.subplots(1,1, figsize=(15,5), dpi=100, facecolor='w', edgecolor='k')

            v1 = sns.violinplot(data = avg_roi_df[avg_roi_df['sj'] == 'sub-{sj}'.format(sj = pp)], 
                                x = 'ROI', y = 'value', hue = plot_hue,
                                order = pp_ROI_dict.keys(),
                                cut=0, inner='box',
                                linewidth=2.7, saturation = 1, ax = ax1) 
            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)

            plt.xlabel('ROI',fontsize = 20,labelpad=18)
            plt.ylabel('RSQ',fontsize = 20,labelpad=18)
            plt.ylim(0,1)
            fig.savefig(fig_name.replace('flatmap','violinplot'))

        # if we provided several participants, make group plot
        if len(participant_list) > 1:

            fig, ax1 = plt.subplots(1,1, figsize=(15,5), dpi=100, facecolor='w', edgecolor='k')

            v1 = sns.violinplot(data = avg_roi_df.groupby(['sj', 'ROI', 'model', 'iterative']).mean().reset_index(),
                                x = 'ROI', y = 'value', 
                                order = pp_ROI_dict.keys(),
                                cut=0, inner='box', hue = plot_hue, 
                                linewidth=2.7,saturation = 1, ax = ax1) 
            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)

            plt.xlabel('ROI',fontsize = 20,labelpad=18)
            plt.ylabel('RSQ',fontsize = 20,labelpad=18)
            plt.ylim(0,1)
            fig.savefig(op.join(figures_pth, op.split(fig_name)[-1].replace('flatmap','violinplot').replace('sub-{sj}'.format(sj = pp),'sub-GROUP')))
            
    def plot_ecc_size(self, participant_list = [], group_estimates = {}, ses = 'mean',  run_type = 'mean',
                        figures_pth = None, model_name = 'gauss', n_bins_dist = 8, 
                        vmin1 = {'ecc': 0, 'size': 0}, vmax1 = {'ecc': 5.5, 'size': 17}):
        
        ## make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.figures_pth, 'ecc_size', self.pRFModelObj.fitfolder['pRF'])
            
        # dataframe to store binned values
        avg_bin_df = pd.DataFrame()
        
        ## loop over participants in list
        for pp in participant_list:

            ## load ROI dict for participant
            pp_ROI_dict = self.load_ROIs_dict(sub_id = pp)

            # make path to save sub-specific figures
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp))
            os.makedirs(sub_figures_pth, exist_ok=True)
                
            ## use RSQ as alpha level for flatmaps
            r2 = group_estimates['sub-{sj}'.format(sj = pp)]['r2']
            alpha_level = np.ones(r2.shape[0]) #self.MRIObj.mri_utils.normalize(np.clip(r2, 0, .6)) # normalize 
            alpha_level[np.where((np.isnan(r2)))[0]] = np.nan
                
            ## get ECCENTRICITY estimates
            eccentricity = self.pRFModelObj.get_eccentricity(xx = group_estimates['sub-{sj}'.format(sj = pp)]['x'],
                                                             yy = group_estimates['sub-{sj}'.format(sj = pp)]['y'],
                                                             rsq = r2)
            
            ## plot rsq values on flatmap surface ##
            fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_task-{tsk}_acq-{acq}_space-{space}_ses-{ses}_run-{run}_model-{model}_flatmap_ECC.png'.format(sj=pp, tsk = 'pRF',
                                                                                                            acq = self.MRIObj.acq, space = self.MRIObj.sj_space,
                                                                                                            ses=ses, run = run_type, model = model_name))

            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            self.plot_utils.plot_flatmap(eccentricity, 
                                        pysub = self.get_pysub_name(sub_id = pp), cmap = 'viridis', 
                                        vmin1 = vmin1['ecc'], vmax1 = vmax1['ecc'],
                                        est_arr2 = alpha_level,
                                        vmin2 = 0, vmax2 = 1, 
                                        fig_abs_name = fig_name)
            
            ## get SIZE estimates 
            size_fwhmaxmin = self.pRFModelObj.fwhmax_fwatmin(model_name, 
                                                                group_estimates['sub-{sj}'.format(sj = pp)])
            
            self.plot_utils.plot_flatmap(size_fwhmaxmin[0], 
                                        pysub = self.get_pysub_name(sub_id = pp), cmap = 'cubehelix', 
                                        vmin1 = vmin1['size'], vmax1 = vmax1['size'],
                                        est_arr2 = alpha_level,
                                        vmin2 = 0, vmax2 = 1, 
                                        fig_abs_name = fig_name.replace('ECC', 'SIZE-fwhmax'))
            
            # also plot just size estimate
            self.plot_utils.plot_flatmap(group_estimates['sub-{sj}'.format(sj = pp)]['size'], 
                                        pysub = self.get_pysub_name(sub_id = pp), cmap = 'cubehelix', 
                                        vmin1 = 0, vmax1 = 6,
                                        est_arr2 = alpha_level,
                                        vmin2 = 0, vmax2 = 1, 
                                        fig_abs_name = fig_name.replace('ECC', 'SIZE'))
            
            ## GET values per ROI ##
            ecc_pp_roi_df = self.MRIObj.mri_utils.get_estimates_roi_df(pp, eccentricity, 
                                                                    ROIs_dict = pp_ROI_dict, 
                                                                    model = model_name)
    
            size_fwhm_pp_roi_df = self.MRIObj.mri_utils.get_estimates_roi_df(pp, size_fwhmaxmin[0], 
                                                                    ROIs_dict = pp_ROI_dict, 
                                                                    model = model_name)
            
            size_pp_roi_df = self.MRIObj.mri_utils.get_estimates_roi_df(pp, group_estimates['sub-{sj}'.format(sj = pp)]['size'], 
                                                                    ROIs_dict = pp_ROI_dict, 
                                                                    model = model_name)
            
            rsq_pp_roi_df = self.MRIObj.mri_utils.get_estimates_roi_df(pp, r2, 
                                                                    ROIs_dict = pp_ROI_dict, 
                                                                    model = model_name)

            # merge them into one
            df_ecc_siz = pd.merge(ecc_pp_roi_df.rename(columns={'value': 'ecc'}),
                                size_pp_roi_df.rename(columns={'value': 'size'}))
            df_ecc_siz = pd.merge(df_ecc_siz, rsq_pp_roi_df.rename(columns={'value': 'rsq'}))
            df_ecc_siz = pd.merge(df_ecc_siz, size_fwhm_pp_roi_df.rename(columns={'value': 'size_fwhm'}))

            ## drop the nans
            df_ecc_siz = df_ecc_siz[~np.isnan(df_ecc_siz.rsq)]

            ##### plot unbinned df - SIZE #########
            sns.set(font_scale=1.3)
            sns.set_style("ticks")

            g = sns.lmplot(x="ecc", y="size", hue = 'ROI', data = df_ecc_siz, scatter_kws={'alpha':0.05},
                        scatter=True, palette = self.ROI_pallete) #, markers=['^', 's', 'o', 'v', 'D', 'h', 'P', '.', ','])

            ax = plt.gca()
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            ax.axes.set_xlim(vmin1['ecc'], vmax1['ecc'])
            #ax.axes.set_ylim(vmin1['size'], 6)
            ax.set_xlabel('pRF eccentricity [deg]', fontsize = 20, labelpad = 15)
            ax.set_ylabel('pRF size [deg]', fontsize = 20, labelpad = 15)
            sns.despine(offset=15)
            # to make legend full alpha
            for lh in g._legend.legendHandles: 
                lh.set_alpha(1)
            fig2 = plt.gcf()

            fig_name = fig_name.replace('ECC', 'ECCvsSIZE_UNbinned')
            fig2.savefig(fig_name, dpi=100,bbox_inches = 'tight')

            ##### plot unbinned df - SIZE FWHM #########
            sns.set(font_scale=1.3)
            sns.set_style("ticks")

            g = sns.lmplot(x="ecc", y="size_fwhm", hue = 'ROI', data = df_ecc_siz, scatter_kws={'alpha':0.05},
                        scatter=True, palette = self.ROI_pallete) #, markers=['^', 's', 'o', 'v', 'D', 'h', 'P', '.', ','])

            ax = plt.gca()
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            ax.axes.set_xlim(vmin1['ecc'], vmax1['ecc'])
            #ax.axes.set_ylim(vmin1['size'], vmax1['size'])
            ax.set_xlabel('pRF eccentricity [deg]', fontsize = 20, labelpad = 15)
            ax.set_ylabel('pRF size FWHMax [deg]', fontsize = 20, labelpad = 15)
            sns.despine(offset=15)
            # to make legend full alpha
            for lh in g._legend.legendHandles: 
                lh.set_alpha(1)
            fig2 = plt.gcf()

            fig_name = fig_name.replace('SIZE', 'SIZE_FWHM')
            fig2.savefig(fig_name, dpi=100,bbox_inches = 'tight')

            ## bin it, for cleaner plot
            for r_name in pp_ROI_dict.keys()   :

                mean_x, _, mean_y, _ = self.MRIObj.mri_utils.get_weighted_bins (df_ecc_siz.loc[(df_ecc_siz['ROI'] == r_name)],
                                                                                x_key = 'ecc', y_key = 'size_fwhm', 
                                                                                weight_key = 'rsq', sort_key = 'ecc', n_bins = n_bins_dist)

                avg_bin_df = pd.concat((avg_bin_df,
                                        pd.DataFrame({ 'sj': np.tile('sub-{sj}'.format(sj = pp), len(mean_x)),
                                                    'ROI': np.tile(r_name, len(mean_x)),
                                                    'ecc': mean_x,
                                                    'size_fwhm': mean_y
                                        })))

            ##### plot binned df #########
            sns.set(font_scale=1.3)
            sns.set_style("ticks")

            g = sns.lmplot(x="ecc", y="size_fwhm", hue = 'ROI', data = avg_bin_df.loc[avg_bin_df['sj'] == 'sub-{sj}'.format(sj = pp)], 
                           scatter_kws={'alpha':0.15}, scatter=True, palette = self.ROI_pallete) #, markers=['^', 's', 'o', 'v', 'D', 'h', 'P', '.', ','])

            ax = plt.gca()
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            ax.axes.set_xlim(vmin1['ecc'], vmax1['ecc'])
            ax.axes.set_ylim(vmin1['size'], vmax1['size'])
            ax.set_xlabel('pRF eccentricity [deg]', fontsize = 20, labelpad = 15)
            ax.set_ylabel('pRF size FWHMax [deg]', fontsize = 20, labelpad = 15)
            sns.despine(offset=15)
            # to make legend full alpha
            for lh in g._legend.legendHandles: 
                lh.set_alpha(1)
            fig2 = plt.gcf()

            fig_name = fig_name.replace('_UNbinned', '_binned')
            fig2.savefig(fig_name, dpi=100,bbox_inches = 'tight')

        if len(participant_list) > 1:

            ##### plot binned df for GROUP #########
            sns.set(font_scale=1.3)
            sns.set_style("ticks")

            g = sns.lmplot(x="ecc", y="size_fwhm", hue = 'ROI', data = avg_bin_df, 
                        scatter=True, palette = self.ROI_pallete,  
                        x_bins = n_bins_dist) #, markers=['^', 's', 'o', 'v', 'D', 'h', 'P', '.', ','])

            ax = plt.gca()
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            ax.axes.set_xlim(vmin1['ecc'], vmax1['ecc'])
            ax.axes.set_ylim(vmin1['size'], vmax1['size'])
            ax.set_xlabel('pRF eccentricity [deg]', fontsize = 20, labelpad = 15)
            ax.set_ylabel('pRF size FWHMax [deg]', fontsize = 20, labelpad = 15)
            sns.despine(offset=15)
            # to make legend full alpha
            for lh in g._legend.legendHandles: 
                lh.set_alpha(1)
            fig2 = plt.gcf()

            fig2.savefig(op.join(figures_pth, op.split(fig_name)[-1].replace('sub-{sj}'.format(sj = pp),'sub-GROUP')),
                            dpi=100,bbox_inches = 'tight')

    def plot_exponent(self, participant_list = [], group_estimates = {}, ses = 'mean',  run_type = 'mean',
                        figures_pth = None, model_name = 'gauss', vmin1 = 0, vmax1 = 1):
        
        ## make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.figures_pth, 'exponent', self.pRFModelObj.fitfolder['pRF'])
            
        # save values per roi in dataframe
        avg_roi_df = pd.DataFrame()
        
        ## loop over participants in list
        for pp in participant_list:

            ## load ROI dict for participant
            pp_ROI_dict = self.load_ROIs_dict(sub_id = pp)

            # make path to save sub-specific figures
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp))
            os.makedirs(sub_figures_pth, exist_ok=True)

            ## plot rsq values on flatmap surface ##
            fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_task-{tsk}_acq-{acq}_space-{space}_ses-{ses}_run-{run}_model-{model}_flatmap_Exponent.png'.format(sj=pp, tsk = 'pRF',
                                                                                                            acq = self.MRIObj.acq, space = self.MRIObj.sj_space,
                                                                                                            ses=ses, run = run_type, model = model_name))

            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            ## use RSQ as alpha level for flatmaps
            r2 = group_estimates['sub-{sj}'.format(sj = pp)]['r2']
            alpha_level = np.ones(r2.shape[0]) #self.MRIObj.mri_utils.normalize(np.clip(r2, 0, .6)) # normalize 
            alpha_level[np.where((np.isnan(r2)))[0]] = np.nan

            self.plot_utils.plot_flatmap(group_estimates['sub-{sj}'.format(sj = pp)]['ns'], 
                                        pysub = self.get_pysub_name(sub_id = pp), cmap = 'magma', 
                                        vmin1 = vmin1, vmax1 = vmax1, 
                                        est_arr2 = alpha_level,
                                        vmin2 = 0, vmax2 = 1, 
                                        fig_abs_name = fig_name)
            
            ## GET values per ROI ##
            ns_pp_roi_df = self.MRIObj.mri_utils.get_estimates_roi_df(pp, group_estimates['sub-{sj}'.format(sj = pp)]['ns'], 
                                                                    ROIs_dict = pp_ROI_dict, 
                                                                    model = model_name)
    
            rsq_pp_roi_df = self.MRIObj.mri_utils.get_estimates_roi_df(pp, r2, 
                                                                    ROIs_dict = pp_ROI_dict, 
                                                                    model = model_name)

            # merge them into one
            df_ns = pd.merge(ns_pp_roi_df.rename(columns={'value': 'exponent'}),
                            rsq_pp_roi_df.rename(columns={'value': 'rsq'}))

            ## drop the nans
            df_ns = df_ns[~np.isnan(df_ns.rsq)]

            #### plot distribution ###
            fig, ax1 = plt.subplots(1,1, figsize=(20,7.5), dpi=100, facecolor='w', edgecolor='k')

            v1 = pt.RainCloud(data = df_ns, move = .2, alpha = .9,
                        x = 'ROI', y = 'exponent', pointplot = False, hue = 'ROI',
                        palette = self.ROI_pallete, ax = ax1)
            
            # quick fix for legen
            handles = [mpatches.Patch(color = self.ROI_pallete[k], label = k) for k in pp_ROI_dict.keys()]
            ax1.legend(loc = 'upper right',fontsize=8, handles = handles, title="ROIs")#, fancybox=True)

            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)

            plt.xlabel('ROI',fontsize = 20,labelpad=18)
            plt.ylabel('Exponent',fontsize = 20,labelpad=18)
            plt.ylim(0,1)
            fig.savefig(fig_name.replace('flatmap','violinplot'))

            ## concatenate average per participant, to make group plot
            avg_roi_df = pd.concat((avg_roi_df, df_ns))
            
        # if we provided several participants, make group plot
        if len(participant_list) > 1:

            fig, ax1 = plt.subplots(1,1, figsize=(15,5), dpi=100, facecolor='w', edgecolor='k')

            v1 = sns.pointplot(data = avg_roi_df.groupby(['sj', 'ROI'])['exponent'].mean().reset_index(),
                                x = 'ROI', y = 'exponent', color = 'k', markers = 'D', #scale = 1, 
                                palette = self.ROI_pallete, order = pp_ROI_dict.keys(), 
                                dodge = False, join = False, ci=68, ax = ax1)
            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            sns.stripplot(data = avg_roi_df.groupby(['sj', 'ROI'])['exponent'].mean().reset_index(), 
                          x = 'ROI', y = 'exponent', #hue = 'sj', palette = sns.color_palette("husl", len(participant_list)),
                            order = pp_ROI_dict.keys(),
                            color="gray", alpha=0.5, ax=ax1)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)

            plt.xlabel('ROI',fontsize = 20,labelpad=18)
            plt.ylabel('Exponent',fontsize = 20,labelpad=18)
            plt.ylim(0,1)

            fig.savefig(op.join(figures_pth, op.split(fig_name)[-1].replace('flatmap','violinplot').replace('sub-{sj}'.format(sj = pp),'sub-GROUP')))

    def plot_pa(self, participant_list = [], group_estimates = {}, ses = 'mean',  run_type = 'mean',
                        figures_pth = None, model_name = 'gauss', n_bins_colors = 256, max_x_lim = 5.5, angle_thresh = 3*np.pi/4,
                        colormap_list = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb', '#3d549f','#655099','#ad5a9b','#dd3933']):
        
        ## make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.figures_pth, 'polar_angle', self.pRFModelObj.fitfolder['pRF'])
            
        # get matplotlib color map from segmented colors
        PA_cmap = self.plot_utils.make_colormap(colormap_list, bins = n_bins_colors, 
                                                    cmap_name = 'PA_mackey_custom',
                                                    discrete = False, add_alpha = False, return_cmap = True)

        ## loop over participants in list
        for pp in participant_list:

            # make path to save sub-specific figures
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp))
            os.makedirs(sub_figures_pth, exist_ok=True)
                
            ## use RSQ as alpha level for flatmaps
            r2 = group_estimates['sub-{sj}'.format(sj = pp)]['r2']
            alpha_level = np.ones(r2.shape[0]) #self.MRIObj.mri_utils.normalize(np.clip(r2, 0, .6)) # normalize 
            alpha_level[np.where((np.isnan(r2)))[0]] = np.nan

            ## position estimates
            xx = group_estimates['sub-{sj}'.format(sj = pp)]['x']
            yy = group_estimates['sub-{sj}'.format(sj = pp)]['y']

            ## calculate polar angle (normalize PA between 0 and 1)
            polar_angle_norm = self.pRFModelObj.get_polar_angle(xx = xx, yy = yy, rsq = r2, 
                                                            pa_transform = 'norm', angle_thresh = None)
               
            ## plot rsq values on flatmap surface ##
            fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_task-{tsk}_acq-{acq}_space-{space}_ses-{ses}_run-{run}_model-{model}_flatmap_PA.png'.format(sj=pp, tsk = 'pRF',
                                                                                                            acq = self.MRIObj.acq, space = self.MRIObj.sj_space,
                                                                                                            ses=ses, run = run_type, model = model_name))

            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            self.plot_utils.plot_flatmap(polar_angle_norm, 
                                        pysub = self.get_pysub_name(sub_id = pp), cmap = PA_cmap, 
                                        vmin1 = 0, vmax1 = 1,
                                        est_arr2 = alpha_level,
                                        vmin2 = 0, vmax2 = 1, 
                                        with_colorbar = False,
                                        fig_abs_name = fig_name)
            
            ## also plot non-uniform color wheel ##
            cmap_pa_sns = sns.hls_palette(as_cmap=True, h = 0.01, s=.9, l=.65)
            pa_transformed = self.pRFModelObj.get_polar_angle(xx = xx, yy = yy, rsq = r2, 
                                                            pa_transform = 'flip', angle_thresh = angle_thresh)

            self.plot_utils.plot_flatmap(pa_transformed, 
                                        pysub = self.get_pysub_name(sub_id = pp), cmap = cmap_pa_sns, 
                                        vmin1 = -angle_thresh, vmax1 = angle_thresh, 
                                        est_arr2 = alpha_level,
                                        vmin2 = 0, vmax2 = 1, 
                                        with_colorbar = False,
                                        fig_abs_name = fig_name.replace('_PA', '_PAnonUNI'))

            # plot x and y separately, for sanity check
            # XX
            self.plot_utils.plot_flatmap(xx, 
                                        pysub = self.get_pysub_name(sub_id = pp), cmap = 'BuBkRd_alpha_2D', 
                                        vmin1 = -max_x_lim, vmax1 = max_x_lim, 
                                        est_arr2 = alpha_level,
                                        vmin2 = 0, vmax2 = 1, 
                                        fig_abs_name = fig_name.replace('_PA', '_XX'))
            # YY
            self.plot_utils.plot_flatmap(yy, 
                                        pysub = self.get_pysub_name(sub_id = pp), cmap = 'BuBkRd_alpha_2D', 
                                        vmin1 = -max_x_lim, vmax1 = max_x_lim, 
                                        est_arr2 = alpha_level,
                                        vmin2 = 0, vmax2 = 1, 
                                        fig_abs_name = fig_name.replace('_PA', '_YY'))

        ## plot the colorwheels as figs
            
        # non uniform colorwheel
        self.plot_utils.plot_pa_colorwheel(resolution=800, angle_thresh = angle_thresh, cmap_name = cmap_pa_sns, 
                                        continuous = True, fig_name = op.join(figures_pth, 'hsv_seaborn'))

        # uniform colorwheel, continuous
        self.plot_utils.plot_pa_colorwheel(resolution=800, angle_thresh = np.pi, 
                                        cmap_name = colormap_list, 
                                        continuous = True, fig_name = op.join(figures_pth, 'PA_mackey'))

        # uniform colorwheel, discrete
        self.plot_utils.plot_pa_colorwheel(resolution=800, angle_thresh = np.pi, 
                                        cmap_name = colormap_list, 
                                        continuous = False, fig_name = op.join(figures_pth, 'PA_mackey'))

    def plot_VFcoverage(self, participant_list = [], group_estimates = {}, ses = 'mean',  run_type = 'mean',
                        figures_pth = None, model_name = 'gauss', max_ecc_ext_dict = {}):

        """
        Plot visual field coverage - hexbins for each ROI
        for all participants in list
        """

        ## make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.figures_pth, 'VF_coverage', self.pRFModelObj.fitfolder['pRF'])
            
        hemi_labels = ['LH', 'RH']

        # save values per roi in dataframe
        df_merge = pd.DataFrame()

        ## loop over participants in list
        for pp in participant_list:

            ## load ROI dict for participant, for each hemisphere
            pp_ROI_dict = {}
            pp_ROI_dict['LH'] = self.load_ROIs_dict(sub_id = pp, hemisphere = 'LH')
            pp_ROI_dict['RH'] = self.load_ROIs_dict(sub_id = pp, hemisphere = 'RH')

            # make path to save sub-specific figures
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp))
            os.makedirs(sub_figures_pth, exist_ok=True)

            df_hemi = pd.DataFrame()

            for hemi in hemi_labels:
                ## GET values per ROI ##
                xx_pp_roi_df = self.MRIObj.mri_utils.get_estimates_roi_df(pp, group_estimates['sub-{sj}'.format(sj = pp)]['x'], 
                                                                        ROIs_dict = pp_ROI_dict[hemi], 
                                                                        model = model_name)
                yy_pp_roi_df = self.MRIObj.mri_utils.get_estimates_roi_df(pp, group_estimates['sub-{sj}'.format(sj = pp)]['y'], 
                                                                        ROIs_dict = pp_ROI_dict[hemi], 
                                                                        model = model_name)

                tmp_df = pd.merge(xx_pp_roi_df.rename(columns={'value': 'xx'}),
                                  yy_pp_roi_df.rename(columns={'value': 'yy'}))
                tmp_df['hemisphere'] = hemi

                df_hemi = pd.concat((df_hemi, tmp_df))

            ## save in merged DF
            df_merge = pd.concat((df_merge, df_hemi))

            ## plot rsq values on flatmap surface ##
            fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_task-{tsk}_acq-{acq}_space-{space}_ses-{ses}_run-{run}_model-{model}_VFcoverage.png'.format(sj=pp, tsk = 'pRF',
                                                                                                            acq = self.MRIObj.acq, space = self.MRIObj.sj_space,
                                                                                                            ses=ses, run = run_type, model = model_name))

            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            # actually plot hexabins
            for r_name in pp_ROI_dict[hemi].keys():

                f, ss = plt.subplots(1, 1, figsize=(8,4.5))

                ss.hexbin(df_merge[(df_merge['hemisphere'] == 'LH') & \
                                    (df_merge['ROI'] == r_name) & \
                                    (df_merge['sj'] == 'sub-{sj}'.format(sj = pp))].xx.values,
                        df_merge[(df_merge['hemisphere'] == 'LH')& \
                                    (df_merge['ROI'] == r_name) & \
                                    (df_merge['sj'] == 'sub-{sj}'.format(sj = pp))].yy.values,
                    gridsize=15, 
                    cmap='Greens',
                    extent= np.array([-1, 1, -1, 1]) * max_ecc_ext_dict['sub-{sj}'.format(sj = pp)],
                    bins='log',
                    linewidths=0.0625,
                    edgecolors='black',
                    alpha=1)

                ss.hexbin(df_merge[(df_merge['hemisphere'] == 'RH')& \
                                    (df_merge['ROI'] == r_name) & \
                                    (df_merge['sj'] == 'sub-{sj}'.format(sj = pp))].xx.values,
                        df_merge[(df_merge['hemisphere'] == 'RH')& \
                                    (df_merge['ROI'] == r_name) & \
                                    (df_merge['sj'] == 'sub-{sj}'.format(sj = pp))].yy.values,
                    gridsize=15, 
                    cmap='Reds',
                    extent= np.array([-1, 1, -1, 1]) * max_ecc_ext_dict['sub-{sj}'.format(sj = pp)],
                    bins='log',
                    linewidths=0.0625,
                    edgecolors='black',
                    alpha=.5)

                plt.xticks(fontsize = 20)
                plt.yticks(fontsize = 20)
                plt.tight_layout()
                plt.xlim(-max_ecc_ext_dict['sub-{sj}'.format(sj = pp)], max_ecc_ext_dict['sub-{sj}'.format(sj = pp)]) #-6,6)#
                plt.ylim(-max_ecc_ext_dict['sub-{sj}'.format(sj = pp)], max_ecc_ext_dict['sub-{sj}'.format(sj = pp)]) #-6,6)#
                ss.set_aspect('auto')
                # set middle lines
                ss.axvline(0, -max_ecc_ext_dict['sub-{sj}'.format(sj = pp)], max_ecc_ext_dict['sub-{sj}'.format(sj = pp)], lw=0.25, color='w')
                ss.axhline(0, -max_ecc_ext_dict['sub-{sj}'.format(sj = pp)], max_ecc_ext_dict['sub-{sj}'.format(sj = pp)], lw=0.25, color='w')

                # custom lines only to make labels
                custom_lines = [Line2D([0], [0], color='g',alpha=0.5, lw=4),
                                Line2D([0], [0], color='r',alpha=0.5, lw=4)]

                plt.legend(custom_lines, hemi_labels, fontsize = 18)
                fig_hex = plt.gcf()
                fig_hex.savefig(fig_name.replace('_VFcoverage','_VFcoverage_{rn}'.format(rn = r_name)))
        

        if len(participant_list) > 1:
            
            fig_name = op.join(figures_pth, op.split(fig_name)[-1].replace('sub-{sj}'.format(sj = pp),'sub-GROUP'))

            # actually plot hexabins
            for r_name in pp_ROI_dict[hemi].keys():

                f, ss = plt.subplots(1, 1, figsize=(8,4.5))

                ss.hexbin(df_merge[(df_merge['hemisphere'] == 'LH') & \
                                    (df_merge['ROI'] == r_name)].xx.values,
                        df_merge[(df_merge['hemisphere'] == 'LH')& \
                                    (df_merge['ROI'] == r_name)].yy.values,
                    gridsize=15, 
                    cmap='Greens',
                    extent= np.array([-1, 1, -1, 1]) * max_ecc_ext_dict['sub-{sj}'.format(sj = pp)],
                    bins='log',
                    linewidths=0.0625,
                    edgecolors='black',
                    alpha=1)

                ss.hexbin(df_merge[(df_merge['hemisphere'] == 'RH')& \
                                    (df_merge['ROI'] == r_name)].xx.values,
                        df_merge[(df_merge['hemisphere'] == 'RH')& \
                                    (df_merge['ROI'] == r_name)].yy.values,
                    gridsize=15, 
                    cmap='Reds',
                    extent= np.array([-1, 1, -1, 1]) * max_ecc_ext_dict['sub-{sj}'.format(sj = pp)],
                    bins='log',
                    linewidths=0.0625,
                    edgecolors='black',
                    alpha=.5)

                plt.xticks(fontsize = 20)
                plt.yticks(fontsize = 20)
                plt.tight_layout()
                plt.xlim(-max_ecc_ext_dict['sub-{sj}'.format(sj = pp)], max_ecc_ext_dict['sub-{sj}'.format(sj = pp)]) #-6,6)#
                plt.ylim(-max_ecc_ext_dict['sub-{sj}'.format(sj = pp)], max_ecc_ext_dict['sub-{sj}'.format(sj = pp)]) #-6,6)#
                ss.set_aspect('auto')
                # set middle lines
                ss.axvline(0, -max_ecc_ext_dict['sub-{sj}'.format(sj = pp)], max_ecc_ext_dict['sub-{sj}'.format(sj = pp)], lw=0.25, color='w')
                ss.axhline(0, -max_ecc_ext_dict['sub-{sj}'.format(sj = pp)], max_ecc_ext_dict['sub-{sj}'.format(sj = pp)], lw=0.25, color='w')

                # custom lines only to make labels
                custom_lines = [Line2D([0], [0], color='g',alpha=0.5, lw=4),
                                Line2D([0], [0], color='r',alpha=0.5, lw=4)]

                plt.legend(custom_lines, hemi_labels, fontsize = 18)
                fig_hex = plt.gcf()
                fig_hex.savefig(fig_name.replace('_VFcoverage','_VFcoverage_{rn}'.format(rn = r_name)))


    def plot_singlevert_pRF(self, participant, 
                    ses = 'ses-mean', run_type = 'mean', vertex = None, ROI = None,
                    prf_model_name = 'gauss', file_ext = '_cropped_dc_psc.npy', 
                    fit_now = False, figures_pth = None):

        """

        Function to plot single vertex timecourse

        Parameters
        ----------
        participant : str
            subject ID
        task: str
            task identifier --> might not be needed if I make this func pRF timecourse specific
        ses: str
            session of input data
        run_type : str
            type of run of input data (ex: 1/mean)
        vertex: int
            vertex index 
        ROI: str
            roi name
        prf_model_name: str
            name of prf model to/that was fit
        file_ext: str
            file extension of the post processed data
        fit_now: bool
            if we want to fit the timecourse now
        figures_pth: str
            where to save the figures

        """
            
        # make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.outputdir, 'single_vertex', self.MRIObj.params['mri']['fitting']['pRF']['fit_folder'], 'sub-{sj}'.format(sj = participant), ses)
        
        os.makedirs(figures_pth, exist_ok=True)

        # if we want to fit it now
        if fit_now:
            print('Fitting estimates')
            estimates_dict, data_arr = self.pRFModelObj.fit_data(participant, self.pp_prf_models, 
                                                                    vertex = vertex, 
                                                                    run_type = run_type, ses = ses,
                                                                    model2fit = prf_model_name, xtol = 1e-2,
                                                                    file_ext = file_ext)

        else:
            print('Loading estimates')
            ## load estimates to make it easier to load later
            estimates_keys_dict, _ = self.pRFModelObj.load_pRF_model_estimates(participant,
                                                                        ses = ses, run_type = run_type, 
                                                                        model_name = prf_model_name, 
                                                                        iterative = True,
                                                                        fit_hrf = self.pRFModelObj.fit_hrf)

            # when loading, dict has key-value pairs stored,
            # need to convert it to make it in same format as when fitting on the spot
            keys = self.pRFModelObj.get_prf_estimate_keys(prf_model_name = prf_model_name)
            
            estimates_dict = {}
            estimates_dict['it_{name}'.format(name = prf_model_name)] = np.stack((estimates_keys_dict[val][vertex] for val in keys))[np.newaxis,...]

            ## load data array
            bold_filelist = self.pRFModelObj.get_bold_file_list(participant, task = 'pRF', ses = ses, file_ext = file_ext)
            data_arr = self.pRFModelObj.get_data4fitting(bold_filelist, task = 'pRF', run_type = run_type, chunk_num = None, vertex = vertex, 
                                baseline_interval = 'empty_long', ses = ses, return_filenames = False)

        ## if we fitted hrf, need to also get that from params
        ## and set model array
        
        # define spm hrf
        spm_hrf = self.pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].create_hrf(hrf_params = [1, 1, 0],
                                                                                                                    onset=self.pRFModelObj.hrf_onset)

        if self.pRFModelObj.fit_hrf:
            hrf = self.pp_prf_models[ 'sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].create_hrf(hrf_params = [1.0,
                                                                                                                                estimates_dict['it_{name}'.format(name = prf_model_name)][0][-3],
                                                                                                                                estimates_dict['it_{name}'.format(name = prf_model_name)][0][-2]],
                                                                                                                    onset=self.pRFModelObj.hrf_onset)
        
            self.pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].hrf = hrf

            model_arr = self.pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].return_prediction(*list(estimates_dict['it_{name}'.format(name = prf_model_name)][0, :-3]))
        
        else:
            self.pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].hrf = spm_hrf

            model_arr = self.pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].return_prediction(*list(estimates_dict['it_{name}'.format(name = prf_model_name)][0, :-1]))
        
        
        # get array with name of condition per TR, to plot in background
        ## get behavioral info 
        mri_beh = preproc_behdata.PreprocBeh(self.MRIObj)

        condition_per_TR = mri_utils.crop_shift_arr(mri_beh.pRF_bar_pass_all, 
                                        crop_nr = self.pRFModelObj.crop_TRs_num['pRF'], 
                                        shift = self.pRFModelObj.shift_TRs_num)

        ## actually plot

        # set figure name
        fig_name = 'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_roi-{roi}_vertex-{vert}.png'.format(sj = participant,
                                                                                                acq = self.MRIObj.acq,
                                                                                                space = self.MRIObj.sj_space,
                                                                                                run = run_type,
                                                                                                model = prf_model_name,
                                                                                                roi = str(ROI),
                                                                                                vert = str(vertex))
        if not fit_now:
            fig_name = fig_name.replace('.png', '_loaded.png')


        if self.pRFModelObj.fit_hrf:

            fig_name = fig_name.replace('.png','_withHRF.png') 

            ## also plot hrf shapes for comparison
            fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)

            time_sec = np.linspace(0,len(hrf[0]) * self.MRIObj.TR, num = len(hrf[0])) # array in seconds

            axis.plot(time_sec, spm_hrf[0],'grey',label='spm hrf')
            axis.plot(time_sec, hrf[0],'red',label='fitted hrf')
            axis.set_xlim(self.pRFModelObj.hrf_onset, 25)
            axis.legend(loc='upper right',fontsize=10) 
            axis.set_xlabel('Time (TR)',fontsize=10, labelpad=10)
            #plt.show()
            fig.savefig(op.join(figures_pth, 'HRF_model-{model}_roi-{roi}_vertex-{vert}.png'.format(model = prf_model_name,
                                                                                                    roi = str(ROI), 
                                                                                                    vert = str(vertex)))) 

        # plot data with model
        fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)

        # plot data with model
        time_sec = np.linspace(0,len(model_arr[0,...]) * self.MRIObj.TR, num = len(model_arr[0,...])) # array in seconds
            
        axis.plot(time_sec, model_arr[0,...], c = 'red', lw = 3, 
                                            label = 'model R$^2$ = %.2f'%estimates_dict['it_{name}'.format(name = prf_model_name)][0][-1], 
                                            zorder = 1)
        #axis.scatter(time_sec, data_reshape[ind_max_rsq,:], marker='v',s=15,c='k',label='data')
        axis.plot(time_sec, data_arr[0,...],'k--',label='data')
        
        axis.set_xlabel('Time (s)',fontsize = 20, labelpad = 5)
        axis.set_ylabel('BOLD signal change (%)',fontsize = 20, labelpad = 5)
        axis.set_xlim(0, len(model_arr[0,...]) * self.MRIObj.TR)
        
        axis.legend(loc='upper left',fontsize = 10) 

        # plot axis vertical bar on background to indicate stimulus display time
        for i,cond in enumerate(condition_per_TR):

            if cond in ['L-R','R-L']: # horizontal bar passes will be darker 
                plt.axvspan(i * self.MRIObj.TR, i * self.MRIObj.TR + self.MRIObj.TR, facecolor = '#8f0000', alpha=0.1)
                
            elif cond in ['U-D','D-U']: # vertical bar passes will be lighter 
                plt.axvspan(i * self.MRIObj.TR, i * self.MRIObj.TR + self.MRIObj.TR, facecolor = '#ff0000', alpha=0.1)
                
        #plt.show()
        fig.savefig(op.join(figures_pth, fig_name))

    def open_click_viewer(self, participant, task2viz = 'pRF',
                    ses = 'ses-mean', run_type = 'mean',
                    prf_model_name = 'gauss', file_ext = '_cropped_dc_psc.npy', rsq_threshold = .1):

        """

        Visualize pRF and FA estimates
        with interactive figure that shows timecourse on click

        Note - requires that we have (at least) pRF estimates saved 

        Parameters
        ----------
        participant : str
            subject ID
        task2viz: str
            task identifier 
        ses: str
            session of input data
        run_type : str
            type of run of input data (ex: 1/mean)
        prf_model_name: str
            name of prf model that was fit
        file_ext: str
            file extension of the post processed data
        rsq_threshold: float
            minimum RSQ threshold to use for figures

        """

        # general 
        n_bins_colors = 256

        ## load pRF data array
        bold_filelist = self.pRFModelObj.get_bold_file_list(participant, task = 'pRF', ses = ses, file_ext = file_ext)
        #print(bold_filelist)
        pRF_data_arr = self.pRFModelObj.get_data4fitting(bold_filelist, task = 'pRF', run_type = run_type, 
                                            baseline_interval = 'empty_long', ses = ses, return_filenames = False)

        # FA_data_arr = self.pRFModelObj.get_data4fitting(bold_filelist, task = 'FA', run_type = '1', 
        #                                     baseline_interval = 'empty', ses = ses, return_filenames = False)

        max_ecc_ext = self.pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['prf_stim'].screen_size_degrees/2

        ## Load click viewer plotted object
        click_plotter = click_viewer.visualize_on_click(self.MRIObj, pRFModelObj = self.pRFModelObj,
                                                        pRF_data = pRF_data_arr,
                                                        prf_dm = self.pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['prf_stim'].design_matrix,
                                                        pysub = self.pysub['sub-{pp}'.format(pp = participant)],
                                                        max_ecc_ext = max_ecc_ext)

        ## set figure, and also load estimates and models
        click_plotter.set_figure(participant,
                                        prf_ses = ses, prf_run_type = run_type, pRFmodel_name = prf_model_name,
                                        task2viz = task2viz)

        ## mask the estimates
        print('masking estimates')

        # get estimate keys
        keys = self.pRFModelObj.get_prf_estimate_keys(prf_model_name = prf_model_name)

        click_plotter.pp_prf_est_dict = self.pRFModelObj.mask_pRF_model_estimates(click_plotter.pp_prf_est_dict, 
                                                                    ROI = None,
                                                                    estimate_keys = keys,
                                                                    x_ecc_lim = [- max_ecc_ext, max_ecc_ext],
                                                                    y_ecc_lim = [- max_ecc_ext, max_ecc_ext],
                                                                    rsq_threshold = rsq_threshold,
                                                                    pysub = self.pysub['sub-{pp}'.format(pp = participant)]
                                                                    )

        ## calculate pa + ecc + size
        nan_mask = np.where((np.isnan(click_plotter.pp_prf_est_dict['r2'])) | (click_plotter.pp_prf_est_dict['r2'] < rsq_threshold))[0]
        
        complex_location = click_plotter.pp_prf_est_dict['x'] + click_plotter.pp_prf_est_dict['y'] * 1j # calculate eccentricity values

        polar_angle = np.angle(complex_location)
        polar_angle_norm = ((polar_angle + np.pi) / (np.pi * 2.0))
        polar_angle_norm[nan_mask] = np.nan

        eccentricity = np.abs(complex_location)
        eccentricity[nan_mask] = np.nan

        if prf_model_name in ['dn', 'dog']:
            size_fwhmax, fwatmin = plot_utils.fwhmax_fwatmin(prf_model_name, click_plotter.pp_prf_est_dict)
        else: 
            size_fwhmax = plot_utils.fwhmax_fwatmin(prf_model_name, click_plotter.pp_prf_est_dict)

        size_fwhmax[nan_mask] = np.nan

        ## make alpha mask
        alpha_level = mri_utils.normalize(np.clip(click_plotter.pp_prf_est_dict['r2'], rsq_threshold, .6)) # normalize 
        alpha_level[nan_mask] = np.nan

        ## set flatmaps ##

        ## pRF rsq
        click_plotter.images['pRF_rsq'] = plot_utils.get_flatmaps(click_plotter.pp_prf_est_dict['r2'], 
                                                                    vmin1 = 0, vmax1 = .8,
                                                                    pysub = self.pysub['sub-{pp}'.format(pp = participant)], 
                                                                    cmap = 'Reds')
        ## pRF Eccentricity

        # make costum coor map
        ecc_cmap = plot_utils.make_colormap(colormap = ['#dd3933','#f3eb53','#7cb956','#82cbdb','#3d549f'],
                                            bins = n_bins_colors, cmap_name = 'ECC_mackey_costum', 
                                            discrete = False, add_alpha = False, return_cmap = True)


        click_plotter.images['ecc'] = plot_utils.make_raw_vertex_image(eccentricity, 
                                                                            cmap = ecc_cmap, 
                                                                            vmin = 0, vmax = 6, 
                                                                            data2 = alpha_level, 
                                                                            vmin2 = 0, vmax2 = 1, 
                                                                            subject = self.pysub['sub-{pp}'.format(pp = participant)], data2D = True)

        ## pRF Size
        click_plotter.images['size_fwhmax'] = plot_utils.make_raw_vertex_image(size_fwhmax, 
                                                    cmap = 'hot', vmin = 0, vmax = 14, #7, 
                                                    data2 = alpha_level, 
                                                    vmin2 = 0, vmax2 = 1, 
                                                    subject = self.pysub['sub-{pp}'.format(pp = participant)], data2D = True)

        ## pRF Polar Angle
       
        # get matplotlib color map from segmented colors
        PA_cmap = plot_utils.make_colormap(colormap = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb',
                                                    '#3d549f','#655099','#ad5a9b','#dd3933'], bins = n_bins_colors, 
                                                    cmap_name = 'PA_mackey_costum',
                                                    discrete = False, add_alpha = False, return_cmap = True)

        click_plotter.images['PA'] = plot_utils.make_raw_vertex_image(polar_angle_norm, 
                                                    cmap = PA_cmap, vmin = 0, vmax = 1, 
                                                    data2 = alpha_level, 
                                                    vmin2 = 0, vmax2 = 1, 
                                                    subject = self.pysub['sub-{pp}'.format(pp = participant)], data2D = True)

        ## pRF Exponent 
        if prf_model_name == 'css':
            click_plotter.images['ns'] = plot_utils.get_flatmaps(click_plotter.pp_prf_est_dict['ns'], 
                                                                vmin1 = 0, vmax1 = 1,
                                                                pysub = self.pysub['sub-{pp}'.format(pp = participant)], 
                                                                cmap = 'plasma')

        
        cortex.quickshow(click_plotter.images['pRF_rsq'], fig = click_plotter.flatmap_ax,
                        with_rois = False, with_curvature = True, with_colorbar=False, 
                        with_sulci = True, with_labels = False)

        click_plotter.full_fig.canvas.mpl_connect('button_press_event', click_plotter.onclick)
        click_plotter.full_fig.canvas.mpl_connect('key_press_event', click_plotter.onkey)

        plt.show()


class FAViewer(Viewer):


    def __init__(self, MRIObj, outputdir = None, pRFModelObj = None, FAModelObj = None, pysub = 'hcp_999999', use_atlas = None):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
        pRFModelObj: pRF Model object
            object from one of the classes defined in prf_model.pRF_model
        outputdir: str
            path to save plots
        pysub: str
            basename of pycortex subject folder, where we drew all ROIs. 
        """

        # need to initialize parent class (Model), indicating output infos
        super().__init__(MRIObj, pysub = pysub, outputdir = outputdir, use_atlas = use_atlas)

        ## output path to save plots
        self.figures_pth = op.join(self.outputdir)
        os.makedirs(self.figures_pth, exist_ok=True)

        # Load pRF and FA model objects
        self.pRFModelObj = pRFModelObj
        self.FAModelObj = FAModelObj

    def plot_spcorrelations(self, participant, fig_basename = None):

        """
        Plot split half correlations used in GLM single fit
        to make noise mask 
        """

        ## path to files
        fitpath = op.join(self.FAModelObj.outputdir, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant))

        ## plot correlations and binary masks
        for task in self.MRIObj.tasks:

            corr_arr = np.load(op.join(fitpath, 'spcorrelation_task-{tsk}.npy'.format(tsk = task)))

            self.plot_utils.plot_flatmap(corr_arr, 
                                        pysub = self.get_pysub_name(sub_id = participant), cmap='hot', 
                                        vmin1 = 0, vmax1 = 1, 
                                        fig_abs_name = op.join(op.split(fig_basename)[0], 'spcorrelation_task-{tsk}_{bn}'.format(bn = op.split(fig_basename)[-1],
                                                                                              tsk = task)))

            binary_arr = np.load(op.join(fitpath, 'binary_mask_spcorrelation_task-{tsk}.npy'.format(tsk = task)))

            self.plot_utils.plot_flatmap(binary_arr, 
                                        pysub = self.get_pysub_name(sub_id = participant), cmap='hot', 
                                        vmin1 = 0, vmax1 = 1, 
                                        fig_abs_name = op.join(op.split(fig_basename)[0], 'binary_mask_spcorrelation_task-{tsk}_{bn}'.format(bn = op.split(fig_basename)[-1],
                                                                                              tsk = task)))

    def plot_glmsingle_estimates(self, participant_list = [], model_type = ['A','D'],
                                    mask_bool_df = None, stim_on_screen = [], mask_arr = True):

        """
        Plot split half correlations used in GLM single fit
        to make noise mask 
        """

        ## load pRF estimates for all participants 
        # store in dict, for ease of access
        print('Loading iterative estimates')
        group_estimates, group_prf_models = self.pRFModelObj.load_pRF_model_estimates(participant_list = participant_list,
                                                                    ses = 'mean', run_type = 'mean', 
                                                                    model_name = self.pRFModelObj.model_type['pRF'], 
                                                                    iterative = True,
                                                                    mask_bool_df = mask_bool_df, stim_on_screen = stim_on_screen,
                                                                    fit_hrf = self.pRFModelObj.fit_hrf)

        ## mask the estimates, if such is the case
        if mask_arr:
            print('masking estimates')

            # get estimate keys
            keys = self.pRFModelObj.get_prf_estimate_keys(prf_model_name = self.pRFModelObj.model_type['pRF'])

            # get screen lim for all participants
            max_ecc_ext = {'sub-{sj}'.format(sj = pp): group_prf_models['sub-{sj}'.format(sj = pp)]['ses-{s}'.format(s = 'mean')]['prf_stim'].screen_size_degrees/2 for pp in participant_list}

            final_estimates = {'sub-{sj}'.format(sj = pp): self.pRFModelObj.mask_pRF_model_estimates(group_estimates['sub-{sj}'.format(sj = pp)], 
                                                                                estimate_keys = keys,
                                                                                x_ecc_lim = np.array([- 1, 1]) * max_ecc_ext['sub-{sj}'.format(sj = pp)],
                                                                                y_ecc_lim = np.array([- 1, 1]) * max_ecc_ext['sub-{sj}'.format(sj = pp)],
                                                                                rsq_threshold = self.rsq_threshold_plot) for pp in participant_list}
        else:
            final_estimates = group_estimates

        # iterate over participant list
        for pp in participant_list:

            ## output path to save plots
            output_pth = op.join(self.figures_pth, 'glmsingle_estimates', 'sub-{sj}'.format(sj = pp))
            os.makedirs(output_pth, exist_ok=True)

            for name in model_type:
                ## load estimates dict
                estimates_dict = self.FAModelObj.load_estimates(pp, model_type = name)

                ## plot R2 on flatmap surface ##
                r2 = estimates_dict['onoffR2'] if name == 'A' else estimates_dict['R2']

                fig_name = op.join(output_pth,
                                'R2_Model-{m}_flatmap_sub-{sj}_acq-{acq}.png'.format(sj = pp, 
                                                                                    m = name,
                                                                                    acq=self.MRIObj.sj_space))
                self.plot_utils.plot_flatmap(r2, 
                                            pysub = self.get_pysub_name(sub_id = pp), cmap='hot', 
                                            vmin1 = 0, vmax1 = 50, 
                                            fig_abs_name = fig_name)
                
                ## plot average betas
                avg_betas = estimates_dict['betasmd'][...,0] if name == 'A' else np.mean(estimates_dict['betasmd'], axis = -1)

                self.plot_utils.plot_flatmap(avg_betas, 
                                            pysub = self.get_pysub_name(sub_id = pp), cmap='RdBu_r', 
                                            vmin1 = -2, vmax1 = 2, 
                                            fig_abs_name = fig_name.replace('R2_', 'Betas_'))
                
                ## plot betas with pRF threshold
                avg_betas[np.isnan(final_estimates['sub-{sj}'.format(sj = pp)]['r2'])] = np.nan

                self.plot_utils.plot_flatmap(avg_betas, 
                                            pysub = self.get_pysub_name(sub_id = pp), cmap='RdBu_r', 
                                            vmin1 = -2, vmax1 = 2, 
                                            fig_abs_name = fig_name.replace('R2_', 'Betas_pRF_'))
                
                # if not on-off model
                if name != 'A':
                    ## plot beta standard deviation, to see how much they vary
                    _, std_surf = self.FAModelObj.get_singletrial_estimates(estimate_arr = estimates_dict['betasmd'], 
                                                                            single_trl_DM = self.FAModelObj.load_single_trl_DM(pp), 
                                                                            return_std = True)
                    
                    self.plot_utils.plot_flatmap(np.mean(std_surf, axis = 0), 
                                            pysub = self.get_pysub_name(sub_id = pp), cmap='gnuplot', 
                                            vmin1 = 0, vmax1 = 1.5, 
                                            fig_abs_name = fig_name.replace('R2_', 'Betas_SD_'))

                # if full model    
                if name == 'D':
                    ## plot FracRidge
                    self.plot_utils.plot_flatmap(estimates_dict['FRACvalue'], 
                                            pysub = self.get_pysub_name(sub_id = pp), cmap='copper', 
                                            vmin1 = 0, vmax1 = 1, 
                                            fig_abs_name = fig_name.replace('R2_', 'FRACvalue_'))
                    
                    ## plot Noise pool
                    self.plot_utils.plot_flatmap(estimates_dict['noisepool'], 
                                            pysub = self.get_pysub_name(sub_id = pp), cmap='hot', 
                                            vmin1 = 0, vmax1 = 1, 
                                            fig_abs_name = fig_name.replace('R2_', 'NoisePool_'))
                    
                    ## and plot binary masks + correlations used to make noise pool
                    self.plot_spcorrelations(pp, fig_basename = fig_name.replace('R2_', ''))

    def plot_betas_2D(self, DF_betas_bar_coord = {}, ROI_list = [], orientation_bars = 'parallel_vertical',
                            max_ecc_ext = 5.5, fig_name = None, bar_color2plot = None, transpose_fig = False):

        """
        Plot model beta values (according to pRF x,y coordinates) in visual space
        for different ROIs

        Parameters
        ----------
        DF_betas_bar_coord: dataframe
            FA beta values dataframe for a participant, with relevant prf estimates (x,y,r2)
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        ROI_list: list/arr
            list with ROI names to plot
        max_ecc_ext: float
            eccentricity limit (screen) for plotting
        fig_name: str
            if given, will save plot with absolute figure name
        bar_color2plot: str
            attended bar color. if given, will plot betas for that bar color, else will average across colors
        """

        # if no ROI specified, then plot all
        if len(ROI_list) == 0:
            ROI_list = DF_betas_bar_coord.ROI.unique()

        ## for bars going left to right (vertical orientation)
        if orientation_bars == 'parallel_vertical':
            coord_list = self.FAModelObj.bar_x_coords_pix
        elif orientation_bars == 'parallel_horizontal':
            coord_list = self.FAModelObj.bar_y_coords_pix
        else:
            raise ValueError('Cross sections not implemented yet')
        
        ## if we want to plot estimates for specific bar color
        if bar_color2plot:
            DF_betas_bar_coord = DF_betas_bar_coord[DF_betas_bar_coord['attend_color'] == bar_color2plot].dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas']) # drop nans
        else:
            # average them, if we dont care
            DF_betas_bar_coord = DF_betas_bar_coord.dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas']).groupby(['prf_x_coord', 'prf_y_coord', 'prf_rsq_coord', 'Att_bar_coord', 'UAtt_bar_coord',
                                                                            'ROI', 'sj'])['betas'].mean().reset_index()

        ### now plot all combinations
        for roi_name in ROI_list:
        
            fig, axs = plt.subplots(nrows = len(coord_list), ncols=len(coord_list), figsize=(4.5 * len(coord_list),4.5 * len(coord_list)), sharex=False, sharey=False)
            
            ## make array with figure axis positions (6*6 = 36x2)
            position_matrix = np.stack((np.meshgrid(np.arange(len(coord_list)), np.arange(len(coord_list)))), axis = 2).reshape(-1,2)

            # if we want to transpose figure over diagonal
            if transpose_fig:
                position_matrix = np.array([np.flip(pair) for pair in position_matrix])

            # counter
            counter = 0

            for Att_bar_coord in coord_list:
                
                for UAtt_bar_coord in coord_list:
                                            
                    df2plot = DF_betas_bar_coord[(DF_betas_bar_coord['ROI'] == roi_name) &\
                                    (DF_betas_bar_coord['Att_bar_coord'] == Att_bar_coord) &\
                                    (DF_betas_bar_coord['UAtt_bar_coord'] == UAtt_bar_coord)]
                    
                    if not df2plot.empty: # if dataframe not empty

                        g = sns.scatterplot(x='prf_x_coord', y='prf_y_coord', hue_norm=(-2, 2),
                                    hue='betas', palette='coolwarm', s=20, linewidth=.3, legend=False, 
                                    data = df2plot, ax = axs[tuple(position_matrix[counter])])
                        g.set(xlim = np.array([- 1, 1]) * max_ecc_ext, 
                            ylim= np.array([- 1, 1]) * max_ecc_ext)
                        axs[tuple(position_matrix[counter])].axhline(y=0, c="0", lw=.3)
                        axs[tuple(position_matrix[counter])].axvline(x=0, c="0", lw=.3)
                        plt.gcf().tight_layout()
                        g.set(xlabel = 'x coordinates')
                        g.set(ylabel = 'y coordinates')
                        axs[tuple(position_matrix[counter])].tick_params(axis='both', labelsize=14)

                        # Create a Rectangle patch
                        # for unattended bar
                        unatt_rect = mpatches.Rectangle((self.convert_pix2dva(UAtt_bar_coord - self.FAModelObj.bar_width_pix[0]/2), 
                                                -self.convert_pix2dva(self.MRIObj.screen_res[1]/2)), 
                                                self.convert_pix2dva(self.FAModelObj.bar_width_pix[0]), 
                                                self.convert_pix2dva(self.MRIObj.screen_res[1]), 
                                                linewidth=1, edgecolor='k', facecolor='#969696', alpha = .15, zorder = 10)
                        axs[tuple(position_matrix[counter])].add_patch(unatt_rect) # Add the patch to the Axes
                        axs[tuple(position_matrix[counter])].patches[-1].set_hatch('///')

                        # for attended bar
                        att_rect = mpatches.Rectangle((self.convert_pix2dva(Att_bar_coord - self.FAModelObj.bar_width_pix[0]/2), 
                                                -self.convert_pix2dva(self.MRIObj.screen_res[1]/2)), 
                                                self.convert_pix2dva(self.FAModelObj.bar_width_pix[0]), 
                                                self.convert_pix2dva(self.MRIObj.screen_res[1]), 
                                                linewidth=1, edgecolor='k', facecolor='#8d9e59', alpha = .15, zorder = 10)
                        axs[tuple(position_matrix[counter])].add_patch(att_rect) # Add the patch to the Axes
                        #axs[row_ind][col_ind].patches[-1].set_hatch('*')

                        # add legend
                        handleA = mpatches.Patch(facecolor = '#8d9e59', edgecolor = 'k', label = 'target')
                        handleB= mpatches.Patch( facecolor = '#969696', edgecolor = 'k', label = 'distractor', hatch = '///')
                        leg = axs[tuple(position_matrix[counter])].legend(handles = [handleA,handleB], loc = 'upper right')
                    else:
                        axs[tuple(position_matrix[counter])].set_visible(False)
                            
                    counter +=1

                # add colorbar
                norm = plt.Normalize(-2, 2)
                sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
                sm.set_array([])
                plt.gcf().tight_layout()

                cb_ax = fig.add_axes([1,.124,.01,.754])
                cb_ax.tick_params(labelsize=15) 
                fig.colorbar(sm, orientation='vertical', cax = cb_ax)
                
            if fig_name:
                os.makedirs(op.split(fig_name)[0], exist_ok=True)
                fig.savefig(fig_name.replace('.png', '_{rn}.png'.format(rn = roi_name)), dpi = 200, bbox_inches="tight")
    
    def plot_betas_1D(self, DF_betas_bar_coord = {}, ROI_list = [], orientation_bars = 'parallel_vertical',
                            max_ecc_ext = 5.5, fig_name = None, bar_color2plot = None, bin_size = None, bin_bool = True, error_type = 'std',
                            transpose_fig = False):

        """
        Plot model beta values (according to pRF x,y coordinates) binned over 1D coordinates
        for different ROIs

        Parameters
        ----------
        DF_betas_bar_coord: dataframe
            FA beta values dataframe for a participant, with relevant prf estimates (x,y,r2)
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        ROI_list: list/arr
            list with ROI names to plot
        max_ecc_ext: float
            eccentricity limit (screen) for plotting
        fig_name: str
            if given, will save plot with absolute figure name
        bar_color2plot: str
            attended bar color. if given, will plot betas for that bar color, else will average across colors
        """

        # if no ROI specified, then plot all
        if len(ROI_list) == 0:
            ROI_list = DF_betas_bar_coord.ROI.unique()

        ## for bars going left to right (vertical orientation)
        if orientation_bars == 'parallel_vertical':
            coord_list = self.FAModelObj.bar_x_coords_pix

        elif orientation_bars == 'parallel_horizontal':
            coord_list = self.FAModelObj.bar_y_coords_pix

        else:
            raise ValueError('Cross sections not implemented yet')
        
        ## if we want to plot estimates for specific bar color
        if bar_color2plot:
            if isinstance(bar_color2plot, str):
                color_list = [bar_color2plot]
            else:
                color_list = bar_color2plot # assumes list, so will plot both
        else:
            # average them, if we dont care
            color_list = [None]

        # if no bin size given, assumes 1/3 of bar width
        if bin_size is None:
            bin_size = self.convert_pix2dva(self.FAModelObj.bar_width_pix[0]/3)
            
        DF_betas_bar_coord1D = pd.DataFrame()

        for cn in color_list:
                
                if bin_bool:
                    ### get betas binned over 1D coordinate
                    DF_betas_bar_coord1D = pd.concat((DF_betas_bar_coord1D,
                                                    self.FAModelObj.get_betas_binned1D_df(DF_betas_bar_coord = DF_betas_bar_coord, 
                                                                        ROI_list = ROI_list, orientation_bars = 'parallel_vertical', 
                                                                        max_ecc_ext = max_ecc_ext, 
                                                                        bin_size = bin_size, 
                                                                        bar_color2bin = cn)
                                                    ))
                else:
                    ### get betas over 1D coordinate (NOT BINNED)
                    DF_betas_bar_coord1D = pd.concat((DF_betas_bar_coord1D,
                                                    self.FAModelObj.get_betas_1D_df(DF_betas_bar_coord = DF_betas_bar_coord, 
                                                                        ROI_list = ROI_list, orientation_bars = 'parallel_vertical', 
                                                                        bar_color2bin = cn)
                                                    ))

        ### now plot all combinations (rows - unattended bar pos changes, column, attend bar pos changes)
        for roi_name in ROI_list:
        
            fig, axs = plt.subplots(nrows= len(coord_list), ncols=len(coord_list), figsize=(4.5 * len(coord_list), 4.5 * len(coord_list)), sharex=False, sharey=False)
            
            ## make array with figure axis positions (6*6 = 36x2)
            position_matrix = np.stack((np.meshgrid(np.arange(len(coord_list)), np.arange(len(coord_list)))), axis = 2).reshape(-1,2)

            # if we want to transpose figure over diagonal
            if transpose_fig:
                position_matrix = np.array([np.flip(pair) for pair in position_matrix])

            # counter
            counter = 0

            for Att_bar_coord in coord_list:
                
                for UAtt_bar_coord in coord_list:
                        
                    df2plot = DF_betas_bar_coord1D[(DF_betas_bar_coord1D['ROI'] == roi_name) &\
                                                (DF_betas_bar_coord1D['Att_bar_coord'] == Att_bar_coord) &\
                                                (DF_betas_bar_coord1D['UAtt_bar_coord'] == UAtt_bar_coord)]
                    
                    if not df2plot.empty: # if dataframe not empty
                    
                        df2plot.sort_values('prf_x_coord')

                        if len(color_list) > 1:
                            df1 = df2plot[df2plot['attend_color'] == color_list[0]]
                            df1.sort_values('prf_x_coord')
                            axs[tuple(position_matrix[counter])].plot(df1['prf_x_coord'], df1['betas'], 
                                                    c = self.MRIObj.params['plotting']['cond_colors'][color_list[0]], 
                                                    label = color_list[0])

                            df2 = df2plot[df2plot['attend_color'] == color_list[1]]
                            df2.sort_values('prf_x_coord')
                            axs[tuple(position_matrix[counter])].plot(df2['prf_x_coord'], df2['betas'], 
                                                    c = self.MRIObj.params['plotting']['cond_colors'][color_list[1]], 
                                                    label = color_list[1])
                            axs[tuple(position_matrix[counter])].legend()

                            if bin_bool:
                                axs[tuple(position_matrix[counter])].errorbar(df1['prf_x_coord'], df1['betas'], yerr=df1[error_type], fmt='o',
                                                            c = self.MRIObj.params['plotting']['cond_colors'][color_list[0]])
                                axs[tuple(position_matrix[counter])].errorbar(df2['prf_x_coord'], df2['betas'], yerr=df2[error_type], fmt='o',
                                                            c = self.MRIObj.params['plotting']['cond_colors'][color_list[1]])
                            else:
                                axs[tuple(position_matrix[counter])].scatter(df1['prf_x_coord'], df1['betas'], 
                                                        c = self.MRIObj.params['plotting']['cond_colors'][color_list[0]])
                                axs[tuple(position_matrix[counter])].scatter(df2['prf_x_coord'], df2['betas'], 
                                                        c = self.MRIObj.params['plotting']['cond_colors'][color_list[1]])

                        else:
                            axs[tuple(position_matrix[counter])].plot(df2plot['prf_x_coord'], df2plot['betas'], c = '#598a9e')
                            if bin_bool:
                                axs[tuple(position_matrix[counter])].errorbar(df2plot['prf_x_coord'], df2plot['betas'], yerr=df2plot[error_type], fmt='o', c = '#598a9e')
                            else:
                                axs[tuple(position_matrix[counter])].scatter(df2plot['prf_x_coord'], df2plot['betas'], c = '#598a9e')

                            # add legend
                            handleA = mpatches.Patch(facecolor = '#8d9e59', edgecolor = 'k', label = 'target')
                            handleB= mpatches.Patch( facecolor = '#969696', edgecolor = 'k', label = 'distractor', hatch = '///')
                            leg = axs[tuple(position_matrix[counter])].legend(handles = [handleA,handleB], loc = 'upper right')

                        axs[tuple(position_matrix[counter])].set_xlim(np.array([- 1, 1]) * max_ecc_ext)
                        axs[tuple(position_matrix[counter])].set_ylim(np.array([- 1.5, 5.5]))

                        axs[tuple(position_matrix[counter])].axhline(y=0, c="0", lw=.3)
                        axs[tuple(position_matrix[counter])].axvline(x=0, c="0", lw=.3)
                        plt.gcf().tight_layout()
                        axs[tuple(position_matrix[counter])].set_xlabel('x coordinates')
                        axs[tuple(position_matrix[counter])].set_ylabel('beta PSC')
                        axs[tuple(position_matrix[counter])].tick_params(axis='both', labelsize=14)

                        # Create a Rectangle patch
                        # for unattended bar
                        unatt_rect = mpatches.Rectangle((self.convert_pix2dva(UAtt_bar_coord - self.FAModelObj.bar_width_pix[0]/2), 
                                                -10), 
                                                self.convert_pix2dva(self.FAModelObj.bar_width_pix[0]), 
                                                20, 
                                                linewidth=1, edgecolor='k', facecolor='#969696', alpha = .15, zorder = 10)
                        axs[tuple(position_matrix[counter])].add_patch(unatt_rect) # Add the patch to the Axes
                        axs[tuple(position_matrix[counter])].patches[-1].set_hatch('///')

                        # for attended bar
                        att_rect = mpatches.Rectangle((self.convert_pix2dva(Att_bar_coord - self.FAModelObj.bar_width_pix[0]/2), 
                                                -10), 
                                                self.convert_pix2dva(self.FAModelObj.bar_width_pix[0]), 
                                                20, 
                                                linewidth=1, edgecolor='k', facecolor='#8d9e59', alpha = .15, zorder = 10)
                        axs[tuple(position_matrix[counter])].add_patch(att_rect) # Add the patch to the Axes
                        #axs[row_ind][col_ind].patches[-1].set_hatch('*')

                    else:
                        axs[tuple(position_matrix[counter])].set_visible(False)
                        
                    counter +=1

            if fig_name:
                os.makedirs(op.split(fig_name)[0], exist_ok=True)
                fig.savefig(fig_name.replace('.png', '_{rn}.png'.format(rn = roi_name)), dpi = 200, bbox_inches="tight")
    
    def plot_betas1D_distance(self, DF_betas_bar_coord = {}, ROI_list = [], orientation_bars = 'parallel_vertical',
                                    fig_name = None, bar_color2plot = None, avg_bool = True):

        """
        Plot model beta values (according to pRF x,y coordinates) averaged within each bar position,
        as a function of distance between bars (for different ROIs)

        Parameters
        ----------
        DF_betas_bar_coord: dataframe
            FA beta values dataframe for a participant, with relevant prf estimates (x,y,r2)
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        ROI_list: list/arr
            list with ROI names to plot
        fig_name: str
            if given, will save plot with absolute figure name
        bar_color2plot: str
            attended bar color. if given, will plot betas for that bar color, else will average across colors
        """

        # if no ROI specified, then plot all
        if len(ROI_list) == 0:
            ROI_list = DF_betas_bar_coord.ROI.unique()

        ## for bars going left to right (vertical orientation)
        if orientation_bars == 'parallel_vertical':
            coord_list = self.FAModelObj.bar_x_coords_pix
        elif orientation_bars == 'parallel_horizontal':
            coord_list = self.FAModelObj.bar_y_coords_pix
        else:
            raise ValueError('Cross sections not implemented yet')
        
        ## if we want to plot estimates for specific bar color
        if bar_color2plot:
            DF_betas_bar_coord = DF_betas_bar_coord[DF_betas_bar_coord['attend_color'] == bar_color2plot].dropna(subset=['prf_x_coord', 'prf_y_coord', 'betas']) # drop nans
        
        ## get df with average beta per position
        DF_betas_bar_avg1D = self.FAModelObj.get_betas_bar_1D_df(DF_betas_bar_coord = DF_betas_bar_coord, 
                                                                    ROI_list = ROI_list, 
                                                                    orientation_bars = orientation_bars, 
                                                                    bar_color2bin = bar_color2plot,
                                                                    avg_bool = avg_bool)

        ### now plot values for different attended bar positions
        for roi_name in ROI_list:
        
            fig, axs = plt.subplots(nrows= len(coord_list), ncols=1, figsize=(18, 7.5 * len(coord_list)), sharex=False, sharey=False)
            row_ind = 0

            for Att_bar_coord in coord_list:
                    
                df2plot = DF_betas_bar_avg1D[(DF_betas_bar_avg1D['ROI'] == roi_name) &\
                                            (DF_betas_bar_avg1D['Att_bar_coord'] == Att_bar_coord)]
                df2plot.sort_values('dist_bars')

                v1 = sns.lineplot(data = df2plot.reset_index(drop=True), 
                                x = 'dist_bars', y = 'betas', hue = 'bar_type',
                            palette = {'target': '#779e00', 'distractor': '#969696'}, ax = axs[row_ind])
                v2 = sns.scatterplot(data = df2plot.reset_index(drop=True), 
                                x = 'dist_bars', y = 'betas', hue = 'bar_type', markers = 'D',
                            palette = {'target': '#779e00', 'distractor': '#969696'}, ax = axs[row_ind])

                v1.set(xlabel=None)
                v1.set(ylabel=None)
                axs[row_ind].set_xticks(np.arange(-5,6)) 
                axs[row_ind].set_xticklabels(np.arange(-5,6))
                axs[row_ind].tick_params(axis='both', labelsize=12)

                axs[row_ind].set_xlabel('Distractor distance relative to target',fontsize = 16) #,labelpad=18)
                axs[row_ind].set_ylabel('Average beta within bar',fontsize = 16) #,labelpad=18)
                axs[row_ind].set_ylim(-1, 3)

                axs[row_ind].set_title('Attended bar x = {val} pix'.format(val = Att_bar_coord), fontsize = 20)

                # quick fix for legen
                handles = [mpatches.Patch(color = val, label = key) for key, val in {'target': '#779e00', 'distractor': '#969696'}.items()]
                axs[row_ind].legend(loc = 'upper right',fontsize=12, handles = handles, title="Bar")#, fancybox=True)

                row_ind += 1

            if fig_name:
                os.makedirs(op.split(fig_name)[0], exist_ok=True)
                fig.savefig(fig_name.replace('.png', '_{rn}.png'.format(rn = roi_name)), dpi = 200, bbox_inches="tight")

            ## also plot it grouped (so only distance, ignore bar position)
            fig, ax1 = plt.subplots(1,1, figsize=(18, 7.5))

            df2plot = DF_betas_bar_avg1D[(DF_betas_bar_avg1D['ROI'] == roi_name)]
            df2plot.sort_values('dist_bars')

            v1 = sns.pointplot(data = df2plot, 
                            x = 'dist_bars', y = 'betas', hue = 'bar_type',
                        palette = {'target': '#779e00', 'distractor': '#969696'},
                        markers = 'D', dodge = .2, join = False, ci=68, ax = ax1, n_boot=5000)

            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            sns.stripplot(data = df2plot, 
                        x = 'dist_bars', y = 'betas', hue = 'bar_type', jitter = True,
                        palette = {'target': '#8d9e59', 'distractor': '#969696'}, dodge = .2,
                        alpha=0.4, ax=ax1)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)

            plt.xlabel('Distractor distance relative to target',fontsize = 16,labelpad=18)
            plt.ylabel('Average beta within bar',fontsize = 16,labelpad=18)
            plt.ylim(0, 2.75) #(0.5, 2.5)

            # quick fix for legen
            handles = [mpatches.Patch(color = val, label = key) for key, val in {'target': '#779e00', 'distractor': '#969696'}.items()]
            ax1.legend(loc = 'upper right',fontsize=12, handles = handles, title="Bar")#, fancybox=True)

            if fig_name:
                os.makedirs(op.split(fig_name)[0], exist_ok=True)
                fig.savefig(fig_name.replace('.png', '_combined_{rn}.png'.format(rn = roi_name)), dpi = 200, bbox_inches="tight")

            ## also plot it grouped for absolute distance (ignore side of distactor bar)
            fig, ax1 = plt.subplots(1,1, figsize=(18, 7.5))

            df2plot = DF_betas_bar_avg1D[(DF_betas_bar_avg1D['ROI'] == roi_name)]
            df2plot['dist_bars'] = np.absolute(df2plot.dist_bars.values)
            df2plot.sort_values('dist_bars')

            v1 = sns.pointplot(data = df2plot, 
                            x = 'dist_bars', y = 'betas', hue = 'bar_type',
                        palette = {'target': '#779e00', 'distractor': '#969696'},
                        markers = 'D', dodge = .2, join = False, ci=68, ax = ax1, n_boot=5000)

            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            sns.stripplot(data = df2plot, 
                        x = 'dist_bars', y = 'betas', hue = 'bar_type', jitter = True,
                        palette = {'target': '#8d9e59', 'distractor': '#969696'}, dodge = .2,
                        alpha=0.4, ax=ax1)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)

            plt.xlabel('Absolute Distractor distance relative to target',fontsize = 16,labelpad=18)
            plt.ylabel('Average beta within bar',fontsize = 16,labelpad=18)
            plt.ylim(0, 2.75) #(0.5, 2.5)

            # quick fix for legen
            handles = [mpatches.Patch(color = val, label = key) for key, val in {'target': '#779e00', 'distractor': '#969696'}.items()]
            ax1.legend(loc = 'upper right',fontsize=12, handles = handles, title="Bar")#, fancybox=True)

            if fig_name:
                os.makedirs(op.split(fig_name)[0], exist_ok=True)
                fig.savefig(fig_name.replace('.png', '_collapsed_{rn}.png'.format(rn = roi_name)), dpi = 200, bbox_inches="tight")

    def plot_betas_coord(self, participant_list = [], model_type = 'D', mask_bool_df = None, stim_on_screen = [], mask_arr = True, rsq_threshold = .1,
                                att_color_ses_run_dict = {}, file_ext = '_cropped.npy', orientation_bars = 'parallel_vertical', ROI_list = ['V1']):

        """
        Plot beta estimates from GLMsingle relative to pRF coordinates
        in 2D and 1D binned plots

        Parameters
        ----------
        participant_list : list
            list of subject ID
        model_type: str
            GLMsingle model type (ex: D)
        mask_bool_df: dataframe
            if dataframe given, will be used to mask pRF design matrix (from participant behavioral performance)
        stim_on_screen: arr
            boolean array with moments where pRF stim was on screen
        mask_arr: bool
            if we want to mask pRF estimates
        rsq_threshold: float
            rsq threshold to mask pRF estimates
        att_color_ses_run_dict: dict
            dict with info for each participant, indicating session and run number for same attended color
        file_ext: str
            file extension for FA files to load
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        ROI_list: list/arr
            list with ROI names to plot
        """

        ## path to store plots
        output_pth = op.join(self.figures_pth, 'betas_coord')

        ## load pRF estimates for all participants 
        # store in dict, for ease of access
        print('Loading iterative estimates')
        group_prf_estimates, group_prf_models = self.pRFModelObj.load_pRF_model_estimates(participant_list = participant_list,
                                                                    ses = 'mean', run_type = 'mean', 
                                                                    model_name = self.pRFModelObj.model_type['pRF'], 
                                                                    iterative = True,
                                                                    mask_bool_df = mask_bool_df, stim_on_screen = stim_on_screen,
                                                                    fit_hrf = self.pRFModelObj.fit_hrf)

        ## mask the estimates, if such is the case
        if mask_arr:
            print('masking estimates')

            # get estimate keys
            keys = self.pRFModelObj.get_prf_estimate_keys(prf_model_name = self.pRFModelObj.model_type['pRF'])

            # get screen lim for all participants
            max_ecc_ext = {'sub-{sj}'.format(sj = pp): group_prf_models['sub-{sj}'.format(sj = pp)]['ses-{s}'.format(s = 'mean')]['prf_stim'].screen_size_degrees/2 for pp in participant_list}

            prf_estimates = {'sub-{sj}'.format(sj = pp): self.pRFModelObj.mask_pRF_model_estimates(group_prf_estimates['sub-{sj}'.format(sj = pp)], 
                                                                                estimate_keys = keys,
                                                                                x_ecc_lim = np.array([- 1, 1]) * max_ecc_ext['sub-{sj}'.format(sj = pp)],
                                                                                y_ecc_lim = np.array([- 1, 1]) * max_ecc_ext['sub-{sj}'.format(sj = pp)],
                                                                                rsq_threshold = rsq_threshold) for pp in participant_list}
        else:
            prf_estimates = group_prf_estimates

        # iterate over participant list
        for pp in participant_list:

            ## load ROI dict for participant
            pp_ROI_dict = self.load_ROIs_dict(sub_id = pp)

            ## output path to save plots for participants
            sub_figures_pth = op.join(output_pth, 'sub-{sj}'.format(sj = pp))
            os.makedirs(sub_figures_pth, exist_ok=True)

            ## load GLMsingle estimates dict
            GLMsing_estimates_dict = self.FAModelObj.load_estimates(pp, model_type = model_type)

            ## load single trial DM
            single_trl_DM = self.FAModelObj.load_single_trl_DM(pp)

            ## get DF with betas and coordinates
            # for vertical parallel bar positions
            DF_betas_bar_coord = self.FAModelObj.get_betas_coord_df(pp, betas_arr = GLMsing_estimates_dict['betasmd'], 
                                                                single_trl_DM = single_trl_DM, 
                                                                att_color_ses_run = att_color_ses_run_dict['sub-{sj}'.format(sj = pp)], 
                                                                file_ext = file_ext, 
                                                                ROIs_dict = pp_ROI_dict, 
                                                                prf_estimates = prf_estimates, 
                                                                orientation_bars = orientation_bars)

            ## 2D plot betas for each attended bar color separately + averaged
            for cn in ['color_red', 'color_green', None]:

                # absolute figure name
                fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_acq-{acq}_space-{space}_model-{model}_bar_orientation-{ori}_GLMsingle_betas2D.png'.format(sj=pp, acq = self.MRIObj.acq, 
                                                                                                                                space = self.MRIObj.sj_space,
                                                                                                            model = model_type, ori = orientation_bars))
                
                if cn is not None:
                    fig_name = fig_name.replace('.png', '_attend-{cn}.png'.format(cn = cn))
                
                self.plot_betas_2D(DF_betas_bar_coord = DF_betas_bar_coord, ROI_list = ROI_list, 
                                    orientation_bars = orientation_bars,
                                    max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], 
                                    bar_color2plot = cn, 
                                    transpose_fig = False,
                                    fig_name = fig_name) 
            
            ## plot betas over 1D coordinates --> all values (messy, might remove later)
            for cn in [['color_red', 'color_green'], None]:
                
                # absolute figure name
                fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_acq-{acq}_space-{space}_model-{model}_bar_orientation-{ori}_GLMsingle_betas1D.png'.format(sj=pp, acq = self.MRIObj.acq, 
                                                                                                                                       space = self.MRIObj.sj_space,
                                                                                                            model = model_type, ori = orientation_bars))
                if cn is not None:
                    fig_name = fig_name.replace('.png', '_per_color.png')

                self.plot_betas_1D(DF_betas_bar_coord = DF_betas_bar_coord, ROI_list = ROI_list, 
                                    orientation_bars = orientation_bars, 
                                    max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                    error_type = 'sem', bin_size = None, bin_bool = False,
                                    transpose_fig = False,
                                    fig_name = fig_name) 

            ## plot betas over 1D coordinates --> BINNED
            for cn in [['color_red', 'color_green'], None]:
                
                # absolute figure name
                fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_acq-{acq}_space-{space}_model-{model}_bar_orientation-{ori}_GLMsingle_betas1D_binned.png'.format(sj=pp, acq = self.MRIObj.acq, 
                                                                                                                                       space = self.MRIObj.sj_space,
                                                                                                            model = model_type, ori = orientation_bars))
                if cn is not None:
                    fig_name = fig_name.replace('.png', '_per_color.png')

                self.plot_betas_1D(DF_betas_bar_coord = DF_betas_bar_coord, ROI_list = ROI_list, 
                                    orientation_bars = orientation_bars,
                                    max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                    error_type = 'sem', bin_size = None, bin_bool = True,
                                    transpose_fig = False,
                                    fig_name = fig_name) 
                
                ## do same but when bin == BAR WIDTH
                self.plot_betas_1D(DF_betas_bar_coord = DF_betas_bar_coord, ROI_list = ROI_list, 
                                    orientation_bars = orientation_bars,
                                    max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                    error_type = 'sem', bin_size = self.convert_pix2dva(self.FAModelObj.bar_width_pix[0]), bin_bool = True,
                                    transpose_fig = False,
                                    fig_name = fig_name.replace('binned', 'binned_bar')) 
                
    
    def plot_att_coord(self, participant_list = [], model_type = 'D', mask_bool_df = None, stim_on_screen = [], mask_arr = True, rsq_threshold = .1,
                                att_color_ses_run_dict = {}, file_ext = '_cropped.npy', orientation_bars = 'parallel_vertical', ROI_list = ['V1']):

        """
        Plot attention modulation, calculated from GLMsingle beta estimates, relative to pRF coordinates
        in 2D and 1D binned plots

        Parameters
        ----------
        participant_list : list
            list of subject ID
        model_type: str
            GLMsingle model type (ex: D)
        mask_bool_df: dataframe
            if dataframe given, will be used to mask pRF design matrix (from participant behavioral performance)
        stim_on_screen: arr
            boolean array with moments where pRF stim was on screen
        mask_arr: bool
            if we want to mask pRF estimates
        rsq_threshold: float
            rsq threshold to mask pRF estimates
        att_color_ses_run_dict: dict
            dict with info for each participant, indicating session and run number for same attended color
        file_ext: str
            file extension for FA files to load
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        ROI_list: list/arr
            list with ROI names to plot
        """

        ## path to store plots
        output_pth = op.join(self.figures_pth, 'attention_coord')

        ## load pRF estimates for all participants 
        # store in dict, for ease of access
        print('Loading iterative estimates')
        group_prf_estimates, group_prf_models = self.pRFModelObj.load_pRF_model_estimates(participant_list = participant_list,
                                                                    ses = 'mean', run_type = 'mean', 
                                                                    model_name = self.pRFModelObj.model_type['pRF'], 
                                                                    iterative = True,
                                                                    mask_bool_df = mask_bool_df, stim_on_screen = stim_on_screen,
                                                                    fit_hrf = self.pRFModelObj.fit_hrf)

        ## mask the estimates, if such is the case
        if mask_arr:
            print('masking estimates')

            # get estimate keys
            keys = self.pRFModelObj.get_prf_estimate_keys(prf_model_name = self.pRFModelObj.model_type['pRF'])

            # get screen lim for all participants
            max_ecc_ext = {'sub-{sj}'.format(sj = pp): group_prf_models['sub-{sj}'.format(sj = pp)]['ses-{s}'.format(s = 'mean')]['prf_stim'].screen_size_degrees/2 for pp in participant_list}

            prf_estimates = {'sub-{sj}'.format(sj = pp): self.pRFModelObj.mask_pRF_model_estimates(group_prf_estimates['sub-{sj}'.format(sj = pp)], 
                                                                                estimate_keys = keys,
                                                                                x_ecc_lim = np.array([- 1, 1]) * max_ecc_ext['sub-{sj}'.format(sj = pp)],
                                                                                y_ecc_lim = np.array([- 1, 1]) * max_ecc_ext['sub-{sj}'.format(sj = pp)],
                                                                                rsq_threshold = rsq_threshold) for pp in participant_list}
        else:
            prf_estimates = group_prf_estimates

        # iterate over participant list
        for pp in participant_list:

            ## load ROI dict for participant
            pp_ROI_dict = self.load_ROIs_dict(sub_id = pp)

            ## output path to save plots
            sub_figures_pth = op.join(output_pth, 'sub-{sj}'.format(sj = pp))
            os.makedirs(sub_figures_pth, exist_ok=True)

            ## load GLMsingle estimates dict
            GLMsing_estimates_dict = self.FAModelObj.load_estimates(pp, model_type = model_type)

            ## load single trial DM
            single_trl_DM = self.FAModelObj.load_single_trl_DM(pp)

            ## get DF with betas and coordinates
            # for vertical parallel bar positions
            DF_betas_bar_coord = self.FAModelObj.get_betas_coord_df(pp, betas_arr = GLMsing_estimates_dict['betasmd'], 
                                                                single_trl_DM = single_trl_DM, 
                                                                att_color_ses_run = att_color_ses_run_dict['sub-{sj}'.format(sj = pp)], 
                                                                file_ext = file_ext, ROIs_dict = pp_ROI_dict, 
                                                                prf_estimates = prf_estimates, 
                                                                orientation_bars = orientation_bars)
                
            ## get attentional modulation df 
            # subtract average bar position from each trial type
            attention_coord_df = self.FAModelObj.get_attention_coord_df(DF_betas_bar_coord = DF_betas_bar_coord, 
                                                               ROI_list = ROI_list, orientation_bars = orientation_bars,
                                                               average = False)
            ## get similar df but now for UNattended modulation (so relative to distractor)
            distractor_coord_df = self.FAModelObj.get_distractor_coord_df(DF_betas_bar_coord = DF_betas_bar_coord, 
                                                               ROI_list = ROI_list, orientation_bars = orientation_bars,
                                                               average = False)
            
            ## subtraction of both
            subtract_coord_df = self.FAModelObj.get_attention_coord_flipped_df(DF_A = attention_coord_df, 
                                                                            DF_B = distractor_coord_df,
                                                                ROI_list = ROI_list, orientation_bars = orientation_bars,
                                                                average = False)
            
            ## 2D plot attentional modulation for each attended bar color separately + averaged
            for cn in ['color_red', 'color_green', None]:

                # absolute figure name
                fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_acq-{acq}_space-{space}_model-{model}_bar_orientation-{ori}_GLMsingle_TargetEffect2D.png'.format(sj=pp, acq = self.MRIObj.acq, 
                                                                                                                                space = self.MRIObj.sj_space,
                                                                                                            model = model_type, ori = orientation_bars))
                
                if cn is not None:
                    fig_name = fig_name.replace('.png', '_attend-{cn}.png'.format(cn = cn))
                
                self.plot_betas_2D(DF_betas_bar_coord = attention_coord_df, ROI_list = ROI_list, 
                                    orientation_bars = orientation_bars,
                                    max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                    transpose_fig = False,
                                    fig_name = fig_name) 
                
                ## same for distractor
                self.plot_betas_2D(DF_betas_bar_coord = distractor_coord_df, ROI_list = ROI_list, 
                                    orientation_bars = orientation_bars,
                                    max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                    transpose_fig = False,
                                    fig_name = fig_name.replace('Target', 'Distractor')) 
                
            ## plot betas over 1D coordinates
            for cn in [['color_red', 'color_green'], None]:

                # absolute figure name
                fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_acq-{acq}_space-{space}_model-{model}_bar_orientation-{ori}_GLMsingle_TargetEffect1D.png'.format(sj=pp, acq = self.MRIObj.acq, 
                                                                                                                                       space = self.MRIObj.sj_space,
                                                                                                                                       model = model_type, ori = orientation_bars))
                
                if cn is not None:
                    fig_name = fig_name.replace('.png', '_per_color.png')

                self.plot_betas_1D(DF_betas_bar_coord = attention_coord_df, ROI_list = ROI_list, 
                                    orientation_bars = orientation_bars, bin_bool = False,
                                    max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                    error_type = 'sem', bin_size = None,
                                    transpose_fig = False,
                                    fig_name = fig_name) 
                
                ## same for distractor
                self.plot_betas_1D(DF_betas_bar_coord = distractor_coord_df, ROI_list = ROI_list, 
                                    orientation_bars = orientation_bars, bin_bool = False,
                                    max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                    error_type = 'sem', bin_size = None,
                                    transpose_fig = False,
                                    fig_name = fig_name.replace('Target', 'Distractor')) 
            
            ## plot betas binned over 1D coordinates
            for cn in [['color_red', 'color_green'], None]:

                # absolute figure name
                fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_acq-{acq}_space-{space}_model-{model}_bar_orientation-{ori}_GLMsingle_TargetEffect1D_binned.png'.format(sj=pp, acq = self.MRIObj.acq, 
                                                                                                                                       space = self.MRIObj.sj_space,
                                                                                                                                       model = model_type, ori = orientation_bars))
                
                if cn is not None:
                    fig_name = fig_name.replace('.png', '_per_color.png')

                self.plot_betas_1D(DF_betas_bar_coord = attention_coord_df, ROI_list = ROI_list, 
                                    orientation_bars = orientation_bars, bin_bool = True,
                                    max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                    error_type = 'sem', bin_size = None,
                                    transpose_fig = False,
                                    fig_name = fig_name) 
                
                ## same for distractor
                self.plot_betas_1D(DF_betas_bar_coord = distractor_coord_df, ROI_list = ROI_list, 
                                    orientation_bars = orientation_bars, bin_bool = True,
                                    max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                    error_type = 'sem', bin_size = None,
                                    transpose_fig = False,
                                    fig_name = fig_name.replace('Target', 'Distractor')) 
                
                ## and for distractor transposed --> to check
                self.plot_betas_1D(DF_betas_bar_coord = distractor_coord_df, ROI_list = ROI_list, 
                                    orientation_bars = orientation_bars, bin_bool = True,
                                    max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                    error_type = 'sem', bin_size = None,
                                    transpose_fig = True,
                                    fig_name = fig_name.replace('Target', 'DistractorTransposed')) 

                ## also for subtraction of both
                self.plot_betas_1D(DF_betas_bar_coord = subtract_coord_df, ROI_list = ROI_list, 
                                    orientation_bars = orientation_bars, bin_bool = True,
                                    max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                    error_type = 'sem', bin_size = None,
                                    transpose_fig = False,
                                    fig_name = fig_name.replace('Target', 'TargetMinusDistractor')) 

                
            ## plot betas binned over 1D coordinates --> bin size = bar width
            for cn in [['color_red', 'color_green'], None]:

                # absolute figure name
                fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_acq-{acq}_space-{space}_model-{model}_bar_orientation-{ori}_GLMsingle_TargetEffect1D_binned_bar.png'.format(sj=pp, acq = self.MRIObj.acq, 
                                                                                                                                       space = self.MRIObj.sj_space,
                                                                                                                                       model = model_type, ori = orientation_bars))
                
                if cn is not None:
                    fig_name = fig_name.replace('.png', '_per_color.png')

                self.plot_betas_1D(DF_betas_bar_coord = attention_coord_df, ROI_list = ROI_list, 
                                    orientation_bars = orientation_bars, bin_bool = True,
                                    max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                    error_type = 'sem', bin_size = self.convert_pix2dva(self.FAModelObj.bar_width_pix[0]),
                                    transpose_fig = False,
                                    fig_name = fig_name) 
                
                ## same for distractor
                self.plot_betas_1D(DF_betas_bar_coord = distractor_coord_df, ROI_list = ROI_list, 
                                    orientation_bars = orientation_bars, bin_bool = True,
                                    max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                    error_type = 'sem', bin_size = self.convert_pix2dva(self.FAModelObj.bar_width_pix[0]),
                                    transpose_fig = False,
                                    fig_name = fig_name.replace('Target', 'Distractor')) 
                
                ## and for distractor transposed --> to check
                self.plot_betas_1D(DF_betas_bar_coord = distractor_coord_df, ROI_list = ROI_list, 
                                    orientation_bars = orientation_bars, bin_bool = True,
                                    max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                    error_type = 'sem', bin_size = self.convert_pix2dva(self.FAModelObj.bar_width_pix[0]),
                                    transpose_fig = True,
                                    fig_name = fig_name.replace('Target', 'DistractorTransposed')) 
                
                ## also for subtraction of both
                self.plot_betas_1D(DF_betas_bar_coord = subtract_coord_df, ROI_list = ROI_list, 
                                    orientation_bars = orientation_bars, bin_bool = True,
                                    max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                    error_type = 'sem', bin_size = self.convert_pix2dva(self.FAModelObj.bar_width_pix[0]),
                                    transpose_fig = False,
                                    fig_name = fig_name.replace('Target', 'TargetMinusDistractor')) 
                
                
    def plot_att_modulation(self, participant_list = [], model_type = 'D', mask_bool_df = None, stim_on_screen = [], mask_arr = True, rsq_threshold = .1,
                                att_color_ses_run_dict = {}, file_ext = '_cropped.npy', orientation_bars = 'parallel_vertical', ROI_list = ['V1']):

        """
        Plot attention modulation, calculated from GLMsingle beta estimates, relative to pRF coordinates
        in 2D and 1D binned plots --> from case where we subtracted betas from flipped trials

        Parameters
        ----------
        participant_list : list
            list of subject ID
        model_type: str
            GLMsingle model type (ex: D)
        mask_bool_df: dataframe
            if dataframe given, will be used to mask pRF design matrix (from participant behavioral performance)
        stim_on_screen: arr
            boolean array with moments where pRF stim was on screen
        mask_arr: bool
            if we want to mask pRF estimates
        rsq_threshold: float
            rsq threshold to mask pRF estimates
        att_color_ses_run_dict: dict
            dict with info for each participant, indicating session and run number for same attended color
        file_ext: str
            file extension for FA files to load
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        ROI_list: list/arr
            list with ROI names to plot
        """

        ## path to store plots
        output_pth = op.join(self.figures_pth, 'attention_flipped_coord')

        ## load pRF estimates for all participants 
        # store in dict, for ease of access
        print('Loading iterative estimates')
        group_prf_estimates, group_prf_models = self.pRFModelObj.load_pRF_model_estimates(participant_list = participant_list,
                                                                    ses = 'mean', run_type = 'mean', 
                                                                    model_name = self.pRFModelObj.model_type['pRF'], 
                                                                    iterative = True,
                                                                    mask_bool_df = mask_bool_df, stim_on_screen = stim_on_screen,
                                                                    fit_hrf = self.pRFModelObj.fit_hrf)

        ## mask the estimates, if such is the case
        if mask_arr:
            print('masking estimates')

            # get estimate keys
            keys = self.pRFModelObj.get_prf_estimate_keys(prf_model_name = self.pRFModelObj.model_type['pRF'])

            # get screen lim for all participants
            max_ecc_ext = {'sub-{sj}'.format(sj = pp): group_prf_models['sub-{sj}'.format(sj = pp)]['ses-{s}'.format(s = 'mean')]['prf_stim'].screen_size_degrees/2 for pp in participant_list}

            prf_estimates = {'sub-{sj}'.format(sj = pp): self.pRFModelObj.mask_pRF_model_estimates(group_prf_estimates['sub-{sj}'.format(sj = pp)], 
                                                                                estimate_keys = keys,
                                                                                x_ecc_lim = np.array([- 1, 1]) * max_ecc_ext['sub-{sj}'.format(sj = pp)],
                                                                                y_ecc_lim = np.array([- 1, 1]) * max_ecc_ext['sub-{sj}'.format(sj = pp)],
                                                                                rsq_threshold = rsq_threshold) for pp in participant_list}
        else:
            prf_estimates = group_prf_estimates

        # iterate over participant list
        for pp in participant_list:

            ## load ROI dict for participant
            pp_ROI_dict = self.load_ROIs_dict(sub_id = pp)

            ## output path to save plots
            sub_figures_pth = op.join(output_pth, 'sub-{sj}'.format(sj = pp))
            os.makedirs(sub_figures_pth, exist_ok=True)

            ## load GLMsingle estimates dict
            GLMsing_estimates_dict = self.FAModelObj.load_estimates(pp, model_type = model_type)

            ## load single trial DM
            single_trl_DM = self.FAModelObj.load_single_trl_DM(pp)

            ## get DF with betas and coordinates
            # for vertical parallel bar positions
            DF_betas_bar_coord = self.FAModelObj.get_betas_coord_df(pp, betas_arr = GLMsing_estimates_dict['betasmd'], 
                                                                single_trl_DM = single_trl_DM, 
                                                                att_color_ses_run = att_color_ses_run_dict['sub-{sj}'.format(sj = pp)], 
                                                                file_ext = file_ext, ROIs_dict = pp_ROI_dict, 
                                                                prf_estimates = prf_estimates, 
                                                                orientation_bars = orientation_bars)
                
            ## get attentional modulation df 
            # subtract average bar position from each trial type
            attention_mod_df = pd.DataFrame()

            for cn in ['color_red', 'color_green', None]:
                attention_mod_df = pd.concat((attention_mod_df,
                                              self.FAModelObj.get_betas_subtract_reverse_df(DF_betas_bar_coord = DF_betas_bar_coord, 
                                                               ROI_list = ROI_list, orientation_bars = orientation_bars,
                                                               bar_color = cn)
                                              ))
                
                if cn is None: ## to check demeaned case
                    demean_attention_mod_df = self.FAModelObj.demean_betas_df(DF_betas_bar_coord = attention_mod_df[attention_mod_df['attend_color'].isna()],
                                                                ROI_list = ROI_list, orientation_bars = orientation_bars,
                                                                bar_color = cn)
            
            ## 2D plot attentional modulation for each attended bar color separately + averaged
            for cn in ['color_red', 'color_green', None]:

                # absolute figure name
                fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_acq-{acq}_space-{space}_model-{model}_bar_orientation-{ori}_GLMsingle_BetasSubtractReverse2D.png'.format(sj=pp, acq = self.MRIObj.acq, 
                                                                                                                                space = self.MRIObj.sj_space,
                                                                                                            model = model_type, ori = orientation_bars))
                
                if cn is not None:
                    fig_name = fig_name.replace('.png', '_attend-{cn}.png'.format(cn = cn))

                ## select df for appropriate input -> should implement in function later
                if cn is None:
                    input_df = attention_mod_df[attention_mod_df['attend_color'].isna()]
                else:
                    input_df = attention_mod_df.dropna(subset=['attend_color'])
                
                ## actually plot
                self.plot_betas_2D(DF_betas_bar_coord = input_df, ROI_list = ROI_list, 
                                            orientation_bars = orientation_bars,
                                            max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], 
                                            bar_color2plot = cn, transpose_fig=False,
                                            fig_name = fig_name) 
                
            ## plot betas over 1D coordinates
            for cn in [['color_red', 'color_green'], None]:

                # absolute figure name
                fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_acq-{acq}_space-{space}_model-{model}_bar_orientation-{ori}_GLMsingle_BetasSubtractReverse1D.png'.format(sj=pp, acq = self.MRIObj.acq, 
                                                                                                                                       space = self.MRIObj.sj_space,
                                                                                                                                       model = model_type, ori = orientation_bars))
                
                if cn is not None:
                    fig_name = fig_name.replace('.png', '_per_color.png')

                ## select df for appropriate input -> should implement in function later
                if cn is None:
                    input_df = attention_mod_df[attention_mod_df['attend_color'].isna()]
                else:
                    input_df = attention_mod_df.dropna(subset=['attend_color'])
                
                ## actually plot
                self.plot_betas_1D(DF_betas_bar_coord = input_df, ROI_list = ROI_list, 
                                            orientation_bars = orientation_bars, bin_bool = False,
                                            max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                            error_type = 'sem', bin_size = None,
                                            transpose_fig=False,
                                            fig_name = fig_name) 
                
                if cn is None: ## also plot demeaned plot, to check
                    self.plot_betas_1D(DF_betas_bar_coord = demean_attention_mod_df,
                                       ROI_list = ROI_list, 
                                        orientation_bars = orientation_bars, bin_bool = False,
                                        max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                        error_type = 'sem', bin_size = None,
                                        transpose_fig=False,
                                        fig_name = fig_name.replace('.png', '_demean.png')) 
            
            ## plot betas binned over 1D coordinates
            for cn in [['color_red', 'color_green'], None]:

                # absolute figure name
                fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_acq-{acq}_space-{space}_model-{model}_bar_orientation-{ori}_GLMsingle_BetasSubtractReverse1D_binned.png'.format(sj=pp, acq = self.MRIObj.acq, 
                                                                                                                                       space = self.MRIObj.sj_space,
                                                                                                                                       model = model_type, ori = orientation_bars))
                
                if cn is not None:
                    fig_name = fig_name.replace('.png', '_per_color.png')

                 ## select df for appropriate input -> should implement in function later
                if cn is None:
                    input_df = attention_mod_df[attention_mod_df['attend_color'].isna()]
                else:
                    input_df = attention_mod_df.dropna(subset=['attend_color'])
                
                ## actually plot
                self.plot_betas_1D(DF_betas_bar_coord = input_df, ROI_list = ROI_list, 
                                            orientation_bars = orientation_bars, bin_bool = True,
                                            max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                            error_type = 'sem', bin_size = None,
                                            transpose_fig=False,
                                            fig_name = fig_name) 

                if cn is None: ## also plot demeaned plot, to check
                    self.plot_betas_1D(DF_betas_bar_coord = demean_attention_mod_df,
                                       ROI_list = ROI_list, 
                                        orientation_bars = orientation_bars, bin_bool = True,
                                        max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                        error_type = 'sem', bin_size = None,
                                        transpose_fig=False,
                                        fig_name = fig_name.replace('.png', '_demean.png')) 
                
            ## plot betas binned over 1D coordinates --> bin == bar width
            for cn in [['color_red', 'color_green'], None]:

                # absolute figure name
                fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_acq-{acq}_space-{space}_model-{model}_bar_orientation-{ori}_GLMsingle_BetasSubtractReverse1D_binned_bar.png'.format(sj=pp, acq = self.MRIObj.acq, 
                                                                                                                                       space = self.MRIObj.sj_space,
                                                                                                                                       model = model_type, ori = orientation_bars))
                
                if cn is not None:
                    fig_name = fig_name.replace('.png', '_per_color.png')

                 ## select df for appropriate input -> should implement in function later
                if cn is None:
                    input_df = attention_mod_df[attention_mod_df['attend_color'].isna()]
                else:
                    input_df = attention_mod_df.dropna(subset=['attend_color'])
                
                ## actually plot
                self.plot_betas_1D(DF_betas_bar_coord = attention_mod_df, ROI_list = ROI_list, 
                                            orientation_bars = orientation_bars, bin_bool = True,
                                            max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                            error_type = 'sem', bin_size = self.convert_pix2dva(self.FAModelObj.bar_width_pix[0]),
                                            transpose_fig=False,
                                            fig_name = fig_name) 
                
                if cn is None: ## also plot demeaned plot, to check
                    self.plot_betas_1D(DF_betas_bar_coord = demean_attention_mod_df,
                                       ROI_list = ROI_list, 
                                        orientation_bars = orientation_bars, bin_bool = True,
                                        max_ecc_ext = max_ecc_ext['sub-{sj}'.format(sj = pp)], bar_color2plot = cn, 
                                        error_type = 'sem', bin_size = self.convert_pix2dva(self.FAModelObj.bar_width_pix[0]),
                                        transpose_fig=False,
                                        fig_name = fig_name.replace('.png', '_demean.png')) 
    
    def plot_betas_bar_dist(self, participant_list = [], model_type = 'D', mask_bool_df = None, stim_on_screen = [], mask_arr = True, rsq_threshold = .1,
                                att_color_ses_run_dict = {}, file_ext = '_cropped.npy', orientation_bars = 'parallel_vertical', ROI_list = ['V1']):

        """
        Plot average GLMsingle beta estimates, for each bar,
        as a function of the distance between bars

        Parameters
        ----------
        participant_list : list
            list of subject ID
        model_type: str
            GLMsingle model type (ex: D)
        mask_bool_df: dataframe
            if dataframe given, will be used to mask pRF design matrix (from participant behavioral performance)
        stim_on_screen: arr
            boolean array with moments where pRF stim was on screen
        mask_arr: bool
            if we want to mask pRF estimates
        rsq_threshold: float
            rsq threshold to mask pRF estimates
        att_color_ses_run_dict: dict
            dict with info for each participant, indicating session and run number for same attended color
        file_ext: str
            file extension for FA files to load
        orientation_bars: str
            string with descriptor for bar orientations (crossed, parallel_vertical or parallel_horizontal)
        ROI_list: list/arr
            list with ROI names to plot
        """

        ## path to store plots
        output_pth = op.join(self.figures_pth, 'betas_bar_distance')

        ## load pRF estimates for all participants 
        # store in dict, for ease of access
        print('Loading iterative estimates')
        group_prf_estimates, group_prf_models = self.pRFModelObj.load_pRF_model_estimates(participant_list = participant_list,
                                                                    ses = 'mean', run_type = 'mean', 
                                                                    model_name = self.pRFModelObj.model_type['pRF'], 
                                                                    iterative = True,
                                                                    mask_bool_df = mask_bool_df, stim_on_screen = stim_on_screen,
                                                                    fit_hrf = self.pRFModelObj.fit_hrf)

        ## mask the estimates, if such is the case
        if mask_arr:
            print('masking estimates')

            # get estimate keys
            keys = self.pRFModelObj.get_prf_estimate_keys(prf_model_name = self.pRFModelObj.model_type['pRF'])

            # get screen lim for all participants
            max_ecc_ext = {'sub-{sj}'.format(sj = pp): group_prf_models['sub-{sj}'.format(sj = pp)]['ses-{s}'.format(s = 'mean')]['prf_stim'].screen_size_degrees/2 for pp in participant_list}

            prf_estimates = {'sub-{sj}'.format(sj = pp): self.pRFModelObj.mask_pRF_model_estimates(group_prf_estimates['sub-{sj}'.format(sj = pp)], 
                                                                                estimate_keys = keys,
                                                                                x_ecc_lim = np.array([- 1, 1]) * max_ecc_ext['sub-{sj}'.format(sj = pp)],
                                                                                y_ecc_lim = np.array([- 1, 1]) * max_ecc_ext['sub-{sj}'.format(sj = pp)],
                                                                                rsq_threshold = rsq_threshold) for pp in participant_list}
        else:
            prf_estimates = group_prf_estimates

        # iterate over participant list
        for pp in participant_list:

            ## load ROI dict for participant
            pp_ROI_dict = self.load_ROIs_dict(sub_id = pp)

            ## output path to save plots for participants
            sub_figures_pth = op.join(output_pth, 'sub-{sj}'.format(sj = pp))
            os.makedirs(sub_figures_pth, exist_ok=True)

            ## load GLMsingle estimates dict
            GLMsing_estimates_dict = self.FAModelObj.load_estimates(pp, model_type = model_type)

            ## load single trial DM
            single_trl_DM = self.FAModelObj.load_single_trl_DM(pp)

            ## get DF with betas and coordinates
            # for vertical parallel bar positions
            DF_betas_bar_coord = self.FAModelObj.get_betas_coord_df(pp, betas_arr = GLMsing_estimates_dict['betasmd'], 
                                                                single_trl_DM = single_trl_DM, 
                                                                att_color_ses_run = att_color_ses_run_dict['sub-{sj}'.format(sj = pp)], 
                                                                file_ext = file_ext, ROIs_dict = pp_ROI_dict, 
                                                                prf_estimates = prf_estimates, 
                                                                orientation_bars = orientation_bars)
            
            ## plot betas binned over 1D coordinates
            for cn in ['color_red', 'color_green', None]:
                
                # absolute figure name
                fig_name = op.join(sub_figures_pth,
                            'sub-{sj}_acq-{acq}_space-{space}_model-{model}_bar_orientation-{ori}_GLMsingle_average_betas_dist.png'.format(sj=pp, acq = self.MRIObj.acq, 
                                                                                                                                       space = self.MRIObj.sj_space,
                                                                                                            model = model_type, ori = orientation_bars))
                if cn is not None:
                    fig_name = fig_name.replace('.png', '_attend-{cn}.png'.format(cn = cn))

                self.plot_betas1D_distance(DF_betas_bar_coord = DF_betas_bar_coord, ROI_list = ROI_list, 
                                            orientation_bars = orientation_bars,
                                            bar_color2plot = cn, 
                                            avg_bool = False,
                                            fig_name = fig_name) 


    def plot_singlevert_FA(self, participant, 
                                ses = 1, run_type = '1', vertex = None, ROI = None,
                                prf_model_name = 'gauss', prf_ses = 'ses-mean', prf_run_type = 'run-mean', 
                                file_ext = '_cropped_confound_psc.npy', 
                                fit_now = False, figures_pth = None, fa_model_name = 'full_stim'):

        ## make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.outputdir, 'single_vertex', self.MRIObj.params['mri']['fitting']['FA']['fit_folder'][fa_model_name], 
                                                                'sub-{sj}'.format(sj = participant), 'ses-{s}'.format(s = ses))
        
        os.makedirs(figures_pth, exist_ok=True)

        print('Loading pRF estimates')
        pp_prf_estimates, pp_prf_models = self.pRFModelObj.load_pRF_model_estimates(participant,
                                                                    ses = prf_ses, run_type = prf_run_type,
                                                                    model_name = prf_model_name, 
                                                                    iterative = True,
                                                                    fit_hrf = self.pRFModelObj.fit_hrf)

        ## load FA data array
        bold_filelist = self.FAModelObj.get_bold_file_list(participant, task = 'FA', ses = ses, file_ext = file_ext)
        data_arr, train_file_list = self.FAModelObj.get_data4fitting(bold_filelist, task = 'FA', run_type = run_type, chunk_num = None, vertex = vertex, 
                            baseline_interval = 'empty', ses = ses, return_filenames = True)   

        ## get run and ses from file
        run_num, ses_num = mri_utils.get_run_ses_from_str(train_file_list[0])                                            
        
        ## select timecourse of single vertex
        data_arr = data_arr[0,0,...]
        
        ## Load/fit estimates for model
        # to make model timecourse

        self.FAModelObj.fit_hrf = self.pRFModelObj.fit_hrf

        if fa_model_name == 'full_stim':

            ## get bounds used for prf estimates of specific model
            pp_prf_stim = pp_prf_models['sub-{sj}'.format(sj = participant)][prf_ses]['prf_stim']
            prf_bounds = self.pRFModelObj.get_fit_startparams(max_ecc_size = pp_prf_stim.screen_size_degrees/2.0)[self.pRFModelObj.model_type['pRF']]['bounds']

            prf_pars2vary = ['betas']

            # if we want to fit it now
            if fit_now:
                print('Fitting estimates')

                ## fit data 
                results_list = self.FAModelObj.fit_data(participant, pp_prf_estimates, 
                                            ses = ses, run_type = run_type,
                                            chunk_num = None, vertex = vertex, ROI = ROI,
                                            prf_model_name = prf_model_name, file_ext = file_ext, 
                                            save_estimates = False,
                                            prf_pars2vary = prf_pars2vary, reg_name = 'full_stim', bar_keys = ['att_bar', 'unatt_bar'],
                                            xtol = 1e-3, ftol = 1e-4, n_jobs = 16, prf_bounds = prf_bounds) 

                ## get estimates dataframe
                estimates_df = results_list[0]

            else:
                ## setup fitting to get relevant variables
                _, _ = self.FAModelObj.setup_vars4fitting(participant, pp_prf_estimates, ses = ses,
                                                            run_type = run_type, chunk_num = None, vertex = vertex, ROI = ROI,
                                                            prf_model_name = prf_model_name, file_ext = file_ext, 
                                                            fit_overlap = False, fit_full_stim = True)

                ## load estimates 
                print('Loading estimates')
                estimates_df = self.FAModelObj.load_FA_model_estimates(participant, ses = ses, run = 'r{r}s{s}'.format(r = run_num, s = ses_num), 
                                                                        run_type = run_type, 
                                                                        model_name = fa_model_name, prf_model_name = prf_model_name, 
                                                                        fit_hrf = self.FAModelObj.fit_hrf, outdir = self.FAModelObj.outdir)

            ## transform estimates dataframe into dictionary
            tc_dict = estimates_df[estimates_df.vertex == vertex].to_dict('r')[0]

            ## turn parameters and bounds into arrays because of scipy minimize
            # but save dict keys to guarantee order is correct
            parameters_keys = list(tc_dict.keys())

            # set parameters and bounds into list, to conform to scipy minimze format
            tc_pars = np.array([tc_dict[key] for key in parameters_keys])
            
            ## get prediction array
            model_arr = self.FAModelObj.get_fit_timecourse(tc_pars, reg_name = 'full_stim', 
                                                            bar_keys = ['att_bar', 'unatt_bar'], parameters_keys = parameters_keys)

            model_arr = model_arr[0]

        elif fa_model_name == 'glm':

            # if we want to fit it now
            if fit_now:
                print('Fitting estimates')

                ## fit data 
                results_list = self.FAModelObj.fit_data(participant, pp_prf_estimates, 
                                            ses = ses, run_type = run_type,
                                            chunk_num = None, vertex = vertex, ROI = ROI,
                                            prf_model_name = prf_model_name, file_ext = file_ext, 
                                            save_estimates = False,
                                            fit_overlap = False, fit_full_stim = True) 

                ## get estimates dataframe
                estimates_df = results_list[0]

            else:
                ## setup fitting to get relevant variables
                _, _ = self.FAModelObj.setup_vars4fitting(participant, pp_prf_estimates, ses = ses,
                                                            run_type = run_type, chunk_num = None, vertex = vertex, ROI = ROI,
                                                            prf_model_name = prf_model_name, file_ext = file_ext, 
                                                            fit_overlap = False, fit_full_stim = True)

                ## load estimates 
                print('Loading estimates')
                estimates_df = self.FAModelObj.load_FA_model_estimates(participant, ses = ses, run = 'r{r}s{s}'.format(r = run_num, s = ses_num), 
                                                                        run_type = run_type, 
                                                                        model_name = fa_model_name, prf_model_name = prf_model_name, 
                                                                        fit_hrf = self.FAModelObj.fit_hrf, outdir = self.FAModelObj.outdir)

            ## if provided, get nuisance regressors
            if self.FAModelObj.add_nuisance_reg:

                # get ses and run number 
                #run_num, ses_num = mri_utils.get_run_ses_from_str(self.FAModelObj.train_file_list[0]) ## assumes we are fitting one run, will need to change later if such is the case

                confounds_df = self.FAModelObj.load_nuisance_df(participant, run_num = run_num, ses_num = ses_num)
            else:
                confounds_df = []
                self.nuisance_reg_names = None


            vertex_df = estimates_df[estimates_df.vertex == vertex]

            ## get DM
            design_matrix, all_regressor_names = self.FAModelObj.get_fa_glm_dm({key: vertex_df[key].values[0] for key in self.FAModelObj.prf_est_keys}, 
                                                nuisances_df = confounds_df, bar_keys = ['att_bar', 'unatt_bar'])

            ## dot design matrix with betas to get prediction timecourse
            model_arr = design_matrix.T.dot(vertex_df[all_regressor_names].values[0])


        # get rsq val for plotting
        #r2 = estimates_df[estimates_df.vertex == vertex]['r2'].values[0]
        r2 = mri_utils.calc_rsq(data_arr, model_arr)

        ## actually plot

        # set figure name
        fig_name = 'sub-{sj}_task-FA_acq-{acq}_space-{space}_run-{run}_model-{model}_pRFmodel-{pmodel}_roi-{roi}_vertex-{vert}.png'.format(sj = participant,
                                                                                                acq = self.MRIObj.acq,
                                                                                                space = self.MRIObj.sj_space,
                                                                                                run = run_type,
                                                                                                model = fa_model_name,
                                                                                                pmodel = prf_model_name,
                                                                                                roi = str(ROI),
                                                                                                vert = str(vertex))
        if not fit_now:
            fig_name = fig_name.replace('.png', '_loaded.png')

        if self.FAModelObj.fit_hrf:

            fig_name = fig_name.replace('.png','_withHRF.png') 

            ## make both hrfs 
            hrf = pp_prf_models['sub-{sj}'.format(sj = participant)][prf_ses]['%s_model'%prf_model_name].create_hrf(hrf_params = [1, estimates_df[estimates_df.vertex == vertex]['hrf_derivative'].values[0],
                                                                                                                            estimates_df[estimates_df.vertex == vertex]['hrf_dispersion'].values[0]], 
                                                                                                                            osf = self.FAModelObj.osf * self.FAModelObj.MRIObj.TR, 
                                                                                                                            onset = self.FAModelObj.hrf_onset)
            
            spm_hrf = pp_prf_models['sub-{sj}'.format(sj = participant)][prf_ses]['%s_model'%prf_model_name].create_hrf(hrf_params = [1, 1, 0], 
                                                                                                            osf = self.FAModelObj.osf * self.FAModelObj.MRIObj.TR, 
                                                                                                            onset = self.FAModelObj.hrf_onset)

            ## also plot hrf shapes for comparison
            fig, axis = plt.subplots(1,figsize=(12,5),dpi=100)

            time_sec = np.linspace(0,len(hrf[0]) * self.MRIObj.TR, num = len(hrf[0])) # array in seconds

            axis.plot(time_sec, spm_hrf[0],'grey',label='spm hrf')
            axis.plot(time_sec, hrf[0],'blue',label='fitted hrf')
            #axis.set_xlim(self.FAModelObj.hrf_onset, 25)
            axis.legend(loc='upper right',fontsize=10) 
            axis.set_xlabel('Time (s)',fontsize=10, labelpad=10)
            #plt.show()
            fig.savefig(op.join(figures_pth, 'HRF_model-{model}_pRFmodel-{pmodel}_run-{run}_roi-{roi}_vertex-{vert}.png'.format(model = fa_model_name,
                                                                                                                    pmodel = prf_model_name,
                                                                                                                    run = run_type,
                                                                                                                    roi = str(ROI), 
                                                                                                                    vert = str(vertex)))) 
        
        # plot data with model
        fig, axis = plt.subplots(1, figsize=(12,5), dpi=100)

        # plot data with model
        time_sec = np.linspace(0, len(model_arr) * self.MRIObj.TR, num = len(model_arr)) # array in seconds
            
        axis.plot(time_sec, model_arr, c = 'blue', lw = 3, 
                                            label = 'model R$^2$ = %.2f'%r2, 
                                            zorder = 1)
        #axis.scatter(time_sec, data_reshape[ind_max_rsq,:], marker='v',s=15,c='k',label='data')
        axis.plot(time_sec, data_arr, 'k--', label='data')
        
        axis.set_xlabel('Time (s)',fontsize = 20, labelpad = 5)
        axis.set_ylabel('BOLD signal change (%)',fontsize = 20, labelpad = 5)
        axis.set_xlim(0, len(model_arr) * self.MRIObj.TR)
        
        axis.legend(loc='upper left',fontsize = 10) 

        fig.savefig(op.join(figures_pth, fig_name))

    def open_click_viewer(self, participant, task2viz = 'both',
                    prf_ses = 'ses-mean', prf_run_type = 'mean', 
                    fa_ses = 1, fa_run_type = '1',
                    prf_model_name = 'gauss', fa_model_name = 'full_stim',
                    prf_file_ext = '_cropped_dc_psc.npy', fa_file_ext = '_cropped_LinDetrend_psc.npy', rsq_threshold = .1):

        """

        Visualize pRF and FA estimates
        with interactive figure that shows timecourse on click

        Note - requires that we have (at least) pRF estimates saved 

        Parameters
        ----------
        participant : str
            subject ID
        task2viz: str
            task identifier 
        ses: str
            session of input data
        run_type : str
            type of run of input data (ex: 1/mean)
        prf_model_name: str
            name of prf model that was fit
        file_ext: str
            file extension of the post processed data
        rsq_threshold: float
            minimum RSQ threshold to use for figures

        """

        # general 
        n_bins_colors = 256

        ## load pRF data array
        bold_filelist = self.pRFModelObj.get_bold_file_list(participant, task = 'pRF', ses = prf_ses, file_ext = prf_file_ext)
        #print(bold_filelist)
        pRF_data_arr = self.pRFModelObj.get_data4fitting(bold_filelist, task = 'pRF', run_type = prf_run_type, 
                                            baseline_interval = 'empty_long', ses = prf_ses, return_filenames = False)

        ## load FA data array
        bold_filelist = self.FAModelObj.get_bold_file_list(participant, task = 'FA', ses = fa_ses, file_ext = fa_file_ext)

        FA_data_arr = self.FAModelObj.get_data4fitting(bold_filelist, task = 'FA', run_type = fa_run_type,  
                                                    baseline_interval = 'empty', ses = fa_ses, return_filenames = False) 
        FA_data_arr = FA_data_arr[0] # fa data loaded in [runs, vertex, time]


        max_ecc_ext = self.pp_prf_models['sub-{sj}'.format(sj = participant)][prf_ses]['prf_stim'].screen_size_degrees/2

        ## Load click viewer plotted object
        click_plotter = click_viewer.visualize_on_click(self.MRIObj, pRFModelObj = self.pRFModelObj, FAModelObj = self.FAModelObj,
                                                        pRF_data = pRF_data_arr, FA_data = FA_data_arr,
                                                        prf_dm = self.pp_prf_models['sub-{sj}'.format(sj = participant)][prf_ses]['prf_stim'].design_matrix,
                                                        pysub = self.pysub['sub-{pp}'.format(pp = participant)],
                                                        max_ecc_ext = max_ecc_ext)

        ## set figure, and also load estimates and models -- continue from here !!!!!!!!
        click_plotter.set_figure(participant,
                                        prf_ses = prf_ses, prf_run_type = prf_run_type, pRFmodel_name = prf_model_name,
                                        fa_ses = fa_ses, fa_run_type = fa_run_type, FAmodel_name = fa_model_name,
                                        task2viz = task2viz, fa_file_ext = fa_file_ext)

        ## mask the estimates
        print('masking estimates')

        # get estimate keys
        keys = self.pRFModelObj.get_prf_estimate_keys(prf_model_name = prf_model_name)

        click_plotter.pp_prf_est_dict = self.pRFModelObj.mask_pRF_model_estimates(click_plotter.pp_prf_est_dict, 
                                                                    ROI = None,
                                                                    estimate_keys = keys,
                                                                    x_ecc_lim = [- max_ecc_ext, max_ecc_ext],
                                                                    y_ecc_lim = [- max_ecc_ext, max_ecc_ext],
                                                                    rsq_threshold = rsq_threshold,
                                                                    pysub = self.pysub['sub-{pp}'.format(pp = participant)]
                                                                    )

        ## calculate pa + ecc + size
        nan_mask = np.where((np.isnan(click_plotter.pp_prf_est_dict['r2'])) | (click_plotter.pp_prf_est_dict['r2'] < rsq_threshold))[0]
        
        complex_location = click_plotter.pp_prf_est_dict['x'] + click_plotter.pp_prf_est_dict['y'] * 1j # calculate eccentricity values

        polar_angle = np.angle(complex_location)
        polar_angle_norm = ((polar_angle + np.pi) / (np.pi * 2.0))
        polar_angle_norm[nan_mask] = np.nan

        eccentricity = np.abs(complex_location)
        eccentricity[nan_mask] = np.nan

        if prf_model_name in ['dn', 'dog']:
            size_fwhmax, fwatmin = plot_utils.fwhmax_fwatmin(prf_model_name, click_plotter.pp_prf_est_dict)
        else: 
            size_fwhmax = plot_utils.fwhmax_fwatmin(prf_model_name, click_plotter.pp_prf_est_dict)

        size_fwhmax[nan_mask] = np.nan

        ## make alpha mask
        alpha_level = mri_utils.normalize(np.clip(click_plotter.pp_prf_est_dict['r2'], rsq_threshold, .6)) # normalize 
        alpha_level[nan_mask] = np.nan

        ## set flatmaps ##

        ## pRF rsq
        click_plotter.images['pRF_rsq'] = plot_utils.get_flatmaps(click_plotter.pp_prf_est_dict['r2'], 
                                                                    vmin1 = 0, vmax1 = .8,
                                                                    pysub = self.pysub['sub-{pp}'.format(pp = participant)], 
                                                                    cmap = 'Reds')
        ## pRF Eccentricity

        # make costum coor map
        ecc_cmap = plot_utils.make_colormap(colormap = ['#dd3933','#f3eb53','#7cb956','#82cbdb','#3d549f'],
                                            bins = n_bins_colors, cmap_name = 'ECC_mackey_costum', 
                                            discrete = False, add_alpha = False, return_cmap = True)


        click_plotter.images['ecc'] = plot_utils.make_raw_vertex_image(eccentricity, 
                                                                            cmap = ecc_cmap, 
                                                                            vmin = 0, vmax = 6, 
                                                                            data2 = alpha_level, 
                                                                            vmin2 = 0, vmax2 = 1, 
                                                                            subject = self.pysub['sub-{pp}'.format(pp = participant)], data2D = True)

        ## pRF Size
        click_plotter.images['size_fwhmax'] = plot_utils.make_raw_vertex_image(size_fwhmax, 
                                                    cmap = 'hot', vmin = 0, vmax = 14, #7, 
                                                    data2 = alpha_level, 
                                                    vmin2 = 0, vmax2 = 1, 
                                                    subject = self.pysub['sub-{pp}'.format(pp = participant)], data2D = True)

        ## pRF Polar Angle
       
        # get matplotlib color map from segmented colors
        PA_cmap = plot_utils.make_colormap(colormap = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb',
                                                    '#3d549f','#655099','#ad5a9b','#dd3933'], bins = n_bins_colors, 
                                                    cmap_name = 'PA_mackey_costum',
                                                    discrete = False, add_alpha = False, return_cmap = True)

        click_plotter.images['PA'] = plot_utils.make_raw_vertex_image(polar_angle_norm, 
                                                    cmap = PA_cmap, vmin = 0, vmax = 1, 
                                                    data2 = alpha_level, 
                                                    vmin2 = 0, vmax2 = 1, 
                                                    subject = self.pysub['sub-{pp}'.format(pp = participant)], data2D = True)

        ## pRF Exponent 
        if prf_model_name == 'css':
            click_plotter.images['ns'] = plot_utils.get_flatmaps(click_plotter.pp_prf_est_dict['ns'], 
                                                                vmin1 = 0, vmax1 = 1,
                                                                pysub = self.pysub['sub-{pp}'.format(pp = participant)], 
                                                                cmap = 'plasma')

        ## FA rsq 
        r2_surf = np.zeros(FA_data_arr.shape[0]); r2_surf[:] = np.nan
        r2_surf[click_plotter.pp_fa_est_df.vertex.values] = click_plotter.pp_fa_est_df.r2.values

        if np.any(r2_surf<0): ## if there are negative rsqs (for non-glm fitting)
            click_plotter.images['FA_rsq'] = plot_utils.get_flatmaps(r2_surf, 
                                                                    vmin1 = -.5, vmax1 = .5,
                                                                    pysub = self.pysub['sub-{pp}'.format(pp = participant)], 
                                                                    cmap = 'BuBkRd')

        else:
            click_plotter.images['FA_rsq'] = plot_utils.get_flatmaps(r2_surf, 
                                                                    vmin1 = 0, vmax1 = .8,
                                                                    pysub = self.pysub['sub-{pp}'.format(pp = participant)], 
                                                                    cmap = 'Reds')
        

        
        cortex.quickshow(click_plotter.images['pRF_rsq'], fig = click_plotter.flatmap_ax,
                        with_rois = False, with_curvature = True, with_colorbar=False, 
                        with_sulci = True, with_labels = False)

        click_plotter.full_fig.canvas.mpl_connect('button_press_event', click_plotter.onclick)
        click_plotter.full_fig.canvas.mpl_connect('key_press_event', click_plotter.onkey)

        plt.show()
        

    