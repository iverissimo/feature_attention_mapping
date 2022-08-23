

import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
import utils

import ptitprince as pt # raincloud plots
import matplotlib.patches as mpatches
from  matplotlib.ticker import FuncFormatter

import cortex

import subprocess

from FAM.utils import mri as mri_utils
from FAM.utils import plot as plot_utils
from FAM.processing import preproc_behdata
from FAM.visualize import click_viewer

from PIL import Image, ImageDraw

class pRFViewer:

    def __init__(self, MRIObj, outputdir = None, pRFModelObj = None, pysub = 'hcp_999999'):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
        pRFModelObj: pRF Model object
            object from one of the classes defined in prf_model.pRF_model
            
        """

        # set data object to use later on
        self.MRIObj = MRIObj

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.MRIObj.derivatives_pth,'plots')
        else:
            self.outputdir = outputdir

        # Load pRF model object
        self.pRFModelObj = pRFModelObj
            
        # number of participants to plot
        self.nr_pp = len(self.MRIObj.sj_num)

        self.pysub = pysub

        
   
    def plot_singlevert(self, participant, task = 'pRF',
                    ses = 'ses-mean', run_type = 'mean', vertex = None, ROI = None,
                    prf_model_name = 'gauss', file_ext = '_cropped_dc_psc.npy', 
                    fit_now = False, figures_pth = None):

        """

        Function to plot single vertex timecourse

        """
        
        if task == 'pRF':
            
            # make output folder for figures
            if figures_pth is None:
                figures_pth = op.join(self.outputdir, 'single_vertex', 'pRF_fit', 'sub-{sj}'.format(sj = participant), ses)
            
            if not op.exists(figures_pth):
                os.makedirs(figures_pth)

            # get participant models, which also will load 
            # DM and mask it according to participants behavior
            pp_prf_models = self.pRFModelObj.set_models(participant_list = [participant], 
                                                        mask_DM = True, combine_ses = True)


            # if we want to fit it now
            if fit_now:
                print('Fitting estimates')
                estimates_dict, data_arr = self.pRFModelObj.fit_data(participant, pp_prf_models, 
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
                                                                            iterative = True)

                # when loading, dict has key-value pairs stored,
                # need to convert it to make it in same format as when fitting on the spot
                keys = self.MRIObj.params['mri']['fitting']['pRF']['estimate_keys'][prf_model_name]
                
                if self.pRFModelObj.fit_hrf:
                    keys = keys[:-1]+self.MRIObj.params['mri']['fitting']['pRF']['estimate_keys']['hrf']+['r2']
                
                estimates_dict = {}
                estimates_dict['it_{name}'.format(name = prf_model_name)] = np.stack((estimates_keys_dict[val][vertex] for val in keys))[np.newaxis,...]

                ## load data array
                bold_filelist = self.pRFModelObj.get_bold_file_list(participant, task = 'pRF', ses = ses, file_ext = file_ext)
                data_arr = self.pRFModelObj.get_data4fitting(bold_filelist, run_type = run_type, chunk_num = None, vertex = vertex)

            # if we fitted hrf, need to also get that from params
            # and set model array
            
            # define spm hrf
            spm_hrf = pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].create_hrf(hrf_params = [1, 1, 0],
                                                                                                                     onset=self.pRFModelObj.hrf_onset)

            if self.pRFModelObj.fit_hrf:
                hrf = pp_prf_models[ 'sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].create_hrf(hrf_params = [1.0,
                                                                                                                                 estimates_dict['it_{name}'.format(name = prf_model_name)][0][-3],
                                                                                                                                 estimates_dict['it_{name}'.format(name = prf_model_name)][0][-2]],
                                                                                                                     onset=self.pRFModelObj.hrf_onset)
            
                pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].hrf = hrf

                model_arr = pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].return_prediction(*list(estimates_dict['it_{name}'.format(name = prf_model_name)][0, :-3]))
            
            else:
                pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].hrf = spm_hrf

                model_arr = pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['{name}_model'.format(name = prf_model_name)].return_prediction(*list(estimates_dict['it_{name}'.format(name = prf_model_name)][0, :-1]))
            
            
            # get array with name of condition per TR, to plot in background
            ## get behavioral info 
            mri_beh = preproc_behdata.PreprocBeh(self.MRIObj)

            condition_per_TR = mri_utils.crop_shift_arr(mri_beh.pRF_bar_pass_all, 
                                            crop_nr = self.pRFModelObj.crop_TRs_num, 
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

                axis.plot(spm_hrf[0],'grey',label='spm hrf')
                axis.plot(hrf[0],'red',label='fitted hrf')
                axis.set_xlim(self.pRFModelObj.hrf_onset, 25)
                axis.legend(loc='upper right',fontsize=10) 
                axis.set_xlabel('Time (s)',fontsize=10, labelpad=10)
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
                    prf_model_name = 'gauss', file_ext = '_cropped_dc_psc.npy'):

        """

        visualize pRF and FA estimates
        with interactive figure 
        that shows timecourse on click

        """

        # general 
        n_bins_colors = 256

        ## load pRF data array
        bold_filelist = self.pRFModelObj.get_bold_file_list(participant, task = 'pRF', ses = ses, file_ext = file_ext)
        pRF_data_arr = self.pRFModelObj.get_data4fitting(bold_filelist, run_type = run_type)

        ## load DM
        pp_prf_models = self.pRFModelObj.set_models(participant_list = [participant], 
                                                        mask_DM = True, combine_ses = True)

        max_ecc_ext = pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['prf_stim'].screen_size_degrees/2

        ## Load click viewer plotted object
        click_plotter = click_viewer.visualize_on_click(self.MRIObj, pRFModelObj = self.pRFModelObj,
                                                        pRF_data = pRF_data_arr,
                                                        prf_dm = pp_prf_models['sub-{sj}'.format(sj = participant)][ses]['prf_stim'].design_matrix,
                                                        pysub = self.pysub,
                                                        max_ecc_ext = max_ecc_ext)

        ## set figure, and also load estimates and models
        click_plotter.set_figure(participant,
                                        ses = ses, run_type = run_type, pRFmodel_name = prf_model_name,
                                        task2viz = task2viz)

        ## mask the estimates
        print('masking estimates')

        # get estimate keys
        keys = self.MRIObj.params['mri']['fitting']['pRF']['estimate_keys'][prf_model_name]
        
        if self.pRFModelObj.fit_hrf:
            keys = keys[:-1]+self.MRIObj.params['mri']['fitting']['pRF']['estimate_keys']['hrf']+['r2']

        click_plotter.pp_prf_est_dict = self.pRFModelObj.mask_pRF_model_estimates(click_plotter.pp_prf_est_dict, 
                                                                    ROI = None,
                                                                    estimate_keys = keys,
                                                                    x_ecc_lim = [- max_ecc_ext, max_ecc_ext],
                                                                    y_ecc_lim = [- max_ecc_ext, max_ecc_ext],
                                                                    rsq_threshold = .1,
                                                                    pysub = self.pysub
                                                                    )

        ## calculate pa + ecc + size
        nan_mask = np.where(np.isnan(click_plotter.pp_prf_est_dict['r2']))[0]
        val_mask = np.where(~np.isnan(click_plotter.pp_prf_est_dict['r2']))[0]
        
        complex_location = np.zeros(click_plotter.pp_prf_est_dict['r2'].shape)
        complex_location[val_mask] = click_plotter.pp_prf_est_dict['x'][val_mask] + click_plotter.pp_prf_est_dict['y'][val_mask] * 1j # calculate eccentricity values

        polar_angle = np.angle(complex_location)
        polar_angle[nan_mask] = np.nan
        eccentricity = np.abs(complex_location)
        eccentricity[nan_mask] = np.nan

        if prf_model_name in ['dn', 'dog']:
            size_fwhmax, fwatmin = plot_utils.fwhmax_fwatmin(prf_model_name, click_plotter.pp_prf_est_dict)
        else: 
            size_fwhmax = plot_utils.fwhmax_fwatmin(prf_model_name, click_plotter.pp_prf_est_dict)

        size_fwhmax[nan_mask] = np.nan

        ## set flatmaps ##

        ## pRF rsq
        click_plotter.images['pRF_rsq'] = plot_utils.get_flatmaps(click_plotter.pp_prf_est_dict['r2'], 
                                                                    vmin1 = 0, vmax1 = .8,
                                                                    pysub = self.pysub, 
                                                                    cmap = 'Reds')
        ## pRF Eccentricity

        # make costum coor map
        ecc_cmap = plot_utils.make_colormap(colormap = ['#dd3933','#f3eb53','#7cb956','#82cbdb','#3d549f'],
                                            bins = n_bins_colors, cmap_name = 'ECC_mackey_costum', 
                                            discrete = False, add_alpha = False, return_cmap = True)


        click_plotter.images['ecc'] = plot_utils.make_raw_vertex_image(eccentricity, 
                                                                            cmap = ecc_cmap, 
                                                                            vmin = 0, vmax = 6, 
                                                                            data2 = np.ones(eccentricity.shape), 
                                                                            vmin2 = 0, vmax2 = 1, 
                                                                            subject = self.pysub, data2D = True)

        ## pRF Size
        click_plotter.images['size_fwhmax'] = plot_utils.make_raw_vertex_image(size_fwhmax, 
                                                    cmap = 'hot', vmin = 0, vmax = 7, 
                                                    data2 = np.ones(eccentricity.shape), vmin2 = 0, vmax2 = 1, 
                                                    subject = self.pysub, data2D = True)

        ## pRF Polar Angle
       
        # get matplotlib color map from segmented colors
        PA_cmap = plot_utils.make_colormap(colormap = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb',
                                                    '#3d549f','#655099','#ad5a9b','#dd3933'], bins = n_bins_colors, 
                                                    cmap_name = 'PA_mackey_costum',
                                                    discrete = False, add_alpha = False, return_cmap = True)

        click_plotter.images['PA'] = plot_utils.make_raw_vertex_image(polar_angle, 
                                                    cmap = PA_cmap, vmin = 0, vmax = 1, 
                                                    data2 = np.ones(eccentricity.shape), vmin2 = 0, vmax2 = 1, 
                                                    subject = self.pysub, data2D = True)

        ## pRF Exponent 
        if prf_model_name == 'css':
            click_plotter.images['ns'] = plot_utils.get_flatmaps(click_plotter.pp_prf_est_dict['ns'], 
                                                                vmin1 = 0, vmax1 = 1,
                                                                pysub = self.pysub, 
                                                                cmap = 'plasma')

        
        cortex.quickshow(click_plotter.images['pRF_rsq'], fig = click_plotter.flatmap_ax,
                        with_rois = False, with_curvature = True, with_colorbar=False)

        click_plotter.full_fig.canvas.mpl_connect('button_press_event', click_plotter.onclick)
        click_plotter.full_fig.canvas.mpl_connect('key_press_event', click_plotter.onkey)

        plt.show()


    
    def plot_prf_results(self, participant_list = [], 
                                ses = 'ses-mean', run_type = 'mean', prf_model_name = 'gauss',
                                mask_arr = True, iterative = True, figures_pth = None, use_atlas_rois = True):


        ## Load pRF models for all participants in list
        pp_prf_models = self.pRFModelObj.set_models(participant_list = participant_list, 
                                                        mask_DM = True, combine_ses = True)

        ## stores estimates for all participants in dict, for ease of access
        group_estimates = {}
        group_ROIs = {}
        group_roi_verts = {}
        group_color_codes = {}

        for pp in participant_list:

            
            ## load estimates
            print('Loading iterative estimates')
            estimates_dict, _ = self.pRFModelObj.load_pRF_model_estimates(pp,
                                                                        ses = ses, run_type = run_type, 
                                                                        model_name = prf_model_name, 
                                                                        iterative = iterative)

            ## Get ROI and color codes for plotting
            group_ROIs['sub-{sj}'.format(sj = pp)], group_roi_verts['sub-{sj}'.format(sj = pp)], group_color_codes['sub-{sj}'.format(sj = pp)] = plot_utils.get_rois4plotting(self.MRIObj.params, 
                                                                                                                                        pysub = self.pysub,
                                                                                                                                        use_atlas = use_atlas_rois, 
                                                                                                                                        atlas_pth = op.join(self.MRIObj.derivatives_pth,
                                                                                                                                                            'glasser_atlas','59k_mesh'), 
                                                                                                                                        space = self.MRIObj.sj_space)

            ## mask the estimates, if such is the case
            if mask_arr:
                print('masking estimates')

                # get estimate keys
                keys = self.MRIObj.params['mri']['fitting']['pRF']['estimate_keys'][prf_model_name]
                
                if self.pRFModelObj.fit_hrf:
                    keys = keys[:-1]+self.MRIObj.params['mri']['fitting']['pRF']['estimate_keys']['hrf']+['r2']

                group_estimates['sub-{sj}'.format(sj = pp)] = self.pRFModelObj.mask_pRF_model_estimates(estimates_dict, 
                                                                            ROI = None,
                                                                            estimate_keys = keys,
                                                                            x_ecc_lim = [- pp_prf_models['sub-{sj}'.format(sj = pp)][ses]['prf_stim'].screen_size_degrees/2, 
                                                                                        pp_prf_models['sub-{sj}'.format(sj = pp)][ses]['prf_stim'].screen_size_degrees/2],
                                                                            y_ecc_lim = [- pp_prf_models['sub-{sj}'.format(sj = pp)][ses]['prf_stim'].screen_size_degrees/2, 
                                                                                        pp_prf_models['sub-{sj}'.format(sj = pp)][ses]['prf_stim'].screen_size_degrees/2],
                                                                            rsq_threshold = .1,
                                                                            pysub = self.pysub
                                                                            )
            else:
                group_estimates['sub-{sj}'.format(sj = pp)] = estimates_dict


        ## Now actually plot results
        # 
        ### RSQ ###
        avg_roi_df = self.plot_rsq(participant_list = participant_list, group_estimates = group_estimates, ses = ses, run_type = run_type,
                                            ROIs_dict = group_ROIs, roi_verts_dict = group_roi_verts, color_codes_dict = group_color_codes, model_name = prf_model_name)

        return avg_roi_df


                          

    def plot_rsq(self, participant_list = [], group_estimates = {}, ses = 'ses-mean',  run_type = 'mean',
                        ROIs_dict = {}, roi_verts_dict = {}, color_codes_dict = {}, figures_pth = None, model_name = 'gauss'):
        
        # make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.outputdir, 'rsq', 'pRF_fit')

        avg_roi_df = pd.DataFrame()
        
        ## loop over participants in list
        for pp in participant_list:

            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp), ses)
            
            if not op.exists(sub_figures_pth):
                os.makedirs(sub_figures_pth)

            #### plot flatmap ###
            flatmap = plot_utils.get_flatmaps(group_estimates['sub-{sj}'.format(sj = pp)]['r2'], 
                                                        vmin1 = 0, vmax1 = .8,
                                                        pysub = self.pysub, 
                                                        cmap = 'Reds')
            
            fig_name = op.join(sub_figures_pth,'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_flatmap_RSQ.png'.format(sj = pp,
                                                                                                                                        acq = self.MRIObj.acq,
                                                                                                                                        space = self.MRIObj.sj_space,
                                                                                                                                        run = run_type, 
                                                                                                                                        model = model_name))
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            print('saving %s' %fig_name)
            _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)


            pp_roi_df = plot_utils.get_estimates_roi_df(pp, group_estimates['sub-{sj}'.format(sj = pp)], 
                                                ROIs = ROIs_dict['sub-{sj}'.format(sj = pp)], 
                                                roi_verts = roi_verts_dict['sub-{sj}'.format(sj = pp)], 
                                                est_key = 'r2')

            #### plot distribution ###
            fig, axis = plt.subplots(1, figsize=(10,5), dpi=100, facecolor='w', edgecolor='k')

            v1 = sns.violinplot(data = pp_roi_df, x = 'ROI', y = 'value', 
                                cut=0, inner='box', palette = color_codes_dict['sub-{sj}'.format(sj = pp)], linewidth=1.8, ax = axis) 

            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            #sns.swarmplot(x='ecc', y='cs', data=crwd_df4plot,color=".25",alpha=0.5)
            plt.xticks(fontsize = 10)
            plt.yticks(fontsize = 10)

            plt.xlabel('ROI',fontsize = 15,labelpad=18)
            plt.ylabel('RSQ',fontsize = 15,labelpad=18)
            plt.ylim(0, 1)

            fig.save_fig(fig_name.replace('flatmap','violinplot'))


            ## concatenate average per participant, to make group plot
            avg_roi_df = pd.concat((avg_roi_df,
                                    pp_roi_df.groupby(['sj', 'ROI'])['value'].median().reset_index()))


        return avg_roi_df


