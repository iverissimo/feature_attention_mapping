

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

    def __init__(self, MRIObj, outputdir = None, pRFModelObj = None, pysub = 'hcp_999999', use_sub_rois = True, use_atlas_rois = True, combine_ses = True):
        
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
        use_sub_rois: bool
            if True, will try to find 'hcp_999999_sub-X' by default, if doesnt exist then uses 'hcp_999999'
        use_atlas: bool
            if we want to use the glasser atlas ROIs instead (this is, from the keys conglomerate defined in the params yml)
        combine_ses: bool
            if we want to combine runs from different sessions (relevant for fitting of average across runs)
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

        # get dict of pycortex subject names to use per participant
        self.pysub = self.get_pycortex_sub_dict(participant_list = self.MRIObj.sj_num, pysub = pysub, use_sub_rois = use_sub_rois)
        
        ## if we want to use atlas roi or not
        self.use_atlas_rois = use_atlas_rois

        ## load participant ROIs and color codes
        self.group_ROIs, self.group_roi_verts, self.group_color_codes = plot_utils.get_rois4plotting(self.MRIObj.params, 
                                                                                                    sub_id = self.MRIObj.sj_num,
                                                                                                    pysub = self.pysub, 
                                                                                                    use_atlas = use_atlas_rois, 
                                                                                                    atlas_pth = op.join(self.MRIObj.derivatives_pth,
                                                                                                                        'glasser_atlas','59k_mesh'), 
                                                                                                    space = self.MRIObj.sj_space)
        ## set ROI key names to use in plots
        # doing this to guarantee same order in all plots 
        if self.use_atlas_rois:
            self.ROI_keys = list(self.MRIObj.params['plotting']['ROIs']['glasser_atlas'].keys())
        else:
            self.ROI_keys = self.MRIObj.params['plotting']['ROIs'][self.MRIObj.sj_space]

        ## load participant models
        # which also will load DM and mask it according to participants behavior
        self.pp_prf_models = self.pRFModelObj.set_models(participant_list = self.MRIObj.sj_num, 
                                                    mask_DM = True, combine_ses = combine_ses)


    def get_pycortex_sub_dict(self, participant_list = [], pysub = 'hcp_999999', use_sub_rois = True):

        """ 
        Helper function to get dict with
        pycortex name per participant in participant list 
        
        Parameters
        ----------
        participant_list : list
            list with participant IDs
        use_sub_rois: bool
            if True we will try to find sub specific folder, else will just use pysub
            
        """
        pysub_dict = {}

        for pp in participant_list:

            if use_sub_rois:
                # check if subject pycortex folder exists
                pysub_folder = '{ps}_sub-{pp}'.format(ps = pysub, pp = pp)

                if op.exists(op.join(cortex.options.config.get('basic', 'filestore'), pysub_folder)):
                    print('Participant overlay %s in pycortex filestore, assumes we draw ROIs there'%pysub_folder)
                    pysub_dict['sub-{pp}'.format(pp = pp)] = pysub_folder
                else:
                    pysub_dict['sub-{pp}'.format(pp = pp)] = pysub
            else:
                pysub_dict['sub-{pp}'.format(pp = pp)] = pysub

        return pysub_dict

            
    def get_prf_estimate_keys(self, prf_model_name = 'gauss'):

        """ 
        Helper function to get prf estimate keys
        
        Parameters
        ----------
        prf_model_name : str
            pRF model name (defaults to gauss)
            
        """

        # get estimate key names, which vary per model used
        keys = self.MRIObj.params['mri']['fitting']['pRF']['estimate_keys'][prf_model_name]
        
        if self.pRFModelObj.fit_hrf:
            keys = keys[:-1]+self.MRIObj.params['mri']['fitting']['pRF']['estimate_keys']['hrf']+['r2']

        return keys
                    

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
            keys = self.get_prf_estimate_keys(prf_model_name = prf_model_name)
            
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

    
    def save_estimates4drawing(self, participant, task2draw = 'pRF',
                                    ses = 'ses-mean', run_type = 'mean', pysub = 'hcp_999999',
                                    prf_model_name = 'gauss', file_ext = '_cropped_dc_psc.npy', rsq_threshold = .1):

        """
        Load estimates into pycortex sub specific overlay, to draw ROIs
        - pRF estimates : will load polar angle (with and without alpha level) and eccentricity
        - FA estimates : not implemented yet

        Note - requires that we have (at least) pRF estimates saved 

        Parameters
        ----------
        participant : str
            subject ID
        task2draw: str
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
        pysub: str
            name of pycortex subject folder, to draw ROIs. 
            will try to find 'hcp_999999_sub-X' by default, if doesnt exist then throws error

        """

        # if we provide a list, we want to save average estimates in overlay
        if isinstance(participant, str): 
            
            participant_list = [participant]
            sub_name = participant

            # check if subject pycortex folder exists
            pysub_folder = '{ps}_sub-{pp}'.format(ps = pysub, 
                                                pp = participant)

            if op.exists(op.join(cortex.options.config.get('basic', 'filestore'), pysub_folder)):
                print('Participant overlay %s in pycortex filestore, assumes we draw ROIs there'%pysub_folder)
            else:
                raise NameError('FOLDER %s DOESNT EXIST'%op.join(cortex.options.config.get('basic', 'filestore'), pysub_folder))

        else:
            participant_list = participant
            pysub_folder = pysub
            sub_name = 'group'

        ## get estimates for all participants, if applicable 
        r2_avg = np.array([])
        xx_avg = np.array([])
        yy_avg = np.array([])

        for pp in participant_list:

            ## load model and prf estimates for that participant
            pp_prf_est_dict, _ = self.pRFModelObj.load_pRF_model_estimates(pp, 
                                                                            ses = ses, run_type = run_type, 
                                                                            model_name = prf_model_name, iterative = True,
                                                                            fit_hrf = self.pRFModelObj.fit_hrf)
            ## STACK
            r2_avg = np.vstack([r2_avg, pp_prf_est_dict['r2']]) if r2_avg.size else pp_prf_est_dict['r2']
            xx_avg = np.vstack([xx_avg, pp_prf_est_dict['x'] ]) if xx_avg.size else pp_prf_est_dict['x'] 
            yy_avg = np.vstack([yy_avg, pp_prf_est_dict['y'] ]) if yy_avg.size else pp_prf_est_dict['y'] 

        ## TAKE MEDIAN
        r2_avg = np.nanmedian(r2_avg, axis = 0)
        xx_avg = np.nanmedian(xx_avg, axis = 0)
        yy_avg = np.nanmedian(yy_avg, axis = 0)

        ## calculate pa + ecc + size
        nan_mask = np.where((np.isnan(r2_avg)) | (r2_avg < rsq_threshold))[0]
        
        complex_location = xx_avg + yy_avg * 1j # calculate eccentricity values

        polar_angle = np.angle(complex_location)
        polar_angle_norm = ((polar_angle + np.pi) / (np.pi * 2.0))
        polar_angle_norm[nan_mask] = np.nan

        eccentricity = np.abs(complex_location)
        eccentricity[nan_mask] = np.nan

        ## make alpha mask
        alpha_level = mri_utils.normalize(np.clip(r2_avg, rsq_threshold, .6)) # normalize 
        alpha_level[nan_mask] = np.nan

        ## set flatmaps ##
        images = {}
        n_bins_colors = 256

        ## pRF rsq
        images['pRF_rsq'] = plot_utils.get_flatmaps(r2_avg, 
                                                    vmin1 = 0, vmax1 = .8,
                                                    pysub = pysub_folder, 
                                                    cmap = 'Reds')
        ## pRF Eccentricity

        # make costum coor map
        ecc_cmap = plot_utils.make_colormap(colormap = ['#dd3933','#f3eb53','#7cb956','#82cbdb','#3d549f'],
                                            bins = n_bins_colors, cmap_name = 'ECC_mackey_costum', 
                                            discrete = False, add_alpha = False, return_cmap = True)


        images['ecc'] = plot_utils.make_raw_vertex_image(eccentricity, 
                                                        cmap = ecc_cmap, 
                                                        vmin = 0, vmax = 6, 
                                                        data2 = alpha_level, 
                                                        vmin2 = 0, vmax2 = 1, 
                                                        subject = pysub_folder, data2D = True)

        ## pRF Polar Angle
       
        # get matplotlib color map from segmented colors
        PA_cmap = plot_utils.make_colormap(colormap = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb',
                                                    '#3d549f','#655099','#ad5a9b','#dd3933'], bins = n_bins_colors, 
                                                    cmap_name = 'PA_mackey_costum',
                                                    discrete = False, add_alpha = False, return_cmap = True)

        images['PA_alpha'] = plot_utils.make_raw_vertex_image(polar_angle_norm, 
                                                    cmap = PA_cmap, vmin = 0, vmax = 1, 
                                                    data2 = alpha_level, 
                                                    vmin2 = 0, vmax2 = 1, 
                                                    subject = pysub_folder, data2D = True)

        images['PA'] = plot_utils.make_raw_vertex_image(polar_angle_norm, 
                                                    cmap = PA_cmap, vmin = 0, vmax = 1, 
                                                    data2 = np.ones(alpha_level.shape), 
                                                    vmin2 = 0, vmax2 = 1, 
                                                    subject = pysub_folder, data2D = True)

        ## also make non uniform color wheel, which helps seeing borders
        rgb_pa = plot_utils.get_NONuniform_polar_angle(xx_avg, yy_avg, r2_avg, 
                                                        angle_thresh = 3*np.pi/4, 
                                                        rsq_thresh = 0, 
                                                        pysub = pysub_folder)

        # make ones mask, only for high rsq fits
        ones_mask = np.ones(r2_avg.shape)
        ones_mask[r2_avg < 0.3] = np.nan

        images['PA_half_hemi'] = cortex.VertexRGB(rgb_pa[:, 0], rgb_pa[:, 1], rgb_pa[:, 2],
                                           alpha = ones_mask,
                                           subject = pysub_folder)

        ### ADD TO OVERLAY, TO DRAW BORDERS
        cortex.utils.add_roi(images['pRF_rsq'], name = 'RSQ_sub-{sj}_task-pRF_run-{run}_model-{model}_withHRF-{hrf_bool}'.format(sj = sub_name,
                                                                                                    run = run_type,
                                                                                                    model = prf_model_name,
                                                                                                    hrf_bool = str(self.pRFModelObj.fit_hrf)), 
                                                                                                    open_inkscape = False, add_path = False)
        cortex.utils.add_roi(images['ecc'], name = 'ECC_sub-{sj}_task-pRF_run-{run}_model-{model}_withHRF-{hrf_bool}'.format(sj = sub_name,
                                                                                                    run = run_type,
                                                                                                    model = prf_model_name,
                                                                                                    hrf_bool = str(self.pRFModelObj.fit_hrf)), 
                                                                                                    open_inkscape = False, add_path = False)
        cortex.utils.add_roi(images['PA'], name = 'PA_sub-{sj}_task-pRF_run-{run}_model-{model}_withHRF-{hrf_bool}'.format(sj = sub_name,
                                                                                                    run = run_type,
                                                                                                    model = prf_model_name,
                                                                                                    hrf_bool = str(self.pRFModelObj.fit_hrf)), 
                                                                                                    open_inkscape = False, add_path = False)
        cortex.utils.add_roi(images['PA_alpha'], name = 'PA_alpha_sub-{sj}_task-pRF_run-{run}_model-{model}_withHRF-{hrf_bool}'.format(sj = sub_name,
                                                                                                    run = run_type,
                                                                                                    model = prf_model_name,
                                                                                                    hrf_bool = str(self.pRFModelObj.fit_hrf)), 
                                                                                                    open_inkscape = False, add_path = False)
        cortex.utils.add_roi(images['PA_half_hemi'], name = 'PA_half_hemi_sub-{sj}_task-pRF_run-{run}_model-{model}_withHRF-{hrf_bool}'.format(sj = sub_name,
                                                                                                    run = run_type,
                                                                                                    model = prf_model_name,
                                                                                                    hrf_bool = str(self.pRFModelObj.fit_hrf)), 
                                                                                                    open_inkscape = False, add_path = False)

        print('Done')


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
        keys = self.get_prf_estimate_keys(prf_model_name = prf_model_name)

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

    
    def plot_prf_results(self, participant_list = [], 
                                ses = 'ses-mean', run_type = 'mean', prf_model_name = 'gauss', max_ecc_ext = 5,
                                mask_arr = True, rsq_threshold = .1, iterative = True, figures_pth = None):


        ## stores estimates for all participants in dict, for ease of access
        group_estimates = {}
  
        for pp in participant_list:

            #max_ecc_ext = self.pp_prf_models['sub-{sj}'.format(sj = pp)][ses]['prf_stim'].screen_size_degrees/2
            
            ## load estimates
            print('Loading iterative estimates')
            estimates_dict, _ = self.pRFModelObj.load_pRF_model_estimates(pp,
                                                                        ses = ses, run_type = run_type, 
                                                                        model_name = prf_model_name, 
                                                                        iterative = iterative,
                                                                        fit_hrf = self.pRFModelObj.fit_hrf)

            ## mask the estimates, if such is the case
            if mask_arr:
                print('masking estimates')

                # get estimate keys
                keys = self.get_prf_estimate_keys(prf_model_name = prf_model_name)

                group_estimates['sub-{sj}'.format(sj = pp)] = self.pRFModelObj.mask_pRF_model_estimates(estimates_dict, 
                                                                            ROI = None,
                                                                            estimate_keys = keys,
                                                                            x_ecc_lim = [- max_ecc_ext, max_ecc_ext],
                                                                            y_ecc_lim = [- max_ecc_ext, max_ecc_ext],
                                                                            rsq_threshold = rsq_threshold,
                                                                            pysub = self.pysub['sub-{pp}'.format(pp = pp)]
                                                                            )
            else:
                group_estimates['sub-{sj}'.format(sj = pp)] = estimates_dict


        ## Now actually plot results
        # 
        ### RSQ ###
        self.plot_rsq(participant_list = participant_list, group_estimates = group_estimates, ses = ses, run_type = run_type,
                                            model_name = prf_model_name, figures_pth = figures_pth)

        ### ECC and SIZE ###
        self.plot_ecc_size(participant_list = participant_list, group_estimates = group_estimates, ses = ses, run_type = run_type,
                                            model_name = prf_model_name, figures_pth = figures_pth)

        ### EXPONENT ###
        if prf_model_name == 'css':
            self.plot_exponent(participant_list = participant_list, group_estimates = group_estimates, ses = ses, run_type = run_type,
                                            model_name = prf_model_name, figures_pth = figures_pth)

        ### POLAR ANGLE ####
        self.plot_pa(participant_list = participant_list, group_estimates = group_estimates, ses = ses, run_type = run_type,
                                        model_name = prf_model_name, figures_pth = figures_pth, 
                                        n_bins_colors = 256, max_x_lim = max_ecc_ext, angle_thresh = 3*np.pi/4)


    def plot_rsq(self, participant_list = [], group_estimates = {}, ses = 'ses-mean',  run_type = 'mean',
                        figures_pth = None, model_name = 'gauss'):
        
        # make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.outputdir, 'rsq', self.MRIObj.params['mri']['fitting']['pRF']['fit_folder'])

        # save values per roi in dataframe
        avg_roi_df = pd.DataFrame()
        
        ## loop over participants in list
        for pp in participant_list:
            
            # make path to save sub-specific figures
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp), ses)
            os.makedirs(sub_figures_pth, exist_ok=True)

            #### plot flatmap ###
            flatmap = plot_utils.get_flatmaps(group_estimates['sub-{sj}'.format(sj = pp)]['r2'], 
                                                        vmin1 = 0, vmax1 = .8,
                                                        pysub = self.pysub['sub-{pp}'.format(pp = pp)], 
                                                        cmap = 'Reds')
            
            fig_name = op.join(sub_figures_pth,'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_flatmap_RSQ.png'.format(sj = pp,
                                                                                                                                        acq = self.MRIObj.acq,
                                                                                                                                        space = self.MRIObj.sj_space,
                                                                                                                                        run = run_type, 
                                                                                                                                        model = model_name))
            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            print('saving %s' %fig_name)
            _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

            ## get estimates per ROI
            pp_roi_df = plot_utils.get_estimates_roi_df(pp, group_estimates['sub-{sj}'.format(sj = pp)], 
                                                ROIs = self.group_ROIs['sub-{sj}'.format(sj = pp)], 
                                                roi_verts = self.group_roi_verts['sub-{sj}'.format(sj = pp)], 
                                                est_key = 'r2')

            #### plot distribution ###
            fig, axis = plt.subplots(1, figsize=(10,5), dpi=100, facecolor='w', edgecolor='k')

            v1 = sns.violinplot(data = pp_roi_df, x = 'ROI', y = 'value', 
                                order = self.ROI_keys,
                                cut=0, inner='box', palette = self.group_color_codes['sub-{sj}'.format(sj = pp)], linewidth=1.8, ax = axis) 

            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            #sns.swarmplot(x='ecc', y='cs', data=crwd_df4plot,color=".25",alpha=0.5)
            plt.xticks(fontsize = 10)
            plt.yticks(fontsize = 10)

            plt.xlabel('ROI',fontsize = 15,labelpad=18)
            plt.ylabel('RSQ',fontsize = 15,labelpad=18)
            plt.ylim(0, 1)

            fig.savefig(fig_name.replace('flatmap','violinplot'))

            ## concatenate average per participant, to make group plot
            avg_roi_df = pd.concat((avg_roi_df,
                                    pp_roi_df.groupby(['sj', 'ROI'])['value'].median().reset_index()))

        # if we provided several participants, make group plot
        if len(participant_list) > 1:

            fig, axis = plt.subplots(1, figsize=(10,5), dpi=100, facecolor='w', edgecolor='k')

            v1 = sns.violinplot(data = avg_roi_df, x = 'ROI', y = 'value', 
                                order = self.ROI_keys,
                                cut=0, inner='box', palette = self.group_color_codes['sub-{sj}'.format(sj = pp)], 
                                linewidth=1.8, ax = axis)

            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            sns.stripplot(data = avg_roi_df, x = 'ROI', y = 'value', 
                            order = self.ROI_keys,
                            color="white", alpha=0.5)
            plt.xticks(fontsize = 10)
            plt.yticks(fontsize = 10)

            plt.xlabel('ROI',fontsize = 15,labelpad=18)
            plt.ylabel('RSQ',fontsize = 15,labelpad=18)
            plt.ylim(0, 1)

            fig.savefig(op.join(figures_pth, op.split(fig_name)[-1].replace('flatmap','violinplot').replace('sub-{sj}'.format(sj = pp),'sub-GROUP')))

    
    def compare_pRF_model_rsq(self, participant_list = [], ses = 'ses-mean', run_type = 'mean', 
                                prf_model_list = ['gauss', 'css'],
                                mask_arr = True, rsq_threshold = .1, figures_pth = None):


        ## stores estimates for all participants in dict, for ease of access
        group_estimates = {}
        pp_model_roi_df = pd.DataFrame()

        # if we only provided one model name, assumes we want to compare grid to iterative rsq
        stage_names = ['iterative', 'grid'] if len(prf_model_list) == 1 else ['iterative']

        # make general output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.outputdir, 'rsq', self.MRIObj.params['mri']['fitting']['pRF']['fit_folder'])

        for pp in participant_list:

            max_ecc_ext = self.pp_prf_models['sub-{sj}'.format(sj = pp)][ses]['prf_stim'].screen_size_degrees/2

            ## make sub specific fig path
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp), ses)
            os.makedirs(sub_figures_pth, exist_ok=True)

            group_estimates['sub-{sj}'.format(sj = pp)] = {}

            ## iterate over models
            for mod_name in prf_model_list:

                # get estimate keys
                keys = self.get_prf_estimate_keys(prf_model_name = mod_name)

                group_estimates['sub-{sj}'.format(sj = pp)][mod_name] = {}
                
                ## iterate over fit stages
                for stage in stage_names:

                    it_bool = True if stage == 'iterative' else False

                    ## load estimates
                    print('Loading {st} estimates for model {n}'.format(st = stage, n = mod_name))
                    estimates_dict, _ = self.pRFModelObj.load_pRF_model_estimates(pp,
                                                                                ses = ses, run_type = run_type, 
                                                                                model_name = mod_name, 
                                                                                iterative = it_bool,
                                                                                fit_hrf = self.pRFModelObj.fit_hrf)

                    ## mask the estimates, if such is the case
                    if mask_arr:
                        group_estimates['sub-{sj}'.format(sj = pp)][mod_name][stage] = self.pRFModelObj.mask_pRF_model_estimates(estimates_dict, 
                                                                                                    ROI = None,
                                                                                                    estimate_keys = keys,
                                                                                                    x_ecc_lim = [- max_ecc_ext, max_ecc_ext],
                                                                                                    y_ecc_lim = [- max_ecc_ext, max_ecc_ext],
                                                                                                    rsq_threshold = rsq_threshold,
                                                                                                    pysub = self.pysub['sub-{pp}'.format(pp = pp)]
                                                                                                    )
                    else:
                        group_estimates['sub-{sj}'.format(sj = pp)][mod_name][stage] = estimates_dict

                
                #### plot flatmap ###
                if len(prf_model_list) == 1:

                    ## plot rsq value difference from grid to iterative
                    flatmap = plot_utils.get_flatmaps(group_estimates['sub-{sj}'.format(sj = pp)][mod_name]['iterative']['r2'] - group_estimates['sub-{sj}'.format(sj = pp)][mod_name]['grid']['r2'], 
                                                                vmin1 = -.1, vmax1 = .1,
                                                                pysub = self.pysub['sub-{pp}'.format(pp = pp)], 
                                                                cmap = 'BuBkRd')
                    
                    fig_name = op.join(sub_figures_pth,'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_flatmap_grid2iterative_diff_RSQ.png'.format(sj = pp,
                                                                                                                                                                acq = self.MRIObj.acq,
                                                                                                                                                                space = self.MRIObj.sj_space,
                                                                                                                                                                run = run_type, 
                                                                                                                                                                model = mod_name))
                    if self.pRFModelObj.fit_hrf:
                        fig_name = fig_name.replace('.png','_withHRF.png') 

                    print('saving %s' %fig_name)
                    _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

                ## get roi rsq for each model ########
                pp_model_roi_df = pd.concat((pp_model_roi_df,
                                            plot_utils.get_estimates_roi_df(pp, group_estimates['sub-{sj}'.format(sj = pp)][mod_name]['iterative'], 
                                                                                            ROIs = self.group_ROIs['sub-{sj}'.format(sj = pp)], 
                                                                                            roi_verts = self.group_roi_verts['sub-{sj}'.format(sj = pp)], 
                                                                                            est_key = 'r2',
                                                                                            model = mod_name)
                                            ))

            ### plot flatmaps comparing gauss with other models ###
            if len(prf_model_list) > 1:

                non_gauss_mod = [mn for mn in prf_model_list if mn != 'gauss']

                for non_g in non_gauss_mod:
                    flatmap = plot_utils.get_flatmaps(group_estimates['sub-{sj}'.format(sj = pp)][non_g]['iterative']['r2'] - group_estimates['sub-{sj}'.format(sj = pp)]['gauss']['iterative']['r2'], 
                                                                vmin1 = -.1, vmax1 = .1,
                                                                pysub = self.pysub['sub-{pp}'.format(pp = pp)], 
                                                                cmap = 'BuBkRd')
                    
                    fig_name = op.join(sub_figures_pth,'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_flatmap_gauss2{model}_diff_RSQ.png'.format(sj = pp,
                                                                                                                                                        acq = self.MRIObj.acq,
                                                                                                                                                        space = self.MRIObj.sj_space,
                                                                                                                                                        run = run_type, 
                                                                                                                                                        model = non_g))
                    if self.pRFModelObj.fit_hrf:
                        fig_name = fig_name.replace('.png','_withHRF.png') 

                    print('saving %s' %fig_name)
                    _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

                #### plot distribution ###
                fig, axis = plt.subplots(1, figsize=(10,5), dpi=100, facecolor='w', edgecolor='k')

                v1 = sns.violinplot(data = pp_model_roi_df, x = 'ROI', y = 'value', hue = 'model',
                                    order = self.ROI_keys,
                                    cut=0, inner='box',linewidth=1.8, ax = axis) 
                v1.set(xlabel=None)
                v1.set(ylabel=None)
                plt.margins(y=0.025)
                #sns.swarmplot(x='ecc', y='cs', data=crwd_df4plot,color=".25",alpha=0.5)
                plt.xticks(fontsize = 10)
                plt.yticks(fontsize = 10)

                plt.xlabel('ROI',fontsize = 15,labelpad=18)
                plt.ylabel('RSQ',fontsize = 15,labelpad=18)
                plt.ylim(0, 1)
                fig.savefig(fig_name.replace('flatmap','violinplot'))

        # if we provided several participants, make group plot
        if len(participant_list) > 1:

            fig, axis = plt.subplots(1, figsize=(10,5), dpi=100, facecolor='w', edgecolor='k')

            v1 = sns.violinplot(data = pp_model_roi_df.groupby(['sj', 'ROI','model'])['value'].median().reset_index(), 
                                        x = 'ROI', y = 'value',  hue = 'model',
                                        order = self.ROI_keys,
                                        cut=0, inner='box',
                                        linewidth=1.8, ax = axis)

            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            # sns.stripplot(data = pp_model_roi_df.groupby(['sj', 'ROI','model'])['value'].median().reset_index(), x = 'ROI', y = 'value', 
            #                 order=self.MRIObj.params['plotting']['ROIs']['glasser_atlas'].keys(),
            #                 color="white", alpha=0.5)
            plt.xticks(fontsize = 10)
            plt.yticks(fontsize = 10)

            plt.xlabel('ROI',fontsize = 15,labelpad=18)
            plt.ylabel('RSQ',fontsize = 15,labelpad=18)
            plt.ylim(0, 1)

            fig_name = op.join(figures_pth,'sub-GROUP_task-pRF_acq-{acq}_space-{space}_run-{run}_violinplot_modelcomparison_RSQ.png'.format(acq = self.MRIObj.acq,
                                                                                                                                                        space = self.MRIObj.sj_space,
                                                                                                                                                        run = run_type))
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png')

            fig.savefig(fig_name)
            
        return pp_model_roi_df


    def plot_ecc_size(self, participant_list = [], group_estimates = {}, ses = 'ses-mean',  run_type = 'mean',
                        figures_pth = None, model_name = 'gauss', n_bins_colors = 256, n_bins_dist = 8, 
                        max_ecc_ext = 5, max_size_ext = 14):
        
        ### make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.outputdir, 'ecc_size', self.MRIObj.params['mri']['fitting']['pRF']['fit_folder'])
            
        ## make costum ECC color map
        ecc_cmap = plot_utils.make_colormap(colormap = ['#dd3933','#f3eb53','#7cb956','#82cbdb','#3d549f'],
                                            bins = n_bins_colors, cmap_name = 'ECC_mackey_costum', 
                                            discrete = False, add_alpha = False, return_cmap = True)
        
        avg_bin_df = pd.DataFrame()
        
        ## loop over participants in list
        for pp in participant_list:

            ## make sub specific folder
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp), ses)
            os.makedirs(sub_figures_pth, exist_ok=True)
                
            ## use RSQ as alpha level for flatmaps
            r2 = group_estimates['sub-{sj}'.format(sj = pp)]['r2']
            alpha_level = mri_utils.normalize(np.clip(r2, 0, .6)) # normalize 
                
            ## get ECCENTRICITY estimates
            complex_location = group_estimates['sub-{sj}'.format(sj = pp)]['x'] + group_estimates['sub-{sj}'.format(sj = pp)]['y'] * 1j # calculate eccentricity values
            eccentricity = np.abs(complex_location)

            #### plot flatmap ###
            flatmap =  plot_utils.make_raw_vertex_image(eccentricity, 
                                                        cmap = ecc_cmap, 
                                                        vmin = 0, vmax = 6, 
                                                        data2 = alpha_level, 
                                                        vmin2 = 0, vmax2 = 1, 
                                                        subject = self.pysub['sub-{pp}'.format(pp = pp)], 
                                                        data2D = True)
            
            fig_name = op.join(sub_figures_pth,'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_flatmap_ECC.png'.format(sj = pp,
                                                                                                                                        acq = self.MRIObj.acq,
                                                                                                                                        space = self.MRIObj.sj_space,
                                                                                                                                        run = run_type, 
                                                                                                                                        model = model_name))
            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            print('saving %s' %fig_name)
            _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

            
            ## get SIZE estimates
            if model_name in ['dn', 'dog']:
                size_fwhmax, fwatmin = plot_utils.fwhmax_fwatmin(model_name, 
                                                                 group_estimates['sub-{sj}'.format(sj = pp)])
            else: 
                size_fwhmax = plot_utils.fwhmax_fwatmin(model_name, 
                                                        group_estimates['sub-{sj}'.format(sj = pp)])
                
            #### plot flatmap ###
            flatmap =  plot_utils.make_raw_vertex_image(size_fwhmax, 
                                                        cmap = 'hot', 
                                                        vmin = 0, vmax = 7, 
                                                        data2 = alpha_level, 
                                                        vmin2 = 0, vmax2 = 1, 
                                                        subject = self.pysub['sub-{pp}'.format(pp = pp)], 
                                                        data2D = True)
            
            fig_name = op.join(sub_figures_pth,'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_flatmap_SIZE-fwhmax.png'.format(sj = pp,
                                                                                                                                        acq = self.MRIObj.acq,
                                                                                                                                        space = self.MRIObj.sj_space,
                                                                                                                                        run = run_type, 
                                                                                                                                        model = model_name))
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            print('saving %s' %fig_name)
            _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

            ## GET values per ROI ##
            ecc_pp_roi_df = plot_utils.get_estimates_roi_df(pp, eccentricity, 
                                                ROIs = self.group_ROIs['sub-{sj}'.format(sj = pp)], 
                                                roi_verts = self.group_roi_verts['sub-{sj}'.format(sj = pp)], 
                                                model = model_name)
            
            size_pp_roi_df = plot_utils.get_estimates_roi_df(pp, size_fwhmax, 
                                                ROIs = self.group_ROIs['sub-{sj}'.format(sj = pp)], 
                                                roi_verts = self.group_roi_verts['sub-{sj}'.format(sj = pp)], 
                                                model = model_name)
            
            rsq_pp_roi_df = plot_utils.get_estimates_roi_df(pp, r2, 
                                                ROIs = self.group_ROIs['sub-{sj}'.format(sj = pp)], 
                                                roi_verts = self.group_roi_verts['sub-{sj}'.format(sj = pp)], 
                                                model = model_name)

            # merge them into one
            df_ecc_siz = pd.merge(ecc_pp_roi_df.rename(columns={'value': 'ecc'}),
                                size_pp_roi_df.rename(columns={'value': 'size'}))
            df_ecc_siz = pd.merge(df_ecc_siz, rsq_pp_roi_df.rename(columns={'value': 'rsq'}))

            ## drop the nans
            df_ecc_siz = df_ecc_siz[~np.isnan(df_ecc_siz.rsq)]

            ##### plot unbinned df #########
            sns.set(font_scale=1.3)
            sns.set_style("ticks")

            g = sns.lmplot(x="ecc", y="size", hue = 'ROI', data = df_ecc_siz, scatter_kws={'alpha':0.05},
                        scatter=True, palette = self.group_color_codes['sub-{sj}'.format(sj = pp)]) #, markers=['^', 's', 'o', 'v', 'D', 'h', 'P', '.', ','])

            ax = plt.gca()
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            ax.axes.set_xlim(0, max_ecc_ext)
            ax.axes.set_ylim(0.5, max_size_ext)
            ax.set_xlabel('pRF eccentricity [deg]', fontsize = 20, labelpad = 15)
            ax.set_ylabel('pRF size FWHMax [deg]', fontsize = 20, labelpad = 15)
            sns.despine(offset=15)
            # to make legend full alpha
            for lh in g._legend.legendHandles: 
                lh.set_alpha(1)
            fig2 = plt.gcf()

            fig_name = op.join(sub_figures_pth,'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_ecc_vs_size_UNbinned.png'.format(sj = pp,
                                                                                                                                        acq = self.MRIObj.acq,
                                                                                                                                        space = self.MRIObj.sj_space,
                                                                                                                                        run = run_type, 
                                                                                                                                        model = model_name))
            
            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            fig2.savefig(fig_name, dpi=100,bbox_inches = 'tight')

            ## bin it, for cleaner plot
            for r_name in self.group_ROIs['sub-{sj}'.format(sj = pp)]:

                mean_x, _, mean_y, _ = plot_utils.get_weighted_bins(df_ecc_siz.loc[(df_ecc_siz['ROI'] == r_name) & \
                                                                                     (df_ecc_siz['rsq'].notna())],
                                                                                    x_key = 'ecc', y_key = 'size', weight_key = 'rsq', n_bins = n_bins_dist)

                avg_bin_df = pd.concat((avg_bin_df,
                                        pd.DataFrame({ 'sj': np.tile('sub-{sj}'.format(sj = pp), len(mean_x)),
                                                    'ROI': np.tile(r_name, len(mean_x)),
                                                    'ecc': mean_x,
                                                    'size': mean_y
                                        })))

            ##### plot binned df #########
            sns.set(font_scale=1.3)
            sns.set_style("ticks")

            g = sns.lmplot(x="ecc", y="size", hue = 'ROI', data = avg_bin_df.loc[avg_bin_df['sj'] == 'sub-{sj}'.format(sj = pp)], scatter_kws={'alpha':0.15},
                        scatter=True, palette = self.group_color_codes['sub-{sj}'.format(sj = pp)]) #, markers=['^', 's', 'o', 'v', 'D', 'h', 'P', '.', ','])

            ax = plt.gca()
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            ax.axes.set_xlim(0, max_ecc_ext)
            ax.axes.set_ylim(0.5, max_size_ext)
            ax.set_xlabel('pRF eccentricity [deg]', fontsize = 20, labelpad = 15)
            ax.set_ylabel('pRF size FWHMax [deg]', fontsize = 20, labelpad = 15)
            sns.despine(offset=15)
            # to make legend full alpha
            for lh in g._legend.legendHandles: 
                lh.set_alpha(1)
            fig2 = plt.gcf()

            fig_name = op.join(sub_figures_pth,'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_ecc_vs_size_binned.png'.format(sj = pp,
                                                                                                                                        acq = self.MRIObj.acq,
                                                                                                                                        space = self.MRIObj.sj_space,
                                                                                                                                        run = run_type, 
                                                                                                                                        model = model_name))
            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            fig2.savefig(fig_name, dpi=100,bbox_inches = 'tight')

        if len(participant_list) > 1:

            ##### plot binned df for GROUP #########
            sns.set(font_scale=1.3)
            sns.set_style("ticks")

            g = sns.lmplot(x="ecc", y="size", hue = 'ROI', data = avg_bin_df, 
                        scatter=True, palette = self.group_color_codes['sub-{sj}'.format(sj = pp)],  
                        x_bins = n_bins_dist) #, markers=['^', 's', 'o', 'v', 'D', 'h', 'P', '.', ','])

            ax = plt.gca()
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            ax.axes.set_xlim(0, max_ecc_ext)
            ax.axes.set_ylim(0.5, max_size_ext)
            ax.set_xlabel('pRF eccentricity [deg]', fontsize = 20, labelpad = 15)
            ax.set_ylabel('pRF size FWHMax [deg]', fontsize = 20, labelpad = 15)
            sns.despine(offset=15)
            # to make legend full alpha
            for lh in g._legend.legendHandles: 
                lh.set_alpha(1)
            fig2 = plt.gcf()

            fig_name = op.join(figures_pth,'sub-GROUP_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_ecc_vs_size_binned.png'.format(acq = self.MRIObj.acq,
                                                                                                                                        space = self.MRIObj.sj_space,
                                                                                                                                        run = run_type, 
                                                                                                                                        model = model_name))
            # if we fitted hrf, then add that to fig name
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            fig2.savefig(fig_name, dpi=100,bbox_inches = 'tight')


    def plot_exponent(self, participant_list = [], group_estimates = {}, ses = 'ses-mean',  run_type = 'mean',
                        figures_pth = None, model_name = 'gauss'):
        
        ### make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.outputdir, 'exponent', self.MRIObj.params['mri']['fitting']['pRF']['fit_folder'])
            
        
        avg_df = pd.DataFrame()
        
        ## loop over participants in list
        for pp in participant_list:

            ## make sub specific folder
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp), ses)
            os.makedirs(sub_figures_pth, exist_ok=True)
                
            ## use RSQ as alpha level for flatmaps
            r2 = group_estimates['sub-{sj}'.format(sj = pp)]['r2']
            alpha_level = mri_utils.normalize(np.clip(r2, 0, .6)) # normalize 
                
            #### plot flatmap ###
            flatmap =  plot_utils.make_raw_vertex_image(group_estimates['sub-{sj}'.format(sj = pp)]['ns'], 
                                                        cmap = 'plasma', 
                                                        vmin = 0, vmax = 1, 
                                                        data2 = alpha_level, 
                                                        vmin2 = 0, vmax2 = 1, 
                                                        subject = self.pysub['sub-{pp}'.format(pp = pp)], 
                                                        data2D = True)
            
            fig_name = op.join(sub_figures_pth,'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_flatmap_N.png'.format(sj = pp,
                                                                                                                                        acq = self.MRIObj.acq,
                                                                                                                                        space = self.MRIObj.sj_space,
                                                                                                                                        run = run_type, 
                                                                                                                                        model = model_name))
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            print('saving %s' %fig_name)
            _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

            
            ## GET values per ROI ##
            ns_pp_roi_df = plot_utils.get_estimates_roi_df(pp, group_estimates['sub-{sj}'.format(sj = pp)]['ns'], 
                                                ROIs = self.group_ROIs['sub-{sj}'.format(sj = pp)], 
                                                roi_verts = self.group_roi_verts['sub-{sj}'.format(sj = pp)], 
                                                model = model_name)
            
            rsq_pp_roi_df = plot_utils.get_estimates_roi_df(pp, r2, 
                                                ROIs = self.group_ROIs['sub-{sj}'.format(sj = pp)], 
                                                roi_verts = self.group_roi_verts['sub-{sj}'.format(sj = pp)], 
                                                model = model_name)

            # merge them into one
            df_ns = pd.merge(ns_pp_roi_df.rename(columns={'value': 'exponent'}),
                                rsq_pp_roi_df.rename(columns={'value': 'rsq'}))

            ## drop the nans
            df_ns = df_ns[~np.isnan(df_ns.rsq)]

            #### plot distribution ###
            fig, axis = plt.subplots(1, figsize=(10,5), dpi=100, facecolor='w', edgecolor='k')

            v1 = sns.violinplot(data = df_ns, x = 'ROI', y = 'exponent', 
                                cut=0, inner='box', palette = self.group_color_codes['sub-{sj}'.format(sj = pp)], linewidth=1.8, ax = axis) 

            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            #sns.swarmplot(x='ecc', y='cs', data=crwd_df4plot,color=".25",alpha=0.5)
            plt.xticks(fontsize = 10)
            plt.yticks(fontsize = 10)

            plt.xlabel('ROI',fontsize = 15,labelpad=18)
            plt.ylabel('Exponent',fontsize = 15,labelpad=18)
            plt.ylim(0, 1)

            fig.savefig(fig_name.replace('flatmap','violinplot'))
            
            
            ## concatenate average per participant, to make group plot
            avg_df = pd.concat((avg_df,
                                    df_ns.groupby(['sj', 'ROI'])['exponent'].median().reset_index()))
            
            
        if len(participant_list) > 1:

            fig, axis = plt.subplots(1, figsize=(10,5), dpi=100, facecolor='w', edgecolor='k')

            v1 = sns.violinplot(data = avg_df, x = 'ROI', y = 'exponent', 
                                order = self.ROI_keys,
                                cut=0, inner='box', palette = self.group_color_codes['sub-{sj}'.format(sj = pp)], 
                                linewidth=1.8, ax = axis)

            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            sns.stripplot(data = avg_df, x = 'ROI', y = 'exponent', 
                            order = self.ROI_keys,
                            color="white", alpha=0.5)
            plt.xticks(fontsize = 10)
            plt.yticks(fontsize = 10)

            plt.xlabel('ROI',fontsize = 15,labelpad=18)
            plt.ylabel('Exponent',fontsize = 15,labelpad=18)
            plt.ylim(0, 1)

            fig.savefig(op.join(figures_pth, op.split(fig_name)[-1].replace('flatmap','violinplot').replace('sub-{sj}'.format(sj = pp),'sub-GROUP')))


    def plot_pa(self, participant_list = [], group_estimates = {}, ses = 'ses-mean',  run_type = 'mean',
                        figures_pth = None, model_name = 'gauss', n_bins_colors = 256, max_x_lim = 5, angle_thresh = 3*np.pi/4):
        
        # make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.outputdir, 'polar_angle', self.MRIObj.params['mri']['fitting']['pRF']['fit_folder'])

        ## loop over participants in list
        for pp in participant_list:

            ## make sub specific folder
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp), ses)
            os.makedirs(sub_figures_pth, exist_ok=True)

            ## use RSQ as alpha level for flatmaps
            r2 = group_estimates['sub-{sj}'.format(sj = pp)]['r2']
            alpha_level = mri_utils.normalize(np.clip(r2, 0, .6)) # normalize 
            alpha_level[np.where((np.isnan(r2)))[0]] = np.nan
               
            ## position estimates
            xx = group_estimates['sub-{sj}'.format(sj = pp)]['x']
            yy = group_estimates['sub-{sj}'.format(sj = pp)]['y']

            ## calculate polar angle 
            complex_location = xx + yy * 1j 
            
            polar_angle = np.angle(complex_location)
            polar_angle_norm = ((polar_angle + np.pi) / (np.pi * 2.0)) # normalize PA between 0 and 1
            polar_angle_norm[np.where((np.isnan(r2)))[0]] = np.nan

            # get matplotlib color map from segmented colors
            PA_cmap = plot_utils.make_colormap(colormap = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb',
                                                        '#3d549f','#655099','#ad5a9b','#dd3933'], bins = n_bins_colors, 
                                                        cmap_name = 'PA_mackey_costum',
                                                        discrete = False, add_alpha = False, return_cmap = True)

            #### plot flatmap ###
            # of PA 
            flatmap = plot_utils.make_raw_vertex_image(polar_angle_norm, 
                                                        cmap = PA_cmap, vmin = 0, vmax = 1, 
                                                        data2 = alpha_level, 
                                                        vmin2 = 0, vmax2 = 1, 
                                                        subject = self.pysub['sub-{pp}'.format(pp = pp)], 
                                                        data2D = True)

            fig_name = op.join(sub_figures_pth,'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_flatmap_PAalpha.png'.format(sj = pp,
                                                                                                                                        acq = self.MRIObj.acq,
                                                                                                                                        space = self.MRIObj.sj_space,
                                                                                                                                        run = run_type, 
                                                                                                                                        model = model_name))
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            print('saving %s' %fig_name)
            _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

            flatmap = plot_utils.make_raw_vertex_image(polar_angle_norm, 
                                                        cmap = PA_cmap, vmin = 0, vmax = 1, 
                                                        data2 = np.ones(alpha_level.shape), 
                                                        vmin2 = 0, vmax2 = 1, 
                                                        subject = self.pysub['sub-{pp}'.format(pp = pp)], 
                                                        data2D = True)

            fig_name = op.join(sub_figures_pth,'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_flatmap_PA.png'.format(sj = pp,
                                                                                                                                        acq = self.MRIObj.acq,
                                                                                                                                        space = self.MRIObj.sj_space,
                                                                                                                                        run = run_type, 
                                                                                                                                        model = model_name))
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            print('saving %s' %fig_name)
            _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

            # also plot non-uniform color wheel
            rgb_pa = plot_utils.get_NONuniform_polar_angle(xx, yy, r2, 
                                                            angle_thresh = angle_thresh, 
                                                            rsq_thresh = 0, 
                                                            pysub = self.pysub['sub-{pp}'.format(pp = pp)])

            # make ones mask,
            ones_mask = np.ones(r2.shape)
            ones_mask[np.where((np.isnan(r2)))[0]] = np.nan

            flatmap = cortex.VertexRGB(rgb_pa[:, 0], rgb_pa[:, 1], rgb_pa[:, 2],
                                            alpha = ones_mask,
                                            subject = self.pysub['sub-{pp}'.format(pp = pp)])

            fig_name = op.join(sub_figures_pth,'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_flatmap_PAnonUNI.png'.format(sj = pp,
                                                                                                                                        acq = self.MRIObj.acq,
                                                                                                                                        space = self.MRIObj.sj_space,
                                                                                                                                        run = run_type, 
                                                                                                                                        model = model_name))
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            print('saving %s' %fig_name)
            _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

            # plot x and y separately, for sanity check
            # x values
            flatmap = cortex.Vertex2D(xx, alpha_level,
                                        self.pysub['sub-{pp}'.format(pp = pp)],
                                        vmin = -max_x_lim, vmax= max_x_lim,
                                        vmin2 = 0, vmax2= 1,
                                        cmap='BuBkRd_alpha_2D')

            fig_name = op.join(sub_figures_pth,'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_flatmap_XX.png'.format(sj = pp,
                                                                                                                                        acq = self.MRIObj.acq,
                                                                                                                                        space = self.MRIObj.sj_space,
                                                                                                                                        run = run_type, 
                                                                                                                                        model = model_name))
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            print('saving %s' %fig_name)
            _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)

            # y values
            flatmap = cortex.Vertex2D(yy, alpha_level,
                                        self.pysub['sub-{pp}'.format(pp = pp)],
                                        vmin = -max_x_lim, vmax= max_x_lim,
                                        vmin2 = 0, vmax2= 1,
                                        cmap='BuBkRd_alpha_2D')

            fig_name = op.join(sub_figures_pth,'sub-{sj}_task-pRF_acq-{acq}_space-{space}_run-{run}_model-{model}_flatmap_XX.png'.format(sj = pp,
                                                                                                                                        acq = self.MRIObj.acq,
                                                                                                                                        space = self.MRIObj.sj_space,
                                                                                                                                        run = run_type, 
                                                                                                                                        model = model_name))
            if self.pRFModelObj.fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            print('saving %s' %fig_name)
            _ = cortex.quickflat.make_png(fig_name, flatmap, recache=False,with_colorbar=True,with_curvature=True,with_sulci=True,with_labels=False)


            ## plot the colorwheels as figs
            
            # non uniform colorwheel
            plot_utils.plot_pa_colorwheel(resolution=800, angle_thresh = angle_thresh, cmap_name = 'hsv', 
                                            continuous = True, fig_name = op.join(sub_figures_pth, 'hsv'))

            # uniform colorwheel, continuous
            plot_utils.plot_pa_colorwheel(resolution=800, angle_thresh = np.pi, 
                                                    cmap_name = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb','#3d549f','#655099','#ad5a9b','#dd3933'], 
                                            continuous = True, fig_name = op.join(sub_figures_pth, 'PA_mackey'))

            # uniform colorwheel, discrete
            plot_utils.plot_pa_colorwheel(resolution=800, angle_thresh = np.pi, 
                                                    cmap_name = ['#ec9b3f','#f3eb53','#7cb956','#82cbdb','#3d549f','#655099','#ad5a9b','#dd3933'], 
                                            continuous = False, fig_name = op.join(sub_figures_pth, 'PA_mackey'))


class FAViewer(pRFViewer):


    def __init__(self, MRIObj, outputdir = None, pRFModelObj = None, FAModelObj = None, pysub = 'hcp_999999', use_sub_rois = True, use_atlas_rois = True, combine_ses = True):
        
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
        use_sub_rois: bool
            if True, will try to find 'hcp_999999_sub-X' by default, if doesnt exist then uses 'hcp_999999'
        use_atlas: bool
            if we want to use the glasser atlas ROIs instead (this is, from the keys conglomerate defined in the params yml)
        combine_ses: bool
            if we want to combine runs from different sessions (relevant for fitting of average across runs)
        """

        # need to initialize parent class (Model), indicating output infos
        super().__init__(MRIObj = MRIObj, outputdir = outputdir, pRFModelObj = pRFModelObj, pysub = pysub, 
                            use_sub_rois = use_sub_rois, use_atlas_rois = use_atlas_rois, combine_ses = combine_ses)

        self.FAModelObj = FAModelObj


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
        keys = self.get_prf_estimate_keys(prf_model_name = prf_model_name)

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
        

    