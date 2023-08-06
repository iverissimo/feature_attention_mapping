

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

        # ### POLAR ANGLE ####
        # self.plot_pa(participant_list = participant_list, group_estimates = final_estimates, ses = ses, run_type = run_type,
        #                                 model_name = prf_model_name, 
        #                                 n_bins_colors = 256, max_x_lim = max_ecc_ext, angle_thresh = 3*np.pi/4)

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
                                        pysub = self.pysub, cmap = 'BuBkRd', 
                                        vmin1 = vmin1, vmax1 = vmax1, 
                                        fig_abs_name = fig_name)
            
            ## iterate over models
            for ind, mod_name in enumerate(prf_model_list):

                ## concatenate estimates per ROI per participant, to make group plot
                avg_roi_df = pd.concat((avg_roi_df,
                                        self.MRIObj.mri_utils.get_estimates_roi_df(pp, estimates_pp = final_estimates[mod_name][it_key[ind]]['sub-{sj}'.format(sj = pp)], 
                                                                            ROIs_dict = self.ROIs_dict, 
                                                                            est_key = 'r2', model = mod_name,
                                                                            iterative = it_bool[ind])
                                        ))
                                    
            fig, ax1 = plt.subplots(1,1, figsize=(15,5), dpi=100, facecolor='w', edgecolor='k')

            v1 = sns.violinplot(data = avg_roi_df[avg_roi_df['sj'] == 'sub-{sj}'.format(sj = pp)], 
                                x = 'ROI', y = 'value', hue = plot_hue,
                                order = self.ROIs_dict.keys(),
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
                                order = self.ROIs_dict.keys(),
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
                        vmin1 = {'ecc': 0, 'size': 0}, vmax1 = {'ecc': 5.5, 'size': 15}):
        
        ## make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.figures_pth, 'ecc_size', self.pRFModelObj.fitfolder['pRF'])
            
        # dataframe to store binned values
        avg_bin_df = pd.DataFrame()
        
        ## loop over participants in list
        for pp in participant_list:

            # make path to save sub-specific figures
            sub_figures_pth = op.join(figures_pth, 'sub-{sj}'.format(sj = pp))
            os.makedirs(sub_figures_pth, exist_ok=True)
                
            ## use RSQ as alpha level for flatmaps
            r2 = group_estimates['sub-{sj}'.format(sj = pp)]['r2']
            alpha_level = self.MRIObj.mri_utils.normalize(np.clip(r2, 0, .6)) # normalize 
                
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
                                        pysub = self.pysub, cmap = 'viridis', 
                                        vmin1 = vmin1['ecc'], vmax1 = vmax1['ecc'],
                                        est_arr2 = alpha_level,
                                        vmin2 = 0, vmax2 = 1, 
                                        fig_abs_name = fig_name)
            
            ## get SIZE estimates 
            size_fwhmaxmin = self.pRFModelObj.fwhmax_fwatmin(model_name, 
                                                                group_estimates['sub-{sj}'.format(sj = pp)])
            
            self.plot_utils.plot_flatmap(size_fwhmaxmin[0], 
                                        pysub = self.pysub, cmap = 'cubehelix', 
                                        vmin1 = vmin1['size'], vmax1 = vmax1['size'],
                                        est_arr2 = alpha_level,
                                        vmin2 = 0, vmax2 = 1, 
                                        fig_abs_name = fig_name.replace('ECC', 'SIZE-fwhmax'))
            
            ## GET values per ROI ##
            ecc_pp_roi_df = self.MRIObj.mri_utils.get_estimates_roi_df(pp, eccentricity, 
                                                                    ROIs_dict = self.ROIs_dict, 
                                                                    model = model_name)
    
            size_pp_roi_df = self.MRIObj.mri_utils.get_estimates_roi_df(pp, size_fwhmaxmin[0], 
                                                                    ROIs_dict = self.ROIs_dict, 
                                                                    model = model_name)
            
            rsq_pp_roi_df = self.MRIObj.mri_utils.get_estimates_roi_df(pp, r2, 
                                                                    ROIs_dict = self.ROIs_dict, 
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
                        scatter=True, palette = self.ROI_pallete) #, markers=['^', 's', 'o', 'v', 'D', 'h', 'P', '.', ','])

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

            fig_name = fig_name.replace('ECC', 'ECCvsSIZE_UNbinned')
            fig2.savefig(fig_name, dpi=100,bbox_inches = 'tight')

            ## bin it, for cleaner plot
            for r_name in self.ROIs_dict.keys()   :

                mean_x, _, mean_y, _ = self.MRIObj.mri_utils.get_weighted_bins (df_ecc_siz.loc[(df_ecc_siz['ROI'] == r_name)],
                                                                                x_key = 'ecc', y_key = 'size', 
                                                                                weight_key = 'rsq', sort_key = 'ecc', n_bins = n_bins_dist)

                avg_bin_df = pd.concat((avg_bin_df,
                                        pd.DataFrame({ 'sj': np.tile('sub-{sj}'.format(sj = pp), len(mean_x)),
                                                    'ROI': np.tile(r_name, len(mean_x)),
                                                    'ecc': mean_x,
                                                    'size': mean_y
                                        })))

            ##### plot binned df #########
            sns.set(font_scale=1.3)
            sns.set_style("ticks")

            g = sns.lmplot(x="ecc", y="size", hue = 'ROI', data = avg_bin_df.loc[avg_bin_df['sj'] == 'sub-{sj}'.format(sj = pp)], 
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

            g = sns.lmplot(x="ecc", y="size", hue = 'ROI', data = avg_bin_df, 
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
            alpha_level = self.MRIObj.mri_utils.normalize(np.clip(r2, 0, .6)) # normalize 

            self.plot_utils.plot_flatmap(group_estimates['sub-{sj}'.format(sj = pp)]['ns'], 
                                        pysub = self.pysub, cmap = 'magma', 
                                        vmin1 = vmin1, vmax1 = vmax1, 
                                        est_arr2 = alpha_level,
                                        vmin2 = 0, vmax2 = 1, 
                                        fig_abs_name = fig_name)
            
            ## GET values per ROI ##
            ns_pp_roi_df = self.MRIObj.mri_utils.get_estimates_roi_df(pp, group_estimates['sub-{sj}'.format(sj = pp)]['ns'], 
                                                                    ROIs_dict = self.ROIs_dict, 
                                                                    model = model_name)
    
            rsq_pp_roi_df = self.MRIObj.mri_utils.get_estimates_roi_df(pp, r2, 
                                                                    ROIs_dict = self.ROIs_dict, 
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
            handles = [mpatches.Patch(color = self.ROI_pallete[k], label = k) for k in self.ROIs_dict.keys()]
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
                                palette = self.ROI_pallete, order = self.ROIs_dict.keys(), 
                                dodge = False, join = False, ci=68, ax = ax1)
            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            sns.stripplot(data = avg_roi_df.groupby(['sj', 'ROI'])['exponent'].mean().reset_index(), 
                          x = 'ROI', y = 'exponent', #hue = 'sj', palette = sns.color_palette("husl", len(participant_list)),
                            order = self.ROIs_dict.keys(),
                            color="gray", alpha=0.5, ax=ax1)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)

            plt.xlabel('ROI',fontsize = 20,labelpad=18)
            plt.ylabel('Exponent',fontsize = 20,labelpad=18)
            plt.ylim(0,1)

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
        super().__init__(MRIObj = MRIObj, outputdir = outputdir, pRFModelObj = pRFModelObj, pysub = pysub, use_atlas = use_atlas)

        self.FAModelObj = FAModelObj

    def plot_spcorrelations(self, participant):

        """
        Plot split half correlations used in GLM single fit
        to make noise mask 
        """

        ## path to files
        fitpath = op.join(self.FAModelObj.outputdir, self.MRIObj.sj_space, 'sub-{sj}'.format(sj = participant))

    def plot_glmsingle_estimates(self, participant, model_type = ['A','D']):

        """
        Plot split half correlations used in GLM single fit
        to make noise mask 
        """

        ## output path to save plots
        output_pth = op.join(self.figures_pth, 'glmsingle_estimates', 'sub-{sj}'.format(sj = participant))
        os.makedirs(output_pth, exist_ok=True)

        for name in model_type:
            ## load estimates dict
            estimates_dict = self.FAModelObj.load_estimates(participant, model_type = name)

            ## plot R2 on flatmap surface ##
            r2 = estimates_dict['onoffR2'] if name == 'A' else estimates_dict['R2']

            fig_name = op.join(output_pth,
                            'R2_Model-{m}_flatmap_sub-{sj}_acq-{acq}.png'.format(sj = participant, 
                                                                                m = name,
                                                                                acq=self.MRIObj.sj_space))
            self.plot_utils.plot_flatmap(r2, 
                                        pysub = self.pysub, cmap='hot', 
                                        vmin1 = 0, vmax1 = 50, 
                                        fig_abs_name = fig_name)
            
            ## plot average betas
            avg_betas = estimates_dict['betasmd'][...,0] if name == 'A' else np.mean(estimates_dict['betasmd'], axis = -1)

            fig_name = op.join(output_pth,
                            'Betas_Model-{m}_flatmap_sub-{sj}_acq-{acq}.png'.format(sj = participant, 
                                                                                m = name,
                                                                                acq=self.MRIObj.sj_space))
            self.plot_utils.plot_flatmap(avg_betas, 
                                        pysub = self.pysub, cmap='RdBu_r', 
                                        vmin1 = -2, vmax1 = 2, 
                                        fig_abs_name = fig_name)

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
        

    