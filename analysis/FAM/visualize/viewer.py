import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
from matplotlib import cycler
import pandas as pd
import seaborn as sns
import yaml

import ptitprince as pt # raincloud plots
import matplotlib.patches as mpatches
from  matplotlib.ticker import FuncFormatter

import cortex

import subprocess

from FAM.utils.plot import PlotUtils

class Viewer:

    def __init__(self, MRIObj, outputdir = None, pysub = 'hcp_999999', use_atlas = None):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
        outputdir: str
            path to save plots
        pysub: str
            basename of pycortex subject folder, where we drew all ROIs, sulci etc 
        use_atlas: str
            If we want to use atlas ROIs (ex: glasser, wang) or not [default].
        """

        # set data object to use later on
        self.MRIObj = MRIObj

        # if output dir not defined, then make it in derivatives
        if outputdir is None:
            self.outputdir = op.join(self.MRIObj.derivatives_pth,'plots')
        else:
            self.outputdir = outputdir
            
        # number of participants to plot
        self.nr_pp = len(self.MRIObj.sj_num)

        # pycortex subject
        if self.MRIObj.sj_space in ['fsnative']: # if using subject specific surfs
            self.pysub = self.MRIObj.sj_space
            self.use_fs_label = True
        else:
            self.pysub = pysub
            self.use_fs_label = False

        ## set variables useful when loading ROIs
        if use_atlas is None:
            self.plot_key = self.MRIObj.sj_space 
            self.annot_filename = ''
        else:
            self.plot_key = use_atlas
            self.annot_filename = self.MRIObj.atlas_annot[self.plot_key ]
        
        self.use_atlas = use_atlas

        ## get vertices for each relevant ROI
        self.ROIs_dict = {}

        # if we are using atlas ROIs, then can already load here and avoid further reloading
        if isinstance(self.use_atlas, str):
            self.ROIs_dict[self.use_atlas] = self.MRIObj.mri_utils.get_ROIs_dict(sub_id = None, pysub = self.pysub, use_atlas = self.use_atlas, 
                                                            annot_filename = self.annot_filename, hemisphere = 'BH',
                                                            ROI_labels = self.MRIObj.params['plotting']['ROIs'][self.plot_key],
                                                            freesurfer_pth = self.MRIObj.freesurfer_pth, 
                                                            use_fs_label = self.use_fs_label)

        # set some generic variables useful for plotting
        self.bar_cond_colors = self.MRIObj.params['plotting']['cond_colors']
        self.ROI_pallete = self.MRIObj.params['plotting']['ROI_pal']
        self.rsq_threshold_plot = self.MRIObj.params['plotting']['rsq_threshold']

        # initialize utilities class
        self.plot_utils = PlotUtils() 
        
        ## adding fun palettes for plots --> taken from ewenme's ghibli package for R/ggplot2.
        self.ghibli_palettes = {'MarnieLight1':cycler('color', ['#95918E','#AF9699','#80C7C9','#8EBBD2','#E3D1C3','#B3DDEB','#F3E8CC']),
        'MarnieMedium1':cycler('color', ['#28231D','#5E2D30','#008E90','#1C77A3','#C5A387','#67B8D6','#E9D097']),
        'MarnieDark1':cycler('color', ['#15110E','#2F1619','#004749','#0E3B52','#635143','#335D6B','#73684C']),
        'MarnieLight2':cycler('color', ['#8E938D','#94A39C','#97B8AF','#A2D1BD','#C0CDBC','#ACD2A3','#E6E58B']),
        'MarnieMedium2':cycler('color', ['#1D271C','#274637','#2C715F','#44A57C','#819A7A','#58A449','#CEC917']),
        'MarnieDark2':cycler('color', ['#0E130D','#14231C','#17382F','#22513D','#404D3C','#2C5223','#66650B']),
        'PonyoLight':cycler('color', ['#A6A0A0','#ADB7C0','#94C5CC','#F4ADB3','#EEBCB1','#ECD89D','#F4E3D3']),
        'PonyoMedium':cycler('color', ['#4C413F','#5A6F80','#278B9A','#E75B64','#DE7862','#D8AF39','#E8C4A2']),
        'PonyoDark':cycler('color', ['#262020','#2D3740','#14454C','#742D33','#6E3C31','#6C581D','#746353']),
        'LaputaLight':cycler('color', ['#898D90','#8D93A1','#9F99B5','#AFACC9','#D7CADE','#DAEDF3','#F7EABD']),
        'LaputaMedium':cycler('color', ['#14191F','#1D2645','#403369','#5C5992','#AE93BE','#B4DAE5','#F0D77B']),
        'LaputaDark':cycler('color', ['#090D10','#0D1321','#1F1935','#2F2C49','#574A5E','#5A6D73','#776A3D']),
        'MononokeLight':cycler('color', ['#838A90','#BA968A','#9FA7BE','#B3B8B1','#E7A79B','#F2C695','#F5EDC9']),
        'MononokeMedium':cycler('color', ['#06141F','#742C14','#3D4F7D','#657060','#CD4F38','#E48C2A','#EAD890']),
        'MononokeDark':cycler('color', ['#030A10','#3A160A','#1F273E','#333831','#67271B','#724615','#756D49']),
        'SpiritedLight':cycler('color', ['#8F9297','#9A9C97','#C19A9B','#C7C0C8','#B4DCF5','#E1D7CB','#DBEBF8']),
        'SpiritedMedium':cycler('color', ['#1F262E','#353831','#833437','#8F8093','#67B9E9','#C3AF97','#B7D9F2']),
        'SpiritedDark':cycler('color', ['#0F1217','#1A1C17','#411A1B','#474048','#345C75','#61574B','#5B6B78']),
        'YesterdayLight':cycler('color', ['#768185','#7E8C97','#88988D','#9DAFC3','#B1D5BB','#ECE28B','#C3DAEA']),
        'YesterdayMedium':cycler('color', ['#061A21','#132E41','#26432F','#4D6D93','#6FB382','#DCCA2C','#92BBD9']),
        'YesterdayDark':cycler('color', ['#030E12','#0B1924','#15251A','#2A3C50','#3E6248','#796F18','#506777']),
        'KikiLight':cycler('color', ['#8E8C8F','#9A9AA2','#D98594','#86C2DA','#D0C1AA','#C0DDE1','#E9DBD0']),
        'KikiMedium':cycler('color', ['#1C1A1F','#333544','#B50A2A','#0E84B4','#9E8356','#7EBAC2','#D1B79E']),
        'KikiDark':cycler('color', ['#0E0C0F','#1A1A22','#590514','#06425A','#50412A','#405D61','#695B50']),
        'TotoroLight':cycler('color', ['#85898A','#959492','#AC9D96','#A8A6A9','#A1B1C8','#D6C0A9','#DCD3C4']),
        'TotoroMedium':cycler('color', ['#0A1215','#2D2A25','#583B2B','#534C53','#446590','#AD8152','#BBA78C']),
        'TotoroDark':cycler('color', ['#05090A','#151412','#2C1D16','#282629','#213148','#564029','#5C5344'])
        }

    def set_palette(self, palette = ''):
        try: 
            plt.rcParams['axes.prop_cycle'] = self.ghibli_palettes[palette]
        except:
            raise Exception('Palette not available.')

    def load_ROIs_dict(self, sub_id = None, hemisphere = 'BH'):

        """
        Load ROIs dict, for the participant
        Wrapper to avoid unnecessary repeating code

        Parameters
        ----------
        sub_id : str
            participant ID
        """

        if isinstance(self.use_atlas, str): # if we are using atlas ROIs
            pp_ROI_dict = self.ROIs_dict[self.use_atlas]
        else: 
            if 'sub-{sj}'.format(sj = sub_id) not in list(self.ROIs_dict.keys()): # if using participant specifc ROIs, load them if not loaded yet
                self.ROIs_dict['sub-{sj}'.format(sj = sub_id)] = self.MRIObj.mri_utils.get_ROIs_dict(sub_id = sub_id, pysub = self.pysub, use_atlas = self.use_atlas, 
                                                                                            annot_filename = self.annot_filename, hemisphere = 'BH',
                                                                                            ROI_labels = self.MRIObj.params['plotting']['ROIs'][self.plot_key],
                                                                                            freesurfer_pth = self.MRIObj.freesurfer_pth, 
                                                                                            use_fs_label = self.use_fs_label)
            pp_ROI_dict = self.ROIs_dict['sub-{sj}'.format(sj = sub_id)]

        if hemisphere == 'BH':
            return pp_ROI_dict
        else:
            # number of vertices in one hemisphere (for bookeeping) 
            if self.use_fs_label:
                # load surface vertices, for each hemi, as dict
                n_verts_dict = self.MRIObj.mri_utils.load_FS_nverts_nfaces(sub_id = sub_id, freesurfer_pth = self.MRIObj.freesurfer_pth, return_faces = False)
                hemi_vert_num = n_verts_dict['lh']
            else:
                hemi_vert_num = cortex.db.get_surfinfo(self.pysub).left.shape[0] 

            # iterate over rois and get vertices
            hemi_pp_ROI_dict = {Rkey: (verts[np.where(verts < hemi_vert_num)[0]] if hemisphere == 'LH' else verts[np.where(verts >= hemi_vert_num)[0]]) for Rkey, verts in pp_ROI_dict.items()}

            return hemi_pp_ROI_dict

    def get_pysub_name(self, sub_id = None):

        """
        Get pysubject folder name to use when plotting
        depends on usage of atlas vs sub-specific ROIs

        Parameters
        ----------
        sub_id : str
            participant ID
        """
        if isinstance(self.use_atlas, str):
            sub_pysub = self.pysub
        else:
            # subject pycortex folder
            sub_pysub = 'sub-{pp}_{ps}'.format(ps = self.pysub, pp = sub_id)
        
        return sub_pysub

    def plot_rsq(self, participant_list = [], group_estimates = {}, ses = 'mean',  run_type = 'mean',
                        model_name = 'gauss', task = 'pRF', figures_pth = None, vmin1 = 0, vmax1 = .8,
                        fit_hrf = True, save_flatmap = False, angles2plot_list = ['lateral_left', 'lateral_right', 'back', 'medial_right', 'medial_left']):
        
        """
        plot R2 estimates of model fit (for either task)

        Parameters
        ----------
        participant_list: list
            list with participant ID 
        group_estimates: dict
            estimates for all participants
        """
        
        # make output folder for figures
        if figures_pth is None:
            figures_pth = op.join(self.outputdir, 'rsq')

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
                            'sub-{sj}_task-{tsk}_acq-{acq}_space-{space}_ses-{ses}_run-{run}_model-{model}_flatmap_RSQ.png'.format(sj=pp, tsk = task,
                                                                                                            acq = self.MRIObj.acq, space = self.MRIObj.sj_space,
                                                                                                            ses=ses, run = run_type, model = model_name))

            # if we fitted hrf, then add that to fig name
            if fit_hrf:
                fig_name = fig_name.replace('.png','_withHRF.png') 

            # if we want to save flatmap (note - requires flattened surface overlay file)
            if save_flatmap:
                self.plot_utils.plot_flatmap(group_estimates['sub-{sj}'.format(sj = pp)]['r2'], 
                                            pysub = self.get_pysub_name(sub_id = pp), cmap = 'hot', 
                                            vmin1 = vmin1, vmax1 = vmax1, 
                                            fig_abs_name = fig_name)
            ## plot inflated
            elif len(angles2plot_list) > 0:
                self.plot_inflated(pp, est_arr1 = group_estimates['sub-{sj}'.format(sj = pp)]['r2'], 
                                    vmin1 = vmin1, vmax1 = vmax1,
                                    cmap='hot', fig_abs_name = fig_name.replace('_flatmap', ''), 
                                    recache = True, overlays_visible=[], cmap2str = True, 
                                    angles2plot_list = angles2plot_list, 
                                    unfold_type = 'inflated')

            ## get estimates per ROI
            pp_roi_df = self.MRIObj.mri_utils.get_estimates_roi_df(pp, estimates_pp = group_estimates['sub-{sj}'.format(sj = pp)], 
                                                ROIs_dict = pp_ROI_dict, 
                                                est_key = 'r2', model = model_name)

            #### plot distribution ###
            fig, ax1 = plt.subplots(1,1, figsize=(20,7.5), dpi=100, facecolor='w', edgecolor='k')

            v1 = pt.RainCloud(data = pp_roi_df, move = .2, alpha = .9,
                        x = 'ROI', y = 'value', pointplot = False, hue = 'ROI',
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
            plt.ylabel('RSQ',fontsize = 20,labelpad=18)
            plt.ylim(0,1)
            fig.savefig(fig_name.replace('flatmap','violinplot'))

            ## concatenate average per participant, to make group plot
            avg_roi_df = pd.concat((avg_roi_df, pp_roi_df))

        # if we provided several participants, make group plot
        if len(participant_list) > 1:

            fig, ax1 = plt.subplots(1,1, figsize=(15,5), dpi=100, facecolor='w', edgecolor='k')

            v1 = sns.pointplot(data = avg_roi_df.groupby(['sj', 'ROI'])['value'].mean().reset_index(),
                                x = 'ROI', y = 'value', color = 'k', markers = 'D', #scale = 1, 
                                palette = self.ROI_pallete, order = pp_ROI_dict.keys(), 
                                dodge = False, join = False, ci=68, ax = ax1)
            v1.set(xlabel=None)
            v1.set(ylabel=None)
            plt.margins(y=0.025)
            sns.stripplot(data = avg_roi_df.groupby(['sj', 'ROI'])['value'].mean().reset_index(), 
                          x = 'ROI', y = 'value', #hue = 'sj', palette = sns.color_palette("husl", len(participant_list)),
                            order = pp_ROI_dict.keys(),
                            color="gray", alpha=0.5, ax=ax1)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)

            plt.xlabel('ROI',fontsize = 20,labelpad=18)
            plt.ylabel('RSQ',fontsize = 20,labelpad=18)
            plt.ylim(0,1)

            fig.savefig(op.join(figures_pth, op.split(fig_name)[-1].replace('flatmap','violinplot').replace('sub-{sj}'.format(sj = pp),'sub-GROUP')))

    def plot_inflated(self, participant, est_arr1 = None, est_arr2 = None, verts = None, 
                            vmin1 = 0, vmax1 = .8, vmin2 = None, vmax2 = None, 
                            cmap='hot', fig_abs_name = None, recache = False, with_colorbar = True,
                            with_curvature = True, with_sulci = True, with_labels=False,
                            curvature_brightness = 0.4, curvature_contrast = 0.1, with_rois = True,
                            overlays_visible=[],
                            cmap2str = True, angles2plot_list = ['lateral_left', 'lateral_right'], unfold_type = 'inflated'):

        # first check if sub in filestore
        self.plot_utils.add_FSsub_db('sub-{sj}'.format(sj = participant), 
                                    cx_subject = self.get_pysub_name(sub_id = participant), 
                                    freesurfer_subject_dir = self.MRIObj.freesurfer_pth)
        
        # get pycortex vertex data object
        inflated_surf = self.plot_utils.prepare_inflated(est_arr1, est_arr2 = est_arr2, verts = verts,
                                                         vmin2 = vmin2, vmax2 = vmax2, 
                                                        pysub = self.get_pysub_name(sub_id = participant), cmap = cmap, 
                                                        vmin1 = vmin1, vmax1 = vmax1, cmap2str = cmap2str)
        
        ## pop-up web browser to check 
        #cortex.webshow(inflated_surf, recache = True)
        
        ## get dict with browser parameters to make figures
        angles2plot_dict = {key: self.MRIObj.params['plotting']['webview']['angle_params'][key] for key in angles2plot_list}

        ## save inflated views
        self.save_inflated_3Dviews(inflated_surf, viewer_angles_dict = angles2plot_dict, 
                                    base_name = fig_abs_name.replace('.png', ''), 
                                    unfold_type = unfold_type, overlays_visible = overlays_visible)

    def save_inflated_3Dviews(self, flatmap, viewer_angles_dict = None, base_name = None, unfold_type = 'inflated',
                                    overlays_visible=['sulci']):

        """
        Function to make and save inflated 3D views 
        from different views
        Note - if running in notebook, will kill kernel
        """

        if viewer_angles_dict is None:
            viewer_angles_dict = self.MRIObj.params['plotting']['webview']['angle_params']

        list_angles = list(viewer_angles_dict.items())
        #list_surfaces = [(unfold_type, self.MRIObj.params['plotting']['webview']['unfold_params'][unfold_type]) for i in range(len(viewer_angles_dict))]
        list_surfaces = [unfold_type for i in range(len(viewer_angles_dict))]

        # save inflated 3D screenshots 
        for ind, surf_name in enumerate(list_surfaces):
            cortex.export.save_3d_views(flatmap, base_name = base_name,
                                        list_angles = [list_angles[ind]], list_surfaces = [surf_name],        
                                        viewer_params=dict(labels_visible = [],
                                                            overlays_visible = overlays_visible, recache=True),
                                        size=(1024 * 4, 768 * 4), trim=True, sleep=10)

        # ## make multipanel figure
        # for filename, angle in zip(filenames, list_angles):
        #     plt.imshow(plt.imread(filename))
        #     plt.axis('off')
        #     plt.title(angle)
        #     plt.show()

            
    def add_data2FSsurface(self, participant, data_arr = None, mask_arr = None,
                                surf_name = '', freesurfer_pth = None, overwrite = False,
                                vmin = 0, vmax = 1, cmap = 'hot', n_bins = 20):

        """
        Add subject data as a custom surface file
        in their freesurfer directory (or FS-like directory)

        Parameters
        ----------
        participant : str 
            subject ID
        data_arr: array
            data array that we want to add to a specific surface
        mask_arr: array (bool)
            if given, will be used to mask the data to be added to the surface
        surf_name: str
            name to give this new custom surface
        freesurfer_pth: str
            absolute path to freesurfer files

        """

        if freesurfer_pth is None:
            freesurfer_pth = self.MRIObj.freesurfer_pth 

        # subject path for freesurfer files
        sub_freesurfer_pth = op.join(freesurfer_pth, 'sub-{sj}'.format(sj = participant))

        if not op.isdir(sub_freesurfer_pth):
            raise ValueError('Subject directory {sdir} DOES NOT EXIST'.format(sdir = sub_freesurfer_pth))
        else:
            sub_custom_surf_pth = op.join(sub_freesurfer_pth, 'custom_surf')
            print('Saving custom surfaces in {cdir}'.format(cdir = sub_custom_surf_pth))
            os.makedirs(sub_custom_surf_pth, exist_ok=True)

        # load surface vertices, for each hemi, as dict
        n_verts_dict, n_faces_dict = self.MRIObj.mri_utils.load_FS_nverts_nfaces(sub_id = participant, 
                                                                                freesurfer_pth = freesurfer_pth, 
                                                                                return_faces = True)

        ## mask data, if mask was provided
        if mask_arr is not None:
            mask_arr = mask_arr.astype(bool)
            data_out = np.zeros_like(data_arr, dtype=float)
            data_out[mask_arr] = data_arr[mask_arr]
            data_out[~mask_arr] = vmin - 1 # to hide non-relevant vertices from surface
        else:
            data_out = data_arr

        ## actually add curvature
        # left hemi
        lh_surf = op.join(sub_custom_surf_pth, 'lh.{surf_name}'.format(surf_name = surf_name))

        if op.exists(lh_surf) and not overwrite:
            print('left hemi surface file already in dir, skipping')
        else:
            self.MRIObj.mri_utils.FS_write_curv(fn = lh_surf, 
                                                curv = data_out[:n_verts_dict['lh']], 
                                                fnum = n_faces_dict['lh'])

        # right hemi
        rh_surf = op.join(sub_custom_surf_pth, 'rh.{surf_name}'.format(surf_name = surf_name))

        if op.exists(rh_surf) and not overwrite:
            print('left hemi surface file already in dir, skipping')
        else:
            self.MRIObj.mri_utils.FS_write_curv(fn = rh_surf, 
                                                curv = data_out[n_verts_dict['lh']:],
                                                fnum = n_faces_dict['rh'])
            
        ## now make overlay_custom str
        # which will have cmap levels to use when loading surface in freeview
        cmap_obj = self.plot_utils.make_colormap(colormap = cmap, bins = 256, add_alpha = False, invert_alpha = False, 
                                                cmap_name = 'custom_surf', discrete = False, return_cmap = True)
        cmap_arr = self.plot_utils.cmap2array(cmap_obj, n_colors = n_bins, vmin = 0, vmax = 1, include_alpha = False)

        # colormap steps (within data range of values)
        data_steps = np.linspace(vmin, vmax, n_bins)

        ## actually save overlay custom str
        # left hemi
        overlay_filename = op.join(sub_custom_surf_pth, '{surf_name}_overlay'.format(surf_name = surf_name))
        
        if op.exists(overlay_filename) and not overwrite:
            print('overlay custom cmap file already in dir, skipping')
        else:
            overlay_custom_str = ['{val},{r},{g},{b}'.format(val = np.round(dval,2), 
                                                    r = int(cmap_arr[i][0]*255),
                                                    g = int(cmap_arr[i][1]*255),
                                                    b = int(cmap_arr[i][2]*255)) for i, dval in enumerate(data_steps)]
            overlay_custom_str = ','.join(overlay_custom_str)

            self.plot_utils.save_str2file(txt = overlay_custom_str, filename = overlay_filename)

        # return full path to surfaces
        return {'hemi-L': lh_surf, 'hemi-R': rh_surf}, overlay_filename

    def open_surf_freeview(self, participant, surf_names = [], freesurfer_pth = None, surf_type = ['inflated'], screenshot_filename = None, 
                                show_colorbar = True):

        """
        Write and call freeview bash command
        to open the specific participant surface(s) + the corresponding overlay custom cmap

        Parameters
        ----------
        participant : str 
            subject ID
        surf_name: list
            list of strs with custom surface names to load
        freesurfer_pth: str
            absolute path to freesurfer files
        surf_type: list
            list of str with type of surface to load (inflated [default], pial, sphere)
        """

        if freesurfer_pth is None:
            freesurfer_pth = self.MRIObj.freesurfer_pth 

        # subject path for freesurfer files
        sub_freesurfer_pth = op.join(freesurfer_pth, 'sub-{sj}'.format(sj = participant))

        if not op.isdir(sub_freesurfer_pth):
            raise ValueError('Subject directory {sdir} DOES NOT EXIST'.format(sdir = sub_freesurfer_pth))
        else:
            sub_custom_surf_pth = op.join(sub_freesurfer_pth, 'custom_surf')
            print('Loading custom surfaces in {cdir}'.format(cdir = sub_custom_surf_pth))

        ## write command
        working_string = """#!/bin/bash

export SUBJECTS_DIR=$DATADIR

cd $DATADIR

freeview -f """

        for stype in surf_type: # iterate over surface types

            for ind, surf_file in enumerate(surf_names): # iterate over data surfaces

                # load appropriate cmap values
                ovfile = open(op.join(sub_custom_surf_pth, '{surf_name}_overlay'.format(surf_name = surf_file)), 'r')
                ovfile_str = ovfile.read()

                # add to command
                fs_cmd = 'sub-$SJ_NR/surf/lh.{stype}:overlay={sfile}:overlay_custom={ovfile} '.format(stype = stype,
                                                            sfile = op.join(sub_custom_surf_pth, 'lh.{surf_name}'.format(surf_name = surf_file)),
                                                            ovfile = ovfile_str
                                                            )
                fs_cmd += fs_cmd.replace('lh.', 'rh.') # add right hemisphere as well

                working_string += fs_cmd

        ## replace folder path and sub number
        working_string = working_string.replace('$DATADIR', freesurfer_pth) 
        working_string = working_string.replace('$SJ_NR', participant) 

        # if we want to save image --> should turn into function. also need to adapt for when loading several overlay layers, and want to screenshot each
        if screenshot_filename is not None:

            # load camera params
            cam_params = self.MRIObj.params['plotting']['freeview']['camera_params']

            ## add relevant params to command string
            photo_cmd = working_string+' -ss {image_name} --colorscale \
                --camera Azimuth {cam_azimuth} \
                    Zoom {cam_zoom} Elevation {cam_elevation} \
                        Roll {cam_roll} '.format(image_name = screenshot_filename.replace('.png', '_backview.png'),
                                                 cam_azimuth = cam_params['azimuth']['back'],
                                                 cam_zoom = cam_params['zoom']['back'],
                                                 cam_elevation = cam_params['elevation']['back'],
                                                 cam_roll = cam_params['roll']['back'])
            ## actually call command
            os.system(photo_cmd) 
        else:
            ## actually call command
            if show_colorbar:
                working_string += ' --colorscale'
            os.system(working_string)
        
    def convert_pix2dva(self, val_pix):

        """
        Convert pixel value to dva
        """
        return val_pix * self.MRIObj.mri_utils.dva_per_pix(height_cm = self.MRIObj.params['monitor']['height'], 
                                                        distance_cm = self.MRIObj.params['monitor']['distance'], 
                                                        vert_res_pix = self.MRIObj.screen_res[1])