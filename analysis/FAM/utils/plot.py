from genericpath import isdir
import numpy as np
import os
from os import path as op
import pandas as pd
from tqdm import tqdm

## plotting packages
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import seaborn as sns
import cortex
from PIL import Image, ImageDraw

## imaging, processing, stats packages
import nibabel as nib
from statsmodels.stats import weightstats

from scipy.signal import periodogram


# local packages
from FAM.utils.general import Utils

class PlotUtils(Utils):

    def __init__(self):
        
        """__init__
        constructor for utilities plotting class 
            
        """

    def rose_plot(self, angles, ax = None, bins = 16, use_density = True, offset = 0, color = 'g', alpha = 0.5, **param_dict):
        
        """
        Plot polar histogram of angles on a figure's axis. 

        Parameters
        ----------
        angles : array
            angle to plot, are expected in radians
        ax: figure axis
            where to plot --> NOTE: ax must have been created using subplot_kw=dict(projection='polar')

        """

        # Wrap angles to [-pi, pi)
        angles = (angles + np.pi) % (2*np.pi) - np.pi

        # Set bins symmetrically around zero
        # To have a bin edge at zero use an even number of bins
        if bins % 2:
            bins += 1
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

        # Bin data and record counts
        count, bin = np.histogram(angles, bins=bins)

        # Compute width of each bin
        widths = np.diff(bin)

        # By default plot density (frequency potentially misleading)
        if use_density:
            # Area to assign each bin
            area = count / angles.size
            # Calculate corresponding bin radius
            radius = (area / np.pi)**.5
        else:
            radius = count
            
        # Plot data on ax
        ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths, edgecolor='0.5', fill=True, linewidth=1,color=color,alpha=alpha)

        # Set the direction of the zero angle
        ax.set_theta_offset(offset)

        # Remove ylabels, they are mostly obstructive and not informative
        ax.set_yticks([])

        label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
        ax.set_xticklabels(label)

        #return ax
    
    def get_flatmaps(self, est_arr1, est_arr2 = None, 
                                vmin1 = 0, vmax1 = .8, vmin2 = None, vmax2 = None,
                                pysub = 'hcp_999999', cmap = 'BuBkRd'):

        """
        Helper function to set and return flatmap  

        Parameters
        ----------
        est_arr1 : array
            data array
        est_arr2 : array
            data array
        cmap : str
            string with colormap name
        vmin1: int/float
            minimum value est_arr1
        vmin2: int/float
            minimum value est_arr2
        vmax1: int/float 
            maximum value est_arr1
        vmax2: int/float 
            maximum value est_arr2
        subject: str
            overlay subject name to use
        """

        # if two arrays provided, then fig is 2D
        if est_arr2 is not None:
            flatmap = cortex.Vertex2D(est_arr1, est_arr2,
                                    pysub,
                                    vmin = vmin1, vmax = vmax1,
                                    vmin2 = vmin2, vmax2 = vmax2,
                                    cmap = cmap)
        else:
            flatmap = cortex.Vertex(est_arr1, 
                                    pysub,
                                    vmin = vmin1, vmax = vmax1,
                                    cmap = cmap)

        return flatmap

    def make_raw_vertex_image(self, data1, cmap = 'hot', vmin = 0, vmax = 1, 
                          data2 = [], vmin2 = 0, vmax2 = 1, pysub = 'hcp_999999', data2D = False):  
    
        """ function to fix web browser bug in pycortex
            allows masking of data with nans
        
        Parameters
        ----------
        data1 : array
            data array
        data2 : array
            alpha array
        cmap : str
            string with colormap name (not the alpha version)
        vmin: int/float
            minimum value
        vmax: int/float 
            maximum value
        vmin2: int/float
            minimum value
        vmax2: int/float 
            maximum value
        pysub: str
            overlay subject name to use
        data2D: bool
            if we want to add alpha or not
        
        Outputs
        -------
        vx_fin : VertexRGB
            vertex object to call in webgl
        
        """
        
        # Get curvature
        curv = cortex.db.get_surfinfo(pysub, type = 'curvature', recache=False)#,smooth=1)
        # Adjust curvature contrast / color. Alternately, you could work
        # with curv.data, maybe threshold it, and apply a color map.     
        curv.data[curv.data>0] = .1
        curv.data[curv.data<=0] = -.1
        #curv.data = np.sign(curv.data.data) * .25
        
        curv.vmin = -1
        curv.vmax = 1
        curv.cmap = 'gray'
        
        # Create display data 
        vx = cortex.Vertex(data1, pysub, cmap = cmap, vmin = vmin, vmax = vmax)
        
        # Pick an arbitrary region to mask out
        # (in your case you could use np.isnan on your data in similar fashion)
        if data2D:
            data2[np.isnan(data2)] = vmin2
            norm2 = colors.Normalize(vmin2, vmax2)  
            alpha = np.clip(norm2(data2), 0, 1)
        else:
            alpha = ~np.isnan(data1) #(data < 0.2) | (data > 0.4)
        alpha = alpha.astype(np.float)
        
        # Map to RGB
        vx_rgb = np.vstack([vx.raw.red.data, vx.raw.green.data, vx.raw.blue.data])
        vx_rgb[:,alpha>0] = vx_rgb[:,alpha>0] * alpha[alpha>0]
        
        curv_rgb = np.vstack([curv.raw.red.data, curv.raw.green.data, curv.raw.blue.data])
        # do this to avoid artifacts where curvature gets color of 0 valur of colormap
        curv_rgb[:,np.where((vx_rgb > 0))[-1]] = curv_rgb[:,np.where((vx_rgb > 0))[-1]] * (1-alpha)[np.where((vx_rgb > 0))[-1]]

        # Alpha mask
        display_data = curv_rgb + vx_rgb 

        # Create vertex RGB object out of R, G, B channels
        vx_fin = cortex.VertexRGB(*display_data, pysub, curvature_brightness = 0.4, curvature_contrast = 0.1)

        return vx_fin

    def zoom_to_roi(self, pysub, roi = None, hem = 'left', margin=10.0, ax=None):

        """
        Plot zoomed in view of flatmap, around a given ROI.
        need to give it the flatmap axis as ref, so it know what to do

        Parameters
        ----------
        pysub : str
            Name of the pycortex subject
        roi: str
            name of the ROI to zoom into
        hem: str
            left or right hemisphere
        margin: float
            margin around ROI - will add/subtract to axis max and min
        ax: figure axis
            where to plot (needs to be an axis where a flatmap is already plotted)
        """

        roi_verts = cortex.get_roi_verts(pysub, roi)[roi]
        roi_map = cortex.Vertex.empty(pysub)
        roi_map.data[roi_verts] = 1

        (lflatpts, lpolys), (rflatpts, rpolys) = cortex.db.get_surf(pysub, "flat",
                                                                    nudge=True)
        sel_pts = dict(left=lflatpts, right=rflatpts)[hem]
        roi_pts = sel_pts[np.nonzero(getattr(roi_map, hem))[0],:2]

        xmin, ymin = roi_pts.min(0) - margin
        xmax, ymax = roi_pts.max(0) + margin
        
        ax.axis([xmin, xmax, ymin, ymax])

    def plot_flatmap(self, est_arr1, est_arr2 = None, verts = None, pysub = 'hcp_999999',
                        vmin1 = 0, vmax1 = .8, vmin2 = None, vmax2 = None, 
                        cmap='hot', fig_abs_name = None, recache = False, with_colorbar = True,
                        with_curvature = True, with_sulci = True, with_labels=False,
                        curvature_brightness = 0.4, curvature_contrast = 0.1, with_rois = True,
                        zoom2ROI = None, hemi_list = ['left', 'right'], figsize=(15,5), dpi=300, margin = 10):

        """
        plot flatmap of data (1D)
        with option to only show select vertices

        Parameters
        ----------
        est_arr1 : array
            data array
        est_arr2 : array
            data array
        verts: array
            list of vertices to select
        cmap : str
            string with colormap name
        vmin1: int/float
            minimum value est_arr1
        vmin2: int/float
            minimum value est_arr2
        vmax1: int/float 
            maximum value est_arr1
        vmax2: int/float 
            maximum value est_arr2
        fig_abs_name: str
            if provided, will save figure with this absolute name
        zoom2ROI: str
            if we want to zoom into an ROI, provide ROI name
        hemi_list: list/arr
            when zooming, which hemisphere to look at (can also be both)
        """

        # subselect vertices, if provided
        if verts is not None:
            surface_arr1 = np.zeros(est_arr1.shape[0])
            surface_arr1[:] = np.nan
            surface_arr1[verts] = est_arr1[verts]
            if est_arr2 is not None:
                surface_arr2 = np.zeros(est_arr2.shape[0])
                surface_arr2[:] = np.nan
                surface_arr2[verts] = est_arr2[verts]
            else:
                surface_arr2 = None
        else:
            surface_arr1 = est_arr1
            surface_arr2 = est_arr2

        if isinstance(cmap, str):
            flatmap = self.get_flatmaps(surface_arr1, est_arr2 = surface_arr2, 
                                vmin1 = vmin1, vmax1 = vmax1, vmin2 = vmin2, vmax2 = vmax2,
                                cmap = cmap, pysub = pysub)
        else:
            if surface_arr2 is None:
                data2D = False
            else:
                data2D = True
            flatmap = self.make_raw_vertex_image(surface_arr1, 
                                                cmap = cmap, 
                                                vmin = vmin1, vmax = vmax1, 
                                                data2 = surface_arr2, 
                                                vmin2 = vmin2, vmax2 = vmax2, 
                                                pysub = pysub, data2D = data2D)

        if len(hemi_list)>1 and zoom2ROI is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = figsize, dpi = dpi)
        else:
            fig, ax1 =  plt.subplots(1, figsize = figsize, dpi = dpi)

        cortex.quickshow(flatmap, fig = ax1, recache = recache, with_colorbar = with_colorbar, with_rois = with_rois,
                                with_curvature = with_curvature, with_sulci = with_sulci, with_labels = with_labels,
                                curvature_brightness = curvature_brightness, curvature_contrast = curvature_contrast)
        
        if zoom2ROI is not None:
            # Zoom on just one hemisphere
            self.zoom_to_roi(pysub, roi = zoom2ROI, hem = hemi_list[0], ax=ax1, margin = margin)

            if len(hemi_list)>1:
                cortex.quickshow(flatmap, fig = ax2, recache = recache, with_colorbar = with_colorbar, with_rois = with_rois,
                                with_curvature = with_curvature, with_sulci = with_sulci, with_labels = with_labels,
                                curvature_brightness = curvature_brightness, curvature_contrast = curvature_contrast)
                # Zoom on just one region
                self.zoom_to_roi(pysub, roi = zoom2ROI, hem = hemi_list[1], ax=ax2, margin = margin)

        # if we provide absolute name for figure, then save there
        if fig_abs_name is not None:

            fig_pth = op.split(fig_abs_name)[0]
            # if output path doesn't exist, create it
            os.makedirs(fig_pth, exist_ok = True)

            print('saving %s' %fig_abs_name)
            if zoom2ROI is not None:
                fig.savefig(fig_abs_name, dpi = dpi)
            else:
                _ = cortex.quickflat.make_png(fig_abs_name, flatmap, recache = recache, with_colorbar = with_colorbar, with_rois = with_rois,
                                                    with_curvature = with_curvature, with_sulci = with_sulci, with_labels = with_labels,
                                                    curvature_brightness = curvature_brightness, curvature_contrast = curvature_contrast)
            
        return flatmap

    def make_colormap(self, colormap = 'rainbow_r', bins = 256, add_alpha = True, invert_alpha = False, 
                      cmap_name = 'custom', discrete = False, return_cmap = False):

        """ make custom colormap
        can add alpha channel to colormap,
        and save to pycortex filestore
        
        Parameters
        ----------
        colormap : str or List/arr
            if string then has to be a matplolib existent colormap
            if list/array then contains strings with color names, to create linear segmented cmap
        bins : int
            number of bins for colormap
        add_alpha: bool
            if we want to add an alpha channel
        invert_alpha : bool
            if we want to invert direction of alpha channel
            (y can be from 0 to 1 or 1 to 0)
        cmap_name : str
            new cmap filename, final one will have _alpha_#-bins added to it
        discrete : bool
            if we want a discrete colormap or not (then will be continuous)
        return_cmap: bool
            if we want to return the cmap itself or the absolute path to new colormap
        """
        
        if isinstance(colormap, str): # if input is string (so existent colormap)

            # get colormap
            cmap = cm.get_cmap(colormap)

        elif isinstance(colormap, list) or isinstance(colormap, np.ndarray): # is list of strings
            cvals  = np.arange(len(colormap))
            norm = plt.Normalize(min(cvals),max(cvals))
            tuples = list(zip(map(norm,cvals), colormap))
            cmap = colors.LinearSegmentedColormap.from_list("", tuples)
            
            if discrete == True: # if we want a discrete colormap from list
                cmap = colors.ListedColormap(colormap)
                bins = int(len(colormap))
        
        else: # assumes it is colormap object
            cmap = colormap #(range(256)) # note, to get full colormap we cannot make it discrete

        # convert into array
        cmap_array = cmap(range(bins))

        # reshape array for map
        new_map = []
        for i in range(cmap_array.shape[-1]):
            new_map.append(np.tile(cmap_array[...,i],(bins,1)))

        new_map = np.moveaxis(np.array(new_map), 0, -1)
        
        if add_alpha: 
            # make alpha array
            if invert_alpha == True: # in case we want to invert alpha (y from 1 to 0 instead pf 0 to 1)
                _, alpha = np.meshgrid(np.linspace(0, 1, bins, endpoint=False), 1-np.linspace(0, 1, bins))
            else:
                _, alpha = np.meshgrid(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins, endpoint=False))

            # add alpha channel
            new_map[...,-1] = alpha
            cmap_ext = (0,1,0,1)
        else:
            new_map = new_map[:1,...].copy() 
            cmap_ext = (0,100,0,1)
        
        fig = plt.figure(figsize=(1,1))
        ax = fig.add_axes([0,0,1,1])
        # plot 
        plt.imshow(new_map,
        extent = cmap_ext,
        origin = 'lower')
        ax.axis('off')

        if add_alpha: 
            rgb_fn = op.join(op.split(cortex.database.default_filestore)[
                            0], 'colormaps', cmap_name+'_alpha_bins_%d.png'%bins)
        else:
            rgb_fn = op.join(op.split(cortex.database.default_filestore)[
                            0], 'colormaps', cmap_name+'_bins_%d.png'%bins)
        #misc.imsave(rgb_fn, new_map)
        plt.savefig(rgb_fn, dpi = 200,transparent=True)

        if return_cmap:
            return cmap
        else:
            return rgb_fn 

    def make_2D_colormap(self, rgb_color = '101', bins = 50, scale=[1,1]):
        
        """
        generate 2D basic colormap, from RGB combination,
        and save to pycortex filestore

        Parameters
        ----------
        rgb_color: str
            combination of rgb values (ex: 101 means it will use red and blue)
        bins: int
            number of color bins between min and max value
        scale: arr/list
            int/float with how much to scale each color (ex: 1 == full red)
        
        """
        
        ##generating grid of x bins
        x,y = np.meshgrid(
            np.linspace(0,1*scale[0],bins),
            np.linspace(0,1*scale[1],bins)) 
        
        # define color combination for plot
        if rgb_color=='101': #red blue
            col_grid = np.dstack((x,np.zeros_like(x), y))
            name='RB'
        elif rgb_color=='110': # red green
            col_grid = np.dstack((x, y,np.zeros_like(x)))
            name='RG'
        elif rgb_color=='011': # green blue
            col_grid = np.dstack((np.zeros_like(x),x, y))
            name='GB'
        
        fig = plt.figure(figsize=(1,1))
        ax = fig.add_axes([0,0,1,1])
        # plot 
        plt.imshow(col_grid,
        extent = (0,1,0,1),
        origin = 'lower')
        ax.axis('off')

        rgb_fn = op.join(op.split(cortex.database.default_filestore)[
                            0], 'colormaps', 'custom2D_'+name+'_bins_%d.png'%bins)

        plt.savefig(rgb_fn, dpi = 200)
        
        return rgb_fn

    def add_data2overlay(self, flatmap = None, name = ''):

        """
        Helper func to add data to overlay.
        Useful for ROI drawing
        
        Parameters
        ----------
        flatmap: pycortex data object
            XD vertex data
        name: str
            name for data layer that will be added
        """

        # ADD ROI TO OVERLAY
        cortex.utils.add_roi(flatmap, name = name, open_inkscape=False)

    def plot_2D_DM(self, dm_array, filename):

        """
        Function to plot design matrix frame by frame 
        and save movie in folder

        Parameters
        ----------
        dm_array: array
            design matrix [N x N x TR]
        filename: str
            absolute filename for output movie 
        """

        # if output path doesn't exist, create it
        outfolder = op.split(filename)[0]
        os.makedirs(outfolder, exist_ok=True)

        print('saving files in %s'%outfolder)

        dm_array = (dm_array * 255).astype(np.uint8)

        for w in tqdm(range(dm_array.shape[-1])):
            im = Image.fromarray(dm_array[...,w])
            im.save(op.join(outfolder,"DM_TR-%s.png"%str(w).zfill(4)))  

        ## save as video
        img_name = op.join(outfolder,'DM_TR-%4d.png')
        os.system("ffmpeg -r 6 -start_number 0 -i %s -vcodec mpeg4 -y %s"%(img_name, filename)) 

    def get_NONuniform_polar_angle(self, xx = [], yy = [], rsq = [], angle_thresh = 3*np.pi/4, 
                                   rsq_thresh = 0, pysub = 'hcp_999999'):

        """
        Helper function to transform polar angle values into RGB values
        guaranteeing a non-uniform representation
        (this is, when we want to use half the color wheel to show the pa values)
        (useful for better visualization of boundaries)
        
        Parameters
        ----------
        xx : arr
            array with x position values
        yy : arr
            array with y position values
        rsq: arr
            rsq values, to be used as alpha level/threshold
        angle_thresh: float
            value upon which to make it red for this hemifield (above angle or below 1-angle will be red in a retinotopy hsv color wheel)
        rsq_thresh: float/int
            minimum rsq threshold to use 
        pysub: str
            name of pycortex subject folder
        """

        hsv_angle = []
        hsv_angle = np.ones((len(rsq), 3))

        ## calculate polar angle
        polar_angle = np.angle(xx + yy * 1j)

        ## set normalized polar angle (0-1), and make nan irrelevant vertices
        hsv_angle[:, 0] = np.nan 
        hsv_angle[:, 0][rsq > rsq_thresh] = ((polar_angle + np.pi) / (np.pi * 2.0))[rsq > rsq_thresh]

        ## normalize angle threshold for overepresentation
        angle_thresh_norm = (angle_thresh + np.pi) / (np.pi * 2.0)

        ## get mid vertex index (diving hemispheres)
        left_index = cortex.db.get_surfinfo(pysub).left.shape[0] 

        ## set angles within threh interval to 0
        ind_thresh = np.where((hsv_angle[:left_index, 0] > angle_thresh_norm) | (hsv_angle[:left_index, 0] < 1-angle_thresh_norm))[0]
        hsv_angle[:left_index, 0][ind_thresh] = 0

        ## now take angles from RH (thus LVF) 
        #### ATENÇÃO -> minus sign to flip angles vertically (then order of colors same for both hemispheres) ###
        # also normalize it
        hsv_angle[left_index:, 0] = ((np.angle(-1*xx + yy * 1j) + np.pi) / (np.pi * 2.0))[left_index:]

        # set angles within threh interval to 0
        ind_thresh = np.where((hsv_angle[left_index:, 0] > angle_thresh_norm) | (hsv_angle[left_index:, 0] < 1-angle_thresh_norm))[0]
        hsv_angle[left_index:, 0][ind_thresh] = 0

        ## make final RGB array
        rgb_angle = np.ones((len(rsq), 3))
        rgb_angle[:] = np.nan

        rgb_angle[rsq > rsq_thresh] = colors.hsv_to_rgb(hsv_angle[rsq > rsq_thresh])

        return rgb_angle

    def plot_pa_colorwheel(self, resolution=800, angle_thresh = 3*np.pi/4, cmap_name = 'hsv', continuous = True, fig_name = None):

        """
        Helper function to create colorwheel image
        for polar angle plots returns 
        Parameters
        ----------
        resolution : int
            resolution of mesh
        angle_thresh: float
            value upon which to make it red for this hemifield (above angle or below 1-angle will be red in a retinotopy hsv color wheel)
            if angle threh different than PI then assumes non uniform colorwheel
        cmap_name: str/list
            colormap name (if string) or list of colors to use for colormap
        continuous: bool
            if continuous colormap or binned
        """

        ## make circle
        circle_x, circle_y = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
        circle_radius = np.sqrt(circle_x**2 + circle_y**2)
        circle_pa = np.arctan2(circle_y, circle_x) # all polar angles calculated from our mesh
        circle_pa[circle_radius > 1] = np.nan # then we're excluding all parts of bitmap outside of circle

        if isinstance(cmap_name, str):

            cmap = plt.get_cmap('hsv')
            norm = colors.Normalize(-angle_thresh, angle_thresh) # normalize between the point where we defined our color threshold
        
        elif isinstance(cmap_name, list) or isinstance(cmap_name, np.ndarray):

            if continuous:
                cvals  = np.arange(len(cmap_name))
                norm = plt.Normalize(min(cvals),max(cvals))
                tuples = list(zip(map(norm,cvals), cmap_name))
                
                colormap = colors.LinearSegmentedColormap.from_list("", tuples)
                norm = colors.Normalize(-angle_thresh, angle_thresh) 

            else:
                colormap = colors.ListedColormap(cmap_name)
                #boundaries = np.linspace(0,1,len(cmap_name))
                #norm = colors.BoundaryNorm(boundaries, colormap.N, clip=True)
                norm = colors.Normalize(-angle_thresh, angle_thresh) 
        
        else:
            cmap = cmap_name
            norm = colors.Normalize(-angle_thresh, angle_thresh) # normalize between the point where we defined our color threshold

        fig = plt.figure(figsize=(5,5))

        # non-uniform colorwheel
        if angle_thresh != np.pi:
            
            ## for LH (RVF)
            circle_pa_left = circle_pa.copy()
            ## between thresh angle make it red
            #circle_pa_left[(circle_pa_left < -angle_thresh) | (circle_pa_left > angle_thresh)] = angle_thresh

            plt.imshow(circle_pa_left, cmap=cmap, norm=norm,origin='lower') # origin lower because imshow flips it vertically, now in right order for VF
            plt.axis('off')

            fig.savefig('{fn}_colorwheel_4LH-RVF.png'.format(fn = fig_name),dpi=100)

            ## for RH (LVF)
            circle_pa_right = circle_pa.copy()
            circle_pa_right = np.fliplr(circle_pa_right)
            ## between thresh angle make it red
            #circle_pa_right[(circle_pa_right < -angle_thresh) | (circle_pa_right > angle_thresh)] = angle_thresh

            plt.imshow(circle_pa_right, cmap=cmap, norm=norm,origin='lower')
            plt.axis('off')

            plt.savefig('{fn}_colorwheel_4RH-LVF.png'.format(fn = fig_name),dpi=100)

        else:
            plt.imshow(circle_pa, cmap = colormap, norm=norm, origin='lower')
            plt.axis('off')

            if continuous:
                fig.savefig('{fn}_colorwheel_continuous.png'.format(fn = fig_name),dpi=100)
            else:
                fig.savefig('{fn}_colorwheel_discrete.png'.format(fn = fig_name),dpi=100)

    def plot_periodogram(self, axis, timecourse = None, TR = 1.6):

        """
        plot power spectral density
            
        """

        sampling_frequency = 1 / TR  
        freq, power = periodogram(timecourse, fs = sampling_frequency)#, detrend = False)
        
        axis.plot(freq, power, 'g-', alpha = .8, label='data')

        axis.set_xlabel('Frequency (Hz)',fontsize = 15, labelpad = 10)
        axis.set_ylabel('Power (dB)',fontsize = 15, labelpad = 10)

        axis.axvline(x=0.01,color='r',ls='dashed', lw=2)
        
        return axis



    



