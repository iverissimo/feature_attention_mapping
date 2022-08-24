import numpy as np
import os
from os import path as op
import pandas as pd


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



def get_flatmaps(est_arr1, est_arr2 = None, 
                            vmin1 = 0, vmax1 = .8, vmin2 = None, vmax2 = None,
                            pysub = 'hcp_999999', cmap = 'BuBkRd'):

    """
    Helper function to set and return flatmap  

    Parameters
    ----------
    est_arr1 : array
        data array
    cmap : str
        string with colormap name
    vmin: int/float
        minimum value
    vmax: int/float 
        maximum value
    subject: str
        overlay subject name to use
    """

    # if two arrays provided, then fig is 2D
    if est_arr2:
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


def make_raw_vertex_image(data1, cmap = 'hot', vmin = 0, vmax = 1, 
                          data2 = [], vmin2 = 0, vmax2 = 1, subject = 'fsaverage', data2D = False):  
    
    """ function to fix web browser bug in pycortex
        allows masking of data with nans
    
    Parameters
    ----------
    data1 : array
        data array
    cmap : str
        string with colormap name (not the alpha version)
    vmin: int/float
        minimum value
    vmax: int/float 
        maximum value
    subject: str
        overlay subject name to use
    
    Outputs
    -------
    vx_fin : VertexRGB
        vertex object to call in webgl
    
    """
    
    # Get curvature
    curv = cortex.db.get_surfinfo(subject, type = 'curvature', recache=False)#,smooth=1)
    # Adjust curvature contrast / color. Alternately, you could work
    # with curv.data, maybe threshold it, and apply a color map.     
    curv.data[curv.data>0] = .1
    curv.data[curv.data<=0] = -.1
    #curv.data = np.sign(curv.data.data) * .25
    
    curv.vmin = -1
    curv.vmax = 1
    curv.cmap = 'gray'
    
    # Create display data 
    vx = cortex.Vertex(data1, subject, cmap = cmap, vmin = vmin, vmax = vmax)
    
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
    vx_fin = cortex.VertexRGB(*display_data, subject, curvature_brightness = 0.4, curvature_contrast = 0.1)

    return vx_fin


def make_colormap(colormap = 'rainbow_r', bins = 256, add_alpha = True, invert_alpha = False, cmap_name = 'costum',
                      discrete = False, return_cmap = False):

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
    invert_alpha : bool
        if we want to invert direction of alpha channel
        (y can be from 0 to 1 or 1 to 0)
    cmap_name : str
        new cmap filename, final one will have _alpha_#-bins added to it
    discrete : bool
        if we want a discrete colormap or not (then will be continuous)
    Outputs
    -------
    rgb_fn : str
        absolute path to new colormap
    """
    
    if isinstance(colormap, str): # if input is string (so existent colormap)

        # get colormap
        cmap = cm.get_cmap(colormap)

    else: # is list of strings
        cvals  = np.arange(len(colormap))
        norm = plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(norm,cvals), colormap))
        cmap = colors.LinearSegmentedColormap.from_list("", tuples)
        
        if discrete == True: # if we want a discrete colormap from list
            cmap = colors.ListedColormap(colormap)
            bins = int(len(colormap))

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


def create_glasser_df(path2file):

    """ Function to create glasser dataframe
     with ROI names, colors (RGBA) and vertex indices

    Parameters
    ----------
    path2file : str 
        path to the parcelation label file
    """
    
    # we read in the atlas data, which consists of 180 separate regions per hemisphere. 
    # These are labeled separately, so the labels go to 360.
    cifti = nib.load(op.join(path2file,
                         'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii'))

    # get index data array
    cifti_data = np.array(cifti.get_fdata(dtype=np.float32))[0]
    # from header get label dict with key + rgba
    cifti_hdr = cifti.header
    label_dict = cifti_hdr.get_axis(0)[0][1]
    
    ## make atlas data frame
    atlas_df = pd.DataFrame(columns = ['ROI', 'index','R','G','B','A'])

    for key in label_dict.keys():

        if label_dict[key][0] != '???': 
            atlas_df = atlas_df.append(pd.DataFrame({'ROI': label_dict[key][0].replace('_ROI',''),
                                                     'index': key,
                                                     'R': label_dict[key][1][0],
                                                     'G': label_dict[key][1][1],
                                                     'B': label_dict[key][1][2],
                                                     'A': label_dict[key][1][3]
                                                    }, index=[0]),ignore_index=True)
            
    return atlas_df, cifti_data


def get_weighted_bins(data_df, x_key = 'ecc', y_key = 'size', weight_key = 'rsq', n_bins = 10):

    """ 
    
    Get weighted bins from dataframe, sorted by one of the variables

    """
    
    # sort values by eccentricity
    data_df = data_df.sort_values(by=[x_key])

    #divide in equally sized bins
    bin_size = int(len(data_df)/n_bins) 
    
    mean_x = []
    mean_x_std = []
    mean_y = []
    mean_y_std = []
    
    # for each bin calculate rsq-weighted means and errors of binned ecc/gain 
    for j in range(n_bins): 
        
        mean_x.append(weightstats.DescrStatsW(data_df[bin_size * j:bin_size * (j+1)][x_key],
                                              weights = data_df[bin_size * j:bin_size * (j+1)][weight_key]).mean)
        mean_x_std.append(weightstats.DescrStatsW(data_df[bin_size * j:bin_size * (j+1)][x_key],
                                                  weights = data_df[bin_size * j:bin_size * (j+1)][weight_key]).std_mean)

        mean_y.append(weightstats.DescrStatsW(data_df[bin_size * j:bin_size * (j+1)][y_key],
                                              weights = data_df[bin_size * j:bin_size*(j+1)][weight_key]).mean)
        mean_y_std.append(weightstats.DescrStatsW(data_df[bin_size * j:bin_size * (j+1)][y_key],
                                                  weights = data_df[bin_size * j:bin_size * (j+1)][weight_key]).std_mean)

    return mean_x, mean_x_std, mean_y, mean_y_std


def get_rois4plotting(params, pysub = 'hcp_999999', use_atlas = True, atlas_pth = '', space = 'fsLR_den-170k'):

    """ 
    helper function to get ROI names, vertice index and color palette
   to be used in plotting scripts
    
    Parameters
    ----------
    params : dict
        yaml dict with task related infos  
    """ 
    
    roi_verts = {} #empty dictionary  
    
    if use_atlas:
        # Get Glasser atlas
        atlas_df, atlas_array = create_glasser_df(atlas_pth)

        # ROI names
        ROIs = list(params['plotting']['ROIs']['glasser_atlas'].keys())
        # colors
        color_codes = {key: params['plotting']['ROIs']['glasser_atlas'][key]['color'] for key in ROIs}

        # get vertices for ROI
        for _,key in enumerate(ROIs):
            roi_verts[key] = np.hstack((np.where(atlas_array == ind)[0] for ind in atlas_df[atlas_df['ROI'].isin(params['plotting']['ROIs']['glasser_atlas'][key]['ROI'])]['index'].values))

    else:
        # set ROI names
        ROIs = params['plotting']['ROIs'][space]

        # dictionary with one specific color per group - similar to fig3 colors
        ROI_pal = params['plotting']['ROI_pal']
        color_codes = {key: ROI_pal[key] for key in ROIs}

        # get vertices for ROI
        for _,val in enumerate(ROIs):
            print(val)
            roi_verts[val] = cortex.get_roi_verts(pysub,val)[val]
            
    return ROIs, roi_verts, color_codes



def fwhmax_fwatmin(model, estimates, normalize_RFs=False, return_profiles=False):
    
    """
    taken from marco aqil's code, all credits go to him
    """
    
    model = model.lower()
    x=np.linspace(-50,50,1000).astype('float32')

    prf = estimates['betas'] * np.exp(-0.5*x[...,np.newaxis]**2 / estimates['size']**2)
    vol_prf =  2*np.pi*estimates['size']**2

    if 'dog' in model or 'dn' in model:
        srf = estimates['sa'] * np.exp(-0.5*x[...,np.newaxis]**2 / estimates['ss']**2)
        vol_srf = 2*np.pi*estimates['ss']*2

    if normalize_RFs==True:

        if model == 'gauss':
            profile =  prf / vol_prf
        elif model == 'css':
            #amplitude is outside exponent in CSS
            profile = (prf / vol_prf)**estimates['ns'] * estimates['betas']**(1 - estimates['ns'])
        elif model =='dog':
            profile = prf / vol_prf - \
                       srf / vol_srf
        elif 'dn' in model:
            profile = (prf / vol_prf + estimates['nb']) /\
                      (srf / vol_srf + estimates['sb']) - estimates['nb']/estimates['sb']
    else:
        if model == 'gauss':
            profile = prf
        elif model == 'css':
            #amplitude is outside exponent in CSS
            profile = prf**estimates['ns'] * estimates['betas']**(1 - estimates['ns'])
        elif model =='dog':
            profile = prf - srf
        elif 'dn' in model:
            profile = (prf + estimates['nb'])/(srf + estimates['sb']) - estimates['nb']/estimates['sb']


    half_max = np.max(profile, axis=0)/2
    fwhmax = np.abs(2*x[np.argmin(np.abs(profile-half_max), axis=0)])


    if 'dog' in model or 'dn' in model:

        min_profile = np.min(profile, axis=0)
        fwatmin = np.abs(2*x[np.argmin(np.abs(profile-min_profile), axis=0)])

        result = fwhmax, fwatmin
    else:
        result = fwhmax

    if return_profiles:
        return result, profile.T
    else:
        return result


def import_fmriprep2pycortex(source_directory, sj, dataset=None, ses=None, acq=None):
    
    """Import a subject from fmriprep-output to pycortex
    
    Parameters
    ----------
    source_directory : string
       Local directory that contains both fmriprep and freesurfer subfolders 
    sj : string
        Fmriprep subject name (without "sub-")
    dataset : string
       If you have multiple fmriprep outputs from different datasets, use this attribute
       to add a prefix to every subject id ('ds01.01' rather than '01')
    ses : string, optional
       BIDS session that contains the anatomical data
    acq : string, optional
        If we intend to specific the acquisition of the T1w file (for naming purposes)
    """
    if dataset is not None:
        pycortex_sub = '{ds}.{sub}'.format(ds=dataset, sub=sj)
    else:
        pycortex_sub = '{sub}'.format(sub=sj)

    if pycortex_sub in cortex.database.db.subjects.keys():
        print('subject %s already in filestore, will not overwrite'%pycortex_sub)
    else:
        
        # import subject into pycortex database
        cortex.fmriprep.import_subj(subject = sj, source_dir = source_directory, 
                             session = ses, dataset = dataset, acq = acq)


def plot_pRF_DM(dm_array, filename):

    """
    Function to plot design matrix frame by frame 
    and save movie in folder

    """

    # if output path doesn't exist, create it

    outfolder = op.split(filename)[0]

    if not op.isdir(outfolder): 
        os.makedirs(outfolder)
    print('saving files in %s'%filename)

    dm_array = (dm_array * 255).astype(np.uint8)

    for w in range(dm_array.shape[-1]):
        im = Image.fromarray(dm_array[...,w])
        im.save(op.join(outfolder,"DM_TR-%s.png"%str(w).zfill(4)))  

    ## save as video
    img_name = op.join(outfolder,'DM_TR-%4d.png')
    os.system("ffmpeg -r 6 -start_number 0 -i %s -vcodec mpeg4 -y %s"%(img_name, filename))  


def get_estimates_roi_df(participant, est_pp_dict, ROIs = None, roi_verts = None, est_key = 'r2', model = 'gauss'):

    """

    Helper function to get estimates dataframe values for each ROI
    will select values based on est key param 

    """

    ## save rsq values in dataframe, for plotting
    df_est = pd.DataFrame({'sj': [], 'index': [], 'ROI': [], 'value': [], 'model': []})

    for idx,rois_ks in enumerate(ROIs): 
        
        # mask estimates
        print('masking estimates for ROI %s'%rois_ks)
        
        roi_arr = est_pp_dict[est_key][roi_verts[rois_ks]]

        df_est = pd.concat((df_est,
                            pd.DataFrame({'sj': np.tile('sub-{sj}'.format(sj = participant), len(roi_arr)), 
                                        'index': roi_verts[rois_ks], 
                                        'ROI': np.tile(rois_ks, len(roi_arr)), 
                                        'value': roi_arr,
                                        'model': np.tile(model, len(roi_arr))})
                        ))

    return df_est
