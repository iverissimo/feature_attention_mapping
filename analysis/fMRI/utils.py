
# script to calculate tSNR
# for the different sequences piloted


import numpy as np
import os
from os import path as op
import nibabel as nib

from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

import cortex
from cortex import fmriprep

from PIL import Image, ImageDraw

from matplotlib import cm
import matplotlib.colors



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
        fmriprep.import_subj(subject = sj, source_dir = source_directory, 
                             session = ses, dataset = dataset, acq = acq)


def get_tsnr(data,affine,file_name):
    """
    Compute the tSNR of nifti file
    and generate the equivalent nifti SNR 3Ds. 
    """ 

    if not op.exists(file_name): 
        print('making %s'%file_name)
    
        mean_d = np.mean(data,axis=-1)
        std_d = np.std(data,axis=-1)
        
        tsnr = mean_d/std_d
        #tsnr[np.where(np.isinf(tsnr))] = np.nan
        
        tsnr_image = nib.nifti1.Nifti1Image(tsnr,affine)
        
        nib.save(tsnr_image,file_name)

    else:
        print('already exists, skipping %s'%file_name)
        tsnr = nib.load(file_name)
        tsnr = np.array(tsnr.dataobj)
    
    return tsnr



def correlate_vol(data1,data2,outfile):
    """
    Compute Pearson correlation between 2 of nifti files
    and generate the equivalent correlation nifti. 
    """ 

    if not op.exists(outfile): 
        print('making %s'%outfile)
    
        # get affine for one of the runs
        nibber = nib.load(data1)
        affine = nibber.affine
        
        # load data 
        data1 = np.array(nib.load(data1).dataobj)
        data2 = np.array(nib.load(data2).dataobj)
        
        #- Calculate the number of voxels (number of elements in one volume)
        n_voxels = np.prod(data1.shape[:-1])

        #- Reshape 4D array to 2D array n_voxels by n_volumes
        data1_2d = np.reshape(data1, (n_voxels, data1.shape[-1]))
        data2_2d = np.reshape(data2, (n_voxels, data2.shape[-1]))

        #- Make a 1D array of size (n_voxels,) to hold the correlation values
        correlations_1d = np.zeros((n_voxels,))

        #- Loop over voxels filling in correlation at this voxel
        for i in range(n_voxels):
            correlations_1d[i] = np.corrcoef(data1_2d[i, :], data2_2d[i, :])[0, 1]
            
        #- Reshape the correlations array back to 3D
        correlations = np.reshape(correlations_1d, data1.shape[:-1])
        
        corr_image = nib.nifti1.Nifti1Image(correlations,affine)
        
        nib.save(corr_image,outfile)

    else:
        print('already exists, skipping %s'%outfile)
        correlations = nib.load(outfile)
        correlations = np.array(correlations.dataobj)
    
    return correlations


def filter_data(file, outdir, filter_type = 'HPgauss', plot_vert=False, **kwargs):
    
    """ 
    Generic filtering function, implemented different types of filters
    High pass filter NIFTI run with gaussian kernel
    
    Parameters
    ----------
    file : str/list/array
        absolute filename to be filtered (or list of filenames)
    outdir : str
        path to save new file
    filter_type : str
        type of filter to use, defaults to gaussian kernel high pass
    
    Outputs
    -------
    out_file: str
        absolute output filename (or list of filenames)
    
    """
    
    # check if single filename or list of filenames
    
    if isinstance(file, list) or isinstance(file, np.ndarray): 
        file_list = file  
    else:
        file_list = [file]
      
    # store output filename in list
    outfiles = []
    
    # for each file, do the same
    for input_file in file_list:
        
        # get file extension
        file_extension = '.{a}.{b}'.format(a = input_file.rsplit('.', 2)[-2],
                                   b = input_file.rsplit('.', 2)[-1])
        # set output filename
        output_file = op.join(outdir, 
                    op.split(input_file)[-1].replace(file_extension,'_{filt}{ext}'.format(filt = filter_type,
                                                                                           ext = file_extension)))
        # if file already exists, skip
        if op.exists(output_file): 
            print('already exists, skipping %s'%output_file)
        
        else:
            print('making %s'%output_file)
            
            # load file
            nibber = nib.load(input_file)

            # way depends on type of extension
            if file_extension == '.func.gii':
                data = np.array([nibber.darrays[i].data for i in range(len(nibber.darrays))]) #load surface data
            else:
                affine = nibber.affine
                data = np.array(nibber.get_fdata())

    
            ### implement filter types, by calling their specific functions

            if filter_type == 'HPgauss':

                data_filt = gausskernel_data(data, **kwargs)
                
            #elif filter_type == 'sg':
                
            elif filter_type == 'dc':
                raise NameError('Not implemented')
                
            else:
                raise NameError('Not implemented')
                
            # if plotting true, make figure of voxel with high variance,
            # to compare the difference
            if plot_vert == True:
                
                ind2plot = np.argwhere(np.std(data, axis=0)==np.max(np.std(data, axis=0)))[0][0]
                fig = plt.figure()
                plt.plot(data[...,ind2plot], color='dimgray',label='Original data')
                plt.plot(data_filt[...,ind2plot], color='mediumseagreen',label='Filtered data')

                plt.xlabel('Time (TR)')
                plt.ylabel('Signal amplitude (a.u.)')
                plt.legend(loc = 'upper right')

                fig.savefig(output_file.replace(file_extension,'_vertex_%i.png'%ind2plot))
            

            ## save filtered file
            # again, way depends on type of extension
            if file_extension == '.func.gii':
                darrays = [nib.gifti.gifti.GiftiDataArray(d) for d in data_filt]
                output_image = nib.gifti.gifti.GiftiImage(header = nibber.header, 
                                                                  extra = nibber.extra, 
                                                                  darrays = darrays)
            else:
                output_image = nib.nifti1.Nifti1Image(data_filt, affine, header = nibber.header)

            # actually save
            nib.save(output_image,output_file)

        # append out files
        outfiles.append(output_file)
        
    # if input file was not list, then return output that is also not list
    if isinstance(file, list) or isinstance(file, np.ndarray): 
        outfiles = outfiles[0] 
    
    return outfiles

def gausskernel_data(data, TR = 1.2, cut_off_hz = 0.01, **kwargs):
    
    """ 
    High pass filter array with gaussian kernel
    
    Parameters
    ----------
    data : arr
        data array
    TR : float
        TR for run
    cut_off_hz : float
        cut off frequency to filter
    
    Outputs
    -------
    data_filt: arr
        filtered array
    """ 
        
    # save shape, for file reshpaing later
    arr_shape = data.shape
    
    sigma = (1/cut_off_hz) / (2 * TR) 

    # reshape to 2D if necessary
    if len(arr_shape)>2:
        data = np.reshape(data, (-1, data.shape[-1])) 

    # filter signal
    filtered_signal = np.array(Parallel(n_jobs=2)(delayed(gaussian_filter)(i, sigma=sigma) for _,i in enumerate(data))) 

    # add mean image back to avoid distribution around 0
    data_filt = data - filtered_signal + np.mean(filtered_signal, axis=0)
    data_filt = data_filt.reshape(*arr_shape)
    
    return data_filt


def psc(file, outpth, file_extension = '_psc.nii.gz'):

    """ percent signal change nii file
    Parameters
    ----------
    file : str
        absolute filename for nifti
    outpth: str
        path to save new files
    extension: str
        file extension
    Outputs
    -------
    output: str
        absolute filename for psc nifti
    
    """
    
    # output filename
    output = op.join(outpth,op.split(file)[-1].replace('.nii.gz',file_extension))
    
    if not op.exists(output): 
        print('making %s'%output)

        nibber = nib.load(file)
        affine = nibber.affine
        data = np.array(nibber.dataobj)
        
        # reshape to 2D
        data_reshap = np.reshape(data, (-1, data.shape[-1])) 
        
        # psc signal
        mean_signal = data_reshap.mean(axis = -1)[..., np.newaxis] 
        data_psc = (data_reshap - mean_signal)/np.absolute(mean_signal)
        data_psc *= 100
        data_psc = data_psc.reshape(*data.shape)

        output_image = nib.nifti1.Nifti1Image(data_psc,affine,header=nibber.header)
        nib.save(output_image,output)

    else:
        print('already exists, skipping %s'%output)


    return output


def avg_nii(files, outpth):

    """ percent signal change gii file
    Parameters
    ----------
    files : list
        list of strings with absolute filename for nifti
    out_pth: str
        path to save new files
    extension: str
        file extension
    Outputs
    -------
    output: str
        absolute filename for psc nifti
    """
    
    # sort files
    files.sort()
    # output filename
    output = op.join(outpth,op.split(files[0])[-1].replace('run-1','run-average'))
    
    if not op.exists(output): 
        print('making %s'%output)
        
        for ind, run in enumerate(files):

            nibber = nib.load(run)
            affine = nibber.affine
            data = np.array(nibber.dataobj)
            
            if ind == 0:
                data_avg = data.copy()[np.newaxis,...] 
            else:
                data_avg = np.vstack((data_avg,data.copy()[np.newaxis,...]))

        # average
        data_avg = np.mean(data_avg,axis=0)

        output_image = nib.nifti1.Nifti1Image(data_avg,affine,header=nibber.header)
        nib.save(output_image,output)

    else:
        print('already exists, skipping %s'%output)
        
    return output

    
def make_pRF_DM(output,params,save_imgs=False,downsample=None):
    
    """Make design matrix for pRF task
    
    Parameters
    ----------
    output : string
       absolute output name for DM
    params : yml dict
        with experiment params
    save_imgs : bool
       if we want to save images in folder, for sanity check
    """
    
    if not op.exists(output): 
        print('making %s'%output)

        if not op.exists(op.split(output)[0]): # make base dir to save files
            os.makedirs(op.split(output)[0])
        
        # general infos
        TR = params['mri']['TR']
        bar_width = params['prf']['bar_width_ratio'] 

        screen_res = params['window']['size']
        if params['window']['display'] == 'square': # if square display
            screen_res = np.array([screen_res[1], screen_res[1]])

        if downsample != None: # if we want to downsample screen res
            screen_res = (screen_res*downsample).astype(int)

        # number TRs per condition
        TR_conditions = {'L-R': params['prf']['bar_pass_hor_TR'],
                         'R-L': params['prf']['bar_pass_hor_TR'],
                         'U-D': params['prf']['bar_pass_ver_TR'],
                         'D-U': params['prf']['bar_pass_ver_TR'],
                         'empty': params['prf']['empty_TR']}

        # order of conditions in run
        bar_pass_direction = params['prf']['bar_pass_direction']
        
        # get total number of TRs in run
        # list of bar orientation at all TRs
        total_TR = 0
        for _,bartype in enumerate(bar_pass_direction):
            total_TR += TR_conditions[bartype]

        # all possible positions in pixels for for midpoint of
        # y position for vertical bar passes, 
        ver_y = screen_res[1]*np.linspace(0,1, TR_conditions['U-D'])
        # x position for horizontal bar passes 
        hor_x = screen_res[0]*np.linspace(0,1, TR_conditions['L-R'])        

        # coordenates for bar pass, for PIL Image
        coordenates_bars = {'L-R': {'upLx': hor_x-0.5*bar_width*screen_res[0], 'upLy': np.repeat(screen_res[1],TR_conditions['L-R']),
                                     'lowRx': hor_x+0.5*bar_width*screen_res[0], 'lowRy': np.repeat(0,TR_conditions['L-R'])},
                            'R-L': {'upLx': np.array(list(reversed(hor_x-0.5*bar_width*screen_res[0]))), 'upLy': np.repeat(screen_res[1],TR_conditions['R-L']),
                                     'lowRx': np.array(list(reversed(hor_x+0.5*bar_width*screen_res[0]))), 'lowRy': np.repeat(0,TR_conditions['R-L'])},
                            'U-D': {'upLx': np.repeat(0,TR_conditions['U-D']), 'upLy': ver_y+0.5*bar_width*screen_res[1],
                                     'lowRx': np.repeat(screen_res[0],TR_conditions['U-D']), 'lowRy': ver_y-0.5*bar_width*screen_res[1]},
                            'D-U': {'upLx': np.repeat(0,TR_conditions['D-U']), 'upLy': np.array(list(reversed(ver_y+0.5*bar_width*screen_res[1]))),
                                     'lowRx': np.repeat(screen_res[0],TR_conditions['D-U']), 'lowRy': np.array(list(reversed(ver_y-0.5*bar_width*screen_res[1])))}
                             }

        # save screen display for each TR
        visual_dm_array = np.zeros((total_TR, screen_res[0],screen_res[1]))
        counter = 0
        for ind,bartype in enumerate(bar_pass_direction): # loop over bar pass directions

            for i in range(TR_conditions[bartype]):

                img = Image.new('RGB', tuple(screen_res)) # background image

                if bartype not in np.array(['empty']): # if not empty screen
                    # set draw method for image
                    draw = ImageDraw.Draw(img)
                    # add bar, coordinates (upLx, upLy, lowRx, lowRy)
                    draw.rectangle(tuple([coordenates_bars[bartype]['upLx'][i],coordenates_bars[bartype]['upLy'][i],
                                        coordenates_bars[bartype]['lowRx'][i],coordenates_bars[bartype]['lowRy'][i]]), 
                                   fill = (255,255,255),
                                   outline = (255,255,255))

                visual_dm_array[counter, ...] = np.array(img)[:,:,0][np.newaxis,...]
                counter += 1

        # swap axis to have time in last axis [x,y,t]
        visual_dm = visual_dm_array.transpose([1,2,0]) 
        
        # save design matrix
        np.save(output, visual_dm)
        
    else:
        print('already exists, skipping %s'%output)
        
        # load
        visual_dm = np.load(output)
        
    #if we want to save the images
    if save_imgs == True:
        outfolder = op.split(output)[0]

        visual_dm = visual_dm.astype(np.uint8)

        for w in range(visual_dm.shape[-1]):
            im = Image.fromarray(visual_dm[...,w])
            im.save(op.join(outfolder,"DM_TR-%i.png"%w))      
            
    return visual_dm


def save_estimates(filename, estimates, vox_indices, data_filename):
    
    """
    re-arrange estimates from 2D to 4D
    and save in folder
    
    Parameters
    ----------
    filename : str
        absolute filename of estimates to be saved
    estimates : arr
        2d estimates (datapoints,estimates)
    vox_indices : list
        list of tuples, with voxel indices 
        (to reshape estimates according to original data shape)
    data_filename: str
        absolute filename of original data fitted
        
    Outputs
    -------
    out_file: str
        absolute output filename
    
    """ 
    # load nifti image to get header and shape
    data_img = nib.load(data_filename)
    data = data_img.get_fdata()
    
    # Re-arrange data
    estimates_mat = np.zeros((data.shape[0],data.shape[1],data.shape[2],estimates.shape[-1]))
    estimates_mat[:] = np.nan
    
    for est,vox in enumerate(vox_indices):
        estimates_mat[vox] = estimates[est]
        
    # Save estimates data
    new_img = nib.Nifti1Image(dataobj = estimates_mat, affine = data_img.affine, header = data_img.header)
    new_img.to_filename(filename)
    
    return filename



def combine_slices(file_list,outdir,num_slices=89, ax=2):
    
    """ High pass filter NIFTI run with gaussian kernel
    
    Parameters
    ----------
    file_list : list
        list of absolute filenames of all volumes to combine
    outdir : str
        path to save new file
    num_slices : int
        number of slices to combine
    ax: int
        which ax to stack slices
    
    Outputs
    -------
    out_file: str
        absolute output filename
    
    """
    

    for num in np.arange(num_slices):
        
        vol = [x for _,x in enumerate(file_list) if '_slice-{num}.nii.gz'.format(num = str(num).zfill(2)) in x]
        
        if len(vol)==0: # if empty
            raise NameError('Slice %s doesnt exist!'%str(num).zfill(2)) 
        
        else:
            nibber = nib.load(vol[0])
            data = np.array(nibber.dataobj)
            data = np.take(data, indices = num, axis=ax)
            
            if num == 0: # for first slice    
                outdata = data[np.newaxis,...] 
            else:
                outdata = np.vstack((outdata,data[np.newaxis,...] ))
                
    outdata = np.moveaxis(outdata,0,ax)
    
    out_file = op.split(vol[0])[-1].replace('_slice-{num}.nii.gz'.format(num = str(num).zfill(2)),'.nii.gz')
    out_file = op.join(outdir,out_file)
    
    # Save estimates data
    new_img = nib.Nifti1Image(dataobj = outdata, affine = nibber.affine, header = nibber.header)
    new_img.to_filename(out_file)
    
    return out_file

  
def add_alpha2colormap(colormap = 'rainbow_r', bins = 256, invert_alpha = False, cmap_name = 'costum',
                      discrete = False):

    """ add alpha channel to colormap,
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
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
        
        if discrete == True: # if we want a discrete colormap from list
            cmap = matplotlib.colors.ListedColormap(colormap)
            bins = int(len(colormap))

    # convert into array
    cmap_array = cmap(range(bins))
    
    # make alpha array
    if invert_alpha == True: # in case we want to invert alpha (y from 1 to 0 instead pf 0 to 1)
        _, alpha = np.meshgrid(np.linspace(0, 1, bins, endpoint=False), 1-np.linspace(0, 1, bins))
    else:
        _, alpha = np.meshgrid(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins, endpoint=False))
    
    # reshape array for map
    new_map = []
    for i in range(cmap_array.shape[-1]):
        new_map.append(np.tile(cmap_array[...,i],(bins,1)))

    new_map = np.moveaxis(np.array(new_map), 0, -1)

    # add alpha channel
    new_map[...,-1] = alpha
    
    fig = plt.figure(figsize=(1,1))
    ax = fig.add_axes([0,0,1,1])
    # plot 
    plt.imshow(new_map,
    extent = (0,1,0,1),
    origin = 'lower')
    ax.axis('off')

    rgb_fn = os.path.join(os.path.split(cortex.database.default_filestore)[
                          0], 'colormaps', cmap_name+'_alpha_bins_%d.png'%bins)

    #misc.imsave(rgb_fn, new_map)
    plt.savefig(rgb_fn, dpi = 200,transparent=True)
       
    return rgb_fn  