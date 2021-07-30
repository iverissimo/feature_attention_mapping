
# useful functions to use in other scripts

import os
import hedfpy
import numpy as np

from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns


def edf2h5(edf_files, hdf_file, pupil_hp = 0.01, pupil_lp = 6.0):
    
    """ convert edf files (can be several)
    into one hdf5 file, for later analysis
    
    Parameters
    ----------
    edf_files : List/arr
        list of absolute filenames for edf files
    hdf_file : str
        absolute filename of output hdf5 file

    Outputs
    -------
    all_alias: List
        list of strings with alias for each run
    
    """
    
    # first check if hdf5 already exists
    if os.path.isfile(hdf_file):
        print('The file %s already exists, skipping'%hdf_file)
        
        all_alias = [os.path.split(ef)[-1].replace('.edf','') for _,ef in enumerate(edf_files)]
        
    else:
        ho = hedfpy.HDFEyeOperator(input_object=hdf_file)

        all_alias = []

        for ef in edf_files:
            alias = os.path.splitext(os.path.split(ef)[1])[0] #name of data for that run
            ho.add_edf_file(ef)
            ho.edf_message_data_to_hdf(alias = alias) #write messages ex_events to hdf5
            ho.edf_gaze_data_to_hdf(alias = alias, pupil_hp = pupil_hp, pupil_lp = pupil_lp) #add raw and preprocessed data to hdf5   

            all_alias.append(alias)
    
    return all_alias



def dva_per_pix(height_cm,distance_cm,vert_res_pix):

    """ calculate degrees of visual angle per pixel, 
    to use for screen boundaries when plotting/masking
    
    Parameters
    ----------
    height_cm : int
        screen height/width
    distance_cm: float
        screen distance (same unit as height)
    vert_res_pix : int
        vertical/horizontal resolution of screen
    
    Outputs
    -------
    deg_per_px : float
        degree (dva) per pixel
    
    """

    # screen size in degrees / vertical resolution
    deg_per_px = (2.0 * np.degrees(np.arctan(height_cm /(2.0*distance_cm))))/vert_res_pix

    return deg_per_px


def mean_dist_deg(xx, yy, origin = [None], screen_res = [1920,1080], screen_width = 69.8, screen_dist = 210):
    
    """ calculate mean distance in deg (and std)
    from an origin point 
    given list of positions in pix
    
    Parameters
    ----------
    xx : list/array
        list of x gaze positions
    yy : list/array
        list of y gaze positions
    origin : list
        origin point [x,y]
    screen_res : list
        resolution of screen
    screen_width : float
        screen width 
    screen_dist: float
        screen distance (same units as width)

    
    Outputs
    -------
    mean_dist_deg : float
        in degree (dva)
    std_dist_deg : float
        in degree (dva)
    
    """
    
    # calculate degrees per pixel for the setup
    deg_per_px = (2.0 * np.degrees(np.arctan(screen_width /(2.0*screen_dist))))/screen_res[0]
    
    # calculate distance of gaze from origin
    if np.array(origin).any() == None: # if not given, defaults to center of screen
        origin = np.array(screen_res)/2
    
    dist_pix = np.sqrt((np.array(xx-origin[0])**2)+(np.array(yy-origin[1])**2))
    
    # convert from pixel to dva
    dist_deg = dist_pix*deg_per_px
    
    # calculate mean and std
    mean_dist_deg = np.mean(dist_deg)
    std_dist_deg = np.std(dist_deg)
    
    return mean_dist_deg, std_dist_deg


def plot_gaze_kde(df_gaze, outpath, run = 0, conditions = ['green_horizontal','green_vertical','red_horizontal','red_vertical'],
                 screen = [1920,1080], downsample = 10, color = {'green_horizontal':(0,1,0),'green_vertical':(0,1,0),
                                                               'red_horizontal':(1,0,0),'red_vertical': (1,0,0)}):
    
    """ plot kde per run
    
    Parameters
    ----------
    df_gaze : pd dataframe
        with gaze data
    outpath : str
        absolute path to save plot
    run : int
        run to plot
    downsample : int
        value to downsample gaze data, to make it faster
    
    """
    # plot gaze density

    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(30,15))
    #fig.subplots_adjust(hspace = .25, wspace=.001)

    plt_counter = 0

    for i in range(2):

        for w in range(2):

            data_x = df_gaze.loc[(df_gaze['run'] == run) &
                                        (df_gaze['attend_condition'] == conditions[plt_counter])]['gaze_x'].values[0]
            data_y = df_gaze.loc[(df_gaze['run'] == run) &
                                        (df_gaze['attend_condition'] == conditions[plt_counter])]['gaze_y'].values[0]

            # turn string list to actual list (workaround for pandas)
            if type(data_x) != list:

                data_x = literal_eval(data_x)[::downsample] 
                data_y = literal_eval(data_y)[::downsample]
            
            else:
                data_x = data_x[::downsample]
                data_y = data_y[::downsample]

            # get mean gaze and std
            mean_gaze, mean_std = mean_dist_deg(data_x, data_y)
            
            # downsample data to make kde faster
            a = sns.kdeplot(ax = axs[i,w], x = data_x, y = data_y, fill = True, color = color[conditions[plt_counter]])
            a.tick_params(labelsize=15)

            axs[i][w].set_title(conditions[plt_counter],fontsize=18)
            axs[i][w].text(10, 10, 'mean gaze distance from center = %.2f +/- %.2f dva'%(mean_gaze, mean_std),
                          fontsize = 15)

            axs[i][w].set_ylim(0, screen[1])
            axs[i][w].set_xlim(0, screen[0])

            axs[i][w].axvline(screen[0]/2, lw=0.5, color='k',alpha=0.5)
            axs[i][w].axhline(screen[1]/2, lw=0.5, color='k',alpha=0.5)

            axs[i][w].add_artist(plt.Circle((screen[0]/2, screen[1]/2), radius=102, color='grey',alpha=0.5 , fill=False)) # add circle of 1dva radius, for reference 

            plt_counter += 1
            
            
    fig.savefig(os.path.join(outpath,'gaze_KDE_run-%s.png' %str(run).zfill(2)))



def plot_sacc_hist(df_sacc, outpath, run = 0, conditions = ['green_horizontal','green_vertical','red_horizontal','red_vertical'],
                 color = {'green_horizontal':(0,1,0),'green_vertical':(0,1,0),'red_horizontal':(1,0,0),'red_vertical': (1,0,0)}):
    
    
    """ plot saccade histogram
    
    Parameters
    ----------
    df_sacc : pd dataframe
        with saccade data
    outpath : str
        absolute path to save plot
    run : int
        run to plot
    
    """
    # plot gaze density

    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(30,15))
    #fig.subplots_adjust(hspace = .25, wspace=.001)

    plt_counter = 0

    for i in range(2):

        for w in range(2):

            amp = df_sacc.loc[(df_sacc['run'] == run) &
                                (df_sacc['attend_condition'] == conditions[plt_counter])]['expanded_amplitude'].values[0]

            if amp == [0]: # if 0, then no saccade

                amp = [np.nan]

            a = sns.histplot(ax = axs[i,w], 
                            x = amp,
                            color = color[conditions[plt_counter]])
            a.tick_params(labelsize=15)
            a.set_xlabel('Amplitude (degrees)',fontsize=15, labelpad = 15)

            axs[i][w].set_title(conditions[plt_counter],fontsize=18)
            axs[i][w].axvline(0.5, lw=0.5, color='k',alpha=0.5,linestyle='--')
            
            # count number of saccades with amplitude bigger than 0.5 deg
            sac_count = len(np.where(np.array(amp) >= 0.5)[0])
            axs[i][w].text(0.7, 0.9,'%i saccades > 0.5deg'%(sac_count), 
                           ha='center', va='center', transform=axs[i][w].transAxes,
                          fontsize = 15)

            plt_counter += 1
            
            
    fig.savefig(os.path.join(outpath,'sacc_hist_run-%s.png' %str(run).zfill(2)))



def get_saccade_angle(arr, angle_unit='radians'):
    
    """
    convert vector position of saccade to angle
    given a list of vector locations (N x 2)
    """
    
    # compute complex location
    complex_list = [sac[0] + sac[1]*1j for _,sac in enumerate(arr)]
    
    if angle_unit == 'degrees':
        deg_unit = True
    else:
        deg_unit = False
    
    # actually calculate angle
    angles = np.angle(complex_list, deg = deg_unit)
    
    return list(angles)


def rose_plot(ax, angles, bins=16, density=None, offset=0, lab_unit="degrees",
              start_zero=False,color='g',alpha=0.5, **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    """

    if not np.isnan(angles[0]):    
        # Wrap angles to [-pi, pi)
        angles = (angles + np.pi) % (2*np.pi) - np.pi

        # Set bins symetrically around zero
        if start_zero:
            # To have a bin edge at zero use an even number of bins
            if bins % 2:
                bins += 1
            bins = np.linspace(-np.pi, np.pi, num=bins+1)

        # Bin data and record counts
        count, bin = np.histogram(angles, bins=bins)

        # Compute width of each bin
        widths = np.diff(bin)

        # By default plot density (frequency potentially misleading)
        if density is None or density is True:
            # Area to assign each bin
            area = count / angles.size
            # Calculate corresponding bin radius
            radius = (area / np.pi)**.5
        else:
            radius = count

        # Plot data on ax
        ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
               edgecolor='0.5', fill=True, linewidth=1,color=color,alpha=alpha)

        # Set the direction of the zero angle
        ax.set_theta_offset(offset)

        # Remove ylabels, they are mostly obstructive and not informative
        ax.set_yticks([])

        if lab_unit == "radians":
            label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                      r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
            ax.set_xticklabels(label)

