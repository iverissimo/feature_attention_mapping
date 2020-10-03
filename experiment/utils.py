
import numpy as np
import os, sys
import math
from psychopy import visual, tools


def jitter(arr,max_val=1,min_val=0.5):

    """ Add random jitter to an array
    
    Parameters
    ----------
    arr : array
        List/array (N,) or (N,2) of values to add jitter to
    max_val : int/float
        maximun amount to add/subtract
    min_val: int/float
        minimum amount to add/subtract
        
    """

    # element positions (#elements,(x,y))
    size_arr = len(arr)

    # add some randomly uniform jitter 
    jit = np.concatenate((np.random.uniform(-max_val,-min_val,math.floor(size_arr * .5)),
                          np.random.uniform(min_val,max_val,math.ceil(size_arr * .5))))
    np.random.shuffle(jit)

    arr += jit

    return(arr)


def near_power_of_2(x,near='previous'):
    """ Get nearest power of 2
    
    Parameters
    ----------
    x : int/float
        value for which we want to find the nearest power of 2
    near : str
        'previous' or 'next' to indicate if floor or ceiling power of 2        
    """
    if x == 0:
        val = 1
    else:
        if near == 'previous':
            val = 2**math.floor(math.log2(x))
        elif near == 'next':
            val = 2**math.ceil(math.log2(x))

    return val


def update_display(ElementArrayStim, background_settings, condition_settings,
                   bar_midpoint_at_TR, bar_direction_at_TR, this_phase,
                   bar_width_pix, monitor, screen=np.array([1680,1050])):
    
    """ update display elements, given background settings and 
    new settings to change (for bar(s))
    
    Parameters
    ----------
    ElementArrayStim : Psychopy ElementArrayStim
        stimulus class defines a field of elements whose behaviour can be independently controlled
    background_settings : dict
        dictionary with keys = ElementArrayStim settings, and values corresponding to the backgroung settings
        (note that len of arrays should be equal to total number of grid positions)
    condition_settings: dict
        dictionary with all condition settings
    bar_midpoint_at_TR: arr
        numpy array with mid point position of bar (N,[x,y]) with N=number of bars on screen
    bar_direction_at_TR: arr
        array of strings with bar direction(s) at that TR
    this_phase: arr
        array of strings with name of condition(s) for that "phase"
    bar_width_pix: arr
        width of bar in pixels for each resolution
    monitor: object
        monitor object (to get monitor references for deg2pix transformation)
    screen: arr
        array with display resolution
        
    """
    # make sure "all" inputs are arrays, avoid crashing
    bar_midpoint_at_TR = np.array([bar_midpoint_at_TR])
    bar_direction_at_TR = np.array([bar_direction_at_TR])
    this_phase = np.array([this_phase])
    
    # define new element settings, by first copying the background ones
    new_contrs = background_settings['contrs'].copy()
    new_sfs = background_settings['sfs'].copy()
    new_oris = background_settings['oris'].copy(); np.random.shuffle(new_oris) # shuffle the orientations
    new_colors = background_settings['colors'].copy()
    new_elementTex = background_settings['elementTex']
    
    
    # update those settings in loop
    for ind in range(len(this_phase)): # iterate settings for number of bars on screen
        
        # first define bar width in pixels (might depend if vertical or horizontal bar pass)
        # and bounds for x and y positions
        
        if bar_direction_at_TR[ind] in np.array(['L-R','R-L']): # if horizontal bar pass
                        
            x_bounds = np.array([bar_midpoint_at_TR[ind][0] - bar_width_pix[0]/2,
                                 bar_midpoint_at_TR[ind][0] + bar_width_pix[0]/2])
            y_bounds = np.array([-screen[1]/2,
                                 screen[1]/2])

        elif bar_direction_at_TR[ind] in np.array(['U-D','D-U']): # if vertical bar pass
            
            x_bounds = np.array([-screen[0]/2,
                                 screen[0]/2])
            y_bounds = np.array([bar_midpoint_at_TR[ind][1] - bar_width_pix[1]/2, 
                                 bar_midpoint_at_TR[ind][1] + bar_width_pix[1]/2])
            

            
        # check which grid positions are within bounds for this conditions
        bar_ind = np.where(((background_settings['xys'][...,0]>=min(x_bounds))&
                            (background_settings['xys'][...,0]<=max(x_bounds))&
                            (background_settings['xys'][...,1]>=min(y_bounds))&
                            (background_settings['xys'][...,1]<=max(y_bounds))
                            ))[0]
        
        
        # update element contrasts
        new_contrs[bar_ind] = condition_settings[this_phase[ind]]['element_contrast']
        
        # update element spatial frequency
        element_sfs_pix = tools.monitorunittools.deg2pix(condition_settings[this_phase[ind]]['element_sf'], 
                                                         monitor) # (transform cycles/degree to cycles/pixel)
        new_sfs[bar_ind] = element_sfs_pix
        
        # update element orientation (half ori1, half ori2)
        ori_arr = np.concatenate((np.ones((math.floor(len(bar_ind) * .5))) * condition_settings[this_phase[ind]]['element_ori'][0], 
                                  np.ones((math.ceil(len(bar_ind) * .5))) * condition_settings[this_phase[ind]]['element_ori'][1]))
        
        # add some jitter to the orientations
        ori_arr = jitter(ori_arr,
                         max_val = condition_settings[this_phase[ind]]['ori_jitter_max'],
                         min_val = condition_settings[this_phase[ind]]['ori_jitter_min']) 

        np.random.shuffle(ori_arr) # shuffle the orientations
        
        for w,val in enumerate(ori_arr): 
            new_oris[bar_ind[w]] = val
         
        # update element colors
        new_colors[bar_ind] = np.array(condition_settings[this_phase[ind]]['element_color'])
        
        # update element texture
        
        if this_phase[ind] in ('color_green','color_red'):
            
            # to make colored gabor, need to do it a bit differently (psychopy forces colors to be opposite)
            grat_res = near_power_of_2(background_settings['sizes'][0],near='previous') # use power of 2 as grating res, to avoid error
            grating = visual.filters.makeGrating(res=grat_res)

            # initialise a 'black' texture 
            colored_grating = np.ones((grat_res, grat_res, 3)) #* -1.0
            
            # replace the red/green channel with the grating
            if this_phase[ind] == 'color_red': 
                colored_grating[..., 0] = grating 
            else:
                colored_grating[..., 1] = grating 

            new_elementTex = colored_grating
        else:
            new_elementTex = 'sin'

            
    # actually set the settings
    ElementArrayStim.setContrs(new_contrs)#, log=False)
    ElementArrayStim.setSfs(new_sfs)
    ElementArrayStim.setOris(new_oris)
    ElementArrayStim.setColors(new_colors)
    ElementArrayStim.setTex(new_elementTex)#, log=False)

    return(ElementArrayStim)


