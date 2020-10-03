
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


def get_bar_positions(grid_pos,bar_midpoint_at_TR, bar_direction_at_TR,
                      bar_width_pix, screen=np.array([1680,1050]), num_bar=1):
    
    """ function to subselect bar positions 
    
    Parameters
    ----------
    grid_pos : arr
        numpy array with all possible grid positions (N,2) -> (number of positions, [x,y])
    bar_midpoint_at_TR: arr
        numpy array with mid point position of bar(s) (B,[x,y]) with B=number of bars on screen
    bar_direction_at_TR: arr
        numpy array of strings with bar direction(s) at that TR
    bar_width_pix: float/arr
        width of bar(s) in pixels for each resolution. 
        If float or array (1,) then same width used for all bars
    num_bar: int
        number of bars to be displayed
                
    """
    
    # make sure "all" inputs are arrays, avoid crashing
    bar_midpoint_at_TR = np.array([bar_midpoint_at_TR])
    bar_direction_at_TR = np.array([bar_direction_at_TR])
    bar_width_pix = np.array([bar_width_pix])
    
    # background positions will be same as grid (initially)
    background_pos = grid_pos.copy()
    
    # define dictionary to save positions and number of elements
    # of all objects (bar(s) and background)
    output_dict = {}
    
    if all(x == num_bar for x in [bar_midpoint_at_TR.shape[0], bar_direction_at_TR.shape[0]]):
    
        if bar_width_pix.shape[0] == 1:
            print('Only one bar width given, using same width for all bars')
            bar_width_pix = np.repeat(bar_width_pix,num_bar)
            
        # iterate for number of bars on screen
        for ind in range(num_bar): 

            # first define bar width in pixels (might depend if vertical or horizontal bar pass)
            # and bounds for x and y positions

            if bar_direction_at_TR[ind] in np.array(['L-R','R-L']): # if horizontal bar pass

                x_bounds = np.array([bar_midpoint_at_TR[ind][0] - bar_width_pix[ind]/2,
                                     bar_midpoint_at_TR[ind][0] + bar_width_pix[ind]/2])
                y_bounds = np.array([-screen[1]/2,
                                     screen[1]/2])

            elif bar_direction_at_TR[ind] in np.array(['U-D','D-U']): # if vertical bar pass

                x_bounds = np.array([-screen[0]/2,
                                     screen[0]/2])
                y_bounds = np.array([bar_midpoint_at_TR[ind][1] - bar_width_pix[ind]/2, 
                                     bar_midpoint_at_TR[ind][1] + bar_width_pix[ind]/2])


            # check which grid positions are within bounds for this conditions
            bar_ind = np.where(((grid_pos[...,0]>=min(x_bounds))&
                                (grid_pos[...,0]<=max(x_bounds))&
                                (grid_pos[...,1]>=min(y_bounds))&
                                (grid_pos[...,1]<=max(y_bounds))
                                ))[0]

            # append to dictionary 
            output_dict['bar%i'%ind] = {'xys': grid_pos[bar_ind], 
                                         'nElements': grid_pos[bar_ind].shape[0]}
            
            # remove bar positions from background positions
            background_pos = np.delete(background_pos, bar_ind, axis=0)
        
        output_dict['background'] = {'xys': background_pos, 
                                    'nElements': background_pos.shape[0]}

    else:
        raise ValueError('Number of bars different from shape of input arrays')
    

    return(output_dict)


def update_elements(win, condition_settings, this_phase, elem_positions, nElements,
                   monitor, screen=np.array([1680,1050])):
    
    """ update element array settings, returning an element array stim
    
    Parameters
    ----------
    win:
    	Window object
    condition_settings: dict
        dictionary with all condition settings
    this_phase: str
        string with name of condition to be displayed
    elem_positions: arr
         numpy array with element positions (N,2) -> (number of positions, [x,y])
    nElements: int
        number of elements in array
    monitor: object
        monitor object (to get monitor references for deg2pix transformation)
    screen: arr
        array with display resolution
        
    """
    
    # update element sizes
    element_sizes_px = tools.monitorunittools.deg2pix(condition_settings[this_phase]['element_size'], 
                                                      monitor) 
    element_sizes = np.ones((nElements)) * element_sizes_px 
        
    # update element texture
    if this_phase in ('color_green','color_red'):

        # to make colored gabor, need to do it a bit differently (psychopy forces colors to be opposite)
        grat_res = near_power_of_2(element_sizes[0],near='previous') # use power of 2 as grating res, to avoid error
        grating = visual.filters.makeGrating(res=grat_res)

        # initialise a 'black' texture 
        colored_grating = np.ones((grat_res, grat_res, 3)) #* -1.0

        # replace the red/green channel with the grating
        if this_phase == 'color_red': 
            colored_grating[..., 0] = grating 
        else:
            colored_grating[..., 1] = grating 

        elementTex = colored_grating
    else:
        elementTex = 'sin'
    
    
    # update element contrasts
    element_contrast =  np.ones((nElements)) * condition_settings[this_phase]['element_contrast']
    
    # update element spatial frequency
    element_sfs_pix = tools.monitorunittools.deg2pix(condition_settings[this_phase]['element_sf'], 
                                                     monitor) # (transform cycles/degree to cycles/pixel)
    element_sfs = np.ones((nElements)) * element_sfs_pix
        
    # update element orientation (half ori1, half ori2)
    ori_arr = np.concatenate((np.ones((math.floor(nElements * .5))) * condition_settings[this_phase]['element_ori'][0], 
                              np.ones((math.ceil(nElements * .5))) * condition_settings[this_phase]['element_ori'][1]))

    # add some jitter to the orientations
    ori_arr = jitter(ori_arr,
                     max_val = condition_settings[this_phase]['ori_jitter_max'],
                     min_val = condition_settings[this_phase]['ori_jitter_min']) 

    np.random.shuffle(ori_arr) # shuffle the orientations

    # update element opacities
    element_opacities = np.ones((nElements))

    # update element colors 
    element_color = np.ones((int(np.round(nElements)),3)) * np.array(condition_settings[this_phase]['element_color'])
    
    ElementArrayStim = visual.ElementArrayStim(win = win, 
												nElements = nElements,
                                                units = 'pix', 
                                                elementTex = elementTex, 
                                                elementMask = 'gauss',
                                                sizes = element_sizes, 
                                                sfs = element_sfs, 
                                                xys = elem_positions, 
                                                oris = ori_arr,
                                                contrs = element_contrast, 
                                                colors = element_color, 
                                                colorSpace = 'rgb') 

    return(ElementArrayStim)





