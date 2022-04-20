
import numpy as np
import os, sys
import os.path as op
import math
import random
import pandas as pd
import yaml

from psychopy import visual, tools, colors, event
import psychopy.tools.colorspacetools as ct
import itertools

import time
import colorsys
import seaborn as sns


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
    size_arr = arr.shape[0]
    dim = arr.shape[-1] if len(arr.shape) == 2 else 1
    
    for k in range(dim):

        # add some randomly uniform jitter 
        jit = np.concatenate((np.random.uniform(-max_val,-min_val,math.floor(size_arr * .5)),
                              np.random.uniform(min_val,max_val,math.ceil(size_arr * .5))))
        np.random.shuffle(jit)
        
        if k == 0 and dim == 1:
            output = arr + jit
        elif k == 0:
            output = arr[...,k] + jit
        else:
            output = np.vstack((output,arr[...,k] + jit))
    
    if dim > 1:
        output = output.T
    
    return(output)


def rgb255_2_hsv(arr):
    
    """ convert RGB 255 to HSV
    
    Parameters
    ----------
    arr: list/array
        1D list of rgb values
        
    """
    
    rgb_norm = np.array(arr)/255
    
    hsv_color = np.array(colorsys.rgb_to_hsv(rgb_norm[0],rgb_norm[1],rgb_norm[2]))
    hsv_color[0] = hsv_color[0] * 360
    
    return hsv_color


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


def get_object_positions(grid_pos,bar_midpoint_at_TR, bar_pass_direction_at_TR,
                      bar_width_pix, screen=np.array([1680,1050]), num_bar=1):
    
    """ function to subselect bar positions and
    return bar and background element positions (and number of elements for each object)
    
    Parameters
    ----------
    grid_pos : arr
        numpy array with all possible grid positions (N,2) -> (number of positions, [x,y])
    bar_midpoint_at_TR: arr
        numpy array with mid point position of bar(s) (B,[x,y]) with B=number of bars on screen
    bar_pass_direction_at_TR: arr
        numpy array of strings with bar direction(s) at that TR
    bar_width_pix: arr
        width of bar(s) in pixels for each resolution. 
        If float or array (1,) then same width used for all bars
    num_bar: int
        number of bars to be displayed
                
    """

    # define dictionary to save positions and number of elements
    # of all objects (bar(s) and background)
    output_dict = {}

    if np.isnan(bar_midpoint_at_TR).any():# or np.isnan(bar_pass_direction_at_TR).any(): # when nan, position is whole background

        output_dict['background'] = {'xys': grid_pos, 
                                    'nElements': grid_pos.shape[0]}
    else:

        # make sure "all" inputs are 2d arrays, if not make them. avoids crashing. 
        bar_midpoint_at_TR = np.array(bar_midpoint_at_TR) if len(np.array(bar_midpoint_at_TR).shape)>1 else np.array([bar_midpoint_at_TR])
        bar_pass_direction_at_TR = np.array(bar_pass_direction_at_TR) if len(np.array(bar_pass_direction_at_TR).shape)>0 else np.array([bar_pass_direction_at_TR])
        bar_width_pix = np.array(bar_width_pix)
        
        all_bar_ind = [] # append all bar position indices to later remove from background
 
        if all(x == num_bar for x in [bar_midpoint_at_TR.shape[0], bar_pass_direction_at_TR.shape[0]]):
        
  
            # iterate for number of bars on screen
            for ind in range(num_bar): 

                # first define bar width in pixels (might depend if vertical or horizontal bar pass)
                # and bounds for x and y positions

                if bar_pass_direction_at_TR[ind] in np.array(['L-R','R-L','horizontal']): # if horizontal bar pass

                    x_bounds = np.array([bar_midpoint_at_TR[ind][0] - bar_width_pix[0]/2,
                                         bar_midpoint_at_TR[ind][0] + bar_width_pix[0]/2])
                    y_bounds = np.array([-screen[1]/2,
                                         screen[1]/2])

                elif bar_pass_direction_at_TR[ind] in np.array(['U-D','D-U','vertical']): # if vertical bar pass

                    x_bounds = np.array([-screen[0]/2,
                                         screen[0]/2])
                    y_bounds = np.array([bar_midpoint_at_TR[ind][1] - bar_width_pix[1]/2, 
                                         bar_midpoint_at_TR[ind][1] + bar_width_pix[1]/2])


                # check which grid positions are within bounds for this conditions
                bar_ind = np.where(((grid_pos[...,0]>=min(x_bounds))&
                                    (grid_pos[...,0]<=max(x_bounds))&
                                    (grid_pos[...,1]>=min(y_bounds))&
                                    (grid_pos[...,1]<=max(y_bounds))
                                    ))[0]

                # append to dictionary 
                output_dict['bar%i'%ind] = {'xys': grid_pos[bar_ind], 
                                            'nElements': grid_pos[bar_ind].shape[0]}
                
                for _,p in enumerate(bar_ind):
                    all_bar_ind.append(p)
                
            # make mask to get background positions
            mask = np.ones(len(grid_pos), np.bool)
            mask[all_bar_ind] = 0
            
            output_dict['background'] = {'xys': grid_pos[mask], 
                                         'nElements': grid_pos[mask].shape[0]}

        else:
            raise ValueError('Number of bars different from shape of input arrays')
        

    return(output_dict)


def update_elements(ElementArrayStim, condition_settings, this_phase, elem_positions, grid_pos,
                   	monitor, screen = np.array([1680,1050]), position_jitter = None, orientation = True, 
                    background_contrast = None, luminance = None, update_settings = False, new_color = False, 
                    override_contrast = False, contrast_val = 1):
    
    """ update element array settings
    
    Parameters
    ----------
    ElementArrayStim: Psychopy object
    	ElementArrayStim to be updated 
    condition_settings: dict
        dictionary with all condition settings
    this_phase: str
        string with name of condition to be displayed
    elem_positions: arr
         numpy array with element positions to be updated and shown (N,2) -> (number of positions, [x,y])
         to be used for opacity update
    grid_pos: arr
        numpy array with element positions (N,2) of whole grid -> (number of positions, [x,y])
    monitor: object
        monitor object (to get monitor references for deg2pix transformation)
    screen: arr
        array with display resolution
    luminance: float or None
        luminance increment to alter color (used for flicker task)
    update_settings: bool
        choose if we want to update settings or not (mainly for color changes)
    new_color: array
        if we are changing color to be one not represented in settings (ca also be False if no new color used)
        
    """

    # we might be using diferent colors than the main 2, so set that straight
    if this_phase in list(condition_settings.keys()):
        main_color = this_phase 
        color_arr = condition_settings[this_phase]['element_color']
    else: 
        if this_phase in list(condition_settings['color_red']['task_color'].keys()):
            main_color = 'color_red'
        elif this_phase in list(condition_settings['color_green']['task_color'].keys()):
            main_color = 'color_green'
        color_arr = condition_settings[main_color]['task_color'][this_phase]['element_color']
    
    # set number of elements
    nElements = grid_pos.shape[0]

    ## to make colored gabor, need to do it a bit differently (psychopy forces colors to be opposite)
    # get rgb color and convert to hsv
    if np.array(new_color).any():
        hsv_color = rgb255_2_hsv(new_color)
        print('new hsv is %s'%str(hsv_color))
    else:
        hsv_color = rgb255_2_hsv(color_arr)

    if luminance != None: # if we want to make color more or less luminant

        hsv_color[-1] = luminance
        hsv_color[-1] = np.clip(hsv_color[-1],0.00001,1) # clip it so it doesn't go above 100% or below 0.0001% (latter avoids 0 division)
        print(hsv_color)

        # update settings dict with new color 
        updat_color_arr = [float(x*255.) for x in colorsys.hsv_to_rgb(hsv_color[0]/360.,hsv_color[1],hsv_color[2])]
        if this_phase in list(condition_settings.keys()):
            condition_settings[this_phase]['element_color'] = updat_color_arr
        else:
            condition_settings[main_color]['task_color'][this_phase]['element_color'] = updat_color_arr 


    grat_res = near_power_of_2(ElementArrayStim.sizes[0][0],near='previous') # use power of 2 as grating res, to avoid error
    
    # initialise grating
    grating = visual.filters.makeGrating(res=grat_res)
    grating_norm = (grating - np.min(grating))/(np.max(grating) - np.min(grating)) # normalize between 0 and 1
    
    # initialise a base texture 
    colored_grating = np.ones((grat_res, grat_res, 3)) 

    # replace the base texture red/green channel with the element color value, and the value channel with the grating

    colored_grating[..., 0] = hsv_color[0]
    colored_grating[..., 1] = hsv_color[1]
    colored_grating[..., 2] = grating_norm * hsv_color[2]

    elementTex = ct.hsv2rgb(colored_grating) # convert back to rgb

    # update element colors to color of the patch 
    element_color = np.ones((int(np.round(nElements)),3)) 
    
    # update element spatial frequency
    element_sfs = np.ones((nElements)) * condition_settings[main_color]['element_sf'] # in cycles/gabor width

    # update element orientation randomly
    if orientation == True:
        element_ori = np.random.uniform(0,360,nElements)
        ElementArrayStim.setOris(element_ori)


    # update element opacities

    # make grid and element position lists of lists
    list_grid_pos = [list(val) for _,val in enumerate(grid_pos)]
    list_elem_pos = [list(val) for _,val in enumerate(elem_positions)]

    # get indices of where one is in other
    list_indices = [list_grid_pos.index(list_elem_pos[i]) for i in range(len(list_elem_pos))]

    # set element contrasts
    element_contrast =  np.zeros(len(grid_pos))
    if override_contrast:
        element_contrast[list_indices] = contrast_val
    else:
        element_contrast[list_indices] = condition_settings[main_color]['element_contrast']
    #element_contrast[list_indices] = background_contrast if background_contrast != None else condition_settings[main_color]['element_contrast']
    
    # set opacities
    element_opacities = np.zeros(len(grid_pos))
    element_opacities[list_indices] = 1

    if position_jitter != None: # if we want to add jitter to (x,y) center of elements
        element_pos = jitter(grid_pos,
                            max_val = position_jitter,
                            min_val = 0)

        ElementArrayStim.setXYs(element_pos)

    # set all of the above settings
    ElementArrayStim.setTex(elementTex)
    ElementArrayStim.setSfs(element_sfs)
    ElementArrayStim.setOpacities(element_opacities)
    ElementArrayStim.setColors(element_color)
    ElementArrayStim.setContrs(element_contrast)
    print(element_contrast[list_indices[0]])

    # return updated settings, if such is the case
    if update_settings == True: 
        return(ElementArrayStim,condition_settings)
    else:
        return(ElementArrayStim)


def get_non_overlapping_indices(arr_shape=[2,8]):
    
    """ get array of indices, that don't overlap
    useful to make sure two bars with same orientation 
    don't overlap spatially
    
    Parameters
    ----------
    arr_shape : list/arr
        shape of indice arr -> [number of bars, number of positions]
        
    """ 
    # initialize empty array
    ind = np.empty((arr_shape[0],), dtype=list)
    
    # get indices for all possible horizontal bar positions
    for w in range(arr_shape[0]):
        ind[w] = np.arange(arr_shape[1])
        np.random.shuffle(ind[w])

        if w>0:
            while any(ind[w-1] == ind[w]): # shuffle until sure that bars in different positions
                np.random.shuffle(ind[w])


    return ind


def repeat_random_lists(arr,num_rep):
    
    """ repeat array, shuffled and stacked horizontally
    
    Parameters
    ----------
    arr : list/arr 
        array to repeat
    num_rep: int
        number of repetions
        
    """ 
    # initialize empty array
    new_arr = np.empty((num_rep,), dtype=list)
    
    # get indices for all possible horizontal bar positions
    for w in range(num_rep):
        random.shuffle(arr)
        new_arr[w] = arr.copy()

    return new_arr



def set_bar_positions(pos_dict = {'horizontal': [], 'vertical': []},
                     attend_condition = 'color_red', unattend_condition = 'color_green',
                      attend_orientation = ['vertical','horizontal'],
                      unattend_orientation = ['vertical','horizontal']):
    
    """ set bar positions for all feature trials
    
    Parameters
    ----------
    pos_dict: dict
        dictionary with bars positions
        "horizontal" array of shape (H,2) -> (number of possible horizontal positions, [x,y])
        with midpoint coordinates for horizontal bars
        "vertical" array of shape (H,2) -> (number of possible vertical positions, [x,y])
        with midpoint coordinates for vertical bars
    attend_condition: str
        name of attended condition (color)
    unattend_condition: str
        name of UNattended condition (color)
    attend_orientation: list/array
        possible bar orientations for attended condition
    unattend_orientation: list/array
        possible bar orientations for UNattended condition
        
    """
    
    # total number of trials
    num_trials = len(attend_orientation)*(pos_dict['horizontal'].shape[0] * pos_dict['vertical'].shape[0] + \
                                          pos_dict['horizontal'].shape[0] * (pos_dict['horizontal'].shape[0]-1))

    print('number of bar trials is %i'%num_trials)
    
    # define dictionary to save positions and directions
    # of all bars
    output_dict = {'attended_bar': {'color': attend_condition,
                                   'bar_midpoint_at_TR': [],
                                   'bar_pass_direction_at_TR': []},
                   'unattended_bar': {'color': unattend_condition,
                                      'bar_midpoint_at_TR': [],
                                      'bar_pass_direction_at_TR': []}
                  }

    # append all postions in dict 
    for att_ori in attend_orientation:

        for unatt_ori in unattend_orientation:

            if att_ori != unatt_ori: # if bar orientations orthogonal

                indice_pairs = list((x,y) for x in np.arange(pos_dict[att_ori].shape[0]) for y in np.arange(pos_dict[unatt_ori].shape[0]))

            else: # if bar orientations the same

                indice_pairs = list(itertools.permutations(np.arange(pos_dict[att_ori].shape[0]), 2))

            # fill attended dict
            output_dict['attended_bar']['bar_midpoint_at_TR'].append(np.array([pos_dict[att_ori][i] for i in np.array(indice_pairs)[...,0]]))
            output_dict['attended_bar']['bar_pass_direction_at_TR'].append(np.tile(att_ori, np.array(indice_pairs).shape[0]))

            # fill unattended dict
            output_dict['unattended_bar']['bar_midpoint_at_TR'].append(np.array([pos_dict[unatt_ori][i] for i in np.array(indice_pairs)[...,-1]]))
            output_dict['unattended_bar']['bar_pass_direction_at_TR'].append(np.tile(unatt_ori, np.array(indice_pairs).shape[0]))

    ## reshape and reorder arrays

    # make random indices
    random_ind = np.arange(num_trials)
    np.random.shuffle(random_ind)  

    for key in output_dict.keys():

        output_dict[key]['bar_midpoint_at_TR'] = np.vstack((v for v in output_dict[key]['bar_midpoint_at_TR']))[random_ind]
        output_dict[key]['bar_pass_direction_at_TR'] = np.hstack((v for v in output_dict[key]['bar_pass_direction_at_TR']))[random_ind]
    


    return(output_dict)

    
    
def leave_one_out_lists(input_list):

    """ make list of lists, leaving one item out in each iteration
    
    Parameters
    ----------
    input_list : list/arr
        list with all items
            
    """

    out_lists = []
    for x in input_list:
        out_lists.append([y for y in input_list if y != x])

    return out_lists




def gradual_shift(curr_point,
                  end_point = [12, 0.3], intersect = 0,
                  x_step = .5, slope = None, L = 0.3, function = 'logistic'):
    
    """ gradual increase/decrease values according to distribution
    
    Parameters
    ----------
    curr_point : list/array
        [x,y] for the current point to be updated
    end_point : list/array
        [x,y] for the ending point of the function
    intersect : int/float
        point where function intersects y-axis
    x_step : int/float
        time step (granularity) for function
    slope : int/float
        steepness of curve/line
    L : int/float
        curve's maximum value
    function : str
        name of mathematical function to be used (ex: 'linear', 'logistic')
        
    """
    # define x coordinate for next point, given step
    x_next = curr_point[0] + x_step
    
    if function == 'linear':
        
        # define slope for function
        # if not given, calculate the slope of function, given the end point
        k = end_point[1]/end_point[0] if slope == None else slope
        
        # define y coordinates for next point
        y_next = k * x_next + intersect
        
        
    elif function == 'logistic':
        
        # define slope for function
        # if not given, calculate the slope of function, given the end point
        k = 1 if slope == None else slope
        
        # define y coordinates for next point
        y_next = L / (1 + np.exp(-k*(x_next-end_point[0]/2)))
        
    
    # define next point array
    if (x_next >= end_point[0]) or (k > 0 and y_next >= end_point[1]) or (k < 0 and y_next <= end_point[1]): # if endpoint reached             
        # ensure saturation 
        x_next = end_point[0]
        y_next = end_point[1]         
    
        
    return x_next, y_next
    

def draw_instructions(win, instructions, keys = ['b'], visual_obj = [], 
                      color = (1, 1, 1), font = 'Helvetica Neue', pos = (0, 0), height = 30, #.65,
                        italic = True, alignHoriz = 'center', alignVert = 'center'):
    
    """ draw instructions on screen
    
    Parameters
    ----------
    win : object
        window object to draw on
    instructions : str
        instruction string to draw 
    key: list
        list of keys to skip instructions
    visual_obj: list
        if not empty, should have psychopy visual objects (to add to the display ex: side rectangles to limit display)
        
    """
    
    text = visual.TextStim(win = win,
                        text = instructions,
                        color = color, 
                        font = font, 
                        pos = pos, 
                        height = height,
                        italic = italic, 
                        alignHoriz = alignHoriz, 
                        alignVert = alignVert
                        )
    
    # draw text again
    text.draw()

    if len(visual_obj)>0:
        for w in range(len(visual_obj)):
            visual_obj[w].draw()
            
    win.flip()

    key_pressed = event.waitKeys(keyList = keys)

    return(key_pressed)


def randomize_conditions(cond_list):
    
    """ randomize condition names to attend in block
    
    Parameters
    ----------
    cond_list : array/list
        List/array (N,) of strings with condition names        
    """

    key_list = []
    for key in cond_list:
        if key != 'background': # we don't want to show background gabors
            key_list.append(key)

    np.random.shuffle(key_list)
    
    return np.array(key_list)


def save_bar_position(bar_dict, output_path):
    
    """ get bar position dictionary (with all positions for whole run), convert to pandas df and 
    save into appropriate output folder
    
    Parameters
    ----------
    bar_dict : dict
        position dictionary
    num_miniblock: int
        number of miniblocks
    output_path: str
        absolute path to output file
        
    """
    
    df_bar_position = pd.DataFrame(columns=['attend_condition', 'color', 'bar_midpoint_at_TR', 'bar_pass_direction_at_TR'])

    for key in bar_dict.keys():

        attend = 1 if key == 'attended_bar' else 0 

        df_bar_position = df_bar_position.append({'attend_condition': attend,
                                                'color': bar_dict[key]['color'],
                                                'bar_midpoint_at_TR': bar_dict[key]['bar_midpoint_at_TR'],
                                                'bar_pass_direction_at_TR': bar_dict[key]['bar_pass_direction_at_TR']
                                                    }, ignore_index=True)  

    df_bar_position.to_pickle(output_path)



def define_feature_trials(bar_pass_direction, bar_dict, empty_TR = 20, task_trial_TR = 2):
    
    """ create feature trials based on order of "type of stimuli" throught experiment  
    and bar positions in run. Outputs number and type of trials, and bar direction and midpoint position
    per trial
    
    Parameters
    ----------
    bar_pass_direction: array/list
        list with order of "type of stimuli" throught experiment
    bar_dict : dict
        position dictionary
    empty_TR: int
        number of TRs for empty intervals of experiment
    task_trial_TR: int
        number of TRs for task trials of experiment (bar presentation + ITI)
    mini_block_trials: int
        number of trials for a miniblock of experiment
        
    """
    
    # number of trials for actual task (Note - can be different than TR)
    num_task_trials = len(bar_dict['attended_bar']['bar_pass_direction_at_TR'])
    
    # create as many trials as TRs
    trial_number = 0
    bar_pass_direction_all = [] # list of lists with bar orientation at all TRs
    bar_pos_array = [] # list of lists with bar midpoint (x,y) for all TRs (if nan, then show background)
    trial_type_all = [] # list of lists with trial type at all TRs,

    for _,bartype in enumerate(bar_pass_direction):
        if bartype in np.array(['empty']): # empty screen
            trial_number += empty_TR
            trial_type_all = trial_type_all + np.repeat(bartype,empty_TR).tolist()
            bar_pass_direction_all = bar_pass_direction_all + np.repeat(bartype,empty_TR).tolist()

            for i in range(empty_TR):
                temp_pos_list = []
                for _,key in enumerate(bar_dict.keys()):
                    temp_pos_list.append(np.array([np.nan,np.nan], dtype=float)) 
                bar_pos_array.append(np.array(temp_pos_list).astype('float'))

        elif 'task' in bartype: # bars on screen
            trial_number += task_trial_TR * num_task_trials # NOTE one feature trial is 1TR of bar display + 1TR of empty screen
            trial_type_all = trial_type_all + list([bartype]+list(np.tile('empty', task_trial_TR-1))) * num_task_trials

            for t in range(num_task_trials):

                temp_dir_list = [] # temporary helper lists
                temp_pos_list = [] 

                for _,key in enumerate(bar_dict.keys()):
                    temp_dir_list.append(bar_dict[key]['bar_pass_direction_at_TR'][t])
                    temp_pos_list.append(bar_dict[key]['bar_midpoint_at_TR'][t])


                bar_pass_direction_all.append(temp_dir_list)
                bar_pos_array.append(np.array(temp_pos_list).astype('float'))
                
                # rest of TRs will be empty
                for i in range(task_trial_TR-1):
                    bar_pass_direction_all.append('empty')
                    
                    temp_pos_list = []
                    for _,key in enumerate(bar_dict.keys()):
                        temp_pos_list.append(np.array([np.nan,np.nan], dtype=float)) 
                    bar_pos_array.append(np.array(temp_pos_list).astype('float'))

    
    return trial_number, np.array(trial_type_all), np.array(bar_pass_direction_all), np.array(bar_pos_array)



def save_all_TR_info(bar_dict = [], trial_type = [], ecc_ind = {},
                     task_colors = {}, task_color_ind = {}, crossing_ind = [], output_path = ''):
    
    """ save all relevant trial infos in pandas df and 
    save into appropriate output folder
    
    Parameters
    ----------
    bar_dict : dict
        position dictionary
    trial_type : list/arr
        list of type of trial ('empty', 'task') for all TRs
    attend_color: str
        name of color that is attended
    output_path: str
        absolute path to output file
    hemifield: list/arr
        list of hemifield placement of attended bar, for all TRS (if no bar on screen then nan)
    crossing_ind: list/arr
        list of lists with plotting indices,for all TRS (if no bar on screen then nan)
        [useful for crossings (to know which bars on top)]
        
    """
    
    # get colors for attended task
    c_att = np.array([task_colors[bar_dict['attended_bar']['color']][v] for v in task_color_ind[bar_dict['attended_bar']['color']]])
    attend_task_color = np.full(len(trial_type), None)
    attend_task_color[np.where(trial_type == 'task')[0]] = c_att

    # do same for unattended task
    c_unatt = np.array([task_colors[bar_dict['unattended_bar']['color']][v] for v in task_color_ind[bar_dict['unattended_bar']['color']]])
    unattend_task_color = np.full(len(trial_type), None)
    unattend_task_color[np.where(trial_type == 'task')[0]] = c_unatt
    
    # get ecc - attended
    ecc_att = ecc_ind[bar_dict['attended_bar']['color']]
    attend_ecc = np.full(len(trial_type), None)
    attend_ecc[np.where(trial_type == 'task')[0]] = ecc_att

    # get ecc - unattended
    ecc_unatt = ecc_ind[bar_dict['unattended_bar']['color']]
    unattend_ecc = np.full(len(trial_type), None)
    unattend_ecc[np.where(trial_type == 'task')[0]] = ecc_unatt

    
    df_out = pd.DataFrame(columns=['trial_num','trial_type', 
                                   'attend_color', 'attend_task_color',
                                   'unattend_color', 'unattend_task_color',
                                   'bars', 'attend_ecc_ind', 'unattend_ecc_ind', 'crossing_ind'])
    
    for trl in range(len(trial_type)):
        
        df_out = df_out.append({'trial_num': trl, 
                                'trial_type': trial_type[trl],
                                'attend_color': bar_dict['attended_bar']['color'],
                                'attend_task_color' : attend_task_color[trl],
                                'unattend_color': bar_dict['unattended_bar']['color'],
                                'unattend_task_color': unattend_task_color[trl],
                                'bars': bar_dict.keys(),
                                'attend_ecc_ind': attend_ecc[trl], 
                                'unattend_ecc_ind': unattend_ecc[trl],
                                'crossing_ind': crossing_ind[trl],
                              }, ignore_index=True) 
        
    df_out.to_csv(output_path, index = False, header=True)


def get_square_positions(grid_pos, ecc_midpoint_at_trial, bar_width_pix, screen=np.array([1680,1050])):
    
    """ function to subselect square positions and
    return square and background element positions (and number of elements for each object)
    
    Parameters
    ----------
    grid_pos : arr
        numpy array with all possible grid positions (N,2) -> (number of positions, [x,y])
    ecc_midpoint_at_trial: float
        eccentricity (in pixels) of bar position for trial (if empty, then nan) 
    bar_width_pix: arr
        width of bar(s) in pixels for each resolution. 
        If float or array (1,) then same width used for all bars
                
    """
    
    # define dictionary to save positions and number of elements
    # of all objects (bar(s) and background)
    output_dict = {}

    if np.isnan(ecc_midpoint_at_trial).any(): # when nan, position is whole background

        output_dict['background'] = {'xys': grid_pos, 
                                    'nElements': grid_pos.shape[0]}
    else:
        
        # set bounds of the outer square
        x_bounds = np.array([ecc_midpoint_at_trial - bar_width_pix[0]/2,
                             np.abs(ecc_midpoint_at_trial) + bar_width_pix[0]/2])
        y_bounds = np.array([ecc_midpoint_at_trial - bar_width_pix[1]/2, 
                            np.abs(ecc_midpoint_at_trial) + bar_width_pix[1]/2])
        
        # check which grid positions are within bounds for this conditions
        bar_ind = np.where(((grid_pos[...,0]>=min(x_bounds))&
                            (grid_pos[...,0]<=max(x_bounds))&
                            (grid_pos[...,1]>=min(y_bounds))&
                            (grid_pos[...,1]<=max(y_bounds))
                            ))[0]

        # outer square position 
        outer_xys = grid_pos[bar_ind]
        
        # set bounds of the inner square
        x_bounds = np.array([ecc_midpoint_at_trial + bar_width_pix[0]/2,
                             np.abs(ecc_midpoint_at_trial) - bar_width_pix[0]/2])
        y_bounds = np.array([ecc_midpoint_at_trial + bar_width_pix[1]/2, 
                            np.abs(ecc_midpoint_at_trial) - bar_width_pix[1]/2])

        # check which outer square positions are not within inner square
        bar_ind = np.where(((outer_xys[...,0]<=min(x_bounds))|
                            (outer_xys[...,0]>=max(x_bounds))|
                            (outer_xys[...,1]<=min(y_bounds))|
                            (outer_xys[...,1]>=max(y_bounds))
                            ))[0]

        # append to dictionary 
        output_dict['bar0'] = {'xys': outer_xys[bar_ind], 
                                'nElements': outer_xys[bar_ind].shape[0]}
        
        ## make mask to get background positions
        # check which positions  within inner square
        inner_square_ind = np.where(((grid_pos[...,0]>min(x_bounds))&
                                    (grid_pos[...,0]<max(x_bounds))&
                                    (grid_pos[...,1]>min(y_bounds))&
                                    (grid_pos[...,1]<max(y_bounds))
                                    ))[0]
        backg_ind = np.concatenate((np.where(((grid_pos[...,0]<np.min(output_dict['bar0']['xys'][...,0]))|
                                    (grid_pos[...,0]>np.max(output_dict['bar0']['xys'][...,0]))|
                                    (grid_pos[...,1]<np.min(output_dict['bar0']['xys'][...,1]))|
                                    (grid_pos[...,1]>np.max(output_dict['bar0']['xys'][...,1]))
                                    ))[0],inner_square_ind))

        output_dict['background'] = {'xys': grid_pos[backg_ind], 
                                     'nElements': grid_pos[backg_ind].shape[0]}
        
    return(output_dict)



def get_average_color(filedir, settings, updated_color_names = ['orange','yellow','blue'],
                     color_categories = ['color_red', 'color_green'], average_ecc = True, ecc_ind = [0,1,2]):
    
    """ get average color 
    
    Parameters
    ----------
    filedir : str
        absolute directory where the new settings files are
    settings: dict
        settings dict, to be updated
    updated_color_names: array/list
        array of strings with names of colors to be updated
    color_categories: array/list
        names of general color categories, for bookeeping
    average_ecc: bool
        average over eccentricities?
    ecc_ind: array/list
        eccentricity indices to consider
            
    """
    
    # get settings files for all trials of flicker task
    flicker_files = [op.join(filedir,x) for _,x in enumerate(os.listdir(filedir)) if 'trial' in x and x.endswith('_updated_settings.yml')]
    all_trials = []
        
    for col in updated_color_names:
        
        new_color = []
        
        # loop over eccentricities
        for e in ecc_ind:
            
            # filenames for that color and ecc
            c_files = [file for file in flicker_files if col in file and 'ecc-%i'%e in file]

            if len(c_files) == 0:
                print('No files found for color %s and ecc %i, keeping initial settings'%(col, e))
            else:
                ecc_color = []
                for file in c_files:
                
                    # load updated settings for each trial 
                    with open(file, 'r', encoding='utf8') as f_in:
                        updated_settings = yaml.safe_load(f_in)

                    if col in color_categories: # if general color category (red, green)
                        ecc_color.append(updated_settings[col]['element_color'])
                    
                    elif col in ['pink','orange']: # if color variant from red
                        ecc_color.append(updated_settings['color_red']['task_color'][col]['element_color'])
                    
                    elif col in ['yellow','blue']: # if color variant from red
                        ecc_color.append(updated_settings['color_green']['task_color'][col]['element_color'])
                
                new_color.append(list(np.mean(ecc_color, axis=0)))
            
                # if we want to average over eccentricities
                if average_ecc: 
                    # actually update color in settings file
                    mean_col = list(np.mean(new_color, axis=0))
                    if col in color_categories:
                        settings['stimuli']['conditions'][col]['element_color'] = mean_col
                        print('new rgb255 for %s is %s'%(col,str(settings['stimuli']['conditions'][col]['element_color'])))
                    elif col in ['pink','orange']:
                        settings['stimuli']['conditions']['color_red']['task_color'][col]['element_color'] = mean_col
                        print('new rgb255 for %s is %s'%(col,str(settings['stimuli']['conditions']['color_red']['task_color'][col]['element_color'])))
                    elif col in ['yellow','blue']:
                        settings['stimuli']['conditions']['color_green']['task_color'][col]['element_color'] = mean_col
                        print('new rgb255 for %s is %s'%(col,str(settings['stimuli']['conditions']['color_green']['task_color'][col]['element_color'])))

                else:
                    all_trials.append(ecc_color)
                    print('NOT IMPLEMENTED YET - decide where to store ecc colors!!')
        
    ###### for now, to check, NEED TO CHANGE #######
    if average_ecc: 
        return settings
    else: 
        return all_trials 


def get_true_responses(bar_responses,drop_nan = False):
    
    """
    given array of trial position of attended bar
    output array of true responses for run
    (same vs dif, nan for first trial if drop_nan = False)
    """
    
    true_responses = []
    for i in range(len(bar_responses)):

        if i == 0:
            if drop_nan == False:
                true_responses.append(np.nan)

        elif bar_responses[i] == bar_responses[i-1]:
            true_responses.append('same')

        elif bar_responses[i] != bar_responses[i-1]:
            true_responses.append('different')
    true_responses = np.array(true_responses)
    
    return true_responses

    
def get_bar_eccentricity(all_bar_pos, 
                        hor_bar_pos_pix = [], 
                        ver_bar_pos_pix = [], 
                        bar_key = 'attended_bar'):

    """
    get eccentricity indice for bars on all trials
    returns array of ecc, with 0 - nearest and 2 being furthest
    """

    ecc_trials = []

    for t, bpos in enumerate(all_bar_pos[bar_key]['bar_midpoint_at_TR']):

        if all_bar_pos[bar_key]['bar_pass_direction_at_TR'][t] == 'horizontal':

            ind_pos = np.where(np.sum(hor_bar_pos_pix == bpos, axis = -1) == 2)[0][0]

        elif all_bar_pos[bar_key]['bar_pass_direction_at_TR'][t] == 'vertical':

            ind_pos = np.where(np.sum(ver_bar_pos_pix == bpos, axis = -1) == 2)[0][0]

        # adjust ecc indice to be between 0 and 2 (0 - nearest, 2 - furthest)
        val = ind_pos - len(ver_bar_pos_pix)/2
        if val < 0: 
            val = np.abs(val)-1

        ecc_trials.append(int(val))
        
    return np.array(ecc_trials)


class StaircaseCostum():
    
    def __init__(self,
                 startVal,
                 stepSize = .1,  # stepsize
                 nUp = 1,
                 nDown = 3,  # correct responses before stim goes down
                 minVal = 0,
                 maxVal = 1):
        
        # input variables
        self.startVal = startVal
        
        self.nUp = nUp
        self.nDown = nDown
        
        self.stepSize = stepSize
        
        self.minVal = minVal
        self.maxVal = maxVal
    

        self.data = []
        self.intensities = [startVal]
        self.reversalIntensities = []
        
        # correct since last stim change (minus are incorrect):
        self.correctCounter = 0
        self.incorrectCounter = 0
        self._nextIntensity = startVal
        
        self.increase = False
        self.decrease = False
        
        
    def addResponse(self, result):
        
        # add response to data
        self.data.append(result)
        
        # increment the counter of correct scores
        if result == 1:
            
            self.correctCounter += 1
            
            if self.correctCounter >= self.nDown:
                    
                self.decrease = True
                # reset counter
                self.correctCounter = 0
                    
        elif result == 0:
            
            self.incorrectCounter += 1
            
            if self.incorrectCounter >= self.nUp:
            
                self.increase = True
                # reset both counters
                self.correctCounter = 0
                self.incorrectCounter = 0
                
                    
        # calculate next intensity
        self.calculateNextIntensity()
        
        
    def calculateNextIntensity(self):
        
            
        # add reversal info
        if self.increase or self.decrease:
            self.reversalIntensities.append(self.intensities[-1])
            
        if self.increase:
            
            self._nextIntensity += self.stepSize
        
            # check we haven't gone out of the legal range
            if (self.maxVal is not None) and (self._nextIntensity > self.maxVal):
                self._nextIntensity = self.maxVal
                
            self.increase = False
            
        elif self.decrease:
            
            self._nextIntensity -= self.stepSize
        
            # check we haven't gone out of the legal range
            if (self.minVal is not None) and (self._nextIntensity < self.minVal):
                self._nextIntensity = self.minVal
                
            self.decrease = False
        
        # append intensities
        self.intensities.append(self._nextIntensity)
              
            
    def mean(self):
        
        return np.array(self.intensities).mean()
    
    def sd(self):
        
        return np.array(self.intensities).std()


def make_lum_plots(all_ecc_colors, out_dir = '', updated_color_names = ['orange','yellow','blue'], num_ecc = 3):
    
    # tile the keys, to make it easier to make dataframe
    color_names = np.repeat(updated_color_names, num_ecc)
    
    all_ecc_dict = {'color': [], 'ecc': [], 'R': [], 'G': [], 'B': [], 'luminance': []}
    for i, name in enumerate(color_names):
        
        if name == 'orange':
            ecc = i
        elif name == 'yellow':
            ecc = i-3
        elif name == 'blue':
            ecc = i-3*2
        
        for t in range(np.array(all_ecc_colors[i]).shape[0]):

            all_ecc_dict['color'].append(name)
            all_ecc_dict['ecc'].append(int(ecc))
            all_ecc_dict['R'].append(np.array(all_ecc_colors[i])[...,0][t])
            all_ecc_dict['G'].append(np.array(all_ecc_colors[i])[...,1][t])
            all_ecc_dict['B'].append(np.array(all_ecc_colors[i])[...,2][t])
            all_ecc_dict['luminance'].append(rgb255_2_hsv(all_ecc_colors[i][t])[-1])
        
    # convert to dataframe
    df_colors = pd.DataFrame(all_ecc_dict)
    
    ## make quick bar plot
    ax = sns.barplot(x = 'color', y = 'luminance', data = df_colors, hue = 'ecc')
    fig = ax.get_figure()
    fig.savefig(op.join(out_dir,"luminance_across_ecc.png")) 

    return df_colors
 
        