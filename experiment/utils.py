
import numpy as np
import os, sys
import math
import random
import pandas as pd

from psychopy import visual, tools, colors, event
import itertools

import time
import colorsys


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
                   	monitor, screen = np.array([1680,1050]), position_jitter = None, orientation = True, background_contrast = None):
    
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
        
    """
    
    # set number of elements
    nElements = grid_pos.shape[0]

    ## to make colored gabor, need to do it a bit differently (psychopy forces colors to be opposite)
    # get rgb color and convert to hsv
    hsv_color = rgb255_2_hsv(condition_settings[this_phase]['element_color'])

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

    elementTex = colors.hsv2rgb(colored_grating) # convert back to rgb

    # update element colors to color of the patch 
    element_color = np.ones((int(np.round(nElements)),3)) 
    
    # update element spatial frequency
    element_sfs = np.ones((nElements)) * condition_settings[this_phase]['element_sf'] # in cycles/gabor width

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
    element_contrast[list_indices] = background_contrast if background_contrast != None else condition_settings[this_phase]['element_contrast']
    
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
    ElementArrayStim.setContrs(element_contrast)
    ElementArrayStim.setSfs(element_sfs)
    ElementArrayStim.setColors(element_color)
    ElementArrayStim.setOpacities(element_opacities)


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



def set_bar_positions(attend_block_conditions,horizontal_pos,vertical_pos,
                         mini_blocks = 4, num_bars = 4, num_ver_bars = 2, num_hor_bars = 2):
    
    """ set bar positions for all feature trials
    
    Parameters
    ----------
    attend_block_conditions : arr
        array of strings with attended condition for each block (will then be the first condition of each block
        of the return dictionary)
    horizontal_pos: arr
        array of shape (H,2) -> (number of possible horizontal positions, [x,y])
        with midpoint coordinates for horizontal bars
    vertical_pos: arr
        array of shape (V,2) -> (number of possible vertical positions, [x,y])
        with midpoint coordinates for vertical bars
    mini_blocks: int
        number of mini blocks in run
    num_bars: int
        number of bars to be displayed simultaneously
    num_ver_bars: int
        number of vertical bars to be displayed simultaneously
    num_hor_bars:
        number of horizontal bars to be displayed simultaneously
        
    """
    
    # make list of bar conditon names per mini block
    # to associate names to position
    bar_list = np.empty([mini_blocks, num_bars], dtype=list)

    for b in range(mini_blocks):
        # get name of non attended positions for that block
        non_attend_cond = [x for x in attend_block_conditions if x != attend_block_conditions[b]]

        for c in range(num_bars):
            if c == 0:
                bar_list[b][c] = attend_block_conditions[b]
            else:
                bar_list[b][c] = non_attend_cond[c-1]

    # define dictionary to save positions and directions
    # of all bars
    output_dict = {}
    for blk in range(mini_blocks):
        output_dict['mini_block_%i'%blk] = {}

    num_trials = horizontal_pos.shape[0] * vertical_pos.shape[0] # all trials
    num_conditions = horizontal_pos.shape[1] + vertical_pos.shape[1] # all conditions
    
    # actually store positions
    for blk in range(mini_blocks):

        # first define for all conditions in block, which will be 
        # vertical bar pass, which will be horizontal bar pass
        for k,cond in enumerate(bar_list[blk]):

            cond_position = []

            if 'vertical' in cond:
                cond_direction = np.repeat('horizontal',num_trials)
            elif 'horizontal' in cond:
                cond_direction = np.repeat('vertical',num_trials)

            # append to dictionary 
            output_dict['mini_block_%i'%blk][cond] = {'bar_midpoint_at_TR': cond_position, 
                                                     'bar_pass_direction_at_TR': cond_direction}  

        # now according to bar direction (horizontal vs vertical)
        # set x,y coordinates for bar midpoint

        ## get non overlapping indices for vertical and horizontal bar positions
        # initialize empty lists for indices
        cond_ind_ver = np.empty([num_ver_bars, vertical_pos.shape[0]], dtype=list)
        cond_ind_hor = np.empty([num_hor_bars, horizontal_pos.shape[0]], dtype=list)

        # get non-overlapping indices for vertical, throughout trials
        for v in range(vertical_pos.shape[0]):

            cond_ind_ver[0][v], cond_ind_ver[1][v] = get_non_overlapping_indices(arr_shape=[num_ver_bars,vertical_pos.shape[0]])

        # get non-overlapping indices for horizontal, throughout trials
        hor1, hor2 = get_non_overlapping_indices(arr_shape=[num_hor_bars,horizontal_pos.shape[0]])

        for h in range(len(hor1)):

            cond_ind_hor[0][h] = np.repeat(hor1[h],horizontal_pos.shape[0])
            cond_ind_hor[1][h] = np.repeat(hor2[h],horizontal_pos.shape[0])

        # reshape the indice arrays, and put in new arrays
        ind_ver = np.empty([num_ver_bars, ], dtype=list)
        ind_hor = np.empty([num_hor_bars, ], dtype=list)

        ind_hor[0] = np.concatenate(cond_ind_hor[0]).ravel()
        ind_hor[1] = np.concatenate(cond_ind_hor[1]).ravel()
        ind_ver[0] = np.concatenate(cond_ind_ver[0]).ravel()
        ind_ver[1] = np.concatenate(cond_ind_ver[1]).ravel()

        # make indice array, that is shuffled, to randomly draw positions
        random_ind = np.arange(num_trials)
        np.random.shuffle(random_ind)

        for trl in range(num_trials): # for each trial
    
            vert_bool = False # boolean markers to keep track of conditions
            hor_bool = False

            for k,cond in enumerate(bar_list[blk]): # iterate per condition

                # get coordinates for vertical bars
                if output_dict['mini_block_%i'%blk][cond]['bar_pass_direction_at_TR'][trl] == 'vertical':

                    m = 0 if vert_bool == False else 1
                    coord = vertical_pos[ind_ver[m][random_ind[trl]]] # save coordinates
                    vert_bool = True # update bool marker

                # get coordinates for horizontal bars
                elif output_dict['mini_block_%i'%blk][cond]['bar_pass_direction_at_TR'][trl] == 'horizontal':

                    m = 0 if hor_bool == False else 1
                    coord = horizontal_pos[ind_hor[m][random_ind[trl]]] # save coordinates
                    hor_bool = True # update bool marker

                # now append coordinates to corresponding condition list
                output_dict['mini_block_%i'%blk][cond]['bar_midpoint_at_TR'].append(coord)


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
                      color = (1, 1, 1), font = 'Helvetica Neue', pos = (0, 0), height = .65,
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


def save_bar_position(bar_dict,num_miniblock, output_path):
    
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
    
    df_bar_position = pd.DataFrame(columns=['mini_block','attend_condition', 'condition', 'bar_midpoint_at_TR', 'bar_pass_direction_at_TR'])

    for blk in range(num_miniblock):
    
        for i,cond in enumerate(bar_dict['mini_block_%i'%blk].keys()):

            attend = 1 if i==0 else 0

            df_bar_position = df_bar_position.append({'mini_block': blk, 
                                                      'attend_condition': attend,
                                                      'condition': cond,
                                                      'bar_midpoint_at_TR': bar_dict['mini_block_%i'%blk][cond]['bar_midpoint_at_TR'],
                                                      'bar_pass_direction_at_TR': bar_dict['mini_block_%i'%blk][cond]['bar_pass_direction_at_TR']
                                                     }, ignore_index=True) 


    df_bar_position.to_pickle(output_path)




def define_feature_trials(bar_pass_direction, bar_dict, empty_TR = 20, cue_TR = 3, mini_block_TR = 64):
    
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
    cue_TR: int
        number of TRs for cue intervals of experiment
    mini_block_TR: int
        number of TRs for miniblocks of experiment
        
    """
    
    #create as many trials as TRs
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
                bar_pos_array.append(np.array([np.nan,np.nan]))

        elif 'cue' in bartype: # cue on screen
            trial_number += cue_TR
            trial_type_all = trial_type_all + np.repeat(bartype,cue_TR).tolist()
            bar_pass_direction_all = bar_pass_direction_all + np.repeat('empty',cue_TR).tolist()

            for i in range(cue_TR):
                bar_pos_array.append(np.array([np.nan,np.nan]))

        elif 'mini_block' in bartype: # bars on screen
            trial_number += 2*mini_block_TR  # NOTE one feature trial is 1TR of bar display + 1TR of empty screen
            trial_type_all = trial_type_all + list([bartype,'empty'])*mini_block_TR

            # get mini block condition keys
            miniblock_cond_keys = list(bar_dict[bartype].keys())

            for t in range(mini_block_TR):

                temp_dir_list = [] # temporary helper lists
                temp_pos_list = [] 

                for _,key in enumerate(miniblock_cond_keys):
                    temp_dir_list.append(bar_dict[bartype][key]['bar_pass_direction_at_TR'][t])
                    temp_pos_list.append(bar_dict[bartype][key]['bar_midpoint_at_TR'][t])


                bar_pass_direction_all.append(temp_dir_list)
                bar_pass_direction_all.append('empty')
                bar_pos_array.append(temp_pos_list)
                bar_pos_array.append(np.array([np.nan,np.nan]))

    
    return trial_number, np.array(trial_type_all), np.array(bar_pass_direction_all), np.array(bar_pos_array)


def save_all_TR_info(bar_dict, trial_type, attend_block_conditions, hemifield, crossing_ind, output_path):
    
    """ save all relevant trial infos in pandas df and 
    save into appropriate output folder
    
    Parameters
    ----------
    bar_dict : dict
        position dictionary
    trial_type : list/arr
        list of type of trial ('cue', 'empty', 'miniblock') for all TRs
    attend_block_conditions: list/arr
        list of strings to attend in each miniblock
    output_path: str
        absolute path to output file
    hemifield: list/arr
        list of hemifield placement of attended bar, for all TRS (if no bar on screen then nan)
    crossing_ind: list/arr
        list of lists with plotting indices,for all TRS (if no bar on screen then nan)
        [useful for crossings (to know which bars on top)]
        
    """
    
    df_out = pd.DataFrame(columns=['trial_num','trial_type', 
                                   'attend_condition', 'bars', 
                                   'hemifield', 'crossing_ind'])

    counter = 0
    
    for trl in range(len(trial_type)):
        
        if trial_type[trl] == 'cue_%i'%(counter+1):
            counter += 1
        
        df_out = df_out.append({'trial_num': trl, 
                                'trial_type': trial_type[trl],
                                'attend_condition': attend_block_conditions[counter],
                                'bars': bar_dict['mini_block_%i'%counter].keys(),
                                'hemifield': hemifield[trl],
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
        
        # make mask to get background positions
        mask = ~np.array([item in output_dict['bar0']['xys'] for item in grid_pos])

        output_dict['background'] = {'xys': grid_pos,#[mask], 
                                     'nElements': grid_pos.shape[0]}
        
    return(output_dict)

