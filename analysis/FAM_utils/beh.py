
# useful functions to use in other scripts

import os
import numpy as np
import colorsys


def get_true_response(arr):

    """ (for feature task)
    get valid ("true") responses for miniblock,
    this is if attended bar in same or different hemifield than prev trial
    
    Parameters
    ----------
    arr : array/list
        hemifield of attended bar
        
    """
    
    out_arr = np.array(['same' if arr[i]==arr[i+1] else 'different' for i in range(len(arr)-1)])
    
    return out_arr



def get_pp_responses(trial_IDs,events,params):
    
    """ (for feature task)
    get participant responses for miniblock, from events df.
    
    return array of responses and onsets
    
    Parameters
    ----------
    trial_IDs : array/list
        list of trial number for that miniblock/run
    events: pandas DF
        dataframe with events (onset, type, etc)
        
    """
    
    responses = []
    responses_onset = []
    
    for i,trl in enumerate(trial_IDs): 
        
        if i!=0: # we ignore first trial because they're not supposed to reply (same/dif task)
            
            # filter df for responses for trial and trial+1(both before new trial is shown)
            response_df = events.loc[(events['event_type']=='response')&(events['trial_nr'].isin([trl,trl+1]))&((events['response']!='t'))] 
            
            if response_df.empty: # if no response given
                responses.append(np.nan)
                responses_onset.append(np.nan)
                
            else:
                if response_df['response'].values[0] in params['keys']['right_index']: # right index finger, different side keys
                    
                    responses.append('different')
                
                elif response_df['response'].values[0] in params['keys']['left_index']: # left index finger, same side keys
                    
                    responses.append('same')
                
                responses_onset.append(response_df['onset'].values[0])
    
    return responses,responses_onset



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