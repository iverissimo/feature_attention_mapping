

# import relevant packages
import sys
import os
#import appnope
from session import PRFSession, FeatureSession, FlickerSession, PylinkEyetrackerSession


# define main function
def main():
    
    # take user input
    
    # define participant number and open json parameter file
    if len(sys.argv) < 2:
        raise NameError('Please add subject number (ex:1) '
                        'as 1st argument in the command line!')

    elif len(sys.argv) < 3:
        raise NameError('Please add run number (ex:1) '
                        'as 2nd argument in the command line!')
    
    sj_num = str(sys.argv[1]).zfill(2) # subject number
    run_num = str(sys.argv[2]).zfill(2) # run number
    
    print('Running experiment for subject-%s, run-%s'%(sj_num,run_num))

    mac_bool = '' # check if running on mac or not, workaround retina display issue
    while mac_bool not in ('y','yes','n','no'):
        mac_bool = input('Running experiment on macbook (y/n)?: ')

    mac_bool = True if (mac_bool=='y' or mac_bool=='yes') else False
    if mac_bool:
        print('Running experiment on macbook, defining display accordingly')


    exp_type = ''
    while exp_type not in ('standard','feature','flicker'):
        exp_type = input('Standard pRF mapping or Feature mapping (standard/feature/flicker)?: ')

    print('Running %s pRF mapping for subject-%s, run-%s'%(exp_type,sj_num,run_num))


    #appnope.nope() # disable power saving feature of Mac
    
    # make output dir
    base_dir = os.path.split(os.getcwd())[0] # main path for all folders of project
    output_dir = os.path.join(base_dir,'output','PRF'+exp_type,'data','sub-{sj}'.format(sj=sj_num))

    # if output path doesn't exist, create it
    if not os.path.isdir(output_dir): 
        os.makedirs(output_dir)
    print('saving files in %s'%output_dir)

    # string for output data
    output_str = 'sub-{sj}_ses-01_task-PRF{task}_run-{run}'.format(sj=sj_num,run=run_num,task=exp_type)

    # load approriate class object to be run
    if exp_type == 'standard': # run standard pRF mapper

        bckg_contrast = '' # define if run starts with or without background
        while bckg_contrast not in ('y','yes','n','no'):
            bckg_contrast = input('Start pRF run with or without background (y/n)?: ')

        bckg_contrast = True if (bckg_contrast=='y' or bckg_contrast=='yes') else False
        if bckg_contrast:
            print('Running Standard pRF mapping')

        exp_sess = PRFSession(output_str = output_str,
                              output_dir = output_dir,
                              settings_file = 'experiment_settings.yml',
                              macbook_bool = mac_bool,
                              background = bckg_contrast,
                              eyetracker_on = False)

    elif exp_type == 'feature': # run feature pRF mapper
         exp_sess = FeatureSession(output_str = output_str,
                                  output_dir = output_dir,
                                  settings_file = 'experiment_settings.yml',
                                  macbook_bool = mac_bool,
                                  eyetracker_on = False)

    elif exp_type == 'flicker': # run feature pRF mapper
         exp_sess = FlickerSession(output_str = output_str,
                                  output_dir = output_dir,
                                  settings_file = 'experiment_settings.yml',
                                  macbook_bool = mac_bool,
                                  eyetracker_on = False)
   	                            
    exp_sess.run()


if __name__ == '__main__':
    main()



