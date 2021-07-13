

# import relevant packages
import sys
import os
#import appnope
from session import PRFSession, FeatureSession, FlickerSession, PylinkEyetrackerSession, PracticeFeatureSession


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
    while exp_type not in ('standard','feature','flicker','practice'):
        exp_type = input('Standard pRF mapping or Feature mapping (standard/feature/flicker/practice)?: ')

    if exp_type == 'practice':
        practice_type = ''
        while practice_type not in ('prf','feature'):
            practice_type = input('Practicing pRF or Feature mapping (prf/feature)?: ')
        exp_type = '{exp}_{typ}'.format(exp=exp_type,typ=practice_type)

    print('Running %s pRF mapping for subject-%s, run-%s'%(exp_type,sj_num,run_num))

    
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
        exp_sess = PRFSession(output_str = output_str,
                              output_dir = output_dir,
                              settings_file = 'experiment_settings.yml',
                              macbook_bool = mac_bool,
                              eyetracker_on = False)

    elif exp_type == 'feature': # run feature pRF mapper
        exp_sess = FeatureSession(output_str = output_str,
                                  output_dir = output_dir,
                                  settings_file = 'experiment_settings.yml',
                                  macbook_bool = mac_bool,
                                  eyetracker_on = True)

    elif exp_type == 'flicker': # run flicker pRF mapper
        exp_sess = FlickerSession(output_str = output_str,
                                  output_dir = output_dir,
                                  settings_file = 'experiment_settings.yml',
                                  macbook_bool = mac_bool,
                                  eyetracker_on = False)

    elif exp_type == 'practice_feature': # run practice feature mapper
        exp_sess = PracticeFeatureSession(output_str = output_str,
                                          output_dir = output_dir,
                                          settings_file = 'experiment_settings.yml',
                                          macbook_bool = mac_bool,
                                          eyetracker_on = False)
       	                            
    
    exp_sess.run()


if __name__ == '__main__':
    main()



