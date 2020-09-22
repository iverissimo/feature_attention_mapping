

# import relevant packages
import sys
import os
import appnope
from session import PRFSession, FeatureSession


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

    exp_type = ''
    while exp_type not in ('standard','feature'):
        exp_type = input('Standard pRF mapping or Feature mapping (standard/feature)?: ')

    print('Running %s pRF mapping for subject-%s, run-%s'%(exp_type,sj_num,run_num))


    appnope.nope() # disable power saving feature of Mac
    
    # make output dir
    base_dir = os.path.split(os.getcwd())[0] # main path for all folders of project
    output_dir = os.path.join(base_dir,'output','PRF'+exp_type,'sub-{sj}'.format(sj=sj_num))

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
                              settings_file='experiment_settings.yml')

    elif exp_type == 'feature': # run feature pRF mapper
         exp_sess = FeatureSession(output_str = output_str,
                              output_dir = output_dir,
                              settings_file='experiment_settings.yml')
   	                            
    exp_sess.run()


if __name__ == '__main__':
    main()



