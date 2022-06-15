import os, sys
import os.path as op
import subprocess as sb

import yaml

# load settings from yaml
with open(op.join(op.split(op.split(os.getcwd())[0])[0],'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)

# define participant number
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex: 001) '
                    'as 1st argument in the command line!')
elif len(sys.argv)<3:
    raise NameError('Please specify where running data (local vs lisa)'
                    'as 2nd argument in the command line!')
elif len(sys.argv)<4:
    raise NameError('Please specify if viewing or making movie (view, movie)'
                    'as 3rd argument in the command line!') 

else:
    sj = str(sys.argv[1]).zfill(3) #fill subject number with 00 in case user forgets
    base_dir = str(sys.argv[2]) # which machine we run the data
    check_type = str(sys.argv[3]) # which freesurfer command

# path to store freesurfer outputs, in derivatives
freesurfer_datadir = op.join(params['mri']['paths'][base_dir]['root'],'derivatives','freesurfer')
print('freesurfer files in %s'%freesurfer_datadir)

if check_type == 'view':

    batch_string = """#!/bin/bash

    conda activate i36

    export SUBJECTS_DIR=$DATADIR

    cd $DATADIR
    # sub-$SJ_NR/mri/T2.mgz \

    freeview -v \
        sub-$SJ_NR/mri/T1.mgz \
        sub-$SJ_NR/mri/wm.mgz \
        sub-$SJ_NR/mri/brainmask.mgz \
        sub-$SJ_NR/mri/aseg.mgz:colormap=lut:opacity=0.2 \
        -f \
        sub-$SJ_NR/surf/lh.white:edgecolor=blue \
        sub-$SJ_NR/surf/lh.pial:edgecolor=red \
        sub-$SJ_NR/surf/rh.white:edgecolor=blue \
        sub-$SJ_NR/surf/rh.pial:edgecolor=red

    """

    working_string = batch_string.replace('$SJ_NR', sj)
    working_string = working_string.replace('$DATADIR', freesurfer_datadir)

    os.system(working_string)

elif check_type == 'movie':

    # output for images and movie produced
    out_dir = op.join(params['mri']['paths'][base_dir]['root'],'derivatives','check_segmentations','sub-{sj}'.format(sj=sj))

    if not op.exists(out_dir):
        os.makedirs(out_dir)

    batch_string = """#!/bin/bash

    conda activate i36

    export SUBJECTS_DIR=$DATADIR

    cd $DATADIR

    freeview -v \
        sub-$SJ_NR/mri/T1.mgz:grayscale=10,100 \
        -f \
        sub-$SJ_NR/surf/lh.white:edgecolor=blue \
        sub-$SJ_NR/surf/lh.pial:edgecolor=red \
        sub-$SJ_NR/surf/rh.white:edgecolor=blue \
        sub-$SJ_NR/surf/rh.pial:edgecolor=red \
        -viewport sagittal \
        -slice {$XPOS} 127 127 \
        -ss {$OPFN}

    """

    working_string = batch_string.replace('$SJ_NR', sj)
    working_string = working_string.replace('$DATADIR', freesurfer_datadir)

    # number of slices for saggital view
    sag_slices = range(77,268) #248)

    for slice in sag_slices:
        if not op.exists(op.join(out_dir, str(slice).zfill(3) + '.png')): # if image already in dir, skip
            plot_slice = working_string.replace('$XPOS', str(slice).zfill(3))
            plot_slice = plot_slice.replace('$OPFN', op.join(out_dir, str(slice).zfill(3) + '.png'))

            os.system(plot_slice)

    subject = 'sub-{sj}'.format(sj=sj)
    convert_command = f'ffmpeg -framerate 5 -pattern_type glob -i "{out_dir}/*.png" -b:v 2M -c:v mpeg4 {out_dir}/{subject}.mp4'
    sb.call(convert_command, shell=True)