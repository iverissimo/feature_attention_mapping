
import numpy as np
import os, sys
import os.path as op
import pandas as pd
import yaml
import glob
import json

from shutil import copy2, copyfile
import subprocess

import nibabel as nib

from nilearn.plotting import plot_anat
import matplotlib.pyplot as plt
import seaborn as sns

from FAM.utils import mri as mri_utils


class PreprocMRI:

    def __init__(self, MRIObj):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
            
        """

        # set data object to use later on
        self.MRIObj = MRIObj


    def BiasFieldCorrec(self, participant, file_type='T2w', input_pth = None):
        
        """
        Run bias field correction to structural files

        Parameters
        ----------
        participant : str
            participant number
        file_type : str
            file type to correct ('T2w' or 'T1w')
        input_pth: str
            path to look for files, if None then will get them from sourcedata/.../anat folder

        """ 
        
        if input_pth is None:
            input_pth = glob.glob(op.join(self.MRIObj.sourcedata_pth, 'sub-{sj}'.format(sj=participant), 'ses-*', 'anat'))[0]
        
        # list original (uncorrected) files (can be T1 or T2)
        orig_files = [op.join(input_pth,run) for run in os.listdir(input_pth) if run.endswith('{file}.nii.gz'.format(file=file_type))]

        # make output folder to store copy of original, tmp and output files
        out_pth = self.MRIObj.bfc_pth
        
        if not op.exists(out_pth):
            os.makedirs(out_pth)
        print('saving files in %s'%out_pth)
        
        # loop over files we want to bias field correct
        for orig in orig_files:  

            # check if outfolder exists
            outfolder = op.join(out_pth, op.split(orig)[-1].replace('.nii.gz',''))

            # if exists and bias field corrected file there, skip
            if op.exists(outfolder) and op.exists(op.join(outfolder,'bico_'+op.split(orig)[-1])):

                print('BIAS FIELD CORRECTION ALREDY PERFORMED ON %s,\nSKIPPING'%orig)

            else:
                # proceed
                if not op.exists(outfolder):
                    os.makedirs(outfolder)
                print('saving files in %s'%outfolder)
                
                batch_string = """#!/bin/bash
            
                echo "moving $ORIG to new folder"
                mv $ORIG $OUTFOLDER # move original file to tmp folder

                cd $OUTFOLDER # go to the folder

                pigz -d $INPUT.gz # unzip the .nii.gz file

                cp $INPUT uncorr.nii

                echo "running SPM"
                cp $REPO/Bias_field_script_job.m ./Bias_field_script_job.m # copy matlab script to here

                $MATLAB -nodesktop -nosplash -r "Bias_field_script_job" # execute the SPM script in matlab

                mv muncorr.nii bico_$INPUT # rename output file

                rm uncorr.nii

                pigz bico_$INPUT

                echo "moving corrected $INPUT to original folder"
                cp bico_$INPUT.gz new_$INPUT.gz
                mv new_$INPUT.gz $ORIG # move to sourcedata again

                echo SUCCESS

                """

                batch_dir = op.join(self.MRIObj.proj_root_pth,'batch')
                if not op.exists(batch_dir):
                        os.makedirs(batch_dir)
                        
                keys2replace = {'$SJ_NR': str(participant).zfill(3),
                        '$ORIG': orig, 
                        '$OUTFOLDER': outfolder, 
                        '$INPUT': op.split(orig)[-1].replace('.nii.gz','.nii'),
                        '$REPO': self.MRIObj.repo_pth,
                        '$MATLAB': self.MRIObj.matlab_pth,
                        '$ROOTFOLDER': self.MRIObj.proj_root_pth 
                         }

                # replace all key-value pairs in batch string
                for key, value in keys2replace.items():
                    batch_string = batch_string.replace(key, value)

                # run it
                js_name = op.join(batch_dir, 'BFC-' + op.split(orig)[-1].replace('.nii.gz','.nii') + '.sh')
                of = open(js_name, 'w')
                of.write(batch_string)
                of.close()

                os.system('sh ' + js_name) 
                
                # make image with before vs after, for comparison
                # and save in dir
                uncorr_file = op.join(outfolder,op.split(orig)[-1])[:-3] # remove gz
                corr_file = op.join(outfolder,'bico_'+op.split(orig)[-1])

                vmax = 50000 if 'T1w' in corr_file else 200000

                # create a figure with multiple axes to plot each anatomical image
                fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

                plot_anat(uncorr_file,vmax=vmax, title='uncorrected', axes=axes[0],cut_coords=(0,-4,-2))
                plot_anat(corr_file,vmax=vmax, title='corrected', axes=axes[1],cut_coords=(0,-4,-2))

                # save the output figure with all the anatomical images
                fig.savefig("%s_BFC.png"%op.join(outfolder,op.split(orig)[-1].split('.')[0]),dpi=100)
                
        
    def check_anatpreproc(self, T2file = True):
        
        """
        Check if we ran preprocessing for anatomical data
        
        Parameters
        ----------
        T2file : bool
            if participant has T2w file (or if we want to check for it anyway)
        """ 
        
        # loop over participants
        for pp in self.MRIObj.sj_num:

            # path for sourcedata anat files of that participant
            anat_pth = glob.glob(op.join(self.MRIObj.sourcedata_pth, 'sub-{sj}'.format(sj=pp), 'ses-*', 'anat'))[0]
            

            #### first check if anat files are in correct orientation, ######
            # if not we need to reorient 
            # to avoid running into issues with linescanning pipeline
            anat_preproc_sub = glob.glob(op.join(self.MRIObj.anat_preproc_pth,'sub-{sj}'.format(sj=pp), 'ses-*', 'anat'))[0]
            anat_preproc_files = [op.join(anat_preproc_sub,val) for val in os.listdir(anat_preproc_sub) if val.endswith('.nii.gz')]
            
            # check first file in list, should be representative of rest
            ori_val = subprocess.check_output('fslval {file} qform_xorient'.format(file=anat_preproc_files[0]), shell=True)
            ori_val = str(ori_val.decode('ascii').strip())

            if ori_val != 'Left-to-Right':
                print('anat file not RAS, reorienting')
                self.reorient_nii_2RAS(participant=pp, input_pth = anat_preproc_sub)

            
            ##### if we collected T2w files for participant #######
            if T2file:
                
                # check for T2w files in sourcedata
                T2w_files = [op.join(anat_pth, val) for val in os.listdir(anat_pth) if val.endswith('_T2w.nii.gz')]

                if len(T2w_files) == 0:
                    raise NameError('No T2w files in {folder}!'.format(folder=anat_pth))
                else:
                    # check if we bias field correct T2 files
                    if self.MRIObj.params['mri']['preproc']['BFC_T2']:
                        bfc_sj = [op.join(self.MRIObj.bfc_pth, val) for val in os.listdir(self.MRIObj.bfc_pth) if val.startswith('sub-{sj}'.format(sj=pp)) and \
                                                                                                                  val.endswith('_T2w')]
                        # if no BFC folder or folder empty
                        if (len(bfc_sj) == 0) or (len(os.listdir(bfc_sj[0])) == 0):
                            
                            # run BFC
                            print('Running BFC for participant {pp}'.format(pp = pp))
                            self.BiasFieldCorrec(participant=pp)
                    
                    print('Participant T2w files already processed')
                    
                    
            ##### check if masked T1w file in anat folder #####
            masked_T1_files = [op.join(anat_pth, val) for val in os.listdir(anat_pth) if val.endswith('_desc-masked_T1w.nii.gz')]
            
            if len(masked_T1_files) == 0:
                print('No  masked T1w files in {folder}!'.format(folder=anat_pth))
                
                # check if files in anat_preprocessing folder and we forgot to copy
                masked_T1_pth = op.join(self.MRIObj.anat_preproc_pth, 'derivatives', 'masked_mp2rage', 
                                       'sub-{sj}'.format(sj=pp), 'ses-1', 'anat')
                
                if not op.exists(masked_T1_pth):
                    os.makedirs(masked_T1_pth)
                print('saving files in %s'%masked_T1_pth)
                
                masked_T1_files = [op.join(masked_T1_pth, val) for val in os.listdir(masked_T1_pth) if val.endswith('_desc-masked_T1w.nii.gz')]
                
                # if they exist, copy them
                if len(masked_T1_files) > 0:
                    for f in masked_T1_files:
                        os.system('cp {mask_file} {sourcedata_file}'.format(mask_file = f,
                                                                           sourcedata_file = op.join(anat_pth, op.split(f)[-1])))
                
                # if not, then ask user to run linescanning pipeline in new terminal       
                else:
                    print('Linescanning anatomical processing pipeline not run!! open new terminal and do following:')
                    pipeline_txt = """\nconda activate fam_anat\nbash\nsource linescanning/shell/spinoza_setup\n
                    
                    master -s {sub} -m 04
                    master -s {sub} -m 05b
                    master -s {sub} -m 07
                    master -s {sub} -m 08
                    master -s {sub} -m 09
                    master -s {sub} -m 11
                    master -s {sub} -m 12
                    master -s {sub} -m 13
                    
                    """
                    print(pipeline_txt.format(sub=pp))
                    
            else:
                print('Participant T1w files already processed')
                
                
                
    def call_freesurfer(self, cmd='all', wf_dir = '/scratch/FAM_wf', batch_dir ='/home/inesv/batch'):
        
        """
        Run FREESURFER 7.2 on T1w and T2w (if available)
        (to be done before fmriprep call)
        
        NOTE - needs to be run in slurm system!!
        
        Parameters
        ----------
        cmd : str
            freesurfer command ('all', 'pial', 't2')
        wf_dir : str
            workflow directory (only releavnt for slurm)
        batch_dir: str
            path to store .sh jobs, for later check of what was actually ran
        """ 
        
        if self.MRIObj.base_dir == 'local':
            raise NameError('Dont run freesurfer locally - only implemented in slurm systems')
        
        # loop over participants
        for pp in self.MRIObj.sj_num:
            
            ## freesurfer command ##
            freesurfer_cmd = 'recon-all -s {sj} -hires '.format(sj=pp)
            
            # path to store freesurfer outputs, in derivatives
            out_dir = op.join(self.MRIObj.freesurfer_pth, 'sub-{sj}'.format(sj=pp))
            print('saving files in %s'%out_dir)

            if not op.exists(out_dir):
                os.makedirs(out_dir)
            elif len(os.listdir(out_dir)) > 0 and freesurfer_cmd == 'all':
                overwrite = ''
                while overwrite not in ('y','yes','n','no'):
                    overwrite = input('dir already has files, continue with recon-all\n(y/yes/n/no)?: ')
                if overwrite in ['no','n']:
                    raise NameError('directory already has files\nstopping analysis!')
                    
            # path for sourcedata anat files of that participant
            anat_pth = glob.glob(op.join(self.MRIObj.sourcedata_pth, 'sub-{sj}'.format(sj=pp), 'ses-*', 'anat'))[0]
            
            # T1 and T2 filenames
            t1_filename = [op.join(anat_pth,run) for run in os.listdir(anat_pth) if run.endswith('masked_T1w.nii.gz')]
            t2_filename = [op.join(anat_pth,run) for run in os.listdir(anat_pth) if run.endswith('_T2w.nii.gz')]

            
            if (len(t1_filename) == 0) and (len(t2_filename) == 0):
                raise NameError('No source files present!! Check whats up')
            
            ### EDIT COMMAND STRING ####
            
            if cmd == 'pial':
                print("running pial fixes")
                freesurfer_cmd += '-autorecon-pial '
                
            elif cmd == 't2':
                print("running pial fixes taking into account T2 or FLAIR images")
                freesurfer_cmd += '-T2 {T2_file} -T2pial -autorecon3 '.format(T2_file = t2_filename[0].replace(anat_pth, op.join(wf_dir, 'anat')))
            
            elif cmd == 'all':
                print("running full pipeline (recon all)")
                
                # loop over t1 if we have several, will be averaged
                for t1 in t1_filename:
                    freesurfer_cmd += '-i {T1_file} '.format(T1_file = t1.replace(anat_pth, op.join(wf_dir, 'anat')))
                # add t2, if exists
                if len(t2_filename) > 0:
                    freesurfer_cmd += '-T2 {T2_file} -T2pial '.format(T2_file = t2_filename[0].replace(anat_pth, op.join(wf_dir, 'anat')))
            
                freesurfer_cmd += '-all '
            
            
            # set batch string
            batch_string = """#!/bin/bash
#SBATCH -t 96:00:00
#SBATCH -N 1 --mem=65536
#SBATCH --cpus-per-task=16
#SBATCH -v
#SBATCH --output=$BD/slurm_FREESURFER_%A.out

# call the programs
echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

conda activate i38

# make working directory in node
mkdir $WF_DIR
cp -r $ANATDIR $WF_DIR

wait
cp -r $OUTDIR/$SJ_NR $WF_DIR

wait
export SUBJECTS_DIR=$WF_DIR

wait
cd $SUBJECTS_DIR

wait
if [ "$CMD" == all ]; then
    # remove sub folder (empty anyway) to avoid freesurfer complaints
    rm -r $WF_DIR/$SJ_NR 
fi

wait
$FS_CMD

wait
rsync -chavzP $WF_DIR/$SJ_NR/ $OUTDIR/$SJ_NR

wait          # wait until programs are finished

echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"
"""
         
            #os.chdir(batch_dir)

            keys2replace = {'$SJ_NR': 'sub-{sj}'.format(sj=pp),
                            '$ANATDIR': anat_pth,
                            '$OUTDIR': op.split(out_dir)[0], 
                            '$FS_CMD': freesurfer_cmd,
                            '$CMD': cmd,
                            '$WF_DIR': wf_dir,
                            '$BD': batch_dir
                             }

            # replace all key-value pairs in batch string
            for key, value in keys2replace.items():
                batch_string = batch_string.replace(key, value)

            print(batch_string)
            
            # run it
            js_name = op.join(batch_dir, 'FREESURFER7_sub-{sj}_FAM_prefmriprep.sh'.format(sj=pp))
            of = open(js_name, 'w')
            of.write(batch_string)
            of.close()

            print('submitting ' + js_name + ' to queue')
            print(batch_string)
            os.system('sbatch ' + js_name)

            
    def call_fmriprep(self, data_type='anat', wf_dir = '/scratch/FAM_wf', batch_dir ='/home/inesv/batch',
                     partition_name = None, node_name = None, use_fmap = True):
        
        """
        Run FMRIPREP on anat or functional data
        
        NOTE - needs to be run in slurm system!!
        
        Parameters
        ----------
        data_type : str
            if we want to run it on 'anat' or 'func'
        wf_dir : str
            workflow directory (only releavnt for slurm)
        batch_dir: str
            path to store .sh jobs, for later check of what was actually ran
        """ 
        
        if self.MRIObj.base_dir == 'local':
            raise NameError('Dont run freesurfer locally - only implemented in slurm systems')
        
        # loop over participants
        for pp in self.MRIObj.sj_num:
            
            fmriprep_cmd = """#!/bin/bash
#SBATCH -t 40:00:00
#SBATCH -N 1 --mem=90G
#SBATCH -v
#SBATCH --output=$BD/slurm_FMRIPREP_%A.out\n"""
            
            if partition_name is not None:
                fmriprep_cmd += '#SBATCH --partition {p}\n'.format(p=partition_name)
            if node_name is not None:
                fmriprep_cmd += '#SBATCH -w {n}\n'.format(n=node_name)
            
            # make fmriprep folder if it does not exist
            #if not op.exists(self.MRIObj.fmriprep_pth):
            #    os.makedirs(self.MRIObj.fmriprep_pth)
                
            if data_type == 'anat':
                fmriprep_cmd +="""\n# call the programs
echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

# make working directory in node
mkdir $WF_DIR

wait

PYTHONPATH="" singularity run --cleanenv -B /project/projects_verissimo -B $WF_DIR \
$SINGIMG \
$ROOTFOLDER/sourcedata $ROOTFOLDER/derivatives/fmriprep/ participant \
--participant-label $SJ_NR --fs-subjects-dir $ROOTFOLDER/derivatives/freesurfer/ \
--output-space T1w \
--nthread 16 --mem_mb 5000 --low-mem --fs-license-file $FREESURFER/license.txt \
--anat-only \
-w $WF_DIR

wait          # wait until programs are finished

echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"
"""
                
            else:
                
                fmriprep_cmd +="""# call the programs
echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

# make working directory in node
mkdir $WF_DIR

wait

PYTHONPATH="" singularity run --cleanenv -B /project/projects_verissimo -B $WF_DIR \
$SINGIMG \
$ROOTFOLDER/sourcedata $ROOTFOLDER/derivatives/fmriprep/ participant \
--participant-label $SJ_NR --fs-subjects-dir $ROOTFOLDER/derivatives/freesurfer/ \
--output-space T1w fsnative fsaverage MNI152NLin2009cAsym --cifti-output 170k \
--bold2t1w-init register --nthread 16 --mem_mb 5000 --low-mem --fs-license-file $FREESURFER/license.txt \
$FMAP_CMD --bold2t1w-dof 6 --stop-on-first-crash --verbose --skip_bids_validation --dummy-scans 5 \
-w $WF_DIR

wait          # wait until programs are finished

echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"
"""

            if use_fmap:
                fmap_cmd = '--use-syn-sdc --force-syn'  
            else:  
                fmap_cmd = '--use-syn-sdc --force-syn --ignore fieldmaps'
            
            #os.chdir(batch_dir)
            batch_string = fmriprep_cmd

            keys2replace = {'$SJ_NR': 'sub-{sj}'.format(sj=pp),
                            '$SINGIMG': op.join(self.MRIObj.params['mri']['paths'][self.MRIObj.base_dir]['singularity'], self.MRIObj.params['mri']['fmriprep_sing']),
                            '$ROOTFOLDER': op.split(self.MRIObj.sourcedata_pth)[0],
                            '$WF_DIR': wf_dir,
                            '$BD': batch_dir,
                            '$FMAP_CMD': fmap_cmd
                             }

            # replace all key-value pairs in batch string
            for key, value in keys2replace.items():
                batch_string = batch_string.replace(key, value)
                
            print(batch_string)
            
            # run it
            js_name = op.join(batch_dir, 'FMRIPREP_sub-{sj}_FAM_{data}.sh'.format(sj=pp, data=data_type))
            of = open(js_name, 'w')
            of.write(batch_string)
            of.close()

            print('submitting ' + js_name + ' to queue')
            #print(batch_string)
            os.system('sbatch ' + js_name)


    def NORDIC(self, participant, input_pth = None, output_pth = None, calc_tsnr=False):
        
        """
        Run NORDIC on functional files
        matlab script from (https://github.com/SteenMoeller/NORDIC_Raw/blob/main/NIFTI_NORDIC.m),
        all credits go to the developers

        NOTE - requires phase (_bold_phase.nii.gz) and mag (_bold.nii.gz) data to be stored in input folder

        Parameters
        ----------
        participant : str
            participant number
        input_pth: str
            path to look for files, if None then will get them from NORDIC/pre_nordic/sub-X/ses-1/ folder

        """ 
        
        ## zero pad participant number, just in case
        participant = str(participant).zfill(3)

        ## in case we want to calculate tSNR
        sub_tsnr = {'pre_nordic': [], 'post_nordic': []}
        
        ## to save shell scripts created
        batch_dir = op.join(self.MRIObj.proj_root_pth,'batch')
        if not op.exists(batch_dir):
            os.makedirs(batch_dir)
        
        ## set input path where standard files are stored
        if input_pth is None:
            input_pth = op.join(self.MRIObj.nordic_pth, 'pre_nordic', 'sub-{sj}'.format(sj=participant), 'ses-1')
            
        # if input path doesnt exist or is empty
        if not op.exists(input_pth) or len(os.listdir(input_pth)) == 0:
            raise NameError('No files found in {pth}'.format(pth=input_pth))
                
        ## output path to copy NORDIC files 
        # (if not set, then will copy to sourcedata folder)
        if output_pth is None:
            output_pth = op.join(self.MRIObj.sourcedata_pth, 'sub-{sj}'.format(sj=participant), 'ses-1', 'func')
            
        # list original (uncorrected) mag files 
        input_mag = [op.join(input_pth,run) for run in os.listdir(input_pth) if run.endswith('_bold.nii.gz') \
            and 'acq-standard' in run and 'phase' not in run]
        
        # if mag files not in sourcedata, 
        # copy them there (we still want to process the non-nordic data)
        for file in input_mag:
            outfile = op.join(output_pth, op.split(file)[-1])

            if op.exists(outfile):
                print('already exists %s'%outfile)
            else:
                copy2(file, outfile)
                print('file copied to %s'%outfile)
        
        # make post_nordic folder (for intermediate files)
        post_nordic = input_pth.replace('pre_nordic', 'post_nordic')
        if not op.exists(post_nordic):
            os.makedirs(post_nordic)
            
        print('saving files in %s'%post_nordic)
        
        # loop over files, make sure using correct phase
        for mag_filename in input_mag:
            
            # phase filename
            phase_filename = mag_filename.replace('_bold.nii.gz', '_bold_phase.nii.gz')
            
            #
            nordic_nii = op.join(post_nordic, op.split(mag_filename)[-1].replace('acq-standard','acq-nordic'))
            
            # if file aready exists, skip
            if op.exists(nordic_nii):
                print('NORDIC ALREADY PERFORMED ON %s,\nSKIPPING'%nordic_nii)
                
            else:
                batch_string = """#!/bin/bash
                
echo "applying nordic to $INMAG"
cd $FILEPATH # go to the folder
cp $REPO/NIFTI_NORDIC.m ./NIFTI_NORDIC.m # copy matlab script to here

$MATLAB -nodesktop -nosplash -r "NIFTI_NORDIC('$INMAG', '$INPHASE', '$OUTFILE'); quit;" # execute the NORDIC script in matlab

wait 
pigz $OUTFILE.nii # compress file

wait
mv $OUTFILE.nii.gz $OUTPATH # move to post nordic folder

"""
                keys2replace = {'$FILEPATH': self.MRIObj.nordic_pth,
                            '$REPO': self.MRIObj.repo_pth,
                            '$INMAG': mag_filename,
                            '$INPHASE': phase_filename, 
                            '$OUTFILE': op.split(nordic_nii)[-1].replace('.nii.gz',''),
                            '$OUTPATH': nordic_nii,
                            '$MATLAB': self.MRIObj.matlab_pth
                            }
                    
                # replace all key-value pairs in batch string
                for key, value in keys2replace.items():
                    batch_string = batch_string.replace(key, value)
                    
                # run it
                js_name = op.join(batch_dir, 'NORDIC-' + op.split(nordic_nii)[-1].replace('.nii.gz','.sh'))
                of = open(js_name, 'w')
                of.write(batch_string)
                of.close()

                print(batch_string)
                os.system('sh ' + js_name) 
                
                # copy file to sourcedata
                copy2(nordic_nii, op.join(output_pth, op.split(nordic_nii)[-1]))
                print('file copied to %s'%op.join(output_pth, op.split(nordic_nii)[-1]))

            if calc_tsnr:
                # calculate tSNR before nordic
                sub_tsnr['pre_nordic'].append(mri_utils.get_tsnr(mag_filename, return_mean = True))

                # calculate tSNR after nordic
                sub_tsnr['post_nordic'].append(mri_utils.get_tsnr(nordic_nii, return_mean = True))

        # make plot of tSNR for comparison
        if calc_tsnr:
            # create a figure with multiple axes to plot each bar
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))

            sns.barplot(data = pd.DataFrame(sub_tsnr['pre_nordic']), ax=ax1)
            ax1.set_ylim(0,30)
            ax1.set_xticks([])
            ax1.set_ylabel('Mean tSNR', fontsize=20)
            ax1.set_xlabel('pre-NORDIC', fontsize=20)

            sns.barplot(data = pd.DataFrame(sub_tsnr['post_nordic']), ax=ax2)
            ax2.set_ylim(0,30)
            ax2.set_xticks([])
            ax2.set_xlabel('post-NORDIC', fontsize=20)
            # save the output figure with all the anatomical images
            fig.savefig(op.join(op.split(nordic_nii)[0], 'mean_tSNR_NORDIC.png'))


    def check_funcpreproc(self):
            
        """
        Check if we ran preprocessing for functional data

        """ 

        # loop over participants
        for pp in self.MRIObj.sj_num:
            
            # and over sessions (if more than one)
            for ses in self.MRIObj.session['sub-{sj}'.format(sj=pp)]:

                # path for sourcedata func files of that participant
                func_pth = op.join(self.MRIObj.sourcedata_pth, 'sub-{sj}'.format(sj=pp), ses, 'func')
                # path for sourcedata fmap files of that participant
                fmap_pth = op.join(self.MRIObj.sourcedata_pth, 'sub-{sj}'.format(sj=pp), ses, 'fmap')
                
                ## Run NORDIC on func files ##
                #set pre-NORDIC dir 
                sub_prenordic = op.join(self.MRIObj.nordic_pth, 'pre_nordic', 'sub-{sj}'.format(sj=pp), ses)
                
                # actually run it
                print('Running NORDIC for participant {pp}, session-{ses}'.format(pp = pp, ses = ses))
                self.NORDIC(participant = pp, input_pth = sub_prenordic, output_pth = func_pth, calc_tsnr=True)

                print('updating jason files')
                self.update_jsons(participant = pp, input_pth = func_pth, json_folder = 'func',
                                    parrec_pth = op.join(self.MRIObj.proj_root_pth, 'raw_data', 'parrec', 
                                    'sub-{sj}'.format(sj = pp), ses))
                
                ## check fmaps ##
                # to see if we cropped initial dummy scans
                print('Cropping fieldmaps for participant {pp}, session-{ses}'.format(pp = pp, ses = ses))
                self.crop_fieldmaps(participant = pp, input_pth = fmap_pth, dummys = self.MRIObj.params['mri']['dummy_TR'])

                ## update fieldmap params (specifically effective echo spacing)
                print('updating jason files')
                self.update_jsons(participant = pp, input_pth = fmap_pth, json_folder = 'fmap',
                                    parrec_pth = op.join(self.MRIObj.proj_root_pth, 'raw_data', 'parrec', 
                                    'sub-{sj}'.format(sj = pp), ses))

    
    def update_jsons(self, participant, input_pth = None, parrec_pth = None, json_folder = 'fmap'):

        """
        Update json params for a given file type
        given raw PAR/REC header info
        (default is fmap epis, to added effective echospacing, but want to generalize to all later on)

        """ 
                
        ## zero pad participant number, just in case
        participant = str(participant).zfill(3)

        ## set input path where sourcedata json files are
        if input_pth is None:
            input_pth = op.join(self.MRIObj.sourcedata_pth, 'sub-{sj}'.format(sj=participant), 'ses-1', json_folder)

        ## set parrec path where raw data PAR/REC files are
        if parrec_pth is None:
            parrec_pth = op.join(self.MRIObj.proj_root_pth, 'raw_data', 'parrec', 'sub-{sj}'.format(sj=participant), 'ses-1')


        ## for fieldmap data json
        if json_folder == 'fmap':

            ## get json file list we want to update
            json_files = [op.join(input_pth, val) for val in os.listdir(input_pth) if 'dir-PA' in val \
                            and val.endswith('_epi.json')]; json_files.sort()

            ## get PAR/REC file list 
            parrec_files = [op.join(parrec_pth, val) for val in os.listdir(parrec_pth) if 'TOPUP' in val \
                            and val.endswith('.PAR')]

            ## loop over runs
            for r in range(self.MRIObj.params['mri']['nr_runs']):
                
                # check if run file exists
                jfile = [val for val in json_files if 'run-%i'%(r+1) in val or 'run-0%i'%(r+1) in val]

                if len(jfile)>0:
                    ## load jason file
                    jfile = jfile[0]
                    with open(jfile) as f:
                        json_data = json.load(f)
                        
                    # check if we have respective parrec
                    parfile = [val for val in parrec_files if 'run-%i'%(r+1) in val or 'run-0%i'%(r+1) in val]

                    if len(parfile)>0:
                        par_data = nib.parrec.load(parfile[0])
                    
                        ## get waterfat shift value and calculate other necessart params for fmriprep ##
                        
                        #BIDS wants TRT to be specified for epi files. long story short, we MUST to put that AND EES in
                        WFS = par_data.header.get_water_fat_shift()
                        #magnetic field strength * water fat difference in ppm * gyromagnetic hydrogen ratio
                        WFS_hz = 7 * 3.35 * 42.576
                        TRT = WFS/WFS_hz
                        epi_factor = par_data.header.general_info['epi_factor']
                        #trt/(epi factor +1)
                        EES = TRT / (epi_factor+1)
                        
                        ## update params
                        json_data['WaterFatShift'] = WFS
                        json_data['EffectiveEchoSpacing'] = EES
                        json_data['TotalReadoutTime'] = TRT
                        json_data['EPIFactor'] = epi_factor
                        json_data['SliceTiming'] = list(np.tile(np.linspace(0, json_data['RepetitionTime'],
                                                        int(par_data.header.general_info['max_slices']/json_data['MultiBandAccelerationFactor']),endpoint=False),
                                                            json_data['MultiBandAccelerationFactor']))
                        
                        ## and save 
                        with open(jfile, 'w') as f:
                            json.dump(json_data, f, indent=4)
                        
                    else:
                        print('No parrec file for topup run-%i, not updating params'%(r+1))

                else:
                    print('No json file for topup run-%i'%(r+1))
                

        ## for functional data json
        elif json_folder == 'func':

            ## loop over tasks
            for tsk in self.MRIObj.params['general']['tasks']:

                ## get json file list we want to update
                json_files = [op.join(input_pth, val) for val in os.listdir(input_pth) if 'task-{tsk}'.format(tsk=tsk) in val \
                                and val.endswith('_bold.json')]; json_files.sort()
            
                ## get PAR/REC file list 
                parrec_files = [op.join(parrec_pth, val) for val in os.listdir(parrec_pth) if 'task-{tsk}'.format(tsk=tsk) in val \
                                and val.endswith('.PAR')]

                ## loop over runs
                for r in range(self.MRIObj.params['mri']['nr_runs']):
                    
                    # check if run file exists
                    jfile = [val for val in json_files if 'run-%i'%(r+1) in val or 'run-0%i'%(r+1) in val]

                    if len(jfile)>0:

                        for runj in jfile: ## because there are usually 2 files for a same run (nordic and not nordic)
                            ## load jason file
                            with open(runj) as f:
                                json_data = json.load(f)
                                
                            # check if we have respective parrec
                            parfile = [val for val in parrec_files if 'run-%i'%(r+1) in val or 'run-0%i'%(r+1) in val]

                            if len(parfile)>0:
                                par_data = nib.parrec.load(parfile[0])
                            
                                ## get waterfat shift value and calculate other necessart params for fmriprep ##
                                
                                #BIDS wants TRT to be specified for epi files. long story short, we MUST to put that AND EES in
                                WFS = par_data.header.get_water_fat_shift()
                                #magnetic field strength * water fat difference in ppm * gyromagnetic hydrogen ratio
                                WFS_hz = 7 * 3.35 * 42.576
                                TRT = WFS/WFS_hz
                                epi_factor = par_data.header.general_info['epi_factor']
                                #trt/(epi factor +1)
                                EES = TRT / (epi_factor+1)
                                
                                ## update params
                                json_data['WaterFatShift'] = WFS
                                json_data['EffectiveEchoSpacing'] = EES
                                json_data['TotalReadoutTime'] = TRT
                                json_data['EPIFactor'] = epi_factor
                                json_data['SliceTiming'] = list(np.tile(np.linspace(0, json_data['RepetitionTime'],
                                                                int(par_data.header.general_info['max_slices']/json_data['MultiBandAccelerationFactor']),endpoint=False),
                                                                    json_data['MultiBandAccelerationFactor']))
                                
                                ## and save 
                                with open(runj, 'w') as f:
                                    json.dump(json_data, f, indent=4)
                                
                            else:
                                print('No parrec file for func run-%i, not updating params'%(r+1))

                    else:
                        print('No json file for task %s func run-%i'%(tsk,(r+1)))


    def crop_fieldmaps(self, participant, dummys = 5, input_pth = None, output_pth = None):

        """
        Crop fieldmaps to remove dummy TRs

        Parameters
        ----------
        participant : str
            participant number
        dummys: int
            number of dummy TRs used (and thus that we want to remove)
        input_pth: str
            path to look for files, if None then will get them from sourcedata/sub-X/ses-1/fmap folder
        output_pth: str
            path to save original files, if None then will save them in root/orig_fmap/sub-X folder

        """

        ## zero pad participant number, just in case
        participant = str(participant).zfill(3)

        ## set input path where fmaps are
        if input_pth is None:
            input_pth = op.join(self.MRIObj.sourcedata_pth, 'sub-{sj}'.format(sj=participant), 'ses-1', 'fmap')

        # list of original niftis
        orig_nii_files = [op.join(input_pth, val) for val in os.listdir(input_pth) if val.endswith('_epi.nii.gz')]

        ## set output path where we want to store original (uncropped) fmaps
        if output_pth is None:
            output_pth = op.join(self.MRIObj.proj_root_pth, 'orig_fmaps' , 'sub-{sj}'.format(sj=participant))

        if not op.isdir(output_pth):
            os.makedirs(output_pth)

        # then for each file
        for file in orig_nii_files:

            # first check size of file, to see if it was already cropped
            file_trs = subprocess.check_output('fslnvols {file}'.format(file=file), shell=True)
            file_trs = int(file_trs.decode('ascii'))

            if file_trs > dummys:

                # copy the original to the new folder
                ogfile = op.join(output_pth, op.split(file)[-1])

                if op.exists(ogfile):
                    print('already exists %s'%ogfile)
                else:
                    copy2(file, ogfile)
                    print('file copied to %s'%ogfile)

                # and crop the one in sourcedata
                os.system('fslroi {old} {new} {tr_n} {tr_n2}'.format(old = file, new = file, 
                                                                    tr_n = int(file_trs-dummys), 
                                                                    tr_n2 = dummys))
            else:
                print('already cropped {file}, nr TRs is {nt}'.format(file=file, 
                                                                    nt=file_trs))


    def reorient_nii_2RAS(self, participant, input_pth = None, output_pth = None):

        """
        Reorient niftis to RAS
        (useful for anat files after dcm2niix)

        Parameters
        ----------
        participant : str
            participant number
        input_pth: str
            path to look for files, if None then will get them from root/anat_preprocessing/sub-X/ses-1/anat folder
        output_pth: str
            path to save original files, if None then will save them in root/orig_anat/sub-X folder

        """

        ## zero pad participant number, just in case
        participant = str(participant).zfill(3)

        ## set input path where fmaps are
        if input_pth is None:
            input_pth = op.join(self.MRIObj.anat_preproc_pth, 'sub-{sj}'.format(sj=participant), 'ses-1', 'anat')

        # list of original niftis
        orig_nii_files = [op.join(input_pth, val) for val in os.listdir(input_pth) if val.endswith('.nii.gz')]

        ## set output path where we want to store original (uncropped) fmaps
        if output_pth is None:
            output_pth = op.join(self.MRIObj.proj_root_pth, 'orig_anat' , 'sub-{sj}'.format(sj=participant))

        if not op.isdir(output_pth):
            os.makedirs(output_pth)

        # then for each file
        for file in orig_nii_files:

            # copy the original to the new folder
            ogfile = op.join(output_pth, op.split(file)[-1])

            if op.exists(ogfile):
                print('already exists %s'%ogfile)
            else:
                copy2(file, ogfile)
                print('file copied to %s'%ogfile)

            # reorient all files to RAS+ (used by nibabel & fMRIprep) 
            orig_img = nib.load(file)
            orig_img_hdr = orig_img.header

            qform = orig_img_hdr['qform_code'] # set qform code to original

            canonical_img = nib.as_closest_canonical(orig_img)

            if qform != 0:
                canonical_img.header['qform_code'] = np.array([qform], dtype=np.int16)
            else:
                # set to 1 if original qform code = 0
                canonical_img.header['qform_code'] = np.array([1], dtype=np.int16)

            nib.save(canonical_img, file)


    def call_mriqc(self, wf_dir = '/scratch/FAM_wf', batch_dir ='/home/inesv/batch'):
            
        """
        Run MRIQC on anat or functional data

        Parameters
        ----------
        wf_dir : str
            workflow directory (only releavnt for slurm)
        batch_dir: str
            path to store .sh jobs, for later check of what was actually ran
        """ 

        # path to singularity image
        sing_img = op.join(self.MRIObj.params['mri']['paths'][self.MRIObj.base_dir]['singularity'], self.MRIObj.params['mri']['mriqc_sing'])

        # loop over participants
        for pp in self.MRIObj.sj_num:

            # path to store mriqc outputs, in derivatives
            out_dir = op.join(self.MRIObj.derivatives_pth, 'mriqc', 'sub-{sj}'.format(sj=pp))
            if not op.exists(out_dir):
                os.makedirs(out_dir)
            print('saving files in %s'%out_dir)

            # if running in local machine
            if self.MRIObj.base_dir == 'local':
                batch_string = """#!/bin/bash
conda activate i38
wait

docker run -it --rm \
-v $ROOTFOLDER/sourcedata:/data:ro \
-v $ROOTFOLDER/derivatives/mriqc/sub-$SJ_NR:/out \
poldracklab/mriqc:latest /data /out participant --participant_label $SJ_NR
"""
            
            else:
                batch_string = """#!/bin/bash
#SBATCH -t 40:00:00
#SBATCH -N 1 --mem=65536
#SBATCH --cpus-per-task=16
#SBATCH -v
#SBATCH --output=$BD/slurm_MRIQC_%A.out

# call the programs
echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"
conda activate i38

echo "Running MRIQC on participant sub-$SJ_NR"

# make working directory in node
mkdir $WF_DIR

wait
singularity run --cleanenv -B /project/projects_verissimo -B $WF_DIR \
$SINGIMG \
$ROOTFOLDER/sourcedata $ROOTFOLDER/derivatives/mriqc/sub-$SJ_NR \
participant --participant-label $SJ_NR --hmc-fsl --float32 -w $WF_DIR

wait          # wait until programs are finished
echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"

"""

            keys2replace = {'$SJ_NR': pp,
                            '$SINGIMG': sing_img,
                            '$ROOTFOLDER': op.split(self.MRIObj.sourcedata_pth)[0],
                            '$WF_DIR': wf_dir,
                            '$BD': batch_dir
                             }

            # replace all key-value pairs in batch string
            for key, value in keys2replace.items():
                batch_string = batch_string.replace(key, value)
                
            print(batch_string)

            # run it
            js_name = op.join(batch_dir, 'MRIQC_sub-{sj}_FAM.sh'.format(sj=pp))
            of = open(js_name, 'w')
            of.write(batch_string)
            of.close()

            print('submitting ' + js_name + ' to queue')
            os.system('sh ' + js_name) if self.MRIObj.base_dir == 'local' else os.system('sbatch ' + js_name)


    def post_fmriprep_proc(self, output_pth = None, tasks = ['pRF', 'FA'], save_subcortical = True, hemispheres = ['hemi-L','hemi-R']):

        """
        Run final processing steps on functional data (after fmriprep)
        """ 

        ## some relevant parameters
        acq = self.MRIObj.params['mri']['acq'] # if using standard files or nordic files
        space = self.MRIObj.params['mri']['space'] # subject space
        file_ext = self.MRIObj.params['mri']['file_ext'][space] # file extension
        confound_ext = self.MRIObj.params['mri']['confounds']['file_ext'] # file extension

        # loop over participants
        for pp in self.MRIObj.sj_num:
            
            # and over sessions (if more than one)
            for ses in self.MRIObj.session['sub-{sj}'.format(sj=pp)]:
                
                if output_pth is None:
                    output_pth = op.join(self.MRIObj.derivatives_pth, 'post_fmriprep', space, 'sub-{sj}'.format(sj=pp), ses)

                # if output path doesn't exist, create it
                if not op.isdir(output_pth): 
                    os.makedirs(output_pth)
                print('saving files in %s'%output_pth)

                # get list of functional files to process, per task
                fmriprep_pth = op.join(self.MRIObj.derivatives_pth, 'fmriprep', 'sub-{sj}'.format(sj=pp), ses, 'func')

                for tsk in tasks:
                    print('Processign bold files from task-{t}'.format(t=tsk))

                    # bold files
                    bold_files = [op.join(fmriprep_pth,run) for run in os.listdir(fmriprep_pth) if 'space-{sp}'.format(sp=space) in run \
                        and 'acq-{a}'.format(a=acq) in run and 'task-{t}'.format(t=tsk) in run and run.endswith(file_ext)]

                    # confounds
                    confound_files = [op.join(fmriprep_pth,run) for run in os.listdir(fmriprep_pth) if 'acq-{a}'.format(a=acq) in run \
                        and 'task-{t}'.format(t=tsk) in run and run.endswith(confound_ext)]

                    ### load and convert files in numpy arrays, to make format issue obsolete ###
                    # note, if we need headers or affines later on, we will need to get them from fmriprep folder
                    bold_files = mri_utils.load_data_save_npz(bold_files, output_pth, save_subcortical=save_subcortical)

                    ### crop files, due to dummies TRs that were saved ##
                    # and extra ones, if we want to
                    crop_TR = self.MRIObj.params['mri']['dummy_TR'] + self.MRIObj.params[tsk]['crop_TR'] if self.MRIObj.params[tsk]['crop'] == True else self.MRIObj.params['mri']['dummy_TR'] 

                    proc_files = mri_utils.crop_epi(bold_files, output_pth, num_TR_crop = crop_TR)

                    ### filtering ###
                    # if regressing confounds
                    if self.MRIObj.params[tsk]['regress_confounds']: 
    
                        ## first sub select confounds that we are using, and store in output dir
                        confounds_list = mri_utils.select_confounds(confound_files, output_pth, reg_names = self.MRIObj.params['mri']['confounds']['regs'],
                                                                    CumulativeVarianceExplained = self.MRIObj.params['mri']['confounds']['CumulativeVarianceExplained'],
                                                                    select =  'num', num_components = 5, num_TR_crop = crop_TR)
                        
                        ## regress out confounds, 
                        ## and percent signal change
                        proc_files = mri_utils.regressOUT_confounds(proc_files, confounds_list, output_pth, TR = self.MRIObj.params['mri']['TR'], plot_vert = True)

                    else: 
                        ## filter files, to remove drifts ##
                        proc_files = mri_utils.filter_data(proc_files, output_pth, filter_type = self.MRIObj.params['mri']['filtering']['type'], 
                                                first_modes_to_remove = self.MRIObj.params['mri']['filtering']['first_modes_to_remove'], plot_vert = True)
                        
                        ### percent signal change ##
                        proc_files = mri_utils.psc_epi(proc_files, output_pth)


                    ## make new outdir, to save final files that will be used for further analysis
                    # avoids mistakes later on
                    final_output_dir =  op.join(output_pth, 'processed')
                    # if output path doesn't exist, create it
                    if not op.isdir(final_output_dir): 
                        os.makedirs(final_output_dir)
                    print('saving FINAL processed files in %s'%final_output_dir)

                    ## average all runs for pRF task
                    if tsk == 'pRF':
                        
                        if '.func.gii' in file_ext: # combine hemispheres into one array
                            
                            hemi_files = []
                            
                            for hemi in hemispheres:
                                hemi_files.append(mri_utils.average_epi([val for val in proc_files if hemi in val], 
                                                            final_output_dir, method = 'mean'))
                            proc_files = hemi_files
                            
                        else:
                            proc_files = mri_utils.average_epi(proc_files, final_output_dir, method = 'mean')
                        
                    else:
                        # save FA files in final output folder too
                        for f in proc_files:
                            copyfile(f, op.join(final_output_dir,op.split(f)[-1]))





