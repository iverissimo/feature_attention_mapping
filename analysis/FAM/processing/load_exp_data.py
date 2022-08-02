import numpy as np
import os, sys
import os.path as op
import pandas as pd
import yaml
import glob

from nilearn.plotting import plot_anat
import matplotlib.pyplot as plt


class FAMData:
    
    """FAMData
    Class that loads relevant paths and settings for FAM data
    """
    
    def __init__(self, params, sj_num, exclude_sj = [], base_dir = None):
        
        """__init__
        constructor for class, takes experiment params and subject num as input
        
        Parameters
        ----------
        params : str/yaml dict
            where parameters from experiment and analysis live
        sj_num : str/list/arr
            participant number(s)
        exclude_sj: list/arr
            list with subject numbers to exclude
            
        """
        
        # set params
        
        if isinstance(params, str):
            # load settings from yaml
            with open(params, 'r') as f_in:
                self.params = yaml.safe_load(f_in)
        else:
            self.params = params
            
        
        # excluded participantsa
        self.exclude_sj = exclude_sj
            
        ## set some paths
        # which machine we run the data
        if base_dir is None:
            self.base_dir = self.params['general']['current_dir'] 
        else:
            self.base_dir = base_dir
        
        # project root folder
        self.proj_root_pth = self.params['mri']['paths'][self.base_dir]['root']
        
        # sourcedata dir
        self.sourcedata_pth = op.join(self.proj_root_pth,'sourcedata')
        
        # derivatives dir
        self.derivatives_pth = op.join(self.proj_root_pth,'derivatives')
        
        ## set sj number
        if sj_num in ['group', 'all']: # if we want all participants in sourcedata folder
            sj_num = [op.split(val)[-1].zfill(3)[4:] for val in glob.glob(op.join(self.sourcedata_pth, 'sub-*'))]
            self.sj_num = [val for val in sj_num if val not in self.exclude_sj ]
        
        elif isinstance(sj_num, list) or isinstance(sj_num, np.ndarray): # if we provide list of sj numbers
            self.sj_num = [str(s).zfill(3) for s in sj_num if str(s).zfill(3) not in self.exclude_sj ]
        
        else:
            self.sj_num = [str(sj_num).zfill(3)] # if only one participant, put in list to make life easier later
        
        ## get session number (can be more than one)
        self.session = {}
        for s in self.sj_num:
            self.session['sub-{sj}'.format(sj=s)] = [op.split(val)[-1] for val in glob.glob(op.join(self.sourcedata_pth, 'sub-{sj}'.format(sj=s), 'ses-*'))] 
        
        
    
class MRIData(FAMData):
    
    """MRIData

    Class that loads relevant paths and settings for (f)MRI data
    """
    
    def __init__(self, params, sj_num, exclude_sj = [], repo_pth = '', base_dir = None):  # initialize child class

        """ Initializes MRIData object. 

        Parameters
        ----------
        params : str/yaml dict
            where parameters from experiment and analysis live
        sj_num : str/list/arr
            participant number(s)
        exclude_sj: list/arr
            list with subject numbers to exclude
        repo_pth: str
            string with absolute path where module is installed in system - needed to get MATLAB .m files 
        """

        # need to initialize parent class (BehTask), indicating output infos
        super().__init__(params = params, sj_num = sj_num, exclude_sj = exclude_sj, base_dir=base_dir)

        ## some paths
        # anat preprocessing path, 
        # where we run linescanning preprocessing pipeline
        self.anat_preproc_pth = op.join(self.proj_root_pth, 'anat_preprocessing')

        # path to BFC files
        self.bfc_pth = op.join(self.proj_root_pth, 'BiasFieldCorrection')

        # path to NORDIC files
        self.nordic_pth = op.join(self.proj_root_pth, 'NORDIC')
        
        # path to freesurfer
        self.freesurfer_pth = op.join(self.derivatives_pth, 'freesurfer')
        
        # path to fmriprep
        self.fmriprep_pth = op.join(self.derivatives_pth, 'fmriprep')

        # path to repo install (needed to run mat files)
        self.repo_pth = repo_pth
        
        
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
        
        # matlab install location
        self.matlab_pth = self.params['mri']['paths'][self.base_dir]['matlab']
        
        if input_pth is None:
            input_pth = glob.glob(op.join(self.sourcedata_pth, 'sub-{sj}'.format(sj=participant), 'ses-*', 'anat'))[0]
        
        # list original (uncorrected) files (can be T1 or T2)
        orig_files = [op.join(input_pth,run) for run in os.listdir(input_pth) if run.endswith('{file}.nii.gz'.format(file=file_type))]

        # make output folder to store copy of original, tmp and output files
        out_pth = self.bfc_pth
        
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

                batch_dir = op.join(self.proj_root_pth,'batch')
                if not op.exists(batch_dir):
                        os.makedirs(batch_dir)
                        
                keys2replace = {'$SJ_NR': str(participant).zfill(3),
                        '$ORIG': orig, 
                        '$OUTFOLDER': outfolder, 
                        '$INPUT': op.split(orig)[-1].replace('.nii.gz','.nii'),
                        '$REPO': self.repo_pth,
                        '$MATLAB': self.matlab_pth,
                        '$ROOTFOLDER': self.proj_root_pth 
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
        for pp in self.sj_num:
            
            # path for sourcedata anat files of that participant
            anat_pth = glob.glob(op.join(self.sourcedata_pth, 'sub-{sj}'.format(sj=pp), 'ses-*', 'anat'))[0]
            
            ##### if we collected T2w files for participant #######
            if T2file:
                
                # check for T2w files in sourcedata
                T2w_files = [op.join(anat_pth, val) for val in os.listdir(anat_pth) if val.endswith('_T2w.nii.gz')]

                if len(T2w_files) == 0:
                    raise NameError('No T2w files in {folder}!'.format(folder=anat_pth))
                else:
                    # check if we bias field correct T2 files
                    if self.params['mri']['preproc']['BFC_T2']:
                        bfc_sj = [op.join(self.bfc_pth, val) for val in os.listdir(self.bfc_pth) if val.startswith('sub-{sj}'.format(sj=pp)) and \
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
                masked_T1_pth = op.join(self.anat_preproc_pth, 'derivatives', 'masked_mp2rage', 
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
        """ 
        
        if self.base_dir == 'local':
            raise NameError('Dont run freesurfer locally - only implemented in slurm systems')
        
        # loop over participants
        for pp in self.sj_num:
            
            ## freesurfer command ##
            freesurfer_cmd = 'recon-all -s {sj} -hires '.format(sj=pp)
            
            # path to store freesurfer outputs, in derivatives
            out_dir = op.join(self.freesurfer_pth, 'sub-{sj}'.format(sj=pp))
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
            anat_pth = glob.glob(op.join(self.sourcedata_pth, 'sub-{sj}'.format(sj=pp), 'ses-*', 'anat'))[0]
            
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
                     partition_name = None, node_name = None):
        
        """
        Run FMRIPREP on anat or functional data
        
        NOTE - needs to be run in slurm system!!
        
        Parameters
        ----------
        data_type : str
            if we want to run it on 'anat' or 'func'
        """ 
        
        if self.base_dir == 'local':
            raise NameError('Dont run freesurfer locally - only implemented in slurm systems')
        
        # loop over participants
        for pp in self.sj_num:
            
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
            #if not op.exists(self.fmriprep_pth):
            #    os.makedirs(self.fmriprep_pth)
                
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
--use-syn-sdc --force-syn --bold2t1w-dof 6 --stop-on-first-crash --verbose --skip_bids_validation --dummy-scans 5 \
-w $WF_DIR

wait          # wait until programs are finished

echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"
"""

            
            #os.chdir(batch_dir)
            batch_string = fmriprep_cmd

            keys2replace = {'$SJ_NR': 'sub-{sj}'.format(sj=pp),
                            '$SINGIMG': self.params['mri']['paths'][self.base_dir]['singularity'],
                            '$ROOTFOLDER': op.split(self.sourcedata_pth)[0],
                            '$WF_DIR': wf_dir,
                            '$BD': batch_dir
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