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
    
    def __init__(self, params, sj_num, exclude_sj = []):
        
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
        self.base_dir = self.params['general']['current_dir'] 
        
        # project root folder
        self.proj_root_pth = self.params['mri']['paths'][self.base_dir]['root']
        
        # sourcedata dir
        self.sourcedata_pth = self.params['mri']['paths'][self.base_dir]['sourcedata']
        
        # derivatives dir
        self.derivatives_pth = self.params['mri']['paths'][self.base_dir]['derivatives']
        
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
    
    def __init__(self, params, sj_num, exclude_sj = [], repo_pth = ''):  # initialize child class

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
        super().__init__(params = params, sj_num = sj_num, exclude_sj = exclude_sj)

        ## some paths
        # anat preprocessing path, 
        # where we run linescanning preprocessing pipeline
        self.anat_preproc_pth = op.join(self.proj_root_pth, 'anat_preprocessing')

        # path to BFC files
        self.bfc_pth = op.join(self.proj_root_pth, 'BiasFieldCorrection')

        # path to NORDIC files
        self.nordic_pth = op.join(self.proj_root_pth, 'NORDIC')

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
        """ 
        
        # loop over participants
        for pp in self.sj_num:
            
            # path for sourcedata anat files of that participant
            anat_pth = glob.glob(op.join(self.sourcedata_pth, 'sub-{sj}'.format(sj=pp), 'ses-*', 'anat'))[0]
            
            # if we collected T2w files for participant
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