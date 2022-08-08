import numpy as np
import os.path as op
import yaml
import glob


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
            
        # excluded participants
        self.exclude_sj = exclude_sj
        if len(self.exclude_sj)>0:
            self.exclude_sj = [str(val).zfill(3) for val in exclude_sj]

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

        # path to matlab install
        if self.base_dir == 'local':
            self.matlab_pth = self.params['mri']['paths'][self.base_dir]['matlab']


        
        
    