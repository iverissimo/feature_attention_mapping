import numpy as np
import os.path as op
import yaml
import glob

from FAM.utils.beh import BehUtils
from FAM.utils.mri import MRIUtils
#from FAM.utils.eye import EyeUtils

class FAMData:
    
    """FAMData
    Class that loads relevant paths and settings for FAM data
    """
    
    def __init__(self, params, sj_num, exclude_sj = [], base_dir = 'local', wf_dir = None):
        
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

        # relevant tasks
        self.tasks = ['pRF', 'FA']

        # timing
        self.TR = self.params['mri']['TR']
            
        # excluded participants
        self.exclude_sj = exclude_sj
        if len(self.exclude_sj)>0:
            print('Excluding participants {expp}'.format(expp = exclude_sj))
            self.exclude_sj = [str(val).zfill(3) for val in exclude_sj]
            
        ## set some paths
        # which machine we run the data
        self.base_dir = base_dir
        
        # project root folder
        self.proj_root_pth = self.params['mri']['paths'][self.base_dir]['root']

        # in case we are computing things in a different workflow dir
        # useful when fitting models in /scratch node
        if wf_dir is not None:
            self.proj_root_pth = wf_dir
        
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
        

class BehData(FAMData):

    """BehData

    Class that loads relevant paths and settings for behavioral data
    """
    
    def __init__(self, params, sj_num, exclude_sj = [], repo_pth = '', base_dir = None, wf_dir = None):  # initialize child class

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
        super().__init__(params = params, sj_num = sj_num, exclude_sj = exclude_sj, base_dir = base_dir, wf_dir = wf_dir)

        ## relevant file extensions
        self.events_ext = '_events.tsv'
        self.trial_info_ext = '_trial_info.csv'
        self.bar_pos_ext = '_bar_positions.pkl'

        ## some relevant params
        # session type (if beh, then training session, if func then scanning session)
        self.ses_type = ['beh','func'] 
        # color categories used (with two per color category)
        self.color_categories_dict = self.params['general']['task_colors']
        # actual colors used
        self.bar_colors = [element for sublist in self.color_categories_dict.values() for element in sublist]

        ## some pRF params relevant for setting task
        self.pRF_bar_pass = self.params['pRF']['bar_pass_direction']
        self.pRF_nr_TRs = self.params['pRF']['num_TRs'] 

        ## FA bar duration
        self.FA_bars_phase_dur = self.params['FA']['bars_phase_dur']
        ## some FA params relevant for setting task
        self.FA_bar_pass = self.params['FA']['bar_pass_direction']
        self.FA_nr_TRs = {'empty_TR': self.params['FA']['empty_TR'],
                          'task_trial_TR': self.params['FA']['task_trial_TR']}

        self.FA_num_bar_position = self.params['FA']['num_bar_position']

        # initialize utilities class
        self.beh_utils = BehUtils() 


class MRIData(BehData):
    
    """MRIData

    Class that loads relevant paths and settings for (f)MRI data
    """
    
    def __init__(self, params, sj_num, exclude_sj = [], repo_pth = '', base_dir = None, wf_dir = None):  # initialize child class

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
        super().__init__(params = params, sj_num = sj_num, exclude_sj = exclude_sj, base_dir = base_dir, wf_dir = wf_dir)

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

        ## some relevant params
        self.acq = self.params['mri']['acq'] # if using standard files or nordic files
        self.sj_space = self.params['mri']['space'] # subject space
        self.file_ext = self.params['mri']['file_ext'][self.sj_space] # file extension
        self.confound_ext = self.params['mri']['confounds']['file_ext'] # file extension
        self.mri_nr_runs = self.params['mri']['nr_runs']

        # path to post-fmriprep files
        self.postfmriprep_pth = op.join(self.derivatives_pth, 'post_fmriprep', self.sj_space)

        # atlas annotation file path
        self.atlas_annot = {'glasser': op.join(self.derivatives_pth, 'atlas', 'glasser','59k_mesh', 
                                        self.params['plotting']['glasser_annot']),
                            'wang': op.join(self.derivatives_pth, 'atlas', 'wang',
                                        self.params['plotting']['wang_annot'])}
        
        # pycortex subject 
        self.pysub = self.params['plotting']['pycortex_sub']

        ## number of cropped TRs
        # due to dummies TRs that were saved and extra ones as defined in params
        self.mri_nr_cropTR = {'pRF': self.params['mri']['dummy_TR'],
                              'FA': self.params['mri']['dummy_TR']}
        if self.params['pRF']['crop'] == True: 
            self.mri_nr_cropTR['pRF'] += self.params['pRF']['crop_TR']
        if self.params['FA']['crop'] == True: 
            self.mri_nr_cropTR['FA'] += self.params['FA']['crop_TR']

        # initialize utilities class
        self.mri_utils = MRIUtils() 
        
    