import os, sys
import os.path as op
import numpy as np
import argparse
import time


class Batcher:
    
    def __init__(self, MRIObj, log_dir = '/home/$USER/batch', wf_dir = '/scratch-shared/$USER/FAM'):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        MRIObj : MRIData object
            object from one of the classes defined in processing.load_exp_data
            
        """

        # set data object to use later on
        self.MRIObj = MRIObj
        
        # some useful vars
        self.log_dir = log_dir # where log files will be stored
        self.wf_dir = wf_dir # scratch folder where processes are run
        self.group_bool = False # if we want to create a script to run across participants
        self.send_email = False
        self.partition_name = None
        self.node_name = None
        self.batch_mem_Gib = None
        
    def setup_slurm_pars(self, n_hours = 2, n_tasks = 16, n_nodes = 1, n_cpus_task = 4):
        
        """Setup some general SLURM parameters
        """
        
        self.run_time = '{h}:00:00'.format(h = str(n_hours)) 
        self.n_nodes = n_nodes
        self.n_tasks = n_tasks
        self.n_cpus_task = n_cpus_task
          
    def create_sh_script_str(self, participant_list = [], step_type = 'fitmodel', taskname = 'pRF', concurrent_job = True,
                                n_jobs = 8, n_batches = 10, chunk_data = None, fit_hrf = True, use_rsync = False):
        
        """Make full batch script for analysis 
        """
        
        # for FA specific analysis we use FS ROI labels
        use_fs_label = True if taskname == 'FA' else False
        
        # get base format for bash 
        bash_basetxt = self.make_base_script(use_rsync = use_rsync)
        
        # if fitting model, we also need to copy fit folder from derivatives
        if step_type == 'fitmodel':
            copy_cmd = self.copy_fit_deriv(use_rsync = use_rsync, task = taskname, use_fs_label = use_fs_label)
        else:
            copy_cmd = ''
        
        # get analysis specific part of batch script
        fit_cmd = self.make_step_script(participant_list = participant_list, step_type = step_type, 
                                        concurrent_job = concurrent_job,
                                        n_jobs = n_jobs, n_batches = n_batches, chunk_data = chunk_data, 
                                        fit_hrf = fit_hrf)
        
        ## replace common commands and scratch dir in base string
        working_string = bash_basetxt.replace('$COPY_CMD', copy_cmd)
        working_string = working_string.replace('$PY_CMD', fit_cmd)
        working_string = working_string.replace('$TMPDIR', self.wf_dir) 
        working_string = working_string.replace('$LOGFILENAME', 'slurm_FAM_{skey}'.format(skey = step_type)) 
        
        # replace list of participants
        working_string = working_string.replace('$PP_LIST', ' '.join(participant_list))

        # replace with parameters from MRI object
        working_string = working_string.replace('$SPACE', self.MRIObj.sj_space)
        if self.MRIObj.sj_space == 'T1w':
           working_string = working_string.replace('$PRFSPACE', 'fsnative') 
        else:
            working_string = working_string.replace('$PRFSPACE', self.MRIObj.sj_space)
        working_string = working_string.replace('$FITFOLDER', self.MRIObj.params['mri']['fitting'][taskname]['fit_folder'])
        working_string = working_string.replace('$DERIV_DIR', self.MRIObj.derivatives_pth)
        working_string = working_string.replace('$SOURCE_DIR', self.MRIObj.sourcedata_pth)
        
        return working_string
               
    def submit_jobs(self, participant_list = [], step_type = 'fitmodel', taskname = 'pRF', concurrent_job = True,
                        n_jobs = 8, n_batches = 10, chunk_data = None, fit_hrf = True, use_rsync = False, 
                        dry_run = False, prf_model_name = 'gauss', fa_model_name = 'glmsingle',
                        run_type = 'mean', ses2fit = 'mean'):
        
        """script to actually submit the jobs
        """
        
        if concurrent_job and len(participant_list) > 1:
            self.group_bool = True
        
        # create full script str
        working_str = self.create_sh_script_str(participant_list = participant_list, step_type = step_type, 
                                                taskname = taskname, concurrent_job = concurrent_job,
                                                n_jobs = n_jobs, n_batches = n_batches, chunk_data = chunk_data, fit_hrf = fit_hrf, 
                                                use_rsync = use_rsync)
        
        # replace model specific parts, if applicable
        working_str = working_str.replace('$TASK', taskname)
        working_str = working_str.replace('$SES', ses2fit)
        working_str = working_str.replace('$RUN', run_type)
        working_str = working_str.replace('$PRFMODEL', prf_model_name)
        working_str = working_str.replace('$FAMODEL', fa_model_name)

        # batch filename
        if concurrent_job and len(participant_list) > 1:
            js_name = op.join(self.log_dir, '{skey}_sub-GROUP_FAM.sh'.format(skey = step_type))
            job_file_list = [js_name]
            working_str_list = [working_str]
        else:
            js_name = op.join(self.log_dir, '{skey}_sub-$SJ_NR_FAM.sh'.format(skey = step_type))
            
            job_file_list = []
            working_str_list = []
            
            # loop over participants
            for pp in participant_list:
                working_str_list.append(working_str.replace('$SJ_NR', pp))
                job_file_list.append(js_name.replace('$SJ_NR', pp))
              
        # iterate over jobs, print them for inspection and submit  
        for i in range(len(job_file_list)):
            
            print(working_str_list[i])
            
            if dry_run == False:
                of = open(job_file_list[i], 'w')
                of.write(working_str_list[i])
                of.close()
                
                print('submitting ' + job_file_list[i] + ' to queue')
                os.system('sbatch ' + job_file_list[i])
        
    def make_step_script(self, participant_list = [], step_type = 'fitmodel', concurrent_job = True,
                            n_jobs = 8, n_batches = 10, chunk_data = None, fit_hrf = True):
        
        """Make analysis step specific script
        """
        
        if step_type == 'post_fmriprep':
            
            # set general analysis command iterating over participant list
            if concurrent_job:
                fit_cmd = """declare -a pp_arr=($PP_LIST)\n"""+ \
                    """for ((i = 0; i < ${#pp_arr[@]}; i++)); do\n"""+ \
                    """(\n  python process_data.py --subject ${pp_arr[$i]} --step post_fmriprep --dir slurm\n) &\n"""+ \
                    """done\n\nwait\n\n"""
            else:
                fit_cmd = """python process_data.py --subject $SJ_NR --step post_fmriprep --dir slurm\n"""+ \
                    """done\n\nwait\n\n"""
                
        elif step_type == 'fitmodel':
            
            # set general analysis command iterating over participant list
            if concurrent_job:
                fit_cmd = """declare -a pp_arr=($PP_LIST)\n"""+ \
                    """for ((i = 0; i < ${#pp_arr[@]}; i++)); do\n"""+ \
                    """(\n  python run_analysis.py --subject ${pp_arr[$i]} --cmd fitmodel --task $TASK --dir slurm """+ \
                    """--ses2fit $SES --run_type $RUN --prf_model_name $PRFMODEL """+ \
                    """--fa_model_name $FAMODEL """+ \
                    """--n_jobs {n_jobs} --n_batches {n_batches} """.format(n_jobs = n_jobs, n_batches = n_batches)+ \
                    """--wf_dir $TMPDIR """
                    
                # if chunking data
                if chunk_data is not None:
                    fit_cmd += """--chunk_num $CH --total_chunks $TOTALCH """

                # if we want to fit hrf
                if fit_hrf:
                    fit_cmd += """--fit_hrf """
                    
                fit_cmd += """\n) &\n"""+ \
                    """done\n\nwait"""
            else:
                # set fitting model command 
                fit_cmd = """python run_analysis.py --subject $SJ_NR --cmd fitmodel --task $TASK --dir slurm """+ \
                    """--ses2fit $SES --run_type $RUN --prf_model_name $PRFMODEL """+ \
                    """--fa_model_name $FAMODEL """+ \
                    """--n_jobs {n_jobs} --n_batches {n_batches} """.format(n_jobs = n_jobs, n_batches = n_batches)+ \
                    """--wf_dir $TMPDIR """
                
                # if chunking data
                if chunk_data is not None:
                    fit_cmd += """--chunk_num $CH --total_chunks $TOTALCH """

                # if we want to fit hrf
                if fit_hrf:
                    fit_cmd += """--fit_hrf """

            fit_cmd += """\n\n"""
             
        return fit_cmd
    
    def make_base_script(self, use_rsync = False):
        
        """Make generic SLURM script
        
        which implies setting up the number of nodes, runtime etc
        copying/rsyncing the derivatives and sourcedata to the node
        and then rsyncing the derivatives back to the project folder
        """
        
        ## initialize batch script, with general node specs
        slurm_cmd = """#!/bin/bash\n#SBATCH -t {rtime}\n#SBATCH -N {n_nodes}\n"""+ \
        """#SBATCH -v\n#SBATCH --ntasks-per-node={ntasks}\n"""+ \
        """#SBATCH --cpus-per-task={n_cpus_task}\n"""+ \
        """#SBATCH --output={log_dir}/$LOGFILENAME_%A.out\n\n"""
        slurm_cmd = slurm_cmd.format(rtime = self.run_time, n_nodes = self.n_nodes, log_dir = self.log_dir,
                                     n_cpus_task = self.n_cpus_task, ntasks = self.n_tasks)
        
        ## if we want a specific node/partition
        if self.partition_name is not None:
            slurm_cmd += '#SBATCH --partition={p}\n'.format(p=self.partition_name)
        if self.node_name is not None:
            slurm_cmd += '#SBATCH -w {n}\n'.format(n=self.node_name)

        ## add memory for node
        if self.batch_mem_Gib is not None:
            slurm_cmd += '#SBATCH --mem={mem}G\n'.format(mem=self.batch_mem_Gib)
            
        ## rsync general folders that should be needed for 
        if use_rsync:
            slurm_cmd += self.rsync_deriv()
        else:
            slurm_cmd += self.cp_deriv() # or copy if more efficient
            
        ## add option to copy different directory to tmp dir (might be needed when, for example, fitting models)
        slurm_cmd += """$COPY_CMD\n\n"""
        
        ## call final part (with actual command)
        bash_string = slurm_cmd + \
            """$PY_CMD\nwait # wait until programs are finished\n\n"""+ \
            """rsync -chavP --exclude=".*" $TMPDIR/derivatives/ $DERIV_DIR --no-compress\n\nwait\n\n$END_EMAIL\n"""
        
        ## if we want to send email
        if self.send_email:
            bash_string = bash_string.replace('$START_EMAIL', 'echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"')
            bash_string = bash_string.replace('$END_EMAIL', 'echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"')
        
        return bash_string
            
    def copy_fit_deriv(self, use_rsync = False, task = 'pRF', use_fs_label = False):
        
        """Copy extra folder from derivatives, with fits
        + freesurfer folder with subject ROI labels
        """
        
        # if running for group
        if self.group_bool:
            if task == 'pRF':
                if use_rsync:
                    fit_cmd = """mkdir -p $TMPDIR/derivatives/$FITFOLDER/$SPACE\n\nwait\n\n"""+ \
                    """if [ -d "$DERIV_DIR/$FITFOLDER/$SPACE" ]\nthen\n"""+ \
                    """    rsync -chavP --exclude=".*" $DERIV_DIR/$FITFOLDER/$SPACE/ $TMPDIR/derivatives/$FITFOLDER/$SPACE --no-compress\nfi\n\nwait\n\n"""
                else:
                    fit_cmd = """mkdir -p $TMPDIR/derivatives/$FITFOLDER/$SPACE\n\nwait\n\n"""+ \
                    """if [ -d "$DERIV_DIR/$FITFOLDER/$SPACE" ]\nthen\n"""+ \
                    """    cp -r $DERIV_DIR/$FITFOLDER/$SPACE $TMPDIR/derivatives/$FITFOLDER/\nfi\n\nwait\n\n"""    
                
            elif task == 'FA':
                # if we are fitting FA, then also need to copy pRF estimates to scratch
                if use_rsync:
                    fit_cmd = """mkdir -p $TMPDIR/derivatives/{$FITFOLDER,$PRFFITFOLDER}/{$SPACE, $PRFSPACE}/\n\nwait\n\n"""+ \
                    """if [ -d "$DERIV_DIR/$PRFFITFOLDER/$PRFSPACE/" ]\nthen\n"""+ \
                    """    rsync -chavP --exclude=".*" $DERIV_DIR/$PRFFITFOLDER/$PRFSPACE/ $TMPDIR/derivatives/$PRFFITFOLDER/$PRFSPACE--no-compress\nfi\n\n"""+ \
                    """if [ -d "$DERIV_DIR/$FITFOLDER/$SPACE" ]\nthen\n"""+ \
                    """    rsync -chavP --exclude=".*" $DERIV_DIR/$FITFOLDER/$SPACE/ $TMPDIR/derivatives/$FITFOLDER/$SPACE --no-compress\nfi\n\n"""+ \
                    """wait\n\n"""
                else:
                    fit_cmd = """mkdir -p $TMPDIR/derivatives/{$FITFOLDER,$PRFFITFOLDER}/{$SPACE, $PRFSPACE}/\n\nwait\n\n"""+ \
                    """if [ -d "$DERIV_DIR/$PRFFITFOLDER/$PRFSPACE" ]\nthen\n"""+ \
                    """    cp -r $DERIV_DIR/$PRFFITFOLDER/$PRFSPACE $TMPDIR/derivatives/$PRFFITFOLDER/\nfi\n\n"""+ \
                    """if [ -d "$DERIV_DIR/$FITFOLDER/$SPACE" ]\nthen\n"""+ \
                    """    cp -r $DERIV_DIR/$FITFOLDER/$SPACE $TMPDIR/derivatives/$FITFOLDER/\nfi\n\n"""+ \
                    """wait\n\n"""
                fit_cmd = fit_cmd.replace('$PRFFITFOLDER', self.MRIObj.params['mri']['fitting']['pRF']['fit_folder'])
        else:
            if task == 'pRF':
                if use_rsync:
                    fit_cmd = """mkdir -p $TMPDIR/derivatives/$FITFOLDER/$SPACE/sub-$SJ_NR\n\nwait\n\n"""+ \
                    """if [ -d "$DERIV_DIR/$FITFOLDER/$SPACE/sub-$SJ_NR" ]\nthen\n"""+ \
                    """    rsync -chavP --exclude=".*" $DERIV_DIR/$FITFOLDER/$SPACE/sub-$SJ_NR/ $TMPDIR/derivatives/$FITFOLDER/$SPACE/sub-$SJ_NR --no-compress\nfi\n\nwait\n\n"""
                else:
                    fit_cmd = """mkdir -p $TMPDIR/derivatives/$FITFOLDER/$SPACE/sub-$SJ_NR\n\nwait\n\n"""+ \
                    """if [ -d "$DERIV_DIR/$FITFOLDER/$SPACE/sub-$SJ_NR" ]\nthen\n"""+ \
                    """    cp -r $DERIV_DIR/$FITFOLDER/$SPACE/sub-$SJ_NR $TMPDIR/derivatives/$FITFOLDER/$SPACE/\nfi\n\nwait\n\n"""    
                
            elif task == 'FA':
                # if we are fitting FA, then also need to copy pRF estimates to scratch
                if use_rsync:
                    fit_cmd = """mkdir -p $TMPDIR/derivatives/{$FITFOLDER,$PRFFITFOLDER}/{$SPACE, $PRFSPACE}/sub-$SJ_NR\n\nwait\n\n"""+ \
                    """if [ -d "$DERIV_DIR/$PRFFITFOLDER/$PRFSPACE/sub-$SJ_NR" ]\nthen\n"""+ \
                    """    rsync -chavP --exclude=".*" $DERIV_DIR/$PRFFITFOLDER/$PRFSPACE/sub-$SJ_NR/ $TMPDIR/derivatives/$PRFFITFOLDER/$PRFSPACE/sub-$SJ_NR --no-compress\nfi\n\n"""+ \
                    """if [ -d "$DERIV_DIR/$FITFOLDER/$SPACE/sub-$SJ_NR" ]\nthen\n"""+ \
                    """    rsync -chavP --exclude=".*" $DERIV_DIR/$FITFOLDER/$SPACE/sub-$SJ_NR/ $TMPDIR/derivatives/$FITFOLDER/$SPACE/sub-$SJ_NR --no-compress\nfi\n\n"""+ \
                    """wait\n\n"""
                else:
                    fit_cmd = """mkdir -p $TMPDIR/derivatives/{$FITFOLDER,$PRFFITFOLDER}/{$SPACE, $PRFSPACE}/sub-$SJ_NR\n\nwait\n\n"""+ \
                    """if [ -d "$DERIV_DIR/$PRFFITFOLDER/$PRFSPACE/sub-$SJ_NR" ]\nthen\n"""+ \
                    """    cp -r $DERIV_DIR/$PRFFITFOLDER/$PRFSPACE/sub-$SJ_NR $TMPDIR/derivatives/$PRFFITFOLDER/$PRFSPACE/\nfi\n\n"""+ \
                    """if [ -d "$DERIV_DIR/$FITFOLDER/$SPACE/sub-$SJ_NR" ]\nthen\n"""+ \
                    """    cp -r $DERIV_DIR/$FITFOLDER/$SPACE/sub-$SJ_NR $TMPDIR/derivatives/$FITFOLDER/$SPACE/\nfi\n\n"""+ \
                    """wait\n\n"""
                fit_cmd = fit_cmd.replace('$PRFFITFOLDER', self.MRIObj.params['mri']['fitting']['pRF']['fit_folder'])
         
        # if our analysis requires freesurfer labels   
        if use_fs_label:
            fit_cmd += self.copy_FS_deriv(use_rsync = use_rsync)
            
        return fit_cmd
    
    def copy_FS_deriv(self, use_rsync = False):
        
        """For some analysis we also use FS subject labels, to define ROIs
        so we also need to add that to the script
        """
        
        # if running for group
        if self.group_bool:
            if use_rsync:
                copyFS_cmd = """mkdir -p $TMPDIR/derivatives/freesurfer\nwait\n"""+ \
                    """rsync -chavP --exclude=".*" $DERIV_DIR/freesurfer/ $TMPDIR/derivatives/freesurfer --no-compress\nwait\n\n"""
            else:
                copyFS_cmd = """mkdir -p $TMPDIR/derivatives/freesurfer\nwait\n"""+ \
                    """cp -r $DERIV_DIR/freesurfer $TMPDIR/derivatives/\nwait\n\n"""
        else:
            if use_rsync:
                copyFS_cmd = """mkdir -p $TMPDIR/derivatives/freesurfer/sub-$SJ_NR\nwait\n"""+ \
                    """rsync -chavP --exclude=".*" $DERIV_DIR/freesurfer/sub-$SJ_NR/ $TMPDIR/derivatives/freesurfer/sub-$SJ_NR --no-compress\nwait\n\n"""
            else:
                copyFS_cmd = """mkdir -p $TMPDIR/derivatives/freesurfer/sub-$SJ_NR\nwait\n"""+ \
                    """cp -r $DERIV_DIR/freesurfer/sub-$SJ_NR $TMPDIR/derivatives/freesurfer\nwait\n\n"""
                    
        return copyFS_cmd
            
    def rsync_deriv(self):
    
        """Snippet for rsyncing postfmriprep derivatives
        """
        
        # if we want to run it for the group, then copy dir for all participants
        if self.group_bool:
            cmd = """# call the programs\n$START_EMAIL\n\n"""+\
            """# make derivatives dir in node and sourcedata because we want to access behav files\n"""+ \
            """mkdir -p $TMPDIR/derivatives/post_fmriprep/$SPACE\n"""+ \
            """mkdir -p $TMPDIR/sourcedata\n\nwait\n\n"""+\
            """rsync -chavP --exclude=".*" $DERIV_DIR/post_fmriprep/$SPACE/ $TMPDIR/derivatives/post_fmriprep/$SPACE --no-compress\n\nwait\n\n"""+\
            """rsync -chavP --exclude=".*" $SOURCE_DIR/ $TMPDIR/sourcedata --no-compress\n\nwait\n\n"""
        else:
            cmd = """# call the programs\n$START_EMAIL\n\n"""+\
            """# make derivatives dir in node and sourcedata because we want to access behav files\n"""+ \
            """mkdir -p $TMPDIR/derivatives/post_fmriprep/$SPACE/sub-$SJ_NR\n"""+ \
            """mkdir -p $TMPDIR/sourcedata/sub-$SJ_NR\n\nwait\n\n"""+\
            """rsync -chavP --exclude=".*" $DERIV_DIR/post_fmriprep/$SPACE/sub-$SJ_NR/ $TMPDIR/derivatives/post_fmriprep/$SPACE/sub-$SJ_NR --no-compress\n\nwait\n\n"""+\
            """rsync -chavP --exclude=".*" $SOURCE_DIR/sub-$SJ_NR/ $TMPDIR/sourcedata/sub-$SJ_NR --no-compress\n\nwait\n\n"""
            
        return cmd

    def cp_deriv(self):
        
        """Snippet for cp postfmriprep derivatives
        """
        
        # if we want to run it for the group, then copy dir for all participants
        if self.group_bool:
            cmd = """# call the programs\n$START_EMAIL\n\n"""+\
            """# make derivatives dir in node and sourcedata because we want to access behav files\n"""+ \
            """mkdir -p $TMPDIR/derivatives/post_fmriprep/$SPACE\n"""+ \
            """mkdir -p $TMPDIR/sourcedata\n\nwait\n\n"""+\
            """cp -r $DERIV_DIR/post_fmriprep/$SPACE $TMPDIR/derivatives/post_fmriprep/\n\nwait\n\n"""+\
            """cp -r $SOURCE_DIR/ $TMPDIR/\n\nwait\n\n"""
        else:
            cmd = """# call the programs\n$START_EMAIL\n\n"""+\
            """# make derivatives dir in node and sourcedata because we want to access behav files\n"""+ \
            """mkdir -p $TMPDIR/derivatives/post_fmriprep/$SPACE/sub-$SJ_NR\n"""+ \
            """mkdir -p $TMPDIR/sourcedata/sub-$SJ_NR\n\nwait\n\n"""+\
            """cp -r $DERIV_DIR/post_fmriprep/$SPACE/sub-$SJ_NR $TMPDIR/derivatives/post_fmriprep/$SPACE/\n\nwait\n\n"""+\
            """cp -r $SOURCE_DIR/sub-$SJ_NR $TMPDIR/sourcedata/\n\nwait\n\n"""
            
        return cmd