import numpy as np
import os, sys
import os.path as op

import cortex

import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from nilearn import image

import yaml
from utils import * #import script to use relevante functions


# load settings from yaml
with open(op.join(op.split(op.split(os.getcwd())[0])[0],'exp_params.yml'), 'r') as f_in:
            params = yaml.safe_load(f_in)


# define participant number
if len(sys.argv)<3: 
    raise NameError('Please add subject number (ex: 001) '
                    'as 1st argument in the command line!')
elif len(sys.argv)<2:
    raise NameError('Please specify where running data (local vs lisa)'
                    'as 2nd argument in the command line!')

else:
    sj = str(sys.argv[1]).zfill(3) #fill subject number with 00 in case user forgets
    base_dir = str(sys.argv[2]) # which machine we run the data

# session number 
ses = params['general']['session']


# save values in DF
tSNR_list = []
   
# load paths

sourcedata_pth = op.join(params['mri']['paths'][base_dir], 'sourcedata','sub-{sj}'.format(sj=sj),
                        'ses-{ses}'.format(ses=ses),'func') 
output_pth = op.join(params['mri']['paths'][base_dir],'derivatives','pre_fmriprep','sub-{sj}'.format(sj=sj),
                        'ses-{ses}'.format(ses=ses)) 

# make output dir
if not op.exists(output_pth):
    print('output dir does not existing, saving files in %s'%output_pth)
    os.makedirs(output_pth)


# get list of func epis
vol_list = [os.path.join(sourcedata_pth,run) for _,run in enumerate(os.listdir(sourcedata_pth)) 
            if run.endswith('_bold.nii.gz')]

print('%i functional files found in %s'%(len(vol_list),sourcedata_pth))

## COMPUTE TSNR

# to save tSNR plots
tsnr_path = os.path.join(output_pth,'tSNR')
if not os.path.exists(tsnr_path):
    print('output dir does not exist, saving files in %s'%tsnr_path)
    os.makedirs(tsnr_path)
    
# iterate over runs
for _,acq_type in enumerate(['standard','nordic']):

    for _,task in enumerate(params['general']['tasks']):

        acq_vols = [vol for _,vol in enumerate(vol_list) if 'acq-{acq}_'.format(acq=acq_type) in vol 
                    and 'task-{task}_'.format(task=task) in vol]

        print('%i functional files found for task %s'%(len(acq_vols),task))

        for run in np.arange(len(acq_vols)):  
            
            vol = [x for _,x in enumerate(acq_vols) if 'run-{run}'.format(run=str(run+1)) in x][0]

            nibber = nib.load(vol)
            affine = nibber.affine
            data = np.array(nibber.dataobj)

            # name for tSNR nifti
            outfile = os.path.join(tsnr_path,os.path.split(vol)[-1].replace('.nii.gz','_tSNR.nii.gz'))

            img_tsnr = get_tsnr(data,affine,outfile)
            
            # determine voxel indices, 
            # in case need to use later
            vox_indices = [(xx,yy,zz) for xx in range(img_tsnr.shape[0]) for yy in range(img_tsnr.shape[1]) for zz in range(img_tsnr.shape[2])]

            for vox in vox_indices: # go through voxels, save in DF for later plotting

                dictionary_data = {'tsnr': img_tsnr[vox],
                                    'vox_ind': vox,
                                    'acq': acq_type, 
                                    'run': str(run+1),
                                    'task': task}
                tSNR_list.append(dictionary_data)
            
        


# convert dictionary list to dataframe
df_tSNR = pd.DataFrame.from_dict(tSNR_list)

# compute mean
masked_tsnr_list = []

for _,acq_type in enumerate(['standard','nordic']):

    for _,task in enumerate(params['general']['tasks']):

        for _,run in enumerate(df_tSNR['run'].unique()): 

            dictionary_data = {'tsnr':  np.nanmean(df_tSNR.loc[(df_tSNR['acq']==acq_type)&
                                 (df_tSNR['task']==task)&
                                 (df_tSNR['run']==run)]['tsnr'].values),
                                'acq': acq_type, 
                                'run': run,
                                'task': task}

            masked_tsnr_list.append(dictionary_data)


# convert dictionary list to dataframe
df_masked_tSNR = pd.DataFrame.from_dict(masked_tsnr_list)

# make barplot to compare
# mean tSNR pre and post nordic
fig = plt.figure(num=None, figsize=(10,7.5), dpi=100, facecolor='w', edgecolor='k')


b1 = sns.barplot(x='task', y='tsnr',hue='acq', data=df_masked_tSNR, capsize=.2,linewidth=1.8)

b1.set(xlabel=None)
b1.set(ylabel=None)
plt.margins(y=0.025)
#sns.swarmplot(x='ecc', y='cs', data=crwd_df4plot,color=".25",alpha=0.5)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)

plt.xlabel('task',fontsize = 20,labelpad=18)
plt.ylabel('mean tSNR',fontsize = 20,labelpad=18)
plt.ylim(0,60)

fig.savefig(os.path.join(tsnr_path,'nordic_comparison_mean_tSNR.svg'), dpi=100)
















