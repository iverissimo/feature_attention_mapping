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
with open(os.path.join(os.path.split(os.getcwd())[0],'exp_params.yml'), 'r') as f_in:
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


# save values in list, for later use
dictionary_list = []

# for webshow visualization of all volumes
DS = {}

for _,nord_bol in enumerate([False,True]): 
    
    # load paths

    if nord_bol:
        preproc = 'nordic'
        pycortex_sub = '{ds}.{sub}'.format(ds='nordic', sub=sj)
        dataset = 'nordic'

    else:
        preproc = 'standard'
        pycortex_sub = '{sub}'.format(sub=sj)
        dataset = None

    derivatives_pth = params['mri']['paths'][base_dir][preproc]
    fit_pth =  op.join(derivatives_pth,'pRF_fitting') # for estimates output
    output_pth =  op.join(derivatives_pth,'rsquared') # for estimates output

    if not op.exists(output_pth):
        print('output dir does not existing, saving files in %s'%output_pth)
        os.makedirs(output_pth)
        
    # make pycortex subject
    import_fmriprep2pycortex(derivatives_pth, sj, dataset=dataset, ses='01', acq='MP2RAGE')
    
    # get T1w from subject, to use later
    t1w = cortex.db.get_anat(pycortex_sub)
    
    # iterate over runs
    for _,acq_type in enumerate(params['general']['acq_type']):
    
        # first combine separate estimates files
        # to unique file
        vols = [op.join(fit_pth,x) for _,x in enumerate(os.listdir(fit_pth)) if '-{acq}_'.format(acq=acq_type) in x and '_slice-' in x]
        vols.sort()
        
        # estimates data
        data = combine_slices(vols, fit_pth, num_slices=89, ax=2)
        
        # get rsq data for volume
        rsq_img = nib.load(data)
        rsq_arr = rsq_img.dataobj[...,-1] 

        # and save in output dir
        rsq_filename = op.split(data)[-1].replace('.nii.gz','_rsq.nii.gz')
        rsq_filename = op.join(output_pth,rsq_filename)

        if op.exists(rsq_filename):
            print('already exists %s'%rsq_filename)
        else:
            rsq_out = nib.nifti1.Nifti1Image(rsq_arr, affine=rsq_img.affine, header=rsq_img.header)
            nib.save(rsq_out,rsq_filename)
            
        # save values in DF
        # determine voxel indices, 
        # in case need to use later
        vox_indices = [(xx,yy,zz) for xx in range(rsq_arr.shape[0]) for yy in range(rsq_arr.shape[1]) for zz in range(rsq_arr.shape[2])]
        
        for vox in vox_indices: # go through voxels, save in DF for later plotting
            
            dictionary_data = {'rsq': rsq_arr[vox],
                                'vox_ind': vox,
                                'acq': acq_type, 
                                'NORDIC': nord_bol}
            dictionary_list.append(dictionary_data)
        
        # resample image to t1w space
        # no registration is performed: the image should already be aligned
        data_t1w = image.resample_to_img(nib.load(data), t1w)
        rsq_t1w = np.array(data_t1w.dataobj)[...,-1]
        
        # name for dataset in browser
        if nord_bol:
            string = 'rsq_acq-{acq}_NORDIC'
        else:
            string = 'rsq_acq-{acq}'

        DS[string.format(acq=acq_type)] = cortex.Volume(rsq_t1w.T, 
                                                        pycortex_sub, 
                                                        'identity',
                                                        vmin=0, vmax=0.6,cmap='hot')

        
# convert dictionary list to dataframe
df_rsq = pd.DataFrame.from_dict(dictionary_list)

# maskout nan and low rsq voxels
rsq_thresh = 0.125
masked_df_rsq = df_rsq.loc[df_rsq['rsq']>rsq_thresh]

fig = plt.figure(num=None, figsize=(10,7.5), dpi=100, facecolor='w', edgecolor='k')

ax = sns.violinplot(x="acq", y="rsq", hue="NORDIC",
                    data=masked_df_rsq, palette="muted")

ax.set(xlabel=None)
ax.set(ylabel=None)
plt.margins(y=0.025)
#sns.swarmplot(x='ecc', y='cs', data=crwd_df4plot,color=".25",alpha=0.5)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)

plt.xlabel('acquisition',fontsize = 20,labelpad=18)
plt.ylabel('RSQ',fontsize = 20,labelpad=18)
plt.ylim(0,1)

fig.savefig(os.path.join(output_pth,'nordic_comparison_rsq_violin_thresh-%0.2f.svg'%rsq_thresh), dpi=100)


fig = plt.figure(num=None, figsize=(10,7.5), dpi=100, facecolor='w', edgecolor='k')

sns.set_theme(style="darkgrid")
sns.set(font_scale=1.5) 

b = sns.displot(
    masked_df_rsq, x="rsq", col="acq", row="NORDIC",
    binwidth=0.05, facet_kws=dict(margin_titles=True),
    )

plt.xlim(0.1, 0.8)
b.savefig(op.join(output_pth,'nordic_comparison_rsq_hist_thresh-%0.2f.svg'%rsq_thresh))


# show in browser
cortex.webgl.show(data=DS,recache=True)





