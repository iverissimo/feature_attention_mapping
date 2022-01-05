#!/bin/bash

echo "Running pre-fmriprep scripts"
machine=$1
sj=$2

conda activate i36 # activate environment where relevant packages installed

echo "running $sj on $machine"

if [ "$machine" == local ]; then

    echo "performing scripts that require MATLAB (only works locally for now)"

    python BiasFieldCorrec.py $sj $machine

elif [ "$machine" == lisa ]; then

    #SBATCH -t 96:00:00
    #SBATCH -N 1 --mem=65536
    #SBATCH --cpus-per-task=16
    #SBATCH -v
    #SBATCH --output=/home/inesv/batch/slurm_pre-fMRIprep_%A.out
    
    # call the programs
    echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"



else
    echo "invalid machine name $machine"
    exit 1
fi

