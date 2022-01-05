#!/bin/bash

echo "Running pre-fmriprep scripts"
echo "Enter machine were we are running scripts (local vs lisa)"
read machine

echo "Enter subject number"
read sj

if [ "$machine" == local ] ||  [ "$machine" == lisa ]; then
    echo "running on $machine"
else
    echo "invalid machine name $machine"
    exit 1
fi

conda activate i36 # activate environment where relevant packages installed

echo "pre-processign anatomicals for $sj"

if [ "$machine" == local ]; then

    echo "running on $machine"

elif [ "$machine" == lisa ]; then

else
    echo "invalid machine name $machine"
    exit 1
fi

python BiasFieldCorrec.py $sj $machine