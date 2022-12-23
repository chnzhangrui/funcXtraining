#!/bin/bash

cd /afs/cern.ch/work/z/zhangr/FCG/FastCaloChallenge/training
source /afs/cern.ch/work/z/zhangr/HH4b/hh4bStat/scripts/setup.sh

task=$1
input=$2
output=$3

model=`echo $output | cut -d '_' -f 1`
config_mask=`echo $output | cut -d '_' -f 2`
config=`echo $config_mask | cut -d '-' -f 1`
mask=`echo $config_mask | cut -d '-' -f 2 | cut -d 'M' -f 2`

if [[ $mask == ?(-)+([0-9]) ]]; then
    version='v2'
    addition="--mask $mask"
else
    version='v1'
    addition=""
fi

if [[ ${task} == *'train'* ]]; then
    command="python train.py -i ${input} -m ${model} -o ../output/dataset1/${version}/${output} -c ../config/config_${config}.json ${addition}"
else
    command="python evaluate.py -i ${input} -t ../output/dataset1/${version}/${output}"
fi
echo $command
$command
cd -
