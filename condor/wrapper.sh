#!/bin/bash

cd /afs/cern.ch/work/z/zhangr/FCG/FastCaloChallenge/training
source /afs/cern.ch/work/z/zhangr/HH4b/hh4bStat/scripts/setup.sh

task=$1
input=$2
output=$3

model=`echo $output | cut -d '_' -f 1`
config=`echo $output | cut -d '_' -f 2`
config=`echo $config | cut -d '-' -f 1`

if [[ ${task} == *'train'* ]]; then
    command="python train.py -i ${input} -m ${model} -o ../output/dataset1/v1/${output} -c ../config/config_${config}.json"
else
    command="python evaluate.py -i ${input} -t ../output/dataset1/v1/${output}"
fi
echo $command
$command

