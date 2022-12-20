#!/bin/bash

cd /afs/cern.ch/work/z/zhangr/FCG/FastCaloChallenge/training
source /afs/cern.ch/work/z/zhangr/HH4b/hh4bStat/scripts/setup.sh

task=$1
input=$2
model=$3

if [[ ${task} == *'train'* ]]; then
    command="python train.py -i ${input} -o ../output/dataset1/v1/${model} -c ../config/config_${model}.json"
else
    command="python evaluate.py -i ${input} -t ../output/dataset1/v1/${model}"
fi
echo $command
$command

