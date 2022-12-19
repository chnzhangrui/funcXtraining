#!/bin/bash

cd /afs/cern.ch/work/z/zhangr/FCG/FastCaloChallenge/training
source /afs/cern.ch/work/z/zhangr/HH4b/hh4bStat/scripts/setup.sh

input=$1
model=$2

command="python train.py -i ${input} -o ../output/dataset1/v1/${model} -c ../config/config_${model}.json"
echo $command
$command

