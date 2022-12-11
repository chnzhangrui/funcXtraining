#!/bin/bash

if [[ $HOSTNAME == "lxplus"* ]] || [[ $HOSTNAME == "pcuw"* ]] || [[ $HOSTNAME == *".cern.ch" ]]; then
    __conda_setup="$('/afs/cern.ch/user/z/zhangr/work/anaconda3.7/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/afs/cern.ch/user/z/zhangr/work/anaconda3.7/etc/profile.d/conda.sh" ]; then
            . "/afs/cern.ch/user/z/zhangr/work/anaconda3.7/etc/profile.d/conda.sh"
        else
            export PATH="/afs/cern.ch/user/z/zhangr/work/anaconda3.7/bin:$PATH"
        fi
    fi
    unset __conda_setup
else
    __conda_setup="$('/Users/zhangrui/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/Users/zhangrui/opt/anaconda3/etc/profile.d/conda.sh" ]; then
            . "/Users/zhangrui/opt/anaconda3/etc/profile.d/conda.sh"
        else
            export PATH="/Users/zhangrui/opt/anaconda3/bin:$PATH"
        fi
    fi
    unset __conda_setup
fi

conda activate roopy38a
which python
python --version
