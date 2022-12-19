Training
```
- export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase # use your path
- source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
- lsetup "views LCG_100cuda x86_64-centos7-gcc8-opt"
- 
- source /afs/cern.ch/work/z/zhangr/HH4b/hh4bStat/scripts/setup.sh
python train.py    -i ../input/dataset1/dataset_1_photons_1.hdf5 -o ../output/dataset1/v1/
python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v1/
```
