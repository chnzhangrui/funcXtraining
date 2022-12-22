Training
```
source /afs/cern.ch/work/z/zhangr/HH4b/hh4bStat/scripts/setup.sh

python train.py    -i ../input/dataset1/dataset_1_photons_1.hdf5 -o ../output/dataset1/v1/GANv1_GANv1
python train.py    -i ../input/dataset1/dataset_1_photons_1.hdf5 -o ../output/dataset1/v1/GANv1_GANv1 -c ../config/config_GANv1.json
python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v1/GANv1_GANv1
```
