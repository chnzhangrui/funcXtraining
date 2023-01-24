## Executing locally

```
cd training
```

Get input data
```
mkdir ../input/dataset1/
wget https://zenodo.org/record/6368338/files/dataset_1_photons_1.hdf5 ../input/dataset1/
```

Training
```
source /afs/cern.ch/work/z/zhangr/HH4b/hh4bStat/scripts/setup.sh

python train.py    -i ../input/dataset1/dataset_1_photons_1.hdf5 -o ../output/dataset1/v1/GANv1_GANv1 -c ../config/config_GANv1.json
python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v1/GANv1_GANv1
```

Best config
```
photon:
python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v2/BNswish_hpo4-M1
pions:
python evaluate.py -i ../input/dataset1/dataset_1_pions_1.hdf5 -t ../output/dataset1/v2/BNReLU_hpo4-M1000
```

## Training in parallel over multiple models with FuncX

### Step 0: Create a Conda environment on local and remote resources
Instructions can be found here: [Conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

The same process needs to be followed on both local and remote resources.

While Conda is not required, it will facilitate installation of the same versions of Python on both local and remote resources (required by FuncX), particular if Python versions differ by default.

### Step 1: Install this repo on both local and remote resources

Installing the repository installs the modules within this repo, in addition
to the program's dependencies, into the Conda environment. 

**Local**

Since access to the modules within the repository (e.g., training/train.py)
is required to launch the program, we will first clone this repository.

```
git clone https://github.com/ValHayot/funcXtraining.git
```

Once the repository is cloned, we can then install it inside our Conda enviornment

```
cd funcXtraining
pip install -e .
```

The `-e` flag is applied such that the code in the repository can be modified
and our environment will use the updated code when executed.

**Remote**

On the remote resource, we do not need to manually execute any of the repository modules, and therefore we can simply install the repository contents
directly into our Conda environment

```pip install git+https://github.com/ValHayot/funcXtraining.git```

### Step 2: Install and configure the FuncX enpoint on the remote resource
1. Install FuncX within the remote Conda environment using the following
command:

```pip install funcx-endpoint```

2. Configure the endpoint on the remote resource.

```
funcx-endpoint configure funcxtraining
```

This creates a default configuration of an endpoint, which may need to be
adapted to function appropriately on the remote resource (e.g., change the default provider
from `LocalProvider` to a cluster provider and add `conda activate <environment>` to the 
`worker_init` of the provider). Examples of
configuration files can be found [here](https://funcx.readthedocs.io/en/latest/endpoints.html)

The default FuncX configuration that has been generated can be found at `~/.funcx/funcxtraining/config.py` on the remote resource.

### Step 3: Start the FuncX endpoint and retrieve its endpoint ID
Once the endpoint is appropriately configured on the remote resource, it can
be started on the resource by executing the following command:

`funcx-endpoint start funcxtraining`

To obtain the endpoint ID, we can execute the following command:

`funcx-endpoint list`

The endpoint ID reflected here will be passed as an argument to 
`training/train.py` and the FuncX client started on the local resource will
know to send tasks to this endpoint.

### Step 4: Download the input data

The input data can be downloaded using the following command from the repository's root directory:

```
mkdir -p input/dataset1
wget https://zenodo.org/record/6368338/files/dataset_1_photons_1.hdf5 input/dataset1/
```

### Step 5: Train desired model(s).

Available models to train are: `GANv1`, `BNReLU`, `BNswish`, `BNswishHe`,
`BNLeakyReLU`, `noBN`, `SN`. The models to train in parallel can be selected
using the `--models` option (e.g., `--models GANv1 BNReLU BNswish`). If this flag is not provided, it will train all the models in parallel.

The `--endpoint-id` option lets us specify the endpoint ID to send our tasks to
and the `--example-run` flag lets us reduce the number of max iterations to 2 and the number of checkpoint intervals to one.

An example command may look like (trains all models):

`python training/train.py -i input/dataset1/dataset_1_photons_1.hdf5 -o output/dataset1/v1 -c config/config_GANv1.json --endpoint-id <endpoint_id> --example-run`