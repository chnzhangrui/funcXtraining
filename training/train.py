# use code from https://github.com/CaloChallenge/homepage/blob/main/code/HighLevelFeatures.ipynb
from argparse import ArgumentParser
from HighLevelFeatures import HighLevelFeatures
import numpy as np
import h5py, os, json
import matplotlib.pyplot as plt
from model import WGANGP
from pdb import set_trace

def particle_mass(particle=None):
    if 'photon' in particle or particle == 22:
        mass = 0
    elif 'electron' in particle or particle == 11:
        mass = 0.512
    elif 'pion' in particle or particle == 211:
        mass = 139.6
    elif 'proton' in particle or particle == 2212:
        mass = 938.27
    return mass

def kin_to_label(kin):
    kin_min = np.min(kin)
    kin_max = np.max(kin)
    return np.log(kin / kin_min) / np.log(kin_max / kin_min)


def main(args):
    # creating instance of HighLevelFeatures class to handle geometry based on binning file
    input_file = args.input_file
    particle = input_file.split('/')[-1].split('_')[-2][:-1]
    #hlf = HighLevelFeatures(particle, filename=f'{os.path.dirname(input_file)}/binning_dataset_1_{particle}s.xml')
    print('\033[92m[INFO] Run\033[0m', particle, input_file)
    
    # loading the .hdf5 datasets
    photon_file = h5py.File(f'{input_file}', 'r')
    
    mass = particle_mass(particle)
    kin = np.sqrt( np.square(photon_file['incident_energies'][:]) + np.square(mass) ) - mass
    label_kin = kin_to_label(kin)
    
    X_train = photon_file['showers'][:] / kin
    
    if 'photon' in particle:
        hp_config = {
            'model': 'BNswish',
            'G_size': 1,
            'D_size': 1,
            'G_lr': 1E-4,
            'D_lr': 1E-4,
            'G_beta1': 0.5,
            'G_beta1': 0.5,
            'batchsize': 1024,
            'dgratio': 8,
            'latent_dim': 50,
            'lam': 3,
            'conditional_dim': label_kin.shape[1],
            'generatorLayers': [50, 100, 200],
            'nvoxels': X_train.shape[1],
            'use_bias': True,
        }
    else: # pion
        hp_config = {
            'model': 'noBN',
            'G_size': 1,
            'D_size': 1,
            'G_lr': 1E-4,
            'D_lr': 1E-4,
            'G_beta1': 0.5,
            'G_beta1': 0.5,
            'batchsize': 1024,
            'dgratio': 5,
            'latent_dim': 50,
            'lam': 10,
            'conditional_dim': label_kin.shape[1],
            'generatorLayers': [50, 100, 200],
            'discriminatorLayers': [800, 400, 200],
            'nvoxels': X_train.shape[1],
            'use_bias': True,
        }
    if args.config:
        from quickstats.utils.common_utils import combine_dict
        hp_config = combine_dict(hp_config, json.load(open(args.config, 'r')))

    job_config = {
        'particle': particle+'s',
        'eta_slice': '20_25',
        'checkpoint_interval': 1000 if not args.debug else 10,
        'output': args.output_path,
        'max_iter': 1E6 if not args.debug else 100,
        'cache': False,
    }
    
    wgan = WGANGP(job_config=job_config, hp_config=hp_config, logger=__file__)
    wgan.train(X_train, label_kin)

if __name__ == '__main__':

    """Get arguments from command line."""
    parser = ArgumentParser(description="\033[92mConfig for training.\033[0m")
    parser.add_argument('-i', '--input_file', type=str, required=False, default='', help='Training h5 file name (default: %(default)s)')
    parser.add_argument('-o', '--output_path', type=str, required=True, default='../output/dataset1/v1', help='Training h5 file path (default: %(default)s)')
    parser.add_argument('-c', '--config', type=str, required=False, default=None, help='External config file (default: %(default)s)')
    parser.add_argument('--debug', required=False, action='store_true', help='Debug mode (default: %(default)s)')

    args = parser.parse_args()
    main(args)
