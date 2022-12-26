# use code from https://github.com/CaloChallenge/homepage/blob/main/code/HighLevelFeatures.ipynb
from argparse import ArgumentParser
from HighLevelFeatures import HighLevelFeatures
import numpy as np
import h5py, os, json
import matplotlib.pyplot as plt
from model import WGANGP
from pdb import set_trace
from common import *
np.set_printoptions(suppress=True)

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
    return np.log10(kin / kin_min) / np.log10(kin_max / kin_min)

def apply_mask(mask, X_train, input_file):
    np.seterr(divide = 'ignore', invalid='ignore')
    event_energy_before = X_train.sum(axis=1)[:]

    # mask too low energy to zeros
    if isinstance(mask, (int, float)):
        X_train[X_train < (mask / 1000)] = 0
    elif isinstance(mask, dict):
        # X_train is un-sorted!
        energies = get_energies(input_file)
        for k,m in mask.items():
            X_train[np.logical_and(energies == k, X_train < (m / 1000))] = 0
    else:
        raise NotImplementedError

    # plot energy change before and after masking
    event_energy_after  = X_train.sum(axis=1)[:]
    event_energy = np.concatenate([event_energy_before.reshape(-1,1), event_energy_after.reshape(-1,1)], axis=1)

    categories, vector_list  = split_energy(input_file, event_energy)
    fig, axes = plot_frame(categories, xlabel="Rel. change in E total", ylabel="Events")
    for index, energy in enumerate(categories):
        ax = axes[index]
        before, after = vector_list[index][:,0], vector_list[index][:,1]
        x = 1 - np.divide(after, before, out=np.zeros_like(before), where=before!=0)
        if x.max() < 1E-4:
            high = 1E-4
        elif x.max() < 1E-3:
            high = 1E-3
        elif x.max() < 1E-2:
            high = 1E-2
        elif x.max() < 0.1:
            high = 0.1
        else:
            high = 1
        print(x.max(), energy, high)
        n, _, _ = ax.hist(x, bins=100, range=(0,high))
        ax.set_yscale('symlog')
        ax.set_ylim(bottom=0)
        if isinstance(mask, (int, float)):
            mask_legend = f'Mask {mask} keV\nMax {high}'
        elif isinstance(mask, dict):
            if mask[energy] < 1E3:
                mask_legend = f'Mask {mask[energy]} keV\nMax {high}'
            elif mask[energy] < 1E6:
                mask_legend = f'Mask {mask[energy]/1E3} MeV\nMax {high}'
            else:
                mask_legend = f'Mask {mask[energy]/1E6} GeV\nMax {high}'
        ax.text(0.98, 0.88, mask_legend, transform=ax.transAxes, va="top", ha="right", fontsize=15)
    ax = axes[-1]
    ax.axis("on")
    x = 1 - event_energy_after / event_energy_before
    if x.max() < 1E-4:
        high = 1E-4
    elif x.max() < 1E-3:
        high = 1E-3
    elif x.max() < 1E-2:
        high = 1E-2
    elif x.max() < 0.1:
        high = 0.1
    else:
        high = 1
    ax.hist(x, bins=100, range=(0,high))
    ax.set_yscale('symlog')
    ax.set_ylim(bottom=0)
    os.makedirs(args.output_path, exist_ok=True)
    particle = input_file.split('/')[-1].split('_')[-2][:-1]
    plt.savefig(os.path.join(args.output_path, f'mask_{particle}_{args.mask}keV.pdf'))
    print('\033[92m[INFO] Mask\033[0m', args.mask, mask, '[keV] for voxel energy')
        
    # return masked input
    return X_train

def main(args):
    # creating instance of HighLevelFeatures class to handle geometry based on binning file
    input_file = args.input_file
    particle = input_file.split('/')[-1].split('_')[-2][:-1]
    #hlf = HighLevelFeatures(particle, filename=f'{os.path.dirname(input_file)}/binning_dataset_1_{particle}s.xml')
    print('\033[92m[INFO] Run\033[0m', particle, input_file)
    
    # loading the .hdf5 datasets
    photon_file = h5py.File(f'{input_file}', 'r')
    
    mass = particle_mass(particle)
    energies = photon_file['incident_energies'][:]
    kin = np.sqrt( np.square(energies) + np.square(mass) ) - mass
    label_kin = kin_to_label(kin)
    
    X_train = photon_file['showers'][:]
    if args.mask is not None:
        if args.mask < 0:
            mask = list(np.unique(energies)/256 * abs(args.mask)) # E/256 * (-mask)
            mask = dict(zip(list(np.unique(energies)), mask))
        else:
            mask = args.mask
        X_train = apply_mask(mask, X_train, input_file)
    X_train /= kin

    if 'photon' in particle:
        hp_config = {
            'model': args.model if args.model else 'BNswish',
            'G_size': 1,
            'D_size': 1,
            'optimizer': 'adam',
            'G_lr': 1E-4,
            'D_lr': 1E-4,
            'G_beta1': 0.5,
            'G_beta1': 0.5,
            'batchsize': 1024,
            'datasize': X_train.shape[0],
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
            'model': args.model if args.model else 'noBN',
            'G_size': 1,
            'D_size': 1,
            'optimizer': 'adam',
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
    parser.add_argument('-m', '--model', type=str, required=False, default=None, help='Model name (default: %(default)s)')
    parser.add_argument('--mask', type=float, required=False, default=None, help='Mask low noisy voxels in keV (default: %(default)s)')
    parser.add_argument('--debug', required=False, action='store_true', help='Debug mode (default: %(default)s)')

    args = parser.parse_args()
    main(args)
