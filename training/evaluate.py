import json
from argparse import ArgumentParser
from HighLevelFeatures import HighLevelFeatures
import numpy as np
import h5py, os
import pandas as pd
import matplotlib.pyplot as plt
from model import WGANGP
from train import particle_mass, kin_to_label
from quickstats.utils.common_utils import execute_multi_tasks
from itertools import repeat
from glob import glob
from pdb import set_trace

def get_truth_E(args):
    # creating instance of HighLevelFeatures class to handle geometry based on binning file
    particle = args.input_file.split('/')[-1].split('_')[-2][:-1]
    photon_file = h5py.File(f'{args.input_file}', 'r')

    hlf = HighLevelFeatures(particle, filename=f'{os.path.dirname(args.input_file)}/binning_dataset_1_{particle}s.xml')
    hlf.CalculateFeatures(photon_file['showers'][:])

    Etot = hlf.GetEtot()
    energies = photon_file['incident_energies'][:]
    if np.all(np.mod(energies, 1) == 0):
        energies = energies.astype(int)
    else:
        raise ValueError
    categories = np.unique(energies)
    joint_array = np.concatenate([energies, Etot.reshape(-1,1)], axis=1)
    joint_array = joint_array[joint_array[:, 0].argsort()]
    Etot_list = np.split(joint_array[:,1], np.unique(joint_array[:, 0], return_index=True)[1][1:])
    return categories, Etot_list

def get_gan_E(args, model_i):
    particle = args.input_file.split('/')[-1].split('_')[-2][:-1]
    photon_file = h5py.File(f'{args.input_file}', 'r')
    mass = particle_mass(particle)
    energies = photon_file['incident_energies'][:]
    kin = np.sqrt( np.square(energies) + np.square(mass) ) - mass
    label_kin = kin_to_label(kin)

    config = json.load(open(os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', 'train', 'config.json')))

    wgan = WGANGP(job_config=config['job_config'], hp_config=config['hp_config'], logger=__file__)
    Egan = wgan.predict(model_i=model_i, labels=label_kin)
    Egan *= kin
    Egan = np.array(Egan).sum(axis=-1)

    if np.all(np.mod(energies, 1) == 0):
        energies = energies.astype(int)
    else:
        raise ValueError
    categories = np.unique(energies)
    joint_array = np.concatenate([energies, Egan.reshape(-1,1)], axis=1)
    joint_array = joint_array[joint_array[:, 0].argsort()]
    Egan_list = np.split(joint_array[:,1], np.unique(joint_array[:, 0], return_index=True)[1][1:])

    return categories, Egan_list

def chi2testWW(y1, y1_err, y2, y2_err):
    zeros = (y1 == 0) * (y2 == 0)
    ndf = y1.size - 1 - zeros.sum()
    if zeros.sum():
        y1 = np.delete(y1, np.where(zeros))
        y2 = np.delete(y2, np.where(zeros))
        y1_err = np.delete(y1_err, np.where(zeros))
        y2_err = np.delete(y2_err, np.where(zeros))

    W1, W2 = y1.sum(), y2.sum()
    delta = W1 * y2 - W2 * y1
    sigma = W1 * W1 * y2_err * y2_err + W2 * W2 * y1_err * y1_err
    chi2 = (delta * delta / sigma).sum()
    return chi2, ndf



def plot_E(args, categories, Etot_list, Egan_list, model_i):
    particle = args.input_file.split('/')[-1].split('_')[-2][:-1]
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
    particle_latex_name = {
        'photon': r"$\gamma$",
        'photons': r"$\gamma$",
        'pion': r"$\pi$",
        'pions': r"$\pi$",
    }
    results = [('ckpt', model_i)]
    dict([(f'{energy} MeV', 0) for energy in categories])


    bins = 30
    ndf_tot = chi2_tot = 0
    for index, energy in enumerate(categories):
        ax = axes[(index) // 4, (index) % 4]
        y_tot, x_tot, _ = plt.hist(Etot_list[index], bins=bins, label='G4', density=False, color='k', linestyle='-', alpha=0.8, linewidth=2.)
        y_gan, x_gan, _ = plt.hist(Egan_list[index], bins=x_tot, label='GAN', density=False, color='r', linestyle='--', alpha=0.8, linewidth=2.)
        y_tot_err = np.sqrt(y_tot)
        y_gan_err = np.sqrt(y_gan)
        chi2, ndf = chi2testWW(y_tot, y_tot_err, y_gan, y_gan_err)
        chi2_tot += chi2
        ndf_tot += ndf
        energy_legend = (str(round(energy / 1000, 1)) + " GeV") if energy > 1024 else (str(energy) + " MeV")
        results.append((f'{energy} MeV', chi2/ndf))
        ax.tick_params(axis="both", direction="in")
        ax.text(0.01, 0.99, "$\chi^2$:{:.1f}".format(chi2 / ndf), transform=ax.transAxes, va="top", ha="left", fontsize=20)
        ax.text(0.99, 0.99, energy_legend, transform=ax.transAxes, va="top", ha="right", fontsize=20)
        ax.set_xlabel("Energy [MeV]")
        ax.set_ylabel("Probability")


    chi2_o_ndf = chi2_tot/ndf_tot
    results.insert(0, (f'All', chi2_o_ndf))
    ax = axes[(index + 1) // 4, (index + 1) % 4]
    ax.axis("off")
    ax.text(0.0, 0.85, "ATLAS", weight=1000, transform=ax.transAxes, fontsize=20)
    ax.text(0.0, 0.7, "Simulation Internal", transform=ax.transAxes, fontsize=20)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:2], labels=["Geant4", "GAN"], loc="center", frameon=False, fontsize=20)
    eta_min, eta_max = tuple(args.eta_slice.split('_'))
    ax.text(0.0, 0.0, particle_latex_name[particle]+ ", " + str("{:.2f}".format(int(eta_min) / 100, 2)) + r"$<|\eta|<$" + str("{:.2f}\n".format((int(eta_max)) / 100, 2)) + "Iter: {}\n$\chi^2$/NDF = {:.0f}/{:.0f}\n= {:.1f}".format(int(model_i), chi2_tot, ndf_tot, chi2_o_ndf), transform=ax.transAxes, fontsize=20)

    plt.tight_layout()
    plot_name = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', os.path.splitext(os.path.basename(__file__))[0], f'plot_{particle}_{args.eta_slice}_{model_i}.pdf')
    plt.savefig(plot_name)
    return dict(results)

def plot_model_i(args, model_i):
    particle = args.input_file.split('/')[-1].split('_')[-2][:-1]
    df_name = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', os.path.splitext(os.path.basename(__file__))[0], f'chi2.csv')
    plot_name = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', os.path.splitext(os.path.basename(__file__))[0], f'plot_{particle}_{args.eta_slice}_{model_i}.pdf')
    if os.path.exists(df_name) and os.path.exists(plot_name):
        df = pd.read_csv(df_name)
        if model_i in df['ckpt'].values:
            print('\033[92m[INFO] Cache\033[0m', 'model', model_i)
            return df[df['ckpt'] == model_i].to_dict(orient='records')[0]
    print('\033[92m[INFO] Evaluate\033[0m', 'model', model_i)

    categories, Etot_list = get_truth_E(args)
    categories, Egan_list = get_gan_E(args, model_i=model_i)
    chi2_results = plot_E(args, categories, Etot_list, Egan_list, model_i)
    return chi2_results

def main(args):
    particle = args.input_file.split('/')[-1].split('_')[-2][:-1]
    models = glob(os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', 'checkpoints', 'model-*.index'))
    print('\033[92m[INFO] Evaluate\033[0m', particle, args.input_file, f'| {len(models)} models')
    models = [m.split('/')[-1].split('-')[-1].split('.')[0] for m in models]
    arguments = (repeat(args), models)
    results = execute_multi_tasks(plot_model_i, *arguments, parallel=-1)
    df = pd.DataFrame(results)
    df_name = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', os.path.splitext(os.path.basename(__file__))[0], f'chi2.csv')
    df.to_csv(df_name, index=False)
    print('\033[92m[INFO] Save to\033[0m', df_name)
    
if __name__ == '__main__':

    """Get arguments from command line."""
    parser = ArgumentParser(description="\033[92mConfig for training.\033[0m")
    parser.add_argument('-i', '--input_file', type=str, required=False, default='', help='Training h5 file name (default: %(default)s)')
    parser.add_argument('-t', '--train_path', type=str, required=True, default='../output/dataset1/v1', help='--out_path from train.py (default: %(default)s)')
    parser.add_argument('-e', '--eta_slice', type=str, required=False, default='20_25', help='--out_path from train.py (default: %(default)s)')
    parser.add_argument('-p', '--parallel', type=int, required=False, default=-1, help='number of CPUs to use to parallelise evaluation, 0 for debug (default: %(default)s)')
    parser.add_argument('--debug', required=False, action='store_true', help='Debug mode (default: %(default)s)')

    args = parser.parse_args()
    main(args)
