import json, time
from argparse import ArgumentParser
from HighLevelFeatures import HighLevelFeatures
import numpy as np
import h5py, os
import pandas as pd
import matplotlib.pyplot as plt
from model import WGANGP
from train import particle_mass, kin_to_label
from quickstats.utils.common_utils import execute_multi_tasks
from quickstats.maths.numerics import get_bins_given_edges
from itertools import repeat
from glob import glob
from pdb import set_trace

def get_truth_E(args):
    # creating instance of HighLevelFeatures class to handle geometry based on binning file
    particle = args.input_file.split('/')[-1].split('_')[-2][:-1]
    photon_file = h5py.File(f'{args.input_file}', 'r')

    hlf = HighLevelFeatures(particle, filename=f'{os.path.dirname(args.input_file)}/binning_dataset_1_{particle}s.xml')
    hlf.CalculateFeatures(photon_file['showers'][:])

    GeV = 1000
    Etot = hlf.GetEtot() / GeV
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
    GeV = 1000
    Egan = np.array(Egan).sum(axis=-1) / GeV

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


    nbins = 30
    ndf_tot = chi2_tot = 0
    for index, energy in enumerate(categories):
        ax = axes[(index) // 4, (index) % 4]
        median = np.median(Etot_list[index])
        high = median + min([np.absolute(np.max(Etot_list[index]) - median), np.absolute(np.quantile(Etot_list[index], q=1-0.05) - median) * 2, np.absolute(np.quantile(Etot_list[index], q=1-0.16) - median) * 10])
        low  = median - min([np.absolute(np.min(Etot_list[index]) - median), np.absolute(np.quantile(Etot_list[index], q=0.05) - median) * 2, np.absolute(np.quantile(Etot_list[index], q=1-0.16) - median) * 10])
        bins = get_bins_given_edges(low, high, nbins, 3)
        y_tot, x_tot, _ = ax.hist(np.clip(Etot_list[index], bins[0], bins[-1]), bins=bins, label='G4', histtype='step', density=False, color='k', linestyle='-', alpha=0.8, linewidth=2.)
        y_gan, x_gan, _ = ax.hist(np.clip(Egan_list[index], bins[0], bins[-1]), bins=bins, label='GAN', histtype='step', density=False, color='r', linestyle='--', alpha=0.8, linewidth=2.)
        y_tot_err = np.sqrt(y_tot)
        y_gan_err = np.sqrt(y_gan)
        chi2, ndf = chi2testWW(y_tot, y_tot_err, y_gan, y_gan_err)
        chi2_tot += chi2
        ndf_tot += ndf
        energy_legend = (str(round(energy / 1000, 1)) + " GeV") if energy > 1024 else (str(energy) + " MeV")
        results.append((f'{energy} MeV', chi2/ndf))
        ax.tick_params(axis="both", direction="in")
        ax.text(0.02, 0.98, energy_legend, transform=ax.transAxes, va="top", ha="left", fontsize=20)
        ax.text(0.02, 0.88, "$\chi^2$:{:.1f}".format(chi2 / ndf), transform=ax.transAxes, va="top", ha="left", fontsize=20)
        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel("Probability")

    handles, labels = ax.get_legend_handles_labels()

    chi2_o_ndf = chi2_tot/ndf_tot
    results.insert(1, (f'All', chi2_o_ndf))
    ax = axes[(index + 1) // 4, (index + 1) % 4]
    ax.axis("off")
    ax.legend(handles=handles[:2], labels=["Geant4", "GAN"], loc="upper left", frameon=False, fontsize=20)
    #ax.text(0.0, 0.85, "ATLAS", weight=1000, transform=ax.transAxes, fontsize=20)
    #ax.text(0.0, 0.7, "Simulation Internal", transform=ax.transAxes, fontsize=20)
    eta_min, eta_max = tuple(args.eta_slice.split('_'))
    ax.text(0.0, 0.1, particle_latex_name[particle]+ ", " + str("{:.2f}".format(int(eta_min) / 100, 2)) + r"$<|\eta|<$" + str("{:.2f}\n".format((int(eta_max)) / 100, 2)) + "Iter: {}\n$\chi^2$/NDF = {:.0f}/{:.0f}\n= {:.1f}".format(int(model_i), chi2_tot, ndf_tot, chi2_o_ndf), transform=ax.transAxes, fontsize=20)

    plt.tight_layout()
    plot_name = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', os.path.splitext(os.path.basename(__file__))[0], f'plot_{particle}_{args.eta_slice}_{model_i}.pdf')
    plt.savefig(plot_name)
    return dict(results)

def plot_model_i(args, model_i):
    start_time = time.time()
    particle = args.input_file.split('/')[-1].split('_')[-2][:-1]
    df_name = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', os.path.splitext(os.path.basename(__file__))[0], f'chi2.csv')
    plot_name = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', os.path.splitext(os.path.basename(__file__))[0], f'plot_{particle}_{args.eta_slice}_{model_i}.pdf')
    if os.path.exists(df_name) and os.path.exists(plot_name):
        df = pd.read_csv(df_name)
        if model_i in df['ckpt'].values:
            chi2_results = df[df['ckpt'] == model_i].to_dict(orient='records')[0]
            print('\033[92m[INFO] Cache\033[0m', 'model', model_i, 'chi2', chi2_results['All'])
            return chi2_results

    categories, Etot_list = get_truth_E(args)
    categories, Egan_list = get_gan_E(args, model_i=model_i)
    chi2_results = plot_E(args, categories, Etot_list, Egan_list, model_i)
    end_time = time.time()
    print('\033[92m[INFO] Evaluate result\033[0m', 'model', model_i, 'chi2', chi2_results['All'], f'time {end_time-start_time:.1f}s')
    return chi2_results

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def best_ckpt(args, df):
    particle = args.input_file.split('/')[-1].split('_')[-2][:-1]
    best_folder = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', 'selected')
    chi_name = os.path.join(best_folder, 'chi2.pdf')
    if os.path.exists(chi_name) and os.path.exists(best_folder):
        return

    os.makedirs(best_folder, exist_ok=True)
    particle_latex_name = {
        'photon': r"$\gamma$",
        'photons': r"$\gamma$",
        'pion': r"$\pi$",
        'pions': r"$\pi$",
    }
    best_x = df[df['All'] == df['All'].min()]['ckpt'] * 1000
    best_y = df['All'].min()
    x = df['ckpt'] * 1000
    y = df['All']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    plt.scatter(x, y, c="k")
    plt.scatter(best_x, best_y, c="r")
    ax.set_xlabel("Iterations", fontsize=20)
    ax.set_ylabel("$\chi^{2}$/NDF", fontsize=20)
    ax.tick_params(axis="both", which="major", width=2, length=8, labelsize=20, direction="in")
    ax.tick_params(axis="both", which="minor", width=2, length=5, labelsize=20, direction="in")
    ax.minorticks_on()
    #ax.text(0.40, 0.92, "ATLAS", weight=1000, fontsize=20, transform=plt.gca().transAxes)
    #ax.text(0.55, 0.92, "Simulation Internal", fontsize=20, transform=plt.gca().transAxes)
    eta_min, eta_max = tuple(args.eta_slice.split('_'))
    ax.text(0.40, 0.74, particle_latex_name[particle] + ", " + str("{:.2f}".format(int(eta_min) / 100, 2)) + r"$<|\eta|<$" + str("{:.2f}\n".format((int(eta_min) + 5) / 100, 2)) + "Best Epoch: {}, $\chi^2$/NDF = {:.1f}".format(int(best_x), best_y), transform=plt.gca().transAxes, fontsize=20)
    plt.tight_layout()
    plt.savefig(chi_name)

    csv_name = os.path.join(best_folder, 'chi2.csv')
    best_df = df[df['All'] == df['All'].min()]
    best_df.to_csv(csv_name, index=False)

    models = glob(os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', 'checkpoints', f'model-{int(best_df["ckpt"])}*'))
    for model in models:
        os.system(f'cp {model} {best_folder}')  
    plot_name = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', os.path.splitext(os.path.basename(__file__))[0], f'plot_{particle}_{args.eta_slice}_{int(best_df["ckpt"])}.pdf')
    os.system(f'cp {plot_name} {best_folder}')  


def main(args):
    particle = args.input_file.split('/')[-1].split('_')[-2][:-1]
    models = glob(os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', 'checkpoints', 'model-*.index'))
    print('\033[92m[INFO] Evaluate\033[0m', particle, args.input_file, f'| {len(models)} models')

    models = [int(m.split('/')[-1].split('-')[-1].split('.')[0]) for m in models]
    models.sort()

    arguments = (repeat(args), models)
    results = execute_multi_tasks(plot_model_i, *arguments, parallel=int(os.cpu_count()/2))
    df = pd.DataFrame(results).sort_values(by=['ckpt'])

    df_name = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', os.path.splitext(os.path.basename(__file__))[0], f'chi2.csv')
    df.to_csv(df_name, index=False)
    print('\033[92m[INFO] Save to\033[0m', df_name)
    best_ckpt(args, df)
    
if __name__ == '__main__':

    """Get arguments from command line."""
    parser = ArgumentParser(description="\033[92mConfig for training.\033[0m")
    parser.add_argument('-i', '--input_file', type=str, required=False, default='', help='Training h5 file name (default: %(default)s)')
    parser.add_argument('-t', '--train_path', type=str, required=True, default='../output/dataset1/v1', help='--out_path from train.py (default: %(default)s)')
    parser.add_argument('-e', '--eta_slice', type=str, required=False, default='20_25', help='--out_path from train.py (default: %(default)s)')
    parser.add_argument('--debug', required=False, action='store_true', help='Debug mode (default: %(default)s)')

    args = parser.parse_args()
    main(args)