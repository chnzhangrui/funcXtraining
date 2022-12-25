import json, time
from argparse import ArgumentParser
from HighLevelFeatures import HighLevelFeatures
import numpy as np
import h5py, os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from model import WGANGP
from train import particle_mass, kin_to_label
from quickstats.utils.common_utils import execute_multi_tasks
from quickstats.maths.numerics import get_bins_given_edges
from itertools import repeat
from glob import glob
from common import *
from pdb import set_trace

def get_E_truth(input_file, mode='total'):
    # creating instance of HighLevelFeatures class to handle geometry based on binning file
    particle = input_file.split('/')[-1].split('_')[-2][:-1]
    photon_file = h5py.File(f'{input_file}', 'r')

    hlf = HighLevelFeatures(particle, filename=f'{os.path.dirname(input_file)}/binning_dataset_1_{particle}s.xml')
    hlf.CalculateFeatures(photon_file['showers'][:])

    if mode == 'total':
        E_tot = hlf.GetEtot()
    elif mode == 'voxel':
        E_vox = photon_file['showers'][:]

    if mode == 'total':
        vector = E_tot.reshape(-1,1)
    elif mode == 'voxel':
        vector = E_vox

    categories, vector_list = split_energy(input_file, vector)
    return categories, vector_list

def get_E_gan(model_i, input_file, train_path, eta_slice, mode='total'):

    particle = input_file.split('/')[-1].split('_')[-2][:-1]
    photon_file = h5py.File(f'{input_file}', 'r')
    mass = particle_mass(particle)
    energies = photon_file['incident_energies'][:]
    kin = np.sqrt( np.square(energies) + np.square(mass) ) - mass
    label_kin = kin_to_label(kin)

    config = json.load(open(os.path.join(train_path, f'{particle}s_eta_{eta_slice}', 'train', 'config.json')))

    wgan = WGANGP(job_config=config['job_config'], hp_config=config['hp_config'], logger=__file__)
    E_vox = wgan.predict(model_i=model_i, labels=label_kin)
    E_vox *= kin
    E_tot = np.array(E_vox).sum(axis=-1)

    if mode == 'total':
        vector = E_tot.reshape(-1,1)
    elif mode == 'voxel':
        vector = E_vox

    categories, vector_list = split_energy(input_file, vector)
    return categories, vector_list

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

def plot_Egan(args, categories, Etot_list, Egan_list, model_i):
    particle = args.input_file.split('/')[-1].split('_')[-2][:-1]
    fig, axes = plot_frame(categories, xlabel="Energy [GeV]", ylabel="Probability")
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
        # Convert energy to GeV
        GeV = 1000
        etot = Etot_list[index] / GeV
        egan = Egan_list[index] / GeV

        ax = axes[index]
        median = np.median(etot)
        high = median + min([np.absolute(np.max(etot) - median), np.absolute(np.quantile(etot, q=1-0.05) - median) * 2, np.absolute(np.quantile(etot, q=1-0.16) - median) * 10])
        low  = median - min([np.absolute(np.min(etot) - median), np.absolute(np.quantile(etot, q=0.05) - median) * 2, np.absolute(np.quantile(etot, q=1-0.16) - median) * 10])
        bins = get_bins_given_edges(low, high, nbins, 3)
        y_tot, x_tot, _ = ax.hist(np.clip(etot, bins[0], bins[-1]), bins=bins, label='G4', histtype='step', density=False, color='k', linestyle='-', alpha=0.8, linewidth=2.)
        y_gan, x_gan, _ = ax.hist(np.clip(egan, bins[0], bins[-1]), bins=bins, label='GAN', histtype='step', density=False, color='r', linestyle='--', alpha=0.8, linewidth=2.)
        y_tot_err = np.sqrt(y_tot)
        y_gan_err = np.sqrt(y_gan)
        chi2, ndf = chi2testWW(y_tot, y_tot_err, y_gan, y_gan_err)
        chi2_tot += chi2
        ndf_tot += ndf
        energy_legend = (str(round(energy / 1000, 1)) + " GeV") if energy > 1024 else (str(energy) + " MeV")
        results.append((f'{energy} MeV', chi2/ndf))
        ax.text(0.02, 0.88, "$\chi^2$:{:.1f}".format(chi2 / ndf), transform=ax.transAxes, va="top", ha="left", fontsize=20)

    handles, labels = ax.get_legend_handles_labels()

    chi2_o_ndf = chi2_tot/ndf_tot
    results.insert(1, (f'All', chi2_o_ndf))
    ax = axes[-1]
    ax.legend(handles=handles[:2], labels=["Geant4", "GAN"], loc="upper left", frameon=False, fontsize=20)
    eta_min, eta_max = tuple(args.eta_slice.split('_'))
    ax.text(0.0, 0.1, particle_latex_name[particle]+ ", " + str("{:.2f}".format(int(eta_min) / 100, 2)) + r"$<|\eta|<$" + str("{:.2f}\n".format((int(eta_max)) / 100, 2)) + "Iter: {}\n$\chi^2$/NDF = {:.0f}/{:.0f}\n= {:.1f}".format(int(model_i)*1000, chi2_tot, ndf_tot, chi2_o_ndf), transform=ax.transAxes, fontsize=20)

    plt.tight_layout()
    plot_name = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', os.path.splitext(os.path.basename(__file__))[0], f'plot_{particle}_{args.eta_slice}_{model_i}.pdf')
    plt.savefig(plot_name)
    fig.clear()
    plt.close(fig)
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

    categories, Etot_list = get_E_truth(args.input_file)
    truth_time = time.time() - start_time
    start_time = time.time()
    categories, Egan_list = get_E_gan(model_i=model_i, input_file=args.input_file, train_path=args.train_path, eta_slice=args.eta_slice)
    gan_time = time.time() - start_time
    start_time = time.time()
    chi2_results = plot_Egan(args, categories, Etot_list, Egan_list, model_i)
    plot_time = time.time() - start_time
    print('\033[92m[INFO] Evaluate result\033[0m', 'model', model_i, 'chi2', f'{chi2_results["All"]:.2f}', f'time (truth) {truth_time:.1f}s (gan) {gan_time:.1f}s (plot) {plot_time:.1f}s')
    return chi2_results

def plot_energy_vox(categories, E_vox_list, label_list=None, nvox='all', output=None):
    np.seterr(divide = 'ignore', invalid='ignore')
    GeV = 1 # no energy correction
    if nvox == 'all': loop = ['all']
    else: loop = range(nvox)
    for vox_i in loop:
        fig, axes = plot_frame(categories, xlabel=f"Log(Energy of voxel {vox_i} [MeV])", ylabel="Events")
        for index, energy in enumerate(categories):
            ax = axes[index]
            for i, E_list in enumerate(E_vox_list):
                if nvox == 'all':
                    x = np.log10(E_list[index][:,:].flatten() / GeV)
                else:
                    x = np.log10(E_list[index][:,vox_i].flatten() / GeV)
                if i == 0:
                    low, high = np.nanmin(x[x != -np.inf]), np.max(x)
                ax.hist(x, range=(low,high), bins=40, histtype='step', label=None if label_list is None else label_list[i]) # [GeV]
            ax.axvline(x=-3, ymax=0.5, color='r', ls='--', label='MeV')
            ax.axvline(x=-6, ymax=0.5, color='b', ls='--', label='keV')
            ax.ticklabel_format(style='plain')
            ax.ticklabel_format(useOffset=False, style='plain')
            ax.set_yscale('log')
            ax.legend(loc='center left')

        ax = axes[-1]
        plt.tight_layout()
        if output is not None:
            plot_name = output.format(vox_i=vox_i)
            os.makedirs(os.path.dirname(plot_name), exist_ok=True)
            plt.savefig(plot_name)
            print('\033[92m[INFO] Save to\033[0m', plot_name)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def best_ckpt(args, df):
    particle = args.input_file.split('/')[-1].split('_')[-2][:-1]
    best_folder = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', 'selected')
    chi_name = os.path.join(best_folder, 'chi2.pdf')
    if not os.path.exists(chi_name):
        os.makedirs(best_folder, exist_ok=True)
        particle_latex_name = {
            'photon': r"$\gamma$",
            'photons': r"$\gamma$",
            'pion': r"$\pi$",
            'pions': r"$\pi$",
        }
        best_x = int(df[df['All'] == df['All'].min()]['ckpt'] * 1000)
        best_y = float(df['All'].min())
        x = df['ckpt'] * 1000
        y = df['All']

        categories = [int(i.replace(' MeV', '')) for i in df if 'MeV' in i]
        chi2_list = [df[f'{c} MeV'].values for c in categories]
        fig, axes = plot_frame(categories, xlabel="Iterations", ylabel="$\chi^{2}$/NDF")
        for index, energy in enumerate(categories):
            ax = axes[index]
            ax.scatter(x, chi2_list[index], c="k", edgecolors="k", alpha=0.9)
            best_x_i = int(df[df[f'{energy} MeV'] == df[f'{energy} MeV'].min()]['ckpt'] * 1000)
            best_y_i = df[f'{energy} MeV'].min()
            best_y_j = float(df[df['ckpt']==int(best_x/1000)][f'{energy} MeV'])
            ax.scatter(best_x_i, best_y_i, c="orange")
            ax.scatter(best_x, best_y_j, c="r")
            ax.text(0.98, 0.98, "Iter {}\n$\chi^2$ = {:.1f}\nSel. iter {}\n$\chi^2$ = {:.1f}".format(best_x_i, best_y_i, best_x, best_y_j), transform=ax.transAxes, va="top", ha="right", fontsize=10, bbox=dict(facecolor='w', alpha=0.8, edgecolor='w'))
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(max(0, ymin), min(50, ymax))

        ax = axes[-1]
        ax.axis("on")
        ax.scatter(x, y, c="k", edgecolors="k", alpha=0.9)
        ax.scatter(best_x, best_y, c="r")
        eta_min, eta_max = tuple(args.eta_slice.split('_'))
        ax.text(0.98, 0.98, particle_latex_name[particle] + "\n" + str("{:.2f}".format(int(eta_min) / 100, 2)) + r"$<|\eta|<$" + str("{:.2f}\n".format((int(eta_min) + 5) / 100, 2)) + "Iter {}\n$\chi^2$ = {:.1f}".format(best_x, best_y), transform=ax.transAxes, va="top", ha="right", fontsize=15, bbox=dict(facecolor='w', alpha=0.8, edgecolor='w'))
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(max(0, ymin), min(50, ymax))
        plt.tight_layout()
        plt.savefig(chi_name)
        print('\033[92m[INFO] Save to\033[0m', chi_name)
        fig.clear()
        plt.close(fig)

    csv_name = os.path.join(best_folder, 'chi2.csv')
    best_df = df[df['All'] == df['All'].min()]
    if not os.path.exists(csv_name):
        best_df.to_csv(csv_name, index=False)

        models = glob(os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', 'checkpoints', f'model-{int(best_df["ckpt"])}*'))
        for model in models:
            os.system(f'cp {model} {best_folder}')  
        plot_name = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', os.path.splitext(os.path.basename(__file__))[0], f'plot_{particle}_{args.eta_slice}_{int(best_df["ckpt"])}.pdf')
        os.system(f'cp {plot_name} {best_folder}')  

    vox_name = os.path.join(best_folder, f'mask_{particle}_{args.eta_slice}_{int(best_df["ckpt"])}_all.pdf')
    if not os.path.exists(vox_name):
        # Plot 'masking' distribution; 'masking' means to remove voxel energies below a threshold of 1keV or 1MeV
        categories, E_gan_list = get_E_gan(model_i=int(best_df["ckpt"]), input_file=args.input_file, train_path=args.train_path, eta_slice=args.eta_slice, mode='voxel')
        categories, E_tru_list = get_E_truth(args.input_file, mode='voxel')
        plot_energy_vox(categories, [E_tru_list, E_gan_list], label_list=['Geant4', 'GAN'], nvox='all', output=vox_name)


def main(args):
    particle = args.input_file.split('/')[-1].split('_')[-2][:-1]
    models = glob(os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', 'checkpoints', 'model-*.index'))
    if not models:
        print('\033[91m[ERROR] No model is found at\033[0m', os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}', 'checkpoints', 'model-*.index'))
        return
    else:
        print('\033[92m[INFO] Evaluate\033[0m', particle, args.input_file, f'| {len(models)} models')

    models = [int(m.split('/')[-1].split('-')[-1].split('.')[0]) for m in models]
    models.sort(reverse = True)

    arguments = (repeat(args), models)
    results = execute_multi_tasks(plot_model_i, *arguments, parallel=-1)
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
