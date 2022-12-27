import numpy as np
import h5py, os
import pandas as pd
import matplotlib.pyplot as plt
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
    return np.log10(kin / kin_min) / np.log10(kin_max / kin_min)

def get_kin(input_file):
    particle = input_file.split('/')[-1].split('_')[-2][:-1]
    photon_file = h5py.File(f'{input_file}', 'r')
    mass = particle_mass(particle)
    energies = photon_file['incident_energies'][:]
    kin = np.sqrt( np.square(energies) + np.square(mass) ) - mass
    return kin, particle

def plot_frame(categories, xlabel, ylabel, label_pos='left', add_summary_panel=True):
    if add_summary_panel:
        categories = np.append(categories, 0)
    length = len(categories)
    width = int(np.ceil(np.sqrt(length)))
    height = int(np.ceil(length / width))
    fig, axes = plt.subplots(nrows=width, ncols=height, figsize=(4*width, 4*height))
    for index, energy in enumerate(categories):
        ax = axes[(index) // 4, (index) % 4]
        ax.tick_params(axis="both", which="major", width=1, length=6, labelsize=10, direction="in")
        ax.tick_params(axis="both", which="minor", width=0.5, length=3, labelsize=10, direction="in")
        ax.minorticks_on()
        if index == length-1 and add_summary_panel:
            ax.axis("off")
        else:
            if isinstance(energy, str):
                energy_legend = energy
            else:
                energy_legend = (str(round(energy / 1000, 1)) + " GeV") if energy > 1024 else (str(energy) + " MeV")
            if label_pos == 'left':
                ax.text(0.02, 0.98, energy_legend, transform=ax.transAxes, va="top", ha="left", fontsize=20)
            elif label_pos == 'right':
                ax.text(0.98, 0.98, energy_legend, transform=ax.transAxes, va="top", ha="right", fontsize=20)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
    return fig, axes.flatten()

def get_energies(input_file):
    photon_file = h5py.File(f'{input_file}', 'r')
    energies = photon_file['incident_energies'][:]
    if np.all(np.mod(energies, 1) == 0):
        energies = energies.astype(int)
    else:
        raise ValueError
    return energies

def get_counts(input_file):
    energies = get_energies(input_file)
    categories = np.unique(energies)

    counts = [np.count_nonzero(energies == c) for c in categories]
    return categories, counts

def split_energy(input_file, vector):
    '''
        Input: h5file and vector with length of nevents
        Output: a list of vectors splitted by energies, and the energies
    '''
    energies = get_energies(input_file)
    categories, counts = get_counts(input_file)

    joint_array = np.concatenate([energies, vector], axis=1)
    joint_array = joint_array[joint_array[:, 0].argsort()]
    vector_list = np.split(joint_array[:,1:], np.unique(joint_array[:, 0], return_index=True)[1][1:])
    return categories, vector_list

def plot_energy_vox(categories, E_vox_list, label_list=None, kin_list=None, nvox='all', output=None, logx=True, particle=None, draw_ref=True, xlabel='E'):
    particle_latex_name = {
        'photon': r"$\gamma$",
        'photons': r"$\gamma$",
        'pion': r"$\pi$",
        'pions': r"$\pi$",
    }
    np.seterr(divide = 'ignore', invalid='ignore')
    GeV = 1 # no energy correction
    if nvox == 'all': loop = ['all']
    else: loop = range(nvox)
    for vox_i in loop:
        fig, axes = plot_frame(categories, xlabel=xlabel, ylabel="Events", label_pos='right')
        colors = ['k', 'r']
        for index, energy in enumerate(categories):
            ax = axes[index]
            for icurve, E_list in enumerate(E_vox_list):
                if nvox == 'all':
                    if kin_list is not None:
                        x = E_list[index][:,:] / kin_list[index]
                    else:
                        x = E_list[index][:,:]
                else:
                    if kin_list is not None:
                        x = E_list[index][:,vox_i] / kin_list[index]
                    else:
                        x = E_list[index][:,vox_i]
                if logx:
                    x = - np.log10(x.flatten() / GeV)
                else:
                    x = x.flatten() / GeV
                if icurve == 0:
                    low, high = np.nanmin(x[x != -np.inf]), np.nanmax(x[x != np.inf])
                ax.hist(x, range=(low,high), bins=40, histtype='step', color=colors[icurve], label=None if label_list is None else label_list[icurve]) # [GeV]
            if draw_ref is None:
                if logx:
                    ax.axvline(x=-3, ymax=0.5, color='orange', ls='--', label='MeV')
                    ax.axvline(x=-6, ymax=0.5, color='b', ls='--', label='keV')
                else:
                    ax.axvline(x=1, ymax=0.5, color='orange', ls='--', label='MeV')
                    ax.axvline(x=1E-3, ymax=0.5, color='b', ls='--', label='keV')
            ax.ticklabel_format(style='plain')
            ax.ticklabel_format(useOffset=False, style='plain')
            ax.set_yscale('log')
            if logx:
                ax.legend(loc='center left')
            else:
                ax.legend(loc='center right')

        ax = axes[-1]
        ax.text(0.5, 0.5, particle_latex_name[particle], transform=ax.transAxes, fontsize=20)
        plt.tight_layout()
        if output is not None:
            plot_name = output.format(vox_i=vox_i)
            os.makedirs(os.path.dirname(plot_name), exist_ok=True)
            plt.savefig(plot_name)
            print('\033[92m[INFO] Save to\033[0m', plot_name)

