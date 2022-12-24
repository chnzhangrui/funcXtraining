import numpy as np
import h5py, os
import pandas as pd
import matplotlib.pyplot as plt
from pdb import set_trace

def plot_frame(categories, xlabel, ylabel, summary_panel=True):
    if summary_panel:
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
        if index == length-1 and summary_panel:
            energy_legend = 'All energies'
            ax.axis("off")
        else:
            energy_legend = (str(round(energy / 1000, 1)) + " GeV") if energy > 1024 else (str(energy) + " MeV")
        ax.text(0.02, 0.98, energy_legend, transform=ax.transAxes, va="top", ha="left", fontsize=20)
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

