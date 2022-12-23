import os
from argparse import ArgumentParser
from glob import glob
import pandas as pd
from pdb import set_trace

def main(args):
    files = glob(f'{args.input}/*/*')

    def get_info(path):
        task = path.split('/')[-2].split('_')
        job = path.split('/')[-1].split('_')
        return tuple([task[0], task[1], job[0], job[2], job[3]])

    pid = {
        'pions': 211,
        'photons': 22,
        'electrons': 11,
        'protons': 2212,
    }

    results = {'particle': [], 'eta': [], 'model': [], 'hp': [], 'chi2': [], 'iter': []}
    for ifile in files:
        model, hp, particle, eta_min, eta_max = get_info(ifile)
        csv_file = f'{ifile}/selected/chi2.csv'
        if not os.path.exists(csv_file):
            print('\033[92m[INFO] Unfinished job\033[0m', f'{csv_file}')
            continue
        df = pd.read_csv(f'{csv_file}')
        chi2 = float(df['All'])
        iteration = int(df['ckpt'])

        results['particle'].append(particle)
        results['eta'].append(int(eta_min))
        results['model'].append(model)
        results['hp'].append(hp)
        results['chi2'].append(chi2)
        results['iter'].append(iteration)
    pd.DataFrame.from_dict(results).sort_values(by=['particle', 'eta', 'model', 'hp']).to_csv(f'{args.input}/results.csv', index=False)
    print('\033[92m[INFO] Save to\033[0m', f'{args.input}/results.csv')

if __name__ == '__main__':

    """Get arguments from command line."""
    parser = ArgumentParser(description="\033[92mConfig for training.\033[0m")
    parser.add_argument('-i', '--input', type=str, required=False, default='../output/dataset1/v1/', help='Training h5 file name (default: %(default)s)')

    args = parser.parse_args()
    main(args)
