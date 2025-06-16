import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from argparse import ArgumentParser
from .resistance_equations import tio2_linear_conductance

def list_csv_files(directory):
    csv_files = []
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(foldername, filename)
                csv_files.append(file_path)
    return csv_files

def get_conductance_bins(N, g_step=1e-10, g_parallel=1e-7, n_thresh=10000):
    resistance_equation = tio2_linear_conductance(g_step, g_parallel, n_thresh) # "Low" and "high" resistance i.e. high and low conductance, respectively
    ns = np.arange(n_thresh, N+1)
    conductance_values = 1/resistance_equation(ns)
    # Bins: -infinity, lowest_value + 1/2 bin width, ..., highest_value - 1/2 bin width, ~ +infinity
    step_size = g_step # (conductance_values[-1] - conductance_values[0]) / N # The bin size is linear in the conductance, rather than the resistance 
    conductance_bin_boundaries = np.concatenate([[1e-100], (conductance_values + (0.5*step_size))[:-1], [1e100]]) # Convert the upper bin limit to capture all very high resistances.
    # Note that if quantised in this way, the number of bins remains equal to N+1
    return conductance_values, conductance_bin_boundaries

def quantise(data_path, N, g_step, g_parallel, n_thresh=10000, interval=1, base_directory='.', initial_only=False, quantise=True):
    conductance_values, conductance_bin_boundaries = get_conductance_bins(N, g_step, g_parallel, n_thresh)
    # The boundaries should be ~ -infinity, lowest_value + 1/2 bin width, ..., highest_value - 1/2 bin width, ~ +infinity
    # i.e. we should add half the bin width and exclude the highest value, replacing it with infinity
    csvs = list_csv_files(data_path)
    if len(csvs) == 0:
        print('NOTE: No CSVs were found for the specified path...')
    all = np.empty((0, 2))
    for filename in tqdm(csvs):
        x = pd.read_csv(filename)
        if quantise:
            conductance_bin_indices = np.digitize(1/x['r'], conductance_bin_boundaries)
            quantized_x_conductance = conductance_values[conductance_bin_indices]
        else:
            quantized_x_conductance = 1/x['r']
        if initial_only:
            pairs = np.array([[quantized_x_conductance[0], quantized_x_conductance[interval]],])
        else:
            pairs = np.stack([quantized_x_conductance[::interval][:-1], quantized_x_conductance[::interval][1:]], axis=-1)
        all = np.concatenate([all, pairs], axis=0)
        np.save(os.path.join(base_directory, 'pairs_{}.npy'.format(interval)), all)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('--N', type=int, default=1e6)
    parser.add_argument('--g-low', type=float, default=1e-06)
    parser.add_argument('--g-high', type=float, default=1e-02)
    parser.add_argument('--interval', type=int, default=1, help='The interval between reads for the output pairs.')
    parsed = parser.parse_args()
    quantise(parsed.data_path, parsed.N, parsed.g_low, parsed.g_high, parsed.interval)
