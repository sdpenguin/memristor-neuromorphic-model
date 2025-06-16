''' This script fits a stretched exponential model, and a linear drift model, to the drift dataset.
    It also extends the evaluation of the fit of the linear model by testing its abilities to interpolate and to extrapolate (performance on unseen data by fitting to a selected subset (either random or for a given range of resistances)).
'''

import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns

from tqdm import tqdm
from scipy.optimize import curve_fit

from memristor_neuromorphic_model.resistance_equations import tio2_linear_conductance
from memristor_neuromorphic_model.quantise import quantise, get_conductance_bins

# Plot settings
plt.rcParams['figure.figsize'] = (20, 8)
plt.rcParams['font.size'] = 30
sns.set_style('dark')

# Parameters
n_thresh = 10000
N = 2*n_thresh
g_step = 1e-7
g_parallel = 1e-10

timestep = 10000 # The time in seconds between pairs of values

# Indices for he resistances/conductances (removing high and low resistance bins from consideration)
lower_index = 6500
upper_index = 9910
random_seed = 12

# Directories
quantised_data_directory = 'data/quantised_drift'
drift_data_directory = 'data/drift'
results_directory = 'figs/readout_fitting'

# Models
def stretched_exponential_difference(r, tau, beta, gamma):
    # Use the stretched exponential model to predict the difference in the resistance after a set amount of time
    return gamma + (r-gamma)*np.exp(-(timestep/tau)**beta) - r

rram_equation = tio2_linear_conductance(g_step, g_parallel, n_thresh)
def linear_conductance(r, a):
    # Estimate the initial value of n through the model, then use this to estimate a linear difference in the number of "high"-state switches
    # Please note that g_high and g_low here refer to the high and low conductances, rather than the conductances corresponding to high and low resistive states
    n = rram_equation(r, inverse=True)
    return rram_equation(n-a) - r

def linear_resistance(r, m, c):
    # Use the stretched exponential model to predict the difference in the resistance after a set amount of time
    return m*r + c - r

def resistance_difference_to_conductance(initial_resistance, difference):
    final_resistance = initial_resistance + difference
    final_conductance = 1/final_resistance
    initial_conductance = 1/initial_resistance
    return final_conductance - initial_conductance

def parameter_fitting(pairs, train_indices, test_indices, stats_array, truncated_resistances, plot_suffix=''):
    '''
        pairs: The input values formatted as conductances in an array of (initial, final) format with the array length being equal to the number of pairs.
    '''
    np.random.default_rng(random_seed)

    params, _ = curve_fit(stretched_exponential_difference, 1/pairs[train_indices][:,0], 1/pairs[train_indices][:,1]-1/pairs[train_indices][:,0], bounds=[[0, -10, 0], [1e7, 10, 1e7]], p0=[1e5, 1, 1e6])
    print("Parameters for stretched exponential difference model: {}. Change for initial resistance of 50kohms: {}".format((params[0], params[1], params[2]), stretched_exponential_difference(50000, params[0], params[1], params[2])))
    params2, _ = curve_fit(linear_conductance, 1/pairs[train_indices][:, 0], 1/pairs[train_indices][:, 1] - 1/pairs[train_indices][:, 0], bounds=[[0], [1e3]], p0=[10])
    print("Parameters for linear conductance model: {}. Change for initial resistance of 50kohms: {}".format(params2, linear_conductance(50000, params2[0])))
    params3, _ = curve_fit(linear_resistance, 1/pairs[train_indices][:, 0], 1/pairs[train_indices][:, 1] - 1/pairs[train_indices][:, 0], bounds=[[0, -1e7], [10, 1e7]], p0=[1, 0])
    print("Parameters for linear resistance model: {}. Change for initial resistance of 50kohms: {}".format(params2, linear_resistance(50000, params3[0], params3[1])))

    # N.B. pairs are conductances, 1/pairs are resistances

    # Compute the train errors
    train_y = 1/pairs[train_indices][:,1]-1/pairs[train_indices][:,0] # The real differences in resistance
    
    stretched_y_train = stretched_exponential_difference(1/pairs[train_indices][:, 0], params[0], params[1], params[2])
    linear_conductance_y_train = linear_conductance(1/pairs[train_indices][:, 0], params2[0])
    linear_resistance_y_train = linear_resistance(1/pairs[train_indices][:, 0], params3[0], params3[1])

    def get_error(indices, y, y_pred):
        err = np.mean(np.abs(y - y_pred))
        return err # We return the absolute, rather than squared, to avoid excessive contributions from outliers
        # ratio_err = np.mean(np.abs((y/(1/pairs[indices][:, 0])) - (y_pred/(1/pairs[indices][:, 0]))))
        # err = np.mean((resistance_difference_to_conductance(1/pairs[indices][:, 0], y)-resistance_difference_to_conductance(1/pairs[indices][:, 0], y_pred))**2)
        # ratio_err = np.mean((resistance_difference_to_conductance(1/pairs[indices][:, 0], y)/(pairs[indices][:, 0])-resistance_difference_to_conductance(1/pairs[indices][:, 0], y_pred)/(pairs[indices][:, 0]))**2)
        # return err, ratio_err

    stretched_train_error = get_error(train_indices, train_y, stretched_y_train)
    linear_conductance_train_error = get_error(train_indices, train_y, linear_conductance_y_train)
    linear_resistance_train_error = get_error(train_indices, train_y, linear_resistance_y_train)
    
    # Compute the test errors
    test_y = 1/pairs[test_indices][:,1]-1/pairs[test_indices][:,0]
    
    stretched_y_test = stretched_exponential_difference(1/pairs[test_indices][:, 0], params[0], params[1], params[2])
    linear_conductance_y_test = linear_conductance(1/pairs[test_indices][:, 0], params2[0])
    linear_resistance_y_test = linear_resistance(1/pairs[test_indices][:, 0], params3[0], params3[1])
    
    stretched_test_error = get_error(test_indices, test_y, stretched_y_test)
    linear_conductance_test_error = get_error(test_indices, test_y, linear_conductance_y_test)
    linear_resistance_test_error = get_error(test_indices, test_y, linear_resistance_y_test)

    stats_array_differences = stats_array[:, 0][stats_array[:, 0] != None] - truncated_resistances[stats_array[:, 0] != None]
    plt.clf()
    plt.scatter(truncated_resistances[stats_array[:, 0] != None], stats_array_differences, alpha=1.0, label='Data', color='orange')
    for window_size, col in zip([10, 50], ['blue', 'green', 'red']):
        window = np.ones(window_size)/window_size
        avgs = np.convolve(stats_array_differences, window, mode='valid')
        plt.plot(truncated_resistances[stats_array[:, 0] != None][window_size//2:-window_size//2+1], avgs, label='Moving Average ({})'.format(window_size), color=col, linewidth=5)
    # plt.plot(truncated_resistances[stats_array[:, 0] != None], stretched_exponential_difference(truncated_resistances[stats_array[:, 0] != None], params[0], params[1], params[2]), label='Streched Exponential Model', color='yellow', linewidth=5)
    plt.plot(truncated_resistances[stats_array[:, 0] != None], linear_conductance(truncated_resistances[stats_array[:, 0] != None], params2[0]), label='Linear Conductance Model', color='red', linewidth=5)
    plt.ylim([0, 4000])
    plt.ylabel('Mean Difference in Resistance')
    plt.xlabel('Initial Resistance')
    plt.legend()
    plt.savefig(os.path.join(results_directory, 'mean_change_in_resistance_{}.pdf'.format(plot_suffix)))
    plt.close()
    print('Stretched train/test error: {} : {}'.format(stretched_train_error, stretched_test_error))
    print('Linear conductance train/test error: {} : {}'.format(linear_conductance_train_error, linear_conductance_test_error))
    print('Linear resistance train/test error: {} : {}'.format(linear_resistance_train_error, linear_resistance_test_error))
    print('\n')

def main():
    if not os.path.exists(quantised_data_directory):
        os.mkdir(quantised_data_directory)
    if not os.path.exists(os.path.join(quantised_data_directory, 'pairs_{}.npy'.format(int(timestep//10)))):
        print('Quantising drift data according to timestep of {}'.format(timestep))
        quantise(drift_data_directory, N, g_step, g_parallel, n_thresh, interval=timestep//10, base_directory=quantised_data_directory, initial_only=False) # We divide by 10, because of the 10 second sampling rate of the data collection process
    pairs = np.load(os.path.join(quantised_data_directory, 'pairs_{}.npy').format(int(timestep//10)))

    conductances, conductance_bin_boundaries = get_conductance_bins(N, g_step, g_parallel, n_thresh)
    resistances = np.flip(1 / conductances)
    resistance_bin_boundaries = np.flip(1 / conductance_bin_boundaries)

    truncated_resistances = resistances[lower_index:upper_index]
    truncated_conductances = conductances[N+1-n_thresh-upper_index:N+1-n_thresh-lower_index]

    stats_array = np.empty((0, 2))
    min_points = 1 # The minimum number of points necessary for stat calculations
    for i in range(lower_index, upper_index):
        indices = np.argwhere(1/pairs[:, 0] == resistances[i])
        if len(indices) >= min_points:
            vals = (1/pairs[:, 1])[indices]
            mean = np.mean(vals)
            std = np.std(vals)
        else:
            mean = None
            std = None
        stats_array = np.concatenate([stats_array, [[mean, std]]], axis=0)

    stats_array_differences = stats_array[:, 0][stats_array[:, 0] != None] - truncated_resistances[stats_array[:, 0] != None]
    plt.clf()
    plt.scatter(truncated_resistances[stats_array[:, 0] != None], stats_array_differences, alpha=0.5, label='Data', color='orange')
    for window_size, col in zip([5, 50], ['blue', 'green', 'red']):
        window = np.ones(window_size)/window_size
        avgs = np.convolve(stats_array_differences, window, mode='valid')
        plt.plot(truncated_resistances[stats_array[:, 0] != None][window_size//2:-window_size//2+1], avgs, label='Moving Average ({})'.format(window_size), color=col)
    plt.ylim([0, 4000])
    plt.ylabel('Mean Difference in Resistance')
    plt.xlabel('Initial Resistance')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(results_directory, 'drift_data_moving_averages_timestep{}.pdf'.format(timestep)))

    # Take a random subset of indices
    np.random.seed(random_seed)
    train_indices_random = np.random.choice(len(pairs), math.ceil(len(pairs)*0.8), replace=False)
    test_indices_random = np.array([x for x in np.arange(len(pairs)) if x not in train_indices_random])

    # Take the central 20% of indices, leaving 40% on either side
    train_indices_interpolate = np.concatenate((np.arange(math.ceil(len(pairs)*0.4)), np.arange(math.ceil(len(pairs)*0.6), len(pairs))))
    test_indices_interpolate = np.arange(math.ceil(len(pairs)*0.4), math.ceil(len(pairs)*0.6))

    # Take the last 10% of indices, leaving 90% to come before
    # train_indices_extrapolate = np.arange(math.ceil(len(pairs)*0.2), len(pairs))
    # test_indices_extrapolate = np.arange(0, math.ceil(len(pairs)*0.2))

    parameter_fitting(pairs, np.arange(len(pairs)), np.arange(len(pairs)), stats_array, truncated_resistances, plot_suffix='all')
    parameter_fitting(pairs, train_indices_random, test_indices_random, stats_array, truncated_resistances, plot_suffix='random')
    parameter_fitting(pairs, train_indices_interpolate, test_indices_interpolate, stats_array, truncated_resistances, plot_suffix='interpolate')
    # parameter_fitting(pairs, train_indices_extrapolate, test_indices_extrapolate, stats_array, truncated_resistances, plot_suffix='extrapolate')

if __name__ == '__main__':
    main()
