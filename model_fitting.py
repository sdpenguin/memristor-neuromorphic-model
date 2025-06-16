import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from argparse import Namespace

from memristor_neuromorphic_model.model import evolution
from memristor_neuromorphic_model.resistance_equations import tio2_linear_conductance
from memristor_neuromorphic_model.rate_equations import boltzmann
from memristor_neuromorphic_model.volatility_equations import joule_heating
from memristor_neuromorphic_model.quantise import quantise, get_conductance_bins

# Plot settings
plt.rcParams['figure.figsize'] = (20, 8)
plt.rcParams['font.size'] = 30
sns.set_style('dark')

n_diff = 3.62426514

lower_index = 6500
upper_index = 9910

k_b = 1.38064e-23 # Boltzmann's Constant
q = 1.602176e-19 # The electronic charge

T = 300 # Temperature in Kelvin

n_thresh = 10000
N = 2*n_thresh
g_step = 1e-7
g_parallel = 1e-10

rate_factor = 1.0 # Leveraging the (limited) number of states available

total_time = 10000 # The total time to run the simulation for
timestep = 100 # The timestep for updating the temperature due to Joule heating (and heat dissipation)

joule_heating_parameters = Namespace(Cth=3.84e-14, Rth=4e4)

results_directory = 'figs/model_fitting'
quantised_data_directory = 'data/quantised_drift/'
drift_data_directory = 'data/drift/'

samples = 100 # 1000

np.random.seed(42) # Ensure that data is reproducible

def main():
    resistance_equation = tio2_linear_conductance(g_step, g_parallel, n_thresh)
    def linear_conductance(r, a):
        # Estimate the initial value of n through the model, then use this to estimate a linear difference in the number of "high"-state switches
        # Please note that g_high and g_low here refer to the high and low conductances, rather than the conductances corresponding to high and low resistive states
        n = resistance_equation(r, inverse=True)
        return resistance_equation(n-a) - r
    
    starting_resistances = resistance_equation(np.array([x for x in range(N+1-upper_index, N+1-lower_index, 10)]))[2:]

    plt.plot(starting_resistances[:], linear_conductance(starting_resistances[:], n_diff), label='Linear conductance model', color='black', linewidth=8)
    for i, V_offset in enumerate([0.5, 0.05, 0.01, 0.005]):
        # V_offset = 0.001 # Arbitrary value for V_offset
        print('V_offset: {}'.format(V_offset))
        V_a = k_b * T / q * - np.log(n_diff / (total_time * ((n_thresh)) * np.sinh(V_offset*q/2/k_b/T)*2))
        print("V_a:", V_a)
        n_eq = N / (np.exp(q*V_offset/k_b/T) + 1)
        print("N_eq:", n_eq)

        changes = []
        for resistance in starting_resistances[1:]:
            n = resistance_equation(resistance, inverse=True)
            changes.append(resistance_equation(n + (-(n)*1/(np.exp((V_a*q - V_offset*q/2)/k_b/(T))) + (N-n)*1/(np.exp((V_a*q + V_offset*q/2)/k_b/(T))))*total_time) - resistance)
        plt.plot(starting_resistances[1:], changes, label=('$V_{off}$ = ' + str(V_offset) + ", $V_a$ = " + str(round(V_a, 2)) + ", $N_{eq} = $" + str(round(n_eq, 2))), linestyle=('--' if i%2==0 else 'dotted'), linewidth=8, alpha=0.8)
        plt.legend()
        # plt.show()
        plt.xlabel('$R(t)$')
        plt.ylabel('$R(t + \Delta T) - R(t)$')
        plt.grid()
    if not os.path.exists(results_directory):
        os.mkdir(results_directory)
    plt.savefig(os.path.join(results_directory, 'fitting_to_linear_conductance.pdf'))

    V_offset = 0.05
    V_a = k_b * T / q * - np.log(n_diff / (total_time * ((n_thresh)) * np.sinh(V_offset*q/2/k_b/T)*2))

    simulated_stats = []
    for n_initial in tqdm(resistance_equation(starting_resistances, inverse=True)):
        final_rs = []
        for i in range(samples): # Number of simulations per starting resistance
            ns, Rs, Ts, Vs_measured, Vs_transients, times = evolution(timestep, [], [], 1.0, N, n_initial, T, total_time, V_a, V_offset, boltzmann, resistance_equation, [joule_heating], [joule_heating_parameters], rate_factor=rate_factor, simulation_no=i, quiet=True)
            final_rs.append(Rs[-1])
        simulated_stats.append([np.mean(final_rs), np.std(final_rs)])
    simulated_stats = np.array(simulated_stats) # Align with truncated_resistances

    # Load quantised pairs to generate statistics
    if not os.path.exists(quantised_data_directory):
        os.mkdir(quantised_data_directory)
    if not os.path.exists(os.path.join(quantised_data_directory, 'pairs_{}.npy'.format(int(total_time//10)))):
        print('Quantising drift data according to timestep of {}'.format(total_time))
        quantise(drift_data_directory, N, g_step, g_parallel, n_thresh, interval=total_time//10, base_directory=quantised_data_directory, initial_only=False) # We divide by 10, because of the 10 second sampling rate of the data collection process
    pairs = np.load(os.path.join(quantised_data_directory, 'pairs_{}.npy').format(int(total_time//10)))

    conductances, conductance_bin_boundaries = get_conductance_bins(N, g_step, g_parallel, n_thresh)
    resistances = np.flip(1 / conductances)
    truncated_resistances = resistances[lower_index:upper_index]

    stats_array = np.empty((0, 2))
    min_points = 2 # The minimum number of points necessary for stat calculations
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

    window = 5
    means_convolved = np.convolve(stats_array_differences, [1/window for _ in range(window)], mode='valid')
    stds_convolved = np.convolve(stats_array[:, 1][stats_array[:, 0] != None], [1/window for _ in range(window)], mode='valid')

    plt.clf()
    # plt.plot(truncated_resistances[stats_array[:, 0] != None][window//2:-window//2+1], means_convolved, label='Data statistics (moving average)')
    plt.plot(truncated_resistances[stats_array[:, 0] != None][window//2:-window//2+1], means_convolved, label='Data statistics (moving average)', color='blue')
    plt.fill_between(np.asarray(truncated_resistances[stats_array[:, 0] != None][window//2:-window//2+1], dtype=np.float64), np.asarray(means_convolved - stds_convolved, dtype=np.float64), np.asarray(means_convolved + stds_convolved, dtype=np.float64), color='blue', alpha=0.3)
    # plt.errorbar(starting_resistances[:], simulated_stats[:, 0][:] - starting_resistances[:], yerr=simulated_stats[:, 1], label='Simulation statistics', linewidth=3)
    plt.fill_between(np.asarray(starting_resistances[:], dtype=np.float64), np.asarray(simulated_stats[:, 0][:] - starting_resistances[:] - simulated_stats[:, 1], dtype=np.float64), np.asarray(simulated_stats[:, 0][:] - starting_resistances[:] + simulated_stats[:, 1], dtype=np.float64), color='green', alpha=0.3)
    plt.plot(starting_resistances[:], linear_conductance(starting_resistances[:], n_diff), label='Linear conductance model', linewidth=3, alpha=0.8, color='red')
    plt.plot(starting_resistances[:], simulated_stats[:, 0][:] - starting_resistances[:], label='Simulation statistics', linewidth=3, color='green')
    plt.legend()
    plt.xlabel('$R(t)$')
    plt.ylabel('$R(t + \Delta T) - R(t)$')
    plt.savefig(os.path.join(results_directory, 'model_change_in_resistance.pdf'))

if __name__=='__main__':
    main()
