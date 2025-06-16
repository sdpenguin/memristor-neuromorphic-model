import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
import time

from argparse import Namespace
from tqdm import tqdm

# Note that N, g_low, g_high have been specified above
from memristor_neuromorphic_model.model import evolution
from memristor_neuromorphic_model.resistance_equations import tio2_linear_conductance
from memristor_neuromorphic_model.voltage_pulses import get_ascending_pulses
from memristor_neuromorphic_model.volatility_equations import joule_heating, voltage_induced
from memristor_neuromorphic_model.rate_equations import boltzmann
from memristor_neuromorphic_model.sampling import get_sampled


plt.rcParams['figure.figsize'] = (20, 8)
plt.rcParams['font.size'] = 40
sns.set_style('darkgrid')
plt.rcParams['lines.linewidth'] = 4

results_directory = 'figs/discrete_comparison'

from memristor_neuromorphic_model.param_defaults import *

V_offset = 0.05
V_a = 0.40049

discrete_timesteps = [1e-2, 1e-1, 1, 2]
N_master = 200000
n_thresh_master = N_master//2
Ns = [5000, 10000, 100000]
g_step_master = 1e-7 # This should be scaled according to the value of N
g_parallel_master = 1e-7

joule_heating_parameters = Namespace(Cth=3.84e-14, Rth=4e4)
voltage_induced_parameters = Namespace(factor=10, time_constant=10)

trials = 100

total_time = 150
timestep = 0.1 # Do not sample in the case of the event based model
sampling_time = 0.1

Vs, Vs_times = get_ascending_pulses(-0.3, -0.02, 0.5, 10)

initial_resistance = 30e3

np.random.seed(42)

def main():
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    resistance_equation = tio2_linear_conductance(g_step_master, g_parallel_master, n_thresh_master)
    all_samples_master = []
    simulation_runtimes = []
    for i in tqdm(range(trials)):
        start_state = resistance_equation(initial_resistance, inverse=True)
        start_time = time.time()
        ns, Rs, Ts, Vs_measured, Vs_transients, times = evolution(timestep, Vs, Vs_times, v_factor, N_master, start_state, T, total_time, V_a, V_offset, boltzmann, resistance_equation, [joule_heating, voltage_induced], [joule_heating_parameters, voltage_induced_parameters], rate_factor=rate_factor, quiet=True, simulation_no=i)
        simulation_runtimes.append(time.time() - start_time)
        sampled = get_sampled([Rs,], [times,], sampling_time, total_time)[0]
        all_samples_master.append(sampled)
    all_samples_master = np.array(all_samples_master, dtype=np.float64)
    means_master = np.mean(all_samples_master, axis=0)
    stds_master = np.std(all_samples_master, axis=0)
    np.save(os.path.join(results_directory, 'simulation_runtimes.npy'), simulation_runtimes)

    sampled_times = np.arange(0, total_time + sampling_time, sampling_time)

    plt.figure()
    plt.plot(sampled_times, means_master, label='Ground Truth', color='green')
    plt.fill_between(sampled_times, means_master - stds_master, means_master + stds_master, color='green', alpha=0.3)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Resistance ($\Omega$)')
    plt.savefig(os.path.join(results_directory, 'ground_truth_simulation.pdf'), bbox_inches='tight')

    plt.clf()
    simulation_runtimes = [[] for _ in range(len(Ns))]
    all_all_samples = []
    for colour_i, N_curr in enumerate(Ns):
        g_step_curr = g_step_master*(N_master/N_curr)
        # g_step_curr = g_step_master/rate_factor_curr
        g_parallel_curr = g_parallel_master*(N_master/N_curr)
        n_thresh_curr = N_curr//2
        # rate_factor_curr = rate_factor*(N_master/N_master)
        resistance_equation = tio2_linear_conductance(g_step_curr, g_parallel_curr, n_thresh_curr)
        all_samples = []
        for i in tqdm(range(trials)):
            start_state = resistance_equation(initial_resistance, inverse=True)
            start_time = time.time()
            ns, Rs, Ts, Vs_measured, Vs_transients, times = evolution(timestep, Vs, Vs_times, v_factor, N_curr, start_state, T, total_time, V_a, V_offset, boltzmann, resistance_equation, [joule_heating, voltage_induced], [joule_heating_parameters, voltage_induced_parameters], rate_factor=rate_factor, quiet=True, simulation_no=i)
            simulation_runtimes[colour_i].append(time.time() - start_time)
            sampled = get_sampled([Rs,], [times,], sampling_time, total_time)[0]
            all_samples.append(sampled)
        all_samples = np.array(all_samples, dtype=np.float64)
        all_all_samples.append(all_samples)
        means = np.mean(all_samples, axis=0)
        stds = np.std(all_samples, axis=0)
        plt.plot(sampled_times, means, label='Event (N = {})'.format(N_curr), color=(colour_i*0.3, 0.0, 1-colour_i*0.3))
        plt.fill_between(sampled_times, means - stds, means + stds, color=(colour_i*0.3, 0.0, 1-colour_i*0.3), alpha=0.3)
    np.save(os.path.join(results_directory, 'simulation_runtimes_event.npy'), simulation_runtimes)
    plt.plot(sampled_times, means_master, label='Event (N = {})'.format(N_master), color='green')
    plt.fill_between(sampled_times, means_master - stds_master, means_master + stds_master, color='green', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Resistance ($\Omega$)')
    plt.ylim([0, 40e3])
    plt.legend(fontsize=35)
    plt.savefig(os.path.join(results_directory, 'simulation_N_comparison.pdf'), bbox_inches='tight')
    all_all_samples = np.array(all_all_samples)
    errors_event = np.mean((all_all_samples - all_samples_master)**2, axis=(1,2))
    np.save(os.path.join(results_directory, 'errors_event.npy'), errors_event)

    plt.clf()
    simulation_runtimes = [[] for _ in range(len(discrete_timesteps))]
    all_all_samples = []
    for colour_i, discrete_timestep in enumerate(discrete_timesteps):
        resistance_equation = tio2_linear_conductance(g_step_master, g_parallel_master, n_thresh_master)
        all_samples = []
        for i in tqdm(range(trials)):
            start_state = resistance_equation(initial_resistance, inverse=True)
            start_time = time.time()
            ns, Rs, Ts, Vs_measured, Vs_transients, times = evolution(discrete_timestep, Vs, Vs_times, v_factor, N_master, start_state, T, total_time, V_a, V_offset, boltzmann, resistance_equation, [joule_heating, voltage_induced], [joule_heating_parameters, voltage_induced_parameters], rate_factor=rate_factor, quiet=True, simulation_no=i, method='continuous_normal')
            simulation_runtimes[colour_i].append(time.time() - start_time)
            sampled = get_sampled([Rs,], [times,], sampling_time, total_time)[0]
            all_samples.append(sampled)
        all_samples = np.array(all_samples, dtype=np.float64)
        all_all_samples.append(all_samples)
        means = np.mean(all_samples, axis=0)
        stds = np.std(all_samples, axis=0)
        plt.plot(sampled_times, means, label='Discrete ($h$ = {})'.format(discrete_timestep), color=(colour_i*0.3, 0.0, 1-colour_i*0.3))
        plt.fill_between(sampled_times, means - stds, means + stds, color=(colour_i*0.3, 0.0, 1-colour_i*0.3), alpha=0.3)
    np.save(os.path.join(results_directory, 'simulation_runtimes_discrete.npy'), simulation_runtimes)
    plt.plot(sampled_times, means_master, label='Event-Based', color='green')
    plt.fill_between(sampled_times, means_master - stds_master, means_master + stds_master, color='green', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Resistance ($\Omega$)')
    plt.ylim([0, 40e3])
    plt.legend(fontsize=35)
    plt.savefig(os.path.join(results_directory, 'simulation_discrete_timestep_comparison.pdf'), bbox_inches='tight')
    all_all_samples = np.array(all_all_samples)
    errors_discrete = np.mean((all_all_samples - all_samples_master)**2, axis=(1,2))
    np.save(os.path.join(results_directory, 'errors_discrete.npy'), errors_discrete)

def plot_runtimes():
    runtimes_vanilla = np.load(os.path.join(results_directory, 'simulation_runtimes.npy'))
    mean_vanilla = np.mean(runtimes_vanilla)
    std_vanilla = np.std(runtimes_vanilla)
    runtimes_event = np.load(os.path.join(results_directory, 'simulation_runtimes_event.npy'))
    means_event = np.mean(runtimes_event, axis=1)
    stds_event = np.std(runtimes_event, axis=1)
    runtimes_discrete = np.load(os.path.join(results_directory, 'simulation_runtimes_discrete.npy'))
    means_discrete = np.mean(runtimes_discrete, axis=1)
    stds_discrete = np.std(runtimes_discrete, axis=1)
    errors_event = np.load(os.path.join(results_directory, 'errors_event.npy'))
    errors_discrete = np.load(os.path.join(results_directory, 'errors_discrete.npy'))
    # Add the vanilla runtime/error
    means_event = np.concatenate([[mean_vanilla,], means_event])
    errors_event = np.concatenate([[0,], errors_event])
    plt.clf()
    # plt.plot(means_event, errors_event, label='Event-based')
    plt.plot(means_discrete, errors_discrete, label='Discrete Timestep', marker='x', linestyle='--', markersize=40, linewidth=5)
    plt.scatter(mean_vanilla, [0], marker="X", s=1000, color='red', label='Event Based (N={})'.format(N_master))
    plt.xlabel('Runtime (s)')
    plt.ylabel('Mean Squared Error ($\Omega^2$)')
    plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(results_directory, 'runtimes_vs_errors.pdf'), bbox_inches='tight')

if __name__=='__main__':
    # main()
    plot_runtimes()
