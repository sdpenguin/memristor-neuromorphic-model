import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns

from argparse import Namespace

from memristor_neuromorphic_model.model import evolution
from memristor_neuromorphic_model.resistance_equations import tio2_linear_conductance
from memristor_neuromorphic_model.voltage_pulses import get_alternating_pulses, get_ascending_pulses, get_descending_pulses
from memristor_neuromorphic_model.rate_equations import boltzmann
from memristor_neuromorphic_model.volatility_equations import joule_heating, voltage_induced

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 40
sns.set_style('darkgrid')
plt.rcParams['lines.linewidth'] = 4

results_directory = 'figs/model_simulation'

from memristor_neuromorphic_model.param_defaults import *

n_thresh = 10000
N = 2*n_thresh
g_step = 1e-7
g_parallel = 1e-10

total_time = 10000 # The total time to run the simulation for
timestep = 0.1 # The discrete timestep (for updating volatility parameters)
# Please note that for the purposes of the drift simulation, the temperature will not change, so we set this to a high value

joule_heating_parameters = Namespace(Cth=3.84e-14, Rth=4e4)
voltage_induced_parameters = Namespace(factor=10, time_constant=10)

resistance_equation = tio2_linear_conductance(g_step, g_parallel, n_thresh)

V_offset = 0.05
V_a = 0.40049

np.random.seed(42) # Ensure consistent figure generation

def hysteresis():
    plt.clf()
    plt.figure(figsize=(8, 4))
    plt.tight_layout()
    Vs_times = np.arange(500) / 50
    Vs = 0.5*np.sin(Vs_times/np.pi*2)
    plt.plot(Vs_times, Vs)
    plt.xlabel('t (s)')
    plt.ylabel('V(t)')
    plt.savefig(os.path.join(results_directory, 'hysteresis_voltage.pdf'), bbox_inches='tight')

    plt.clf()
    plt.figure(figsize=(8, 4))
    # plt.tight_layout()
    start_resistance = 5e3
    # for start_resistance in np.arange(10e3, 100e3+1, 10e3):
    start_state = resistance_equation(start_resistance, inverse=True)
    print('Starting resistance ', resistance_equation(start_state), 'Starting state: ', start_state)
    ns, Rs, Ts, Vs_measured, Vs_transients, times = evolution(timestep, Vs, Vs_times, v_factor, N, start_state, T, 10, V_a, V_offset, boltzmann, resistance_equation, [joule_heating, voltage_induced], [joule_heating_parameters, voltage_induced_parameters], rate_factor=rate_factor, quiet=False)

    plt.plot(np.array(Vs_measured), np.array(Vs_measured)/np.array(Rs), marker='x')
    plt.xlabel('v(t)')
    plt.ylabel('i(t)')
    plt.savefig(os.path.join(results_directory, 'hysteresis.pdf'), bbox_inches='tight')

def plot_experiment(Vs, Vs_times, figure_name, save_legend=False):
    plt.clf()
    # Note: we make ax2 first so that lines plotted on ax1 appear over those of ax2
    fig, ax2 = plt.subplots()
    plt.tight_layout()
    ax1 = ax2.twinx()
    for i, start_resistance in enumerate(np.arange(10e3, 100e3+1, 10e3)):
        start_state = resistance_equation(start_resistance, inverse=True)
        print('Starting resistance ', resistance_equation(start_state), 'Starting state: ', start_state)
        ns, Rs, Ts, Vs_measured, Vs_transients, times = evolution(timestep, Vs, Vs_times, v_factor, N, start_state, T, total_time, V_a, V_offset, boltzmann, resistance_equation, [joule_heating, voltage_induced], [joule_heating_parameters, voltage_induced_parameters], rate_factor=rate_factor, quiet=True, simulation_no=i)
        ax1.set_xlabel('$t$ (s)')
        ax1.set_ylabel('$R(t)$ ($\Omega$)')
        ax1.plot(times, Rs, color=(start_resistance/100e3, 0.0, ((100e3-start_resistance)/100e3)), label='{:.0f}k$\Omega$'.format(start_resistance/1e3))
    ax1.set_ylim([0, 150e3])
    ax2.set_ylabel('$V(t)$ (V)')
    ax2.plot(times, Vs_measured, color='green', label='V(t)')
    ax2.set_xlabel('Time (s)')
    if save_legend:
        legend = fig.legend()
        # --- Extract handles/labels from figure (if not already stored) ---
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        # Combine them
        handles = handles1 + handles2
        labels = labels1 + labels2
        # --- Create a new figure for the standalone legend ---
        fig_legend = plt.figure(figsize=(3, 2))
        fig_legend.legend(handles, labels, loc='center', frameon=False)
        # --- Save as PDF ---
        fig_legend.savefig(os.path.join(results_directory, 'legend.pdf'), bbox_inches='tight', dpi=300)
        plt.close(fig_legend)
        legend.remove()
    fig.savefig(os.path.join(results_directory, figure_name), bbox_inches='tight')

def pulse_experiments():
    plt.clf()
    Vs, Vs_times = get_descending_pulses(-0.2, -0.01, 0.1, 100*10)
    plot_experiment(Vs, Vs_times, 'descending_negative.pdf', save_legend=True)
    
    Vs, Vs_times = get_ascending_pulses(-0.2, -0.01, 0.1, 100*10)
    plot_experiment(Vs, Vs_times, 'ascending_negative.pdf')

    Vs, Vs_times = get_ascending_pulses(0.2, 0.01, 0.1, 100*10)
    plot_experiment(Vs, Vs_times, 'ascending_positive.pdf')

    Vs, Vs_times = get_descending_pulses(0.07, 0.01, 0.1, 100*10)
    plot_experiment(Vs, Vs_times, 'descending_positive.pdf')

    Vs, Vs_times = get_alternating_pulses(0.2, 10, 5e-3, 100*10)
    plot_experiment(Vs, Vs_times, 'alternating.pdf')

if __name__=='__main__':
    hysteresis()
    pulse_experiments()
