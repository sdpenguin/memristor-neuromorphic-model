import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns

from argparse import Namespace

# Note that N, g_low, g_high have been specified above
from memristor_neuromorphic_model.model import evolution
from memristor_neuromorphic_model.resistance_equations import tio2_linear_conductance
from memristor_neuromorphic_model.voltage_pulses import get_pulses
from memristor_neuromorphic_model.volatility_equations import voltage_induced, current_induced
from memristor_neuromorphic_model.rate_equations import boltzmann

plt.rcParams['figure.figsize'] = (20, 8)
plt.rcParams['font.size'] = 25
sns.set_style('dark')
plt.rcParams['lines.linewidth'] = 4

results_directory = 'figs/neuromorphic'

from memristor_neuromorphic_model.param_defaults import *

volatilty_params_frequency = Namespace(factor=500, time_constant=10)
volatility_params_spiking = Namespace(factor=10000000, time_constant=1)

np.random.seed(42) # Ensure consistent figure generation

def frequency_dependent_potentiation():
    # Parameters
    n_thresh = 10000
    g_step = 1e-7
    g_parallel = 1e-10
    resistance_equation = tio2_linear_conductance(g_step, g_parallel, n_thresh)
    N = 2*n_thresh

    total_time = 100 # The total time to run the simulation for
    timestep = 0.1 # The discrete simulation interval

    V_offset = 0.05
    V_a = 0.40049

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    start_resistance = 20e3
    freq_upper = 0.2
    freq_lower = 0.1
    for frequency in np.arange(freq_lower, freq_upper, 0.03):
        # duty_cycle = 0.005
        total = 1/(frequency)
        active = 0.15 # total*duty_cycle
        inactive = total - active # *(1-duty_cycle)
        time_to_apply = 0.75
        Vs, Vs_times = get_pulses(0.1, active, total, int(time_to_apply/active))
        start_state = resistance_equation(start_resistance, inverse=True)
        print('Starting resistance ', resistance_equation(start_state), 'Starting state: ', start_state)
        ns, Rs, Ts, Vs_measured, Vs_transients, times = evolution(timestep, Vs, Vs_times, v_factor, N, start_state, T, total_time, V_a, V_offset, boltzmann, resistance_equation, transient_equations=[voltage_induced], transient_params=[volatilty_params_frequency], rate_factor=rate_factor, quiet=True, simulation_no=frequency)
        ax1.set_xlabel('$t$ (s)')
        ax1.set_ylabel('$R(t)$ ($\Omega$)')
        ax1.plot(times, Rs, color=((frequency-freq_lower)/(freq_upper-freq_lower), 0.0, ((freq_upper-frequency)/(freq_upper-freq_lower))), label=("R(t) (" + str(np.round(frequency, 2)) + "Hz)"))
        ax2.plot(times, np.array(Vs_transients)-1, color=(0, ((freq_upper-frequency)/(freq_upper-freq_lower)), 0.0), label=("$\\rho(t)$ (" + str(np.round(frequency, 2)) + "Hz)")) # , label="$\\rho (t)$")
        ax2.plot(times, Vs_measured, color=(((frequency-freq_lower)/(freq_upper-freq_lower)), ((frequency-freq_lower)/(freq_upper-freq_lower)), 0.0), alpha=0.4) # , label='V(t)')
    ax1.set_ylim([0, 40e3])
    ax2.set_ylabel('$\\rho(t)$')
    ax2.text(1, 0.1, 'V(t)', fontsize=30, bbox=dict(facecolor=(1, 1, 1), alpha=0.0))
    ax1.legend(loc='upper center', bbox_to_anchor=(0.64, 0.5))
    ax2.legend(loc='upper center', bbox_to_anchor=(0.85, 0.5))
    plt.savefig(os.path.join(results_directory, 'frequency_dependent_switching.pdf'))

    E_a = V_a * q
    E_offset = V_offset * q
    v_divider = 1.0

    plt.clf()
    rho = np.arange(0, 40, 0.1)
    for V in np.arange(-0.2, 0.21, 0.1):
        n_eq = N / (np.exp((V*q + E_offset)/(k_b*T*(1+rho))) + 1)
        plt.plot(rho, n_eq, label='V={}'.format(np.round(V, 2)))
        plt.xlabel('$\\rho$')
        plt.ylabel('$n_{eq}$')

    equilibrium_y = n_thresh  # Example equilibrium y-coordinate
    plt.axhline(y=equilibrium_y, color='r', linestyle='--', label='$n_{thresh}$')

    # Add label for the equilibrium point
    # plt.text(max(rho), equilibrium_y, '$n_{thresh}$', color='r', verticalalignment='top', horizontalalignment='right')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(results_directory, 'rho_impact.pdf'))

def spike_generation():
    V_a = 1.0
    V_offset = -0.8

    N = 1000000
    n_thresh = int(N*0.8)
    g_step = 1e-8
    g_parallel = 1e-6

    V_transient_factor = 10000000 # 5e3 # The factor to multiply the instantaneous voltage by to obtain the steady state transient volatility.
    V_transient_tc = 1 # Characteristic timescale for the transient volatility decay.

    total_time = 2 # The total time to run the simulation for
    timestep = 1e-2 # The interval for updating the temperature due to Joule heating (and heat dissipation)
    # Please note that for the purposes of the drift simulation, the temperature will not change, so we set this to a high value

    resistance_equation = tio2_linear_conductance(g_step, g_parallel, n_thresh)

    plt.clf()
    for V in np.arange(-0.0, -0.181, -0.06):
        Vs = [V]
        Vs_times = [0.0]
        total_time=5.0
        start_state = resistance_equation(1e4, inverse=True)
        ns, Rs, Ts, Vs_measured, Vs_transients, times = evolution(timestep, Vs, Vs_times, v_factor, N, start_state, T, total_time, V_a, V_offset, boltzmann, resistance_equation, transient_equations=[current_induced], transient_params=[volatility_params_spiking], rate_factor=rate_factor, quiet=True, simulation_no=V)
        plt.plot(times, np.abs(Vs_measured/(np.array(Rs)+1e3)), label='V = {}'.format(np.round(V, 2)))
        plt.legend()
    plt.xlabel('t (s)')
    plt.ylabel('I(t)')
    plt.savefig(os.path.join(results_directory, 'synaptic_responses.pdf'))

if __name__ == '__main__':
    frequency_dependent_potentiation()
    spike_generation()
