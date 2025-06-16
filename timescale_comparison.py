import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import seaborn as sns

from tqdm import tqdm
from scipy.optimize import curve_fit

# Note that N, g_low, g_high have been specified above
from memristor_neuromorphic_model.resistance_equations import tio2_linear_conductance
from memristor_neuromorphic_model.quantise import quantise, get_conductance_bins

plt.rcParams['figure.figsize'] = (20, 8)
plt.rcParams['font.size'] = 30
sns.set_style('dark')

quantised_data_directory = 'data/quantised_drift'
drift_data_directory = 'data/drift'
results_directory = 'figs/timescale_comparison'

timesteps = np.array([1000, 5000, 10000]) # np.arange(1000, 10001, 5000)

n_thresh = 10000
N = 2*n_thresh
g_step = 1e-7
g_parallel = 1e-10

if __name__ == '__main__':
    for timestep in tqdm(timesteps):
        if not os.path.exists(os.path.join(quantised_data_directory, 'pairs_{}.npy'.format(int(timestep//10)))):
            quantise(drift_data_directory, N, g_step, g_parallel, n_thresh, interval=timestep//10, base_directory=quantised_data_directory, initial_only=False) # The interval is the number of interavls between reads to load

    rram_equation = tio2_linear_conductance(g_step, g_parallel, n_thresh)
    def linear_conductance(r, a):
        # Estimate the initial value of n through the model, then use this to estimate a linear difference in the number of "high"-state switches
        # Please note that g_high and g_low here refer to the high and low conductances, rather than the conductances corresponding to high and low resistive states
        n = rram_equation(r, inverse=True)
        return rram_equation(n-a) - r
    conductances, conductance_bin_boundaries = get_conductance_bins(N, g_step, g_parallel, n_thresh)
    resistances = np.flip(1 / conductances)
    resistance_bin_boundaries = np.flip(1 / conductance_bin_boundaries)
    lower_index = 6500
    upper_index = 9910
    min_points = 2 # The minimum number of points necessary for stat calculations

    params_sup = []
    for timestep in timesteps:
        pairs = np.load(os.path.join(quantised_data_directory, 'pairs_{}.npy'.format(int(timestep//10))))

        stats_array = np.empty((0, 2))

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
        truncated_resistances = resistances[lower_index:upper_index]
        truncated_conductances = conductances[N+1-n_thresh-upper_index:N+1-n_thresh-lower_index]
        
        stats_array_differences = stats_array[:, 0][stats_array[:, 0] != None] - truncated_resistances[stats_array[:, 0] != None]
        # smoothing = 0.9
        # avgs = []
        # avg = stats_array_differences[0]
        # for i in range(len(stats_array_differences)):
        #     avg = smoothing*avg + (1-smoothing)*(stats_array_differences[i])
        #     avgs.append(avg)
        window = np.ones(10)/10
        avgs = np.convolve(stats_array_differences, window, mode='same')
        
        params2, _ = curve_fit(linear_conductance, truncated_resistances[stats_array[:, 0] != None], avgs, bounds=[[0], [1e3]], p0=[10])
        params_sup.append(params2[0])
        print(params2[0])
        plt.plot(truncated_resistances[stats_array[:, 0] != None], avgs, label='Moving Average {}'.format(timestep), color=(timestep/np.max(timesteps), 0.0, (1-timestep/np.max(timesteps))), linewidth=5)
        plt.plot(truncated_resistances[stats_array[:, 0] != None], linear_conductance(truncated_resistances[stats_array[:, 0] != None], params2[0]), label='Linear Model {}'.format(timestep), color=(0.0, timestep/np.max(timesteps), 0.0), linewidth=5)
        plt.ylabel('$R(t + \Delta T) - R(t)$', fontsize=30)
        plt.xlabel('$R(t)$', fontsize=30)
        plt.legend(fontsize=25)
        # plt.title('Moving Average', fontsize=20)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.savefig(os.path.join(results_directory, 'linear_model_timestep_fits.pdf'), bbox_inches='tight')

    plt.clf()
    plt.plot(timesteps, params_sup, label='Best fit parameters', linewidth=5, linestyle='--', marker='x', markersize=40)
    plt.grid()
    plt.xlabel('Timestep')
    plt.ylabel('a')
    # plt.xlim([0, 10000])
    plt.ylim(0, 4)
    # def linear(x, m):
    #     return m*x
    # params_linear = curve_fit(linear, timesteps, params_sup)
    # plt.plot(np.arange(0, 10000), params_linear[0]*np.arange(0, 10000), label='Best linear approximation')
    plt.xlabel('Delay $\Delta T$ (s)')
    plt.ylabel('Best fit $a$')
    plt.legend()
    plt.savefig(os.path.join(results_directory, 'linear_model_timestep_comparison.pdf'), bbox_inches='tight')
