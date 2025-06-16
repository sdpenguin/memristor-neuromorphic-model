import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from memristor_neuromorphic_model.quantise import quantise

plt.rcParams['figure.figsize'] = (20, 8)
plt.rcParams['font.size'] = 40
sns.set_style('dark')

drift_data_directory = 'data/drift'
unquantised_data_directory = 'data/processed_drift'
results_directory = 'figs/dataset_plots'

n_thresh = 10000
N = 2*n_thresh
g_step = 1e-7
g_parallel = 1e-10
timestep = 10000

def main():
    if not os.path.exists(os.path.join(unquantised_data_directory, 'pairs_{}.npy'.format(int(1)))):
        quantise(drift_data_directory, N, g_step, g_parallel, n_thresh, interval=1, base_directory=unquantised_data_directory, initial_only=False, quantise=False)
    pairs = np.load(os.path.join(unquantised_data_directory, 'pairs_{}.npy'.format(int(1))))
    plt.hist(1/pairs[:, 0], bins=100)
    plt.xlabel('Resistance ($\Omega$)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(results_directory, 'dataset_histogram_unquantised.pdf'), bbox_inches='tight')


if __name__=='__main__':
    main()
