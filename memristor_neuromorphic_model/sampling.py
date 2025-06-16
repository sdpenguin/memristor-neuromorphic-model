import numpy as np
from tqdm import tqdm

def get_sampled(resistances_all, times_all, sampling_step, total_time, quiet=True):
    '''
        Inputs:

        resistance_all: An array of resistance series.
        times_all: An array of times series individually matching the shapes of resistance series.
        sampling_step: How often to generate a sample.
        total_time: The maximum time to sample for (usually the minimum time the simulation was run for).

        Returns:

        sampled_resistances_all: An array of sampled resistances of length equal to the input resistances_all array.
    '''
    sampled_resistances_all = []
    i = 1
    for resistances, times in zip(resistances_all, times_all):
        if not quiet:
            print('Sampling sequence {} out of {}.'.format(i, len(resistances_all)))
        curr_time = 0.0
        sampled_resistances = []
        ts = []
        iterator = zip(resistances, times)
        if not quiet:
            iterator = tqdm(iterator)
        index = 0
        while curr_time <= total_time + sampling_step/2: # Account for floating point errors in the addition
            if (index == len(resistances)-1) or (times[index] >= curr_time):
                sampled_resistances.append(resistances[index])
                curr_time += sampling_step
            else:
                index += 1
        sampled_resistances_all.append(sampled_resistances)
        i += 1

    return sampled_resistances_all
