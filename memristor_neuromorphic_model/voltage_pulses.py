import numpy as np

def get_pulses(magnitude, length, interval, number, offset=0.0):
    '''
        Generates `number` pulses of frequency 1/`interval`, with active times of `length`.
        Inputs:

        magnitude: The amplitude of the pulses (V).
        length: The active time of the pulses (s).
        interval: The inter-pulse interval (s).
        number: The number of pulses.
        offset: An optional offset time at which to start the pulses (s).

        Returns:
        
        (Vs, Vs_times)
    '''
    Vs = np.concatenate([np.array([magnitude, 0.0]) for x in range(number)])
    Vs_times = np.concatenate([np.array([offset + interval*x, offset + interval*x + length]) for x in range(number)])
    return Vs, Vs_times

def get_ascending_pulses(max_magnitude, step, duty_cycle, interval, offset=0.0):
    '''
        Inputs:

        max_magnitude: The maximum magnitude of pulse (V).
        step: The step to increase the magnitude after each pulse (V).
        duty_cycle: The percentage of the time period during which the voltage pulse is active.
        interval: The interpulse interval (s).
        offset: An optional offset time at which to start the pulses (s).
    '''

    Vs = np.arange(step, max_magnitude+(step/10), step)
    Vs = [0.0 if i % 2 == 0 else Vs[i//2] for i in range(len(Vs)*2+1)]
    length = duty_cycle*interval
    Vs_times = np.concatenate([np.array(0.0).reshape(1)] + [np.array([offset + interval*x, offset + interval*x+length]) for x in range(len(Vs)//2)])
    return Vs, Vs_times

def get_descending_pulses(max_magnitude, step, duty_cycle, interval, offset=0.0):
    '''
        Inputs:

        max_magnitude: The maximum magnitude of pulse (V).
        step: The step to increase the magnitude after each pulse (V).
        duty_cycle: The percentage of the time period during which the voltage pulse is active.
        interval: The interpulse interval (s).
        offset: An optional offset time at which to start the pulses (s).
    '''

    Vs = np.arange(max_magnitude, 0.0, -step)
    Vs = [0.0 if i % 2 == 0 else Vs[i//2] for i in range(len(Vs)*2+1)]
    length = duty_cycle*interval
    Vs_times = np.concatenate([np.array(0.0).reshape(1)] + [np.array([offset + interval*x, offset + interval*x+length]) for x in range(len(Vs)//2)])
    return Vs, Vs_times

def get_alternating_pulses(magnitude, number_of_pulses, duty_cycle, interval, offset=0.0):
    '''
    Inputs:

    magnitude: the magnitude of the alternating voltage pulses.
    number_of_pulses: The number of positive and negative pulses to generate. Note that the total number of pulses is double this value.
    duty_cycle: The percentage of the time period during which the voltage pulse is active.
    interval: The interpulse interval (s).
    offset: An optional offset time at which to start the pulses (s).
    '''

    Vs = []
    for i in range(number_of_pulses):
        Vs = Vs + [-magnitude, magnitude]
    Vs = [0.0 if i % 2 == 0 else Vs[i//2] for i in range(len(Vs)*2+1)]
    length = duty_cycle*interval
    Vs_times = np.concatenate([np.array(0.0).reshape(1)] + [np.array([offset + interval*x, offset + interval*x+length]) for x in range(len(Vs)//2)])
    return Vs, Vs_times
