import numpy as np
import datetime

from .volatility_equations import joule_heating

def ttf(lmbda):
    ''' Time to faliure distribution for a rate parameter lambda. '''
    return - np.log(1-np.random.uniform()) / max(lmbda, 1e-10) # 1e-10 to avoid division by 0 if the rate is minimal

def evolution(timestep, Vs, Vs_times, v_factor, N, n_initial, T, total_time, V_a, V_offset, rate_equation, resistance_equation, transient_equations, transient_params, rate_factor=1.0, simulation_no='?', method='event', quiet=False):
    '''
        method: [event, continuous_poisson, continuous_normal]
    '''
    Vs = np.concatenate([np.array([0.0]), np.array(Vs)])
    Vs_times = np.concatenate([np.array([0.0]), np.array(Vs_times)]) # We prepend a zero for programming convenience
    V_index = 1 # To account for the prepended zero, we start indexing at 0
    times = [0.0] # The starting time (offset)
    ns = [n_initial] # The initial number of metastable switches in the low resistance sate
    Rs = [resistance_equation(ns[-1])]
    Ts = [T] # Start at ambient temperature
    Vs_measured = [0.0]
    transients = [[1.0 for _ in range(len(transient_equations))]]
    r_down = 0
    r_up = 0

    previous_time = datetime.datetime.now()
    output_first_time = 1

    rng = np.random.default_rng()
    while(times[-1] < total_time):
        # Simulation logging
        if datetime.datetime.now() - previous_time >= datetime.timedelta(seconds=5) or output_first_time:
            if not quiet:
                print("Simulation no. {}: {}/{}s Resistance: {} Temperature: {} r_down: {} r_up: {} ns: {}".format(simulation_no, times[-1], total_time, Rs[-1], Ts[-1], r_down, r_up, ns[-1]))
            previous_time = datetime.datetime.now()
            if output_first_time:
                output_first_time += 1
                if output_first_time > 5:
                    output_first_time = False

        # Resistance more likely to increase (r_up is higher) for a positive voltage
        r_down = rate_factor*rate_equation(V_a, V_offset, v_factor*Vs_measured[-1], Ts[-1], transients[-1])
        r_up = rate_factor*rate_equation(V_a, V_offset, v_factor*Vs_measured[-1], Ts[-1], transients[-1], up=True)
        lmbda_down = r_down*((N-ns[-1]))
        lmbda_up = r_up*(ns[-1])
        
        # State and Readout Update
        if method == 'event':
            ttf_down = ttf(lmbda_down)
            ttf_up = ttf(lmbda_up)
            if ns[-1] == 0: # Ensure that there is a state transition to n if there are no n state switches left
                ttf_up = ttf_down + 1.0
            elif ns[-1] == N: # Ensure that there is a state transition to m if there are no m state switches left
                ttf_down = ttf_up + 1.0
            time_delta = min(ttf_up, ttf_down, timestep)
        else:
            time_delta = timestep
        if V_index < len(Vs_times):
            time_delta_V = Vs_times[V_index] - times[-1]
        else:
            time_delta_V = time_delta + 1
        if time_delta_V < time_delta:
            time_delta = time_delta_V
            V_index += 1
        
        # For discrete timestep method
        if method == 'continuous_poisson': # May be inaccurate for small values of timestep*(lmbda_down + lmbda_up)
            avg_change = rng.poisson(lmbda_up*timestep) - rng.poisson(lmbda_down*timestep)
            ns.append(ns[-1] + avg_change)
        elif method == 'continuous_normal':
            avg_change = np.random.randn()*np.sqrt(lmbda_down*timestep) + lmbda_down*timestep - np.random.randn()*np.sqrt(lmbda_up*timestep) - lmbda_up*timestep
            ns.append(ns[-1] + avg_change)
        
        # Update parameters for given time delta
        # Update transients
        new_transients = []
        for i, transient in enumerate(transients[-1]):
            new_transients.append(transient_equations[i](transient, Vs_measured[-1], Vs_measured[-1]/Rs[-1], Rs[-1], T, time_delta, transient_params[i]))
        transients.append(new_transients)
        # Update states and resistance
        if method == 'event':
            if (time_delta < ttf_down) and (time_delta < ttf_up):
                pass # No change to state for this time delta
            elif ttf_down < ttf_up:
                ns[-1] += 1
            elif ttf_up < ttf_down:
                ns[-1] -= 1
        if ns[-1] < 0:
            ns = 0
        elif ns[-1] > N:
            ns[-1] = N
        # Update resistance readout
        Rs.append(resistance_equation(ns[-1]))
        # Update voltage input
        Vs_measured.append(Vs[V_index-1])
        # Update times
        times.append(times[-1] + time_delta)
    return ns, Rs, Ts, Vs_measured, transients, times
