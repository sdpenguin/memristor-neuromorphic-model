import numpy as np

''' Volatility equations. Should take the current transient value (phi) and the environmental/circuit inputs (V, I, R, T) as inputs, and return the updated transient. '''

def voltage_induced(phi_v, V, I, R, T, timestep, transient_params):
    '''Get the new transient value.

        V_transient: The most recent transient volatility value.
        V_transient_factor: The factor to multiply the applied voltage by to obtain the maximum transient value.
        V_transient_tc: The time constant for reaching the maximum transient volatility value.
        V: The most recent voltage.
        deltat: The time to simulate for.
    '''
    phi_v = phi_v - 1
    return 1 + phi_v + (transient_params.factor*np.abs(V)-phi_v)*timestep/transient_params.time_constant

def current_induced(phi_i, V, I, R, T, timestep, transient_params):
    '''Analagous to the voltage induced version, but for current. '''
    phi_i = phi_i - 1
    return 1 + phi_i + (transient_params.factor*np.abs(I)-phi_i)*timestep/transient_params.time_constant

def joule_heating(phi_T, V, I, R, T, timestep, transient_params):
    ''' Apply Newton's Law of Cooling and Joule heating in the solution to a differential equation
        that assumes a constant voltage input for deltaT seconds to calculate a new temperature.
        Computes a factor that updates the temperature to reflect the internal temperature of the device (relative to the bath temperature).
    
        T: The bath (ambient) temperature.

        Transient parameters:
        Rth: The thermal resistance.
        Cth: The thermal capacity.
        
        Some useful links:
        https://en.wikipedia.org/wiki/Newton%27s_law_of_cooling
        https://www.umsl.edu/~physics/files/pdfs/Electricity%20and%20Magnetism%20Lab/Exp6.JouleHeating.pdf
    '''
    T_internal = phi_T*T
    tau_th = transient_params.Rth*transient_params.Cth
    T_delta = ((T + (V**2/R)*transient_params.Rth - T_internal)/tau_th)*timestep
    T_new = np.minimum(T_internal + T_delta, T + V**2/R*transient_params.Rth)
    T_new = np.maximum(T_new, T) # The temperature should never fall below the ambient temperature T_base
    # The volatilty is computed as the factor to multiply the denominator of the rate equation (the base temperature) by (i.e. the temperature factor in this case)
    return T_new/T 
