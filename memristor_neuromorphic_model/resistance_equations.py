import numpy as np

def basic_linear_conductance(low=1e2, high=1e6):
    ''' An RRAM resistance equation.
        The low is the low resistance state.
        The high is the high resistance state. '''
    low_g = 1/low
    high_g = 1/high
    def resistance_equation(x, N, extra_m_states=0, extra_n_states=0, inverse=False):
        ''' Calculates the resistance from the number of switches in the low state (and the total number of switches).
            NOTE: The extra states are not currently used for resistance calculation. '''
        if inverse:
            gs = 1/x
            ns = N*(gs - high_g)/(low_g - high_g)
            return np.round(ns).astype(int)
        else:
            ns = x
            ms = N - ns
            return 1/((low_g)*ns/N + (high_g)*ms/N)
    return resistance_equation

def tio2_linear_conductance(g_step=1e-10, g_parallel=1e-7, n_thresh=10000):
    ''' An RRAM resistance equation.
        g_step is the linear conductance increase over time.
        g_parallel is the parallel resistance (upper bounding the highest resistance possible).
        n_thresh is the threshold state, at or below which n will not have an impact on the output resistance. '''
    def resistance_equation(x, inverse=False):
        ''' Calculates the resistance from the number of switches in the low state (and the total number of switches).
            NOTE: The extra states are not currently used for resistance calculation. '''
        if inverse:
            gs = 1/x
            ns = n_thresh + (1/g_step)*(gs - g_parallel) # If the resistance is at 1/g_parallel, assume that n is n_thresh (rather than being 0 or anywhere in between 0 and n_thresh, which are also possible values) - i.e. assume we are in the active switching region.
            return np.round(ns).astype(int)
        else:
            ns = x
            return 1/((g_step)*np.maximum(ns - n_thresh, 0) + (g_parallel))
    return resistance_equation

def tio2_square_conductance(g_step=1e-10, g_parallel=1e-7, n_thresh=10000, factor=1.0):
    ''' An RRAM resistance equation.
        g_step is the linear conductance increase over time.
        g_parallel is the parallel resistance (upper bounding the highest resistance possible).
        n_thresh is the threshold state, at or below which n will not have an impact on the output resistance. 
        Here we assume that the conductance increases with the square of the state - i.e. the n measures the width of the conductive filament.
        The conductive filament area impacts the resistance. '''
    def resistance_equation(x, inverse=False):
        ''' Calculates the resistance from the number of switches in the low state (and the total number of switches).
            NOTE: The extra states are not currently used for resistance calculation. '''
        if inverse:
            gs = 1/x
            ns = n_thresh + (1/factor)*((1/g_step)*(gs - g_parallel))**(3/2)
            return np.round(ns).astype(int)
        else:
            ns = x
            return 1/((g_step)*((factor*np.maximum(ns-n_thresh, 0))**(2/3)) + (g_parallel))
    return resistance_equation

resistance_equation_dict = {
    'basic_linear_conductance' : basic_linear_conductance,
    'tio2_linear_conductance' : tio2_linear_conductance,
    'tio2_square_conductance' : tio2_square_conductance
}
