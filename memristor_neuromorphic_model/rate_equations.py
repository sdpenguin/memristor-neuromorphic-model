import numpy as np

k_b = 1.38064e-23 # Boltzmann's Constant
q = 1.602176e-19 # The electronic charge

''' A rate equation maps inputs to an associated (Poisson) rate associated with switching.
    This should always be a positive quantity. '''

def boltzmann(V_a, V_offset, V, T, phis, up=False):
    phi_tot = 1.0
    for phi in phis:
        phi_tot *= phi
    rate = np.exp(-q*(V_a + (-1 if up else 1)*(V_offset + V)/2)/k_b/(T*phi_tot))
    return max(0, rate) # To account for floating point errors which make the rate negative

def fermi(V_a, V_offset, V, T, phis, up=False):
    phi_tot = 1.0
    for phi in phis:
        phi_tot *= phi
    rate = 1/(1+np.exp(q*(V_a + (-1 if up else 1)*(V_offset + V)/2)/k_b/(T*phi)))
    return max(0, rate)

rate_equation_dict = {
    'boltzmann' : boltzmann,
    'fermi': fermi
}
