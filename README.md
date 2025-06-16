# Event-Based Simulation of Stochastic Memristive Devices for Neuromorphic Computing

This repository contains the code and data associated with the paper:

W. El-Geresy, C. Papavassiliou, and D. Gündüz, ‘Event-Based Simulation of Stochastic Memristive Devices for Neuromorphic Computing’, arXiv:2407.04718 [physics], Jun. 2024, doi: 10.48550/arXiv.2407.04718.

Which is available [here](https://arxiv.org/abs/2407.04718).

## Installing the Python Module

To install `memristor_neuromorphic_model` as a Python module, run:

`` pip install . ``

This will allow `memristor_neuromorphic_model` to be imported as a module and used externally.

## The Model

The model is composed of three components. There are functions for each of the components available in the following modules:

### Rate Equations

``memristor_neuromorphic_model.rate_equation``

Implemented:
- Boltzmann (Recommended)
- Fermi (Generalised Metastable Switch Model) (TODO: Citation)

### Volatility Equations

``memristor_neuromorphic_model.volatility_equations``

Implemented:
- Voltage Induced (structural disruption)
- Current Induced (structural disruption)
- Joule Heating

### Readout Equations

``memristor_neuromorphic_model.readout_equations``

Implemented:
- Linear conductance
- TiO2 inear conductance (with threshold)
- Square linear conductance

## Using the model

The most basic way to use the model is to run the function `evolution` from `memristor_volatility_model.model`.

You will need to choose from the above components for your particular model implementation, and specify parameters.

## Scripts

### Plotting the Drift Data

One can plot the drift dataset using the following:

``python plot_drift_data.py``

### Fitting to the Drift Data

To fit the parameters of the resistance equation to the drift data, we can run the following:

``python readout_equation_fitting.py``

This will also compare to two other models: a linear model and a stretched exponential model.

We can evaluate the goodness of fit of the proposed linear conductance model across timescales using:

``python timescale_comparison.py``

We can then derive parameter pairs (V_a and V_offset) to fit to the linear conductance model by using:

``python model_fitting.py``

### Running the Neuromorphic Experiments

Demonstration of frequency dependent potentiation and spiking behaviour through choice of model parameters:

``python neuromorphic_experiments.py``

### Experimental Simulations

We can simulate a pinched hysteresis loop using the fit model and associated switching experiments for different pulse sequences:

``model_experiments.py``

### Discrete Time Comparison

We can evaluate the efficiency of the event-based model compared to a discrete-time simulation using the following:

`` python discrete_evolution_comparison.py ``
