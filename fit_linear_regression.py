#!/usr/bin/env python

"""
Fit linear regression...

This is using PYMC3, everything needs to be cast to float 64

(Don't need for this example...)
THEANO_FLAGS='floatX=float64' ./run_pymc_wrapper.py


That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (03.01.2020)"
__email__ = "mdekauwe@gmail.com"


import os
import sys
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import pandas as pd
import theano
import theano.tensor as tt

@theano.as_op(itypes=[tt.dscalar, tt.dscalar], otypes=[tt.dvector])
def linear_model(intercept, slope):

    # figure out how to pass
    size = 200
    x = np.linspace(0, 1, size)

    y = slope * x + intercept
    y = y.astype(np.float64)

    return y


# set up Observations...
size = 200
x = np.linspace(0, 1, size)

# y = m * x + c
intercept = 1.0
slope = 2.0
truth = slope * x + intercept

# add noise
obs = truth + np.random.normal(scale=.5, size=size)
obs = obs.astype(np.float64)

#fig = plt.figure(figsize=(6, 6))
#ax = fig.add_subplot(111, xlabel='x', ylabel='y')
#ax.plot(x, obs, 'o', label='observed')
#ax.plot(x, truth, label='truth', lw=2.)
#plt.legend(loc=0)
#plt.show()



with pm.Model() as model:

    # Define priors
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.0)
    intercept = pm.Normal('intercept', mu=0.0, sigma=20.0)
    slope = pm.Normal('slope', mu=0.0, sigma=20.0)

    # Define likelihood
    mod = linear_model(intercept, slope)
    likelihood = pm.Normal('y_obs', mu=mod, sd=sigma, observed=obs)

    # Inference

    # The NUTS won't work with the "blackbox" model setup like this as it
    # doesn't have a gradient, so we can only use Slice or Metropolis...
    #step = pm.NUTS() # Hamiltonian MCMC with No U-Turn Sampler -

    #step = pm.Slice()
    step = pm.Metropolis()
    trace = pm.sample(10000, step=step, cores=3, progressbar=True)

plt.figure(figsize=(7, 7))
pm.traceplot(trace[100:])
#plt.tight_layout()
plt.show()

print(pm.summary(trace))
