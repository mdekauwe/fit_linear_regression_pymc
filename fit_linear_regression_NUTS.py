#!/usr/bin/env python

"""
Fit linear regression...

The NUTS sampler does not work when we specify our own "blackbox" model,
following this blog to get around the need for a gradient ...

https://docs.pymc.io/notebooks/blackbox_external_likelihood.html

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
import pymc3 as pm
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt
import numpy as np


from blackbox_log_likelihood import LogLikeWithGrad, LogLikeGrad


def my_likelihood(theta, x, obs, sigma):
    """
    A Gaussian log-likelihood function for a model with parameters given
    in theta
    """

    model = linear_model(x, theta)

    return np.sum( -(0.5/sigma**2)*np.sum((obs - model)**2) )

def linear_model(x, theta):

    """
    This could be any model...
    """
    intercept, slope = theta  # unpack line gradient and intercept

    y = slope * x + intercept
    #y = y.astype(np.float64)

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
#obs = obs.astype(np.float64)

uncert = 0.1 * np.abs(obs)
#uncert = uncert.astype(np.float64)


#fig = plt.figure(figsize=(6, 6))
#ax = fig.add_subplot(111, xlabel='x', ylabel='y')
#ax.plot(x, obs, 'o', label='observed')
#ax.plot(x, truth, label='truth', lw=2.)
#plt.legend(loc=0)
#plt.show()

# create our Op
logl = LogLikeWithGrad(my_likelihood, obs, x, uncert)

ndraws = 3000  # number of draws from the distribution
nburn = 1000   # number of "burn-in points" (which we'll discard)

with pm.Model() as model:

    # Define priors
    intercept = pm.Normal('intercept', mu=0.0, sigma=20.0)
    slope = pm.Normal('slope', mu=0.0, sigma=20.0)

    # convert to tensor vectors
    theta = tt.as_tensor_variable([intercept, slope])

    # use a DensityDist (use a lamdba function to "call" the Op)
    pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})

    # Inference
    step = pm.NUTS() # Hamiltonian MCMC with No U-Turn Sampler
    #step = pm.Slice()
    #step = pm.Metropolis()
    trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True)

plt.figure(figsize=(7, 7))
pm.traceplot(trace[100:])
#plt.tight_layout()
plt.show()

print(pm.summary(trace))
