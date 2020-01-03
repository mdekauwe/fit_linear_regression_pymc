#!/usr/bin/env python

"""
Fit linear regression...

The NUTS sampler does not work when we specify our own "blackbox" model, follow
this blog to get around the need for a gradient...

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
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import pandas as pd
import theano
import theano.tensor as tt


class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    #otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)
    otypes = [tt.dvector] # outputs a vector for the log likelihood)


    def __init__(self, loglike, obs, sigma):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.obs = obs
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.obs, self.sigma)

        outputs[0][0] = np.array(logl) # output the log-likelihood

def my_likelihood(theta, obs, sigma):
    """
    A Gaussian log-likelihood function for a model with parameters given
    in theta
    """

    model = run_and_unpack_cable(theta)

    return -(0.5/sigma**2)*np.sum((obs - model)**2)
    #return np.sum(-(0.5/sigma**2)*np.sum((obs - model)**2))

def my_loglikelihood(theta, obs, sigma):

    model = run_and_unpack_cable(theta)

    return -np.sum(0.5 * \
            (np.log(2. * np.pi * sigma ** 2.) + ((obs - model) / sigma) ** 2))

def linear_model(theta):

    intercept, slope = theta  # unpack line gradient and intercept

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

uncert = 0.1 * np.abs(obs)
uncert = uncert.astype(np.float64)


#fig = plt.figure(figsize=(6, 6))
#ax = fig.add_subplot(111, xlabel='x', ylabel='y')
#ax.plot(x, obs, 'o', label='observed')
#ax.plot(x, truth, label='truth', lw=2.)
#plt.legend(loc=0)
#plt.show()

logl = LogLike(my_likelihood, obs, uncert)


with pm.Model() as model:

    # Define priors
    intercept = pm.Normal('intercept', mu=0.0, sigma=20.0)
    slope = pm.Normal('slope', mu=0.0, sigma=20.0)

    theta = tt.as_tensor_variable([intercept, slope])

    # use a DensityDist (use a lamdba function to "call" the Op)
    pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})

    # Inference
    step = pm.NUTS() # Hamiltonian MCMC with No U-Turn Sampler
    #step = pm.Slice()
    #step = pm.Metropolis()

    trace = pm.sample(1000, step=step, cores=2, progressbar=True)

plt.figure(figsize=(7, 7))
pm.traceplot(trace[100:])
#plt.tight_layout()
plt.show()

print(pm.summary(trace))
