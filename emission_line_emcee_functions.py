import numpy as np
import emcee
import math
import os
import model_line_functions as mlf

# class MCMC_functions(model, model_bounds):
# 	def __init__(self):
# 		self.model = model
# 		self.model_args = model_args
# 		self.model_bounds = model_bounds

#define the log likelihood function
def lnlike(theta, x, y, yerr):
	mod = mlf.OIII_doub_gaussian(x, theta)
	return -0.5*sum(((y - mod)**2)/(yerr**2))

#define the log prior function
def lnprior(theta):
	# if len(theta) > 3:
	# 	if ( 0.0005 <= theta[0] <= 0.002) and (3.5 <= theta[1] <= 4.5) and (0.0 <= theta[2] <= 100) and (0.0 <= theta[3] <= 100):
	# 		return 0.0
	# 	return -np.inf
	# else:
	if ( 0.0005 <= theta[0] <= 0.002) and (3.5 <= theta[1] <= 4.5) and (0.0 <= theta[2] <= 100):
		return 0.0
	return -np.inf

#define log postierior to sovle with emcee
def lnprob(theta, x, y, yerr):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, x, y, yerr)

def run_emcee(lnprob, ndim, nwalkers, nchains, thetaGuess, args):

	pos = [thetaGuess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)

	print "Burning in ..."
	pos, prob, state = sampler.run_mcmc(pos, 200)

	sampler.reset()

	print "Running MCMC ..."
	pos, prob, state = sampler.run_mcmc(pos, 500, rstate0=state)

	flat_samples = sampler.flatchain
	samples = sampler.chain[:, :, :].reshape((-1, ndim))
	mc_results = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

	return flat_samples, mc_results
