import numpy as np
import emcee
import math
import os
import matplotlib.pyplot as plt
import model_line_functions as mlf

class MCMC_functions():
	def __init__(self, model, model_bounds, args):
		self.model = model
		self.model_bounds = model_bounds
		self.args = args

	#define the log likelihood function
	def lnlike(self, theta, x, y, yerr):
		mod = self.model(x, theta)
		return -0.5*sum(((y - mod)**2)/(yerr**2))

	#define the log prior function
	def lnprior(self, theta):
		if len(theta) > 3:
			if ( self.model_bounds[0][0] <= theta[0] <= self.model_bounds[0][1]) and (self.model_bounds[1][0] <= theta[1] <= self.model_bounds[1][1]) and (self.model_bounds[2][0] <= theta[2] <= self.model_bounds[2][1]) and (self.model_bounds[3][0] <= theta[2] <= self.model_bounds[3][1]):
				return 0.0
			return -np.inf
		else:
			if ( self.model_bounds[0][0] <= theta[0] <= self.model_bounds[0][1]) and (self.model_bounds[1][0] <= theta[1] <= self.model_bounds[1][1]) and (self.model_bounds[2][0] <= theta[2] <= self.model_bounds[2][1]):
				return 0.0
			return -np.inf

	#define log postierior to sovle with emcee
	def lnprob(self, theta, x, y, yerr):
		lp = self.lnprior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp + self.lnlike(theta, x, y, yerr)

	def run_emcee(self, lnprob, ndim, nwalkers, nchains, thetaGuess):

		pos = [thetaGuess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=self.args)

		print "Burning in ..."
		pos, prob, state = sampler.run_mcmc(pos, 200)

		sampler.reset()

		print "Running MCMC ..."
		pos, prob, state = sampler.run_mcmc(pos, 500, rstate0=state)

		flat_samples = sampler.flatchain
		samples = sampler.chain[:, :, :].reshape((-1, ndim))
		mc_results = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

		return flat_samples, mc_results

	def plot_results(self, mc_results):
		wl_sol, dat, disp = self.args

		sol    = [i[0] for i in mc_results]
		up_lim = [i[1] for i in mc_results]
		lo_lim = [i[2] for i in mc_results]

		up_lim_theta = np.add(sol,up_lim)
		lo_lim_theta = np.subtract(sol,lo_lim)

		plt.plot(wl_sol, dat)
		plt.plot(wl_sol, self.model(wl_sol, theta = sol), color='red')
		plt.plot(wl_sol, self.model(wl_sol, theta = up_lim_theta), color='red', ls=':')
		plt.plot(wl_sol, self.model(wl_sol, theta = lo_lim_theta), color='red', ls=':')
		plt.show()
