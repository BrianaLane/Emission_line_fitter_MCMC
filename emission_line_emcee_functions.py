import numpy as np
import emcee
import math
import string as st
import os
import pandas as pd
from scipy.integrate import quad
import matplotlib.pyplot as plt
import model_line_functions as mlf

class MCMC_functions():
	def __init__(self, num, lines, model, model_bounds, args):
		self.num = num
		self.lines = lines
		self.wl_lines = [int(st.split(w, ']')[1]) for w in lines]
		self.model = model
		self.model_bounds = model_bounds
		self.args = args
		self.int_flux = 0.0
		self.mc_results = 0.0
		self.flux = 0.0
		self.flux_err = 0.0

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

	def run_emcee(self, ndim, nwalkers, nchains, thetaGuess):

		pos = [thetaGuess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
		sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, args=self.args)

		print "Burning in ..."
		pos, prob, state = sampler.run_mcmc(pos, 200)

		sampler.reset()

		print "Running MCMC ..."
		pos, prob, state = sampler.run_mcmc(pos, 500, rstate0=state)

		flat_samples = sampler.flatchain
		samples = sampler.chain[:, :, :].reshape((-1, ndim))
		mc_results = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

		self.mc_results = mc_results

		return flat_samples, mc_results

	def gaussian(self, x, z, sig, inten):
		mu = 100*(1+z)
		return inten * (np.exp(-0.5*np.power(x - mu, 2.) / (np.power(sig, 2.))))

	def integrate_flux(self):
		mu = 100*(1+self.mc_results[0][0])
		a = [i[0] for i in self.mc_results]

		tot_flux = []
		tot_flux_err= []

		I = quad(self.gaussian, mu-100, mu+100, args=(a[0], a[1], a[2]))
		tot_flux.append(I[0])
		tot_flux_err.append(I[1])

		if len(a)>3:
			I2 = quad(self.gaussian, mu-100, mu+100, args=(a[0], a[1], a[3]))
			tot_flux.append(I2[0])
			tot_flux_err.append(I2[1])

		self.flux = tot_flux
		self.flux_err = tot_flux_err

		return tot_flux, tot_flux_err

	def plot_results(self):
		wl_sol, dat, disp = self.args

		sol    = [i[0] for i in self.mc_results]
		up_lim = [i[1] for i in self.mc_results]
		lo_lim = [i[2] for i in self.mc_results]

		up_lim_theta = np.add(sol,up_lim)
		lo_lim_theta = np.subtract(sol,lo_lim)

		fig, ax = plt.subplots(figsize=(13, 8))

		plt.plot(wl_sol, dat)
		plt.plot(wl_sol, self.model(wl_sol, theta = sol), color='red')
		plt.plot(wl_sol, self.model(wl_sol, theta = up_lim_theta), color='red', ls=':')
		plt.plot(wl_sol, self.model(wl_sol, theta = lo_lim_theta), color='red', ls=':')

		plt.text(0.05,0.82,'flux '+str(self.lines[0])+': '+str(round(self.flux[0],2))+'+/-'+str(round(self.flux_err[0],2)), 
			transform = ax.transAxes, color='navy',size='medium', bbox=dict(facecolor='none', edgecolor='navy', pad=10.0))
		if len(sol)>3:
			plt.text(0.05,0.72,'flux '+str(self.lines[1])+': '+str(round(self.flux[1],2))+'+/-'+str(round(self.flux_err[1],2)), 
				transform = ax.transAxes, color='navy',size='medium', bbox=dict(facecolor='none', edgecolor='navy', pad=10.0))

		#Set plot labels
		plt.title('Spectrum Fit: '+str(self.num))
		plt.ylabel('Flux (ergs/s/cm^2/A)')
		plt.xlabel('Wavelength (A)')

		#sets plotting speed and closes the plot before opening a new one
		plt.pause(0.01)
		plt.close()

	def write_results(self, df):
		for l in range(len(self.lines)):
			df[self.lines[l]][self.num] = self.flux[l]
			df[self.lines[l]+'_e'][self.num] = self.flux_err[l]
		return df



