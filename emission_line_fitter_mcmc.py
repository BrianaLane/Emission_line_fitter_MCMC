import numpy as np
import math
import string
from scipy.integrate import quad
import subprocess
import os.path as op
import sys
import itertools
import model_line_functions as mlf
import emission_line_emcee_functions as elef
import matplotlib.pyplot as plt

#****************************************#
# Load in 2D spectral Data and Residuals #
#****************************************#

#data_file 	= '/Users/Briana/Documents/Grad_School/M82/ppxf/FcalFeSpesvp0054.fits'
#wl_sol_file = '/Users/Briana/Documents/Grad_School/M82/ppxf/WavelengthSolu_M82_F1_B.npy'

gasfit_file = '/Users/Briana/Documents/Grad_School/M82/ppxf/pPXF_VP/results/FcalFeSpesvp0054_GASFIT.npy'
newgal_file = '/Users/Briana/Documents/Grad_School/M82/ppxf/pPXF_VP/results/FcalFeSpesvp0054_gal.npy'
galfit_file = '/Users/Briana/Documents/Grad_School/M82/ppxf/pPXF_VP/results/FcalFeSpesvp0054_BESTFIT.npy'
wl_sol_file = '/Users/Briana/Documents/Grad_School/M82/ppxf/pPXF_VP/results/FcalFeSpesvp0054_wave.npy'

gasfit = np.load(gasfit_file)
galfit = np.load(galfit_file)
newgal = np.load(newgal_file)
wl_sol = np.load(wl_sol_file)

residuals = np.subtract(newgal, galfit)
dat       = np.add(gasfit, residuals)

#************************#
# User Defined Variables #
#************************#
outname = 'F1_B_OIII_gasResid.dat'

line = 'OIII_Hb_trip'
z		= 0.000677

thetaGuess = [z, 4, 4, 4] #z, sig, inten1, (inten2)
model_bounds = [(0.0005,0.002), (3.5,4.5), (0.0, 20.0), (0.0, 20.0)] #z, sig, inten1, (inten2)
ndim, nwalkers = len(thetaGuess), 100
nchains = (50, 100) #(200, 500)

#+++++++++++++++++++ end user defined variables +++++++++++++++++++++#

line_dict = {'OII':			{'mod':mlf.OII_gaussian,			'wl':[3727]},
			'Hb':			{'mod':mlf.Hb_gaussian,				'wl':[4861]},
			'OIII_doub':	{'mod':mlf.OIII_doub_gaussian,		'wl':[4959, 5007]},
			'SII_doub':		{'mod':mlf.SII_doub_gaussian,		'wl':[6717, 6731]},
			'OIII_Hb_trip':	{'mod':mlf.OIII_Hb_trip_gaussian,	'wl':[4861, 4959, 5007]},
			'NII_Ha_trip':	{'mod':mlf.NII_Ha_trip_gaussian,	'wl':[6549, 6562, 6583]}}

# trim the data and residual spectra around the line(s) to be fit
dat, residuals, wl_sol = mlf.trim_spec_for_model(line, dat, residuals, wl_sol)

#+++++++++++++++++++ this is where I need to add iteration over the data cube +++++++++++++++++++++#

#for i in range(np.shape(dat)[0]):
for i in range(3):
	#define the arguments containing the observed data for emcee
	args=(wl_sol, dat[i], np.std(residuals[i])**2)

	#build and MCMC object for that line
	OII_MCMC = elef.MCMC_functions(line_dict[line]['mod'], model_bounds, args)
	#call to run emcee
	flat_samp, mc_results = OII_MCMC.run_emcee(ndim, nwalkers, nchains, thetaGuess)
	print mc_results
	#calculate integrated flux of lines
	flux, flux_err = OII_MCMC.integrate_flux()
	#plot the results of the emcee
	OII_MCMC.plot_results()





