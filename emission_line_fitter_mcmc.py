import numpy as np
import math
import string
import matplotlib.pyplot as plt
from scipy.integrate import quad
import subprocess
import os.path as op
import sys
import itertools
import model_line_functions as mlf
import emission_line_emcee_functions as elef

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
nchains = (200, 500)

#+++++++++++++++++++ end user defined variables +++++++++++++++++++++#

model_dict = {'OII': mlf.OII_gaussian,
			'Hb': mlf.Hb_gaussian,
			'OIII_doub': mlf.OIII_doub_gaussian,
			'SII_doub': mlf.SII_doub_gaussian,
			'OIII_Hb_trip': mlf.OIII_Hb_trip_gaussian,
			'NII_Ha_trip': mlf.NII_Ha_trip_gaussian}

# trim the data and residual spectra around the line(s) to be fit
dat, residuals, wl_sol = mlf.trim_spec_for_model(line, dat, residuals, wl_sol)

#+++++++++++++++++++ this is where I need to add iteration over the data cube +++++++++++++++++++++#
#define the arguments containing the observed data for emcee
args=(wl_sol, dat[0][0], np.std(residuals[0][0])**2)

#build and MCMC object for that line
OII_MCMC = elef.MCMC_functions(model_dict[line], model_bounds, args)
#call to run emcee
flat_samp, mc_results = OII_MCMC.run_emcee(OII_MCMC.lnprob, ndim, nwalkers, nchains, thetaGuess)
print mc_results 
#plot the results of the emcee
OII_MCMC.plot_results(mc_results)





