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

#************************#
# User Defined Variables #
#************************#
#data_file 	= '/Users/Briana/Documents/Grad_School/M82/ppxf/FcalFeSpesvp0054.fits'
#wl_sol_file = '/Users/Briana/Documents/Grad_School/M82/ppxf/WavelengthSolu_M82_F1_B.npy'

gasfit_file = '/Users/Briana/Documents/Grad_School/M82/ppxf/pPXF_VP/results/FcalFeSpesvp0054_GASFIT.npy'
newgal_file = '/Users/Briana/Documents/Grad_School/M82/ppxf/pPXF_VP/results/FcalFeSpesvp0054_gal.npy'
galfit_file = '/Users/Briana/Documents/Grad_School/M82/ppxf/pPXF_VP/results/FcalFeSpesvp0054_BESTFIT.npy'
wl_sol_file = '/Users/Briana/Documents/Grad_School/M82/ppxf/pPXF_VP/results/FcalFeSpesvp0054_wave.npy'

outname = 'F1_B_OIII_gasResid.dat'

line = 'OIII_Hb_trip'
z		= 0.000677

thetaGuess = [z, 4, 4,4] #z, sig, inten1, (inten2)
ndim, nwalkers = len(thetaGuess), 100
nchains = (200, 500)

#+++++++++++++++++++ end user defined variables +++++++++++++++++++++#

model_dict = {'OII':mlf.OII_gaussian,
			'Hb': mlf.Hb_gaussian,
			'OIII_doub': mlf.OIII_doub_gaussian,
			'SII_doub': mlf.SII_doub_gaussian,
			'OIII_Hb_trip': mlf.OIII_Hb_trip_gaussian,
			'NII_Ha_trip': mlf.NII_Ha_trip_gaussian}

wl_trim_dict = {'OII':(3700.0, 3760.0),
				'Hb': (4830.0, 4890.0),
				'OIII_doub': (4930.0, 5050.0),
				'SII_doub': (6700.0, 6750.0),
				'OIII_Hb_trip': (4830.0, 5050.0),
				'NII_Ha_trip': (6520.0, 6610.0)}

gasfit = np.load(gasfit_file)
galfit = np.load(galfit_file)
newgal = np.load(newgal_file)
wl_sol = np.load(wl_sol_file)

inds = np.where((wl_trim_dict[line][0] < wl_sol) & (wl_sol < wl_trim_dict[line][1]))

gasfit = gasfit[:, inds]
galfit = galfit[:, inds]
newgal = newgal[:, inds]
wl_sol = wl_sol[inds]

residuals = np.subtract(newgal, galfit)
dat       = np.add(gasfit, residuals)

#+++++++++++++++++++ this is where I need to add iteration over the data cube +++++++++++++++++++++#

args=(wl_sol, dat[0][0], np.std(residuals[0][0])**2)

OII_MCMC = elef.MCMC_functions(model_dict[line], 20)

flat_samp, mc_results = OII_MCMC.run_emcee(OII_MCMC.lnprob, ndim, nwalkers, nchains, thetaGuess, args)
print mc_results 

if len(thetaGuess) > 3:

	z_mc, sig_mc, inten1_mc, inten2_mc = mc_results 

	plt.plot(wl_sol, dat[0][0])
	plt.plot(wl_sol, model_dict[line](wl_sol, theta = (z_mc[0], sig_mc[0], inten1_mc[0], inten2_mc[0])), color='red')
	plt.plot(wl_sol, model_dict[line](wl_sol, theta = (z_mc[0]+z_mc[1], sig_mc[0]+sig_mc[1], inten1_mc[0]+inten1_mc[1], inten2_mc[0]+inten2_mc[1])), color='red', ls=':')
	plt.plot(wl_sol, model_dict[line](wl_sol, theta = (z_mc[0]-z_mc[2], sig_mc[0]-sig_mc[2], inten1_mc[0]-inten1_mc[2], inten2_mc[0]-inten2_mc[2])), color='red', ls=':')
	plt.show()

else:

	z_mc, sig_mc, inten_mc = mc_results 

	plt.plot(wl_sol, dat[0][0])
	plt.plot(wl_sol, model_dict[line](wl_sol, theta = (z_mc[0], sig_mc[0], inten_mc[0])), color='red')
	plt.plot(wl_sol, model_dict[line](wl_sol, theta = (z_mc[0]+z_mc[1], sig_mc[0]+sig_mc[1], inten_mc[0]+inten_mc[1])), color='red', ls=':')
	plt.plot(wl_sol, model_dict[line](wl_sol, theta = (z_mc[0]-z_mc[2], sig_mc[0]-sig_mc[2], inten_mc[0]-inten_mc[2])), color='red', ls=':')
	plt.show()



