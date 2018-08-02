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

line = 'OIII_Hb'
z		= 0.000677
wl_trim_range = 40


#+++++++++++++++++++ end user defined variables +++++++++++++++++++++#

#For triplets: need to put the strongest line first and the weaker line second. The H line is last 
line_wls    = {'OII':[3727], 'OIII5007':[5007], 'OIII4959':[4959],'Hbonly':[4861], 'OIII_Hb':[5007,4959,4861], "NII_Ha":[6583,6549,6562], "SII":[6731,6717]}
line_ratios = {'OIII_Hb':3, 'NII_Ha':2.5, 'OIII_doub':3}

gasfit = np.load(gasfit_file)
galfit = np.load(galfit_file)
newgal = np.load(newgal_file)
wl_sol = np.load(wl_sol_file)

inds = np.where((4930.0 < wl_sol) & (wl_sol < 5050.0))

gasfit = gasfit[:, inds]
galfit = galfit[:, inds]
newgal = newgal[:, inds]
wl_sol = wl_sol[inds]

residuals = np.subtract(newgal, galfit)
dat       = np.add(gasfit, residuals)

thetaGuess = [z, 4, 4]
ndim, nwalkers = len(thetaGuess), 100
args=(wl_sol, dat[0][0], np.std(residuals[0][0])**2)
nchains = (200, 500)

flat_samp, mc_results = elef.run_emcee(elef.lnprob, ndim, nwalkers, nchains, thetaGuess, args)
print mc_results 

z_mc, sig_mc, inten_mc = mc_results 

plt.plot(wl_sol, dat[0][0])
plt.plot(wl_sol, mlf.OIII_doub_gaussian(wl_sol, theta = (z_mc[0], sig_mc[0], inten_mc[0])), color='red')
plt.plot(wl_sol, mlf.OIII_doub_gaussian(wl_sol, theta = (z_mc[0]+z_mc[1], sig_mc[0]+sig_mc[1], inten_mc[0]+inten_mc[1])), color='red', ls=':')
plt.plot(wl_sol, mlf.OIII_doub_gaussian(wl_sol, theta = (z_mc[0]-z_mc[2], sig_mc[0]-sig_mc[2], inten_mc[0]-inten_mc[2])), color='red', ls=':')
plt.show()



