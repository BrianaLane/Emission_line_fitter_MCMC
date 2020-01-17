import numpy as np
import pandas as pd 
import os.path as op

import model_line_functions as mlf
import emission_line_emcee_functions as elef
import matplotlib.pyplot as plt

#****************************************#
# Load in 2D spectral Data and Residuals #
#****************************************#

# M82
#data_file 	= '/Users/Briana/Documents/Grad_School/M82/ppxf/FcalFeSpesvp0054.fits'
#wl_sol_file = '/Users/Briana/Documents/Grad_School/M82/ppxf/WavelengthSolu_M82_F1_B.npy'

gasfit_file = '/Users/Briana/Documents/Grad_School/M82/ppxf/pPXF_VP/results/F1/FcalFeSpesvp0054_GASFIT.npy'
newgal_file = '/Users/Briana/Documents/Grad_School/M82/ppxf/pPXF_VP/results/F1/FcalFeSpesvp0054_gal.npy'
galfit_file = '/Users/Briana/Documents/Grad_School/M82/ppxf/pPXF_VP/results/F1/FcalFeSpesvp0054_BESTFIT.npy'
wl_sol_file = '/Users/Briana/Documents/Grad_School/M82/ppxf/pPXF_VP/results/F1/FcalFeSpesvp0054_wave.npy'

gasfit = np.load(gasfit_file)
galfit = np.load(galfit_file)
newgal = np.load(newgal_file)
wl_sol = np.load(wl_sol_file)

residuals = np.subtract(newgal, galfit)
fit       = np.subtract(galfit, gasfit)
dat       = np.subtract(newgal, fit)

#************************#
# User Defined Variables #
#************************#
pd_dataframe = 'M82_F1.csv'

line = 'OIII_Hb_trip'
z		= 0.000677

res = 5.3 #AA
sigma_guess = np.sqrt(res)
thetaGuess = [z, sigma_guess, 4, 4,] # m, b, z, sig, inten1, (inten2)
model_bounds = [(0.0005,0.002), (2.0,5.0), (0.0, 20.0), (0.0, 20.0)] # m, b, z, sig, inten1, (inten2)
ndim, nwalkers = len(thetaGuess), 100
nchains = (200, 500) #(200, 500)

make_dataframe = False

#+++++++++++++++++++ end user defined variables +++++++++++++++++++++#

#*****************************#
# Initialize Pandas Dataframe #
#*****************************#
if make_dataframe:
	index = np.arange(np.shape(dat)[0])
	columns = ['[OII]3727', '[Hb]4861', '[OIII]4959', '[OIII]5007', '[Ha]6562', '[NII]6549', '[NII]6583', '[SII]6717', '[SII]6731', 
				'[OII]3727_e', '[Hb]4861_e', '[OIII]4959_e', '[OIII]5007_e', '[Ha]6562_e', '[NII]6549_e', '[NII]6583_e', '[SII]6717_e', '[SII]6731_e']
	df = pd.DataFrame(index=index, columns=columns)

	#assign the dtypes to be float for the line fluxes
	#and objects for the errors since they will be a list of 2 values
	dtypes = {k: float for k in columns[:9]}
	dtypes.update({k: object for k in columns[10:]})
	df = df.astype(dtypes)

else:
	df = pd.read_csv(pd_dataframe)

# trim the data and residual spectra around the line(s) to be fit
dat, residuals, wl_sol = mlf.trim_spec_for_model(line, dat, residuals, wl_sol, z)

#for i in index:
for i in range(5):
	print 'Spec '+str(i)
	#define the arguments containing the observed data for emcee
	args=(wl_sol, dat[i], residuals[i])

	#build an MCMC object for that line
	OII_MCMC = elef.MCMC_functions(i, line, model_bounds, args, fit_cont=False, abs_params=(0.0, 0.0))
	#call to run emcee
	flat_samp, mc_results = OII_MCMC.run_emcee(ndim, nwalkers, nchains, thetaGuess)
	r = [i[0] for i in mc_results]
	print r, '\n'
	#calculate integrated flux of lines
	flux, flux_err = OII_MCMC.calculate_flux()
	#plot the results of the emcee
	OII_MCMC.plot_results()
	#write the results to a pandas dataframe
	new_df = OII_MCMC.write_results(df)

new_df.to_csv(pd_dataframe)
print new_df.head(7)






