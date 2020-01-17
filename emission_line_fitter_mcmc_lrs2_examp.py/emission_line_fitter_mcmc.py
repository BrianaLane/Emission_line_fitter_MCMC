import numpy as np
import pandas as pd 
import os.path as op
import glob as glob
import matplotlib.pyplot as plt

import model_line_functions as mlf
import emission_line_emcee_functions as elef

#************************#
# User Defined Variables #
#************************#

arm = 'uv'
fit_cont = True

#for each arm used define the line(s) to be fit and the arm's resolution in AA
arm_dict = {'uv':('OII_doub',1.68), 'orange':('OIII_Hb_trip_abs', 4.04)}
line_ID = arm_dict[arm][0]

ppxf_folders = glob.glob('/Volumes/Briana_mac3/HPS/LRS2_reduction_greg/ppxf_results/HPS*')
#ppxf_folders = ppxf_folders[1:3]
names = [i.split('/')[-1] for i in ppxf_folders]

hps = pd.read_csv('/Volumes/Briana_mac3/Green_Peas/HPS_cat_table.dat')

#sigma_guess  = (arm_dict[arm][2] - arm_dict[arm][1])/2.0
sigma_guess = np.sqrt(arm_dict[arm][1])
inten_guess  = 4.0
m_guess, b_guess = 0.0, 0.0
nwalkers = 200
nchains = (200, 1000) 

pd_dataframe = '/Volumes/Briana_mac3/HPS/LRS2_reduction_greg/hps_line_fits.csv'
make_dataframe = False

#+++++++++++++++++++ end user defined variables +++++++++++++++++++++#

#*****************************#
# Initialize Pandas Dataframe #
#*****************************#
columns = ['[OII]3726', '[OII]3729', '[Hb]4861', '[OIII]4959', '[OIII]5007', '[OII]3726_e','[OII]3729_e', '[Hb]4861_e', '[OIII]4959_e', '[OIII]5007_e']
if make_dataframe:
	index = names
	df = pd.DataFrame(index=index, columns=columns)

	#assign the dtypes to be float for the line fluxes
	#and objects for the errors since they will be a list of 2 values
	dtypes = {k: float for k in columns[:4]}
	dtypes.update({k: object for k in columns[5:]})
	df = df.astype(dtypes)

else:
	df = pd.read_csv(pd_dataframe, index_col=0)
	dtypes = {k: float for k in columns[:4]}
	dtypes.update({k: object for k in columns[5:]})
	df = df.astype(dtypes)

# print df
# print '\n'

for i in range(len(names)):

	obj = names[i]
	ppxf_path = ppxf_folders[i]
	gasfit_file = glob.glob(op.join(ppxf_path, obj+'*'+arm+'_GASFIT.npy'))
	galfit_file = glob.glob(op.join(ppxf_path, obj+'*'+arm+'_BESTFIT.npy'))
	newgal_file = glob.glob(op.join(ppxf_path, obj+'*'+arm+'_gal.npy'))
	wl_sol_file = glob.glob(op.join(ppxf_path, obj+'*'+arm+'_wave.npy'))

	gasfit = np.load(gasfit_file[0])
	galfit = np.load(galfit_file[0])
	newgal = np.load(newgal_file[0])
	wl_sol = np.load(wl_sol_file[0])

	residuals = np.subtract(newgal, galfit)
	fit       = np.subtract(galfit, gasfit)
	dat       = np.subtract(newgal, fit)
	dat = newgal

	z = float(hps.loc[hps['HPS_name'] == obj].OII_z_x.values[0])
	if fit_cont:
		model_bounds = [(z-0.002,z+0.002), (sigma_guess-0.1, sigma_guess+0.1), (0.0, 20.0), (0.0, 20.0), (-0.01, 0.01), (0.0, 3.0)] # z, sig, inten1, (inten2), m, b
		thetaGuess   = [z, sigma_guess, inten_guess, inten_guess, m_guess, b_guess] # z, sig, inten1, (inten2), m, b
	else:
		model_bounds = [(z-0.002,z+0.002), (sigma_guess-0.1, sigma_guess+1.0), (0.0, 20.0), (0.0, 20.0)] #z, sig, inten1, (inten2)
		thetaGuess   = [z, sigma_guess, inten_guess, inten_guess] # z, sig, inten1, (inten2)
	ndim = len(thetaGuess)

	if arm == 'orange':
		abs_params = (3.0, 1.0)
		#abs_params = float(hps.loc[hps['HPS_name'] == obj].Hb_abs_params.values[0])
	else:
		abs_params = (0.0, 0.0)
 
	# trim the data and residual spectra around the line(s) to be fit
	dat, residuals, wl_sol = mlf.trim_spec_for_model(line_ID, dat, residuals, wl_sol, z)

	print 'OBJECT ' + obj + ' z:'+str(z)

	#define the arguments containing the observed data for emcee
	args=(wl_sol, dat, residuals)

	#build an MCMC object for that line
	OII_MCMC = elef.MCMC_functions(obj, line_ID, model_bounds, args, fit_cont, abs_params)
	#call to run emcee
	flat_samp, mc_results = OII_MCMC.run_emcee(ndim, nwalkers, nchains, thetaGuess)
	r = [i[0] for i in mc_results]
	print mc_results
	#calculate integrated flux of lines
	flux, flux_err = OII_MCMC.calculate_flux()
	#plot the results of the emcee
	OII_MCMC.plot_results()
	#write the results to a pandas dataframe
	new_df = OII_MCMC.write_results(df)

new_df.to_csv(pd_dataframe)
print new_df










