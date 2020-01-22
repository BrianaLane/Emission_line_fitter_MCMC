import numpy as np
import pandas as pd 
import os.path as op
import glob as glob
import matplotlib.pyplot as plt
from astropy.io import fits

import model_line_functions as mlf
import emission_line_emcee_functions as elef

#************************#
# User Defined Variables #
#************************#

pd_dataframe = '/Volumes/B_SS/HDX_OIII/HDX_O3_dataframes/hdx_oiii_line_fits.csv'
data_path = '/Volumes/B_SS/HDX_OIII/LRS2_data'
plot_path = data_path+'/ELF_plots'

make_dataframe = True
fit_cont = True

#*****************************#
# Initialize Pandas Dataframe #
#*****************************#
if make_dataframe:
	df = pd.read_csv('/Volumes/B_SS/HDX_OIII/hdx_oiii_lrs2.csv')
	line_columns = ['[OII]3726', '[OII]3729', '[NeIII]3870', '[OIII]4363', '[Hb]4861', 
				'[OIII]4959', '[OIII]5007', '[OI]6300', '[NII]6549', '[Ha]6562', 
				'[NII]6583', '[SII]6717', '[SII]6731']
	err_columns =  ['[OII]3726_e','[OII]3729_e', '[NeIII]3870_e', '[OIII]4363_e', '[Hb]4861_e', 
				'[OIII]4959_e', '[OIII]5007_e', '[OI]6300_e', '[NII]6549_e', '[Ha]6562_e', 
				'[NII]6583_e', '[SII]6717_e', '[SII]6731_e']
	for c in line_columns:
		df[c] = np.NaN
	for e in err_columns:
		df[e] = None

else:
	df = pd.read_csv(pd_dataframe)

print(df.columns)
print(df.head(4))
print('\n')

#*********************#
# Define lines to fit #
#*********************#
#define the wavelength limits of each LRS2 channel
lrs2_dict = {'uv':    {'wave':(3700, 4700),  'res':1.68},
			'orange': {'wave':(4600, 7000),  'res':4.04},
			'red':    {'wave':(6500, 8420),  'res':3.00},
			'farred': {'wave':(8180, 10500), 'res':3.67}}

#for each arm used define the line(s) to be fit and the arm's resolution in AA
mod_dict = {'OII_doub'    :3727, 
			'NeIII'       :3869,
			'OIII_Te'     :4363,
			'OIII_Hb_trip':4861,
			'OI'          :6300,
			'NII_Ha_trip' :6562,
			'SII_doub'    :6717}


#++++++++++++++++++++++++++++++ end user defined variables ++++++++++++++++++++++++++++++++#

#*******************************#
# Define model fitting function #
#*******************************#

def fit_model(line_mod, obj, arm, chan, num):
	lines = mlf.line_dict[line_mod]['lines']
	z = obj['z']
	print(' ',line_mod, arm)

	#find correct fits file
	prop_name = obj['Proposal']
	obs_date = str(int(obj['Date']))
	obs_num = '{:07}'.format(int(obj[chan]))
	spec_lis = glob.glob(data_path+'/'+prop_name+'/spectrum_'+obs_date+'_'+obs_num+'_exp01_'+arm+'.fits')

	#load in fits file information 
	dat_tot, hdr = fits.getdata(spec_lis[0], 0, header=True)
	dat  = dat_tot[1,:]*1e16
	err  = dat_tot[3,:]*1e16 #for OIII new observations
	wave = dat_tot[0,:]

	# trim the data and residual spectra around the line(s) to be fit
	dat, err, wave = mlf.trim_spec_for_model(line_mod, dat, err, wave, z)

	sigma_guess = lrs2_dict[arm]['res']/2*np.sqrt(2*np.log(2))
	inten_guess  = 2.0
	m_guess, b_guess = 0.0, 0.0
	nwalkers = 200
	nchains = (300, 1000) 

	if len(mlf.line_dict[line_mod]['lines']) == 1:
		model_bounds = [(z-0.001,z+0.001), (sigma_guess-0.5, sigma_guess+0.5), (-100.0, 1000.), (-0.01, 0.01), (0.0, 3.0)] # z, sig, inten1, (inten2), m, b
		thetaGuess   = [z, sigma_guess, inten_guess, m_guess, b_guess] # z, sig, inten1, (inten2), m, b
	else:
		model_bounds = [(z-0.001,z+0.001), (sigma_guess-0.5, sigma_guess+0.5), (0.0, 1000.), (0.0, 1000.), (-0.01, 0.01), (0.0, 3.0)] # z, sig, inten1, (inten2), m, b
		thetaGuess   = [z, sigma_guess, 0.0, inten_guess, m_guess, b_guess] # z, sig, inten1, (inten2), m, b

	ndim = len(thetaGuess)
	args=(wave, dat, err)

	#build an MCMC object for that line
	line_MCMC = elef.MCMC_functions(num, line_mod, model_bounds, args, fit_cont)
	#call to run emcee
	flat_samp, mc_results = line_MCMC.run_emcee(ndim, nwalkers, nchains, thetaGuess)

	#calculate integrated flux of lines
	flux, flux_err = line_MCMC.calculate_flux()
	print '	Lines:',lines
	print '	Flux:',flux
	print '	flux_err:',flux_err
	print '\n'

	#plot the results of the emcee
	line_MCMC.plot_results(name=plot_path+'/ELF_'+str(obj['Object'])+'_'+line_mod)

	#update dataframe with flux values and errors 
	for l in range(len(lines)):
		col   = lines[l]
		col_e = lines[l]+'_e'
		val   = np.round(flux[l],3)
		val_e = [np.round(flux_err[0][l],3), np.round(flux_err[1][l],3)]

		df.at[d, col] = val
		df.at[d, col_e] = val_e


#*********************#
# Run for all objects #
#*********************#

for d in range(len(df)):
	obj = df.iloc[d]
	z = obj['z']
	print('OBJECT ' + str(obj['Object']) + ' z:'+str(z))
	#redshift the values in the model dictionary
	new_dict = {keys:values*(1+z) for keys, values in mod_dict.items()}

	if obj['B_Spectrum']==True:
		uv_models = [key for (key, value) in new_dict.items() if value <= 4650]
		or_models = [key for (key, value) in new_dict.items() if (value > 4650) & (value <= 6750)]
		if len(uv_models) > 0:
			for mod in uv_models:
				fit_model(mod, obj, 'uv', 'blue_obs', d)
		if len(or_models) > 0:
			for mod in or_models:
				fit_model(mod, obj, 'orange', 'blue_obs', d)

	if obj['R_Spectrum']==True:
		re_models = [key for (key, value) in new_dict.items() if (value > 6750) & (value <= 8300)]
		fr_models = [key for (key, value) in new_dict.items() if (value > 8300)]
		if len(re_models) > 0:
			for mod in re_models:
				fit_model(mod, obj, 'red', 'red_obs', d)
		if len(fr_models) > 0:
			for mod in fr_models:
				fit_model(mod, obj, 'farred', 'red_obs', d)

df.to_csv(pd_dataframe)
print df





