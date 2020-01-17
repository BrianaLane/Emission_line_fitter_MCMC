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

arm = 'uv'
#pd_dataframe = '/Volumes/Bri_USB/hps_lrs2_fits_wNe_4.csv'
pd_dataframe = '/Users/Briana/Documents/Grad_School/HPS/HPS_dataframes/hps_lsr2_fits_wNe_OIIIobs.csv'
make_dataframe = False

#get the data
spec_uv = glob.glob('/Volumes/BRIANA_3T/external_backup_3/HPS/LRS2_reduction_greg/reductions3/20*/lrs2/lrs2*/exp*/lrs2/spectrum_uv.fits')
spec_or = glob.glob('/Volumes/BRIANA_3T/external_backup_3/HPS/LRS2_reduction_greg/reductions3/20*/lrs2/lrs2*/exp*/lrs2/spectrum_orange.fits')
spec_re = glob.glob('/Volumes/BRIANA_3T/external_backup_3/HPS/LRS2_reduction_greg/reductions3/20*/lrs2/lrs2*/exp*/lrs2/spectrum_red.fits')
spec_fr = glob.glob('/Volumes/BRIANA_3T/external_backup_3/HPS/LRS2_reduction_greg/reductions3/20*/lrs2/lrs2*/exp*/lrs2/spectrum_farred.fits')

# spec_uv = glob.glob('/Users/Briana/Documents/Grad_School/HPS/LRS2_reductions/OIII_objects/fit_spec/spectrum*_uv.fits')
# spec_or = glob.glob('/Users/Briana/Documents/Grad_School/HPS/LRS2_reductions/OIII_objects/fit_spec/spectrum*_orange.fits')
# spec_re = glob.glob('/Users/Briana/Documents/Grad_School/HPS/LRS2_reductions/OIII_objects/fit_spec/spectrum*_red.fits')
# spec_fr = glob.glob('/Users/Briana/Documents/Grad_School/HPS/LRS2_reductions/OIII_objects/fit_spec/spectrum*_farred.fits')

fit_cont = True

#for each arm used define the line(s) to be fit and the arm's resolution in AA
arm_dict = {'uv':{'line':'OII_doub', 'res':1.68, 'dat':spec_uv}, 
			'uv2':{'line':'NeIII', 'res':1.68, 'dat':spec_uv},
			'orange':{'line':'OIII_Hb_trip', 'res':4.04, 'dat':spec_or},
			'red':{'line':'NII_Ha_trip', 'res':3.00, 'dat':spec_re},
			'farred':{'line':'NII_Ha_trip', 'res':3.67, 'dat':spec_fr}}

print(len(spec_uv), len(spec_or))

dates = [i.split('/')[-6] for i in arm_dict[arm]['dat']]
obj_num = [int(i.split('/')[-4].split('s2')[1]) for i in arm_dict[arm]['dat']]
exp_num = [i.split('/')[-3] for i in arm_dict[arm]['dat']]

# dates = [i.split('/')[-1].split('_')[1] for i in arm_dict[arm]['dat']] #4
# obj_num = [int(i.split('/')[-1].split('_')[2]) for i in arm_dict[arm]['dat']] #6
# exp_num = [i.split('/')[-1].split('_')[3] for i in arm_dict[arm]['dat']] #7

# dates = [i.split('/')[4] for i in arm_dict[arm]['dat']] #4
# obj_num = [int(i.split('/')[6].split('s2')[1]) for i in arm_dict[arm]['dat']] #6
# exp_num = [i.split('/')[7] for i in arm_dict[arm]['dat']] #7

hps = pd.read_csv('/Users/Briana/Documents/Grad_School/HPS/HPS_dataframes/HPS_cat_table.dat')

sigma_guess = arm_dict[arm]['res']/2*np.sqrt(2*np.log(2))
inten_guess  = 2.0
m_guess, b_guess = 0.0, 0.0
nwalkers = 200
nchains = (300, 1000) 

line_ID = arm_dict[arm]['line']
lines   = mlf.line_dict[line_ID]['lines']
index = np.arange(len(arm_dict[arm]['dat']))

#+++++++++++++++++++ end user defined variables +++++++++++++++++++++#

#*****************************#
# Initialize Pandas Dataframe #
#*****************************#
columns = ['HPS_name', 'Obs_ID', '[OII]3726', '[OII]3729', '[NeIII]3870',  '[Hb]4861', '[OIII]4959', '[OIII]5007', '[NII]6549', '[Ha]6562', '[NII]6583', '[OII]3726_e','[OII]3729_e', '[NeIII]3870_e', '[Hb]4861_e', '[OIII]4959_e', '[OIII]5007_e', '[NII]6549_e', '[Ha]6562_e', '[NII]6583_e']
if make_dataframe:
	df = pd.DataFrame(index=index, columns=columns)

else:
	df = pd.read_csv(pd_dataframe)

#assign the dtypes to be float for the line fluxes
#and objects for the errors since they will be a list of 2 values
dtypes = {k: object for k in columns[:1]}
dtypes.update({k: float for k in columns[2:10]})
dtypes.update({k: object for k in columns[11:]})
df = df.astype(dtypes)

print df.head(4)
print '\n'

for i in range(len(index)):

	files = arm_dict[arm]['dat']
	obs_ID = dates[i]+'_obs'+str(obj_num[i])+'_'+str(exp_num[i])
	print obs_ID

	dat_tot, hdr = fits.getdata(files[i], 0, header=True)
	dat  = dat_tot[1,:]*1e16
	#err  = dat_tot[3,:]*1e16 #for OIII new observations
	err  = dat_tot[2,:]*1e16 #for old observations
	wave = dat_tot[0,:]

	#get header parameters for that object
	obj = hdr['OBJECT'].split('_')[0][0:16]
	z = float(hps.loc[hps['HPS_name'] == obj].OII_z_x.values[0])
	if np.isnan(z):
		print 'USING OIII z'
		z = float(hps.loc[hps['HPS_name'] == obj].OIII1_z.values[0])

	print obj, z

	#if obj == 'HPS100021+021223':
	if obj == 'HPS030655+000213':

		if fit_cont:
			if arm == 'uv2':
				print sigma_guess, z
				model_bounds = [(z-0.001,z+0.001), (sigma_guess-0.5, sigma_guess+0.5), (-100.0, 1000.), (-0.01, 0.01), (0.0, 3.0)] # z, sig, inten1, (inten2), m, b
				thetaGuess   = [z, sigma_guess, inten_guess, m_guess, b_guess] # z, sig, inten1, (inten2), m, b
			else:
				print sigma_guess, z
				model_bounds = [(z-0.001,z+0.001), (sigma_guess-0.5, sigma_guess+0.5), (0.0, 1000.), (0.0, 1000.), (-0.01, 0.01), (0.0, 3.0)] # z, sig, inten1, (inten2), m, b
				thetaGuess   = [z, sigma_guess, 0.0, inten_guess, m_guess, b_guess] # z, sig, inten1, (inten2), m, b
		else:
			if arm == 'uv2':
				model_bounds = [(z-0.001,z+0.001), (sigma_guess-0.5, sigma_guess+0.5), (-100.0, 1000.)] #z, sig, inten1, (inten2)
				thetaGuess   = [z, sigma_guess, inten_guess] # z, sig, inten1, (inten2)
			else:
				model_bounds = [(z-0.001,z+0.001), (sigma_guess-0.5, sigma_guess+0.5), (0.0, 1000.), (0.0, 1000.)] #z, sig, inten1, (inten2)
				thetaGuess   = [z, sigma_guess, inten_guess, inten_guess] # z, sig, inten1, (inten2)
		ndim = len(thetaGuess)
	 
		# trim the data and residual spectra around the line(s) to be fit
		dat, err, wave = mlf.trim_spec_for_model(line_ID, dat, err, wave, z)

		print 'OBJECT ' + obj + ' z:'+str(z)

		#define the arguments containing the observed data for emcee
		args=(wave, dat, err)

		#build an MCMC object for that line
		OII_MCMC = elef.MCMC_functions(i, line_ID, model_bounds, args, fit_cont)
		#call to run emcee
		flat_samp, mc_results = OII_MCMC.run_emcee(ndim, nwalkers, nchains, thetaGuess)

		#calculate integrated flux of lines
		flux, flux_err = OII_MCMC.calculate_flux()
		print 'Lines:',lines
		print 'Flux:',flux
		print 'flux_err:',flux_err
		print '\n'

		#plot the results of the emcee
		OII_MCMC.plot_results(name=obj+'_'+line_ID)

		if make_dataframe:
			#write the results to a pandas dataframe
			for l in range(len(lines)):
				col   = lines[l]
				col_e = lines[l]+'_e'
				val   = np.round(flux[l],3)
				val_e = [np.round(flux_err[0][l],3), np.round(flux_err[1][l],3)]

				df.at[i, col] = val
				df.at[i, col_e] = val_e

			df.at[i, 'HPS_name'] = obj
			df.at[i, 'Obs_ID'] = obs_ID

		else:
			#write the results to a pandas dataframe
			for l in range(len(lines)):
				col   = lines[l]
				col_e = lines[l]+'_e'
				row   = df.index[(df['HPS_name'] == obj) & (df['Obs_ID'] == obs_ID)].tolist()
				#row   = df.index[(df['HPS_name'] == obj)].tolist()
				val   = np.round(flux[l],3)
				val_e = [np.round(flux_err[0][l],3), np.round(flux_err[1][l],3)]

				print row, col

				df.at[row[0], col] = val
				df.at[row[0], col_e] = val_e

#df.to_csv(pd_dataframe)
#print df










