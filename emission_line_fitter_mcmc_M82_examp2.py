import numpy as np
import pandas as pd 
import os.path as op
import glob as glob
from scipy import interpolate
from astropy.io import fits
import time

import model_line_functions as mlf
import emission_line_emcee_functions as elef
import matplotlib.pyplot as plt

#*********************************#
# Load in M82 dataframe and paths #
#*********************************#

df = pd.read_csv('/Users/Briana/Documents/Grad_School/M82/M82_dataframes/M82_dataframe_Wabs.csv',index_col=0)
df.drop(columns=['Unnamed: 0.1'], inplace=True)

df['[OI]6300'] = np.zeros(len(df['[OII]3727']))
df['[OI]6300_e'] = np.zeros(len(df['[OII]3727_e']))

df['[OI]6300_e'] = df['[OI]6300_e'].astype(object)
df['[OII]3727_e'] = df['[OII]3727_e'].astype(object)
df['[Hb]4861_e'] = df['[Hb]4861_e'].astype(object)
df['[OIII]5007_e'] = df['[OIII]5007_e'].astype(object)
df['[Ha]6562_e'] = df['[Ha]6562_e'].astype(object)
df['[NII]6583_e'] = df['[NII]6583_e'].astype(object)
df['[SII]6717_e'] = df['[SII]6717_e'].astype(object)
df['[SII]6731_e'] = df['[SII]6731_e'].astype(object)

print df.columns

data_path = '/Volumes/B_SS/M82/VP_M82_data'
wave_path = '/Volumes/B_SS/M82/wave_sols'
ppxf_path = '/Users/Briana/Documents/Grad_School/M82/ppxf/pPXF_VP/results/combine/'

model_cont = np.load(ppxf_path+'resultsM82_F1_combine_spec_CONFIT.npy')
model_wave = np.load(ppxf_path+'resultsM82_F1_combine_spec_wave.npy')

#************************#
# User Defined Variables #
#************************#

band = 'B'
z = 0.000677
res = 5.3 #AA

#mod_dict = {'B': ['OII','OIII_Hb_trip'], 'R': ['NII_Ha_trip', 'SII_doub']}
mod_dict = {'B': ['OI'], 'R': ['NII_Ha_trip', 'SII_doub']}
abs_dict = {'B':'Hb', 'R':'Ha'}

abs_line = abs_dict[band]
mod_lis = mod_dict[band]

sigma_guess = np.sqrt(res)

nchains = (200, 500) #(200, 500)
nwalkers = 100

#+++++++++++++++++++ end user defined variables +++++++++++++++++++++#

#*****************************#
# Initialize Pandas Dataframe #
#*****************************#


dither_lis = df[df['chan']==band]['dither'].unique()
print 'Found', len(dither_lis), 'dithers for band ', band

for d in range(len(dither_lis)):
#for d in [66]:

	t0 = time.time()

	dith = dither_lis[d]
	dither_name = "{:04d}".format(dith)
	dither_num = df[(df['dither'] == dith) & (df['chan']==band)]['dith_num'].iloc[0]
	date = df[(df['dither'] == dith) & (df['chan']==band)]['date'].iloc[0]
	field = df[(df['dither'] == dith) & (df['chan']==band)]['field'].iloc[0]

	d_file_name = data_path+'/*/*'+str(date)+'*/*/M82_'+str(field)+'/FcalFeSpesvp'+str(dither_name)+'.fits'
	print(d_file_name)
	d_file = glob.glob(d_file_name)[0]

	e_file_name = data_path+'/*/*'+str(date)+'*/*/M82_'+str(field)+'/e.FcalFeSpesvp'+str(dither_name)+'.fits'
	e_file = glob.glob(e_file_name)[0]

	wave_file = wave_path+'/WavelengthSolu_M82_'+str(field)+'_'+str(band)+'.npy'

	dat_big, hdr     = fits.getdata(d_file, 0, header=True)
	dat_e, hdr_e = fits.getdata(e_file, 0, header=True)
	wave = np.load(wave_file)

	#plt.plot(wave, dat_big[0], color='purple', label='dat')

	for m in range(len(mod_lis)):

		line = mod_lis[m]
		lines_lis = mlf.line_dict[line]['lines']
		num_lines = len(lines_lis)

		if num_lines > 1:
			thetaGuess = [z, sigma_guess, 4.0, 4.0, 0.0, 0.0] # z, sig, inten1, (inten2), m, b
			model_bounds = [(0.0005,0.002), (sigma_guess-0.5, sigma_guess+1.5), (-1.0, 20.0), (-10.0, 1000.0), (-0.01, 0.01), (0.0, 3.0)] #z, sig, inten1, (inten2), m, b
		else:
			thetaGuess = [z, sigma_guess, 4.0, 0.0, 0.0] # z, sig, inten1, m, b
			model_bounds = [(0.0005,0.002), (sigma_guess-0.5, sigma_guess+1.5), (-1.0, 20.0), (-0.01, 0.01), (0.0, 3.0)] #z, sig, inten1, m, b

		ndim = len(thetaGuess)

		# trim the data and residual spectra around the line(s) to be fit
		dat, residuals, wl_sol = mlf.trim_spec_for_model(line, dat_big, dat_e, wave, z)

		f_int = interpolate.interp1d(model_wave, model_cont, fill_value='extrapolate')
		model_interp = f_int(wl_sol)

		spec_scale = 1e17
		dat = dat*spec_scale
		residuals = residuals*spec_scale
		model_interp = model_interp*spec_scale

		plt.plot(wl_sol, model_interp, color='yellow', label='mod')

		for f in range(np.shape(dat)[0]):
		#for f in range(5):

			ind = df[(df['dither']==dith) & (df['chan']==band) & (df['fib_num']==f)].index

			print ind.values, dith, dither_num, date, field, band, f, line

			scale_ratio = df.loc[ind][abs_dict[band]+'_scale'].values[0]
			scaled_model = model_interp*scale_ratio

			spec = dat[f] - scaled_model

			args=(wl_sol, spec, residuals[f])

			# plt.plot(wl_sol, spec, color='red', label='spec')
			# plt.plot(wl_sol, residuals[f], color='orange', label='resid')
			# plt.plot(wl_sol, scaled_model, color='blue', label='scal_mod')
			# plt.plot(wl_sol, dat[f], color='grey', label='dat')
			# plt.legend()
			# plt.show()

			#build an MCMC object for that line
			line_MCMC = elef.MCMC_functions(f, line, model_bounds, args, fit_cont=True)
			#call to run emcee
			flat_samp, mc_results = line_MCMC.run_emcee(ndim, nwalkers, nchains, thetaGuess)
			#calculate integrated flux of lines
			flux, flux_err = line_MCMC.calculate_flux()
			#plot the results of the emcee
			#line_MCMC.plot_results()
			# #write the results to a pandas dataframe
			new_df = line_MCMC.write_results(df, ind=ind)

			for l in range(len(lines_lis)):
				col   = lines_lis[l]
				col_e = lines_lis[l]+'_e'
				row   = ind
				val   = np.round(flux[l],3)
				val_e = str([np.round(flux_err[0][l],3), np.round(flux_err[1][l],3)])

				df.at[row, col] = val
				df.at[row, col_e] = val_e

	t1 = time.time()
	print 'TIME: ', t1-t0

	df.to_csv('/Users/Briana/Documents/Grad_School/M82/M82_dataframes/M82_dataframe_Wfits_oi.csv')

df.to_csv('/Users/Briana/Documents/Grad_School/M82/M82_dataframes/M82_dataframe_Wfits_oi_copy.csv')
print df.loc[2942:2950,['[OI]6300', '[OI]6300_e' ]]







