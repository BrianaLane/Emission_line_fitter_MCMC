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

df_path = '/Users/Briana/Documents/Grad_School/M82/M82_dataframes/M82_dataframe_master_SII.csv'

df = pd.read_csv(df_path)

line_dict = {'[Ha]6562': {'chan':'R', 'name':'Ha'},
                 '[Hb]4861': {'chan':'B', 'name':'Hb'},
                 '[OII]3727':{'chan':'B', 'name':'OII3727'},
                 '[OIII]5007':{'chan':'B', 'name':'OIII5007'},
                 '[OI]6300':{'chan':'B', 'name':'OI6300'},
                 '[NII]6583':{'chan':'R', 'name':'NII6583'},
                 '[SII]6717':{'chan':'R', 'name':'SII6717'},
                 '[SII]6731':{'chan':'R', 'name':'SII6731'}}

for l in line_dict:
	df[l+'_z'].astype(object)
	df[l+'_sig'].astype(object)
	df[l+'_int'].astype(object)
	df[l+'_b'].astype(object)
	df[l+'_m'].astype(object)


print(df.columns.values)

data_path = '/Volumes/B_SS/M82/VP_M82_data'
wave_path = '/Volumes/B_SS/M82/wave_sols'
ppxf_path = '/Users/Briana/Documents/Grad_School/M82/ppxf/pPXF_VP/results/combine/'

model_cont = np.load(ppxf_path+'resultsM82_F1_combine_spec_CONFIT.npy')
model_wave = np.load(ppxf_path+'resultsM82_F1_combine_spec_wave.npy')

#************************#
# User Defined Variables #
#************************#

line = 'SII_doub'
band = 'R'
z = 0.000677
res = 5.3 #AA

abs_dict = {'B':'Hb', 'R':'Ha'}
sigma_guess = np.sqrt(res)

nchains = (200, 500) #(200, 500)
nwalkers = 100

line_dict = {'[OI]6300':   {'chan':'R', 'bp':(6290, 6315)},
             '[OII]3727':  {'chan':'B', 'bp':(3725, 3740)}, 
             '[Hb]4861':   {'chan':'B', 'bp':(4860, 4875)},
             '[OIII]5007': {'chan':'B', 'bp':(5005, 5017)},
             '[Ha]6562':   {'chan':'R', 'bp':(6560, 6575)},
             '[NII]6583':  {'chan':'R', 'bp':(6580, 6600)}, 
             '[SII]6717':  {'chan':'R', 'bp':(6710, 6730)},
             '[SII]6731':  {'chan':'R', 'bp':(6730, 6745)}}

#+++++++++++++++++++ end user defined variables +++++++++++++++++++++#

#*****************************#
# Initialize Pandas Dataframe #
#*****************************#


dither_ind = df[(df['chan']==band) & (df['fib_num']==1)]['dither'].index
print('Found', len(dither_ind), 'dithers for band ', band)

for d in range(len(dither_ind)):
#for d in [66]:

	t0 = time.time()

	dith_df = df.iloc[dither_ind[d]]

	dith = dith_df['dither']
	dither_name = "{:04d}".format(dith)
	dither_num = dith_df['dith_num']
	date = dith_df['date']
	field = dith_df['field']

	d_file_name = data_path+'/*/*'+str(date)+'*/*/M82_'+str(field)+'/FcalFeSpesvp'+str(dither_name)+'.fits'
	d_file = glob.glob(d_file_name)[0]

	e_file_name = data_path+'/*/*'+str(date)+'*/*/M82_'+str(field)+'/e.FcalFeSpesvp'+str(dither_name)+'.fits'
	e_file = glob.glob(e_file_name)[0]

	wave_file = wave_path+'/WavelengthSolu_M82_'+str(field)+'_'+str(band)+'.npy'

	dat_big, hdr = fits.getdata(d_file, 0, header=True)
	dat_e, hdr_e = fits.getdata(e_file, 0, header=True)
	wave = np.load(wave_file)

	#plt.plot(wave, dat_big[0], color='purple', label='dat')

	lines_lis = mlf.line_dict[line]['lines']
	num_lines = len(lines_lis)

	# trim the data and residual spectra around the line(s) to be fit
	dat, residuals, wl_sol = mlf.trim_spec_for_model(line, dat_big, dat_e, wave, z)

	f_int = interpolate.interp1d(model_wave, model_cont, fill_value='extrapolate')
	model_interp = f_int(wl_sol)

	spec_scale = 1e17
	dat = dat*spec_scale
	residuals = residuals*spec_scale
	model_interp = model_interp*spec_scale

	#plt.plot(wl_sol, model_interp, color='yellow', label='mod')

	for f in range(np.shape(dat)[0]):
	#for f in [100]:

		ind = df[(df['dither']==dith) & (df['chan']==band) & (df['field']==field) & (df['fib_num']==f)].index

		print('['+str(d+f)+']','['+str(d)+']', ind.values, dith, dither_num, date, field, band, f, line)

		scale_ratio = df.loc[ind][abs_dict[band]+'_scale'].values[0]
		scaled_model = model_interp*scale_ratio

		spec = dat[f] - scaled_model

		inten_guess_lis = []
		for l in lines_lis:
			wl_inds = np.where((line_dict[l]['bp'][0] < wl_sol) & (wl_sol < line_dict[l]['bp'][1]))
			inten_guess_lis.append((np.max(spec[wl_inds])))

		if num_lines > 1:
			thetaGuess = [z, sigma_guess, inten_guess_lis[0], inten_guess_lis[1], 0.0, 0.0] # z, sig, inten1, (inten2), m, b
			model_bounds = [(0.0005,0.002), (sigma_guess-0.5, sigma_guess+1.5), (-1.0, 20.0), (-10.0, 1000.0), (-0.01, 0.01), (0.0, 3.0)] #z, sig, inten1, (inten2), m, b
		else:
			thetaGuess = [z, sigma_guess, inten_guess_lis[0], 0.0, 0.0] # z, sig, inten1, m, b
			model_bounds = [(0.0005,0.002), (sigma_guess-0.5, sigma_guess+1.5), (-1.0, 20.0), (-0.01, 0.01), (0.0, 3.0)] #z, sig, inten1, m, b

		ndim = len(thetaGuess)

		args=(wl_sol, spec, residuals[f])

		# plt.plot(wl_sol, spec, color='red', label='spec')
		# plt.plot(wl_sol, residuals[f], color='orange', label='resid')
		# plt.plot(wl_sol, scaled_model, color='blue', label='scal_mod')
		# plt.plot(wl_sol, dat[f], color='grey', label='dat')
		# for l in range(len(lines_lis)):
		# 	plt.scatter(np.average(line_dict[lines_lis[l]]['bp']), inten_guess_lis[l], color='green', s=50)
		# plt.legend()
		# plt.show()

		#build an MCMC object for that line
		line_MCMC = elef.MCMC_functions(f, line, model_bounds, args, fit_cont=True)
		#call to run emcee
		flat_samp, mc_results = line_MCMC.run_emcee(ndim, nwalkers, nchains, thetaGuess)

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

			df.at[row, col+'_z'] = str(list(mc_results[0]))
			df.at[row, col+'_sig'] = str(list(mc_results[1]))
			df.at[row, col+'_int'] = str(list(mc_results[l+2]))
			df.at[row, col+'_m'] = str(list(mc_results[2+num_lines]))
			df.at[row, col+'_b'] = str(list(mc_results[2+num_lines+1]))

	t1 = time.time()
	print('TIME: ', t1-t0)

	df.to_csv(df_path)

#df.to_csv('/Users/Briana/Documents/Grad_School/M82/M82_dataframes/M82_dataframe_master.csv', index=False)







