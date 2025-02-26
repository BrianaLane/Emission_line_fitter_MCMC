import numpy as np
import pandas as pd 
import os.path as op
import glob as glob
from scipy import interpolate
from astropy.io import fits
import time
import pickle

import model_line_functions as mlf
import emission_line_emcee_functions as elef
import matplotlib.pyplot as plt

master_spec_path = 'M82_master_spectra/'

master_spec_R  = fits.getdata(master_spec_path+'all_fib_spec_R_scaled.fits')
master_spec_R_wcont  = fits.getdata(master_spec_path+'all_fib_spec_R.fits')
master_spec_R_e = fits.getdata(master_spec_path+'all_fib_spec_R_e.fits')
wave_R = np.load(master_spec_path+'WavelengthSolu_M82_R.npy')

master_spec_B  = fits.getdata(master_spec_path+'all_fib_spec_B_scaled.fits')
master_spec_B_wcont  = fits.getdata(master_spec_path+'all_fib_spec_B.fits')
master_spec_B_e = fits.getdata(master_spec_path+'all_fib_spec_B_e.fits')
wave_B = np.load(master_spec_path+'WavelengthSolu_M82_B.npy')

master_df = pd.read_csv(master_spec_path+'M82_dataframe_master.csv', low_memory=False)

master_df['[Hb]4861_test'] = np.zeros(len(master_df['[Hb]4861']))
master_df['[Hb]4861_e_test'] = np.zeros(len(master_df['[Hb]4861']))

df_B = master_df[(master_df['chan']== 'B') & (master_df['field'] != 'F4b')].reset_index()
df_R = master_df[master_df['chan']== 'R'].reset_index()

#define the sigma guess based on the resolution
z_m82 = 0.000677
vp_res = 5.3 #AA
sigma_guess = np.sqrt(vp_res)

#define initial parameter guess (z, sig, inten1, (inten2), m, b)
# thetaGuess = [z_m82, sigma_guess, 10.0, 10.0, 0.0, 0.0] #doublet
thetaGuess = [z_m82, sigma_guess, 10.0, 0.0, 0.0] 

#define the model bounds for thetaGuess (z, sig, inten1, (inten2), m, b)
#model_bounds = [(0.0005,0.002), (sigma_guess-0.5, sigma_guess+1.5), (-10.0, 1000.0), (-10.0, 1000.0), (-0.01, 0.01), (0.0, 3.0)] #doublet
model_bounds = [(0.0005,0.002), (sigma_guess-0.5, sigma_guess+1.5), (-10.0, 1000.0), (-0.01, 0.01), (0.0, 3.0)]

#number of dimensions is len of parameters (thetaGuess)
ndim = len(thetaGuess)

#define mcmc parameters
nchains = (500, 200) 
nwalkers = 100

spec_scale = 1e17 #define scale factor for fitting fluxes

line='Hb'
spec_trim, err_trim, wl_sol_trim = mlf.trim_spec_for_model(line, master_spec_B, master_spec_B_e, wave_B, z_m82)

for f in range(5):

    ind = int(df_B['master_ind_B'].values[f])
    print('spec:', f, ';ind:', ind)

    spec = spec_trim[ind]*spec_scale
    err_spec = err_trim[ind]*spec_scale
    
    args=(wl_sol_trim, spec, err_spec)

    #build an MCMC object for that line
    line_MCMC = elef.MCMC_functions(f, line, model_bounds, args, fit_cont=True)
    #call to run emcee
    flat_samp, mc_results = line_MCMC.run_emcee(ndim, nwalkers, nchains, thetaGuess)
    #calculate integrated flux of lines
    flux, flux_err = line_MCMC.calculate_flux()
    
    # #write the results to a pandas dataframe
    new_df = line_MCMC.write_results(master_df, ind=ind)


master_df.to_csv(master_spec_path+'M82_dataframe_master_copy.csv')

