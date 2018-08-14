import numpy as np
import string as st

#*****************************#
# Continuum Subtracted Models #
#*****************************#

def OII_gaussian(x, theta):
	z, sig, inten = theta
	mu = 3727*(1+z)
	return inten * (np.exp(-0.5*np.power(x - mu, 2.) / (np.power(sig, 2.))))

def Hb_gaussian(x, theta):
	z, sig, inten = theta
	mu = 4861*(1+z)
	return inten * (np.exp(-0.5*np.power(x - mu, 2.) / (np.power(sig, 2.))))

#fit the OIII doublet and fixes the ratio to 2.89
def OIII_doub_gaussian(x, theta):
	z, sig, inten = theta
	mu1 = 5007*(1+z)
	mu2 = 4959*(1+z)
	return (inten * (np.exp(-0.5*np.power(x - mu1, 2.) / (np.power(sig, 2.))))) + \
		((inten/2.98) * (np.exp(-0.5*np.power(x - mu2, 2.) / (np.power(sig, 2.)))))

#fits independent SII doublet, no fixed ratio so get 2 intensity values
def SII_doub_gaussian(x, theta):
	z, sig, inten1, inten2 = theta
	mu1 = 6731*(1+z)
	mu2 = 6717*(1+z)
	return (inten1 * (np.exp(-0.5*np.power(x - mu1, 2.) / (np.power(sig, 2.))))) + \
		(inten2 * (np.exp(-0.5*np.power(x - mu2, 2.) / (np.power(sig, 2.)))))

#fit the OIII doublet and fixes the ratio to 2.89. Also fits Hb which is blueward of the doublet 
def OIII_Hb_trip_gaussian(x, theta):
	z, sig, inten1, inten2 = theta
	mu1 = 5007*(1+z)
	mu2 = 4959*(1+z)
	mu3 = 4861*(1+z)
	return ( inten1 * (np.exp(-0.5*np.power(x - mu1, 2.) / (np.power(sig, 2.))))) + \
		((inten1/2.98) * (np.exp(-0.5*np.power(x - mu2, 2.) / (np.power(sig, 2.))))) + \
		(inten2 * (np.exp(-0.5*np.power(x - mu3, 2.) / (np.power(sig, 2.)))))

#fit the NII doublet and fixes the ratio to 2.5 and also fits Ha which is between the lines 
def NII_Ha_trip_gaussian(x, theta):
	z, sig, inten1, inten2 = theta
	mu1 = 6583*(1+z)
	mu2 = 6562*(1+z)
	mu3 = 6549*(1+z)
	return ( inten1 * (np.exp(-0.5*np.power(x - mu1, 2.) / (np.power(sig, 2.))))) + \
		(inten2 * (np.exp(-0.5*np.power(x - mu2, 2.) / (np.power(sig, 2.))))) + \
		((inten1/2.5) * (np.exp(-0.5*np.power(x - mu3, 2.) / (np.power(sig, 2.)))))

#********************#
# Trim Spec Function #
#********************#

line_dict = {'OII':			{'mod':OII_gaussian,			'lines':['[OII]3727'],				'trim':(3700.0,3760.0)},
			'Hb':			{'mod':Hb_gaussian,				'lines':['[Hb]4861'],				'trim':(4830.0, 4890.0)},
			'OIII_doub':	{'mod':OIII_doub_gaussian,		'lines':['[OIII]5007'],				'trim':(4930.0, 5050.0)},
			'SII_doub':		{'mod':SII_doub_gaussian,		'lines':['[SII]6731','[SII]6717'],	'trim':(6700.0, 6750.0)},
			'OIII_Hb_trip':	{'mod':OIII_Hb_trip_gaussian,	'lines':['[OIII]5007','[Hb]4861'],	'trim':(4830.0, 5050.0)},
			'NII_Ha_trip':	{'mod':NII_Ha_trip_gaussian,	'lines':['[NII]6583','[Ha]6562'],	'trim':(6520.0, 6610.0)}}

def trim_spec_for_model(line, dat, residuals, wl):
	min_wave = line_dict[line]['trim'][0]
	max_wave = line_dict[line]['trim'][1]
	inds = np.where((min_wave < wl) & (wl < max_wave))

	dat       = dat[:, inds]
	residuals = residuals[:, inds]
	wl = wl[inds]

	return np.vstack(dat), np.vstack(residuals), wl
