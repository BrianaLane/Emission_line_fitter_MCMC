import numpy as np

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
	mu2 = 4939*(1+z)
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

line_dict = {'OII':			{'mod':OII_gaussian,			'wl':[3727]},
			'Hb':			{'mod':Hb_gaussian,				'wl':[4861]},
			'OIII_doub':	{'mod':OIII_doub_gaussian,		'wl':[4959, 5007]},
			'SII_doub':		{'mod':SII_doub_gaussian,		'wl':[6717, 6731]},
			'OIII_Hb_trip':	{'mod':OIII_Hb_trip_gaussian,	'wl':[4861, 4959, 5007]},
			'NII_Ha_trip':	{'mod':NII_Ha_trip_gaussian,	'wl':[6549, 6562, 6583]}}

def trim_spec_for_model(line, dat, residuals, wl):
	min_wave = line_dict[line]['wl'][0] - 50
	max_wave = line_dict[line]['wl'][-1] + 50
	inds = np.where((min_wave < wl) & (wl < max_wave))

	dat       = dat[:, inds]
	residuals = residuals[:, inds]
	wl = wl[inds]

	return np.vstack(dat), np.vstack(residuals), wl
