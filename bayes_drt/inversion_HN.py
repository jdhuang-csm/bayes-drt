import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import pystan as stan
import warnings
import os
from copy import deepcopy
import glob

from .inversion import Inverter
from .matrices import get_basis_func
from .utils import rel_round
from . import peak_fit as pf
from .stan_models import save_pickle, load_pickle

script_dir = os.path.dirname(os.path.realpath(__file__))

warnings.simplefilter('always',UserWarning)
warnings.simplefilter('once',RuntimeWarning)

k_B = 8.617333e-5

# # load stan models so that HMC fits can be loaded
# stan_models = glob.glob(os.path.join(script_dir,'stan_model_files','*.pkl'))
# for model in stan_models:
	# load_pickle(model) 

class Inverter_HN(Inverter):
	def map_fit(self, frequencies, Z, part='both', scale_Z=True, init_from_ridge=False, nonneg=False, outliers=False,
				check_outliers=True, sigma_min=0.002, max_iter=50000, random_seed=1234, inductance_scale=1,
				outlier_lambda=10, ridge_kw={}, add_stan_data={}, model_str=None, fitY=False, SA=False, SASY=False):
		"""
		Obtain the maximum a posteriori estimate of the defined distribution(s) (and all model parameters).
		
		Parameters:
		-----------
		frequencies: array
			Measured frequencies
		Z: complex array
			Measured (complex) impedance values. Must have same length as frequencies
		part: str, optional (default: 'both')
			Which part of the impedance data to fit. Options: 'both', 'real', 'imag'
		scale_Z: bool, optional (default: True)
			If True, scale impedance by the factor sqrt(N)/std(|Z|) to normalize for magnitude and sample size
		init_from_ridge: bool, optional (default: False)
			If True, use the hyperparametric ridge solution to initialize the Bayesian fit. 
			Only valid for single-distribution fits
		nonneg: bool, optional (default: False)
			If True, constrain the DRT to non-negative values
		outliers: bool, optional (default: False)
			If True, enable outlier identification via independent error contribution variable
		sigma_min: float, optional (default: 0.002)
			Impedance error floor. This is necessary to avoid sampling/optimization errors.
			Values smaller than the default (0.002) may enable slightly closer fits of very clean data,
			but may also result in sampling/optimization errors that yield unexpected results.
		max_iter: int, optional (default: 50000)
			Maximum number of iterations to allow the optimizer to perform
		random_seed: int, optional (default: 1234)
			Random seed for optimizer
		inductance_scale: float, optional (default: 1)
			Scale (std of normal prior) of the inductance. Lower values will constrain the inductance,
			which may be helpful if the estimated inductance is anomalously large (this may occur if your
			measured impedance data does not extend to high frequencies, i.e. 1e5-1e6 Hz)
		outlier_lambda: float, optional (default: 5)
			Lambda parameter (inverse scale) of the exponential prior on the outlier error contribution.
			Smaller values will make it easier for points to be flagged as outliers 
		ridge_kw: dict, optional (default: {})
			Keyword arguments to pass to ridge_fit if init_from_ridge==True.
		"""
		# load stan model
		if model_str is None:
			model_str = 'Series_HN_pos'
			if ordered:
				model_str += '_ordered'
			model_str += '_StanModel.pkl'
		model = load_pickle(os.path.join(script_dir,'stan_model_files',model_str))
		self.stan_model_name = model_str
		model_type = model_str.split('_')[0]

		# perform scaling
		Z_scaled = self._scale_Z(Z,'map')
		self.f_train = frequencies
		
		# prepare data for stan model
		dat = self._prep_stan_data(frequencies,Z_scaled,part,model_type,sigma_min,mode='optimize',inductance_scale=inductance_scale)
		# if tau boundaries not specified, allow HN peaks to extend 2 decades beyond measurement range
		if min_tau is None:
			min_tau = np.min(1/(2*np.pi*frequencies))/100
		if max_tau is None:
			max_tau = np.max(1/(2*np.pi*frequencies))*100
		dat['min_tau_HN'] = min_tau
		dat['max_tau_HN'] = max_tau
		dat['K'] = num_HN
		# dat['ups_alpha'] = 20
		# dat['ups_beta'] = 0.5
		dat['R_scale_alpha'] = 1
		dat['R_scale_beta'] = 1
		dat.update(add_stan_data)
		self._stan_input = dat.copy()
		
		# Create tau grid for convenience, using min_tau and max_tau
		num_decades = int(np.ceil(np.log10(max_tau)) - np.floor(np.log10(min_tau)))
		for distribution in self.distributions.keys():
			self.distributions[distribution]['tau'] = np.logspace(np.log10(min_tau),np.log10(max_tau),num_decades*10 + 1)
		
		# Get initial parameter values
		iv = {}
		if np.sum([init_from_ridge,init_from_map,init_drt_fit is not None,init_values is not None]) > 1:
			raise ValueError('Only one of init_from_ridge, init_from_map, and init_drt_fit may be used')
		if init_from_ridge:
			if len(self.distributions) > 1:
				raise ValueError('Ridge initialization can only be performed for single-distribution fits')
			else:
				init = self._get_init_from_ridge(frequencies,Z,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
				iv.update(init)
		elif init_from_map:
			init = self._get_init_from_map(frequencies,Z,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
			iv.update(init)
		elif init_drt_fit is not None:
			init = self._get_init_from_drt(init_drt_fit,num_HN,inductance_scale,min_tau,max_tau,peakfit_kw)
			iv.update(init)
		else:
			# distribute lntau values uniformly
			iv['lntau_HN'] = np.linspace(np.log(min_tau*10),np.log(max_tau/10),num_HN)
			iv['R_HN'] = np.ones(num_HN)*0.1
			iv['alpha_HN'] = np.ones(num_HN)*0.95
			iv['beta_HN'] = np.ones(num_HN)*0.8
			iv['upsilon'] = np.ones(num_HN)*10
		
		if nonneg:
			raise ValueError('Outlier model not yet implemented')
			# initialize sigma_out near zero, everything else randomly
			iv['sigma_out_raw'] = np.zeros(2*len(Z)) + 0.1
			dat['so_invscale'] = outlier_lambda
			
		if init_values is not None:
			iv = init_values
		
			
		def init():
			return iv
		self._init_params = iv
				
		# print(iv)
		
		# optimize posterior
		self._opt_result = model.optimizing(dat,iter=max_iter,seed=random_seed,init=init,init_alpha=init_alpha)
		
		# extract coefficients
		self.distribution_fits = {}
		self.error_fit = {}
		if model_type in ['Series','Parallel']:
			dist_name = [k for k,v in self.distributions.items() if v['dist_type']==model_type.lower()][0]
			dist_type = self.distributions[dist_name]['dist_type']
			self.distribution_fits[dist_name] = {'R_HN':self._rescale_coef(self._opt_result['R_HN'],dist_type)}
			self.distribution_fits[dist_name]['tau_HN'] = np.exp(self._opt_result['lntau_HN'])
			self.distribution_fits[dist_name]['alpha_HN'] = self._opt_result['alpha_HN']
			self.distribution_fits[dist_name]['beta_HN'] = self._opt_result['beta_HN']
			
			self.R_inf = self._rescale_coef(self._opt_result['Rinf'],'series')
			self.inductance = self._rescale_coef(self._opt_result['induc'],'series')
		
		# store error structure parameters
		# scaled parameters
		self.error_fit['sigma_min'] = self._rescale_coef(sigma_min,'series')
		for param in ['sigma_tot','sigma_res']:
			self.error_fit[param] = self._rescale_coef(self._opt_result[param],'series')
		# unscaled parameters
		for param in ['alpha_prop','alpha_re','alpha_im']:
			self.error_fit[param] = self._opt_result[param]
		# outlier contribution
		if nonneg:
			self.error_fit['sigma_out'] = self._rescale_coef(self._opt_result['sigma_out'],'series')
		
		self.fit_type = 'map'
		
	def hn_drt_fit(self,frequencies,Z,part='both',scale_Z=True,nonneg=True,outliers=False,
		# HN parameters
		min_tau=None,max_tau=None,num_HN=5,
		# DRT eval
		drt_eval_tau=None,
		
		# Other hyperparameters
		sigma_min=0.002,inductance_scale=1,outlier_lambda=5,sigma_F_min=0.1,
		
		model_str=None,add_stan_data={},
		
		
		# initialization parameters
		init_from_ridge=False,ridge_kw={},#peakfit_kw={},
		init_from_map=False,map_kw={},
		init_values=None,
		# optimization control
		max_iter=50000,init_alpha=1e-3,random_seed=1234):
		"""
		Obtain the maximum a posteriori estimate of the defined distribution(s) (and all model parameters).
		
		Parameters:
		-----------
		frequencies: array
			Measured frequencies
		Z: complex array
			Measured (complex) impedance values. Must have same length as frequencies
		part: str, optional (default: 'both')
			Which part of the impedance data to fit. Options: 'both', 'real', 'imag'
		scale_Z: bool, optional (default: True)
			If True, scale impedance by the factor sqrt(N)/std(|Z|) to normalize for magnitude and sample size
		init_from_ridge: bool, optional (default: False)
			If True, use the hyperparametric ridge solution to initialize the Bayesian fit. 
			Only valid for single-distribution fits
		nonneg: bool, optional (default: False)
			If True, constrain the DRT to non-negative values
		outliers: bool, optional (default: False)
			If True, enable outlier identification via independent error contribution variable
		sigma_min: float, optional (default: 0.002)
			Impedance error floor. This is necessary to avoid sampling/optimization errors.
			Values smaller than the default (0.002) may enable slightly closer fits of very clean data,
			but may also result in sampling/optimization errors that yield unexpected results.
		max_iter: int, optional (default: 50000)
			Maximum number of iterations to allow the optimizer to perform
		random_seed: int, optional (default: 1234)
			Random seed for optimizer
		inductance_scale: float, optional (default: 1)
			Scale (std of normal prior) of the inductance. Lower values will constrain the inductance,
			which may be helpful if the estimated inductance is anomalously large (this may occur if your
			measured impedance data does not extend to high frequencies, i.e. 1e5-1e6 Hz)
		outlier_lambda: float, optional (default: 5)
			Lambda parameter (inverse scale) of the exponential prior on the outlier error contribution.
			Smaller values will make it easier for points to be flagged as outliers 
		ridge_kw: dict, optional (default: {})
			Keyword arguments to pass to ridge_fit if init_from_ridge==True.
		"""
		# load stan model
		if model_str is None:
			model_str = 'Series_HN-DRT_pos'
			model_str += '_StanModel.pkl'
		model = load_pickle(os.path.join(script_dir,'stan_model_files',model_str))
		self.stan_model_name = model_str
		model_type = model_str.split('_')[0]
		
		dist_name = list(self.distributions.keys())[0]
		dist_type = self.distributions[dist_name]['dist_type']

		# # perform scaling
		# Z_scaled = self._scale_Z(Z,'map')
		# self.f_train = frequencies
		# perform scaling and weighting and get matrices
		frequencies, Z_scaled, WZ_re,WZ_im,W_re,W_im, dist_mat = self._prep_matrices(frequencies,Z,part,weights=None,dZ=False,
			scale_Z=scale_Z,penalty='discrete',fit_type='map')
		matrices = dist_mat[dist_name]
		
		
		# prepare data for stan model
		# dat = self._prep_stan_data(frequencies,Z_scaled,part,model_type,sigma_min,mode='optimize',inductance_scale=inductance_scale)
		# if tau boundaries not specified, allow HN peaks to extend 2 decades beyond measurement range
		if min_tau is None:
			min_tau = np.min(self.distributions[dist_name]['tau'])
		if max_tau is None:
			max_tau = np.max(self.distributions[dist_name]['tau'])
			
		num_decades = int(np.ceil(np.log10(max_tau)) - np.floor(np.log10(min_tau)))
			
		# make G matrix
		if drt_eval_tau is None:
			drt_eval_tau = self.distributions[dist_name]['tau']
			# drt_eval_tau = 
		phi = get_basis_func('gaussian')
		G = np.array([phi(np.log(drt_eval_tau/t_m),self.distributions[dist_name]['epsilon']) for t_m in self.distributions[dist_name]['tau']]).T
		
		dat = dict(	
			# Dimensions
			N = len(frequencies),
			K = len(self.distributions[dist_name]['tau']),
			J = num_HN,
			M = len(self.distributions[dist_name]['tau']),
			
			# Matrices
			A_re = matrices['A_re'],
			A_im = matrices['A_im'],
			G = G, # G*x = gamma
			L0 = 1.5*0.24*matrices['L0'],
			L1 = 1.5*0.16*matrices['L1'],
			L2 = 1.5*0.08*matrices['L2'],
			
			# Impedance data
			freq = frequencies,
			Z = np.concatenate((Z_scaled.real,Z_scaled.imag)),
			
			# HN limits
			min_tau_HN = min_tau,
			max_tau_HN = max_tau,
			
			# DRT evaluation
			tau_drt = self.distributions[dist_name]['tau'],
			F_scale = 0.5,
			sigma_F_min = sigma_F_min,
			sigma_F_res_scale = 1,
			alpha_F_prop_scale = 0.1,
			
			# Fixed hyperparameters
			induc_scale = inductance_scale,
			ups_alpha = 0.05,
			ups_beta = 0.1,
			R_hn_scale_alpha = 1,
			R_hn_scale_beta = 1,
			
			# Impedance error structure
			sigma_min = sigma_min,
			sigma_res_scale = 0.05,
			alpha_prop_scale = 0.05,
			alpha_re_scale = 0.05,
			alpha_im_scale = 0.05
		)
		
		dat.update(add_stan_data)
		self._stan_input = dat.copy()
		
		# Get initial parameter values
		iv = {}
		if np.sum([init_from_ridge,init_from_map,init_values is not None]) > 1:
			raise ValueError('Only one of init_from_ridge, init_from_map, and init_drt_fit may be used')
		elif init_from_ridge:
			# default ridge_fit settings
			ridge_defaults = dict(hyper_lambda=True,penalty='integral',reg_ord=2,scale_Z=True,dZ=True,weights='modulus',
				   hl_beta=2.5,lambda_0=1e-2,nonneg=nonneg)
			# update with any user-upplied settings - may overwrite defaults
			ridge_defaults.update(ridge_kw)
			# get initial parameter values from ridge fit
			self.ridge_fit(frequencies,Z,**ridge_defaults)
			
			# scale the coefficients
			coef = self.distribution_fits[dist_name]['coef']
			if dist_type=='series':
				x_star = coef/self._Z_scale
			elif dist_type=='parallel':
				x_star = coef*self._Z_scale
			iv = {'x_drt':x_star}
			iv['induc_raw'] = self.inductance/inductance_scale
			
			# distribute lntau values uniformly
			iv['lntau_hn'] = np.linspace(np.log(min_tau*10),np.log(max_tau/10),num_HN)
			# iv['R_hn'] = np.ones(num_HN)*0.1
			iv['alpha_hn'] = np.ones(num_HN)*0.95
			iv['beta_hn'] = np.ones(num_HN)*0.95
		elif init_from_map:
			# default map_fit settings
			map_defaults = dict(nonneg_drt=nonneg)
			# update with any user-upplied settings - may overwrite defaults
			map_defaults.update(map_kw)
			# Perform map fit
			if hasattr(self,'inv_init')==False:
				self.inv_init = Inverter(basis_freq=self.basis_freq)
			self.inv_init.map_fit(frequencies, Z, **map_defaults)
			
			# coef = self.inv_init.distribution_fits[dist_name]['coef']
			# if dist_type=='series':
				# x_star = coef/self._Z_scale
			# elif dist_type=='parallel':
				# x_star = coef*self._Z_scale
			# iv = {'x_drt':x_star}
			iv = self.inv_init._opt_result.copy()
			# update parameter names
			iv['x_drt'] = iv['x']
			# error structure parameters for drt need _drt suffix
			for param in ['alpha_re_raw','alpha_im_raw','alpha_prop_raw','sigma_res_raw']:
				iv[f'{param}_drt'] = iv[param]
				del iv[param]
				
			# distribute lntau values uniformly
			iv['lntau_hn'] = np.linspace(np.log(min_tau*10),np.log(max_tau/10),num_HN)
			# iv['R_hn'] = np.ones(num_HN)*0.1
			iv['alpha_hn'] = np.ones(num_HN)*0.95
			iv['beta_hn'] = np.ones(num_HN)*0.95
		else:
			# distribute lntau values uniformly
			iv['lntau_hn'] = np.linspace(np.log(min_tau*10),np.log(max_tau/10),num_HN)
			iv['R_hn'] = np.ones(num_HN)*0.1
			iv['alpha_hn'] = np.ones(num_HN)*0.95
			iv['beta_hn'] = np.ones(num_HN)*0.8
			# iv['upsilon'] = np.ones(num_HN)*10	
		
		# if init_from_ridge:
			# if len(self.distributions) > 1:
				# raise ValueError('Ridge initialization can only be performed for single-distribution fits')
			# else:
				# init = self._get_init_from_ridge(frequencies,Z,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
				# iv.update(init)
		# elif init_from_map:
			# init = self._get_init_from_map(frequencies,Z,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
			# iv.update(init)
		# elif init_drt_fit is not None:
			# init = self._get_init_from_drt(init_drt_fit,num_HN,inductance_scale,min_tau,max_tau,peakfit_kw)
			# iv.update(init)
		
		
		# if outliers:
			# raise ValueError('Outlier model not yet implemented')
			# # initialize sigma_out near zero, everything else randomly
			# iv['sigma_out_raw'] = np.zeros(2*len(Z)) + 0.1
			# dat['so_invscale'] = outlier_lambda
			
		if init_values is not None:
			iv = init_values
		
			
		def init():
			return iv
		self._init_params = iv
				
		# print(iv)
		
		# optimize posterior
		self._opt_result = model.optimizing(dat,iter=max_iter,seed=random_seed,init=init,init_alpha=init_alpha)
		
		# extract coefficients
		self.distribution_fits = {}
		self.error_fit = {}
		if model_type in ['Series','Parallel']:
			dist_name = [k for k,v in self.distributions.items() if v['dist_type']==model_type.lower()][0]
			dist_type = self.distributions[dist_name]['dist_type']
			self.distribution_fits[dist_name] = {'R_HN':self._rescale_coef(self._opt_result['R_hn'],dist_type)}
			self.distribution_fits[dist_name]['tau_HN'] = np.exp(self._opt_result['lntau_hn'])
			self.distribution_fits[dist_name]['alpha_HN'] = self._opt_result['alpha_hn']
			self.distribution_fits[dist_name]['beta_HN'] = self._opt_result['beta_hn']
			
			self.R_inf = self._rescale_coef(self._opt_result['Rinf'],'series')
			self.inductance = self._rescale_coef(self._opt_result['induc'],'series')
		
		# store error structure parameters
		# scaled parameters
		self.error_fit['sigma_min'] = self._rescale_coef(sigma_min,'series')
		for param in ['sigma_drt','sigma_hn','sigma_res']:
			self.error_fit[param] = self._rescale_coef(self._opt_result[param],'series')
			if param=='sigma_res':
				self.error_fit[param] *= dat['sigma_hn_scale']
		# unscaled parameters
		for param in ['alpha_prop','alpha_re','alpha_im']:
			self.error_fit[param] = self._opt_result[param]*dat['sigma_hn_scale']
		# outlier contribution
		if outliers:
			self.error_fit['sigma_out'] = self._rescale_coef(self._opt_result['sigma_out'],'series')
		
		self.fit_type = 'map'
		
	def map_activation_fit(self,frequencies,Z,T,T_base,part='both',scale_Z=True,nonneg=True,outliers=False,
		min_tau=None,max_tau=None,ordered=True,num_HN=10,add_stan_data={},
		adjust_temp=False,temp_uncertainty=5,temp_offset_scale=2,
		model_str=None,
		# shapeshift=False,prefactor=False,
		# initialization parameters
		init_from_ridge=False,init_from_map=False,init_basis_freq=None,init_drt_fit=None,init_values=None,init_kw={},peakfit_kw={},
		# optimization control
		sigma_min=0.002,max_iter=50000,random_seed=1234,inductance_scale=1,outlier_lambda=5):
		"""
		Obtain the maximum a posteriori estimate of the defined distribution(s) (and all model parameters).
		
		Parameters:
		-----------
		frequencies: array
			Measured frequencies
		Z: complex array
			Measured (complex) impedance values. Must have same length as frequencies
		part: str, optional (default: 'both')
			Which part of the impedance data to fit. Options: 'both', 'real', 'imag', 'polar'
		scale_Z: bool, optional (default: True)
			If True, scale impedance by the factor sqrt(N)/std(|Z|) to normalize for magnitude and sample size
		init_from_ridge: bool, optional (default: False)
			If True, use the hyperparametric ridge solution to initialize the Bayesian fit. 
			Only valid for single-distribution fits
		nonneg: bool, optional (default: False)
			If True, constrain the DRT to non-negative values
		outliers: bool, optional (default: False)
			If True, enable outlier identification via independent error contribution variable
		sigma_min: float, optional (default: 0.002)
			Impedance error floor. This is necessary to avoid sampling/optimization errors.
			Values smaller than the default (0.002) may enable slightly closer fits of very clean data,
			but may also result in sampling/optimization errors that yield unexpected results.
		max_iter: int, optional (default: 50000)
			Maximum number of iterations to allow the optimizer to perform
		random_seed: int, optional (default: 1234)
			Random seed for optimizer
		inductance_scale: float, optional (default: 1)
			Scale (std of normal prior) of the inductance. Lower values will constrain the inductance,
			which may be helpful if the estimated inductance is anomalously large (this may occur if your
			measured impedance data does not extend to high frequencies, i.e. 1e5-1e6 Hz)
		outlier_lambda: float, optional (default: 5)
			Lambda parameter (inverse scale) of the exponential prior on the outlier error contribution.
			Smaller values will make it easier for points to be flagged as outliers 
		ridge_kw: dict, optional (default: {})
			Keyword arguments to pass to ridge_fit if init_from_ridge==True.
		"""
		# load stan model
		if model_str is None:
			if part=='both':
				model_str = 'Series_HN-activation_pos'
				if ordered:
					model_str += '_ordered'
				if adjust_temp:
					model_str += '_T-adjust'
				# if shapeshift:
					# model_str += '_shapeshift'
				# if prefactor:
					# model_str += '_prefactor'
				model_str += '_StanModel.pkl'
			elif part=='polar':
				model_str = 'Series_HN-activation_polar_pos_StanModel.pkl'
		model = load_pickle(os.path.join(script_dir,'stan_model_files',model_str))
		self.stan_model_name = model_str
		model_type = model_str.split('_')[0]
		
		# if prefactor:
			# warnings.warn('Prediction methods have not been updated for prefactor fits')
		
		# store unique fit temperatures
		temp_start_indices = np.where(np.diff(T)!=0)[0] + 1
		temp_start_indices = np.concatenate(([0],temp_start_indices))
		self.fit_temperatures = T[temp_start_indices]
		self.T_offset = np.zeros(len(self.fit_temperatures))
		if T[0]!=T_base:
			raise ValueError('T_base must be first temperature')

		# perform scaling. Scale based on T_base
		Z_base = Z[np.where(T==T_base)]
		self._scale_Z(Z_base,'map')
		Z_scaled = Z/self._Z_scale
		self.f_train = frequencies
		freq_base = frequencies[np.where(T==T_base)]
		
		rel_Z_scale = np.ones(len(frequencies))
		for temp in np.unique(T):
			tidx = np.where(T==temp)
			Z_T = Z[tidx]
			Zmod_T = (Z_T*Z_T.conjugate())**0.5
			Z_scale_T = np.std(Zmod_T)/np.sqrt(len(Z_T)/81)
			rel_Z_scale[tidx] = Z_scale_T/self._Z_scale
			
		# prepare data for stan model
		dat = self._prep_stan_data(frequencies,Z_scaled,part,model_type,sigma_min,mode='optimize',inductance_scale=inductance_scale)
		# if tau boundaries not specified, allow HN peaks to extend 2 decades beyond measurement range
		if min_tau is None:
			min_tau = np.min(1/(2*np.pi*freq_base))/100
		if max_tau is None:
			max_tau = np.max(1/(2*np.pi*freq_base))*100
		dat['N'] = len(frequencies)
		dat['temp'] = T
		dat['T_base'] = T_base
		dat['min_tau_HN'] = min_tau
		dat['max_tau_HN'] = max_tau
		dat['max_delta_G'] = 2
		dat['max_phi_T'] = 2
		dat['K'] = num_HN
		dat['ln_phi_T_scale'] = 0.2
		dat['R_base_scale_alpha'] = 1
		dat['R_base_scale_beta'] = 1
		dat['rel_Z_scale'] = rel_Z_scale
		
		if adjust_temp:
			dat['temp_start_indices'] = temp_start_indices + 1 # adjust for stan indexing (starts at 1)
			dat['P'] = len(temp_start_indices)
			dat['temp_uncertainty'] = temp_uncertainty
			dat['temp_offset_scale'] = temp_offset_scale
		
		dat.update(add_stan_data)
		self._stan_input = dat.copy()
		
		# if shapeshift:
			# # increment indices for stan (indexing starts at 1)
			# temp_start_indices += 1
			# dat['P'] = len(temp_start_indices) + 1
			# dat['temp_start_indices'] = temp_start_indices
			# dat['sigmaT_alpha_scale'] = 0.1
			# dat['sigmaT_beta_scale'] = 0.1
			# dat['ln_phi_scale'] = 0.2
			# dat['N'] = len(frequencies) # changed from N to 2N; should update other models
			# # dat['rel_Z_scale'] = rel_Z_scale
			
		# if prefactor:
			# dat['n_T_alpha'] = 2
			# dat['n_T_beta'] = 0.5
			# dat['min_n_T'] = -2
			# dat['max_n_T'] = 2
			
		
		# Create tau grid for convenience, using min_tau and max_tau
		num_decades = int(np.ceil(np.log10(max_tau)) - np.floor(np.log10(min_tau)))
		for distribution in self.distributions.keys():
			self.distributions[distribution]['tau'] = np.logspace(np.log10(min_tau),np.log10(max_tau),num_decades*10 + 1)
		
		# Get initial parameter values
		iv = {}
		if np.sum([init_from_ridge,init_from_map,init_drt_fit is not None]) > 1:
			raise ValueError('Only one of init_from_ridge, init_from_map, and init_drt_fit may be used')
		if init_from_ridge:
			if len(self.distributions) > 1:
				raise ValueError('Ridge initialization can only be performed for single-distribution fits')
			else:
				init = self._get_init_from_ridge(freq_base,Z_base,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
				iv.update(init)
				# Update parameter names for activation model
				iv['Rinf_base'] = iv['Rinf']
				iv['Rinf_base_raw'] = init['Rinf_raw']
				iv['lntau_HN_base'] = iv['lntau_HN']
				iv['R_HN_base'] = iv['R_HN']
		elif init_from_map:
			init = self._get_init_from_map(freq_base,Z_base,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
			iv.update(init)
			# Update parameter names for activation model
			iv['Rinf_base'] = iv['Rinf']
			iv['Rinf_base_raw'] = init['Rinf_raw']
			iv['lntau_HN_base'] = iv['lntau_HN']
			iv['R_HN_base'] = iv['R_HN']
		elif init_drt_fit is not None:
			init = self._get_init_from_drt(init_drt_fit,num_HN,inductance_scale,min_tau,max_tau,peakfit_kw)
			iv.update(init)
			# Update parameter names for activation model
			iv['Rinf_base'] = iv['Rinf']
			iv['Rinf_base_raw'] = init['Rinf_raw']
			iv['lntau_HN_base'] = iv['lntau_HN']
			iv['R_HN_base'] = iv['R_HN']
		else:
			# distribute lntau values uniformly
			iv['lntau_HN_base'] = np.linspace(np.log(min_tau*10),np.log(max_tau/10),num_HN)
			iv['R_HN_base'] = np.ones(num_HN)
			iv['alpha_HN'] = np.ones(num_HN)*0.95
			iv['beta_HN'] = np.ones(num_HN)*0.8
			# iv['upsilon'] = np.ones(num_HN)*1
			iv['delta_G'] = np.ones(num_HN)*0.5
			iv['delta_G_Rinf'] = 0.5
			iv['ln_phi_T'] = np.zeros(num_HN)
			iv['induc'] = 1e-8
			iv['induc_raw'] = iv['induc']/inductance_scale
			# if prefactor:
				# iv['n_T'] = np.zeros(num_HN)
				# iv['n_T_raw'] = np.zeros(num_HN)
		
		if outliers:
			raise ValueError('Outlier model not yet implemented')
			# initialize sigma_out near zero, everything else randomly
			iv['sigma_out_raw'] = np.zeros(2*len(Z)) + 0.1
			dat['so_invscale'] = outlier_lambda
			
		# if shapeshift:
			# for param in ['alpha_HN','beta_HN']:
				# iv[param] = np.tile(iv[param],(dat['P'],1))
				
		if init_values is not None:
			# iv.update(init_values)
			iv = init_values
			
		def init():
			return iv
		self._init_params = iv
				
		# print(iv)
		
		# optimize posterior
		self._opt_result = model.optimizing(dat,iter=max_iter,seed=random_seed,init=init)
		
		# extract coefficients
		self.distribution_fits = {}
		self.error_fit = {}
		if model_type in ['Series','Parallel']:
			dist_name = [k for k,v in self.distributions.items() if v['dist_type']==model_type.lower()][0]
			dist_type = self.distributions[dist_name]['dist_type']
			self.distribution_fits[dist_name] = {'R_HN_base':self._rescale_coef(self._opt_result['R_HN_base'],dist_type)}
			self.distribution_fits[dist_name]['tau_HN_base'] = np.exp(self._opt_result['lntau_HN_base'])
			self.distribution_fits[dist_name]['alpha_HN'] = self._opt_result['alpha_HN']
			self.distribution_fits[dist_name]['beta_HN'] = self._opt_result['beta_HN']
			self.distribution_fits[dist_name]['delta_G'] = self._opt_result['delta_G']
			self.distribution_fits[dist_name]['phi_T'] = np.exp(self._opt_result['ln_phi_T'])
			self.distribution_fits[dist_name]['T_base'] = T_base
			
			self.R_inf_base = self._rescale_coef(self._opt_result['Rinf_base'],'series')
			self.R_inf_dG = self._opt_result['delta_G_Rinf']
			self.inductance = self._rescale_coef(self._opt_result['induc'],'series')
			
			# if prefactor:
				# self.distribution_fits[dist_name]['n_T'] = self._opt_result['n_T']
			if adjust_temp:
				# self.T_offset = np.concatenate(([0],self._opt_result['temp_offset']))
				self.T_offset = self._opt_result['temp_offset']
		
		# store error structure parameters
		if part=='both':
			# scaled parameters
			self.error_fit['sigma_min'] = self._rescale_coef(sigma_min,'series')
			for param in ['sigma_tot','sigma_res']:
				self.error_fit[param] = self._rescale_coef(self._opt_result[param],'series')
			# unscaled parameters
			for param in ['alpha_prop','alpha_re','alpha_im']:
				self.error_fit[param] = self._opt_result[param]
			# outlier contribution
			if outliers:
				self.error_fit['sigma_out'] = self._rescale_coef(self._opt_result['sigma_out'],'series')
		elif part=='polar':
			self.error_fit['sigma_Zmod_min'] = self._rescale_coef(sigma_min,'series')
			self.error_fit['sigma_Zmod_res'] = self._rescale_coef(self._opt_result['sigma_Zmod_res'],'series')
			self.error_fit['alpha_prop'] = self._opt_result['alpha_prop']
			
			self.error_fit['sigma_Zphase_min'] = 0.00175
			self.error_fit['sigma_Zphase_res'] = self._opt_result['sigma_Zphase_res']
			
			
			
		
		self.fit_type = 'map-activation'
		
	# =========================
	# Initialization
	# =========================
	def _get_init_from_drt(self,fit_data,num_HN,inductance_scale,min_tau,max_tau,peakfit_kw):
		self.inv_init = Inverter()
		self.inv_init.load_fit_data(fit_data)
		
		# Fit HN peaks to DRT
		dist_name = list(self.inv_init.distributions.keys())[0]
		self.inv_init.fit_HN_peaks(dist_name,**peakfit_kw)
		
		# Initialize parameter values
		pf_params = self.inv_init.extract_HN_info(distribution=dist_name)
		# ignore peakfit peaks outside tau boundaries
		in_idx = np.where((pf_params['tau_0']>=min_tau) & (pf_params['tau_0']<=max_tau))
		for key in ['R','tau_0','alpha','beta']:
			pf_params[key] = pf_params[key][in_idx]
		num_pf_peaks = len(in_idx[0])
		
		if num_pf_peaks > num_HN:
			# raise ValueError('Number of peaks identified by ridge fit ({}) exceeds number of HN basis functions ({}). Increase num_HN or adjust peakfit_kw accordingly'.format(num_pf_peaks,num_HN))
			# if number of peaks from DRT exceeds num_HN, keep largest peaks
			warnings.warn('Number of peaks identified from DRT fit ({}) exceeds number of HN basis functions ({}). Keeping only the largest {} peaks'.format(num_pf_peaks,num_HN,num_HN))
			sort_idx = np.argsort(pf_params['R'])[::-1]
			keep_idx = sort_idx[:num_HN]
			for key in ['R','tau_0','alpha','beta']:
				pf_params[key] = pf_params[key][keep_idx]
			num_pf_peaks = num_HN
			
		init = {}		
		# Set first num_pf_peaks resistances to values identified by peak fit; rest go to small value
		init['R_HN'] = np.zeros(num_HN) + 0.1
		init['R_HN'][:num_pf_peaks] = pf_params['R']/self._Z_scale
		# Set first num_pf_peaks tau values to peak fit values; remainder should be uniformly distributed
		init['lntau_HN'] = np.zeros(num_HN)
		init['lntau_HN'][:num_pf_peaks] = np.log(pf_params['tau_0'])
		init['lntau_HN'][num_pf_peaks:] = np.linspace(np.log(min_tau*10),np.log(max_tau/10),num_HN - num_pf_peaks)
		# Set first num_pf_peaks alpha values to peak fit values; remainder set to 0.95
		init['alpha_HN'] = np.zeros(num_HN) + 0.95
		init['alpha_HN'][:num_pf_peaks] = pf_params['alpha']
		# Set first num_pf_peaks beta values to peak fit values; remainder set to 0.8
		init['beta_HN'] = np.zeros(num_HN) + 0.8
		init['beta_HN'][:num_pf_peaks] = pf_params['beta']
		# Set inductance and R_inf
		init['Rinf'] = self.inv_init.R_inf/self._Z_scale
		init['Rinf_raw'] = init['Rinf']/100
		init['induc'] = self.inv_init.inductance/self._Z_scale
		if init['induc'] <= 0:
			init['induc'] = 1e-10
		init['induc_raw'] = init['induc']/inductance_scale
		# Set upsilon values for fitted peaks to 5; remainder set to 15
		init['upsilon'] = np.zeros(num_HN) + 15
		init['upsilon'][:num_pf_peaks] = 5

		return init
		
	def _get_constrained_init_from_drt(self,fit_data,inductance_scale,peakfit_kw):
		self.inv_init = Inverter()
		self.inv_init.load_fit_data(fit_data)
		
		tau = 1/(2*np.pi*self.basis_freq)
		min_tau = np.min(tau)
		max_tau = np.max(tau)
		num_HN = len(tau)
		
		# Fit HN peaks to DRT
		dist_name = list(self.inv_init.distributions.keys())[0]
		self.inv_init.fit_HN_peaks(dist_name,**peakfit_kw)
		
		# Initialize parameter values
		pf_params = self.inv_init.extract_HN_info(distribution=dist_name)
		# # ignore peakfit peaks outside tau boundaries
		# in_idx = np.where((pf_params['tau_0']>=min_tau) & (pf_params['tau_0']<=max_tau))
		# for key in ['R','tau_0','alpha','beta']:
			# pf_params[key] = pf_params[key][in_idx]
		# num_pf_peaks = len(in_idx[0])
		match_idx = [np.argmin(np.abs(t0-tau)) for t0 in pf_params['tau_0']]
			
		init = {}		
		# Set first num_pf_peaks resistances to values identified by peak fit; rest go to small value
		init['R_HN'] = np.zeros(num_HN) + 0.02
		init['R_HN'][match_idx] = pf_params['R']/self._Z_scale
		# Set first num_pf_peaks tau values to peak fit values; remainder should be uniformly distributed
		init['lntau_HN'] = tau.copy()
		init['lntau_HN'][match_idx] = np.log(pf_params['tau_0'])
		init['lntau_shift'] = np.zeros(num_HN)
		init['lntau_shift'][match_idx] = np.log(pf_params['tau_0']) - np.log(tau[match_idx])
		# Set first num_pf_peaks alpha values to peak fit values; remainder set to 0.95
		init['alpha_HN'] = np.zeros(num_HN) + 0.95
		init['alpha_HN'][match_idx] = pf_params['alpha']
		# Set first num_pf_peaks beta values to peak fit values; remainder set to 0.8
		init['beta_HN'] = np.zeros(num_HN) + 0.8
		init['beta_HN'][match_idx] = pf_params['beta']
		# Set inductance and R_inf
		init['Rinf'] = self.inv_init.R_inf/self._Z_scale
		init['Rinf_raw'] = init['Rinf']/100
		init['induc'] = self.inv_init.inductance/self._Z_scale
		if init['induc'] <= 0:
			init['induc'] = 1e-10
		init['induc_raw'] = init['induc']/inductance_scale
		# Set upsilon values for fitted peaks to 5; remainder set to 15
		init['upsilon'] = np.zeros(num_HN) + 15
		init['upsilon'][match_idx] = 1

		return init
	
	def _get_init_from_ridge(self,frequencies,Z,basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,ridge_kw,peakfit_kw):
		"""Get initial parameter estimate from ridge_fit
		Parameters:
		-----------
		frequencies: array
			Array of frequencies
		Z: array
			Impedance data
		hl_beta: float
			beta regularization parameter
		lambda_0: float
			lambda_0 regularization parameter
		nonneg: bool
			If True, constrain distribution to non-negative values
		inductance_scale: float
			Scale (std of normal prior) of the inductance
		ridge_kw: dict
			
		"""
		dist_name = list(self.distributions.keys())[0]
		dist_type = self.distributions[dist_name]['dist_type']
		
		# default ridge_fit settings
		ridge_defaults = dict(hyper_lambda=True,penalty='integral',reg_ord=2,scale_Z=True,dZ=True,
			   hl_beta=2.5,lambda_0=1e-2,nonneg=nonneg)
		# update with any user-upplied settings - may overwrite defaults
		ridge_defaults.update(ridge_kw)
		# Perform ridge fit
		self.inv_init = Inverter(basis_freq=basis_freq)
		self.inv_init.ridge_fit(frequencies,Z,**ridge_defaults)
		
		# Get initial HN parameter values from ridge fit
		init = self._get_init_from_drt(self.inv_init.save_fit_data(),num_HN,inductance_scale,min_tau,max_tau,peakfit_kw)
		
		return init
		
	def _get_init_from_map(self,frequencies,Z,basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,map_kw,peakfit_kw):
		"""Get initial parameter estimate from ridge_fit
		Parameters:
		-----------
		frequencies: array
			Array of frequencies
		Z: array
			Impedance data
		hl_beta: float
			beta regularization parameter
		lambda_0: float
			lambda_0 regularization parameter
		nonneg: bool
			If True, constrain distribution to non-negative values
		inductance_scale: float
			Scale (std of normal prior) of the inductance
		ridge_kw: dict
			
		"""
		dist_name = list(self.distributions.keys())[0]
		dist_type = self.distributions[dist_name]['dist_type']
		
		# default map_fit settings
		map_defaults = dict(nonneg_drt=nonneg)
		# update with any user-upplied settings - may overwrite defaults
		map_defaults.update(map_kw)
		# Perform map fit
		self.inv_init = Inverter(basis_freq=basis_freq)
		self.inv_init.map_fit(frequencies, Z, **map_defaults)
		
		# Get initial HN parameter values from ridge fit
		init = self._get_init_from_drt(self.inv_init.save_fit_data(which='core'),num_HN,inductance_scale,min_tau,max_tau,peakfit_kw)
		
		return init
		
	def _prep_stan_data(self,frequencies,Z,part,model_type,sigma_min,mode,inductance_scale):
		"""Prepare input data for Stan model. Called by map_fit and bayes_fit methods
		
		Parameters:
		-----------
		frequencies: array
			Measured frequencies
		Z: complex array
			Measured (complex) impedance values
		part: str
			Which part of the impedance data to fit. Options: 'both', 'real', 'imag'
		model_type: str
			Model type indicating distributions to be recovered
		outliers: bool
			Whether or not to enable outlier detection
		sigma_min: float
			Minimum value of error scale
		mode: str
			Solution mode. Options: 'sample', 'optimize'
		
		"""
		if model_type in ['Series','Parallel']:
			if part=='both':
				Z_stack = np.concatenate((Z.real,Z.imag))
				dat = {'N':len(frequencies),
					   'freq':frequencies,
					   'Z':Z_stack,
					   'sigma_min':sigma_min
					   }
			elif part=='polar':
				dat = {'N':len(frequencies),
					   'freq':frequencies,
					   'Zmod': np.real(np.sqrt(Z*Z.conjugate())),
					   'Zphase': np.arctan2(Z.imag,Z.real),
					   'sigma_Zmod_min': sigma_min,
					   'sigma_Zphase_min': 0.00175 # ~0.1 degrees
					   }
				
			if mode=='sample':
				pass
			elif mode=='optimize':
				ups_alpha = 15
				ups_beta = 0.5
				
			dat.update({'induc_scale':inductance_scale,
					   'ups_alpha':ups_alpha,
					   'ups_beta':ups_beta
						}
					)
						
		return dat
	
	# =========================
	# Prediction
	# =========================
	
	def _get_drift_model(self,base_str):
		# get drift model
		model_split = self.stan_model_name.split('_')
		drift_str = [ms for ms in model_split if ms[:len(base_str)]==base_str][0]
		if drift_str==base_str:
			drift_model = ''
		else:
			base_split_len = len(base_str.split('-'))
			drift_model = '-'.join(drift_str.split('-')[base_split_len:])
			
		return drift_model
	
	def _calc_drift_Ft(self,fit,drift_model,times):
		t_i = fit['t_i']
		t_f = fit['t_f']
		
		if drift_model=='RQ-lin-xi':
			F_t = (times - t_i)/(t_f - t_i)
		elif drift_model=='RQ-lin-xf':
			F_t = (times - t_f)/(t_f - t_i)
		elif drift_model=='RQ-xi':
			k_d = fit['k_d']
			F_t = 1 - np.exp(-k_d*times)
		elif drift_model=='RQ-compressed-xi':
			k_d = fit['k_d']
			F_t = (np.exp(-k_d*t_i) - np.exp(-k_d*times))/(np.exp(-k_d*t_i) - np.exp(-k_d*t_f))
			
		return F_t
		
	def _calc_drift_Z(self,fit,drift_model,frequencies,times):
		Z_pred = np.zeros(len(frequencies),dtype=complex)
		if drift_model=='':
			R_i = fit['R_i']
			tau_i = fit['tau_i']
			alpha_HN = fit['alpha_HN']
			beta_HN = fit['beta_HN']
			R_f = fit['R_f']
			k_d = fit['k_d']
			phi_d = fit['phi_d']
			
			# Get impedance for each HN peak
			for k in range(len(R_i)):	
				# Get R(t) and tau(t)
				R_t = R_i[k] + (R_f[k] - R_i[k])*(1 - np.exp(-k_d[k]*times))
				tau_t = tau_i[k] + tau_i[k]*((R_f[k]/R_i[k])**phi_d[k] - 1)*(1 - np.exp(-k_d[k]*times))
				
				# Add impedance for kth HN element
				Z_pred += R_t*pf.HN_impedance(frequencies,tau_t,alpha_HN[k],beta_HN[k])
		elif drift_model.find('RQ') > -1:
			if drift_model.find('xi') > -1:
				R_HN = fit['R_i']
				tau_HN = fit['tau_i']
			elif drift_model.find('xf') > -1:
				R_HN = fit['R_f']
				tau_HN = fit['tau_f']
			alpha_HN = fit['alpha_HN']
			beta_HN = fit['beta_HN']
			
			# Get impedance for each HN peak
			for k in range(len(R_HN)):
				# Z_list = [R*pf.HN_impedance(frequencies,t0,a,b) for R,t0,a,b in zip(R_HN,tau_HN,alpha_HN,beta_HN)]
				# Z_stack = np.vstack(Z_list)
				# # Sum HN impedances
				# Z_pred += np.sum(Z_stack,axis=0)
				Z_pred += R_HN[k]*pf.HN_impedance(frequencies,tau_HN[k],alpha_HN[k],beta_HN[k])
			
			# Z due to time-dependent ZARC
			R_rq = fit['R_rq']
			tau_rq = fit['tau_rq']
			phi_rq = fit['phi_rq']	
			
			F_t = self._calc_drift_Ft(fit,drift_model,times)
			# print(F_t)
			# print(R_rq)
			
			Z_pred += F_t*(R_rq/(1+(tau_rq*1j*2*np.pi*frequencies)**phi_rq))
		
		# # add offsets
		# if drift_model=='':
			# Z_pred += fit['R_inf_i'] + (fit['R_inf_f'] - fit['R_inf_i'])*(1 - np.exp(-fit['R_inf_kd']*times))
		# elif drift_model.find('xi') > -1:
			# Z_pred += fit['R_inf_i'] + (fit['R_inf_f'] - fit['R_inf_i'])*F_t
		# elif drift_model.find('xf') > -1:
			# Z_pred += fit['R_inf_f'] + (fit['R_inf_f'] - fit['R_inf_i'])*F_t
		# Z_pred += 1j*2*np.pi*frequencies*self.inductance
			
		return Z_pred
		
	def _calc_drift_distribution(self,fit,drift_model,eval_tau,time):
		if drift_model=='':
			R_i = fit['R_i']
			tau_i = fit['tau_i']
			alpha_HN = fit['alpha_HN']
			beta_HN = fit['beta_HN']
			R_f = fit['R_f']
			k_d = fit['k_d']
			phi_d = fit['phi_d']
			
			# Get R(t) and tau(t)
			R_t = R_i + (R_f - R_i)*(1 - np.exp(-k_d*time))
			tau_t = tau_i + tau_i*((R_f/R_i)**phi_d - 1)*(1 - np.exp(-k_d*time))
			
			R_HN = R_t
			tau_HN = tau_t
			
			bases = np.array([pf.HN_distribution(eval_tau,t0,a,b) for t0,a,b in zip(tau_HN,alpha_HN,beta_HN)]).T
			F = bases@R_HN
		
		elif drift_model.find('RQ') > -1:
			if drift_model.find('xi') > -1:
				R_HN = fit['R_i']
				tau_HN = fit['tau_i']
			elif drift_model.find('xf') > -1:
				R_HN = fit['R_f']
				tau_HN = fit['tau_f']
			alpha_HN = fit['alpha_HN']
			beta_HN = fit['beta_HN']
			
			# get static DRT
			bases = np.array([pf.HN_distribution(eval_tau,t0,a,b) for t0,a,b in zip(tau_HN,alpha_HN,beta_HN)]).T
			F = bases@R_HN
		
			# Get time-dependent ZARC DRT
			R_rq = fit['R_rq']
			tau_rq = fit['tau_rq']
			phi_rq = fit['phi_rq']	

			F_t = self._calc_drift_Ft(fit,drift_model,time)
			F_rq = (1/(2*np.pi))*np.sin((1-phi_rq)*np.pi)/(np.cosh(phi_rq*np.log(eval_tau/tau_rq))-np.cos((1-phi_rq)*np.pi))
			
			# print(F_rq.shape,R_rq.shape,F_t.shape)
			F += F_rq*R_rq*F_t
			
		return F
	
	def predict_Z(self,frequencies,distributions=None,times=None,T=None,percentile=None,sample_id=None,include_T_offset=False):
		# times does nothing currently - included for compatibility only
		if distributions is not None:
			if type(distributions)==str:
				distributions = [distributions]
		else:
			distributions = list(self.distribution_fits.keys())
			
		# activation-drift fits
		if self.fit_type.find('activation-drift') >= 0:
			# Check for time, temperature, and sample_id
			if T is None:
				raise ValueError('T is required for activation fits')
			elif type(T) in (list,np.ndarray) and len(T)!=len(frequencies):
				raise ValueError('T must either be a single value or array-like of same length as frequencies')
			elif type(T) in (float,int):
				T = np.ones(len(frequencies))*T
				
			if times is None:
				raise ValueError('times are required for drift fits')
			elif len(times)!=len(frequencies):
				raise ValueError('times must be array-like of same length as frequencies')
			
			if self.fit_type.find('combi-activation-drift')>=0:
				if sample_id is None:
					raise ValueError('sample_id is required for combi fit')
				else:
					sample_index = np.where(self.sample_ids==sample_id)
			
			# get drift model
			drift_model = self._get_drift_model('HN-activation-drift')
					
			# get T offset
			if include_T_offset:
				offset = self.get_T_offset(T,sample_id)
				T_orig = deepcopy(T)
				T = T.copy() + offset
			else:
				T_orig = deepcopy(T)
				
			Z_pred = np.zeros(len(frequencies),dtype=complex)
			for dist_name in distributions:
			
				# Unpack parameters for convenience
				fit = self.distribution_fits[dist_name].copy()
				if self.fit_type.find('combi-activation-drift') >= 0:
					# Get parameters for sample_id
					j_params = ['tau_i_base','alpha_HN','beta_HN','delta_G','phi_T']
					if drift_model=='':
						j_params += ['R_i_base','R_f_base','k_base','delta_G_k','phi_d']
					elif drift_model.find('RQ') > -1:
						j_params += ['R_rq','tau_rq','phi_rq']
						if drift_model.find('xi') > -1:
							j_params += ['R_i_base']
						elif drift_model.find('xf') > -1:
							j_params += ['R_f_base']
					for param in j_params:
						fit[param] = fit[param][sample_index].flatten()
				
				# populate tmp_fit with parameters for _calc_drift_Z
				# need to get param values at temperature T
				tmp_fit = deepcopy(fit)
				T_base = fit['T_base']
				K = len(fit['alpha_HN'])
				if drift_model=='':
					for param in ['R_i','R_f','tau_i','k_d']:
						tmp_fit[param] = np.empty((K,len(frequencies)))
				elif drift_model.find('RQ') > -1:
					opt_point = drift_model[drift_model.find('x')+1]
					for param in [f'R_{opt_point}',f'tau_{opt_point}']:
						tmp_fit[param] = np.empty((K,len(frequencies)))
					
					def get_rq_param(fit,param,temp):
						"""func to get drift params - enable prediction even if T not in fit_temperatures"""
						T_idx = np.where(self.fit_temperatures==temp)
						if len(T_idx[0]) > 0:
							out = fit[param][T_idx][0]
						else:
							# treat drift as zero if temp was not fitted
							if param=='R_rq':
								out = 0
							elif param in ['tau_rq','phi_rq','k_d']:
								out = 1
						return out
						
					p_params = ['R_rq','tau_rq','phi_rq']
					if drift_model.find('compressed') > -1:
						p_params += ['k_d']#,'t_i','t_f']
					for param in p_params:
						tmp_fit[param] = np.array([get_rq_param(fit,param,temp) for temp in T_orig])

				for k in range(K):
					# Get R_i,R_f, tau_i, and k at T
					try:
						delta_G_i = fit['delta_G_i']
						delta_G_f = fit['delta_G_f']
					except KeyError:
						delta_G_i = fit['delta_G']
						delta_G_f = fit['delta_G']
					
					if drift_model=='':
						tmp_fit['R_i'][k] = fit['R_i_base'][k]*np.exp((delta_G_i[k]/k_B)*(1/T - 1/T_base))
						tmp_fit['tau_i'][k] = fit['tau_i_base'][k]*np.exp((delta_G_i[k]*fit['phi_T'][k]/k_B)*(1/T - 1/T_base))
						tmp_fit['R_f'][k] = fit['R_f_base'][k]*np.exp((delta_G_f[k]/k_B)*(1/T - 1/T_base))						
						tmp_fit['k_d'][k] = fit['k_base'][k]*np.exp((-fit['delta_G_k'][k]/k_B)*(1/T - 1/T_base))
					elif drift_model.find('RQ') > -1:
						tmp_fit[f'R_{opt_point}'][k] = fit[f'R_{opt_point}_base'][k]*np.exp((delta_G_i[k]/k_B)*(1/T - 1/T_base))
						tmp_fit[f'tau_{opt_point}'][k] = fit[f'tau_{opt_point}_base'][k]*np.exp((delta_G_i[k]*fit['phi_T'][k]/k_B)*(1/T - 1/T_base))
						
					# if self.stan_model_name.find('R-err') > -1:
						# delta_lnR = fit['delta_lnR'][np.where(self.fit_temperatures==T_orig[0]),k][0,0]
						# tmp_fit['R_i'][k] *= np.exp(delta_lnR)
						# tmp_fit['tau_i'][k] *= np.exp(delta_lnR*fit['phi_T'][k])
						
				# print(tmp_fit)
				# print(drift_model)
				Z_pred += self._calc_drift_Z(tmp_fit,drift_model,frequencies,times)
				# F_t = self._calc_drift_Ft(fit,drift_model,times)
						
				# R_i_base = fit['R_i_base']
				# tau_i_base = fit['tau_i_base']
				# alpha_HN = fit['alpha_HN']
				# beta_HN = fit['beta_HN']
				# # delta_G = fit['delta_G']
				# try:
					# delta_G_i = fit['delta_G_i']
					# delta_G_f = fit['delta_G_f']
				# except KeyError:
					# delta_G_i = fit['delta_G']
					# delta_G_f = fit['delta_G']
				# phi_T = fit['phi_T']				
				# T_base = fit['T_base']
				
				# R_f_base = fit['R_f_base']
				# k_base = fit['k_base']
				# delta_G_k = fit['delta_G_k']
				# phi_d = fit['phi_d']
				# try:
					# beta_t = fit['beta_t']
				# except KeyError:
					# beta_t = np.ones(len(k_base))
				
				# # Get impedance for each HN peak
				# for k in range(len(R_i_base)):	
					# # Get R_i,R_f, tau_i, and k at T
					# R_i = R_i_base[k]*np.exp((delta_G_i[k]/k_B)*(1/T - 1/T_base))
					# R_f = R_f_base[k]*np.exp((delta_G_f[k]/k_B)*(1/T - 1/T_base))
					# tau_i = tau_i_base[k]*np.exp((delta_G_i[k]*phi_T[k]/k_B)*(1/T - 1/T_base))
					# k_d = k_base[k]*np.exp((-delta_G_k[k]/k_B)*(1/T - 1/T_base))
					
					# # Get R(t) and tau(t)
					# R_t = R_i + (R_f - R_i)*(1 - np.exp(-(k_d*times)**beta_t[k]))
					# tau_t = tau_i + tau_i*((R_f/R_i)**phi_d[k] - 1)*(1 - np.exp(-(k_d*times)**beta_t[k]))
					
					# # Add impedance for kth HN element
					# Z_pred += R_t*pf.HN_impedance(frequencies,tau_t,alpha_HN[k],beta_HN[k])
					
			
			if self.fit_type.find('combi-activation-drift') >= 0:
				Z_pred += self.R_inf_base[sample_index]*np.exp((self.R_inf_dG[sample_index]/k_B)*(1/T - 1/T_base))
				Z_pred += 1j*2*np.pi*frequencies*self.inductance[sample_index]
			else:
				Z_pred += self.R_inf_base*np.exp((self.R_inf_dG/k_B)*(1/T - 1/T_base))
				Z_pred += 1j*2*np.pi*frequencies*self.inductance	
		
		# drift fits
		elif self.fit_type.find('drift') >= 0:
			if times is None:
				raise ValueError('times are required for drift fits')
			elif len(times)!=len(frequencies):
				raise ValueError('times must be array-like of same length as frequencies')
				
			# get drift model
			drift_model = self._get_drift_model('HN-drift')
			
			Z_pred = np.zeros(len(frequencies),dtype=complex)
			for dist_name in distributions:
				# Unpack parameters for convenience
				fit = self.distribution_fits[dist_name].copy()
				
				Z_pred += self._calc_drift_Z(fit,drift_model,frequencies,times)
				F_t = self._calc_drift_Ft(fit,drift_model,times)
				
				# if drift_model=='':
					# R_i = fit['R_i']
					# tau_i = fit['tau_i']
					# alpha_HN = fit['alpha_HN']
					# beta_HN = fit['beta_HN']
					# R_f = fit['R_f']
					# k_d = fit['k_d']
					# phi_d = fit['phi_d']
					
					# # Get impedance for each HN peak
					# for k in range(len(R_i)):	
						# # Get R(t) and tau(t)
						# R_t = R_i[k] + (R_f[k] - R_i[k])*(1 - np.exp(-k_d[k]*times))
						# tau_t = tau_i[k] + tau_i[k]*((R_f[k]/R_i[k])**phi_d[k] - 1)*(1 - np.exp(-k_d[k]*times))
						
						# # Add impedance for kth HN element
						# Z_pred += R_t*pf.HN_impedance(frequencies,tau_t,alpha_HN[k],beta_HN[k])
				# elif drift_model.find('RQ') > -1:
					# if drift_model.find('xi') > -1:
						# R_HN = fit['R_i']
						# tau_HN = fit['tau_i']
					# elif drift_model.find('xf') > -1:
						# R_HN = fit['R_f']
						# tau_HN = fit['tau_f']
					# alpha_HN = fit['alpha_HN']
					# beta_HN = fit['beta_HN']
					
					# # Get impedance for each HN peak
					# Z_list = [R*pf.HN_impedance(frequencies,t0,a,b) for R,t0,a,b in zip(R_HN,tau_HN,alpha_HN,beta_HN)]
					# Z_stack = np.vstack(Z_list)
					# # Sum HN impedances
					# Z_pred += np.sum(Z_stack,axis=0)
					
					# # Z due to time-dependent ZARC
					# R_rq = fit['R_rq']
					# tau_rq = fit['tau_rq']
					# phi_rq = fit['phi_rq']	
					
					# F_t = self._calc_drift_Ft(fit,drift_model,times)
					
					# Z_pred += F_t*(R_rq/(1+(tau_rq*1j*2*np.pi*frequencies)**phi_rq))
			
			if drift_model=='':
				Z_pred += self.R_inf_i + (self.R_inf_f - self.R_inf_i)*(1 - np.exp(-self.R_inf_kd*times))
			elif drift_model.find('xi') > -1:
				Z_pred += self.R_inf_i + (self.R_inf_f - self.R_inf_i)*F_t
			elif drift_model.find('xf') > -1:
				Z_pred += self.R_inf_f + (self.R_inf_f - self.R_inf_i)*F_t
			Z_pred += 1j*2*np.pi*frequencies*self.inductance	
				
		# activation fits	
		elif self.fit_type.find('activation') >= 0:
			if T is None:
				raise ValueError('T is required for activation fits')
			elif type(T) in (list,np.ndarray) and len(T)!=len(frequencies):
				raise ValueError('T must either be a single value or array-like of same length as frequencies')
			elif type(T) in (float,int):
				T = np.ones(len(frequencies))*T
			
			if self.fit_type.find('combi-activation') >= 0:
				if sample_id is None:
					raise ValueError('sample_id is required for combi fit')
				else:
					sample_index = np.where(self.sample_ids==sample_id)	

			if include_T_offset:
				offset = self.get_T_offset(T,sample_id)		
				T = T.copy() + offset
				
			Z_pred = np.zeros(len(frequencies),dtype=complex)
			for dist_name in distributions:
				# Unpack parameters for convenience
				fit = self.distribution_fits[dist_name].copy()
				if self.fit_type.find('combi-activation') >= 0:
					# Get parameters for sample_id
					for param in ['R_HN_base','tau_HN_base','delta_G','phi_T']:
						fit[param] = fit[param][sample_index].flatten()
						
				R_base = fit['R_HN_base']
				tau_base = fit['tau_HN_base']
				delta_G = fit['delta_G']
				phi = fit['phi_T']
				T_base = fit['T_base']
				
				for temp in np.unique(T):
					tidx = np.where(T==temp)
					freq_T = frequencies[tidx]
					
					# Get shape parameters
					if self.fit_type.find('combi-activation') >= 0:
						# Get shape parameters for sample_id (and T if shapeshift)
						if len(fit['alpha_HN'].shape)==3:
							# shapeshift fit - different alpha matrix for each temperature
							temp_index = np.where(self.fit_temperatures==temp)
							alpha_HN = fit['alpha_HN'][temp_index][0][sample_index].flatten()
							beta_HN = fit['beta_HN'][temp_index][0][sample_index].flatten()
						else:
							alpha_HN = fit['alpha_HN'][sample_index].flatten()
							beta_HN = fit['beta_HN'][sample_index].flatten()
					else:
						# Get shape parameters for sample_id (and T if shapeshift)
						if len(fit['alpha_HN'].shape)==2:
							# shapeshift fit - different alpha vector for each temperature
							temp_index = np.where(self.fit_temperatures==temp)
							alpha_HN = fit['alpha_HN'][temp_index].flatten()
							beta_HN = fit['beta_HN'][temp_index].flatten()
						else:
							alpha_HN = fit['alpha_HN']
							beta_HN = fit['beta_HN']
												
					# Get R and tau at specified temperature
					R_HN = R_base*np.exp((delta_G/k_B)*(1/temp - 1/T_base))
					tau_HN = tau_base*np.exp((delta_G*phi/k_B)*(1/temp - 1/T_base))
					
					# Get impedance for each HN peak
					Z_list = [R*pf.HN_impedance(freq_T,t0,a,b) for R,t0,a,b in zip(R_HN,tau_HN,alpha_HN,beta_HN)]
					Z_stack = np.vstack(Z_list)
					# Sum HN impedances
					Z_pred[tidx] += np.sum(Z_stack,axis=0)
			
			if self.fit_type.find('combi-activation') >= 0:
				Z_pred += self.R_inf_base[sample_index]*np.exp((self.R_inf_dG[sample_index]/k_B)*(1/T - 1/T_base))
				Z_pred += 1j*2*np.pi*frequencies*self.inductance[sample_index]
			else:
				Z_pred += self.R_inf_base*np.exp((self.R_inf_dG/k_B)*(1/T - 1/T_base))
				Z_pred += 1j*2*np.pi*frequencies*self.inductance
		
		# combi fits
		elif self.fit_type.find('combi') >= 0:
			if sample_id is None:
				raise ValueError('sample_id is required for combi fit')
				
			sample_index = np.where(self.sample_ids==sample_id)	
			Z_pred = np.zeros(len(frequencies),dtype=complex)
			
			for dist_name in distributions:
				# Unpack parameters for convenience
				fit = self.distribution_fits[dist_name].copy()
				R_HN = fit['R_HN'][sample_index].flatten()
				tau_HN = fit['tau_HN'][sample_index].flatten()
				alpha_HN = fit['alpha_HN'][sample_index].flatten()
				beta_HN = fit['beta_HN'][sample_index].flatten()
				
				# Get impedance for each HN peak
				Z_list = [R*pf.HN_impedance(frequencies,t0,a,b) for R,t0,a,b in zip(R_HN,tau_HN,alpha_HN,beta_HN)]
				Z_stack = np.vstack(Z_list)
				# Sum HN impedances
				Z_pred += np.sum(Z_stack,axis=0)
				
			Z_pred += self.R_inf[sample_index]
			Z_pred += 1j*2*np.pi*frequencies*self.inductance[sample_index]
		# regular fits
		else:
			Z_pred = np.zeros(len(frequencies),dtype=complex)
			for dist_name in distributions:
				# Unpack parameters for convenience
				fit = self.distribution_fits[dist_name].copy()
				R_HN = fit['R_HN']
				tau_HN = fit['tau_HN']
				alpha_HN = fit['alpha_HN']
				beta_HN = fit['beta_HN']
				
				# Get impedance for each HN peak
				Z_list = [R*pf.HN_impedance(frequencies,t0,a,b) for R,t0,a,b in zip(R_HN,tau_HN,alpha_HN,beta_HN)]
				Z_stack = np.vstack(Z_list)
				# Sum HN impedances
				Z_pred += np.sum(Z_stack,axis=0)
				
			Z_pred += self.R_inf
			Z_pred += 1j*2*np.pi*frequencies*self.inductance
			
		return Z_pred
		
	def predict_distribution(self,distribution=None,eval_tau=None,percentile=None,time=None,T=None,sample_id=None,include_T_offset=False):
		# time does nothing currently - included for compatibility only
		if distribution is None:
			distribution = list(self.distribution_fits.keys())[0]
		
		if eval_tau is None:
			if 'tau' in self.distributions[distribution].keys():
				eval_tau = self.distributions[distribution]['tau']
			else:
				tmin = np.log10(np.min(1/2*np.pi*self.f_train))
				tmax = np.log10(np.max(1/2*np.pi*self.f_train))
				eval_tau = np.logspace(tmin-2,tmax+2,200)
				
		if self.fit_type.find('combi') >= 0:
			if sample_id is None:
				raise ValueError('sample_id is required for combi fit')
			else:
				sample_index = np.where(self.sample_ids==sample_id)
				
		if include_T_offset:
			offset = self.get_T_offset(T,sample_id)
			T_orig = deepcopy(T)
			T = deepcopy(T) + offset
		else:
			T_orig = deepcopy(T)
			
		# activation-drift fits
		if self.fit_type.find('activation-drift') >= 0:
			if T is None:
				raise ValueError('T is required for activation fits')
				
			if time is None:
				raise ValueError('time is required for drift fits')
				
			# get drift model
			drift_model = self._get_drift_model('HN-activation-drift')
			
			# Unpack parameters for convenience
			fit = self.distribution_fits[distribution].copy()
			if self.fit_type.find('combi-activation-drift') >= 0:
				# Get parameters for sample_id
				for param in ['R_i_base','tau_i_base','alpha_HN','beta_HN','delta_G','phi_T','R_f_base','k_base','delta_G_k','phi_d']:
					fit[param] = fit[param][sample_index].flatten()
					
			# populate tmp_fit with parameters for _calc_drift_Z
			# need to get params values at temperature T
			tmp_fit = deepcopy(fit)
			T_base = fit['T_base']
			K = len(fit['alpha_HN'])
			
			if drift_model.find('RQ') > -1:
				def get_rq_param(fit,param,temp):
					"""func to get drift params - enable prediction even if T not in fit_temperatures"""
					T_idx = np.where(self.fit_temperatures==temp)
					if len(T_idx[0]) > 0:
						out = fit[param][T_idx][0]
					else:
						# treat drift as zero if temp was not fitted
						if param=='R_rq':
							out = 0
						elif param in ['tau_rq','phi_rq','k_d']:
							out = 1
					return out
						
				p_params = ['R_rq','tau_rq','phi_rq']
				if drift_model.find('compressed') > -1:
					p_params += ['k_d']#,'t_i','t_f']
				for param in p_params:
					tmp_fit[param] = get_rq_param(fit,param,T_orig)

			# Get R_i,R_f, tau_i, and k at T
			try:
				delta_G_i = fit['delta_G_i']
				delta_G_f = fit['delta_G_f']
			except KeyError:
				delta_G_i = fit['delta_G']
				delta_G_f = fit['delta_G']
			
			if drift_model=='':
				tmp_fit['R_i'] = fit['R_i_base']*np.exp((delta_G_i/k_B)*(1/T - 1/T_base))
				tmp_fit['tau_i'] = fit['tau_i_base']*np.exp((delta_G_i*fit['phi_T']/k_B)*(1/T - 1/T_base))
				tmp_fit['R_f'] = fit['R_f_base']*np.exp((delta_G_f/k_B)*(1/T - 1/T_base))						
				tmp_fit['k_d'] = fit['k_base']*np.exp((-fit['delta_G_k']/k_B)*(1/T - 1/T_base))
			elif drift_model.find('RQ') > -1:
				opt_point = drift_model[drift_model.find('x')+1]
				tmp_fit[f'R_{opt_point}'] = fit[f'R_{opt_point}_base']*np.exp((delta_G_i/k_B)*(1/T - 1/T_base))
				tmp_fit[f'tau_{opt_point}'] = fit[f'tau_{opt_point}_base']*np.exp((delta_G_i*fit['phi_T']/k_B)*(1/T - 1/T_base))
					
			# print(tmp_fit)
			F = self._calc_drift_distribution(tmp_fit,drift_model,eval_tau,time)
			
			return F
			# R_i_base = fit['R_i_base']
			# tau_i_base = fit['tau_i_base']
			# alpha_HN = fit['alpha_HN']
			# beta_HN = fit['beta_HN']
			# # delta_G = fit['delta_G']
			# try:
				# delta_G_i = fit['delta_G_i']
				# delta_G_f = fit['delta_G_f']
			# except KeyError:
				# delta_G_i = fit['delta_G']
				# delta_G_f = fit['delta_G']
			# phi_T = fit['phi_T']				
			# T_base = fit['T_base']
			
			# R_f_base = fit['R_f_base']
			# k_base = fit['k_base']
			# delta_G_k = fit['delta_G_k']
			# phi_d = fit['phi_d']
			
			# # Get R_i,R_f, tau_i, and k at T
			# R_i = R_i_base*np.exp((delta_G_i/k_B)*(1/T - 1/T_base))
			# R_f = R_f_base*np.exp((delta_G_f/k_B)*(1/T - 1/T_base))
			# tau_i = tau_i_base*np.exp((delta_G_i*phi_T/k_B)*(1/T - 1/T_base))
			# k_d = k_base*np.exp((-delta_G_k/k_B)*(1/T - 1/T_base))
			
			# # Get R(t) and tau(t)
			# R_t = R_i + (R_f - R_i)*(1 - np.exp(-k_d*time))
			# tau_t = tau_i + tau_i*((R_f/R_i)**phi_d - 1)*(1 - np.exp(-k_d*time))
			
			# R_HN = R_t
			# tau_HN = tau_t
		
		# drift fits
		elif self.fit_type.find('drift') >= 0:
			if time is None:
				raise ValueError('time is required for drift fits')
				
			# get drift model
			drift_model = self._get_drift_model('HN-drift')
			
			# Unpack parameters for convenience
			fit = self.distribution_fits[distribution].copy()
			F = self._calc_drift_distribution(fit,drift_model,eval_tau,time)
			
			return F
			# if drift_model=='':
				# R_i = fit['R_i']
				# tau_i = fit['tau_i']
				# alpha_HN = fit['alpha_HN']
				# beta_HN = fit['beta_HN']
				# R_f = fit['R_f']
				# k_d = fit['k_d']
				# phi_d = fit['phi_d']
				
				# # Get R(t) and tau(t)
				# R_t = R_i + (R_f - R_i)*(1 - np.exp(-k_d*time))
				# tau_t = tau_i + tau_i*((R_f/R_i)**phi_d - 1)*(1 - np.exp(-k_d*time))
				
				# R_HN = R_t
				# tau_HN = tau_t
			
			# elif drift_model.find('RQ') > -1:
				# if drift_model.find('xi') > -1:
					# R_HN = fit['R_i']
					# tau_HN = fit['tau_i']
				# elif drift_model.find('xf') > -1:
					# R_HN = fit['R_f']
					# tau_HN = fit['tau_f']
				# alpha_HN = fit['alpha_HN']
				# beta_HN = fit['beta_HN']
				
				# # get static DRT
				# bases = np.array([pf.HN_distribution(eval_tau,t0,a,b) for t0,a,b in zip(tau_HN,alpha_HN,beta_HN)]).T
				# F = bases@R_HN
			
				# # Get time-dependent ZARC DRT
				# R_rq = fit['R_rq']
				# tau_rq = fit['tau_rq']
				# phi_rq = fit['phi_rq']	

				# F_t = self._calc_drift_Ft(fit,drift_model,time)
				# F_rq = (1/(2*np.pi))*np.sin((1-phi_rq)*np.pi)/(np.cosh(phi_rq*np.log(eval_tau/tau_rq))-np.cos((1-phi_rq)*np.pi))
				
				# F += F_rq*R_rq*F_t
				
				# return F
		
		# activation fits
		elif self.fit_type.find('activation') >= 0:
			if T is None:
				raise ValueError('T is required for activation fits')
					
			# Unpack parameters for convenience
			fit = self.distribution_fits[distribution].copy()
			if self.fit_type.find('combi-activation') >= 0:
				# Get parameters for sample_id
				for param in ['R_HN_base','tau_HN_base','delta_G','phi_T']:
					fit[param] = fit[param][sample_index].flatten()
					
				# Get shape parameters for sample_id (and T if shapeshift)
				if len(fit['alpha_HN'].shape)==3:
					# shapeshift fit - different alpha matrix for each temperature
					temp_index = np.where(self.fit_temperatures==T)
					alpha_HN = fit['alpha_HN'][temp_index][0][sample_index].flatten()
					beta_HN = fit['beta_HN'][temp_index][0][sample_index].flatten()
				else:
					alpha_HN = fit['alpha_HN'][sample_index].flatten()
					beta_HN = fit['beta_HN'][sample_index].flatten()
			else:
				# Get shape parameters for sample_id (and T if shapeshift)
				if len(fit['alpha_HN'].shape)==2:
					# shapeshift fit - different alpha matrix for each temperature
					temp_index = np.where(self.fit_temperatures==T)
					alpha_HN = fit['alpha_HN'][temp_index].flatten()
					beta_HN = fit['beta_HN'][temp_index].flatten()
				else:
					alpha_HN = fit['alpha_HN']
					beta_HN = fit['beta_HN']
					
			R_base = fit['R_HN_base']
			tau_base = fit['tau_HN_base']
			delta_G = fit['delta_G']
			phi = fit['phi_T']
			T_base = fit['T_base']
			
			if len(alpha_HN.shape)==3:
				# shapeshift fit - different alpha matrix for each temperatures
				temp_index = np.where(self.fit_temperatures==T)
				alpha_HN = alpha_HN[temp_index]
				beta_HN = beta_HN[temp_index]
			
			# Get R and tau at specified temperature
			R_HN = R_base*np.exp((delta_G/k_B)*(1/T - 1/T_base))
			tau_HN = tau_base*np.exp((delta_G*phi/k_B)*(1/T - 1/T_base))
		# Combi fits
		elif self.fit_type.find('combi') >= 0:
			# Unpack parameters for convenience
			fit = self.distribution_fits[distribution].copy()
			R_HN = fit['R_HN'][sample_index].flatten()
			tau_HN = fit['tau_HN'][sample_index].flatten()
			alpha_HN = fit['alpha_HN'][sample_index].flatten()
			beta_HN = fit['beta_HN'][sample_index].flatten()
		else:
			# Unpack parameters for convenience
			fit = self.distribution_fits[distribution].copy()
			R_HN = fit['R_HN']
			tau_HN = fit['tau_HN']
			alpha_HN = fit['alpha_HN']
			beta_HN = fit['beta_HN']
			
		bases = np.array([pf.HN_distribution(eval_tau,t0,a,b) for t0,a,b in zip(tau_HN,alpha_HN,beta_HN)]).T
		F = bases@R_HN
		
		return F
		
	def predict_Rp(self,distributions=None,percentile=None,T=None,times=None,sample_id=None,include_T_offset=False):
		"""Predict polarization resistance
		
		Parameters:
		-----------
		distributions: str or list (default: None)
			Distribution name or list of distribution names for which to sum Rp contributions. 
			If None, include all distributions
		percentile: float (default: None)
			If specified, predict a percentile (0-100) of the polarization resistance. Only applicable for bayes_fit results.
		"""
		if distributions is not None:
			if type(distributions)==str:
				distributions = [distributions]
		else:
			distributions = list(self.distribution_fits.keys())
			
		if len(distributions) > 1:
			Z_range = self.predict_Z(np.array([1e20,1e-20]),distributions=distributions,T=T,times=times,sample_id=sample_id,percentile=percentile,include_T_offset=include_T_offset)
			Rp = np.real(Z_range[1] - Z_range[0])
		else:
			info = self.distributions[distributions[0]]
			# just calculate Z at very high and very low frequencies and take the difference in Z'
			# could do calcs using coefficients, but this is fast and accurate enough for now (and should work for any arbitrary distribution)
			if percentile is None:
				Z_range = self.predict_Z(np.array([1e20,1e-20]),distributions=distributions,T=T,times=times,sample_id=sample_id,include_T_offset=include_T_offset)
				Rp = np.real(Z_range[1] - Z_range[0])
			else:
				# get the distribution of Rp
				Z_mat = self.predict_Z_distribution(np.array([1e20,1e-20]),distributions=distributions,T=T,times=times,sample_id=sample_id)
				Rp_sample = np.real(Z_mat[:,1] - Z_mat[:,0])
				Rp = np.percentile(Rp_sample,percentile)
				
		return Rp
	
	def predict_sigma(self,frequencies,percentile=None,T=None,sample_id=None,times=None,include_T_offset=False):
		if percentile is not None and self.fit_type!='bayes':
			raise ValueError('Percentile prediction is only available for bayes_fit')
			
		# if np.min(rel_round(self.f_train,10)==rel_round(frequencies,10))==True:
			# # if frequencies are training frequencies, just use sigma_tot output
			# if self.fit_type=='bayes' and percentile is not None:
				# sigma_tot = np.percentile(self._sample_result['sigma_tot'],percentile,axis=0)*self._Z_scale
			# elif self.fit_type=='bayes' or self.fit_type[:3]=='map':
				# sigma_tot = self.error_fit['sigma_tot']
			# else:
				# raise ValueError('Error scale prediction only available for bayes_fit and map_fit')
				
			# sigma_re = sigma_tot[:len(self.f_train)].copy()
			# sigma_im = sigma_tot[len(self.f_train):].copy()
		# else:
		
		# if frequencies are not training frequencies, calculate from parameters
		# this doesn't match sigma_tot perfectly
		if self.fit_type=='bayes' and percentile is not None:
			sigma_res = np.percentile(self._sample_result['sigma_res'],percentile)*self._Z_scale
			alpha_prop = np.percentile(self._sample_result['alpha_prop'],percentile)
			alpha_re = np.percentile(self._sample_result['alpha_re'],percentile)
			alpha_im = np.percentile(self._sample_result['alpha_im'],percentile)
			try:
				sigma_out = np.percentile(self._sample_result['sigma_out'],percentile,axis=0)*self._Z_scale
			except ValueError:
				sigma_out = np.zeros(2*len(self.f_train))
		elif self.fit_type=='bayes' or self.fit_type[:3] in ('map','hmc'):
			sigma_res = self.error_fit['sigma_res']
			alpha_prop = self.error_fit['alpha_prop']
			alpha_re = self.error_fit['alpha_re']
			alpha_im = self.error_fit['alpha_im']
			try:
				sigma_out = self.error_fit['sigma_out']
			except KeyError:
				sigma_out = np.zeros(2*len(self.f_train))
		else:
			raise ValueError('Error scale prediction only available for bayes_fit and map_fit')
			
		sigma_min = self.error_fit['sigma_min']
			
		Z_pred = self.predict_Z(frequencies,percentile=percentile,T=T,sample_id=sample_id,times=times,include_T_offset=include_T_offset)
		
		# Get rel_Z_scale for activation and drift fits
		if self.fit_type.find('activation') >= 0 or self.fit_type.find('drift') >= 0:
			if self.fit_type.find('combi-activation') >= 0:
				if type(T) in (float,int):
					# If T is a scalar, rel_Z_scale is a scalar
					# T_base = self.distribution_fits[list(self.distributions.keys()[0]]['T_base']
					if T in self._basis_Z_scale.keys():
						rel_Z_scale = self._basis_Z_scale[T]/self._Z_scale
					else:
						# Estimate the Z_scale for sample_id at T relative to sample mean
						Z_T = Z_pred
						Zmod_T = (Z_T*Z_T.conjugate())**0.5
						Z_scale_T = np.std(Zmod_T)/np.sqrt(len(Z_T)/81)
						sample_factors = []
						for temp in self._basis_Z_scale.keys():
							# Aggregate sample_id Z_scale relative to mean Z_scale for fitted temps
							Z_temp = self.predict_Z(frequencies,T=T,sample_id=sample_id,times=times,include_T_offset=include_T_offset)
							Zmod_temp = (Z_temp*Z_temp.conjugate())**0.5
							Z_scale_temp = np.std(Zmod_temp)/np.sqrt(len(Z_temp)/81)
							sample_factors.append(Z_scale_temp/self._basis_Z_scale[temp])
						sample_factor = np.mean(sample_factors)	
						rel_Z_scale = Z_scale_T/(sample_factor*self._Z_scale)
				else:
					# If T is a vector, rel_Z_scale is a vector
					rel_Z_scale = np.ones(len(frequencies))
					# If any temps are not in fitted temperatures, need to estimate sample_factor
					in_fit_temps = [temp in self._basis_Z_scale.keys() for temp in np.unique(T)]
					if np.sum(in_fit_temps) < len(np.unique(T)):
						sample_factors = []
						for temp in self._basis_Z_scale.keys():
							# Aggregate sample_id Z_scale relative to mean Z_scale for fitted temps
							Z_temp = self.predict_Z(frequencies,T=T,sample_id=sample_id,times=times,include_T_offset=include_T_offset)
							Zmod_temp = (Z_temp*Z_temp.conjugate())**0.5
							Z_scale_temp = np.std(Zmod_temp)/np.sqrt(len(Z_temp)/81)
							sample_factors.append(Z_scale_temp/self._basis_Z_scale[temp])
						sample_factor = np.mean(sample_factors)	
							
					for temp in np.unique(T):
						tidx = np.where(T==temp)
						if T in self._basis_Z_scale.keys():
							rel_Z_scale[tidx] = self._basis_Z_scale[T]/self._Z_scale
						else:
							# Estimate the Z_scale for sample_id at T relative to sample mean
							Z_T = Z_pred[tidx]
							Zmod_T = (Z_T*Z_T.conjugate())**0.5
							Z_scale_T = np.std(Zmod_T)/np.sqrt(len(Z_T)/81)
							rel_Z_scale[tidx] = Z_scale_T/(sample_factor*self._Z_scale)
			# Get relative Z scale for each temperature
			elif self.fit_type.find('activation') >= 0 or self.fit_type.find('activation-drift') >= 0:
				if type(T) in (float,int):
					# If T is a scalar, rel_Z_scale is a scalar
					Z_T = Z_pred
					Zmod_T = (Z_T*Z_T.conjugate())**0.5
					Z_scale_T = np.std(Zmod_T)/np.sqrt(len(Z_T)/81)
					rel_Z_scale = Z_scale_T/self._Z_scale
				else:
					# If T is a vector, rel_Z_scale is a vector
					rel_Z_scale = np.ones(len(frequencies))				
					for temp in np.unique(T):
						tidx = np.where(T==temp)
						Z_T = Z_pred[tidx]
						Zmod_T = (Z_T*Z_T.conjugate())**0.5
						Z_scale_T = np.std(Zmod_T)/np.sqrt(len(Z_T)/81)
						rel_Z_scale[tidx] = Z_scale_T/self._Z_scale
						# print(temp,tidx)
			# For drift fits, get relative Z scale for each distinct measurement
			elif self.fit_type.find('drift') >= 0:
				# identify measurement start and end indices
				start_idx = np.where(np.diff(frequencies) > 0)[0] + 1
				if len(start_idx) > 0:
					meas_start_indices = np.concatenate(([0],start_idx)).astype(int)
				else:
					meas_start_indices = np.array([0],dtype=int)
				meas_end_indices = np.zeros_like(meas_start_indices)
				meas_end_indices[:-1] = meas_start_indices[1:]
				meas_end_indices[-1] = len(frequencies)
				
				rel_Z_scale = np.ones(len(frequencies))	
				for start,end in zip(meas_start_indices,meas_end_indices):
					Z_i = Z_pred[start:end]
					Zmod_i = (Z_i*Z_i.conjugate())**0.5
					Z_scale_i = np.std(Zmod_i)/np.sqrt(len(Z_i)/81)
					rel_Z_scale[start:end] = Z_scale_i/self._Z_scale

						
				
			sigma_base = np.sqrt((sigma_res*rel_Z_scale)**2 + np.min(sigma_out)**2 + (sigma_min*rel_Z_scale)**2)
		else:
			sigma_base = np.sqrt((sigma_res)**2 + np.min(sigma_out)**2 + sigma_min**2)
		
		sigma_re = np.sqrt(sigma_base**2 + (alpha_prop*Z_pred.real)**2 + (alpha_re*Z_pred.real)**2 + (alpha_im*Z_pred.imag)**2)
		sigma_im = np.sqrt(sigma_base**2 + (alpha_prop*Z_pred.imag)**2 + (alpha_re*Z_pred.real)**2 + (alpha_im*Z_pred.imag)**2)
				
		return sigma_re,sigma_im	
		
	def score(self,frequencies,Z,metric='chi_sq',weights=None,part='both',T=None,sample_id=None,times=None,include_T_offset=False):
		weights = self._format_weights(frequencies,Z,weights,part)
		Z_pred = self.predict_Z(frequencies,T=T,sample_id=sample_id,times=times,include_T_offset=include_T_offset)
		if part=='both':
			Z_pred = np.concatenate((Z_pred.real,Z_pred.imag))
			Z = np.concatenate((Z.real,Z.imag))
			weights = np.concatenate((weights.real,weights.imag))
		else:
			Z_pred = getattr(Z_pred,part)
			Z = getattr(Z,part)
			weights = getattr(weights,part)
			
		if metric=='chi_sq':
			score = np.sum(((Z_pred - Z)*weights)**2)/len(frequencies)
		elif metric=='r2':
			score = r2_score(Z,Z_pred,sample_weight=weights)
		else:
			raise ValueError(f"Invalid metric {metric}. Options are 'chi_sq', 'r2'")
			
		return score
		
	def extract_HN_info(self,distribution=None,sort=True,filter_R=False,R_rthresh=0.005,sample_id=None,T=None):			
		# If no distribution specified, use first distribution
		if distribution is None:
			distribution = list(self.distributions.keys())[0]
			
		# get parameters and parse
		if self.fit_type=='map-activation' or self.fit_type=='map-combi-activation':
			params = self.distribution_fits[distribution].copy()
			if self.fit_type=='map-combi-activation':
				# Get parameters for sample_id
				if sample_id is None:
					raise ValueError('sample_id is required for combi fit')
				else:
					sample_index = np.where(self.sample_ids==sample_id)
					
				for param in ['R_HN_base','tau_HN_base','delta_G','phi_T']:
					params[param] = params[param][sample_index].flatten()
					
				# Get shape parameters for sample_id (and T if shapeshift)
				if len(params['alpha_HN'].shape)==3:
					# shapeshift fit - different alpha matrix for each temperature
					alpha_HN = dict(zip(self.fit_temperatures,np.array([alpha_T[sample_index].flatten() for alpha_T in params['alpha_HN']])))
					beta_HN = dict(zip(self.fit_temperatures,np.array([beta_T[sample_index].flatten() for beta_T in params['beta_HN']])))
				else:
					alpha_HN = params['alpha_HN'][sample_index].flatten()
					beta_HN = params['beta_HN'][sample_index].flatten()
			elif self.fit_type=='map-activation':
				# Get shape parameters for sample_id (and T if shapeshift)
				if len(params['alpha_HN'].shape)==2:
					# shapeshift fit - different alpha matrix for each temperature
					alpha_HN = dict(zip(self.fit_temperatures,params['alpha_HN']))
					beta_HN = dict(zip(self.fit_temperatures,params['beta_HN']))
				else:
					alpha_HN = params['alpha_HN']
					beta_HN = params['beta_HN']
				
			
			R_base = params['R_HN_base']
			t0_base = params['tau_HN_base']
			delta_G = params['delta_G']
			phi = params['phi_T']
			
			# sort by time constant
			if sort:
				sort_idx = np.argsort(t0_base)
				R_base = R_base[sort_idx]
				t0_base = t0_base[sort_idx]
				if type(alpha_HN)==np.ndarray:
					alpha_HN = alpha_HN[sort_idx]
					beta_HN = beta_HN[sort_idx]
				else:
					alpha_HN = {k:v[sort_idx] for k,v in alpha_HN.items()}
					beta_HN = {k:v[sort_idx] for k,v in beta_HN.items()}
				delta_G = delta_G[sort_idx]
				phi = phi[sort_idx]
				
			# filter out insignificant peaks
			if filter_R:
				R_max = np.max(np.abs(R_base))
				big_idx = np.where(np.abs(R_base)>=R_max*R_rthresh)
				R_base = R_base[big_idx]
				t0_base = t0_base[big_idx]
				if type(alpha_HN)==np.ndarray:
					alpha_HN = alpha_HN[big_idx]
					beta_HN = beta_HN[big_idx]
				else:
					alpha_HN = {k:v[big_idx] for k,v in alpha_HN.items()}
					beta_HN = {k:v[big_idx] for k,v in beta_HN.items()}
				delta_G = delta_G[big_idx]
				phi = phi[big_idx]
			
			
			# make dict for easy reading
			if T is not None:
				# get parameters at T
				info = {}
				T_base = params['T_base']
				info['num_peaks'] = len(R_base)
				info['R'] = R_base*np.exp((delta_G/k_B)*(1/T - 1/T_base))
				info['tau_0'] = t0_base*np.exp((phi*delta_G/k_B)*(1/T - 1/T_base))
				if type(alpha_HN)==np.ndarray:
					info['alpha'] = alpha_HN
					info['beta'] = beta_HN
				else:
					info['alpha'] = alpha_HN[T]
					info['beta'] = beta_HN[T]
				info['delta_G'] = delta_G
				info['phi_T'] = phi
				
			else:
				# get all parameters to define temperature dependence
				info = {}
				info['num_peaks'] = len(R_base)
				info['R_base'] = R_base
				info['tau_base'] = t0_base
				info['alpha'] = alpha_HN
				info['beta'] = beta_HN
				info['delta_G'] = delta_G
				info['phi_T'] = phi
			
		else:
			params = self.distribution_fits[distribution].copy()
			if self.fit_type=='map-combi':
				# Get parameters for sample_id
				if sample_id is None:
					raise ValueError('sample_id is required for combi fit')
				else:
					sample_index = np.where(self.sample_ids==sample_id)	
				for param in ['R_HN','tau_HN','alpha_HN','beta_HN']:
					params[param] = params[param][sample_index].flatten()
				
			R = params['R_HN']
			t0 = params['tau_HN']
			alpha = params['alpha_HN']
			beta = params['beta_HN']
			
			# sort by time constant
			if sort:
				sort_idx = np.argsort(t0)
				R = R[sort_idx]
				t0 = t0[sort_idx]
				alpha = alpha[sort_idx]
				beta = beta[sort_idx]
				
			# filter out insignificant peaks
			if filter_R:
				R_max = np.max(np.abs(R))
				big_idx = np.where(np.abs(R)>=R_max*R_rthresh)
				R = R[big_idx]
				t0 = t0[big_idx]
				alpha = alpha[big_idx]
				beta = beta[big_idx]
				
			num_peaks = len(R)
			
			# make dict for easy reading
			info = {'num_peaks':num_peaks,
					'R':R,
					'tau_0':t0,
					'alpha':alpha,
					'beta':beta
					}
		
		return info
		 
	def get_sample_index(self,sample_id):
		
		if type(sample_id) in (list,np.ndarray):
			index = np.array([np.where(self.sample_ids==id)[0][0] for id in sample_id])
		else:
			index = np.where(self.sample_ids==sample_id)
		
		return index
		
	def get_T_offset(self,T,sample_id=None):
		"""
		Get temperature offset 
		"""
		
		def _get_T_offset(T,sample_id):
			if self.fit_type.find('combi') >= 0:
				if sample_id is None:
					raise ValueError('sample_id is required for combi fits')
				
				sample_index = self.get_sample_index(sample_id)
				
				if T in self.fit_temperatures:
					T_offset = self.T_offset[sample_index,np.where(self.fit_temperatures==T)][0,0]
				else:
					T_offset = 0
			else:
				if T in self.fit_temperatures:
					T_offset = self.T_offset[np.where(self.fit_temperatures==T)][0]
				else:
					T_offset = 0
					
			return T_offset
		
		# get dimensions
		if type(T) in (list,np.ndarray):
			P = len(T)
		else: 
			P = 1
		if type(sample_id) in (list,np.ndarray):
			J = len(sample_id)
		else: 
			J = 1
			
		# for temp in np.unique(T):
			# if temp not in self.fit_temperatures:
				# warnings.warn(f'{temp} is not in fit_temperatures. T_offset returned as zero')
			
		if J==1 and P==1:
			out = _get_T_offset(T,sample_id)
		else:
			T = np.atleast_1d(T)
			sample_id = np.atleast_1d(sample_id)
			out = np.zeros((J,P))
			for j in range(J):
				out[j] = [_get_T_offset(T[p],sample_id[j]) for p in range(P)]
			if J==1 or P==1:
				out = out.flatten()
			
		return out
	
	
	# ===============================================
	# Methods for saving and loading fits
	# ===============================================
	def get_fit_attributes(self,which='all'):
		fit_attributes = {'common':{'core':['distributions','distribution_fits','f_train','_Z_scale','fit_type','inductance'],'detail':[]},
						'map':{'core':['stan_model_name','error_fit','R_inf'],'detail':['_stan_input','_init_params','_opt_result']},
						'map-combi':{'core':['stan_model_name','error_fit','R_inf','sample_ids'],'detail':['_stan_input','_init_params','_opt_result']},
						'map-activation':{'core':['stan_model_name','error_fit','R_inf_base','R_inf_dG','fit_temperatures','T_offset'],
								'detail':['_stan_input','_init_params','_opt_result']},
						'map-drift':{'core':['stan_model_name','error_fit','R_inf_i','R_inf_f','R_inf_kd'],
								'detail':['_stan_input','_init_params','_opt_result']},
						'map-activation-drift':{'core':['stan_model_name','error_fit','R_inf_base','R_inf_dG','fit_temperatures','T_offset'],
								'detail':['_stan_input','_init_params','_opt_result']},
						'map-combi-activation':{'core':['stan_model_name','error_fit','R_inf_base','R_inf_dG','sample_ids','_basis_Z_scale','fit_temperatures','T_offset'],
								'detail':['_stan_input','_init_params','_opt_result']},
						'map-combi-activation-drift':{'core':['stan_model_name','error_fit','R_inf_base','R_inf_dG','sample_ids','_basis_Z_scale','fit_temperatures','T_offset'],
								'detail':['_stan_input','_init_params','_opt_result']},
								
						'hmc-activation-drift':{'core':['stan_model_name','error_fit','R_inf_base','R_inf_dG','fit_temperatures','T_offset'],
								'detail':['_stan_input','_init_params','_sample_result']},
						'hmc-combi-activation':{'core':['stan_model_name','error_fit','R_inf_base','R_inf_dG','sample_ids','_basis_Z_scale','fit_temperatures','T_offset'],
								'detail':['_stan_input','_init_params','_sample_result']},
					}
		
		if which=='all':
			att = sum([v for v in fit_attributes['common'].values()],[]) + sum([v for v in fit_attributes[self.fit_type].values()],[])
		else:
			att = fit_attributes['common'][which] + fit_attributes[self.fit_type][which]
		
		return att
	
	def save_fit_data(self,filename=None,which='all',add_attributes=[]):
		# get names of attributes to be stored
		store_att = self.get_fit_attributes(which)
		store_att += add_attributes

		fit_data = {}
		for att in store_att:
			fit_data[att] = getattr(self,att)
			
		if filename is not None:
			# save to file
			save_pickle(fit_data,filename)
		else:
			# return dict
			return fit_data
		
	def load_fit_data(self,data):
		if type(data)==str:
			# data is filename - load file
			fit_data = load_pickle(data)
		else:
			# data is dict 
			fit_data = data
		
		for k,v in fit_data.items():
			setattr(self,k,v)
		
	# ===============================================
	# Testing models
	# ===============================================
	def map_fit_constrained(self,frequencies,Z,part='both',scale_Z=True,nonneg=True,outliers=False,
		add_stan_data={},
		# initialization parameters
		init_from_ridge=False,init_from_map=False,init_basis_freq=None,init_drt_fit=None,init_kw={},peakfit_kw={},
		# optimization control
		sigma_min=0.002,max_iter=50000,random_seed=1234,inductance_scale=1,outlier_lambda=5):
		"""
		Obtain the maximum a posteriori estimate of the defined distribution(s) (and all model parameters).
		
		Parameters:
		-----------
		frequencies: array
			Measured frequencies
		Z: complex array
			Measured (complex) impedance values. Must have same length as frequencies
		part: str, optional (default: 'both')
			Which part of the impedance data to fit. Options: 'both', 'real', 'imag'
		scale_Z: bool, optional (default: True)
			If True, scale impedance by the factor sqrt(N)/std(|Z|) to normalize for magnitude and sample size
		init_from_ridge: bool, optional (default: False)
			If True, use the hyperparametric ridge solution to initialize the Bayesian fit. 
			Only valid for single-distribution fits
		nonneg: bool, optional (default: False)
			If True, constrain the DRT to non-negative values
		outliers: bool, optional (default: False)
			If True, enable outlier identification via independent error contribution variable
		sigma_min: float, optional (default: 0.002)
			Impedance error floor. This is necessary to avoid sampling/optimization errors.
			Values smaller than the default (0.002) may enable slightly closer fits of very clean data,
			but may also result in sampling/optimization errors that yield unexpected results.
		max_iter: int, optional (default: 50000)
			Maximum number of iterations to allow the optimizer to perform
		random_seed: int, optional (default: 1234)
			Random seed for optimizer
		inductance_scale: float, optional (default: 1)
			Scale (std of normal prior) of the inductance. Lower values will constrain the inductance,
			which may be helpful if the estimated inductance is anomalously large (this may occur if your
			measured impedance data does not extend to high frequencies, i.e. 1e5-1e6 Hz)
		outlier_lambda: float, optional (default: 5)
			Lambda parameter (inverse scale) of the exponential prior on the outlier error contribution.
			Smaller values will make it easier for points to be flagged as outliers 
		ridge_kw: dict, optional (default: {})
			Keyword arguments to pass to ridge_fit if init_from_ridge==True.
		"""
		# load stan model
		model_str = 'Series_HN-constrained_pos_StanModel.pkl'
		model = load_pickle(os.path.join(script_dir,'stan_model_files',model_str))
		self.stan_model_name = model_str
		model_type = model_str.split('_')[0]

		# perform scaling
		Z_scaled = self._scale_Z(Z,'map')
		self.f_train = frequencies
		
		# prepare data for stan model
		dat = self._prep_stan_data(frequencies,Z_scaled,part,model_type,sigma_min,mode='optimize',inductance_scale=inductance_scale)
		tau = 1/(2*np.pi*self.basis_freq)
		self.distributions[list(self.distributions.keys())[0]]['tau'] = tau
		dat['lntau_basis'] = np.log(tau)
		dat['lntau_box_width'] = np.abs(np.mean(np.diff(np.log(tau))))
		dat['K'] = len(self.basis_freq)
		dat['ups_alpha'] = 20
		dat['ups_beta'] = 0.5
		dat.update(add_stan_data)
		self._stan_input = dat.copy()
		
		num_HN = len(self.basis_freq)
		
		# Get initial parameter values
		iv = {}
		if np.sum([init_from_ridge,init_from_map,init_drt_fit is not None]) > 1:
			raise ValueError('Only one of init_from_ridge, init_from_map, and init_drt_fit may be used')
		if init_from_ridge:
			if len(self.distributions) > 1:
				raise ValueError('Ridge initialization can only be performed for single-distribution fits')
			else:
				init = self._get_init_from_ridge(frequencies,Z,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
				iv.update(init)
		elif init_from_map:
			init = self._get_init_from_map(frequencies,Z,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
			iv.update(init)
		elif init_drt_fit is not None:
			init = self._get_constrained_init_from_drt(init_drt_fit,inductance_scale,peakfit_kw)
			iv.update(init)
		else:
			# distribute lntau values uniformly
			iv['lntau_shift'] = np.zeros(len(tau))
			# iv['R_HN'] = np.ones(num_HN)
			# iv['alpha_HN'] = np.ones(num_HN)*0.95
			# iv['beta_HN'] = np.ones(num_HN)*0.8
			# iv['upsilon'] = np.ones(num_HN)*0.2*dat['ups_alpha']/dat['ups_beta']
		
		if outliers:
			raise ValueError('Outlier model not yet implemented')
			# initialize sigma_out near zero, everything else randomly
			iv['sigma_out_raw'] = np.zeros(2*len(Z)) + 0.1
			dat['so_invscale'] = outlier_lambda
		
			
		def init():
			return iv
		self._init_params = iv
				
		# print(iv)
		
		# optimize posterior
		self._opt_result = model.optimizing(dat,iter=max_iter,seed=random_seed,init=init)
		
		# extract coefficients
		self.distribution_fits = {}
		self.error_fit = {}
		if model_type in ['Series','Parallel']:
			dist_name = [k for k,v in self.distributions.items() if v['dist_type']==model_type.lower()][0]
			dist_type = self.distributions[dist_name]['dist_type']
			self.distribution_fits[dist_name] = {'R_HN':self._rescale_coef(self._opt_result['R_HN'],dist_type)}
			self.distribution_fits[dist_name]['tau_HN'] = np.exp(self._opt_result['lntau_HN'])
			self.distribution_fits[dist_name]['alpha_HN'] = self._opt_result['alpha_HN']
			self.distribution_fits[dist_name]['beta_HN'] = self._opt_result['beta_HN']
			
			self.R_inf = self._rescale_coef(self._opt_result['Rinf'],'series')
			self.inductance = self._rescale_coef(self._opt_result['induc'],'series')
		
		# store error structure parameters
		# scaled parameters
		self.error_fit['sigma_min'] = self._rescale_coef(sigma_min,'series')
		for param in ['sigma_tot','sigma_res']:
			self.error_fit[param] = self._rescale_coef(self._opt_result[param],'series')
		# unscaled parameters
		for param in ['alpha_prop','alpha_re','alpha_im']:
			self.error_fit[param] = self._opt_result[param]
		# outlier contribution
		if outliers:
			self.error_fit['sigma_out'] = self._rescale_coef(self._opt_result['sigma_out'],'series')
		
		self.fit_type = 'map'
		
	def map_combi_fit(self,frequencies,Z,sample_ids,x,y,part='both',scale_Z=True,nonneg=True,outliers=False,
		min_tau=None,max_tau=None,max_comparison_distance=None,num_HN=5,bounded=False,repulse=False,centered=False,add_stan_data={},
		# initialization parameters
		init_from_ridge=False,init_from_map=False,init_basis_freq=None,init_drt_fit=None,init_values=None,init_kw={},peakfit_kw={},
		# optimization control
		sigma_min=0.002,max_iter=50000,random_seed=1234,inductance_scale=1,outlier_lambda=5):
		"""
		Obtain the maximum a posteriori estimate of the defined distribution(s) (and all model parameters).
		
		Parameters:
		-----------
		frequencies: JxN array
			Measured frequencies
		Z: complex JxN array
			Measured (complex) impedance values. Must have same length as frequencies
		x: J-vector
			Vector of x-coordinates
		y: J-vector
			Vector of y-coordinates
		part: str, optional (default: 'both')
			Which part of the impedance data to fit. Options: 'both', 'real', 'imag'
		scale_Z: bool, optional (default: True)
			If True, scale impedance by the factor sqrt(N)/std(|Z|) to normalize for magnitude and sample size
		init_from_ridge: bool, optional (default: False)
			If True, use the hyperparametric ridge solution to initialize the Bayesian fit. 
			Only valid for single-distribution fits
		nonneg: bool, optional (default: False)
			If True, constrain the DRT to non-negative values
		outliers: bool, optional (default: False)
			If True, enable outlier identification via independent error contribution variable
		sigma_min: float, optional (default: 0.002)
			Impedance error floor. This is necessary to avoid sampling/optimization errors.
			Values smaller than the default (0.002) may enable slightly closer fits of very clean data,
			but may also result in sampling/optimization errors that yield unexpected results.
		max_iter: int, optional (default: 50000)
			Maximum number of iterations to allow the optimizer to perform
		random_seed: int, optional (default: 1234)
			Random seed for optimizer
		inductance_scale: float, optional (default: 1)
			Scale (std of normal prior) of the inductance. Lower values will constrain the inductance,
			which may be helpful if the estimated inductance is anomalously large (this may occur if your
			measured impedance data does not extend to high frequencies, i.e. 1e5-1e6 Hz)
		outlier_lambda: float, optional (default: 5)
			Lambda parameter (inverse scale) of the exponential prior on the outlier error contribution.
			Smaller values will make it easier for points to be flagged as outliers 
		ridge_kw: dict, optional (default: {})
			Keyword arguments to pass to ridge_fit if init_from_ridge==True.
		"""
		# load stan model
		# if max_comparison_distance is not None:
		if bounded:
			if repulse:
				if centered:
					model_str = 'Series_HN-combi_pos_trunc_bounded_repulse_centered_StanModel.pkl'
				else:
					model_str = 'Series_HN-combi_pos_trunc_bounded_repulse_StanModel.pkl'
			else:
				model_str = 'Series_HN-combi-trunc-bounded_pos_StanModel.pkl'
		else:
			model_str = 'Series_HN-combi-trunc_pos_StanModel.pkl'	
		# else:
			# model_str = 'Series_HN-combi_pos_StanModel.pkl'	
		model = load_pickle(os.path.join(script_dir,'stan_model_files',model_str))
		self.stan_model_name = model_str
		model_type = model_str.split('_')[0]
		
		# get dimensions
		J = frequencies.shape[0]
		N = frequencies.shape[1]
		K = num_HN
		
		# perform scaling
		Z_scaled = self._scale_Z(Z.flatten(),'map')
		Z_scaled = np.reshape(Z_scaled,Z.shape)
		Z_mat = np.concatenate((Z_scaled.real,Z_scaled.imag),axis=1)
		
		
		self.f_train = frequencies[0]
		self.sample_ids = np.array(sample_ids)
		
		# prepare data for stan model
		dat = {'N': N,
			'J': J,
			'K': K,
			'Z': Z_mat,
			# 'D': D,
			'x_coord': x,
			'y_coord': y,
			'freq':frequencies,
			'induc_scale': inductance_scale,
			'sigma_min':sigma_min,
			'ups_alpha': 1,
			'ups_beta':0.5,
			'sigma_lntau_scale': 0.1,
			'sigma_lnR_scale': 1,
			'sigma_alpha_scale': 0.1,
			'sigma_beta_scale': 0.1,
			# 'dxy_effect_scale': 0.35,
			# 'dist_floor': 1e-3
			}
		
		# if tau boundaries not specified, allow HN peaks to extend 2 decades beyond measurement range
		if min_tau is None:
			min_tau = np.min(1/(2*np.pi*frequencies))/100
		if max_tau is None:
			max_tau = np.max(1/(2*np.pi*frequencies))*100
		dat['min_tau_HN'] = min_tau
		dat['max_tau_HN'] = max_tau
		# dat['K'] = K
		
		if repulse:
			dat['r_lntau_scale'] = 1
		if centered:
			dat['sigma_lntau_HN_dev_alpha'] = 2
			dat['sigma_lntau_HN_dev_beta'] = 0.5
			dat['r_lntau_center_scale'] = 2
		
		# if max_comparison_distance is not None:
		# calculate distance matrix
		dx = np.tile(x,(len(x),1)) - np.tile(x,(len(x),1)).T
		dy = np.tile(y,(len(y),1)) - np.tile(y,(len(y),1)).T
		D = np.sqrt(dx**2 + dy**2)
		# get upper triangle of distance matrix and flatten
		J
		distance_vec = np.zeros(int(J*(J-1)/2))
		pos = 0
		for i in range(J):
			for j in range(i+1,J):
				distance_vec[pos] = D[i,j]
				pos += 1
		# get indices of distances that are below comparison threshold
		if max_comparison_distance is None:
			max_comparison_distance = np.max(distance_vec)
		compare_idx = np.where(distance_vec <= max_comparison_distance)
		# print(compare_idx[0])
		# print(distance_vec)
		# print(distance_vec[compare_idx])
		dat['compare_idx'] = compare_idx[0] + 1 # indexing starts at 0 in stan
		dat['M'] = len(compare_idx[0])
		print('Full distance vector length:',len(distance_vec))
		print('Truncated distance vector length:',dat['M'])
		 
		dat.update(add_stan_data)
		self._stan_input = dat.copy()
		
		# Create tau grid for convenience, using min_tau and max_tau
		num_decades = int(np.ceil(np.log10(max_tau)) - np.floor(np.log10(min_tau)))
		for distribution in self.distributions.keys():
			self.distributions[distribution]['tau'] = np.logspace(np.log10(min_tau),np.log10(max_tau),num_decades*10 + 1)
		
		# Get initial parameter values
		iv = {}
		if np.sum([init_from_ridge,init_from_map,init_drt_fit is not None]) > 1:
			raise ValueError('Only one of init_from_ridge, init_from_map, and init_drt_fit may be used')
		if init_from_ridge:
			if len(self.distributions) > 1:
				raise ValueError('Ridge initialization can only be performed for single-distribution fits')
			else:
				init = self._get_init_from_ridge(frequencies,Z,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
				# Tile the initial parameter vector to make initial matrix
				for param in ['lntau_HN','R_HN','alpha_HN','beta_HN']:
					iv[param] = np.tile(init[param],(J,1))
				iv['Rinf_base'] = np.tile(init['Rinf'],J)
				iv['Rinf_base_raw'] = np.tile(init['Rinf_raw'],J)
				iv['induc'] = np.tile(init['induc'],J)
				iv['induc_raw'] = np.tile(init['induc_raw'],J)
		elif init_from_map:
			init = self._get_init_from_map(frequencies,Z,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
			# Tile the initial parameter vector to make initial matrix
			for param in ['lntau_HN','R_HN','alpha_HN','beta_HN']:
				iv[param] = np.tile(init[param],(J,1))
			iv['Rinf_base'] = np.tile(init['Rinf'],J)
			iv['Rinf_base_raw'] = np.tile(init['Rinf_raw'],J)
			iv['induc'] = np.tile(init['induc'],J)
			iv['induc_raw'] = np.tile(init['induc_raw'],J)
		elif init_drt_fit is not None:
			init = self._get_init_from_drt(init_drt_fit,num_HN,inductance_scale,min_tau,max_tau,peakfit_kw)
			# Tile the initial parameter vector to make initial matrix
			for param in ['lntau_HN','R_HN','alpha_HN','beta_HN']:
				iv[param] = np.tile(init[param],(J,1))
			iv['Rinf_base'] = np.tile(init['Rinf'],J)
			iv['Rinf_base_raw'] = np.tile(init['Rinf_raw'],J)
			iv['induc'] = np.tile(init['induc'],J)
			iv['induc_raw'] = np.tile(init['induc_raw'],J)
			iv['upsilon'] = np.ones(K)#*(dat['ups_alpha']/dat['ups_beta'])
		else:
			# distribute lntau values uniformly
			iv['lntau_HN'] = np.tile(np.linspace(np.log(min_tau*10),np.log(max_tau/10),K),(J,1))
			iv['R_HN'] = np.tile(np.ones(K)*1,(J,1))
			iv['alpha_HN'] = np.tile(np.ones(K)*0.95,(J,1))
			iv['beta_HN'] = np.tile(np.ones(K)*0.8,(J,1))
			iv['upsilon'] = np.ones(K)
		
		if outliers:
			raise ValueError('Outlier model not yet implemented')
			# initialize sigma_out near zero, everything else randomly
			iv['sigma_out_raw'] = np.zeros(2*len(Z)) + 0.1
			dat['so_invscale'] = outlier_lambda
			
		if init_values is not None:
			# if specific initial values provided,
			# update defaults with user-provided values
			iv.update(init_values)
			
		if centered:
			# initialize lntau center and dev
			if 'lntau_HN' in init_values.keys():
				init_values['lntau_HN_center'] = np.mean(init_values['lntau_HN'],axis=0)
				init_values['lntau_HN_dev'] = init_values['lntau_HN'] - np.tile(init_values['lntau_HN_center'],(J,1))
		
			
		def init():
			return iv
		self._init_params = iv
				
		# print(iv)
		
		# optimize posterior
		# use extra small initial step size (init_alpha, default 0.001) to avoid initialization/startup issues - Z_hat ends up inf
		self._opt_result = model.optimizing(dat,iter=max_iter,seed=random_seed,init=init,init_alpha=5e-4)
		
		# extract coefficients
		self.distribution_fits = {}
		self.error_fit = {}
		if model_type in ['Series','Parallel']:
			dist_name = [k for k,v in self.distributions.items() if v['dist_type']==model_type.lower()][0]
			dist_type = self.distributions[dist_name]['dist_type']
			self.distribution_fits[dist_name] = {'R_HN':self._rescale_coef(self._opt_result['R_HN'],dist_type)}
			self.distribution_fits[dist_name]['tau_HN'] = np.exp(self._opt_result['lntau_HN'])
			self.distribution_fits[dist_name]['alpha_HN'] = self._opt_result['alpha_HN']
			self.distribution_fits[dist_name]['beta_HN'] = self._opt_result['beta_HN']
			
			self.R_inf = self._rescale_coef(self._opt_result['Rinf'],'series')
			self.inductance = self._rescale_coef(self._opt_result['induc'],'series')
		
		# store error structure parameters
		# scaled parameters
		self.error_fit['sigma_min'] = self._rescale_coef(sigma_min,'series')
		for param in ['sigma_tot','sigma_res']:
			self.error_fit[param] = self._rescale_coef(self._opt_result[param],'series')
		# unscaled parameters
		for param in ['alpha_prop','alpha_re','alpha_im']:
			self.error_fit[param] = self._opt_result[param]
		# outlier contribution
		if outliers:
			self.error_fit['sigma_out'] = self._rescale_coef(self._opt_result['sigma_out'],'series')
		
		self.fit_type = 'map-combi'
		
	def map_combi_activation_fit(self,frequencies,Z,T,T_base,sample_ids,x,y,part='both',scale_Z=True,nonneg=True,outliers=False,
		min_tau=None,max_tau=None,max_comparison_distance=None,num_HN=5,ordered=True,repulse=False,
		# shapeshift=False,
		add_stan_data={},
		# drt_assist=False,drt_fit_data=None,drt_eval_tau=None,
		adjust_temp=False,temp_uncertainty=10,temp_offset_scale=2,
		sigma_min=0.002,inductance_scale=1,
		mode='optimize',
		# initialization parameters
		init_from_ridge=False,init_from_map=False,init_basis_freq=None,init_drt_fit=None,init_values=None,init_kw={},peakfit_kw={},
		# optimization control
		max_iter=50000,random_seed=1234,history_size=5,
		# sampling control
		warmup=200,sample=200,chains=2
		):
		"""
		Obtain the maximum a posteriori estimate of the defined distribution(s) (and all model parameters).
		
		Parameters:
		-----------
		frequencies: JxN array
			Measured frequencies
		Z: complex JxN array
			Measured (complex) impedance values. Must have same length as frequencies
		x: J-vector
			Vector of x-coordinates
		y: J-vector
			Vector of y-coordinates
		part: str, optional (default: 'both')
			Which part of the impedance data to fit. Options: 'both', 'real', 'imag'
		scale_Z: bool, optional (default: True)
			If True, scale impedance by the factor sqrt(N)/std(|Z|) to normalize for magnitude and sample size
		init_from_ridge: bool, optional (default: False)
			If True, use the hyperparametric ridge solution to initialize the Bayesian fit. 
			Only valid for single-distribution fits
		nonneg: bool, optional (default: False)
			If True, constrain the DRT to non-negative values
			
		adjust_temp: bool, optional (default: False)	
			If True, 
		outliers: bool, optional (default: False)
			If True, enable outlier identification via independent error contribution variable
		sigma_min: float, optional (default: 0.002)
			Impedance error floor. This is necessary to avoid sampling/optimization errors.
			Values smaller than the default (0.002) may enable slightly closer fits of very clean data,
			but may also result in sampling/optimization errors that yield unexpected results.
		max_iter: int, optional (default: 50000)
			Maximum number of iterations to allow the optimizer to perform
		random_seed: int, optional (default: 1234)
			Random seed for optimizer
		inductance_scale: float, optional (default: 1)
			Scale (std of normal prior) of the inductance. Lower values will constrain the inductance,
			which may be helpful if the estimated inductance is anomalously large (this may occur if your
			measured impedance data does not extend to high frequencies, i.e. 1e5-1e6 Hz)
		outlier_lambda: float, optional (default: 5)
			Lambda parameter (inverse scale) of the exponential prior on the outlier error contribution.
			Smaller values will make it easier for points to be flagged as outliers 
		ridge_kw: dict, optional (default: {})
			Keyword arguments to pass to ridge_fit if init_from_ridge==True.
		"""
		# load stan model
		model_str = 'Series_HN-combi-activation_pos_trunc'
		if ordered:
			model_str += '_ordered'
		if repulse:
			model_str += '_repulse'
		# if shapeshift:
			# model_str += '_shapeshift'
		if adjust_temp:
			model_str += '_T-adjust'
		# if drt_assist:
			# model_str += '_drt-assist'
		model_str += '_StanModel.pkl'
		model = load_pickle(os.path.join(script_dir,'stan_model_files',model_str))
		self.stan_model_name = model_str
		model_type = model_str.split('_')[0]
		
		# get dimensions
		J = frequencies.shape[0]
		N = frequencies.shape[1]
		K = num_HN
		
		# Check that temperatures are the same across samples
		T_dev = np.std(T,axis=0)
		if np.max(T_dev) > 1e-6:
			raise ValueError('Measurement temperatures must be the same for all samples')
			
		# store unique fit temperatures
		temp_start_indices = np.where(np.diff(T[0])!=0)[0] + 1
		temp_start_indices = np.concatenate(([0],temp_start_indices))
		self.fit_temperatures = T[0,temp_start_indices]
		self.T_offset = np.zeros((J,len(self.fit_temperatures)))
			
		# perform scaling. Scale based on T_base
		Z_base = Z[np.where(T==T_base)]
		self._scale_Z(Z_base.flatten(),'map')
		Z_scaled = Z/self._Z_scale
		Z_mat = np.concatenate((Z_scaled.real,Z_scaled.imag),axis=1)
		
		self.f_train = frequencies[0]
		freq_base = frequencies[np.where(T==T_base)]
		self.sample_ids = np.array(sample_ids)
		
		# Get relative Z_scale (across all samples) for each temperature
		# Assume same temps used for all samples, so rel_Z_scale is a N-vector, not a JxN matrix
		rel_Z_scale = np.ones(N)
		self._basis_Z_scale = {}
		for temp in np.unique(T):
			# Get all impedance values for temp
			tidx = np.where(T==temp)
			Z_T = Z[tidx]
			Zmod_T = (Z_T*Z_T.conjugate())**0.5
			# Get Z_scale across all samples for temp
			Z_scale_T = np.std(Zmod_T.flatten())/np.sqrt(len(Z_T.flatten())/81)
			# rel_Z_scale for temp is Z_scale_T/(Z_scale at T_base)
			rel_Z_scale[np.where(T[0]==temp)] = Z_scale_T/self._Z_scale
			# store rel_Z_scale for temp - need this for predict_sigma
			self._basis_Z_scale[temp] = Z_scale_T
		
		# prepare data for stan model
		dat = {
			# dimensions
			'N': N,
			'J': J,
			'K': K,
			# impedance data
			'freq':frequencies,
			'Z': Z_mat,
			'rel_Z_scale': rel_Z_scale,
			# temperature
			'temp': T,
			'T_base': T_base,
			# spatial coordinates
			'x_coord': x,
			'y_coord': y,
			# fixed hyperparameters
			'induc_scale': inductance_scale,
			'sigma_min':sigma_min,
			'max_delta_G': 2,
		}
		if mode=='optimize':
			dat.update(
				{
				'R_base_scale_alpha': 1,
				'R_base_scale_beta': 1,
				# scales of spatial standard devs
				'sigmaxy_lntau_scale': 0.2,
				'sigmaxy_lnR_scale': 1,
				'sigmaxy_alpha_scale': 0.1,
				'sigmaxy_beta_scale': 0.1,
				'sigmaxy_deltaG_scale': 0.2,
				'sigmaxy_lnphiT_scale': 0.5,
				
				'ln_phi_T_scale': 0.2,
				}
			)
		elif mode=='sample':
			dat.update(
				{
				'R_base_scale_alpha': 1,
				'R_base_scale_beta': 1,
				# scales of spatial standard devs
				'sigmaxy_lntau_scale': 0.1,
				'sigmaxy_lnR_scale': 0.5,
				'sigmaxy_alpha_scale': 0.05,
				'sigmaxy_beta_scale': 0.05,
				'sigmaxy_deltaG_scale': 0.1,
				'sigmaxy_lnphiT_scale': 0.25,
				
				'ln_phi_T_scale': 0.1,
				}
			)
		
		
		# calculate distance matrix
		dx = np.tile(x,(len(x),1)) - np.tile(x,(len(x),1)).T
		dy = np.tile(y,(len(y),1)) - np.tile(y,(len(y),1)).T
		D = np.sqrt(dx**2 + dy**2)
		# get upper triangle of distance matrix and flatten
		
		distance_vec = np.zeros(int(J*(J-1)/2))
		pos = 0
		for i in range(J):
			for j in range(i+1,J):
				distance_vec[pos] = D[i,j]
				pos += 1
		# get indices of distances that are below comparison threshold
		if max_comparison_distance is None:
			max_comparison_distance = np.max(distance_vec)
		compare_idx = np.where(distance_vec <= max_comparison_distance)
		# print(compare_idx[0])
		# print(distance_vec)
		# print(distance_vec[compare_idx])
		dat['compare_idx'] = compare_idx[0] + 1 # indexing starts at 0 in stan
		dat['M'] = len(compare_idx[0])
		print('Full distance vector length:',len(distance_vec))
		print('Truncated distance vector length:',dat['M'])
		
		# if shapeshift:
			# # update spatial sigma input names
			# for hp in ['lntau','lnR','alpha','beta','deltaG','lnphi']:
				# new_name = f'sigmaxy_{hp}_scale'
				# old_name = f'sigma_{hp}_scale'
				# dat[new_name] = dat[old_name]
				# del dat[old_name]
			# dat['sigmaT_alpha_scale'] = 0.1
			# dat['sigmaT_beta_scale'] = 0.1
			
			# # get & store temperature indices
			# temp_start_indices = np.where(np.diff(T[0])!=0)[0] + 1
			# # increment indices for stan (indexing starts at 1)
			# temp_start_indices += 1
			# dat['P'] = len(temp_start_indices) + 1
			# dat['temp_start_indices'] = temp_start_indices
		
		# # Generate DRT matrix if using DRT assistance
		# if drt_assist:
			# if drt_fit_data is None:
				# raise ValueError('drt_fit_data is required if drt_assist==True')
			# elif len(drt_fit_data)!=len(sample_ids):
				# raise ValueError('drt_fit_data must include one dataset for each sample')
			
			# if drt_eval_tau is None:
				# # If drt_eval_tau not specified, use 10 ppd over training frequencies
				# fmin = np.min(frequencies)
				# fmax = np.max(frequencies)
				# num_decades = int(np.log10(fmax) - np.log10(fmin))
				# drt_eval_tau = np.logspace(np.log10(1/(2*np.pi*fmax)),np.log10(1/(2*np.pi*fmin)),int(num_decades*10 + 1))
			
			# self.inv_init = Inverter()
			# gamma_base = np.zeros((J,len(drt_eval_tau)))
			
			# for j,fit_data in enumerate(drt_fit_data):
				# self.inv_init.load_fit_data(fit_data)
				# gamma_base[j] = self.inv_init.predict_distribution('DRT',drt_eval_tau)
			
			# # scale gamma_base to stan input scale
			# gamma_base /= self._Z_scale
				
			# gamma_weight = 1/(gamma_base + np.tile(np.percentile(gamma_base,80,axis=1),(len(drt_eval_tau),1)).T)
			# gamma_weight *= 3
			
			# dat['G'] = len(drt_eval_tau)
			# dat['tau_drt'] = drt_eval_tau
			# print('G:',len(drt_eval_tau))
			# dat['gamma_base'] = gamma_base
			# dat['gamma_weight'] = gamma_weight
			
		if adjust_temp:
			# increment indices for stan (indexing starts at 1)
			temp_start_indices += 1
			dat['P'] = len(temp_start_indices)
			dat['temp_start_indices'] = temp_start_indices
			dat['temp_uncertainty'] = temp_uncertainty
			dat['temp_offset_scale'] = temp_offset_scale
			dat['sigmaxy_tempoffset_scale'] = 10
			
		if repulse:
			dat['r_lntau_scale'] = 1
		 
		dat.update(add_stan_data)
		self._stan_input = dat.copy()
		
		# Create tau grid for convenience, using min_tau and max_tau
		# if tau boundaries not specified, allow HN peaks to extend 2 decades beyond measurement range
		if min_tau is None:
			min_tau = np.min(1/(2*np.pi*frequencies))/100
		if max_tau is None:
			max_tau = np.max(1/(2*np.pi*frequencies))*100
		num_decades = int(np.ceil(np.log10(max_tau)) - np.floor(np.log10(min_tau)))
		for distribution in self.distributions.keys():
			self.distributions[distribution]['tau'] = np.logspace(np.log10(min_tau),np.log10(max_tau),num_decades*10 + 1)
		
		# Get initial parameter values
		iv = {}
		if np.sum([init_from_ridge,init_from_map,init_drt_fit is not None]) > 1:
			raise ValueError('Only one of init_from_ridge, init_from_map, and init_drt_fit may be used')
		if init_from_ridge:
			if len(self.distributions) > 1:
				raise ValueError('Ridge initialization can only be performed for single-distribution fits')
			else:
				init = self._get_init_from_ridge(frequencies,Z,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
				for param in ['lntau_HN','R_HN','alpha_HN','beta_HN']:
					iv[param] = np.tile(init[param],(J,1))
		elif init_from_map:
			init = self._get_init_from_map(frequencies,Z,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
			for param in ['lntau_HN','R_HN','alpha_HN','beta_HN']:
				iv[param] = np.tile(init[param],(J,1))
		elif init_drt_fit is not None:
			init = self._get_init_from_drt(init_drt_fit,num_HN,inductance_scale,min_tau,max_tau,peakfit_kw)
			# Tile the initial parameter vector to make initial matrix
			for param in ['lntau_HN','R_HN','alpha_HN','beta_HN']:
				iv[param] = np.tile(init[param],(J,1))
			iv['Rinf_base'] = np.tile(init['Rinf'],J)
			iv['Rinf_base_raw'] = np.tile(init['Rinf_raw'],J)
			iv['induc'] = np.tile(init['induc'],J)
			iv['induc_raw'] = np.tile(init['induc_raw'],J)
			# Update parameter names for activation model
			iv['lntau_HN_base'] = iv['lntau_HN']
			iv['R_HN_base'] = iv['R_HN']
		else:
			# distribute lntau values uniformly
			iv['lntau_HN_base'] = np.tile(np.linspace(np.log(min_tau*10),np.log(max_tau/10),K),(J,1))
			iv['R_HN_base'] = np.tile(np.ones(K),(J,1))
			iv['alpha_HN'] = np.tile(np.ones(K)*0.95,(J,1))
			iv['beta_HN'] = np.tile(np.ones(K)*0.8,(J,1))
			iv['delta_G'] = np.tile(np.ones(K)*0.5,(J,1))
			iv['ln_phi_T_raw'] = np.tile(np.zeros(K),(J,1))
			# iv['upsilon'] = np.tile(np.ones(K)*10,(J,1))
		if outliers:
			raise ValueError('Outlier model not yet implemented')
			# initialize sigma_out near zero, everything else randomly
			iv['sigma_out_raw'] = np.zeros(2*len(Z)) + 0.1
			dat['so_invscale'] = outlier_lambda
			
		# if shapeshift:
			# iv['alpha_HN'] = np.tile(iv['alpha_HN'],(dat['P'],1,1))
			# iv['beta_HN'] = np.tile(iv['beta_HN'],(dat['P'],1,1))
			
		if init_values is not None:
			# if specific initial values provided,
			# update defaults with user-provided values
			# iv.update(init_values)
			iv = init_values
		
			
		def init():
			return iv
		self._init_params = iv
				
		# print(iv)
		
		if mode=='optimize':
			# optimize posterior
			# use extra small initial step size (init_alpha, default 0.001) to avoid initialization/startup issues - Z_hat ends up inf
			self._opt_result = model.optimizing(dat,iter=max_iter,seed=random_seed,init=init,history_size=history_size,init_alpha=5e-4)
		
			# extract coefficients
			self.distribution_fits = {}
			self.error_fit = {}
			if model_type in ['Series','Parallel']:
				dist_name = [k for k,v in self.distributions.items() if v['dist_type']==model_type.lower()][0]
				dist_type = self.distributions[dist_name]['dist_type']
				self.distribution_fits[dist_name] = {'R_HN_base':self._rescale_coef(self._opt_result['R_HN_base'],dist_type)}
				self.distribution_fits[dist_name]['tau_HN_base'] = np.exp(self._opt_result['lntau_HN_base'])
				self.distribution_fits[dist_name]['alpha_HN'] = self._opt_result['alpha_HN']
				self.distribution_fits[dist_name]['beta_HN'] = self._opt_result['beta_HN']
				self.distribution_fits[dist_name]['delta_G'] = self._opt_result['delta_G']
				self.distribution_fits[dist_name]['phi_T'] = np.exp(self._opt_result['ln_phi_T'])
				self.distribution_fits[dist_name]['T_base'] = T_base
				
				self.R_inf_base = self._rescale_coef(self._opt_result['Rinf_base'],'series')
				self.R_inf_dG = self._opt_result['delta_G_Rinf']
				self.inductance = self._rescale_coef(self._opt_result['induc'],'series')
				
				if adjust_temp:
					self.T_offset = self._opt_result['temp_offset']
			
			# store error structure parameters
			# scaled parameters
			self.error_fit['sigma_min'] = self._rescale_coef(sigma_min,'series')
			for param in ['sigma_tot','sigma_res']:
				self.error_fit[param] = self._rescale_coef(self._opt_result[param],'series')
			# unscaled parameters
			for param in ['alpha_prop','alpha_re','alpha_im']:
				self.error_fit[param] = self._opt_result[param]
			# outlier contribution
			if outliers:
				self.error_fit['sigma_out'] = self._rescale_coef(self._opt_result['sigma_out'],'series')
		elif mode=='sample':
			# sample posterior
			self._sample_result = model.sampling(dat,warmup=warmup,iter=warmup+sample,chains=chains,seed=random_seed,init=init,
								  control={'adapt_delta':0.9,'adapt_t0':10})
		
			# extract coefficients
			# ****NEED TO UPDATE FOR HMC SAMPLING
			self.distribution_fits = {}
			self.error_fit = {}
			if model_type in ['Series','Parallel']:
				dist_name = [k for k,v in self.distributions.items() if v['dist_type']==model_type.lower()][0]
				dist_type = self.distributions[dist_name]['dist_type']
				self.distribution_fits[dist_name] = {'R_HN_base':self._rescale_coef(self._opt_result['R_HN_base'],dist_type)}
				self.distribution_fits[dist_name]['tau_HN_base'] = np.exp(self._opt_result['lntau_HN_base'])
				self.distribution_fits[dist_name]['alpha_HN'] = self._opt_result['alpha_HN']
				self.distribution_fits[dist_name]['beta_HN'] = self._opt_result['beta_HN']
				self.distribution_fits[dist_name]['delta_G'] = self._opt_result['delta_G']
				self.distribution_fits[dist_name]['phi_T'] = np.exp(self._opt_result['ln_phi_T'])
				self.distribution_fits[dist_name]['T_base'] = T_base
				
				self.R_inf_base = self._rescale_coef(self._opt_result['Rinf_base'],'series')
				self.R_inf_dG = self._opt_result['delta_G_Rinf']
				self.inductance = self._rescale_coef(self._opt_result['induc'],'series')
				
				if adjust_temp:
					self.T_offset = self._opt_result['temp_offset']
			
			# store error structure parameters
			# scaled parameters
			self.error_fit['sigma_min'] = self._rescale_coef(sigma_min,'series')
			for param in ['sigma_tot','sigma_res']:
				self.error_fit[param] = self._rescale_coef(self._opt_result[param],'series')
			# unscaled parameters
			for param in ['alpha_prop','alpha_re','alpha_im']:
				self.error_fit[param] = self._opt_result[param]
			# outlier contribution
			if outliers:
				self.error_fit['sigma_out'] = self._rescale_coef(self._opt_result['sigma_out'],'series')
			
		if mode=='optimize':
			self.fit_type = 'map-combi-activation'
		elif mode=='sample':
			self.fit_type = 'hmc-combi-activation'
		
		
	def map_combi_activation_drift_fit(self,frequencies,Z,T,T_base,times,sample_ids,x,y,part='both',scale_Z=True,nonneg=True,outliers=False,
		min_tau=None,max_tau=None,max_comparison_distance=None,num_HN=5,#repulse=False,
		adjust_temp=False,temp_uncertainty=5,temp_offset_scale=1,
		model_str=None,add_stan_data={},
		# drt_assist=False,drt_fit_data=None,drt_eval_tau=None,
		# initialization parameters
		init_from_ridge=False,init_from_map=False,init_basis_freq=None,init_drt_fit=None,init_values=None,init_kw={},peakfit_kw={},
		# optimization control
		sigma_min=0.002,max_iter=50000,history_size=5,init_alpha=5e-4,random_seed=1234,inductance_scale=1,outlier_lambda=5):
		"""
		Obtain the maximum a posteriori estimate of the defined distribution(s) (and all model parameters).
		
		Parameters:
		-----------
		frequencies: JxN array
			Measured frequencies
		Z: complex JxN array
			Measured (complex) impedance values. Must have same length as frequencies
		x: J-vector
			Vector of x-coordinates
		y: J-vector
			Vector of y-coordinates
		part: str, optional (default: 'both')
			Which part of the impedance data to fit. Options: 'both', 'real', 'imag'
		scale_Z: bool, optional (default: True)
			If True, scale impedance by the factor sqrt(N)/std(|Z|) to normalize for magnitude and sample size
		init_from_ridge: bool, optional (default: False)
			If True, use the hyperparametric ridge solution to initialize the Bayesian fit. 
			Only valid for single-distribution fits
		nonneg: bool, optional (default: False)
			If True, constrain the DRT to non-negative values
			
		adjust_temp: bool, optional (default: False)	
			If True, 
		outliers: bool, optional (default: False)
			If True, enable outlier identification via independent error contribution variable
		sigma_min: float, optional (default: 0.002)
			Impedance error floor. This is necessary to avoid sampling/optimization errors.
			Values smaller than the default (0.002) may enable slightly closer fits of very clean data,
			but may also result in sampling/optimization errors that yield unexpected results.
		max_iter: int, optional (default: 50000)
			Maximum number of iterations to allow the optimizer to perform
		random_seed: int, optional (default: 1234)
			Random seed for optimizer
		inductance_scale: float, optional (default: 1)
			Scale (std of normal prior) of the inductance. Lower values will constrain the inductance,
			which may be helpful if the estimated inductance is anomalously large (this may occur if your
			measured impedance data does not extend to high frequencies, i.e. 1e5-1e6 Hz)
		outlier_lambda: float, optional (default: 5)
			Lambda parameter (inverse scale) of the exponential prior on the outlier error contribution.
			Smaller values will make it easier for points to be flagged as outliers 
		ridge_kw: dict, optional (default: {})
			Keyword arguments to pass to ridge_fit if init_from_ridge==True.
		"""
		# load stan model
		if model_str is None:
			model_str = 'Series_HN-combi-activation-drift_pos_trunc_ordered'
			# if repulse:
				# model_str += '_repulse'
			# if shapeshift:
				# model_str += '_shapeshift'
			if adjust_temp:
				model_str += '_T-adjust'
			# if drt_assist:
				# model_str += '_drt-assist'
			model_str += '_StanModel.pkl'
		model = load_pickle(os.path.join(script_dir,'stan_model_files',model_str))
		self.stan_model_name = model_str
		model_type = model_str.split('_')[0]
		 
		# get dimensions
		J = frequencies.shape[0]
		N = frequencies.shape[1]
		K = num_HN
		
		# Check that temperatures are the same across samples
		T_dev = np.std(T,axis=0)
		if np.max(T_dev) > 1e-6:
			raise ValueError('Measurement temperatures must be the same for all samples')
			
		# store unique fit temperatures
		temp_start_indices = np.where(np.diff(T[0])!=0)[0] + 1
		temp_start_indices = np.concatenate(([0],temp_start_indices))
		self.fit_temperatures = T[0,temp_start_indices]
		self.T_offset = np.zeros((J,len(self.fit_temperatures)))
			
		# perform scaling. Scale based on T_base
		Z_base = Z[np.where(T==T_base)]
		self._scale_Z(Z_base.flatten(),'map')
		Z_scaled = Z/self._Z_scale
		Z_mat = np.concatenate((Z_scaled.real,Z_scaled.imag),axis=1)
		
		self.f_train = frequencies[0]
		freq_base = frequencies[np.where(T==T_base)]
		self.sample_ids = np.array(sample_ids)
		
		# Get relative Z_scale (across all samples) for each temperature
		# Assume same temps used for all samples, so rel_Z_scale is a N-vector, not a JxN matrix
		rel_Z_scale = np.ones(N)
		self._basis_Z_scale = {}
		for temp in np.unique(T):
			# Get all impedance values for temp
			tidx = np.where(T==temp)
			Z_T = Z[tidx]
			Zmod_T = (Z_T*Z_T.conjugate())**0.5
			# Get Z_scale across all samples for temp
			Z_scale_T = np.std(Zmod_T.flatten())/np.sqrt(len(Z_T.flatten())/81)
			# rel_Z_scale for temp is Z_scale_T/(Z_scale at T_base)
			rel_Z_scale[np.where(T[0]==temp)] = Z_scale_T/self._Z_scale
			# store rel_Z_scale for temp - need this for predict_sigma
			self._basis_Z_scale[temp] = Z_scale_T
		
		# prepare data for stan model
		# ---------------------------
		dat = {
			# dimensions
			'N': N,
			'J': J,
			'K': K,
			# spatial coords
			'x_coord': x,
			'y_coord': y,
			# impedance data
			'freq': frequencies,
			'times': times,
			'Z': Z_mat,
			'rel_Z_scale': rel_Z_scale,
			# temperature
			'temp': T,
			'T_base': T_base,
			
			# fixed hyperparameters
			'induc_scale': inductance_scale,
			'sigma_min':sigma_min,
			'R_i_base_scale_alpha': 1,
			'R_i_base_scale_beta': 1,
			# activation
			'dG_alpha': 10,
			'dG_beta': 6,
			'max_delta_G': 2,
			'ln_phi_T_scale': 0.2,
			'max_phi_T': 3,
			# drift
			'sigma_delta_R_scale': 0.01,
			'ln_k_alpha': 5,
			'ln_k_beta': 5,
			'max_delta_G_k': 3,
			'max_phi_d': 2,
			'ln_phi_d_scale': 0.2,
			'min_k': 1e-4,
			'max_k': 1,
			# spatial stds
			'sigmaxy_lntau_scale': 0.1,
			'sigmaxy_lnRi_scale': 1,
			'sigmaxy_alpha_scale': 0.1,
			'sigmaxy_beta_scale': 0.1,
			'sigmaxy_deltaG_scale': 0.2,
			'sigmaxy_lnphiT_scale': 0.5,
			'sigmaxy_lnRf_scale': 1,
			'sigmaxy_lnk_scale': 0.5,
			'sigmaxy_deltaGk_scale': 0.2,
			'sigmaxy_lnphid_scale': 0.5,
			}
		
		
		# calculate distance matrix
		dx = np.tile(x,(len(x),1)) - np.tile(x,(len(x),1)).T
		dy = np.tile(y,(len(y),1)) - np.tile(y,(len(y),1)).T
		D = np.sqrt(dx**2 + dy**2)
		# get upper triangle of distance matrix and flatten
		
		distance_vec = np.zeros(int(J*(J-1)/2))
		pos = 0
		for i in range(J):
			for j in range(i+1,J):
				distance_vec[pos] = D[i,j]
				pos += 1
		# get indices of distances that are below comparison threshold
		if max_comparison_distance is None:
			max_comparison_distance = np.max(distance_vec)
		compare_idx = np.where(distance_vec <= max_comparison_distance)
		# print(compare_idx[0])
		# print(distance_vec)
		# print(distance_vec[compare_idx])
		dat['compare_idx'] = compare_idx[0] + 1 # indexing starts at 0 in stan
		dat['M'] = len(compare_idx[0])
		print('Full distance vector length:',len(distance_vec))
		print('Truncated distance vector length:',dat['M'])
		
		if adjust_temp:
			dat['temp_uncertainty'] = temp_uncertainty
			dat['temp_offset_scale'] = temp_offset_scale
			dat['temp_start_indices'] = temp_start_indices + 1 # adjust for stan indexing (starts at 1)
			dat['sigmaxy_tempoffset_scale'] = 1
			dat['P'] = len(temp_start_indices)
		 
		dat.update(add_stan_data)
		self._stan_input = dat.copy()
		
		# Get initial parameter values
		# ----------------------------
		# Create tau grid for convenience, using min_tau and max_tau
		# if tau boundaries not specified, allow HN peaks to extend 2 decades beyond measurement range
		if min_tau is None:
			min_tau = np.min(1/(2*np.pi*frequencies))/100
		if max_tau is None:
			max_tau = np.max(1/(2*np.pi*frequencies))*100
		num_decades = int(np.ceil(np.log10(max_tau)) - np.floor(np.log10(min_tau)))
		for distribution in self.distributions.keys():
			self.distributions[distribution]['tau'] = np.logspace(np.log10(min_tau),np.log10(max_tau),num_decades*10 + 1)
			
		iv = {}
		if np.sum([init_from_ridge,init_from_map,init_drt_fit is not None]) > 1:
			raise ValueError('Only one of init_from_ridge, init_from_map, and init_drt_fit may be used')
		if init_from_ridge:
			if len(self.distributions) > 1:
				raise ValueError('Ridge initialization can only be performed for single-distribution fits')
			else:
				init = self._get_init_from_ridge(frequencies,Z,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
				for param in ['lntau_HN','R_HN','alpha_HN','beta_HN']:
					iv[param] = np.tile(init[param],(J,1))
		elif init_from_map:
			init = self._get_init_from_map(frequencies,Z,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
			for param in ['lntau_HN','R_HN','alpha_HN','beta_HN']:
				iv[param] = np.tile(init[param],(J,1))
		elif init_drt_fit is not None:
			init = self._get_init_from_drt(init_drt_fit,num_HN,inductance_scale,min_tau,max_tau,peakfit_kw)
			# Tile the initial parameter vector to make initial matrix
			for param in ['lntau_HN','R_HN','alpha_HN','beta_HN']:
				iv[param] = np.tile(init[param],(J,1))
			iv['Rinf_base'] = np.tile(init['Rinf'],J)
			iv['Rinf_base_raw'] = np.tile(init['Rinf_raw'],J)
			iv['induc'] = np.tile(init['induc'],J)
			iv['induc_raw'] = np.tile(init['induc_raw'],J)
			# Update parameter names for activation model
			iv['lntau_HN_base'] = iv['lntau_HN']
			iv['R_HN_base'] = iv['R_HN']
		else:
			# distribute lntau values uniformly
			iv['lntau_i_base'] = np.tile(np.linspace(np.log(min_tau*10),np.log(max_tau/10),K),(J,1))
			iv['R_i_base'] = np.ones((J,K))
			iv['R_f_base'] = iv['R_i_base']
			iv['deltaR_base_raw'] = np.zeros((J,K))
			iv['alpha_HN'] = np.ones((J,K))*0.95
			iv['beta_HN'] = np.ones((J,K))*0.8
			iv['delta_G'] = np.ones((J,K))*0.5
			iv['delta_G_k'] = np.ones((J,K))*0.5
			iv['ln_phi_T_raw'] = np.zeros((J,K))
			iv['ln_phi_d_raw'] = np.zeros((J,K))
			
			# iv['upsilon'] = np.tile(np.ones(K)*10,(J,1))
			
		if adjust_temp:
			iv['temp_offset_raw'] = np.zeros((J,dat['P']))
			
		if init_values is not None:
			# if specific initial values provided,
			# update defaults with user-provided values
			iv = init_values
		
			
		def init():
			return iv
		self._init_params = iv
				
		# print(iv)
		
		# optimize posterior
		# use extra small initial step size (init_alpha, default 0.001) to avoid initialization/startup issues - Z_hat ends up inf
		self._opt_result = model.optimizing(dat,iter=max_iter,seed=random_seed,init=init,init_alpha=init_alpha,history_size=history_size)
		
		# extract coefficients
		self.distribution_fits = {}
		self.error_fit = {}
		if model_type in ['Series','Parallel']:
			dist_name = [k for k,v in self.distributions.items() if v['dist_type']==model_type.lower()][0]
			dist_type = self.distributions[dist_name]['dist_type']
			self.distribution_fits[dist_name] = {'R_i_base':self._rescale_coef(self._opt_result['R_i_base'],dist_type)}
			self.distribution_fits[dist_name]['tau_i_base'] = np.exp(self._opt_result['lntau_i_base'])
			self.distribution_fits[dist_name]['alpha_HN'] = self._opt_result['alpha_HN']
			self.distribution_fits[dist_name]['beta_HN'] = self._opt_result['beta_HN']
			self.distribution_fits[dist_name]['delta_G'] = self._opt_result['delta_G']
			self.distribution_fits[dist_name]['phi_T'] = np.exp(self._opt_result['ln_phi_T'])
			
			# self.distribution_fits[dist_name]['delta_R_base'] = self._rescale_coef(self._opt_result['deltaR_base'],dist_type)
			self.distribution_fits[dist_name]['R_f_base'] = self._rescale_coef(self._opt_result['R_f_base'],dist_type)
			self.distribution_fits[dist_name]['k_base'] = np.exp(self._opt_result['ln_k_base'])
			self.distribution_fits[dist_name]['delta_G_k'] = self._opt_result['delta_G_k']
			self.distribution_fits[dist_name]['phi_d'] = np.exp(self._opt_result['ln_phi_d'])
			
			self.distribution_fits[dist_name]['T_base'] = T_base
			
			self.R_inf_base = self._rescale_coef(self._opt_result['Rinf_base'],'series')
			self.R_inf_dG = self._opt_result['delta_G_Rinf']
			self.inductance = self._rescale_coef(self._opt_result['induc'],'series')
			
			if adjust_temp:
				self.T_offset = self._opt_result['temp_offset']
		
		# store error structure parameters
		# scaled parameters
		self.error_fit['sigma_min'] = self._rescale_coef(sigma_min,'series')
		for param in ['sigma_tot','sigma_res']:
			self.error_fit[param] = self._rescale_coef(self._opt_result[param],'series')
		# unscaled parameters
		for param in ['alpha_prop','alpha_re','alpha_im']:
			self.error_fit[param] = self._opt_result[param]
		# outlier contribution
		if outliers:
			self.error_fit['sigma_out'] = self._rescale_coef(self._opt_result['sigma_out'],'series')
			
		self.fit_type = 'map-combi-activation-drift'
		
	def map_activation_drift_fit(self,frequencies,Z,T,T_base,times,drift_model='',part='both',scale_Z=True,nonneg=True,outliers=False,
		min_tau=None,max_tau=None,num_HN=10,add_stan_data={},ordered=True,#shapeshift=False,
		adjust_temp=False,temp_uncertainty=10,temp_offset_scale=5,
		student_t=False,
		# R_err=False,
		sigma_min=0.002,inductance_scale=1,outlier_lambda=5,
		model_str=None,
		mode='optimize',
		# initialization parameters
		init_from_ridge=False,init_from_map=False,init_basis_freq=None,init_drt_fit=None,init_values=None,init_kw={},peakfit_kw={},
		# optimization control
		max_iter=50000,random_seed=1234,init_alpha=1e-3,
		# sampling control
		warmup=200,sample=200,chains=2
		):
		"""
		Obtain the maximum a posteriori estimate of the defined distribution(s) (and all model parameters).
		
		Parameters:
		-----------
		frequencies: array
			Measured frequencies
		Z: complex array
			Measured (complex) impedance values. Must have same length as frequencies
		part: str, optional (default: 'both')
			Which part of the impedance data to fit. Options: 'both', 'real', 'imag', 'polar'
		scale_Z: bool, optional (default: True)
			If True, scale impedance by the factor sqrt(N)/std(|Z|) to normalize for magnitude and sample size
		init_from_ridge: bool, optional (default: False)
			If True, use the hyperparametric ridge solution to initialize the Bayesian fit. 
			Only valid for single-distribution fits
		nonneg: bool, optional (default: False)
			If True, constrain the DRT to non-negative values
		outliers: bool, optional (default: False)
			If True, enable outlier identification via independent error contribution variable
		sigma_min: float, optional (default: 0.002)
			Impedance error floor. This is necessary to avoid sampling/optimization errors.
			Values smaller than the default (0.002) may enable slightly closer fits of very clean data,
			but may also result in sampling/optimization errors that yield unexpected results.
		max_iter: int, optional (default: 50000)
			Maximum number of iterations to allow the optimizer to perform
		random_seed: int, optional (default: 1234)
			Random seed for optimizer
		inductance_scale: float, optional (default: 1)
			Scale (std of normal prior) of the inductance. Lower values will constrain the inductance,
			which may be helpful if the estimated inductance is anomalously large (this may occur if your
			measured impedance data does not extend to high frequencies, i.e. 1e5-1e6 Hz)
		outlier_lambda: float, optional (default: 5)
			Lambda parameter (inverse scale) of the exponential prior on the outlier error contribution.
			Smaller values will make it easier for points to be flagged as outliers 
		ridge_kw: dict, optional (default: {})
			Keyword arguments to pass to ridge_fit if init_from_ridge==True.
		"""
		# load stan model	
		if model_str is None:
			model_str = 'Series_HN-activation-drift'
			if drift_model!='':
				model_str += f'-{drift_model}'
			if nonneg:
				model_str += '_pos'
			if ordered:
				model_str += '_ordered'
			if student_t:
				model_str += '_student-t'
			# if R_err:
				# model_str += '_R-err'
			if adjust_temp:
				model_str += '_T-adjust'
			model_str += '_StanModel.pkl'
		
		model = load_pickle(os.path.join(script_dir,'stan_model_files',model_str))
		self.stan_model_name = model_str
		model_type = model_str.split('_')[0]
		
		# store unique fit temperatures
		temp_start_indices = np.concatenate(([0],np.where(np.diff(T)!=0)[0] + 1))
		self.fit_temperatures = T[temp_start_indices]
		self.T_offset = np.zeros(len(self.fit_temperatures))
		# if T[0]!=T_base:
			# raise ValueError('T_base must be first temperature')

		# perform scaling. Scale based on T_base
		Z_base = Z[np.where(T==T_base)]
		self._scale_Z(Z_base,'map')
		Z_scaled = Z/self._Z_scale
		self.f_train = frequencies
		freq_base = frequencies[np.where(T==T_base)]
		
		rel_Z_scale = np.ones(len(frequencies))
		for temp in np.unique(T):
			tidx = np.where(T==temp)
			Z_T = Z[tidx]
			Zmod_T = (Z_T*Z_T.conjugate())**0.5
			Z_scale_T = np.std(Zmod_T)/np.sqrt(len(Z_T)/81)
			rel_Z_scale[tidx] = Z_scale_T/self._Z_scale
			
		# prepare data for stan model
		dat = self._prep_stan_data(frequencies,Z_scaled,part,model_type,sigma_min,mode='optimize',inductance_scale=inductance_scale)
		# if tau boundaries not specified, allow HN peaks to extend 2 decades beyond measurement range
		if min_tau is None:
			min_tau = np.min(1/(2*np.pi*freq_base))/10
		if max_tau is None:
			max_tau = np.max(1/(2*np.pi*freq_base))*10
		dat['N'] = len(frequencies)
		dat['temp'] = T
		dat['T_base'] = T_base
		dat['times'] = times
		# dat['min_tau_HN'] = min_tau
		# dat['max_tau_HN'] = max_tau
		dat['max_delta_G'] = 2
		dat['max_phi_T'] = 2
		dat['ln_phi_T_scale'] = 0.2
		dat['K'] = num_HN
		# dat['ups_alpha'] = 1
		# dat['ups_beta'] = 0.5
		dat['rel_Z_scale'] = rel_Z_scale
		
		if drift_model=='':
			dat['R_i_base_scale_alpha'] = 1
			dat['R_i_base_scale_beta'] = 2
			dat['R_f_base_scale_alpha'] = 1
			dat['R_f_base_scale_beta'] = 2
			dat['ln_k_alpha'] = 1
			dat['ln_k_beta'] = 2
			dat['sigma_delta_lnR_scale'] = np.ones(num_HN)*0.15
			dat['min_delta_lnR'] = -5
			dat['max_delta_lnR'] = 5
			
			dat['max_delta_G_k'] = 3
			dat['max_phi_d'] = 2
			dat['ln_phi_d_scale'] = 0.2
			dat['min_k'] = 1e-4
			dat['max_k'] = 1
		elif drift_model.find('RQ') > -1:
			dat['P'] = len(temp_start_indices)
			opt_point = drift_model[drift_model.find('x')+1]
			dat[f'R_{opt_point}_base_scale_alpha'] = 1
			dat[f'R_{opt_point}_base_scale_beta'] = 2
			dat['min_tau_rq'] = min_tau
			dat['max_tau_rq'] = max_tau
			dat['sigma_R_rq_scale'] = np.ones(dat['P'])*0.05
			if drift_model.find('compressed') > -1:
				dat['min_k'] = 1e-4
				dat['max_k'] = 1
			
		if adjust_temp:
			dat['temp_uncertainty'] = temp_uncertainty
			dat['temp_offset_scale'] = temp_offset_scale
			dat['temp_start_indices'] = temp_start_indices + 1 # adjust for stan indexing (starts at 1)
			dat['P'] = len(temp_start_indices)
			
		# if R_err:
			# dat['delta_lnR_scale'] = 0.05
		
		dat.update(add_stan_data)
		self._stan_input = dat.copy()
		
		# Create tau grid for convenience, using min_tau and max_tau
		num_decades = int(np.ceil(np.log10(max_tau)) - np.floor(np.log10(min_tau)))
		for distribution in self.distributions.keys():
			self.distributions[distribution]['tau'] = np.logspace(np.log10(min_tau),np.log10(max_tau),num_decades*10 + 1)
		
		# Get initial parameter values
		iv = {}
		if np.sum([init_from_ridge,init_from_map,init_drt_fit is not None]) > 1:
			raise ValueError('Only one of init_from_ridge, init_from_map, and init_drt_fit may be used')
		if init_from_ridge:
			if len(self.distributions) > 1:
				raise ValueError('Ridge initialization can only be performed for single-distribution fits')
			else:
				init = self._get_init_from_ridge(freq_base,Z_base,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
				iv.update(init)
				# Update parameter names for activation model
				iv['Rinf_base'] = iv['Rinf']
				iv['Rinf_base_raw'] = init['Rinf_raw']
				iv['lntau_HN_base'] = iv['lntau_HN']
				iv['R_HN_base'] = iv['R_HN']
		elif init_from_map:
			init = self._get_init_from_map(freq_base,Z_base,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
			iv.update(init)
			# Update parameter names for activation model
			iv['Rinf_base'] = iv['Rinf']
			iv['Rinf_base_raw'] = init['Rinf_raw']
			iv['lntau_HN_base'] = iv['lntau_HN']
			iv['R_HN_base'] = iv['R_HN']
		elif init_drt_fit is not None:
			init = self._get_init_from_drt(init_drt_fit,num_HN,inductance_scale,min_tau,max_tau,peakfit_kw)
			iv.update(init)
			# Update parameter names for activation model
			iv['Rinf_base'] = iv['Rinf']
			iv['Rinf_base_raw'] = init['Rinf_raw']
			iv['lntau_HN_base'] = iv['lntau_HN']
			iv['R_HN_base'] = iv['R_HN']
		else:
			# distribute lntau values uniformly
			iv['lntau_HN_base'] = np.linspace(np.log(min_tau*10),np.log(max_tau/10),num_HN)
			iv['R_HN_base'] = np.ones(num_HN)
			iv['alpha_HN'] = np.ones(num_HN)*0.95
			iv['beta_HN'] = np.ones(num_HN)*0.8
			iv['upsilon'] = np.ones(num_HN)*1
			iv['delta_G'] = np.ones(num_HN)*0.5
			iv['delta_G_Rinf'] = 0.5
			iv['ln_phi_T'] = np.zeros(num_HN)
			iv['induc'] = 1e-8
			iv['induc_raw'] = iv['induc']/inductance_scale
			iv['deltaR_base_raw'] = np.zeros(num_HN)
			iv['deltaR_base'] = np.zeros(num_HN)
		
		if outliers:
			raise ValueError('Outlier model not yet implemented')
			# initialize sigma_out near zero, everything else randomly
			iv['sigma_out_raw'] = np.zeros(2*len(Z)) + 0.1
			dat['so_invscale'] = outlier_lambda
			
		# if shapeshift:
			# for param in ['alpha_HN','beta_HN']:
				# iv[param] = np.tile(iv[param],(dat['P'],1))
				
		if adjust_temp:
			iv['temp_offset_raw'] = np.zeros(dat['P'])
				
		if init_values is not None:
			# if type(init_values)==dict:
				# iv.update(init_values)
			# elif init_values=='random':
				# iv = 'random'
			iv = init_values.copy()
			
		def init():
			return iv
		self._init_params = iv
				
		# print(iv)
		
		if mode=='optimize':
			# optimize posterior
			self._opt_result = model.optimizing(dat,iter=max_iter,seed=random_seed,init=init,init_alpha=init_alpha)
		
			# extract coefficients
			self.distribution_fits = {}
			self.error_fit = {}
			if model_type in ['Series','Parallel']:
				dist_name = [k for k,v in self.distributions.items() if v['dist_type']==model_type.lower()][0]
				dist_type = self.distributions[dist_name]['dist_type']
				self.distribution_fits[dist_name] = {}
				self.distribution_fits[dist_name]['alpha_HN'] = self._opt_result['alpha_HN']
				self.distribution_fits[dist_name]['beta_HN'] = self._opt_result['beta_HN']
				# self.distribution_fits[dist_name]['delta_G'] = self._opt_result['delta_G']
				try:
					self.distribution_fits[dist_name]['delta_G_i'] = self._opt_result['delta_G_i']
					self.distribution_fits[dist_name]['delta_G_f'] = self._opt_result['delta_G_f']
				except Exception:
					self.distribution_fits[dist_name]['delta_G_i'] = self._opt_result['delta_G']
					self.distribution_fits[dist_name]['delta_G_f'] = self._opt_result['delta_G']
				self.distribution_fits[dist_name]['phi_T'] = np.exp(self._opt_result['ln_phi_T'])
				self.distribution_fits[dist_name]['T_base'] = T_base
				
				# if R_err:
					# self.distribution_fits[dist_name]['delta_lnR'] = self._opt_result['delta_lnR']
				
				if drift_model=='':
					try:
						self.distribution_fits[dist_name]['tau_i_base'] = np.exp(self._opt_result['lntau_i_base'])
					except Exception:
						self.distribution_fits[dist_name]['tau_i_base'] = np.exp(self._opt_result['lntau_f_base'] + self._opt_result['delta_lnR_base']*np.exp(self._opt_result['ln_phi_d']))
					self.distribution_fits[dist_name]['R_i_base'] = self._rescale_coef(self._opt_result['R_i_base'],dist_type)
					self.distribution_fits[dist_name]['R_f_base'] = self._rescale_coef(self._opt_result['R_f_base'],dist_type)
					self.distribution_fits[dist_name]['k_base'] = np.exp(self._opt_result['ln_k_base'])
					self.distribution_fits[dist_name]['delta_G_k'] = self._opt_result['delta_G_k']
					self.distribution_fits[dist_name]['phi_d'] = np.exp(self._opt_result['ln_phi_d'])
					try:
						self.distribution_fits[dist_name]['beta_t'] = self._opt_result['beta_t']
					except KeyError:
						self.distribution_fits[dist_name]['beta_t'] = np.ones(num_HN)
				elif drift_model.find('RQ') > -1:
					self.distribution_fits[dist_name]['R_rq'] = self._rescale_coef(self._opt_result['R_rq'],dist_type)
					self.distribution_fits[dist_name]['tau_rq'] = np.exp(self._opt_result['log_tau_rq'])
					self.distribution_fits[dist_name]['phi_rq'] = self._opt_result['phi_rq']
					self.distribution_fits[dist_name]['t_i'] = np.min(times)
					self.distribution_fits[dist_name]['t_f'] = np.max(times)
					
					opt_point = drift_model[drift_model.find('x')+1] # 'i' or 'f'
					self.distribution_fits[dist_name][f'tau_{opt_point}_base'] = np.exp(self._opt_result[f'lntau_{opt_point}_base'])
					self.distribution_fits[dist_name][f'R_{opt_point}_base'] = self._rescale_coef(self._opt_result[f'R_{opt_point}_base'],dist_type)
					
					if drift_model.find('compressed') > -1:
						self.distribution_fits[dist_name]['k_d'] = np.exp(self._opt_result['ln_k'])
					
				self.R_inf_base = self._rescale_coef(self._opt_result['Rinf_base'],'series')
				self.R_inf_dG = self._opt_result['delta_G_Rinf']
				self.inductance = self._rescale_coef(self._opt_result['induc'],'series')
				
				if adjust_temp:
					self.T_offset = self._opt_result['temp_offset']
			
			# store error structure parameters
			if part=='both':
				# scaled parameters
				self.error_fit['sigma_min'] = self._rescale_coef(sigma_min,'series')
				for param in ['sigma_tot','sigma_res']:
					self.error_fit[param] = self._rescale_coef(self._opt_result[param],'series')
				# unscaled parameters
				for param in ['alpha_prop','alpha_re','alpha_im']:
					self.error_fit[param] = self._opt_result[param]
				# outlier contribution
				if outliers:
					self.error_fit['sigma_out'] = self._rescale_coef(self._opt_result['sigma_out'],'series')
			elif part=='polar':
				self.error_fit['sigma_Zmod_min'] = self._rescale_coef(sigma_min,'series')
				self.error_fit['sigma_Zmod_res'] = self._rescale_coef(self._opt_result['sigma_Zmod_res'],'series')
				self.error_fit['alpha_prop'] = self._opt_result['alpha_prop']
				
				self.error_fit['sigma_Zphase_min'] = 0.00175
				self.error_fit['sigma_Zphase_res'] = self._opt_result['sigma_Zphase_res']
			
		elif mode=='sample':
			# sample posterior
			self._sample_result = model.sampling(dat,warmup=warmup,iter=warmup+sample,chains=chains,seed=random_seed,init=init,
								  control={'adapt_delta':0.9,'adapt_t0':10})
								  
			# extract coefficients
			self.distribution_fits = {}
			self.error_fit = {}
			if model_type in ['Series','Parallel']:
				dist_name = [k for k,v in self.distributions.items() if v['dist_type']==model_type.lower()][0]
				dist_type = self.distributions[dist_name]['dist_type']
				self.distribution_fits[dist_name] = {'R_i_base':self._rescale_coef(np.mean(self._sample_result['R_i_base'],axis=0),dist_type)}
				try:
					self.distribution_fits[dist_name]['tau_i_base'] = np.exp(np.mean(self._sample_result['lntau_i_base'],axis=0))
				except Exception:
					self.distribution_fits[dist_name]['tau_i_base'] = np.exp(self._sample_result['lntau_f_base'] + self._sample_result['delta_lnR_base']*np.exp(self._sample_result['ln_phi_d']))
				self.distribution_fits[dist_name]['alpha_HN'] = np.mean(self._sample_result['alpha_HN'],axis=0)
				self.distribution_fits[dist_name]['beta_HN'] = np.mean(self._sample_result['beta_HN'],axis=0)
				# self.distribution_fits[dist_name]['delta_G'] = self._sample_result['delta_G']
				try:
					self.distribution_fits[dist_name]['delta_G_i'] = np.mean(self._sample_result['delta_G_i'],axis=0)
					self.distribution_fits[dist_name]['delta_G_f'] = np.mean(self._sample_result['delta_G_f'],axis=0)
				except Exception:
					self.distribution_fits[dist_name]['delta_G_i'] = np.mean(self._sample_result['delta_G'],axis=0)
					self.distribution_fits[dist_name]['delta_G_f'] = np.mean(self._sample_result['delta_G'],axis=0)
				self.distribution_fits[dist_name]['phi_T'] = np.exp(np.mean(self._sample_result['ln_phi_T'],axis=0))
				self.distribution_fits[dist_name]['T_base'] = T_base
				
				# self.distribution_fits[dist_name]['delta_R_base'] = self._rescale_coef(self._sample_result['deltaR_base'],dist_type)
				self.distribution_fits[dist_name]['R_f_base'] = self._rescale_coef(np.mean(self._sample_result['R_f_base'],axis=0),dist_type)
				self.distribution_fits[dist_name]['k_base'] = np.exp(np.mean(self._sample_result['ln_k_base'],axis=0))
				self.distribution_fits[dist_name]['delta_G_k'] = np.mean(self._sample_result['delta_G_k'],axis=0)
				self.distribution_fits[dist_name]['phi_d'] = np.exp(np.mean(self._sample_result['ln_phi_d'],axis=0))
				
				self.R_inf_base = self._rescale_coef(np.mean(self._sample_result['Rinf_base']),'series')
				self.R_inf_dG = np.mean(self._sample_result['delta_G_Rinf'])
				self.inductance = self._rescale_coef(np.mean(self._sample_result['induc']),'series')
				
				if adjust_temp:
					self.T_offset = np.mean(self._sample_result['temp_offset'],axis=0)
			
			# store error structure parameters
			if part=='both':
				# scaled parameters
				self.error_fit['sigma_min'] = self._rescale_coef(np.mean(sigma_min),'series')
				self.error_fit['sigma_res'] = self._rescale_coef(np.mean(self._sample_result['sigma_res']),'series')
				self.error_fit['sigma_tot'] = self._rescale_coef(np.mean(self._sample_result['sigma_tot'],axis=0),'series')
				# unscaled parameters
				for param in ['alpha_prop','alpha_re','alpha_im']:
					self.error_fit[param] = np.mean(self._sample_result[param])
				# outlier contribution
				if outliers:
					self.error_fit['sigma_out'] = self._rescale_coef(np.mean(self._sample_result['sigma_out'],axis=0),'series')
			elif part=='polar':
				self.error_fit['sigma_Zmod_min'] = self._rescale_coef(sigma_min,'series')
				self.error_fit['sigma_Zmod_res'] = self._rescale_coef(self._sample_result['sigma_Zmod_res'],'series')
				self.error_fit['alpha_prop'] = self._sample_result['alpha_prop']
				
				self.error_fit['sigma_Zphase_min'] = 0.00175
				self.error_fit['sigma_Zphase_res'] = self._sample_result['sigma_Zphase_res']
		
		if mode=='optimize':
			self.fit_type = 'map-activation-drift'
		elif mode=='sample':
			self.fit_type = 'hmc-activation-drift'
			
	# def extract_fit_params(self,result,mode,scalars,vectors,arrays)
			
	def map_drift_fit(self,frequencies,Z,times,drift_model='',part='both',scale_Z=True,nonneg=True,outliers=False,
		min_tau=None,max_tau=None,num_HN=10,add_stan_data={},ordered=True,#shapeshift=False,
		model_str=None,
		# initialization parameters
		init_from_ridge=False,init_from_map=False,init_basis_freq=None,init_drt_fit=None,init_values=None,init_kw={},peakfit_kw={},
		# optimization control
		sigma_min=0.002,max_iter=50000,random_seed=1234,inductance_scale=1,outlier_lambda=5,init_alpha=1e-3):
		"""
		Obtain the maximum a posteriori estimate of the defined distribution(s) (and all model parameters).
		
		Parameters:
		-----------
		frequencies: array
			Measured frequencies
		Z: complex array
			Measured (complex) impedance values. Must have same length as frequencies
		part: str, optional (default: 'both')
			Which part of the impedance data to fit. Options: 'both', 'real', 'imag', 'polar'
		scale_Z: bool, optional (default: True)
			If True, scale impedance by the factor sqrt(N)/std(|Z|) to normalize for magnitude and sample size
		init_from_ridge: bool, optional (default: False)
			If True, use the hyperparametric ridge solution to initialize the Bayesian fit. 
			Only valid for single-distribution fits
		nonneg: bool, optional (default: False)
			If True, constrain the DRT to non-negative values
		outliers: bool, optional (default: False)
			If True, enable outlier identification via independent error contribution variable
		sigma_min: float, optional (default: 0.002)
			Impedance error floor. This is necessary to avoid sampling/optimization errors.
			Values smaller than the default (0.002) may enable slightly closer fits of very clean data,
			but may also result in sampling/optimization errors that yield unexpected results.
		max_iter: int, optional (default: 50000)
			Maximum number of iterations to allow the optimizer to perform
		random_seed: int, optional (default: 1234)
			Random seed for optimizer
		inductance_scale: float, optional (default: 1)
			Scale (std of normal prior) of the inductance. Lower values will constrain the inductance,
			which may be helpful if the estimated inductance is anomalously large (this may occur if your
			measured impedance data does not extend to high frequencies, i.e. 1e5-1e6 Hz)
		outlier_lambda: float, optional (default: 5)
			Lambda parameter (inverse scale) of the exponential prior on the outlier error contribution.
			Smaller values will make it easier for points to be flagged as outliers 
		ridge_kw: dict, optional (default: {})
			Keyword arguments to pass to ridge_fit if init_from_ridge==True.
		"""
		# load stan model	
		if model_str is None:
			model_str = 'Series_HN-drift'
			if drift_model!='':
				model_str += f'-{drift_model}'
			if nonneg:
				model_str += '_pos'
			if ordered:
				model_str += '_ordered'
			model_str += '_StanModel.pkl'
		
		model = load_pickle(os.path.join(script_dir,'stan_model_files',model_str))
		self.stan_model_name = model_str
		model_type = model_str.split('_')[0]
		
		# identify measurement start and end indices
		start_idx = np.where(np.diff(frequencies) > 0)[0] + 1
		if len(start_idx) > 0:
			meas_start_indices = np.concatenate(([0],start_idx)).astype(int)
		else:
			meas_start_indices = np.array([0],dtype=int)
		meas_end_indices = np.zeros_like(meas_start_indices)
		meas_end_indices[:-1] = meas_start_indices[1:]
		meas_end_indices[-1] = len(frequencies)
		
		# perform scaling. Scale based on first spectrum
		Z_base = Z[meas_start_indices[0]:meas_end_indices[0]]
		self._scale_Z(Z_base,'map')
		Z_scaled = Z/self._Z_scale
		self.f_train = frequencies
		freq_base = frequencies[meas_start_indices[0]:meas_end_indices[0]]
		
		rel_Z_scale = np.ones(len(frequencies))
		for start,end in zip(meas_start_indices,meas_end_indices):
			Z_i = Z[start:end]
			Zmod_i = (Z_i*Z_i.conjugate())**0.5
			Z_scale_i = np.std(Zmod_i)/np.sqrt(len(Z_i)/81)
			rel_Z_scale[start:end] = Z_scale_i/self._Z_scale
			
		# prepare data for stan model 
		dat = self._prep_stan_data(frequencies,Z_scaled,part,model_type,sigma_min,mode='optimize',inductance_scale=inductance_scale)
		# if tau boundaries not specified, allow HN peaks to extend 2 decades beyond measurement range
		if min_tau is None:
			min_tau = np.min(1/(2*np.pi*freq_base))/10
		if max_tau is None:
			max_tau = np.max(1/(2*np.pi*freq_base))*10
		dat['N'] = len(frequencies)
		dat['K'] = num_HN
		
		dat['times'] = times
		dat['rel_Z_scale'] = rel_Z_scale
		
		if drift_model=='':
			dat['R_i_scale_alpha'] = 1
			dat['R_i_scale_beta'] = 2
			dat['sigma_delta_R_scale'] = np.ones(num_HN)*0.05
			dat['sigma_delta_Rinf_scale'] = 0.05
			# dat['deltaR_scale_alpha'] = 5
			# dat['deltaR_scale_beta'] = 0.2
			dat['max_phi_d'] = 2
			dat['ln_phi_d_scale'] = 0.2
			dat['min_k'] = 1e-4
			dat['max_k'] = 1
		elif drift_model.find('RQ') > -1:
			if drift_model.find('xi') > -1:
				dat['R_i_scale_alpha'] = 1
				dat['R_i_scale_beta'] = 2
			elif drift_model.find('xf') > -1:
				dat['R_f_scale_alpha'] = 1
				dat['R_f_scale_beta'] = 2
			dat['sigma_R_rq_scale'] = 0.05
			dat['sigma_delta_Rinf_scale'] = 0.05
			
			# constrain time-dependent ZARC to fall between min_tau and max_tau
			dat['min_tau_rq'] = min_tau
			dat['max_tau_rq'] = max_tau
			
			if drift_model.find('lin')==-1:
				dat['min_k'] = 1e-4
				dat['max_k'] = 1
		
		dat.update(add_stan_data)
		self._stan_input = dat.copy()
		
		# Create tau grid for convenience, using min_tau and max_tau
		num_decades = int(np.ceil(np.log10(max_tau)) - np.floor(np.log10(min_tau)))
		for distribution in self.distributions.keys():
			self.distributions[distribution]['tau'] = np.logspace(np.log10(min_tau),np.log10(max_tau),num_decades*10 + 1)
		
		# Get initial parameter values
		iv = {}
		# if np.sum([init_from_ridge,init_from_map,init_drt_fit is not None]) > 1:
			# raise ValueError('Only one of init_from_ridge, init_from_map, and init_drt_fit may be used')
		# if init_from_ridge:
			# if len(self.distributions) > 1:
				# raise ValueError('Ridge initialization can only be performed for single-distribution fits')
			# else:
				# init = self._get_init_from_ridge(freq_base,Z_base,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
				# iv.update(init)
				# # Update parameter names for activation model
				# iv['Rinf_base'] = iv['Rinf']
				# iv['Rinf_base_raw'] = init['Rinf_raw']
				# iv['lntau_HN_base'] = iv['lntau_HN']
				# iv['R_HN_base'] = iv['R_HN']
		# elif init_from_map:
			# init = self._get_init_from_map(freq_base,Z_base,init_basis_freq,nonneg,inductance_scale,num_HN,min_tau,max_tau,init_kw,peakfit_kw)
			# iv.update(init)
			# # Update parameter names for activation model
			# iv['Rinf_base'] = iv['Rinf']
			# iv['Rinf_base_raw'] = init['Rinf_raw']
			# iv['lntau_HN_base'] = iv['lntau_HN']
			# iv['R_HN_base'] = iv['R_HN']
		# elif init_drt_fit is not None:
			# init = self._get_init_from_drt(init_drt_fit,num_HN,inductance_scale,min_tau,max_tau,peakfit_kw)
			# iv.update(init)
			# # Update parameter names for activation model
			# iv['Rinf_base'] = iv['Rinf']
			# iv['Rinf_base_raw'] = init['Rinf_raw']
			# iv['lntau_HN_base'] = iv['lntau_HN']
			# iv['R_HN_base'] = iv['R_HN']
		# else:
			# # distribute lntau values uniformly
			# iv['lntau_HN_base'] = np.linspace(np.log(min_tau*10),np.log(max_tau/10),num_HN)
			# iv['R_HN_base'] = np.ones(num_HN)
			# iv['alpha_HN'] = np.ones(num_HN)*0.95
			# iv['beta_HN'] = np.ones(num_HN)*0.8
			# iv['upsilon'] = np.ones(num_HN)*1
			# iv['delta_G'] = np.ones(num_HN)*0.5
			# iv['delta_G_Rinf'] = 0.5
			# iv['ln_phi_T'] = np.zeros(num_HN)
			# iv['induc'] = 1e-8
			# iv['induc_raw'] = iv['induc']/inductance_scale
			# iv['detaR_base_raw'] = np.zeros(num_HN)
			# iv['detaR_base'] = np.zeros(num_HN)
		
		# if outliers:
			# raise ValueError('Outlier model not yet implemented')
			# # initialize sigma_out near zero, everything else randomly
			# iv['sigma_out_raw'] = np.zeros(2*len(Z)) + 0.1
			# dat['so_invscale'] = outlier_lambda
			
		# # if shapeshift:
			# # for param in ['alpha_HN','beta_HN']:
				# # iv[param] = np.tile(iv[param],(dat['P'],1))
				
				
		if init_values is not None:
			iv = init_values
			
		def init():
			return iv
		self._init_params = iv
				
		# print(iv)
		
		# optimize posterior
		self._opt_result = model.optimizing(dat,iter=max_iter,seed=random_seed,init=init,init_alpha=init_alpha)
		
		# extract coefficients
		self.distribution_fits = {}
		self.error_fit = {}
		if model_type in ['Series','Parallel']:
			dist_name = [k for k,v in self.distributions.items() if v['dist_type']==model_type.lower()][0]
			dist_type = self.distributions[dist_name]['dist_type']
			
			if drift_model=='':
				self.distribution_fits[dist_name] = {'R_i':self._rescale_coef(self._opt_result['R_i'],dist_type)}
				self.distribution_fits[dist_name]['tau_i'] = np.exp(self._opt_result['lntau_i'])
				self.distribution_fits[dist_name]['alpha_HN'] = self._opt_result['alpha_HN']
				self.distribution_fits[dist_name]['beta_HN'] = self._opt_result['beta_HN']
				
				# self.distribution_fits[dist_name]['delta_R_base'] = self._rescale_coef(self._opt_result['deltaR_base'],dist_type)
				self.distribution_fits[dist_name]['R_f'] = self._rescale_coef(self._opt_result['R_f'],dist_type)
				self.distribution_fits[dist_name]['k_d'] = np.exp(self._opt_result['ln_k'])
				self.distribution_fits[dist_name]['phi_d'] = np.exp(self._opt_result['ln_phi_d'])
				
				self.R_inf_i = self._rescale_coef(self._opt_result['Rinf_i'],'series')
				self.R_inf_f = self._rescale_coef(self._opt_result['Rinf_f'],'series')
				self.R_inf_kd = np.exp(self._opt_result['ln_k_Rinf'])
				self.inductance = self._rescale_coef(self._opt_result['induc'],'series')
			elif drift_model.find('RQ') > -1:
				if drift_model.find('xi') > -1:
					self.distribution_fits[dist_name] = {'R_i':self._rescale_coef(self._opt_result['R_i'],dist_type)}
					self.distribution_fits[dist_name]['tau_i'] = np.exp(self._opt_result['lntau_i'])
				elif drift_model.find('xf') > -1:
					self.distribution_fits[dist_name] = {'R_f':self._rescale_coef(self._opt_result['R_f'],dist_type)}
					self.distribution_fits[dist_name]['tau_f'] = np.exp(self._opt_result['lntau_f'])
				self.distribution_fits[dist_name]['alpha_HN'] = self._opt_result['alpha_HN']
				self.distribution_fits[dist_name]['beta_HN'] = self._opt_result['beta_HN']
				
				self.distribution_fits[dist_name]['R_rq'] = self._rescale_coef(self._opt_result['R_rq'],dist_type)
				self.distribution_fits[dist_name]['tau_rq'] = self._opt_result['tau_rq']
				self.distribution_fits[dist_name]['phi_rq'] = self._opt_result['phi_rq']
				
				
				self.distribution_fits[dist_name]['t_i'] = np.min(times)
				self.distribution_fits[dist_name]['t_f'] = np.max(times)
				if drift_model.find('lin')==-1:
					self.distribution_fits[dist_name]['k_d'] = np.exp(self._opt_result['ln_k'])
				
				self.R_inf_i = self._rescale_coef(self._opt_result['Rinf_i'],'series')
				self.R_inf_f = self._rescale_coef(self._opt_result['Rinf_f'],'series')
				self.inductance = self._rescale_coef(self._opt_result['induc'],'series')
				
		
		# store error structure parameters
		# scaled parameters
		self.error_fit['sigma_min'] = self._rescale_coef(sigma_min,'series')
		for param in ['sigma_tot','sigma_res']:
			self.error_fit[param] = self._rescale_coef(self._opt_result[param],'series')
		# unscaled parameters
		for param in ['alpha_prop','alpha_re','alpha_im']:
			self.error_fit[param] = self._opt_result[param]
		# outlier contribution
		if outliers:
			self.error_fit['sigma_out'] = self._rescale_coef(self._opt_result['sigma_out'],'series')
		
		self.fit_type = 'map-drift'
	

			
		
				  