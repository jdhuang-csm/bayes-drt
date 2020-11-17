import numpy as np
from scipy.linalg import toeplitz, cholesky
from scipy.integrate import quad
from scipy.special import loggamma
from scipy.optimize import least_squares, minimize, minimize_scalar
import pandas as pd
# from sklearn.metrics import r2_score
import cvxopt
import pystan as stan
import warnings
from stan_models import load_pickle
import os
from copy import deepcopy

cvxopt.solvers.options['show_progress'] = False

script_dir = os.path.dirname(os.path.realpath(__file__))

warnings.simplefilter('always',UserWarning)
warnings.simplefilter('once',RuntimeWarning)


class Inverter:
	def __init__(self,basis_freq=None,basis='gaussian',epsilon=None,fit_inductance=True,distributions={'DRT':{'kernel':'DRT'}}): #kernel='DRT',dist_type='series',symmetry='planar',bc=None,ct=False,tk=None
		self._recalc_mat = True
		self.distribution_matrices = {}
		self.set_basis_freq(basis_freq)
		self.set_basis(basis)
		self.set_epsilon(epsilon) # inverse length scale of RBF. mu in paper
		self.set_fit_inductance(fit_inductance)
		self.set_distributions(distributions)
		self.f_train = [0]
		self._Z_scale = 1.0
		
	def set_distributions(self,distributions):
		"""Set kernels for inversion
		
		Parameters:
		-----------
		distributions: dict
			Dict of dicts describing distributions to include. Top-level keys are user-supplied names for distributions. Each dict should include the following keys:
				kernel: 'DRT' or 'DDT'
				dist_type: 'series' or 'parallel' (default 'parallel'). Valid for DDT only
				symmetry: 'planar' or 'spherical' (default 'planar'). Valid for DDT only
				bc: 'transmissive' or 'blocking' (default 'transmissive'). Valid for DDT only
				ct: boolean indicating whether there is a simultaneous charge transfer reaction (default False). Valid for DDT only
				k_ct: apparent 1st-order rate constant for simultaneous charge transfer reaction. Required for DDT if ct==True
				basis_freq: array of frequencies to use as basis. If not specified, use self.basis_freq
				epsilon: epsilon value for basis functions. If not specified, use self.epsilon
		"""
		# perform checks and normalizations
		for name,info in distributions.items():
			if info['kernel']=='DRT':
				# set dist_type to series and warn if overwriting
				if info.get('dist_type','series') !='series':
					warnings.warn("dist_type for DRT kernel must be series. \
					Overwriting supplied dist_type '{}' for distribution '{}' with 'series'".format(name,info['dist_type']))
				info['dist_type'] = 'series'
				# check for invalid keys and warn
				invalid_keys = np.intersect1d(list(info.keys()),['symmetry','bc','ct','k_ct'])
				if len(invalid_keys) > 0:
					warnings.warn("The following keys are invalid for distribution '{}': {}.\
					\n These keys will be ignored".format(name,invalid_keys))
				
			elif info['kernel']=='DDT':
				# check for invalid key-value pairs
				if info.get('dist_type','parallel') not in ['series','parallel']:
					raise ValueError("Invalid dist_type '{}' for distribution '{}'".format(info.get('dist_type','NA'),name))
				elif info.get('symmetry','planar') not in ['planar','spherical']:
					raise ValueError("Invalid symmetry '{}' for distribution '{}'".format(info.get('symmetry','NA'),name))
				elif info.get('bc','transmissive') not in ['transmissive','blocking']:
					raise ValueError("Invalid bc '{}' for distribution '{}'".format(info.get('bc','NA'),name))
				elif info.get('ct',True) not in [True,False]:
					raise ValueError("Invalid ct {} for distribution '{}'".format(info['ct'],name))
				
				# check if k_ct missing
				if info.get('ct',False)==True:
					if 'k_ct' not in info.keys():
						raise ValueError("k_ct must be supplied for distribution '{}' if ct==True".format(name))
					
				# set defaults and update with supplied args
				defaults = {'dist_type':'parallel','symmetry':'planar','bc':'blocking','ct':False}
				defaults.update(info)
				distributions[name] = defaults	
					
			if name not in self.distribution_matrices.keys():
				self.distribution_matrices[name] = {}
			
		self._distributions = distributions
		
		self._recalc_mat = True
		# print('called set_distributions')
		
	def get_distributions(self):
		# print('called get_distributions')
		return self._distributions
		
	distributions = property(get_distributions,set_distributions)
		
		
	def ridge_fit(self,frequencies,Z,part='both',penalty='discrete',reg_ord=2,L1_penalty=0,
		# hyper_lambda parameters
		hyper_lambda=True,hl_solution='analytic',hl_beta=2.5,hl_fbeta=None,lambda_0=1e-2,cv_lambdas=np.logspace(-10,5,31),
		# hyper_weights parameters
		hyper_weights=False,hw_beta=2,hw_wbar=1,
		# gamma distribution hyperparameters
		hyper_a=False,alpha_a=2,hl_beta_a=2,hyper_b=False,sb=1,
		# other parameters
		x0=None,weights=None,xtol=1e-3,max_iter=20,
		scale_Z=True,nonneg=True,
		dZ=True,dZ_power=0.5):
		"""
		weights : str or array (default: None)
			Weights for fit. Standard weighting schemes can be specified by passing 'modulus' or 'proportional'. 
			Custom weights can be passed as an array. If the array elements are real, the weights are applied to both the real and imaginary parts of the impedance.
			If the array elements are complex, the real parts are used to weight the real impedance, and the imaginary parts are used to weight the imaginary impedance.
			If None, all points are weighted equally.
		"""
		# checks
		if penalty in ('discrete','cholesky'):
			if hl_beta <= 1:
				raise ValueError('hl_beta must be greater than 1 for penalty ''cholesky'' and ''discrete''')
		elif penalty=='integral':
			if hl_beta <= 2:
				raise ValueError('hl_beta must be greater than 2 for penalty ''integral''')
		else:
			raise ValueError(f'Invalid penalty argument {penalty}. Options are ''integral'', ''discrete'', and ''cholesky''')
			
		if hyper_lambda and hyper_weights:
			raise ValueError('hyper_lambda and hyper_weights fits cannot be performed simultaneously')
			
		if len(self.distributions) > 1:
			raise ValueError('ridge_fit cannot be used to fit multiple distributions')
			
		self.distribution_fits = {}
		
		
		
		# perform Re-Im CV to get optimal lambda_0
		if lambda_0=='cv':
			lambda_0 = self.ridge_ReImCV(frequencies,Z,lambdas=cv_lambdas,
										penalty=penalty,hl_solution=hl_solution,
										hl_beta=hl_beta,hl_fbeta=hl_fbeta,
										reg_ord=reg_ord,L1_penalty=L1_penalty,
										x0=x0,weights=weights,xtol=xtol,max_iter=max_iter,
										scale_Z=scale_Z,nonneg=nonneg,
										dZ=dZ,dZ_power=dZ_power,
										hyper_a=hyper_a,alpha_a=alpha_a,hl_beta_a=hl_beta_a,hyper_b=hyper_b,sb=sb)
		
		
		# ridge_fit can only handle a single distribution. For convenience, pull distribution info out of dicts
		dist_name = list(self.distributions.keys())[0]
		dist_info = self.distributions[dist_name]
		if dist_info['kernel']!='DRT' and dZ==True:
			warnings.warn('dZ should only be set to True for DRT recovery. Proceeding with dZ=False')
			dZ = False
		
		# set fit target
		if dist_info['dist_type']=='series':
			# use impedance for series distributions (e.g. DRT)
			target = Z
		elif dist_info['dist_type']=='parallel':
			# for parallel distributions, must fit admittance for linearity 
			target = 1/Z
			
		# perform scaling and weighting and get matrices
		frequencies, target_scaled, WT_re,WT_im,W_re,W_im, dist_mat = self._prep_matrices(frequencies,target,part,weights,dZ,scale_Z,penalty,'ridge')
		
		if dist_info['dist_type']=='parallel' and scale_Z:
			# redo the scaling such that Z is still the variable that gets scaled
			# this helps avoid tiny admittances, which get ignored in fitting
			Z_scaled = self._scale_Z(Z,'ridge')
			target_scaled = 1/Z_scaled
			WT_re = W_re@target_scaled.real
			WT_im = W_im@target_scaled.imag
		
		
		# unpack matrices
		matrices = dist_mat[dist_name]
		A_re = matrices['A_re']
		A_im = matrices['A_im']
		WA_re = matrices['WA_re']
		WA_im = matrices['WA_im']
		B = matrices['B']
		if penalty!='integral':
			L0 = matrices['L0']
			L1 = matrices['L1']
			L2 = matrices['L2']
		M0 = matrices.get('M0',None)
		M1 = matrices.get('M1',None)
		M2 = matrices.get('M2',None)
		tau = dist_info['tau']
		epsilon = dist_info['epsilon']
		
		
		# for series distributions, adjust matrices for R_inf and inductance
		if dist_info['dist_type']=='series':
			# add columns to A_re, B, and A_im
			A_re_main = A_re.copy()
			A_re = np.zeros((A_re_main.shape[0],A_re_main.shape[1]+2))
			A_re[:,2:] = A_re_main
			A_re[:,0] = 1
			
			if B is not None:
				B = np.hstack((np.zeros((B.shape[1],2)), B))
			
			A_im_main = A_im.copy()
			A_im = np.zeros((A_im_main.shape[0],A_im_main.shape[1]+2))
			A_im[:,2:] = A_im_main
			if self.fit_inductance:
				A_im[:,1] = 2*np.pi*frequencies*1e-4 # scale the inductance column so that coef[1] doesn't drop below the tolerance of cvxopt
				
			# re-apply weight matrices to expanded A matrices
			WA_re = W_re@A_re
			WA_im = W_im@A_im
				
			# add rows and columns to M matrices
			if M0 is not None:
				M0_main = M0.copy()
				M0 = np.zeros((M0_main.shape[0]+2,M0_main.shape[1]+2))
				M0[2:,2:] = M0_main
			if M1 is not None:
				M1_main = M1.copy()
				M1 = np.zeros((M1_main.shape[0]+2,M1_main.shape[1]+2))
				M1[2:,2:] = M1_main
			if M2 is not None:
				M2_main = M2.copy()
				M2 = np.zeros((M2_main.shape[0]+2,M2_main.shape[1]+2))
				M2[2:,2:] = M2_main
				
			# add columns to L matrices
			if penalty!='integral':
				L0 = np.hstack((np.zeros((A_re.shape[1]-2,2)), L0))
				L1 = np.hstack((np.zeros((A_re.shape[1]-2,2)), L1))
				L2 = np.hstack((np.zeros((A_re.shape[1]-2,2)), L2))
			
		
				
		# convert reg_ord to list
		if type(reg_ord)==int:
			ord_idx = reg_ord
			reg_ord = np.zeros(3)
			reg_ord[ord_idx] = 1
			
		# get penalty matrices
		if penalty in ['integral','cholesky']:
			L2_base = [M0,M1,M2]
		elif penalty=='discrete':
			L2_base = [L.T@L for L in [L0,L1,L2]]
			
		# create L1 penalty vector
		L1_vec = np.ones(A_re.shape[1])*np.pi**0.5/epsilon*L1_penalty
		if dist_info['dist_type']=='series':
			L1_vec[0:2] = 0 # inductor and high-frequency resistor are not penalized

		
		# format hl_beta and lambda_0 for each regularization order
		if type(hl_beta) in (float,int,np.float64):
			hl_beta = np.array([hl_beta]*3)
		else:
			hl_beta = np.array(hl_beta)
		a_list = hl_beta/2
		if penalty=='integral':
			b_list = 0.5*(2*a_list-2)/lambda_0
			hyper_as = np.array([np.ones(A_re.shape[1])*a for a in a_list])
			hyper_bs = np.array([np.ones(A_re.shape[1])*b for b in b_list])
			hyper_lambda0s = (2*hyper_as-2)/(2*hyper_bs)
		else:
			b_list = 0.5*(2*a_list-1)/lambda_0
			hyper_as = np.array([np.ones(A_re.shape[1])*a for a in a_list])
			hyper_bs = np.array([np.ones(A_re.shape[1])*b for b in b_list])
			hyper_lambda0s = (2*hyper_as-1)/(2*hyper_bs)
		hyper_hl_betas = 2*hyper_as
		# print(hyper_lambda0s)
		# print(hyper_hl_betas)
		
		if type(alpha_a) in (float,int):
			alpha_a = 3*[alpha_a]
		if type(hl_beta_a) in (float,int):
			hl_beta_a = 3*[hl_beta_a]
		if type(sb) in (float,int):
			sb = 3*[sb]
		
		# Hyper-lambda fit
		# --------------------------
		if hyper_lambda:
			self._iter_history = []
			iter = 0
			if x0 is not None:
				coef = x0
			else:
				coef = np.zeros(A_re.shape[1])+1e-6
			
			lam_vectors = [np.ones(A_re.shape[1])*lambda_0]*3
			
			# lam_vec[0:2] = 0
			lam_matrices = [np.diag(lam_vec**0.5) for lam_vec in lam_vectors]
			# lam_step = np.zeros_like(lam_vectors[0])
			
			dZ_re = np.ones(A_re.shape[1])
			
			L2_mat = np.zeros_like(L2_base[0])
			for L2b,lam_mat,frac in zip(L2_base,lam_matrices,reg_ord):
				if frac>0:
					L2_mat += frac*(lam_mat@L2b@lam_mat)
			P = WA_re.T@WA_re + WA_im.T@WA_im + L2_mat
			q = (-WA_re.T@WT_re - WA_im.T@WT_im + L1_vec)
			cost = 0.5*coef.T@P@coef + q.T@coef
			
			# if hyper_b:
				# hyper_b_converged = [False]*3
			# else:
				# hyper_b_converged = [True]*3
				
			while iter < max_iter:
			
				
				# L2_mat = lam_mat@L2_base@lam_mat
				# P = WA_re.T@WA_re + WA_im.T@WA_im + L2_mat
				# q = (-WA_re.T@WZ_re + WA_im.T@WZ_im + L1_vec)
				# cost = 0.5*coef.T@P@coef + q.T@coef
				# print('iteration',iter)
				# print('cost before hyper lambda:',cost)
				
				prev_coef = coef.copy()
				prev_lam = lam_vectors.copy()
				# prev_step = lam_step.copy()
				
				if dZ and iter > 0:
					dZ_raw = B@prev_coef
					# scale by tau spacing to get dZ'/dlnt
					dlnt = np.mean(np.diff(np.log(tau)))
					dZ_raw /= (dlnt/0.23026)
					dZ_re[2:] = (np.abs(dZ_raw))**dZ_power
					# for stability, dZ_re must not be 0
					dZ_re[np.abs(dZ_re<1e-8)] = 1e-8
				

				if hyper_b and iter > 0:
					for n,(lam_vec,frac,sbn) in enumerate(zip(lam_vectors,reg_ord,sb)):
						# if hyper_b_converged[n]==False:
						if frac > 0:
							prev_b = hyper_bs[n].copy()
							hb = self._hyper_b(lam_vec,hyper_as[n],sbn)
							# for stability, b must be >0
							hb[hb<1e-8] = 1e-8
							hyper_bs[n] = hb
							hyper_lambda0s[n] = (2*hyper_as[n]-2)/hyper_bs[n]
							# print(hyper_bs[n])
							# if np.mean((hyper_bs[n]-prev_b)/prev_b) < 1e-3:
								# hyper_b_converged[n] = True
				
				if hyper_a and iter>0:
					for n,(lam_vec,frac,aa,ba) in enumerate(zip(lam_vectors,reg_ord,alpha_a,hl_beta_a)):
						# if hyper_b_converged[n]:
						if frac > 0:
							hyper_as[n] = np.ones(len(hyper_bs[n]))*self._hyper_a(lam_vec,hyper_bs[n],aa,ba)
							# print(hyper_as[n])
							hyper_lambda0s[n] = (2*hyper_as[n]-2)/hyper_bs[n]
							hyper_hl_betas[n] = 2*hyper_as[n]
				
				
						
				# solve for lambda
				if penalty in ('discrete','cholesky'):
					if hl_solution=='analytic':
						if hl_fbeta is not None:
							for n,(Ln,frac,hlam0,hhl_beta) in enumerate(zip([L0,L1,L2],reg_ord,hyper_lambda0s,hyper_hl_betas)):
								if frac > 0:
									lam_vectors[n] = self._hyper_lambda_fbeta(Ln,prev_coef/dZ_re,dist_info['dist_type'],hl_fbeta=hl_fbeta,lambda_0=lambda_0)
						else:
							for n,(Ln,frac,hlam0,hhl_beta) in enumerate(zip([L0,L1,L2],reg_ord,hyper_lambda0s,hyper_hl_betas)):
								if frac > 0:
									lam_vectors[n] = self._hyper_lambda_discrete(Ln,prev_coef/dZ_re,dist_info['dist_type'],hl_beta=hhl_beta[2:],lambda_0=hlam0[2:])
						
					elif hl_solution=='lm':
						zeta = (hl_beta-1)/lambda_0
						def jac(x,L,coef):
							# off-diagonal terms are zero
							diag = (L@coef)**2 + zeta - (hl_beta-1)/x
							return np.diag(diag)
							
						def fun(x,L,coef):
							return ((L@coef)**2 + zeta)*x - (hl_beta-1)*np.log(x)
						
						for n,(Ln,frac) in enumerate(zip([L0,L1,L2],reg_ord)):
							if dist_info['dist_type']=='series':
								start = 2
							else:
								start = 0
								
							if frac > 0:
								result = least_squares(fun,prev_lam[start:],jac=jac,method='lm',xtol=lambda_0*1e-3,args=([Ln,coef]),max_nfev=100)
								lam_vectors[n][start:] = result['x']
						
				elif penalty=='integral':
					if hl_solution=='analytic':
						# print('iter',iter)
						# D = np.diag(dZ_re**(-1))
						for n,(L2b,lam_mat,frac,hlam0,hhl_beta) in enumerate(zip(L2_base,lam_matrices,reg_ord,hyper_lambda0s,hyper_hl_betas)):
							if frac > 0:
								# lam_vectors[n] = self._hyper_lambda_integral(L2b,prev_coef,D@lam_mat,hl_beta=hl_beta,lambda_0=lambda_0)
								if n==0:
									factor = 100
								elif n==1:
									factor=10
								else:
									factor=1
								lv = self._hyper_lambda_integral(L2b,factor*prev_coef/dZ_re,lam_mat,hl_beta=hhl_beta,lambda_0=hlam0)
								# handle numerical instabilities that may arise for large lambda_0 and small hl_beta
								lv[lv<=0] = 1e-15
								lam_vectors[n] = lv
								# print(n,lam_vectors[n])
				
				# lam_vec[lam_vec < 0] = 0
				# print(lam_vectors)
				lam_matrices = [np.diag(lam_vec**0.5) for lam_vec in lam_vectors]
				L2_mat = np.zeros_like(L2_base[0])
				D = np.diag(dZ_re**(-1))
				for L2b,lam_mat,frac in zip(L2_base,lam_matrices,reg_ord):
					if frac>0:
						L2_mat += frac*(D@lam_mat@L2b@lam_mat@D)
				
				
				# P = (WA_re.T@WA_re + WA_im.T@WA_im + L2_mat)
				# q = (-WA_re.T@WZ_re + WA_im.T@WZ_im + L1_vec)
				# cost = 0.5*coef.T@P@coef + q.T@coef
				# print('cost after hyper lambda:',cost)
				
				# optimize coef
				result = self._convex_opt(part,WT_re,WT_im,WA_re,WA_im,L2_mat,L1_vec,nonneg)
				coef = np.array(list(result['x']))
				
				P = (WA_re.T@WA_re + WA_im.T@WA_im + L2_mat)
				q = (-WA_re.T@WT_re - WA_im.T@WT_im + L1_vec)
				cost = 0.5*coef.T@P@coef + q.T@coef
				# for frac,lam_vec,ha,hb in zip(reg_ord,lam_vectors,hyper_as,hyper_bs):
					# cost += frac*np.sum((hb*lam_vec - (ha-1)*np.log(lam_vec)))
				# print('cost after coef optimization:',cost)
				
				self._iter_history.append({'lambda_vectors':lam_vectors.copy(),'coef':coef.copy(),'fun':result['primal objective'],'cost':cost,'result':result,'dZ_re':dZ_re.copy(),
										'hyper_bs':hyper_bs.copy(),'hyper_lambda0s':hyper_lambda0s.copy(),'hyper_hl_betas':hyper_hl_betas.copy()})
				
				# check for convergence
				coef_delta = (coef - prev_coef)/prev_coef
				# If inductance not fitted, set inductance delta to zero (inductance goes to random number)
				if self.fit_inductance==False or part=='real':
					coef_delta[1] = 0
				# print(np.mean(np.abs(coef_delta)))
				if np.mean(np.abs(coef_delta)) < xtol:
					break
				elif iter==max_iter-1:
					warnings.warn(f'Hyperparametric solution did not converge within {max_iter} iterations')
				
				iter += 1
			
			cost_hist = [h['cost'] for h in self._iter_history]
			# if cost > np.min(cost_hist):
				# best_idx = np.argmin(cost_hist)
				# warnings.warn('Final log-likelihood is not the minimum. Reverting to iteration {}'.format(best_idx+1))
				# self.opt_result_ = self._iter_history[best_idx]['result']
				# self.coef_ = self._iter_history[best_idx]['coef'].copy()
				# self.lambda_vectors_ = self._iter_history[best_idx]['lambda_vectors'].copy()
				# self.cost_ = cost.copy()
			# else:		

			self.distribution_fits[dist_name] = {'opt_result':result,'coef':coef.copy(),'lambda_vectors':lam_vectors.copy(),'cost':cost.copy()}
			
		# Hyper-weights fit
		# --------------------------
		elif hyper_weights:
			self._iter_history = []
			iter = 0
			# initialize coef and dZ
			coef = np.zeros(A_re.shape[1])+1e-6
			dZ_re = np.ones(A_re.shape[1])
				
			# get w_bar
			wbar = self._format_weights(frequencies,target_scaled,hw_wbar,part)
			# initialize weights at w_bar
			weights = wbar
			# print('wbar:',wbar)
			# wbar_re = np.real(wbar)
			# wbar_im = np.imag(wbar)
			
			lam_vectors = [np.ones(A_re.shape[1])*lambda_0]*3
			lam_matrices = [np.diag(lam_vec**0.5) for lam_vec in lam_vectors]
			L2_mat = np.zeros_like(L2_base[0])
			D = np.diag(dZ_re**(-1))
			for L2b,lam_mat,frac in zip(L2_base,lam_matrices,reg_ord):
				if frac>0:
					L2_mat += frac*(D@lam_mat@L2b@lam_mat@D)
				
			while iter < max_iter:
				
				prev_coef = coef.copy()
				prev_weights = weights.copy()
				
				# calculate new weights
				if iter > 0:
					weights = self._hyper_weights(coef,A_re,A_im,target_scaled,hw_beta,wbar)
				
				# apply weights to A and Z
				W_re = np.diag(np.real(weights))
				W_im = np.diag(np.imag(weights))
				WA_re = W_re@A_re
				WA_im = W_im@A_im
				WT_re = W_re@target_scaled.real
				WT_im = W_im@target_scaled.imag
				
				if dZ and iter > 0:
					dZ_raw = B@prev_coef
					# scale by tau spacing to get dZ'/dlnt
					dlnt = np.mean(np.diff(np.log(tau)))
					dZ_raw /= (dlnt/0.23026)
					dZ_re[2:] = (np.abs(dZ_raw))**dZ_power
					# for stability, dZ_re must not be 0
					dZ_re[np.abs(dZ_re<1e-8)] = 1e-8
				
				
				
				# optimize coef
				result = self._convex_opt(part,WT_re,WT_im,WA_re,WA_im,L2_mat,L1_vec,nonneg)
				coef = np.array(list(result['x']))
				
				P = (WA_re.T@WA_re + WA_im.T@WA_im + L2_mat)
				q = (-WA_re.T@WT_re - WA_im.T@WT_im + L1_vec)
				cost = 0.5*coef.T@P@coef + q.T@coef
				
				self._iter_history.append({'weights':weights.copy(),'coef':coef.copy(),'fun':result['primal objective'],'cost':cost,
											'result':result,'dZ_re':dZ_re.copy()
										})
				
				# check for convergence
				coef_delta = (coef - prev_coef)/prev_coef
				# If inductance not fitted, set inductance delta to zero (inductance goes to random number)
				if self.fit_inductance==False:# or part=='real':
					coef_delta[1] = 0
				# print(np.mean(np.abs(coef_delta)))
				if np.mean(np.abs(coef_delta)) < xtol:
					break
				elif iter==max_iter-1:
					warnings.warn(f'Hyperparametric solution did not converge within {max_iter} iterations')
				
				iter += 1
			
			self.distribution_fits[dist_name] = {'opt_result':result,'coef':coef.copy(),'weights':weights.copy(),'cost':cost.copy()}
		
		# Ordinary ridge fit		
		# --------------------------
		else:
			# create L2 penalty matrix
			lam_vectors = [np.ones(A_re.shape[1])*lambda_0]*3
			lam_matrices = [np.diag(lam_vec**0.5) for lam_vec in lam_vectors]
			L2_mat = np.zeros_like(L2_base[0])
			for L2b,lam_mat,frac in zip(L2_base,lam_matrices,reg_ord):
				if frac>0:
					L2_mat += frac*(lam_mat@L2b@lam_mat)
			
			result = self._convex_opt(part,WT_re,WT_im,WA_re,WA_im,L2_mat,L1_vec,nonneg)
			coef = np.array(list(result['x']))
			P = (WA_re.T@WA_re + WA_im.T@WA_im + L2_mat)
			q = (-WA_re.T@WT_re - WA_im.T@WT_im + L1_vec)
			cost = 0.5*coef.T@P@coef + q.T@coef
			
			self.distribution_fits[dist_name] = {'opt_result':result,'coef':coef.copy(),'cost':cost.copy()}
		
		# If fitted imag part only, optimize high-frequency resistance (offset)
		if part=='imag' and dist_info['dist_type']=='series':
			basis_coef = self.distribution_fits[dist_name]['coef'][2:]
			Zr_pred = A_re[:,2:]@basis_coef
			def res_fun(x):
				return Zr_pred + x - target_scaled.real
				
			result = least_squares(res_fun,x0=target_scaled.real[0])
			self.distribution_fits[dist_name]['coef'][0] = result['x'][0]
		# If fitted real part only, optimize inductance
		elif part=='real' and dist_info['dist_type']=='series' and self.fit_inductance:
			basis_coef = self.distribution_fits[dist_name]['coef'][2:]
			Zi_pred = A_im[:,2:]@basis_coef
			def res_fun(x):
				return Zi_pred + frequencies*2*np.pi*x - target_scaled.imag
				
			result = least_squares(res_fun,x0=1e-7)
			self.distribution_fits[dist_name]['coef'][1] = result['x'][0]
		
		# rescale coefficients if scaling applied to A or Z
		if scale_Z:
			self.distribution_fits[dist_name]['scaled_coef'] = self.distribution_fits[dist_name]['coef'].copy()
			# since the target for ridge_fit changes based on dist_type, rescaling coefficients always works in the same direction
			self.distribution_fits[dist_name]['coef'] = self._rescale_coef(self.distribution_fits[dist_name]['coef'],dist_info['dist_type']) 
		
		# rescale the inductance
		if dist_info['dist_type']=='series':
			self.distribution_fits[dist_name]['coef'][1] *= 1e-4
		
		# If inductance not fitted, set inductance to zero to avoid confusion (goes to random number)
		if dist_info['dist_type']=='series' and (self.fit_inductance==False or part=='real'):
			self.distribution_fits[dist_name]['coef'][1] = 0
			
		# pull R_inf and inductance out of coef
		if dist_info['dist_type']=='series':
			self.R_inf = self.distribution_fits[dist_name]['coef'][0]
			self.inductance = self.distribution_fits[dist_name]['coef'][1]
			self.distribution_fits[dist_name]['coef'] = self.distribution_fits[dist_name]['coef'][2:]
		else: 
			# ridge_fit cannot optimize R_inf and inductance for parallel distributions
			self.R_inf = 0
			self.inductance = 0
			
		# self._recalc_mat = False
		self.fit_type = 'ridge'
		
	def ridge_ReImCV(self,frequencies,Z,lambdas=np.logspace(-10,5,31),**kw):
		"""
		Perform re-im cross-validation to obtain optimal lambda_0 value
		
		Parameters:
		-----------
		frequencies: array
			Measured frequencies
		Z: complex array
			Measured (complex) impedance values
		lambdas: array
			Array of lambda_0 values to evaulate
		kw: various
			Keyword args to pass to ridge_fit
		"""
		
		recv = np.zeros_like(lambdas)
		imcv = np.zeros_like(lambdas)

		for i,lam in enumerate(lambdas):
			self.ridge_fit(frequencies,Z,part='real',lambda_0=lam,hyper_lambda=False,**kw)
			Zi_pred = np.imag(self.predict_Z(frequencies))
			
			self.ridge_fit(frequencies,Z,part='imag',lambda_0=lam,hyper_lambda=False,**kw)
			Zr_pred = np.real(self.predict_Z(frequencies))

			Zr_err = np.sum((Z.real - Zr_pred)**2)
			Zi_err = np.sum((Z.imag - Zi_pred)**2)
			recv[i] = Zr_err
			imcv[i] = Zi_err
		
		# get lambda_0 that minimizes total CV error
		totcv = recv + imcv
		min_lam = lambdas[np.argmin(totcv)]
		if min_lam==np.min(lambdas) or min_lam==np.max(lambdas):
			warnings.warn('Optimal lambda_0 {} determined by Re-Im CV is at the boundary of the evaluated range. Re-run with an expanded lambda_0 range to obtain an accurate estimate of the optimal lambda_0.'.format(lambda_0))
		
		# store DataFrame of results
		self.cv_result = pd.DataFrame(np.array([lambdas,recv,imcv,totcv]).T,columns=['lambda','recv','imcv','totcv'])
		
		return min_lam
		
	def map_fit(self,frequencies,Z,part='both',scale_Z=True,init_from_ridge=False,nonneg_drt=False,outliers=False,sigma_min=0.002,max_iter=50000,random_seed=1234,fitY=False,Yscale=1,SA=False,SASY=False):
		"""
		Obtain the maximum a posteriori estimate of the DRT (and all model parameters).
		
		Parameters:
		-----------
		frequencies: array
			Measured frequencies
		Z: complex array
			Measured (complex) impedance values
		part: str, optional (default: 'both')
			Which part of the impedance data to fit. Options: 'both', 'real', 'imag'
		scale_Z: bool, optional (default: True)
			
		"""
		# load stan model
		model,model_str = self._get_stan_model(nonneg_drt,outliers,fitY,SA)
		self.stan_model_name = model_str
		model_type = model_str.split('_')[0]
		if model_type=='Series-Parallel' and nonneg_drt==False:
			warnings.warn('For mixed series-parallel models, it is highly recommended to set nonnneg_drt=True')

		# perform scaling and weighting and get A and B matrices
		frequencies, Z_scaled, WZ_re,WZ_im,W_re,W_im, dist_mat = self._prep_matrices(frequencies,Z,part,weights=None,dZ=False,scale_Z=scale_Z,penalty='discrete',fit_type='map')
		
		# prepare data for stan model
		Z_scaled *= Yscale
		dat = self._prep_stan_data(frequencies,Z_scaled,part,model_type,dist_mat,outliers,sigma_min,mode='optimize',fitY=fitY,SA=SA,SASY=SASY)
		
		# get initial fit
		"""NEED TO UPDATE"""
		if init_from_ridge:
			init = self._get_init_from_ridge(frequencies,Z,hl_beta=2.5,lambda_0=1e-2,nonneg=nonneg,outliers=outliers)
			self._init_params = init()
		elif outliers:
			# initialize sigma_out near zero, everything else randomly
			iv = {'sigma_out_raw':np.zeros(2*len(Z)) + 0.1}
			def init():
				return iv
		else:
			init = 'random'
		
		# optimize posterior
		self._opt_result = model.optimizing(dat,iter=max_iter,seed=random_seed,init=init)
		
		# extract coefficients
		self.distribution_fits = {}
		if model_type in ['Series','Parallel']:
			dist_name = [k for k,v in self.distributions.items() if v['dist_type']==model_type.lower()][0]
			dist_type = self.distributions[dist_name]['dist_type']
			self.distribution_fits[dist_name] = {'coef':self._rescale_coef(self._opt_result['x'],dist_type)}
			self.distribution_fits[dist_name]['coef'] *= Yscale
			if fitY:
				self.R_inf = 0
				self.inductance = 0
			else:
				self.R_inf = self._rescale_coef(self._opt_result['Rinf'],'series')
				self.inductance = self._rescale_coef(self._opt_result['induc'],'series')
		elif model_type=='Series-Parallel':
			for dist_name,dist_info in self.distributions.items():
				if dist_info['dist_type']=='series':
					self.distribution_fits[dist_name] = {'coef':self._rescale_coef(self._opt_result['xs'],dist_info['dist_type'])}
				elif dist_info['dist_type']=='parallel':
					self.distribution_fits[dist_name] = {'coef':self._rescale_coef(self._opt_result['xp'],dist_info['dist_type'])}
			self.R_inf = self._rescale_coef(self._opt_result['Rinf'],'series')
			self.inductance = self._rescale_coef(self._opt_result['induc'],'series')
		elif model_type=='Series-2Parallel':
			for dist_name,dist_info in self.distributions.items():
				if dist_info['dist_type']=='series':
					self.distribution_fits[dist_name] = {'coef':self._rescale_coef(self._opt_result['xs'],dist_info['dist_type'])}
				elif dist_info['dist_type']=='parallel':
					order = dist_info['order']
					self.distribution_fits[dist_name] = {'coef':self._rescale_coef(self._opt_result[f'xp{order}'],dist_info['dist_type'])}
			self.R_inf = self._rescale_coef(self._opt_result['Rinf'],'series')
			self.inductance = self._rescale_coef(self._opt_result['induc'],'series')	
		
		elif model_type=='MultiDist':
			"""Placeholder"""
			for dist_name,dist_info in self.distributions.items():
				if dist_info['kernel']=='DRT':
					self.distribution_fits[dist_name] = {'coef':self._rescale_coef(self._opt_result['xs'],dist_info['dist_type'])}
				elif dist_info['kernel']=='DDT':
					self.distribution_fits[dist_name] = {'coef':self._rescale_coef(self._opt_result['xp'],dist_info['dist_type'])}
			self.R_inf = self._rescale_coef(self._opt_result['Rinf'],'series')
			self.inductance = self._rescale_coef(self._opt_result['induc'],'series')
		
		self.fit_type = 'map'
		self.sigma_min = sigma_min
		
	def bayes_fit(self,frequencies,Z,part='both',scale_Z=True,init_from_ridge=False,nonneg_drt=False,outliers=False,sigma_min=0.002,
			warmup=200,sample=200,chains=2,random_seed=1234,fitY=False,Yscale=1,SA=False,SASY=False):
			
		# load stan model
		model,model_str = self._get_stan_model(nonneg_drt,outliers,fitY,SA)
		self.stan_model_name = model_str
		model_type = model_str.split('_')[0]
		if model_type=='Series-Parallel' and nonneg_drt==False:
			warnings.warn('For mixed series-parallel models, it is highly recommended to set nonnneg_drt=True')
			
		# perform scaling and weighting and get A and B matrices
		frequencies, Z_scaled, WZ_re,WZ_im,W_re,W_im, dist_mat = self._prep_matrices(frequencies,Z,part,weights=None,dZ=False,scale_Z=scale_Z,penalty='discrete',fit_type='bayes')
		
		# prepare data for stan model
		Z_scaled *= Yscale
		dat = self._prep_stan_data(frequencies,Z_scaled,part,model_type,dist_mat,outliers,sigma_min,mode='sample',fitY=fitY,SA=SA,SASY=SASY)
		
		# get initial fit
		"""NEED TO UPDATE"""
		if init_from_ridge:
			init = self._get_init_from_ridge(frequencies,Z,hl_beta=2.5,lambda_0=1e-2,nonneg=nonneg,outliers=outliers)
			self._init_params = init()
		# elif outliers:
			# # initialize sigma_out near zero, everything else randomly
			# iv = {'sigma_out_raw':np.zeros(2*len(Z)) + 0.1}
			# def init():
				# return iv
		else:
			init = 'random'
		
		# sample from posterior
		self._sample_result = model.sampling(dat,warmup=warmup,iter=warmup+sample,chains=chains,seed=random_seed,init=init,
								  control={'adapt_delta':0.9,'adapt_t0':10})
								  
		# extract coefficients
		self.distribution_fits = {}
		if model_type in ['Series','Parallel']:
			dist_name = [k for k,v in self.distributions.items() if v['dist_type']==model_type.lower()][0]
			dist_type = self.distributions[dist_name]['dist_type']
			self.distribution_fits[dist_name] = {'coef':self._rescale_coef(np.mean(self._sample_result['x'],axis=0),dist_type)}
			self.distribution_fits[dist_name]['coef'] *= Yscale
			if fitY:
				self.R_inf = 0
				self.inductance = 0
			else:
				self.R_inf = self._rescale_coef(np.mean(self._sample_result['Rinf']),'series')
				self.inductance = self._rescale_coef(np.mean(self._sample_result['induc']),'series')
		elif model_type=='Series-Parallel':
			for dist_name,dist_info in self.distributions.items():
				if dist_info['dist_type']=='series':
					self.distribution_fits[dist_name] = {'coef':self._rescale_coef(np.mean(self._sample_result['xs'],axis=0),dist_info['dist_type'])}
				elif dist_info['dist_type']=='parallel':
					self.distribution_fits[dist_name] = {'coef':self._rescale_coef(np.mean(self._sample_result['xp'],axis=0),dist_info['dist_type'])}
			self.R_inf = self._rescale_coef(np.mean(self._sample_result['Rinf']),'series')
			self.inductance = self._rescale_coef(np.mean(self._sample_result['induc']),'series')
		elif model_type=='Series-2Parallel':
			for dist_name,dist_info in self.distributions.items():
				if dist_info['dist_type']=='series':
					self.distribution_fits[dist_name] = {'coef':self._rescale_coef(np.mean(self._sample_result['xs'],axis=0),dist_info['dist_type'])}
				elif dist_info['dist_type']=='parallel':
					order = dist_info['order']
					self.distribution_fits[dist_name] = {'coef':self._rescale_coef(np.mean(self._sample_result[f'xp{order}'],axis=0),dist_info['dist_type'])}
			self.R_inf = self._rescale_coef(np.mean(self._sample_result['Rinf']),'series')
			self.inductance = self._rescale_coef(np.mean(self._sample_result['induc']),'series')	
		
		self.fit_type = 'bayes'
		self.sigma_min = sigma_min
		
	def _get_stan_model(self,nonneg_drt,outliers,fitY,SA):
		"""Get the appropriate Stan model for the distributions. Called by map_fit and bayes_fit methods
		
		Parameters:
		-----------
		nonneg_drt: bool
			If True, constrain DRT to be non-negative. If False, allow negative DRT values
		outliers: bool
			If True, enable outlier detection. If False, do not include outlier error contribution in error model.
		"""
		num_series = len([name for name,info in self.distributions.items() if info['dist_type']=='series'])
		num_par = len([name for name,info in self.distributions.items() if info['dist_type']=='parallel'])
		
		if num_series==1 and num_par==0:		
			model_str = 'Series'
		elif num_series==0 and num_par==1:
			model_str = 'Parallel'
		elif num_series==1 and num_par==1:
			model_str = 'Series-Parallel'
		elif num_series==1 and num_par==2:
			model_str = 'Series-2Parallel'
		else:
			model_str = 'MultiDist'
			warnings.warn('The MultiDist model will handle an arbitrary number of series and/or parallel distributions, but the computational performance and accuracy are suboptimal. \
			Hard-coding your own model will most likely yield better results.')
		
		if nonneg_drt and num_series>=1:
			model_str += '_pos'
			
		if fitY:
			if num_par>=1 and num_series==0:
				model_str += '_fitY'
			else:
				raise ValueError('fitY=True is only valid for parallel distributions')
				
		if SA:
			model_str +='_SA'
			
		if outliers:
			model_str += '_outliers'
			
		model_str += '_StanModel.pkl'
		# print(model_str)
		model = load_pickle(os.path.join(script_dir,'stan_model_files',model_str))
		
		return model, model_str
		
			
	def _get_init_from_ridge(self,frequencies,Z,hl_beta,lambda_0,nonneg,outliers):
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
		outliers: bool
			If True, initialize sigma_out near zero
		"""
		# get initial parameter values from ridge fit
		self.ridge_fit(frequencies,Z,hyper_lambda=True,penalty='integral',reg_ord=2,scale_Z=True,dZ=True,
			   hl_beta=hl_beta,lambda_0=lambda_0,nonneg=nonneg)
		iv = {'beta':self.coef_[2:]/self._Z_scale}
		iv['tau_raw'] = np.zeros(self.A_re.shape[1]-2) + 1
		iv['hfr_raw'] = self.coef_[0]/(100*self._Z_scale)
		iv['induc'] = self.coef_[1]/self._Z_scale
		if iv['induc'] <= 0:
			iv['induc'] = 1e-10
		if outliers:
			iv['sigma_out_raw'] = np.zeros(2*len(Z)) + 0.1
		def init_func():
			return iv

		return init_func
		
	def _prep_stan_data(self,frequencies,Z,part,model_type,dist_mat,outliers,sigma_min,mode,fitY,SA,SASY):
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
		dist_mat: dict
			Distribution matrices dict
		outliers: bool
			Whether or not to enable outlier detection
		sigma_min: float
			Minimum value of error scale
		mode: str
			Solution mode. Options: 'sample', 'optimize'
		"""
		if model_type in ['Series','Parallel']:
			dist_name = [k for k,v in self.distributions.items() if v['dist_type']==model_type.lower()][0]
			matrices = dist_mat[dist_name]
			
			if part=='both':
				Z_stack = np.concatenate((Z.real,Z.imag))
				A_stack = np.concatenate((matrices['A_re'],matrices['A_im']))
			else:
				Z_stack = getattr(Z,part)
				A_stack = matrices['A_{}'.format(part[:2])]
				
			if mode=='sample':
				ups_alpha = 1
				ups_beta = 0.1
				L0 = matrices['L0']
				L1 = matrices['L1']
				L2 = 0.75*matrices['L2']
			
			elif mode=='optimize':
				ups_alpha = 0.05
				ups_beta = 0.1
				L0 = 1.5*0.24*matrices['L0']
				L1 = 1.5*0.16*matrices['L1']
				L2 = 1.5*0.08*matrices['L2'] # consider 0.08-->0.09
				
			dat = {'N':2*len(frequencies),
				   'freq':frequencies,
				   'K':A_stack.shape[1],
				   'A':A_stack,
				   'Z':Z_stack,
				   'N_tilde':2*len(frequencies),
				   'A_tilde':A_stack,
				   'freq_tilde': frequencies,
				   'L0':L0,
				   'L1':L1,
				   'L2':L2,
				   'sigma_min':sigma_min,
				   'ups_alpha':ups_alpha,
				   'ups_beta':ups_beta,
				  }
				  
			if SA:
				A_re = matrices['A_re']
				A_im = matrices['A_im']
				# using modulus of row sums for scale
				re_sum = np.sum(A_re,axis=1)
				im_sum = np.sum(A_im,axis=1)
				
				# mod = np.concatenate((re_sum,im_sum))
				# matrices['SA_re'] = np.diag(1/re_sum)@A_re
				# matrices['SA_im'] = np.diag(1/im_sum)@A_im
				
				Y = 1/Z
				Ymod = np.real((Y*Y.conjugate())**0.5)
				
				mod = Ymod#np.sqrt(re_sum**2 + im_sum**2)
				S = np.diag(1/mod)
				matrices['SA_re'] = S@A_re
				matrices['SA_im'] = S@A_im
				S_inv = np.diag(np.concatenate([mod,mod]))
				# print(mod)
				
				# using modulus of Y for scale
				# mod = Ymod
				# S = np.diag(1/mod)
				# S_inv = np.diag(np.concatenate((mod,mod)))
				# # SY = S@Y.real + 1j*(S@Y.imag)
				# matrices['SA_re'] = S@A_re
				# matrices['SA_im'] = S@A_im
				
				if part=='both':
					SA_stack = np.concatenate((matrices['SA_re'],matrices['SA_im']))
				else:
					SA_stack = matrices['SA_{}'.format(part[:2])]
			
				# SY = np.zeros_like(Y)
				# for A_str in ['A_re','A_im']:
					# A = matrices[A_str]
					# rowsum = np.sum(A,axis=1)
					# # S = np.diag(1/rowsum)
					# S = np.diag(1/Ymod)
					# matrices[A_str] = S@A
					# if A_str[2:]=='re':
						# SY += S@Y.real
					# else:
						# SY += 1j*(S@Y.imag)
				# Y = SY
				
				dat['SA'] = SA_stack
				dat['SA_tilde'] = SA_stack
				dat['S_inv'] = S_inv #np.vstack([np.hstack([S_inv,np.zeros_like(S_inv)]), np.hstack([np.zeros_like(S_inv),S_inv])])
				dat['S'] = dat['S_inv'] #S not actually used (FIX)
				  
			if fitY:
				# Z input will be ignored. Just need to add Y input
				Y = 1/Z
				Ymod = np.real((Y*Y.conjugate())**0.5)
				# # get Y range
				# Y_rng = Y.real[0] - Y.real[-1]
				# A_re_geomean = np.exp(np.sum(np.log(matrices['A_re']),axis=1)/matrices['A_re'].shape[1])
				# A_re_rng = matrices['A_re'].shape[1]*(A_re_geomean[0] - A_re_geomean[-1])
				# x_exp = 1
				
				# self._Y_scale = Y_rng/(A_re_rng*x_exp)
				# self._Y_scale=1
				# Y /= self._Y_scale
				
				
					
				if SASY:
					
					A_re = matrices['A_re']
					A_im = matrices['A_im']
					# using modulus of row sums for scale
					# re_sum = np.sum(A_re,axis=1)
					# im_sum = np.sum(A_im,axis=1)
					# mod = np.sqrt(re_sum**2 + im_sum**2)
					# using modulus of Y for scale
					mod = Ymod
					S = np.diag(1/mod)
					S_inv = np.diag(mod)
					SY = S@Y.real + 1j*(S@Y.imag)
					matrices['SA_re'] = S@A_re
					matrices['SA_im'] = S@A_im
					
					# using Ymod as scale
					# SY = np.zeros_like(Y)
					# for A_str in ['A_re','A_im']:
						# A = matrices[A_str]
						# S = np.diag(1/Ymod)
						# matrices['S'+A_str] = S@A
						# if A_str[2:]=='re':
							# SY += S@Y.real
						# else:
							# SY += 1j*(S@Y.imag)
					
					Y = SY
					
					if part=='both':
						A_stack = np.concatenate((matrices['SA_re'],matrices['SA_im']))
					else:
						A_stack = matrices['SA_{}'.format(part[:2])]
						
					dat['A'] = A_stack
				
				if part=='both':
					Y_stack = np.concatenate((Y.real,Y.imag))
					# A_stack = np.concatenate((matrices['A_re'],matrices['A_im']))
				else:
					Y_stack = getattr(Y,part)
					# A_stack = matrices['A_{}'.format(part[:2])]
					
				
				dat['Y'] = Y_stack
				  
			# if model_type=='Parallel':
				# # scaling factor for parallel coefficients
				# dat['x_scale'] = self._Z_scale**2
				# # print('x_scale:',dat['x_scale'])
				
			if outliers:
				if mode=='optimize':
					dat['so_invscale'] = 5
				elif mode=='sample':
					dat['so_invscale'] = 10
					
		elif model_type=='Series-Parallel':
			if len(self.distributions) > 2:
				raise ValueError('Too many distributions for Series-Parallel model')
			ser_name = [k for k,v in self.distributions.items() if v['dist_type']=='series'][0]
			par_name = [k for k,v in self.distributions.items() if v['dist_type']=='parallel'][0]
			ser_mat = dist_mat[ser_name]
			par_mat = dist_mat[par_name]
			
			if part=='both':
				Z_stack = np.concatenate((Z.real,Z.imag))
				As_stack = np.concatenate((ser_mat['A_re'],ser_mat['A_im']))
				Ap_stack = np.concatenate((par_mat['A_re'],par_mat['A_im']))
			elif part=='real':
				Z_stack = np.concatenate((Z.real,np.zeros_like(Z.imag)))
				As_stack = np.concatenate((ser_mat['A_re'],np.zeros_like(ser_mat['A_im'])))
				Ap_stack = np.concatenate((par_mat['A_re'],np.zeros_like(par_mat['A_im'])))
			elif part=='imag':
				Z_stack = np.concatenate((np.zeros_like(Z.real),Z.imag))
				As_stack = np.concatenate((np.zeros_like(ser_mat['A_re']),ser_mat['A_im']))
				Ap_stack = np.concatenate((np.zeros_like(par_mat['A_re']),par_mat['A_im']))
				

			if mode=='sample':
				ups_alpha = 1
				ups_beta = 0.1
				L0s = ser_mat['L0']
				L1s = ser_mat['L1']
				L2s = 0.75*ser_mat['L2']
				L0p = par_mat['L0']
				L1p = par_mat['L1']
				L2p = 0.75*par_mat['L2']
				x_sum_invscale = 1
				
			elif mode=='optimize':
				ups_alpha = 0.05
				ups_beta = 0.1
				L0s = 1.5*0.24*ser_mat['L0']
				L1s = 1.5*0.16*ser_mat['L1']
				L2s = 1.5*0.08*ser_mat['L2'] # consider 0.08-->0.1
				L0p = 1.5*0.36*par_mat['L0']
				L1p = 1.5*0.16*par_mat['L1']
				L2p = 1.5*0.08*par_mat['L2'] # consider 0.08-->0.1
				x_sum_invscale = 0.
				
			dat = {'N':2*len(frequencies),
				   'freq':frequencies,
				   'Ks':As_stack.shape[1],
				   'Kp':Ap_stack.shape[1],
				   'As':As_stack,
				   'Ap':Ap_stack,
				   'Z':Z_stack,
				   'N_tilde':2*len(frequencies),
				   'As_tilde':As_stack,
				   'Ap_tilde':Ap_stack,
				   'freq_tilde': frequencies,
				   'L0s':L0s,
				   'L1s':L1s,
				   'L2s':L2s,
				   'L0p':L0p,
				   'L1p':L1p,
				   'L2p':L2p,
				   'sigma_min':sigma_min,
				   'ups_alpha':ups_alpha,
				   'ups_beta':ups_beta,
				   'x_sum_invscale':x_sum_invscale,
				   'xp_scale':self.distributions[par_name].get('x_scale',1)#self._Z_scale**2,
				  }
				
			if outliers:
				if mode=='optimize':
					dat['so_invscale'] = 5
				elif mode=='sample':
					dat['so_invscale'] = 10
					
		elif model_type=='Series-2Parallel':
			ser_name = [k for k,v in self.distributions.items() if v['dist_type']=='series'][0]
			par_names = sorted([k for k,v in self.distributions.items() if v['dist_type']=='parallel'])
			par1_name = par_names[0]
			par2_name = par_names[1]
			# store order for parallel distributions
			self.distributions[par1_name]['order'] = 1
			self.distributions[par2_name]['order'] = 2
			ser_mat = dist_mat[ser_name]
			par1_mat = dist_mat[par1_name]
			par2_mat = dist_mat[par2_name]
			
			if part=='both':
				Z_stack = np.concatenate((Z.real,Z.imag))
				As_stack = np.concatenate((ser_mat['A_re'],ser_mat['A_im']))
				Ap1_stack = np.concatenate((par1_mat['A_re'],par1_mat['A_im']))
				Ap2_stack = np.concatenate((par2_mat['A_re'],par2_mat['A_im']))
			else:
				Z_stack = getattr(Z,part)
				As_stack = ser_mat['A_{}'.format(part[:2])]
				Ap1_stack = par1_mat['A_{}'.format(part[:2])]
				Ap2_stack = par2_mat['A_{}'.format(part[:2])]
				
			if mode=='sample':
				ups_alpha = 1
				ups_beta = 0.1
				L0s = ser_mat['L0']
				L1s = ser_mat['L1']
				L2s = 0.75*ser_mat['L2']
				L0p1 = par1_mat['L0']
				L1p1 = par1_mat['L1']
				L2p1 = 0.75*par1_mat['L2']
				L0p2 = par2_mat['L0']
				L1p2 = par2_mat['L1']
				L2p2 = 0.75*par2_mat['L2']
				x_sum_invscale = 0.1
				
			elif mode=='optimize':
				ups_alpha = 0.05
				ups_beta = 0.1
				L0s = 1.5*0.24*ser_mat['L0']
				L1s = 1.5*0.16*ser_mat['L1']
				L2s = 1.5*0.08*ser_mat['L2']
				L0p1 = 1.5*0.36*par1_mat['L0']
				L1p1 = 1.5*0.16*par1_mat['L1']
				L2p1 = 1.5*0.08*par1_mat['L2']
				L0p2 = 1.5*0.36*par2_mat['L0']
				L1p2 = 1.5*0.16*par2_mat['L1']
				L2p2 = 1.5*0.08*par2_mat['L2']
				x_sum_invscale = 0.
				
			dat = {'N':2*len(frequencies),
				   'freq':frequencies,
				   'Ks':As_stack.shape[1],
				   'Kp1':Ap1_stack.shape[1],
				   'Kp2':Ap2_stack.shape[1],
				   'As':As_stack,
				   'Ap1':Ap1_stack,
				   'Ap2':Ap2_stack,
				   'Z':Z_stack,
				   'N_tilde':2*len(frequencies),
				   'As_tilde':As_stack,
				   'Ap1_tilde':Ap1_stack,
				   'Ap2_tilde':Ap2_stack,
				   'freq_tilde': frequencies,
				   'L0s':L0s,
				   'L1s':L1s,
				   'L2s':L2s,
				   'L0p1':L0p1,
				   'L1p1':L1p1,
				   'L2p1':L2p1,
				   'L0p2':L0p2,
				   'L1p2':L1p2,
				   'L2p2':L2p2,
				   'sigma_min':sigma_min,
				   'ups_alpha':ups_alpha,
				   'ups_beta':ups_beta,
				   'x_sum_invscale':x_sum_invscale,
				   'xp1_scale':self.distributions[par1_name].get('x_scale',1),#self._Z_scale**2
				   'xp2_scale':self.distributions[par2_name].get('x_scale',1)#self._Z_scale**2
				  }
				
			if outliers:
				if mode=='optimize':
					dat['so_invscale'] = 5
				elif mode=='sample':
					dat['so_invscale'] = 10
					
		elif model_type=='MultiDist':
			"""placeholder"""
			drt_name = [k for k,v in self.distributions.items() if v['kernel']=='DRT'][0]
			ddt_name = [k for k,v in self.distributions.items() if v['kernel']=='DDT'][0]
			drt_mat = dist_mat[drt_name]
			ddt_mat = dist_mat[ddt_name]
			
			if part=='both':
				Z_stack = np.concatenate((Z.real,Z.imag))
				Ar_stack = np.concatenate((drt_mat['A_re'],drt_mat['A_im']))
				Ad_stack = np.concatenate((ddt_mat['A_re'],ddt_mat['A_im']))
			else:
				Z_stack = getattr(Z,part)
				Ar_stack = drt_mat['A_{}'.format(part[:2])]
				Ad_stack = ddt_mat['A_{}'.format(part[:2])]

			if mode=='sample':
				ups_alpha = 1
				ups_beta = 0.1
				L0r = drt_mat['L0']
				L1r = drt_mat['L1']
				L2r = 0.5*drt_mat['L2']
				L0d = ddt_mat['L0']
				L1d = ddt_mat['L1']
				L2d = 0.5*ddt_mat['L2']
				x_sum_invscale = 0
				
			elif mode=='optimize':
				ups_alpha = 0.05
				ups_beta = 0.1
				L0r = 1.5*0.24*drt_mat['L0']
				L1r = 1.5*0.16*drt_mat['L1']
				L2r = 1.5*0.08*drt_mat['L2']
				L0d = 1.5*0.24*ddt_mat['L0']
				L1d = 1.5*0.16*ddt_mat['L1']
				L2d = 1.5*0.08*ddt_mat['L2']
				x_sum_invscale = 0.
				
			dat = {'N':2*len(frequencies),
				   'freq':frequencies,
				   'Ms':1,
				   'Mp':1,
				   'Ks':[Ar_stack.shape[1]],
				   'Kp':[Ad_stack.shape[1]],
				   'As':Ar_stack,
				   'Ap':Ad_stack,
				   'Z':Z_stack,
				   # 'N_tilde':2*len(frequencies),
				   # 'Ar_tilde':Ar_stack,
				   # 'Ad_tilde':Ad_stack,
				   # 'freq_tilde': frequencies,
				   'L0s':L0r,
				   'L1s':L1r,
				   'L2s':L2r,
				   'L0p':L0d,
				   'L1p':L1d,
				   'L2p':L2d,
				   'sigma_min':sigma_min,
				   'ups_alpha':ups_alpha,
				   'ups_beta':ups_beta,
				   'x_sum_invscale':x_sum_invscale
				  }
				
			if outliers:
				if mode=='optimize':
					dat['so_invscale'] = 5
				elif mode=='sample':
					dat['so_invscale'] = 10
			  
		return dat
			  
	def _hyper_lambda_discrete(self,L,coef,dist_type,hl_beta=2.5,lambda_0=1):
		Lx2 = (L@coef)**2
		#lam = np.ones(self.A_re.shape[1]) #*lambda_0
		lam = 1/(Lx2/(hl_beta-1) + 1/lambda_0)
		if dist_type=='series':
			# add ones for R_ohmic and inductance
			lam = np.hstack(([1,1],lam))
		return lam
		
	def _hyper_lambda_fbeta(self,L,coef,dist_type,hl_fbeta,lambda_0):
		Lx2 = (L@coef)**2
		Lxmax = np.max(Lx2)
		# lam = np.ones(self.A_re.shape[1]) #*lambda_0
		lam = lambda_0/(Lx2/(Lxmax*hl_fbeta) + 1)
		if dist_type=='series':
			# add ones for R_ohmic and inductance
			lam = np.hstack(([1,1],lam))
		return lam
		
	# def _grad_lambda_discrete(self,frequencies,coef,lam_vec,reg_ord,beta=2.5,lambda_0=1):
		# L = construct_L(frequencies,tau=self.tau,basis=self.basis,epsilon=self.epsilon,order=reg_ord)
		# Lx2 = (L@coef)**2
		# zeta = (beta-1)/lambda_0
		# grad = Lx2 + zeta - (beta-1)/lam_vec[2:]
		# return grad
		
	def _hyper_lambda_integral(self,M,coef,lam_mat,hl_beta=2.5,lambda_0=1):
		X = np.diag(coef)
		xlm = X@lam_mat@M@X
		xlm = xlm - np.diag(np.diagonal(xlm))
		C = np.sum(xlm,axis=0)
		
		a = hl_beta/2
		b = 0.5*(2*a-2)/lambda_0
		d = coef**2*np.diagonal(M) + 2*b
		lam = (C**2 - np.sign(C)*C*np.sqrt(4*d*(2*a-2) + C**2) + 2*d*(2*a-2))/(2*d**2)
		return lam
	
	def _hyper_b(self,lam,a,sb):
		K = self.A_re.shape[1]-2
		b = 0.25*(np.sqrt(16*a*K*sb**2 + 4*sb**4*np.sum(lam)**2) -2*np.sum(lam)*sb**2) # b ~ normal(0,sb)
		# b = 0.25*(np.sqrt(16*a*sb**2 + 4*sb**4*lam**2) -2*lam*sb**2) # b_k ~ normal(0,sb))
		return b
	
	def _hyper_a(self,lam,b,alpha_a,beta_a):
		# a is a vector
		# def obj_fun(ak,bk,lk,alpha_a,beta_a):
			# # return -2*ak*np.log(bk*lk) + 2*loggamma(ak) + 4*np.log(ak-2) + 2*((ak-2)*sa)**(-2) # 1/(ak-2) ~ normal(0,sa)
			# return -2*ak*np.log(bk*lk) + 2*loggamma(ak) + 2*beta_a*(ak-1) - 2*(alpha_a-1)*np.log(ak-1) # ak-1 ~ gamma(alpha_a,beta_a)
			
		# a = np.zeros_like(lam)
		# a = [minimize_scalar(obj_fun,method='bounded',bounds=(1,5),args=(bk,lk,alpha_a,beta_a))['x'] for bk,lk in zip(b,lam)]
		
		# a is a scalar
		def obj_fun(a,b,lam,alpha_a,beta_a):
			return -2*a*np.sum(np.log(b*lam)) + 2*loggamma(a) + 2*beta_a*(a-1) - 2*(alpha_a-1)*np.log(a-1) # a-1 ~ gamma(alpha_a,beta_a)
			
		a = minimize_scalar(obj_fun,method='bounded',bounds=(1,5),args=(b,lam,alpha_a,beta_a))['x']
		
		return a
		
	def _hyper_weights(self,coef,A_re,A_im,Z,hw_beta,wbar):
		"""Calculate hyper weights
		
		Parameters:
		-----------
		coef: array
			Current coefficients
		Z: array
			Measured complex impedance
		hw_beta: float
			Beta hyperparameter for weight hyperprior
		wbar: array
			Expected weights
		"""
		# calculate zeta
		zeta_re = hw_beta/np.real(wbar)
		zeta_im = hw_beta/np.imag(wbar)
		
		# calculate residuals
		Z_pred = A_re@coef + 1j*A_im@coef
		resid = Z - Z_pred
		r_re = np.real(resid)
		r_im = np.imag(resid)
		 # calculate MAP weights
		w_re = (np.real(wbar) - 1/zeta_re)/(r_re**2/zeta_re + 1)
		w_im = (np.imag(wbar) - 1/zeta_im)/(r_im**2/zeta_im + 1)
		
		# print(resid[8:13])
		# print(w_im[8:13])
		# print(wbar[8:13])
		
		return w_re + 1j*w_im
		
	
	def _convex_opt(self,part,WZ_re,WZ_im,WA_re,WA_im,L2_mat,L1_vec,nonneg):
		if part=='both':
			P = cvxopt.matrix((WA_re.T@WA_re + WA_im.T@WA_im + L2_mat).T)
			q = cvxopt.matrix((-WA_re.T@WZ_re - WA_im.T@WZ_im + L1_vec).T)
		elif part=='real':
			P = cvxopt.matrix((WA_re.T@WA_re + L2_mat).T)
			q = cvxopt.matrix((-WA_re.T@WZ_re + L1_vec).T)
		else:
			P = cvxopt.matrix((WA_im.T@WA_im + L2_mat).T)
			q = cvxopt.matrix((-WA_im.T@WZ_im + L1_vec).T)
		
		G = cvxopt.matrix(-np.eye(WA_re.shape[1]))
		if nonneg:
			# coefficients must be >= 0
			h = cvxopt.matrix(np.zeros(WA_re.shape[1]))
		else:
			# coefficients can be positive or negative
			h = 10*np.ones(WA_re.shape[1])
			# HFR and inductance must still be nonnegative
			h[0:2] = 0
			# print(h)
			h = cvxopt.matrix(h)
			# print('neg')
			
		return cvxopt.solvers.qp(P,q,G,h)
		
	def _prep_matrices(self,frequencies,Z,part,weights,dZ,scale_Z,penalty,fit_type):
		if len(frequencies)!=len(Z):
			raise ValueError("Length of frequencies and Z must be equal")
			
		if type(Z)!=np.ndarray:
			Z = np.array(Z)
		
		if type(frequencies)!=np.ndarray:
			frequencies = np.array(frequencies)
			
		# sort by descending frequency
		sort_idx = np.argsort(frequencies)[::-1]
		frequencies = frequencies[sort_idx]
		Z = Z[sort_idx]
		
		# check if we need to recalculate A matrices
		freq_subset = False
		if np.min(rel_round(self.f_train,10)==rel_round(frequencies,10))==False:
			# if frequencies are a subset of f_train, we can use submatrices of the existing A matrices
			# instead of recalculating
			if np.min([rel_round(f,10) in rel_round(self.f_train,10) for f in frequencies])==True:
				freq_subset = True
			else:
				# if frequencies have changed and are not a subset of f_train, must recalculate
				self.f_train = frequencies
				self._recalc_mat = True
		else:
			self.f_train = frequencies
		# print(self._recalc_mat)
		# print(freq_subset)
	
		# scale Z
		if scale_Z:
			Z = self._scale_Z(Z,fit_type)
			if type(weights) in (list,np.ndarray):
				weights = np.array(weights)/self._Z_scale
		else:
			self._Z_scale = 1
			
		# create weight matrices
		weights = self._format_weights(frequencies,Z,weights,part)
		W_re = np.diag(np.real(weights))
		W_im = np.diag(np.imag(weights))
		
		# print(self._recalc_mat)
		
		# set up matrices for each distribution
		dist_mat = {} # transient dict to hold matrices for fit, which may be scaled
		for name, info in self.distributions.items():
			temp_dist = deepcopy(self.distributions)
			# set tau and epsilon
			if info.get('basis_freq',self.basis_freq) is None:
				# by default, use 10 ppd for tau spacing regardless of input frequency spacing
				tmin = np.ceil(np.log10(1/(2*np.pi*np.max(frequencies))))
				tmax = np.floor(np.log10(1/(2*np.pi*np.min(frequencies))))
				num_decades = tmax - tmin
				tau = np.logspace(tmin,tmax, int(10*num_decades + 1))
			else:
				tau = 1/(2*np.pi*info.get('basis_freq',self.basis_freq))
			temp_dist[name]['tau'] = tau
			
			if info.get('epsilon',self.epsilon) is None:
				# if neither dist-specific nor class-level epsilon is specified
				dlnt = np.mean(np.diff(np.log(tau)))
				temp_dist[name]['epsilon'] = (1/dlnt)
			elif info.get('epsilon',None) is None:
				# if dist-specific epsilon not specified, but class-level epsilon is present
				temp_dist[name]['epsilon'] = self.epsilon
			epsilon = temp_dist[name].get('epsilon',self.epsilon)
			
			# update distributions without overwriting self._recalc_mat
			recalc_mat = self._recalc_mat
			self.distributions = temp_dist
			self._recalc_mat = recalc_mat
			
			# create A matrices
			if self._recalc_mat==False:
				if freq_subset:
					# frequencies is a subset of f_train - no need to recalc
					# print('freq in f_train')
					f_index = np.array([np.where(rel_round(self.f_train,10)==rel_round(f,10))[0][0] for f in frequencies])
					A_re = self.distribution_matrices[name]['A_re'][f_index,:].copy()
					A_im = self.distribution_matrices[name]['A_im'][f_index,:].copy()
				else:
					A_re = self.distribution_matrices[name]['A_re'].copy()
					A_im = self.distribution_matrices[name]['A_im'].copy()
					
				if dZ and info['kernel']=='DRT':
					if 'B' in self.distribution_matrices[name].keys():
						B = self.distribution_matrices[name]['B'].copy()
					else:
						# Z_re differentiation matrix (B@coef gives approx dZ'/dlnt at each basis tau)
						tau_diff = np.mean(np.diff(np.log(tau)))
						B_start = np.exp(np.log(tau[0]) - tau_diff/2)
						B_end = np.exp(np.log(tau[-1]) + tau_diff/2)
						B_tau = np.logspace(np.log10(B_start),np.log10(B_end),len(tau)+1)
						B_pre = construct_A(1/(2*np.pi*B_tau),'real',tau=tau,basis=self.basis,epsilon=epsilon,
											kernel=info['kernel'],dist_type=info['dist_type'],symmetry=info.get('symmetry',''),
											bc=info.get('bc',''),ct=info.get('ct',False),k_ct=info.get('k_ct',None)
											)
						B = B_pre[1:,:] - B_pre[:-1,:]
						self.distribution_matrices[name]['B'] = B
				else:
					B = None
			
			if self._recalc_mat or 'A_re' not in self.distribution_matrices[name].keys() or 'A_im' not in self.distribution_matrices[name].keys():
				self.distribution_matrices[name]['A_re'] = construct_A(frequencies,'real',tau=tau,basis=self.basis,fit_inductance=self.fit_inductance,epsilon=epsilon,
														kernel=info['kernel'],dist_type=info['dist_type'],symmetry=info.get('symmetry',''),bc=info.get('bc',''),
														ct=info.get('ct',False),k_ct=info.get('k_ct',None)
														)
				self.distribution_matrices[name]['A_im'] = construct_A(frequencies,'imag',tau=tau,basis=self.basis,fit_inductance=self.fit_inductance,epsilon=epsilon,
														kernel=info['kernel'],dist_type=info['dist_type'],symmetry=info.get('symmetry',''),bc=info.get('bc',''),
														ct=info.get('ct',False),k_ct=info.get('k_ct',None)
														)
				A_re = self.distribution_matrices[name]['A_re'].copy()
				A_im = self.distribution_matrices[name]['A_im'].copy()
				
				if dZ and info['kernel']=='DRT':
					# Z_re differentiation matrix (B@coef gives approx dZ'/dlnt at each basis tau)
					tau_diff = np.mean(np.diff(np.log(tau)))
					B_start = np.exp(np.log(tau[0]) - tau_diff/2)
					B_end = np.exp(np.log(tau[-1]) + tau_diff/2)
					B_tau = np.logspace(np.log10(B_start),np.log10(B_end),len(tau)+1)
					B_pre = construct_A(1/(2*np.pi*B_tau),'real',tau=tau,basis=self.basis,epsilon=epsilon,
										kernel=info['kernel'],dist_type=info['dist_type'],symmetry=info.get('symmetry',''),
										bc=info.get('bc',''),ct=info.get('ct',False),k_ct=info.get('k_ct',None)
										)
					B = B_pre[1:,:] - B_pre[:-1,:]
					self.distribution_matrices[name]['B'] = B
				else:
					B = None

			
			
			# apply weights to A
			WA_re = W_re@A_re
			WA_im = W_im@A_im
			
			dist_mat[name] = {}
			
			# calculate L or M matrices
			if penalty=='integral':
				dist_mat[name]['M0'] = construct_M(1/(2*np.pi*tau),basis=self.basis,order=0,epsilon=epsilon)
				dist_mat[name]['M1'] = construct_M(1/(2*np.pi*tau),basis=self.basis,order=1,epsilon=epsilon)
				dist_mat[name]['M2'] = construct_M(1/(2*np.pi*tau),basis=self.basis,order=2,epsilon=epsilon)
				
			elif penalty=='discrete':
				dist_mat[name]['L0'] = construct_L(1/(2*np.pi*tau),tau=tau,basis=self.basis,epsilon=epsilon,order=0)
				dist_mat[name]['L1'] = construct_L(1/(2*np.pi*tau),tau=tau,basis=self.basis,epsilon=epsilon,order=1)
				dist_mat[name]['L2'] = construct_L(1/(2*np.pi*tau),tau=tau,basis=self.basis,epsilon=epsilon,order=2)
				
			elif penalty=='cholesky':
				M0 = construct_M(1/(2*np.pi*tau),basis=self.basis,order=0,epsilon=epsilon)
				M1 = construct_M(1/(2*np.pi*tau),basis=self.basis,order=1,epsilon=epsilon)
				M2 = construct_M(1/(2*np.pi*tau),basis=self.basis,order=2,epsilon=epsilon)
				
				dist_mat[name]['L0'] = cholesky(M0) # scipy cholesky gives upper triangular by default, such that M = L.T@L. Then x.T@M@x = x.T@L.T@L@x = ||L@x||^2
				dist_mat[name]['L1'] = cholesky(M1)
				dist_mat[name]['L2'] = cholesky(M2)
					
				dist_mat[name]['M0'] = M0
				dist_mat[name]['M1'] = M1
				dist_mat[name]['M2'] = M2
				
			# add L and M matrices to self.distribution_matrices
			self.distribution_matrices[name].update(dist_mat[name])
			
			# add matrices to transient dist_mat
			dist_mat[name].update({'A_re':A_re,'A_im':A_im,'WA_re':WA_re,'WA_im':WA_im,'B':B})
				
		# apply weights to Z
		WZ_re = W_re@Z.real
		WZ_im = W_im@Z.imag
		
		self._recalc_mat = False
		
		return frequencies, Z, WZ_re,WZ_im,W_re,W_im, dist_mat
			
			
	def _format_weights(self,frequencies,Z,weights,part):
		"""
		Format real and imaginary weight vectors
		Parameters:
		-----------
		weights : str or array (default: None)
			Weights for fit. Standard weighting schemes can be specified by passing 'unity', 'modulus', or 'proportional'. 
			Custom weights can be passed as an array. If the array elements are real, the weights are applied to both the real and imaginary parts of the impedance.
			If the array elements are complex, the real parts are used to weight the real impedance, and the imaginary parts are used to weight the imaginary impedance.
			If None, all points are weighted equally.
		part : str (default:'both')
			Which part of impedance is being fitted. Options: 'both', 'real', or 'imag'
		"""
		if weights is None or weights=='unity':
			weights = np.ones_like(frequencies)*(1+1j)
		elif type(weights)==str:
			if weights=='modulus':
				weights = (1+1j)/np.sqrt(np.real(Z*Z.conjugate()))
			elif weights=='Orazem':
				weights = (1+1j)/(np.abs(Z.real) + np.abs(Z.imag))
			elif weights=='proportional':
				weights = 1/np.abs(Z.real) + 1j/np.abs(Z.imag)
			elif weights=='prop_adj':
				Zmod = np.real(Z*Z.conjugate())
				weights = 1/(np.abs(Z.real) + np.percentile(Zmod,25)) + 1j/(np.abs(Z.imag) + np.percentile(Zmod,25))
			else:
				raise ValueError(f"Invalid weights argument {weights}. String options are 'unity', 'modulus', 'proportional', and 'prop_adj'")
		elif type(weights) in (float,int):
			# assign constant value
			weights = np.ones_like(frequencies)*(1+1j)*weights
		elif type(weights)==complex:
			# assign constant value
			weights = np.ones_like(frequencies)*weights
		elif len(weights)!=len(frequencies):
			raise ValueError("Weights array must match length of data")
			
		if part=='both':
			if np.min(np.isreal(weights))==True:
				# if weights are real, apply to both real and imag parts
				weights = weights + 1j*weights
			else:
				# if weights are complex, leave them
				pass
		elif part=='real':
			weights = np.real(weights) + 1j*np.ones_like(frequencies)
		elif part=='imag':
			if np.min(np.isreal(weights))==True:
				# if weights are real, apply to imag
				weights = np.ones_like(frequencies) + 1j*weights
			else:
				# if weights are complex, leave them
				pass
		else:
			raise ValueError(f"Invalid part {part}. Options are 'both', 'real', or 'imag'")
			
		return weights
		
	# def _scale_A(self):
		# """Scale A matrices such that, when considered together, the modulus of each column has unit variance"""
		# A_com = self.A_re[:,2:] + 1j*self.A_im[:,2:]
		# A_mod = (A_com*A_com.conjugate())**0.5
		# self._A_scale = np.ones(self.A_re.shape[1])
		# # don't scale high-frequency resistance or inductance
		# self._A_scale[2:] = np.std(A_mod,axis=0)
		# self.A_re_scaled = self.A_re.copy()
		# self.A_im_scaled = self.A_im.copy()
		# self.A_re_scaled /= self._A_scale
		# self.A_im_scaled /= self._A_scale
		# if hasattr(self,'B'):
			# self.B_scaled = self.B/self._A_scale
		
	def _scale_Z(self,Z,fit_type):
		Zmod = (Z*Z.conjugate())**0.5
		
		# adjust the Z scale for pure parallel distributions
		num_series = len([name for name,info in self.distributions.items() if info['dist_type']=='series'])
		num_par = len([name for name,info in self.distributions.items() if info['dist_type']=='parallel'])
		if num_par==1 and num_series==0 and fit_type!='ridge':
			Y = 1/Z
			Ymod = (Y*Y.conjugate())**0.5
			dist_name = [name for name,info in self.distributions.items() if info['dist_type']=='parallel'][0]
			info = self.distributions[dist_name]
			if info['kernel']=='DDT' and info['symmetry']=='planar':
				# scale Z such that the scaled admittance has the desired std
				if info['bc']=='transmissive':
					if fit_type=='map':
						Ystar_std = 14
					elif fit_type=='bayes':
						Ystar_std = 14 #70
				elif info['bc']=='blocking':
					if fit_type=='map':
						Ystar_std = 2.4
					elif fit_type=='bayes':
						Ystar_std = 2.4
				self._Z_scale = Ystar_std*np.sqrt(len(Z)/81)/np.std(Ymod)
				
			else:
				self._Z_scale = np.std(Zmod)/np.sqrt(len(Z)/81)
		else:
			# scale by sqrt(n) as suggested by Ishwaran and Rao (doi: 10.1214/009053604000001147)
			# hyperparameters were selected based on spectra with 81 data points - therefore, scale relative to N=81
			self._Z_scale = np.std(Zmod)/np.sqrt(len(Z)/81)
				
		return Z/self._Z_scale
		
	def _rescale_coef(self,coef,dist_type):
		if dist_type=='series':
			rs_coef = coef*self._Z_scale
		elif dist_type=='parallel':
			rs_coef = coef/self._Z_scale
		return rs_coef
		
	def coef_percentile(self,distribution_name,percentile):
		if self.fit_type=='bayes':
			dist_type = self.distributions[distribution_name]['dist_type']
			model_type = self.stan_model_name.split('_')[0]
			if model_type in ['Series','Parallel']:
				coef = np.percentile(self._sample_result['x'],percentile,axis=0)
			elif model_type=='Series-Parallel':
				if self.distributions[distribution_name]['dist_type']=='series':
					coef = np.percentile(self._sample_result['xs'],percentile,axis=0)
				elif self.distributions[distribution_name]['dist_type']=='parallel':
					coef = np.percentile(self._sample_result['xp'],percentile,axis=0)
			elif model_type=='Series-2Parallel':
				if self.distributions[distribution_name]['dist_type']=='series':
					coef = np.percentile(self._sample_result['xs'],percentile,axis=0)
				elif self.distributions[distribution_name]['dist_type']=='parallel':
					order = self.distributions[distribution_name]['order']
					coef = np.percentile(self._sample_result[f'xp{order}'],percentile,axis=0)
			# rescale coef
			coef = self._rescale_coef(coef,dist_type)
		else:
			raise ValueError('Percentile prediction is only available for bayes_fit')
			
		return coef
		
	def predict_Z(self,frequencies,distributions=None,include_offsets=True,percentile=None):
		"""percentile currently does nothing. This should be updated to use Z_hat from sample_result for bayes_fit only"""
		
		if distributions is not None:
			if type(distributions)==str:
				distributions = [distributions]
		else:
			distributions = [k for k in self.distribution_fits.keys()]
			
		if percentile is not None:
			if self.fit_type!='bayes':
				raise ValueError('Percentile prediction is only available for bayes_fit results')
			elif len(distributions)!=len(self.distributions) or include_offsets==False:
				raise ValueError('If percentile is specified, distributions and include_offsets must be left at their default values.')
				
			if np.min(rel_round(self.f_train,10)==rel_round(frequencies,10))==True:
				# If frequencies are same as f_train, use Z_hat from sample_result
				Z_pred = np.percentile(self._sample_result['Z_hat'],percentile,axis=0)*self._Z_scale
				Z_pred = Z_pred[:len(frequencies)] + 1j*Z_pred[len(frequencies):]
			else:
				# If frequencies are different from f_train, need to calculate
				raise Exception('Percentile prediction not yet developed for frequencies different from training frequencies')
		else:
			# get A matrices for prediction
			pred_mat = {}
			for name in distributions:
				pred_mat[name] = {}
			# check if we need to recalculate A matrices
			freq_subset = False
			if np.min(rel_round(self.f_train,10)==rel_round(frequencies,10))==False:					
				# if frequencies are a subset of f_train, we can use submatrices of the existing A matrices
				# instead of calculating new A matrices
				if np.min([rel_round(f,10) in rel_round(self.f_train,10) for f in frequencies])==True:
					# print('freq in f_train')
					f_index = np.array([np.where(rel_round(self.f_train,10)==rel_round(f,10))[0][0] for f in frequencies])
					
					for name in distributions:
						mat = self.distribution_matrices[name]
						pred_mat[name]['A_re'] = mat['A_re'][f_index,:].copy()
						pred_mat[name]['A_im'] = mat['A_im'][f_index,:].copy()
				# otherwise, we need to calculate A matrices
				else:
					for name in distributions:
						tau = self.distributions[name]['tau']
						epsilon = self.distributions[name]['epsilon']
						pred_mat[name]['A_re'] = construct_A(frequencies,'real',tau=tau,basis=self.basis,fit_inductance=self.fit_inductance,epsilon=epsilon)
						pred_mat[name]['A_im'] = construct_A(frequencies,'imag',tau=tau,basis=self.basis,fit_inductance=self.fit_inductance,epsilon=epsilon)
			else:
				# frequencies are same as f_train. Use existing matrices
				for name in distributions:
					mat = self.distribution_matrices[name]
					pred_mat[name]['A_re'] = mat['A_re'].copy()
					pred_mat[name]['A_im'] = mat['A_im'].copy()
				
			# construct Z_pred
			Z_pred = np.zeros(len(frequencies),dtype=complex)
			
			# add contributions from distributions to Z_pred
			for name, mat in pred_mat.items():
				dist_type = self.distributions[name]['dist_type']
				coef = self.distribution_fits[name]['coef']
				
				if dist_type=='series':
					Z_re = mat['A_re']@coef
					Z_im = mat['A_im']@coef
					Z_pred += Z_re + 1j*Z_im
				elif dist_type=='parallel':
					Y_re = mat['A_re']@coef
					Y_im = mat['A_im']@coef
					Z_pred += 1/(Y_re + 1j*Y_im)
		
			# add contributions from R_inf and inductance
			if include_offsets:
				Z_pred += self.R_inf
				Z_pred += 1j*2*np.pi*frequencies*self.inductance
			
		return Z_pred
		
	def predict_Rp(self,distributions=None):
		"""Predict polarization resistance
		
		Parameters:
		-----------
		distributions: str or list (default: None)
			Distribution name or list of distribution names for which to sum Rp contributions. 
			If None, include all distributions
		"""
		if distributions is not None:
			if type(distributions)==str:
				distributions = [distributions]
		else:
			distributions = [k for k in self.distribution_fits.keys()]
			
		if len(distributions) > 1:
			Z_range = self.predict_Z(np.array([1e20,1e-20]),distributions=distributions)
			Rp = np.real(Z_range[1] - Z_range[0])
		else:
			info = self.distributions[distributions[0]]
			if info['kernel']=='DRT':
				# Rp due to DRT is area under DRT
				Rp = np.sum(self.distribution_fits[distributions[0]]['coef'])*np.pi**0.5/info['epsilon']
			else:
				# just calculate Z at very high and very low frequencies and take the difference in Z'
				# could do calcs using coefficients, but this is fast and accurate enough for now (and should work for any arbitray distribution)
				Z_range = self.predict_Z(np.array([1e20,1e-20]),distributions=distributions)
				Rp = np.real(Z_range[1] - Z_range[0])
				# raise Exception("Haven't implemented Rp calc for non-DRT distributions yet")
					
				
		return Rp
		
	def predict_sigma(self,frequencies,percentile=None):
		if percentile is not None and self.fit_type!='bayes':
			raise ValueError('Percentile prediction is only available for bayes_fit')
			
		if np.min(rel_round(self.f_train,10)==rel_round(frequencies,10))==True:
			# if frequencies are training frequencies, just use sigma_tot output
			if self.fit_type=='bayes':
				if percentile is not None:
					sigma_tot = np.percentile(self._sample_result['sigma_tot'],percentile,axis=0)
				else:
					sigma_tot = np.mean(self._sample_result['sigma_tot'],axis=0)
			elif self.fit_type=='map':
				sigma_tot = self._opt_result['sigma_tot']
			else:
				raise ValueError('Error scale prediction only available for bayes_fit and map_fit')
				
			sigma_re = sigma_tot[:len(self.f_train)]*self._Z_scale
			sigma_im = sigma_tot[len(self.f_train):]*self._Z_scale
		else:
			# if frequencies are not training frequencies, calculate from parameters
			# this doesn't match sigma_tot perfectly
			if self.fit_type=='bayes':
				if percentile is not None:
					sigma_res = np.percentile(self._sample_result['sigma_res'],percentile)
					alpha_prop = np.percentile(self._sample_result['alpha_prop'],percentile)
					alpha_re = np.percentile(self._sample_result['alpha_re'],percentile)
					alpha_im = np.percentile(self._sample_result['alpha_im'],percentile)
					try:
						sigma_out = np.percentile(self._sample_result['sigma_out'],percentile,axis=0)
					except ValueError:
						sigma_out = np.zeros(2*len(self.f_train))
				else:
					sigma_res = np.mean(self._sample_result['sigma_res'])
					alpha_prop = np.mean(self._sample_result['alpha_prop'])
					alpha_re = np.mean(self._sample_result['alpha_re'])
					alpha_im = np.mean(self._sample_result['alpha_im'])
					try:
						sigma_out = np.mean(self._sample_result['sigma_out'],axis=0)
					except ValueError:
						sigma_out = np.zeros(2*len(self.f_train))
			elif self.fit_type=='map':
				sigma_res = self._opt_result['sigma_res']
				alpha_prop = self._opt_result['alpha_prop']
				alpha_re = self._opt_result['alpha_re']
				alpha_im = self._opt_result['alpha_im']
				try:
					sigma_out = self._opt_result['sigma_out']
				except KeyError:
					sigma_out = np.zeros(2*len(self.f_train))
			else:
				raise ValueError('Error scale prediction only available for bayes_fit and map_fit')
				
			# try:
			sigma_min = self.sigma_min
			# except AttributeError:
				# # legacy - for models run before _sigma_min was set by fit methods
				# sigma_min = 0 #***placeholder
				
			Z_pred = self.predict_Z(frequencies,percentile)
			
			# assume none of predicted points are outliers - just get baseline sigma_out contribution
			sigma_base = np.sqrt(sigma_res**2 + np.min(sigma_out)**2 + sigma_min**2)*self._Z_scale
			
			sigma_re = np.sqrt(sigma_base**2 + (alpha_prop*Z_pred.real)**2 + (alpha_re*Z_pred.real)**2 + (alpha_im*Z_pred.imag)**2)
			sigma_im = np.sqrt(sigma_base**2 + (alpha_prop*Z_pred.imag)**2 + (alpha_re*Z_pred.real)**2 + (alpha_im*Z_pred.imag)**2)
				
		return sigma_re,sigma_im	
		
	def score(self,frequencies=None,Z=None,metric='chi_sq',weights=None,part='both'):
		weights = self._format_weights(frequencies,Z,weights,part)
		Z_pred = self.predict(frequencies)
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
			score = r2_score(Z,Z_pred,weights=weights)
		else:
			raise ValueError(f"Invalid metric {metric}. Options are 'chi_sq', 'r2'")
			
		return score	
		
	def predict_distribution(self,name=None,eval_tau=None,percentile=None):
		"""Get fitted distribution(s)
		***Need to reimplement percentile***
		"""
		"Need to reimplement percentile"
		# if percentile is not None:
			# coef = self.coef_percentile(percentile)
		# else:
			# coef = self.coef_
			
		if name is not None:
			# return the specified distribution
			if percentile is not None:
				coef = self.coef_percentile(name,percentile)
			else:
				coef = self.distribution_fits[name]['coef']
				
			epsilon = self.distributions[name]['epsilon']
			basis_tau = self.distributions[name]['tau']
			if eval_tau is None:
				eval_tau = self.distributions[name]['tau']
			phi = get_basis_func(self.basis)
			bases = np.array([phi(np.log(eval_tau/t_m),epsilon) for t_m in basis_tau]).T
			F = bases@coef
			
			return F
		else:
			out = {}
			# return all distributions in a dict
			for name in self.distributions.keys():
				if percentile is not None:
					coef = self.coef_percentile(name,percentile)
				else:
					coef = self.distribution_fits[name]['coef']
				epsilon = self.distributions[name]['epsilon']
				basis_tau = self.distributions[name]['tau']
				if eval_tau is None:
					eval_tau = self.distributions[name]['tau']
				phi = get_basis_func(self.basis)
				bases = np.array([phi(np.log(eval_tau/t_m),epsilon) for t_m in basis_tau]).T
				F = bases@coef
				out[name] = F
			
			return out
		
	# getters and setters to control matrix recalculation
	def get_basis_freq(self):
		return self._basis_freq
	
	def set_basis_freq(self,basis_freq):
		self._basis_freq = basis_freq
		self._recalc_mat = True
		
	basis_freq = property(get_basis_freq,set_basis_freq)
		
	def get_basis(self):
		return self._basis
	
	def set_basis(self,basis):
		self._basis = basis
		self._recalc_mat = True
		
	basis = property(get_basis,set_basis)
	
	def get_epsilon(self):
		return self._epsilon
		
	def set_epsilon(self,epsilon):
		self._epsilon = epsilon
		self._recalc_mat = True
	
	epsilon = property(get_epsilon,set_epsilon)
	
	def get_fit_inductance(self):
		return self._fit_inductance_
		
	def set_fit_inductance(self,fit_inductance):
		self._fit_inductance_ = fit_inductance
		if self._recalc_mat==False and hasattr(self,'A_im'):
			self.A_im[:,1] = -2*np.pi*self.f_train
			
	fit_inductance = property(get_fit_inductance,set_fit_inductance)
	
def rel_round(x,precision):
	"""Round to relative precision
	
	Parameters
	----------
	x : array
		array of numbers to round
	precision : int
		number of digits to keep
	"""
	# add 1e-30 for safety in case of zeros in x
	x_scale = np.floor(np.log10(np.array(x)+1e-30))
	digits = (precision - x_scale).astype(int)
	# print(digits)
	if type(x) in (list,np.ndarray):
		x_round = np.array([round(xi,di) for xi,di in zip(x,digits)])
	else:
		x_round = round(x,digits)
	return x_round
		

def get_basis_func(basis):
	"Generate basis function"
	# y = ln (tau/tau_m)
	if basis=='gaussian':
		def phi(y,epsilon):
			return np.exp(-(epsilon*y)**2)
	elif basis=='Cole-Cole':
		def phi(y,epsilon):
			return (1/(2*np.pi))*np.sin((1-epsilon)*np.pi)/(np.cosh(epsilon*y)-np.cos((1-epsilon)*np.pi))
	elif basis=='Zic':
		def phi(y,epsilon):
			# epsilon unused, included only for compatibility
			return 2*np.exp(y)/(1+np.exp(2*y))
	else:
		raise ValueError(f'Invalid basis {basis}. Options are gaussian')
	return phi
	
def get_A_func(part,basis='gaussian',kernel='DRT',dist_type='series',symmetry='planar',bc=None,ct=False,k_ct=None):
	"""
	Create integrand function for A matrix
	
	Parameters:
	-----------
	part : string
		Part of impedance for which to generate function ('real' or 'imag')
	basis : string, optional (default: 'gaussian')
		basis function
	"""
	phi = get_basis_func(basis)
	
	# y = ln (tau/tau_m)
	if ct==True and k_ct is None:
		raise ValueError('k_ct must be supplied if ct==True')
		
	if kernel=='DRT':
		if dist_type=='series':
			if part=='real':
				def func(y,w_n,t_m,epsilon=1):
					return phi(y,epsilon)/(1+np.exp(2*(y+np.log(w_n*t_m)))) 
			elif part=='imag':
				def func(y,w_n,t_m,epsilon=1):
					return -phi(y,epsilon)*np.exp(y)*w_n*t_m/(1+np.exp(2*(y+np.log(w_n*t_m))))
		else:
			raise ValueError('dist_type for DRT kernel must be series')
	
	elif kernel=='DDT':
		"""Need to add Gerischer-type equivalents"""
		# first define diffusion impedance, Z_D
		if bc=='blocking':
			if symmetry=='planar':
				if ct:
					def Z_D(y,w_n,t_m):
						# coth(x) = 1/tanh(x)
						x = np.sqrt(t_m*np.exp(y)*(k_ct + 1j*w_n))
						return 1/(np.tanh(x)*x)
				else:
					def Z_D(y,w_n,t_m):
						# coth(x) = 1/tanh(x)
						x = np.sqrt(1j*w_n*t_m*np.exp(y))
						return 1/(np.tanh(x)*x)
			# elif symmetry=='cylindrical': # not sure how I_0 and I_1 are defined for cylindrical in Song and Bazant (2018)
			elif symmetry=='spherical':
				if ct:
					def Z_D(y,w_n,t_m):
						x = np.sqrt(t_m*np.exp(y)*(k_ct + 1j*w_n))
						return np.tanh(x)/(x - np.tanh(x))
				else:
					def Z_D(y,w_n,t_m):
						x = np.sqrt(1j*w_n*t_m*np.exp(y))
						return np.tanh(x)/(x - np.tanh(x))
			else:
				raise ValueError(f'Invalid symmetry {symmetry}. Options are planar or spherical for bc=blocking')
		elif bc=='transmissive':
			if symmetry=='planar':
				if ct:
					def Z_D(y,w_n,t_m):
						x = np.sqrt(t_m*np.exp(y)*(k_ct + 1j*w_n))
						return np.tanh(x)/x
				else:
					def Z_D(y,w_n,t_m):
						x = np.sqrt(1j*w_n*t_m*np.exp(y))
						return np.tanh(x)/x
			else:
				raise ValueError(f'Invalid symmetry {symmetry}. Symmetry must be planar for bc=transmissive')
			
			
		# then choose whether to integrate Z_D or Y_D
		if dist_type=='parallel':
			if part=='real':
				def func(y,w_n,t_m,epsilon=1):
					return phi(y,epsilon)*np.real(1/Z_D(y,w_n,t_m))
			elif part=='imag':
				def func(y,w_n,t_m,epsilon=1):
					return phi(y,epsilon)*np.imag(1/Z_D(y,w_n,t_m))
		elif dist_type=='series':
			if part=='real':
				def func(y,w_n,t_m,epsilon=1):
					return phi(y,epsilon)*np.real(Z_D(y,w_n,t_m))
			elif part=='imag':
				def func(y,w_n,t_m,epsilon=1):
					return phi(y,epsilon)*np.imag(Z_D(y,w_n,t_m))
		else: raise ValueError(f'Invalid dist_type {dist_type}. Options are series and parallel')
		
	else:
		raise ValueError(f'Invalid kernel {kernel}. Options are DRT and DDT')

	
	return func			
				
				
def is_loguniform(frequencies):
	"Check if frequencies are uniformly log-distributed"
	fdiff = np.diff(np.log(frequencies))
	if np.std(fdiff)/np.mean(fdiff) <= 0.01:
		return True
	else: 
		return False

def construct_A(frequencies,part,tau=None,basis='gaussian',fit_inductance=False,epsilon=1,
				kernel='DRT',dist_type='series',symmetry='planar',bc=None,ct=False,k_ct=None,
				integrate_method='trapz'):
	"""
	Construct A matrix for DRT. A' and A'' matrices transform DRT coefficients to real 
	and imaginary impedance values, respectively, for given frequencies.
	
	Parameters:
	-----------
	frequencies : array
		Frequencies
	part : string
		Part of impedance for which to construct A matrix ('real' or 'imag')
	tau : array, optional (default: None)
		Time constants at which to center basis functions. If None, use time constants 
		corresponding to frequencies, i.e. tau=1/(2*pi*frequencies)
	basis : string, optional (default: 'gaussian')
		Basis function to use to approximate DRT
	fit_inductance : bool, optional (default: False)
		Whether to include inductive term
	epsilon : float, optional (default: 1)
		Shape parameter for chosen basis function
	"""
	"need to handle basis_kw"
	
	omega = frequencies*2*np.pi
	
	# check if tau is inverse of omega
	if tau is None:
		tau = 1/omega
		tau_eq_omega = True
	elif len(tau)==len(omega):
		if np.min(rel_round(tau,10)==rel_round(1/omega,10)):
			tau_eq_omega = True
		else:
			tau_eq_omega = False
	else:
		tau_eq_omega = False
	
	# check if omega is subset of inverse tau
	# find index where first frequency matches tau
	match = rel_round(1/omega[0],10)==rel_round(tau,10)
	if np.sum(match)==1:
		start_idx = np.where(match==True)[0][0]
		# if tau vector starting at start_idx matches omega, omega is a subset of tau
		if np.min(rel_round(tau[start_idx:start_idx + len(omega)],10)==rel_round(1/omega,10)):
			freq_subset = True
		else:
			freq_subset = False
	elif np.sum(match)==0:
		# if no tau corresponds to first omega, omega is not a subset of tau
		freq_subset = False
	else:
		# if more than one match, must be duplicates in tau
		raise Exception('Repeated tau values')
		
	if freq_subset==False:
		# check if tau is subset of inverse omega
		# find index where first frequency matches tau
		match = rel_round(1/omega,10)==rel_round(tau[0],10)
		if np.sum(match)==1:
			start_idx = np.where(match==True)[0][0]
			# if omega vector starting at start_idx matches tau, tau is a subset of omega
			if np.min(rel_round(omega[start_idx:start_idx + len(tau)],10)==rel_round(1/tau,10)):
				freq_subset = True
			else:
				freq_subset = False
		elif np.sum(match)==0:
			# if no omega corresponds to first tau, tau is not a subset of omega
			freq_subset = False
		else:
			# if more than one match, must be duplicates in omega
			raise Exception('Repeated omega values')
		
	# Determine if A is a Toeplitz matrix
	# Note that when there is simultaneous charge transfer, the matrix is never a Toeplitz matrix
	# because the integrand can no longer be written in terms of w_n*t_m only
	if is_loguniform(frequencies) and ct==False:
		if tau_eq_omega:
			is_toeplitz = True
		elif freq_subset and is_loguniform(tau):
			is_toeplitz = True
		else:
			is_toeplitz = False
	else:
		is_toeplitz = False
		
	# print(part,'is toeplitz',is_toeplitz)
	# print(part,'freq subset',freq_subset)
	
	# get function to integrate
	func = get_A_func(part,basis,kernel,dist_type,symmetry,bc,ct,k_ct)
		
	if is_toeplitz: #is_loguniform(frequencies) and tau_eq_omega:
		# only need to calculate 1st row and column
		w_0 = omega[0]
		t_0 = tau[0]
		if part=='real':
			if basis=='Zic':
				quad_limits = (-100,100)
			elif kernel=='DDT':
				quad_limits = (-20,20)
			else:
				quad_limits = (-np.inf,np.inf)
		elif part=='imag':
			# scipy.integrate.quad is unstable for imag func with infinite limits
			quad_limits = (-20,20)
#		  elif part=='imag':
#			  y = np.arange(-5,5,0.1)
#			  c = [np.trapz(func(y,w_n,t_0),x=y) for w_n in omega]
#			  r = [np.trapz(func(y,w_0,t_m),x=y) for t_m in 1/omega]

		if integrate_method=='quad':
			c = [quad(func,quad_limits[0],quad_limits[1],args=(w_n,t_0,epsilon),epsabs=1e-4)[0] for w_n in omega]
			r = [quad(func,quad_limits[0],quad_limits[1],args=(w_0,t_m,epsilon),epsabs=1e-4)[0] for t_m in tau]
		elif integrate_method=='trapz':
			y = np.linspace(-20,20,1000)
			c = [np.trapz(func(y,w_n,t_0,epsilon),x=y) for w_n in omega]
			r = [np.trapz(func(y,w_0,t_m,epsilon),x=y) for t_m in tau]
		if r[0]!=c[0]:
			print(r[0],c[0])
			raise Exception('First entries of first row and column are not equal')
		A = toeplitz(c,r)
	else:
		# need to calculate all entries
		if part=='real':
			if basis=='Zic':
				quad_limits = (-20,20)
			elif kernel=='DDT':
				quad_limits = (-20,20)
			else:
				quad_limits = (-np.inf,np.inf)
		elif part=='imag':
			# scipy.integrate.quad is unstable for imag func with infinite limits
			quad_limits = (-20,20)
			
		A = np.empty((len(frequencies),len(tau)))
		for n,w_n in enumerate(omega):
			if integrate_method=='quad':
				A[n,:] = [quad(func,quad_limits[0],quad_limits[1],args=(w_n,t_m,epsilon),epsabs=1e-4)[0] for t_m in tau]
			elif integrate_method=='trapz':
				y = np.linspace(-20,20,1000)
				A[n,:] = [np.trapz(func(y,w_n,t_m,epsilon),x=y) for t_m in tau]
			
	return A
	

def construct_L(frequencies,tau=None,basis='gaussian',epsilon=1,order=1):
	"Differentiation matrix. L@coef gives derivative of DRT"
	omega = 2*np.pi*frequencies
	if tau is None:
		# if no time constants given, assume collocated with frequencies
		tau = 1/omega
		
	L = np.zeros((len(omega),len(tau)))
	
	if basis=='gaussian':
		if type(order)==list:
			f0,f1,f2 = order
			def dphi_dy(y,epsilon):
				"Mixed derivative of Gaussian RBF"
				return f0*np.exp(-(epsilon*y)**2) + f1*(-2*epsilon**2*y*np.exp(-(epsilon*y)**2)) \
					+ f2*(-2*epsilon**2 + 4*epsilon**4*y**2)*np.exp(-(epsilon*y)**2)
		elif order==0:
			def dphi_dy(y,epsilon):
				"Gaussian RBF"
				return np.exp(-(epsilon*y)**2)
		elif order==1:
			def dphi_dy(y,epsilon):
				"Derivative of Gaussian RBF"
				return -2*epsilon**2*y*np.exp(-(epsilon*y)**2)
		elif order==2:
			def dphi_dy(y,epsilon):
				"2nd derivative of Gaussian RBF"
				return (-2*epsilon**2 + 4*epsilon**4*y**2)*np.exp(-(epsilon*y)**2)
		elif order==3:
			def dphi_dy(y,epsilon):
				"3rd derivative of Gaussian RBF"
				return (12*epsilon**4*y - 8*epsilon**6*y**3)*np.exp(-(epsilon*y)**2)
		elif order > 0 and order < 1:
			f0 = 1-order
			f1 = order
			def dphi_dy(y,epsilon):
				"Mixed derivative of Gaussian RBF"
				return f0*np.exp(-(epsilon*y)**2) + f1*(-2*epsilon**2*y*np.exp(-(epsilon*y)**2)) 
		elif order > 1 and order < 2:
			f1 = 2-order
			f2 = order-1
			def dphi_dy(y,epsilon):
				"Mixed derivative of Gaussian RBF"
				return f1*(-2*epsilon**2*y*np.exp(-(epsilon*y)**2)) + f2*(-2*epsilon**2 + 4*epsilon**4*y**2)*np.exp(-(epsilon*y)**2)
		else:
			raise ValueError('Order must be between 0 and 3')
	elif basis=='Zic':
		if order==0:
			dphi_dy = get_basis_func(basis)
			
	for n in range(len(omega)):
			L[n,:] = [dphi_dy(np.log(1/(omega[n]*t_m)),epsilon) for t_m in tau]
		
	return L

def get_M_func(basis='gaussian',order=1):
	"""
	Create function for M matrix
	
	Parameters:
	-----------
	basis : string, optional (default: 'gaussian')
		Basis function used to approximate DRT
	order : int, optional (default: 1)
		Order of DRT derivative for ridge penalty
	"""
	
	if order==0:
		if basis=='gaussian':
			def func(w_n,t_m,epsilon):
				a = epsilon*np.log(1/(w_n*t_m))
				return (np.pi/2)**0.5*epsilon**(-1)*np.exp(-(a**2/2))
		else:
			raise ValueError(f'Invalid basis {basis}')
	elif order==1:
		if basis=='gaussian':
			def func(w_n,t_m,epsilon):
				a = epsilon*np.log(1/(w_n*t_m))
				return -(np.pi/2)**0.5*epsilon*(-1+a**2)*np.exp(-(a**2/2))
		else:
			raise ValueError(f'Invalid basis {basis}')
	elif order==2:
		if basis=='gaussian':
			def func(w_n,t_m,epsilon):
				a = epsilon*np.log(1/(w_n*t_m))
				return (np.pi/2)**0.5*epsilon**3*(3-6*a**2+a**4)*np.exp(-(a**2/2))
		else:
			raise ValueError(f'Invalid basis {basis}')
	else:
		 raise ValueError(f'Invalid order {order}')
	return func

def construct_M(frequencies,basis='gaussian',order=1,epsilon=1):
	"""
	Construct M matrix for calculation of DRT ridge penalty. 
	x^T@M@x gives integral of squared derivative of DRT over all ln(tau)
	
	Parameters:
	-----------
	frequencies : array
		Frequencies at which basis functions are centered
	basis : string, optional (default: 'gaussian')
		Basis function used to approximate DRT
	order : int, optional (default: 1)
		Order of derivative to penalize
	epsilon : float, optional (default: 1)
		Shape parameter for chosen basis function
	"""
	omega = frequencies*2*np.pi
	
	if type(order)==list:
		f0,f1,f2 = order
		func0 = get_M_func(basis,0)
		func1 = get_M_func(basis,1)
		func2 = get_M_func(basis,2)
		def func(w_n,t_m,epsilon):
			return f0*func0(w_n,t_m,epsilon) + f1*func1(w_n,t_m,epsilon) + f2*func2(w_n,t_m,epsilon)
			
	else:
		func = get_M_func(basis,order)
		
	
	if is_loguniform(frequencies):
		# only need to calculate 1st column (symmetric toeplitz matrix) IF gaussian - may not apply to other basis functions
		# w_0 = omega[0]
		t_0 = 1/omega[0]
		
		c = [func(w_n,t_0,epsilon) for w_n in omega]
		# r = [quad(func,limits[0],limits[1],args=(w_0,t_m,epsilon),epsabs=1e-4)[0] for t_m in 1/omega]
		# if r[0]!=c[0]:
			# raise Exception('First entries of first row and column are not equal')
		M = toeplitz(c)
	else:
		# need to calculate all entries
		M = np.empty((len(frequencies),len(frequencies)))
		for n,w_n in enumerate(omega):
			M[n,:] = [func(w_n,t_m,epsilon) for t_m in 1/omega]
	
	return M
	
def r2_score(y,y_hat,weights=None):
	"""
	Calculate r^2 score
	
	Parameters:
	-----------
		y: y values
		y_hat: predicted y values
		weights: sample weights
	"""
	if weights is None:
		ss_res = np.sum((y_hat-y)**2)#np.var(y_hat-y)
		ss_tot = np.sum((y - np.mean(y))**2) #np.var(y)
	else:
		ss_res = np.sum(weights*(y_hat-y)**2)
		ss_tot = np.sum(weights*(y-np.average(y,weights=weights))**2)
	return 1-(ss_res/ss_tot)
	
