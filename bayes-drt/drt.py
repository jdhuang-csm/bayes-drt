import numpy as np
from scipy.linalg import toeplitz, cholesky
from scipy.integrate import quad
from scipy.special import loggamma
from scipy.optimize import least_squares, minimize, minimize_scalar
from sklearn.metrics import r2_score
import cvxopt
import pystan as stan
import warnings
from stan_models import load_pickle
import os

cvxopt.solvers.options['show_progress'] = False

script_dir = os.path.dirname(os.path.realpath(__file__))

class DRT():
	def __init__(self,basis_freq=None,basis='gaussian',epsilon=None,fit_inductance=True):
		self._recalc_mat = True
		self.set_basis_freq(basis_freq)
		self.set_basis(basis)
		self.set_epsilon(epsilon) # inverse length scale of RBF. mu in paper
		self.set_fit_inductance(fit_inductance)
		self.f_train = [0]
		self._Z_scale = 1.0
		
	def ridge_fit(self,frequencies,Z,part='both',hyper_lambda=True,hyper_penalty='integral',hyper_method='analytic',
		beta=2.5,lambda_0=1,reg_ord=2,L1_penalty=0,x0=None,weights=None,xtol=1e-3,max_iter=20,scale_A=False,scale_Z=True,
		dZ=True,dZ_power=0.5,hyper_a=False,alpha_a=2,beta_a=2,hyper_b=False,sb=1,nonneg=False):
		"""
		weights : str or array (default: None)
			Weights for fit. Standard weighting schemes can be specified by passing 'modulus' or 'proportional'. 
			Custom weights can be passed as an array. If the array elements are real, the weights are applied to both the real and imaginary parts of the impedance.
			If the array elements are complex, the real parts are used to weight the real impedance, and the imaginary parts are used to weight the imaginary impedance.
			If None, all points are weighted equally.
		"""
		# checks
		if hyper_penalty in ('discrete','cholesky'):
			if beta <= 1:
				raise ValueError('beta must be greater than 1 for hyper_penalty ''cholesky'' and ''discrete''')
		elif hyper_penalty=='integral':
			if beta <= 2:
				raise ValueError('beta must be greater than 2 for hyper_penalty ''integral''')
		else:
			raise ValueError(f'Invalid hyper_penalty argument {hyper_penalty}. Options are ''integral'', ''discrete'', and ''cholesky''')
		
		# perform scaling and weighting and get A and B matrices
		frequencies, Z, A_re,A_im, WA_re,WA_im,WZ_re,WZ_im, B = self._prep_matrices(frequencies,Z,part,weights,dZ,scale_A,scale_Z)
		
		# create L or M matrix
		if type(reg_ord)==int:
			ord_idx = reg_ord
			reg_ord = np.zeros(3)
			reg_ord[ord_idx] = 1
		if hyper_penalty=='integral':
			self.M0 = construct_M(1/(2*np.pi*self.tau),basis=self.basis,order=0,epsilon=self.epsilon)
			self.M1 = construct_M(1/(2*np.pi*self.tau),basis=self.basis,order=1,epsilon=self.epsilon)
			self.M2 = construct_M(1/(2*np.pi*self.tau),basis=self.basis,order=2,epsilon=self.epsilon)
			L2_base = [self.M0,self.M1,self.M2]
		elif hyper_penalty=='discrete':
			self.L0 = construct_L(1/(2*np.pi*self.tau),tau=self.tau,basis=self.basis,epsilon=self.epsilon,order=0)
			self.L1 = construct_L(1/(2*np.pi*self.tau),tau=self.tau,basis=self.basis,epsilon=self.epsilon,order=1)
			self.L2 = construct_L(1/(2*np.pi*self.tau),tau=self.tau,basis=self.basis,epsilon=self.epsilon,order=2)
			L2_base = [L.T@L for L in [self.L0,self.L1,self.L2]]
		elif hyper_penalty=='cholesky':
			self.M0 = construct_M(1/(2*np.pi*self.tau),basis=self.basis,order=0,epsilon=self.epsilon)
			self.M1 = construct_M(1/(2*np.pi*self.tau),basis=self.basis,order=1,epsilon=self.epsilon)
			self.M2 = construct_M(1/(2*np.pi*self.tau),basis=self.basis,order=2,epsilon=self.epsilon)
			self.L0 = np.hstack((np.zeros((A_re.shape[1]-2,2)), cholesky(self.M0[2:,2:])))
			self.L1 = np.hstack((np.zeros((A_re.shape[1]-2,2)), cholesky(self.M1[2:,2:])))
			self.L2 = np.hstack((np.zeros((A_re.shape[1]-2,2)), cholesky(self.M2[2:,2:])))
			L2_base = [L.T@L for L in [self.L0,self.L1,self.L2]]
			
		# create L1 penalty vector
		L1_vec = np.ones(len(self.tau)+2)*np.pi**0.5/self.epsilon*L1_penalty
		L1_vec[0:2] = 0 # inductor and high-frequency resistor are not penalized
		
		if type(beta) in (float,int,np.float64):
			beta = np.array([beta]*3)
		else:
			beta = np.array(beta)
		a_list = beta/2
		if hyper_penalty=='integral':
			b_list = 0.5*(2*a_list-2)/lambda_0
			hyper_as = np.array([np.ones(A_re.shape[1])*a for a in a_list])
			hyper_bs = np.array([np.ones(A_re.shape[1])*b for b in b_list])
			hyper_lambda0s = (2*hyper_as-2)/(2*hyper_bs)
		else:
			b_list = 0.5*(2*a_list-1)/lambda_0
			hyper_as = np.array([np.ones(A_re.shape[1])*a for a in a_list])
			hyper_bs = np.array([np.ones(A_re.shape[1])*b for b in b_list])
			hyper_lambda0s = (2*hyper_as-1)/(2*hyper_bs)
		hyper_betas = 2*hyper_as
		# print(hyper_lambda0s)
		# print(hyper_betas)
		
		if type(alpha_a) in (float,int):
			alpha_a = 3*[alpha_a]
		if type(beta_a) in (float,int):
			beta_a = 3*[beta_a]
		if type(sb) in (float,int):
			sb = 3*[sb]
		
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
			lam_step = np.zeros_like(lam_vectors[0])
			
			dZ_re = np.ones(A_re.shape[1])
			
			L2_mat = np.zeros_like(L2_base[0])
			for L2b,lam_mat,frac in zip(L2_base,lam_matrices,reg_ord):
				if frac>0:
					L2_mat += frac*(lam_mat@L2b@lam_mat)
			P = WA_re.T@WA_re + WA_im.T@WA_im + L2_mat
			q = (-WA_re.T@WZ_re + WA_im.T@WZ_im + L1_vec)
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
				prev_step = lam_step.copy()
				
				if dZ and iter > 0:
					dZ_raw = B@prev_coef
					# scale by tau spacing to get dZ'/dlnt
					dlnt = np.mean(np.diff(np.log(self.tau)))
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
					for n,(lam_vec,frac,aa,ba) in enumerate(zip(lam_vectors,reg_ord,alpha_a,beta_a)):
						# if hyper_b_converged[n]:
						if frac > 0:
							hyper_as[n] = np.ones(len(hyper_bs[n]))*self._hyper_a(lam_vec,hyper_bs[n],aa,ba)
							# print(hyper_as[n])
							hyper_lambda0s[n] = (2*hyper_as[n]-2)/hyper_bs[n]
							hyper_betas[n] = 2*hyper_as[n]
				
				
						
				# solve for lambda
				if hyper_penalty in ('discrete','cholesky'):
					if hyper_method=='analytic':
						for n,(Ln,frac,hlam0,hbeta) in enumerate(zip([self.L0,self.L1,self.L2],reg_ord,hyper_lambda0s,hyper_betas)):
							if frac > 0:
								lam_vec = self._hyper_lambda_discrete(Ln,prev_coef/dZ_re,beta=hbeta[2:],lambda_0=hlam0[2:])
								# damp the lambda step to avoid oscillation
								lam_step = lam_vec - prev_lam[n]
								lam_vectors[n] = prev_lam[n] + lam_step
					# elif hyper_method=='grad':
						
						# lam_grad = self._grad_lambda_discrete(frequencies,prev_coef,prev_lam,reg_ord,beta=beta,lambda_0=lambda_0)
						# # print(lam_grad)
						# # damp the lambda step to avoid oscillation
						# lam_vec[2:] = prev_lam[2:] - lam_grad*0.01
						# lam_vec[lam_vec < 0] = 1e-6
						# # print(lam_vec)
						
					elif hyper_method=='lm':
						zeta = (beta-1)/lambda_0
						def jac(x,L,coef):
							# off-diagonal terms are zero
							diag = (L@coef)**2 + zeta - (beta-1)/x
							return np.diag(diag)
							
						def fun(x,L,coef):
							return ((L@coef)**2 + zeta)*x - (beta-1)*np.log(x)
						
						for n,(Ln,frac) in enumerate(zip([self.L0,self.L1,self.L2],reg_ord)):
							if frac > 0:
								result = least_squares(fun,prev_lam[2:],jac=jac,method='lm',xtol=lambda_0*1e-3,args=([Ln,coef]),max_nfev=100)
								lam_vectors[n][2:] = result['x']
						
					# elif hyper_method=='ssa':
						# w = 0.01
						# v0 = 0.005 # lower value for bernoulli
						# # k = 5 # shape
						# # theta = 1/50 # scale
						# k = beta/2
						# zeta = (beta-1)/lambda_0
						# theta = 2/zeta
					
						# Lx2 = (self.L@coef)**2
						# lam = np.ones(A_re.shape[1])*lambda_0
						# # find the approximate MAP for the spike and the slab separately
						# # zero coefs (large lambdas)
						# lam0 = 1/(Lx2/(beta-1) + v0/lambda_0)
						# # nonzero coefs (small lambdas)
						# lam1 = 1/(Lx2/(beta-1) + 1/lambda_0)
						
						# # get the cost at each MAP lambda
						# def lamcost(lam):
							# return lam*Lx2 - np.log(lam) - 2*np.log(lam**(k-1)*((1-w)*v0**k*np.exp(-lam*v0/theta) +w*np.exp(-lam/theta)))
						# cost0 = lamcost(lam0)
						# cost1 = lamcost(lam1)
						# # compare the costs and choose the lambda with the lower cost
						# costs = np.vstack((cost0,cost1))
						# min_idx = np.argmin(costs,axis=0)
						# lams = np.vstack((lam0,lam1))
						# lam_vec[2:] = np.diag(lams[min_idx])
						
				elif hyper_penalty=='integral':
					if hyper_method=='analytic':
						# print('iter',iter)
						# D = np.diag(dZ_re**(-1))
						for n,(L2b,lam_mat,frac,hlam0,hbeta) in enumerate(zip(L2_base,lam_matrices,reg_ord,hyper_lambda0s,hyper_betas)):
							if frac > 0:
								# lam_vectors[n] = self._hyper_lambda_integral(L2b,prev_coef,D@lam_mat,beta=beta,lambda_0=lambda_0)
								if n==0:
									factor = 100
								elif n==1:
									factor=10
								else:
									factor=1
								lv = self._hyper_lambda_integral(L2b,factor*prev_coef/dZ_re,lam_mat,beta=hbeta,lambda_0=hlam0)
								# handle numerical instabilities that may arise for large lambda_0 and small beta
								lv[lv<=0] = 1e-15
								lam_vectors[n] = lv
								# print(n,lam_vectors[n])
					# elif hyper_method=='lm':
						# zeta = (beta-1)/lambda_0
						# def jac(x,coef):
							# X = np.diag(coef)
							# lam_mat = np.diag(x**0.5)
							# xlm = X@lam_mat@self.M@X
							# xlm = xlm - np.diag(np.diagonal(xlm))
							# C = np.sum(xlm,axis=0)
							# a = coef**2*np.diagonal(self.M) + zeta
							# grad = 1e5*(C/(x**0.5) + a - (beta-1)/x)
							# return grad
							
						# def fun(x,coef):
							# lam_mat = np.diag(x**0.5)
							# return 1e5*coef.T@lam_mat@self.M@lam_mat@coef + np.sum(zeta*x - (beta-1)*np.log(x))
							
						# result = minimize(fun,prev_lam,jac=None,args=(coef),method='L-BFGS-B', tol=1e-20, \
										# bounds=[(0,np.inf) for i in range(len(prev_lam))],\
										# options={'maxfun':10000,'gtol':1e-20}) # L-BFGS-B
										# # options= {'maxiter':100,'gtol':1e-10}) # BFGS
										# # options={'maxiter':10000,'xatol':1e-10,'fatol':1e-10}) # Nelder-Mead
										# # options = {'maxiter':100,'gtol':1e-10})
						# # print(result)
						# lam_vec[2:] = result['x'][2:]
					
				
						
				
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
				result = self._convex_opt(part,WZ_re,WZ_im,WA_re,WA_im,L2_mat,L1_vec,nonneg)
				coef = np.array(list(result['x']))
				
				P = (WA_re.T@WA_re + WA_im.T@WA_im + L2_mat)
				q = (-WA_re.T@WZ_re + WA_im.T@WZ_im + L1_vec)
				cost = 0.5*coef.T@P@coef + q.T@coef
				# for frac,lam_vec,ha,hb in zip(reg_ord,lam_vectors,hyper_as,hyper_bs):
					# cost += frac*np.sum((hb*lam_vec - (ha-1)*np.log(lam_vec)))
				# print('cost after coef optimization:',cost)
				
				self._iter_history.append({'lambda_vectors':lam_vectors.copy(),'coef':coef.copy(),'fun':result['primal objective'],'cost':cost,'result':result,'dZ_re':dZ_re.copy(),
										'hyper_bs':hyper_bs.copy(),'hyper_lambda0s':hyper_lambda0s.copy(),'hyper_betas':hyper_betas.copy()})
				
				# check for convergence
				coef_delta = (coef - prev_coef)/prev_coef
				# If inductance not fitted, set inductance delta to zero (inductance goes to random number)
				if self.fit_inductance==False:
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
			self.opt_result_ = result
			self.coef_ = coef.copy()
			self.lambda_vectors_ = lam_vectors.copy()
			self.cost_ = cost.copy()
					
		else:
			# ordinary ridge
			# create L2 penalty matrix
			lam_vectors = [np.ones(A_re.shape[1])*lambda_0]*3
			lam_matrices = [np.diag(lam_vec**0.5) for lam_vec in lam_vectors]
			L2_mat = np.zeros_like(L2_base[0])
			for L2b,lam_mat,frac in zip(L2_base,lam_matrices,reg_ord):
				if frac>0:
					L2_mat += frac*(lam_mat@L2b@lam_mat)
			# L2_mat = self.M
			self.opt_result_ = self._convex_opt(part,WZ_re,WZ_im,WA_re,WA_im,L2_mat,L1_vec,nonneg)
			self.coef_ = np.array(list(self.opt_result_['x']))
		
		# rescale coefficients if scaling applied to A or Z
		if scale_A or scale_Z:
			self._scaled_coef = self.coef_.copy()
			self.coef_ = self._rescale_coef(self.coef_,scale_A,scale_Z)
		# rescale the inductance
		self.coef_[1] *= 1e-4
		
		# If inductance not fitted, set inductance to zero to avoid confusion (goes to random number)
		if self.fit_inductance==False:
			self.coef_[1] = 0
		
		# If fitted imag part only, set high-frequency resistance to align with start of real impedance
		if part=='imag':
			Z_pred = self.predict(frequencies)
			self.coef_[0] += Z.real[0] - Z_pred[0].real
			
		self._recalc_mat = False
		self.fit_type = 'ridge'
		
	def map_fit(self,frequencies,Z,part='both',scale_A=False,scale_Z=True,dZ=False,init_from_ridge=True,nonneg=False,outliers=False,sigma_min=0.002,max_iter=30000):
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
		scale_A: bool, optional (default: False)
			If True, scale the A matrices such that the modulus of each column (considering both A_re and A_im) has unit variance. Not recommended
		scale_Z: bool, optional (default: True)
			
		"""
		# perform scaling and weighting and get A and B matrices
		frequencies, Z_scaled, A_re,A_im, WA_re,WA_im,WZ_re,WZ_im, B = self._prep_matrices(frequencies,Z,part,weights=None,dZ=dZ,scale_A=scale_A,scale_Z=scale_Z)
		
		# create L matrices
		self.L0 = construct_L(1/(2*np.pi*self.tau),tau=self.tau,basis=self.basis,epsilon=self.epsilon,order=0)
		self.L1 = construct_L(1/(2*np.pi*self.tau),tau=self.tau,basis=self.basis,epsilon=self.epsilon,order=1)
		self.L2 = construct_L(1/(2*np.pi*self.tau),tau=self.tau,basis=self.basis,epsilon=self.epsilon,order=2)
		
		# get initial fit
		if init_from_ridge:
			init = self._get_init_from_ridge(frequencies,Z,beta=2.5,lambda_0=1e-2,nonneg=nonneg,outliers=outliers)
			self._init_params = init()
		else:
			init = 'random'
		
		# prepare data for stan model
		dat = self._prep_stan_data(frequencies,Z_scaled,part,A_re,A_im,B,outliers,sigma_min,mode='optimize')
		
		# load stan model
		model_str = 'drt'
		if dZ:
			model_str += '_dZ'
		else:
			model_str +='_no-dZ'
		
		if nonneg:
			model_str += '_pos'
			
		if outliers:
			model_str += '_outliers'
			
		model_str += '_StanModel.pkl'
		# print(model_str)
		model = load_pickle(os.path.join(script_dir,'stan_model_files',model_str))

		# optimize posterior
		self._opt_result = model.optimizing(dat,iter=max_iter,seed=1234,init=init)
		
		# extract coefficients
		self.coef_ = np.zeros(len(self._opt_result['beta'])+2)
		self.coef_[0] = self._opt_result['hfr']
		self.coef_[1] = self._opt_result['induc']
		self.coef_[2:] = self._opt_result['beta']
		
		# rescale coefficients if scaling applied to A or Z
		if scale_A or scale_Z:
			self._scaled_coef = self.coef_.copy()
			self.coef_ = self._rescale_coef(self.coef_,scale_A,scale_Z)
		
		self.fit_type = 'map'
		self.sigma_min = sigma_min
		
	def bayes_fit(self,frequencies,Z,part='both',scale_A=False,scale_Z=True,dZ=False,init_from_ridge=True,nonneg=False,outliers=False,sigma_min=0.002,
			warmup=200,sample=200,chains=2):
		# perform scaling and weighting and get A and B matrices
		frequencies, Z_scaled, A_re,A_im, WA_re,WA_im,WZ_re,WZ_im, B = self._prep_matrices(frequencies,Z,part,weights=None,dZ=dZ,scale_A=scale_A,scale_Z=scale_Z)
		
		# create L matrices
		self.L0 = construct_L(1/(2*np.pi*self.tau),tau=self.tau,basis=self.basis,epsilon=self.epsilon,order=0)
		self.L1 = construct_L(1/(2*np.pi*self.tau),tau=self.tau,basis=self.basis,epsilon=self.epsilon,order=1)
		self.L2 = construct_L(1/(2*np.pi*self.tau),tau=self.tau,basis=self.basis,epsilon=self.epsilon,order=2)
		
		# get initial fit
		if init_from_ridge:
			init = self._get_init_from_ridge(frequencies,Z,beta=2.5,lambda_0=1e-2,nonneg=nonneg,outliers=outliers)
			self._init_params = init()
		else:
			init = 'random'
		
		# prepare data for stan model
		dat = self._prep_stan_data(frequencies,Z_scaled,part,A_re,A_im,B,outliers,sigma_min,mode='sample')
		
		# load stan model
		model_str = 'drt'
		if dZ:
			model_str += '_dZ'
		else:
			model_str +='_no-dZ'
		
		if nonneg:
			model_str += '_pos'
			
		if outliers:
			model_str += '_outliers'
			
		model_str += '_StanModel.pkl'
		# print(model_str)
		model = load_pickle(os.path.join(script_dir,'stan_model_files',model_str))
			
		
		# sample from posterior
		self._sample_result = model.sampling(dat,warmup=warmup,iter=warmup+sample,chains=chains,seed=1234,init=init,
                                  control={'adapt_delta':0.9,'adapt_t0':10})
		
		# extract coefficients
		self.coef_ = np.zeros(self._sample_result['beta'].shape[1] +2)
		self.coef_[0] = np.mean(self._sample_result['hfr'])
		self.coef_[1] = np.mean(self._sample_result['induc'])
		self.coef_[2:] = np.mean(self._sample_result['beta'],axis=0)
		
		# rescale coefficients if scaling applied to A or Z
		if scale_A or scale_Z:
			self._scaled_coef = self.coef_.copy()
			self.coef_ = self._rescale_coef(self.coef_,scale_A,scale_Z)
		
		self.fit_type = 'bayes'
		self.sigma_min = sigma_min
			
	def _get_init_from_ridge(self,frequencies,Z,beta,lambda_0,nonneg,outliers):
		"""Get initial parameter estimate from ridge_fit
		Parameters:
		-----------
		frequencies: array
			Array of frequencies
		Z: array
			Impedance data
		beta: float
			beta regularization parameter
		lambda_0: float
			lambda_0 regularization parameter
		"""
		# get initial parameter values from ridge fit
		self.ridge_fit(frequencies,Z,hyper_lambda=True,hyper_penalty='integral',reg_ord=2,scale_Z=True,scale_A=False,dZ=True,
			   beta=beta,lambda_0=lambda_0,nonneg=nonneg)
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
		
	def _prep_stan_data(self,frequencies,Z,part,A_re,A_im,B,outliers,sigma_min,mode):
		if part=='both':
			Z_stack = np.concatenate((Z.real,Z.imag))
			A_stack = np.concatenate((A_re[:,2:],-A_im[:,2:]))
		else:
			Z_stack = getattr(Z,part)
			A_stack = getattr(self,'A_{}'.format(part[:2]))[:,2:]
			
		if mode=='sample':
			tau_alpha = 1
			tau_beta = 0.1
			L0 = self.L0[:,2:]
			L1 = self.L1[:,2:]
			L2 = self.L2[:,2:]
			
		elif mode=='optimize':
			tau_alpha = 0.05
			tau_beta = 0.1
			L0 = 1.5*0.24*self.L0[:,2:]
			L1 = 1.5*0.16*self.L1[:,2:]
			L2 = 1.5*0.08*self.L2[:,2:]
			
		dat = {'N':2*len(frequencies),
			   'freq':frequencies,
			   'K':A_stack.shape[1],
			   'x':A_stack,
			   'y':Z_stack,
			   'N_tilde':A_stack.shape[0],
			   'x_tilde':A_stack,
			   'freq_tilde': frequencies,
			   'L0':L0,
			   'L1':L1,
			   'L2':L2,
			   'sigma_min':sigma_min,
			   'tau_alpha':tau_alpha,
			   'tau_beta':tau_beta,
			  }
			  
		if B is not None:
			dat['B'] = B[:,2:]
			
		if outliers:
			if mode=='optimize':
				dat['so_invscale'] = 5
			elif mode=='sample':
				dat['so_invscale'] = 10
			  
		return dat
			  
	def _hyper_lambda_discrete(self,L,coef,beta=2.5,lambda_0=1):
		Lx2 = (L@coef)**2
		lam = np.ones(self.A_re.shape[1]) #*lambda_0
		lam[2:] = 1/(Lx2/(beta-1) + 1/lambda_0)
		return lam
		
	def _grad_lambda_discrete(self,frequencies,coef,lam_vec,reg_ord,beta=2.5,lambda_0=1):
		L = construct_L(frequencies,tau=self.tau,basis=self.basis,epsilon=self.epsilon,order=reg_ord)
		Lx2 = (L@coef)**2
		zeta = (beta-1)/lambda_0
		grad = Lx2 + zeta - (beta-1)/lam_vec[2:]
		return grad
		
	def _hyper_lambda_integral(self,M,coef,lam_mat,beta=2.5,lambda_0=1):
		X = np.diag(coef)
		xlm = X@lam_mat@M@X
		xlm = xlm - np.diag(np.diagonal(xlm))
		C = np.sum(xlm,axis=0)
		
		a = beta/2
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
	
	def _convex_opt(self,part,WZ_re,WZ_im,WA_re,WA_im,L2_mat,L1_vec,nonneg):
		if part=='both':
			P = cvxopt.matrix((WA_re.T@WA_re + WA_im.T@WA_im + L2_mat).T)
			q = cvxopt.matrix((-WA_re.T@WZ_re + WA_im.T@WZ_im + L1_vec).T)
		elif part=='real':
			P = cvxopt.matrix((WA_re.T@WA_re + L2_mat).T)
			q = cvxopt.matrix((-WA_re.T@WZ_re + L1_vec).T)
		else:
			P = cvxopt.matrix((WA_im.T@WA_im + L2_mat).T)
			q = cvxopt.matrix((WA_im.T@WZ_im + L1_vec).T)
		
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
		
	def _prep_matrices(self,frequencies,Z,part,weights,dZ,scale_A,scale_Z):
		if len(frequencies)!=len(Z):
			raise ValueError("Length of frequencies and Z must be equal")
			
		# sort frequencies descending
		frequencies = np.sort(frequencies)[::-1]
		
		# check if we need to recalculate A matrices
		freq_subset = False
		if np.min(rel_round(self.f_train,10)==rel_round(frequencies,10))==False:
			# if frequencies have changed, must recalculate
			self._recalc_mat = True
			# if frequencies are a subset of f_train, we can use submatrices of the existing A matrices
			# instead of recalculating
			if np.min([rel_round(f,10) in rel_round(self.f_train,10) for f in frequencies])==True:
				freq_subset = True
			else:
				self.f_train = frequencies
		else:
			self.f_train = frequencies
		# print(self._recalc_mat)
		
		if type(Z)!=np.ndarray:
			Z = np.array(Z)
		
		if self.basis_freq is None:
			# by default, use 10 ppd for tau spacing regardless of input frequency spacing
			tmin = np.log10(1/(2*np.pi*np.max(frequencies)))
			tmax = np.log10(1/(2*np.pi*np.min(frequencies)))
			num_decades = tmax - tmin
			self.tau = np.logspace(tmin,tmax, int(10*np.ceil(num_decades) + 1))
		else:
			self.tau = 1/(2*np.pi*self.basis_freq)
		
		if self.epsilon is None:
			dlnt = np.mean(np.diff(np.log(self.tau)))
			self.set_epsilon(1/dlnt)
		
		# create A matrices
		if self._recalc_mat==False:
			A_re = self.A_re.copy()
			A_im = self.A_im.copy()
			if dZ:
				if hasattr(self,'B'):
					B = self.B
				else:
					# Z_re differentiation matrix (B@coef gives approx dZ'/dlnt at each basis tau)
					tau_diff = np.mean(np.diff(np.log(self.tau)))
					B_start = np.exp(np.log(self.tau[0]) - tau_diff/2)
					B_end = np.exp(np.log(self.tau[-1]) + tau_diff/2)
					self._B_tau = np.logspace(np.log10(B_start),np.log10(B_end),len(self.tau)+1)
					B_pre = construct_A(1/(2*np.pi*self._B_tau),'real',tau=self.tau,basis=self.basis,epsilon=self.epsilon)
					self.B = B_pre[1:,:] - B_pre[:-1,:]
					B = self.B
		if self._recalc_mat or hasattr(self,'A_re')==False or hasattr(self,'A_im')==False:
			if freq_subset:
				# frequencies is a subset of f_train - no need to recalc
				# print('freq in f_train')
				f_index = np.array([np.where(rel_round(self.f_train,10)==rel_round(f,10))[0][0] for f in frequencies])
				A_re = self.A_re[f_index,:].copy()
				A_im = self.A_im[f_index,:].copy()
				
				if dZ:
					B = self.B
					
			else:
				self.A_re = construct_A(frequencies,'real',tau=self.tau,basis=self.basis,fit_inductance=self.fit_inductance,epsilon=self.epsilon)
				self.A_im = construct_A(frequencies,'imag',tau=self.tau,basis=self.basis,fit_inductance=self.fit_inductance,epsilon=self.epsilon)
				A_re = self.A_re
				A_im = self.A_im.copy()
				
				if dZ:
					# Z_re differentiation matrix (B@coef gives approx dZ'/dlnt at each basis tau)
					tau_diff = np.mean(np.diff(np.log(self.tau)))
					B_start = np.exp(np.log(self.tau[0]) - tau_diff/2)
					B_end = np.exp(np.log(self.tau[-1]) + tau_diff/2)
					self._B_tau = np.logspace(np.log10(B_start),np.log10(B_end),len(self.tau)+1)
					B_pre = construct_A(1/(2*np.pi*self._B_tau),'real',tau=self.tau,basis=self.basis,epsilon=self.epsilon)
					self.B = B_pre[1:,:] - B_pre[:-1,:]
					B = self.B
		# apply scaling
		if scale_A:
			self._scale_A()
			A_re = self.A_re_scaled
			A_im = self.A_im_scaled.copy()
			if dZ:
				B = self.B_scaled
		if scale_Z:
			Zmod = (Z*Z.conjugate())**0.5
			Z = self._scale_Z(Z)
			if type(weights) in (list,np.ndarray):
				weights = np.array(weights)/self._Z_scale
		else:
			self._Z_scale = 1
		# scale the inductance so that it doesn't drop below the tolerance of cvxopt
		# this does not affect map_fit or bayes_fit, since they use separate hfr and induc parameters
		A_im[:,1] *= 1e-4
			
		# create weight matrices
		weights = self._format_weights(frequencies,Z,weights,part)
		W_re = np.diag(np.real(weights))
		W_im = np.diag(np.imag(weights))
		
		# apply weights to A and Z
		WA_re = W_re@A_re
		WA_im = W_im@A_im
		WZ_re = W_re@Z.real
		WZ_im = W_im@Z.imag
		
		if not dZ:
			B = None
		
		return frequencies, Z, A_re,A_im, WA_re,WA_im,WZ_re,WZ_im, B
			
			
	def _format_weights(self,frequencies,Z,weights,part):
		"""
		Format real and imaginary weight vectors
		Parameters:
		-----------
		weights : str or array (default: None)
			Weights for fit. Standard weighting schemes can be specified by passing 'modulus' or 'proportional'. 
			Custom weights can be passed as an array. If the array elements are real, the weights are applied to both the real and imaginary parts of the impedance.
			If the array elements are complex, the real parts are used to weight the real impedance, and the imaginary parts are used to weight the imaginary impedance.
			If None, all points are weighted equally.
		part : str (default:'both')
			Which part of impedance is being fitted. Options: 'both', 'real', or 'imag'
		"""
		if weights is None:
			weights = np.ones_like(frequencies)*(1+1j)
		elif type(weights)==str:
			if weights=='modulus':
				weights = (1+1j)/np.sqrt(np.real(Z*Z.conjugate()))
			elif weights=='proportional':
				weights = 1/np.abs(Z.real) + 1j/np.abs(Z.imag)
			else:
				raise ValueError(f"Invalid weights argument {weights}. String options are 'modulus' and 'proportional'")
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
		
	def _scale_A(self):
		"""Scale A matrices such that, when considered together, the modulus of each column has unit variance"""
		A_com = self.A_re[:,2:] + 1j*self.A_im[:,2:]
		A_mod = (A_com*A_com.conjugate())**0.5
		self._A_scale = np.ones(self.A_re.shape[1])
		# don't scale high-frequency resistance or inductance
		self._A_scale[2:] = np.std(A_mod,axis=0)
		self.A_re_scaled = self.A_re.copy()
		self.A_im_scaled = self.A_im.copy()
		self.A_re_scaled /= self._A_scale
		self.A_im_scaled /= self._A_scale
		if hasattr(self,'B'):
			self.B_scaled = self.B/self._A_scale
		
	def _scale_Z(self,Z):
		Zmod = (Z*Z.conjugate())**0.5
		# scale by sqrt(n) as suggested by Ishwaran and Rao (doi: 10.1214/009053604000001147)
		# hyperparameters were selected based on spectra with 81 data points - therefore, scale relative to n=81
		self._Z_scale = np.std(Zmod)/np.sqrt(len(Z)/81)
		return Z/self._Z_scale
		
	def _rescale_coef(self,coef,scale_A,scale_Z):
		if scale_Z and not scale_A:
			rs_coef = coef*self._Z_scale
		elif scale_A and not scale_Z:
			rs_coef = coef/self._A_scale
		else:
			rs_coef = coef*self._Z_scale/self._A_scale
		return rs_coef
		
	def coef_percentile(self,percentile):
		if self.fit_type=='bayes':
			coef = np.zeros(self._sample_result['beta'].shape[1] + 2)
			coef[0] = np.percentile(self._sample_result['hfr'],percentile)
			coef[1] = np.percentile(self._sample_result['induc'],percentile)
			coef[2:] = np.percentile(self._sample_result['beta'],percentile,axis=0)
			coef *= self._Z_scale
		else:
			raise ValueError('Percentile prediction is only available for bayes_fit')
			
		return coef
		
	def predict(self,frequencies,percentile=None):
		# check if we need to recalculate A matrices
		freq_subset = False
		if np.min(rel_round(self.f_train,10)==rel_round(frequencies,10))==False:					
			# if frequencies are a subset of f_train, we can use submatrices of the existing A matrices
			# instead of calculating new A matrices
			if np.min([rel_round(f,10) in rel_round(self.f_train,10) for f in frequencies])==True:
				# print('freq in f_train')
				f_index = np.array([np.where(rel_round(self.f_train,10)==rel_round(f,10))[0][0] for f in frequencies])
				A_re = self.A_re[f_index,:].copy()
				A_im = self.A_im[f_index,:].copy()
			# otherwise, we need to calculate A matrices
			else:
				A_re = construct_A(frequencies,'real',tau=self.tau,basis=self.basis,fit_inductance=self.fit_inductance,epsilon=self.epsilon)
				A_im = construct_A(frequencies,'imag',tau=self.tau,basis=self.basis,fit_inductance=self.fit_inductance,epsilon=self.epsilon)
		else:
			A_re = self.A_re
			A_im = self.A_im

		# need to fix this - should use the output from sample_result, rather than coef_percentile
		if percentile is not None:
			coef = self.coef_percentile(percentile)
		else:
			coef = self.coef_
			
		Z_re = A_re@coef
		Z_im = -A_im@coef
		return Z_re + 1j*Z_im
		
	def predict_err_scale(self,frequencies,percentile=None):
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
				
			Z_pred = self.predict(frequencies,percentile)
			
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
			score = r2_score(Z,Z_pred,sample_weight=weights)
		else:
			raise ValueError(f"Invalid metric {metric}. Options are 'chi_sq', 'r2'")
			
		return score	
		
	def drt(self,eval_tau=None,percentile=None):
		"""Get distribution of relaxation times"""
		if eval_tau is None:
			eval_tau = self.tau
			
		phi = get_basis_func(self.basis)
		bases = np.array([phi(np.log(eval_tau/t_m),self.epsilon) for t_m in self.tau]).T
		
		if percentile is not None:
			coef = self.coef_percentile(percentile)
		else:
			coef = self.coef_
			
		gamma = bases@coef[2:]
		return gamma
		
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
	if basis=='gaussian':
		def phi(y,epsilon):
			return np.exp(-(epsilon*y)**2)
	elif basis=='Zic':
		def phi(y,epsilon):
			# epsilon unused, included only for compatibility
			return 2*np.exp(y)/(1+np.exp(2*y))
	else:
		raise ValueError(f'Invalid basis {basis}. Options are gaussian')
	return phi
	
def get_A_func(part,basis='gaussian'):
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
		
	if part=='real':
		def func(y,w_n,t_m,epsilon=1):
			return phi(y,epsilon)/(1+np.exp(2*(y+np.log(w_n*t_m)))) 
	elif part=='imag':
		def func(y,w_n,t_m,epsilon=1):
			return phi(y,epsilon)*np.exp(y)*w_n*t_m/(1+np.exp(2*(y+np.log(w_n*t_m))))
	return func
	
def is_loguniform(frequencies):
	"Check if frequencies are uniformly log-distributed"
	fdiff = np.diff(np.log(frequencies))
	if np.std(fdiff)/np.mean(fdiff) <= 0.01:
		return True
	else: 
		return False

def construct_A(frequencies,part,tau=None,basis='gaussian',fit_inductance=False,epsilon=1):
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
	if tau is None:
		tau = 1/omega
		tau_eq_omega = True
	elif np.min(rel_round(tau,10)==rel_round(1/omega,10)):
		tau_eq_omega = True
	else:
		tau_eq_omega = False
	# add 2 columns for high-frequency resistance and inductance
	A = np.zeros((len(frequencies),len(tau)+2))
	
	# get function to integrate
	func = get_A_func(part,basis)
		
	if is_loguniform(frequencies) and tau_eq_omega:
		# only need to calculate 1st row and column
		w_0 = omega[0]
		t_0 = tau[0]
		if part=='real':
			if basis=='Zic':
				limits = (-100,100)
			else:
				limits = (-np.inf,np.inf)
		elif part=='imag':
			# scipy.integrate.quad is unstable for imag func with infinite limits
			limits = (-20,20)
#		  elif part=='imag':
#			  y = np.arange(-5,5,0.1)
#			  c = [np.trapz(func(y,w_n,t_0),x=y) for w_n in omega]
#			  r = [np.trapz(func(y,w_0,t_m),x=y) for t_m in 1/omega]
		c = [quad(func,limits[0],limits[1],args=(w_n,t_0,epsilon),epsabs=1e-4)[0] for w_n in omega]
		r = [quad(func,limits[0],limits[1],args=(w_0,t_m,epsilon),epsabs=1e-4)[0] for t_m in tau]
		if r[0]!=c[0]:
			print(r[0],c[0])
			raise Exception('First entries of first row and column are not equal')
		A_main = toeplitz(c,r)
	else:
		# need to calculate all entries
		if part=='real':
			limits = (-np.inf,np.inf)
		elif part=='imag':
			# scipy.integrate.quad is unstable for imag func with infinite limits
			limits = (-100,100)
		A_main = np.empty((len(frequencies),len(tau)))
		for n,w_n in enumerate(omega):
			A_main[n,:] = [quad(func,limits[0],limits[1],args=(w_n,t_m,epsilon),epsabs=1e-4)[0] for t_m in tau]
			
	if part=='real':
		A[:,0] = 1
	elif part=='imag' and fit_inductance==True:
		A[:,1] = -omega
	A[:,2:] = A_main
		
	return A
	

def construct_L(frequencies,tau=None,basis='gaussian',epsilon=1,order=1):
	"Differentiation matrix. L@x gives derivative of DRT"
	omega = 2*np.pi*frequencies
	if tau is None:
		# if no time constants given, assume collocated with frequencies
		tau = 1/omega
		
	L = np.zeros((len(omega),len(tau)+2))
	
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
			L[n,2:] = [dphi_dy(np.log(1/(omega[n]*t_m)),epsilon) for t_m in tau]
		
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
	
	# add 2 rows/columns for high-frequency resistor and inductance
	# these should not be penalized (?) - leave as zeroes
	M = np.zeros((len(frequencies)+2,len(frequencies)+2))
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
		M_main = toeplitz(c)
	else:
		# need to calculate all entries
		M_main = np.empty((len(frequencies),len(frequencies)))
		for n,w_n in enumerate(omega):
			M_main[n,:] = [func(w_n,t_m,epsilon) for t_m in 1/omega]
	
	M[2:,2:] = M_main
	
	return M
	
