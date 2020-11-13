import GP_DRT
import numpy as np
from scipy.optimize import minimize, curve_fit

print('reloaded GP_utils')

def predict_Z_re(gamma,tau,freq,Z_exp):
	"Predict real impedance using trapezoidal approx"
	Z_re_drt = [np.trapz(gamma/(1+(2*np.pi*f*tau)**2),x=np.log(tau)) for f in freq]
	Z_re_drt = np.array(Z_re_drt)
	# estimate Z_re offset
	def obj_fun(xdata,x):
		return Z_re_drt + x
	res = curve_fit(obj_fun,freq,Z_exp.real)
	return Z_re_drt + res[0]

def gp_fit(freq,Z_exp,freq_star=None,theta0=[1,5,1],max_iter=40):
	"Fitting process based on ex1_simple_ZARC_model.ipynb"
	
	xi_vec = np.log(2*np.pi*freq)
	N_freqs = len(freq)
	
	# initialize the parameter for global 3D optimization to maximize the marginal log-likelihood as shown in eq (31)
	sigma_n,sigma_f,ell = theta0

	theta_0 = np.array([sigma_n, sigma_f, ell])
	global seq_theta
	seq_theta = np.copy(theta_0)
	global seq_NMLL
	seq_NMLL = [GP_DRT.NMLL_fct(theta_0, Z_exp, xi_vec)]
	def print_results(theta):
		global seq_theta
		global seq_NMLL
		seq_theta = np.vstack((seq_theta, theta))
		NMLL = GP_DRT.NMLL_fct(theta, Z_exp, xi_vec)
		seq_NMLL.append(NMLL)
		print('{0:.7f}	{1:.7f}	 {2:.7f}  {3:.3f}'.format(theta[0], theta[1], theta[2], NMLL))
		
	# GP_DRT.grad_NMLL_fct(theta_0, Z_exp, xi_vec)
	print('sigma_n,	  sigma_f,	 ell,		NMLL')

	# minimize the NMLL L(\theta) w.r.t sigma_n, sigma_f, ell using the Newton-CG method as implemented in scipy
	# Limit iterations for reasonable run times - if not converged by 20-30 iterations, usually stuck anyway
	try:
		res = minimize(GP_DRT.NMLL_fct, theta_0, args=(Z_exp, xi_vec), method='Newton-CG', \
					   jac=GP_DRT.grad_NMLL_fct,  callback=print_results, options={'disp': True,'maxiter':max_iter})
		# collect the optimized parameters
		sigma_n, sigma_f, ell = res.x
	except (np.linalg.LinAlgError,OverflowError) as lae:
		print(lae)
		best_idx = np.argmin(seq_NMLL)
		print(f'Choosing iteration with lowest NMLL: {best_idx}')
		if len(seq_theta.shape)==1:
			sigma_n, sigma_f, ell = seq_theta
		else:
			sigma_n, sigma_f, ell = seq_theta[best_idx]
		res = dict(sigma_n=sigma_n,sigma_f=sigma_f,ell=ell,fun=seq_NMLL[best_idx])

	# compute matrices
	K = GP_DRT.matrix_K(xi_vec, xi_vec, sigma_f, ell)
	L_im_K = GP_DRT.matrix_L_im_K(xi_vec, xi_vec, sigma_f, ell)
	L2_im_K = GP_DRT.matrix_L2_im_K(xi_vec, xi_vec, sigma_f, ell)
	Sigma = (sigma_n**2)*np.eye(N_freqs) # add 1e-5 for stability in case sigma_n approaches zero

	# the matrix $\mathcal L^2_{\rm im} \mathbf K + \sigma_n^2 \mathbf I$ whose inverse is needed
	K_im_full = L2_im_K + Sigma

	# Cholesky factorization, L is a lower-triangular matrix
	L = np.linalg.cholesky(K_im_full)

	# solve for alpha
	alpha = np.linalg.solve(L, Z_exp.imag)
	alpha = np.linalg.solve(L.T, alpha)

	# estimate the gamma of eq (21a), the minus sign, which is not included in L_im_K, refers to eq (65)
	gamma_fct_est = -np.dot(L_im_K.T, alpha)

	# covariance matrix
	inv_L = np.linalg.inv(L)
	inv_K_im_full = np.dot(inv_L.T, inv_L)

	# estimate the sigma of gamma for eq (21b)
	cov_gamma_fct_est = K - np.dot(L_im_K.T, np.dot(inv_K_im_full, L_im_K))
	sigma_gamma_fct_est = np.sqrt(np.diag(cov_gamma_fct_est))
	
	# estimate Z_im for xi_vec (added by J. Huang)
	Z_im_vec_est = np.empty_like(xi_vec)
	Sigma_Z_im_vec_est = np.empty_like(xi_vec)
	
	for index, val in enumerate(xi_vec):
		xi_i = np.array([val])
		
		# compute matrices, k_i corresponds to a point in training data
		# k_i = GP_DRT.matrix_K(xi_vec, xi_i, sigma_f, ell)
		# L_im_k_star = GP_DRT.matrix_L_im_K(xi_vec, xi_star, sigma_f, ell)
		L2_im_k_i = GP_DRT.matrix_L2_im_K(xi_vec, xi_i, sigma_f, ell)
		# k_star_star = GP_DRT.matrix_K(xi_star, xi_star, sigma_f, ell)
		# L_im_k_star_star = GP_DRT.matrix_L_im_K(xi_star, xi_star, sigma_f, ell)
		L2_im_k_i_i = GP_DRT.matrix_L2_im_K(xi_i, xi_i, sigma_f, ell)
		
		Z_im_vec_est[index] = np.dot(L2_im_k_i.T, np.dot(inv_K_im_full, Z_exp.imag))
		Sigma_Z_im_vec_est[index] = L2_im_k_i_i-np.dot(L2_im_k_i.T, np.dot(inv_K_im_full, L2_im_k_i))
	
	# predict the DRT and impedance for freq_star
	# -------------------------------------------
	if freq_star is None:
		# assume freq_star is same as freq
		Z_im_vec_star = np.copy(Z_im_vec_est)
		Sigma_Z_im_vec_star = np.copy(Sigma_Z_im_vec_est)

		gamma_vec_star = np.copy(gamma_fct_est)
		Sigma_gamma_vec_star = np.copy(sigma_gamma_fct_est)
	else:
		xi_vec_star = np.log(2*np.pi*freq_star)
		# initialize the imaginary part of impedance vector
		Z_im_vec_star = np.empty_like(xi_vec_star)
		Sigma_Z_im_vec_star = np.empty_like(xi_vec_star)

		gamma_vec_star = np.empty_like(xi_vec_star)
		Sigma_gamma_vec_star = np.empty_like(xi_vec_star)

		# calculate the imaginary part of impedance at each $\xi$ point for the plot
		for index, val in enumerate(xi_vec_star):
			xi_star = np.array([val])

			# compute matrices shown in eq (18), k_star corresponds to a new point
			# k_star = GP_DRT.matrix_K(xi_vec, xi_star, sigma_f, ell) # not needed
			L_im_k_star = GP_DRT.matrix_L_im_K(xi_vec, xi_star, sigma_f, ell)
			L2_im_k_star = GP_DRT.matrix_L2_im_K(xi_vec, xi_star, sigma_f, ell)
			k_star_star = GP_DRT.matrix_K(xi_star, xi_star, sigma_f, ell)
			# L_im_k_star_star = GP_DRT.matrix_L_im_K(xi_star, xi_star, sigma_f, ell) # not needed
			L2_im_k_star_star = GP_DRT.matrix_L2_im_K(xi_star, xi_star, sigma_f, ell)

			# compute Z_im_star mean and standard deviation using eq (26)
			Z_im_vec_star[index] = np.dot(L2_im_k_star.T, np.dot(inv_K_im_full, Z_exp.imag))
			Sigma_Z_im_vec_star[index] = L2_im_k_star_star-np.dot(L2_im_k_star.T, np.dot(inv_K_im_full, L2_im_k_star))
			
			# compute Z_im_star mean and standard deviation
			gamma_vec_star[index] = -np.dot(L_im_k_star.T, np.dot(inv_K_im_full, Z_exp.imag))
			Sigma_gamma_vec_star[index] = np.sqrt(k_star_star-np.dot(L_im_k_star.T, np.dot(inv_K_im_full, L_im_k_star))) 
			
	# estimate Z_re for freq using fine DRT (added by J. Huang)
	Z_re_vec = predict_Z_re(gamma_vec_star,1/(2*np.pi*freq_star),freq,Z_exp)
		
	# create result dict
	result = {'gamma':gamma_fct_est,'sigma_gamma':sigma_gamma_fct_est,
			'gamma_star':gamma_vec_star,'sigma_gamma_star':Sigma_gamma_vec_star,
			'Z_im_fit':Z_im_vec_est,'sigma_Z_im_fit':Sigma_Z_im_vec_est,
			'Z_re_fit':Z_re_vec,
			'Z_im_star':Z_im_vec_star,'sigma_Z_im_star':Sigma_Z_im_vec_star,
			'hyperparams':{'sigma_n':sigma_n, 'sigma_f':sigma_f, 'ell':ell},
			'min_result':res
			}
			
	return result