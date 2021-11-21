import numpy as np
from scipy.linalg import cholesky
from scipy.special import loggamma
from scipy.optimize import least_squares, minimize, minimize_scalar
import pandas as pd
import cvxopt
import warnings
import os
from copy import deepcopy
import matplotlib.pyplot as plt

from .matrices import get_basis_func, construct_A, construct_L, construct_M
from .stan_models import save_pickle, load_pickle
from . import peak_fit as pf
from . import file_load as fl
from . import plotting as bp
from .utils import check_equality, rel_round, get_outlier_thresh, r2_score, \
    get_unit_scale, get_scale_factor, get_factor_from_unit

cvxopt.solvers.options['show_progress'] = False

script_dir = os.path.dirname(os.path.realpath(__file__))

warnings.simplefilter('always', UserWarning)
warnings.simplefilter('once', RuntimeWarning)


class Inverter:
    def __init__(self, basis_freq=None, basis='gaussian', epsilon=None, fit_inductance=True,
                 distributions={'DRT': {'kernel': 'DRT'}}):
        """
		Parameters:
		-----------
		basis_freq: array (default: None)
			Frequencies to use for radial basis functions for all distributions. If None, determine from measurement frequencies.
			If you specify a custom basis_freq, it is highly recommended that you use a spacing of 10 points per decade.
			basis_freq arrays specified for individual distributions in the distributions argument take precedence over this array.
		basis: str (defualt: 'gaussian')
			Type of radial basis function to use. 'gaussian' is currently the only option.
		epsilon: float (default: None)
			Inverse length scale of radial basis functions. If None, determined automatically from basis_freq.
			For best results, leave epsilon at its default value for map_fit and bayes_fit.
			For ridge_fit, epsilon can be tuned as desired.
		fit_inductance: bool (default: True)
			If True, fit inductance; if False, assume inductance is zero. Applies only to ridge_fit.
		distributions: dict (default: {'DRT':{'kernel':'DRT'}})
			Dictionary of distributions to fit. Each key-value pair consists of a distribution name and a nested dict of parameters.
			See set_distributions for details.
		"""
        self._recalc_mat = True
        self.distribution_matrices = {}
        self.set_basis_freq(basis_freq)
        self.set_basis(basis)
        self.set_epsilon(epsilon)  # inverse length scale of RBFs
        self.set_fit_inductance(fit_inductance)
        self.set_distributions(distributions)
        self._cached_distributions = self.distributions.copy()
        self.f_train = [0]
        self.Z_train = None
        self.f_pred = None
        self._Z_scale = 1.0
        self._init_params = {}
        self.distribution_fits = {}
        self._iter_history = None

    def set_distributions(self, distributions):
        """Set kernels for inversion

		Parameters:
		-----------
		distributions: dict
			Dict of dicts describing distributions to include. Top-level keys are user-supplied names for distributions.
			Each nested dict defines a distribution and should include the following keys:
				kernel: 'DRT' or 'DDT'
				dist_type: 'series' or 'parallel' (default 'parallel'). Valid for DDT only
				symmetry: 'planar' or 'spherical' (default 'planar'). Valid for DDT only
				bc: 'transmissive' or 'blocking' (default 'transmissive'). Valid for DDT only
				ct: boolean indicating whether there is a simultaneous charge transfer reaction (default False).
					Valid for DDT only
				k_ct: apparent 1st-order rate constant for simultaneous charge transfer reaction.
					Required for DDT if ct==True
				basis_freq: array of frequencies to use as basis. If not specified, use self.basis_freq
				epsilon: epsilon value for basis functions. If not specified, use self.epsilon
		"""
        # perform checks and normalizations
        for name, info in distributions.items():
            if info['kernel'] == 'DRT':
                # set dist_type to series and warn if overwriting
                if info.get('dist_type', 'series') != 'series':
                    warnings.warn("dist_type for DRT kernel must be series. \
					Overwriting supplied dist_type '{}' for distribution '{}' with 'series'".format(name,
                                                                                                    info['dist_type']))
                info['dist_type'] = 'series'
                # check for invalid keys and warn
                invalid_keys = np.intersect1d(list(info.keys()), ['symmetry', 'bc', 'ct', 'k_ct'])
                if len(invalid_keys) > 0:
                    warnings.warn("The following keys are invalid for distribution '{}': {}.\
					\n These keys will be ignored".format(name, invalid_keys))

            elif info['kernel'] == 'DDT':
                # check for invalid key-value pairs
                if info.get('dist_type', 'parallel') not in ['series', 'parallel']:
                    raise ValueError(
                        "Invalid dist_type '{}' for distribution '{}'".format(info.get('dist_type', 'NA'), name))
                elif info.get('symmetry', 'planar') not in ['planar', 'spherical']:
                    raise ValueError(
                        "Invalid symmetry '{}' for distribution '{}'".format(info.get('symmetry', 'NA'), name))
                elif info.get('bc', 'transmissive') not in ['transmissive', 'blocking']:
                    raise ValueError("Invalid bc '{}' for distribution '{}'".format(info.get('bc', 'NA'), name))
                elif info.get('ct', True) not in [True, False]:
                    raise ValueError("Invalid ct {} for distribution '{}'".format(info['ct'], name))

                # check if k_ct missing
                if info.get('ct', False) == True:
                    if 'k_ct' not in info.keys():
                        raise ValueError("k_ct must be supplied for distribution '{}' if ct==True".format(name))

                # set defaults and update with supplied args
                defaults = {'dist_type': 'parallel', 'symmetry': 'planar', 'bc': 'blocking', 'ct': False}
                defaults.update(info)
                distributions[name] = defaults

            if name not in self.distribution_matrices.keys():
                self.distribution_matrices[name] = {}

        self._distributions = distributions

        self._recalc_mat = True
        self.f_pred = None

    # print('called set_distributions')

    def get_distributions(self):
        # print('called get_distributions')
        return self._distributions

    distributions = property(get_distributions, set_distributions)

    # ===============================================
    # Methods for hyperparametric ridge fit
    # ===============================================
    def ridge_fit(self, frequencies, Z, part='both',
                  # basic options
                  penalty='discrete', reg_ord=2, L1_penalty=0, scale_Z=True, nonneg=True, weights=None, preset=None,
                  # hyper_lambda parameters
                  hyper_lambda=True, hl_solution='analytic', hl_beta=2.5, hl_fbeta=None, lambda_0=1e-2,
                  cv_lambdas=np.logspace(-10, 5, 31),
                  # hyper_weights parameters
                  hyper_weights=False, hw_beta=2, hw_wbar=1,
                  # convex optimization control
                  xtol=1e-3, max_iter=20,
                  # gamma distribution hyperparameters
                  hyper_a=False, alpha_a=2, hl_beta_a=2, hyper_b=False, sb=1,
                  # correct_phase_offset parameters
                  correct_phase_offset=False, IERange=None, lambda_phz=1, init_phase_offset=False,
                  # other parameters
                  x0=None, dZ=False, dZ_power=0.5):
        """
		Perform ridge fit. Only valid for single-distribution fits.

		Parameters:
		-----------
		frequencies: array
			Measured frequencies
		Z: complex array
			Measured (complex) impedance values. Must have same length as frequencies
		part: str, optional (default: 'both')
			Which part of the impedance data to fit. Options: 'both', 'real', 'imag'
		penalty: str, optional (default: 'discrete')
			Type of penalty matrix to apply to the coefficients. Options:
				'discrete': Applies a matrix that yields the derivative of the distribution at discrete values of tau
				'integral': Applies a matrix that yields the integral of the squared derivative of the distribution across all tau
				'cholesky': Applies the Cholesky decomposition of the integral matrix. Tends to induce asymmetry; not recommended
		reg_ord: int or list, optional (default: 2)
			Order of the derivative to regularize. If int, penalize the nth derivative.
			If list, the penalty will consist of a weighted sum of the 0th-2nd derivatives.
			The list must be length 3, with the first entry indicating the weight of the 0th derivative,
			2nd entry indicating the weight of the 1st derivative, and 3rd entry indicating the weight of the 2nd derivative.
		L1_penalty: float, optional (default: 0)
			Magnitude of L1 (LASSO) penalty to apply to coefficients.
			If L1_penalty > 0, the fit becomes an elastic net regression
		scale_Z: bool, optional (default: True)
			If True, scale impedance by the factor sqrt(N)/std(|Z|) to normalize for magnitude and sample size
		nonneg: bool, optional (default: False)
			If True, constrain the distribution to non-negative values
		weights : str or array (default: None)
			Weights for fit. Standard weighting schemes can be specified by passing 'modulus' or 'proportional'.
			Custom weights can be passed as an array. If the array elements are real, the weights are applied to both the real and imaginary parts of the impedance.
			If the array elements are complex, the real parts are used to weight the real impedance, and the imaginary parts are used to weight the imaginary impedance.
			If None, all points are weighted equally.
		preset: str, optional (default: None)
			Name of preset settings to use for optimization. Options and corresponding arguments are as follows:
			'Ciucci': penalty='discrete',lambda_0='cv',hl_fbeta=0.1
			'Huang': penalty='integral',weights='modulus',dZ=True

		hyper_lambda parameters:
		-------------------------
		hyper_lambda: bool, optional (default: True)
			If True, allow the regularization parameter lambda to vary with tau (hierarchical ridge).
			See https://www.sciencedirect.com/science/article/pii/S0013468617314676 for details.
			If False, perform ordinary ridge regression
		hl_solution: str, optional (default: 'analytic')
			Solution method for determining lambda values when hyper_lambda==True. Options:
				'analytic': Use analytic solution. Generally works well
				'lm': Use Levenberg-Marquardt algorithm. May help avoid oscillation
				arising from analytic solution in some cases
		hl_beta: float, optional (default: 2.5)
			Beta hyperparameter of gamma prior for hyper_lambda method.
			Smaller values allow greater variation in the regularization parameter lambda.
			If penalty is 'discrete' or 'cholesky', hl_beta > 1.
			If penalty=='integral', hl_beta > 2
		hl_fbeta: float, optional (default: None)
			If specified, ignore hl_beta and instead set the f_beta hyperparameter, which
			normalizes for the magnitude of the penalty for the recovered distribution.
			Smaller values allow greater variation in the regularization parameter lambda.
			See http://dx.doi.org/10.1016/j.electacta.2015.03.123 Eqs. 35-36 for details
		lambda_0: float, optional (default: 1e-2)
			lambda_0 hyperparameter for hyper_lambda method.
			Larger values result in stronger baseline level of regularization.
			If lambda_0=='cv', perform Re-Im cross-validation to estimate the optimal lambda_0.
		cv_lambdas: array, optional (default: np.logspace(-10,5,31))
			lambda_0 grid for Re-Im cross-validation if lambda_0=='cv'

		hyper_weights parameters:
		-------------------------
		hyper_weights: bool, optional (default: False)
			If true, optimize weights along with coefficients.
			Allows for outlier identification and estimation of relative error structure.
			See https://www.sciencedirect.com/science/article/pii/S0013468617314676 for details
		hw_beta: float, optional (default: 2)
			beta hyperparameter of gamma prior for hyper_weights method.
			Smaller values allow weights to approach zero (outlier) more easily.
			Must be greater than 1
		hw_wbar: float, optional (default: 1)
			Expectation value of gamma prior on weights

		# convex optimization control:
		------------------------------
		xtol: float, optional (default: 1e-3)
			Coefficient tolerance for iterative optimization (hyper_lambda or hyper_weights).
			Optimization stops when mean change in coefficients drops below xtol
		max_iter: int, optional (default: 20)
			Maximum iterations to perform for hyper_lambda or hyper_weights fits

		correct_phase_offset parameters:
		--------------------------------
		correct_phase_offset: bool, optional (default: False)
			If True, correct phase offsets that may occur due to changes in the
			hardware current range.
		IERange: array, optional (default: None)
			Array of I/E ranges (hardware current ranges, represented as ints) used to measure impedance.
			Must have same length as frequencies. Required if correct_phase_offset==True
		lambda_phz: float, optional (default: 1)
			Inverse scale parameter for exponential prior on phase offsets.
			Larger values regularize phase offsets more strongly, leading to smaller offset values.
		init_phase_offset: bool, optional (default: False)
			If True, estimate phase offsets and adjust Z before first DRT fit.

		other parameters:
		-----------------
		dZ: bool, optional (default: False)
		    For testing only
		dZ_power: float, optional (default: 0.5)
		    For testing only
		x0: array, optional (default: None)
		    Initial parameters for optimization. If None, initalize all coefficients near zero.
		"""
        # apply presets
        presets = ['Ciucci', 'Huang']
        if preset is not None:
            if preset not in presets:
                raise ValueError('Invalid preset {}. Options are {}'.format(preset, presets))
            else:
                if preset == 'Ciucci':
                    penalty = 'discrete'
                    lambda_0 = 'cv'
                    hl_fbeta = 0.1
                elif preset == 'Huang':
                    penalty = 'integral'
                    hl_beta = 2.5
                    lambda_0 = 1e-2
                    weights = 'modulus'
                # dZ=True

        # checks
        if penalty in ('discrete', 'cholesky'):
            if hl_beta <= 1:
                raise ValueError('hl_beta must be greater than 1 for penalty ''cholesky'' and ''discrete''')
        elif penalty == 'integral':
            if hl_beta <= 2:
                raise ValueError('hl_beta must be greater than 2 for penalty ''integral''')
        else:
            raise ValueError(
                f'Invalid penalty argument {penalty}. Options are ''integral'', ''discrete'', and ''cholesky''')

        if hyper_lambda and hyper_weights:
            raise ValueError('hyper_lambda and hyper_weights fits cannot be performed simultaneously')

        if len(self.distributions) > 1:
            raise ValueError('ridge_fit cannot be used to fit multiple distributions')

        if correct_phase_offset and IERange is None:
            raise ValueError('IERange must be provided if correct_phase_offset==True')

        self.distribution_fits = {}

        # Get initial estimate of phase offset
        if correct_phase_offset:
            Z_exp = Z.copy()
            Zphz_exp = np.angle(Z_exp, deg=True)
            # find steps in IERange. Go from low frequency to high frequency
            step_indices = np.where(np.diff(IERange[::-1]) != 0)[0] + 1
            # add last index
            step_indices = np.append(step_indices, len(frequencies))

            # initialize offsets at zero
            phase_offsets = np.zeros(len(step_indices))
            offset_vec = np.zeros(len(Z))
            Zphz_adj = Zphz_exp.copy()[::-1]  # low to high freq

            "initial estimate seems to be unnecessary"
            if init_phase_offset:
                for i, idx in enumerate(step_indices[:-1]):
                    # recalculate diff
                    Zphz_diff = np.diff(Zphz_adj)
                    # interpolate the 1st discrete diff at the step
                    Zphz_diff_interp = (Zphz_diff[idx - 2] + Zphz_diff[idx]) / 2
                    # use the interpolated diff to estimate the appropriate Zphz value after the step
                    Zphz_interp = Zphz_adj[idx - 1] + Zphz_diff_interp
                    phase_offsets[i] = Zphz_interp - Zphz_adj[idx]

                    # update offset vector and adjusted Zphz
                    offset_vec[idx:step_indices[i + 1]] += phase_offsets[i]
                    Zphz_adj[idx:step_indices[i + 1]] += phase_offsets[i]

            Zphz_adj = Zphz_adj[::-1]
            offset_vec = offset_vec[::-1]

            Zmod = (Z * Z.conjugate()) ** 0.5
            Z = Zmod * np.cos(np.deg2rad(Zphz_adj)) + 1j * Zmod * np.sin(np.deg2rad(Zphz_adj))
            Z_adj = Z.copy()

        # perform Re-Im CV to get optimal lambda_0
        if lambda_0 == 'cv':
            lambda_0 = self.ridge_ReImCV(frequencies, Z, lambdas=cv_lambdas,
                                         penalty=penalty, hyper_lambda=hyper_lambda, hl_solution=hl_solution,
                                         hl_beta=hl_beta, hl_fbeta=hl_fbeta,
                                         reg_ord=reg_ord, L1_penalty=L1_penalty,
                                         x0=x0, weights=weights, xtol=xtol, max_iter=max_iter,
                                         scale_Z=scale_Z, nonneg=nonneg,
                                         dZ=dZ, dZ_power=dZ_power,
                                         hyper_a=hyper_a, alpha_a=alpha_a, hl_beta_a=hl_beta_a, hyper_b=hyper_b, sb=sb)

        # ridge_fit can only handle a single distribution. For convenience, pull distribution info out of dicts
        dist_name = list(self.distributions.keys())[0]
        dist_info = self.distributions[dist_name]
        if dist_info['kernel'] != 'DRT' and dZ == True:
            warnings.warn('dZ should only be set to True for DRT recovery. Proceeding with dZ=False')
            dZ = False

        # set fit target
        if dist_info['dist_type'] == 'series':
            # use impedance for series distributions (e.g. DRT)
            target = Z
        elif dist_info['dist_type'] == 'parallel':
            # for parallel distributions, must fit admittance for linearity
            target = 1 / Z

        # perform scaling and weighting and get matrices
        frequencies, target_scaled, WT_re, WT_im, W_re, W_im, dist_mat = self._prep_matrices(frequencies, target, part,
                                                                                             weights, dZ, scale_Z,
                                                                                             penalty, 'ridge')
        # refresh dist_info
        dist_info = self.distributions[dist_name]

        if dist_info['dist_type'] == 'parallel' and scale_Z:
            # redo the scaling such that Z is still the variable that gets scaled
            # this helps avoid tiny admittances, which get ignored in fitting
            Z_scaled = self._scale_Z(Z, 'ridge')
            target_scaled = 1 / Z_scaled
            WT_re = W_re @ target_scaled.real
            WT_im = W_im @ target_scaled.imag

        # unpack matrices
        matrices = dist_mat[dist_name]
        A_re = matrices['A_re']
        A_im = matrices['A_im']
        WA_re = matrices['WA_re']
        WA_im = matrices['WA_im']
        B = matrices['B']
        if penalty != 'integral':
            L0 = matrices['L0']
            L1 = matrices['L1']
            L2 = matrices['L2']
        M0 = matrices.get('M0', None)
        M1 = matrices.get('M1', None)
        M2 = matrices.get('M2', None)
        tau = dist_info['tau']
        epsilon = dist_info['epsilon']

        # for series distributions, adjust matrices for R_inf and inductance
        if dist_info['dist_type'] == 'series':
            # add columns to A_re, B, and A_im
            A_re_main = A_re.copy()
            A_re = np.zeros((A_re_main.shape[0], A_re_main.shape[1] + 2))
            A_re[:, 2:] = A_re_main
            A_re[:, 0] = 1

            if B is not None:
                B = np.hstack((np.zeros((B.shape[1], 2)), B))

            A_im_main = A_im.copy()
            A_im = np.zeros((A_im_main.shape[0], A_im_main.shape[1] + 2))
            A_im[:, 2:] = A_im_main
            if self.fit_inductance:
                # scale the inductance column so that coef[1] doesn't drop below the tolerance of cvxopt
                A_im[:, 1] = 2 * np.pi * frequencies * 1e-4

            # re-apply weight matrices to expanded A matrices
            WA_re = W_re @ A_re
            WA_im = W_im @ A_im

            # add rows and columns to M matrices
            if M0 is not None:
                M0_main = M0.copy()
                M0 = np.zeros((M0_main.shape[0] + 2, M0_main.shape[1] + 2))
                M0[2:, 2:] = M0_main
            if M1 is not None:
                M1_main = M1.copy()
                M1 = np.zeros((M1_main.shape[0] + 2, M1_main.shape[1] + 2))
                M1[2:, 2:] = M1_main
            if M2 is not None:
                M2_main = M2.copy()
                M2 = np.zeros((M2_main.shape[0] + 2, M2_main.shape[1] + 2))
                M2[2:, 2:] = M2_main

            # add columns to L matrices
            if penalty != 'integral':
                L0 = np.hstack((np.zeros((A_re.shape[1] - 2, 2)), L0))
                L1 = np.hstack((np.zeros((A_re.shape[1] - 2, 2)), L1))
                L2 = np.hstack((np.zeros((A_re.shape[1] - 2, 2)), L2))

        # convert reg_ord to list
        if type(reg_ord) == int:
            ord_idx = reg_ord
            reg_ord = np.zeros(3)
            reg_ord[ord_idx] = 1

        # get penalty matrices
        if penalty in ['integral', 'cholesky']:
            L2_base = [M0, M1, M2]
        elif penalty == 'discrete':
            L2_base = [L.T @ L for L in [L0, L1, L2]]

        # create L1 penalty vector
        L1_vec = np.ones(A_re.shape[1]) * np.pi ** 0.5 / epsilon * L1_penalty
        if dist_info['dist_type'] == 'series':
            L1_vec[0:2] = 0  # inductor and high-frequency resistor are not penalized

        # format hl_beta and lambda_0 for each regularization order
        if type(hl_beta) in (float, int, np.float64):
            hl_beta = np.array([hl_beta] * 3)
        else:
            hl_beta = np.array(hl_beta)
        a_list = hl_beta / 2
        if penalty == 'integral':
            b_list = 0.5 * (2 * a_list - 2) / lambda_0
            hyper_as = np.array([np.ones(A_re.shape[1]) * a for a in a_list])
            hyper_bs = np.array([np.ones(A_re.shape[1]) * b for b in b_list])
            hyper_lambda0s = (2 * hyper_as - 2) / (2 * hyper_bs)
        else:
            b_list = 0.5 * (2 * a_list - 1) / lambda_0
            hyper_as = np.array([np.ones(A_re.shape[1]) * a for a in a_list])
            hyper_bs = np.array([np.ones(A_re.shape[1]) * b for b in b_list])
            hyper_lambda0s = (2 * hyper_as - 1) / (2 * hyper_bs)
        hyper_hl_betas = 2 * hyper_as
        # print(hyper_lambda0s)
        # print(hyper_hl_betas)

        if type(alpha_a) in (float, int):
            alpha_a = 3 * [alpha_a]
        if type(hl_beta_a) in (float, int):
            hl_beta_a = 3 * [hl_beta_a]
        if type(sb) in (float, int):
            sb = 3 * [sb]

        # Hyper-lambda fit
        # --------------------------
        if hyper_lambda:
            self._iter_history = []
            iter = 0
            if x0 is not None:
                coef = x0
            else:
                coef = np.zeros(A_re.shape[1]) + 1e-6

            lam_vectors = [np.ones(A_re.shape[1]) * lambda_0] * 3

            # lam_vec[0:2] = 0
            lam_matrices = [np.diag(lam_vec ** 0.5) for lam_vec in lam_vectors]
            # lam_step = np.zeros_like(lam_vectors[0])

            dZ_re = np.ones(A_re.shape[1])

            L2_mat = np.zeros_like(L2_base[0])
            for L2b, lam_mat, frac in zip(L2_base, lam_matrices, reg_ord):
                if frac > 0:
                    L2_mat += frac * (lam_mat @ L2b @ lam_mat)
            P = WA_re.T @ WA_re + WA_im.T @ WA_im + L2_mat
            q = (-WA_re.T @ WT_re - WA_im.T @ WT_im + L1_vec)
            cost = 0.5 * coef.T @ P @ coef + q.T @ coef

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
                    dZ_raw = B @ prev_coef
                    # scale by tau spacing to get dZ'/dlnt
                    dlnt = np.mean(np.diff(np.log(tau)))
                    dZ_raw /= (dlnt / 0.23026)
                    # dZ_raw /= np.mean(dZ_raw)  #TEMP
                    dZ_re[2:] = (np.abs(dZ_raw)) ** dZ_power
                    # for stability, dZ_re must not be 0
                    dZ_re[np.abs(dZ_re < 1e-8)] = 1e-8

                if hyper_b and iter > 0:
                    for n, (lam_vec, frac, sbn) in enumerate(zip(lam_vectors, reg_ord, sb)):
                        # if hyper_b_converged[n]==False:
                        if frac > 0:
                            prev_b = hyper_bs[n].copy()
                            hb = self._hyper_b(lam_vec, hyper_as[n], sbn)
                            # for stability, b must be >0
                            hb[hb < 1e-8] = 1e-8
                            hyper_bs[n] = hb
                            hyper_lambda0s[n] = (2 * hyper_as[n] - 2) / hyper_bs[n]
                        # print(hyper_bs[n])
                        # if np.mean((hyper_bs[n]-prev_b)/prev_b) < 1e-3:
                        # hyper_b_converged[n] = True

                if hyper_a and iter > 0:
                    for n, (lam_vec, frac, aa, ba) in enumerate(zip(lam_vectors, reg_ord, alpha_a, hl_beta_a)):
                        # if hyper_b_converged[n]:
                        if frac > 0:
                            hyper_as[n] = np.ones(len(hyper_bs[n])) * self._hyper_a(lam_vec, hyper_bs[n], aa, ba)
                            # print(hyper_as[n])
                            hyper_lambda0s[n] = (2 * hyper_as[n] - 2) / hyper_bs[n]
                            hyper_hl_betas[n] = 2 * hyper_as[n]

                """Correct phase offset"""
                if correct_phase_offset and iter > 0:
                    # get fitted Zphz
                    Z_pred = A_re @ prev_coef + 1j * A_im @ prev_coef
                    Zphz_pred = np.angle(Z_pred, deg=True)
                    # get current Zphz
                    Zphz = np.angle(Z, deg=True)
                    # estimate Zphz error variance
                    Zphz_var = np.var(Zphz - Zphz_pred)

                    # print('Var:',Zphz_var)

                    # optimize phase_offsets
                    def fun(x, step_indices, Zphz_exp, Zphz_pred, lambda_phz):
                        offset_vec = np.zeros(len(frequencies))
                        Zphz_adj = Zphz_exp.copy()[::-1]

                        for i, (idx, offset) in enumerate(zip(step_indices[:-1], x)):
                            offset_vec[idx:step_indices[i + 1]] = offset
                            Zphz_adj[idx:step_indices[i + 1]] += offset

                        # resid = Zphz_adj - Zphz_pred[::-1]
                        # return resid

                        # Zphz residual sum of squares
                        cost = 0.5 * np.sum((Zphz_adj - Zphz_pred[::-1]) ** 2) / Zphz_var
                        # exponential prior on offsets
                        cost += lambda_phz * np.sum(np.abs(x))

                        return cost

                    result = minimize(fun, x0=phase_offsets,
                                      args=(step_indices, Zphz_exp, Zphz_pred, lambda_phz))  # ,method='Nelder-Mead')
                    # result = least_squares(fun,x0=np.zeros(len(step_indices)+1),args=(step_indices,Zphz,Zphz_pred,lambda_phz),method='lm')
                    # print(result)
                    # print('cost:',fun(result['x'],))

                    # get offsets
                    phase_offsets = result['x']
                    # print(phase_offsets)
                    offset_vec = np.zeros(len(frequencies))
                    Zphz_adj = Zphz_exp.copy()[::-1]

                    for i, (idx, offset) in enumerate(zip(step_indices[:-1], phase_offsets)):
                        offset_vec[idx:step_indices[i + 1]] = offset
                        Zphz_adj[idx:step_indices[i + 1]] += offset

                    Zphz_adj = Zphz_adj[::-1]
                    offset_vec = offset_vec[::-1]

                    # overwrite Z_adj with adjusted phase
                    Zmod = (Z * Z.conjugate()) ** 0.5
                    Z_adj = Zmod * np.cos(np.deg2rad(Zphz_adj)) + 1j * Zmod * np.sin(np.deg2rad(Zphz_adj))

                    # handle scaling and weighting
                    if dist_info['dist_type'] == 'series':
                        # use impedance for series distributions (e.g. DRT)
                        target_adj = Z_adj.copy()
                    elif dist_info['dist_type'] == 'parallel':
                        # for parallel distributions, must fit admittance for linearity
                        target_adj = 1 / Z_adj.copy()

                    target_adj *= target_scaled / target

                    WT_re = W_re @ target_adj.real
                    WT_im = W_im @ target_adj.imag

                # solve for lambda
                if penalty in ('discrete', 'cholesky'):
                    if hl_solution == 'analytic':
                        if hl_fbeta is not None:
                            for n, (Ln, frac, hlam0, hhl_beta) in enumerate(
                                    zip([L0, L1, L2], reg_ord, hyper_lambda0s, hyper_hl_betas)):
                                if frac > 0:
                                    lam_vectors[n] = self._hyper_lambda_fbeta(Ln, prev_coef / dZ_re,
                                                                              dist_info['dist_type'], hl_fbeta=hl_fbeta,
                                                                              lambda_0=lambda_0)
                        else:
                            for n, (Ln, frac, hlam0, hhl_beta) in enumerate(
                                    zip([L0, L1, L2], reg_ord, hyper_lambda0s, hyper_hl_betas)):
                                if frac > 0:
                                    lam_vectors[n] = self._hyper_lambda_discrete(Ln, prev_coef / dZ_re,
                                                                                 dist_info['dist_type'],
                                                                                 hl_beta=hhl_beta[2:],
                                                                                 lambda_0=hlam0[2:])

                    elif hl_solution == 'lm':
                        zeta = (hl_beta - 1) / lambda_0

                        def jac(x, L, coef):
                            # off-diagonal terms are zero
                            diag = (L @ coef) ** 2 + zeta - (hl_beta - 1) / x
                            return np.diag(diag)

                        def fun(x, L, coef):
                            return ((L @ coef) ** 2 + zeta) * x - (hl_beta - 1) * np.log(x)

                        for n, (Ln, frac) in enumerate(zip([L0, L1, L2], reg_ord)):
                            if dist_info['dist_type'] == 'series':
                                start = 2
                            else:
                                start = 0

                            if frac > 0:
                                result = least_squares(fun, prev_lam[start:], jac=jac, method='lm',
                                                       xtol=lambda_0 * 1e-3, args=([Ln, coef]), max_nfev=100)
                                lam_vectors[n][start:] = result['x']

                elif penalty == 'integral':
                    if hl_solution == 'analytic':
                        # print('iter',iter)
                        # D = np.diag(dZ_re**(-1))
                        for n, (L2b, lam_mat, frac, hlam0, hhl_beta) in enumerate(
                                zip(L2_base, lam_matrices, reg_ord, hyper_lambda0s, hyper_hl_betas)):
                            if frac > 0:
                                # lam_vectors[n] = self._hyper_lambda_integral(L2b,prev_coef,D@lam_mat,hl_beta=hl_beta,lambda_0=lambda_0)
                                if n == 0:
                                    factor = 100
                                elif n == 1:
                                    factor = 10
                                else:
                                    factor = 1
                                lv = self._hyper_lambda_integral(L2b, factor * prev_coef / dZ_re, lam_mat,
                                                                 hl_beta=hhl_beta, lambda_0=hlam0)
                                # handle numerical instabilities that may arise for large lambda_0 and small hl_beta
                                lv[lv <= 0] = 1e-15
                                lam_vectors[n] = lv
                            # print(n,lam_vectors[n])

                # lam_vec[lam_vec < 0] = 0
                # print(lam_vectors)
                lam_matrices = [np.diag(lam_vec ** 0.5) for lam_vec in lam_vectors]
                L2_mat = np.zeros_like(L2_base[0])
                D = np.diag(dZ_re ** (-1))
                for L2b, lam_mat, frac in zip(L2_base, lam_matrices, reg_ord):
                    if frac > 0:
                        L2_mat += frac * (D @ lam_mat @ L2b @ lam_mat @ D)

                # P = (WA_re.T@WA_re + WA_im.T@WA_im + L2_mat)
                # q = (-WA_re.T@WZ_re + WA_im.T@WZ_im + L1_vec)
                # cost = 0.5*coef.T@P@coef + q.T@coef
                # print('cost after hyper lambda:',cost)

                # optimize coef
                result = self._convex_opt(part, WT_re, WT_im, WA_re, WA_im, L2_mat, L1_vec, nonneg)
                coef = np.array(list(result['x']))

                P = (WA_re.T @ WA_re + WA_im.T @ WA_im + L2_mat)
                q = (-WA_re.T @ WT_re - WA_im.T @ WT_im + L1_vec)
                cost = 0.5 * coef.T @ P @ coef + q.T @ coef
                # for frac,lam_vec,ha,hb in zip(reg_ord,lam_vectors,hyper_as,hyper_bs):
                # cost += frac*np.sum((hb*lam_vec - (ha-1)*np.log(lam_vec)))
                # print('cost after coef optimization:',cost)

                self._iter_history.append(
                    {'lambda_vectors': lam_vectors.copy(), 'coef': coef.copy(), 'fun': result['primal objective'],
                     'cost': cost, 'result': result, 'dZ_re': dZ_re.copy(),
                     'hyper_bs': hyper_bs.copy(), 'hyper_lambda0s': hyper_lambda0s.copy(),
                     'hyper_hl_betas': hyper_hl_betas.copy()})
                if correct_phase_offset:
                    self._iter_history[-1]['phase_offsets'] = phase_offsets.copy()
                    self._iter_history[-1]['offset_vec'] = offset_vec.copy()
                    self._iter_history[-1]['Zphz_adj'] = Zphz_adj.copy()
                    self._iter_history[-1]['Z_adj'] = Z_adj.copy()

                # check for convergence
                coef_delta = (coef - prev_coef) / prev_coef
                # If inductance not fitted, set inductance delta to zero (inductance goes to random number)
                if self.fit_inductance == False or part == 'real':
                    coef_delta[1] = 0
                # print(np.mean(np.abs(coef_delta)))
                if np.mean(np.abs(coef_delta)) < xtol:
                    break
                elif iter == max_iter - 1:
                    warnings.warn(f'Hyperparametric solution did not converge within {max_iter} iterations')

                iter += 1

            # cost_hist = [h['cost'] for h in self._iter_history]
            # if cost > np.min(cost_hist):
            # best_idx = np.argmin(cost_hist)
            # warnings.warn('Final log-likelihood is not the minimum. Reverting to iteration {}'.format(best_idx+1))
            # self.opt_result_ = self._iter_history[best_idx]['result']
            # self.coef_ = self._iter_history[best_idx]['coef'].copy()
            # self.lambda_vectors_ = self._iter_history[best_idx]['lambda_vectors'].copy()
            # self.cost_ = cost.copy()
            # else:

            self.distribution_fits[dist_name] = {'opt_result': result, 'coef': coef.copy(),
                                                 'lambda_vectors': lam_vectors.copy(), 'cost': cost.copy()}

        # Hyper-weights fit
        # --------------------------
        elif hyper_weights:
            self._iter_history = []
            iter = 0
            # initialize coef and dZ
            coef = np.zeros(A_re.shape[1]) + 1e-6
            dZ_re = np.ones(A_re.shape[1])

            # get w_bar
            wbar = self._format_weights(frequencies, target_scaled, hw_wbar, part)
            # initialize weights at w_bar
            weights = wbar
            # print('wbar:',wbar)
            # wbar_re = np.real(wbar)
            # wbar_im = np.imag(wbar)

            lam_vectors = [np.ones(A_re.shape[1]) * lambda_0] * 3
            lam_matrices = [np.diag(lam_vec ** 0.5) for lam_vec in lam_vectors]
            L2_mat = np.zeros_like(L2_base[0])
            D = np.diag(dZ_re ** (-1))
            for L2b, lam_mat, frac in zip(L2_base, lam_matrices, reg_ord):
                if frac > 0:
                    L2_mat += frac * (D @ lam_mat @ L2b @ lam_mat @ D)

            while iter < max_iter:

                prev_coef = coef.copy()
                prev_weights = weights.copy()

                # calculate new weights
                if iter > 0:
                    weights = self._hyper_weights(coef, A_re, A_im, target_scaled, hw_beta, wbar)

                # apply weights to A and Z
                W_re = np.diag(np.real(weights))
                W_im = np.diag(np.imag(weights))
                WA_re = W_re @ A_re
                WA_im = W_im @ A_im
                WT_re = W_re @ target_scaled.real
                WT_im = W_im @ target_scaled.imag

                if dZ and iter > 0:
                    dZ_raw = B @ prev_coef
                    # scale by tau spacing to get dZ'/dlnt
                    dlnt = np.mean(np.diff(np.log(tau)))
                    dZ_raw /= (dlnt / 0.23026)
                    dZ_re[2:] = (np.abs(dZ_raw)) ** dZ_power
                    # for stability, dZ_re must not be 0
                    dZ_re[np.abs(dZ_re < 1e-8)] = 1e-8

                # optimize coef
                result = self._convex_opt(part, WT_re, WT_im, WA_re, WA_im, L2_mat, L1_vec, nonneg)
                coef = np.array(list(result['x']))

                P = (WA_re.T @ WA_re + WA_im.T @ WA_im + L2_mat)
                q = (-WA_re.T @ WT_re - WA_im.T @ WT_im + L1_vec)
                cost = 0.5 * coef.T @ P @ coef + q.T @ coef

                self._iter_history.append(
                    {'weights': weights.copy(), 'coef': coef.copy(), 'fun': result['primal objective'], 'cost': cost,
                     'result': result, 'dZ_re': dZ_re.copy()
                     })

                # check for convergence
                coef_delta = (coef - prev_coef) / prev_coef
                # If inductance not fitted, set inductance delta to zero (inductance goes to random number)
                if self.fit_inductance == False:  # or part=='real':
                    coef_delta[1] = 0
                # print(np.mean(np.abs(coef_delta)))
                if np.mean(np.abs(coef_delta)) < xtol:
                    break
                elif iter == max_iter - 1:
                    warnings.warn(f'Hyperparametric solution did not converge within {max_iter} iterations')

                iter += 1

            self.distribution_fits[dist_name] = {'opt_result': result, 'coef': coef.copy(), 'weights': weights.copy(),
                                                 'cost': cost.copy()}

        # Ordinary ridge fit
        # --------------------------
        else:
            # create L2 penalty matrix
            lam_vectors = [np.ones(A_re.shape[1]) * lambda_0] * 3
            lam_matrices = [np.diag(lam_vec ** 0.5) for lam_vec in lam_vectors]
            L2_mat = np.zeros_like(L2_base[0])
            for L2b, lam_mat, frac in zip(L2_base, lam_matrices, reg_ord):
                if frac > 0:
                    L2_mat += frac * (lam_mat @ L2b @ lam_mat)

            result = self._convex_opt(part, WT_re, WT_im, WA_re, WA_im, L2_mat, L1_vec, nonneg)
            coef = np.array(list(result['x']))
            P = (WA_re.T @ WA_re + WA_im.T @ WA_im + L2_mat)
            q = (-WA_re.T @ WT_re - WA_im.T @ WT_im + L1_vec)
            cost = 0.5 * coef.T @ P @ coef + q.T @ coef

            self.distribution_fits[dist_name] = {'opt_result': result, 'coef': coef.copy(), 'cost': cost.copy()}

        # If fitted imag part only, optimize high-frequency resistance (offset)
        if part == 'imag' and dist_info['dist_type'] == 'series':
            basis_coef = self.distribution_fits[dist_name]['coef'][2:]
            Zr_pred = A_re[:, 2:] @ basis_coef

            def res_fun(x):
                return Zr_pred + x - target_scaled.real

            result = least_squares(res_fun, x0=target_scaled.real[0])
            self.distribution_fits[dist_name]['coef'][0] = result['x'][0]
        # If fitted real part only, optimize inductance
        elif part == 'real' and dist_info['dist_type'] == 'series' and self.fit_inductance:
            basis_coef = self.distribution_fits[dist_name]['coef'][2:]
            Zi_pred = A_im[:, 2:] @ basis_coef

            def res_fun(x):
                return Zi_pred + frequencies * 2 * np.pi * 1e-4 * x - target_scaled.imag

            result = least_squares(res_fun, x0=1e-7)
            self.distribution_fits[dist_name]['coef'][1] = result['x'][0]

        # rescale coefficients if scaling applied to A or Z
        if scale_Z:
            self.distribution_fits[dist_name]['scaled_coef'] = self.distribution_fits[dist_name]['coef'].copy()
            # since the target for ridge_fit changes based on dist_type, rescaling coefficients always works in the same direction
            self.distribution_fits[dist_name]['coef'] = self._rescale_coef(self.distribution_fits[dist_name]['coef'],
                                                                           dist_info['dist_type'])

        # rescale the inductance
        if dist_info['dist_type'] == 'series':
            self.distribution_fits[dist_name]['coef'][1] *= 1e-4

        # If inductance not fitted, set inductance to zero to avoid confusion (goes to random number)
        if dist_info['dist_type'] == 'series' and not self.fit_inductance:
            self.distribution_fits[dist_name]['coef'][1] = 0

        # pull R_inf and inductance out of coef
        if dist_info['dist_type'] == 'series':
            self.R_inf = self.distribution_fits[dist_name]['coef'][0]
            self.inductance = self.distribution_fits[dist_name]['coef'][1]
            self.distribution_fits[dist_name]['coef'] = self.distribution_fits[dist_name]['coef'][2:]
        else:
            # ridge_fit cannot optimize R_inf and inductance for parallel distributions
            self.R_inf = 0
            self.inductance = 0

        self.fit_type = 'ridge'

    def ridge_ReImCV(self, frequencies, Z, lambdas=np.logspace(-10, 5, 31), **kw):
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

        for i, lam in enumerate(lambdas):
            self.ridge_fit(frequencies, Z, part='real', lambda_0=lam, **kw)
            Zi_pred = np.imag(self.predict_Z(frequencies))

            self.ridge_fit(frequencies, Z, part='imag', lambda_0=lam, **kw)
            Zr_pred = np.real(self.predict_Z(frequencies))

            Zr_err = np.sum((Z.real - Zr_pred) ** 2)
            Zi_err = np.sum((Z.imag - Zi_pred) ** 2)
            recv[i] = Zr_err
            imcv[i] = Zi_err

        # get lambda_0 that minimizes total CV error
        totcv = recv + imcv
        min_lam = lambdas[np.argmin(totcv)]
        if min_lam == np.min(lambdas) or min_lam == np.max(lambdas):
            warnings.warn(
                'Optimal lambda_0 {} determined by Re-Im CV is at the boundary of the evaluated range. Re-run with an expanded lambda_0 range to obtain an accurate estimate of the optimal lambda_0.'.format(
                    min_lam))

        # store DataFrame of results
        self.cv_result = pd.DataFrame(np.array([lambdas, recv, imcv, totcv]).T,
                                      columns=['lambda', 'recv', 'imcv', 'totcv'])

        return min_lam

    def _hyper_lambda_discrete(self, L, coef, dist_type, hl_beta=2.5, lambda_0=1):
        Lx2 = (L @ coef) ** 2
        # lam = np.ones(self.A_re.shape[1]) #*lambda_0
        lam = 1 / (Lx2 / (hl_beta - 1) + 1 / lambda_0)
        if dist_type == 'series':
            # add ones for R_ohmic and inductance
            lam = np.hstack(([1, 1], lam))
        return lam

    def _hyper_lambda_fbeta(self, L, coef, dist_type, hl_fbeta, lambda_0):
        Lx2 = (L @ coef) ** 2
        Lxmax = np.max(Lx2)
        # lam = np.ones(self.A_re.shape[1]) #*lambda_0
        lam = lambda_0 / (Lx2 / (Lxmax * hl_fbeta) + 1)
        if dist_type == 'series':
            # add ones for R_ohmic and inductance
            lam = np.hstack(([1, 1], lam))
        return lam

    # def _grad_lambda_discrete(self,frequencies,coef,lam_vec,reg_ord,beta=2.5,lambda_0=1):
    # L = construct_L(frequencies,tau=self.tau,basis=self.basis,epsilon=self.epsilon,order=reg_ord)
    # Lx2 = (L@coef)**2
    # zeta = (beta-1)/lambda_0
    # grad = Lx2 + zeta - (beta-1)/lam_vec[2:]
    # return grad

    def _hyper_lambda_integral(self, M, coef, lam_mat, hl_beta=2.5, lambda_0=1):
        X = np.diag(coef)
        xlm = X @ lam_mat @ M @ X
        xlm = xlm - np.diag(np.diagonal(xlm))
        C = np.sum(xlm, axis=0)

        a = hl_beta / 2
        b = 0.5 * (2 * a - 2) / lambda_0
        d = coef ** 2 * np.diagonal(M) + 2 * b
        lam = (C ** 2 - np.sign(C) * C * np.sqrt(4 * d * (2 * a - 2) + C ** 2) + 2 * d * (2 * a - 2)) / (2 * d ** 2)
        return lam

    def _hyper_b(self, lam, a, sb):
        K = self.A_re.shape[1] - 2
        b = 0.25 * (np.sqrt(16 * a * K * sb ** 2 + 4 * sb ** 4 * np.sum(lam) ** 2) - 2 * np.sum(
            lam) * sb ** 2)  # b ~ normal(0,sb)
        # b = 0.25*(np.sqrt(16*a*sb**2 + 4*sb**4*lam**2) -2*lam*sb**2) # b_k ~ normal(0,sb))
        return b

    def _hyper_a(self, lam, b, alpha_a, beta_a):
        # a is a vector
        # def obj_fun(ak,bk,lk,alpha_a,beta_a):
        # # return -2*ak*np.log(bk*lk) + 2*loggamma(ak) + 4*np.log(ak-2) + 2*((ak-2)*sa)**(-2) # 1/(ak-2) ~ normal(0,sa)
        # return -2*ak*np.log(bk*lk) + 2*loggamma(ak) + 2*beta_a*(ak-1) - 2*(alpha_a-1)*np.log(ak-1) # ak-1 ~ gamma(alpha_a,beta_a)

        # a = np.zeros_like(lam)
        # a = [minimize_scalar(obj_fun,method='bounded',bounds=(1,5),args=(bk,lk,alpha_a,beta_a))['x'] for bk,lk in zip(b,lam)]

        # a is a scalar
        def obj_fun(a, b, lam, alpha_a, beta_a):
            return -2 * a * np.sum(np.log(b * lam)) + 2 * loggamma(a) + 2 * beta_a * (a - 1) - 2 * (
                    alpha_a - 1) * np.log(a - 1)  # a-1 ~ gamma(alpha_a,beta_a)

        a = minimize_scalar(obj_fun, method='bounded', bounds=(1, 5), args=(b, lam, alpha_a, beta_a))['x']

        return a

    def _hyper_weights(self, coef, A_re, A_im, Z, hw_beta, wbar):
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
        zeta_re = hw_beta / np.real(wbar)
        zeta_im = hw_beta / np.imag(wbar)

        # calculate residuals
        Z_pred = A_re @ coef + 1j * A_im @ coef
        resid = Z - Z_pred
        r_re = np.real(resid)
        r_im = np.imag(resid)
        # calculate MAP weights
        w_re = (np.real(wbar) - 1 / zeta_re) / (r_re ** 2 / zeta_re + 1)
        w_im = (np.imag(wbar) - 1 / zeta_im) / (r_im ** 2 / zeta_im + 1)

        # print(resid[8:13])
        # print(w_im[8:13])
        # print(wbar[8:13])

        return w_re + 1j * w_im

    def _convex_opt(self, part, WZ_re, WZ_im, WA_re, WA_im, L2_mat, L1_vec, nonneg):
        if part == 'both':
            P = cvxopt.matrix((WA_re.T @ WA_re + WA_im.T @ WA_im + L2_mat).T)
            q = cvxopt.matrix((-WA_re.T @ WZ_re - WA_im.T @ WZ_im + L1_vec).T)
        elif part == 'real':
            P = cvxopt.matrix((WA_re.T @ WA_re + L2_mat).T)
            q = cvxopt.matrix((-WA_re.T @ WZ_re + L1_vec).T)
        else:
            P = cvxopt.matrix((WA_im.T @ WA_im + L2_mat).T)
            q = cvxopt.matrix((-WA_im.T @ WZ_im + L1_vec).T)

        G = cvxopt.matrix(-np.eye(WA_re.shape[1]))
        if nonneg:
            # coefficients must be >= 0
            h = cvxopt.matrix(np.zeros(WA_re.shape[1]))
        else:
            # coefficients can be positive or negative
            h = 10 * np.ones(WA_re.shape[1])
            # HFR and inductance must still be nonnegative
            h[0:2] = 0
            # print(h)
            h = cvxopt.matrix(h)
        # print('neg')

        return cvxopt.solvers.qp(P, q, G, h)

    # ===============================================
    # Methods for fitting hierarchical Bayesian model
    # ===============================================
    def fit(self, frequencies, Z, part='both', scale_Z=True, nonneg=False, outliers=False, check_outliers=True,
            init_from_ridge=False, ridge_kw={},
            sigma_min=0.002, inductance_scale=1, outlier_lambda=None,
            mode='optimize', random_seed=1234,
            # Optimization control
            max_iter=50000,
            # Sampling control
            warmup=200, samples=200, chains=2,
            add_stan_data={}, model_str=None,
            fitY=False, SA=False, SASY=False):
        """
        Fit the defined distribution(s) using the calibrated hierarchical Bayesian model.
        Model may be fitted either via optimization (maximum a posteriori estimate) or HMC sampling.

        Parameters:
        -----------
        frequencies: array
            Measured frequencies
        Z: complex array
            Measured (complex) impedance values. Must have same length as frequencies
        part: str, optional (default: 'both')
            Which part of the impedance data to fit. Options: 'both', 'real', 'imag'
        scale_Z: bool, optional (default: True)
            If True, scale impedance by the factor sqrt(N)/std(|Z|) to normalize for magnitude and sample size.
            Model is calibrated for scaled data.
        nonneg: bool, optional (default: False)
            If True, constrain the DRT to non-negative values
        outliers: bool or str, optional (default: False)
            If True, use outlier-robust error model.
            If 'auto', check for likely outliers and automatically determine whether to use outlier model.
            If False, use regular error model.
            Set to True if you know your data contains outliers, set to False if you know it doesn't, or
            set to 'auto' to let the system make the determination (this is especially useful for batch fits
            in which some spectra contain outliers and others don't).
        check_outliers: bool, optional (default: True)
            If True, check for likely outliers after performing MAP fit.
            If False, skip outlier check.
            May find possible outliers that were not identified by initial check when outliers='auto'
            due to better estimate of error structure from MAP fit.
        init_from_ridge: bool, optional (default: False)
            If True, use the hyperparametric ridge solution to initialize the Bayesian fit.
            Only valid for single-distribution fits
        ridge_kw: dict, optional (default: {})
            Keyword arguments to pass to ridge_fit if init_from_ridge==True.
        sigma_min: float, optional (default: 0.002)
            Impedance error floor. This is necessary to avoid sampling/optimization errors.
            Values smaller than the default (0.002) may enable slightly closer fits of very clean data,
            but may also result in sampling/optimization errors that yield unexpected results.
        inductance_scale: float, optional (default: 1)
            Scale (std of normal prior) of the inductance. Lower values will constrain the inductance,
            which may be helpful if the estimated inductance is anomalously large (this may occur if your
            measured impedance data does not extend to high frequencies, i.e. 1e5-1e6 Hz)
        outlier_lambda: float, optional (default: None)
            Lambda parameter (inverse scale) of the exponential prior on the outlier error contribution.
            Smaller values will make it easier for points to be flagged as outliers.
            Sampling and optimization modes may require different values. Defaults to 10 for both
        mode: str, optional (default: 'optimize')
            Solution mode. If 'optimize', use the L-BFGS-B algorithm to obtain the MAP estimate of the solution.
            If 'sample', use HMC sampling to estimate the posterior distribution.
        random_seed: int, optional (default: 1234)
            Random seed for optimization or sampling
        max_iter: int, optional (default: 50000)
            Maximum number of iterations to allow the optimizer to perform. Only used when mode='optimize'.
        warmup: int, optional (default: 200)
            Number of warmup or burn-in samples to draw. These samples will not contribute to the esimate of the
            posterior distribution. Only used when mode='sample'.
        sample: int, optional (default: 200)
            Number of samples to draw after warm-up. These samples constitute the estimate of the posterior
            distribution. The total number of samples will be sample*chains. Only used when mode='sample'.
        chains: int, optional (default: 2)
            Number of chains to sample in parallel. Only used when mode='sample'.
        add_stan_data: dict, optional (default: {})
            Additional parameters to provide as data inputs to the Stan model. Can be used to adjust hyperparameters
            if you know what you're doing and/or are developing a model.
        model_str: str, optional (default: None)
            String to specify different model file. For troubleshooting and model development
        fitY: bool, optional (default: False)
            If True, fit admittance. Only valid when fitting a parallel distribution.
        SA, SASY: bool
            For testing only
        """
        # perform scaling and weighting and get A and B matrices
        frequencies, Z_scaled, WZ_re, WZ_im, W_re, W_im, dist_mat = self._prep_matrices(frequencies, Z, part,
                                                                                        weights=None, dZ=False,
                                                                                        scale_Z=scale_Z,
                                                                                        penalty='discrete',
                                                                                        fit_type='map')

        # get initial fit
        if init_from_ridge:
            if len(self.distributions) > 1:
                raise ValueError('Ridge initialization can only be performed for single-distribution fits')
            else:
                init = self._get_init_from_ridge(frequencies, Z, mode, nonneg=nonneg, outliers=outliers,
                                                 inductance_scale=inductance_scale, ridge_kw=ridge_kw)
                self._init_params = init()
        else:
            init = 'random'

        # check for outliers. Use more stringent threshold to avoid false positives
        if outliers == 'auto':
            # If initial ridge fit performed, use existing ridge fit. Otherwise perform new ridge fit
            if init_from_ridge:
                use_existing_fit = True
            else:
                use_existing_fit = False
            outlier_idx = self.check_outliers(frequencies, Z, threshold=4, use_existing_fit=use_existing_fit,
                                              **ridge_kw)

            if len(outlier_idx) > 0:
                outliers = True
                warnings.warn(
                    'Identified likely outliers at indices {}, f={} Hz. An outlier-robust error model will be used. To disable this behavior, pass outliers=False.'.format(
                        outlier_idx, frequencies[outlier_idx]))
            else:
                outliers = False

        # load stan model
        if model_str is None:
            model, model_str = self._get_stan_model(nonneg, outliers, False, None, fitY, SA)
        else:
            model = load_pickle(os.path.join(script_dir, 'stan_model_files', model_str))
        self.stan_model_name = model_str
        model_type = model_str.split('_')[0]
        if model_type == 'Series-Parallel' and nonneg == False:
            warnings.warn('For mixed series-parallel models, it is highly recommended to set nonnneg_drt=True')

        # prepare data for stan model
        dat = self._prep_stan_data(frequencies, Z_scaled, part, model_type, dist_mat, outliers, sigma_min,
                                   mode=mode,
                                   inductance_scale=inductance_scale, outlier_lambda=outlier_lambda,
                                   fitY=fitY, SA=SA, SASY=SASY)

        # add user-supplied stan inputs
        dat.update(add_stan_data)

        if outliers:
            # outlier models have been updated to use N instead of 2*N
            # other models will be updated later
            dat['N'] = len(frequencies)
        self._stan_input = dat.copy()

        # optimize or sample posterior
        if mode == 'optimize':
            self._opt_result = model.optimizing(dat, iter=max_iter, seed=random_seed, init=init)
        elif mode == 'sample':
            self._sample_result = model.sampling(dat, warmup=warmup, iter=warmup + samples, chains=chains,
                                                 seed=random_seed,
                                                 init=init,
                                                 control={'adapt_delta': 0.9, 'adapt_t0': 10})

        # extract coefficients
        self.distribution_fits = {}
        self.error_fit = {}
        if model_type in ['Series', 'Parallel']:
            dist_name = [k for k, v in self.distributions.items() if v['dist_type'] == model_type.lower()][0]
            dist_type = self.distributions[dist_name]['dist_type']
            self.distribution_fits[dist_name] = {'coef': self._extract_parameter('x', dist_type, mode)}

        elif model_type == 'Series-Parallel':
            for dist_name, dist_info in self.distributions.items():
                if dist_info['dist_type'] == 'series':
                    self.distribution_fits[dist_name] = {
                        'coef': self._extract_parameter('xs', dist_info['dist_type'], mode)}
                elif dist_info['dist_type'] == 'parallel':
                    self.distribution_fits[dist_name] = {
                        'coef': self._extract_parameter('xp', dist_info['dist_type'], mode)}

        elif model_type == 'Series-2Parallel':
            for dist_name, dist_info in self.distributions.items():
                if dist_info['dist_type'] == 'series':
                    self.distribution_fits[dist_name] = {
                        'coef': self._extract_parameter('xs', dist_info['dist_type'], mode)}
                elif dist_info['dist_type'] == 'parallel':
                    order = dist_info['order']
                    self.distribution_fits[dist_name] = {
                        'coef': self._extract_parameter(f'xp{order}', dist_info['dist_type'], mode)}

        elif model_type == 'MultiDist':
            """Placeholder"""
            for dist_name, dist_info in self.distributions.items():
                if dist_info['kernel'] == 'DRT':
                    self.distribution_fits[dist_name] = {
                        'coef': self._extract_parameter('xs', dist_info['dist_type'], mode)}
                elif dist_info['kernel'] == 'DDT':
                    self.distribution_fits[dist_name] = {
                        'coef': self._extract_parameter('xp', dist_info['dist_type'], mode)}
        if not fitY:
            self.R_inf = self._extract_parameter('Rinf', 'series', mode)
            self.inductance = self._extract_parameter('induc', 'series', mode)
        else:
            self.R_inf = 0
            self.inductance = 0

        # store error structure parameters
        # scaled parameters
        self.error_fit['sigma_min'] = self._rescale_coef(sigma_min, 'series')
        for param in ['sigma_tot', 'sigma_res']:
            self.error_fit[param] = self._extract_parameter(param, 'series', mode)
        # unscaled parameters
        for param in ['alpha_prop', 'alpha_re', 'alpha_im']:
            self.error_fit[param] = self._extract_parameter(param, None, mode)
        # outlier contribution
        if outliers == True:
            self.error_fit['sigma_out'] = self._extract_parameter('sigma_out', 'series', mode)

        if mode == 'optimize':
            self.fit_type = 'map'
        elif mode == 'sample':
            self.fit_type = 'bayes'

        # check if outliers were missed
        if outliers == False and check_outliers:
            outlier_idx = self.check_outliers(frequencies, Z, threshold=3.5, use_existing_fit=True)
            if len(outlier_idx) > 0:
                warnings.warn(
                    'Possible outliers were identified at indices {}, f={} Hz. Check the residuals and consider re-running with outliers=True'.format(
                        outlier_idx, frequencies[outlier_idx]))

    def drift_map_fit(self, frequencies, Z, times, drift_model='x1', part='both', scale_Z=True, init_from_ridge=False,
                      nonneg=False, outliers=False,
                      model_str=None, init_values=None,
                      sigma_min=0.002, max_iter=50000, random_seed=1234, inductance_scale=1, outlier_lambda=5,
                      ridge_kw={}, add_stan_data={},
                      fitY=False, Yscale=1, SA=False, SASY=False):
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
            model, model_str = self._get_stan_model(nonneg, outliers, True, drift_model, fitY, SA)
        else:
            model = load_pickle(os.path.join(script_dir, 'stan_model_files', model_str))
        self.stan_model_name = model_str
        model_type = model_str.split('_')[0]
        if model_type == 'Series-Parallel' and nonneg == False:
            warnings.warn('For mixed series-parallel models, it is highly recommended to set nonnneg_drt=True')

        # perform scaling and weighting and get A and B matrices
        frequencies, Z_scaled, WZ_re, WZ_im, W_re, W_im, dist_mat = self._prep_matrices(frequencies, Z, part,
                                                                                        weights=None, dZ=False,
                                                                                        scale_Z=scale_Z,
                                                                                        penalty='discrete',
                                                                                        fit_type='map', sort_desc=False)

        # prepare data for stan model
        Z_scaled *= Yscale
        dat = self._prep_stan_data(frequencies, Z_scaled, part, model_type, dist_mat, outliers, sigma_min,
                                   mode='optimize',
                                   inductance_scale=inductance_scale, outlier_lambda=outlier_lambda,
                                   fitY=fitY, SA=SA, SASY=SASY)

        dat['time'] = times
        if drift_model in ('x1', 'x2'):
            dat['min_tau_x1'] = 200
            dat['max_tau_x1'] = 10000
            dat['min_tau_x2'] = 500
            dat['max_tau_x2'] = 10000
        elif drift_model == 'dx':
            dat['min_tau_dx'] = 200
            dat['max_tau_dx'] = 10000
        elif drift_model in ('RQ', 'RQ-lin', 'RQ-from-final', 'RQ-lin-from-final'):
            # constrain time-dependent ZARC to fall within basis_tau
            min_taus = []
            max_taus = []
            for dist, info in self.distributions.items():
                min_taus.append(np.min(info['tau']))
                max_taus.append(np.max(info['tau']))

            dat['min_tau_rq'] = np.min(min_taus)  # 1/(2*np.pi*frequencies))
            dat['max_tau_rq'] = np.max(max_taus)  # 1/(2*np.pi*frequencies))

            # set k boundaries for nonlinear model
            if drift_model in ('RQ', 'RQ-from-final'):
                dat['min_k'] = 1e-4
                dat['max_k'] = 1
        elif drift_model == 'dx-lin':
            dat['dx_scale_fixed'] = 1

        # add user-supplied stan data
        dat.update(add_stan_data)

        self._stan_input = dat.copy()

        # set initial parameter values
        if init_values is not None:
            iv = init_values
        else:
            if drift_model in ('x1', 'x2'):
                iv = {'log_tau_x1': np.log(500), 'log_tau_x2': np.log(500), 'log_tau_Rinf': np.log(600)}
            elif drift_model == 'dx':
                iv = {'log_tau_dx': np.log(1000)}
            elif drift_model == 'dx-lin':
                iv = {'delta_Rinf': 0}
            elif drift_model in ('RQ', 'RQ-lin', 'RQ-from-final', 'RQ-lin-from-final'):
                iv = {'phi_rq': 0.5, 'delta_Rinf': 0}
            ## TEMP ##
            # iv['tau_rq'] = 0.1
            # iv['log_tau_rq'] = np.log(iv['tau_rq'])

        if outliers:
            # initialize sigma_out near zero
            iv['sigma_out_raw'] = np.zeros(2 * len(Z)) + 0.1

        if init_from_ridge:
            # get initial fit from ridge solution
            if len(self.distributions) > 1:
                raise ValueError('Ridge initialization can only be performed for single-distribution fits')
            else:
                init = self._get_init_from_ridge(frequencies, Z, mode, nonneg=nonneg, outliers=outliers,
                                                 inductance_scale=inductance_scale, ridge_kw=ridge_kw)
                self._init_params = init()
                dist_name = list(self.distributions.keys())[0]
                iv_ridge = init()
                iv_ridge['x0'] = iv_ridge['x'].copy()
                iv_ridge['Rinf0_raw'] = iv_ridge['Rinf_raw']
                if drift_model in ('x1', 'x2'):
                    iv_ridge['x1'] = iv_ridge['x'].copy()
                    iv_ridge['x2'] = np.zeros(len(iv_ridge['x0'])) + 1e-3
                elif drift_model == 'dx-lin':
                    iv_ridge['dx'] = np.zeros(len(iv_ridge['x0'])) + 1e-3
                elif drift_model in ('RQ-from-final', 'RQ-lin-from-final'):
                    iv_ridge['x1'] = iv_ridge['x'].copy()
                    iv_ridge['Rinf1_raw'] = iv_ridge['Rinf_raw']

                iv.update(iv_ridge)

        def init():
            return iv

        # print(iv)

        # optimize posterior
        self._opt_result = model.optimizing(dat, iter=max_iter, seed=random_seed, init=init)

        # extract coefficients
        self.distribution_fits = {}
        self.error_fit = {}
        self.drift_offsets = {}
        if model_type in ['Series', 'Parallel']:
            # get distribution name and type
            dist_name = [k for k, v in self.distributions.items() if v['dist_type'] == model_type.lower()][0]
            dist_type = self.distributions[dist_name]['dist_type']
            if drift_model in ('RQ-from-final', 'RQ-lin-from-final'):
                # get final coefficients
                self.distribution_fits[dist_name] = {'x1': self._rescale_coef(self._opt_result['x1'], dist_type)}
                self.distribution_fits[dist_name]['x1'] *= Yscale
            else:
                # get initial coefficients
                self.distribution_fits[dist_name] = {'x0': self._rescale_coef(self._opt_result['x0'], dist_type)}
                self.distribution_fits[dist_name]['x0'] *= Yscale

            if drift_model in ('x1', 'x2'):
                # number of drift processes
                num_proc = int(drift_model[-1])
                for n in range(1, num_proc + 1):
                    # get coefficients for each drift process
                    self.distribution_fits[dist_name][f'x{n}'] = self._rescale_coef(self._opt_result[f'x{n}'],
                                                                                    dist_type)
                    self.distribution_fits[dist_name][f'tau_x{n}'] = np.exp(self._opt_result[f'log_tau_x{n}'])

                    self.distribution_fits[dist_name][f'x{n}'] *= Yscale

                if fitY:
                    self.R_inf = 0
                    self.inductance = 0
                else:
                    # store R_inf vector for all times
                    self.R_inf = self._rescale_coef(self._opt_result['Rinf'], 'series')
                    # R_inf is time-dependent - store parameters
                    self.drift_offsets['Rinf_0'] = self._rescale_coef(100 * self._opt_result['Rinf0_raw'], 'series')
                    self.drift_offsets['delta_Rinf'] = self._rescale_coef(100 * self._opt_result['dRinf_raw'], 'series')
                    self.drift_offsets['tau_Rinf'] = np.exp(self._opt_result['log_tau_Rinf'])
                    # inductance is constant
                    self.inductance = self._rescale_coef(self._opt_result['induc'], 'series')
            elif drift_model == 'dx':
                # get dx info
                self.distribution_fits[dist_name]['dx'] = self._rescale_coef(self._opt_result['dx'], dist_type)
                self.distribution_fits[dist_name]['dx'] *= Yscale

                self.distribution_fits[dist_name]['tau_dx'] = np.exp(self._opt_result['log_tau_dx'])

                if fitY:
                    self.R_inf = 0
                    self.inductance = 0
                else:
                    # store R_inf vector for all times
                    self.R_inf = self._rescale_coef(self._opt_result['Rinf'], 'series')
                    # R_inf is time-dependent - store parameters
                    self.drift_offsets['Rinf_0'] = self._rescale_coef(100 * self._opt_result['Rinf0_raw'], 'series')
                    self.drift_offsets['delta_Rinf'] = self._rescale_coef(100 * self._opt_result['dRinf_raw'], 'series')
                    self.drift_offsets['tau_Rinf'] = np.exp(self._opt_result['log_tau_Rinf'])
                    # inductance is constant
                    self.inductance = self._rescale_coef(self._opt_result['induc'], 'series')
            elif drift_model == 'dx-lin':
                # get dx
                self.distribution_fits[dist_name]['dx'] = self._rescale_coef(self._opt_result['dx'], dist_type)
                self.distribution_fits[dist_name]['dx'] *= dat['dx_scale_fixed']
                self.distribution_fits[dist_name]['dx'] *= Yscale
                # slope for F(t)
                self.distribution_fits[dist_name]['m_Ft'] = 1 / np.max(times)

                if fitY:
                    self.R_inf = 0
                    self.inductance = 0
                else:
                    # store R_inf vector for all times
                    self.R_inf = self._rescale_coef(self._opt_result['Rinf'], 'series')
                    # R_inf is time-dependent - store parameters
                    self.drift_offsets['Rinf_0'] = self._rescale_coef(100 * self._opt_result['Rinf0_raw'], 'series')
                    self.drift_offsets['delta_Rinf'] = self._rescale_coef(self._opt_result['delta_Rinf'], 'series')
                    # inductance is constant
                    self.inductance = self._rescale_coef(self._opt_result['induc'], 'series')

            elif drift_model in ('RQ', 'RQ-lin', 'RQ-from-final', 'RQ-lin-from-final'):
                # extract time-dependent ZARC (RQ) parameters
                self.distribution_fits[dist_name]['R_rq'] = self._rescale_coef(self._opt_result['R_rq'], dist_type)
                self.distribution_fits[dist_name]['phi_rq'] = self._opt_result['phi_rq']
                self.distribution_fits[dist_name]['tau_rq'] = self._opt_result['tau_rq']

                if drift_model in ('RQ', 'RQ-from-final'):
                    self.distribution_fits[dist_name]['k_d'] = np.exp(self._opt_result['ln_k'])
                elif drift_model == 'RQ-lin':
                    # slope for F(t)
                    self.distribution_fits[dist_name]['m_Ft'] = 1 / np.max(times)
                elif drift_model == 'RQ-lin-from-final':
                    # slope for F(t)
                    self.distribution_fits[dist_name]['t_i'] = np.min(times)
                    self.distribution_fits[dist_name]['t_f'] = np.max(times)

                if fitY:
                    self.R_inf = 0
                    self.inductance = 0
                else:
                    # store R_inf vector for all times
                    self.R_inf = self._rescale_coef(self._opt_result['Rinf'], 'series')
                    # R_inf is time-dependent - store parameters
                    if drift_model in ('RQ-from-final', 'RQ-lin-from-final'):
                        self.drift_offsets['Rinf_1'] = self._rescale_coef(100 * self._opt_result['Rinf1_raw'], 'series')
                    else:
                        self.drift_offsets['Rinf_0'] = self._rescale_coef(100 * self._opt_result['Rinf0_raw'], 'series')
                    self.drift_offsets['delta_Rinf'] = self._rescale_coef(self._opt_result['delta_Rinf'], 'series')
                    # inductance is constant
                    self.inductance = self._rescale_coef(self._opt_result['induc'], 'series')

        # store error structure parameters
        # scaled parameters
        self.error_fit['sigma_min'] = self._rescale_coef(sigma_min, 'series')
        for param in ['sigma_tot', 'sigma_res']:
            self.error_fit[param] = self._rescale_coef(self._opt_result[param], 'series')
        # unscaled parameters
        for param in ['alpha_prop', 'alpha_re', 'alpha_im']:
            self.error_fit[param] = self._opt_result[param]
        # outlier contribution
        if outliers:
            self.error_fit['sigma_out'] = self._rescale_coef(self._opt_result['sigma_out'], 'series')

        self.fit_type = 'map-drift'

    def _get_stan_model(self, nonneg, outliers, drift, drift_model, fitY, SA):
        """Get the appropriate Stan model for the distributions. Called by map_fit and bayes_fit methods

		Parameters:
		-----------
		nonneg: bool
			If True, constrain DRT to be non-negative. If False, allow negative DRT values
		outliers: bool
			If True, enable outlier detection. If False, do not include outlier error contribution in error model.
		"""
        num_series = len([name for name, info in self.distributions.items() if info['dist_type'] == 'series'])
        num_par = len([name for name, info in self.distributions.items() if info['dist_type'] == 'parallel'])

        if num_series == 1 and num_par == 0:
            model_str = 'Series'
        elif num_series == 0 and num_par == 1:
            model_str = 'Parallel'
        elif num_series == 1 and num_par == 1:
            model_str = 'Series-Parallel'
        elif num_series == 1 and num_par == 2:
            model_str = 'Series-2Parallel'
        else:
            model_str = 'MultiDist'
            warnings.warn('The MultiDist model will handle an arbitrary number of series and/or parallel distributions, but the computational performance and accuracy are suboptimal. \
			Hard-coding your own model will most likely yield better results.')

        if nonneg and num_series >= 1:
            model_str += '_pos'

        if drift:
            model_str += f'_drift-{drift_model}'  # '_drift_2dx'

        if fitY:
            if num_par >= 1 and num_series == 0:
                model_str += '_fitY'
            else:
                raise ValueError('fitY=True is only valid for parallel distributions')

        if SA:
            model_str += '_SA'

        if outliers:
            model_str += '_outliers'

        model_str += '_StanModel.pkl'
        # print(model_str)
        model = load_pickle(os.path.join(script_dir, 'stan_model_files', model_str))

        return model, model_str

    def _get_init_from_ridge(self, frequencies, Z, mode, nonneg, outliers, inductance_scale, ridge_kw):
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
		inductance_scale: float
			Scale (std of normal prior) of the inductance
		ridge_kw: dict

		"""
        dist_name = list(self.distributions.keys())[0]
        dist_type = self.distributions[dist_name]['dist_type']

        # default ridge_fit settings
        ridge_defaults = dict(preset='Huang', nonneg=nonneg)
        # update with any user-upplied settings - may overwrite defaults
        ridge_defaults.update(ridge_kw)
        # get initial parameter values from ridge fit
        self.ridge_fit(frequencies, Z, **ridge_defaults)

        # scale the coefficients
        coef = self.distribution_fits[dist_name]['coef']
        if dist_type == 'series':
            x_star = coef / self._Z_scale
        elif dist_type == 'parallel':
            x_star = coef * self._Z_scale
        iv = {'x': x_star}

        # estimate distribution complexity and initialize upsilon accordingly
        q = self._calc_q(mode, reg_strength=[1, 1, 1])
        if mode == 'optimize':
            # Only initialize upsilon for optimization. Let Stan initialize randomly for sampling
            iv['ups_raw'] = q * 0.5 / 0.15

        # input other parameters
        iv['Rinf'] = self.R_inf / self._Z_scale
        iv['Rinf_raw'] = iv['Rinf'] / 100
        iv['induc'] = self.inductance / self._Z_scale
        if iv['induc'] <= 0:
            iv['induc'] = 1e-10
        iv['induc_raw'] = iv['induc'] / inductance_scale

        if outliers is True or outliers is 'auto':
            # identify likely outliers. Use less stringent threshold to avoid missing outliers
            outlier_idx = self.check_outliers(frequencies, Z, threshold=3, use_existing_fit=True)
            if outliers is True or len(outlier_idx) > 0:
                # initialize sigma_out_raw
                sigma_out_raw = np.zeros(len(Z)) + 0.1
                sigma_out_raw[outlier_idx] = 1
                iv['sigma_out_raw'] = sigma_out_raw

        def init_func():
            return iv

        return init_func

    def _prep_stan_data(self, frequencies, Z, part, model_type, dist_mat, outliers, sigma_min, mode, inductance_scale,
                        outlier_lambda, fitY, SA, SASY):
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

        if outlier_lambda is None:
            if mode == 'optimize':
                outlier_lambda = 10
            elif mode == 'sample':
                outlier_lambda = 10

        if model_type in ['Series', 'Parallel']:
            dist_name = [k for k, v in self.distributions.items() if v['dist_type'] == model_type.lower()][0]
            matrices = dist_mat[dist_name]

            if part == 'both':
                Z_stack = np.concatenate((Z.real, Z.imag))
                A_stack = np.concatenate((matrices['A_re'], matrices['A_im']))
            else:
                Z_stack = getattr(Z, part)
                A_stack = matrices['A_{}'.format(part[:2])]

            if mode == 'sample':
                ups_alpha = 1
                ups_beta = 0.1
                L0 = matrices['L0']
                L1 = matrices['L1']
                L2 = 0.75 * matrices['L2']

            elif mode == 'optimize':
                ups_alpha = 0.05
                ups_beta = 0.1
                L0 = 1.5 * 0.24 * matrices['L0']
                L1 = 1.5 * 0.16 * matrices['L1']
                L2 = 1.5 * 0.08 * matrices['L2']  # consider 0.08-->0.09

            dat = {'N': 2 * len(frequencies),
                   'freq': frequencies,
                   'K': A_stack.shape[1],
                   'A': A_stack,
                   'Z': Z_stack,
                   'N_tilde': 2 * len(frequencies),
                   'A_tilde': A_stack,
                   'freq_tilde': frequencies,
                   'L0': L0,
                   'L1': L1,
                   'L2': L2,
                   'sigma_min': sigma_min,
                   'ups_alpha': ups_alpha,
                   'ups_beta': ups_beta,
                   'induc_scale': inductance_scale
                   }

            if SA:
                A_re = matrices['A_re']
                A_im = matrices['A_im']
                # using modulus of row sums for scale
                re_sum = np.sum(A_re, axis=1)
                im_sum = np.sum(A_im, axis=1)

                # mod = np.concatenate((re_sum,im_sum))
                # matrices['SA_re'] = np.diag(1/re_sum)@A_re
                # matrices['SA_im'] = np.diag(1/im_sum)@A_im

                Y = 1 / Z
                Ymod = np.real((Y * Y.conjugate()) ** 0.5)

                mod = Ymod  # np.sqrt(re_sum**2 + im_sum**2)
                S = np.diag(1 / mod)
                matrices['SA_re'] = S @ A_re
                matrices['SA_im'] = S @ A_im
                S_inv = np.diag(np.concatenate([mod, mod]))
                # print(mod)

                # using modulus of Y for scale
                # mod = Ymod
                # S = np.diag(1/mod)
                # S_inv = np.diag(np.concatenate((mod,mod)))
                # # SY = S@Y.real + 1j*(S@Y.imag)
                # matrices['SA_re'] = S@A_re
                # matrices['SA_im'] = S@A_im

                if part == 'both':
                    SA_stack = np.concatenate((matrices['SA_re'], matrices['SA_im']))
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
                dat[
                    'S_inv'] = S_inv  # np.vstack([np.hstack([S_inv,np.zeros_like(S_inv)]), np.hstack([np.zeros_like(S_inv),S_inv])])
                dat['S'] = dat['S_inv']  # S not actually used (FIX)

            if fitY:
                # Z input will be ignored. Just need to add Y input
                Y = 1 / Z
                Ymod = np.real((Y * Y.conjugate()) ** 0.5)
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
                    S = np.diag(1 / mod)
                    S_inv = np.diag(mod)
                    SY = S @ Y.real + 1j * (S @ Y.imag)
                    matrices['SA_re'] = S @ A_re
                    matrices['SA_im'] = S @ A_im

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

                    if part == 'both':
                        A_stack = np.concatenate((matrices['SA_re'], matrices['SA_im']))
                    else:
                        A_stack = matrices['SA_{}'.format(part[:2])]

                    dat['A'] = A_stack

                if part == 'both':
                    Y_stack = np.concatenate((Y.real, Y.imag))
                # A_stack = np.concatenate((matrices['A_re'],matrices['A_im']))
                else:
                    Y_stack = getattr(Y, part)
                # A_stack = matrices['A_{}'.format(part[:2])]

                dat['Y'] = Y_stack

            # if model_type=='Parallel':
            # # scaling factor for parallel coefficients
            # dat['x_scale'] = self._Z_scale**2
            # # print('x_scale:',dat['x_scale'])

            if outliers:
                dat['sigma_out_lambda'] = outlier_lambda

                if mode == 'optimize':
                    dat['sigma_out_alpha'] = 2
                elif mode == 'sample':
                    dat['sigma_out_alpha'] = 5
                dat['sigma_out_beta'] = 1
            # if mode=='optimize':
            # dat['so_invscale'] = 5
            # elif mode=='sample':
            # dat['so_invscale'] = 10

        elif model_type == 'Series-Parallel':
            if len(self.distributions) > 2:
                raise ValueError('Too many distributions for Series-Parallel model')
            ser_name = [k for k, v in self.distributions.items() if v['dist_type'] == 'series'][0]
            par_name = [k for k, v in self.distributions.items() if v['dist_type'] == 'parallel'][0]
            ser_mat = dist_mat[ser_name]
            par_mat = dist_mat[par_name]

            if part == 'both':
                Z_stack = np.concatenate((Z.real, Z.imag))
                As_stack = np.concatenate((ser_mat['A_re'], ser_mat['A_im']))
                Ap_stack = np.concatenate((par_mat['A_re'], par_mat['A_im']))
            elif part == 'real':
                Z_stack = np.concatenate((Z.real, np.zeros_like(Z.imag)))
                As_stack = np.concatenate((ser_mat['A_re'], np.zeros_like(ser_mat['A_im'])))
                Ap_stack = np.concatenate((par_mat['A_re'], np.zeros_like(par_mat['A_im'])))
            elif part == 'imag':
                Z_stack = np.concatenate((np.zeros_like(Z.real), Z.imag))
                As_stack = np.concatenate((np.zeros_like(ser_mat['A_re']), ser_mat['A_im']))
                Ap_stack = np.concatenate((np.zeros_like(par_mat['A_re']), par_mat['A_im']))

            if mode == 'sample':
                ups_alpha = 1
                ups_beta = 0.1
                L0s = ser_mat['L0']
                L1s = ser_mat['L1']
                L2s = 0.75 * ser_mat['L2']
                L0p = par_mat['L0']
                L1p = par_mat['L1']
                L2p = 0.75 * par_mat['L2']
                x_sum_invscale = 1

            elif mode == 'optimize':
                ups_alpha = 0.05
                ups_beta = 0.1
                L0s = 1.5 * 0.24 * ser_mat['L0']
                L1s = 1.5 * 0.16 * ser_mat['L1']
                L2s = 1.5 * 0.08 * ser_mat['L2']  # consider 0.08-->0.1
                L0p = 1.5 * 0.36 * par_mat['L0']
                L1p = 1.5 * 0.16 * par_mat['L1']
                L2p = 1.5 * 0.08 * par_mat['L2']  # consider 0.08-->0.1
                x_sum_invscale = 0.

            dat = {'N': 2 * len(frequencies),
                   'freq': frequencies,
                   'Ks': As_stack.shape[1],
                   'Kp': Ap_stack.shape[1],
                   'As': As_stack,
                   'Ap': Ap_stack,
                   'Z': Z_stack,
                   'N_tilde': 2 * len(frequencies),
                   'As_tilde': As_stack,
                   'Ap_tilde': Ap_stack,
                   'freq_tilde': frequencies,
                   'L0s': L0s,
                   'L1s': L1s,
                   'L2s': L2s,
                   'L0p': L0p,
                   'L1p': L1p,
                   'L2p': L2p,
                   'sigma_min': sigma_min,
                   'ups_alpha': ups_alpha,
                   'ups_beta': ups_beta,
                   'induc_scale': inductance_scale,
                   'x_sum_invscale': x_sum_invscale,
                   'xp_scale': self.distributions[par_name].get('x_scale', 1)  # self._Z_scale**2,
                   }

            if outliers:
                dat['so_invscale'] = outlier_lambda
            # if mode=='optimize':
            # dat['so_invscale'] = 5
            # elif mode=='sample':
            # dat['so_invscale'] = 10

        elif model_type == 'Series-2Parallel':
            ser_name = [k for k, v in self.distributions.items() if v['dist_type'] == 'series'][0]
            par_names = sorted([k for k, v in self.distributions.items() if v['dist_type'] == 'parallel'])
            par1_name = par_names[0]
            par2_name = par_names[1]
            # store order for parallel distributions
            self.distributions[par1_name]['order'] = 1
            self.distributions[par2_name]['order'] = 2
            ser_mat = dist_mat[ser_name]
            par1_mat = dist_mat[par1_name]
            par2_mat = dist_mat[par2_name]

            if part == 'both':
                Z_stack = np.concatenate((Z.real, Z.imag))
                As_stack = np.concatenate((ser_mat['A_re'], ser_mat['A_im']))
                Ap1_stack = np.concatenate((par1_mat['A_re'], par1_mat['A_im']))
                Ap2_stack = np.concatenate((par2_mat['A_re'], par2_mat['A_im']))
            else:
                Z_stack = getattr(Z, part)
                As_stack = ser_mat['A_{}'.format(part[:2])]
                Ap1_stack = par1_mat['A_{}'.format(part[:2])]
                Ap2_stack = par2_mat['A_{}'.format(part[:2])]

            if mode == 'sample':
                ups_alpha = 1
                ups_beta = 0.1
                L0s = ser_mat['L0']
                L1s = ser_mat['L1']
                L2s = 0.75 * ser_mat['L2']
                L0p1 = par1_mat['L0']
                L1p1 = par1_mat['L1']
                L2p1 = 0.75 * par1_mat['L2']
                L0p2 = par2_mat['L0']
                L1p2 = par2_mat['L1']
                L2p2 = 0.75 * par2_mat['L2']
                x_sum_invscale = 0.1

            elif mode == 'optimize':
                ups_alpha = 0.05
                ups_beta = 0.1
                L0s = 1.5 * 0.24 * ser_mat['L0']
                L1s = 1.5 * 0.16 * ser_mat['L1']
                L2s = 1.5 * 0.08 * ser_mat['L2']
                L0p1 = 1.5 * 0.36 * par1_mat['L0']
                L1p1 = 1.5 * 0.16 * par1_mat['L1']
                L2p1 = 1.5 * 0.08 * par1_mat['L2']
                L0p2 = 1.5 * 0.36 * par2_mat['L0']
                L1p2 = 1.5 * 0.16 * par2_mat['L1']
                L2p2 = 1.5 * 0.08 * par2_mat['L2']
                x_sum_invscale = 0.

            dat = {'N': 2 * len(frequencies),
                   'freq': frequencies,
                   'Ks': As_stack.shape[1],
                   'Kp1': Ap1_stack.shape[1],
                   'Kp2': Ap2_stack.shape[1],
                   'As': As_stack,
                   'Ap1': Ap1_stack,
                   'Ap2': Ap2_stack,
                   'Z': Z_stack,
                   'N_tilde': 2 * len(frequencies),
                   'As_tilde': As_stack,
                   'Ap1_tilde': Ap1_stack,
                   'Ap2_tilde': Ap2_stack,
                   'freq_tilde': frequencies,
                   'L0s': L0s,
                   'L1s': L1s,
                   'L2s': L2s,
                   'L0p1': L0p1,
                   'L1p1': L1p1,
                   'L2p1': L2p1,
                   'L0p2': L0p2,
                   'L1p2': L1p2,
                   'L2p2': L2p2,
                   'sigma_min': sigma_min,
                   'ups_alpha': ups_alpha,
                   'ups_beta': ups_beta,
                   'induc_scale': inductance_scale,
                   'x_sum_invscale': x_sum_invscale,
                   'xp1_scale': self.distributions[par1_name].get('x_scale', 1),  # self._Z_scale**2
                   'xp2_scale': self.distributions[par2_name].get('x_scale', 1)  # self._Z_scale**2
                   }

            if outliers:
                dat['so_invscale'] = outlier_lambda
            # if mode=='optimize':
            # dat['so_invscale'] = 5
            # elif mode=='sample':
            # dat['so_invscale'] = 10

        elif model_type == 'MultiDist':
            """placeholder"""
            drt_name = [k for k, v in self.distributions.items() if v['kernel'] == 'DRT'][0]
            ddt_name = [k for k, v in self.distributions.items() if v['kernel'] == 'DDT'][0]
            drt_mat = dist_mat[drt_name]
            ddt_mat = dist_mat[ddt_name]

            if part == 'both':
                Z_stack = np.concatenate((Z.real, Z.imag))
                Ar_stack = np.concatenate((drt_mat['A_re'], drt_mat['A_im']))
                Ad_stack = np.concatenate((ddt_mat['A_re'], ddt_mat['A_im']))
            else:
                Z_stack = getattr(Z, part)
                Ar_stack = drt_mat['A_{}'.format(part[:2])]
                Ad_stack = ddt_mat['A_{}'.format(part[:2])]

            if mode == 'sample':
                ups_alpha = 1
                ups_beta = 0.1
                L0r = drt_mat['L0']
                L1r = drt_mat['L1']
                L2r = 0.5 * drt_mat['L2']
                L0d = ddt_mat['L0']
                L1d = ddt_mat['L1']
                L2d = 0.5 * ddt_mat['L2']
                x_sum_invscale = 0

            elif mode == 'optimize':
                ups_alpha = 0.05
                ups_beta = 0.1
                L0r = 1.5 * 0.24 * drt_mat['L0']
                L1r = 1.5 * 0.16 * drt_mat['L1']
                L2r = 1.5 * 0.08 * drt_mat['L2']
                L0d = 1.5 * 0.24 * ddt_mat['L0']
                L1d = 1.5 * 0.16 * ddt_mat['L1']
                L2d = 1.5 * 0.08 * ddt_mat['L2']
                x_sum_invscale = 0.

            dat = {'N': 2 * len(frequencies),
                   'freq': frequencies,
                   'Ms': 1,
                   'Mp': 1,
                   'Ks': [Ar_stack.shape[1]],
                   'Kp': [Ad_stack.shape[1]],
                   'As': Ar_stack,
                   'Ap': Ad_stack,
                   'Z': Z_stack,
                   # 'N_tilde':2*len(frequencies),
                   # 'Ar_tilde':Ar_stack,
                   # 'Ad_tilde':Ad_stack,
                   # 'freq_tilde': frequencies,
                   'L0s': L0r,
                   'L1s': L1r,
                   'L2s': L2r,
                   'L0p': L0d,
                   'L1p': L1d,
                   'L2p': L2d,
                   'sigma_min': sigma_min,
                   'ups_alpha': ups_alpha,
                   'ups_beta': ups_beta,
                   'induc_scale': inductance_scale,
                   'x_sum_invscale': x_sum_invscale
                   }

            if outliers:
                dat['so_invscale'] = outlier_lambda
            # if mode=='optimize':
            # dat['so_invscale'] = 5
            # elif mode=='sample':
            # dat['so_invscale'] = 10

        return dat

    # ===============================================
    # Matrix preparation & other preprocessing methods
    # ===============================================
    def _prep_matrices(self, frequencies, Z, part, weights, dZ, scale_Z, penalty, fit_type, sort_desc=True):
        if len(frequencies) != len(Z):
            raise ValueError("Length of frequencies and Z must be equal")

        if type(Z) != np.ndarray:
            Z = np.array(Z)

        if type(frequencies) != np.ndarray:
            frequencies = np.array(frequencies)

        # sort by descending frequency
        if sort_desc:
            sort_idx = np.argsort(frequencies)[::-1]
            frequencies = frequencies[sort_idx]
            Z = Z[sort_idx]

        # store Z
        self.Z_train = Z

        # check if we need to recalculate matrices due to change in self.distributions
        # A change in self.distributions may not be caught by set_distributions because
        # set_distributions is only called if we do self.distributions = new_distributions.
        # If we simply update one parameter for a distribution (e.g. self.distributions['DRT']['epsilon'] = 5),
        # set_distributions never gets called, and thus _recalc_mat does not get reset to True.
        if check_equality(self.distributions, self._cached_distributions) == False:
            self._recalc_mat = True
            self.f_pred = None

        # check if we need to recalculate matrices due to change in measurement frequencies
        freq_subset = False
        if np.min(rel_round(self.f_train, 10) == rel_round(frequencies, 10)) == False:
            # if frequencies are a subset of f_train, we can use submatrices of the existing A matrices
            # instead of recalculating
            if np.min([rel_round(f, 10) in rel_round(self.f_train, 10) for f in frequencies]) == True:
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
            Z = self._scale_Z(Z, fit_type)
            if type(weights) in (list, np.ndarray):
                weights = np.array(weights) / self._Z_scale
        else:
            self._Z_scale = 1

        # create weight matrices
        weights = self._format_weights(frequencies, Z, weights, part)
        W_re = np.diag(np.real(weights))
        W_im = np.diag(np.imag(weights))

        # print(self._recalc_mat)

        # set up matrices for each distribution
        dist_mat = {}  # transient dict to hold matrices for fit, which may be scaled
        for name, info in self.distributions.items():
            temp_dist = deepcopy(self.distributions)
            # set tau and epsilon
            if info.get('basis_freq', self.basis_freq) is None:
                # If basis_freq not specified, go one decade beyond measured frequency range in each direction
                # by default, use 10 ppd for tau spacing regardless of input frequency spacing
                tmin = np.log10(1 / (2 * np.pi * np.max(frequencies))) - 1
                tmax = np.log10(1 / (2 * np.pi * np.min(frequencies))) + 1
                num_decades = tmax - tmin
                tau = np.logspace(tmin, tmax, int(10 * num_decades + 1))
            else:
                tau = 1 / (2 * np.pi * info.get('basis_freq', self.basis_freq))
            temp_dist[name]['tau'] = tau

            if info.get('epsilon', self.epsilon) is None:
                # if neither dist-specific nor class-level epsilon is specified
                dlnt = np.mean(np.diff(np.log(tau)))
                temp_dist[name]['epsilon'] = (1 / dlnt)
            elif info.get('epsilon', None) is None:
                # if dist-specific epsilon not specified, but class-level epsilon is present
                temp_dist[name]['epsilon'] = self.epsilon
            epsilon = temp_dist[name].get('epsilon', self.epsilon)

            # update distributions without overwriting self._recalc_mat
            recalc_mat = self._recalc_mat
            self.distributions = temp_dist
            self._recalc_mat = recalc_mat

            # create A matrices
            if self._recalc_mat == False:
                if freq_subset:
                    # frequencies is a subset of f_train - no need to recalc
                    # print('freq in f_train')
                    f_index = np.array(
                        [np.where(rel_round(self.f_train, 10) == rel_round(f, 10))[0][0] for f in frequencies])
                    A_re = self.distribution_matrices[name]['A_re'][f_index, :].copy()
                    A_im = self.distribution_matrices[name]['A_im'][f_index, :].copy()
                else:
                    A_re = self.distribution_matrices[name]['A_re'].copy()
                    A_im = self.distribution_matrices[name]['A_im'].copy()

                if dZ and info['kernel'] == 'DRT':
                    if 'B' in self.distribution_matrices[name].keys():
                        B = self.distribution_matrices[name]['B'].copy()
                    else:
                        # Z_re differentiation matrix (B@coef gives approx dZ'/dlnt at each basis tau)
                        tau_diff = np.mean(np.diff(np.log(tau)))
                        B_start = np.exp(np.log(tau[0]) - tau_diff / 2)
                        B_end = np.exp(np.log(tau[-1]) + tau_diff / 2)
                        B_tau = np.logspace(np.log10(B_start), np.log10(B_end), len(tau) + 1)
                        B_pre = construct_A(1 / (2 * np.pi * B_tau), 'real', tau=tau, basis=self.basis, epsilon=epsilon,
                                            kernel=info['kernel'], dist_type=info['dist_type'],
                                            symmetry=info.get('symmetry', ''),
                                            bc=info.get('bc', ''), ct=info.get('ct', False), k_ct=info.get('k_ct', None)
                                            )
                        B = B_pre[1:, :] - B_pre[:-1, :]
                        self.distribution_matrices[name]['B'] = B
                else:
                    B = None

            if self._recalc_mat or 'A_re' not in self.distribution_matrices[name].keys() or 'A_im' not in \
                    self.distribution_matrices[name].keys():
                self.distribution_matrices[name]['A_re'] = construct_A(frequencies, 'real', tau=tau, basis=self.basis,
                                                                       fit_inductance=self.fit_inductance,
                                                                       epsilon=epsilon,
                                                                       kernel=info['kernel'],
                                                                       dist_type=info['dist_type'],
                                                                       symmetry=info.get('symmetry', ''),
                                                                       bc=info.get('bc', ''),
                                                                       ct=info.get('ct', False),
                                                                       k_ct=info.get('k_ct', None)
                                                                       )
                self.distribution_matrices[name]['A_im'] = construct_A(frequencies, 'imag', tau=tau, basis=self.basis,
                                                                       fit_inductance=self.fit_inductance,
                                                                       epsilon=epsilon,
                                                                       kernel=info['kernel'],
                                                                       dist_type=info['dist_type'],
                                                                       symmetry=info.get('symmetry', ''),
                                                                       bc=info.get('bc', ''),
                                                                       ct=info.get('ct', False),
                                                                       k_ct=info.get('k_ct', None)
                                                                       )
                A_re = self.distribution_matrices[name]['A_re'].copy()
                A_im = self.distribution_matrices[name]['A_im'].copy()

                if dZ and info['kernel'] == 'DRT':
                    # Z_re differentiation matrix (B@coef gives approx dZ'/dlnt at each basis tau)
                    tau_diff = np.mean(np.diff(np.log(tau)))
                    B_start = np.exp(np.log(tau[0]) - tau_diff / 2)
                    B_end = np.exp(np.log(tau[-1]) + tau_diff / 2)
                    B_tau = np.logspace(np.log10(B_start), np.log10(B_end), len(tau) + 1)
                    B_pre = construct_A(1 / (2 * np.pi * B_tau), 'real', tau=tau, basis=self.basis, epsilon=epsilon,
                                        kernel=info['kernel'], dist_type=info['dist_type'],
                                        symmetry=info.get('symmetry', ''),
                                        bc=info.get('bc', ''), ct=info.get('ct', False), k_ct=info.get('k_ct', None)
                                        )
                    B = B_pre[1:, :] - B_pre[:-1, :]
                    self.distribution_matrices[name]['B'] = B
                else:
                    B = None

            # apply weights to A
            WA_re = W_re @ A_re
            WA_im = W_im @ A_im

            dist_mat[name] = {}

            # calculate L or M matrices
            if penalty == 'integral':
                dist_mat[name]['M0'] = construct_M(1 / (2 * np.pi * tau), basis=self.basis, order=0, epsilon=epsilon)
                dist_mat[name]['M1'] = construct_M(1 / (2 * np.pi * tau), basis=self.basis, order=1, epsilon=epsilon)
                dist_mat[name]['M2'] = construct_M(1 / (2 * np.pi * tau), basis=self.basis, order=2, epsilon=epsilon)

            elif penalty == 'discrete':
                dist_mat[name]['L0'] = construct_L(1 / (2 * np.pi * tau), tau=tau, basis=self.basis, epsilon=epsilon,
                                                   order=0)
                dist_mat[name]['L1'] = construct_L(1 / (2 * np.pi * tau), tau=tau, basis=self.basis, epsilon=epsilon,
                                                   order=1)
                dist_mat[name]['L2'] = construct_L(1 / (2 * np.pi * tau), tau=tau, basis=self.basis, epsilon=epsilon,
                                                   order=2)

            elif penalty == 'cholesky':
                M0 = construct_M(1 / (2 * np.pi * tau), basis=self.basis, order=0, epsilon=epsilon)
                M1 = construct_M(1 / (2 * np.pi * tau), basis=self.basis, order=1, epsilon=epsilon)
                M2 = construct_M(1 / (2 * np.pi * tau), basis=self.basis, order=2, epsilon=epsilon)

                dist_mat[name]['L0'] = cholesky(
                    M0)  # scipy cholesky gives upper triangular by default, such that M = L.T@L. Then x.T@M@x = x.T@L.T@L@x = ||L@x||^2
                dist_mat[name]['L1'] = cholesky(M1)
                dist_mat[name]['L2'] = cholesky(M2)

                dist_mat[name]['M0'] = M0
                dist_mat[name]['M1'] = M1
                dist_mat[name]['M2'] = M2

            # add L and M matrices to self.distribution_matrices
            self.distribution_matrices[name].update(dist_mat[name])

            # add matrices to transient dist_mat
            dist_mat[name].update({'A_re': A_re, 'A_im': A_im, 'WA_re': WA_re, 'WA_im': WA_im, 'B': B})

        # apply weights to Z
        WZ_re = W_re @ Z.real
        WZ_im = W_im @ Z.imag

        self._recalc_mat = False
        self._cached_distributions = self.distributions.copy()

        return frequencies, Z, WZ_re, WZ_im, W_re, W_im, dist_mat

    def _format_weights(self, frequencies, Z, weights, part):
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
        if weights is None or weights == 'unity':
            weights = np.ones_like(frequencies) * (1 + 1j)
        elif type(weights) == str:
            if weights == 'modulus':
                weights = (1 + 1j) / np.sqrt(np.real(Z * Z.conjugate()))
            elif weights == 'Orazem':
                weights = (1 + 1j) / (np.abs(Z.real) + np.abs(Z.imag))
            elif weights == 'proportional':
                weights = 1 / np.abs(Z.real) + 1j / np.abs(Z.imag)
            elif weights == 'prop_adj':
                Zmod = np.real(Z * Z.conjugate())
                weights = 1 / (np.abs(Z.real) + np.percentile(Zmod, 25)) + 1j / (
                        np.abs(Z.imag) + np.percentile(Zmod, 25))
            else:
                raise ValueError(
                    f"Invalid weights argument {weights}. String options are 'unity', 'modulus', 'proportional', and 'prop_adj'")
        elif type(weights) in (float, int):
            # assign constant value
            weights = np.ones_like(frequencies) * (1 + 1j) * weights
        elif type(weights) == complex:
            # assign constant value
            weights = np.ones_like(frequencies) * weights
        elif len(weights) != len(frequencies):
            raise ValueError("Weights array must match length of data")

        if part == 'both':
            if np.min(np.isreal(weights)) == True:
                # if weights are real, apply to both real and imag parts
                weights = weights + 1j * weights
            else:
                # if weights are complex, leave them
                pass
        elif part == 'real':
            weights = np.real(weights) + 1j * np.ones_like(frequencies)
        elif part == 'imag':
            if np.min(np.isreal(weights)) == True:
                # if weights are real, apply to imag
                weights = np.ones_like(frequencies) + 1j * weights
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

    def _scale_Z(self, Z, fit_type):
        Zmod = (Z * Z.conjugate()) ** 0.5

        # adjust the Z scale for pure parallel distributions
        num_series = len([name for name, info in self.distributions.items() if info['dist_type'] == 'series'])
        num_par = len([name for name, info in self.distributions.items() if info['dist_type'] == 'parallel'])
        if num_par == 1 and num_series == 0 and fit_type != 'ridge':
            Y = 1 / Z
            Ymod = (Y * Y.conjugate()) ** 0.5
            dist_name = [name for name, info in self.distributions.items() if info['dist_type'] == 'parallel'][0]
            info = self.distributions[dist_name]
            if info['kernel'] == 'DDT' and info['symmetry'] == 'planar':
                # scale Z such that the scaled admittance has the desired std
                if info['bc'] == 'transmissive':
                    if fit_type == 'map':
                        Ystar_std = 14
                    elif fit_type == 'bayes':
                        Ystar_std = 14  # 70
                elif info['bc'] == 'blocking':
                    if fit_type == 'map':
                        Ystar_std = 2.4
                    elif fit_type == 'bayes':
                        Ystar_std = 2.4
                self._Z_scale = Ystar_std * np.sqrt(len(Z) / 81) / np.std(Ymod)

            else:
                self._Z_scale = np.std(Zmod) / np.sqrt(len(Z) / 81)
        else:
            # scale by sqrt(n) as suggested by Ishwaran and Rao (doi: 10.1214/009053604000001147)
            # hyperparameters were selected based on spectra with 81 data points - therefore, scale relative to N=81
            self._Z_scale = np.std(Zmod) / np.sqrt(len(Z) / 81)

        return Z / self._Z_scale

    def _rescale_coef(self, coef, dist_type):
        if dist_type == 'series':
            rs_coef = coef * self._Z_scale
        elif dist_type == 'parallel':
            rs_coef = coef / self._Z_scale
        return rs_coef

    def _calc_q(self, mode, distribution_name=None, reg_strength=[1, 1, 1]):
        """
		Calculate distribution complexity
		Parameters:
		-----------
		distribution_name: str, optional (default: None)
			Name of distribution for which to calculate complexity
			If not specified, uses first distribution in self.distributions
		mode: str
			Solver mode. Determines scaling of differentiation matrices
			Options: 'optimize', 'sample'
			If None, regularization strengths are taken from reg_strength argument
		reg_strength: array-like, optional (default: [1,1,1])
			Regularization strength of 0th, 1st, and 2nd derivatives of the distribution, respectively.
			Defaults to [1,1,1] (equal strengths)
		"""
        if distribution_name is None:
            distribution_name = list(self.distributions.keys())[0]
        dist_mat = self.distribution_matrices[distribution_name]
        L0 = dist_mat['L0'].copy()
        L1 = dist_mat['L1'].copy()
        L2 = dist_mat['L2'].copy()

        # if self.fit_type=='ridge':
        x = self.distribution_fits[distribution_name]['coef'].copy()

        x /= self._Z_scale

        # apply scaling to differentiation matrices
        if mode == 'optimize':
            L0 *= 1.5 * 0.24
            L1 *= 1.5 * 0.16
            L2 *= 1.5 * 0.08
        elif mode == 'sample':
            L2 *= 0.75
        # get regularization strengths
        d0, d1, d2 = reg_strength

        q = np.sqrt(d0 * (L0 @ x) ** 2 + d1 * (L1 @ x) ** 2 + d2 * (L2 @ x) ** 2)

        return q

    def _extract_parameter(self, stan_key, dist_type, mode):
        """
        Extract parameter from stan model result
        Parameters
        ----------
        stan_key:
        dist_type
        mode

        Returns
        -------

        """
        if mode == 'optimize':
            if stan_key in ['alpha_prop', 'alpha_re', 'alpha_im']:
                # Error structure coefficients - not scaled
                return self._opt_result[stan_key]
            else:
                # Scale all other parameters
                return self._rescale_coef(self._opt_result[stan_key], dist_type)
        elif mode == 'sample':
            if stan_key in ['alpha_prop', 'alpha_re', 'alpha_im']:
                # Error structure coefficients - not scaled
                return np.mean(self._sample_result[stan_key])
            else:
                return self._rescale_coef(np.mean(self._sample_result[stan_key], axis=0), dist_type)

    def _get_stan_coef_name(self, distribution_name):
        """Get stan model coefficient name for distribution

		Parameters:
		-----------
		distribution_name: str
			Name of distribution
		"""
        dist_type = self.distributions[distribution_name]['dist_type']
        model_type = self.stan_model_name.split('_')[0]
        if model_type in ['Series', 'Parallel']:
            coef_name = 'x'
        elif model_type == 'Series-Parallel':
            if self.distributions[distribution_name]['dist_type'] == 'series':
                coef_name = 'xs'
            elif self.distributions[distribution_name]['dist_type'] == 'parallel':
                coef_name = 'xp'
        elif model_type == 'Series-2Parallel':
            if self.distributions[distribution_name]['dist_type'] == 'series':
                coef_name = 'xs'
            elif self.distributions[distribution_name]['dist_type'] == 'parallel':
                order = self.distributions[distribution_name]['order']
                coef_name = f'xp{order}'

        return coef_name

    def coef_percentile(self, distribution_name, percentile):
        """Get percentile for distribution coefficients

		Parameters:
		-----------
		distribution_name: str
			Name of distribution
		percentile: float
			Percentile (0-100) to calculate
		"""
        if self.fit_type == 'bayes':
            dist_type = self.distributions[distribution_name]['dist_type']
            coef_name = self._get_stan_coef_name(distribution_name)
            coef = np.percentile(self._sample_result[coef_name], percentile, axis=0)
            # rescale coef
            coef = self._rescale_coef(coef, dist_type)
        else:
            raise ValueError('Percentile prediction is only available for bayes_fit')

        return coef

    # ===============================================
    # Prediction
    # ===============================================
    def _get_prediction_matrices(self, frequencies, distributions):

        if self.f_pred is not None:
            pred_mat = {}
            for name in distributions:
                pred_mat[name] = {}
            # check if we need to recalculate A matrices
            freq_subset = False
            if np.min(rel_round(self.f_pred, 10) == rel_round(frequencies, 10)) == False:
                # frequencies differ from f_pred
                if np.min([rel_round(f, 10) in rel_round(self.f_pred, 10) for f in frequencies]) == True:
                    # if frequencies are a subset of f_pred, we can use submatrices of the existing A matrices
                    # instead of calculating new A matrices
                    f_index = np.array(
                        [np.where(rel_round(self.f_pred, 10) == rel_round(f, 10))[0][0] for f in frequencies])

                    for name in distributions:
                        mat = self.prediction_matrices[name]
                        pred_mat[name]['A_re'] = mat['A_re'][f_index, :].copy()
                        pred_mat[name]['A_im'] = mat['A_im'][f_index, :].copy()
                else:
                    # otherwise, we need to calculate A matrices
                    for name in distributions:
                        info = self.distributions[name]
                        tau = self.distributions[name]['tau']
                        epsilon = self.distributions[name]['epsilon']
                        pred_mat[name]['A_re'] = construct_A(frequencies, 'real', tau=tau, basis=self.basis,
                                                             fit_inductance=self.fit_inductance, epsilon=epsilon,
                                                             kernel=info['kernel'], dist_type=info['dist_type'],
                                                             symmetry=info.get('symmetry', ''), bc=info.get('bc', ''),
                                                             ct=info.get('ct', False), k_ct=info.get('k_ct', None)
                                                             )
                        pred_mat[name]['A_im'] = construct_A(frequencies, 'imag', tau=tau, basis=self.basis,
                                                             fit_inductance=self.fit_inductance, epsilon=epsilon,
                                                             kernel=info['kernel'], dist_type=info['dist_type'],
                                                             symmetry=info.get('symmetry', ''), bc=info.get('bc', ''),
                                                             ct=info.get('ct', False), k_ct=info.get('k_ct', None)
                                                             )
                    self.prediction_matrices = pred_mat
                    self.f_pred = frequencies
            else:
                # frequencies are same as f_pred. Use existing matrices
                for name in distributions:
                    mat = self.prediction_matrices[name]
                    pred_mat[name]['A_re'] = mat['A_re'].copy()
                    pred_mat[name]['A_im'] = mat['A_im'].copy()

        elif self.f_pred is None:
            pred_mat = {}
            dist_mat_exists = True
            for name in distributions:
                pred_mat[name] = {}
                if len(self.distribution_matrices[name]) == 0:
                    dist_mat_exists = False
            # check if we need to recalculate A matrices
            freq_subset = False
            if np.min(rel_round(self.f_train, 10) == rel_round(frequencies, 10)) == False or dist_mat_exists == False:
                # frequencies differ from f_train OR distribution_matrices is empty (could happen if used load_fit_data with core data only)
                if np.min([rel_round(f, 10) in rel_round(self.f_train, 10) for f in
                           frequencies]) == True and dist_mat_exists:
                    # if frequencies are a subset of f_train, we can use submatrices of the existing A matrices
                    # instead of calculating new A matrices
                    f_index = np.array(
                        [np.where(rel_round(self.f_train, 10) == rel_round(f, 10))[0][0] for f in frequencies])

                    for name in distributions:
                        mat = self.distribution_matrices[name]
                        pred_mat[name]['A_re'] = mat['A_re'][f_index, :].copy()
                        pred_mat[name]['A_im'] = mat['A_im'][f_index, :].copy()
                else:
                    # otherwise, we need to calculate A matrices
                    for name in distributions:
                        info = self.distributions[name]
                        tau = self.distributions[name]['tau']
                        epsilon = self.distributions[name]['epsilon']
                        pred_mat[name]['A_re'] = construct_A(frequencies, 'real', tau=tau, basis=self.basis,
                                                             fit_inductance=self.fit_inductance, epsilon=epsilon,
                                                             kernel=info['kernel'], dist_type=info['dist_type'],
                                                             symmetry=info.get('symmetry', ''), bc=info.get('bc', ''),
                                                             ct=info.get('ct', False), k_ct=info.get('k_ct', None)
                                                             )
                        pred_mat[name]['A_im'] = construct_A(frequencies, 'imag', tau=tau, basis=self.basis,
                                                             fit_inductance=self.fit_inductance, epsilon=epsilon,
                                                             kernel=info['kernel'], dist_type=info['dist_type'],
                                                             symmetry=info.get('symmetry', ''), bc=info.get('bc', ''),
                                                             ct=info.get('ct', False), k_ct=info.get('k_ct', None)
                                                             )
            else:
                # frequencies are same as f_train. Use existing matrices
                for name in distributions:
                    mat = self.distribution_matrices[name]
                    pred_mat[name]['A_re'] = mat['A_re'].copy()
                    pred_mat[name]['A_im'] = mat['A_im'].copy()
            self.f_pred = frequencies
            self.prediction_matrices = pred_mat

        return pred_mat

    def predict_Z(self, frequencies, times=None, distributions=None, include_offsets=True, percentile=None):
        """Predict impedance from recovered distributions

		Parameters:
		-----------
		frequencies: array
			Frequencies at which to predict impedance
		distributions: str or list (default: None)
			Distribution name or list of distribution names for which to sum Rp contributions.
			If None, include all distributions
		include_offsets: bool (default: True)
			If True, include contributions of R_inf and inductance. If False, include only contributions from distributions.
		percentile: float (default: None)
			If specified, predict a percentile (0-100) of the posterior distribution. Only applicable for bayes_fit results.
		"""
        if distributions is not None:
            if type(distributions) == str:
                distributions = [distributions]
        else:
            distributions = [k for k in self.distribution_fits.keys()]

        if percentile is not None:
            if self.fit_type != 'bayes':
                raise ValueError('Percentile prediction is only available for bayes_fit results')

            if len(distributions) != len(self.distributions) or include_offsets == False:
                warnings.warn(
                    'If percentile is specified, all distributions and offsets should be included for meaningful results')
            # If distributions or offsets are excluded, the CI will be falsely wide due to correlations between different distributions and offsets

            if np.min(rel_round(self.f_train, 10) == rel_round(frequencies, 10)) == True and len(distributions) == len(
                    self.distributions) and include_offsets == True:
                # If frequencies are same as f_train AND all distributions and offsets included, can use Z_hat from sample_result
                Z_pred = np.percentile(self._sample_result['Z_hat'], percentile, axis=0) * self._Z_scale
                Z_pred = Z_pred[:len(frequencies)] + 1j * Z_pred[len(frequencies):]
            else:
                # If frequencies are different from f_train OR not all distributions/offsets included, need to calculate
                # get A matrices for prediction
                pred_mat = self._get_prediction_matrices(frequencies, distributions)

                # calculate Zhat for each HMC sample, then get percentile
                num_samples = len(self._sample_result['Rinf'])
                Z_pred_matrix = np.zeros((num_samples, len(frequencies)), dtype=complex)

                for name, mat in pred_mat.items():
                    dist_type = self.distributions[name]['dist_type']
                    coef_name = self._get_stan_coef_name(name)
                    coef_matrix = self._sample_result[coef_name]
                    coef_matrix = self._rescale_coef(coef_matrix, dist_type)

                    if dist_type == 'series':
                        Z_re = coef_matrix @ mat['A_re'].T
                        Z_im = coef_matrix @ mat['A_im'].T
                        Z_pred_matrix += Z_re + 1j * Z_im
                    elif dist_type == 'parallel':
                        Y_re = coef_matrix @ mat['A_re'].T
                        Y_im = coef_matrix @ mat['A_im'].T
                        Z_pred_matrix += 1 / (Y_re + 1j * Y_im)
                if include_offsets:
                    # add contributions from R_inf and inductance
                    Z_pred_matrix += np.ones((num_samples, len(frequencies))) * self._rescale_coef(
                        self._sample_result['Rinf'], 'series')[:, None]
                    Z_pred_matrix += 1j * 2 * np.pi * frequencies * self._rescale_coef(self._sample_result['induc'],
                                                                                       'series')[:, None]

                Z_pred_re = np.percentile(np.real(Z_pred_matrix), percentile, axis=0)
                Z_pred_im = np.percentile(np.imag(Z_pred_matrix), percentile, axis=0)
                Z_pred = Z_pred_re + 1j * Z_pred_im

        else:
            # get A matrices for prediction
            pred_mat = self._get_prediction_matrices(frequencies, distributions)

            # construct Z_pred
            Z_pred = np.zeros(len(frequencies), dtype=complex)

            # add contributions from distributions to Z_pred
            if self.fit_type == 'map-drift':
                if times is None:
                    raise ValueError('Data collection times must be provided for drift prediction')
                elif len(times) != len(frequencies):
                    raise ValueError('times must have same length as frequencies')

                for name, mat in pred_mat.items():
                    dist_type = self.distributions[name]['dist_type']
                    # determine number of processes
                    model_split = self.stan_model_name.split('_')
                    drift_str = [ms for ms in model_split if ms[:5] == 'drift'][0]
                    drift_model = '-'.join(drift_str.split('-')[1:])

                    if drift_model in ('x1', 'x2'):
                        num_proc = int(drift_str[-1])

                        # pull x0-x1
                        x0 = self.distribution_fits[name]['x0']
                        x1 = self.distribution_fits[name]['x1']
                        tau_x1 = self.distribution_fits[name]['tau_x1']

                        # create time matrix
                        T = np.tile(times, (len(x0), 1))

                        # construct time-dependent X matrix
                        X0 = np.tile(x0, (len(times), 1)).T
                        X1 = np.tile(x1, (len(times), 1)).T
                        X = X0 + (X1 - X0) * (1 - np.exp(-T / tau_x1))

                        # add additional processes
                        if num_proc > 1:
                            for n in range(2, num_proc + 1):
                                xn = self.distribution_fits[name][f'x{n}']
                                tau_xn = self.distribution_fits[name][f'tau_x{n}']
                                Xn = np.tile(xn, (len(times), 1)).T
                                X += Xn * (1 - np.exp(-T / tau_xn))

                        # calculate AX
                        AX_re = mat['A_re'] * X.T
                        AX_im = mat['A_im'] * X.T

                        if dist_type == 'series':
                            Z_re = np.sum(AX_re, axis=1)
                            Z_im = np.sum(AX_im, axis=1)
                            Z_pred += Z_re + 1j * Z_im
                        elif dist_type == 'parallel':
                            Y_re = np.sum(AX_re, axis=1)
                            Y_im = np.sum(AX_im, axis=1)
                            Z_pred += 1 / (Y_re + 1j * Y_im)

                        if include_offsets:
                            # Rinf is time-dependent
                            R_inf = self.drift_offsets['Rinf_0'] + self.drift_offsets['delta_Rinf'] * (
                                    1 - np.exp(-times / self.drift_offsets['tau_Rinf']))
                            Z_pred += R_inf
                            # inductance is constant
                            Z_pred += 1j * 2 * np.pi * frequencies * self.inductance
                    elif drift_model == 'dx':
                        # pull x0 and dx
                        x0 = self.distribution_fits[name]['x0']
                        dx = self.distribution_fits[name]['dx']
                        tau_dx = self.distribution_fits[name]['tau_dx']

                        # create time matrix
                        T = np.tile(times, (len(x0), 1))

                        # construct time-dependent X matrix
                        X0 = np.tile(x0, (len(times), 1)).T
                        DX = np.tile(dx, (len(times), 1)).T
                        X = X0 + DX * (1 - np.exp(-T / tau_dx))

                        # calculate AX
                        AX_re = mat['A_re'] * X.T
                        AX_im = mat['A_im'] * X.T

                        if dist_type == 'series':
                            Z_re = np.sum(AX_re, axis=1)
                            Z_im = np.sum(AX_im, axis=1)
                            Z_pred += Z_re + 1j * Z_im
                        elif dist_type == 'parallel':
                            Y_re = np.sum(AX_re, axis=1)
                            Y_im = np.sum(AX_im, axis=1)
                            Z_pred += 1 / (Y_re + 1j * Y_im)

                        if include_offsets:
                            # Rinf is time-dependent
                            R_inf = self.drift_offsets['Rinf_0'] + self.drift_offsets['delta_Rinf'] * (
                                    1 - np.exp(-times / self.drift_offsets['tau_Rinf']))
                            Z_pred += R_inf
                            # inductance is constant
                            Z_pred += 1j * 2 * np.pi * frequencies * self.inductance

                    elif drift_model == 'dx-lin':
                        # pull x0 and dx
                        x0 = self.distribution_fits[name]['x0']
                        dx = self.distribution_fits[name]['dx']
                        m_Ft = self.distribution_fits[name]['m_Ft']

                        # create f(t) vector and matrix
                        f_t = times * self.distribution_fits[name]['m_Ft']
                        F_t = np.tile(f_t, (len(x0), 1))

                        # construct time-dependent X matrix
                        X0 = np.tile(x0, (len(times), 1)).T
                        DX = np.tile(dx, (len(times), 1)).T
                        X = X0 + DX * F_t

                        # calculate AX
                        AX_re = mat['A_re'] * X.T
                        AX_im = mat['A_im'] * X.T

                        if dist_type == 'series':
                            Z_re = np.sum(AX_re, axis=1)
                            Z_im = np.sum(AX_im, axis=1)
                            Z_pred += Z_re + 1j * Z_im
                        elif dist_type == 'parallel':
                            Y_re = np.sum(AX_re, axis=1)
                            Y_im = np.sum(AX_im, axis=1)
                            Z_pred += 1 / (Y_re + 1j * Y_im)

                        if include_offsets:
                            # Rinf is time-dependent
                            R_inf = self.drift_offsets['Rinf_0'] + self.drift_offsets['delta_Rinf'] * f_t
                            Z_pred += R_inf
                            # inductance is constant
                            Z_pred += 1j * 2 * np.pi * frequencies * self.inductance

                    elif drift_model in ('RQ-lin', 'RQ'):

                        # get initial coefs
                        x0 = self.distribution_fits[name]['x0']

                        # Z due to initial coefs
                        if dist_type == 'series':
                            Z_re = mat['A_re'] @ x0
                            Z_im = mat['A_im'] @ x0
                            Z_pred += Z_re + 1j * Z_im
                        elif dist_type == 'parallel':
                            Y_re = mat['A_re'] @ x0
                            Y_im = mat['A_im'] @ x0
                            Z_pred += 1 / (Y_re + 1j * Y_im)

                        # Z due to time-dependent ZARC
                        R_rq = self.distribution_fits[name]['R_rq']
                        tau_rq = self.distribution_fits[name]['tau_rq']
                        phi_rq = self.distribution_fits[name]['phi_rq']
                        if drift_model == 'RQ':
                            k_d = self.distribution_fits[name]['k_d']
                            F_t = 1 - np.exp(-k_d * times)
                        elif drift_model == 'RQ-lin':
                            F_t = times * self.distribution_fits[name]['m_Ft']
                        Z_pred += F_t * (R_rq / (1 + (tau_rq * 1j * 2 * np.pi * frequencies) ** phi_rq))

                        if include_offsets:
                            # Rinf is time-dependent
                            R_inf = self.drift_offsets['Rinf_0'] + self.drift_offsets['delta_Rinf'] * F_t
                            Z_pred += R_inf
                            # inductance is constant
                            Z_pred += 1j * 2 * np.pi * frequencies * self.inductance

                    elif drift_model in ('RQ-from-final', 'RQ-lin-from-final'):

                        # get initial coefs
                        x1 = self.distribution_fits[name]['x1']

                        # Z due to initial coefs
                        if dist_type == 'series':
                            Z_re = mat['A_re'] @ x1
                            Z_im = mat['A_im'] @ x1
                            Z_pred += Z_re + 1j * Z_im
                        elif dist_type == 'parallel':
                            Y_re = mat['A_re'] @ x1
                            Y_im = mat['A_im'] @ x1
                            Z_pred += 1 / (Y_re + 1j * Y_im)

                        # Z due to time-dependent ZARC
                        R_rq = self.distribution_fits[name]['R_rq']
                        tau_rq = self.distribution_fits[name]['tau_rq']
                        phi_rq = self.distribution_fits[name]['phi_rq']
                        if drift_model == 'RQ-from-final':
                            k_d = self.distribution_fits[name]['k_d']
                            F_t = -np.exp(-k_d * times)
                        elif drift_model == 'RQ-lin-from-final':
                            t_i = self.distribution_fits[name]['t_i']
                            t_f = self.distribution_fits[name]['t_f']
                            F_t = (times - t_f) / (t_f - t_i)

                        Z_pred += F_t * (R_rq / (1 + (tau_rq * 1j * 2 * np.pi * frequencies) ** phi_rq))

                        if include_offsets:
                            # Rinf is time-dependent
                            R_inf = self.drift_offsets['Rinf_1'] + self.drift_offsets['delta_Rinf'] * F_t
                            Z_pred += R_inf
                            # inductance is constant
                            Z_pred += 1j * 2 * np.pi * frequencies * self.inductance

            else:
                for name, mat in pred_mat.items():
                    dist_type = self.distributions[name]['dist_type']
                    coef = self.distribution_fits[name]['coef']

                    if dist_type == 'series':
                        Z_re = mat['A_re'] @ coef
                        Z_im = mat['A_im'] @ coef
                        Z_pred += Z_re + 1j * Z_im
                    elif dist_type == 'parallel':
                        Y_re = mat['A_re'] @ coef
                        Y_im = mat['A_im'] @ coef
                        Z_pred += 1 / (Y_re + 1j * Y_im)

                # add contributions from R_inf and inductance
                if include_offsets:
                    Z_pred += self.R_inf
                    Z_pred += 1j * 2 * np.pi * frequencies * self.inductance

        return Z_pred

    def predict_Z_distribution(self, frequencies, distributions=None, include_offsets=True):
        """Predict posterior distribution of impedance. Only applicable for fits obtained with bayes_fit

		Parameters:
		-----------
		frequencies: array
			Frequencies at which to predict impedance
		distributions: str or list (default: None)
			Distribution name or list of distribution names for which to sum Rp contributions.
			If None, include all distributions
		include_offsets: bool (default: True)
			If True, include contributions of R_inf and inductance. If False, include only contributions from distributions.
		percentile: float (default: None)
			If specified, predict a percentile (0-100) of the posterior distribution. Only applicable for bayes_fit results.

		Returns:
		Z_pred_matrix: array
			Array of sampled impedance vectors. Each row is a sample
		"""
        if self.fit_type != 'bayes':
            raise ValueError('predict_Z_distribution is only available for bayes_fit results')

        if distributions is not None:
            if type(distributions) == str:
                distributions = [distributions]
        else:
            distributions = [k for k in self.distribution_fits.keys()]

        if len(distributions) != len(self.distributions) or include_offsets == False:
            warnings.warn(
                'All distributions and offsets should be included for meaningful results from predict_Z_distribution')
        # If distributions or offsets are excluded, the CI will be falsely wide due to correlations between different distributions and offsets

        if np.min(rel_round(self.f_train, 10) == rel_round(frequencies, 10)) == True and len(distributions) == len(
                self.distributions) and include_offsets == True:
            # If frequencies are same as f_train AND all distributions and offsets included, can use Z_hat from sample_result
            Z_pred_split = self._sample_result['Z_hat'] * self._Z_scale
            Z_pred_matrix = Z_pred_split[:, :len(frequencies)] + 1j * Z_pred_split[:, len(frequencies):]
        else:
            # If frequencies are different from f_train OR not all distributions/offsets included, need to calculate
            # get A matrices for prediction
            pred_mat = self._get_prediction_matrices(frequencies, distributions)

            # calculate Zhat for each HMC sample, then get percentile
            num_samples = len(self._sample_result['Rinf'])
            Z_pred_matrix = np.zeros((num_samples, len(frequencies)), dtype=complex)

            for name, mat in pred_mat.items():
                dist_type = self.distributions[name]['dist_type']
                coef_name = self._get_stan_coef_name(name)
                coef_matrix = self._sample_result[coef_name]
                coef_matrix = self._rescale_coef(coef_matrix, dist_type)

                if dist_type == 'series':
                    Z_re = coef_matrix @ mat['A_re'].T
                    Z_im = coef_matrix @ mat['A_im'].T
                    Z_pred_matrix += Z_re + 1j * Z_im
                elif dist_type == 'parallel':
                    Y_re = coef_matrix @ mat['A_re'].T
                    Y_im = coef_matrix @ mat['A_im'].T
                    Z_pred_matrix += 1 / (Y_re + 1j * Y_im)
            if include_offsets:
                # add contributions from R_inf and inductance
                Z_pred_matrix += np.ones((num_samples, len(frequencies))) * self._rescale_coef(
                    self._sample_result['Rinf'], 'series')[:, None]
                Z_pred_matrix += 1j * 2 * np.pi * frequencies * self._rescale_coef(self._sample_result['induc'],
                                                                                   'series')[:, None]

            return Z_pred_matrix

    def predict_Rp(self, distributions=None, percentile=None, time=None):
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
            if type(distributions) == str:
                distributions = [distributions]
        else:
            distributions = [k for k in self.distribution_fits.keys()]

        if len(distributions) > 1:
            Z_range = self.predict_Z(np.array([1e20, 1e-20]), distributions=distributions, percentile=percentile)
            Rp = np.real(Z_range[1] - Z_range[0])
        else:
            info = self.distributions[distributions[0]]
            if info['kernel'] == 'DRT' and self.fit_type != 'map-drift' and 'coef' in self.distribution_fits[
                distributions[0]].keys():
                # Rp due to DRT is area under DRT
                if percentile is None:
                    Rp = np.sum(self.distribution_fits[distributions[0]]['coef']) * np.pi ** 0.5 / info['epsilon']
                else:
                    if self.fit_type != 'bayes':
                        raise ValueError('Percentile prediction is only available for bayes_fit results')
                    else:
                        coef_name = self._get_stan_coef_name(distributions[0])
                        coef_matrix = self._sample_result[coef_name]
                        coef_matrix = self._rescale_coef(coef_matrix, 'series')
                        Rp_array = np.sum(coef_matrix, axis=1) * np.pi ** 0.5 / info['epsilon']
                        Rp = np.percentile(Rp_array, percentile)
            else:
                # just calculate Z at very high and very low frequencies and take the difference in Z'
                # could do calcs using coefficients, but this is fast and accurate enough for now (and should work for any arbitrary distribution)
                if percentile is None:
                    if time is not None:
                        times = np.array([time, time])
                    elif self.fit_type == 'map-drift':
                        raise ValueError('Time must be provided for drift prediction')
                    Z_range = self.predict_Z(np.array([1e20, 1e-20]), distributions=distributions,
                                             times=np.array([time, time]))
                    Rp = np.real(Z_range[1] - Z_range[0])
                else:
                    # get the distribution of Rp
                    Z_mat = self.predict_Z_distribution(np.array([1e20, 1e-20]), distributions=distributions)
                                                        # times=np.array([time, time]))
                    Rp_sample = np.real(Z_mat[:, 1] - Z_mat[:, 0])
                    Rp = np.percentile(Rp_sample, percentile)

        return Rp

    def predict_sigma(self, frequencies, percentile=None, times=None):
        if percentile is not None and self.fit_type != 'bayes':
            raise ValueError('Percentile prediction is only available for bayes_fit')

        if np.min(rel_round(self.f_train, 10) == rel_round(frequencies, 10)) == True:
            # if frequencies are training frequencies, just use sigma_tot output
            if self.fit_type == 'bayes' and percentile is not None:
                sigma_tot = np.percentile(self._sample_result['sigma_tot'], percentile, axis=0) * self._Z_scale
            elif self.fit_type == 'bayes' or self.fit_type[:3] == 'map':
                sigma_tot = self.error_fit['sigma_tot']
            else:
                raise ValueError('Error scale prediction only available for bayes_fit and map_fit')

            sigma_re = sigma_tot[:len(self.f_train)].copy()
            sigma_im = sigma_tot[len(self.f_train):].copy()
        else:
            # if frequencies are not training frequencies, calculate from parameters
            # this doesn't match sigma_tot perfectly
            if self.fit_type == 'bayes' and percentile is not None:
                sigma_res = np.percentile(self._sample_result['sigma_res'], percentile) * self._Z_scale
                alpha_prop = np.percentile(self._sample_result['alpha_prop'], percentile)
                alpha_re = np.percentile(self._sample_result['alpha_re'], percentile)
                alpha_im = np.percentile(self._sample_result['alpha_im'], percentile)
                try:
                    sigma_out = np.percentile(self._sample_result['sigma_out'], percentile, axis=0) * self._Z_scale
                except ValueError:
                    sigma_out = np.zeros(2 * len(self.f_train))
            elif self.fit_type == 'bayes' or self.fit_type[:3] == 'map':
                sigma_res = self.error_fit['sigma_res']
                alpha_prop = self.error_fit['alpha_prop']
                alpha_re = self.error_fit['alpha_re']
                alpha_im = self.error_fit['alpha_im']
                try:
                    sigma_out = self.error_fit['sigma_out']
                except KeyError:
                    sigma_out = np.zeros(2 * len(self.f_train))
            else:
                raise ValueError('Error scale prediction only available for bayes_fit and map_fit')

            sigma_min = self.error_fit['sigma_min']

            Z_pred = self.predict_Z(frequencies, percentile=percentile, times=times)
            # assume none of predicted points are outliers - just get baseline sigma_out contribution
            sigma_base = np.sqrt(sigma_res ** 2 + np.min(sigma_out) ** 2 + sigma_min ** 2)

            sigma_re = np.sqrt(sigma_base ** 2 + (alpha_prop * Z_pred.real) ** 2 + (alpha_re * Z_pred.real) ** 2 + (
                    alpha_im * Z_pred.imag) ** 2)
            sigma_im = np.sqrt(sigma_base ** 2 + (alpha_prop * Z_pred.imag) ** 2 + (alpha_re * Z_pred.real) ** 2 + (
                    alpha_im * Z_pred.imag) ** 2)

        return sigma_re, sigma_im

    def score(self, frequencies, Z, metric='chi_sq', weights=None, part='both', times=None):
        weights = self._format_weights(frequencies, Z, weights, part)
        Z_pred = self.predict_Z(frequencies, times=times)
        if part == 'both':
            Z_pred = np.concatenate((Z_pred.real, Z_pred.imag))
            Z = np.concatenate((Z.real, Z.imag))
            weights = np.concatenate((weights.real, weights.imag))
        else:
            Z_pred = getattr(Z_pred, part)
            Z = getattr(Z, part)
            weights = getattr(weights, part)

        if metric == 'chi_sq':
            score = np.sum(((Z_pred - Z) * weights) ** 2) / len(frequencies)
        elif metric == 'r2':
            score = r2_score(Z, Z_pred, weights=weights)
        else:
            raise ValueError(f"Invalid metric {metric}. Options are 'chi_sq', 'r2'")

        return score

    def predict_distribution(self, name=None, eval_tau=None, percentile=None, time=None):
        """Get fitted distribution(s)
		"""

        if self.fit_type == 'map-drift':
            # evaluate the distribution at the specified time
            if time is None:
                raise ValueError('time must be supplied for drift fit')

            # If distribution name not specified, use first distribution
            if name is None:
                name = list(self.distributions.keys())[0]

            model_split = self.stan_model_name.split('_')
            drift_str = [ms for ms in model_split if ms[:5] == 'drift'][0]
            drift_model = '-'.join(drift_str.split('-')[1:])

            if drift_model in ('x1', 'x2'):
                # pull x0-x1
                x0 = self.distribution_fits[name]['x0']
                x1 = self.distribution_fits[name]['x1']
                tau_x1 = self.distribution_fits[name]['tau_x1']

                x = x0 + (x1 - x0) * (1 - np.exp(-time / tau_x1))

                # determine number of processes
                num_proc = int(drift_str[-1])

                # add additional processes
                if num_proc > 1:
                    for n in range(2, num_proc + 1):
                        xn = self.distribution_fits[name][f'x{n}']
                        tau_xn = self.distribution_fits[name][f'tau_x{n}']

                        x += xn * (1 - np.exp(-time / tau_xn))

                epsilon = self.distributions[name]['epsilon']
                basis_tau = self.distributions[name]['tau']
                if eval_tau is None:
                    eval_tau = self.distributions[name]['tau']
                phi = get_basis_func(self.basis)
                bases = np.array([phi(np.log(eval_tau / t_m), epsilon) for t_m in basis_tau]).T
                F = bases @ x

                return F

            elif drift_model == 'dx':
                # pull x0 and dx
                x0 = self.distribution_fits[name]['x0']
                dx = self.distribution_fits[name]['dx']
                tau_dx = self.distribution_fits[name]['tau_dx']

                x = x0 + dx * (1 - np.exp(-time / tau_dx))

                epsilon = self.distributions[name]['epsilon']
                basis_tau = self.distributions[name]['tau']
                if eval_tau is None:
                    eval_tau = self.distributions[name]['tau']
                phi = get_basis_func(self.basis)
                bases = np.array([phi(np.log(eval_tau / t_m), epsilon) for t_m in basis_tau]).T
                F = bases @ x

                return F

            elif drift_model == 'dx-lin':
                # pull x0 and dx
                x0 = self.distribution_fits[name]['x0']
                dx = self.distribution_fits[name]['dx']
                m_Ft = self.distribution_fits[name]['m_Ft']

                x = x0 + dx * time * m_Ft

                epsilon = self.distributions[name]['epsilon']
                basis_tau = self.distributions[name]['tau']
                if eval_tau is None:
                    eval_tau = self.distributions[name]['tau']
                phi = get_basis_func(self.basis)
                bases = np.array([phi(np.log(eval_tau / t_m), epsilon) for t_m in basis_tau]).T
                F = bases @ x

                return F

            elif drift_model in ('RQ', 'RQ-lin'):
                # get initial DRT
                x0 = self.distribution_fits[name]['x0']

                if eval_tau is None:
                    eval_tau = self.distributions[name]['tau']
                epsilon = self.distributions[name]['epsilon']
                basis_tau = self.distributions[name]['tau']

                phi = get_basis_func(self.basis)
                bases = np.array([phi(np.log(eval_tau / t_m), epsilon) for t_m in basis_tau]).T
                F0 = bases @ x0

                # get time-dependent ZARC DRT
                R_rq = self.distribution_fits[name]['R_rq']
                tau_rq = self.distribution_fits[name]['tau_rq']
                phi_rq = self.distribution_fits[name]['phi_rq']
                if drift_model == 'RQ':
                    k_d = self.distribution_fits[name]['k_d']
                    F_t = 1 - np.exp(-k_d * time)
                elif drift_model == 'RQ-lin':
                    F_t = time * self.distribution_fits[name]['m_Ft']
                F_rq = (1 / (2 * np.pi)) * np.sin((1 - phi_rq) * np.pi) / (
                        np.cosh(phi_rq * np.log(eval_tau / tau_rq)) - np.cos((1 - phi_rq) * np.pi))

                F = F0 + F_t * R_rq * F_rq

                return F

            elif drift_model in ('RQ-from-final', 'RQ-lin-from-final'):
                # get initial DRT
                x1 = self.distribution_fits[name]['x1']

                if eval_tau is None:
                    eval_tau = self.distributions[name]['tau']
                epsilon = self.distributions[name]['epsilon']
                basis_tau = self.distributions[name]['tau']

                phi = get_basis_func(self.basis)
                bases = np.array([phi(np.log(eval_tau / t_m), epsilon) for t_m in basis_tau]).T
                F1 = bases @ x1

                # get time-dependent ZARC DRT
                R_rq = self.distribution_fits[name]['R_rq']
                tau_rq = self.distribution_fits[name]['tau_rq']
                phi_rq = self.distribution_fits[name]['phi_rq']
                if drift_model == 'RQ-from-final':
                    k_d = self.distribution_fits[name]['k_d']
                    F_t = -np.exp(-k_d * time)
                elif drift_model == 'RQ-lin-from-final':
                    t_i = self.distribution_fits[name]['t_i']
                    t_f = self.distribution_fits[name]['t_f']
                    F_t = (time - t_f) / (t_f - t_i)

                F_rq = (1 / (2 * np.pi)) * np.sin((1 - phi_rq) * np.pi) / (
                        np.cosh(phi_rq * np.log(eval_tau / tau_rq)) - np.cos((1 - phi_rq) * np.pi))

                F = F1 + F_t * R_rq * F_rq

                return F

        else:
            if name is not None:
                # return the specified distribution
                if percentile is not None:
                    coef = self.coef_percentile(name, percentile)
                else:
                    coef = self.distribution_fits[name]['coef']

                epsilon = self.distributions[name]['epsilon']
                basis_tau = self.distributions[name]['tau']
                if eval_tau is None:
                    eval_tau = self.distributions[name]['tau']
                phi = get_basis_func(self.basis)
                bases = np.array([phi(np.log(eval_tau / t_m), epsilon) for t_m in basis_tau]).T
                F = bases @ coef

                return F
            else:
                # out = {}
                # # return all distributions in a dict
                # for name in self.distributions.keys():

                # return first distribution
                name = list(self.distributions.keys())[0]
                if percentile is not None:
                    coef = self.coef_percentile(name, percentile)
                else:
                    coef = self.distribution_fits[name]['coef']
                epsilon = self.distributions[name]['epsilon']
                basis_tau = self.distributions[name]['tau']
                if eval_tau is None:
                    # don't overwrite eval_tau - need to maintain across multiple distributions
                    etau = self.distributions[name]['tau']
                else:
                    etau = eval_tau
                phi = get_basis_func(self.basis)
                bases = np.array([phi(np.log(eval_tau / t_m), epsilon) for t_m in basis_tau]).T
                F = bases @ coef
                # 	out[name] = F
                # return out

                return F

    def check_outliers(self, frequencies, Z, threshold, use_existing_fit, **ridge_kw):
        """
		Check for outliers in the impedance data.
		If Inverter instance has been fitted, use existing fit.
		Otherwise, use ridge_fit to check for outliers.
		Also used as a check when outliers='auto' in map_fit and bayes_fit.

		Parameters:
		-----------
		frequencies: array
			Array of measurement frequencies
		Z: complex array
			Array of complex impedance values
		threshold: float
			Threshold for outlier flagging. If fit type is ridge, threshold is the number of IQRs by which a residual
			must exceed the 75th percentile to be flagged as an outlier. If fit type is map or bayes, threshold is minimum
			z-score of residual required to be flagged as an outlier.
		use_existing_fit: bool
			If True, use the existing fit to check for outliers (if a fit exists for the provided dataset).
			If False, use a new ridge fit to check for outliers.
		ridge_kw:
			Keyword args to pass to ridge_fit if Inverter instance has not yet been fitted.
			Ignored if Inverter has already been fitted

		Returns:
		--------
		outlier_idx: array
			Indices of likely outliers
		"""
        # Check if instance has already been fitted to this data
        if check_equality(frequencies, self.f_train) and check_equality(Z, self.Z_train) \
                and not self._recalc_mat and hasattr(self, 'distribution_fits'):
            fit_exists = True
        else:
            fit_exists = False

        # If no fit exists or use_existing_fit == False, perform ridge fit
        if not (use_existing_fit and fit_exists):
            self.ridge_fit(frequencies, Z, preset='Huang', **ridge_kw)

        # Get residuals
        Z_err = self.predict_Z(frequencies) - Z

        if self.fit_type == 'ridge':
            # no error structure estimate - use IQR to identify possible outliers
            Zmod = np.sqrt(Z.real ** 2 + Z.imag ** 2)
            # get thresholds for real and imaginary residuals
            re_thresh = get_outlier_thresh(np.abs(Z_err.real / Zmod), iqr_factor=threshold)
            im_thresh = get_outlier_thresh(np.abs(Z_err.imag / Zmod), iqr_factor=threshold)

            outlier_idx = np.argwhere(
                (Z_err.real / Zmod) ** 2 + (Z_err.imag / Zmod) ** 2 >= re_thresh ** 2 + im_thresh ** 2)
        elif self.fit_type in ('map', 'bayes'):
            # Use fitted error structure to identify possible outliers
            sigma_re, sigma_im = self.predict_sigma(frequencies)

            # get real and imag z-scores
            zs_re = Z_err.real / sigma_re
            zs_im = Z_err.imag / sigma_im
            # Get combined z-score and apply threshold
            zs_tot = np.sqrt((zs_re ** 2 + zs_im ** 2) / 2)
            outlier_idx = np.argwhere(zs_tot > threshold)

        return outlier_idx

    # ===============================================
    # Peak fitting
    # ===============================================
    def fit_peaks(self, distribution=None, eval_tau=None, percentile=None, time=None,
                  check_shoulders=True, weights=None, prom_rthresh=0.001, R_rthresh=0.005,
                  l1_penalty=0, l2_penalty=0.01,
                  check_chi_sq=False, chi_sq_thresh=0.5, chi_sq_delta=0.3,
                  fit_data=False, frequencies=None, Z=None, Z_weights=None, lambda_x=10):
        """
        Fit Havriliak-Negami (HN) peaks to the recovered distribution.
		Uses a peak detection algorithm to determine the number of peaks,
		then optimizes the HN peak parameters to best fit the distribution
		and/or impedance data.

        Parameters
        ----------
        distribution : str, optional (default: None)
            Name of distribution to fit. If None, use first distribution in self.distributions
        eval_tau : array, optional (default: None)
            tau grid over which to fit distribution. If None, go one decade beyond basis tau in each direction
        percentile : float, optional (default: None)
            Percentile of credible interval to fit. Only applies to HMC fits (mode='sample')
        time : float, optional (default: None)
            Time at which to evaluate distribution. Only applies to drift fits
        check_shoulders : bool, optional (default: True)
            If True, identify shoulder peaks by searching for peaks in the first derivative of the distribution.
        weights : array, optional (default: None)
            Weights to use for fitting peaks. Must match length of eval_tau
        prom_rthresh : float, optional (default: 0.001)
            Relative prominence threshold for identifying peaks using scipy.signal.fined_peaks.
            Threshold is fraction of the largest peak.
        R_rthresh : float, optional (default: 0.005)
            Relative resistance threshold for keeping fitted peaks.
            Threshold is fraction of polarization resistance.
        l1_penalty : float, optional (default: 0)
            L1 penalty strength to apply to peak resistance values.
        l2_penalty : float, optional (default: 0.01)
            L2 penalty strength to apply to peak resistance values.
        check_chi_sq : bool, optional (default: False)
            If True, check the chi square value after fitting and search for additional peaks if chi_sq > chi_sq_thresh.
        chi_sq_thresh : float, optional (default: 0.5)
            chi square threshold for searching for additional peaks.
        chi_sq_delta : float, optional (default: 0.3)
            Minimum improvement in chi square required to keep additional peaks identified when check_chi_sq=True.
        fit_data : bool, optional (default: False)
            If True, fit peaks both to distribution and to impedance data
        frequencies : array, optional (default: None)
            Array of measured frequencies. Only used when fit_data=True.
        Z : array, optional (default: None)
            Array of impedance values. Only used when fit_data=True.
        Z_weights : array or str, optional (default: None)
            Array of weights or name of weighting scheme to use for fitting impedance. Only used when fit_data=True.
        lambda_x :
            Magnitude of penalty for deviations from parameter values obtained by fitting peaks directly to the
            distribution. Only used when fit_data=True.
        """
        # If no distribution specified, use first distribution
        if distribution is None:
            distribution = list(self.distributions.keys())[0]
        # If eval_tau not given, go one decade beyond basis tau in each direction to capture all peaks
        if eval_tau is None:
            basis_tau = self.distributions[distribution]['tau']
            tmin = np.log10(np.min(basis_tau)) - 1
            tmax = np.log10(np.max(basis_tau)) + 1
            num_decades = tmax - tmin
            eval_tau = np.logspace(tmin, tmax, int(10 * num_decades + 1))

        F = self.predict_distribution(distribution, eval_tau, percentile, time)

        # Determine whether negative peaks need to be fitted
        if np.min(F) >= 0:
            nonneg = True
        else:
            nonneg = False

        Rp = self.predict_Rp()

        # Fit HN model to distribution
        x = pf.fit_peaks(eval_tau, F, Rp, weights=weights, nonneg=nonneg, check_shoulders=check_shoulders,
                         prom_rthresh=prom_rthresh, R_rthresh=R_rthresh, check_chi_sq=check_chi_sq,
                         chi_sq_thresh=chi_sq_thresh, chi_sq_delta=chi_sq_delta,
                         l1_penalty=l1_penalty, l2_penalty=l2_penalty)

        # Re-fit HN model to data
        if fit_data:
            if frequencies is None or Z is None:
                raise ValueError('frequencies and Z must be provided if fit_data==True')
            ##**will have to adjust R_inf for drift models**
            result = pf.fit_data(x, frequencies, Z, R_inf=self.R_inf, inductance=self.inductance,
                                 weights=Z_weights, lambda_x=lambda_x)
            x = result['x']

        # sort by time constant
        t0 = np.exp(x[1::4])
        sort_idx = np.argsort(t0)
        x_sorted = np.empty(len(x))
        for i, idx in enumerate(sort_idx):
            x_sorted[4 * i: 4 * (i + 1)] = x[4 * idx: 4 * (idx + 1)]

        self.distribution_fits[distribution]['peak_params'] = x_sorted

        # Calculate and store chi_sq for convenience
        self.distribution_fits[distribution]['peak_chi_sq'] = self.score_peak_fit(eval_tau=eval_tau,
                                                                                  distribution=distribution,
                                                                                  weights=weights,
                                                                                  percentile=percentile,
                                                                                  time=time)

    def fit_peaks_constrained(self, tau0_guess, distribution=None, eval_tau=None, percentile=None, time=None,
                              sigma_lntau=5, lntau_uncertainty=3, weights=None, l2_penalty=0.01):
        # If no distribution specified, use first distribution
        if distribution is None:
            distribution = list(self.distributions.keys())[0]
        # If eval_tau not given, go one decade beyond basis tau in each direction to capture all peaks
        if eval_tau is None:
            basis_tau = self.distributions[distribution]['tau']
            tmin = np.log10(np.min(basis_tau)) - 1
            tmax = np.log10(np.max(basis_tau)) + 1
            num_decades = tmax - tmin
            eval_tau = np.logspace(tmin, tmax, int(10 * num_decades + 1))

        F = self.predict_distribution(distribution, eval_tau, percentile, time)

        # Determine whether negative peaks need to be fitted
        if np.min(F) >= 0:
            nonneg = True
        else:
            nonneg = False

        Rp = self.predict_Rp()

        # Fit HN model to distribution
        result = pf.constrained_peak_fit(eval_tau, F, tau0_guess, Rp, nonneg, lntau_uncertainty, sigma_lntau, weights,
                                         l2_penalty)
        self.distribution_fits[distribution]['peak_params'] = result['x']

        # Calculate and store chi_sq for convenience
        self.distribution_fits[distribution]['peak_chi_sq'] = self.score_peak_fit(eval_tau=eval_tau,
                                                                                  distribution=distribution,
                                                                                  weights=weights,
                                                                                  percentile=percentile,
                                                                                  time=time)

    def predict_peak_distribution(self, eval_tau=None, distribution=None, peak_index=None):
        """
		Predict distribution from peak fit

		Parameters:
		-----------
		eval_tau: array, optional (default: None)
			tau values at which to evaluate the distribution.
			If None, use basis tau
		distribution: str, optional (default: None)
			Name of distribution to predict.
			If None, use first distribution
		peak_index : int, optional (default: False)
		    Index of peak to evaluate. If None, sum contributions of all peaks.
		"""
        # If no distribution specified, use first distribution
        if distribution is None:
            distribution = list(self.distributions.keys())[0]

        # If eval_tau not given, go one decade beyond basis tau in each direction to capture all peaks
        if eval_tau is None:
            basis_tau = self.distributions[distribution]['tau']
            tmin = np.log10(np.min(basis_tau)) - 1
            tmax = np.log10(np.max(basis_tau)) + 1
            num_decades = tmax - tmin
            eval_tau = np.logspace(tmin, tmax, int(10 * num_decades + 1))

        if peak_index is not None:
            # Get parameters for specified peaks
            peak_params = self.distribution_fits[distribution]['peak_params'][4 * peak_index:4 * peak_index + 4]
        else:
            # Get parameters for all peaks
            peak_params = self.distribution_fits[distribution]['peak_params']

        F = pf.evaluate_fit_distribution(peak_params, eval_tau)

        return F

    def predict_peak_Z(self, frequencies, distribution=None):
        """
		Predict impedance from HN model fit

		Parameters:
		-----------
		frequencies: array
			Frequencies at which to evaluate impedance
		distribution: str, optional (default: None)
			Name of distribution to predict.
			If None, use first distribution
		"""
        # If no distribution specified, use first distribution
        if distribution is None:
            distribution = list(self.distributions.keys())[0]

        Z = pf.evaluate_fit_impedance(self.distribution_fits[distribution]['peak_params'], frequencies, self.R_inf,
                                      self.inductance)

        return Z

    def extract_peak_info(self, distribution=None, sort=True):
        """
        Extract dict of peak parameters.
        Parameters
        ----------
        distribution : str, optional (default: None)
            Name of distribution for which to extract peak parameters.
            If None, use first distribution in self.distributions
        sort : bool, optional (default: True)
            If True, sort peaks by ascending time constant
        Returns
        -------
        info : dict
            Dict of peak parameters
        """
        # If no distribution specified, use first distribution
        if distribution is None:
            distribution = list(self.distributions.keys())[0]

        # get parameters and parse
        params = self.distribution_fits[distribution]['peak_params']
        num_peaks = int(len(params) / 4)

        R = params[::4]
        t0 = np.exp(params[1::4])
        alpha = params[2::4]
        beta = params[3::4]

        # sort by time constant
        if sort:
            sort_idx = np.argsort(t0)
            R = R[sort_idx]
            t0 = t0[sort_idx]
            alpha = alpha[sort_idx]
            beta = beta[sort_idx]

        # make dict for easy reading
        info = {'num_peaks': num_peaks,
                'chi_sq': self.distribution_fits[distribution]['peak_chi_sq'],
                'R': R,
                'tau_0': t0,
                'alpha': alpha,
                'beta': beta
                }

        return info

    def score_peak_fit(self, eval_tau=None, distribution=None, weights=None, percentile=None, time=None):
        # If no distribution specified, use first distribution
        if distribution is None:
            distribution = list(self.distributions.keys())[0]

        # If eval_tau not given, use basis tau
        if eval_tau is None:
            eval_tau = self.distributions[distribution]['tau']

        # Get distribution and HN fit
        F = self.predict_distribution(distribution, eval_tau, percentile, time)
        F_fit = pf.evaluate_fit_distribution(self.distribution_fits[distribution]['peak_params'], eval_tau)

        # If no weights specified, use hybrid weighting scheme
        if weights is None:
            weights = 1 / (F + np.percentile(F, 80))

        # Calculate chi_sq
        resid = F_fit - F
        chi_sq = np.sum((resid * weights) ** 2)

        return chi_sq

    # ===============================================
    # Plotting
    # ===============================================
    def plot_distribution(self, ax=None, distribution=None, tau_plot=None, plot_bounds=True, plot_ci=True,
                          label='', ci_label='', unit_scale='auto', freq_axis=True, area=None, normalize=False,
                          predict_kw={}, **kw):
        """
        Plot the specified distribution as a function of tau.

        Parameters
        ----------
        ax : matplotlib axis
            Axis on which to plot
        distribution : str, optional (default: None)
            Name of distribution to plot. If None, first distribution in self.distributions will be used
        tau_plot : array, optonal (default: None)
            Time constant grid over which to evaluate the distribution.
            If None, a grid extending one decade beyond the basis time constants in each direction will be used.
        plot_bounds : bool, optional (default: True)
            If True, indicate frequency bounds of experimental data with vertical lines.
            Requires that DataFrame of experimental data be passed for df argument
        plot_ci : bool, optional (default: True)
            If True, plot the 95% credible interval of the distribution (if available).
        label : str, optional (default: '')
            Label for matplotlib
        unit_scale : str, optional (default: 'auto')
            Scaling unit prefix. If 'auto', determine from data.
            Options are 'mu', 'm', '', 'k', 'M', 'G'
        freq_axis : bool, optional (default: True)
            If True, add a secondary x-axis to display frequency
        area : float, optional (default: None)
            Active area. If provided, plot the area-normalized distribution
        normalize : bool, optional (default: False)
            If True, normalize the distribution such that the integrated area is 1
        predict_kw : dict, optional (default: {})
            Keyword args to pass to Inverter predict_distribution() method
        kw : keyword args, optional
            Keyword args to pass to maplotlib.pyplot.plot
        Returns
        -------
        ax : matplotlib axis
            Axis on which distribution is plotted
        """
        # Construct dataframe from fitted data
        df = fl.construct_eis_df(self.f_train, self.Z_train)

        ax = bp.plot_distribution(df, self, ax, distribution, tau_plot, plot_bounds, plot_ci,
                                  label, ci_label, unit_scale, freq_axis, area, normalize,
                                  predict_kw, **kw)

        return ax

    def plot_fit(self, axes=None, plot_type='all', bode_cols=['Zreal', 'Zimag'], plot_data=True, color='k',
                 f_pred=None, label='', data_label='', unit_scale='auto', area=None, predict_kw={}, data_kw={},
                 **kw):
        """
        Plot fit of impedance data.
        Parameters
        ----------
        df : pandas DataFrame
            Dataframe of fitted impedance data
        inv : Inverter instance
            Fitted Inverter instance
        axes : array, optional (default: None)
            Array or list of axes on which to plot. If None, axes will be created
        plot_type : str, optional (default: 'all')
            Type of plot(s) to create. Options:
                'all': Nyquist and Bode plots
                'nyquist': Nyquist plot only
                'bode': Bode plots only
        bode_cols : list, optional (default: ['Zmod', 'Zphz'])
            List of data columns to plot in Bode plots. Options: 'Zreal', 'Zimag', 'Zmod', 'Zphz'
            Only used if plot_type in ('all', 'bode')
        plot_data : bool, optional (default: True)
            If True, scatter data and plot fit line. If False, plot fit line only.
        color : str, optional (default: 'k')
            Color for fit line
        f_pred : array, optional (default: None)
            Frequencies for which to plot fit line. If None, use data frequencies
        label : str, optional (default: '')
            Label for fit line
        data_label : str, optional (default: '')
            Label for data points
        unit_scale : str, optional (default: 'auto')
            Scaling unit prefix. If 'auto', determine from data.
            Options are 'mu', 'm', '', 'k', 'M', 'G'
        area : float, optional (default: None)
            Active area in cm^2. If specified, plot area-normalized impedance.
        predict_kw : dict, optional (default: {})
            Keywords to pass to self.predict_Z
        data_kw : dict, optional (default: {})
            Keywords to pass to matplotlib.pyplot.scatter when plotting data points
        kw : dict, optional (default: {})
            Keywords to pass to matplotlib.pyplot.plot when plotting fit line
        Returns
        -------
        axes : array
            Axes on which fit is plotted
        """
        # Construct dataframe from fitted data
        df = fl.construct_eis_df(self.f_train, self.Z_train)

        axes = bp.plot_fit(df, self, axes, plot_type, bode_cols, plot_data, color, f_pred, label, data_label,
                           unit_scale, area,
                           predict_kw, data_kw, **kw)

        return axes

    def plot_residuals(self, axes=None, unit_scale='auto', plot_ci=True, predict_kw={}):
        """
        Plot the real and imaginary impedance residuals

        Parameters
        ----------
        df : pandas DataFrame
            Dataframe of fitted impedance data
        inv : Inverter instance
            Fitted Inverter instance
        axes : array, optional (default: None)
            Array or list of axes on which to plot. If None, axes will be created
        unit_scale : str, optional (default: 'auto')
            Scaling unit prefix. If 'auto', determine from data.
            Options are 'mu', 'm', '', 'k', 'M', 'G'
        plot_ci : bool, optional (default: True)
            If True, plot the 99.7% credible interval (+/- 3 sigma) of residuals (if available).
        predict_kw : dict, optional (default: {})
            Keywords to pass to self.predict_Z
        Returns
        -------
        axes : array
            Axes on which residuals are plotted
        """
        # Construct dataframe from fitted data
        df = fl.construct_eis_df(self.f_train, self.Z_train)

        axes = bp.plot_residuals(df, self, axes, unit_scale, plot_ci, predict_kw)

        return axes

    def plot_full_results(self, bode_cols=['Zreal', 'Zimag'], plot_data=True, color='k', axes=None,
                          tau_plot=None, f_pred=None, plot_ci=True, plot_drt_ci=True, predict_kw={}):
        """
        Plot the impedance fit, fitted distribution, and impedance residuals.

        Parameters
        ----------
        df : pandas DataFrame
            Dataframe of fitted impedance data
        inv : Inverter instance
            Fitted Inverter instance
        axes : array, optional (default: None)
            Array or list of axes on which to plot. If None, axes will be created
        bode_cols : list, optional (default: ['Zmod', 'Zphz'])
            List of data columns to plot in Bode plots. Options: 'Zreal', 'Zimag', 'Zmod', 'Zphz'
            Only used if plot_type in ('all', 'bode')
        plot_data : bool, optional (default: True)
            If True, scatter data and plot fit line. If False, plot fit line only.
        color : str, optional (default: 'k')
            Color for fit line
        tau_plot : array, optonal (default: None)
            Time constant grid over which to evaluate the distribution.
            If None, a grid extending one decade beyond the basis time constants in each direction will be used.
        f_pred : array, optional (default: None)
            Frequencies for which to plot fit line. If None, use data frequencies
        plot_ci : bool, optional (default: True)
            If True, plot the 99.7% credible interval (+/- 3 sigma) of residuals (if available).
        plot_drt_ci : bool, optional (default: True)
            If True, plot the 95% credible interval of the distribution (if available).
        predict_kw : dict, optional (default: {})
            Keywords to pass to self.predict_Z

        Returns
        -------
        axes : array
            Axes on which results are plotted
        """
        # Construct dataframe from fitted data
        df = fl.construct_eis_df(self.f_train, self.Z_train)

        axes = bp.plot_full_results(df, self, axes, bode_cols, plot_data, color, tau_plot, f_pred, plot_ci, plot_drt_ci,
                                    predict_kw)

        return axes

    def plot_peak_fit(self, ax=None, distribution=None, tau_plot=None, plot_bounds=False, plot_ci=False,
                      plot_distribution=True, plot_individual_peaks=False,
                      peak_fit_label=None, distribution_label='$\gamma$', ci_label='95% CI',
                      unit_scale='auto', freq_axis=True, area=None, normalize=False,
                      predict_kw={}, distribution_kw={}, **kw):
        """
        Plot Havriliak-Negami peak fit of the distribution. Only valid if fit_peaks has been executed.

        Parameters
        ----------
        ax : matplotlib axis, optional (default: None)
            Axis on which to plat
        distribution : str, optional (default: None)
            Name of distribution to plot. If None, first distribution in self.distributions will be used
        tau_plot : array, optonal (default: None)
            Time constant grid over which to evaluate the distribution.
            If None, a grid extending one decade beyond the basis time constants in each direction will be used.
        plot_bounds : bool, optional (default: False)
            If True, indicate frequency bounds of experimental data with vertical lines.
            Requires that DataFrame of experimental data be passed for df argument
        plot_ci : bool, optional (default: False)
            If True, plot the 95% credible interval of the distribution (if available).
        plot_distribution : bool, optional (default: True)
            If True, plot the fitted distribution and the peak fit. If False, plot only the peak fit.
        plot_individual_peaks : bool, optional (default: False)
            If True, plot each peak as its own series. If False, plot the total peak fit.
        peak_fit_label : str, optional (default: 'Peak fit')
            Label for peak fit
        distribution_label : str, optional (default: '$\gamma$'$)
            Label for fitted distribution
        ci_label : str, optional (default: '95% CI')
            Label for credible interval of fitted distribution
        unit_scale : str, optional (default: 'auto')
            Scaling unit prefix. If 'auto', determine from data.
            Options are 'mu', 'm', '', 'k', 'M', 'G'
        freq_axis : bool, optional (default: True)
            If True, add a secondary x-axis to display frequency
        area : float, optional (default: None)
            Active area. If provided, plot the area-normalized distribution
        normalize : bool, optional (default: False)
            If True, normalize the distribution such that the integrated area is 1
        predict_kw : dict, optional (default: {})
            Keyword args to pass to Inverter predict_distribution() method
        distribution_kw : dict, optional (default: {})
            Keyword args to pass to matplotlib.plot when plotting distribution
        kw : keyword args, optional
            Keyword args to pass to matplotlib.plot when plotting peak fit(s)

        Returns
        -------
        ax : matplotlib axis
            Axis on which distribution is plotted
        """
        # Construct dataframe from fitted data
        df = fl.construct_eis_df(self.f_train, self.Z_train)

        distribution_defaults = {'color': 'k'}
        distribution_defaults.update(distribution_kw)

        if ax is None:
            fig, ax = plt.subplots(figsize=(3.5, 2.75))

        # If no distribution specified, use first distribution
        if distribution is None:
            distribution = list(self.distributions.keys())[0]

        # If tau_plot not given, go one decade beyond basis tau in each direction
        if tau_plot is None:
            basis_tau = self.distributions[distribution]['tau']
            tmin = np.log10(np.min(basis_tau)) - 1
            tmax = np.log10(np.max(basis_tau)) + 1
            num_decades = tmax - tmin
            tau_plot = np.logspace(tmin, tmax, int(20 * num_decades + 1))

        if not plot_distribution:
            # Hide distribution
            distribution_defaults['alpha'] = 0
            distribution_label = ''

        # Plot distribution
        ax = self.plot_distribution(ax, distribution, tau_plot, plot_bounds, plot_ci,
                                    distribution_label, ci_label, unit_scale, freq_axis, area, normalize,
                                    predict_kw, **distribution_defaults)

        # Plot peak fit
        if unit_scale == 'auto':
            scale_factor = get_scale_factor(df, area)
        else:
            scale_factor = get_factor_from_unit(unit_scale)

        if plot_individual_peaks:
            peak_info = self.extract_peak_info()
            if peak_fit_label is None:
                peak_fit_label = [f'Peak {i + 1}' for i in range(peak_info['num_peaks'])]
            for i in range(peak_info['num_peaks']):
                F_i = self.predict_peak_distribution(tau_plot, distribution, i)
                ax.plot(tau_plot, F_i / scale_factor, label=peak_fit_label[i], **kw)
        else:
            if peak_fit_label is None:
                peak_fit_label = 'Peak fit'
            F_peaks = self.predict_peak_distribution(tau_plot, distribution)
            ax.plot(tau_plot, F_peaks / scale_factor, label=peak_fit_label, **kw)

        if len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend()

        return ax

    # ===============================================
    # Methods for saving and loading fits
    # ===============================================
    def get_fit_attributes(self, which='all'):
        fit_attributes = {
            'common': {
                'core': ['distributions', 'distribution_fits', 'f_train', 'Z_train', '_Z_scale', 'fit_type',
                         'R_inf', 'inductance'],
                'detail': ['distribution_matrices']
            },
            'ridge': {'core': [], 'detail': ['_iter_history']},
            'map': {'core': ['stan_model_name', 'error_fit'],
                    'detail': ['_stan_input', '_init_params', '_opt_result']},
            'bayes': {'core': ['stan_model_name', '_sample_result', 'error_fit'],
                      'detail': ['_stan_input', '_init_params']},
            'map-drift': {'core': ['stan_model_name', 'error_fit', 'drift_offsets'],
                          'detail': ['_stan_input', '_init_params', '_opt_result']}
        }

        if which == 'all':
            att = sum([v for v in fit_attributes['common'].values()], []) + sum(
                [v for v in fit_attributes[self.fit_type].values()], [])
        else:
            att = fit_attributes['common'][which] + fit_attributes[self.fit_type][which]

        return att

    def save_fit_data(self, filename=None, which='all'):
        """
        Save fit data to a file
        Parameters
        ----------
        filename : str (default: None)
            Path to file in which to save fit data. If None, return a dict of fit data
        which : str (default: 'all')
            Which data to save. Options:
                'core': save essential fit parameters only. Does not save matrices required for prediction (these can be
                    recalculated easily), initialization parameters, stan model inputs, or extraneous hyperparameters.
                    Recommended if saving a large number of fits to reduce storage requirements.
                'detail': save detail data and parameters only. Does NOT save essential parameters.
                'all': save both core and detail parameters and data. Requires substantial storage. Recommended if
                    saving a small number of fits to maintain access to all attributes.
        Returns
        -------
        fit_data : dict
            Dict of fit data. Only returned if filename is None
        """
        # get names of attributes to be stored
        store_att = self.get_fit_attributes(which)

        fit_data = {}
        for att in store_att:
            fit_data[att] = getattr(self, att)

        if filename is not None:
            # save to file
            save_pickle(fit_data, filename)
        else:
            # return dict
            return fit_data

    def load_fit_data(self, data):
        if type(data) == str:
            # data is filename - load file
            fit_data = load_pickle(data)
        else:
            # data is dict
            fit_data = data

        # f_train_old = self.f_train.copy()
        f_pred_old = deepcopy(self.f_pred)
        self._cached_distributions = self.distributions.copy()

        for k, v in fit_data.items():
            setattr(self, k, v)

        # If distribution_matrices was not stored, check if we can reuse existing prediction_matrices.
        # distribution_matrices is controlled by recalc_mat, which is already set to True when set_distributions is called.
        # (Also, we only care about distribution_matrices for fitting,, and if we run a new fit we're overwriting the loaded fit data anyway...)
        if 'distribution_matrices' not in fit_data.keys():
            # if distributions have changed, must recalculate prediction_matrices
            # otherwise, existing matrices can be used
            if check_equality(self.distributions,
                              self._cached_distributions):  # and np.min(rel_round(self.f_train,10)==rel_round(f_train_old,10))==True:
                self.f_pred = f_pred_old
            # print('No recalc')
        # else:
        # print('recalc')

    # ===============================================
    # Getters and setters to control matrix recalculation
    # ===============================================
    def get_basis_freq(self):
        return self._basis_freq

    def set_basis_freq(self, basis_freq):
        self._basis_freq = basis_freq
        self._recalc_mat = True
        self.f_pred = None

    basis_freq = property(get_basis_freq, set_basis_freq)

    def get_basis(self):
        return self._basis

    def set_basis(self, basis):
        self._basis = basis
        self._recalc_mat = True
        self.f_pred = None

    basis = property(get_basis, set_basis)

    def get_epsilon(self):
        return self._epsilon

    def set_epsilon(self, epsilon, override_distributions=False):
        self._epsilon = epsilon
        self._recalc_mat = True
        if override_distributions:
            for name in self.distributions.keys():
                self.distributions[name]['epsilon'] = epsilon
        self.f_pred = None

    epsilon = property(get_epsilon, set_epsilon)

    def get_fit_inductance(self):
        return self._fit_inductance_

    def set_fit_inductance(self, fit_inductance):
        self._fit_inductance_ = fit_inductance
        if self._recalc_mat == False and hasattr(self, 'A_im'):
            self.A_im[:, 1] = -2 * np.pi * self.f_train

    fit_inductance = property(get_fit_inductance, set_fit_inductance)
