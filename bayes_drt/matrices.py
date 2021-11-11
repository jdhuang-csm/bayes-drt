import numpy as np
from scipy.integrate import quad
from scipy.linalg import toeplitz

from bayes_drt.utils import rel_round, is_loguniform


def get_basis_func(basis):
    "Generate basis function"
    # y = ln (tau/tau_m)
    if basis == 'gaussian':
        def phi(y, epsilon):
            return np.exp(-(epsilon * y) ** 2)
    elif basis == 'Cole-Cole':
        def phi(y, epsilon):
            return (1 / (2 * np.pi)) * np.sin((1 - epsilon) * np.pi) / (
                    np.cosh(epsilon * y) - np.cos((1 - epsilon) * np.pi))
    elif basis == 'Zic':
        def phi(y, epsilon):
            # epsilon unused, included only for compatibility
            return 2 * np.exp(y) / (1 + np.exp(2 * y))
    else:
        raise ValueError(f'Invalid basis {basis}. Options are gaussian')
    return phi


def get_A_func(part, basis='gaussian', kernel='DRT', dist_type='series', symmetry='planar', bc=None, ct=False,
               k_ct=None):
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
    if ct == True and k_ct is None:
        raise ValueError('k_ct must be supplied if ct==True')

    if kernel == 'DRT':
        if dist_type == 'series':
            if part == 'real':
                def func(y, w_n, t_m, epsilon=1):
                    return phi(y, epsilon) / (1 + np.exp(2 * (y + np.log(w_n * t_m))))
            elif part == 'imag':
                def func(y, w_n, t_m, epsilon=1):
                    return -phi(y, epsilon) * np.exp(y) * w_n * t_m / (1 + np.exp(2 * (y + np.log(w_n * t_m))))
        else:
            raise ValueError('dist_type for DRT kernel must be series')

    elif kernel == 'DDT':
        """Need to add Gerischer-type equivalents"""
        # first define diffusion impedance, Z_D
        if bc == 'blocking':
            if symmetry == 'planar':
                if ct:
                    def Z_D(y, w_n, t_m):
                        # coth(x) = 1/tanh(x)
                        x = np.sqrt(t_m * np.exp(y) * (k_ct + 1j * w_n))
                        return 1 / (np.tanh(x) * x)
                else:
                    def Z_D(y, w_n, t_m):
                        # coth(x) = 1/tanh(x)
                        x = np.sqrt(1j * w_n * t_m * np.exp(y))
                        return 1 / (np.tanh(x) * x)
            # elif symmetry=='cylindrical': # not sure how I_0 and I_1 are defined for cylindrical in Song and Bazant (2018)
            elif symmetry == 'spherical':
                if ct:
                    def Z_D(y, w_n, t_m):
                        x = np.sqrt(t_m * np.exp(y) * (k_ct + 1j * w_n))
                        return np.tanh(x) / (x - np.tanh(x))
                else:
                    def Z_D(y, w_n, t_m):
                        x = np.sqrt(1j * w_n * t_m * np.exp(y))
                        return np.tanh(x) / (x - np.tanh(x))
            else:
                raise ValueError(f'Invalid symmetry {symmetry}. Options are planar or spherical for bc=blocking')
        elif bc == 'transmissive':
            if symmetry == 'planar':
                if ct:
                    def Z_D(y, w_n, t_m):
                        x = np.sqrt(t_m * np.exp(y) * (k_ct + 1j * w_n))
                        return np.tanh(x) / x
                else:
                    def Z_D(y, w_n, t_m):
                        x = np.sqrt(1j * w_n * t_m * np.exp(y))
                        return np.tanh(x) / x
            else:
                raise ValueError(f'Invalid symmetry {symmetry}. Symmetry must be planar for bc=transmissive')

        # then choose whether to integrate Z_D or Y_D
        if dist_type == 'parallel':
            if part == 'real':
                def func(y, w_n, t_m, epsilon=1):
                    return phi(y, epsilon) * np.real(1 / Z_D(y, w_n, t_m))
            elif part == 'imag':
                def func(y, w_n, t_m, epsilon=1):
                    return phi(y, epsilon) * np.imag(1 / Z_D(y, w_n, t_m))
        elif dist_type == 'series':
            if part == 'real':
                def func(y, w_n, t_m, epsilon=1):
                    return phi(y, epsilon) * np.real(Z_D(y, w_n, t_m))
            elif part == 'imag':
                def func(y, w_n, t_m, epsilon=1):
                    return phi(y, epsilon) * np.imag(Z_D(y, w_n, t_m))
        else:
            raise ValueError(f'Invalid dist_type {dist_type}. Options are series and parallel')

    else:
        raise ValueError(f'Invalid kernel {kernel}. Options are DRT and DDT')

    return func


def construct_A(frequencies, part, tau=None, basis='gaussian', fit_inductance=False, epsilon=1,
                kernel='DRT', dist_type='series', symmetry='planar', bc=None, ct=False, k_ct=None,
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

    omega = frequencies * 2 * np.pi

    # check if tau is inverse of omega
    if tau is None:
        tau = 1 / omega
        tau_eq_omega = True
    elif len(tau) == len(omega):
        if np.min(rel_round(tau, 10) == rel_round(1 / omega, 10)):
            tau_eq_omega = True
        else:
            tau_eq_omega = False
    else:
        tau_eq_omega = False

    # check if omega is subset of inverse tau
    # find index where first frequency matches tau
    match = rel_round(1 / omega[0], 10) == rel_round(tau, 10)
    if np.sum(match) == 1:
        start_idx = np.where(match == True)[0][0]
        # if tau vector starting at start_idx matches omega, omega is a subset of tau
        if np.min(rel_round(tau[start_idx:start_idx + len(omega)], 10) == rel_round(1 / omega, 10)):
            freq_subset = True
        else:
            freq_subset = False
    elif np.sum(match) == 0:
        # if no tau corresponds to first omega, omega is not a subset of tau
        freq_subset = False
    else:
        # if more than one match, must be duplicates in tau
        raise Exception('Repeated tau values')

    if freq_subset == False:
        # check if tau is subset of inverse omega
        # find index where first frequency matches tau
        match = rel_round(1 / omega, 10) == rel_round(tau[0], 10)
        if np.sum(match) == 1:
            start_idx = np.where(match == True)[0][0]
            # if omega vector starting at start_idx matches tau, tau is a subset of omega
            if np.min(rel_round(omega[start_idx:start_idx + len(tau)], 10) == rel_round(1 / tau, 10)):
                freq_subset = True
            else:
                freq_subset = False
        elif np.sum(match) == 0:
            # if no omega corresponds to first tau, tau is not a subset of omega
            freq_subset = False
        else:
            # if more than one match, must be duplicates in omega
            raise Exception('Repeated omega values')

    # Determine if A is a Toeplitz matrix
    # Note that when there is simultaneous charge transfer, the matrix is never a Toeplitz matrix
    # because the integrand can no longer be written in terms of w_n*t_m only
    if is_loguniform(frequencies) and ct == False:
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
    func = get_A_func(part, basis, kernel, dist_type, symmetry, bc, ct, k_ct)

    if is_toeplitz:  # is_loguniform(frequencies) and tau_eq_omega:
        # only need to calculate 1st row and column
        w_0 = omega[0]
        t_0 = tau[0]
        if part == 'real':
            if basis == 'Zic':
                quad_limits = (-100, 100)
            elif kernel == 'DDT':
                quad_limits = (-20, 20)
            else:
                quad_limits = (-np.inf, np.inf)
        elif part == 'imag':
            # scipy.integrate.quad is unstable for imag func with infinite limits
            quad_limits = (-20, 20)
        #		  elif part=='imag':
        #			  y = np.arange(-5,5,0.1)
        #			  c = [np.trapz(func(y,w_n,t_0),x=y) for w_n in omega]
        #			  r = [np.trapz(func(y,w_0,t_m),x=y) for t_m in 1/omega]

        if integrate_method == 'quad':
            c = [quad(func, quad_limits[0], quad_limits[1], args=(w_n, t_0, epsilon), epsabs=1e-4)[0] for w_n in omega]
            r = [quad(func, quad_limits[0], quad_limits[1], args=(w_0, t_m, epsilon), epsabs=1e-4)[0] for t_m in tau]
        elif integrate_method == 'trapz':
            y = np.linspace(-20, 20, 1000)
            c = [np.trapz(func(y, w_n, t_0, epsilon), x=y) for w_n in omega]
            r = [np.trapz(func(y, w_0, t_m, epsilon), x=y) for t_m in tau]
        if r[0] != c[0]:
            print(r[0], c[0])
            raise Exception('First entries of first row and column are not equal')
        A = toeplitz(c, r)
    else:
        # need to calculate all entries
        if part == 'real':
            if basis == 'Zic':
                quad_limits = (-20, 20)
            elif kernel == 'DDT':
                quad_limits = (-20, 20)
            else:
                quad_limits = (-np.inf, np.inf)
        elif part == 'imag':
            # scipy.integrate.quad is unstable for imag func with infinite limits
            quad_limits = (-20, 20)

        A = np.empty((len(frequencies), len(tau)))
        for n, w_n in enumerate(omega):
            if integrate_method == 'quad':
                A[n, :] = [quad(func, quad_limits[0], quad_limits[1], args=(w_n, t_m, epsilon), epsabs=1e-4)[0] for t_m
                           in tau]
            elif integrate_method == 'trapz':
                y = np.linspace(-20, 20, 1000)
                A[n, :] = [np.trapz(func(y, w_n, t_m, epsilon), x=y) for t_m in tau]

    return A


def construct_L(frequencies, tau=None, basis='gaussian', epsilon=1, order=1):
    "Differentiation matrix. L@coef gives derivative of DRT"
    omega = 2 * np.pi * frequencies
    if tau is None:
        # if no time constants given, assume collocated with frequencies
        tau = 1 / omega

    L = np.zeros((len(omega), len(tau)))

    if basis == 'gaussian':
        if type(order) == list:
            f0, f1, f2 = order

            def dphi_dy(y, epsilon):
                "Mixed derivative of Gaussian RBF"
                return f0 * np.exp(-(epsilon * y) ** 2) + f1 * (-2 * epsilon ** 2 * y * np.exp(-(epsilon * y) ** 2)) \
                       + f2 * (-2 * epsilon ** 2 + 4 * epsilon ** 4 * y ** 2) * np.exp(-(epsilon * y) ** 2)
        elif order == 0:
            def dphi_dy(y, epsilon):
                "Gaussian RBF"
                return np.exp(-(epsilon * y) ** 2)
        elif order == 1:
            def dphi_dy(y, epsilon):
                "Derivative of Gaussian RBF"
                return -2 * epsilon ** 2 * y * np.exp(-(epsilon * y) ** 2)
        elif order == 2:
            def dphi_dy(y, epsilon):
                "2nd derivative of Gaussian RBF"
                return (-2 * epsilon ** 2 + 4 * epsilon ** 4 * y ** 2) * np.exp(-(epsilon * y) ** 2)
        elif order == 3:
            def dphi_dy(y, epsilon):
                "3rd derivative of Gaussian RBF"
                return (12 * epsilon ** 4 * y - 8 * epsilon ** 6 * y ** 3) * np.exp(-(epsilon * y) ** 2)
        elif order > 0 and order < 1:
            f0 = 1 - order
            f1 = order

            def dphi_dy(y, epsilon):
                "Mixed derivative of Gaussian RBF"
                return f0 * np.exp(-(epsilon * y) ** 2) + f1 * (-2 * epsilon ** 2 * y * np.exp(-(epsilon * y) ** 2))
        elif order > 1 and order < 2:
            f1 = 2 - order
            f2 = order - 1

            def dphi_dy(y, epsilon):
                "Mixed derivative of Gaussian RBF"
                return f1 * (-2 * epsilon ** 2 * y * np.exp(-(epsilon * y) ** 2)) + f2 * (
                        -2 * epsilon ** 2 + 4 * epsilon ** 4 * y ** 2) * np.exp(-(epsilon * y) ** 2)
        else:
            raise ValueError('Order must be between 0 and 3')
    elif basis == 'Zic':
        if order == 0:
            dphi_dy = get_basis_func(basis)

    for n in range(len(omega)):
        L[n, :] = [dphi_dy(np.log(1 / (omega[n] * t_m)), epsilon) for t_m in tau]

    return L


def get_M_func(basis='gaussian', order=1):
    """
	Create function for M matrix

	Parameters:
	-----------
	basis : string, optional (default: 'gaussian')
		Basis function used to approximate DRT
	order : int, optional (default: 1)
		Order of DRT derivative for ridge penalty
	"""

    if order == 0:
        if basis == 'gaussian':
            def func(w_n, t_m, epsilon):
                a = epsilon * np.log(1 / (w_n * t_m))
                return (np.pi / 2) ** 0.5 * epsilon ** (-1) * np.exp(-(a ** 2 / 2))
        else:
            raise ValueError(f'Invalid basis {basis}')
    elif order == 1:
        if basis == 'gaussian':
            def func(w_n, t_m, epsilon):
                a = epsilon * np.log(1 / (w_n * t_m))
                return -(np.pi / 2) ** 0.5 * epsilon * (-1 + a ** 2) * np.exp(-(a ** 2 / 2))
        else:
            raise ValueError(f'Invalid basis {basis}')
    elif order == 2:
        if basis == 'gaussian':
            def func(w_n, t_m, epsilon):
                a = epsilon * np.log(1 / (w_n * t_m))
                return (np.pi / 2) ** 0.5 * epsilon ** 3 * (3 - 6 * a ** 2 + a ** 4) * np.exp(-(a ** 2 / 2))
        else:
            raise ValueError(f'Invalid basis {basis}')
    else:
        raise ValueError(f'Invalid order {order}')
    return func


def construct_M(frequencies, basis='gaussian', order=1, epsilon=1):
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
    omega = frequencies * 2 * np.pi

    if type(order) == list:
        f0, f1, f2 = order
        func0 = get_M_func(basis, 0)
        func1 = get_M_func(basis, 1)
        func2 = get_M_func(basis, 2)

        def func(w_n, t_m, epsilon):
            return f0 * func0(w_n, t_m, epsilon) + f1 * func1(w_n, t_m, epsilon) + f2 * func2(w_n, t_m, epsilon)

    else:
        func = get_M_func(basis, order)

    if is_loguniform(frequencies):
        # only need to calculate 1st column (symmetric toeplitz matrix) IF gaussian - may not apply to other basis functions
        # w_0 = omega[0]
        t_0 = 1 / omega[0]

        c = [func(w_n, t_0, epsilon) for w_n in omega]
        # r = [quad(func,limits[0],limits[1],args=(w_0,t_m,epsilon),epsabs=1e-4)[0] for t_m in 1/omega]
        # if r[0]!=c[0]:
        # raise Exception('First entries of first row and column are not equal')
        M = toeplitz(c)
    else:
        # need to calculate all entries
        M = np.empty((len(frequencies), len(frequencies)))
        for n, w_n in enumerate(omega):
            M[n, :] = [func(w_n, t_m, epsilon) for t_m in 1 / omega]

    return M