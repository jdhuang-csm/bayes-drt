import numpy as np

from scipy.signal import find_peaks
from scipy.optimize import least_squares


def HN_distribution(tau, t0, alpha, beta):
    """
	Analytical DRT of Havriliak-Negami relaxation.
	When alpha=1, equivalent to ZARC/RQ element.
	When beta=1, equivalent to Cole-Davidson.
	When alpha=0.5 and beta=1, equivalent to Gerischer element.
	
	Parameters:
	-----------
	tau: array
		Timescales at which to evaluate the DRT
	t0: float
		Characteristic time constant
	alpha: float
		First shape parameter. Ranges from 0 to 1
	beta: float
		Second shape parameter. Ranges from 0 to 1
	"""
    theta = np.arctan2(np.sin(np.pi * beta), ((tau / t0) ** beta + np.cos(np.pi * beta)))
    g = (1 / np.pi) * (tau / t0) ** (beta * alpha) * np.sin(alpha * theta) / (
                1 + 2 * np.cos(np.pi * beta) * (tau / t0) ** beta + (tau / t0) ** (2 * beta)) ** (alpha / 2)
    return g


def HN_impedance(freq, t0, alpha, beta):
    omega = freq * 2 * np.pi
    return 1 / (1 + (1j * omega * t0) ** beta) ** alpha


def evaluate_fit_distribution(x, tau):
    if len(x) % 4 != 0:
        raise ValueError('Number of parameters must be a multiple of 4')

    gamma = np.zeros_like(tau)

    num_peaks = int(len(x) / 4)
    for i in range(0, num_peaks):
        xi = x[4 * i:4 * i + 4]
        R, log_t0, alpha, beta = xi
        gamma += R * HN_distribution(tau, np.exp(log_t0), alpha, beta)

    return gamma


def evaluate_fit_impedance(x, freq, R_inf=0, inductance=0):
    if len(x) % 4 != 0:
        raise ValueError('Number of parameters must be a multiple of 4')

    Z = np.zeros(len(freq), dtype=complex)

    num_peaks = int(len(x) / 4)
    for i in range(0, num_peaks):
        xi = x[4 * i:4 * i + 4]
        R, log_t0, alpha, beta = xi
        Z += R * HN_impedance(freq, np.exp(log_t0), alpha, beta)

    Z += R_inf + 1j * inductance * freq * 2 * np.pi

    return Z


def peak_fit_residuals(x, tau, gamma, Rp, weights, l1_penalty, l2_penalty):
    resid = evaluate_fit_distribution(x, tau) - gamma
    l1 = np.sqrt(np.abs(x[::4] / Rp)) * l1_penalty
    l2 = (x[::4] / Rp) * l2_penalty
    Rp_resid = 2 * (np.sum(x[::4]) - Rp) / Rp  # add a penalty for mismatch with Rp from DRT
    return np.concatenate([resid * weights, l1, l2, [Rp_resid]])


def fit_peaks(tau, gamma, Rp, weights=None, nonneg=True, check_shoulders=False, prom_rthresh=0.001, R_rthresh=0.005,
              check_chi_sq=False, chi_sq_thresh=0.4, chi_sq_delta=0.2, l1_penalty=0, l2_penalty=0.01):
    """
	Fit HN peaks to distribution. 
	First identify number of peaks and their locations, 
	then optimize HN parameters to match gamma.
	
	Parameters:
	-----------
	tau: array
		Array of tau values at which gamma is evaluated
	gamma: array
		Distribution to fit
	weights: array, optional (default: None)
		Weights for fitting gamma. Must match length of gamma
		
	
	"""

    if nonneg:
        x = fit_pos_peaks(tau, gamma, Rp, weights, check_shoulders, prom_rthresh, R_rthresh, check_chi_sq, chi_sq_thresh,
                          chi_sq_delta, None, l1_penalty, l2_penalty)
    else:
        # Fit positive and negative portions of gamma separately
        gamma_zeros = np.array([gamma, np.zeros_like(gamma)])
        gamma_pos = np.max(gamma_zeros, axis=0)
        gamma_neg = np.min(gamma_zeros, axis=0)
        # print(gamma_neg)
        min_weight_deno = np.percentile(np.abs(gamma), 80)
        x_pos = fit_pos_peaks(tau, gamma_pos, Rp, weights, check_shoulders, prom_rthresh, R_rthresh, check_chi_sq,
                              chi_sq_thresh, chi_sq_delta, min_weight_deno, l1_penalty, l2_penalty)
        x_neg = fit_pos_peaks(tau, -gamma_neg, Rp, weights, check_shoulders, prom_rthresh, R_rthresh, check_chi_sq,
                              chi_sq_thresh, chi_sq_delta, min_weight_deno, l1_penalty, l2_penalty)
        x_neg[0::4] *= -1  # make R values negative in x_neg
        x0 = np.concatenate((x_pos, x_neg))

        # Perform a final fit of positive and negative peaks together
        # set bounds: 0<=alpha<=1, 0<=beta<=1. log_tau must be +/-0.1 from value obtained from separate fits
        weights = 1 / (gamma + min_weight_deno)
        lb = np.zeros_like(x0)
        ub = np.zeros_like(x0)
        num_peaks = int(len(x0) / 4)
        for i in range(0, num_peaks):
            xi = x0[4 * i:4 * i + 4]
            R, log_t0, alpha, beta = xi
            lb[4 * i:4 * i + 4] = [-np.inf, log_t0 - 0.1, 0, 0]
            ub[4 * i:4 * i + 4] = [np.inf, log_t0 + 0.1, 1, 1]

        result = least_squares(peak_fit_residuals, x0, args=(tau, gamma, Rp, weights, l1_penalty, l2_penalty),
                               bounds=(lb, ub))
        x = filter_peaks(result['x'], R_rthresh, Rp)

    return x


def fit_pos_peaks(tau, gamma, Rp, weights=None, check_shoulders=False, prom_rthresh=0.001, R_rthresh=0.005,
                  check_chi_sq=False, chi_sq_thresh=0.4, chi_sq_delta=0.2, min_weight_deno=None,
                  l1_penalty=0, l2_penalty=0.01):
    """
	Fit HN peaks to distribution. 
	First identify number of peaks and their locations, 
	then optimize HN parameters to match gamma.
	
	Parameters:
	-----------
	tau: array
		Array of tau values at which gamma is evaluated
	gamma: array
		Distribution to fit
	weights: array, optional (default: None)
		Weights for fitting gamma. Must match length of gamma
		
	
	"""
    if len(tau) != len(gamma):
        raise ValueError('tau and gamma must have same length')

    # identify peaks
    peaks, properties = find_peaks(gamma, width=1, prominence=prom_rthresh * Rp)  # np.max(gamma))
    if len(peaks) == 0:
        return []


    # get initial parameter estimates
    x0 = np.zeros(len(peaks) * 4)  # each peak has 4 parameters: R, t0, gamma, beta
    for i, peak in enumerate(peaks):
        width = properties['widths'][i]
        start = int(peak - width)
        end = int(peak + width)
        R = np.trapz(gamma[start:end], np.log(tau[start:end]))
        t0 = tau[peak]
        alpha = 0.99  # 1 is symmetric
        beta = 0.8  # may be able to estimate from width
        x0[4 * i:4 * i + 4] = [R, np.log(t0), alpha, beta]

    if weights is None:
        if min_weight_deno is None:
            min_weight_deno = max(np.percentile(gamma, 80), np.max(gamma) / 50)
        weights = 1 / (gamma + min_weight_deno)
    # print(gamma, np.percentile(gamma,80))
    # print(weights)
    elif len(weights) != len(gamma):
        raise ValueError('Length of weights must match length of gamma')

    # fit parameters to gamma
    # set bounds: R>=0, 0<=alpha<=1, 0<=beta<=1. log_tau must be +/-0.25 from estimate from find_peaks
    lb = np.zeros_like(x0)
    ub = np.zeros_like(x0)
    num_peaks = int(len(x0) / 4)
    for i in range(0, num_peaks):
        xi = x0[4 * i:4 * i + 4]
        R, log_t0, alpha, beta = xi
        lb[4 * i:4 * i + 4] = [0, log_t0 - 0.25, 0, 0]
        ub[4 * i:4 * i + 4] = [np.inf, log_t0 + 0.25, 1, 1]

    result = least_squares(peak_fit_residuals, x0, args=(tau, gamma, Rp, weights, l1_penalty, l2_penalty),
                           bounds=(lb, ub))

    # only keep peaks that meet relative R threshold
    x_filter = filter_peaks(result['x'], R_rthresh, Rp)
    num_peaks = int(len(x_filter) / 4)

    if check_shoulders:
        # check for shoulders or minor peaks missed by find_peaks
        # use unfiltered peaks to avoid re-identifying insignificant peaks
        gamma_fit = evaluate_fit_distribution(result['x'], tau)
        dg = np.diff(gamma)
        pos_peaks, _ = find_peaks(dg)
        neg_peaks, _ = find_peaks(-dg)

        # print('pos and neg peaks:', len(neg_peaks), len(pos_peaks))

        if neg_peaks[0] < pos_peaks[0]:
            # if first peak is negative, assume that a positive peak precedes it but is not captured in the tau range
            pos_peaks = np.insert(pos_peaks, 0, 0)
        if pos_peaks[-1] > neg_peaks[-1]:
            # If last peak is positive, assume that a negative peak follows it but is not captured in the tau range
            neg_peaks = np.append(neg_peaks, len(tau) - 1)

        new_peaks = []
        new_peak_widths = []
        # print(pos_peaks,neg_peaks)
        if len(pos_peaks) == len(neg_peaks):
            for pos, neg in zip(pos_peaks, neg_peaks):
                # a peak should exist between each positive peak and the following negative peak
                peak_idx = np.where((pos <= peaks) & (peaks <= neg))
                if len(peak_idx[0]) == 0:
                    # If peak does not already exist in the interval, add a new peak

                    # check relative area and max height of residual in potential shoulder region
                    # rarea = np.sum((gamma-gamma_fit)[pos:neg]/np.sum(gamma))
                    # rheight = np.max((gamma-gamma_fit)[pos:neg])/np.mean(gamma)
                    # if rarea > 0.00 and rheight >= 0.1:
                    # If area exceeds 0.5% of total area and max height exceeds 10% of mean(gamma), mark new peak
                    # Place new peak at largest residual between pos and neg peaks
                    new_peak_idx = pos + np.argmax((gamma - gamma_fit)[pos:neg])
                    new_peaks.append(new_peak_idx)
                    new_peak_widths.append(pos - neg)
        else:
            raise Exception('Different numbers of pos and neg peaks!')

        if len(new_peaks) > 0:
            # print('Found {} shoulder peak(s)'.format(len(new_peaks)))
            x0 = np.zeros(len(x_filter) + len(new_peaks) * 4)
            x0[:len(x_filter)] = x_filter
            for i, peak in enumerate(new_peaks):
                width = new_peak_widths[i]
                start = int(peak - width)
                end = int(peak + width)
                R = np.trapz((gamma - gamma_fit)[start:end], np.log(tau[start:end]))
                if R <= 0:
                    R = gamma[peak]
                t0 = tau[peak]
                alpha = 0.99
                beta = 0.8
                x0[4 * (i + num_peaks):4 * (i + num_peaks) + 4] = [R, np.log(t0), alpha, beta]

            # set bounds: R>=0, 0<=alpha<=1, 0<=beta<=1. log_tau must be +/-0.25 from previous estimate
            lb = np.zeros_like(x0)
            ub = np.zeros_like(x0)
            num_peaks = int(len(x0) / 4)
            for i in range(0, num_peaks):
                xi = x0[4 * i:4 * i + 4]
                R, log_t0, alpha, beta = xi
                lb[4 * i:4 * i + 4] = [0, log_t0 - 0.25, 0, 0]
                ub[4 * i:4 * i + 4] = [np.inf, log_t0 + 0.25, 1, 1]
            # print(x0)
            result = least_squares(peak_fit_residuals, x0, args=(tau, gamma, Rp, weights, l1_penalty, l2_penalty),
                                   bounds=(lb, ub))

            x_filter = filter_peaks(result['x'], R_rthresh, Rp)

    if check_chi_sq:
        # Check chi_sq of fit. If above threshold, add a peak and re-fit
        # May be helpful for capturing smooth shoulders that are not
        # distinct enough to be caught by check_shoulders
        chi_sq = np.sum((resid(x_filter, weights)) ** 2)
        num_peaks = int(len(x_filter) / 4)
        if chi_sq > chi_sq_thresh:
            # If chi_sq exceeds threshold, add peak and re-fit
            x0 = np.zeros(len(x_filter) + 4)
            x0[:len(x_filter)] = x_filter
            # initialize new peak at largest misfit
            gamma_fit = evaluate_fit_distribution(x_filter, tau)
            peak = np.argmax(gamma - gamma_fit)
            R = np.trapz((gamma - gamma_fit), np.log(tau))
            if R <= 0:
                R = gamma[peak]
            t0 = tau[peak]
            alpha = 0.99
            beta = 0.8
            x0[len(x_filter):] = [R, np.log(t0), alpha, beta]

            # Set bounds: R>=0, 0<=alpha<=1, 0<=beta<=1.
            # For existing peaks, log_tau must be +/-0.25 from previous estimate
            # For new peak, allow log_tau to go anywhere within tau
            lb = np.zeros_like(x0)
            ub = np.zeros_like(x0)
            for i in range(0, num_peaks + 1):
                xi = x0[4 * i:4 * i + 4]
                R, log_t0, alpha, beta = xi
                if i == num_peaks:
                    # Allow new peak tau_0 to go anywhere within tau range
                    logtau_min = np.log(np.min(tau))
                    logtau_max = np.log(np.max(tau))
                else:
                    # Constrain log_tau for existing peaks
                    logtau_min = log_t0 - 0.25
                    logtau_max = log_t0 + 0.25
                lb[4 * i:4 * i + 4] = [0, log_t0 - 0.25, 0, 0]
                ub[4 * i:4 * i + 4] = [np.inf, log_t0 + 0.25, 1, 1]

            result = least_squares(peak_fit_residuals, x0, args=(tau, gamma, Rp, weights, l1_penalty, l2_penalty),
                                   bounds=(lb, ub))

            x_filter_new = filter_peaks(result['x'], R_rthresh, Rp)
            new_chi_sq = np.sum((resid(x_filter_new, weights)) ** 2)
            # check if chi_sq improvement exceeds chi_sq_delta
            if new_chi_sq <= chi_sq - chi_sq_delta:
                x_filter = x_filter_new

    return x_filter


def fit_data(x0, freq, Z, R_inf=0, inductance=0, weights=None, lambda_x=10):
    # get weights
    if weights is None or weights == 'unity':
        weights = np.ones_like(freq) * (1 + 1j)
    elif type(weights) == str:
        if weights == 'modulus':
            weights = (1 + 1j) / np.sqrt(np.real(Z * Z.conjugate()))
        elif weights == 'Orazem':
            weights = (1 + 1j) / (np.abs(Z.real) + np.abs(Z.imag))
        elif weights == 'proportional':
            weights = 1 / np.abs(Z.real) + 1j / np.abs(Z.imag)
        elif weights == 'prop_adj':
            Zmod = np.real(Z * Z.conjugate())
            weights = 1 / (np.abs(Z.real) + np.percentile(Zmod, 25)) + 1j / (np.abs(Z.imag) + np.percentile(Zmod, 25))
        else:
            raise ValueError(
                f"Invalid weights argument {weights}. String options are 'unity', 'modulus', 'proportional', and 'prop_adj'")
    elif type(weights) in (float, int):
        # assign constant value
        weights = np.ones_like(freq) * (1 + 1j) * weights
    elif type(weights) == complex:
        # assign constant value
        weights = np.ones_like(freq) * weights
    elif len(weights) != len(freq):
        raise ValueError("Weights array must match length of data")

    # print(np.mean(np.abs(Z.real*weights.real)))
    # print(np.mean(np.abs(Z.imag*weights.imag)))
    flat_weights = np.concatenate((weights.real, weights.imag))

    def resid(x, flat_weights, lambda_x):
        # get Z residuals
        Z_resid = evaluate_fit_impedance(x, freq, R_inf, inductance) - Z
        flat_resid = np.concatenate((Z_resid.real, Z_resid.imag))
        Z_resid = flat_resid * flat_weights
        Z_resid_scaled = Z_resid / len(Z_resid)

        # parameter residuals
        x_resid = x - x0
        R_resid = x_resid[::4] / (0.05 * x0[::4])  # sigma = 0.05*R_0
        logt_resid = x_resid[1::4] / 0.2
        alpha_resid = x_resid[2::4] / 0.15
        beta_resid = x_resid[3::4] / 0.15
        x_resid_scaled = np.concatenate((R_resid, logt_resid, alpha_resid, beta_resid))
        x_resid_scaled /= len(x0)

        return np.concatenate((Z_resid_scaled, lambda_x * x_resid_scaled))

    # set bounds
    lb = np.zeros_like(x0)
    ub = np.zeros_like(x0)
    num_peaks = int(len(x0) / 4)
    for i in range(0, num_peaks):
        xi = x0[4 * i:4 * i + 4]
        R, log_t0, alpha, beta = xi
        lb[4 * i:4 * i + 4] = [0, log_t0 - 1, 0, 0]
        ub[4 * i:4 * i + 4] = [np.inf, log_t0 + 1, 1, 1]

    result = least_squares(resid, x0, args=(flat_weights, lambda_x),
                           bounds=(lb, ub))  # =([0,-np.inf,0,0]*len(peaks),[np.inf,np.inf,1,1]*len(peaks)))

    # print(np.sum(Z_resid**2),np.sum(R_resid**2),np.sum(logt_resid**2),np.sum(alpha_resid**2),np.sum(beta_resid**2))

    return result


def filter_peaks(x, rthresh, Rp):
    # get R values relative to max
    R_vals = x[::4]
    R_rel = np.abs(R_vals / Rp)  #/ np.max(np.abs(R_vals))
    # get indices of significant peaks
    big_idx = np.where(R_rel >= rthresh)

    # place significant peak parameters in new x vector
    x_out = np.zeros(4 * len(big_idx[0]))
    for i, idx in enumerate(big_idx[0]):
        x_out[4 * i:4 * i + 4] = x[4 * idx:4 * idx + 4]

    return x_out


def constrained_peak_fit(tau, gamma, tau0_guess, Rp, nonneg, lntau_uncertainty=3, sigma_lntau=5, weights=None,
                         l2_penalty=0.01):
    num_peaks = len(tau0_guess)

    if len(tau) != len(gamma):
        raise ValueError('tau and gamma must have same length')

    if weights is None:
        weights = 1 / (gamma + np.percentile(np.abs(gamma), 80))
    elif len(weights) != len(gamma):
        raise ValueError('Length of weights must match length of gamma')

    # Initial parameter guesses
    x0 = np.zeros(num_peaks * 4)  # each peak has 4 parameters: R, t0, gamma, beta
    for n in range(num_peaks):
        peak_width = 4  # peak width in ln_tau space
        start = np.argmin(np.abs(tau - (tau0_guess[n] * np.exp(-peak_width / 2))))
        end = np.argmin(np.abs(tau - (tau0_guess[n] * np.exp(peak_width / 2))))
        R = np.trapz(gamma[start:end], np.log(tau[start:end]))
        t0 = tau0_guess[n]
        alpha = 0.99
        beta = 0.8
        x0[n * 4:n * 4 + 4] = [R, np.log(t0), alpha, beta]

    # fit parameters to gamma
    def resid(x, weights):
        gamma_resid = evaluate_fit_distribution(x, tau) - gamma
        lntau0 = x[1::4]
        tau_resid = (lntau0 - np.log(tau0_guess))
        l2 = (x[::4] / Rp) * l2_penalty
        Rp_resid = 2 * (np.sum(x[::4]) - Rp) / Rp  # add a penalty for mismatch with Rp from DRT
        return np.concatenate((gamma_resid * weights, tau_resid / sigma_lntau, l2, [Rp_resid]))

    # set bounds: R>=0, 0<=alpha<=1, 0<=beta<=1. log_tau must be +/-lntau_uncertainty from tau0_guess
    lb = np.zeros_like(x0)
    ub = np.zeros_like(x0)
    num_peaks = int(len(x0) / 4)
    for i in range(0, num_peaks):
        xi = x0[4 * i:4 * i + 4]
        R, log_t0, alpha, beta = xi
        if nonneg:
            # Constrain R to non-negative values
            R_lb = 0
            R_ub = np.inf
        else:
            # Constrain R to have same sign as initial estimate from integration
            if R > 0:
                R_lb = 0
                R_ub = np.inf
            else:
                R_lb = -np.inf
                R_ub = 0
        lb[4 * i:4 * i + 4] = [R_lb, log_t0 - lntau_uncertainty, 0, 0]
        ub[4 * i:4 * i + 4] = [R_ub, log_t0 + lntau_uncertainty, 1, 1]

    result = least_squares(resid, x0, args=(weights, ), bounds=(lb, ub))

    return result
