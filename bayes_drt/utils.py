import re
import numpy as np
import pandas as pd


# Data scaling
# ------------
def get_unit_scale(df, area=None):
    """ Get unit scale (mu, m, k, M, G) for EIS data"""
    if area is None:
        area = 1
    unit_map = {-2: '$\mu$', -1: 'm', 0: '', 1: 'k', 2: 'M', 3: 'G'}
    Z_max = max(df['Zreal'].max(), df['Zimag'].abs().max())
    Z_max *= area
    Z_ord = np.floor(np.log10(Z_max) / 3)
    unit_scale = unit_map.get(Z_ord, '')
    return unit_scale


def get_scale_factor(df, area=None):
    if area is None:
        area = 1
    Z_max = max(df['Zreal'].max(), df['Zimag'].abs().max())
    Z_max *= area
    Z_ord = np.floor(np.log10(Z_max) / 3)
    return 10 ** (3 * Z_ord)


def get_factor_from_unit(unit_scale):
    unit_map = {-2: '$\mu$', -1: 'm', 0: '', 1: 'k', 2: 'M', 3: 'G'}
    pwr_map = {v: k for k, v in unit_map.items()}
    pwr = pwr_map[unit_scale]
    return 10 ** (3 * pwr)


def get_common_unit_scale(df_list, aggregate='min'):
    """
	Get common unit scale for multiple datasets
	Parameters:
		df_list: list of DataFrames
		aggregate: method for choosing common scale. Defaults to min (smallest scale)
	"""
    unit_map = {-2: '$\mu$', -1: 'm', 0: '', 1: 'k', 2: 'M', 3: 'G'}
    rev_map = {v: k for k, v in unit_map.items()}
    units = [get_unit_scale(df) for df in df_list]
    unit_nums = [rev_map[u] for u in units]
    common_num = getattr(np, aggregate)(unit_nums)
    common_unit = unit_map.get(common_num, '')
    return common_unit


def polar_from_complex(data):
    if type(data) == pd.core.frame.DataFrame:
        Zmod = (data['Zreal'].values ** 2 + data['Zimag'].values ** 2) ** 0.5
        Zphz = (180 / np.pi) * np.arctan(data['Zimag'].values / data['Zreal'].values)
    elif type(data) == np.ndarray:
        Zmod = ((data * data.conjugate()) ** 0.5).real
        Zphz = (180 / np.pi) * np.arctan(data.imag / data.real)

    return Zmod, Zphz


def complex_from_polar(data):
    if type(data) == pd.core.frame.DataFrame:
        Zmod = data['Zmod'].values
        Zphz = data['Zphz'].values
    elif type(data) == np.ndarray:
        Zmod = data[:, 0]
        Zphz = data[:, 1]

    Zreal = Zmod * np.cos(np.pi * Zphz / 180)
    Zimag = Zmod * np.sin(np.pi * Zphz / 180)

    return Zreal, Zimag


# Miscellaneous functions
# -----------------------
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def camel_case_split(identifier):
    # from https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z0-9])|(?<=[A-Z0-9])(?=[A-Z0-9][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def check_equality(a, b):
    """
	Convenience function for testing equality of arrays and dictionaries containing arrays

	Parameters:
	-----------
	a: dict or array
		First object
	b: dict or array
		Second object
	"""
    out = True
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        out = False

    return out


def rel_round(x, precision):
    """Round to relative precision

	Parameters
	----------
	x : array
		array of numbers to round
	precision : int
		number of digits to keep
	"""
    # add 1e-30 for safety in case of zeros in x
    x_scale = np.floor(np.log10(np.array(x) + 1e-30))
    digits = (precision - x_scale).astype(int)
    # print(digits)
    if type(x) in (list, np.ndarray):
        x_round = np.array([round(xi, di) for xi, di in zip(x, digits)])
    else:
        x_round = round(x, digits)
    return x_round


def is_loguniform(frequencies):
    "Check if frequencies are uniformly log-distributed"
    fdiff = np.diff(np.log(frequencies))
    if np.std(fdiff) / np.mean(fdiff) <= 0.01:
        return True
    else:
        return False


def get_outlier_thresh(y, iqr_factor=3):
    "Get outlier detection threshold using IQR"
    iqr = np.percentile(y, 75) - np.percentile(y, 25)
    return np.percentile(y, 75) + iqr_factor * iqr


def r2_score(y, y_hat, weights=None):
    """
	Calculate r^2 score

	Parameters:
	-----------
		y: y values
		y_hat: predicted y values
		weights: sample weights
	"""
    if weights is None:
        ss_res = np.sum((y_hat - y) ** 2)  # np.var(y_hat-y)
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # np.var(y)
    else:
        ss_res = np.sum(weights * (y_hat - y) ** 2)
        ss_tot = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
    return 1 - (ss_res / ss_tot)