import glob
import os
import re
from io import StringIO

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import binom
from scipy.stats import mode, iqr

from bayes_drt.file_load import get_timestamp, read_eis, read_jv, get_file_source
from bayes_drt.plotting import plot_bode, plot_nyquist, plot_jv, plot_ocv
from bayes_drt.utils import get_unit_scale, camel_case_split


def lsv_resistance(df):
    """
	Calculate resistance from LSV file
	Args:
		df: dataframe of LSV data
	"""
    return np.polyfit(df['Im'], df['Vf'], deg=1)[0]


def extract_eis_HFR(datadir, filter_func, ignore_files=[], **est_HFR_kw):
    """
	Extract HFR from multiple EIS files and load into a DataFrame

	Args:
		datadir: data file directory
		filter_func: function to select files to load. Should return True when passed desired filenames.
			Ex: filter_func = lambda file: file.split('_')[2].replace('.DTA','')=='TimedRampDown'
		ignore_files: list of filenames to ignore (exclude from loading even if they meet filter_func conditions)
		est_HFR_kw: kwargs for estimate_HFR()
	"""
    files = [file for file in os.listdir(datadir) if filter_func(file) == True]
    # sort files by time
    files = sorted(files, key=lambda x: get_timestamp(os.path.join(datadir, x)).timestamp())
    columns = ['T_str', 'time', 'HFR', 'file']
    df = pd.DataFrame(columns=columns)

    for filename in files:
        if filename not in ignore_files:
            file = os.path.join(datadir, filename)
            temp = filename.split('_')[1]
            data = read_eis(file)
            time = get_timestamp(file).timestamp()
            HFR = estimate_HFR(data, **est_HFR_kw)
            df = df.append(pd.Series([temp, time, HFR, filename], index=columns), ignore_index=True)

    df['T'] = df['T_str'].str[:-1].astype(int)

    return df


def flag_eis_diffs(df, n=3, iqr_bound=5, cols=['Zmod', 'Zphz'], scaling='unity', show_plots=False,
                   direction=['fwd', 'rev']):
    bad_idx = []
    if scaling in ('modulus', 'mixed'):
        # get frequency spacing
        dx = mode(np.log(df['Freq']).values[1:] - np.log(df['Freq']).values[:-1]).mode[0]
        # get variance of nth derivative
        var_n = binom(2 * n, n) * df['Zmod'].values ** 2 / (dx ** (2 * n))
        # weight is inverse std
        mod_weights = 1 / var_n ** 0.5

    if type(direction) == str:
        direction = [direction]

    mod_df = df.copy()
    plot_col_idx = {}
    plot_col_diff = {}
    for col in cols:
        # do differences forward and in reverse to catch all points
        for drct in direction:  # ,'rev']:
            mod_df[f'{drct}_diff_{col}'] = np.nan
            # filter by nth derivative
            if drct == 'fwd':
                diff = np.diff(df[col], n=n) / (np.diff(np.log(df['Freq']), n=1)[n - 1:]) ** n
                if scaling == 'unity':
                    weights = np.ones_like(diff)
                elif scaling == 'modulus':
                    weights = mod_weights[n:]  # 1/(df['Zmod'].values[n:]
                elif scaling == 'mixed':
                    if col == 'Zphz':
                        weights = np.ones_like(diff)
                    else:
                        weights = mod_weights[n:]

                diff *= weights
                mod_df.loc[mod_df.index.min() + n:, f'{drct}_diff_{col}'] = diff
                plot_col_diff[(col, drct)] = diff
            else:
                diff = np.diff(df[col].values[::-1], n=n) / (np.diff(np.log(df['Freq'].values[::-1]), n=1)[n - 1:]) ** n
                if scaling == 'unity':
                    weights = np.ones_like(diff)
                elif scaling == 'modulus':
                    weights = mod_weights[::-1][n:]  # 1/(df['Zmod'].values[::-1][n:]*np.exp(9.825*n))
                elif scaling == 'mixed':
                    if col == 'Zphz':
                        weights = np.ones_like(diff)
                    else:
                        weights = mod_weights[::-1][n:]
                diff *= weights
                mod_df.loc[:mod_df.index.max() - n, f'{drct}_diff_{col}'] = diff[::-1]
                plot_col_diff[(col, drct)] = diff[::-1]
            # get indexes of data outside bounds
            col_idx = (np.where(np.abs(diff - np.median(diff)) > iqr_bound * iqr(diff))[0]).astype(
                int)  # + np.ceil(n/2)).astype(int)

            if drct == 'rev':
                col_idx = len(df) - col_idx - 1

            # print(col,drct,sorted(col_idx))
            bad_idx += list(col_idx)

            plot_col_idx[(col, drct)] = col_idx.astype(int)

    bad_idx = np.unique(bad_idx).astype(int)

    mod_df['flag'] = 0
    if len(bad_idx) > 0:
        mod_df.iloc[bad_idx, -1] = 1

    if show_plots:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].get_shared_x_axes().join(axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])
        bad_df = mod_df[mod_df['flag'] == 1]
        good_df = mod_df[mod_df['flag'] == 0]
        unit_scale = get_unit_scale(mod_df)
        # plot nth diff vs. frequency
        for col, ax in zip(cols, axes[0]):
            # obsolete
            # diff = np.diff(df[col],n=n)/np.diff(np.log(df['Freq']),n=1)[n-1:]
            # if weighting=='unity':
            # weights = np.ones_like(diff)
            # elif weighting=='modulus':
            # weights = 1/df['Zmod'].values[n:]
            # elif weighting=='mixed':
            # if col=='Zphz':
            # weights = np.ones_like(diff)
            # else:
            # weights = mod_weights[n:]
            # diff *= weights
            # # color the points that triggered the flag based on nth diff, not the points actually flagged
            # col_idx = np.where(np.abs(diff-np.median(diff)) > iqr_bound*iqr(diff))[0] + n

            for drct in direction:  # ,'rev']:
                diff = plot_col_diff[(col, drct)]
                col_idx = plot_col_idx[(col, drct)]
                if drct == 'fwd':
                    ax.scatter(df['Freq'][n:], diff, s=8, c='k', label='Fwd Diff')
                    ax.scatter(df.iloc[col_idx + n, :]['Freq'], diff[col_idx], s=20, c='k', edgecolor='r',
                               linewidth=0.8)
                    ax.axhline(np.median(diff) - iqr_bound * iqr(diff), c='r', lw=1, label='Fwd Bound')
                    ax.axhline(np.median(diff) + iqr_bound * iqr(diff), c='r', lw=1)
                # print(col,drct,iqr_bound*iqr(diff),diff[col_idx])
                elif drct == 'rev':
                    ax.scatter(df['Freq'][:-n], diff, s=8, c='gray', label='Rev Diff')
                    # print(col_idx[::-1]-2, bad_df.index)
                    # print(col,drct,np.median(diff),iqr_bound*iqr(diff),diff[col_idx-n])
                    ax.scatter(df.iloc[col_idx - n, :]['Freq'], diff[col_idx - n], s=20, c='gray', edgecolor='r',
                               linewidth=0.8)
                    ax.axhline(np.median(diff) - iqr_bound * iqr(diff), c='r', lw=1, ls='--', label='Rev Bound')
                    ax.axhline(np.median(diff) + iqr_bound * iqr(diff), c='r', lw=1, ls='--')
            ax.set_xscale('log')
            ax.set_title(col)
            ax.set_xlabel('Frequency')
            ax.set_ylabel('nth Discrete Difference')
            ax.legend(fontsize=9)

        # plot bode with flagged points in red
        plot_bode(good_df, axes=axes[1], cols=cols)
        plot_bode(bad_df, axes=axes[1], cols=cols, c='r')

        fig.tight_layout()

    return mod_df


def flag_eis_points(df, n=3, iqr_bound=5, cols=['Zmod', 'Zphz'], scaling='unity', fill_rolling_mean=False,
                    show_plots=False, direction=['fwd', 'rev'],
                    trim_method='direction', trim_offset=-2, axes=None, plot_kw={'s': 8}):  # ,trim_start_consec_pts=5):
    """
	Flag bad points in EIS data using finite differencing on Zmod and Zphz

	Args:
		df: data DataFrame
		n: order of discrete differencing. Should be 3 for best results (I think - n>3 might also work)
		iqr_bound: bound on nth discrete difference. Points that are above or below iqr_bound*iqr will be flagged
		scaling: how to weight/scale differences. Options:
			'unity': assume constant variance
			'modulus': assume variance of both Zmod and Zphz is proportional to Zmod**2
			'mixed': assume variance of Zmod is proportional to Zmod**2, variance of Zphz is constant
		fill_rolling_mean: if True, fill in values for bad frequencies by interpolating rolling average. Not recommended
		show_plots: if True, show plots illustrating data processing
	Returns:
		DataFrame with 'flag' column added. 'flag' value of 1 indicates bad data point
	"""

    bad_idx = []
    if scaling in ('modulus', 'mixed'):
        # get frequency spacing
        dx = mode(np.log(df['Freq']).values[1:] - np.log(df['Freq']).values[:-1]).mode[0]
        # get variance of nth derivative
        var_n = binom(2 * n, n) * df['Zmod'].values ** 2 / (dx ** (2 * n))
        # weight is inverse std
        mod_weights = 1 / var_n ** 0.5

    if type(direction) == str:
        direction = [direction]

    trim_options = ['direction', 'full', 'none']
    if trim_method not in trim_options:
        raise ValueError(f'Invalid trim_method {trim_method}. Options: {trim_options}')
    # number of points to trim from start and end of each range:
    # n trims all points except supposed bad point. Very conservative, may leave actual bad points in
    # n-1 leaves an extra point on each side of supposed bad point
    # n-2 leaves 2 extra points on each side of supposed bad point
    trim_len = n + trim_offset

    plot_col_idx = {}
    plot_col_diff = {}
    for drct in direction:
        # do differences forward and in reverse to catch all points
        drct_idx = []
        for col in cols:
            # filter by nth derivative
            if drct == 'fwd':
                diff = np.diff(df[col], n=n) / (np.diff(np.log(df['Freq']), n=1)[n - 1:]) ** n
                if scaling == 'unity':
                    weights = np.ones_like(diff)
                elif scaling == 'modulus':
                    weights = mod_weights[n:]  # 1/(df['Zmod'].values[n:]
                elif scaling == 'mixed':
                    if col == 'Zphz':
                        weights = np.ones_like(diff)
                    else:
                        weights = mod_weights[n:]

                diff *= weights
                plot_col_diff[(col, drct)] = diff
            else:
                diff = np.diff(df[col].values[::-1], n=n) / (np.diff(np.log(df['Freq'].values[::-1]), n=1)[n - 1:]) ** n
                if scaling == 'unity':
                    weights = np.ones_like(diff)
                elif scaling == 'modulus':
                    weights = mod_weights[::-1][n:]  # 1/(df['Zmod'].values[::-1][n:]*np.exp(9.825*n))
                elif scaling == 'mixed':
                    if col == 'Zphz':
                        weights = np.ones_like(diff)
                    else:
                        weights = mod_weights[::-1][n:]
                diff *= weights
                plot_col_diff[(col, drct)] = diff[::-1]
            # get indexes of data outside bounds
            # fluctuation in diff shows up 1 point after the errant point (subtract 1). Diff starts at nth point (add n). The errant point is at the diff index plus n minus 1
            col_idx = (np.where(np.abs(diff - np.median(diff)) > iqr_bound * iqr(diff))[0]).astype(
                int)  # + n-1).astype(int)
            # for plotting, track the actual points that triggered the flags, not the flagged points. Align diff index with function index
            plot_idx = col_idx  # - (n-1)
            # print('Pre-condense:',col,drct,col_idx)
            # a single bad point cascades to n subsequent points in the diff. Condense the ranges accordingly
            # Still flag one point on each side of the point thought to be "bad" as it may be unclear which point is actually bad
            # (i.e., a transition from a bad point to a good point may make it look like the good point is actually the errant one)
            # if len(col_idx) > 0:
            # rng_end_idx = np.where(np.diff(col_idx)!=1)[0]
            # rng_start_idx = np.insert((rng_end_idx + 1),0,0)
            # rng_end_idx = np.append(rng_end_idx,len(col_idx)-1)
            # trimmed_ranges = [np.arange(col_idx[start],max(col_idx[start]+1,col_idx[end] - (n-2))) for start,end in zip(rng_start_idx,rng_end_idx)]
            # col_idx = np.concatenate([r for r in trimmed_ranges])
            # print('Post-condense:',col,drct,col_idx)

            # check last point - won't be flagged above due to centering
            # if np.abs(diff[-1]-np.median(diff)) > iqr_bound*iqr(diff):
            # col_idx = np.insert(col_idx, len(col_idx), n + len(diff) - 1)
            if drct == 'rev':
                col_idx = len(df) - col_idx - 1 + (-1)  # +((n-1)-1)
                plot_idx = len(df) - plot_idx - 1

            # print(col,drct,sorted(col_idx))
            # concatenate all the flags determined in the same direction
            drct_idx += list(col_idx)

            plot_col_idx[(col, drct)] = plot_idx.astype(int)
        drct_idx = np.unique(drct_idx)

        if trim_method == 'direction':
            if len(drct_idx) > 0:
                print('Pre-trim:', drct_idx)
                # a single bad point cascades to n points in the diff. Trim the ranges accordingly
                # do this after aggregating all flags in one direction to ensure that contiguous ranges of bad points are not lost by prematurely condensing ranges
                rng_end_idx = np.where(np.diff(drct_idx) != 1)[0]
                rng_start_idx = np.insert((rng_end_idx + 1), 0, 0)
                rng_end_idx = np.append(rng_end_idx, len(drct_idx) - 1)
                # trim logic: trim the end unless it truncates the range to length zero.
                trimmed_ranges = [np.arange(drct_idx[start],
                                            max(drct_idx[start] + 1, drct_idx[end] - (trim_len - 1))
                                            )
                                  for start, end in zip(rng_start_idx, rng_end_idx)]
                # the very last points in the spectra should not be trimmed
                if drct_idx[-1] == len(df) - 1:
                    start = rng_start_idx[-1]
                    end = rng_end_idx[-1]
                    # reset the range end to the last point
                    trimmed_ranges[-1] = np.arange(drct_idx[start], drct_idx[end])
                drct_idx = np.concatenate([r for r in trimmed_ranges])
                print('Post-trim:', drct_idx)
        bad_idx += list(drct_idx)

    bad_idx = np.unique(bad_idx).astype(int)

    if trim_method == 'full':
        if len(bad_idx) > 0:
            print('Pre-trim:', bad_idx)
            # a single bad point cascades to n subsequent points in the diff. Condense (trim) the ranges accordingly
            # do this after aggregating all flags to ensure that contiguous ranges of bad points are not lost by prematurely condensing ranges
            rng_end_idx = np.where(np.diff(bad_idx) != 1)[0]
            rng_start_idx = np.insert((rng_end_idx + 1), 0, 0)
            rng_end_idx = np.append(rng_end_idx, len(bad_idx) - 1)

            # trim logic: use the trim unless it truncates the range to length zero.
            # min(bad_idx[start] + trim, bad_idx[end]-trim): start at the smaller of start + trim and end - trim
            # max( bad_idx[start], <above>): if end-trim < start, just start at start (i.e. don't extend the range just because it's short)
            # same logic applies to end of range
            trimmed_ranges = [np.arange(max(bad_idx[start], min(bad_idx[start] + trim_len, bad_idx[end] - trim_len)),
                                        min(bad_idx[end] + 1,
                                            max(bad_idx[start] + trim_len + 1, bad_idx[end] - (trim_len - 1)))
                                        )
                              for start, end in zip(rng_start_idx, rng_end_idx)]

            # the very first and very last points in the spectra should not be trimmed
            if bad_idx[0] == 0:
                start = rng_start_idx[0]
                end = rng_end_idx[0]
                # reset the range start to point 0
                trimmed_ranges[0] = np.arange(bad_idx[start], min(bad_idx[end] + 1, max(bad_idx[start] + trim_len + 1,
                                                                                        bad_idx[end] - (trim_len - 1))))
            if bad_idx[-1] == len(df) - 1:
                start = rng_start_idx[-1]
                end = rng_end_idx[-1]
                # reset the range end to the last point
                trimmed_ranges[-1] = np.arange(
                    max(bad_idx[start], min(bad_idx[start] + trim_len, bad_idx[end] - trim_len)),
                    bad_idx[rng_end_idx[-1]])
            bad_idx = np.concatenate([r for r in trimmed_ranges])
            print('Post-trim:', bad_idx)

    # if len(bad_idx) >= trim_start_consec_pts:
    # # if the first trim_start_consec_pts points are all flagged, also flag the first n-1 points
    # if np.sum(bad_idx[:trim_start_consec_pts] == np.arange(n-1, n-1+trim_start_consec_pts))==trim_start_consec_pts:
    # bad_idx = np.concatenate((np.arange(0,n-1),bad_idx))

    # print(bad_idx)

    mod_df = df.copy()
    mod_df['flag'] = 0
    if len(bad_idx) > 0:
        mod_df.iloc[bad_idx, -1] = 1

    if fill_rolling_mean:
        # get rolling mean
        ma = df.rolling(5, center=True).mean()
        mod_df['Zmod_filled'] = mod_df['Zmod']
        mod_df['Zphz_filled'] = mod_df['Zphz']
        bad_df = mod_df[mod_df['flag'] == 1]
        for col in cols:
            # interpolate rolling mean to fill bad data points
            mod_df.loc[bad_idx, col + '_filled'] = bad_df.apply(
                lambda r: np.interp(r['Freq'], ma['Freq'][::-1], ma[col][::-1]), axis=1)
        mod_df['Zreal_filled'] = mod_df['Zmod_filled'] * np.cos(2 * np.pi * mod_df['Zphz_filled'] / 360)
        mod_df['Zimag_filled'] = mod_df['Zmod_filled'] * np.sin(2 * np.pi * mod_df['Zphz_filled'] / 360)

    if show_plots:
        if axes is None:
            fig, axes = plt.subplots(3, 2, figsize=(10, 10))
        else:
            fig = axes.ravel()[0].get_figure()
        axes[0, 0].get_shared_x_axes().join(axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])
        bad_df = mod_df[mod_df['flag'] == 1]
        good_df = mod_df[mod_df['flag'] == 0]
        unit_scale = get_unit_scale(mod_df)
        # plot nth diff vs. frequency
        for col, ax in zip(cols, axes[0]):
            # obsolete
            # diff = np.diff(df[col],n=n)/np.diff(np.log(df['Freq']),n=1)[n-1:]
            # if weighting=='unity':
            # weights = np.ones_like(diff)
            # elif weighting=='modulus':
            # weights = 1/df['Zmod'].values[n:]
            # elif weighting=='mixed':
            # if col=='Zphz':
            # weights = np.ones_like(diff)
            # else:
            # weights = mod_weights[n:]
            # diff *= weights
            # # color the points that triggered the flag based on nth diff, not the points actually flagged
            # col_idx = np.where(np.abs(diff-np.median(diff)) > iqr_bound*iqr(diff))[0] + n

            for drct in direction:  # ,'rev']:
                diff = plot_col_diff[(col, drct)]
                col_idx = plot_col_idx[(col, drct)]
                if drct == 'fwd':
                    ax.scatter(df['Freq'][n:], diff, c='k', label='Fwd Diff', **plot_kw)
                    ax.scatter(df.iloc[col_idx + n, :]['Freq'], diff[col_idx], c='k', edgecolor='r', linewidth=0.8,
                               **plot_kw)
                    ax.axhline(np.median(diff) - iqr_bound * iqr(diff), c='r', lw=1, label='Fwd Bound')
                    ax.axhline(np.median(diff) + iqr_bound * iqr(diff), c='r', lw=1)
                # print(col,drct,iqr_bound*iqr(diff),diff[col_idx])
                elif drct == 'rev':
                    ax.scatter(df['Freq'][:-n], diff, c='gray', label='Rev Diff', **plot_kw)
                    # print(col_idx[::-1]-2, bad_df.index)
                    # print(col,drct,np.median(diff),iqr_bound*iqr(diff),diff[col_idx-n])
                    ax.scatter(df.iloc[col_idx - n, :]['Freq'], diff[col_idx - n], c='gray', edgecolor='r',
                               linewidth=0.8, **plot_kw)
                    ax.axhline(np.median(diff) - iqr_bound * iqr(diff), c='r', lw=1, ls='--', label='Rev Bound')
                    ax.axhline(np.median(diff) + iqr_bound * iqr(diff), c='r', lw=1, ls='--')
            ax.set_xscale('log')
            ax.set_title(col)
            ax.set_xlabel('Frequency')
            ax.set_ylabel('nth Discrete Difference')
            ax.legend(fontsize=9)

        # plot bode with flagged points in red
        plot_bode(good_df, axes=axes[1], cols=cols, **plot_kw)
        plot_bode(bad_df, axes=axes[1], cols=cols, c='r', **plot_kw)
        if fill_rolling_mean:
            # plot interpolated points
            fdf = mod_df.copy()
            for col in ['Zreal', 'Zimag', 'Zmod', 'Zphz']:
                fdf[col] = fdf[col + '_filled']
            fdf = fdf.loc[bad_idx, :]
            plot_bode(fdf, axes=axes[1], c='g')
        # plot nyquist with flagged points in red
        plot_nyquist(good_df, ax=axes[2, 0], unit_scale=unit_scale, **plot_kw)
        plot_nyquist(bad_df, ax=axes[2, 0], c='r', unit_scale=unit_scale, **plot_kw)
        if fill_rolling_mean:
            plot_nyquist(fdf, ax=axes[2, 0], c='g', unit_scale=unit_scale, **plot_kw)
        axes[2, 0].set_title('Nyquist')

        axes[2, 1].axis('off')
        fig.tight_layout()

    return mod_df


def calc_sigma(R, d, t, units='mm'):
    """
	Calculate conductivity in S/cm given resistance and cell dimensions
	Assumes button cell geometry

	Args:
		R: resistance (ohm)
		d: diameter
		t: thickness
		units: units for d and t. Default mm. Options: 'mm', 'cm'
	"""
    # convert to cm
    if units == 'mm':
        d, t = d / 10, t / 10
    elif units != 'cm':
        raise ValueError(f'Units arg {units} not recognized. Valid units are ''mm'',''cm''')
    a = np.pi * (d / 2) ** 2
    sigma = t / (R * a)
    return sigma


def aggregate_prop(df, by, property_col, aggregate):
    grp_df = df.groupby(by)
    if aggregate == 'end':
        prop = np.array([gdf.loc[gdf['time'].idxmax(), property_col] for name, gdf in grp_df])
    elif aggregate == 'start':
        prop = np.array([gdf.loc[gdf['time'].idxmin(), property_col] for name, gdf in grp_df])
    else:
        prop = getattr(grp_df[property_col], aggregate)().values
    return prop


def calc_G_act(df, property_col, aggregate, return_fit=False):
    """
	Calculate activation energy (in eV) from EIS data

	Args:
		df: DataFrame with property by temperature (in C)
		aggregate: function to use to aggregate multiple values for same temperature
		property_col: column name of property for which to calculate activation energy
		return_fit: if True, return fit coefficients in addition to G_act
	Returns:
		G_act: activation energy in eV
		fit: fit coefficients for log(prop) vs. 1/T (if return_fit==True)
	"""
    prop_agg = aggregate_prop(df, 'T', property_col, aggregate)
    temps = np.unique(df['T'])

    # grp_df = df.groupby('T')
    # temps = np.array(list(grp_df.groups.keys())) #np.zeros(len(grp_df))
    # sigma_agg = np.zeros(len(grp_df))

    # for i, (T, df) in enumerate(grp_df):
    # if aggregate in ['start','end']:
    # if aggregate=='start':
    # agg_time = df['time'].min()
    # else:
    # agg_time = df['time'].max()
    # sigma_agg[i] = float(df[df['time']==agg_time][property_col])
    # else:
    # sigma_agg[i] = getattr(df[property_col],aggregate)()

    T_inv = 1 / (273 + temps)
    # print(sigma_agg)
    fit = np.polyfit(T_inv, np.log(prop_agg), deg=1)
    k_B = 8.617e-5  # eV/K

    if return_fit == True:
        return -k_B * fit[0], fit
    else:
        return -k_B * fit[0]


def jv_multiplot(files, color_by='T', aggregate=False, file_sequence=['file_type', 'T', 'aflow', 'cflow'], area=None,
                 ax=None, label_color=None):
    """
	Plot multiple j-V curves on same axes
	Args:
		files: files to plot
		color_by: fieldname to determine series colors
		aggregate: how to aggregate multiple files with same value of color_by field. Options:
			False: plot all files, ordered by time
			'max_pwr': plot file with max power
	"""
    if ax is None:
        fig, ax = plt.subplots()

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['.', '^', 'v', 's', 'D', '*']

    # temps = [int(os.path.basename(file).split('_')[1].replace('C','')) for file in files]
    times = [get_timestamp(f) for f in files]
    infos = [get_file_info(file, file_sequence) for file in files]

    jvdf = pd.DataFrame(np.array([files, times]).T, columns=['file', 'time'])
    jvdf = jvdf.join(pd.DataFrame(infos))

    if label_color is None:
        label_color = dict(zip(jvdf[color_by].unique(), default_colors))
    if color_by == 'T':
        label_units = '$^\circ$C'
    else:
        label_units = ''

    for label, gdf in jvdf.groupby(color_by):
        if len(gdf) > 1:
            if aggregate == False:
                gdf = gdf.sort_values('time')
                gdf.index = np.arange(len(gdf))
                for i, row in gdf.iterrows():
                    df = read_jv(row['file'])
                    plot_jv(df, area=area, plot_pwr=True, ax=ax, label='{}{} ({})'.format(label, label_units, i + 1),
                            marker=markers[i], markersize=5, c=label_color[label],
                            pwr_kw=dict(ls=':', marker=markers[i], markersize=3, c=label_color[label]))
            else:
                if aggregate == 'max_pwr':
                    idx = gdf['file'].map(lambda x: read_jv(x)['Pwr'].abs().max()).idxmax()
                elif aggregate == 'min_pwr':
                    idx = gdf['file'].map(lambda x: read_jv(x)['Pwr'].abs().max()).idxmin()
                elif aggregate == 'first':
                    idx = gdf['file'].map(lambda x: get_timestamp(x)).idxmin()
                elif aggregate == 'last':
                    idx = gdf['file'].map(lambda x: get_timestamp(x)).idxmax()
                else:
                    raise ValueError(f'Invalid aggregate method {aggregate} specified')

                df = read_jv(gdf.loc[idx, 'file'])
                plot_jv(df, area=area, plot_pwr=True, ax=ax, label=f'{label}{label_units}', marker=markers[0],
                        markersize=5, c=label_color[label],
                        pwr_kw=dict(ls=':', marker=markers[0], markersize=5, c=label_color[label]))

        else:
            df = read_jv(gdf['file'].min())
            plot_jv(df, area=area, plot_pwr=True, ax=ax, label=f'{label}{label_units}', marker=markers[0], markersize=5,
                    c=label_color[label],
                    pwr_kw=dict(ls=':', marker=markers[0], markersize=5, c=label_color[label]))


def generate_plots(plot_types, datadir=None, savefigs=False, savedir='./plots', area=None,
                   ocv_kw={}, nyquist_kw={}, bode_kw={}, jv_kw={}):
    """
	Generate basic plots for files in directory

	Args:
		plot_types: list of plot types to generate. Options: 'nyquist','bode','ocv','jv'
		datadir: data directory
		savefigs: if True, save generated figures
		area: cell area (cm2)
		plot_kw:
	"""
    # allowed_plot_types = ['nyquist','bode','ocv','jv']#,'nyquist-bode'

    if datadir is None:
        datadir = os.getcwd()

    plotdir = os.path.join(datadir, savedir)

    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    # get cell info from datadir
    cell = get_cell_name(datadir)

    # set kw defaults and update with any user-specified params
    # so that user doesn't have to re-enter all defaults to change one thing
    jv_default = {'plot_pwr': True, 'marker': '.', 'pwr_kw': {'ls': ':', 'marker': '.'}}
    jv_default.update(jv_kw)
    jv_kw = jv_default

    ocv_default = {'filter_func': lambda x: x[0:3] in ('OCV', 'EIS') and x[-3:] == 'DTA'}
    ocv_default.update(ocv_kw)
    ocv_kw = ocv_default

    for plot_type in plot_types:
        if plot_type in ['nyquist', 'bode']:
            start_str = 'EIS'
        elif plot_type == 'jv':
            start_str = 'PWRPOLARIZATION'

        if plot_type == 'ocv':
            plot_ocv(datadir, **ocv_kw)
            plt.title(cell, wrap=True)
            if savefigs is True:
                fig = plt.gcf()
                fig.savefig(os.path.join(plotdir, 'OCV_plot.png'), dpi=500)
        else:
            files = glob.glob(os.path.join(datadir, start_str + '*.DTA'))

            for file in files:
                info = get_file_info(file)
                if plot_type == 'nyquist':
                    df = read_eis(file)
                    ax = plot_nyquist(df, area=area, **nyquist_kw)
                    ax.text(0.97, 0.05, '{}$^\circ$C, {}, {}'.format(info['T'], info['aflow'], info['cflow']),
                            transform=ax.transAxes, ha='right')
                elif plot_type == 'bode':
                    df = read_eis(file)
                    axes = plot_bode(df, area=area, **bode_kw)
                    ax = axes[0]
                    ax.text(0.03, 0.9, '{}$^\circ$C, {}, {}'.format(info['T'], info['aflow'], info['cflow']),
                            transform=ax.transAxes)
                elif plot_type == 'jv':
                    df = read_jv(file)
                    ax = plot_jv(df, area=area, **jv_kw)
                    ax.text(0.5, 0.05, '{}$^\circ$C, {}, {}'.format(info['T'], info['aflow'], info['cflow']),
                            transform=ax.transAxes, ha='center')

                ax.set_title(cell, wrap=True)

                if savefigs is True:
                    fig = plt.gcf()
                    fname = os.path.basename(file)
                    fig.savefig(os.path.join(plotdir, fname.replace('DTA', 'png')), dpi=500)


def plot_eis_prop(eis_df, property_col, label='', aggregate=['start', 'end', 'max', 'mean'], ax=None, **plt_kw):
    """
	Plot EIS-derived property as a function of temperature

	Args:
	  eis_df: DataFrame with time, temperature, and property to plot
	  property_col: column name of property to plot
	  label: label for legend. Aggregate name will be affixed to end of provided label
	  aggregate: list of aggregate functions to use to aggregate multiple property values for each temperature.
		  Options: start, end, or any built-in pandas aggregate function
	  ax: axis on which to plot
	  plt_kw: plot keyword args
	"""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    grp_df = eis_df.groupby('T')
    temps = np.array(list(grp_df.groups.keys()))
    prop_agg = {}
    for agg in aggregate:
        prop_agg[agg] = aggregate_prop(eis_df, 'T', property_col, agg)  # np.zeros(len(grp_df))

    # for i, (T, df) in enumerate(grp_df):
    # for agg in aggregate:
    # if agg in ['start','end']:
    # if agg=='start':
    # agg_time = df['time'].min()
    # else:
    # agg_time = df['time'].max()
    # sigma_agg[agg][i] = float(df[df['time']==agg_time][property_col])
    # else:
    # sigma_agg[agg][i] = getattr(df[property_col],agg)()

    T_inv = 1000 / (273 + temps)
    for agg, prop in prop_agg.items():
        if label in ('', None):
            lab_prefix = ''
        else:
            lab_prefix = label + ' '
        ax.semilogy(T_inv, prop, label=lab_prefix + agg, **plt_kw)
    if label is not None:
        ax.legend()

    # label untransformed T on top axis
    if ax is not None:
        # get twin ax if already exists
        for other_ax in ax.figure.axes:
            if other_ax is ax:
                ax2 = None
            elif other_ax.bbox.bounds == ax.bbox.bounds:
                ax2 = other_ax
                break
            else:
                ax2 = None
    if ax2 is None:
        # create twin ax
        ax2 = ax.twiny()
    # set to same scale
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(T_inv)
    ax2.set_xticklabels(temps.astype(int))

    ax.set_xlabel(r'$1000/T \ (\mathrm{K}^{-1})$')
    ax2.set_xlabel(r'$T \ (^\circ \mathrm{C})$')
    # ax.set_ylabel(r'$\sigma$ (S/cm)')

    fig.tight_layout()

    return ax


def plot_arrhenius_fit(df, property_col, aggregate, ax=None, **plt_kw):
    G_act, fit = calc_G_act(df, property_col, aggregate, return_fit=True)
    if ax is None:
        fig, ax = plt.subplots()

    temps = np.unique(df['T'])
    T_inv = 1 / (temps + 273)
    y_fit = np.exp(np.polyval(fit, T_inv))
    ax.plot(1000 * T_inv, y_fit, **plt_kw)

    if plt_kw.get('label', '') != '':
        ax.legend()
    return ax


def get_file_info(file, sequence=['file_type', 'T', 'aflow', 'cflow']):
    """
	Get information from filename

	Args:
		file: filename (basename or full path)
		sequence: list of identifiers in the order that they appear in the filename (separated by _)
	"""
    fname = os.path.basename(file).replace('.DTA', '')
    info = dict(zip(sequence, fname.split('_')))

    info['T'] = int(info['T'][:info['T'].find('C')])
    for flow in ('aflow', 'cflow'):
        try:
            if info[flow].find('sccm') > 0:
                rate, gas = info[flow].split('sccm')
                gas = ' '.join(camel_case_split(gas))
                info[flow] = ' '.join([rate, 'SCCM', gas])
            else:
                info[flow] = ' '.join(camel_case_split(info[flow]))
        except KeyError:
            pass

    return info


def read_nleis_data(file):
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()
    source = get_file_source(file)

    if source == 'gamry':
        # get number of points measured
        num_freq_start = txt.find('NUMFREQ')
        num_freq_end = txt.find('\n', num_freq_start + 1)
        num_freq_line = txt[num_freq_start:num_freq_end]
        num_freq = int(num_freq_line.split('\t')[2])

        frequency_data = {}

        for n in range(num_freq):
            fra_start = txt.find(f'FREQUENCY {n}')
            if n == num_freq - 1:
                fra_end = txt.find('ZCURVE')
            else:
                fra_end = txt.find('FREQUENCY {}'.format(n + 1))
            fra_txt = txt[fra_start:fra_end]

            # get frequency
            freq_line = fra_txt[:fra_txt.find('\n')]
            requested_freq = float(freq_line.split('\t')[1].replace('Requested Freq (Hz):', '').strip())
            actual_freq = float(freq_line.split('\t')[2].replace('Actual Freq (Hz):', '').strip())

            # get header
            header_start = fra_txt.find('\n', fra_txt.find('\n') + 1) + 1
            header_end = fra_txt.find('\n', header_start)
            header = fra_txt[header_start:header_end].split('\t')
            # if table is indented, ignore empty left column
            if header[0] == '':
                usecols = header[1:]
            else:
                usecols = header

            fra_table = fra_txt[fra_txt.find('\n', header_end + 1) + 1:]
            fra_data = pd.read_csv(StringIO(fra_table), sep='\t', header=None, names=header, usecols=usecols)

            frequency_data[n] = {'requested_freq': requested_freq, 'actual_freq': actual_freq, 'data': fra_data}

        return frequency_data


def get_cell_name(datadir):
    datadir = os.path.abspath(datadir)

    if os.path.basename(os.path.split(datadir)[0]) == 'Win10 Gamry data':
        # datadir is one level below Gamry data
        celldir = os.path.basename(datadir)
    else:
        # datadir is two levels below Gamry data - need info from both top dir and subdir
        celldir = os.path.join(os.path.basename(os.path.split(datadir)[0]), os.path.basename(datadir))

    dirsplit = [txt.replace('-', ' ') for txt in re.split('_|/|\\\\', celldir.strip('./'))]
    cell = dirsplit[0] + ' ' + ' | '.join(dirsplit[1:])

    return cell