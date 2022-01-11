import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import file_load as fl
from .utils import get_unit_scale, get_scale_factor, get_factor_from_unit


# ---------------------------
# Functions for plotting data
# ---------------------------
def plot_ocv(datadir, filter_func=None, files=None, ax=None, invert='auto', same_color=True, **plt_kw):
    # get files
    if filter_func is None and files is None:
        # if no filter or files specified, get all OCV files
        filter_func = lambda x: x[0:3] in ('OCV', 'OCP') and x[-3:] == 'DTA'
        files = [f for f in os.listdir(datadir) if filter_func(f)]
    elif files and not filter_func:
        if type(files) == str:
            # if single file specified, convert to 1-element list
            files = [files]
    elif filter_func and not files:
        files = [f for f in os.listdir(datadir) if filter_func(f)]
    elif filter_func and files:
        raise ValueError('Both filter_func and files have been specified. Please specify only one')

    dfs = [fl.read_ocv(os.path.join(datadir, file)) for file in files]
    dfs = [df for df in dfs if len(df) > 0]
    start_times = [df['timestamp'][0] for df in dfs]
    start_time = min(start_times)

    ts_func = lambda ts: (ts - start_time).dt.total_seconds() / 3600

    if ax is None:
        fig, ax = plt.subplots()

    if invert == 'auto':
        # choose sign based on max voltage
        tdf = pd.concat(dfs, ignore_index=True)
        V_sign = np.sign(tdf.loc[tdf['Vf'].abs().idxmax(), 'Vf'])
    elif invert == True:
        V_sign = -1
    elif invert == False:
        V_sign = 1

    for df in dfs:
        if 'c' not in plt_kw and 'color' not in plt_kw and same_color == True:
            # if no color specified and same color desired, set color to first default color
            default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            plt_kw['c'] = default_colors[0]

        ax.plot(ts_func(df['timestamp']), V_sign * df['Vf'], **plt_kw)

    ax.set_xlabel('Time / h')
    ax.set_ylabel('OCV / V')


def plot_jv(df, area=None, plot_pwr=True, ax=None, pwr_kw={'marker': 'o', 'mfc': 'white'}, marker='o', **plt_kw):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if area is not None:
        # if area given, convert to densities
        df = df.copy()
        df['Im'] /= area
        df['Pwr'] /= area

    ax.plot(1000 * df['Im'].abs(), df['Vf'].abs(), marker=marker, **plt_kw)
    if area is None:
        ax.set_xlabel('$j$ / mA')
    else:
        ax.set_xlabel('$j$ / mA$\cdot$cm$^{-2}$')
    ax.set_ylabel('$V$ / V')
    if 'label' in plt_kw.keys():
        ax.legend()

    if plot_pwr is True:
        # plot power on same axes

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
            ax2 = ax.twinx()

        ax2.plot(1000 * df['Im'].abs(), 1000 * df['Pwr'].abs().values, **pwr_kw)

        if area is None:
            ax2.set_ylabel('$P$ / mW')
        else:
            ax2.set_ylabel('$P$ / mW$\cdot$cm$^{-2}$')

        ax2.set_ylim(0, max(np.max(1000 * df['Pwr'].abs().values) * 1.1, ax2.get_ylim()[1]))

    fig.tight_layout()

    ax.set_xlim(0, max(1.1 * np.max(1000 * df['Im'].abs()), ax.get_xlim()[1]))
    ax.set_ylim(0, max(1.1 * np.max(df['Vf'].abs()), ax.get_ylim()[1]))

    return ax


def plot_nyquist(df, area=None, ax=None, label='', plot_func='scatter', unit_scale='auto', set_aspect_ratio=True,
                 **kw):
    """
	Generate Nyquist plot.

	Parameters
	----------
    df : pandas DataFrame
        DataFrame of impedance data
    area : float, optional (default: None)
        Active area in cm^2. If provided, plot area-normalized impedance
    ax : matplotlib axis, optional (default: None)
        Axis on which to plot. If None, axis will be created.
    label : str, optional (default: '')
        Label for data
    plot_func : str, optional (default: 'scatter')
        Name of matplotlib.pyplot plotting function to use. Options: 'scatter', 'plot'
    unit_scale: str, optional (default: 'auto')
        Scaling unit prefix. If 'auto', determine from data.
        Options are 'mu', 'm', '', 'k', 'M', 'G'
    set_aspect_ratio : bool, optional (default: True)
        If True, ensure that visual scale of x and y axes is the same.
        If False, use matplotlib's default scaling.
    kw:
        Keywords to pass to matplotlib.pyplot.plot_func
	"""
    df = df.copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 2.75))

    if area is not None:
        # if area given, convert to ASR
        df['Zreal'] *= area
        df['Zimag'] *= area

    # get/set unit scale
    unit_map = {-2: '$\mu$', -1: 'm', 0: '', 1: 'k', 2: 'M', 3: 'G'}
    if unit_scale == 'auto':
        unit_scale = get_unit_scale(df, area)
        Z_ord = [k for k, v in unit_map.items() if v == unit_scale][0]
    elif unit_scale is None:
        unit_scale = ''
        Z_ord = 0
    else:
        Z_ord = [k for k, v in unit_map.items() if v == unit_scale][0]

    # scale data
    df['Zreal'] /= 10 ** (Z_ord * 3)
    df['Zimag'] /= 10 ** (Z_ord * 3)

    if plot_func == 'scatter':
        scatter_defaults = {'s': 10, 'alpha': 0.5}
        scatter_defaults.update(kw)
        ax.scatter(df['Zreal'], -df['Zimag'], label=label, **scatter_defaults)
    elif plot_func == 'plot':
        ax.plot(df['Zreal'], -df['Zimag'], label=label, **kw)
    else:
        raise ValueError(f'Invalid plot type {plot_func}. Options are scatter, plot')

    if area is not None:
        ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega\cdot \mathrm{{cm}}^2$')
        ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega\cdot \mathrm{{cm}}^2$')
    else:
        ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$')
        ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$')

    if label != '':
        ax.legend()

    if set_aspect_ratio:
        # make scale of x and y axes the same
        fig = ax.get_figure()

        # if data extends beyond axis limits, adjust to capture all data
        ydata_range = df['Zimag'].max() - df['Zimag'].min()
        xdata_range = df['Zreal'].max() - df['Zreal'].min()
        if np.min(-df['Zimag']) < ax.get_ylim()[0]:
            if np.min(-df['Zimag']) >= 0:
                # if data doesn't go negative, don't let y-axis go negative
                ymin = max(0, np.min(-df['Zimag']) - ydata_range * 0.1)
            else:
                ymin = np.min(-df['Zimag']) - ydata_range * 0.1
        else:
            ymin = ax.get_ylim()[0]
        if np.max(-df['Zimag']) > ax.get_ylim()[1]:
            ymax = np.max(-df['Zimag']) + ydata_range * 0.1
        else:
            ymax = ax.get_ylim()[1]
        ax.set_ylim(ymin, ymax)

        if df['Zreal'].min() < ax.get_xlim()[0]:
            if df['Zreal'].min() >= 0:
                # if data doesn't go negative, don't let x-axis go negative
                xmin = max(0, df['Zreal'].min() - xdata_range * 0.1)
            else:
                xmin = df['Zreal'].min() - xdata_range * 0.1
        else:
            xmin = ax.get_xlim()[0]
        if df['Zreal'].max() > ax.get_xlim()[1]:
            xmax = df['Zreal'].max() + xdata_range * 0.1
        else:
            xmax = ax.get_xlim()[1]
        ax.set_xlim(xmin, xmax)

        # get data range
        yrng = ax.get_ylim()[1] - ax.get_ylim()[0]
        xrng = ax.get_xlim()[1] - ax.get_xlim()[0]

        # get axis dimensions
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height

        yscale = yrng / height
        xscale = xrng / width

        if yscale > xscale:
            # expand the x axis
            diff = (yscale - xscale) * width
            xmin = max(0, ax.get_xlim()[0] - diff / 2)
            mindelta = ax.get_xlim()[0] - xmin
            xmax = ax.get_xlim()[1] + diff - mindelta

            ax.set_xlim(xmin, xmax)
        elif xscale > yscale:
            # expand the y axis
            diff = (xscale - yscale) * height
            if min(np.min(-df['Zimag']), ax.get_ylim()[0]) >= 0:
                # if -Zimag doesn't go negative, don't go negative on y-axis
                ymin = max(0, ax.get_ylim()[0] - diff / 2)
                mindelta = ax.get_ylim()[0] - ymin
                ymax = ax.get_ylim()[1] + diff - mindelta
            else:
                negrng = abs(ax.get_ylim()[0])
                posrng = abs(ax.get_ylim()[1])
                negoffset = negrng * diff / (negrng + posrng)
                posoffset = posrng * diff / (negrng + posrng)
                ymin = ax.get_ylim()[0] - negoffset
                ymax = ax.get_ylim()[1] + posoffset

            ax.set_ylim(ymin, ymax)

    return ax


def plot_bode(df, area=None, axes=None, label='', plot_func='scatter', cols=['Zmod', 'Zphz'], unit_scale='auto',
              invert_phase=True, invert_Zimag=True, **kw):
    """
    Generate Bode plots.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame of impedance data
    area : float, optional (default: None)
        Active area in cm^2. If provided, plot area-normalized impedance.
    axes : array, optional (default: None)
        List or array of axes on which to plot. If None, axes will be created.
    label : str, optional (default: '')
        Label for data
    plot_func : str, optional (default: 'scatter')
        Name of matplotlib.pyplot plotting function to use. Options: 'scatter', 'plot'
    cols : list, optional (default: ['Zmod', 'Zphz'])
        List of data columns to plot. Options: 'Zreal', 'Zimag', 'Zmod', 'Zphz'
    unit_scale: str, optional (default: 'auto')
        Scaling unit prefix. If 'auto', determine from data.
        Options are 'mu', 'm', '', 'k', 'M', 'G'
    invert_phase : bool, optional (default: True)
        If True, plot negative phase
    invert_Zimag : bool, optional (default: True)
        If True, plot negative Zimag
    kw:
        Keywords to pass to matplotlib.pyplot.plot_func
    """
    df = df.copy()
    # formatting for columns
    col_dict = {'Zmod': {'units': '$\Omega$', 'label': '$|Z|$', 'scale': 'log'},
                'Zphz': {'units': '$^\circ$', 'label': r'$\theta$', 'scale': 'linear'},
                'Zreal': {'units': '$\Omega$', 'label': '$Z^\prime$', 'scale': 'linear'},
                'Zimag': {'units': '$\Omega$', 'label': '$Z^{\prime\prime}$', 'scale': 'linear'}
                }

    if type(axes) not in [list, np.ndarray, tuple] and axes is not None:
        axes = [axes]

    if axes is None:
        fig, axes = plt.subplots(1, len(cols), figsize=(3 * len(cols), 2.75))
    else:
        fig = axes[0].get_figure()
    # ax1,ax2 = axes

    if area is not None:
        for col in ['Zreal', 'Zimag', 'Zmod']:
            if col in df.columns:
                df[col] *= area

    # get/set unit scale
    unit_map = {-2: '$\mu$', -1: 'm', 0: '', 1: 'k', 2: 'M', 3: 'G'}
    if unit_scale == 'auto':
        unit_scale = get_unit_scale(df, area)
        Z_ord = [k for k, v in unit_map.items() if v == unit_scale][0]
    elif unit_scale is None:
        unit_scale = ''
        Z_ord = 0
    else:
        Z_ord = [k for k, v in unit_map.items() if v == unit_scale][0]

    # scale data
    for col in ['Zreal', 'Zimag', 'Zmod']:
        if col in df.columns:
            df[col] /= 10 ** (Z_ord * 3)

    if invert_Zimag:
        df['Zimag'] *= -1

    if invert_phase:
        df['Zphz'] *= -1

    if plot_func == 'scatter':
        scatter_defaults = {'s': 10, 'alpha': 0.5}
        scatter_defaults.update(kw)
        for col, ax in zip(cols, axes):
            ax.scatter(df['Freq'], df[col], label=label, **scatter_defaults)
    elif plot_func == 'plot':
        for col, ax in zip(cols, axes):
            ax.plot(df['Freq'], df[col], label=label, **kw)
    else:
        raise ValueError(f'Invalid plot type {plot_func}. Options are scatter, plot')

    for ax in axes:
        ax.set_xlabel('$f$ / Hz')
        ax.set_xscale('log')

    def ax_title(col, area):
        cdict = col_dict.get(col, {})
        if area is not None and cdict.get('units', '') == '$\Omega$':
            title = '{} / {}{}$\cdot\mathrm{{cm}}^2$'.format(cdict.get('label', col), unit_scale,
                                                             cdict.get('units', ''))
        elif cdict.get('units', '') == '$\Omega$':
            title = '{} / {}{}'.format(cdict.get('label', col), unit_scale, cdict.get('units', ''))
        else:
            title = '{} / {}'.format(cdict.get('label', col), cdict.get('units', 'a.u.'))

        if col == 'Zimag' and invert_Zimag:
            title = '$-$' + title
        return title

    for col, ax in zip(cols, axes):
        ax.set_ylabel(ax_title(col, area))
        ax.set_yscale(col_dict.get(col, {}).get('scale', 'linear'))
        if col_dict.get(col, {}).get('scale', 'linear') == 'log':
            # if y-axis is log-scaled, manually set limits
            # sometimes matplotlib gets it wrong
            ymin = min(ax.get_ylim()[0], df[col].min() / 2)
            ymax = max(ax.get_ylim()[1], df[col].max() * 2)
            ax.set_ylim(ymin, ymax)

    for ax in axes:
        # manually set x axis limits - sometimes matplotlib doesn't get them right
        fmin = min(df['Freq'].min(), ax.get_xlim()[0] * 5)
        fmax = max(df['Freq'].max(), ax.get_xlim()[1] / 5)
        ax.set_xlim(fmin / 5, fmax * 5)

    # if area is not None:
    # ax1.set_ylabel('$Z_{\mathrm{mod}} \ (\Omega\cdot \mathrm{cm}^2)$')
    # else:
    # ax1.set_ylabel('$Z_{\mathrm{mod}} \ (\Omega)$')

    # ax1.set_yscale('log')
    # ax2.set_ylabel('$Z_{\mathrm{phz}} \ (^\circ)$')

    fig.tight_layout()

    return axes


def plot_eis(df, plot_type='all', area=None, axes=None, label='', plot_func='scatter', unit_scale='auto',
             bode_cols=['Zmod', 'Zphz'], set_aspect_ratio=True, **kw):
    """
    Plot eis data in Nyquist and/or Bode plot(s)
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame of impedance data
    plot_type : str, optional (default: 'all')
        Type of plot(s) to create. Options:
            'all': Nyquist and Bode plots
            'nyquist': Nyquist plot only
            'bode': Bode plots only
    area : float, optional (default: None)
        Active area in cm^2. If provided, plot area-normalized impedance
    axes : array, optional (default: None)
        Axes on which to plot. If None, axes will be created
    label : str, optional (default: '')
        Label for data
    plot_func : str, optional (default: 'scatter')
        Name of matplotlib.pyplot function to use. Options: 'scatter', 'plot'
    unit_scale: str, optional (default: 'auto')
        Scaling unit prefix. If 'auto', determine from data.
        Options are 'mu', 'm', '', 'k', 'M', 'G'
    bode_cols : list, optional (default: ['Zmod', 'Zphz'])
        List of data columns to plot in Bode plots. Options: 'Zreal', 'Zimag', 'Zmod', 'Zphz'
        Only used if plot_type in ('all', 'bode')
    set_aspect_ratio : bool, optional (default: True)
        If True, ensure that visual scale of x and y axes is the same for Nyquist plot.
        Only used if plot_type in ('all', 'nyquist')
    kw :
        Keywords to pass to matplotlib.pyplot.plot_func

    Returns
    -------

    """
    if plot_type == 'bode':
        axes = plot_bode(df, area=area, label=label, axes=axes, plot_func=plot_func, cols=bode_cols,
                         unit_scale=unit_scale, **kw)
    elif plot_type == 'nyquist':
        axes = plot_nyquist(df, area=area, ax=axes, label=label, plot_func=plot_func, unit_scale=unit_scale,
                            set_aspect_ratio=set_aspect_ratio, **kw)
    elif plot_type == 'all':
        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(9, 2.75))
            ax1, ax2, ax3 = axes.ravel()
        else:
            ax1, ax2, ax3 = axes.ravel()
            fig = axes.ravel()[0].get_figure()

        # Nyquist plot
        plot_nyquist(df, area=area, ax=ax1, label=label, plot_func=plot_func, unit_scale=unit_scale,
                     set_aspect_ratio=set_aspect_ratio, **kw)

        # Bode plots
        plot_bode(df, area=area, label=label, axes=(ax2, ax3), plot_func=plot_func, cols=bode_cols,
                  unit_scale=unit_scale,
                  **kw)

        fig.tight_layout()
    else:
        raise ValueError(f'Invalid plot_type {plot_type}. Options: all, bode, nyquist')

    return axes


# ----------------------------------
# Functions for plotting DRT results
# ----------------------------------
def plot_distribution(df, inv, ax=None, distribution=None, tau_plot=None, plot_bounds=True, plot_ci=True,
                      label='', ci_label='', unit_scale='auto', freq_axis=True, area=None, normalize=False,
                      predict_kw={}, **kw):
    """
    Plot the specified distribution as a function of tau.

	Parameters
	----------
    df : pandas DataFrame
        DataFrame containing experimental EIS data. Used only for scaling and frequency bounds
        If None is passed, scaling will not be performed and frequency bounds will not be drawn
    inv : Inverter instance
        Fitted Inverter instance
    ax : matplotlib axis
        Axis on which to plot
    distribution : str, optional (default: None)
        Name of distribution to plot. If None, first distribution in inv.distributions will be used
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
        If True, normalize the distribution such that the polarization resistance is 1
    predict_kw : dict, optional (default: {})
        Keyword args to pass to Inverter predict_distribution() method
    kw : keyword args, optional
        Keyword args to pass to maplotlib.pyplot.plot
    Returns
    -------
    ax : matplotlib axis
        Axis on which distribution is plotted
	"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 2.75))

    # If no distribution specified, use first distribution
    if distribution is None:
        distribution = list(inv.distributions.keys())[0]

    # If tau_plot not given, go one decade beyond basis tau in each direction
    if tau_plot is None:
        basis_tau = inv.distributions[distribution]['tau']
        tmin = np.log10(np.min(basis_tau)) - 1
        tmax = np.log10(np.max(basis_tau)) + 1
        num_decades = tmax - tmin
        tau_plot = np.logspace(tmin, tmax, int(20 * num_decades + 1))

    F_pred = inv.predict_distribution(distribution, tau_plot, **predict_kw)

    if normalize and area is not None:
        raise ValueError('If normalize=True, area cannot be specified.')

    if area is not None:
        if df is not None:
            df = df.copy()
            for col in ['Zmod', 'Zreal', 'Zimag']:
                df[col] *= area
        F_pred *= area

    if normalize:
        Rp_kw = predict_kw.copy()
        # if time given, calculate Rp at given time
        if 'time' in predict_kw.keys():
            Rp_kw['times'] = [predict_kw['time'], predict_kw['time']]
            del Rp_kw['time']
        Rp = inv.predict_Rp(**Rp_kw)
        F_pred /= Rp

    if unit_scale == 'auto':
        if normalize:
            unit_scale = ''
        elif df is not None:
            unit_scale = get_unit_scale(df, area)
        else:
            unit_map = {-2: '$\mu$', -1: 'm', 0: '', 1: 'k', 2: 'M', 3: 'G'}
            F_max = np.max(F_pred)
            F_ord = np.floor(np.log10(F_max) / 3)
            unit_scale = unit_map.get(F_ord, '')
    scale_factor = get_factor_from_unit(unit_scale)

    ax.plot(tau_plot, F_pred / scale_factor, label=label, **kw)

    if plot_ci:
        if inv.fit_type.find('bayes') >= 0:
            F_lo = inv.predict_distribution(distribution, tau_plot, percentile=2.5, **predict_kw)
            F_hi = inv.predict_distribution(distribution, tau_plot, percentile=97.5, **predict_kw)
            if area is not None:
                F_lo *= area
                F_hi *= area
            if normalize:
                F_lo /= Rp
                F_hi /= Rp
            ax.fill_between(tau_plot, F_lo / scale_factor, F_hi / scale_factor, color='k', alpha=0.2, label=ci_label)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau$ / s')

    if plot_bounds:
        if df is not None:
            ax.axvline(1 / (2 * np.pi * df['Freq'].max()), c='k', ls=':', alpha=0.6, zorder=-10)
            ax.axvline(1 / (2 * np.pi * df['Freq'].min()), c='k', ls=':', alpha=0.6, zorder=-10)

    if area is not None:
        ax.set_ylabel(fr'$\gamma \, (\ln{{\tau}})$ / {unit_scale}$\Omega\cdot\mathrm{{cm}}^2$')
    elif normalize:
        ax.set_ylabel(fr'$\gamma \, (\ln{{\tau}}) / R_p$')
    else:
        ax.set_ylabel(fr'$\gamma \, (\ln{{\tau}})$ / {unit_scale}$\Omega$')

    # add freq axis to DRT plot
    if freq_axis:
        # check for existing twin axis
        all_axes = ax.figure.axes
        ax2 = None
        for other_ax in all_axes:
            if other_ax.bbox.bounds == ax.bbox.bounds and other_ax is not ax:
                ax2 = other_ax
                break
            else:
                continue

        if ax2 is None:
            ax2 = ax.twiny()

        ax2.set_xscale('log')
        ax2.set_xlim(ax.get_xlim())
        f_powers = np.arange(7, -4.1, -2)
        f_ticks = 10 ** f_powers
        ax2.set_xticks(1 / (2 * np.pi * f_ticks))
        ax2.set_xticklabels(['$10^{{{}}}$'.format(int(p)) for p in f_powers])
        ax2.set_xlabel('$f$ / Hz')

    # Indicate zero if necessary
    if np.min(F_pred) >= 0:
        ax.set_ylim(0, ax.get_ylim()[1])
    else:
        ax.axhline(0, c='k', lw=0.5)

    return ax


def plot_fit(df, inv, axes=None, plot_type='all', bode_cols=['Zreal', 'Zimag'], plot_data=True, color='k',
                 f_pred=None, label='', data_label='', unit_scale='auto', area=None, predict_kw={}, data_kw={}, **kw):
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
        Keywords to pass to inv.predict_Z
    data_kw : dict, optional (default: {})
        Keywords to pass to matplotlib.pyplot.scatter when plotting data points
    kw : dict, optional (default: {})
        Keywords to pass to matplotlib.pyplot.plot when plotting fit line
    Returns
    -------
    axes : array
        Axes on which fit is plotted
    """
    if axes is None:
        if plot_type == 'nyquist':
            fig, axes = plt.subplots(figsize=(3.5, 2.75))
        elif plot_type == 'bode':
            fig, axes = plt.subplots(1, len(bode_cols), figsize=(3 * len(bode_cols), 2.75))
        elif plot_type == 'all':
            fig, axes = plt.subplots(1, len(bode_cols) + 1, figsize=(3 * (len(bode_cols) + 1), 2.75))

    if unit_scale == 'auto':
        unit_scale = get_unit_scale(df, area)

    freq = df['Freq'].values
    if f_pred is None:
        f_pred = freq

    Z_pred = inv.predict_Z(f_pred, **predict_kw)
    df_pred = fl.construct_eis_df(f_pred, Z_pred)

    data_defaults = dict(s=10, alpha=0.5)
    data_defaults.update(data_kw)

    # plot Z fit
    if plot_type == 'all':
        if plot_data:
            plot_eis(df, area=area, axes=axes, label=data_label, unit_scale=unit_scale, bode_cols=bode_cols,
                     **data_defaults)
        plot_eis(df_pred, area=area, axes=axes, label=label, plot_func='plot', unit_scale=unit_scale,
                 bode_cols=bode_cols, color=color, **kw)
    elif plot_type == 'nyquist':
        if plot_data:
            plot_nyquist(df, area=area, ax=axes, label=data_label, unit_scale=unit_scale, **data_defaults)
        plot_nyquist(df_pred, area=area, ax=axes, label=label, plot_func='plot', unit_scale=unit_scale, color=color,
                     **kw)
    elif plot_type == 'bode':
        if plot_data:
            plot_bode(df, cols=bode_cols, axes=axes, label=data_label, area=area, unit_scale=unit_scale,
                      **data_defaults)
        plot_bode(df_pred, axes=axes, cols=bode_cols, plot_func='plot', color=color, unit_scale=unit_scale,
                  label=label, area=area, **kw)

    return axes


def plot_residuals(df, inv, axes=None, unit_scale='auto', plot_ci=True, predict_kw={}):
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
        Keywords to pass to inv.predict_Z
    Returns
    -------
    axes : array
        Axes on which residuals are plotted
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(6, 2.75))

    freq, Z = fl.get_fZ(df)
    Z_pred = inv.predict_Z(freq, **predict_kw)

    df_err = fl.construct_eis_df(freq, Z_pred - Z)
    if unit_scale == 'auto':
        err_scale = get_scale_factor(df_err)
        unit_scale = get_unit_scale(df_err)
    else:
        err_scale = get_factor_from_unit(unit_scale)

    plot_bode(df_err, axes=axes, s=10, alpha=0.5, cols=['Zreal', 'Zimag'], unit_scale=unit_scale, label='Residuals')
    if (inv.fit_type == 'bayes' or inv.fit_type[:3] == 'map') and plot_ci:
        sigma_re, sigma_im = inv.predict_sigma(freq, **predict_kw)
        axes[0].fill_between(freq, -3 * sigma_re / err_scale, 3 * sigma_re / err_scale, color='k', alpha=0.15,
                             label='$\pm 3 \sigma$')
        axes[1].fill_between(freq, -3 * sigma_im / err_scale, 3 * sigma_im / err_scale, color='k', alpha=0.15,
                             label='$\pm 3 \sigma$')

    for ax in axes:
        ax.axhline(0, c='k', lw=0.5)

    axes[1].legend()

    axes[0].set_ylabel(fr'$\hat{{Z}}^{{\prime}}-Z^{{\prime}}$ / {unit_scale}$\Omega$')
    axes[1].set_ylabel(fr'$-(\hat{{Z}}^{{\prime\prime}}-Z^{{\prime\prime}})$ / {unit_scale}$\Omega$')

    return axes


def plot_full_results(df, inv, axes=None, bode_cols=['Zreal', 'Zimag'], plot_data=True, color='k',
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
        Keywords to pass to inv.predict_Z

    Returns
    -------
    axes : array
        Axes on which results are plotted
    """
    if axes is None:
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    else:
        fig = axes.ravel()[0].get_figure()

    unit_scale = get_unit_scale(df)
    scale_factor = get_scale_factor(df)

    # plot Z fit
    plot_fit(df, inv, axes[0], bode_cols=bode_cols, plot_data=plot_data, color=color, f_pred=f_pred, label='Fit',
             data_label='Data', predict_kw=predict_kw)

    # plot DRT
    if 'times' in predict_kw.keys():
        # Plot DRT at initial and final times
        tmp_kw = predict_kw.copy()
        del tmp_kw['times']
        tmp_kw['time'] = predict_kw['times'][0]
        plot_distribution(df, inv, axes[1, 0], color='k', plot_bounds=plot_data, tau_plot=tau_plot, predict_kw=tmp_kw,
                          label='Initial', plot_ci=plot_drt_ci)
        tmp_kw['time'] = predict_kw['times'][-1]
        plot_distribution(df, inv, axes[1, 0], color='r', plot_bounds=plot_data, tau_plot=tau_plot, predict_kw=tmp_kw,
                          label='Final', plot_ci=plot_drt_ci)
        axes[1, 0].legend()
    else:
        plot_distribution(df, inv, axes[1, 0], color=color, plot_bounds=plot_data, tau_plot=tau_plot, ci_label='95% CI',
                          predict_kw=predict_kw, plot_ci=plot_drt_ci)

        if plot_drt_ci and inv.fit_type == 'bayes':
            axes[1, 0].legend()

    # plot error
    plot_residuals(df, inv, axes[1, 1:], plot_ci=plot_ci, predict_kw=predict_kw)

    axes[0, 0].axhline(0, color='k', lw=0.5)
    axes[0, 2].axhline(0, color='k', lw=0.5)

    fig.tight_layout()

    return axes
