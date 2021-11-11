# functions for plotting DRT fits
# should move potting functions over from eis_utils at some point
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import eis_utils as gt

def plot_distribution(df,inv,ax,distribution='DRT',tau_plot=np.logspace(-8,3,200),plot_bounds=True,plot_ci=True,
					label='',ci_label='',unit_scale='auto',freq_axis=True,area=None,normalize=False,predict_kw={},**kw):
	"""
	Parameters:
	----------
		df: pandas DataFrame
			DataFrame containing experimental EIS data. Used only for scaling and frequency bounds
			If None is passed, scaling will not be performed and frequency bounds will not be drawn
		inv: Inverter instance
			Fitted Inverter instance
		ax: matplotlib axis
			Axis to plot on
		distribution: str, optional (default:'DRT')
			Name of distribution to plot
		tau_plot: array, optonal (default:np.logspace(-8,3,200))
			Time constant grid over which to evaluate the distribution
		plot_bounds: bool, optional (default: True)
			If True, indicate frequency bounds of experimental data with vertical lines.
			Requires that DataFrame of experimental data be passed for df argument
		label: str, optional (default: '')
			Label for matplotlib
		unit_scale: str, optional (default: 'auto')
			Scaling unit prefix. If 'auto', determine from data. 
			Options are 'mu', 'm', '', 'k', 'M', 'G'
		freq_axis: bool, optional (default: True)
			If True, add a secondary x-axis to display frequency
		area: float, optional (default: None)
			Active area. If provided, plot the area-normalized distribution
		normalize: bool, optional (default: False)
			If True, normalize the distribution such that the polarization resistance is 1
		predict_kw: dict, optional (default: {})
			Keyword args to pass to Inverter predict_distribution() method
		**kw: keyword args, optional
			Keyword args to pass to maplotlib.pyplot.plot
	"""
	F_pred = inv.predict_distribution(distribution,tau_plot,**predict_kw)
	
	if normalize and area is not None:
		raise ValueError('If normalize=True, area cannot be specified.')
		
	if area is not None:
		if df is not None: 
			for col in ['Zmod','Zreal','Zimag']:
				df[col] *= area
		F_pred *= area
		
	if normalize:
		Rp_kw = predict_kw.copy()
		# if time given, calculate Rp at given time
		if 'time' in predict_kw.keys():
			Rp_kw['times'] = [predict_kw['time'],predict_kw['time']]
			del Rp_kw['time']
		Rp = inv.predict_Rp(**Rp_kw)
		F_pred /= Rp
		
	if unit_scale=='auto':
		if normalize:
			unit_scale = ''
		elif df is not None:
			unit_scale = gt.get_unit_scale(df,area)
		else:
			unit_map = {-2:'$\mu$',-1:'m',0:'',1:'k',2:'M',3:'G'}
			F_max = np.max(F_pred)
			F_ord = np.floor(np.log10(F_max)/3)
			unit_scale = unit_map.get(F_ord,'')
	scale_factor = gt.get_factor_from_unit(unit_scale)
	
	ax.plot(tau_plot,F_pred/scale_factor,label=label,**kw)
	
	if plot_ci:
		if inv.fit_type.find('bayes') >= 0:
			F_lo = inv.predict_distribution(distribution,tau_plot,percentile=2.5,**predict_kw)
			F_hi = inv.predict_distribution(distribution,tau_plot,percentile=97.5,**predict_kw)
			if area is not None:
				F_lo *= area
				F_hi *= area
			if normalize:
				F_lo /= Rp
				F_hi /= Rp
			ax.fill_between(tau_plot,F_lo/scale_factor,F_hi/scale_factor,color='k',alpha=0.2,label=ci_label)
	
	ax.set_xscale('log')
	ax.set_xlabel(r'$\tau$ / s')
	
	if plot_bounds:
		if df is not None:
			ax.axvline(1/(2*np.pi*df['Freq'].max()),c='k',ls=':',alpha=0.6,zorder=-10)
			ax.axvline(1/(2*np.pi*df['Freq'].min()),c='k',ls=':',alpha=0.6,zorder=-10)
	
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
			if other_ax.bbox.bounds==ax.bbox.bounds and other_ax is not ax:
				ax2 = other_ax
				break
			else:
				continue
				
		if ax2 is None:
			ax2 = ax.twiny()
		
		ax2.set_xscale('log')
		ax2.set_xlim(ax.get_xlim())
		f_powers = np.arange(7,-4.1,-2)
		f_ticks = 10**f_powers
		ax2.set_xticks(1/(2*np.pi*f_ticks))
		ax2.set_xticklabels(['$10^{{{}}}$'.format(int(p)) for p in f_powers])
		ax2.set_xlabel('$f$ / Hz')
	
	ax.axhline(0,c='k',lw=0.5)

def plot_resid(df,inv,axes,unit_scale='auto',plot_ci=True,predict_kw={}):
	freq = df['Freq'].values
	Z = df['Zreal'].values + 1j*df['Zimag'].values
	Z_pred = inv.predict_Z(freq,**predict_kw)
	
	df_err = gt.construct_eis_df(freq,Z_pred-Z)
	if unit_scale=='auto':
		err_scale = gt.get_scale_factor(df_err)
		unit_scale = gt.get_unit_scale(df_err)
	else:
		err_scale = gt.get_factor_from_unit(unit_scale)
		
	gt.plot_bode(df_err,axes=axes,s=10,alpha=0.5,cols=['Zreal','Zimag'],unit_scale=unit_scale)
	if (inv.fit_type=='bayes' or inv.fit_type[:3]=='map') and plot_ci:
		sigma_re,sigma_im = inv.predict_sigma(freq,**predict_kw)
		axes[0].fill_between(freq,-3*sigma_re/err_scale,3*sigma_re/err_scale,color='k',alpha=0.15,label='$\pm 3 \sigma$')
		axes[1].fill_between(freq,-3*sigma_im/err_scale,3*sigma_im/err_scale,color='k',alpha=0.15,label='$\pm 3 \sigma$')
	
	for ax in axes:
		ax.axhline(0,c='k',lw=0.5)
		
	axes[0].set_ylabel(fr'$\hat{{Z}}^{{\prime}}-Z^{{\prime}}$ / {unit_scale}$\Omega$')
	axes[1].set_ylabel(fr'$-(\hat{{Z}}^{{\prime\prime}}-Z^{{\prime\prime}})$ / {unit_scale}$\Omega$')
	
def plot_drt_fit(df,inv,axes,plot_type='all',bode_cols=['Zreal','Zimag'],plot_data=True,color='k',
				 f_pred=None,label='',data_label='',unit_scale='auto',area=None,predict_kw={},data_kw={},**kw):
	if unit_scale=='auto':
		unit_scale = gt.get_unit_scale(df,area)
	freq = df['Freq'].values
	if f_pred is None:
		f_pred = freq
		
	Z_pred = inv.predict_Z(f_pred,**predict_kw)
	df_pred = gt.construct_eis_df(f_pred,Z_pred)
	
	data_defaults = dict(s=10,alpha=0.5)
	data_defaults.update(data_kw)
		
	# plot Z fit
	if plot_type=='all':	
		if plot_data:
			gt.plot_full_eis(df,bode_cols=bode_cols,axes=axes,label=data_label,area=area,unit_scale=unit_scale,**data_defaults)
		gt.plot_full_eis(df_pred,axes=axes,bode_cols=bode_cols,plot_func='plot',color=color,unit_scale=unit_scale,label=label,area=area,**kw)
	elif plot_type=='nyquist':
		if plot_data:
			gt.plot_nyquist(df,ax=axes,label=data_label,area=area,unit_scale=unit_scale,**data_defaults)
		gt.plot_nyquist(df_pred,ax=axes,plot_func='plot',color=color,unit_scale=unit_scale,label=label,area=area,**kw)
	elif plot_type=='bode':
		if plot_data:
			gt.plot_bode(df,cols=bode_cols,axes=axes,label=data_label,area=area,unit_scale=unit_scale,**data_defaults)
		gt.plot_bode(df_pred,axes=axes,cols=bode_cols,plot_func='plot',color=color,unit_scale=unit_scale,label=label,area=area,**kw)
	
def plot_drt_result(df,inv,bode_cols=['Zreal','Zimag'],plot_data=True,color='k',axes=None,
					tau_plot=np.logspace(-8,3,200),f_pred=None,plot_ci=True,plot_drt_ci=True,predict_kw={}):
	if axes is None:
		fig,axes = plt.subplots(2,3,figsize=(9,6))
	else:
		fig = axes.ravel()[0].get_figure()

	unit_scale = gt.get_unit_scale(df)
	scale_factor = gt.get_scale_factor(df)

	# plot Z fit
	plot_drt_fit(df,inv,axes[0],bode_cols=bode_cols,color=color,f_pred=f_pred,plot_data=plot_data,predict_kw=predict_kw)

	# plot DRT
	if 'times' in predict_kw.keys():
		# Plot DRT at initial and final times
		tmp_kw = predict_kw.copy()
		del tmp_kw['times']
		tmp_kw['time'] = predict_kw['times'][0]
		plot_distribution(df,inv,axes[1,0],color='k',plot_bounds=plot_data,tau_plot=tau_plot,predict_kw=tmp_kw,label='Initial',plot_ci=plot_drt_ci)
		tmp_kw['time'] = predict_kw['times'][-1]
		plot_distribution(df,inv,axes[1,0],color='r',plot_bounds=plot_data,tau_plot=tau_plot,predict_kw=tmp_kw,label='Final',plot_ci=plot_drt_ci)
		axes[1,0].legend()
	else:
		plot_distribution(df,inv,axes[1,0],color=color,plot_bounds=plot_data,tau_plot=tau_plot,predict_kw=predict_kw,plot_ci=plot_drt_ci)

	# plot error
	plot_resid(df,inv,axes[1,1:],plot_ci=plot_ci,predict_kw=predict_kw)

	axes[0,0].axhline(0,color='k',lw=0.5)
	axes[0,2].axhline(0,color='k',lw=0.5)
	
	fig.tight_layout()
		
	return axes