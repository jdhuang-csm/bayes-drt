# Module for plotting and fitting EIS data
# (C) Jake Huang 2020
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, basinhopping, differential_evolution, curve_fit, least_squares
from datetime import datetime, timedelta
import itertools
import re
from scipy.stats import iqr, mode
from scipy.special import binom
import inspect
from copy import copy
import warnings
from io import StringIO

#------------------
# Misc functions
#------------------
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
	
def fit_r_squared(x,y,fit,weights=None):
	"""
	Calculate r squared for polynomial fit
	
	Args:
		x: x values
		y: y values
		fit: numpy polyfit output, or array of coefficients
		weights: sample weights
	"""
	y_hat = np.polyval(fit,x)
	return r_squared(y,y_hat,weights)
	
def r_squared(y,y_hat,weights=None):
	"""
	Calculate r squared for 
	
	Args:
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
	
def reg_degree_polyfit(x,y,alpha,min_r2=0,weights=None,verbose=False):
	"""
	Regularized-degree polynomial fit. L2 regularization penalty applied to polynomial degree
	
	Args:
		x: x values
		y: y values
		alpha: regularization strength
		min_r2: minimum r2. If specified, degree will be increased until this min value is achieved, even if overall score decreases
		weights: weights for fit
		verbose: if True, print info about best fit, plus previous and next degree fits
	"""
	
	best_score = -np.inf
	deg = 1
	r2 = -np.inf
	while deg < len(x):
		fit = np.polyfit(x,y,deg=deg,w=weights)
		last_r2 = r2
		r2 = fit_r_squared(x,y,fit,weights=weights)
		score = r2 - alpha*deg**2
		if score > best_score:# or r2 < min_r2:
			#print(f'Deg {deg}, Case 1,','r2={},last r2={}'.format(round(r2,5),round(last_r2,5)))
			best_fit = fit
			best_score = score
			best_deg = deg
			deg += 1
		elif last_r2 < min_r2:# and r2 >= min_r2:
			#print(f'Deg {deg}, Case 2,','r2={},last r2={}'.format(round(r2,5),round(last_r2,5)))
			best_fit = fit
			best_score = score
			best_deg = deg
			deg += 1
			#break
		else:
			break
	
	if verbose==True:
		print('Best score: degree={}, r2={}, score={}'.format(best_deg,round(fit_r_squared(x,y,best_fit,w=weights),5), round(best_score,5)))
		if best_deg > 1:
			prev_r2 = fit_r_squared(x,y,np.polyfit(x,y,deg=deg-2,w=weights),w=weights)
			prev_score = prev_r2 - alpha*(deg-2)**2
			print('Previous degree: degree={}, r2={}, score={}'.format(deg-2,round(prev_r2,5), round(prev_score,5)))
		print('Next degree: degree={}, r2={}, score={}'.format(deg,round(fit_r_squared(x,y,fit,weights),5), round(score,5)))
	
	return best_fit

#---------------------
# File loading
#---------------------
def source_extension(source):
	"""Get file extension for source"""
	extensions = {'gamry':'.DTA','zplot':'.z'}
	return extensions[source]

def get_file_source(file):
	"""Determine file source"""
	try:
		with open(file,'r') as f:
			txt = f.read()
	except UnicodeDecodeError:
		with open(file,'r',encoding='latin1') as f:
			txt = f.read()
	# determine	format
	if txt.split('\n')[0]=='EXPLAIN':
		source = 'gamry'
	elif txt.split('\n')[0]=='ZPLOT2 ASCII':
		source='zplot'
	
	return source

def get_timestamp(file):
	"""Get experiment start timestamp from file"""
	try:
		with open(file,'r') as f:
			txt = f.read()
	except UnicodeDecodeError:
		with open(file,'r',encoding='latin1') as f:
			txt = f.read()
		
	source = get_file_source(file)

	if source=='gamry':
		date_start = txt.find('DATE')
		date_end = txt[date_start:].find('\n') + date_start
		date_line = txt[date_start:date_end]
		date = date_line.split('\t')[2]

		time_start = txt.find('TIME')
		time_end = txt[time_start:].find('\n') + time_start
		time_line = txt[time_start:time_end]
		time = time_line.split('\t')[2]
		
		timestr = date + ' ' + time
		dt = datetime.strptime(timestr,"%m/%d/%Y %H:%M:%S")
		
	elif source=='zplot':
		date_start = txt.find('Date')
		date_end = txt[date_start:].find('\n') + date_start
		date_line = txt[date_start:date_end]
		date = date_line.split()[1]

		time_start = txt.find('Time')
		time_end = txt[time_start:].find('\n') + time_start
		time_line = txt[time_start:time_end]
		time = time_line.split()[1]
		
		timestr = date + ' ' + time
		dt = datetime.strptime(timestr,"%m-%d-%Y %H:%M:%S")
		
	return dt
	
def get_file_info(file,sequence=['file_type','T','aflow','cflow']):
	"""
	Get information from filename
	
	Args:
		file: filename (basename or full path)
		sequence: list of identifiers in the order that they appear in the filename (separated by _)
	"""
	fname = os.path.basename(file).replace('.DTA','')
	info = dict(zip(sequence,fname.split('_')))
	
	info['T'] = int(info['T'][:info['T'].find('C')])
	for flow in ('aflow','cflow'):
		try:
			if info[flow].find('sccm') > 0:
				rate,gas = info[flow].split('sccm')
				gas = ' '.join(camel_case_split(gas))
				info[flow] = ' '.join([rate,'SCCM',gas])
			else:
				info[flow] = ' '.join(camel_case_split(info[flow]))
		except KeyError:
			pass
	
	return info

def read_eis_zdata(file):
	"""read EIS zcurve data from Gamry .DTA file"""
	try:
		with open(file,'r') as f:
			txt = f.read()
	except UnicodeDecodeError:
		with open(file,'r',encoding='latin1') as f:
			txt = f.read()
	source = get_file_source(file)
	
	if source=='gamry':
		#find start of zcurve data
		zidx = txt.find('ZCURVE')
		#check for experiment aborted flag
		if txt.find('EXPERIMENTABORTED') > -1:
			skipfooter = len(txt[txt.find('EXPERIMENTABORTED'):].split('\n')) - 1
		else:
			skipfooter = 0

		#preceding text
		pretxt = txt[:zidx]

		#zcurve data
		ztable = txt[zidx:]
		#column headers are next line after ZCURVE TABLE line
		header_start = ztable.find('\n') + 1
		header_end = header_start + ztable[header_start:].find('\n')
		header = ztable[header_start:header_end].split('\t')
		#units are next line after column headers
		unit_end = header_end + 1 + ztable[header_end + 1:].find('\n')
		units = ztable[header_end + 1:unit_end].split('\t')
		#determine # of rows to skip by counting line breaks in preceding text
		skiprows = len(pretxt.split('\n')) + 2

		# if table is indented, ignore empty left column
		if header[0] == '':
			usecols = header[1:]
		else:
			usecols = header
			
		# if extra tab at end of data rows, add an extra column to header to match (for Igor data)
		first_data_row = ztable[unit_end+1: unit_end+ 1 + ztable[unit_end+1:].find('\n')]
		if first_data_row.split('\t')[-1]=='':
			header = header + ['extra_tab']

		#read data to DataFrame
		#python engine required to use skipfooter
		data = pd.read_csv(file,sep='\t',skiprows=skiprows,header=None,names=header,usecols=usecols,skipfooter=skipfooter,engine='python')
		
	elif source=='zplot':
		#find start of zcurve data
		zidx = txt.find('End Comments')

		#preceding text
		pretxt = txt[:zidx]

		#z data
		ztable = txt[zidx:]
		#column headers are in line above "End Comments"
		header = pretxt.split('\n')[-2].strip().split('\t')
		
		#determine # of rows to skip by counting line breaks in preceding text
		skiprows = len(pretxt.split('\n'))

		# if table is indented, ignore empty left column
		if header[0] == '':
			usecols = header[1:]
		else:
			usecols = header

		# read data to DataFrame
		data = pd.read_csv(file,sep='\t',skiprows=skiprows,header=None,names=header,usecols=usecols)
		
		# rename to standard format
		rename = {"Z'(a)":"Zreal","Z''(b)":"Zimag","Freq(Hz)":"Freq"}
		data = data.rename(rename,axis=1)
		
		# calculate Zmod and Zphz
		Zmod, Zphz = bode_from_complex(data)
		data['Zmod'] = Zmod
		data['Zphz'] = Zphz
	
	return data
	
def read_nleis_data(file):
	try:
		with open(file,'r') as f:
			txt = f.read()
	except UnicodeDecodeError:
		with open(file,'r',encoding='latin1') as f:
			txt = f.read()
	source = get_file_source(file)
	
	if source=='gamry':
		# get number of points measured
		num_freq_start = txt.find('NUMFREQ')
		num_freq_end = txt.find('\n',num_freq_start+1)
		num_freq_line = txt[num_freq_start:num_freq_end]
		num_freq = int(num_freq_line.split('\t')[2])
		
		frequency_data = {}
		
		for n in range(num_freq):
			fra_start = txt.find(f'FREQUENCY {n}')
			if n==num_freq-1:
				fra_end = txt.find('ZCURVE')
			else:
				fra_end = txt.find('FREQUENCY {}'.format(n+1))
			fra_txt = txt[fra_start:fra_end]
			
			# get frequency
			freq_line = fra_txt[:fra_txt.find('\n')]
			requested_freq = float(freq_line.split('\t')[1].replace('Requested Freq (Hz):','').strip())
			actual_freq = float(freq_line.split('\t')[2].replace('Actual Freq (Hz):','').strip())
			
			# get header
			header_start = fra_txt.find('\n',fra_txt.find('\n')+1) + 1
			header_end = fra_txt.find('\n',header_start)
			header = fra_txt[header_start:header_end].split('\t')
			# if table is indented, ignore empty left column
			if header[0] == '':
				usecols = header[1:]
			else:
				usecols = header
				
			fra_table = fra_txt[fra_txt.find('\n',header_end+1) + 1:]
			fra_data = pd.read_csv(StringIO(fra_table),sep='\t',header=None,names=header,usecols=usecols)
			
			frequency_data[n] = {'requested_freq':requested_freq,'actual_freq':actual_freq,'data':fra_data}
		
		return frequency_data
		
	
def read_jv_data(file,source='gamry'):
	"""read from manual jV txt file"""
	try:
		with open(file,'r') as f:
			txt = f.read()
	except UnicodeDecodeError:
		with open(file,'r',encoding='latin1') as f:
			txt = f.read()
		
	if source=='manual':
		"""Manually created j-V txt file"""
		jv_idx = txt.find('Current')
		pretxt = txt[:jv_idx]
		skiprows = len(pretxt.split('\n'))-1
		data = pd.read_csv(file,sep='\t',skiprows=skiprows)
	elif source=='gamry':
		#find start of curve data
		cidx = txt.find('CURVE\tTABLE')

		#preceding text
		pretxt = txt[:cidx]

		#curve data
		ctable = txt[cidx:]
		#column headers are next line after ZCURVE TABLE line
		header_start = ctable.find('\n') + 1
		header_end = header_start + ctable[header_start:].find('\n')
		header = ctable[header_start:header_end].split('\t')
		#units are next line after column headers
		unit_end = header_end + 1 + ctable[header_end + 1:].find('\n')
		units = ctable[header_end + 1:unit_end].split('\t')
		#determine # of rows to skip by counting line breaks in preceding text
		skiprows = len(pretxt.split('\n')) + 2
		
		#if table is indented, ignore empty left column
		if header[0] == '':
			usecols = header[1:]
		else:
			usecols = header
		#read data to DataFrame
		data = pd.read_csv(file,sep='\t',skiprows=skiprows,header=None,names=header,usecols=usecols)
	else:
		raise ValueError(f'Invalid source {source}. Options are ''gamry'', ''manual''')
		
	return data	

def read_ocv_data(file,file_type='auto'):
	"""
	read OCV data from Gamry .DTA file
	
	Args:
		file: file to read
		file_type: file type. Options are 'ocv','eis'
	"""
	try:
		with open(file,'r') as f:
			txt = f.read()
	except UnicodeDecodeError:
		with open(file,'r',encoding='latin1') as f:
			txt = f.read()
	
	if file_type=='auto':
		file_type = os.path.basename(file).split('_')[0].lower()[:3]
	#find start (and end, if needed) of ocv data
	if file_type=='ocv':
		cidx = txt.find('CURVE\tTABLE')
		skipfooter = 0
	elif file_type=='eis':
		cidx = txt.find('OCVCURVE\tTABLE')
		post_txt = txt[txt.find('EOC\tQUANT'):]
		skipfooter = len(post_txt.split('\n')) - 1
		
	if cidx==-1:
		# coudn't find OCV curve data in file
		# return empty dataframe 
		return pd.DataFrame([])
	else:
		#preceding text
		pretxt = txt[:cidx]

		#ocv curve data
		ctable = txt[cidx:]
		#column headers are next line after ZCURVE TABLE line
		header_start = ctable.find('\n') + 1
		header_end = header_start + ctable[header_start:].find('\n')
		header = ctable[header_start:header_end].split('\t')
		#units are next line after column headers
		unit_end = header_end + 1 + ctable[header_end + 1:].find('\n')
		units = ctable[header_end + 1:unit_end].split('\t')
		#determine # of rows to skip by counting line breaks in preceding text
		skiprows = len(pretxt.split('\n')) + 2
		
		#if table is indented, ignore empty left column
		if header[0] == '':
			usecols = header[1:]
		else:
			usecols = header
		#read data to DataFrame
		data = pd.read_csv(file,sep='\t',skiprows=skiprows,skipfooter=skipfooter,header=None,names=header,usecols=usecols,engine='python')
		
		#get timestamp
		dt = get_timestamp(file)
		#time_col = np.intersect1d(['Time','T'],data.columns) # EIS files in Repeating jv-EIS files have column named 'Time' instead of 'T'
		data['timestamp'] = [dt + timedelta(seconds=t) for t in data['T']]
		
		return data
		
def read_gen_curve_data(file):
	"""
	read generic curve data from Gamry .DTA file
	
	Args:
		file: file to read
	"""
	try:
		with open(file,'r') as f:
			txt = f.read()
	except UnicodeDecodeError:
		with open(file,'r',encoding='latin1') as f:
			txt = f.read()
	
	#find start of curve data	
	cidx = txt.find('CURVE\tTABLE')
	skipfooter = 0
		
	if cidx==-1:
		# coudn't find OCV curve data in file
		# return empty dataframe 
		return pd.DataFrame([])
	else:
		#preceding text
		pretxt = txt[:cidx]

		#ocv curve data
		ctable = txt[cidx:]
		#column headers are next line after ZCURVE TABLE line
		header_start = ctable.find('\n') + 1
		header_end = header_start + ctable[header_start:].find('\n')
		header = ctable[header_start:header_end].split('\t')
		#units are next line after column headers
		unit_end = header_end + 1 + ctable[header_end + 1:].find('\n')
		units = ctable[header_end + 1:unit_end].split('\t')
		#determine # of rows to skip by counting line breaks in preceding text
		skiprows = len(pretxt.split('\n')) + 2
		
		#if table is indented, ignore empty left column
		if header[0] == '':
			usecols = header[1:]
		else:
			usecols = header
		#read data to DataFrame
		data = pd.read_csv(file,sep='\t',skiprows=skiprows,skipfooter=skipfooter,header=None,names=header,usecols=usecols,engine='python')
		
		#get timestamp
		dt = get_timestamp(file)
		#time_col = np.intersect1d(['Time','T'],data.columns) # EIS files in Repeating jv-EIS files have column named 'Time' instead of 'T'
		data['timestamp'] = [dt + timedelta(seconds=t) for t in data['T']]
		
		return data
	
def read_lsv_data(file):
	"""read LSV data from Gamry .DTA file"""
	try:
		with open(file,'r') as f:
			txt = f.read()
	except UnicodeDecodeError:
		with open(file,'r',encoding='latin1') as f:
			txt = f.read()
	#find start of curve data
	cidx = txt.find('CURVE\tTABLE')

	#preceding text
	pretxt = txt[:cidx]

	#LSV curve data
	ctable = txt[cidx:]
	#column headers are next line after CURVE TABLE line
	header_start = ctable.find('\n') + 1
	header_end = header_start + ctable[header_start:].find('\n')
	header = ctable[header_start:header_end].split('\t')
	#units are next line after column headers
	unit_end = header_end + 1 + ctable[header_end + 1:].find('\n')
	units = ctable[header_end + 1:unit_end].split('\t')
	#determine # of rows to skip by counting line breaks in preceding text
	skiprows = len(pretxt.split('\n')) + 2
	
	#if table is indented, ignore empty left column
	if header[0] == '':
		usecols = header[1:]
	else:
		usecols = header
	#read data to DataFrame
	data = pd.read_csv(file,sep='\t',skiprows=skiprows,header=None,names=header,usecols=usecols)
	
	return data
	
def get_cell_name(datadir):
	datadir = os.path.abspath(datadir)
	
	if os.path.basename(os.path.split(datadir)[0])=='Win10 Gamry data':
		# datadir is one level below Gamry data
		celldir = os.path.basename(datadir)
	else:
		# datadir is two levels below Gamry data - need info from both top dir and subdir
		celldir = os.path.join(os.path.basename(os.path.split(datadir)[0]),os.path.basename(datadir))
	
	dirsplit = [txt.replace('-',' ') for txt in re.split('_|/|\\\\',celldir.strip('./'))]
	cell = dirsplit[0] + ' ' + ' | '.join(dirsplit[1:])
	
	return cell
	
#---------------------------------
# Data processing
#---------------------------------

def bode_from_complex(data):
	if type(data)==pd.core.frame.DataFrame:
		Zmod = (data['Zreal'].values**2 + data['Zimag'].values**2)**0.5
		Zphz = (180/np.pi)*np.arctan(data['Zimag'].values/data['Zreal'].values)
	elif type(data)==np.ndarray:
		Zmod = ((data*data.conjugate())**0.5).real
		Zphz = (180/np.pi)*np.arctan(data.imag/data.real)
		
	return Zmod, Zphz
	
def complex_from_bode(data):
	if type(data)==pd.core.frame.DataFrame:
		Zmod = data['Zmod'].values
		Zphz = data['Zphz'].values
	elif type(data)==np.ndarray:
		Zmod = data[:,0]
		Zphz = data[:,1]
		
	Zreal = Zmod*np.cos(np.pi*Zphz/180)
	Zimag = Zmod*np.sin(np.pi*Zphz/180)
	
	return Zreal,Zimag

def lsv_resistance(df):
	"""
	Calculate resistance from LSV file
	Args:
		df: dataframe of LSV data
	"""
	return np.polyfit(df['Im'],df['Vf'],deg=1)[0]


def extract_eis_HFR(datadir,filter_func,ignore_files=[],**est_HFR_kw):
	"""
	Extract HFR from multiple EIS files and load into a DataFrame
	
	Args:
		datadir: data file directory
		filter_func: function to select files to load. Should return True when passed desired filenames. 
			Ex: filter_func = lambda file: file.split('_')[2].replace('.DTA','')=='TimedRampDown'
		ignore_files: list of filenames to ignore (exclude from loading even if they meet filter_func conditions)
		est_HFR_kw: kwargs for estimate_HFR()
	"""
	files = [file for file in os.listdir(datadir) if filter_func(file)==True]
	#sort files by time
	files = sorted(files,key=lambda x: get_timestamp(os.path.join(datadir,x)).timestamp())
	columns=['T_str','time','HFR','file']
	df = pd.DataFrame(columns=columns)
	
	for filename in files:
		if filename not in ignore_files:
			file = os.path.join(datadir,filename)
			temp = filename.split('_')[1]
			data = read_eis_zdata(file)
			time = get_timestamp(file).timestamp()
			HFR = estimate_HFR(data,**est_HFR_kw)
			df = df.append(pd.Series([temp,time,HFR,filename],index=columns),ignore_index=True)
	
	df['T'] = df['T_str'].str[:-1].astype(int)
	
	return df

def flag_eis_diffs(df,n=3,iqr_bound=5,cols=['Zmod','Zphz'],scaling='unity',show_plots=False,direction=['fwd','rev']):
	bad_idx = []
	if scaling in ('modulus','mixed'):
		# get frequency spacing
		dx = mode(np.log(df['Freq']).values[1:] - np.log(df['Freq']).values[:-1]).mode[0]
		# get variance of nth derivative
		var_n = binom(2*n,n)*df['Zmod'].values**2/(dx**(2*n))
		# weight is inverse std
		mod_weights = 1/var_n**0.5
		
	if type(direction)==str:
		direction = [direction]
	
	mod_df = df.copy()
	plot_col_idx = {}
	plot_col_diff = {}
	for col in cols:
		# do differences forward and in reverse to catch all points
		for drct in direction:#,'rev']:
			mod_df[f'{drct}_diff_{col}'] = np.nan
			# filter by nth derivative
			if drct=='fwd':
				diff = np.diff(df[col],n=n)/(np.diff(np.log(df['Freq']),n=1)[n-1:])**n
				if scaling=='unity':
					weights = np.ones_like(diff)
				elif scaling=='modulus':
					weights = mod_weights[n:] #1/(df['Zmod'].values[n:]
				elif scaling=='mixed':
					if col=='Zphz':
						weights = np.ones_like(diff)
					else:
						weights = mod_weights[n:]
						
				diff *= weights
				mod_df.loc[mod_df.index.min()+n:,f'{drct}_diff_{col}'] = diff
				plot_col_diff[(col,drct)] = diff
			else:
				diff = np.diff(df[col].values[::-1],n=n)/(np.diff(np.log(df['Freq'].values[::-1]),n=1)[n-1:])**n
				if scaling=='unity':
					weights = np.ones_like(diff)                      
				elif scaling=='modulus':
					weights = mod_weights[::-1][n:] #1/(df['Zmod'].values[::-1][n:]*np.exp(9.825*n))
				elif scaling=='mixed':
					if col=='Zphz':
						weights = np.ones_like(diff)
					else:
						weights = mod_weights[::-1][n:]
				diff *= weights
				mod_df.loc[:mod_df.index.max()-n,f'{drct}_diff_{col}'] = diff[::-1]
				plot_col_diff[(col,drct)] = diff[::-1]
			# get indexes of data outside bounds
			col_idx = (np.where(np.abs(diff-np.median(diff)) > iqr_bound*iqr(diff))[0]).astype(int) # + np.ceil(n/2)).astype(int)

			if drct=='rev':
				col_idx = len(df) - col_idx - 1
			
			# print(col,drct,sorted(col_idx))
			bad_idx += list(col_idx)
			
			plot_col_idx[(col,drct)] = col_idx.astype(int)
	
	bad_idx = np.unique(bad_idx).astype(int)
	
	
	mod_df['flag'] = 0
	if len(bad_idx) > 0:
		mod_df.iloc[bad_idx,-1] = 1
	
	if show_plots:
		fig, axes = plt.subplots(2,2,figsize=(10,8))
		axes[0,0].get_shared_x_axes().join(axes[0,0],axes[0,1],axes[1,0],axes[1,1])
		bad_df = mod_df[mod_df['flag']==1]
		good_df = mod_df[mod_df['flag']==0]
		unit_scale = get_unit_scale(mod_df)
		# plot nth diff vs. frequency
		for col, ax in zip(cols,axes[0]):
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
			
			for drct in direction:#,'rev']:
				diff = plot_col_diff[(col,drct)]
				col_idx = plot_col_idx[(col,drct)]
				if drct=='fwd':
					ax.scatter(df['Freq'][n:],diff,s=8,c='k',label='Fwd Diff')
					ax.scatter(df.iloc[col_idx+n,:]['Freq'],diff[col_idx],s=20,c='k',edgecolor='r',linewidth=0.8)
					ax.axhline(np.median(diff)-iqr_bound*iqr(diff),c='r',lw=1,label='Fwd Bound')
					ax.axhline(np.median(diff)+iqr_bound*iqr(diff),c='r',lw=1)
					# print(col,drct,iqr_bound*iqr(diff),diff[col_idx])
				elif drct=='rev':
					ax.scatter(df['Freq'][:-n],diff,s=8,c='gray',label='Rev Diff')
					# print(col_idx[::-1]-2, bad_df.index)
					# print(col,drct,np.median(diff),iqr_bound*iqr(diff),diff[col_idx-n])
					ax.scatter(df.iloc[col_idx-n,:]['Freq'],diff[col_idx-n],s=20,c='gray',edgecolor='r',linewidth=0.8)
					ax.axhline(np.median(diff)-iqr_bound*iqr(diff),c='r',lw=1,ls='--',label='Rev Bound')
					ax.axhline(np.median(diff)+iqr_bound*iqr(diff),c='r',lw=1,ls='--')
			ax.set_xscale('log')
			ax.set_title(col)
			ax.set_xlabel('Frequency')
			ax.set_ylabel('nth Discrete Difference')
			ax.legend(fontsize=9)
		
		# plot bode with flagged points in red
		plot_bode(good_df,axes=axes[1],cols=cols)
		plot_bode(bad_df,axes=axes[1],cols=cols,c='r')

		fig.tight_layout()
		
	return mod_df
	
	
def flag_eis_points(df,n=3,iqr_bound=5,cols=['Zmod','Zphz'],scaling='unity',fill_rolling_mean=False,show_plots=False,direction=['fwd','rev'],
					trim_method='direction',trim_offset=-2,axes=None,plot_kw={'s':8}): #,trim_start_consec_pts=5):
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
	if scaling in ('modulus','mixed'):
		# get frequency spacing
		dx = mode(np.log(df['Freq']).values[1:] - np.log(df['Freq']).values[:-1]).mode[0]
		# get variance of nth derivative
		var_n = binom(2*n,n)*df['Zmod'].values**2/(dx**(2*n))
		# weight is inverse std
		mod_weights = 1/var_n**0.5
		
	if type(direction)==str:
		direction = [direction]
	
	trim_options = ['direction','full','none']
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
			if drct=='fwd':
				diff = np.diff(df[col],n=n)/(np.diff(np.log(df['Freq']),n=1)[n-1:])**n
				if scaling=='unity':
					weights = np.ones_like(diff)
				elif scaling=='modulus':
					weights = mod_weights[n:] #1/(df['Zmod'].values[n:]
				elif scaling=='mixed':
					if col=='Zphz':
						weights = np.ones_like(diff)
					else:
						weights = mod_weights[n:]
						
				diff *= weights
				plot_col_diff[(col,drct)] = diff
			else:
				diff = np.diff(df[col].values[::-1],n=n)/(np.diff(np.log(df['Freq'].values[::-1]),n=1)[n-1:])**n
				if scaling=='unity':
					weights = np.ones_like(diff)                      
				elif scaling=='modulus':
					weights = mod_weights[::-1][n:] #1/(df['Zmod'].values[::-1][n:]*np.exp(9.825*n))
				elif scaling=='mixed':
					if col=='Zphz':
						weights = np.ones_like(diff)
					else:
						weights = mod_weights[::-1][n:]
				diff *= weights
				plot_col_diff[(col,drct)] = diff[::-1]
			# get indexes of data outside bounds
			# fluctuation in diff shows up 1 point after the errant point (subtract 1). Diff starts at nth point (add n). The errant point is at the diff index plus n minus 1
			col_idx = (np.where(np.abs(diff-np.median(diff)) > iqr_bound*iqr(diff))[0]).astype(int) # + n-1).astype(int)
			# for plotting, track the actual points that triggered the flags, not the flagged points. Align diff index with function index
			plot_idx = col_idx #- (n-1)
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
			if drct=='rev':
				col_idx = len(df) - col_idx - 1 + (-1) #+((n-1)-1)
				plot_idx = len(df) - plot_idx - 1
			
			# print(col,drct,sorted(col_idx))
			# concatenate all the flags determined in the same direction
			drct_idx += list(col_idx)
			
			plot_col_idx[(col,drct)] = plot_idx.astype(int)
		drct_idx = np.unique(drct_idx)
		
		if trim_method=='direction':
			if len(drct_idx) > 0:
				print('Pre-trim:',drct_idx)
				# a single bad point cascades to n points in the diff. Trim the ranges accordingly
				# do this after aggregating all flags in one direction to ensure that contiguous ranges of bad points are not lost by prematurely condensing ranges
				rng_end_idx = np.where(np.diff(drct_idx)!=1)[0]
				rng_start_idx = np.insert((rng_end_idx + 1),0,0)
				rng_end_idx = np.append(rng_end_idx,len(drct_idx)-1)
				# trim logic: trim the end unless it truncates the range to length zero.
				trimmed_ranges = [np.arange( drct_idx[start],
											 max(drct_idx[start]+1, drct_idx[end] - (trim_len-1))
											 )
										for start,end in zip(rng_start_idx,rng_end_idx)]
				# the very last points in the spectra should not be trimmed
				if drct_idx[-1]==len(df)-1:
					start = rng_start_idx[-1]
					end = rng_end_idx[-1]
					# reset the range end to the last point
					trimmed_ranges[-1] = np.arange(drct_idx[start],drct_idx[end])
				drct_idx = np.concatenate([r for r in trimmed_ranges])
				print('Post-trim:',drct_idx)
		bad_idx += list(drct_idx)
		
	bad_idx = np.unique(bad_idx).astype(int)
	
	if trim_method=='full':
		if len(bad_idx) > 0:
			print('Pre-trim:',bad_idx)
			# a single bad point cascades to n subsequent points in the diff. Condense (trim) the ranges accordingly
			# do this after aggregating all flags to ensure that contiguous ranges of bad points are not lost by prematurely condensing ranges
			rng_end_idx = np.where(np.diff(bad_idx)!=1)[0]
			rng_start_idx = np.insert((rng_end_idx + 1),0,0)
			rng_end_idx = np.append(rng_end_idx,len(bad_idx)-1)
			
			# trim logic: use the trim unless it truncates the range to length zero.
				# min(bad_idx[start] + trim, bad_idx[end]-trim): start at the smaller of start + trim and end - trim
				# max( bad_idx[start], <above>): if end-trim < start, just start at start (i.e. don't extend the range just because it's short)
				# same logic applies to end of range
			trimmed_ranges = [np.arange( max(bad_idx[start],min(bad_idx[start]+trim_len,bad_idx[end]-trim_len)),
										 min(bad_idx[end]+1,max(bad_idx[start]+trim_len+1,bad_idx[end] - (trim_len-1)))
										)
									for start,end in zip(rng_start_idx,rng_end_idx)]
			
			# the very first and very last points in the spectra should not be trimmed
			if bad_idx[0]==0:
				start = rng_start_idx[0]
				end = rng_end_idx[0]
				# reset the range start to point 0
				trimmed_ranges[0] = np.arange(bad_idx[start],min(bad_idx[end]+1,max(bad_idx[start]+trim_len+1,bad_idx[end] - (trim_len-1))))
			if bad_idx[-1]==len(df)-1:
				start = rng_start_idx[-1]
				end = rng_end_idx[-1]
				# reset the range end to the last point
				trimmed_ranges[-1] = np.arange(max(bad_idx[start],min(bad_idx[start]+trim_len,bad_idx[end]-trim_len)),bad_idx[rng_end_idx[-1]])
			bad_idx = np.concatenate([r for r in trimmed_ranges])
			print('Post-trim:',bad_idx)
		
	# if len(bad_idx) >= trim_start_consec_pts:
		# # if the first trim_start_consec_pts points are all flagged, also flag the first n-1 points
		# if np.sum(bad_idx[:trim_start_consec_pts] == np.arange(n-1, n-1+trim_start_consec_pts))==trim_start_consec_pts:
			# bad_idx = np.concatenate((np.arange(0,n-1),bad_idx))
		
	#print(bad_idx)
	
	mod_df = df.copy()
	mod_df['flag'] = 0
	if len(bad_idx) > 0:
		mod_df.iloc[bad_idx,-1] = 1
	
	if fill_rolling_mean:
		# get rolling mean
		ma = df.rolling(5,center=True).mean()
		mod_df['Zmod_filled'] = mod_df['Zmod']
		mod_df['Zphz_filled'] = mod_df['Zphz']
		bad_df = mod_df[mod_df['flag']==1]
		for col in cols:
			# interpolate rolling mean to fill bad data points
			mod_df.loc[bad_idx,col + '_filled'] = bad_df.apply(lambda r: np.interp(r['Freq'],ma['Freq'][::-1],ma[col][::-1]), axis=1)
		mod_df['Zreal_filled'] = mod_df['Zmod_filled']*np.cos(2*np.pi*mod_df['Zphz_filled']/360)
		mod_df['Zimag_filled'] = mod_df['Zmod_filled']*np.sin(2*np.pi*mod_df['Zphz_filled']/360)
	
	if show_plots:
		if axes is None:
			fig, axes = plt.subplots(3,2,figsize=(10,10))
		else:
			fig = axes.ravel()[0].get_figure()
		axes[0,0].get_shared_x_axes().join(axes[0,0],axes[0,1],axes[1,0],axes[1,1])
		bad_df = mod_df[mod_df['flag']==1]
		good_df = mod_df[mod_df['flag']==0]
		unit_scale = get_unit_scale(mod_df)
		# plot nth diff vs. frequency
		for col, ax in zip(cols,axes[0]):
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
			
			for drct in direction:#,'rev']:
				diff = plot_col_diff[(col,drct)]
				col_idx = plot_col_idx[(col,drct)]
				if drct=='fwd':
					ax.scatter(df['Freq'][n:],diff,c='k',label='Fwd Diff',**plot_kw)
					ax.scatter(df.iloc[col_idx+n,:]['Freq'],diff[col_idx],c='k',edgecolor='r',linewidth=0.8,**plot_kw)
					ax.axhline(np.median(diff)-iqr_bound*iqr(diff),c='r',lw=1,label='Fwd Bound')
					ax.axhline(np.median(diff)+iqr_bound*iqr(diff),c='r',lw=1)
					# print(col,drct,iqr_bound*iqr(diff),diff[col_idx])
				elif drct=='rev':
					ax.scatter(df['Freq'][:-n],diff,c='gray',label='Rev Diff',**plot_kw)
					# print(col_idx[::-1]-2, bad_df.index)
					# print(col,drct,np.median(diff),iqr_bound*iqr(diff),diff[col_idx-n])
					ax.scatter(df.iloc[col_idx-n,:]['Freq'],diff[col_idx-n],c='gray',edgecolor='r',linewidth=0.8,**plot_kw)
					ax.axhline(np.median(diff)-iqr_bound*iqr(diff),c='r',lw=1,ls='--',label='Rev Bound')
					ax.axhline(np.median(diff)+iqr_bound*iqr(diff),c='r',lw=1,ls='--')
			ax.set_xscale('log')
			ax.set_title(col)
			ax.set_xlabel('Frequency')
			ax.set_ylabel('nth Discrete Difference')
			ax.legend(fontsize=9)
		
		# plot bode with flagged points in red
		plot_bode(good_df,axes=axes[1],cols=cols,**plot_kw)
		plot_bode(bad_df,axes=axes[1],cols=cols,c='r',**plot_kw)
		if fill_rolling_mean:
			# plot interpolated points
			fdf = mod_df.copy()
			for col in ['Zreal','Zimag','Zmod','Zphz']:
				fdf[col] = fdf[col + '_filled']
			fdf = fdf.loc[bad_idx,:]
			plot_bode(fdf,axes=axes[1],c='g')
		# plot nyquist with flagged points in red
		plot_nyquist(good_df,ax=axes[2,0],unit_scale=unit_scale,**plot_kw)
		plot_nyquist(bad_df,ax=axes[2,0],c='r',unit_scale=unit_scale,**plot_kw)
		if fill_rolling_mean:
			plot_nyquist(fdf,ax=axes[2,0],c='g',unit_scale=unit_scale,**plot_kw)
		axes[2,0].set_title('Nyquist')
			
		axes[2,1].axis('off')
		fig.tight_layout()
		
	return mod_df

	
def calc_sigma(R,d,t,units='mm'):
	"""
	Calculate conductivity in S/cm given resistance and cell dimensions
	Assumes button cell geometry
	
	Args:
		R: resistance (ohm)
		d: diameter 
		t: thickness 
		units: units for d and t. Default mm. Options: 'mm', 'cm'
	"""
	#convert to cm
	if units=='mm':
		d,t = d/10, t/10
	elif units!='cm':
		raise ValueError(f'Units arg {units} not recognized. Valid units are ''mm'',''cm''')
	a = np.pi*(d/2)**2
	sigma = t/(R*a)
	return sigma
	

def aggregate_prop(df,by,property_col,aggregate):
	grp_df = df.groupby(by)
	if aggregate=='end':
		prop = np.array([gdf.loc[gdf['time'].idxmax(),property_col] for name, gdf in grp_df])
	elif aggregate=='start':
		prop = np.array([gdf.loc[gdf['time'].idxmin(),property_col] for name, gdf in grp_df])
	else:
		prop = getattr(grp_df[property_col],aggregate)().values
	return prop
	
def calc_G_act(df,property_col,aggregate,return_fit=False):
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
	prop_agg = aggregate_prop(df,'T',property_col,aggregate)
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
			
	T_inv = 1/(273+temps)
	#print(sigma_agg)
	fit = np.polyfit(T_inv,np.log(prop_agg),deg=1)
	k_B = 8.617e-5 #eV/K
	
	if return_fit==True:
		return -k_B*fit[0], fit
	else:
		return -k_B*fit[0]
		
def get_unit_scale(df):
	""" Get unit scale (mu, m, k, M, G) for EIS data"""
	unit_map = {-2:'$\mu$',-1:'m',0:'',1:'k',2:'M',3:'G'}
	Z_max = max(df['Zreal'].max(),df['Zimag'].abs().max())
	Z_ord = np.floor(np.log10(Z_max)/3)
	unit_scale = unit_map.get(Z_ord,'')
	return unit_scale
	
def get_scale_factor(df):
	Z_max = max(df['Zreal'].max(),df['Zimag'].abs().max())
	Z_ord = np.floor(np.log10(Z_max)/3)
	return 10**(3*Z_ord)
	
def get_common_unit_scale(df_list,aggregate='min'):
	"""
	Get common unit scale for multiple datasets
	Parameters:
		df_list: list of DataFrames
		aggregate: method for choosing common scale. Defaults to min (smallest scale) 
	"""
	unit_map = {-2:'$\mu$',-1:'m',0:'',1:'k',2:'M',3:'G'}
	rev_map = {v:k for k,v in unit_map.items()}
	units = [get_unit_scale(df) for df in df_list]
	unit_nums = [rev_map[u] for u in units]
	common_num = getattr(np,aggregate)(unit_nums)
	common_unit = unit_map.get(common_num,'')
	return common_unit
	
#---------------------------------
# Plotting
#---------------------------------		
def plot_ocv(datadir, filter_func=None, files=None, ax=None, invert='auto', same_color=True,**plt_kw):
	#get files
	if filter_func is None and files is None:
		#if no filter or files specified, get all OCV files
		filter_func = lambda x: x[0:3]=='OCV' and x[-3:]=='DTA'
		files = [f for f in os.listdir(datadir) if filter_func(f)]
	elif files and not filter_func:
		if type(files)==str:
			#if single file specified, convert to 1-element list
			files = [files]
	elif filter_func and not files:
		files = [f for f in os.listdir(datadir) if filter_func(f)]
	elif filter_func and files:
		raise ValueError('Both filter_func and files have been specified. Please specify only one')
		
	dfs = [read_ocv_data(os.path.join(datadir,file)) for file in files]
	dfs = [df for df in dfs if len(df) > 0]
	start_times = [df['timestamp'][0] for df in dfs]
	start_time = min(start_times)
	
	ts_func = lambda ts: (ts - start_time).dt.total_seconds()/3600

	if ax is None:
		fig, ax = plt.subplots()
	
	if invert=='auto':
		# choose sign based on max voltage
		tdf = pd.concat(dfs,ignore_index=True)
		V_sign = np.sign(tdf.loc[tdf['Vf'].abs().idxmax(),'Vf'])
	elif invert==True:
		V_sign = -1
	elif invert==False:
		V_sign = 1
	
	for df in dfs:
		if 'c' not in plt_kw and 'color' not in plt_kw and same_color==True:
			#if no color specified and same color desired, set color to first default color
			default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
			plt_kw['c'] = default_colors[0]
			
		ax.plot(ts_func(df['timestamp']),V_sign*df['Vf'],**plt_kw)

	ax.set_xlabel('Time (h)')
	ax.set_ylabel('OCV (V)')
	
def plot_jv(df,area=None,plot_pwr=False,ax=None,pwr_kw={'ls':'--'},**plt_kw):
	if ax is None:
		fig, ax = plt.subplots()
	if area is not None:
		# if area given, convert to densities
		df = df.copy()
		df['Im'] /= area
		df['Pwr'] /= area
		
	ax.plot(1000*df['Im'].abs(),df['Vf'].abs(),**plt_kw)
	if area is None:
		ax.set_xlabel('Current (mA)')
	else:
		ax.set_xlabel('Current Density (mA/cm$^2$)')
	ax.set_ylabel('Voltage (V)')
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
		
		ax2.plot(1000*df['Im'].abs(),1000*df['Pwr'].abs().values,**pwr_kw)
		
		if area is None:
			ax2.set_ylabel('Power (mW)')
		else:
			ax2.set_ylabel('Power Density (mW/cm$^2$)')
	return ax
	
def plot_nyquist(df,area=None,ax=None,label='',plot_func='scatter',unit_scale='auto',label_size=10,eq_xy=True,**kw):
	"""
	Nyquist plot
	
	Args:
		df: dataframe of impedance data
		area: cell area in cm2. If None, plot raw impedance
		label: series label
		plot_func: pyplot plotting function. Options: 'scatter', 'plot'
		eq_xy: if True, ensure that scale of x and y axes is the same (i.e. xmax-xmin = ymax-ymin)
		kw: kwargs for plot_func
	"""
	df = df.copy()
	
	if ax is None:
		fig, ax = plt.subplots()
		
	if area is not None:
		# if area given, convert to ASR
		df['Zreal'] *= area
		df['Zimag'] *= area
	
	# get/set unit scale
	unit_map = {-2:'$\mu$',-1:'m',0:'',1:'k',2:'M',3:'G'}
	if unit_scale=='auto':
		unit_scale = get_unit_scale(df)
		Z_ord = [k for k,v in unit_map.items() if v==unit_scale][0]
	elif unit_scale is None:
		unit_scale=''
		Z_ord = 0
	else:
		Z_ord = [k for k,v in unit_map.items() if v==unit_scale][0]
	
	# scale data	
	df['Zreal'] /= 10**(Z_ord*3)
	df['Zimag'] /= 10**(Z_ord*3)
		
	if plot_func=='scatter':
		if 's' not in kw.keys():
			# default point size
			kw['s'] = 8
		ax.scatter(df['Zreal'],-df['Zimag'],label=label,**kw)
	elif plot_func=='plot':
		ax.plot(df['Zreal'],-df['Zimag'],label=label,**kw)
	else:
		raise ValueError(f'Invalid plot type {plot_func}. Options are scatter, plot')
	
	if area is not None:
		ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega\cdot \mathrm{{cm}}^2$',size=label_size)
		ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega\cdot \mathrm{{cm}}^2$',size=label_size)
	else:
		ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$',size=label_size)
		ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$',size=label_size)
		
	if label!='':
		ax.legend()
			
	if eq_xy:
		# make scale of x and y axes the same
		fig = ax.get_figure()
		
		# get data range
		yrng = ax.get_ylim()[1] - ax.get_ylim()[0]
		xrng = ax.get_xlim()[1] - ax.get_xlim()[0]
		
		# get axis dimensions
		bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
		width, height = bbox.width, bbox.height
		width, height
		
		yscale = yrng/height
		xscale = xrng/width
		
		if yscale > xscale:
			# expand the x axis 
			diff = (yscale - xscale)*width
			xmin = max(0,ax.get_xlim()[0] - diff/2)
			mindelta = ax.get_xlim()[0] - xmin
			xmax = ax.get_xlim()[1] + diff - mindelta
			
			ax.set_xlim(xmin,xmax)
		elif xscale > yscale:
			# expand the y axis
			diff = (xscale - yscale)*height
			if np.min(-df['Zimag']) >= 0:
				# if -Zimag doesn't go negative, don't go negative on y-axis
				ymin = max(0,ax.get_ylim()[0] - diff/2)
				mindelta = ax.get_ylim()[0] - ymin
				ymax = ax.get_ylim()[1] + diff - mindelta
			else:
				negrng = abs(ax.get_ylim()[0])
				posrng = abs(ax.get_ylim()[1])
				negoffset = negrng*diff/(negrng + posrng)
				posoffset = posrng*diff/(negrng + posrng)
				ymin = ax.get_ylim()[0] - negoffset
				ymax = ax.get_ylim()[1] + posoffset
			
			ax.set_ylim(ymin,ymax)
	
	return ax

def plot_bode(df,area=None,axes=None,label='',plot_func='scatter',cols=['Zmod','Zphz'],unit_scale='auto',invert_Zimag=True,**kw):
	"""
	Bode plot
	
	Args:
		df: dataframe of impedance data
		area: cell area in cm2. If None, plot raw impedance
		label: series label
		plot_func: pyplot plotting function. Options: 'scatter', 'plot'
		cols: which columns to plot vs. frequency. Defaults to Zmod and Zphz
		unit_scale: impedance scale. Options are 'auto', 'mu', 'm', '', 'k', 'M'
		kw: kwargs for plot_func
	"""
	df = df.copy()
	# formatting for columns
	col_dict = {'Zmod':{'units':'$\Omega$','label':'$Z_{\mathrm{mod}}$','scale':'log'},
				'Zphz':{'units':'$^\circ$','label':'$Z_{\mathrm{phz}}$','scale':'linear'},
				'Zreal':{'units':'$\Omega$','label':'$Z^\prime$','scale':'linear'},
				'Zimag':{'units':'$\Omega$','label':'$Z^{\prime\prime}$','scale':'linear'}
				}
	
	if axes is None:
		fig, axes = plt.subplots(1,2,figsize=(8,3))
	else:
		fig = axes[0].get_figure()
	ax1,ax2 = axes
	
	if area is not None:
		for col in ['Zreal','Zimag','Zmod']:
			if col in df.columns:
				df[col] *= area
		
	# get/set unit scale	
	unit_map = {-2:'$\mu$',-1:'m',0:'',1:'k',2:'M',3:'G'}
	if unit_scale=='auto':
		unit_scale = get_unit_scale(df)
		Z_ord = [k for k,v in unit_map.items() if v==unit_scale][0]
	elif unit_scale is None:
		unit_scale=''
		Z_ord = 0
	else:
		Z_ord = [k for k,v in unit_map.items() if v==unit_scale][0]
	
	# scale data
	for col in ['Zreal','Zimag','Zmod']:
		if col in df.columns:
			df[col] /= 10**(Z_ord*3)
	
	if invert_Zimag:
		df['Zimag'] *= -1
	
	
	if plot_func=='scatter':
		if 's' not in kw.keys():
			# default point size
			kw['s'] = 8
		ax1.scatter(df['Freq'],df[cols[0]],label=label,**kw)
		ax2.scatter(df['Freq'],df[cols[1]],label=label,**kw)
	elif plot_func=='plot':
		ax1.plot(df['Freq'],df[cols[0]],label=label,**kw)
		ax2.plot(df['Freq'],df[cols[1]],label=label,**kw)
	else:
		raise ValueError(f'Invalid plot type {plot_func}. Options are scatter, plot')
	
	for ax in axes:
		ax.set_xlabel('$f$ / Hz')
		ax.set_xscale('log')
		
	def ax_title(col,area):
		cdict = col_dict.get(col,{})
		if area is not None and cdict.get('units','')=='$\Omega$':
			title = '{} / {}{}$\cdot\mathrm{{cm}}^2$'.format(cdict.get('label',col),unit_scale,cdict.get('units',''))
		elif cdict.get('units','')=='$\Omega$':
			title = '{} / {}{}'.format(cdict.get('label',col),unit_scale,cdict.get('units',''))
		else:
			title = '{} / {}'.format(cdict.get('label',col),cdict.get('units','a.u.'))
			
		if col=='Zimag' and invert_Zimag:
			title = '$-$' + title
		return title
	
	for col, ax in zip(cols,axes):
		ax.set_ylabel(ax_title(col,area))
		ax.set_yscale(col_dict.get(col,{}).get('scale','linear'))
		
	for ax in axes:
		# manually set x axis limits - sometimes matplotlib doesn't get them right
		fmin = df['Freq'].min()
		fmax = df['Freq'].max()
		ax.set_xlim(fmin/5,fmax*5)
	
	
	# if area is not None:
		# ax1.set_ylabel('$Z_{\mathrm{mod}} \ (\Omega\cdot \mathrm{cm}^2)$')
	# else:
		# ax1.set_ylabel('$Z_{\mathrm{mod}} \ (\Omega)$')
	
	# ax1.set_yscale('log')
	# ax2.set_ylabel('$Z_{\mathrm{phz}} \ (^\circ)$')
	
	fig.tight_layout()
	
	return axes
	
def plot_full_eis(df,area=None,axes=None,label='',plot_func='scatter',unit_scale='auto',bode_cols=['Zmod','Zphz'],eq_xy=True,**kw):
	if axes is None:
		# fig = plt.figure(figsize=(8,8))
		# ax1 = plt.subplot2grid((2,2),(0,0),colspan=2)
		# ax2 = plt.subplot2grid((2,2),(1,0))
		# ax3 = plt.subplot2grid((2,2),(1,1))
		# axes = np.array([ax1,ax2,ax3])
		fig,axes = plt.subplots(1,3,figsize=(9,2.5))
		ax1,ax2,ax3 = axes.ravel()
	else:
		ax1,ax2,ax3 = axes.ravel()
		fig = axes.ravel()[0].get_figure()
	
	#Nyquist plot
	plot_nyquist(df,area=area,label=label,ax=ax1,plot_func=plot_func,unit_scale=unit_scale,eq_xy=eq_xy,**kw)
	
	#Bode plots
	plot_bode(df,area=area,label=label,axes=(ax2,ax3),plot_func=plot_func,cols=bode_cols,unit_scale=unit_scale,**kw)
	
	fig.tight_layout()
	
	return axes
	
def compare_fits(data,gamry_params,py_params,model):
	"""
	Plot comparison of Nyquist and Bode plots for gamry fit and python fit
	"""
	w = data['Freq'].values
	y = data.loc[:,['Zreal','Zimag']].values
	weights = 1/(data['Zmod']).values
	
	#Nyquist plot
	fig, ax = plt.subplots(figsize=(8,6))
	ax.scatter(data['Zreal'],-data['Zimag'],s=6,label='Measured')

	Z_fc_gam = model(w,**gamry_params)
	ax.plot(Z_fc_gam.real,-Z_fc_gam.imag,'k',label='Gamry fit')

	Z_fc_py = model(w,**py_params)
	ax.plot(Z_fc_py.real,-Z_fc_py.imag,'r',label='Python fit')

	ax.legend()
	ax.set_xlabel('$Z_{real}$ ($\Omega \cdot$cm$^2$)')
	ax.set_ylabel('$-Z_{imag}$ ($\Omega \cdot$cm$^2$)')

	#Bode plots
	fig2, (ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
	ax1.semilogx(w,data['Zreal'],'.',label='Measured')
	ax1.semilogx(w,Z_fc_gam.real,'k',label='Gamry fit')
	ax1.semilogx(w,Z_fc_py.real,'r',label='Python fit')
	ax1.set_title('Real')
	ax1.set_xlabel('Frequency (Hz)')
	ax1.set_ylabel('$Z_{real}$')
	
	ax2.semilogx(w,-data['Zimag'],'.',label='Measured')
	ax2.semilogx(w,-Z_fc_gam.imag,'k',label='Gamry fit')
	ax2.semilogx(w,-Z_fc_py.imag,'r',label='Python fit')
	ax2.set_title('Imag')
	ax2.set_xlabel('Frequency (Hz)')
	ax2.set_ylabel('-$Z_{imag}$')
	fig2.tight_layout()
	
def plot_model(model,params,w=None,area=None,plot_type='all',plot_func='plot',axes=None,label='',unit_scale='auto',mark_peaks=False,c=None,**kw):
	if w is None:
		w = np.logspace(-2,6)
	Z = np.array(model(w,**params))
	
	data = pd.DataFrame(np.array([w,Z.real,Z.imag]).T,columns=['Freq','Zreal','Zimag'])
	data['Zmod'] = (Z*Z.conjugate())**0.5
	data['Zphz'] = (180/np.pi)*np.arctan(Z.imag/Z.real)
	
	# if area is not None:
		# data['Zreal']*=area
		# data['Zimag']*=area
		# data['Zmod']*=area
	
	if plot_type=='nyquist':
		axes = plot_nyquist(data,ax=axes,label=label,plot_func=plot_func,area=area,unit_scale=unit_scale,c=c,**kw)
	elif plot_type=='bode':
		axes = plot_bode(data,axes=axes,label=label,plot_func=plot_func,area=area,c=c,**kw)
	elif plot_type=='all':
		if axes is None:
			fig, axes = plt.subplots(1,3,figsize=(12,4))
		plot_full_eis(data,axes=axes,label=label,plot_func=plot_func,area=area,unit_scale=unit_scale,c=c,**kw)
		# plot_nyquist(data,ax=axes.ravel()[0],label=label,plot_func=plot_func,area=area,unit_scale=unit_scale,c=c,**kw)
		# plot_bode(data,axes=axes.ravel()[1:],label=label,plot_func=plot_func,area=area,c=c,**kw)
	else:
		raise ValueError(f'Invalid plot type {plot_type}. Options are nyquist, bode, all')	
	if label != '':
		if type(axes) in (tuple,list):
			for tax in axes:
				tax.legend()
		elif type(axes)==np.ndarray:
			for tax in axes.ravel():
				tax.legend()
		else:
			axes.legend()
			
	# plot model peak frequencies
	if mark_peaks:
		f_peaks = np.array(var_RC_peak_frequencies(params))
		Z_peaks = model(f_peaks,**params)
		peak_df = construct_eis_df(f_peaks,Z_peaks)
		if plot_type=='nyquist':
			plot_nyquist(peak_df,ax=axes,marker='x',s=50,unit_scale=unit_scale,area=area,c=c)
		elif plot_type=='bode':
			plot_bode(peak_df,axes=axes,marker='x',s=50,area=area,c=c)
		elif plot_type=='all':
			plot_nyquist(peak_df,ax=axes.ravel()[0],marker='x',s=50,unit_scale=unit_scale,area=area,c=c)
			plot_bode(peak_df,axes=axes.ravel()[1:],marker='x',s=50,area=area,c=c)
	return axes
			
def mark_model_peaks(params,model,area=None,plot_type='all',axes=None,label='',marker='x',s=50,c='r',unit_scale='auto',**kw):
	"""
	Mark peak RQ frequencies on Nyquist and/or Bode plots
	
	Parameters:
		params: dict of EC model parameters
		model: EC model
		area: cell area
		plot_type: which type of plot(s) to generate. Options: ''nyquist'', ''bode'', ''all''
		axes: axis or axes on which to plot
		label: legend label for peak markers
		marker: marker type
		s: marker size
		c: marker color
		unit_scale: unit scale for Nyquist plot
		kw: kwargs to pass to plt.scatter
	"""
	f_peaks = np.array(var_RC_peak_frequencies(params))
	Z_peaks = model(f_peaks,**params)
	peak_df = construct_eis_df(f_peaks,Z_peaks)
	
	if plot_type=='nyquist':
		axes = plot_nyquist(peak_df,ax=axes,label=label,area=area,unit_scale=unit_scale,marker=marker,s=s,c=c,**kw)
	elif plot_type=='bode':
		axes = plot_bode(peak_df,axes=axes,label=label,area=area,marker=marker,s=s,c=c,**kw)
	elif plot_type=='all':
		if axes is None:
			fig, axes = plt.subplots(1,3,figsize=(12,4))
		plot_nyquist(peak_df,ax=axes.ravel()[0],c='r',marker='x',s=50,label=label,unit_scale=unit_scale,area=area)
		plot_bode(peak_df,axes=axes.ravel()[1:],c='r',marker='x',s=50,label=label,area=area)
	else:
		raise ValueError(f'Invalid plot type {plot_type}. Options are nyquist, bode, all')	
	if label != '':
		if type(axes) in (tuple,list):
			for tax in axes:
				tax.legend()
		elif type(axes)==np.ndarray:
			for tax in axes.ravel():
				tax.legend()
		else:
			axes.legend()
	
def plot_fit(data,params,model,f_model=None,axes=None,unit_scale='auto',area=None,bode_cols=['Zmod','Zphz'],mark_peaks=False,fit_color='k',fit_kw={},**data_kw):
	w = data['Freq'].values
	if f_model is None:
		f_model = w
	elif f_model is 'fill':
		f_model = np.logspace(np.log10(np.min(w)),np.log10(np.max(w)), 100)
	y = data.loc[:,['Zreal','Zimag']].values
	weights = 1/(data['Zmod']).values
	
	if axes is None:
		fig,axes = plt.subplots(1,3,figsize=(9,3))
		ax1,ax2,ax3 = axes 
		# fig = plt.figure(figsize=(8,8))
		# ax1 = plt.subplot2grid((2,2),(0,0),colspan=2)
		# ax2 = plt.subplot2grid((2,2),(1,0))
		# ax3 = plt.subplot2grid((2,2),(1,1))
		# axes = np.array([ax1,ax2,ax3])
	else:
		ax1,ax2,ax3 = axes.ravel()
		fig = axes.ravel()[0].get_figure()
	
	Z_fit = model(f_model,**params)
	fit_df = pd.DataFrame(np.array([Z_fit.real,Z_fit.imag]).T,columns=['Zreal','Zimag'])
	fit_df['Freq'] = f_model
	fit_df['Zmod'] = (Z_fit*Z_fit.conjugate())**0.5
	fit_df['Zphz'] = (180/np.pi)*np.arctan(Z_fit.imag/Z_fit.real)
	
	if unit_scale=='auto':
		unit_scale = get_unit_scale(data)
	
	#Nyquist plot
	#ax.scatter(data['Zreal'],-data['Zimag'],s=6,label='Measured')
	plot_nyquist(data,label='Measured',ax=ax1,unit_scale=unit_scale,area=area,**data_kw)
	plot_nyquist(fit_df,c=fit_color,ax=ax1,label='Fit',plot_func='plot',unit_scale=unit_scale,area=area,**fit_kw)
	#ax1.plot(Z_fc.real,-Z_fc.imag,'k',label='Fit')

	ax1.legend()
	
	#Bode plots
	plot_bode(data,axes=(ax2,ax3),label='Measured',area=area,cols=bode_cols,unit_scale=unit_scale,**data_kw)
	plot_bode(fit_df,axes=(ax2,ax3),label='Fit',c=fit_color,plot_func='plot',area=area,cols=bode_cols,unit_scale=unit_scale,**fit_kw)
	
	# plot model peak frequencies
	if mark_peaks:
		f_peaks = np.array(var_RC_peak_frequencies(params))
		Z_peaks = model(f_peaks,**params)
		peak_df = construct_eis_df(f_peaks,Z_peaks)
		plot_nyquist(peak_df,ax=ax1,c=fit_color,marker='x',s=50,label='RQ Peak Frequencies',unit_scale=unit_scale,area=area)
		plot_bode(peak_df,axes=(ax2,ax3),c=fit_color,marker='x',s=50,label='RQ Peak Frequencies',area=area,cols=bode_cols)
	
	ax2.legend()
	ax3.legend()
	
	for ax in [ax2,ax3]:
		# manually set x axis limits - sometimes matplotlib doesn't get them right
		fmin = min(data['Freq'].min(),np.min(f_model))
		fmax = max(data['Freq'].max(),np.max(f_model))
		ax.set_xlim(fmin/5,fmax*5)
		
	fig.tight_layout()
	
	return axes
	
def jv_multiplot(files,color_by='T',aggregate=False,file_sequence=['file_type','T','aflow','cflow'],area=None,ax=None,label_color=None):
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
	markers = ['.','^','v','s','D','*']
	
	#temps = [int(os.path.basename(file).split('_')[1].replace('C','')) for file in files]
	times = [get_timestamp(f) for f in files]
	infos = [get_file_info(file,file_sequence) for file in files]
	
	jvdf = pd.DataFrame(np.array([files,times]).T,columns=['file','time'])
	jvdf = jvdf.join(pd.DataFrame(infos))
	
	if label_color is None:
		label_color = dict(zip(jvdf[color_by].unique(),default_colors))
	if color_by=='T':
		label_units = '$^\circ$C'
	else:
		label_units = ''
	
	for label, gdf in jvdf.groupby(color_by):
		if len(gdf) > 1:
			if aggregate==False:
				gdf = gdf.sort_values('time')
				gdf.index = np.arange(len(gdf))
				for i, row in gdf.iterrows():
					df = read_jv_data(row['file'])
					plot_jv(df,area=area,plot_pwr=True,ax=ax,label='{}{} ({})'.format(label,label_units,i+1),
							   marker=markers[i],markersize=5,c=label_color[label],
							   pwr_kw=dict(ls=':',marker=markers[i],markersize=3,c=label_color[label]))
			else:
				if aggregate=='max_pwr':
					idx = gdf['file'].map(lambda x: read_jv_data(x)['Pwr'].abs().max()).idxmax()
				elif aggregate=='min_pwr':
					idx = gdf['file'].map(lambda x: read_jv_data(x)['Pwr'].abs().max()).idxmin()
				elif aggregate=='first':
					idx = gdf['file'].map(lambda x: get_timestamp(x)).idxmin()
				elif aggregate=='last':
					idx = gdf['file'].map(lambda x: get_timestamp(x)).idxmax()
				else:
					raise ValueError(f'Invalid aggregate method {aggregate} specified')
				
				df = read_jv_data(gdf.loc[idx,'file'])
				plot_jv(df,area=area,plot_pwr=True,ax=ax,label=f'{label}{label_units}',marker=markers[0],markersize=5,c=label_color[label],
					  pwr_kw=dict(ls=':',marker=markers[0],markersize=5,c=label_color[label]))
				
		else:
			df = read_jv_data(gdf['file'].min())
			plot_jv(df,area=area,plot_pwr=True,ax=ax,label=f'{label}{label_units}',marker=markers[0],markersize=5,c=label_color[label],
					  pwr_kw=dict(ls=':',marker=markers[0],markersize=5,c=label_color[label]))
						
	
def generate_plots(plot_types,datadir=None,savefigs=False,savedir='./plots',area=None,
				   ocv_kw={},nyquist_kw={},bode_kw={},jv_kw={}):
	"""
	Generate basic plots for files in directory
	
	Args:
		plot_types: list of plot types to generate. Options: 'nyquist','bode','ocv','jv'
		datadir: data directory
		savefigs: if True, save generated figures
		area: cell area (cm2)
		plot_kw:
	"""
	#allowed_plot_types = ['nyquist','bode','ocv','jv']#,'nyquist-bode'

	if datadir is None:
		datadir = os.getcwd()
		
	plotdir = os.path.join(datadir,savedir)
	
	if not os.path.exists(plotdir):
		os.mkdir(plotdir)
		
	# get cell info from datadir
	cell = get_cell_name(datadir)
				
	# set kw defaults and update with any user-specified params
	# so that user doesn't have to re-enter all defaults to change one thing
	jv_default={'plot_pwr':True,'marker':'.','pwr_kw':{'ls':':','marker':'.'}}
	jv_default.update(jv_kw)
	jv_kw = jv_default
	
	ocv_default = {'filter_func':lambda x: x[0:3] in ('OCV','EIS') and x[-3:]=='DTA'}
	ocv_default.update(ocv_kw)
	ocv_kw = ocv_default
	
	for plot_type in plot_types:
		if plot_type in ['nyquist','bode']:
			start_str = 'EIS'
		elif plot_type=='jv':
			start_str = 'PWRPOLARIZATION'
		
		if plot_type=='ocv':
			plot_ocv(datadir,**ocv_kw)
			plt.title(cell,wrap=True)
			if savefigs is True:
				fig = plt.gcf()
				fig.savefig(os.path.join(plotdir,'OCV_plot.png'),dpi=500)
		else:
			files = glob.glob(os.path.join(datadir,start_str + '*.DTA'))
			
			for file in files:
				info = get_file_info(file)
				if plot_type=='nyquist':
					df = read_eis_zdata(file)
					ax = plot_nyquist(df,area=area,**nyquist_kw)
					ax.text(0.97,0.05,'{}$^\circ$C, {}, {}'.format(info['T'],info['aflow'],info['cflow']),transform=ax.transAxes,ha='right')
				elif plot_type=='bode':
					df = read_eis_zdata(file)
					axes = plot_bode(df,area=area,**bode_kw)
					ax = axes[0]
					ax.text(0.03,0.9,'{}$^\circ$C, {}, {}'.format(info['T'],info['aflow'],info['cflow']),transform=ax.transAxes)
				elif plot_type=='jv':
					df = read_jv_data(file)
					ax = plot_jv(df,area=area,**jv_kw)
					ax.text(0.5,0.05,'{}$^\circ$C, {}, {}'.format(info['T'],info['aflow'],info['cflow']),transform=ax.transAxes,ha='center')
				
				ax.set_title(cell,wrap=True)
				
				if savefigs is True:
					fig = plt.gcf()
					fname = os.path.basename(file)
					fig.savefig(os.path.join(plotdir,fname.replace('DTA','png')),dpi=500)
					
def plot_eis_prop(eis_df,property_col,label='',aggregate=['start','end','max','mean'],ax=None,**plt_kw):
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
		prop_agg[agg] = aggregate_prop(eis_df,'T',property_col,agg) #np.zeros(len(grp_df))

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
				
	T_inv = 1000/(273+temps)
	for agg, prop in prop_agg.items():
		if label in('',None):
			lab_prefix = ''
		else:
			lab_prefix = label + ' '
		ax.semilogy(T_inv,prop,label=lab_prefix+agg,**plt_kw)
	if label is not None:
		ax.legend()

	#label untransformed T on top axis
	if ax is not None:
		#get twin ax if already exists
		for other_ax in ax.figure.axes:
			if other_ax is ax:
				ax2 = None
			elif other_ax.bbox.bounds == ax.bbox.bounds:
				ax2 = other_ax
				break
			else:
				ax2 = None
	if ax2 is None:
		#create twin ax
		ax2 = ax.twiny()
	#set to same scale
	ax2.set_xlim(ax.get_xlim())
	ax2.set_xticks(T_inv)
	ax2.set_xticklabels(temps.astype(int))
	
	ax.set_xlabel(r'$1000/T \ (\mathrm{K}^{-1})$')
	ax2.set_xlabel(r'$T \ (^\circ \mathrm{C})$')
	#ax.set_ylabel(r'$\sigma$ (S/cm)')
	
	fig.tight_layout()
	
	return ax
	
def plot_arrhenius_fit(df,property_col,aggregate,ax=None,**plt_kw):
	G_act, fit = calc_G_act(df,property_col,aggregate,return_fit=True)
	if ax is None:
		fig, ax = plt.subplots()
		
	temps = np.unique(df['T'])
	T_inv = 1/(temps + 273)
	y_fit = np.exp(np.polyval(fit,T_inv))
	ax.plot(1000*T_inv,y_fit,**plt_kw)
	
	if plt_kw.get('label','') != '':
		ax.legend()
	return ax
	
#---------------------------------	
# Equivalent circuit modeling
#---------------------------------
def Z_cpe(w,Q,n):
	"Impedance of CPE"
	#Q=Y_0, n=a in Gamry
	#Z = (1/(Q*w**n))*np.exp(-np.pi*n*1j/2)
	
	Z = 1/((1j*w*2*np.pi)**n*Q)
	#equiv to: Z = (1/(Q*w**n))*1j**(-n)
	return Z
	
def Z_C(w,C):
	return 1/(2*np.pi*1j*w*C)

def Z_L(w,L):
	"Impedance of inductor"
	return w*L*1j*2*np.pi

def Z_O(w,Y,B):
	"Impedance of O diffusion element (porous bounded Warburg)"
	return (1/(Y*(1j*w*2*np.pi)**0.5))*np.tanh(B*(1j*w*2*np.pi)**0.5)
	
def Z_fO(w,Y,t0,nf):
	"Impedance of fractal O diffusion element (fractal porous bounded Warburg)"
	return (1/(Y*(1j*t0*w*2*np.pi)**nf))*np.tanh((1j*w*t0*2*np.pi)**nf)
	
def Z_ger(w,Y,t0):
	"Gerischer impedance"
	return 1/(Y*np.sqrt(1 + 1j*2*np.pi*w*t0))
	
def Z_HN(w,Rct,t0,nu,beta):
	"Havriliak-Negami impedance"
	return Rct/(1+(1j*2*np.pi*w*t0)**nu)**beta
	
def Z_par(Z1,Z2):
	"parallel impedance"
	return 1/(1/Z1 + 1/Z2)

def Z_fuelcell(w,HFR,Rf_c,Yo_c,a_c,Rf_a,Yo_a,a_a,Lstray):
	Z_a = Z_par(Z_cpe(w,Yo_a,a_a), Rf_a)
	Z_c = Z_par(Z_cpe(w,Yo_c,a_c), Rf_c)
	return Z_L(w,Lstray) + HFR + Z_a + Z_c
	
def Z_var_num_RC(w,HFR,Lstray,**RC_params):
	"""
	Impedance of circuit with 1-n parallel RC circuits in series with a resistor (HFR) and an inductor (Lstray)
	Args:
		w: frequency (Hz)
		HFR: high-frequency resistance
		Lstray: inductance
		RC_params: parameters for each parallel RC circuit. keys: R, Q, n
	"""
	#Z_RC = [Z_par(Z_cpe(w,p['Q'],p['n']), p['R']) for p in RC_params]
	num_RC = int(len(RC_params)/3)
	
	Z_RC = [Z_par(Z_cpe(w,RC_params[f'Q{i}'],RC_params[f'n{i}']), RC_params[f'R{i}']) for i in range(num_RC)]
	
	return Z_L(w,Lstray) + HFR + np.sum(Z_RC,axis=0)
	
def Z_var_num_RC_RL(w,HFR,Lstray,R_L,**RC_params):
	"""
	Impedance of circuit with 1-n parallel RC circuits in series with a resistor (HFR) and parallel RL element (R_L, Lstray)
	Args:
		w: frequency (Hz)
		HFR: high-frequency resistance
		Lstray: inductance
		R_L: resistance of resistor in parallel with inductor
		RC_params: parameters for each parallel RC circuit. keys: R, Q, n
	"""
	#Z_RC = [Z_par(Z_cpe(w,p['Q'],p['n']), p['R']) for p in RC_params]
	num_RC = int(len(RC_params)/3)
	
	Z_RC = [Z_par(Z_cpe(w,RC_params[f'Q{i}'],RC_params[f'n{i}']), RC_params[f'R{i}']) for i in range(num_RC)]
	
	return Z_par(R_L,Z_L(w,Lstray)) + HFR + np.sum(Z_RC,axis=0)
	
def Z_var_num_RC_noL(w,HFR,**RC_params):
    """
    Impedance of circuit with 1-n parallel RC circuits in series with a resistor (HFR) only
    Args:
        w: frequency (Hz)
        HFR: high-frequency resistance
        Lstray: inductance
        RC_params: parameters for each parallel RC circuit. keys: R, Q, n
    """
    #Z_RC = [Z_par(Z_cpe(w,p['Q'],p['n']), p['R']) for p in RC_params]
    num_RC = int(len(RC_params)/3)

    Z_RC = [Z_par(Z_cpe(w,RC_params[f'Q{i}'],RC_params[f'n{i}']), RC_params[f'R{i}']) for i in range(num_RC)]

    return HFR + np.sum(Z_RC,axis=0)
	
	
def Z_var_num_RC_RL_LRC(w,HFR,Lstray,R_L,R_lrc,L_lrc,C_lrc,**RC_params):
	"""
	Impedance of circuit with 1-n parallel RC circuits in series with a resistor (HFR) and parallel RL element (R_L, Lstray) and a parallel LRC element (R_lrc and C_lrc in parallel with L_lrc)
	LRC circuit allows fitting of low-frequency curl that goes below x-axis and moves to left with decreasing frequency
	Args:
		w: frequency (Hz)
		HFR: high-frequency resistance
		Lstray: inductance
		R_L: resistance of resistor in parallel with inductor
		RC_params: parameters for each parallel RC circuit. keys: R, Q, n
	"""
	#Z_RC = [Z_par(Z_cpe(w,p['Q'],p['n']), p['R']) for p in RC_params]
	num_RC = int(len(RC_params)/3)

	Z_RC = [Z_par(Z_cpe(w,RC_params[f'Q{i}'],RC_params[f'n{i}']), RC_params[f'R{i}']) for i in range(num_RC)]

	return Z_par(R_L,Z_L(w,Lstray)) + Z_par(Z_L(w,L_lrc), Z_C(w,C_lrc) + R_lrc) + HFR + np.sum(Z_RC,axis=0)
	
def Z_var_num_RC2(w,HFR,Lstray,**RC_params):
	"""
	Impedance of circuit with 1-n parallel RC circuits in series with a resistor (HFR) and an inductor (Lstray)
	Args:
		w: frequency (Hz)
		HFR: high-frequency resistance
		Lstray: inductance
		RC_params: parameters for each parallel RC circuit. keys: R, Q, n, on (order matters!)
	"""
	
	def RC_switch(on):
		if on >= 1:
			return 1
		else:
			return 0
			
	def Z_RC_element(w,el_params):
		# params: R, Q, n, on
		if RC_switch(el_params[3])==0:
			return np.zeros_like(w)
		else:
			return Z_par(el_params[0],Z_cpe(w,el_params[1],el_params[2]))*RC_switch(el_params[3])
		
	num_RC = int(len(RC_params)/4)
	
	Z_rc = np.sum([Z_RC_element(w,list(RC_params.values())[i*4:i*4+4]) for i in range(num_RC)],axis=0)
	
	return Z_L(w,Lstray) + HFR + Z_rc

def construct_eis_df(f,Z):
	"""
	Construct dataframe from complex impedance array
	Parameters:
		f: frequency array
		Z: complex impedance array
	"""
	df = pd.DataFrame(f,columns=['Freq'])
	df['Zreal'] = Z.real
	df['Zimag'] = Z.imag
	df['Zmod'] = ((Z*Z.conjugate())**0.5).real
	df['Zphz'] = (180/np.pi)*np.arctan(Z.imag/Z.real)
	
	return df
	

def chi_sq(y, y_fit, weights):
	"""
	Weighted sum of squared residuals
	
	Parameters:
	-----------
	y: actual data points (nxp)
	y_fit: fitted data points (nxp)
	weights: weights to apply to squared residuals. n-vector or nx2
	"""
	if len(weights.shape)==1:
		weights = weights.reshape(-1,1)
	elif weights.shape[1]!=2:
		raise ValueError('Invalid shape for weights: {}'.format(weights.shape))
		
	x2 = np.sum(np.sum((y-y_fit)**2*weights**2,axis=1))
	# x2 = np.sum(np.sum((y-y_fit)**2,axis=1)*weights**2)
		
	return x2

def ec_chi_sq(params,w,y,weights,model,normalize='deg'):
	"""
	Chi squared for equivalent circuit model. 
	
	Parameters:
	-----------
	params: dict of model parameters
	w: frequencies
	y: measured impedance data: nx2 matrix of Zreal, Zimag
	weights: weights for squared residuals (n-vector)
	model: equivalent circuit model
	normalize: normalization method. Options:
		'deg': normalize by degrees of freedom, i.e. len(y) - len(params)
		'n': normalize by number of observations, i.e. len(y)
		False: don't normalize
	"""	  
	Zfit = model(w,**params)
	y_fit = np.array([Zfit.real,Zfit.imag]).T
	
	x2 = chi_sq(y,y_fit,weights) #+ np.sum((x < 0).astype(int)*1000)
	
	if normalize=='deg':
		x2 /= (len(y) - len(params))
	elif normalize=='n':
		x2 /= len(y)
	elif normalize is not False:
		raise ValueError(f'Invalid normalize option {normalize}. Options are ''deg'', ''n'', False')
	
	return x2
		
	
def chi_sq_from_df(df,params,model,normalize='deg',weighting='modulus'):
	"""
	Convenience function for getting chi squared from dataframe
	"""
	
	w = df['Freq'].values
	y = df[['Zreal','Zimag']].values
	if weighting=='modulus':
		weights = 1/(df['Zmod']).values
	elif weighting=='proportional':
		weights = 1/y
	elif weighting=='hybrid_modulus':
		weights = 1/(np.abs(y)*df['Zmod'].values.reshape(-1,1))**0.5
	elif weighting=='unity':
		weights = np.ones_like(w)
		
	return ec_chi_sq(params,w,y,weights,model,normalize=normalize)
	

def fuelcell_chi_sq(x,w,y,weights,est_HFR=0,alpha=0,normalize='deg'):
	"""
	Regularized chi squared for fuel cell EC model. Used for Nelder-Mead optimization
	Regularization penalty is difference between model HFR and estimated HFR
	
	Parameters:
	-----------
	x: vector of square roots of model parameters (square root necessary to bound Nelder-Mead to positive values)
		Order: HFR, Rf_c, Yo_c, a_c, Rf_a, Yo_a, a_a, Lstray
	w: frequencies
	y: measured impedance data: nx2 matrix of Zreal, Zimag
	weights: weights for squared residuals (n-vector)
	est_HFR: estimated HFR. If alpha=0, this does not matter
	alpha: regularization weight
	"""
	varnames = ['HFR','Rf_c','Yo_c','a_c','Rf_a','Yo_a','a_a','Lstray']
	params = dict(zip(varnames,x**2))
		
	return ec_chi_sq(params,w,y,weights,Z_fuelcell,normalize) + alpha*(params['HFR'] - est_HFR)**2
	
def estimate_HFR(data,n_pts_extrap=20,alpha=2e-4,min_r2=0,verbose=0,plot_fit=False,ax=None):
	"""
	Estimate HFR from impedance data by interpolating Zreal intercept
	If data does not cross Zreal axis, extrapolate using regularized-degree polynomial fit
	
	Parameters:
	data: dataframe of impedance data read from Gamry DTA file
	n_pts_extrap: if extrapolation required, number of high-frequency data points to fit
	alpha: if extrapolation required, regularization strength for fit
	verbose: 0: no messages; 1: print whether fitted or interpolated; 2: print whether fitted or interpolated and fit info
	"""
	
	end_idx = data[data['Zimag']<0].index.min()
	if end_idx==0:
		#if data starts above x axis, use first n points for fitting and extrapolation
		fit_data = data.iloc[:n_pts_extrap,:]
		extrap_flag = True
	elif np.isnan(end_idx):
		#if data never gets above x axis, use last n points for fitting and extrapolation
		fit_data = data.iloc[-n_pts_extrap:,:]
		extrap_flag = True
	else:
		extrap_flag = False
		
	if extrap_flag:
		#if high-frequency data does not cross Zreal axis, fit and extrapolate
		#weighted fit - give higher weight to points closer to intercept
		fit = reg_degree_polyfit(fit_data['Zreal'],fit_data['Zimag'],weights=(1/fit_data['Zimag'])**2,alpha=alpha,min_r2=min_r2)
		roots = np.roots(fit)
		real_roots = np.real(roots[np.iscomplex(roots)==False])
		min_idx = np.abs(real_roots - fit_data['Zreal'].values[0]).argmin()
		HFR = real_roots[min_idx]
		if verbose==1:
			print('Extrapolated HFR')
		if verbose==2:
			r2 = fit_r_squared(fit_data['Zreal'],fit_data['Zimag'],fit,w=(1/fit_data['Zimag'])**2)
			print('Extrapolated HFR. Degree = {}, r2 = {}'.format(len(fit)-1,round(r2,5)))
	else:
		#else, simply sort and interpolate
		#limit data to points up to first negative Zimag to avoid oscillation when sorting by Zimag
		srt = data.loc[:end_idx,:].sort_values(by='Zimag',axis=0)
		HFR = np.interp(0,srt['Zimag'],srt['Zreal'])
		if verbose in (1,2):
			print('Interpolated HFR')
			
	if plot_fit is True:
		if extrap_flag:
			if ax is None:
				fig, ax = plt.subplots()
			ax.scatter(data['Zreal'][:n_pts_extrap+3],-data['Zimag'][:n_pts_extrap+3],s=10)
			x = np.arange(HFR,fit_data['Zreal'].max(),0.1)
			y_fit = np.polyval(fit,x)
			ax.plot(x,-y_fit,ls='-')
			deg = len(fit)
			r2 = round(fit_r_squared(fit_data['Zreal'],fit_data['Zimag'],fit,w=(1/fit_data['Zimag'])**2),4)
			rHFR = round(HFR,3)
			ax.text(0.1,0.9,f'$R_{{\Omega}}$: {rHFR} $\Omega$\nDegree: {deg}\n$r^2$: {r2}',transform=ax.transAxes,va='top')
			ax.set_ylabel(r'$-Z_{\mathrm{imag}} \ (\Omega)$')
			ax.set_xlabel(r'$Z_{\mathrm{real}} \ (\Omega)$')
	
	
	return HFR
	
	
	
def fit_ec_model(data,model,init_params=None,normalize='deg',alpha=0,n_restarts=10,est_HFR=True,weighting='modulus',return_result=False,simplex_params={},random_seed=None,**est_HFR_kw):
	"""
	Fit equivalent circuit model using Nelder-Mead downhill simplex method.
	Adds random noise to the init_params (+/- 50%) and uses these as starting guesses for optimization.
	Minimizes objective_func with regularization penalty for deviation of fitted HFR from estimated HFR
	Runs with n_restarts different initial parameter sets and keeps best result to attempt to find global minimum.
	
	Parameters:
	-----------
	data: dataframe of impedance data containing Freq, Zreal, Zimag, and Zmod columns
	objective_func: cost function to minimize. Args: (x,w,y,weights, eHFR, alpha)
	init_params: dict of model parameters from which to start optimization
	alpha: regularization factor for HFR deviance from estimated HFR
	n_restarts: number of times to restart optimization from randomized initial parameters
	est_HFR: if True, interpolate Zdata to estimate HFR and use this estimate in init_params
	est_HFR_kw: kwargs to pass to estimate_HFR
	
	Returns scipy.optimize output for best result. result['x']**2 gives optimized parameters
	"""
	w = data['Freq'].values
	y = data.loc[:,['Zreal','Zimag']].values
	if weighting=='modulus':
		weights = 1/(data['Zmod']).values
	elif weighting=='proportional':
		weights = 1/y
	elif weighting=='hybrid_modulus':
		weights = 1/(np.abs(y)*data['Zmod'].values.reshape(-1,1))**0.5
	elif weighting=='unity':
		weights = np.ones_like(w)
	else:
		raise ValueError('Invalid weighting {}. Options are ''modulus'', ''proportional'', ''hybrid_modulus'', ''unity'''.format(weighting))

	if init_params is None:
		# get param names from model argspec
		param_names = inspect.getfullargspec(model)[0]
		param_names.remove('w')
		init_params = dict(zip(param_names,np.ones(len(param_names))))
	else:
		param_names = list(init_params.keys())
		init_params = init_params.copy()
		# # ensure that order of init_params matches argspec
		# init_params = {k:init_params[k] for k in param_names}
	
	def objective_func(x,w,y,weights, eHFR, alpha):
		params = dict(zip(param_names,x**2))
		
		cs = ec_chi_sq(params,w,y,weights,model,normalize=normalize)
		
		# apply a hefty penalty to prevent non-physical n values
		n_vals = np.array([v for k,v in params.items() if k[0]=='n'])# or k=='nu' or k=='beta')])
		n_penalty = sum(n_vals[n_vals > 1])*1e3*cs
		
		return  cs + n_penalty
		
	#estimate HFR if specified
	if est_HFR==True:
		eHFR = estimate_HFR(data,**est_HFR_kw)
		init_params['HFR'] = eHFR
	else:
		eHFR = 0
		if alpha!=0:
			print('''Warning: alpha is non-zero but HFR is not being estimated. This should only be run this way if the HFR in init_params is a reasonably accurate estimate of the actual HFR. 
			Otherwise, set alpha to 0 or est_HFR to True''')
			
	start_vals = np.array(list(init_params.values()))
			
	simplex_defaults = {'shift_factor':2,'n_restarts':5}
	simplex_defaults.update(simplex_params)
	simplex_params = simplex_defaults
	
	# randomly shift the starting parameters and optimize to attempt to find the global min
	best_fun = np.inf
	best_steps = 0
	# initialize RandomState
	randstate = np.random.RandomState(random_seed)
	
	for i in range(simplex_params['n_restarts']):
		if i==0:
			# on first attempt, just use the starting parameters determined above
			init_vals = start_vals
		else:
			# on subsequent attempts, randomly shift the starting paramters
			rands = randstate.rand(len(start_vals))
			# multiply or divide the start_vals by random factors up to shift_factor
			# transform linear [0,1) range to logarithmic [1/shift_factor,shift_factor) range
			factors = (1/simplex_params['shift_factor'])*np.exp(rands*2*np.log(simplex_params['shift_factor']))
			init_vals = start_vals*factors # 0.95*(rands/np.max(np.abs(rands)))*start_vals
		#print(init_vals)
		result = minimize(fun=objective_func,x0=init_vals**(1/2), args=(w,y,weights,eHFR,alpha), method='Nelder-Mead', #tol=1e-10,
						  options=dict(maxiter=10000,adaptive=True))#,bounds=[(0,None)]*len(init_vals))
						  
		

		#print(result)
		if result.fun < best_fun:
			#init_vals = result.x.copy()**2
			best_fun = copy(result.fun)
			best_result = result.copy()
			best_steps = i + 1
		#else:
			#init_vals = prev_vals

		# print(dict(zip(init_params.keys(),result['x']**2)))
		# print('fun: ',result.fun)
		
	print('Best result {:.2e} achieved within {} restarts'.format(best_fun,best_steps))

	best_params = dict(zip(param_names,best_result['x']**2))
	
	if return_result:
		return best_params, best_result
	else:
		return best_params

def regfit_ec_model(data,objective_func,init_params,alpha,n_restarts=10,est_HFR=True,weighting='inverse',**est_HFR_kw):
	# this should be incorporated into the standard fit_ec_model by simply adding a relax option
	"""
	Fit equivalent circuit model using Nelder-Mead downhill simplex method with initial regularization, but final fit unregularized
	First runs n_restarts optimizations with randomized init_params and HFR regularization penalty, and keeps best regularized result.
	Uses best regularized parameters as starting guess for unregularized final fit
	
	Parameters:
	-----------
	data: dataframe of impedance data containing Freq, Zreal, Zimag, and Zmod columns
	objective_func: cost function to minimize. Args: (x,w,y,weights, eHFR, alpha)
	init_params: dict of model parameters from which to start optimization
	alpha: regularization factor for HFR deviance from estimated HFR
	n_restarts: number of times to restart optimization from randomized initial parameters
	est_HFR: if True, interpolate Zdata to estimate HFR and use this estimate in init_params
	
	Returns scipy.optimize output for final result. result['x']**2 gives optimized parameters
	"""
	w = data['Freq'].values
	y = data.loc[:,['Zreal','Zimag']].values
	if weighting=='inverse':
		weights = 1/(data['Zmod']).values
	elif weighting=='equal':
		weights = np.ones_like(w)
	else:
		raise ValueError('Invalid weighting {}. Options are ''inverse'', ''equal'''.format(weighting))
	
	#optimize regularized fit
	best_result = fit_ec_model(data,objective_func,init_params,alpha,n_restarts,est_HFR,weighting,**est_HFR_kw)
	#use best regularized result as starting point for unregularized fit
	unreg_init = best_result['x'].copy()
	result = minimize(fun=objective_func,x0=unreg_init, args=(w,y,weights,0,0), method='Nelder-Mead', #tol=1e-10,
						  options=dict(maxiter=10000,adaptive=True))#,bounds=[(0,None)]*len(init_vals))
	
	print(f'Unregularized result: {result.fun}')

	return result

#----------------------------
#variable RC circuit fitting
#----------------------------
def calculate_weights(data,weighting='modulus',split_character=0.5):
	"""Calculate weights for complex impedance chi squared"""
	f = data['Freq'].values
	y = data.loc[:,['Zreal','Zimag']].values
	if weighting=='modulus':
		weights = 1/(data['Zmod']).values
	elif weighting=='proportional':
		weights = 1/(y**2)**0.5
	elif weighting=='hybrid_modulus':
		mod_root = np.abs(y)**split_character * data['Zmod'].values.reshape(-1,1)**(1-split_character)
		weights = 1/((mod_root)**2)**0.5
		# weights = 1/(np.abs(y)*data['Zmod'].values.reshape(-1,1))**0.5
	elif weighting=='unity':
		weights = np.ones_like(f)
	else:
		raise ValueError('Invalid weighting {}. Options are ''modulus'', ''proportional'', ''hybrid_modulus'', ''unity'''.format(weighting))
	return weights

	
def evaluate_param_window(data,params,model,param,normalize='n',weighting='modulus',bounds=(0.95,1.05),n_points=10):
	factors = np.linspace(bounds[0],bounds[1],n_points)
	param_vals = factors*params[param]
	mod_params = params.copy()
	func_vals = np.empty(len(factors))
	for i,pv in enumerate(param_vals):	
		mod_params[param] = pv
		func_vals[i] = chi_sq_from_df(data,mod_params,model,normalize,weighting)
	return param_vals, func_vals
	
def plot_param_windows(data,params,model,plot_params='all',normalize='n',weighting='modulus',bounds=(0.95,1.05),n_points=10,ncol=3,subplot_dims=(3.5,3),sharey=False):
	if plot_params=='all':
		plot_params = list(params.keys())
	nrow = int(np.ceil(len(plot_params)/ncol))
	fig, axes = plt.subplots(nrow,ncol,figsize=(ncol*subplot_dims[0],nrow*subplot_dims[1]),sharey=sharey)
	
	for param,ax in zip(plot_params,axes.ravel()):
		pv,fv = evaluate_param_window(data,params,model,param,normalize,weighting,bounds,n_points)
		ax.plot(pv,fv)
		ax.set_xlabel(param)
		ax.set_ylabel('Error')
		ax.ticklabel_format(scilimits=(-3,3))
	for ax in axes.ravel()[len(plot_params):]:
		ax.axis('off')
	
	fig.tight_layout()
		
	
def Z_var_num_RC_RL(w,HFR,Lstray,R_L,**RC_params):
	"""
	Impedance of circuit with 1-n parallel RC circuits in series with a resistor (HFR) and parallel RL element (R_L, Lstray)
	Args:
		w: frequency (Hz)
		HFR: high-frequency resistance
		Lstray: inductance
		R_L: resistance of resistor in parallel with inductor
		RC_params: parameters for each parallel RC circuit. keys: R, Q, n
	"""
	#Z_RC = [Z_par(Z_cpe(w,p['Q'],p['n']), p['R']) for p in RC_params]
	num_RC = int(len(RC_params)/3)
	
	Z_RC = [Z_par(Z_cpe(w,RC_params[f'Q{i}'],RC_params[f'n{i}']), RC_params[f'R{i}']) for i in range(num_RC)]
	
	return Z_par(R_L,Z_L(w,Lstray)) + HFR + np.sum(Z_RC,axis=0)
	

def get_model_func(model):
	nonRC_param_names = inspect.getfullargspec(model)[0]
	nonRC_param_names.remove('w')

	def model_func(w,*args):
		params = dict(zip(nonRC_param_names,args))
		RC_param_vals = args[len(nonRC_param_names):]
		num_RC = int(len(RC_param_vals)/3)
		RC_param_names = sum([[f'R{i}',f'Q{i}',f'n{i}'] for i in range(num_RC)],[])
		RC_params  = dict(zip(RC_param_names,RC_param_vals))
		params.update(RC_params)
		Z_model = model(w,**params)
		return np.hstack([Z_model.real,Z_model.imag])
	
	return model_func	
		
	
def model_func(w,HFR,Lstray,R_L,*RC_params):
	"""
	Impedance of circuit with 1-n parallel RC circuits in series with a resistor (HFR) and parallel RL element (R_L, Lstray)
	Args:
		w: frequency (Hz)
		HFR: high-frequency resistance
		Lstray: inductance
		R_L: resistance of resistor in parallel with inductor
		RC_params: parameters for each parallel RC circuit. keys: R, Q, n
	"""
	#Z_RC = [Z_par(Z_cpe(w,p['Q'],p['n']), p['R']) for p in RC_params]
	num_RC = int(len(RC_params)/3)
	
	Z_RC = [Z_par(RC_params[i*3],Z_cpe(w,RC_params[i*3+1],RC_params[i*3+2]), ) for i in range(num_RC)]
	
	Z_model = Z_par(R_L,Z_L(w,Lstray)) + HFR + np.sum(Z_RC,axis=0) 
	return np.hstack([Z_model.real,Z_model.imag])


def fit_var_RC(data,alpha,max_fun,model=Z_var_num_RC,init_params=None,max_L=1e-5,min_geo_gain=5,min_ari_gain=0.05,min_num_RC=1,max_num_RC=3,est_HFR=True,relax=False,
	method='simplex',direction='ascending',early_stop=True,
	err_peak_log_sep=1,weighting='modulus',weight_split_character=0.5,frequency_bounds=None,random_seed=None,
	simplex_params={'shift_factor':2,'n_restarts':5},grid_search_params={'grid_density':3,'n_best':10},
	global_params={'algorithm':'basinhopping','n_restarts':1,'shift_factor':2,'algorithm_kwargs':{}},
	return_info=False,return_history=False,**est_HFR_kw):
	"""
	Fit equivalent circuit model with variable number of parallel RC elements using Nelder-Mead downhill simplex method
	Uses grid search or random parameter sampling to find a global minimum
	Increase the number of RC elements until the target objective function is achieved
	
	Parameters:
		data: dataframe of impedance data containing Freq, Zreal, Zimag, and Zmod columns
		alpha: regularization factor for HFR deviance from estimated HFR
		max_fun: maximum acceptable value of objective function. RC elements will be added until the objective function is less than or equal to this value
		model: EC model to fit. Options: Z_var_num_RC, Z_var_num_RC_RL
		init_params: dict of model parameters from which to start optimization. Can contain parameters for any number of RC elements
		max_L: maximum allowable inductance 
		max_num_RC: maximum number of RC elements to allow
		min_geo_gain: minimum geometric improvement in objective function required to allow addition of an RC element (i.e. min_geo_gain=5 requires a 5-fold improvement in objective function to add an element)
		min_ari_gain: minimum arithmetic improvement in objective function required to allow addition of an RC element (i.e. min_ari_gain=0.05 requires an improvement of 0.05 in objective function to add an element)
			If both min_geo_gain and min_ari_gain are specified, any element addition that satisfies EITHER condition will be kept (not both!)
			This allows for fitting noisy spectra, for which adding an additional element may significantly improve the fit with a large arithmetic gain but relatively small geometric gain,
			as well as cleaner spectra, where an additional element may result in a small arithmetic gain due to the small value of the objective function, but a large geometric gain.
		n_restarts: if grid_search is False, number of times to restart optimization from randomized initial parameters
		est_HFR: if True, interpolate Zdata to estimate HFR and use this estimate in init_params
		relax: if True, perform a final parameter optimization without HFR regularization, using the best regularized fit as the starting point. Default False
		method: optimization method. Options:
			''simplex'': Nelder-Mead simplex method. Can be run from multiple starting points in parameter space in order to increase likelihood of finding global min
			''global'': Use a global optimization method. Recommended method
			''grid_search'': Use a grid search  to explore parameter space, then use several of the best parameter sets as starting points for local Nelder-mead optimization. Not recommmended
		direction: direction for circuit element addition/subtraction
			'ascending': start at min_num_RC and add elements up to max_num_RC. Best for methods 'simplex', 'curve_fit'
			'descending': start at max_num_RC + 1 and remove elements down to min_num_RC. May be useful for method 'global'
			'overshoot': start at min_num_RC and add elements up to max_num_RC + 1, then remove the final element and refit at max_num_RC. May be useful for method 'global'
		early_stop: if True, stop increasing num_RC if 2 consecutive element additions fail to produce a qualifying reduction of the objective function
		err_peak_log_sep: minimum logarithmic (base 10 - order of magnitude) frequency separation between error peaks. Used for identifying maximum error when initializing next RQ element		
		shift_factor: maximum factor by which to multiply/divide initial parameters if using random parameter sampling. E.g. a value of 2 allows initial parameter values to at most be doubled or halved
		weighting: weighting to use for fit. Options: 
			'unity': weight = 1
			'modulus': weight = 1/Zmod
			'proportional': real weight = 1/Zreal, imag weight = 1/Zimag
			'hybrid_modulus': mix between regular modulus and split modulus:
				real weight = (((1/Zmod)^(1-split_character)*(1/Zreal)^split_character)^2)^1/2
				imag weight = (((1/Zmod)^(1-split_character)*(1/Zimag)^split_character)^2)^1/2
		weight_split_character: if weighting=='hybrid_modulus', determines how much split character the hybrid modulus has. 0 is regular modulus, 1 is split modulus, 0.5 is equal split and regular character
		frequency_bounds: bounds on allowed RQ peak frequencies. (min,max) tuple. None indicates no bounds
		random_seed: int or None. If int, initialize a RandomState with this seed and use it for random parameter sampling to guarantee repeatability
		grid_search: if True, use a grid search to identify best starting params for optimization. if False, use random parameter space sampling
		grid_search_params: dict of params for grid search.
			grid_density: number of different values to test for each parameter
			n_best: number of param sets from grid search to use as starting points for optimization
		return_info: if True, return best_fun and num_RC after best_params
		return_history: if True, return history of params and fun for each num_RC tested
		est_HFR_kw: kwargs to pass to estimate_HFR	
		
	Returns: 
		return_info=False,return_history=False: best_params (dict of optimized parameters)
		return_info=True,return_history=False: best_params, best_fun, num_RC
		return_info=True,return_history=True: best_params, best_fun, num_RC, history
		return_info=False,return_history=True: best_params, history
	"""
	
	# get non-RC param names from model argspec
	nonRC_param_names = inspect.getfullargspec(model)[0]
	nonRC_param_names.remove('w')
	
	Zmag = data['Zmod'].max()
	
	if frequency_bounds is None:
		if 'Lstray' in nonRC_param_names:
			def objective_func(param_roots,param_names,w,y,weights,eHFR,alpha):
				params = dict(zip(param_names,param_roots**2))
				err = ec_chi_sq(params,w,y,weights,model,normalize='n')
				
				# apply a hefty penalty to prevent non-physical n values
				n_vals = np.array([v for k,v in params.items() if (k[0]=='n' or k=='nu' or k=='beta')])
				n_penalty = sum(n_vals[n_vals > 1])*1000*Zmag
				
				# apply a hefty penalty to prevent high inductance
				if params['Lstray'] > max_L:
					L_penalty = 1e6*(params['Lstray'] - max_L)*Zmag
				else:
					L_penalty = 0
				
				return err + n_penalty + alpha*(params['HFR'] - eHFR)**2 + L_penalty
		else:
			def objective_func(param_roots,param_names,w,y,weights,eHFR,alpha):
				params = dict(zip(param_names,param_roots**2))
				err = ec_chi_sq(params,w,y,weights,model,normalize='n')
				
				# apply a hefty penalty to prevent non-physical n values
				n_vals = np.array([v for k,v in params.items() if (k[0]=='n' or k=='nu' or k=='beta')])
				n_penalty = sum(n_vals[n_vals > 1])*1000*Zmag
				
				return err + n_penalty + alpha*(params['HFR'] - eHFR)**2
	elif len(frequency_bounds)==2:
		if 'Lstray' in nonRC_param_names:
			def objective_func(param_roots,param_names,w,y,weights,eHFR,alpha):
				params = dict(zip(param_names,param_roots**2))
				err = ec_chi_sq(params,w,y,weights,model,normalize='n')
				
				# apply a hefty penalty to prevent non-physical n values
				n_vals = np.array([v for k,v in params.items() if (k[0]=='n' or k=='nu' or k=='beta')])
				n_penalty = sum(n_vals[n_vals > 1])*1000*Zmag
				
				# apply a hefty penalty to prevent high inductance
				if params['Lstray'] > max_L:
					L_penalty = 1e6*(params['Lstray'] - max_L)*Zmag
				else:
					L_penalty = 0
					
				# apply a hefty penalty to keep peak frequencies inside bounds
				f_peaks = var_RC_peak_frequencies(params)
				above_freq = np.array([np.log(max(f,frequency_bounds[1])) - np.log(frequency_bounds[1]) for f in f_peaks])
				below_freq = np.array([np.log(frequency_bounds[0]) - np.log(min(f,frequency_bounds[0])) for f in f_peaks])
				fRQ_penalty = np.sum(above_freq*1000*Zmag + below_freq*1000*Zmag)
				
				return err + n_penalty + alpha*(params['HFR'] - eHFR)**2 + L_penalty + fRQ_penalty
		else:
			def objective_func(param_roots,param_names,w,y,weights,eHFR,alpha):
				params = dict(zip(param_names,param_roots**2))
				err = ec_chi_sq(params,w,y,weights,model,normalize='n')
				
				# apply a hefty penalty to prevent non-physical n values
				n_vals = np.array([v for k,v in params.items() if (k[0]=='n' or k=='nu' or k=='beta')])
				n_penalty = sum(n_vals[n_vals > 1])*1000*Zmag
					
				# apply a hefty penalty to keep peak frequencies inside bounds
				f_peaks = var_RC_peak_frequencies(params)
				above_freq = np.array([np.log(max(f,frequency_bounds[1])) - np.log(frequency_bounds[1]) for f in f_peaks])
				below_freq = np.array([np.log(frequency_bounds[0]) - np.log(min(f,frequency_bounds[0])) for f in f_peaks])
				fRQ_penalty = np.sum(above_freq*1000*Zmag + below_freq*1000*Zmag)
				
				return err + n_penalty + alpha*(params['HFR'] - eHFR)**2 + fRQ_penalty
	else:
		raise ValueError('Invalid argument for frequency_bounds. Must be a 2-length tuple or list, or None for no bounds')
		
	"""Attempt to use correlation between real and imag errors as an additional penalty term to help find global minimum. Didn't work well"""
	# def objective_func(param_roots,param_names,w,y,weights,eHFR,alpha,beta,sig_err):	
		# params = dict(zip(param_names,param_roots**2))
		# # need Z_fit for error correlation - calculate explicitly instead of using ec_chi_sq to avoid calculating twice
		# Zfit = model(w,**params)
		# y_fit = np.array([Zfit.real,Zfit.imag]).T
		
		# if len(weights.shape)==1:
			# weights = weights.reshape(-1,1)
		# elif weights.shape[1]!=2:
			# raise ValueError('Invalid shape for weights: {}'.format(weights.shape))
		
		# y_err = (y-y_fit)**2*weights**2
		# x2 = np.sum(y_err)
		# # normalize by number of points
		# x2 /= len(y)
		
		# # apply a hefty penalty to prevent non-physical n values
		# n_vals = np.array([v for k,v in params.items() if k[0]=='n'])
		# n_penalty = sum(n_vals[n_vals > 1])*1000
		
		# # apply a penalty for real and imag error correlation
		# y_tot = np.sum(y_err,axis=1)
		# # limit to significant errors to avoid treating cluster of points near zero error as highly correlated
		# y_err_sig = y_err[np.where(y_tot>=sig_err)]
		# # calculate robust correlation coefficient between real and imag errors
		# if len(y_err_sig) > 0:
			# rs = spearmanr(y_err_sig)
			# #print(rs.pvalue)
			# corr_penalty = -beta*np.log(rs.pvalue)
		# else:
			# # if no errors meet significance threshold, no penalty
			# corr_penalty = 0
			
		# return x2 + n_penalty + alpha*(params['HFR'] - eHFR)**2 + corr_penalty
		
	
	
	# get model func for curve_fit
	"""placeholder"""
	model_func = get_model_func(model)
	# get order of max Zreal
	R_ord = np.floor(np.log10(data['Zreal'].max()))
	
	# get frequency, Z values, and weights
	w = data['Freq'].values
	y = data.loc[:,['Zreal','Zimag']].values
	weights = calculate_weights(data,weighting=weighting,split_character=weight_split_character)
	
	# initialize RandomState
	randstate = np.random.RandomState(random_seed)
	
	# initialize lists for storing results for each num_RC
	history = {}
	num = []
	num_fun = []
	num_params = []
	num_new_params = []
	idx_maxerr_hist = [] # history of max error indexes
	
	if init_params is not None:
		# avoid overwriting passed parameters
		init_params = init_params.copy()
	
	##---------------------------------
	## Ascending
	##---------------------------------
	if direction in ('ascending','overshoot'):
	
		if init_params is None:
			init_params = {}
			if 'HFR' in nonRC_param_names:
				init_params['HFR'] = 1
			if 'Lstray' in nonRC_param_names:
				init_params['Lstray'] = 1e-6
			# ensure that order of init_params matches argspec
			init_params = {k:init_params.get(k,1) for k in nonRC_param_names}
			# add params for first RC element
			# set R to same order of magnitude as max Zreal
			init_params.update({'R0':10**R_ord,'Q0':1e-3,'n0':0.5})
		else:
			init_num_RC = int((len(init_params) - len(nonRC_param_names))/3)
			if init_num_RC==0:
				default_RC = {'R0':10**R_ord,'Q0':1e-3,'n0':0.5}
				init_params.update(default_RC)
			param_names = nonRC_param_names + sum([[f'{k}{i}' for k in ['R','Q','n']] for i in range(init_num_RC)],[])
			# ensure that order of init_params matches argspec. If RC params not specified, use defaults
			# Don't use dict.get() in order to throw an error if params (besides 1st RC params) are missing
			init_params = {k:init_params[k] for k in param_names}

		#initial parameter values
		#estimate HFR if specified
		if est_HFR==True:
			eHFR = estimate_HFR(data,**est_HFR_kw)
			init_params['HFR'] = eHFR
			if eHFR < 0:
				init_params['HFR'] = 0
				alpha = 0
				print("""Warning: Estimated HFR is negative. Estimated HFR set to 0, alpha set to 0""")
		else:
			eHFR = 0
			if alpha!=0:
				print('''Warning: alpha is non-zero but HFR is not being estimated. This should only be run this way if the HFR in init_params is a reasonably accurate estimate of the actual HFR. 
				Otherwise, set alpha to 0 or est_HFR to True''')
				
		print('Initial parameters: {}'.format(init_params))
		print('Initial peak frequencies:',var_RC_peak_frequencies(init_params))
		
		
		#determine # of RC elements in initial parameters
		#First two params are HFR and Lstray. Each RC element has 3 params: R, Q, n
		init_num_RC = max(int((len(init_params) - len(nonRC_param_names))/3), min_num_RC)
		
		n = init_num_RC
		
		if direction=='overshoot':
			end_RC = max_num_RC + 1
		else:
			end_RC = max_num_RC
		
		while n <= end_RC:
			if n > int((len(init_params) - len(nonRC_param_names))/3):
				while n > int((len(init_params) - len(nonRC_param_names))/3):
					# add RQ elements until we reach desired number of elements 
					
					# find frequency with largest error in current fit
					Z_fit = model(data['Freq'],**init_params)
					y_fit = np.array([np.real(Z_fit),np.imag(Z_fit)]).T
					if len(weights.shape)==1:
						shaped_weights = weights.reshape(-1,1)
					else:
						shaped_weights = weights
					y_errs = np.sum((y-y_fit)**2*shaped_weights**2,axis=1)
					# aggregate local errors with moving average
					y_errs_agg = np.array([np.mean(y_errs[max(i-5,0):min(i+5,len(y_errs))]) for i in range(len(y_errs))])
					
					# ignore points below Nyquist x-axis (positive phase/inductive)
					ignore_idx = list(np.where(data['Zimag'].values > 0)[0])
					# don't initialize new elements outside user-specified frequency bounds
					if frequency_bounds is not None:
						ignore_idx += list(np.where((data['Freq'].values < frequency_bounds[0]) | (data['Freq'].values > frequency_bounds[1]))[0])
					# don't initialize new elements at edges of measured frequency range
					ignore_idx += [0,1]
					ignore_idx += list(np.arange(len(y_errs_agg)-2,len(y_errs_agg),1).astype(int))
					
					# don't initialize new elements in same location as previously initialized elements
					for idx in idx_maxerr_hist:
						add_idx = np.where(np.abs(np.log10(w) - np.log10(w[idx])) <= err_peak_log_sep)
						# only ignore previous initialization location if an element was actually placed near it
						if np.max([np.min(data['Freq'].values[add_idx]) <= f <= np.max(data['Freq'].values[add_idx]) for f in var_RC_peak_frequencies(init_params)])==True:
							ignore_idx += list(add_idx[0])
						# OR if a new element has been initialized in the same location twice already
						elif idx_maxerr_hist.count(idx) > 1:
							ignore_idx += list(add_idx[0])
						
					ignore_idx = np.unique(ignore_idx)
					#print(ignore_idx)
					
					if len(ignore_idx) > 0:
						y_errs_agg[ignore_idx] = 0
					
					idx_maxerr = np.where(y_errs_agg==np.max(y_errs_agg))[0][0]
					# if idx_maxerr in idx_maxerr_hist:
						# # if the max error is still at a previous max error frequency, assume we can't fit it by adding another RCPE here.
						# # instead add a RCPE at the second highest error
						# min_err_peak_sep=5 # separation between error peaks (in index space)
						# delete_idx = []
						# for idx in idx_maxerr_hist:
							# if idx - min_err_peak_sep >=0:
								# start_idx = idx - min_err_peak_sep
							# else:
								# start_idx = 0
							# end_idx = idx + min_err_peak_sep
							# delete_idx += list(np.arange(start_idx,end_idx))
						# delete_idx = np.unique(delete_idx)
						
						# y_errs = np.delete(y_errs,delete_idx)
						# idx_maxerr = np.where(y_errs==np.max(y_errs))[0][0]
					idx_maxerr_hist.append(idx_maxerr)
					row = data.iloc[idx_maxerr,:]
					# assuming that there is a poorly fitted semicircle located here, add new RCPE circuit with peak at this frequency
					w_maxerr = row['Freq']*2*np.pi
					n_new=0.7
					R_new = -2*row['Zimag']/np.tan(np.pi*n_new/4) # RCPE reaches max Zimag at phase 45*n degrees. R is then related to peak Zimag: -Zimag_peak = R*tan(45*n)/2
					Q_new = (1/R_new)*(1/w_maxerr)**n_new #w_peak = 1/(RQ)^(1/n) 
					RC_idx = int((len(init_params) - len(nonRC_param_names))/3)
					new_RQ = {f'R{RC_idx}':R_new, f'Q{RC_idx}':Q_new, f'n{RC_idx}':n_new}
					num_new_params.append(new_RQ)
					print('New RQ element: peak frequency {:.2e}, params {}'.format(w_maxerr/(2*np.pi),new_RQ))
					init_params.update(new_RQ)
			else:
				idx_maxerr = 'NA'
				new_RQ = 'NA'
			start_vals = np.array(list(init_params.values()))
			
			if method=='simplex':
				simplex_defaults = {'shift_factor':2,'n_restarts':5}
				simplex_defaults.update(simplex_params)
				simplex_params = simplex_defaults
				
				# randomly shift the starting parameters and optimize to attempt to find the global min
				best_fun = np.inf
				best_steps = 0
				for i in range(simplex_params['n_restarts']):
					if i==0:
						# on first attempt, just use the starting parameters determined above
						init_vals = start_vals
					else:
						# on subsequent attempts, randomly shift the starting paramters
						rands = randstate.rand(len(start_vals))
						# multiply or divide the start_vals by random factors up to shift_factor
						# transform linear [0,1) range to logarithmic [1/shift_factor,shift_factor) range
						factors = (1/simplex_params['shift_factor'])*np.exp(rands*2*np.log(simplex_params['shift_factor']))
						init_vals = start_vals*factors # 0.95*(rands/np.max(np.abs(rands)))*start_vals
					#print(init_vals)
					result = minimize(fun=objective_func,x0=init_vals**(1/2), args=(init_params.keys(),w,y,weights,eHFR,alpha), method='Nelder-Mead', #tol=1e-10,
									  options=dict(maxiter=10000,adaptive=True))#,bounds=[(0,None)]*len(init_vals))
									  
					

					#print(result)
					if result.fun < best_fun:
						#init_vals = result.x.copy()**2
						best_fun = copy(result.fun)
						best_result = result.copy()
						best_steps = i + 1
					#else:
						#init_vals = prev_vals

					# print(dict(zip(init_params.keys(),result['x']**2)))
					# print('fun: ',result.fun)
				print('{} RC element(s): Best result {:.2e} achieved within {} restarts'.format(n,best_fun,best_steps))
				
			elif method=='curve_fit':
				simplex_defaults = {'shift_factor':2,'n_restarts':5}
				simplex_defaults.update(simplex_params)
				simplex_params = simplex_defaults
				
				# set bounds
				l_bounds = np.zeros(len(init_params))
				u_bounds = np.zeros(len(init_params))
				for i,k in enumerate(init_params.keys()):
					if k[0]=='n':
						u_bounds[i] = 1
					else:
						u_bounds[i] = np.inf
						
				bounds = (l_bounds,u_bounds)
				
				# transform weights to sigma values
				if len(weights.shape)==1:
					sigma = np.hstack([weights,weights])**0.5
				else:
					sigma = np.hstack([weights[:,0],weights[:,0]])**0.5
				
				# randomly shift the starting parameters and optimize to attempt to find the global min
				best_fun = np.inf
				best_steps = 0
				for attempt in range(simplex_params['n_restarts']):
					if attempt==0:
						# on first attempt, just use the starting parameters determined above
						init_vals = start_vals
					else:
						# on subsequent attempts, randomly shift the starting paramters
						rands = randstate.rand(len(start_vals))
						# multiply or divide the start_vals by random factors up to shift_factor
						# transform linear [0,1) range to logarithmic [1/shift_factor,shift_factor) range
						factors = (1/simplex_params['shift_factor'])*np.exp(rands*2*np.log(simplex_params['shift_factor']))
						init_vals = start_vals*factors # 0.95*(rands/np.max(np.abs(rands)))*start_vals
					
					# keep init vals within bounds
					for i, v in enumerate(init_vals):
						if v < bounds[0][i]:
							init_vals[i] = bounds[0][i]
						elif v > bounds[1][i]:
							init_vals[i] = bounds[1][i]
					
					popt, pcov = curve_fit(model_func,xdata=w,ydata=np.hstack([data['Zreal'].values,data['Zimag'].values]),p0=init_vals,ftol=1e-13,maxfev=int(1e5),bounds=bounds,sigma=sigma)
					fun = objective_func(popt**0.5,init_params.keys(),w,y,weights,eHFR,alpha)
					result = {'fun':fun,'x':popt**0.5}
					
					# result = 
					
					
					
					#print(result)
					if fun < best_fun:
						#init_vals = result.x.copy()**2
						best_fun = copy(fun)
						best_result = copy(result)
						best_steps = attempt + 1
					#else:
						#init_vals = prev_vals

					# print(dict(zip(init_params.keys(),result['x']**2)))
					# print('fun: ',result.fun)
				print('{} RC element(s): Best result {} achieved within {} restarts'.format(n,round(best_fun,6),best_steps))
				
				
			elif method=='global':
				global_defaults = {'algorithm':'basinhopping','n_restarts':1,'shift_factor':2,'algorithm_kwargs':{}}
				global_defaults.update(global_params)
				global_params = global_defaults
				
				best_fun = np.inf
				best_steps = 0
				for i in range(global_params['n_restarts']):
					# if i==0:
						# # on first attempt, just use the starting parameters determined above
						# init_vals = start_vals
					# else:
						# # on subsequent attempts, randomly shift the starting paramters
						# rands = randstate.rand(len(start_vals))
						# # multiply or divide the start_vals by random factors up to shift_factor
						# # transform linear [0,1) range to logarithmic [1/shift_factor,shift_factor) range
						# factors = (1/global_params['shift_factor'])*np.exp(rands*2*np.log(global_params['shift_factor']))
						# init_vals = start_vals*factors # 0.95*(rands/np.max(np.abs(rands)))*start_vals
					
					seed = random_seed + i
					init_vals = start_vals
					
					if global_params['algorithm']=='basinhopping':
						algorithm_kwargs = {'niter':40,'T':max_fun*100,'stepsize':10}
						algorithm_kwargs.update(global_params['algorithm_kwargs'])
						
						with warnings.catch_warnings():
							# too many warnings about invalid values/overflow with basinhopping
							warnings.simplefilter('ignore')
							result = basinhopping(func=objective_func,x0=init_vals**(1/2), 
												minimizer_kwargs={'args':(init_params.keys(),w,y,weights,eHFR,alpha)},#,'method':'Nelder-Mead','options':dict(maxiter=1000,adaptive=True)},
												seed=seed,**algorithm_kwargs
												)
											
					elif global_params['algorithm']=='differential_evolution':
						algorithm_kwargs = {'maxiter':100}
						algorithm_kwargs.update(global_params['algorithm_kwargs'])
						bound_func = lambda tup: (0,1) if tup[0][0]=='n' else (tup[1]*1e-3,tup[1]*1e3)
						bounds = [bound_func(t) for t in init_params.items()]
						# print(dict(zip(init_params.keys(),bounds)))
						result = differential_evolution(func=objective_func,bounds=bounds,
											args=(init_params.keys(),w,y,weights,eHFR,alpha),
											seed=seed,**algorithm_kwargs
											)
											
					if result.fun < best_fun:
						#init_vals = result.x.copy()**2
						best_fun = copy(result.fun)
						best_result = result.copy()
						best_steps = i + 1
									  
				# print(best_result)
				best_fun = best_result['fun']
				print('{} RC element(s): Best result {} achieved within {} restarts'.format(n,round(best_fun,6),best_steps))
				
			elif method=='grid_search':
				# set default params and update any user-specified params
				search_params = {'grid_density':3,'n_best':10}
				search_params.update(grid_search_params)
				
				n_var = len(init_params)
				grid_density = search_params['grid_density']

				# initialize parameter grid with values from init_params
				param_grid = np.reshape(list(init_params.values())*(grid_density**n_var),(grid_density**n_var,n_var))

				# create grid of factors
				grid_1d = np.arange(0.25,1.75+1e-10,1.5/(grid_density-1))
				iter_grid = [grid_1d for i in range(n_var)]
				if est_HFR is True:
					grid_HFR = np.arange(0.75,1.25+1e-10,0.5/(grid_density-1))
					iter_grid[list(init_params.keys()).index('HFR')] = grid_HFR

				factors = np.array([p for p in itertools.product(*iter_grid)])
				param_grid *= factors

				# evaluate objective_func for each set of parameters in grid
				fun_grid = np.array([objective_func(p**0.5,init_params.keys(),w,y,weights,eHFR,alpha) for p in param_grid])
				# get the n_best best parameter sets
				top_params = param_grid[np.argsort(fun_grid)[:search_params['n_best']]]
				
				# run the optimization starting from each selected set of parameters
				best_fun = np.inf
				for init_vals in top_params:
					result = minimize(fun=objective_func,x0=init_vals**(1/2), args=(init_params.keys(),w,y,weights,eHFR,alpha), method='Nelder-Mead', #tol=1e-10,
									  options=dict(maxiter=10000,adaptive=True))#,bounds=[(0,None)]*len(init_vals))

					print(result.fun)
					if result.fun < best_fun:
						best_fun = result.fun.copy()
						best_result = result.copy()
				print('{} RC element(s): Best result from grid search optimization is {}'.format(n,round(best_fun,5)))
						
			if relax==True:
				if alpha==0:
					print('Warning: alpha is zero (unregularized), and passing relax=True will thus have little effect on the result')
				# use the best regularized fit as starting point for a final unregularized optimization
				unreg_init = best_result['x'].copy()
				best_result = minimize(fun=objective_func,x0=unreg_init, args=(init_params.keys(),w,y,weights,0,0), method='Nelder-Mead', #tol=1e-10,
								  options=dict(maxiter=10000,adaptive=True))#,bounds=[(0,None)]*len(init_vals))
								  
				best_fun = copy(best_result.fun)
				print('Unregularized result: {}'.format(round(best_fun,5)))
			
			history[n] = {'fun':best_fun,'params':dict(zip(init_params.keys(),best_result['x']**2)),'idx_maxerr':idx_maxerr,'new_RQ_params':new_RQ}
			num.append(n)
			num_fun.append(best_fun)
			num_params.append(dict(zip(init_params.keys(),best_result['x']**2)))
			
			if best_fun > max_fun:
				# if did not reach the goal function value, check progress 
				if len(num_fun) > 2 and early_stop==True:
					# if we've performed at least 3 steps, check progress
					# get last improvement satisfying min_gain
					geo_gain = np.array(num_fun[:-1])/np.array(num_fun[1:])
					ari_gain = np.array(num_fun[:-1]) - np.array(num_fun[1:])
					if np.max(geo_gain) > min_geo_gain:
						last_geo_gain = np.max(np.where(geo_gain > min_geo_gain))
					else:
						last_geo_gain = -1
					if np.max(ari_gain) > min_ari_gain:
						last_ari_gain = np.max(np.where(ari_gain > min_ari_gain))
					else:
						last_ari_gain = -1
					
					last_gain_idx = max(last_ari_gain,last_geo_gain)
					
					if num[last_gain_idx + 1] >= n - 1:
						# if fit improved on this step or last step, continue
						add_element = True
					else:
						# if fit has not improved in last 2 steps, stop
						add_element = False
				else:
					# if fewer than 3 steps performed or early_stop turned off, continue
					add_element = True
				
				if add_element and n < end_RC:
					# add an element and re-optimize
					# use optimized params as starting point
					init_params = dict(zip(init_params.keys(),best_result['x']**2))
					n += 1
				elif n==end_RC and direction=='overshoot' and end_RC > max_num_RC:
					# remove the smallest element and re-optimize
					init_params = dict(zip(init_params.keys(),best_result['x']**2))
					sorted_params = sort_RQ(init_params,by='R',descending=True)
					n -= 1
					removed_RQ = {}
					for k in [f'R{n}',f'Q{n}',f'n{n}']:
						removed_RQ[k] = sorted_params.pop(k)
					init_params = sorted_params
					
					
					print(f'Overshoot complete with {end_RC} elements. Returning to {n} elements after removing smallest RQ element:',removed_RQ)
					
					# reset the endpoint
					end_RC = max_num_RC
					
					
				else:
					break
			else:
				break
		num_fun = np.array(num_fun)
			
	##---------------------------------
	# descending
	##---------------------------------
	elif direction=='descending':
		#start with extra RQ elements and remove one at a time
		init_num_RC = max_num_RC + 1
		if init_params is None:
			init_params = {}
			if 'HFR' in nonRC_param_names:
				init_params['HFR'] = 1
			if 'Lstray' in nonRC_param_names:
				init_params['Lstray'] = 1e-6
			# ensure that order of init_params matches argspec
			init_params = {k:init_params.get(k,1) for k in nonRC_param_names}
			# add params for RQ elements
			# set R to same order of magnitude as max Zreal
			for n in range(init_num_RC):
				init_params.update({f'R{n}':10**R_ord,f'Q{n}':1e-3,f'n{n}':0.5})
			
		#estimate HFR if specified
		if est_HFR==True:
			eHFR = estimate_HFR(data,**est_HFR_kw)
			init_params['HFR'] = eHFR
			if eHFR < 0:
				init_params['HFR'] = 0
				alpha = 0
				print("""Warning: Estimated HFR is negative. Estimated HFR set to 0, alpha set to 0""")
		else:
			eHFR = 0
			if alpha!=0:
				print('''Warning: alpha is non-zero but HFR is not being estimated. This should only be run this way if the HFR in init_params is a reasonably accurate estimate of the actual HFR. 
				Otherwise, set alpha to 0 or est_HFR to True''')
				
		print('Initial parameters: {}'.format(init_params))
		
		n = init_num_RC
		while n >= 1:
			if n < int((len(init_params) - len(nonRC_param_names))/3):
				# remove the lowest-resistance RQ element
				sorted_params = sort_RQ(init_params,by='R',descending=True)
				removed_RQ = {}
				for k in [f'R{n}',f'Q{n}',f'n{n}']:
					removed_RQ[k] = sorted_params.pop(k)
				init_params = sorted_params
				print('Removed RQ element:',removed_RQ)
			else:
				removed_RQ = 'NA'
			start_vals = np.array(list(init_params.values()))
			
			if method=='global':
				global_defaults = {'algorithm':'basinhopping','n_restarts':1,'shift_factor':2,'algorithm_kwargs':{}}
				global_defaults.update(global_params)
				global_params = global_defaults
				
				best_fun = np.inf
				best_steps = 0
				for i in range(global_params['n_restarts']):
					# if i==0:
						# # on first attempt, just use the starting parameters determined above
						# init_vals = start_vals
					# else:
						# # on subsequent attempts, randomly shift the starting paramters
						# rands = randstate.rand(len(start_vals))
						# # multiply or divide the start_vals by random factors up to shift_factor
						# # transform linear [0,1) range to logarithmic [1/shift_factor,shift_factor) range
						# factors = (1/global_params['shift_factor'])*np.exp(rands*2*np.log(global_params['shift_factor']))
						# init_vals = start_vals*factors # 0.95*(rands/np.max(np.abs(rands)))*start_vals
					
					seed = random_seed + i
					init_vals = start_vals
					
					if global_params['algorithm']=='basinhopping':
						algorithm_kwargs = {'niter':40,'T':max_fun*100,'stepsize':10}
						algorithm_kwargs.update(global_params['algorithm_kwargs'])
						
						with warnings.catch_warnings():
							# too many warnings about invalid values/overflow with basinhopping
							warnings.simplefilter('ignore')
							result = basinhopping(func=objective_func,x0=init_vals**(1/2), 
												minimizer_kwargs={'args':(init_params.keys(),w,y,weights,eHFR,alpha)},#,'method':'Nelder-Mead','options':dict(maxiter=1000,adaptive=True)},
												seed=seed,**algorithm_kwargs
												)
											
					elif global_params['algorithm']=='differential_evolution':
						algorithm_kwargs = {'maxiter':100}
						algorithm_kwargs.update(global_params['algorithm_kwargs'])
						bound_func = lambda tup: (0,1) if tup[0][0]=='n' else (tup[1]*1e-3,tup[1]*1e3)
						bounds = [bound_func(t) for t in init_params.items()]
						# print(dict(zip(init_params.keys(),bounds)))
						result = differential_evolution(func=objective_func,bounds=bounds,
											args=(init_params.keys(),w,y,weights,eHFR,alpha),
											seed=randstate,**algorithm_kwargs
											)
											
					if result.fun < best_fun:
						#init_vals = result.x.copy()**2
						best_fun = copy(result.fun)
						best_result = result.copy()
						best_steps = i + 1
									  
				best_fun = best_result['fun']
				
					
				print('{} RC element(s): Best result {} achieved within {} restarts'.format(n,round(best_fun,6),best_steps))
							
			if relax==True:
				if alpha==0:
					print('Warning: alpha is zero (unregularized), and passing relax=True will thus have little effect on the result')
				# use the best regularized fit as starting point for a final unregularized optimization
				unreg_init = best_result['x'].copy()
				best_result = minimize(fun=objective_func,x0=unreg_init, args=(init_params.keys(),w,y,weights,0,0), method='Nelder-Mead', #tol=1e-10,
								  options=dict(maxiter=10000,adaptive=True))#,bounds=[(0,None)]*len(init_vals))
								  
				best_fun = best_result.fun.copy()
				print('Unregularized result: {}'.format(round(best_fun,5)))
			
			history[n] = {'fun':best_fun,'params':dict(zip(init_params.keys(),best_result['x']**2)),'removed_RQ_params':removed_RQ}
			num.append(n)
			num_fun.append(best_fun)
			num_params.append(dict(zip(init_params.keys(),best_result['x']**2)))
			 
			if len(num_fun) > 1:
				# if we've performed at least 2 steps, check progress
				geo_delta = num_fun[-1]/num_fun[-2] 
				ari_delta = num_fun[-1] - num_fun[-2]
				
				# if the increase in obj func due to removing the last element exceeds min_geo_gain or min_ari_gain, stop
				if geo_delta > min_geo_gain:
					remove_element = False
				elif ari_delta > min_ari_gain:
					remove_element = False
				else:
					remove_element = True
			else:
				# if only 1 step performed, continue
				remove_element = True
			
			if remove_element:
				# remove an element and re-optimize
				# use optimized params as starting point
				init_params = dict(zip(init_params.keys(),best_result['x']**2))
				n -= 1
			else:
				break
				
		# placeholder: reverse num_fun to match ascending direction. Then same logic below can be used
		# next: write separate logic for descending direction
		num_fun = np.array(num_fun)[::-1]
			
	
	if best_fun <= max_fun:
		# if satisfactory fit found, check geo_gain and ari_gain requirements and report appropriate result accordingly
		if len(num_fun) > 1:
			if num_fun[-2]/num_fun[-1] > min_geo_gain or num_fun[-2] - num_fun[-1] > min_ari_gain:
				print(f'Achieved target objective function with {n} RC elements')
				best_params = num_params[-1]
			else:	
				print('Achieved target objective function with {} RC elements, but did not satisfy min_gain. Reverting to {} element(s)'.format(n,n-1))
				best_params = num_params[-2]
				best_fun = num_fun[-2]
		else:
			print(f'Achieved target objective function with {n} RC elements')
			best_params = num_params[-1]
	else:
		# if no satisfactory fit found, find the last result that satisfied gain requirements
		if len(num_fun) > 1:
			# get last improvement satisfying min_gain
			geo_gain = num_fun[:-1]/num_fun[1:]
			ari_gain = num_fun[:-1] - num_fun[1:]
			if np.max(geo_gain) > min_geo_gain:
				last_geo_gain = np.max(np.where(geo_gain > min_geo_gain))
			else:
				last_geo_gain = -1
			if np.max(ari_gain) > min_ari_gain:
				last_ari_gain = np.max(np.where(ari_gain > min_ari_gain))
			else:
				last_ari_gain = -1
				
			last_gain_idx = max(last_ari_gain,last_geo_gain)
				
			if last_gain_idx + 1 == len(geo_gain):
				print(f'Could not achieve target objective function with {n} RC elements, but satisfied min_gain')
				best_params = num_params[-1]
			else:
				n_revert = num[last_gain_idx + 1]
				#print(gain, last_gain_idx, n_revert)
				best_params = num_params[last_gain_idx + 1]
				best_fun = num_fun[last_gain_idx + 1]
				print(f'Could not achieve target objective function with {n} RC elements, and did not satisfy min_gain. Reverting to {n_revert} element(s)')
		else:
			print(f'Could not achieve target objective function with {max_num_RC} RC elements')
			best_params = num_params[-1]
				
	out = [best_params]
	
	if return_info:
		num_RC = int((len(best_params)-len(nonRC_param_names))/3)
		out += [best_fun, num_RC]
	if return_history:
		# num_res = [{'params':p, 'fun':f,'idx_maxerr':idx,'new_RQ_params':rqp} for p,f,idx,rqp in zip(num_params,num_fun,['NA'] + idx_maxerr_hist,['NA'] + num_new_params)]
		# hist = dict(zip(num,num_res))
		out += [history]
	if len(out)==1:
		# if only output is params, pull out of list
		out = out[0]
	return out
###=========================================================================	
	
	
def RQ_peak_frequency(R,Q,n):
	"""Calculate peak frequency of parallel RQ element"""
	return (1/(2*np.pi))*1/(R*Q)**(1/n)
	
def var_RC_peak_frequencies(params):
	"""Calculate peak frequencies of parallel RQ elements in model"""
	RQ_keys = [k for k in params.keys() if k[0] in ['R','Q','n'] and is_number(k[1])]
	if len(RQ_keys)%3!=0:
		raise Exception('Wrong number of RQ keys identified. Check parameter names')
	num_RQ = int(len(RQ_keys)/3)
	f_peaks = [RQ_peak_frequency(params[f'R{i}'],params[f'Q{i}'],params[f'n{i}']) for i in range(num_RQ)]
	return f_peaks #sorted(f_peaks)[::-1]
	
def sort_RQ(params,by='frequency',descending=True):
	"""
	Sort RQ elements
	
	Parameters:
		params: dict of parameters. May include additional equivalent circuit elements beyond RQ elements
		by: metric by which to sort. Options:
			''frequency'': sort by RQ peak frequency
			''R'': sort by resistance
		descending: if True, sort by metric in descending order
	Returns:
		sorted_params: dict of params with RQ elements sorted as specified
	"""
	RQ_keys = [k for k in params.keys() if k[0] in ['R','Q','n'] and is_number(k[1])]
	non_RQ_keys = [k for k in params.keys() if k not in RQ_keys]
	if len(RQ_keys)%3!=0:
		raise Exception('Wrong number of RQ keys identified. Check parameter names')
	num_RQ = int(len(RQ_keys)/3)
	
	if by=='frequency':
		f_peaks = var_RC_peak_frequencies(params)
		sort_idx = np.argsort(f_peaks)
	elif by=='R':
		Rs = [params[f'R{i}'] for i in range(num_RQ)]
		sort_idx = np.argsort(Rs)
		
	if descending==True:
		sort_idx = sort_idx[::-1]
		
	sorted_params = {k:params[k] for k in non_RQ_keys}
	for new_idx,old_idx in enumerate(sort_idx):
		sorted_RQ = {f'{k}{new_idx}':params[f'{k}{old_idx}'] for k in ['R','Q','n']}
		sorted_params.update(sorted_RQ)
		
	return sorted_params


def fit_error(data,params,model=Z_var_num_RC,weighting='modulus',weight_split_character=0.5):
	# set weights
	y = data.loc[:,['Zreal','Zimag']].values
	weights = calculate_weights(data,weighting=weighting,split_character=weight_split_character)
	# check/reshape weights
	if len(weights.shape)==1:
		weights = weights.reshape(-1,1)
	elif weights.shape[1]!=2:
		raise ValueError('Invalid shape for weights: {}'.format(weights.shape))	
		
	# df for calculating model errors at measured frequencies
	Z_fit = model(data['Freq'],**params)
	Z_fit = pd.DataFrame(np.array([Z_fit.real,Z_fit.imag]).T,columns=['Zreal','Zimag'])
	#calculate errors
	y_fit = Z_fit.values
	y_err = (y-y_fit)**2*weights**2 
	err_df = pd.DataFrame(data['Freq'])
	err_df['real'] = y_err[:,0]
	err_df['imag'] = y_err[:,1]
	err_df['tot'] = err_df['real'] + err_df['imag']
	return err_df
	
def plot_fit_error(data,params,model=Z_var_num_RC,weighting='modulus',weight_split_character=0.5,axes=None):
		
	if axes is None:
		fig, axes = plt.subplots(1,3,figsize=(12,4))
		
	# set weights
	y = data.loc[:,['Zreal','Zimag']].values
	weights = calculate_weights(data,weighting=weighting,split_character=weight_split_character)
	# check/reshape weights
	if len(weights.shape)==1:
		weights = weights.reshape(-1,1)
	elif weights.shape[1]!=2:
		raise ValueError('Invalid shape for weights: {}'.format(weights.shape))	
		
	# df for calculating model errors at measured frequencies
	Z_fit = model(data['Freq'],**params)
	Z_fit = pd.DataFrame(np.array([Z_fit.real,Z_fit.imag]).T,columns=['Zreal','Zimag'])
	#calculate errors
	y_fit = Z_fit.values
	y_err = (y-y_fit)**2*weights**2 
	Z_fit['e_real'] = y_err[:,0]
	Z_fit['e_imag'] = y_err[:,1]
	Z_fit['e_tot'] = Z_fit['e_real'] + Z_fit['e_imag']
	
	# plot error
	axes[0].plot(data['Freq'],Z_fit['e_real'])
	axes[1].plot(data['Freq'],Z_fit['e_imag'])
	axes[2].plot(data['Freq'],Z_fit['e_tot'])
	
	for ax in axes:
		ax.set_xscale('log')
		ax.ticklabel_format(axis='y',scilimits=(-3,3))
		ax.set_xlabel('Frequency (Hz)')
	axes[0].set_title('$Z^\prime$ Error')
	axes[1].set_title('$Z^{\prime\prime}$ Error')
	axes[2].set_title('Total Error')
		
	fig.tight_layout()
	
def plot_var_RC_fit_path(data,history,model=Z_var_num_RC,weighting='modulus',weight_split_character=0.5,num_RC=None,subplot_dims=(3.25,3),w_model='fill',chosen_num=None):
	if num_RC is None:
		# if no n specified, plot all ns in history
		num_RC = list(history.keys())
	elif type(num_RC) == int:
		# if scalar, convert to list
		num_RC = [num_RC]
		
	fig, axes = plt.subplots(len(num_RC),4,figsize=(subplot_dims[0]*4,subplot_dims[1]*len(num_RC)))
	if len(num_RC)==1:
		# if only one n, add axis to axes to make zip work
		axes = axes[None,:]
		
	# set shared axes by column
	col1 = list(axes[:,1])
	axes[0,1].get_shared_x_axes().join(*col1)
	axes[0,1].get_shared_y_axes().join(*col1)
	col2 = list(axes[:,2])
	axes[0,2].get_shared_y_axes().join(*col2)
	col3 = list(axes[:,3])
	# axes[0,3].get_shared_y_axes().join(*col3)
	
	# set w_model
	if w_model is None:
		w_model = np.logspace(-2,6,100)
	elif w_model is 'fill':
		w_model = np.logspace(np.log10(np.min(data['Freq'])),np.log10(np.max(data['Freq'])), 100)
		
	# set weights
	y = data.loc[:,['Zreal','Zimag']].values
	weights = calculate_weights(data,weighting=weighting,split_character=weight_split_character)
	# check/reshape weights
	if len(weights.shape)==1:
		weights = weights.reshape(-1,1)
	elif weights.shape[1]!=2:
		raise ValueError('Invalid shape for weights: {}'.format(weights.shape))	
		
	# define parallel RCPE circuit for plotting new elements
	def Z_RQ(w,offset,R,Q,n):
		"offset is just an artifical param to place the RQ element at the right place on the real axis"
		return offset + Z_par(R,Z_cpe(w,Q,n))
		
	unit_scale = get_unit_scale(data)
	
	for num,axrow in zip(num_RC,axes):
		params = history[num]['params']
		# df for calculating model errors at measured frequencies
		Z_fit = model(data['Freq'],**params)
		Z_fit = pd.DataFrame(np.array([np.real(Z_fit),np.imag(Z_fit)]).T,columns=['Zreal','Zimag'])
		# df for plotting model
		Z_fit_smooth = model(w_model,**params)
		Z_fit_smooth = pd.DataFrame(np.array([Z_fit_smooth.real,Z_fit_smooth.imag]).T,columns=['Zreal','Zimag'])
		#calculate errors
		y_fit = Z_fit.values
		y_err = (y-y_fit)**2*weights**2 
		Z_fit['e_real'] = y_err[:,0]
		Z_fit['e_imag'] = y_err[:,1]
		Z_fit['e_tot'] = Z_fit['e_real'] + Z_fit['e_imag']
		Z_fit['e_agg'] = np.array([np.mean(Z_fit['e_tot'].values[max(i-5,0):min(i+5,len(Z_fit))]) for i in range(len(Z_fit))])

		# plot nyquist fit
		plot_nyquist(data,ax=axrow[1],c='k',alpha=0.7,unit_scale=unit_scale)
		plot_model(model,params,plot_type='nyquist',axes=axrow[1],w=w_model,label='Current Model',c='k',alpha=0.7,unit_scale=unit_scale)
		axrow[1].ticklabel_format(axis='both',scilimits=(-3,3))
		axrow[1].set_title('Nyquist Plot')
		
		# plot Zreal and Zimag fits
		axrow[2].scatter(data['Freq'],data['Zreal'],c='k',s=8)
		p1 = axrow[2].plot(w_model,Z_fit_smooth['Zreal'],c='k',label='Z$^\prime$ Fit')
		axrow[2].set_xscale('log')
		axrow[2].set_yscale('log')
		ax2 = axrow[2].twinx()
		ax2.scatter(data['Freq'],-data['Zimag'],c='b',s=8)
		p2 = ax2.plot(w_model,-Z_fit_smooth['Zimag'],c='b',label='Z$^{\prime\prime}$ Fit')
		ax2.ticklabel_format(axis='y',scilimits=(-3,3))
		axrow[2].legend(handles=[p1[0],p2[0]],labels=['Z$^\prime$ Fit','Z$^{\prime\prime}$ Fit'])
		axrow[2].set_xlabel('Frequency (Hz)')
		axrow[2].set_title('Z$^\prime$ and Z$^{{\prime\prime}}$ Fits')
		
		# plot error
		axrow[3].plot(data['Freq'],Z_fit['e_real'],label='Real',c='k')
		axrow[3].plot(data['Freq'],Z_fit['e_imag'],label='Imag',c='b')
		# axrow[3].plot(data['Freq'],Z_fit['e_tot'],label='Total',c='gray',lw=2)
		# ax3 = axrow[3].twinx()
		axrow[3].plot(data['Freq'],Z_fit['e_agg'],label='Aggregate',c='gray',lw=2)
		axrow[3].set_xscale('log')
		axrow[3].ticklabel_format(axis='y',scilimits=(-3,3))
		axrow[3].legend()
		axrow[3].set_xlabel('Frequency (Hz)')
		axrow[3].set_title('Model Error')
		
		if num < np.max(list(history.keys())):
			row_maxerr = data.iloc[history[num+1]['idx_maxerr'],:]
			# mark the chosen max error frequency
			axrow[2].axvline(row_maxerr['Freq'],c='r',ls='--')
			axrow[3].axvline(row_maxerr['Freq'],c='r',ls='--',label='Target')
			axrow[3].legend()
			# plot the new RQ element on the Nyquist
			new_RQ_params = {k[0]:v for k,v in history[num+1]['new_RQ_params'].items()} #remove the number from each param name
			new_RQ_params['offset'] = row_maxerr['Zreal'] - new_RQ_params['R']/2
			# new_RQ_params.update(history[num+1]['new_RQ_params'])
			plot_model(Z_RQ,new_RQ_params,axes=axrow[1],plot_type='nyquist',w=w_model,label='Next RQ Element',c='r',unit_scale=unit_scale)
			
		# model info in leftmost column
		txt = '{} RQ Element(s)\nTotal Error: {:.2e}'.format(num,history[num]['fun'])
		axrow[0].text(0.5,0.5,txt,transform=axrow[0].transAxes,size=13,ha='center')
		if num==chosen_num:
			axrow[0].text(0.5,0.75,'Selected Model',transform=axrow[0].transAxes,size=13,ha='center',color='b',fontweight='bold')
		axrow[0].axis('off')
		
	fig.tight_layout()
		
	
def fit_var_RC2(data,alpha,gamma,init_params=None,max_num_RC=3,n_restarts=10,est_HFR=True,relax=False,weighting='inverse',
	grid_search=False,grid_search_params={},return_info=False,**est_HFR_kw):
	"""
	Fit equivalent circuit model with variable number of parallel RC elements using Nelder-Mead downhill simplex method
	Uses grid search or random parameter sampling to find a global minimum
	Optimize the number of RC elements simultaneously with the circuit parameters
	
	Parameters:
		data: dataframe of impedance data containing Freq, Zreal, Zimag, and Zmod columns
		alpha: regularization factor for HFR deviance from estimated HFR
		gamma: regularization factor for number of RC elements
		init_params: dict of model parameters from which to start optimization. If None, default values will be used based on max_num_RC
		max_num_RC: maximum number of RC elements to allow
		n_restarts: if grid_search is False, number of times to restart optimization from randomized initial parameters
		est_HFR: if True, interpolate Zdata to estimate HFR and use this estimate in init_params
		relax: if True, perform a final parameter optimization without HFR regularization, using the best regularized fit as the starting point. Default False
		weighting: weighting to use for fit. Options: 
			'inverse': weight = 1/Zmod
			'equal': weight = 1
		grid_search: if True, use a grid search to identify best starting params for optimization. if False, use random parameter space sampling
		grid_search_params: dict of params for grid search.
			grid_density: number of different values to test for each parameter
			n_best: number of param sets from grid search to use as starting points for optimization
		est_HFR_kw: kwargs to pass to estimate_HFR	
		
	Returns: dict of optimized parameters
	"""
	
	if init_params is None:
		R_ord = np.floor(np.log10(data['Zreal'].max()))
		new_RC_params={'R':10**R_ord,'Q':1e-3,'n':0.5,'on':1}
		RC_params = list(new_RC_params.values())*max_num_RC
		init_param_vals = [10,1e-6] + RC_params
		param_names = ['HFR','Lstray'] + sum([[f'R{i}',f'Q{i}',f'n{i}',f'on{i}'] for i in range(max_num_RC)],[])
		init_params = dict(zip(param_names,init_param_vals))
	
	def RC_switch(on):
		if on >= 1:
			return 1
		else:
			return 0
			
	def Z_RC_element(w,el_params):
		# params: R, Q, n, on
		return Z_par(el_params[0],Z_cpe(w,el_params[1],el_params[2]))*RC_switch(el_params[3])
		
	def Z_model(w,params):
		# four params for each RC element
		Z_rc = np.sum([Z_RC_element(w,params[i*4+2:i*4+6]) for i in range(max_num_RC)],axis=0)
		return params[0] + Z_L(w,params[1]) + Z_rc
	
	def num_on(params):
		return sum([RC_switch(params[i*4+5]) for i in range(max_num_RC)])
	
	def objective_func(param_roots,w,y,weights,eHFR,alpha,gamma):
		params = param_roots**2
		Z_fit = Z_model(w,params)
		y_hat = np.array([Z_fit.real,Z_fit.imag]).T
		err = chi_sq(y,y_hat,weights)/(len(y)-len(params))
		
		# apply a hefty penalty to prevent non-physical n values
		n_vals = np.array([params[i*4+4] for i in range(max_num_RC)])
		n_penalty = sum(n_vals[n_vals > 1])*1000
		
		return err + alpha*(params[0] - eHFR)**2 + gamma*num_on(params) + n_penalty
	
	w = data['Freq'].values
	y = data.loc[:,['Zreal','Zimag']].values
	if weighting=='inverse':
		weights = 1/(data['Zmod']).values
	elif weighting=='equal':
		weights = np.ones_like(w)
	else:
		raise ValueError('Invalid weighting {}. Options are ''inverse'', ''equal'''.format(weighting))

	#initial parameter values
	#estimate HFR if specified
	if est_HFR==True:
		eHFR = estimate_HFR(data,**est_HFR_kw)
		init_params['HFR'] = eHFR
		if eHFR < 0:
			init_params['HFR'] = 0
			alpha = 0
			print("""Warning: Estimated HFR is negative. Estimated HFR set to 0, alpha set to 0""")
	else:
		eHFR = 0
		if alpha!=0:
			print('''Warning: alpha is non-zero but HFR is not being estimated. This should only be run this way if the HFR in init_params is a reasonably accurate estimate of the actual HFR. 
			Otherwise, set alpha to 0 or est_HFR to True''')
	
	#determine # of RC elements in initial parameters
	#First two params are HFR and Lstray. Each RC element has 3 params: R, Q, n
	
	start_vals = np.array(list(init_params.values()))
		
	if grid_search is False:
		# randomly shift the starting parameters and optimize to attempt to find the global min
		best_fun = np.inf
		best_steps = 0
		for i in range(n_restarts):

			init_vals = start_vals + 1.9*(np.random.rand(len(start_vals))-0.5)*start_vals
			#print(init_vals)
			result = minimize(fun=objective_func,x0=init_vals**(1/2), args=(w,y,weights,eHFR,alpha,gamma), method='Nelder-Mead', #tol=1e-10,
							  options=dict(maxiter=10000,adaptive=True))#,bounds=[(0,None)]*len(init_vals))

			#print(result)
			if result.fun < best_fun:
				#init_vals = result.x.copy()**2
				best_fun = result.fun.copy()
				best_result = result.copy()
				best_steps = i + 1

			# print(dict(zip(init_params.keys(),result['x']**2)))
			# Z_fit = Z_model(w,result['x']**2)
			# y_hat = np.array([Z_fit.real,Z_fit.imag]).T
			# err = chi_sq(y,y_hat,weights,normalize=True)
			# print('err: ',err)
			# print('fun: ',result.fun)
			# print('num RC: ', num_on(best_result['x']**2))
		print('Best result {} achieved with {} RC elements within {} restarts'.format(round(best_fun,5),num_on(best_result['x']**2),best_steps))
		
	elif grid_search is True:
		# set default params and update any user-specified params
		search_params = {'grid_density':3,'n_best':10}
		search_params.update(grid_search_params)
		
		n_var = len(init_params)
		grid_density = search_params['grid_density']

		# initialize parameter grid with values from init_params
		param_grid = np.reshape(list(init_params.values())*(grid_density**n_var),(grid_density**n_var,n_var))

		# create grid of factors
		grid_1d = np.arange(0.25,1.75+1e-10,1.5/(grid_density-1))
		iter_grid = [grid_1d for i in range(n_var)]
		if est_HFR is True:
			grid_HFR = np.arange(0.75,1.25+1e-10,0.5/(grid_density-1))
			iter_grid[list(init_params.keys()).index('HFR')] = grid_HFR

		factors = np.array([p for p in itertools.product(*iter_grid)])
		param_grid *= factors

		# evaluate objective_func for each set of parameters in grid
		fun_grid = np.array([objective_func(p**0.5,w,y,weights,eHFR,alpha,gamma) for p in param_grid])
		# get the n_best best parameter sets
		top_params = param_grid[np.argsort(fun_grid)[:search_params['n_best']]]
		
		# run the optimization starting from each selected set of parameters
		best_fun = np.inf
		for init_vals in top_params:
			result = minimize(fun=objective_func,x0=init_vals**(1/2), args=(w,y,weights,eHFR,alpha,gamma), method='Nelder-Mead', #tol=1e-10,
							  options=dict(maxiter=10000,adaptive=True))#,bounds=[(0,None)]*len(init_vals))

			print(result.fun)
			if result.fun < best_fun:
				best_fun = result.fun.copy()
				best_result = result.copy()
		print('Best result from grid search optimization is {}'.format(round(best_fun,5)))
				
	if relax==True:
		if alpha==0:
			print('Warning: alpha is zero (unregularized), and passing relax=True will thus have little effect on the result')
		# use the best regularized fit as starting point for a final unregularized optimization
		unreg_init = best_result['x'].copy()
		best_result = minimize(fun=objective_func,x0=unreg_init, args=(w,y,weights,0,0,gamma), method='Nelder-Mead', #tol=1e-10,
						  options=dict(maxiter=10000,adaptive=True))#,bounds=[(0,None)]*len(init_vals))
						  
		best_fun = best_result.fun.copy()
		print('Unregularized result: {}'.format(round(best_fun,5)))
	
	opt_params =  dict(zip(param_names,best_result['x']**2))
	# set params for switched off elements to zero for readability
	for i in range(max_num_RC):
		if RC_switch(opt_params[f'on{i}'])==0:
			off_dict = {f'R{i}':0,f'Q{i}':0,f'n{i}':0,f'on{i}':0}
			opt_params.update(off_dict)
	
	# if best_fun > max_fun:
		# # if did not reach the goal function value, add an element and re-optimize
		# # use optimized params as starting point
		# init_params = dict(zip(init_params.keys(),best_result['x']**2))
		# n += 1
	# else:
		# break
			
	# if best_fun <= max_fun:
		# print(f'Achieved target objective function with {n} RC elements')
	# else:
		# print(f'Could not achieve target objective function with {max_num_RC} RC elements')
	if return_info is True:
		return opt_params, best_fun, num_on(best_result['x']**2)
	else:
		return opt_params

		

def fit_var_RC_with_clean(file,return_df=False,flag_params={'n':3,'iqr_bound':5},**fit_params):
	"""
	Read eis file, flag and remove errant points, and fit cleaned data
	Parameters:
		file: EIS data file
		return_df: if True, return flagged dataframe
		flag_params: dict of parameters to pass to flag_eis_points
		fit_params: kwargs to pass to fit_var_RC
		
	Returns: 
		mdf: flagged dataframe (if return_df is True)
		fit_result: output from fit_var_RC. Dict or list, depending on options specified in fit_params
	"""
	df = read_eis_zdata(file)
	mdf = flag_eis_points(df,**flag_params)
	clean_df = mdf[mdf['flag']==0]
	
	#params,fun,num,hist = fit_var_RC(mdf,**fit_params)
	fit_result = fit_var_RC_test(clean_df,**fit_params)
	
	if return_df:
		return mdf, fit_result
	else:
		return fit_result
		
#temporary fit with clean function for testing. Need in module for parallelization		
def fit_with_clean(file,flag_params={'n':3,'iqr_bound':5},**fit_params):
	df = read_eis_zdata(file)
	mdf = flag_eis_points(df,**flag_params)
	clean_df = mdf[mdf['flag']==0]
	
	params,fun,num,hist = fit_var_RC_test(mdf,**fit_params)
	cparams,cfun,cnum,chist = fit_var_RC_test(clean_df,**fit_params)
	
	return mdf, params, fun, num,hist, cparams, cfun, cnum, chist

def fit_with_clean2(file,flag_params={'n':3,'iqr_bound':5},**fit_params):
	df = read_eis_zdata(file)
	mdf = flag_eis_points(df,**flag_params)
	clean_df = mdf[mdf['flag']==0]
	
	params,fun,num = fit_var_RC2(mdf,**fit_params)
	cparams,cfun,cnum = fit_var_RC2(clean_df,**fit_params)
	
	return mdf, params, fun, num, cparams, cfun, cnum
	
	

		