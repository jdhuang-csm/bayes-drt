# Module for plotting and fitting EIS data
# (C) Jake Huang 2020
import os
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

import warnings

from .utils import polar_from_complex


# ---------------------
# File loading
# ---------------------
def source_extension(source):
    """Get file extension for source"""
    extensions = {'gamry': '.DTA', 'zplot': '.z'}
    return extensions[source]


def get_file_source(file):
    """Determine file source"""
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()
    # determine	format
    if txt.split('\n')[0] == 'EXPLAIN':
        source = 'gamry'
    elif txt.split('\n')[0] == 'ZPLOT2 ASCII':
        source = 'zplot'

    return source


def get_timestamp(file):
    """Get experiment start timestamp from file"""
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()

    source = get_file_source(file)

    if source == 'gamry':
        date_start = txt.find('DATE')
        date_end = txt[date_start:].find('\n') + date_start
        date_line = txt[date_start:date_end]
        date = date_line.split('\t')[2]

        time_start = txt.find('TIME')
        time_end = txt[time_start:].find('\n') + time_start
        time_line = txt[time_start:time_end]
        time = time_line.split('\t')[2]

        timestr = date + ' ' + time
        dt = datetime.strptime(timestr, "%m/%d/%Y %H:%M:%S")

    elif source == 'zplot':
        date_start = txt.find('Date')
        date_end = txt[date_start:].find('\n') + date_start
        date_line = txt[date_start:date_end]
        date = date_line.split()[1]

        time_start = txt.find('Time')
        time_end = txt[time_start:].find('\n') + time_start
        time_line = txt[time_start:time_end]
        time = time_line.split()[1]

        timestr = date + ' ' + time
        dt = datetime.strptime(timestr, "%m-%d-%Y %H:%M:%S")

    return dt


def read_eis(file, warn=True):
    """read EIS zcurve data from Gamry .DTA file"""
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()
    source = get_file_source(file)

    if source == 'gamry':
        # find start of zcurve data
        zidx = txt.find('ZCURVE')
        # check for experiment aborted flag
        if txt.find('EXPERIMENTABORTED') > -1:
            skipfooter = len(txt[txt.find('EXPERIMENTABORTED'):].split('\n')) - 1
        else:
            skipfooter = 0

        # preceding text
        pretxt = txt[:zidx]

        # zcurve data
        ztable = txt[zidx:]
        # column headers are next line after ZCURVE TABLE line
        header_start = ztable.find('\n') + 1
        header_end = header_start + ztable[header_start:].find('\n')
        header = ztable[header_start:header_end].split('\t')
        # units are next line after column headers
        unit_end = header_end + 1 + ztable[header_end + 1:].find('\n')
        units = ztable[header_end + 1:unit_end].split('\t')
        # determine # of rows to skip by counting line breaks in preceding text
        skiprows = len(pretxt.split('\n')) + 2

        # if table is indented, ignore empty left column
        if header[0] == '':
            usecols = header[1:]
        else:
            usecols = header

        # if extra tab at end of data rows, add an extra column to header to match (for Igor data)
        first_data_row = ztable[unit_end + 1: unit_end + 1 + ztable[unit_end + 1:].find('\n')]
        if first_data_row.split('\t')[-1] == '':
            header = header + ['extra_tab']

        # read data to DataFrame
        # python engine required to use skipfooter
        data = pd.read_csv(file, sep='\t', skiprows=skiprows, header=None, names=header, usecols=usecols,
                           skipfooter=skipfooter, engine='python')

        # add timestamp
        try:
            dt = get_timestamp(file)
            time_col = np.intersect1d(['Time', 'T'], data.columns)[
                0]  # EIS files in Repeating jv-EIS files have column named 'Time' instead of 'T'
            data['timestamp'] = [dt + timedelta(seconds=t) for t in data[time_col]]
        except Exception:
            if warn:
                warnings.warn(f'Reading timestamp failed for file {file}')

    elif source == 'zplot':
        # find start of zcurve data
        zidx = txt.find('End Comments')

        # preceding text
        pretxt = txt[:zidx]

        # z data
        ztable = txt[zidx:]
        # column headers are in line above "End Comments"
        header = pretxt.split('\n')[-2].strip().split('\t')

        # determine # of rows to skip by counting line breaks in preceding text
        skiprows = len(pretxt.split('\n'))

        # if table is indented, ignore empty left column
        if header[0] == '':
            usecols = header[1:]
        else:
            usecols = header

        # read data to DataFrame
        data = pd.read_csv(file, sep='\t', skiprows=skiprows, header=None, names=header, usecols=usecols)

        # rename to standard format
        rename = {"Z'(a)": "Zreal", "Z''(b)": "Zimag", "Freq(Hz)": "Freq"}
        data = data.rename(rename, axis=1)

        # calculate Zmod and Zphz
        Zmod, Zphz = polar_from_complex(data)
        data['Zmod'] = Zmod
        data['Zphz'] = Zphz

    return data


def read_jv(file, source='gamry'):
    """read from manual jV txt file"""
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()

    if source == 'manual':
        """Manually created j-V txt file"""
        jv_idx = txt.find('Current')
        pretxt = txt[:jv_idx]
        skiprows = len(pretxt.split('\n')) - 1
        data = pd.read_csv(file, sep='\t', skiprows=skiprows)
    elif source == 'gamry':
        # find start of curve data
        cidx = txt.find('CURVE\tTABLE')

        # preceding text
        pretxt = txt[:cidx]

        # curve data
        ctable = txt[cidx:]
        # column headers are next line after ZCURVE TABLE line
        header_start = ctable.find('\n') + 1
        header_end = header_start + ctable[header_start:].find('\n')
        header = ctable[header_start:header_end].split('\t')
        # units are next line after column headers
        unit_end = header_end + 1 + ctable[header_end + 1:].find('\n')
        units = ctable[header_end + 1:unit_end].split('\t')
        # determine # of rows to skip by counting line breaks in preceding text
        skiprows = len(pretxt.split('\n')) + 2

        # if table is indented, ignore empty left column
        if header[0] == '':
            usecols = header[1:]
        else:
            usecols = header
        # read data to DataFrame
        data = pd.read_csv(file, sep='\t', skiprows=skiprows, header=None, names=header, usecols=usecols)
    else:
        raise ValueError(f'Invalid source {source}. Options are ''gamry'', ''manual''')

    return data


def read_ocv(file, file_type='auto'):
    """
	read OCV data from Gamry .DTA file

	Args:
		file: file to read
		file_type: file type. Options are 'ocv','eis'
	"""
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()

    if file_type == 'auto':
        file_type = os.path.basename(file).split('_')[0].lower()[:3]
    # find start (and end, if needed) of ocv data
    if file_type in ('ocv', 'ocp'):
        cidx = txt.find('CURVE\tTABLE')
        skipfooter = 0
    elif file_type == 'eis':
        cidx = txt.find('OCVCURVE\tTABLE')
        post_txt = txt[txt.find('EOC\tQUANT'):]
        skipfooter = len(post_txt.split('\n')) - 1

    if cidx == -1:
        # coudn't find OCV curve data in file
        # return empty dataframe
        return pd.DataFrame([])
    else:
        # preceding text
        pretxt = txt[:cidx]

        # ocv curve data
        ctable = txt[cidx:]
        # column headers are next line after ZCURVE TABLE line
        header_start = ctable.find('\n') + 1
        header_end = header_start + ctable[header_start:].find('\n')
        header = ctable[header_start:header_end].split('\t')
        # units are next line after column headers
        unit_end = header_end + 1 + ctable[header_end + 1:].find('\n')
        units = ctable[header_end + 1:unit_end].split('\t')
        # determine # of rows to skip by counting line breaks in preceding text
        skiprows = len(pretxt.split('\n')) + 2

        # if table is indented, ignore empty left column
        if header[0] == '':
            usecols = header[1:]
        else:
            usecols = header
        # read data to DataFrame
        data = pd.read_csv(file, sep='\t', skiprows=skiprows, skipfooter=skipfooter, header=None, names=header,
                           usecols=usecols, engine='python')

        # get timestamp
        dt = get_timestamp(file)
        time_col = np.intersect1d(['Time', 'T'], data.columns)[
            0]  # EIS files in Repeating jv-EIS files have column named 'Time' instead of 'T'
        data['timestamp'] = [dt + timedelta(seconds=t) for t in data[time_col]]

        return data


def read_gen_curve(file):
    """
	read generic curve data from Gamry .DTA file

	Args:
		file: file to read
	"""
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()

    # find start of curve data
    cidx = txt.find('CURVE\tTABLE')
    skipfooter = 0

    if cidx == -1:
        # coudn't find OCV curve data in file
        # return empty dataframe
        return pd.DataFrame([])
    else:
        # preceding text
        pretxt = txt[:cidx]

        # ocv curve data
        ctable = txt[cidx:]
        # column headers are next line after ZCURVE TABLE line
        header_start = ctable.find('\n') + 1
        header_end = header_start + ctable[header_start:].find('\n')
        header = ctable[header_start:header_end].split('\t')
        # units are next line after column headers
        unit_end = header_end + 1 + ctable[header_end + 1:].find('\n')
        units = ctable[header_end + 1:unit_end].split('\t')
        # determine # of rows to skip by counting line breaks in preceding text
        skiprows = len(pretxt.split('\n')) + 2

        # if table is indented, ignore empty left column
        if header[0] == '':
            usecols = header[1:]
        else:
            usecols = header
        # read data to DataFrame
        data = pd.read_csv(file, sep='\t', skiprows=skiprows, skipfooter=skipfooter, header=None, names=header,
                           usecols=usecols, engine='python')

        # get timestamp
        dt = get_timestamp(file)
        # time_col = np.intersect1d(['Time','T'],data.columns) # EIS files in Repeating jv-EIS files have column named 'Time' instead of 'T'
        data['timestamp'] = [dt + timedelta(seconds=t) for t in data['T']]

        return data


def read_lsv(file):
    """read LSV data from Gamry .DTA file"""
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()
    # find start of curve data
    cidx = txt.find('CURVE\tTABLE')

    # preceding text
    pretxt = txt[:cidx]

    # LSV curve data
    ctable = txt[cidx:]
    # column headers are next line after CURVE TABLE line
    header_start = ctable.find('\n') + 1
    header_end = header_start + ctable[header_start:].find('\n')
    header = ctable[header_start:header_end].split('\t')
    # units are next line after column headers
    unit_end = header_end + 1 + ctable[header_end + 1:].find('\n')
    units = ctable[header_end + 1:unit_end].split('\t')
    # determine # of rows to skip by counting line breaks in preceding text
    skiprows = len(pretxt.split('\n')) + 2

    # if table is indented, ignore empty left column
    if header[0] == '':
        usecols = header[1:]
    else:
        usecols = header
    # read data to DataFrame
    data = pd.read_csv(file, sep='\t', skiprows=skiprows, header=None, names=header, usecols=usecols)

    return data


# -----------------------------------------
# Convenience functions for data extraction
# -----------------------------------------
def get_fZ(df):
    """Get frequency and Z from DataFrame"""
    freq = df['Freq'].values
    Z = df['Zreal'].values + 1j * df['Zimag'].values

    return freq, Z


