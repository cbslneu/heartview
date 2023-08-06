import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import datetime as dt
import heartpy as hp
import plotly.graph_objects as go
from zipfile import ZipFile, ZipExtFile

def get_fs(file):
    """Get the sampling rate from an Empatica E4 file."""
    contents = pd.read_csv(file, header = None, nrows = 2, usecols = [0])
    fs = contents.iloc[1].item()
    return fs

def get_start_time(file):
    """Get the Unix-formatted start time of an Emmpatica E4 recording."""
    contents = pd.read_csv(file, header = None, nrows = 2, usecols = [0])
    if type(file) is ZipExtFile:
        if 'IBI' in file.name:
            start = contents.loc[0, 0]
        else:
            start = contents.iloc[0].item()
    else:
        if file.endswith('IBI.csv'):
            start = contents.loc[0, 0]
        else:
            start = contents.iloc[0].item()
    return start

def preprocess_e4(e4_zip):
    """Pre-process Empatica E4 data."""

    with ZipFile(e4_zip) as z:
        bvp = pd.read_csv(z.open('BVP.csv'), header = 1, names = ['BVP'])
        bvp_fs = get_fs(z.open('BVP.csv'))
        bvp_start = get_start_time(z.open('BVP.csv'))
        hr = pd.read_csv(z.open('HR.csv'), header = 1, names = ['HR'])
        ibi = pd.read_csv(z.open('IBI.csv'), header = 0,
                          names = ['Seconds', 'IBI'])
        ibi_start = get_start_time(z.open('IBI.csv'))
        acc = pd.read_csv(z.open('ACC.csv'), header = 1,
                          names = ['X', 'Y', 'Z'])
        acc_fs = get_fs(z.open('ACC.csv'))
        acc_start = get_start_time(z.open('ACC.csv'))

    # Pre-process IBI
    ibi.insert(0, 'Timestamp', (ibi['Seconds'] + ibi_start).apply(
        lambda t: dt.datetime.fromtimestamp(t)))

    # Pre-process BVP
    bvp_end = bvp_start + (len(bvp) / bvp_fs)
    bvp.insert(0, 'Unix', np.arange(bvp_start, bvp_end, 1 / bvp_fs))
    bvp.insert(1, 'Timestamp', bvp['Unix'].apply(
        lambda unix: dt.datetime.fromtimestamp(unix)))
    for n in range(len(ibi)):
        ix = round(bvp_start + ibi.loc[n, 'Seconds'])
        bvp.loc[(bvp['Unix'] >= ix) & (bvp['Unix'] <= ix), 'Peak'] = 1

    # Pre-process ACC
    acc_end = acc_start + (len(acc) / acc_fs)
    acc = acc.apply(lambda x: (x / 64) * 9.81)
    acc.insert(0, 'Timestamp', np.arange(acc_start, acc_end, (1/ acc_fs)))
    acc['Timestamp'] = acc['Timestamp'].apply(lambda t: dt.datetime.fromtimestamp(t))
    acc['Magnitude'] = np.sqrt(acc[['X', 'Y', 'Z']].apply(lambda x: x ** 2).sum(axis = 1))

    # Pre-process HR
    hr_end = bvp_start + len(hr)
    hr.insert(0, 'Unix', np.arange(bvp_start, hr_end))
    hr.insert(1, 'Timestamp', hr['Unix'].apply(
        lambda t: dt.datetime.fromtimestamp(t)))

    # -- check for and add invalid beats
    hr.insert(hr.shape[1], 'Invalid', hr['HR'].apply(
        lambda x: 1 if x < 30 or x > 220 else 0))

    # -- get peak locations
    for n in range(len(ibi)):
        ix = round(ibi_start + ibi.loc[n, 'Seconds'])
        hr.loc[(hr['Unix'] >= ix) & (hr['Unix'] <= ix), 'Peak'] = 1

    e4_data = {'bvp': bvp,
               'hr': hr,
               'ibi': ibi,
               'acc': acc,
               'fs': bvp_fs,
               'start time': bvp_start}

    return e4_data

# def preprocess_e4(e4_zip):
#     """Pre-process Empatica E4 data."""
#
#     with ZipFile(e4_zip) as z:
#         bvp = pd.read_csv(z.open('BVP.csv'), header = 1, names = ['BVP'])
#         bvp_fs = get_fs(z.open('BVP.csv'))
#         start = get_start_time(z.open('BVP.csv'))
#         hr = pd.read_csv(z.open('HR.csv'), header = 1, names = ['HR'])
#         ibi = pd.read_csv(z.open('IBI.csv'), header = 0,
#                           names = ['Seconds', 'IBI'])
#         ibi_start = get_start_time(z.open('IBI.csv'))
#
#     # Pre-process IBI
#     ibi.insert(0, 'Timestamp', (ibi['Seconds'] + ibi_start).apply(
#         lambda t: dt.datetime.fromtimestamp(t)))
#
#     # Pre-process BVP
#     end = start + (len(bvp) / bvp_fs)
#     bvp.insert(0, 'Unix', np.arange(start, end, 1 / bvp_fs))
#     bvp.insert(1, 'Timestamp', bvp['Unix'].apply(
#         lambda unix: dt.datetime.fromtimestamp(unix)))
#     for n in range(len(ibi)):
#         ix = round(start + ibi.loc[n, 'Seconds'])
#         bvp.loc[(bvp['Unix'] >= ix) & (bvp['Unix'] <= ix), 'Peak'] = 1
#
#     # Pre-process HR
#     hr_end = start + len(hr)
#     hr.insert(0, 'Unix', np.arange(start, hr_end))
#     hr.insert(1, 'Timestamp', hr['Unix'].apply(
#         lambda t: dt.datetime.fromtimestamp(t)))
#
#     # -- check for and add invalid beats
#     hr.insert(hr.shape[1], 'Invalid', hr['HR'].apply(
#         lambda x: 1 if x < 30 or x > 220 else 0))
#
#     # -- get peak locations
#     for n in range(len(ibi)):
#         ix = round(ibi_start + ibi.loc[n, 'Seconds'])
#         hr.loc[(hr['Unix'] >= ix) & (hr['Unix'] <= ix), 'Peak'] = 1
#
#     e4_data = {'bvp': bvp,
#                'hr': hr,
#                'ibi': ibi,
#                'fs': bvp_fs,
#                'start time': start}
#
#     return e4_data

def get_e4_peaks(ibi, fs, start_time):
    """Get peak locations from Empatica E4 IBI file."""

    # ibi = pd.read_csv(ibi_file, header = 0, names = ['Seconds', 'IBI'])
    ibi['IBI'] = ibi['IBI'] * 1000
    ibi['HR'] = 60000 / ibi['IBI']
    end = start_time + ibi.iloc[-1]['Seconds']

    df = pd.DataFrame()
    df['Unix'] = np.arange(start_time, end, 1/fs)
    df['Timestamp'] = df['Unix'].apply(lambda t: dt.datetime.fromtimestamp(t))
    df['Second'] = pd.Series(df.index).apply(lambda x: (x // fs) + 1)
    df['Peak'] = np.nan

    for n in range(len(ibi)):
        unix = start_time + ibi.loc[n, 'Seconds']
        IBI = ibi.loc[n, 'IBI']
        HR = ibi.loc[n, 'HR']
        df.loc[(df['Unix'] >= unix) & (df['Unix'] < (unix + 1/fs)), 'Peak'] = 1
        df.loc[(df['Unix'] >= unix) & (df['Unix'] < (unix + 1/fs)), 'IBI'] = IBI
        df.loc[(df['Unix'] >= unix) & (df['Unix'] < (unix + 1/fs)), 'HR'] = HR

    # for n in range(len(ibi)):
    #     unix = start_time + ibi['Seconds'].iloc[n]
    #     IBI = ibi['IBI'].iloc[n]
    #     HR = ibi['HR'].iloc[n]
    #     df.at[df[(df['Unix'] >= unix) &
    #            (df['Unix'] < unix + 1 / fs)].index.values.item(), 'Peak'] = 1
    #     df.at[df[(df['Unix'] >= unix) &
    #            (df['Unix'] < unix + 1 / fs)].index.values.item(), 'IBI'] = IBI
    #     df.at[df[(df['Unix'] >= unix) &
    #              (df['Unix'] < unix + 1 / fs)].index.values.item(), 'HR'] = HR
    return df

def get_e4_interval_data(df, seg_size):
    """Get second-by-second HR values from Empatica E4 IBI values."""
    interval_data = pd.DataFrame()
    for s in df['Second'].unique().tolist():
        subset = df.loc[df['Second'].between(s, s + 1)]
        ts = subset['Timestamp'].iloc[0]
        n_peaks = subset['Peak'].sum()
        ibis = subset['IBI']
        r_hr = 1 / subset['HR']
        mean_ibi = pd.Series(ibis, dtype = 'float64').mean()
        mean_hr = 1 / (pd.Series(r_hr, dtype = 'float64').mean())
        df.loc[subset.index, 'Mean IBI'] = mean_ibi
        df.loc[subset.index, 'Mean HR'] = mean_hr

        interval_data = pd.concat([interval_data, pd.DataFrame.from_records([{
            'Second': s,
            'Timestamp': ts,
            'Mean HR': mean_hr,
            'Mean IBI': mean_ibi,
            '# R Peaks': n_peaks}]
        )], ignore_index = True)
    interval_data.insert(0, 'Segment', interval_data['Second'].apply(
        lambda x: (x // seg_size) + 1))
    return interval_data

def get_ppg_peaks(ppg_signal, fs):
    """Get peak locations from an array of PPG data."""
    ppg = ppg_signal.values.flatten()
    wd, m = hp.process(hrdata = ppg, sample_rate = fs,
                       bpmmin = 30, bpmmax = 220)
    peaks = wd['peaklist']
    return peaks