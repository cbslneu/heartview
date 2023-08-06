from os import listdir, path
from scipy.signal import hilbert, filtfilt, butter, iirnotch
from itertools import islice
from dateutil import parser
import ast
import pyedflib
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px

def get_fs(edf):
    """Get ECG sampling rate from an Actiwave Cardio device."""
    f = pyedflib.EdfReader(edf)
    signal_labels = f.getSignalLabels()
    for chn in range(len(signal_labels)):
        if 'ECG' in signal_labels[chn]:
            ecg_chn = chn
    fs = f.getSampleFrequency(ecg_chn)
    f.close()
    return fs

def read_actiwave(edf):
    """
    Read ECG and acceleration data from an Actiwave Cardio device.

    Parameters
    ----------
    edf : str
        The path to the EDF file.

    Returns
    -------
    ecg : pd.DataFrame
        The ECG data containing 'Timestamp' and 'mV' columns.
    acc : pd.DataFrame
        The acceleration data containing 'Timestamp', 'X', 'Y', and 'Z'
        columns.

    """
    f = pyedflib.EdfReader(edf)
    start = dt.datetime.timestamp(f.getStartdatetime())
    end = start + f.getFileDuration()
    ecg, acc = pd.DataFrame(), pd.DataFrame()
    signal_labels = f.getSignalLabels()
    ecg_chn = [i for i in range(len(signal_labels))
               if 'ECG' in signal_labels[i]]
    acc_chn = [i for i in range(len(signal_labels))
               if 'X' in signal_labels[i]
               or 'Y' in signal_labels[i]
               or 'Z' in signal_labels[i]]
    acc_sig = dict(zip(['X', 'Y', 'Z'], acc_chn))

    ecg_fs = f.getSampleFrequency(ecg_chn[0])
    acc_fs = f.getSampleFrequency(acc_chn[0])

    # Get ECG data
    ecg['Timestamp'] = np.arange(start, end, (1 / ecg_fs))
    ecg['mV'] = pd.Series(f.readSignal(ecg_chn[0]) / 1000)
    ecg['Timestamp'] = ecg['Timestamp'].apply(
        lambda t: dt.datetime.fromtimestamp(t))

    # Get acceleration data
    acc['Timestamp'] = np.arange(start, end, (1/ acc_fs))
    for k, v in acc_sig.items():
        acc[k] = pd.Series(f.readSignal(v))
    acc['Magnitude'] = np.sqrt(acc[['X', 'Y', 'Z']].apply(
        lambda x: x ** 2).sum(axis = 1))
    acc['Timestamp'] = acc['Timestamp'].apply(
        lambda t: dt.datetime.fromtimestamp(t))

    f.close()

    return ecg, acc

# RUNNING MEAN FILTER
def _running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# BASELINE AND MUSCLE NOISE FILTER
def baseline_muscle_filter(data, lowcut, highcut, fs, order = 4):
    """Filter out baseline wander and muscle noise.

    Parameters
    ----------
    data : array_like
        An array containing the noisy signal data.
    lowcut : int, float
        The lower cut-off frequency.
    highcut : int, float
        The higher cut-off frequency.
    fs : int, float
        The sampling rate of the signal recording.
    order : int
        The filter order, i.e., the number of samples required to
        produce the desired filtered output; by default, 4.
    """
    if fs <= 90:
        nyq = 0.91 * fs
    else:
        nyq = 0.5 * fs
    high = highcut / nyq
    low = lowcut / nyq
    b, a = butter(order, [low, high], btype = 'bandpass')
    filtered = filtfilt(b, a, data)
    return filtered

# POWERLINE INTERFERENCE NOTCH FILTER AT 60 Hz
def powerline_int_filter(data, fs, q, freq = 60):
    """Filter out powerline interference at a specified frequency.

    Parameters
    ----------
    data : array_like
        An array containing the noisy signal data.
    fs : int, float
        The sampling rate of the signal recording.
    q : float
        The quality factor, i.e., how narrow or wide the stopband is for
        a notch filter. A higher quality factor indicates a narrower
        bandpass.
    freq : int, float
        The frequency to filter out (e.g., 50 or 60 Hz); by default, 60 Hz.

    Returns
    -------
    filtered : array_like
        An array containing the filtered signal data.
    """
    b, a = iirnotch(freq, q, fs)
    filtered = filtfilt(b, a, data)
    return filtered

# SHANNON ENERGY R PEAK DETECTION FUNCTION
# Credit: https://github.com/nsunami/Shannon-Energy-R-Peak-Detection
def detect_rpeaks(df, signal, fs):
    """Detect locations of R peaks with Shannon energy & smooth
    envelope extraction.

    Parameters
    ----------
    df : pd.DataFrame
        The data frame containing the pre-processed ECG data.
    signal : str
        The name of the column containing the filtered ECG data.
    fs : int, float
        The sampling rate of the signal recording.

    Returns
    -------
    id_maxes : np.array
        Array containing the index locations of all detected peaks.
    """
    array = np.array(df[signal])
    # Eq. 1 - Differentiate the filtered signal
    dn = (np.append(array[1:], 0) - array)

    # Eq. 2 - Normalize the differentiated signal
    dtn = dn / (np.max(abs(dn)))

    # Eq. 3 - Compute absolute value
    an = abs(dtn)

    # Eq. 4 - Compute energy value
    en = an ** 2

    # Eq. 5 - Compute Shannon entropy value
    sen = -abs(dtn) * np.log10(abs(dtn))

    # Eq. 6 - Compute Shannon energy value
    sn = -(dtn ** 2) * np.log10(dtn ** 2)

    # Zero-phase filtering, Running mean
    # https://dsp.stackexchange.com/a/27317
    # 360 samples/s : 55 samples (length)
    # 500 samples/s : 76 samples
    # Normal QRS duration is .12 sec. 500 * .12 = 60
    window_len = int(fs / (360 / 55))
    # window_len = 79

    # Moving average of the filtered signal
    sn_f = np.insert(_running_mean(sn, window_len), 0, [0] * (window_len - 1))

    # Hilbert Transformation
    zn = np.imag(hilbert(sn_f))

    # Moving Average of the Hilbert Transformed Signal
    # 2.5 sec from Manikanda (900 samples)
    # 2.5 sec in 500 Hz == 1250 samples
    ma_len = int(fs * 2.5)
    zn_ma = np.insert(_running_mean(zn, ma_len), 0, [0] * (ma_len - 1))

    # Get the difference between the Hilbert signal and the MA filtered signal
    zn_ma_s = zn - zn_ma

    # Look for zero crossings: https://stackoverflow.com/a/28766902/6205282
    # Original paper: +/- 25 frames from the ref point (360Hz)
    # if 500Hz---35 frames. With some trials, it seems that ~60 is good.

    # Find locations of zero-crossings (from negative to positive)
    idx = np.argwhere(np.diff(np.sign(zn_ma_s)) > 0).flatten().tolist()

    # Prepare a container for windows
    idx_search = []
    id_maxes = np.empty(0, dtype = int)
    search_window_half = round(fs * .12)  # <------------ Parameter
    for n in idx:
        lows = np.arange(n - search_window_half, n)
        highs = np.arange(n + 1, (n + search_window_half + 1))
        if highs[-1] > len(array):
            highs = np.delete(highs,
                              np.arange(np.where(highs == len(array))[0],
                                        len(highs)))
        ecg_window = np.concatenate((lows, [n], highs))
        idx_search.append(ecg_window)
        ecg_window_wave = array[ecg_window]
        peak_loc = ecg_window[
            np.where(ecg_window_wave == np.max(ecg_window_wave))[0]]
        if peak_loc.item() > 0:
            id_maxes = np.append(id_maxes, peak_loc)

    # df.loc[id_maxes, 'Peak'] = 1

    return id_maxes

# IBI EXTRACTION
def compute_ibis(data, ts_col, fs, seg_size, peaks_ix):
    """Compute interbeat intervals from peak locations in ECG data.

    Parameters
    ----------
    data : pd.DataFrame
        The data frame containing the ECG data and a timestamp column.
    ts_col : str
        The name of the column containing timestamp values.
    fs : int
        The sampling rate.
    seg_size : int
        The window size of each segment.
    peaks_ix : array_like
        An array of indices corresponding to peak occurrences.

    Returns
    -------
    ibi : pd.DataFrame
        A data frame containing IBI values.
    """
    data.index = data.index.astype(int)
    data['Segment'] = pd.Series(data.index).apply(
        lambda x: (((x // fs) + 1) // seg_size) + 1)

    seg = 1
    chunk = int(fs * seg_size)
    for n in range(0, len(data), chunk):
        data.loc[n:(n + chunk - 1), 'Segment'] = seg
        seg += 1
    data[ts_col] = pd.to_datetime(data[ts_col])
    ts = data.loc[peaks_ix, [ts_col, 'Segment']].reset_index(drop = True)
    segs = []
    ibis = []
    for n in range(1, len(ts)):
        seg = ts.loc[n, 'Segment'].item()
        segs.append(seg)
        nn_s = (data.loc[n, ts_col] - data.loc[n - 1, ts_col]).total_seconds()
        nn_ms = nn_s * 1000
        ibis.append(nn_ms)
    ibi = pd.DataFrame({'Segment': segs,
                        'Timestamp': ts.loc[1:, ts_col],
                        'IBI': ibis}).reset_index(drop = True)
    return ibi

# SECOND-BY-SECOND PRE-PROCESSING
def get_seconds(df, peaks_col, fs, seg_size):
    """Get second-by-second HR, IBI, and peak counts from ECG data.

    Parameters
    ----------
    df : pd.DataFrame
        The data frame containing the ECG data and a timestamp column.
    peaks_col : str
        The name of the column containing peak instances.
    fs : int
        The sampling rate.
    seg_size : int
        The window size of each segment.

    Returns
    -------
    interval_data : pd.DataFrame
        A data frame containing IBI values.
    """

    # Add 'segment' and 'second' columns
    df.index = df.index.astype(int)
    df['Second'] = pd.Series(df.index).apply(lambda x: (x // fs) + 1)
    df['Segment'] = df['Second'].apply(lambda x: (x // seg_size) + 1)

    # Create interval data
    interval_data = pd.DataFrame()
    for s in df['Second'].unique().tolist():

        # Count peaks in each second
        n_peaks = df.loc[df['Second'] == s, peaks_col].sum()

        # Get timestamp of each second
        ts = df.loc[df['Second'] == s, 'Timestamp'].values[0]

        # Look at current second and one second ahead
        subset = df.loc[df['Second'].between(s, s + 1)]

        # Get the indices of the peaks in the subset
        peak_indices = subset[subset[peaks_col] == 1].index.values

        # Get the IBIs and HRs
        ibis = []
        heartrates = []
        for ix in range(len(peak_indices) - 1):
            ibi = ((peak_indices[ix + 1] - peak_indices[ix]) / fs) * 1000
            hr = 60000 / ibi
            r_hr = 1 / hr
            ibis.append(ibi)
            heartrates.append(r_hr)

        # Add mean HR and IBI values (HR is calculated with harmonic mean)
        mean_ibi = pd.Series(ibis, dtype = 'float64').mean()
        mean_hr = 1 / (pd.Series(heartrates, dtype = 'float64').mean())

        interval_data = pd.concat([
            interval_data, pd.DataFrame.from_records([{
                'Second': s,
                'Timestamp': ts,
                'Mean HR': mean_hr,
                'Mean IBI': mean_ibi,
                '# R Peaks': n_peaks}])],
            ignore_index = True)

    interval_data.insert(0, 'Segment', interval_data['Second'].apply(
        lambda x: (x // seg_size) + 1))
    interval_data.insert(
        interval_data.shape[1], 'Invalid', interval_data['Mean HR'].apply(
            lambda x: 1 if x < 30 or x > 220 else 0))
    return interval_data
