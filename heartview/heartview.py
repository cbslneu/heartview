from zipfile import ZipFile, ZipExtFile
from tqdm import tqdm
from scipy.signal import resample as scipy_resample
from plotly.subplots import make_subplots
from heartview.pipeline.ACC import compute_magnitude
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import pyedflib

class Actiwave:
    """
    A class for convenient pre-processing of data from the Actiwave Cardio
    device.

    Parameters/Attributes
    ---------------------
    file : str
        The path of the Actiwave Cardio device file saved in European
        Data Format (.edf).
    """

    def __init__(self, file):
        """
        Initialize the Actiwave object.

        Parameters
        ----------
        file : str
            The path of the Actiwave Cardio device file saved in European
            Data Format (.edf).
        """
        if not file.endswith(('.edf', '.EDF')):
            raise ValueError(
                'Invalid file path. The `file` parameter must take a string '
                'value ending in \'.EDF\' or \'.edf\'.')
        else:
            self.file = file

    def preprocess(self, time_aligned = False):
        """
        Pre-process electrocardiograph (ECG) and acceleration data from
        an Actiwave Cardio file.

        Parameters
        ----------
        time_aligned : bool, optional
            Whether to time-align ECG and acceleration data based on the
            sampling rate of the ECG data; by default, False.

        Returns
        -------
        tuple or pandas.DataFrame
            If `time_aligned` is False, returns a tuple (`ecg`, `acc`),
            where `ecg` is a DataFrame containing the pre-processed ECG data
            and `acc` is a DataFrame containing the pre-processed X-, Y-, and
            Z-axis acceleration data. If `time_aligned` is True, returns a
            single DataFrame containing time-synced ECG and acceleration
            data according to the ECG data's timestamps.
        """
        f = pyedflib.EdfReader(self.file)
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
        ecg['ECG'] = pd.Series(f.readSignal(ecg_chn[0]) / 1000)
        ecg['Timestamp'] = ecg['Timestamp'].apply(
            lambda t: dt.datetime.utcfromtimestamp(t))

        # Get ACC data
        acc['Timestamp'] = np.arange(start, end, (1 / acc_fs))
        for k, v in acc_sig.items():
            acc[k] = pd.Series(f.readSignal(v))
        acc['Magnitude'] = np.sqrt(acc[['X', 'Y', 'Z']].apply(
            lambda x: x ** 2).sum(axis = 1))
        acc['Timestamp'] = acc['Timestamp'].apply(
            lambda t: dt.datetime.utcfromtimestamp(t))
        f.close()

        if time_aligned:
            resampled = pd.DataFrame()
            for col in ['X', 'Y', 'Z']:
                rs = scipy_resample(acc[col], len(ecg))
                resampled = pd.concat(
                    [resampled, pd.Series(rs, name = col)], axis = 1)
            preprocessed = pd.concat([ecg, resampled], axis = 1)
            return preprocessed
        else:
            return ecg, acc

    def get_ecg_fs(self):
        """
        Get the sampling rate of ECG data from an Actiwave Cardio device.

        Returns
        -------
        fs : int, float
            The sampling rate of the ECG recording.
        """
        f = pyedflib.EdfReader(self.file)
        signal_labels = f.getSignalLabels()
        for chn in range(len(signal_labels)):
            if 'ECG' in signal_labels[chn]:
                ecg_chn = chn
        try:
            fs = f.getSampleFrequency(ecg_chn)
            return fs
        except NameError:
            raise NameError('No ECG channel found.')
        finally:
            f.close()

    def get_acc_fs(self):
        """
        Get the sampling rate of accelerometer data from an Actiwave Cardio
        device.

        Returns
        -------
        fs : int, float
            The sampling rate of the accelerometer recording.
        """
        f = pyedflib.EdfReader(self.file)
        signal_labels = f.getSignalLabels()
        for chn in range(len(signal_labels)):
            if 'X' in signal_labels[chn]:
                acc_chn = chn
        try:
            fs = f.getSampleFrequency(acc_chn)
            return fs
        except NameError:
            raise NameError('No ACC channels found.')
        finally:
            f.close()

# ======================== Empatica E4 Pre-Processing ========================
class Empatica:
    """
    A class to conveniently pre-process data from Empatica devices.

    Parameters/Attributes
    ---------------------
    file : str
        The path of the Empatica archive file with a '.zip' extension.
    """

    def __init__(self, file):
        """
        Initialize the Empatica object.

        Parameters
        ----------
        file : str
            The path of the Empatica archive file with a '.zip' extension.
        """
        if not file.endswith(('.zip', '.ZIP')):
            raise ValueError(
                'Invalid file path. The `file` parameter must take a string '
                'value ending in \'.zip\' or \'.ZIP\'.')
        else:
            self.file = file

    def preprocess(self, time_aligned = False):
        """
        Pre-process all data from the Empatica E4.

        Parameters
        ----------
        time_aligned : bool, optional
            Whether to time-align all data based on the signal with the
            highest sampling rate (i.e. blood volume pulse); by default,
            False.

        Returns
        -------
        dict
            A dictionary containing the recording start time, sampling
            rates, and DataFrames pre-processed from the Empatica E4.

            If `time_aligned` is False:
                'acc' : pandas.DataFrame
                    A DataFrame containing the pre-processed ACC data with
                    corresponding timestamps.
                'bvp' : pandas.DataFrame
                    A DataFrame containing the pre-processed BVP data with
                    corresponding timestamps.
                'eda' : pandas.DataFrame
                    A DataFrame containing the pre-processed EDA data with
                    corresponding timestamps.
                'hr' : pandas.DataFrame
                    A DataFrame containing the pre-processed HR data with
                    corresponding timestamps.
                'ibi' : pandas.DataFrame
                    A DataFrame containing the pre-processed IBI data with
                    corresponding timestamps and seconds elapsed since the
                    start time of the IBI recording.
                'temp' : pandas.DataFrame
                    A DataFrame containing the pre-processed temperature
                    data with corresponding timestamps.
                'start_time' : float
                    The Unix-formatted start time of the E4 recording.
                'bvp_fs' : float
                    The sampling rate of the BVP recording.
                'eda_fs' : float
                    The sampling rate of the EDA recording.

            If `time_aligned` is True:
                'hrv' : pandas.DataFrame
                    A DataFrame containing time-synced BVP, HR, IBI,
                    and acceleration data.
                'eda' : pandas.DataFrame
                    A DataFrame containing time-synced EDA, temperature,
                    and acceleration data.
                'start_time' : float
                    The Unix-formatted start time of the E4 recording.
                'bvp_fs' : float
                    The sampling rate of the BVP recording.
                'eda_fs' : float
                    The sampling rate of the EDA recording.

        Examples
        --------
        >>> from heartview import heartview
        >>> e4_archive = 'Sample_E4_Data.zip'
        >>> E4 = heartview.Empatica(e4_archive)
        >>> ALL_E4_DATA = E4.preprocess()
        """
        with ZipFile(self.file, 'r') as archive:
            e4_files = archive.namelist()
            for file in e4_files:
                if 'ACC' in file:
                    with archive.open(file) as acc_file:
                        acc_data, _, _ = self.get_acc()
                if 'BVP' in file:
                    with archive.open(file) as bvp_file:
                        bvp_data, start_time, bvp_fs = self.get_bvp()
                if 'EDA' in file:
                    with archive.open(file) as eda_file:
                        eda_data, start_time, eda_fs = self.get_eda()
                if 'HR' in file:
                    with archive.open(file) as hr_file:
                        hr_data, _, _ = self.get_hr()
                if 'IBI' in file:
                    with archive.open(file) as ibi_file:
                        ibi_data, _ = self.get_ibi()
                if 'TEMP' in file:
                    with archive.open(file) as temp_file:
                        temp_data, _, _ = self.get_temp()

        if time_aligned:
            # Merge IBI and HR values into BVP data frame
            full_hrv = pd.merge_asof(
                bvp_data, ibi_data.drop(['Seconds'], axis = 1),
                on = 'Timestamp', direction = 'nearest')
            full_hrv = pd.merge_asof(
                full_hrv, hr_data,
                on = 'Timestamp', direction = 'nearest')
            bvp_ts = bvp_data['Timestamp'].values
            ibi_ts = ibi_data['Timestamp'].values
            hr_ts = hr_data['Timestamp'].values
            ibi_insertion_points = np.searchsorted(bvp_ts, ibi_ts) - 1
            hr_insertion_points = np.searchsorted(bvp_ts, hr_ts)
            full_hrv.loc[~np.isin(np.arange(len(full_hrv)),
                                  ibi_insertion_points), 'IBI'] = np.nan
            full_hrv.loc[~np.isin(np.arange(len(full_hrv)),
                                  hr_insertion_points), 'HR'] = np.nan

            # Resample acceleration data to match BVP and EDA sampling rates
            acc_rs = pd.DataFrame()
            acc_cols = ['X', 'Y', 'Z', 'Magnitude']
            for ref_data in [bvp_data, eda_data]:
                acc_rs[acc_cols] = acc_data[acc_cols].apply(
                    lambda a: scipy_resample(a, len(ref_data)))
                if ref_data is bvp_data:
                    full_hrv = pd.merge(full_hrv, acc_rs,
                                        left_index = True, right_index = True)
                else:
                    full_eda = pd.merge(eda_data, acc_rs,
                                        left_index = True, right_index = True)
            return {'hrv': full_hrv,
                    'eda': full_eda,
                    'start_time': start_time,
                    'bvp_fs': bvp_fs,
                    'eda_fs': eda_fs}

        else:
            return {'acc': acc_data,
                    'bvp': bvp_data,
                    'eda': eda_data,
                    'hr': hr_data,
                    'ibi': ibi_data,
                    'temp': temp_data,
                    'start_time': start_time,
                    'bvp_fs': bvp_fs,
                    'eda_fs': eda_fs}

    def get_acc(self):
        """
        Get the pre-processed acceleration data and its start time and
        sampling rate from the Empatica E4.

        Returns
        -------
        tuple
            A tuple containing the pre-processed blood volume pulse (BVP)
            data and its corresponding start time and sampling rate.

            acc : pandas.DataFrame
                A DataFrame containing the pre-processed BVP data with
                corresponding timestamps.
            acc_start : float
                The Unix-formatted start time of the BVP recording.
            acc_fs : int
                The sampling rate of the BVP data.
        """
        with ZipFile(self.file, 'r') as archive:
            e4_files = archive.namelist()
            acc_file = None
            for file in e4_files:
                if 'ACC' in file:
                    acc_file = file
                    break
            if acc_file is None:
                raise ValueError('No "ACC.csv" file found.')
            with archive.open(file) as acc_file:
                acc, acc_start, acc_fs = self._get_e4_data(
                    acc_file, name = ['X', 'Y', 'Z'])
                acc = acc.apply(lambda x: (x / 64) * 9.81
                                if x.name != 'Timestamp' else x)
                acc['Magnitude'] = compute_magnitude(
                    acc['X'], acc['Y'], acc['Z'])
            return acc, acc_start, acc_fs

    def get_bvp(self):
        """
        Get the raw BVP data and its start time and sampling rate from the
        Empatica E4.

        Returns
        -------
        tuple
            A tuple containing the pre-processed blood volume pulse (BVP)
            data and its corresponding start time and sampling rate.

            bvp : pandas.DataFrame
                A DataFrame containing the pre-processed BVP data with
                corresponding timestamps.
            bvp_start : float
                The Unix-formatted start time of the BVP recording.
            bvp_fs : int
                The sampling rate of the BVP data.
        """
        with ZipFile(self.file, 'r') as archive:
            e4_files = archive.namelist()
            bvp_file = None
            for file in e4_files:
                if 'BVP' in file:
                    bvp_file = file
                    break
            if bvp_file is None:
                raise ValueError('No "BVP.csv" file found.')
            with archive.open(bvp_file) as bvp_file:
                bvp, bvp_start, bvp_fs = self._get_e4_data(
                    bvp_file, name = 'BVP')
            return bvp, bvp_start, bvp_fs

    def get_eda(self):
        """
        Get the pre-processed electrodermal activity (EDA) data and its
        recording start time and sampling rate from the Empatica E4.

        Returns
        -------
        tuple
            A tuple containing the pre-processed EDA data and its
            corresponding start time and sampling rate.

            eda : pandas.DataFrame
                A DataFrame containing the pre-processed EDA data with
                corresponding timestamps.
            eda_start : float
                The Unix-formatted start time of the EDA recording.
            eda_fs : int
                The sampling rate of the EDA data.
        """
        with ZipFile(self.file, 'r') as archive:
            e4_files = archive.namelist()
            eda_file = None
            for file in e4_files:
                if 'EDA' in file:
                    eda_file = file
                    break
            if eda_file is None:
                raise ValueError('No "EDA.csv" file found.')
            with archive.open(eda_file) as eda_file:
                eda, eda_start, eda_fs = self._get_e4_data(
                    eda_file, name = 'EDA')
            return eda, eda_start, eda_fs

    def get_hr(self):
        """
        Get the pre-processed heart rate (HR) data, start time of the
        first HR measurement, and sampling rate from the Empatica E4.

        Returns
        -------
        hr : pandas.DataFrame
            A DataFrame containing the pre-processed HR data with
            corresponding timestamps.
        hr_start : float
            The Unix-formatted start time of the HR measurements.
        hr_fs : int
            The sampling rate of the BVP data.
        """
        with ZipFile(self.file, 'r') as archive:
            e4_files = archive.namelist()
            hr_file = None
            for file in e4_files:
                if 'HR' in file:
                    hr_file = file
                    break
            if hr_file is None:
                raise ValueError('No "HR.csv" file found.')
            with archive.open(file) as hr_file:
                hr, hr_start, hr_fs = self._get_e4_data(
                    hr_file, name = 'HR')
            return hr, hr_start, hr_fs

    def get_ibi(self):
        """
        Get the pre-processed interbeat interval (IBI) data and the start
        time of the first interval from the Empatica E4.

        Returns
        -------
        ibi : pandas.DataFrame
            A DataFrame containing the pre-processed IBI data with
            corresponding timestamps.
        ibi_start : int
            The Unix-formatted start time of the IBI data.
        """
        with ZipFile(self.file, 'r') as archive:
            e4_files = archive.namelist()
            ibi_file = None
            for file in e4_files:
                if 'IBI' in file:
                    ibi_file = file
                    break
            if ibi_file is None:
                raise ValueError('No "IBI.csv" file found.')
            with archive.open(file) as ibi_file:
                ibi = pd.read_csv(ibi_file, header = 0,
                                  names = ['Seconds', 'IBI'])
                ibi_file.seek(0)
                ibi_start = self._get_e4_start_time(ibi_file)
                ibi['IBI'] *= 1000
                ibi.insert(
                    0, 'Timestamp', (ibi['Seconds'] + ibi_start).apply(
                        lambda t: dt.datetime.utcfromtimestamp(t)))
            return ibi, ibi_start

    def get_temp(self):
        """
        Get the raw skin temperature data and its recording start time and
        sampling rate from the Empatica E4.

        Returns
        -------
        tuple
            A tuple containing the pre-processed skin temperature data and its
            corresponding start time and sampling rate.

            temp : pandas.DataFrame
                A DataFrame containing the pre-processed temperature data with
                corresponding timestamps.
            temp_start : float
                The Unix-formatted start time of the temperature recording.
            temp_fs : int
                The sampling rate of the temperature data.
        """
        with ZipFile(self.file, 'r') as archive:
            e4_files = archive.namelist()
            temp_file = None
            for file in e4_files:
                if 'TEMP' in file:
                    temp_file = file
                    break
            if temp_file is None:
                raise ValueError('No "TEMP.csv" file found.')
            with archive.open(temp_file) as temp_file:
                temp, temp_start, temp_fs = self._get_e4_data(
                    temp_file, name = 'Temp')
            return temp, temp_start, temp_fs

    def get_e4_beats(self, bvp_data, ibi_data, start_time,
                     show_progress = True):
        """
        Get locations of beats from Empatica E4 interbeat interval (IBI)
        data relative to its blood volume pulse (BVP) data.

        Parameters
        ----------
        bvp_data : pandas.DataFrame
            A DataFrame containing the Empatica E4 BVP data, outputted from
            `Empatica.preprocess()`.
        ibi_data : pandas.DataFrame
            A DataFrame containing the Empatica E4 IBI data, outputted from
            `Empatica.preprocess()`.
        start_time : int
            The Unix timestamp of the recording start time.
        show_progress : bool, optional
            Whether to display a progress bar while the function runs; by
            default, True.

        Returns
        -------
        e4_beats : list
            A list containing the indices of beats extracted from IBI data of
            the Empatica E4.
        """
        ibi = ibi_data.copy()
        bvp = bvp_data.copy()
        ibi['Unix'] = ibi['Seconds'] + start_time
        ibi['Timestamp'] = ibi['Unix'].apply(
            lambda t: dt.datetime.utcfromtimestamp(t))
        bvp['Timestamp'] = pd.to_datetime(bvp['Timestamp'])
        e4_beats = []
        for t in tqdm(ibi['Timestamp'], disable = not show_progress):
            time_diff = np.abs(bvp['Timestamp'] - t)
            closest_ix = time_diff.idxmin()
            e4_beats.append(closest_ix)
        return e4_beats

    def _get_e4_data(self, file, name):
        """Get pre-processed data from an Empatica E4 file."""
        if not isinstance(name, list) and not isinstance(name, str):
            raise ValueError('The `name` parameter must take either a string '
                             'or a list of strings.')
        else:
            if isinstance(name, list):
                col_name = name
            else:
                col_name = [name]
        data = pd.read_csv(file, header = 1, names = col_name)
        if isinstance(file, str):
            fs = self._get_e4_fs(file)
            start_time = self._get_e4_start_time(file)
        else:
            if hasattr(file, 'seek'):
                file.seek(0)
                fs = self._get_e4_fs(file)
                file.seek(0)
                start_time = self._get_e4_start_time(file)
        timestamps = pd.date_range(
            start = pd.to_datetime(start_time, unit = 's'),
            periods = len(data), freq = f'{1 / fs}S')
        timestamps = pd.Series(timestamps, name = 'Timestamp')
        data = pd.merge(timestamps, data,
                        left_index = True, right_index = True)
        return data, start_time, fs

    def _get_e4_fs(self, file):
        """Get the sampling rate from an Empatica E4 file."""
        contents = pd.read_csv(file, header = None, nrows = 2, usecols = [0])
        fs = contents.iloc[1].item()
        return fs

    def _get_e4_start_time(self, file):
        """Get the Unix-formatted start time of an Empatica E4 recording."""
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

# ======================== Other Data Pre-Processing =========================
def get_duration(data, fs, unit = 'sec'):
    """
    Get the duration of a signal.

    Parameters
    ----------
    data : array_like
        An array or DataFrame containing the signal.
    fs : int
        The sampling rate of the data.
    unit : str
        The unit in which the duration should be calculated; by default,
        in seconds (`sec`).

    Returns
    -------
    dur : float
        The duration of the signal.
    """

    dur = len(data) / fs
    if unit not in ['sec', 's', 'min', 'm', 'hour', 'h']:
        raise ValueError('The `unit` parameter must take \'sec\', \'min\', '
                         'or \'hour\'.')
    else:
        if unit in ('min', 'm'):
            return round((dur / 60), 2)
        if unit == ('hour', 'h'):
            return round(((dur / 60) / 60), 2)
    return round(dur, 2)

def segment_data(data, fs, seg_size):
    """
    Segment data into specific window sizes.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data to be segmented.
    fs : int
        The sampling rate of the data.
    seg_size : int
        The window size, in seconds, into which the data should be
        segmented.

    Returns
    -------
    df : pd.DataFrame
        The original DataFrame with data segmented with labels in a
        'Segment' column.
    """
    df = data.copy()
    df.insert(0, 'Segment', 0)
    segment = 1
    for n in range(0, len(df), int(seg_size * fs)):
        df.loc[n:(n + int(seg_size * fs)), 'Segment'] = segment
        segment += 1
    return df

def compute_ibis(data, fs, beats_ix, ts_col = None):
    """
    Compute interbeat intervals from beat locations in electrocardiograph
     (ECG) or photoplethysmograph (PPG) data.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the pre-processed ECG/PPG data.
    fs : int
        The sampling rate of the ECG/PPG data.
    beats_ix : array_like
        An array of indices corresponding to beat occurrences.
    ts_col : str
        The name of the column in `data` containing timestamp values; by
        default, None.

    Returns
    -------
    ibi : pd.DataFrame
        A DataFrame containing timestamps and IBI values.
    """

    df = data.copy()
    ibis = (np.diff(beats_ix) / fs) * 1000
    if ts_col is not None:
        ibi = df[[ts_col]].copy()
    else:
        ibi = pd.DataFrame({'Sample': np.arange(len(df)) + 1})
    for n, ix in enumerate(beats_ix[1:]):
        ibi.loc[ix, 'IBI'] = ibis[n]
    return ibi

def plot_cardio_signals(signal, fs, ibi, signal_type, x = 'Timestamp',
                        y = 'Filtered', acc = None, seg_num = 1,
                        seg_size = 60, title = None):
    """
    Create subplots of the electrocardiograph (ECG) or photoplethysmograph
    (PPG), interbeat interval (IBI), and acceleration data (if any).

    Parameters
    ----------
    signal : pandas.DataFrame
        A DataFrame containing the pre-processed ECG or PPG data with beat
        and artifact occurrences in a "Beat" and "Artifact" column.
    fs : int
        The sampling rate of the ECG or PPG data.
    ibi : pandas.DataFrame
        A DataFrame containing IBI values in an "IBI" column.
    signal_type : str
        The type of cardiovascular data being plotted. This must be either
        'ECG' or 'PPG'.
    x : str, optional
        The name of the column of values in the `signal` DataFrame to plot
        along the x-axis; by default, 'Timestamp'.
    y : str, optional
        The column name of values to plot along the y-axis; by default,
        'Filtered'.
    acc : pandas.DataFrame, optional
        A DataFrame containing pre-processed acceleration data with
        magnitude values in a "Magnitude" column.
    seg_num : int
        The segment to plot.
    seg_size : int
        The size of the segment, in seconds; by default, 60.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A figure containing subplots of ECG or PPG data with beat annotations
        and its corresponding IBI data.

    See Also
    --------
    heartview.compute_ibis : Compute IBIs in a DataFrame time-aligned to its
    corresponding cardiovascular data.
    """

    seg_start = int((seg_num - 1) * seg_size * fs)
    seg_end = int(seg_start + (fs * seg_size))
    for df in [signal, ibi]:
        df[x] = pd.to_datetime(df[x])
    signal_segment = signal.iloc[seg_start:seg_end]
    ibi_segment = ibi.iloc[seg_start:seg_end].dropna()

    x_array = signal_segment[x]
    if not pd.api.types.is_datetime64_any_dtype(x_array):
        artifact_hover = '<b>Potential Artifact</b> <extra></extra>'
        beat_hover = '<b>Beat</b> <extra></extra>'
    else:
        artifact_hover = '<b>Potential Artifact</b>: %{x|%H:%M:%S.%3f} ' \
                         '<extra></extra>'
        beat_hover = '<b>Beat</b>: %{x|%H:%M:%S.%3f} <extra></extra>'
    if signal_type == 'PPG' or signal_type == 'BVP':
        y_axis = 'bvp'
    else:
        y_axis = 'mV'

    if acc is not None:
        fig = make_subplots(rows = 3, cols = 1,
                            shared_xaxes = True,
                            vertical_spacing = 0.02,
                            row_heights = [0.25, 0.50, 0.25])

        # ACC subplot
        acc = scipy_resample(acc['Magnitude'], len(signal))
        acc_segment = acc[seg_start:seg_end]
        fig.add_trace(
            go.Scatter(
                x = x_array,
                y = acc_segment,
                name = 'ACC',
                line = dict(color = 'forestgreen', width = 1.5),
                hovertemplate = '<b>ACC</b>: %{y:.2f} m/s² <extra></extra>'),
            row = 1, col = 1)
        fig.update_yaxes(
            title_text = 'm/s²',
            title_standoff = 5,
            row = 1, col = 1,
            showgrid = True,
            gridwidth = 0.5,
            gridcolor = 'lightgrey',
            griddash = 'dot',
            tickcolor = 'grey',
            linecolor = 'grey')

        # ECG/PPG subplot
        fig.add_trace(
            go.Scatter(
                x = x_array,
                y = signal_segment[y],
                name = signal_type,
                showlegend = True,
                line = dict(color = '#3562bd', width = 1.5),
                hovertemplate = f'<b>{signal_type}:</b> %{{y:.2f}} {y_axis} '
                                f'<extra></extra>'),
            row = 2, col = 1)
        fig.update_yaxes(
            title_text = y_axis,
            title_standoff = 5,
            row = 2, col = 1,
            showgrid = True,
            gridwidth = 0.5,
            gridcolor = 'lightgrey',
            griddash = 'dot',
            tickcolor = 'grey',
            linecolor = 'grey')

        # IBI subplot
        fig.add_trace(
            go.Scatter(
                x = ibi_segment[x],
                y = ibi_segment['IBI'],
                name = 'IBI',
                line = dict(color = '#eb4034', width = 1.5),
                hovertemplate = '<b>IBI</b>: %{y:.2f} ms <extra></extra>'),
            row = 3, col = 1)
        fig.update_yaxes(
            title_text = 'ms',
            row = 3, col = 1,
            title_standoff = 1,
            showgrid = True,
            gridwidth = 0.5,
            gridcolor = 'lightgrey',
            griddash = 'dot',
            tickcolor = 'grey',
            linecolor = 'grey')

        # Detected beats
        fig.add_trace(
            go.Scatter(
                x = signal_segment.loc[signal_segment.Beat == 1, x],
                y = signal_segment.loc[signal_segment.Beat == 1, y],
                name = 'Detected Beat',
                showlegend = True,
                mode = 'markers',
                marker = dict(color = '#f9c669', size = 6),
                hovertemplate = beat_hover),
            row = 2, col = 1)

        # Artifactual beats
        fig.add_trace(
            go.Scatter(
                x = signal_segment.loc[signal_segment.Artifact == 1, x],
                y = signal_segment.loc[signal_segment.Artifact == 1, y],
                name = 'Potential Artifact',
                showlegend = True,
                mode = 'markers',
                marker = dict(color = 'red'),
                hovertemplate = artifact_hover),
            row = 2, col = 1)

    else:
        fig = make_subplots(rows = 2, cols = 1,
                            shared_xaxes = True,
                            vertical_spacing = 0.02,
                            row_heights = [0.6, 0.4])

        # ECG/PPG subplot
        fig.add_trace(
            go.Scatter(
                x = x_array,
                y = signal_segment[y],
                name = signal_type,
                showlegend = True,
                line = dict(color = '#3562bd', width = 1.5),
                hovertemplate = f'<b>{signal_type}:</b> %{{y:.2f}} {y_axis} '
                                f'<extra></extra>'),
            row = 1, col = 1)
        fig.update_yaxes(
            title_text = y_axis,
            row = 1, col = 1,
            title_standoff = 5,
            showgrid = True,
            gridwidth = 0.5,
            gridcolor = 'lightgrey',
            griddash = 'dot',
            tickcolor = 'grey',
            linecolor = 'grey')

        # IBI subplot
        fig.add_trace(
            go.Scatter(
                x = ibi_segment[x],
                y = ibi_segment['IBI'],
                name = 'IBI',
                line = dict(color = '#eb4034', width = 1.5),
                hovertemplate = '<b>IBI</b>: %{y:.2f} ms <extra></extra>'),
            row = 2, col = 1)
        fig.update_yaxes(
            title_text = 'ms',
            row = 2, col = 1, title_standoff = 1,
            showgrid = True,
            gridwidth = 0.5,
            gridcolor = 'lightgrey',
            griddash = 'dot',
            tickcolor = 'grey',
            linecolor = 'grey')

        # Detected beats
        fig.add_trace(
            go.Scatter(
                x = signal_segment.loc[signal_segment.Beat == 1, x],
                y = signal_segment.loc[signal_segment.Beat == 1, y],
                name = 'Detected Beat',
                showlegend = True,
                mode = 'markers',
                marker = dict(color = '#f9c669', size = 6),
                hovertemplate = beat_hover),
            row = 1, col = 1)

        # Artifactual beats
        fig.add_trace(
            go.Scatter(
                x = signal_segment.loc[signal_segment.Artifact == 1, x],
                y = signal_segment.loc[signal_segment.Artifact == 1, y],
                name = 'Potential Artifact',
                showlegend = True,
                mode = 'markers',
                marker = dict(color = 'red'),
                hovertemplate = artifact_hover),
            row = 1, col = 1)

    # Format shared x-axis
    x_min = signal_segment[x].min()
    x_max = signal_segment[x].max()
    fig.update_xaxes(
        tickfont = dict(size = 14),
        tickcolor = 'grey',
        linecolor = 'grey',
        range = [x_min, x_max])

    # Format figure
    fig.update_layout(
        height = 450,
        title_text = title,
        template = 'simple_white',
        font = dict(family = 'Poppins', color = 'black'),
        legend = dict(
            font = dict(size = 16),
            orientation = 'h',
            yanchor = 'bottom',
            y = 1.05,
            xanchor = 'right',
            x = 1.0),
        annotations = [dict(
            text = x.capitalize(),
            x = 0.5,
            y = -0.22,
            showarrow = False,
            xref = 'paper',
            yref = 'paper',
            font = dict(size = 16)
        )],
        margin = dict(l = 20, r = 20, t = 60, b = 70)
    )
    return fig

def plot_signal(df, x, y, fs, seg_size = 60, segment = 1, n_segments = 1,
                signal_type = None, peaks = None):
    """
    Visualize a signal.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the signal data.
    x : str
        The column containing the x-axis value (e.g., `'Time'`).
    y : str, list
        The column(s) of the signal data (y-axis values).
    fs : int, float
        The sampling rate.
    seg_size : int
        The size of the segment, in seconds; by default, 60.
    segment : int, float, None
        The segment number; by default, 1. For example, segment `1`
        denotes the first segment of the recording. This argument can also
        be set to `None` if `df` contains a 'Segment' column.
    n_segments : int, float
        The number of segments to be visualized; by default, 1.
    signal_type : str
        The type of signal being plotted (i.e., 'ecg', 'bvp', 'acc',
        'ibi'); by default, None.
    peaks : str
        The column containing peak occurrences, i.e., a sequence of
        `0` and/or `1` denoting False or True occurrences of peaks.
        By default, peaks will be plotted on the first trace.

    Returns
    -------
    fig : matplotlib.axes.AxesSubplot
        The signal visualization.
    """

    if segment is None and \
            'segment' in [c.lower() for c in df.columns.tolist()]:
        seg = df.loc[(df.Segment >= 1) & (df.Segment <= 2)]
    else:
        start = int(segment - 1) * seg_size * fs
        end = int(((segment - 1) + n_segments) * seg_size * fs)
        seg = df.iloc[start:end]

    # Set plotting parameters
    plt.rcParams['font.size'] = 14
    palette1 = {'blue': '#4c73c2',
                'red': '#eb4034',
                'green': '#63b068',
                'grey': '#bdbdbd'}
    palette2 = ['#ec2049', '#176196', '#f7db4f', '#63b068']

    # Set up the figure
    fig = go.Figure()

    # Plot a single signal
    if not isinstance(y, list):
        fig.add_trace(go.Scatter(
            x = seg[x],
            y = seg[y],
            mode = 'lines',
            hovertemplate = '%{x}' + '<br>%{y:.2f}' + '<extra></extra>',
            name = f'{y}'))

        # Add peaks
        if peaks != None:
            fig.add_trace(go.Scatter(
                x = seg[x],
                y = np.where(seg[peaks] == 1, seg[y], np.nan),
                mode = 'markers',
                marker = dict(size = 8, color = 'gold', line_width = 1),
                hovertemplate = '<b>Peak</b>: %{y} <extra></extra>',
                name = 'Peaks'))
        fig.update_layout(yaxis_title = y)

    # Plot multiple signals
    else:
        for yval in range(len(y)):
            fig.add_trace(go.Scatter(
                x = seg[x],
                y = seg[y[yval]],
                mode = 'lines',
                line = dict(color = palette2[yval]),
                hovertemplate = '%{x}' + '<br>%{y:.2f}' + '<extra></extra>',
                name = f'{y[yval]}'))

        # Add peaks
        if peaks is not None:
            fig.add_trace(go.Scatter(
                x = seg[x],
                y = np.where(seg[peaks] == 1, seg[y[0]], np.nan),
                mode = 'markers',
                marker = dict(size = 8, color = 'gold', line_width = 1),
                hovertemplate = '<b>Peak</b>: %{y} <extra></extra>',
                name = 'Peaks'))

    # Format the plot
    fig.update_layout(
        xaxis_title = x,
        template = 'simple_white',
        height = 300,
        margin = dict(l = 10, r = 30, b = 50, t = 50, pad = 3)
    )
    # Label axes and set trace colors according to signal type
    if signal_type == 'ecg' or signal_type == 'bvp':
        if isinstance(y, list):
            for d in range(len(fig.data)):
                fig.data[d].line.color = palette2[d]
            return fig.update_layout(yaxis_title = signal_type.upper())
        else:
            return fig.update_traces(
                line_color = palette1['blue']).update_layout(yaxis_title = y)
    elif signal_type == 'acc':
        if isinstance(y, list):
            for d in range(len(fig.data)):
                fig.data[d].line.color = palette2[d]
            return fig.update_layout(yaxis_title = 'm/s<sup>2</sup>')
        else:
            return fig.update_traces(
                line_color = palette1['green']).update_layout(
                yaxis_title = 'm/s<sup>2</sup>')
    elif signal_type == 'ibi':
        if isinstance(y, list):
            for d in range(len(fig.data)):
                fig.data[d].line.color = palette2[d]
            return fig.update_layout(yaxis_title = 'ms')
        else:
            return fig.update_traces(
                line_color = palette1['red']).update_layout(yaxis_title = 'ms')
    else:
        return fig

def plot_ibi_from_ecg(df, x, y, segment, n_segments):
    """
    Visualize an IBI series generated from ECG data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the signal data.
    x : str
        The column containing the x-axis value (e.g., `'Time'`).
    y : str
        The column containing the IBI series (e.g., `'IBI'`).
    segment : int, float
        The segment number. For example, segment `1` denotes the first
        segment of the recording.
    n_segments : int, float
        The number of segments to be visualized; by default, 1.

    Returns
    -------
    fig : matplotlib.axes.AxesSubplot
        The IBI series plot.
    """
    start = int(segment)
    end = round(segment + n_segments)
    seg = df.loc[df['Segment'].between(start, end, inclusive = 'both')]
    plt.rcParams['font.size'] = 14

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = seg[x],
        y = seg[y],
        mode = 'lines',
        marker = dict(color = '#eb4034'),
        hovertemplate = '%{x}' + '<br>%{y:.2f} ms' + '<extra></extra>',
        name = f'{y}'))

    ymin = np.nanmin(seg[y].values.flatten()) * 0.95
    ymax = np.nanmax(seg[y].values.flatten()) * 1.05
    fig.update_layout(
        yaxis_range = (ymin, ymax),
        yaxis_title = 'IBI (ms)',
        xaxis_title = x,
        template = 'simple_white',
        height = 300,
        margin = dict(l = 50, r = 10, b = 50, t = 30, pad = 10)
    )
    fig.update_yaxes(
        title_standoff = 10)
    return fig