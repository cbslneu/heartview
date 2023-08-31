from os import listdir, path, remove, makedirs
from pathlib import Path
from shutil import rmtree
from zipfile import ZipFile
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pyedflib
import json
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def _clear_temp():
    temp = './temp'
    temp_contents = [f for f in listdir(temp) if
                     not f.startswith('.') and f != '00.txt']
    if len(temp_contents) > 0:
        files = [f for f in temp_contents if
                 not path.isdir(f'{temp}/{f}') and f != '00.txt']
        for f in files:
            remove(temp + '/' + f)
        sessions = [s for s in temp_contents if path.isdir(f'{temp}/{s}')]
        for s in sessions:
            if path.isdir(temp + '/' + s):
                rmtree(temp + '/' + s)
            if path.isfile(temp + '/' + s):
                remove(temp + '/' + s)
    return None

def _check_edf(edf):
    """Check whether the EDF uploaded is a valid Actiwave Cardio file."""
    f = pyedflib.EdfReader(edf)
    signals = f.getSignalLabels()
    if any('ECG0' in s for s in signals):
        return 'ECG'
    else:
        return 'invalid'

def _get_configs():
    cfg_dir = './configs'
    cfgs = [f for f in listdir(cfg_dir) if
            not f.startswith('.') and f != '00.txt']
    if len(cfgs) > 0:
        return cfgs
    else:
        return []

def _create_configs(device, fs, seg_size, filters: list or None, headers: dict):
    """Create a JSON-formatted configuration file of user
    SQA parameters.

    Parameters
    ----------
    fs : int
        The sampling rate of the recording.
    seg_size : int
        The window size of each segment, in seconds (e.g., 60).
    filters : list, None
        A list of ECG filters (i.e., 'powerline-interference'
        and/or 'baseline-wander')
    headers : dict
        A dictionary mapping data variables to headers in the
        uploaded CSV file.
    """

    # Save user configuration
    configs = {"device": device,
               "sampling rate": fs,
               "segment size": seg_size,
               "filters": filters,
               "headers": headers}

    # Serialize JSON
    json_object = json.dumps(configs)

    return json_object

def _load_config(filename):
    """Load a JSON configuration file into a dictionary."""
    cfg = open(filename)
    configs = json.load(cfg)
    return configs

def _export_sqa(file, data_type, type: str):
    """Export the SQA summary data in Zip or Excel format."""
    files = ['./temp/sqa_metrics.csv', './temp/peaks_by_segment.csv']
    if data_type == 'E4':
        files.append(f'./temp/{file}_BVP.csv')
        files.append(f'./temp/{file}_ACC.csv')
        files.append(f'./temp/{file}_IBI.csv')
    else:
        if data_type == 'Actiwave':
            files.append(f'./temp/{file}_ECG.csv')
            files.append(f'./temp/{file}_ACC.csv')
            files.append(f'./temp/{file}_IBI.csv')
        else:
            files.append(f'./temp/{file}_ECG.csv')
            files.append(f'./temp/{file}_IBI.csv')
            if f'{file}_ACC.csv' in listdir('./temp'):
                files.append(f'./temp/{file}_ACC.csv')
    if not path.exists('./downloads/'):
        makedirs('./downloads')
    else:
        pass
    if type == 'zip':
        with ZipFile(f'./downloads/{file}_sqa_summary.zip', 'w') as archive:
            for csv in files:
                archive.write(csv)
    if type == 'excel':
        with pd.ExcelWriter(f'./downloads/{file}_sqa_summary.xlsx') as xlsx:
            for csv in files:
                df = pd.read_csv(csv)
                fname = Path(csv).stem
                df.to_excel(xlsx, sheet_name = fname, index = False)
    return None

def _get_csv_headers(csv):
    """Get the headers of a user-uploaded CSV file in a list."""
    initial = pd.read_csv(csv, nrows = 1)
    headers = initial.columns.tolist()
    return headers

def segment_data(data, fs, seg_size):
    """Segment data into specific window sizes.

    Parameters
    ----------
    data : pd.DataFrame
        The data frame containing the data to be segmented.
    fs : int
        The sampling rate.
    seg_size : int
        The window size, in seconds, into which the data should be
        segmented.

    Returns
    -------
    df : pd.DataFrame
        The original data frame with data segmented with labels in a
        'Segment' column.
    """
    df = data.copy()
    df.insert(0, 'Segment', 0)
    segment = 1
    for n in range(0, len(df), int(seg_size * fs)):
        df.loc[n:(n + int(seg_size * fs)), 'Segment'] = segment
        segment += 1

    return df


def _blank_fig(context):
    """Display the default blank figure."""
    fig = go.Figure(go.Scatter(x = [], y = []))
    fig.update_layout(template = None,
                      paper_bgcolor = 'rgba(0, 0, 0, 0)',
                      plot_bgcolor = 'rgba(0, 0, 0, 0)')
    fig.update_xaxes(showgrid = False,
                     showticklabels = False,
                     zeroline = False)
    fig.update_yaxes(showgrid = False,
                     showticklabels = False,
                     zeroline = False)
    if context == 'pending':
        fig.add_annotation(text = '<i>Input participant data to view...</i>',
                           xref = 'paper', yref = 'paper',
                           font = dict(family = 'Poppins',
                                       size = 14,
                                       color = '#3a4952'),
                           x = 0.5, y = 0.5, showarrow = False)
    if context == 'none':
        fig.add_annotation(text = '<i>No data to view.</i>',
                           xref = 'paper', yref = 'paper',
                           font = dict(family = 'Poppins',
                                       size = 14,
                                       color = '#3a4952'),
                           x = 0.5, y = 0.5, showarrow = False)
    return fig

def _blank_table():
    """Display the default blank table."""
    summary = pd.DataFrame({
        'Signal Quality Metrics': ['Segments with Missing Beats',
                       'Segments with Invalid Beats',
                       'Average Missing Beats/Segment',
                       'File Duration'],
        '': ['N/A', 'N/A', 'N/A', 'N/A']})
    summary.set_index('Signal Quality Metrics', inplace = True)
    return dbc.Table.from_dataframe(
        summary,
        index = True,
        className = 'segmentTable',
        striped = False,
        hover = False,
        bordered = False)


def plot_signal(df, x, y, fs, seg_size = 60, segment = 1, n_segments = 1,
                signal_type = None, peaks = None):
    """Visualize a signal.

    Parameters
    ----------
    df : pandas.DataFrame
        The data frame containing the signal data.
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
    """Visualize an IBI series generated from ECG data.

    Parameters
    ----------
    df : pd.DataFrame
        The data frame containing the signal data.
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