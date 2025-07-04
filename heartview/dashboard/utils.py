from pathlib import Path
from shutil import rmtree
from zipfile import ZipFile
from scipy.signal import resample as scipy_resample
from heartview.heartview import compute_ibis
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pyedflib
import json

# ============================= Startup Functions ============================
def _make_subdirs():
    (Path('.') / 'temp').mkdir(
        parents = True, exist_ok = True)
    (Path('.') / 'beat-editor' / 'data').mkdir(
        parents = True, exist_ok = True)
    (Path('.') / 'beat-editor' / 'export').mkdir(
        parents = True, exist_ok = True)

def _clear_temp():
    temp = Path('.') / 'temp'
    if not temp.exists():
        return None

    for item in temp.iterdir():
        if item.name.startswith('.'):
            continue
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            rmtree(item)
    return None

# ====================== HeartView Dashboard Functions =======================
def _check_edf(edf):
    """Check whether the EDF uploaded is a valid Actiwave Cardio file."""
    f = pyedflib.EdfReader(edf)
    signals = f.getSignalLabels()
    if any('ECG0' in s for s in signals):
        return 'ECG'
    else:
        return 'invalid'

def _get_configs():
    cfg_dir = Path('.') / 'configs'
    cfgs = [f.name for f in cfg_dir.iterdir() if f.is_file() and
            not f.name.startswith('.')]
    if len(cfgs) > 0:
        return cfgs
    else:
        return []

def _create_configs(device, dtype, fs, seg_size, artifact_method, artifact_tol,
                    filter_on, headers):
    """Create a JSON-formatted configuration file of user SQA parameters.

    Parameters
    ----------
    device : str
        The device used to record the data.
    dtype : str
        The type of data being pre-processed (i.e., 'ECG', 'PPG', or 'EDA').
    fs : int
        The sampling rate of the recording.
    seg_size : int
        The window size of each segment, in seconds (e.g., 60).
    filt_on : boolean
        Whether the signal filter was turned on.
    headers : dict
        A dictionary mapping data variables to headers in the
        uploaded CSV file.
    """

    # Save user configuration
    configs = {"device": device,
               "data type": dtype,
               "sampling rate": fs,
               "segment size": seg_size,
               "filters": filter_on,
               "artifact identification method": artifact_method,
               "artifact tolerance": artifact_tol,
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
    temp_dir = Path('temp')
    downloads_dir = Path('downloads')
    downloads_dir.mkdir(parents = True, exist_ok = True)

    files = [temp_dir / f'{file}_SQA.csv']
    # files = [f'.{sep}temp{sep}{file}_SQA.csv']

    if data_type == 'E4':
        files += [
            temp_dir / f'{file}_BVP.csv',
            temp_dir / f'{file}_ACC.csv',
            temp_dir / f'{file}_IBI.csv',
            temp_dir / f'{file}_EDA.csv'
        ]
    elif data_type == 'Actiwave':
        files += [
            temp_dir / f'{file}_ECG.csv',
            temp_dir / f'{file}_ACC.csv',
            temp_dir / f'{file}_IBI.csv'
        ]
    else:  # Generic PPG or CSV input
        files += [
            temp_dir / f'{file}_ECG.csv',
            temp_dir / f'{file}_IBI.csv'
        ]
        acc_file = temp_dir / f'{file}_ACC.csv'
        if acc_file.exists():
            files.append(acc_file)

    if type == 'zip':
        zip_path = downloads_dir / f'{file}_sqa_summary.zip'
        with ZipFile(zip_path, 'w') as archive:
            for csv in files:
                archive.write(csv)
    elif type == 'excel':
        excel_path = downloads_dir / f'{file}_sqa_summary.xlsx'
        with pd.ExcelWriter(excel_path) as xlsx:
            for csv in files:
                df = pd.read_csv(csv)
                sheet_name = csv.stem
                df.to_excel(xlsx, sheet_name = sheet_name, index = False)
    return None

def _get_csv_headers(csv):
    """Get the headers of a user-uploaded CSV file in a list."""
    initial = pd.read_csv(csv, nrows = 1)
    headers = initial.columns.tolist()
    return headers

def _setup_data_ts(csv, dtype: str, dropdown: list):
    """Read uploaded CSV data into a data frame with timestamps."""
    df = pd.read_csv(csv, usecols = dropdown)
    if len(dropdown) > 2:
        df = df.rename(columns = {
            dropdown[0]: 'Timestamp',
            dropdown[1]: dtype,
            dropdown[2]: 'X',
            dropdown[3]: 'Y',
            dropdown[4]: 'Z'})
    else:
        df = df.rename(columns = {
            dropdown[0]: 'Timestamp',
            dropdown[1]: dtype})
    return df

def _setup_data_samples(csv, dtype: str, dropdown: list):
    """Read uploaded CSV data into a data frame with sample counts."""
    df = pd.read_csv(csv, usecols = dropdown)
    if len(dropdown) > 1:
        df = df.rename(columns = {
            dropdown[0]: dtype,
            dropdown[1]: 'X',
            dropdown[2]: 'Y',
            dropdown[3]: 'Z'})
    else:
        df = df.rename(columns = {
            dropdown[0]: dtype})
    samples = np.arange(len(df)) + 1
    df.insert(0, 'Sample', samples)
    return df

def _downsample_data(df, fs, signal_type, beats_ix, artifacts_ix, acc = None):
    """Downsample pre-processed cardio data and any acceleration data for
    quicker plot rendering on the dashboard."""

    # Downsample ECG data
    time = df['Timestamp'] if df.columns[1] == 'Timestamp' else \
        df['Sample']
    ds_factor = int(fs // 125) if fs >= 125 else 1   # minimum is 125 Hz
    ds_fs = int(fs / ds_factor)
    ds_time = time.iloc[::ds_factor].copy()
    ds_time = pd.Series(ds_time.values, name = time.name)

    try:
        ds = scipy_resample(df['Filtered'], len(ds_time))
    except KeyError:
        ds = scipy_resample(df[signal_type], len(ds_time))

    ds = pd.concat([ds_time, pd.Series(ds, name = signal_type)], axis = 1)
    rescaled_beats = np.round((beats_ix / ds_factor)).astype(int)
    rescaled_artifacts = np.round(artifacts_ix / ds_factor).astype(int)
    ds.loc[rescaled_beats, 'Beat'] = 1
    ds.loc[rescaled_artifacts, 'Artifact'] = 1

    # Downsample IBI data
    new_fs = int(fs / ds_factor)
    ds_ibi = compute_ibis(ds, new_fs, rescaled_beats, ts_col = time.name)

    # Downsample acceleration data
    if acc is not None:
        rs = scipy_resample(acc['Magnitude'], len(ds))
        ds_acc = pd.concat([ds_time, pd.Series(rs, name = 'Magnitude')],
                           axis = 1)
    else:
        ds_acc = None
    return ds, ds_ibi, ds_acc, ds_fs

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
        'Data Quality Metrics': ['Segments with Missing Beats',
                                   'Segments with Artifactual Beats',
                                   'Segments with Invalid Beats',
                                   'Average % Missing Beats/Segment',
                                   'Average % Artifactual Beats/Segment'],
        '': ['N/A', 'N/A', 'N/A', 'N/A', 'N/A']})
    summary.set_index('Data Quality Metrics', inplace = True)
    return dbc.Table.from_dataframe(
        summary,
        index = True,
        className = 'segmentTable',
        striped = False,
        hover = False,
        bordered = False)