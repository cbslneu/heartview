import numpy as np
import pandas as pd


def convert_acc(signal, sens, fs):
    """Convert an acceleration signal from G-force to units of m/s^2.

    Parameters
    ----------
    signal: array_like
        An array containing the acceleration signal.
    sens : int
        The G-force sensitivity level of the accelerometer
        (e.g., `2` for \u00B1 2g).
    fs : int
        The sampling rate of the accelerometer recording.

    Returns
    -------
    ms2 : array_like
        An array containing the converted acceleration signal.
    """

    ms2 = signal.copy()
    ms2 = (ms2 / (sens * fs)) * 9.81

    return ms2


def compute_magnitude(x, y, z):
    """Compute the magnitude of a 3-axis accelerometer signal.

    Parameters
    ----------
    x : array_like
        An array containing the x-axis accelerometer data.
    y : array_like
        An array containing the y-axis accelerometer data.
    z : array_like
        An array containing the z-axis accelerometer data.

    Returns
    -------
    magnitude : pandas.Series
        A series of acceleration magnitude values.
    """

    axes = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
    magnitude = np.sqrt(axes.apply(lambda x: x ** 2).sum(axis = 1))

    return magnitude


def compute_auc(df, signal, fs, window):
    """Compute the area-under-the-curve of acceleration magnitude
    across a given window size using Riemann sums.

    Parameters
    ----------
    df : pandas.DataFrame
        The data frame containing the signal data and a column with
        timestamps in datetime format.
    signal : str
        The name of the column containing the acceleration magnitude.
    fs : int, float
        The sampling rate.
    window : int
        The window size (in seconds) by which the areas are computed.

    Returns
    -------
    auc : pandas.DataFrame
        A data frame containing the area under the signal for each
        window and its corresponding timestamp.
    """

    auc = pd.DataFrame()
    auc['AUC'] = df[signal].groupby(df.index // (fs * window)).sum()

    # Look for timestamps column
    t = df.columns[df.dtypes == np.dtype('datetime64[ns]')]
    if len(t) != 0:
        auc.insert(0, 'Timestamp', None)
        auc['Timestamp'] = df[t].groupby(df.index // (fs * window)).first()

    return auc