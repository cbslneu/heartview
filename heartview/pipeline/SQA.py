import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

def evaluate_peaks(df, seg_size):
    """
    Get the number of detected and expected number of peaks per segment.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data frame containing the second-by-second heart
        rates, interbeat intervals, and peak counts.
    seg_size : int
        The segment size of each segment.

    Returns
    -------
    peaks_by_seg : pandas.DataFrame
        A DataFrame with the number of detected and the expected number
        of peaks per segment.

    """
    df.index = df.index.astype(int)
    peaks_by_seg = pd.DataFrame(columns = ['Segment', 'Timestamp',
                                           'Detected', 'Expected'])

    for seg, n in enumerate(range(0, len(df), seg_size), start = 1):
        subset = df.iloc[n: n + seg_size].reset_index(drop = True)
        detected = subset['# R Peaks'].sum()
        expected = subset['Mean HR'].median() * (seg_size / 60)
        peaks_by_seg = pd.concat([peaks_by_seg, pd.DataFrame.from_records([{
            'Segment': seg,
            'Timestamp': subset.loc[0, 'Timestamp'],
            'Detected': detected,
            'Expected': expected
        }])], ignore_index = True)

    return peaks_by_seg

def compute_metrics(df):
    """
    Compute SQA metrics by segment from the `peaks_by_seg` output.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the number of detected and the expected
        number of peaks per segment.

    Returns
    -------
    metrics : pandas.DataFrame
        A DataFrame with computed SQA metrics per segment.

    """
    df.index = df.index.astype(int)
    metrics = df.copy()
    metrics[['Invalid', 'Missing', '% Missing']] = 0

    for n in range(len(metrics)):
        det = metrics.loc[n, 'Detected']
        exp = metrics.loc[n, 'Expected']

        # Label invalid segments
        if (det < 30) or (det > 220):
            metrics.loc[n, 'Invalid'] = 1

        # Compute missing peaks
        if det < exp:
            metrics.loc[n, 'Missing'] = np.abs(det - exp)
            metrics.loc[n, '% Missing'] = np.abs((det - exp) / exp) * 100

    return metrics

def plot_expected2missing(df, title = None):
    """
    Plot the expected-to-missing beats chart.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing SQA metrics per segment.
    title : str
        The title of the chart; by default, None.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The resulting figure of expected-to-missing beats per segment.

    """
    valid = df[df.Invalid == 0].index.values

    fig = go.Figure(data = [
        go.Bar(
            name = 'Expected Peaks',
            x = df.loc[valid, 'Segment'],
            y = df.loc[valid, 'Expected'],
            marker = dict(color = '#f2ce4b'),
            hovertemplate = '<b>Segment %{x}:</b> %{y:.0f} '
                            'expected beats<extra></extra>'
        ),
        go.Bar(
            name = 'Missing Peaks',
            x = df.loc[valid, 'Segment'],
            y = df.loc[valid, 'Missing'],
            opacity = 0.7,
            marker = dict(color = '#fa2a0a'),
            hovertemplate = '<b>Segment %{x}:</b> %{y:.0f} '
                            'missing beats<extra></extra>'
        )
    ])

    invalid = df.index[df.Invalid == 1]

    fig.add_traces(
        go.Bar(
            x = df.loc[invalid, 'Segment'],
            y = df.loc[invalid, 'Detected'],
            name = 'Invalid Segment',
            width = 0.8,
            marker_color = '#e8e8e8',
            hovertemplate = '<b>Segment %{x}:</b> %{y:.0f} '
                            'detected beats<extra></extra>'
        )
    )

    fig.update_layout(
        xaxis_title = 'Segment Number',
        yaxis_title = 'Number of Beats',
        xaxis = dict(tickmode = 'linear', dtick = 5),
        margin = dict(t = 80, r = 20),
        barmode = 'overlay',
        template = 'plotly_white',
        legend = dict(
            orientation = 'h',
            yanchor = 'bottom',
            y = 1.0,
            xanchor = 'right',
            x = 1.0
        )
    )

    if title is not None:
        fig.update_layout(
            title = title
        )

    return fig

def display_summary_table(df):
    """
    Display the SQA summary table.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the SQA metrics per segment.

    Returns
    -------
    table : dash_bootstrap_components.Table
        Summary table for SQA metrics.

    """
    missing_n = len(df.loc[df['Missing'] > 0])
    invalid_n = len(df.loc[df['Invalid'] == 1])
    avg_missing = '{0:.2f}%'.format(df['% Missing'].mean())

    summary = pd.DataFrame({
        'Signal Quality Metrics': ['Segments with Missing Beats',
                                   'Segments with Invalid Beats',
                                   'Average % Missing Beats/Segment'],
        '': [missing_n, invalid_n, avg_missing]
    })

    summary.set_index('Signal Quality Metrics', inplace = True)

    table = dbc.Table.from_dataframe(
        summary,
        index = True,
        className = 'segmentTable',
        striped = False,
        hover = False,
        bordered = False
    )

    return table