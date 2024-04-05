import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from tqdm import tqdm
from math import ceil

# ============================== CARDIOVASCULAR ==============================
class Cardio:
    """
    A class for signal quality assessment on cardiovascular data, including
    electrocardiograph (ECG) or photoplethysmograph (PPG) data.

    Parameters/Attributes
    ---------------------
    fs : int
        The sampling rate of the cardiovascular data.
    """

    def __init__(self, fs):
        """
        Initialize the Cardiovascular object.

        Parameters
        ----------
        fs : int
            The sampling rate of the ECG or PPG recording.
        """
        self.fs = int(fs)

    def compute_metrics(self, data, beats_ix, artifacts_ix, ts_col = None,
                        seg_size = 60, min_hr = 40, rolling_window = None,
                        rolling_step = 15, show_progress = True):
        """
        Compute all SQA metrics for cardiovascular data by segment or
        moving window. Metrics per segment or moving window include numbers
        of detected, expected, missing, and artifactual beats and
        percentages of missing and artifactual beats.

        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame containing pre-processed ECG or PPG data.
        beats_ix : array_like
            An array containing the indices of detected beats.
        artifacts_ix : array_like
            An array containing the indices of artifactual beats.
        seg_size : int
            The segment size in seconds; by default, 60.
        min_hr : int, float
            The minimum acceptable heart rate against which the number of
            beats in the last partial segment will be compared; by default, 40.
        ts_col : str, optional
            The name of the column containing timestamps; by default, None.
            If a string value is given, the output will contain a timestamps
            column.
        rolling_window : int, optional
            The size, in seconds, of the sliding window across which to
            compute the SQA metrics; by default, None.
        rolling_step : int, optional
            The step size, in seconds, of the sliding windows; by default, 15.
        show_progress : bool, optional
            Whether to display a progress bar while the function runs; by
            default, True.

        Returns
        -------
        metrics : pandas.DataFrame
            A DataFrame with all computed SQA metrics per segment.

        Notes
        -----
        If a value is given in the `rolling_window` parameter, the rolling
        window approach will override the segmented approach, ignoring any
        `seg_size` value.
        """
        df = data.copy()
        df.index = df.index.astype(int)
        df.loc[beats_ix, 'Beat'] = 1

        if rolling_window is not None:
            metrics = pd.DataFrame()
            if ts_col is not None:
                seconds = self.get_seconds(data, beats_ix, ts_col,
                                           show_progress = False)
                s = 1
                for n in tqdm(range(0, len(seconds), rolling_step),
                              disable = not show_progress):

                    # Get missing beats
                    window_missing = seconds.iloc[n:(n + rolling_window)]
                    n_expected = window_missing['Mean HR'].median()
                    n_detected = window_missing['N Beats'].sum()
                    n_missing = (n_expected - n_detected) \
                        if n_expected > n_detected else 0
                    perc_missing = round((n_missing / n_expected) * 100, 2)
                    ts = window_missing['Timestamp'].iloc[0]

                    # Summarize artifactual beats
                    artifacts = self.get_artifacts(
                        df, beats_ix, artifacts_ix, seg_size = 1,
                        ts_col = ts_col)
                    window_artifact = artifacts.iloc[n:(n + rolling_window)]
                    n_artifact = window_artifact['N Artifact'].sum()
                    perc_artifact = round((n_artifact / n_detected) * 100, 2)

                    # Output summary
                    metrics = pd.concat([metrics, pd.DataFrame.from_records([{
                        'Moving Window': s,
                        'Timestamp': ts,
                        'N Expected': n_expected,
                        'N Detected': n_detected,
                        'N Missing': n_missing,
                        '% Missing': perc_missing,
                        'N Artifact': n_artifact,
                        '% Artifact': perc_artifact
                    }])], ignore_index = True).reset_index(drop = True)
                    s += 1
            else:
                seconds = self.get_seconds(data, beats_ix,
                                           show_progress = False)
                s = 1
                for n in tqdm(range(0, len(seconds), rolling_step),
                              disable = not show_progress):

                    # Get missing beats
                    window_missing = seconds.iloc[n:(n + rolling_window)]
                    n_expected = window_missing['Mean HR'].median()
                    n_detected = window_missing['N Beats'].sum()
                    n_missing = (n_expected - n_detected) \
                        if n_expected > n_detected else 0
                    perc_missing = round((n_missing / n_expected) * 100, 2)

                    # Get artifactual beats
                    artifacts = self.get_artifacts(
                        df, beats_ix, artifacts_ix, seg_size = 1)
                    window_artifact = artifacts.iloc[n:(n + rolling_window)]
                    n_artifact = window_artifact['N Artifact'].sum()
                    perc_artifact = round((n_artifact / n_detected) * 100, 2)

                    # Output summary
                    metrics = pd.concat([metrics, pd.DataFrame.from_records([{
                        'Moving Window': s,
                        'N Expected': n_expected,
                        'N Detected': n_detected,
                        'N Missing': n_missing,
                        '% Missing': perc_missing,
                        'N Artifact': n_artifact,
                        '% Artifact': perc_artifact
                    }])], ignore_index = True).reset_index(drop = True)
                    s += 1

            # Handle last partial rolling window of data
            last_seg_len = len(seconds) % rolling_window
            if last_seg_len > 0:
                last_detected = metrics['N Detected'].iloc[-1]
                last_expected_ratio = min_hr / \
                                      metrics['N Expected'].iloc[:-1].median()
                last_expected = last_expected_ratio * last_seg_len
                if last_expected > last_detected:
                    last_n_missing = last_expected - last_detected
                    last_perc_missing = round(
                        (last_n_missing / last_expected) * 100, 2)
                else:
                    last_perc_missing = 0
                    last_n_missing = 0
                metrics['N Expected'].iloc[-1] = last_expected
                metrics['N Missing'].iloc[-1] = last_n_missing
                metrics['% Missing'].iloc[-1] = last_perc_missing

        else:
            if ts_col is not None:
                missing = self.get_missing(
                    df, beats_ix, seg_size, min_hr = min_hr, ts_col = ts_col,
                    show_progress = False)
                artifacts = self.get_artifacts(
                    df, beats_ix, artifacts_ix, seg_size, ts_col)
                metrics = pd.merge(missing, artifacts,
                                   on = ['Segment', 'Timestamp'])
                metrics['Invalid'] = metrics['N Detected'].apply(
                    lambda n: 1 if n < min_hr or n > 220 else np.nan)
            else:
                missing = self.get_missing(
                    df, beats_ix, seg_size, show_progress = False)
                artifacts = self.get_artifacts(
                    df, beats_ix, artifacts_ix, seg_size)
                metrics = pd.merge(missing, artifacts, on = ['Segment'])

        metrics['Invalid'] = metrics['N Detected'].apply(
            lambda x: 1 if x < min_hr or x > 220 else np.nan)

        return metrics

    def display_summary_table(self, sqa_df):
        """
        Display the SQA summary table.

        Parameters
        ----------
        sqa_df : pandas.DataFrame
            The DataFrame containing the SQA metrics per segment.

        Returns
        -------
        table : dash_bootstrap_components.Table
            Summary table for SQA metrics.

        """
        missing_n = len(sqa_df.loc[sqa_df['N Missing'] > 0])
        artifact_n = len(sqa_df.loc[sqa_df['N Artifact'] > 0])
        invalid_n = len(sqa_df.loc[sqa_df['Invalid'] == 1])
        avg_missing = '{0:.2f}%'.format(sqa_df['% Missing'].mean())
        avg_artifact = '{0:.2f}%'.format(
            sqa_df.loc[sqa_df['% Artifact'] > 0, '% Artifact'].mean())

        summary = pd.DataFrame({
            'Signal Quality Metrics': ['Segments with Missing Beats',
                                       'Segments with Artifactual Beats',
                                       'Segments with Invalid Beats',
                                       'Average % Missing Beats/Segment',
                                       'Average % Artifactual Beats/Segment'],
            '': [missing_n, artifact_n, invalid_n, avg_missing, avg_artifact]
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

    def get_artifacts(self, data, beats_ix, artifacts_ix,
                      seg_size = 60, ts_col = None):
        """
        Summarize the number and proportion of artifactual beats per segment.

        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame containing the pre-processed ECG or PPG data.
        beats_ix : array_like
            An array containing the indices of detected beats.
        artifacts_ix : array_like
            An array containing the indices of artifactual beats. This is
            outputted from `SQA.Cardio.identify_artifacts()`.
        seg_size : int
            The size of the segment in seconds; by default, 60.
        ts_col : str, optional
            The name of the column containing timestamps; by default, None.
            If a string value is given, the output will contain a timestamps
            column.

        Returns
        -------
        artifacts : pandas.DataFrame
            A DataFrame with the number and proportion of artifactual beats
            per segment.

        See Also
        --------
        SQA.Cardio.identify_artifacts :
            Identify artifactual beats using both or either of the methods.
        """
        df = data.copy()
        df.loc[beats_ix, 'Beat'] = 1
        df.loc[artifacts_ix, 'Artifact'] = 1

        n_seg = ceil(len(df) / (self.fs * seg_size))
        segments = pd.Series(np.arange(1, n_seg + 1))
        n_detected = df.groupby(
            df.index // (self.fs * seg_size))['Beat'].sum()
        n_artifact = df.groupby(
            df.index // (self.fs * seg_size))['Artifact'].sum()
        perc_artifact = round((n_artifact / n_detected) * 100, 2)

        if ts_col is not None:
            timestamps = df.groupby(
                df.index // (self.fs * seg_size)).first()[ts_col]
            artifacts = pd.concat([
                segments,
                timestamps,
                n_artifact,
                perc_artifact,
            ], axis = 1)
            artifacts.columns = [
                'Segment',
                'Timestamp',
                'N Artifact',
                '% Artifact',
            ]
        else:
            artifacts = pd.concat([
                segments,
                n_artifact,
                perc_artifact,
            ], axis = 1)
            artifacts.columns = [
                'Segment',
                'N Artifact',
                '% Artifact',
            ]
        return artifacts

    def identify_artifacts(self, beats_ix, method, initial_hr = None,
                           prev_n = None, neighbors = None, tol = None):
        """
        Identify locations of artifactual beats in cardiovascular data based
        on the criterion beat difference approach by Berntson et al. (1990),
        the Hegarty-Craver et al. (2018) approach, or both.

        Parameters
        ----------
        beats_ix : array_like
            An array containing the indices of detected beats.
        method : str
            The artifact identification method for identifying artifacts.
            This must be 'hegarty', 'cbd', or 'both'.
        initial_hr : int or float, optional
            The heart rate value for the first interbeat interval (IBI) to be
            validated against; by default, 80. Required for 'hegarty' method.
        prev_n : int, optional
            The number of preceding IBIs to validate against; by default, 6.
            Required for 'hegarty' method.
        neighbors : int, optional
            The number of surrounding IBIs with which to derive the criterion
            beat difference score; by default, 5. Required for 'cbd' method.
        tol : float, optional
            A configurable hyperparameter used to fine-tune the stringency of
            the criterion beat difference test; by default, 1. Required for
            'cbd' method.

        Returns
        -------
        artifacts_ix : array_like
            An array containing the indices of identified artifact beats.

        Notes
        -----
        The source code for the criterion beat difference test is from work by
        Hoemann et al. (2020).

        References
        ----------
        Berntson, G., Quigley, K., Jang, J., Boysen, S. (1990). An approach to
        artifact identification: Application to heart period data.
        Psychophysiology, 27(5), 586–598.

        Hegarty-Craver, M. et al. (2018). Automated respiratory sinus
        arrhythmia measurement: Demonstration using executive function
        assessment. Behavioral Research Methods, 50, 1816–1823.

        Hoemann, K. et al. (2020). Context-aware experience sampling reveals
        the scale of variation in affective experience. Scientific
        Reports, 10(1), 1–16.
        """

        def identify_artifacts_hegarty(beats_ix, initial_hr = 80, prev_n = 6):
            """Identify locations of artifactual beats in cardiovascular data
            based on the approach by Hegarty-Craver et al. (2018)."""

            ibis = (np.diff(beats_ix) / self.fs) * 1000
            beats = beats_ix[1:]  # drop the first beat
            artifact_beats = []
            valid_beats = [beats_ix[0]]  # assume first beat is valid

            # Set the initial IBI to compare against
            first_ibi = 60000 / initial_hr

            for n in range(len(ibis)):
                current_ibi = ibis[n]
                current_beat = beats[n]

                # Check against an estimate of the first N IBIs
                if n < prev_n:
                    if n == 0:
                        ibi_estimate = first_ibi
                    else:
                        next_five = np.insert(ibis[:n], 0, first_ibi)
                        ibi_estimate = np.median(next_five)

                # Check against an estimate of the preceding N IBIs
                else:
                    ibi_estimate = np.median(ibis[n - (prev_n):n])

                # Set the acceptable/valid range of IBIs
                low = (29 / 32) * ibi_estimate
                high = (37 / 32) * ibi_estimate

                if low <= current_ibi <= high:
                    valid_beats.append(current_beat)
                else:
                    artifact_beats.append(current_beat)

            return valid_beats, artifact_beats

        def identify_artifacts_cbd(beats_ix, neighbors = 5, tol = 1):
            """Identify locations of abnormal interbeat intervals (IBIs) using
             the criterion beat difference test by Berntson et al. (1990)."""

            # Derive IBIs from beat indices
            ibis = ((np.ediff1d(beats_ix)) / self.fs) * 1000

            # Compute consecutive absolute differences across IBIs
            ibi_diffs = np.abs(np.ediff1d(ibis))

            # Initialize an array to store "bad" IBIs
            ibi_bad = np.zeros(shape = len(ibis))
            artifact_beats = []

            if len(ibi_diffs) < neighbors:
                neighbors = len(ibis)

            for ii in range(len(ibi_diffs)):

                # If there are not enough neighbors in the beginning
                if ii < int(neighbors / 2) + 1:
                    select = np.concatenate(
                        (ibi_diffs[:ii], ibi_diffs[(ii + 1):(neighbors + 1)]))
                    select_ibi = np.concatenate(
                        (ibis[:ii], ibis[(ii + 1):(neighbors + 1)]))

                # If there are not enough neighbors at the end
                elif (len(ibi_diffs) - ii) < (int(neighbors / 2) + 1) and (
                        len(ibi_diffs) - ii) > 1:
                    select = np.concatenate(
                        (ibi_diffs[-(neighbors - 1):ii], ibi_diffs[ii + 1:]))
                    select_ibi = np.concatenate(
                        (ibis[-(neighbors - 1):ii], ibis[ii + 1:]))

                # If there is only one neighbor left to check against
                elif len(ibi_diffs) - ii == 1:
                    select = ibi_diffs[-(neighbors - 1):-1]
                    select_ibi = ibis[-(neighbors - 1):-1]

                else:
                    select = np.concatenate(
                        (ibi_diffs[ii - int(neighbors / 2):ii],
                         ibi_diffs[(ii + 1):(ii + 1 + int(neighbors / 2))]))
                    select_ibi = np.concatenate(
                        (ibis[ii - int(neighbors / 2):ii],
                         ibis[(ii + 1):(ii + 1 + int(neighbors / 2))]))

                # Calculate the quartile deviation
                QD = self._quartile_deviation(select)

                # Calculate the maximum expected difference (MED)
                MED = 3.32 * QD

                # Calculate the minimal artifact difference (MAD)
                MAD = (np.median(select_ibi) - 2.9 * QD) / 3

                # Calculate the criterion beat difference score
                criterion_beat_diff = (MED + MAD) / 2

                # Find indices of IBIs that fail the CBD check
                if (ibi_diffs[ii]) > tol * criterion_beat_diff:

                    bad_neighbors = int(neighbors * 0.25)
                    if ii + (bad_neighbors - 1) < len(beats_ix):
                        artifact_beats.append(beats_ix[ii:(ii +
                                                           bad_neighbors)])
                    else:
                        artifact_beats.append(
                            beats_ix[ii:(ii + (bad_neighbors - 1))])
                    ibi_bad[ii + 1] = 1

            artifact_beats = np.array(artifact_beats).flatten()
            return artifact_beats

        if method == 'hegarty':
            initial_hr = initial_hr if initial_hr is not None else 80
            prev_n = prev_n if prev_n is not None else 6
            _, artifacts_ix = identify_artifacts_hegarty(
                beats_ix, initial_hr, prev_n)
        elif method == 'cbd':
            neighbors = neighbors if neighbors is not None else 5
            tol = tol if tol is not None else 1
            artifacts_ix = identify_artifacts_cbd(
                beats_ix, neighbors, tol)
        elif method == 'both':
            initial_hr = initial_hr if initial_hr is not None else 80
            prev_n = prev_n if prev_n is not None else 6
            neighbors = neighbors if neighbors is not None else 5
            tol = tol if tol is not None else 1
            _, artifact_hegarty = identify_artifacts_hegarty(
                beats_ix, initial_hr, prev_n)
            artifact_cbd = identify_artifacts_cbd(
                beats_ix, neighbors, tol)
            artifacts_ix = np.union1d(artifact_hegarty, artifact_cbd)
        else:
            raise ValueError(
                'Invalid method. Method must be \'hegarty\', \'cbd\', '
                'or \'both\'.')
        return artifacts_ix

    def get_missing(self, data, beats_ix, seg_size = 60, min_hr = 40,
                    ts_col = None, show_progress = True):
        """
        Summarize the number and proportion of missing beats per segment.

        Parameters
        ----------
        data : pandas.DataFrame
            The DataFrame containing the pre-processed ECG or PPG data.
        beats_ix : array-like
            An array containing the indices of detected beats.
        seg_size : int
            The size of the segment in seconds; by default, 60.
        min_hr : int, float
            The minimum acceptable heart rate against which the number of
            beats in the last partial segment will be compared; by default, 40.
        ts_col : str, optional
            The name of the column containing timestamps; by default, None.
            If a string value is given, the output will contain a timestamps
            column.
        show_progress : bool, optional
            Whether to display a progress by while the function runs; by
            default, True.

        Returns
        -------
        missing : pandas.DataFrame
            A DataFrame with detected, expected, and missing numbers of
            beats per segment.
        """
        seconds = self.get_seconds(data, beats_ix, ts_col, show_progress)
        seconds.index = seconds.index.astype(int)

        n_seg = ceil(len(seconds) / seg_size)
        segments = pd.Series(np.arange(1, n_seg + 1))
        n_expected = seconds.groupby(
            seconds.index // seg_size)['Mean HR'].median()
        n_detected = seconds.groupby(
            seconds.index // seg_size)['N Beats'].sum()
        n_missing = (n_expected - n_detected).clip(lower = 0)
        perc_missing = round((n_missing / n_expected) * 100, 2)

        # Handle last partial segment of data
        last_seg_len = len(seconds) % seg_size
        if last_seg_len > 0:
            last_detected = n_detected.iloc[-1]
            last_expected_ratio = min_hr / n_expected.iloc[:-1].median()
            last_expected = last_expected_ratio * last_seg_len
            if last_expected > last_detected:
                last_n_missing = last_expected - last_detected
                last_perc_missing = round(
                    (last_n_missing / last_expected) * 100, 2)
            else:
                last_perc_missing = 0
                last_n_missing = 0
            n_expected.iloc[-1] = last_expected
            n_missing.iloc[-1] = last_n_missing
            perc_missing.iloc[-1] = last_perc_missing

        if ts_col is not None:
            timestamps = seconds.groupby(
                seconds.index // seg_size).first()[ts_col]
            missing = pd.concat([
                segments,
                timestamps,
                n_detected,
                n_expected,
                n_missing,
                perc_missing,
            ], axis = 1)
            missing.columns = [
                'Segment',
                'Timestamp',
                'N Detected',
                'N Expected',
                'N Missing',
                '% Missing',
            ]
        else:
            missing = pd.concat([
                segments,
                n_detected,
                n_expected,
                n_missing,
                perc_missing,
            ], axis = 1)
            missing.columns = [
                'Segment',
                'N Detected',
                'N Expected',
                'N Missing',
                '% Missing',
            ]
        return missing

    def get_seconds(self, data, beats_ix, ts_col = None, show_progress = True):
        """Get second-by-second HR, IBI, and beat counts from ECG or PPG data
        according to the approach by Graham (1978).

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the pre-processed ECG or PPG data.
        beats_ix : array-like
            An array containing the indices of detected beats.
        ts_col : str, optional
            The name of the column containing timestamps; by default, None.
            If a string value is given, the output will contain a timestamps
            column.
        show_progress : bool, optional
            Whether to display a progress bar while the function runs; by
            default, True.

        Returns
        -------
        interval_data : pd.DataFrame
            A DataFrame containing second-by-second HR and IBI values.

        Notes
        -----
        Rows with `NaN` values in the resulting DataFrame `interval_data`
        denote seconds during which no beats in the data were detected.

        References
        ----------
        Graham, F. K. (1978). Constraints on measuring heart rate and period
        sequentially through real and cardiac time. Psychophysiology, 15(5),
        492–495.
        """

        df = data.copy()
        df.index = df.index.astype(int)
        df.loc[beats_ix, 'Beat'] = 1

        interval_data = []

        # Iterate over each second
        s = 1
        for i in tqdm(range(0, len(df), self.fs), disable = not show_progress):

            # Get data at the current second and evaluation window
            current_sec = df.iloc[i:(i + self.fs)]
            if i == 0:
                # Look at current and next second
                window = df.iloc[:(i + self.fs)]
            else:
                # Look at previous, current, and next second
                window = df.iloc[(i - self.fs):(min(i + self.fs, len(df)))]

            # Get mean IBI and HR values from the detected beats
            current_beats = current_sec[current_sec['Beat'] == 1].index.values
            window_beats = window[window['Beat'] == 1].index.values
            ibis = np.diff(window_beats) / self.fs * 1000
            if len(ibis) == 0:
                mean_ibi = np.nan
                mean_hr = np.nan
            else:
                mean_ibi = np.mean(ibis)
                hrs = 60000 / ibis
                r_hrs = 1 / hrs
                mean_hr = 1 / np.mean(r_hrs)

            # Append values for the current second
            if ts_col is not None:
                interval_data.append({
                    'Second': s,
                    'Timestamp': current_sec.iloc[0][ts_col],
                    'Mean HR': mean_hr,
                    'Mean IBI': mean_ibi,
                    'N Beats': len(current_beats)
                })
            else:
                interval_data.append({
                    'Second': s,
                    'Mean HR': mean_hr,
                    'Mean IBI': mean_ibi,
                    'N Beats': len(current_beats)
                })

            s += 1
        interval_data = pd.DataFrame(interval_data)
        return interval_data

    def plot_missing(self, df, invalid_thresh = 30, title = None):
        """
        Plot detected and missing beat counts.

        Parameters
        ----------
        df : pandas.DataFrame()
            The DataFrame containing SQA metrics per segment.
        invalid_thresh : int, float
            The minimum number of beats detected for a segment to be considered
            valid; by default, 30.
        title : str, optional
        """
        max_beats = ceil(df['N Detected'].max() / 10) * 10
        nearest = ceil(max_beats / 2) * 2
        dtick_value = nearest / 5

        fig = go.Figure(
            data = [
                go.Bar(
                    x = df['Segment'],
                    y = df['N Expected'],
                    name = 'Missing',
                    marker = dict(color = '#f2816d'),
                    hovertemplate = '<b>Segment %{x}:</b> %{customdata:.0f} '
                                    'missing<extra></extra>'),
                go.Bar(
                    x = df['Segment'],
                    y = df['N Detected'],
                    name = 'Detected',
                    marker = dict(color = '#313c42'),
                    hovertemplate = '<b>Segment %{x}:</b> %{y:.0f} '
                                    'detected<extra></extra>')
            ]
        )
        fig.data[0].update(customdata = df['N Missing'])

        # Annotate invalid segments
        invalid = []
        for i, (segment_num, n_detected) in enumerate(
                zip(df['Segment'], df['N Detected'])):
            if n_detected < invalid_thresh:
                annotation = dict(
                    x = segment_num,
                    y = n_detected,
                    text = '<b>!</b>',
                    showarrow = False,
                    font = dict(color = '#db0f0f', size = 20),
                    yshift = 10,
                )
                invalid.append(annotation)
        if invalid:
            fig.add_annotation(
                text = '<span style="color: #db0f0f"><b>!</b></span>  '
                       'Invalid Number of Beats ',
                align = 'right',
                showarrow = False,
                xref = 'paper',
                yref = 'paper',
                x = 1,
                y = 1.3)

        fig.update_layout(
            xaxis_title = 'Segment Number',
            xaxis = dict(tickmode = 'linear', dtick = 1),
            yaxis = dict(
                title = 'Number of Beats',
                range = [0, max_beats],
                dtick = dtick_value),
            legend = dict(
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.0,
                xanchor = 'right',
                x = 1.0),
            annotations = invalid,
            font = dict(family = 'Poppins', size = 13),
            height = 289,
            margin = dict(t = 70, r = 20, l = 40, b = 65),
            barmode = 'overlay',
            template = 'simple_white',
        )
        if title is not None:
            fig.update_layout(
                title = title
            )
        return fig

    def plot_artifact(self, df, invalid_thresh = 30, title = None):
        """
        Plot detected and artifact beat counts.

        Parameters
        ----------
        df : pandas.DataFrame()
            The DataFrame containing SQA metrics per segment.
        invalid_thresh : int, float
            The minimum number of beats detected for a segment to be considered
            valid; by default, 30.
        title : str, optional
        """
        max_beats = ceil(df['N Detected'].max() / 10) * 10
        nearest = ceil(max_beats / 2) * 2
        dtick_value = nearest / 5

        fig = go.Figure(
            data = [
                go.Bar(
                    x = df['Segment'],
                    y = df['N Detected'],
                    name = 'Detected',
                    marker = dict(color = '#313c42'),
                    hovertemplate = '<b>Segment %{x}:</b> %{y:.0f} '
                                    'detected<extra></extra>'),
                go.Bar(
                    x = df['Segment'],
                    y = df['N Artifact'],
                    name = 'Artifact',
                    marker = dict(color = '#f2b463'),
                    hovertemplate = '<b>Segment %{x}:</b> %{y:.0f} '
                                    'artifact<extra></extra>')
            ],
        )

        # Annotate invalid segments
        invalid = []
        for i, (segment_num, n_detected) in enumerate(
                zip(df['Segment'], df['N Detected'])):
            if n_detected < invalid_thresh:
                annotation = dict(
                    x = segment_num,
                    y = n_detected,
                    text = '<b>!</b>',
                    showarrow = False,
                    font = dict(color = '#db0f0f', size = 20),
                    yshift = 10,
                )
                invalid.append(annotation)
        if invalid:
            fig.add_annotation(
                text = '<span style="color: #db0f0f"><b>!</b></span>  '
                       'Invalid Number of Beats ',
                align = 'right',
                showarrow = False,
                xref = 'paper',
                yref = 'paper',
                x = 1,
                y = 1.3)

        fig.update_layout(
            xaxis_title = 'Segment Number',
            xaxis = dict(tickmode = 'linear', dtick = 1),
            yaxis = dict(
                title = 'Number of Beats',
                range = [0, max_beats],
                dtick = dtick_value),
            legend = dict(
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.0,
                xanchor = 'right',
                x = 1.0,
                traceorder = 'reversed'),
            annotations = invalid,
            font = dict(family = 'Poppins', size = 13),
            height = 289,
            margin = dict(t = 70, r = 20, l = 40, b = 65),
            barmode = 'overlay',
            template = 'simple_white',
        )
        if title is not None:
            fig.update_layout(
                title = title
            )
        return fig

    def _get_iqr(self, data):
        """Compute the interquartile range of a data array."""
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        return iqr

    def _quartile_deviation(self, data):
        """Compute the quartile deviation in the criterion beat difference
        test."""
        iqr = self._get_iqr(data)
        QD = iqr * 0.5
        return QD