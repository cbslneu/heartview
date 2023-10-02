========
Examples
========

Below are examples of how you can use functions from the HeartView pipeline on
your data. Alternatively, see our |Jupyter notebooks| for full a full
walk-through for each type of data.

Reading and Pre-Processing Data
-------------------------------

|Actiwave Cardio|

Extract and pre-process data from an Actiwave Cardio device.

::

    from heartview.pipeline import ECG

    edf = 'sample_actiwave_data.edf'
    ecg, acc = ECG.read_actiwave(edf)

    display(ecg.head(), acc.head())

Output:

::

                       Timestamp        mV
    0 2016-11-19 10:30:00.000000 -0.053957
    1 2016-11-19 10:30:00.000977 -0.058344
    2 2016-11-19 10:30:00.001953 -0.046353
    3 2016-11-19 10:30:00.002930 -0.036410
    4 2016-11-19 10:30:00.003906 -0.036994

                       Timestamp         X         Y          Z  Magnitude
    0 2016-11-19 10:30:00.000000 -0.460020 -0.876585  10.064994  10.113562
    1 2016-11-19 10:30:00.031250 -0.358126 -0.774691  10.064994  10.101114
    2 2016-11-19 10:30:00.062500 -0.358126 -0.876585  10.064994  10.109439
    3 2016-11-19 10:30:00.093750 -0.358126 -0.876585  10.064994  10.109439
    4 2016-11-19 10:30:00.125000 -0.358126 -0.876585  10.064994  10.109439

|Empatica E4|

Extract and pre-process data from an Empatica E4 device. All pre-processed
data are stored in a Python dictionary.

::

    from heartview.pipeline import PPG

    e4_zip = 'sample_e4_data.zip'
    e4_data = PPG.preprocess_e4(e4_zip)

    e4_data.keys()

Output:

::

    # Under construction


Beat Detection
--------------

|ECG|

Detect R peaks from ECG data collected with the Actiwave Cardio and other
sources. Note that ECG data from other sources will have to be read into a
Pandas data frame prior to this step.

::

    from heartview.pipeline import ECG

    # `ecg` = ECG DataFrame output from `ECG.read_actiwave()`
    # `'mV'` = name of column containing ECG values
    ecg_fs = 1024
    peak_loc = ECG.detect_rpeaks(ecg, 'mV', fs)

    # Save R peak occurrences
    ecg.loc[peak_loc, 'Peak'] = 1

|Empatica E4|

Extract heartbeats from IBI data from the Empatica E4.

::

    from heartview.pipeline import PPG

    # `e4_data` = output from `PPG.preprocess_e4()`
    ibi = e4_data['ibi']
    e4_fs = e4_data['fs']
    start_time = e4_data['start time']

    e4_peaks = PPG.get_e4_peaks(ibi, e4_fs, start_time)
    e4_peaks.head()

Output:

::

    # Under construction


Signal Quality Assessment
-------------------------

|ECG|

::

    from heartview.pipeline import ECG, SQA

    # Get second-by-second data
    ecg_fs = 1024
    seg_size = 60  # seconds
    interval_data = ECG.get_seconds(ecg, 'Peak', ecg_fs, seg_size)

    # Get the expected and detected numbers of peaks by segment
    peaks_by_seg = SQA.evaluate_peaks(interval_data, seg_size)

    peaks_by_seg.head()

Output:

::

    # Under construction


.. |Jupyter notebooks| raw:: html

    <a href="https://github.com/cbslneu/heartview/tree/main/examples" target="_blank">Jupyter notebooks</a>

.. |Actiwave Cardio| raw:: html

    <div style="font-size: 14pt; font-weight: bold; margin-bottom: 10pt">Actiwave Cardio</div>

.. |ECG| raw:: html

    <div style="font-size: 14pt; font-weight: bold; margin-bottom: 10pt">Actiwave Cardio and Other ECG Sources</div>

.. |Empatica E4| raw:: html

    <div style="font-size: 14pt; font-weight: bold; margin-bottom: 10pt">Empatica E4</div>