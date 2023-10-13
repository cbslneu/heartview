.. raw:: html

    <style type="text/css">
        .bolditalic {
            font-weight: bold;
            font-style: italic;
        }
    </style>

.. role:: bolditalic
   :class: bolditalic


The HeartView dashboard was developed in Python and with the Dash framework,
which consists of a Flask server that communicates with front-end React components.

=================
Launching the App
=================

1. Within the activated virtual environment:

::

    (venv) $ python3 app.py

2. Open your web browser and go to: https://127.0.0.1:8050


===================
Terminating the App
===================

1. In your Terminal, press ``CTRL`` + ``c``.
2. Exit the virtual environment by typing ``deactivate``.


===================
Using the Dashboard
===================

Uploading and Setting Up Your Data
..................................

Once the dashboard application has been launched in your web browser, select
your data file. HeartView will accept raw European Data Formatted (.edf) files
from the Actiwave Cardio, archive (.zip) files from the Empatica E4, and
comma-separated values (.csv) files from other ECG sources.

In the example below, a CSV file containing raw data from another ECG source is
uploaded to the dashboard.

.. image:: _static/dashboard-upload-1.png
    :width: 600
    :align: center

| Set the sampling rate and map the headers in your CSV file to their respective timestamp and ECG variables.

.. image:: _static/dashboard-upload-2.png
    :width: 600
    :align: center

| Set the window size (60 seconds, by default), select the ECG filters to apply to your data, and click the "Run" button.

.. image:: _static/dashboard-upload-3.png
    :width: 600
    :align: center

|

Viewing Signal Quality Metrics
..............................

HeartView's main dashboard shows three panels.

.. image:: _static/dashboard-main.png
    :width: 600
    :align: center

| :bolditalic:`Data Summary` displays information about the loaded data file and signal quality metrics, including the number and proportion of invalid segments, as well as segment-by-segment counts and proportions of missing beats.

| :bolditalic:`Expected-to-Missing-Beats` shows an interactive bar chart of the numbers of expected and missing beats and whether they are invalid per segment.

| The lower panel shows the **ECG**, **interbeat interval (IBI)**, and **acceleration** signals pre-processed from the data file.

|

Exporting the Data Summary
..........................

The **Export Summary** button in the :bolditalic:`Data Summary` panel allows you
to download your data summary as a Zip archive file or an Excel workbook.

.. image:: _static/dashboard-export-1.png
    :width: 600
    :align: center

| The resulting file is saved in the `downloads/` folder in your HeartView project directory.

.. image:: _static/dashboard-export-2.png
    :width: 600
    :align: center

| Download an example Excel workbook with the SQA summary :download:`here <../examples/sample_ecg_sqa_summary.xlsx>`.

============================
Creating Configuration Files
============================

The HeartView dashboard allows you to save your data pre-processing parameters
according to your file type in configuration files that can be loaded again in
the future for more convenient pre-processing.

| After setting your data pre-processing parameters in the welcome panel, click the **Save** button to create a JSON configuration file for your parameters.

.. image:: _static/dashboard-config-export.png
    :width: 600
    :align: center

|

===========================
Loading Configuration Files
===========================

Toggle the **Load a Configuration File** switch to display the configuration
file upload field. Click on *Select Configuration File...* to load your JSON
configuration file.

.. image:: _static/dashboard-config-load-1.png
    :width: 600
    :align: center

| |important| You must re-map any headers saved in your JSON configuration file to the dashboard variables.

.. image:: _static/dashboard-config-load-2.png
    :width: 400
    :align: center

.. |important| raw:: html

    <span style="font-size: 12pt; font-weight: bold; color: #ed2d33">Important:</span>