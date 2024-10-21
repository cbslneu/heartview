.. raw:: html

    <style type="text/css">
        .bolditalic {
            font-weight: bold;
            font-style: italic;
        }
    </style>

.. role:: bolditalic
   :class: bolditalic

=======================
Data Quality Assessment
=======================

Uploading and Setting Up Your Data
----------------------------------

Once the dashboard application has been launched in your web browser, select
your data file. HeartView will accept raw European Data Formatted (.edf) files
from the Actiwave Cardio, archive (.zip) files from the Empatica E4, and
comma-separated values (.csv) files from other ECG and PPG sources.

In the example below, a CSV file containing raw data from another ECG source is
uploaded to the dashboard.

.. image:: _static/dashboard-upload-1.png
    :width: 600
    :align: center

| Set the data type and the sampling rate, and map the headers in your CSV
  file to their respective timestamp and ECG variables.

.. image:: _static/dashboard-upload-2.png
    :width: 600
    :align: center

| Set the window size (60 seconds, by default), artifact identification
  tolerance (any floating-point value between 0 and 2), and select whether to
  apply filters to your data. Click the "Run" button to run the pipeline.

.. image:: _static/dashboard-upload-3.png
    :width: 600
    :align: center

Viewing Signal Quality Metrics
------------------------------

HeartView's main dashboard shows three panels.

.. image:: _static/dashboard-main.png
    :width: 600
    :align: center
|
| :bolditalic:`Data Summary` displays information about the loaded data file
  and signal quality metrics, including the number and proportion of invalid
  segments, as well as segment-by-segment counts and proportions of missing
  and artifactual beats.

| :bolditalic:`Data Quality` shows interactive bar charts of numbers of
  artifactual and missing beats against the number of detected beats and
  whether they are invalid per segment.

| :bolditalic:`Signal View` shows the ECG/PPG, interbeat interval (IBI),
  and, if given, acceleration signals pre-processed from the data file.

Exporting the Data Summary
--------------------------

The **Export Summary** button in the :bolditalic:`Data Summary` panel allows you
to download your data summary as a Zip archive file or an Excel workbook.

.. image:: _static/dashboard-export-1.png
    :width: 600
    :align: center

| The resulting file is saved in the `downloads/` folder in your HeartView project directory.

.. image:: _static/dashboard-export-2.png
    :width: 600
    :align: center

| Download an example Excel workbook with the SQA summary `here <../../../examples/sample_ecg_acc_sqa_summary.xlsx>`_.