====================
Welcome to HeartView
====================
.. image:: https://badgen.net/badge/python/3.9+/cyan
.. image:: https://badgen.net/badge/license/GPL-3.0/orange
.. image:: https://badgen.net/badge/contributions/welcome/green

About
-----

HeartView is a Python-based **signal quality assessment pipeline and
dashboard** that visualizes and summarizes segment-by-segment quantification of
missing and artifactual beats in wearable electrocardiograph (ECG) and
photoplethysmograph (PPG) data obtained in research contexts.

In contrast to other existing tools, HeartView provides a graphical user
interface intended to increase efficiency and accessibility for a wider range
of researchers who may not otherwise be able to conduct rigorous quality
assessment on their data. As such, HeartView is meant as **a diagnostic tool
for pre- and post-artifact correction of cardiovascular data**. We aim to help
researchers make more informed decisions about later data cleaning and
processing procedures and the reliability of their data when wearable
biosensor systems are used.

HeartView works with data collected from the Actiwave Cardio, Empatica E4, and 
other ECG devices outputting data in comma-separated value (CSV) format.

Features
--------

* **File Reader**: Read and transform raw ECG, PPG, and accelerometer data
  from Actiwave European Data Format (EDF), Empatica E4 archive (ZIP),
  and CSV files.

* **Configuration File Exporter**: Define and save pipeline parameters in a
  JSON configuration file that can be loaded for use on the same dataset later.

* **Signal Filters**: Filter out noise from baseline wander, muscle (EMG)
  activity, and powerline interference from your ECG or PPG data.

* **Beat Detection**: Extract heartbeats from ECG and PPG data.

* **Visualization Dashboard**: View and interact with our signal quality
  assessment chart and ECG/PPG, interbeat interval (IBI), and acceleration
  signal plots by segment.

* **Signal Quality Metrics**: Generate segment-by-segment information about
  missing and artifactual data.

Citation
--------

If you use this software in your research, please cite `this paper <https://link.springer.com/chapter/10.1007/978-3-031-59717-6_8>`_.

.. code-block:: bibtex

    @inproceedings{Yamane2024,
      author    = {Yamane, N. and Mishra, V. and Goodwin, M.S.},
      title     = {HeartView: An Extensible, Open-Source, Web-Based Signal Quality Assessment Pipeline for Ambulatory Cardiovascular Data},
      booktitle = {Pervasive Computing Technologies for Healthcare. PH 2023},
      series    = {Lecture Notes of the Institute for Computer Sciences, Social Informatics and Telecommunications Engineering},
      volume    = {572},
      year      = {2024},
      editor    = {Salvi, D. and Van Gorp, P. and Shah, S.A.},
      publisher = {Springer, Cham},
      doi       = {10.1007/978-3-031-59717-6_8},
    }

What's New in Version 2.0.1
---------------------------

Pipeline Enhancements
*********************

- Added beat detection algorithms for ECG and PPG pre-processing.
- Added default filters for ECG, PPG, and ACC.
- Implemented artifact identification methods.
- Added progress bars for real-time feedback.

Dashboard Improvements
**********************
- Enhanced handling of uploaded data files.
- Added artifactual beats visualization and metrics.
- Improved segment view selection options.
- Enabled data downsampling for faster chart rendering.

For a full list of changes, see the full |changelog|.

Installation
------------

The HeartView source code is available from |GitHub|:

::

   $ git clone https://github.com/cbslneu/heartview.git

See :doc:`installation` for further info.



.. |ref1| raw:: html

    <a href="https://doi.org/10.1016/j.bspc.2011.03.004" target="_blank">1</a>

.. |GitHub| raw:: html

    <a href="https://github.com/cbslneu/heartview" target="_blank">GitHub</a>

.. |changelog| raw:: html

    <a href="https://github.com/cbslneu/heartview/blob/main/CHANGELOG.md"
    target="_blank">changelog</a>