.. image:: https://badgen.net/badge/python/3.9+/cyan
.. image:: https://badgen.net/badge/license/GPL-3.0/orange
.. image:: https://badgen.net/badge/contributions/welcome/green

====================
Welcome to HeartView
====================

HeartView is a Python-based **signal quality assessment pipeline and
dashboard** that visualizes and summarizes segment-by-segment quantification of
missing and invalid beats in wearable electrocardiograph (ECG) and
photoplethysmograph (PPG) data obtained in research contexts.

In contrast to other existing tools, HeartView provides a graphical user
interface intended to increase efficiency and accessibility for a wider range
of researchers who may not otherwise be able to conduct rigorous quality
assessment on their data. As such, HeartView is meant as **a diagnostic tool
for pre- and post-artifact correction of cardiovascular data**. We aim to help
researchers make more informed decisions about later data cleaning and
processing procedures and the reliabiltiy of their data when wearable biosensor
systems are used.

HeartView works with data collected from the Actiwave Cardio, Empatica E4, and 
other ECG devices outputting data in comma-separated value (CSV) format.

Features
--------

* **File Reader**: Read and transform raw ECG, PPG, and accelerometer data from European Data Format (EDF), archive (ZIP), and CSV files.

* **Configuration File Exporter**: Define and save pipeline parameters in a JSON configuration file that can be loaded for use on the same dataset later.

* **ECG Filters**: Filter out noise from baseline wander, muscle (EMG) activity, and powerline interference from your ECG data.

* **Beat Detection**: Extract heartbeats from your ECG [|ref1|] and Empatica E4 data.

* **Visualization Dashboard**: View and interact with our signal quality assessment chart and ECG/PPG, interbeat interval (IBI), and acceleration signal plots by segment.

* **Signal Quality Metrics**: Generate segment-by-segment information about missing and invalid data.

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