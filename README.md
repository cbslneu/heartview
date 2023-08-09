<div align="center"> 
  <img src="https://github.com/nmy2103/heartview/blob/main/assets/heartview-logo.png?raw=true" height="100">
  <br>
  <img src="https://badgen.net/badge/python/3.9/cyan">
  <img src="https://badgen.net/badge/license/GPL-3.0/orange">
  <img src="https://badgen.net/badge/contributions/welcome/green">
  <br>
  <i>An extensible, open-source, and web-based signal quality assessment pipeline for ambulatory cardiovascular data</i>
  <br>
</div>  
<hr>

HeartView is a Python-based **signal quality assessment pipeline and dashboard** that visualizes and summarizes segment-by-segment quantification of missing and invalid beats in wearable electrocardiograph (ECG) and photoplethysmograph (PPG) data obtained in research contexts.  

In contrast to other existing tools, HeartView provides a graphical user interface intended to increase efficiency and accessibility for a wider range of researchers who may not otherwise be able to conduct rigorous quality assessment on their data. We aim to help researchers make more informed decisions about later data cleaning and processing procedures and the reliabiltiy of their data when wearable biosensor systems are used.  

Currently, HeartView works with data collected from the Actiwave Cardio, Empatica E4, and other ECG devices outputting data in comma-separated value (CSV) format.

## Features
* **File Reader**
<br>Read and transform raw ECG, PPG, and accelerometer data from European Data Format (EDF), archive (ZIP), and CSV files.
* **Configuration File Exporter**
<br>Define and save pipeline parameters in a JSON configuration file that can be loaded for use on the same dataset later.
* **ECG Filters**
<br>Filter out noise from baseline wander, muscle (EMG) activity, and powerline interference from your ECG data.
* **Beat Detection**
<br>Extract heartbeats from your ECG [[1](https://doi.org/10.1016/j.bspc.2011.03.004)] and Empatica E4 data.
* **Visualization Dashboard**
<br>View and interact with our signal quality assessment chart and ECG/PPG, interbeat interval (IBI), and acceleration signal plots by segment.
* **Signal Quality Metrics**
<br>Generate segment-by-segment information about missing and invalid data.

## Installation
```
cd <directory>  # replace <directory> with your directory
git clone https://github.com/nmy2103/heartview.git
```

## HeartView Dashboard
### Executing
Open your command line interface and type in:
```
cd /<directory>/heartview
```
```
python3 app.py
```
Open your web browser and go to: http://127.0.0.1:8050/

### Terminating
Within your command line interface, type `CTRL`+`c`.
