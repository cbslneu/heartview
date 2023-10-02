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

In contrast to other existing tools, HeartView provides a graphical user interface intended to increase efficiency and accessibility for a wider range of researchers who may not otherwise be able to conduct rigorous quality assessment on their data. As such, HeartView is meant as a diagnostic tool for pre- and post-artifact correction of cardiovascular data. We aim to help researchers make more informed decisions about later data cleaning and processing procedures and the reliabiltiy of their data when wearable biosensor systems are used.  

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
1. Clone the HeartView GitHub repository into a directory of your choice.
```
cd <directory>  # replace <directory> with your directory
```
```
git clone https://github.com/cbslneu/heartview.git
```
2. Set up and activate a virtual environment in the `heartview` project directory.  
❗️**Note:** If you do not have `virtualenv` installed, run `pip3 install virtualenv` before proceeding below.
```
cd heartview
```
```
virtualenv venv -p python3
```
```
source venv/bin/activate
```
3. Install all project dependencies.
```
pip3 install -r requirements.txt
```

## HeartView Dashboard
### Executing
1. Within the activated virtual environment 
(i.e., `source <directory>/heartview/venv/bin/activate`), run the command:
```
python3 app.py
```
2. Open your web browser and go to: http://127.0.0.1:8050/

### Terminating
1. Kill the dashboard program: press `CTRL`+`c`.
2. Exit the virtual environment: `deactivate`.
