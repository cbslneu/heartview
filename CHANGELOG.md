# Changelog
All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [HeartView 2.0.2] - 2024-07-07

### Added

#### Dashboard
- Dropdown for artifact identification method selection
- Dropdown for beat detection algorithm selection
- Progreess bar in "Export Summary" modal

#### Beat Editor
- Keyboard shortcuts 
- Support for marking portions of segments 'unusable'
- Support for automatically loading existing edits from a saved file on startup
- Documentation pages for installation, quick start guide, and working with 
  Beat Editor files

### Changed

#### Beat Editor
- Legend label from "ECG" to "Signal" for broader cardiac signal type support
- Code in `PlotBeatSegment.js` to `BeatChartContainer.jsx`


## [HeartView 2.0.1] - 2024-10-07

### Added
- To `__init__.py` file within the `heartview` package:
  - Package version

#### Dashboard
- Input field for artifact identification tolerance level

### Changed

#### Dashboard
- Handling of user-inputted artifact identification tolerance level
- Shared x-axis limits in bottom signal plots
- "Export Summary" button to be disabled by default. It becomes enabled 
  once the required data is processed and available for export.

## [HeartView 2.0] - 2024-04-04

### Added
- To `__init__.py` file within the `heartview` package:
  - Package version
  - `cardio_sqa()` function to conveniently initialize `SQA.Cardio` class

#### Pipeline
- **ECG Pre-processing:**
  - Three beat detection algorithms for ECG pre-processing:
    - `ECG.BeatDetectors.pantompkins()`
    - `ECG.BeatDetectors.engzee()`
    - `ECG.BeatDetectors.nabian()`
  - All-in-one default filter: `ECG.Filters.filter_signal()`
- **PPG Pre-processing:**
  - All-in-one default filter: `PPG.Filters.filter_signal()`
- **ACC Pre-processing:**
  - Support for normalization of acceleration magnitude in `ACC.compute_auc()`
- **Signal Quality Assessment:**
  - **`Cardio`** class in `SQA.py` module
  - Two ECG/PPG artifact identification methods:
    - `SQA.Cardio.identify_artifacts_cbd()`
    - `SQA.Cardio.identify_artifacts_hegarty()`
  - Parameters and artifact identification in `SQA.Cardio.compute_metrics()`:
    - Segmentation across rolling windows
    - Number and proportion of artifactual beats per segment
  - Handling of overestimated missingness in last partial segments of data
  - Separate SQA functions for missing and artifactual data:
    - `SQA.Cardio.get_missing()`
    - `SQA.Cardio.get_artifacts()`
  - `tqdm` progress bars in `SQA.py` functions for real-time feedback

#### Dashboard
- Handling of uploaded data files without timestamp data
- Artifactual beats data quality chart
- Artifactual beats metrics to data summary table
- Segment view dropdown and buttons
- ECG and PPG data downsampling in the `run_pipeline` callback function 
  for faster chart rendering

### Changed

#### Pipeline
- Organization of existing functions into new classes:
  - `ECG.BeatDetectors` and `PPG.BeatDetectors`
  - `ECG.Filters` and `PPG.Filters`
  - `SQA.Cardio`
- Several names of functions in `ECG.py` and `PPG.py` modules for clarity
  and consistency
- Usage of `ECG.get_seconds()` to be in the `SQA.py` module
- Organization of Empatica- and Actiwave-specific functions to be in new 
  classes:
  - `heartview.Empatica`
  - `heartview.Actiwave`
- Organization of dashboard-specific functions and scripts to be in the 
  `dashboard` package directory
- New package and function names in the documentation
- Output from dictionary to tuple of BVP, HR, IBI, sampling rate, and start
  time in `PPG.Empatica.preprocess()`

#### Dashboard
- Usage of `dcc.Checklist` to `daq.BooleanSwitch` for filter options
- Display of bottom signal plots to be vertically stacked in one view

### Removed

#### Pipeline
- Accelerometer output from E4 data pre-processing function
- `PPG.get_e4_interval_data()` function

#### Dashboard
- Individual buttons for signal plots
- Segment view slider

## [HeartView 1.0] - 2023-08-03
- Initial release of the `heartview` package
- Included `pipeline` modules for pre-processing and signal quality assessment
  (SQA): `ACC.py`, `ECG.py`, `PPG.py`, and `SQA.py`
- Provided basic functionality for pre-processing ECG and PPG data
- Included a web dashboard for pre-processing, visualization, and SQA of
  ECG and PPG data
- Included example scripts and documentation for usage instructions
