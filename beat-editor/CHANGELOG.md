# Update v1.0.3 - 10/10/2024
Small update to change the timestamp x-axis to reflect local timezone times

# Update v1.0.2 - 10/03/2024 
Minor patch for small bug fixes that we're found post v1.0.1 update

# Major Update 🔧
## 10/04/2024 - v1.0.2
N/A

# Bug fixes 🐛
## 10/04/2024 - v1.0.2
* Fixed unnecessary coordinate plotting while panning with the mouse hovering over either a ECG point or Beat, respectively on Add or Delete Beat mode.
* Auto rerendering was happening once more when an initial export was overwritten, on a fresh app startup. This is now resolved.

# Update v1.0.1 - 10/03/2024
This update contains a mixture of trivial bug fixes to major optimization upgrades to the application. The beat editor has a few new additions of QoL and some minor changes applied from given feedback.

# Major Update 🔧
## 10/03/2024 - v1.0.1
* Refactored file fetching functionality to optimize chart generation and page load up speed.

# Bug fixes 🐛
## 10/03/2024 - v1.0.1
* Fixed issue where attempting to delete a Beat/Artifact on segment switch, "This is not a beat" error populates
* Fixed unwanted coordinate plotting when hitting the 'Reset Zoom' button
* Fixed zoom scale not resetting back to it's original level when switching segments
* Fixed 'Artifact' displaying on the chart legend when there isn't any
* Fixed unwanted auto rerendering when existing file is overwritten

# QOL Updates 👍
## 10/03/2024 - v1.0.1
* Zoom scroll functionality implementation
* Panning feature enabled to complement the zoom scroll addition
* Export confirmation message now displays when the process executes successfully
* Shortcut keys added for Add(A) & Delete(D) Mode

# Minor Changes 🤏
## 10/03/2024 - v1.0.1
* Moved file name to display under the buttons and dropdown on the left side of the page
* Current Segment and # of Beats display added
* Removed "!" from "This is not a beat!", now displays "This is not a beat"
* Updated hex color of "Added Beats" to #02E337 