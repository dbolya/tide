# TIDE Changelog

## v1.0.1
### Error Calculation
 - **Important**: Fixed an oversight where detections ignored by AP calculation were not allowed to contribute to fixing errors. This caused a lot of error in datasets with large amounts of ignore regions (e.g., LVIS) to significantly overrepresent Missed Error and underrepresent either Classification or Localization error. This fix will also slightly change errors for other datasets (< .4 dAP on COCO), but conclusions on LVIS change dramatically.
### Plotting
 - Fixed issue with pie plots sometimes having negative padding.
 - Added automatic rescaling to the bar plots. This scalining will persist for the entire TIDE object, so you can compare between runs.
### Datasets
 - Fixed bug in the LVIS automatic download script that caused it to crash after downloading.


## v1.0.0
 - Initial release.
