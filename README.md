# SACLA_UXRD

This repository holds python code and jupyter notebooks used for data processing during ultrafast x-ray diffraction experiments using the SACLA free-electron laser at the SPring8 synchrotron facility. Contained within this repository are the following:

- `cube_timetool_notebook.ipynb`: a notebook used to process time delay scans into binned "cube" H5 files. This code applies the timing corrections measured by the timetool, and thus can only be used for time delay scans for which timing corrections exist.
- `cube_timetool.py`: a python script version of the code in `cube_timetool_notebook.ipynb` that can be submitted to the cluster as a job.
- `cube_no-timetool_notebook.ipynb`: a notebook used to process scan measurements into binned "cube" H5 files. This code does not apply timetool corrections, and can be used to process scans over any parameter.
- `make_background.py`: a python script that process a static (i.e., non-scan) "dark" background measurement to be subtracted from subsequent measurements. All of the above scripts and notebooks require such a background file to be specified.
