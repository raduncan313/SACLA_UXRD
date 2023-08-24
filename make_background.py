# Usage: python make_background.py <run number>
# This code makes a background H5 file from a "dark" run with all shutters closed and saves it in `bg_dir`
# The resulting H5 file has two datasets:
#    `img`: the sum of all detector images acquired
#    `count`: the number of detector images acquired
# These background files are required to generate binned "cube" datasets

import sys

DataAccessUserAPI_path = '/home/software/SACLA_tool/DataAccessUserAPI/latest/python/lib'
SACLA_python_modules = '/home/software/opt/intel/oneapi/intelpython/python3.7/lib/python3.7/site-packages'

if not DataAccessUserAPI_path in sys.path:
    sys.path.insert(0, DataAccessUserAPI_path)
if not SACLA_python_modules in sys.path:
    sys.path.insert(0, SACLA_python_modules)

import numpy as np
import dbpy
import stpy
import os
import h5py

################ DEFINITIONS ####################
hi_tag = 202301
bl = 3 # specify beamline
run = int(sys.argv[1])

# IDs of the detectors for which to generate background H5 files. One background file will be generated for each detector.
det_IDs = ['MPCCD-1N0-M06-002']
bg_dir = '/work/raduncan/2023A8060_work/backgrounds/'

################# END DEFINITIONS ##############

for det_ID in det_IDs:
    print(f'Making background file for run {run}, detector {det_ID}...')
    taglist = dbpy.read_taglist_bydetid(bl, run, det_ID)
    nptaglist = np.array(taglist)
    count = len(taglist)
    
    # Initialize detector reading objects and cube fields
    reader = stpy.StorageReader(det_ID, bl, (run, run))
    buffer = stpy.StorageBuffer(reader)

    for ii,tag in enumerate(taglist):
        reader.collect(buffer, tag)

        if ii == 0:
            img = buffer.read_det_data(0)
        else:
            img += buffer.read_det_data(0)

    filename = f'run{run:04d}_{det_ID}.h5'
    print(f'Saving background image for run {run} to {bg_dir + filename}...')
    with h5py.File(bg_dir + filename, 'w') as f:
        f.create_dataset('img', data=img)
        f.create_dataset('count', data=count)

print('Done.')