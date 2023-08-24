# Usage: python cube_timetool.py <run> <binval_min> <binval_max> <num_bins>

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
import h5py
from time import time

################ DEFINITIONS (MAY NEED TO BE CHANGED) ###################

# Experiment parameters
hi_tag = 202301
bl = 3 # specify beamline
run = int(sys.argv[1]) # Run number

# Motor names and filter names -- before the experiment check that these are correct
det_ID = 'MPCCD-1N0-M06-002'
xon_name = 'xfel_bl_3_shutter_1_open_valid/status'
xoff_name = 'xfel_bl_3_shutter_1_close_valid/status'
lason_name = 'xfel_bl_3_lh1_shutter_1_open_valid/status'
lasoff_name = 'xfel_bl_3_lh1_shutter_1_close_valid/status'
delay_name = 'xfel_bl_3_st_2_motor_1/position'
tt_delay_name = 'xfel_bl_3_st_1_motor_73/position'
intens_6_name = 'xfel_bl_3_st_2_pd_user_6_fitting_peak/voltage'
intens_7_name = 'xfel_bl_3_st_2_pd_user_7_fitting_peak/voltage'
intens_name = intens_6_name

cube_dir = '/work/raduncan/2023A8060_work/raduncan/cubes_post/' # Directory where cubes are saved
tt_dir = '/work/raduncan/2023A8060_work/timingtool/data/' # Directory where timetool CSV files are saved

# Background information
bg_num = 1292973 # Run number of background scan
bg_dir = '/work/raduncan/2023A8060_work/backgrounds/'

# Conversion factors
pos_to_ps = 0.006671 # Convert delay stage pulse values to picoseconds
pix_to_ps = 0.0024431 # Convert timetool pixel values to picoseconds

# Threshold values
energy_thresh = 2100 # Energy threshold for pixels -- set all pixels below this value to zero
intens_thresh = (-np.inf, np.inf) # Filter for i0 values

# Binning parameters
binval_min = int(sys.argv[2])
binval_max = int(sys.argv[3])
num_bins = int(sys.argv[4])
t0 = -448.82488

################### END DEFINITIONS ########################################

start_time = time()
bin_delta = (binval_max - binval_min)/(num_bins - 1)
scan_var = np.linspace(binval_min, binval_max, num_bins)

print('Loading scan data...')

# load non-detector data
csvcontents = np.genfromtxt(tt_dir + str(run)  + '.csv', delimiter=',', skip_header=2)
nptaglist_csv = csvcontents[:,0].astype('int')
ttpix_csv = csvcontents[:,2]
taglist_csv = tuple(map(int, nptaglist_csv.tolist()))
taglist = dbpy.read_taglist_bydetid(bl, run, det_ID)
nptaglist = np.array(taglist)
num_shots = len(taglist)
intens = np.array(dbpy.read_syncdatalist_float(intens_6_name, hi_tag, taglist)) + np.array(dbpy.read_syncdatalist_float(intens_7_name, hi_tag, taglist))
ttpix = np.zeros(len(taglist))

for ii,tag in enumerate(taglist):
    if tag in taglist_csv:
        ttpix[ii] = ttpix_csv[taglist_csv.index(tag)]
ttps = ttpix*pix_to_ps

lason = np.array(dbpy.read_syncdatalist_float(lason_name, hi_tag, taglist), dtype=bool)
lasoff = np.array(dbpy.read_syncdatalist_float(lasoff_name, hi_tag, taglist), dtype=bool)
xon = np.array(dbpy.read_syncdatalist_float(xon_name, hi_tag, taglist), dtype=bool)
xoff = np.array(dbpy.read_syncdatalist_float(xoff_name, hi_tag, taglist), dtype=bool)
tt_delay = np.array(dbpy.read_syncdatalist_float(tt_delay_name, hi_tag, taglist))*pos_to_ps
delay = np.array(dbpy.read_syncdatalist_float(delay_name, hi_tag, taglist))*pos_to_ps - tt_delay - ttps - t0 + np.nanmean(ttps)

# Create filters
nan_filt = np.logical_and.reduce([np.logical_not(np.isnan(intens)), np.logical_not(np.isnan(ttps))])
intens_filt = np.logical_and((intens > intens_thresh[0]), (intens < intens_thresh[-1]))
in_csv = np.array([tag in taglist_csv for tag in taglist])
x_filt = np.logical_and(xon, np.logical_not(xoff))
las_filt = np.logical_or(np.logical_and(lason, np.logical_not(lasoff)), np.logical_and(lasoff, np.logical_not(lason)))
tot_filt = np.logical_and.reduce([nan_filt, intens_filt, in_csv, x_filt, las_filt])

# Apply filters
intens = intens[tot_filt]
ttps = ttps[tot_filt]
nptaglist = nptaglist[tot_filt]
taglist = tuple(map(int, nptaglist.tolist()))
lason = lason[tot_filt]
lasoff = lasoff[tot_filt]
tt_delay = tt_delay[tot_filt]
delay = delay[tot_filt]

# Generate bin indices for events
bin_inds = np.digitize(delay - bin_delta/2, scan_var, right=True)
bin_inds[bin_inds > num_bins - 1] = num_bins - 1
bin_inds[bin_inds < 0] = 0

# Separate events into laser-on and laser-off
on_tags = tuple(map(int, nptaglist[lason].tolist()))
off_tags = tuple(map(int, nptaglist[lasoff].tolist()))
intens_on = intens[lason]
intens_off = intens[lasoff]
delay_on = delay[lason]
delay_off = delay[lasoff]
bin_inds_on = bin_inds[lason]
bin_inds_off = bin_inds[lasoff]

# Initialize detector reading objects and cube fields
reader = stpy.StorageReader(det_ID, bl, (run, run))
buffer = stpy.StorageBuffer(reader)

i0_on = np.zeros(num_bins)
bin_counts_on = np.zeros(num_bins)

i0_off = np.zeros(num_bins)
bin_counts_off = np.zeros(num_bins)

print(f'Delays ranging from {min(delay)} to {max(delay)} ps')
print(f'Binning into {num_bins} bins from {binval_min} to {binval_max} ps')
print(f'Bin delta is {bin_delta} ps')
print(f'{num_shots} shots before filtering')
print(f'{len(on_tags) + len(off_tags)} shots after filtering')
print(f'{len(on_tags)} laser-on shots, {len(off_tags)} laser-off shots')

print('\nBinning laser-on shots...')
for ii,tag in enumerate(on_tags):
    reader.collect(buffer, tag)
    img = buffer.read_det_data(0)
    i0 = intens_on[ii]
    
    
    if ii == 0:
        imgs_on = np.zeros((num_bins, img.shape[0], img.shape[1]))
        imgs_off = np.zeros((num_bins, img.shape[0], img.shape[1]))
        
        # Load background image
        with h5py.File(bg_dir + f'run{bg_num:04d}_{det_ID}.h5', 'r') as f:
            img_bg = f['img'][:]
            count_bg = f['count'][()]
        
        img_bg /= count_bg
        img_bg = np.reshape(img_bg, img.shape)        
        
    gain = np.copy(buffer.read_det_info(0)['mp_absgain'])
    img -= img_bg
    img *= gain
    
    jj = bin_inds_on[ii]
    img[img < energy_thresh] = 0
    imgs_on[jj,:,:] += img
    i0_on[jj] += i0
    bin_counts_on[jj] += 1
    
    if (ii%100 == 0) and (ii > 0):
        print(f'Binned {ii} shots...')

print('Done binning laser-on shots')

print('\nBinning laser-off shots...')
for ii,tag in enumerate(off_tags):
    reader.collect(buffer, tag)
    img = buffer.read_det_data(0)
    i0 = intens_off[ii]
    
    if ii == 0:
        imgs_off = np.zeros((num_bins, img.shape[0], img.shape[1]))
        if len(on_tags) == 0:
            imgs_on = np.zeros((num_bins, img.shape[0], img.shape[1]))
        
        # Load background
        with h5py.File(bg_dir + f'run{bg_num:04d}_{det_ID}.h5', 'r') as f:
            img_bg = f['img'][:]
            count_bg = f['count'][()]
            
        img_bg /= count_bg
        img_bg = np.reshape(img_bg, img.shape)
    
    gain = np.copy(buffer.read_det_info(0)['mp_absgain'])
    img -= img_bg
    img *= gain
    
    jj = bin_inds_off[ii]
    img[img < energy_thresh] = 0
    imgs_off[jj,:,:] += img
    i0_off[jj] += i0
    bin_counts_off[jj] += 1
    
    if (ii%100 == 0) and (ii > 0):
        print(f'Binned {ii} shots...')

print('Done binning')
end_time = time()
print(f'Took {end_time - start_time} seconds to bin the run.')

# Save cube to hdf5
cubename_on = f'run{run:04d}_on.h5'
cubename_off = f'run{run:04d}_off.h5'
with h5py.File(cube_dir + cubename_on, 'w') as f:
    f.create_dataset('scan_var', data=scan_var)
    f.create_dataset('imgs', data=imgs_on)
    f.create_dataset('i0', data=i0_on)
    f.create_dataset('bin_counts', data=bin_counts_on)

with h5py.File(cube_dir + cubename_off, 'w') as f:
    f.create_dataset('scan_var', data=scan_var)
    f.create_dataset('imgs', data=imgs_off)
    f.create_dataset('i0', data=i0_off)
    f.create_dataset('bin_counts', data=bin_counts_off)

print(f'Saved cubes for run {run} in {cube_dir}')
print('Done.')