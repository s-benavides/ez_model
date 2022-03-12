import os,glob,pathlib
import numpy as np

idir='./'

skip = False

# Searches through all directories in 'Data' folder (which are named after experiments) and imports the data:
dirs = sorted(glob.glob(idir+'c_0*'))

runs = []
for file in dirs:
    run = file.split('/')[1]
    runs.append(run)

# Sort run-names based on value of q_in
runs = sorted(runs, key=lambda x: float(x.split('_')[2].replace('d','.')), reverse=False)

print(runs)

c_0s = [float(x.split('_')[2].replace('d','.')) for x in runs]
print(c_0s)

for ii,run in enumerate(runs):
#    if qins[ii]<0.1:
#        print("Skipping  %s" % run)
    if skip:
        print("Skipping")
        pass
    else:
        print("Working on %s" % run)
        file = glob.glob(idir+run+'/*_scalars.h5')
        if len(file)>0:
            file = file[0]
            print("Removing "+file)
            os.remove(file)
        else:
            print("No file found (possibly deleted).")
