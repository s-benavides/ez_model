#!/bin/bash

# Partition             Nodes   S-C-T   Timelimit
# ---------             -----   -----   ---------
# sched_mit_hill        (32)    2-8-1   12:00:00
# sched_mit_raffaele    (32)    2-10-1  12:00:00
# sched_any_quicktest   2       2-8-1   00:15:00
# newnodes              (32)    2-10-1  12:00:00

# Job
##SBATCH --partition=sched_mit_hill
#SBATCH --partition=newnodes
##SBATCH --partition=sched_any_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
##SBATCH --exclude=node[390-391]
#SBATCH --constraint=centos7
#SBATCH --time=12:00:00 ## Extra 5 mins to do last saves
##SBATCH --time=00:15:00 ## Extra 5 mins to do last saves
#SBATCH -J ez_w60d3d5_128x64  # sensible name for the job

# Setup conda and dedalus environment
minicondahome="/home/santiago_b/miniconda3"
. $minicondahome/etc/profile.d/conda.sh
##source ~/dedalus/dedalus_modules
conda activate ez

# Run scripts
mpiexec -n 16 python3 main_f_parallel.py
