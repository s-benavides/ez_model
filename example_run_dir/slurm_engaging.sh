#!/bin/bash

# Partition             Nodes   S-C-T   Timelimit
# ---------             -----   -----   ---------
# sched_mit_hill        (32)    2-8-1   12:00:00
# sched_mit_raffaele    (32)    2-10-1  12:00:00
# sched_any_quicktest   2       2-8-1   00:15:00
# newnodes              (32)    2-10-1  12:00:00

# Job
##SBATCH --partition=sched_mit_hill
#SBATCH --partition=sched_any_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=0:15:00
#SBATCH -J ez_test  # sensible name for the job

# Setup conda and dedalus environment
minicondahome="/home/santiago_b/miniconda3"
. $minicondahome/etc/profile.d/conda.sh
##source ~/dedalus/dedalus_modules
conda activate ez

# Run scripts
mpiexec -n 2 python3 main_q_parallel.py
