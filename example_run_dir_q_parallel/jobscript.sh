#!/bin/bash
# Job
#SBATCH --partition=newnodes,sched_mit_hill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --time=12:00:00 ## Extra 5 mins to do last saves
#SBATCH -J ez_200x1000_q  # sensible name for the job

# Setup conda environment
minicondahome="/home/user/miniconda3"
. $minicondahome/etc/profile.d/conda.sh
conda activate ez

# Run scripts
mpiexec -n 9 python3 main_q_parallel.py
