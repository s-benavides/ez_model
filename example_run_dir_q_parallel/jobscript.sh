#!/bin/bash
# Job
#SBATCH --partition=newnodes,sched_mit_hill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --time=00:15:00 ## Extra 5 mins to do last saves
#SBATCH -J ez_200x1000_q  # sensible name for the job

# Setup conda environment
minicondahome="/home/user/miniconda3"
. $minicondahome/etc/profile.d/conda.sh
conda activate general

# Run scripts
mpiexec -n $SLURM_NTASKS python3 main_q_parallel.py
