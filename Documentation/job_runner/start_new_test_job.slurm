#!/bin/bash -l
#SBATCH --partition=barnard
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=5100M
#SBATCH -c 4
#SBATCH -A "p_htw_promentat"
unset XDG_RUNTIME_DIR
srun ./new_test_job.sh
