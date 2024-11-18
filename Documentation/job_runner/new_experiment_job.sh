#!/bin/bash
module load release/23.10  Anaconda3 #load the required modules
source activate /projects/p078/p_htw_promentat/conda_ps
cd /projects/p078/p_htw_promentat/smt2020_4/projektseminarRL2024
python3.9 ./exp_set_gen_with_env.py

