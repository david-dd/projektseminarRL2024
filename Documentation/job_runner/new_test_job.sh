#!/bin/bash
module load release/23.10  Anaconda3
source activate /projects/p078/p_htw_promentat/conda_ps
cd /home/tosc270g/projects/projektseminarRL2024
python3.9 ./rl_test_with_env.py
