#!/bin/bash
module load release/23.10  Anaconda3 #load the required modules
source activate /projects/p078/p_htw_promentat/conda_ps
cd /home/tosc270g/projects/projektseminarRL2024
chmod g+w /home/tosc270g/projects/projektseminarRL2024/experiments
python3.9 ./exp_set_gen_with_env.py

