srun --ntasks=1 --cpus-per-task=4 --time=1:00:00 --mem-per-cpu=1700 --pty bash -l #allocate 4 cores for the interactive job
module load release/23.10  Anaconda3 #load the required modules
source activate /projects/p078/p_htw_promentat/conda_ps
cd /projects/p078/p_htw_promentat/smt2020_4/projektseminarRL2024
python3.9 ./exp_set_gen_with_env.py

