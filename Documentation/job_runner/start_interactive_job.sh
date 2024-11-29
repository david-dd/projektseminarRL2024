srun --ntasks=1 --cpus-per-task=4 --time=1:00:00 --mem-per-cpu=1700 --pty bash -l #allocate 4 cores for the interactive job
module load release/23.10  Anaconda3 #load the required modules
cd /home/tosc270g/projects/projektseminarRL2024
