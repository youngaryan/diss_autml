# diss_autml

ssh ach21ag@stanage.shef.ac.uk

srun --pty --cpus-per-task=4 --mem=32G bash -i 

conda activate myspark  

squeue --me 

sbatch ~/sched_scripts/ax_hpo.sh

srun --partition=gpu --qos=gpu --gres=gpu:1 --pty bash


<img width="1146" height="70" alt="image" src="https://github.com/user-attachments/assets/611aecaa-cd86-4930-80e5-8e01682b5885" />
