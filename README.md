# diss_autml

ssh ach21ag@stanage.shef.ac.uk

srun --pty --cpus-per-task=4 --mem=32G bash -i 

conda activate myspark  

squeue --me 

sbatch ~/sched_scripts/ax_hpo.sh

srun --partition=gpu --qos=gpu --gres=gpu:1 --pty bash


<img width="1146" height="70" alt="image" src="https://github.com/user-attachments/assets/97094b30-2410-4a41-8cbc-f986cdf885d4" />



# 1) Make this shell mamba-aware (needed once per shell session)
eval "$(mamba shell hook --shell bash)"

# 2) Activate the env
mamba activate abstention-bench
mamba activate abstentionrl
