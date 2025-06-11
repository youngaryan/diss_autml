# diss_autml


srun --pty --cpus-per-task=4 --mem=32G bash -i 
conda activate myspark  
squeue --me 
sbatch ~/sched_scripts/ax_hpo.sh
srun --partition=gpu --qos=gpu --gres=gpu:1 --pty bash
