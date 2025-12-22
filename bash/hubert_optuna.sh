#!/bin/bash
#SBATCH --job-name=hubert_Optuna_started
#SBATCH --output=/mnt/parscratch/users/ach21ag/private/diss_autml/speecbrain/logs/train_grpo_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00  # 4 days







# 1. Load Anaconda Module
module load Anaconda3/2024.02-1

# 2. ACTIVATE ENVIRONMENT (The Critical Fix)
# We use 'source activate' with the full path you found earlier.
# This works in batch mode where 'mamba' commands usually fail.
source activate /mnt/parscratch/users/ach21ag/private/mamba/envs/icassp

# 3. Move to scripts folder
cd /mnt/parscratch/users/ach21ag/private/diss_autml/speecbrain


echo "Starting Job..."

# 4. Run Steps

echo "Step 1: hubert Optuna started"
python  hubert_hpo_optuna.py

echo "Job Complete"
