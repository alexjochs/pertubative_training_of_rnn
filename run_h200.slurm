#!/bin/bash
#SBATCH --job-name=rnn_es
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=sy-grp
#SBATCH --account=sy-grp
#SBATCH --nodelist=cn-x-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=480G
#SBATCH --time=12:00:00

# Strict error handling
set -e

# Setup logging
mkdir -p logs

# Activate environment
source .venv/bin/activate
echo "Environment activated: $(which python3)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L)"

# Run training
# Using scaling parameters for H200:
# - Batch: 256 (cranked up as requested)
# - Hidden: 4000
# - Pairs: 100
# - Iters: 100
# - Warmup: 15
# - T: 20

python3 perturbative_trained_rnn.py \
    --gpus 8 \
    --iters 100 \
    --pairs 100 \
    --batch 256 \
    --hidden 4000 \
    --log_every 10 \
    --warmup 15 \
    --T 20 \
    --emb_dim 128 \
    --rank 32 \
    --w0_std 1.0 \
    --win_std 0.2 \
    --h_clip 50.0

echo "Training complete."
