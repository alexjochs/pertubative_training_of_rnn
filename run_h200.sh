#!/bin/bash
#SBATCH --job-name=rnn_es
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=sy-grp
#SBATCH --account=sy-grp
#SBATCH --nodelist=cn-x-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --mem=480G
#SBATCH --time=12:00:00

set -euo pipefail

# Create logs directory in case it doesn't exist (recommended to also do this before submitting)
mkdir -p logs

echo "JobID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CWD:  $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

# Load CUDA runtime
module purge
module load cuda/12.8

echo "Loaded modules:"
module list

# GPU visibility (do not fail job if nvidia-smi isn't available in PATH)
echo "nvidia-smi -L:"
nvidia-smi -L || true

# Create venv on shared filesystem if missing
VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv at $VENV_DIR"
  $PYTHON_BIN -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
python -V
which python

# Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel

# Install project requirements
if [ -f "requirements.txt" ]; then
  echo "Installing requirements.txt"
  python -m pip install -r requirements.txt
else
  echo "ERROR: requirements.txt not found in $(pwd)"
  exit 1
fi

# Ensure GPU-enabled PyTorch + torchvision are installed (CUDA 12.4 wheels)
# This will override any CPU-only torch pulled in by requirements.txt.
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Sanity check: confirm CUDA works in PyTorch
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

# Avoid oversubscription
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Run training (match Slurm GPU request)
python3 perturbative_trained_rnn.py \
    --gpus 1 \
    --iters 100 \
    --pairs 10 \
    --batch 10 \
    --hidden 2000 \
    --log_every 1 \
    --warmup 15 \
    --T 20 \
    --emb_dim 128 \
    --rank 32 \
    --w0_std 1.0 \
    --win_std 0.2 \
    --h_clip 50.0

echo "Training complete."