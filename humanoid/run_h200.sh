#!/bin/bash
#SBATCH --job-name=humanoid_es
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=sy-grp
#SBATCH --account=sy-grp
#SBATCH --nodelist=cn-x-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:1
#SBATCH --mem=900G
#SBATCH --time=24:00:00

set -euo pipefail

START_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SEARCH_DIR="$(cd "$START_DIR" && pwd)"
PROJECT_ROOT=""

for _ in 1 2 3 4 5 6; do
  if [ -f "$SEARCH_DIR/requirements.txt" ] && [ -f "$SEARCH_DIR/humanoid/pertubative_trained_rnn_rl.py" ]; then
    PROJECT_ROOT="$SEARCH_DIR"
    break
  fi
  PARENT_DIR="$(dirname "$SEARCH_DIR")"
  if [ "$PARENT_DIR" = "$SEARCH_DIR" ]; then
    break
  fi
  SEARCH_DIR="$PARENT_DIR"
done

if [ -z "$PROJECT_ROOT" ]; then
  echo "ERROR: Could not locate project root from START_DIR=$START_DIR"
  echo "Expected files: requirements.txt and humanoid/pertubative_trained_rnn_rl.py"
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/humanoid/logs"

echo "JobID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CWD:  $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

module purge
module load cuda/12.8

echo "Loaded modules:"
module list

echo "nvidia-smi -L:"
nvidia-smi -L || true

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv at $VENV_DIR"
  $PYTHON_BIN -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -V
which python

python -m pip install --upgrade pip setuptools wheel

if [ -f "requirements.txt" ]; then
  echo "Installing requirements.txt"
  python -m pip install -r requirements.txt
else
  echo "ERROR: requirements.txt not found in $(pwd)"
  exit 1
fi

# Ensure CUDA-enabled PyTorch on the node.
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124

# RL dependencies for MuJoCo Humanoid.
# Pin to versions with prebuilt wheels on Python 3.9 to avoid source builds
# that require MUJOCO_PATH.
python -m pip install --upgrade "gymnasium[mujoco]==0.29.1" "mujoco==2.3.7"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

# Avoid thread oversubscription across many env worker processes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Headless MuJoCo defaults.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python3 humanoid/pertubative_trained_rnn_rl.py \
    --env_candidates "Humanoid-v5,Humanoid-v4" \
    --iters 250 \
    --pairs 512 \
    --sigma 0.03 \
    --theta_lr 0.01 \
    --hidden 1024 \
    --rank 32 \
    --k_in 50 \
    --leak 0.2 \
    --episodes_per_candidate 2 \
    --candidate_chunk 64 \
    --rollout_steps 500 \
    --torch_num_threads 4 \
    --log_every 5 \
    --checkpoint_every 10 \
    --results_root humanoid/results

echo "Humanoid perturbative RL training complete."
