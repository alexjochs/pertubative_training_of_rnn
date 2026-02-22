#!/bin/bash
#SBATCH --job-name=humanoid_mjx_es
#SBATCH --output=search_logs/%x_%j.out
#SBATCH --error=search_logs/%x_%j.err
#SBATCH --partition=sy-grp
#SBATCH --account=sy-grp
#SBATCH --nodelist=cn-x-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00

set -euo pipefail

START_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SEARCH_DIR="$(cd "$START_DIR" && pwd)"
PROJECT_ROOT=""

for _ in 1 2 3 4 5 6; do
  if [ -f "$SEARCH_DIR/humanoid/requirements_mjx.txt" ] && [ -f "$SEARCH_DIR/humanoid/pertubative_trained_rnn_rl.py" ]; then
    PROJECT_ROOT="$SEARCH_DIR"
    break
  fi
  PARENT_DIR="$(dirname "$SEARCH_DIR")"
  if [ "$PARENT_DIR" = "$SEARCH_DIR" ]; then
    break
  fi
  SEARCH_DIR="$PARENT_DIR"
done


cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/humanoid/search_logs"

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

if [ -z "${PYTHON_BIN:-}" ]; then
  for candidate in python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi


PY_VER="$($PYTHON_BIN - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
VENV_DIR=".venv_py${PY_VER/./}"

echo "Selected Python: $PYTHON_BIN ($PY_VER)"
echo "Using venv: $VENV_DIR"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv at $VENV_DIR"
  $PYTHON_BIN -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -V
which python

python -m pip install --upgrade pip setuptools wheel

if [ -f "humanoid/requirements_mjx.txt" ]; then
  echo "Installing humanoid/requirements_mjx.txt"
  python -m pip install -r humanoid/requirements_mjx.txt
else
  echo "ERROR: humanoid/requirements_mjx.txt not found in $(pwd)"
  exit 1
fi

# MJX stack for CUDA 12.x GPUs (validated on this cluster).
python -m pip install --upgrade --only-binary=:all: \
  "jax[cuda12]==0.9.0.1" \
  "mujoco==3.5.0" \
  "mujoco-mjx==3.5.0"

python - <<'PY'
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

print("jax:", jax.__version__)
print("mujoco:", mujoco.__version__)
print("devices:", jax.devices())

x = jnp.ones((1024, 1024), dtype=jnp.float32)
y = x @ x
print("matmul check:", float(y[0, 0]))
print("mjx module:", mjx.__name__)
PY

# Avoid CPU oversubscription and tell JAX to use CUDA on NVIDIA nodes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS
export JAX_PLATFORM_NAME=cuda
export JAX_LOG_COMPILES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92

# Headless MuJoCo defaults.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python - <<'PY'
import jax
print("jax backend:", jax.default_backend())
print("jax devices:", jax.devices())
PY

# Default values
ITERS=5000
PAIRS=16384
SIGMA=0.03
LR=0.01
HIDDEN=1024
RANK=32
K_IN=50
LEAK=0.2
EPISODES=2
CHUNK=32768
STEPS=500
CANDIDATES=1024

# Parse arguments
for i in "$@"; do
  case $i in
    --iters=*)
      ITERS="${i#*=}"
      ;;
    --pairs=*)
      PAIRS="${i#*=}"
      ;;
    --sigma=*)
      SIGMA="${i#*=}"
      ;;
    --theta_lr=*)
      LR="${i#*=}"
      ;;
    --hidden=*)
      HIDDEN="${i#*=}"
      ;;
    --rank=*)
      RANK="${i#*=}"
      ;;
    --k_in=*)
      K_IN="${i#*=}"
      ;;
    --leak=*)
      LEAK="${i#*=}"
      ;;
    --episodes=*)
      EPISODES="${i#*=}"
      ;;
    --chunk=*|--candidate_chunk=*)
      CHUNK="${i#*=}"
      ;;
    --steps=*)
      STEPS="${i#*=}"
      ;;
    --candidates=*|--num_candidates=*)
      CANDIDATES="${i#*=}"
      ;;
    *)
      # Unknown option
      ;;
  esac
done

# Create a descriptive tag for this run
RUN_TAG="it${ITERS}_p${PAIRS}_h${HIDDEN}_k${K_IN}_cand${CANDIDATES}"
echo "Run Tag: $RUN_TAG"

python humanoid/pertubative_trained_rnn_rl.py \
    --env_candidates "Humanoid-v4" \
    --xml_path humanoid/humanoid.xml \
    --iters "$ITERS" \
    --pairs "$PAIRS" \
    --sigma "$SIGMA" \
    --theta_lr "$LR" \
    --hidden "$HIDDEN" \
    --rank "$RANK" \
    --k_in "$K_IN" \
    --leak "$LEAK" \
    --episodes_per_candidate "$EPISODES" \
    --candidate_chunk "$CHUNK" \
    --rollout_steps "$STEPS" \
    --log_every 1 \
    --checkpoint_every 1 \
    --num_candidates "$CANDIDATES" \
    --search_burst_iters 20 \
    --results_root "humanoid/search_results/$RUN_TAG"


# Optionally rename the SLURM output file at the end to include the tag
if [ -n "${SLURM_JOB_ID:-}" ]; then
  NEW_LOG="humanoid/search_logs/humanoid_${RUN_TAG}_${SLURM_JOB_ID}.out"
  echo "Training finished. Copying log to $NEW_LOG"
  cp "search_logs/humanoid_mjx_es_${SLURM_JOB_ID}.out" "$NEW_LOG" || true
fi

echo "Humanoid MJX perturbative RL training complete."
