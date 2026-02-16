#!/bin/bash
#SBATCH --job-name=humanoid_mjx_es
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

if [ -z "$PROJECT_ROOT" ]; then
  echo "ERROR: Could not locate project root from START_DIR=$START_DIR"
  echo "Expected files: humanoid/requirements_mjx.txt and humanoid/pertubative_trained_rnn_rl.py"
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

if [ -z "${PYTHON_BIN:-}" ]; then
  for candidate in python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  echo "ERROR: no Python interpreter found (tried python3.11, python3.10, python3)"
  exit 1
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

# MJX stack for CUDA 12.x GPUs.
# Try latest first; fall back to a known-compatible stack if wheels are unavailable
# on this cluster image (common when glibc is older than manylinux_2_28 baseline).
echo "glibc:"
ldd --version | head -n 1 || true

MJX_MUJOCO_VERSION="${MJX_MUJOCO_VERSION:-3.5.0}"
MJX_FALLBACK_MUJOCO_VERSION="${MJX_FALLBACK_MUJOCO_VERSION:-3.1.6}"
JAX_FALLBACK_VERSION="${JAX_FALLBACK_VERSION:-0.4.30}"

echo "Installing MJX stack (preferred): mujoco/mujoco-mjx ${MJX_MUJOCO_VERSION}"
if ! python -m pip install --upgrade --only-binary=:all: \
    "jax[cuda12]" \
    "mujoco==${MJX_MUJOCO_VERSION}" \
    "mujoco-mjx==${MJX_MUJOCO_VERSION}"; then
  echo "Preferred MJX stack install failed; falling back to mujoco/mujoco-mjx ${MJX_FALLBACK_MUJOCO_VERSION}"
  python -m pip install --upgrade --only-binary=:all: \
    "jax[cuda12]==${JAX_FALLBACK_VERSION}" \
    "mujoco==${MJX_FALLBACK_MUJOCO_VERSION}" \
    "mujoco-mjx==${MJX_FALLBACK_MUJOCO_VERSION}"
fi

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
unset JAX_PLATFORMS
export JAX_PLATFORM_NAME=cuda
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

# Optional preflight (compile-only):
# python humanoid/pertubative_trained_rnn_rl.py \
#     --env_candidates "Humanoid-v4" \
#     --xml_path humanoid/humanoid.xml \
#     --hidden 1024 \
#     --rank 32 \
#     --candidate_chunk 256 \
#     --episodes_per_candidate 2 \
#     --rollout_steps 500 \
#     --dry_run_compile \
#     --dry_run_repeats 2

python humanoid/pertubative_trained_rnn_rl.py \
    --env_candidates "Humanoid-v4" \
    --xml_path humanoid/humanoid.xml \
    --iters 10 \
    --pairs 4096 \
    --sigma 0.03 \
    --theta_lr 0.01 \
    --hidden 1024 \
    --rank 32 \
    --k_in 50 \
    --leak 0.2 \
    --episodes_per_candidate 2 \
    --candidate_chunk 256 \
    --rollout_steps 500 \
    --autotune_warmup_iters 3 \
    --autotune_pairs_cap 32768 \
    --autotune_chunk_cap 1024 \
    --log_every 1 \
    --checkpoint_every 1 \
    --headroom_target_iter_sec 600 \
    --results_root humanoid/results

echo "Humanoid MJX perturbative RL training complete."
