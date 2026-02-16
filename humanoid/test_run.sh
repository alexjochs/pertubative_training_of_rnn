#!/bin/bash
#SBATCH --job-name=humanoid_mjx_test
#SBATCH --output=humanoid/logs/%x_%j.out
#SBATCH --error=humanoid/logs/%x_%j.err
#SBATCH --partition=sy-grp
#SBATCH --account=sy-grp
#SBATCH --nodelist=cn-x-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=00:30:00

set -euo pipefail

# Ensure we are in the repository root if submitted from the 'humanoid' folder
if [[ "$PWD" == */humanoid ]]; then
    cd ..
fi

mkdir -p humanoid/logs

echo "JobID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CWD:  $(pwd)"

module purge
module load cuda/12.8

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

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv at $VENV_DIR"
  $PYTHON_BIN -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade --only-binary=:all: \
  "jax[cuda12]==0.9.0.1" \
  "mujoco==3.5.0" \
  "mujoco-mjx==3.5.0" \
  "numpy" \
  "torch"

# Avoid CPU oversubscription and tell JAX to use CUDA on NVIDIA nodes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset JAX_PLATFORMS
export JAX_PLATFORM_NAME=cuda
export JAX_LOG_COMPILES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92

# Headless MuJoCo defaults.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python humanoid/pertubative_trained_rnn_rl.py \
    --env_candidates "Humanoid-v4" \
    --xml_path humanoid/humanoid.xml \
    --iters 5 \
    --hidden 256 \
    --pairs 32 \
    --episodes_per_candidate 1 \
    --rollout_steps 100 \
    --log_every 1 \
    --results_root humanoid/test_results

echo "Humanoid MJX perturbative RL test complete."
