#!/bin/bash
#SBATCH --job-name=humanoid_test_search
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
#SBATCH --time=00:30:00

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

source "$VENV_DIR/bin/activate"

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

# Test values to quickly verify compile shapes and the halving loop
ITERS=5
PAIRS=128
SIGMA=0.03
LR=0.01
HIDDEN=512
RANK=32
K_IN=50
LEAK=0.2
EPISODES=1
CHUNK=16
STEPS=200
CANDIDATES=16
SEARCH_BURST=3

# Create a descriptive tag for this run
RUN_TAG="test_search_c${CANDIDATES}_i${ITERS}"
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
    --search_burst_iters "$SEARCH_BURST" \
    --results_root "humanoid/search_results/$RUN_TAG"

echo "Humanoid MJX Test completed."
