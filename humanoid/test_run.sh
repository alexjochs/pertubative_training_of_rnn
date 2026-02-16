#!/bin/bash
#SBATCH --job-name=humanoid_test
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=sy-grp
#SBATCH --account=sy-grp
#SBATCH --nodelist=cn-x-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# === ASSUMPTIONS ===
# 1. Run this from: `~/hpc-share/scratch_repos/pertubative_training_of_rnn/humanoid`
# 2. It will create/use a venv at `../.venv_mjx`
# 3. It installs JAX/MJX/MuJoCo specific to CUDA 12.8 on the nodes.

set -euo pipefail
mkdir -p logs

# 1. Load Cluster Modules
module purge
module load cuda/12.8

# 2. Setup/Activate Venv (using a dedicated mjx venv to avoid conflicts)
VENV_PATH="../.venv_mjx"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating new MJX environment..."
    python3 -m venv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"

# 3. Install Dependencies (The "Working" Stack)
echo "Ensuring JAX/MJX stack is installed..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade --only-binary=:all: \
  "jax[cuda12]==0.9.0.1" \
  "mujoco==3.5.0" \
  "mujoco-mjx==3.5.0" \
  "torch" \
  "numpy"

# 4. Verify the stack before running
python -c "import jax; import mujoco; print(f'JAX: {jax.devices()}'); print('MuJoCo Loaded Successfully')"

# 5. Run the Training Script
# env.py, model.py, etc. are in the current folder, so we run direct.
python pertubative_trained_rnn_rl.py \
    --iters 5 \
    --hidden 256 \
    --pairs 32 \
    --results_root test_results

echo "Run complete. Check humanoid/logs/ and humanoid/test_results/"
