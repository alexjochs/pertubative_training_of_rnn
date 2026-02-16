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
#SBATCH --time=00:20:00

# === ASSUMPTIONS ===
# 1. Submit this script from the 'humanoid/' directory: `sbatch test_run.sh`
# 2. Your virtual environment is in the parent directory: '../.venv'
# 3. Logs will be created in 'humanoid/logs/'
# 4. Results will be created in 'humanoid/test_results/'

set -e
mkdir -p logs

# 1. Setup Environment
module purge
module load cuda/12.8
source ../.venv/bin/activate

# 2. Run the test
# We stay in the 'humanoid/' folder so Python finds env.py, model.py, etc. locally.
python pertubative_trained_rnn_rl.py \
    --iters 5 \
    --hidden 256 \
    --pairs 32 \
    --results_root test_results

echo "Test complete."
