#!/bin/bash
# Quick turnaround test script for Humanoid RL
# Run this on the cluster to verify the training loop integrity.

python humanoid/pertubative_trained_rnn_rl.py \
    --iters 5 \
    --hidden 256 \
    --pairs 32 \
    --episodes_per_candidate 1 \
    --rollout_steps 100 \
    --log_every 1 \
    --results_root humanoid/test_results
