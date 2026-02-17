# Humanoid MJX Notes

## Scope
This folder is for the Humanoid RL ES experiment only (`pertubative_trained_rnn_rl.py` + `run_h200.sh`).
It should use its own environment and dependency path, separate from the root perturbative vision script.

## Cluster Facts (learned on this setup)
- CUDA module on cluster: `cuda/12.8`
- Confirmed GPU target: NVIDIA H200
- Preferred Python on cluster: `python3.11` (script auto-selects `python3.11`, then `python3.10`, then `python3`)
- Venv path is versioned: `.venv_py311` (or matching selected Python)

## Known-good MJX Stack (current)
- `jax[cuda12]==0.9.0.1`
- `mujoco==3.5.0`
- `mujoco-mjx==3.5.0`
- `torch` (used for reservoir init and ES optimizer math)
- Installed in `humanoid/run_h200.sh` with `--only-binary=:all:`

## Environment Variables (important)
- `JAX_PLATFORM_NAME=cuda`
- `JAX_PLATFORMS` should be unset (avoid backend probing issues)
- `MUJOCO_GL=egl`
- `PYOPENGL_PLATFORM=egl`
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`

## XML / Model Notes
- Runtime XML is pinned in script launch args:
  - `--xml_path humanoid/humanoid.xml`
- Current XML is Gym-style Humanoid with:
  - `Action dim: 17`
  - `Obs dim: 376` (with current env spec path)
- Solver in XML is set to `CG` (MJX-compatible on this stack).
- Tendons are currently present in `humanoid/humanoid.xml` and load under MJX `3.5.0`.

## Warnings Seen That Are Safe
- `Failed to import warp: No module named 'warp'`
- `Failed to import mujoco_warp: No module named 'warp'`
- These are optional acceleration backends; not required for JAX+MJX GPU execution.

## Run Path
- **Primary launcher**: `humanoid/run_h200.sh`
- **Critical Boilerplate**: Any new batch script **MUST** copy the `run_h200.sh` boilerplate for Python probing (3.11 check), venv creation, and library pins. DO NOT simplify the environment setup; it is fragile on this cluster.
- **Entrypoint**: `humanoid/pertubative_trained_rnn_rl.py`

## Fixed Runtime Shape
- Runtime shape is fixed:
  - `pairs=8192`
  - `candidate_chunk=2048`
- The training entrypoint performs a one-time warmup/compile for this shape and writes a marker in:
  - `humanoid/jax_compile_cache`
- JAX persistent compilation cache is enabled via:
  - `--compile_cache_dir` (defaults to `humanoid/jax_compile_cache`)

## Performance & Optimization Notes (H200 Verified)
- **Throughput**: ~41,000 FPS verified with 16,384 candidates.
- **Physics Stabilization**:
  - `mjx.step` is wrapped in `jax.lax.cond` to catch NaNs in `qpos`/`qvel`/`action` before execution.
  - Solver iterations explicitly clamped to `4` ( Newton) to prevent "silent hangs" on bad states.
- **Chunking Strategy**: 
  - `candidate_chunk=2048` provides good occupancy.
  - `rollout_fn` uses `jax.lax.scan` sequentially over chunks to fit 16k population in memory.
  - Initial compilation takes ~2-3 minutes; subsequent iterations are <10s execution time.

## Known Failure Patterns and Fix Direction
- **Silent Hangs (0 FPS)**:
  - Caused by physics solver infinite loops on NaN states. Fixed by NaN guards in `env.py`.
- `NotImplementedError: mjtSolver.mjSOL_PGS`
  - Use XML solver `CG` or `NEWTON`.
- Backend init weirdness mentioning ROCm while on NVIDIA:
  - Ensure `JAX_PLATFORM_NAME=cuda` and `JAX_PLATFORMS` unset.
- MuJoCo source build attempts:
  - Keep wheel-only installs and pinned versions in `run_h200.sh`.

## Cluster Submission Guide

### 1. Directory Assumptions
- **Submit Site**: Always run `sbatch` from within the `humanoid/` directory.
- **Log Location**: Slurm will write `.out` and `.err` files to `humanoid/logs/`.
- **Imports**: Running from `humanoid/` ensures Python finds `env.py`, `model.py`, and `utils.py` as local imports.

### 2. How to Run
```bash
cd humanoid/
sbatch test_run.sh   # For quick verification (5 iters, 32 pairs)
sbatch run_h200.sh   # For production training (100 iters, 8192 pairs)
```

### 3. Environment Setup (Automatic in SBATCH)
The scripts handle the following automatically:
- **Modules**: `module load cuda/12.8`
- **Venv**: Creates/Uses `../.venv_mjx` (stored in Repo Root).
- **Versioning**: Pins `jax[cuda12]==0.9.0.1` and `mujoco==3.5.0` to match the cluster CUDA version.

### 4. Verifying Success
1. Check `humanoid/logs/` for the latest `.out` file. 
2. Verify JAX output: `jax devices: [gpu(id=0)]`.
3. Training metrics appear in `humanoid/results/` (or `test_results/` for the test script).

## Coordination Notes
- Keep changes in this folder isolated from root project training unless explicitly requested.
- If changing MJX/JAX versions, validate with:
  - printed `jax.__version__`
  - printed `mujoco.__version__`
  - `jax.default_backend()`
  - `jax.devices()`
