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
- Primary launcher:
  - `humanoid/run_h200.sh`
- It installs:
  - `humanoid/requirements_mjx.txt` (minimal base deps)
  - then pinned MJX/JAX stack
- Training entrypoint:
  - `humanoid/pertubative_trained_rnn_rl.py`

## Fixed Runtime Shape
- Runtime shape is fixed:
  - `pairs=8192`
  - `candidate_chunk=256`
- The training entrypoint performs a one-time warmup/compile for this shape and writes a marker in:
  - `humanoid/jax_compile_cache`
- JAX persistent compilation cache is enabled via:
  - `--compile_cache_dir` (defaults to `humanoid/jax_compile_cache`)

## Known Failure Patterns and Fix Direction
- `NotImplementedError: mjtSolver.mjSOL_PGS`
  - Use XML solver `CG` or `NEWTON`.
- Backend init weirdness mentioning ROCm while on NVIDIA:
  - Ensure `JAX_PLATFORM_NAME=cuda` and `JAX_PLATFORMS` unset.
- MuJoCo source build attempts:
  - Keep wheel-only installs and pinned versions in `run_h200.sh`.

## Coordination Notes
- Keep changes in this folder isolated from root project training unless explicitly requested.
- If changing MJX/JAX versions, validate with:
  - printed `jax.__version__`
  - printed `mujoco.__version__`
  - `jax.default_backend()`
  - `jax.devices()`
