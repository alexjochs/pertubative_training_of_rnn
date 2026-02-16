"""
Perturbative ES-trained reservoir policy for MuJoCo Humanoid using MJX.

This is a GPU-oriented migration of the previous Gym VectorEnv + PyTorch version.
Rollouts are evaluated with batched MJX physics and JAX-jitted policy inference.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
except Exception as exc:
    raise RuntimeError(
        "torch is required. Install with: pip install torch"
    ) from exc

try:
    import jax
    import jax.numpy as jnp
except Exception as exc:
    raise RuntimeError(
        "jax is required. Install with: pip install 'jax[cuda12]'"
    ) from exc

try:
    import mujoco
    from mujoco import mjx
except Exception as exc:
    raise RuntimeError(
        "mujoco + mujoco-mjx are required. Install with: pip install mujoco mujoco-mjx"
    ) from exc

FIXED_PAIRS = 8192
FIXED_CANDIDATE_CHUNK = 256
DEFAULT_JAX_COMPILE_CACHE_DIR = os.path.join("humanoid", "jax_compile_cache")


@dataclass
class ReservoirConfig:
    N: int = 1024
    D: int = 376
    A: int = 17
    rank: int = 32
    leak: float = 0.2
    h_clip: float = 80.0
    k_in: int = 50
    w0_std: float = 1.0
    win_std: float = 0.15


@dataclass(frozen=True)
class HumanoidSpec:
    obs_variant: str
    include_contact_cost: bool
    forward_reward_weight: float = 1.25
    ctrl_cost_weight: float = 0.1
    contact_cost_weight: float = 5e-7
    contact_cost_max: float = 10.0
    healthy_reward: float = 5.0
    terminate_when_unhealthy: bool = True
    healthy_z_min: float = 1.0
    healthy_z_max: float = 2.0
    reset_noise_scale: float = 1e-2
    frame_skip: int = 5


class CSVLogger:
    def __init__(self, filepath: str, fieldnames: List[str]):
        self.filepath = filepath
        self.file = open(filepath, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.file.flush()

    def log(self, data: Dict[str, float]) -> None:
        self.writer.writerow(data)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def make_eval_stats() -> Dict[str, Any]:
    return {
        "wall_s": 0.0,
        "chunks": 0,
        "loop_steps": 0,
        "effective_env_steps": 0,
        "actual_env_steps": 0,
    }


def merge_eval_stats(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for key in ("wall_s",):
        dst[key] += float(src[key])
    for key in ("chunks", "loop_steps", "effective_env_steps", "actual_env_steps"):
        dst[key] += int(src[key])


def save_config(args: argparse.Namespace, filepath: str) -> None:
    with open(filepath, "w") as f:
        json.dump(vars(args), f, indent=2)


def rank_transform(values: jnp.ndarray, maximize: bool = True) -> jnp.ndarray:
    n = int(values.shape[0])
    if n <= 1:
        return jnp.zeros_like(values)

    order = jnp.argsort(values)
    ranks = jnp.zeros((n,), dtype=jnp.float32).at[order].set(jnp.arange(n, dtype=jnp.float32))

    if maximize:
        w = ranks / float(n - 1) - 0.5
    else:
        w = -(ranks / float(n - 1) - 0.5)

    return w - jnp.mean(w)


def resolve_env_spec(candidates: List[str]) -> Tuple[str, HumanoidSpec]:
    for env_id in candidates:
        env_id = env_id.strip()
        if not env_id:
            continue

        name = env_id.lower()
        if name.startswith("humanoid-v5"):
            return env_id, HumanoidSpec(obs_variant="v5", include_contact_cost=True)
        if name.startswith("humanoid-v4"):
            # Matches known v4 behavior (contact term effectively disabled).
            return env_id, HumanoidSpec(obs_variant="v4", include_contact_cost=False)
        if name.startswith("humanoid-v3") or name.startswith("humanoid-v2") or name == "humanoid":
            return env_id, HumanoidSpec(obs_variant="v4", include_contact_cost=True)

    raise RuntimeError(
        "Only Humanoid variants are supported in this MJX script. "
        f"Got candidates: {candidates}"
    )


def resolve_humanoid_xml(xml_path: str) -> str:
    candidates: List[str] = []
    if xml_path:
        candidates.append(xml_path)

    env_xml = os.environ.get("HUMANOID_XML", "")
    if env_xml:
        candidates.append(env_xml)

    candidates.append(os.path.join(os.getcwd(), "humanoid", "humanoid.xml"))
    candidates.append(os.path.join(os.getcwd(), "humanoid.xml"))

    try:
        import gymnasium  # type: ignore

        gym_xml = os.path.join(
            os.path.dirname(gymnasium.__file__),
            "envs",
            "mujoco",
            "assets",
            "humanoid.xml",
        )
        candidates.append(gym_xml)
    except Exception:
        pass

    mujoco_dir = os.path.dirname(mujoco.__file__)
    candidates.append(os.path.abspath(os.path.join(mujoco_dir, "..", "..", "share", "mujoco", "model", "humanoid.xml")))
    candidates.append(os.path.join(mujoco_dir, "mjx", "test_data", "humanoid", "humanoid.xml"))

    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)

    checked = "\n  - " + "\n  - ".join(candidates)
    raise RuntimeError(
        "Could not locate humanoid.xml. Provide --xml_path or set HUMANOID_XML. "
        f"Checked:{checked}"
    )


def infer_obs_dim(spec: HumanoidSpec, model: mujoco.MjModel) -> int:
    nq = int(model.nq)
    nv = int(model.nv)
    nbody = int(model.nbody)
    nu = int(model.nu)

    if spec.obs_variant == "v5":
        return (nq - 2) + nv + (nbody - 1) * 10 + (nbody - 1) * 6 + nu + (nbody - 1) * 6

    return (nq - 2) + nv + nbody * 10 + nbody * 6 + nv + nbody * 6


class ReservoirPolicy:
    """
    Reservoir dynamics with trainable low-rank adapter and action head in theta.

    h_{t+1} = (1-leak)h_t + leak * tanh(W0 h_t + U(V^T h_t) + Win x_t + b)
    a_t = tanh(Wa h_t + ba), scaled to action bounds.
    """

    def __init__(
        self,
        cfg: ReservoirConfig,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ):
        self.cfg = cfg

        g = torch.Generator(device="cpu")
        g.manual_seed(1234)
        win_t = torch.randn((cfg.N, cfg.D), generator=g, dtype=torch.float32) * cfg.win_std
        self.Win = jnp.asarray(win_t.numpy(), dtype=jnp.float32)
        self.b = jnp.asarray(torch.zeros((cfg.N,), dtype=torch.float32).numpy(), dtype=jnp.float32)
        self.W0 = self._make_sparse_w0(cfg.N, cfg.k_in, w_std=cfg.w0_std)

        low = jnp.asarray(action_low, dtype=jnp.float32)
        high = jnp.asarray(action_high, dtype=jnp.float32)
        self.action_scale = 0.5 * (high - low)
        self.action_bias = 0.5 * (high + low)

    @property
    def theta_dim(self) -> int:
        n, r, a = self.cfg.N, self.cfg.rank, self.cfg.A
        return 2 * n * r + a * n + a

    @staticmethod
    def _make_sparse_w0(N: int, k_in: int, w_std: float) -> jnp.ndarray:
        print(f"  [Reservoir] Generating sparse W0 ({N}x{N})...", flush=True)
        t0 = time.time()

        g = torch.Generator(device="cpu")
        g.manual_seed(1234)

        scale = w_std / math.sqrt(float(k_in))
        rows = torch.arange(N, dtype=torch.long).repeat_interleave(k_in)
        cols = torch.randint(0, N, (N * k_in,), generator=g, dtype=torch.long)
        vals = torch.randn((N * k_in,), generator=g, dtype=torch.float32) * scale

        idx = torch.stack([rows, cols], dim=0)
        dense = torch.sparse_coo_tensor(idx, vals, (N, N), device="cpu").coalesce().to_dense()

        dt = time.time() - t0
        print(f"  [Reservoir] W0 generated in {dt:.2f}s", flush=True)
        return jnp.asarray(dense.numpy(), dtype=jnp.float32)

    def split_theta(
        self, theta: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        theta shape: [C, theta_dim]
        returns:
          U: [C, N, R]
          V: [C, N, R]
          Wa: [C, A, N]
          ba: [C, A]
        """
        C = int(theta.shape[0])
        n, r, a = self.cfg.N, self.cfg.rank, self.cfg.A

        uv = 2 * n * r
        wa = a * n

        U = theta[:, : n * r].reshape((C, n, r))
        V = theta[:, n * r : uv].reshape((C, n, r))
        Wa = theta[:, uv : uv + wa].reshape((C, a, n))
        ba = theta[:, uv + wa : uv + wa + a].reshape((C, a))
        return U, V, Wa, ba

    def policy_step(
        self,
        h: jnp.ndarray,
        obs: jnp.ndarray,
        cand_U: jnp.ndarray,
        cand_V: jnp.ndarray,
        cand_Wa: jnp.ndarray,
        cand_ba: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        h: [C, E, N], obs: [C, E, D]
        cand_U/V: [C, N, R], cand_Wa: [C, A, N], cand_ba: [C, A]
        returns:
          next_h: [C, E, N]
          action: [C, E, A]
          exploded: [C, E] bool
        """
        C, E, N = h.shape
        D = obs.shape[-1]
        if D != self.cfg.D:
            raise RuntimeError(f"Observation dim mismatch. Expected {self.cfg.D}, got {D}")

        h_flat = h.reshape((C * E, N))
        rec0 = h_flat @ self.W0.T
        rec0 = rec0.reshape((C, E, N))

        low_tmp = jnp.einsum("cen,cnr->cer", h, cand_V)
        low = jnp.einsum("cer,cnr->cen", low_tmp, cand_U)

        inp = jnp.einsum("ced,nd->cen", obs, self.Win)

        pre = rec0 + low + inp + self.b
        nh = jnp.tanh(pre)
        next_h = (1.0 - self.cfg.leak) * h + self.cfg.leak * nh

        exploded = (~jnp.all(jnp.isfinite(next_h), axis=2)) | (
            jnp.max(jnp.abs(next_h), axis=2) > self.cfg.h_clip
        )
        safe = jnp.nan_to_num(next_h, nan=0.0, posinf=0.0, neginf=0.0)
        next_h = jnp.where(exploded[..., None], 0.0, safe)

        act_pre = jnp.einsum("cen,can->cea", next_h, cand_Wa) + cand_ba[:, None, :]
        action = jnp.tanh(act_pre)
        action = action * self.action_scale[None, None, :] + self.action_bias[None, None, :]

        return next_h, action, exploded


def torch_to_jnp(x: torch.Tensor) -> jnp.ndarray:
    return jnp.asarray(x.detach().cpu().numpy(), dtype=jnp.float32)


class CandidateEvaluator:
    def __init__(
        self,
        env_id: str,
        xml_path: str,
        env_spec: HumanoidSpec,
        obs_dim: int,
        action_dim: int,
        chunk_candidates: int,
        episodes_per_candidate: int,
        rollout_steps: int,
        policy: ReservoirPolicy,
    ):
        self.env_id = env_id
        self.xml_path = xml_path
        self.env_spec = env_spec
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_candidates = chunk_candidates
        self.episodes_per_candidate = episodes_per_candidate
        self.rollout_steps = rollout_steps
        self.policy = policy

        self.num_envs = self.chunk_candidates * self.episodes_per_candidate

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        solver = int(self.model.opt.solver)
        solver_cg = int(mujoco.mjtSolver.mjSOL_CG)
        solver_newton = int(mujoco.mjtSolver.mjSOL_NEWTON)
        if solver not in (solver_cg, solver_newton):
            raise RuntimeError(
                "MJX supports solver=CG or solver=NEWTON for this stack. "
                f"Loaded XML has solver={mujoco.mjtSolver(self.model.opt.solver)}: {xml_path}."
            )
        self.mjx_model = mjx.put_model(self.model)

        if int(self.model.nu) != self.action_dim:
            raise RuntimeError(
                f"Action dim mismatch from model {self.model.nu} vs expected {self.action_dim}"
            )

        inferred = infer_obs_dim(env_spec, self.model)
        if inferred != self.obs_dim:
            raise RuntimeError(
                f"Observation dim mismatch from spec/model {inferred} vs expected {self.obs_dim}"
            )

        if self.model.nkey > 0:
            init_qpos_np = np.asarray(self.model.key_qpos[0], dtype=np.float32)
            try:
                init_qvel_np = np.asarray(self.model.key_qvel[0], dtype=np.float32)
            except Exception:
                init_qvel_np = np.zeros((int(self.model.nv),), dtype=np.float32)
        else:
            init_qpos_np = np.asarray(self.model.qpos0, dtype=np.float32)
            init_qvel_np = np.zeros((int(self.model.nv),), dtype=np.float32)

        body_mass = jnp.asarray(self.model.body_mass, dtype=jnp.float32)
        body_mass_sum = jnp.sum(body_mass)
        init_qpos = jnp.asarray(init_qpos_np, dtype=jnp.float32)
        init_qvel = jnp.asarray(init_qvel_np, dtype=jnp.float32)

        frame_skip = int(self.env_spec.frame_skip)
        dt = float(self.model.opt.timestep * frame_skip)
        healthy_z_min = jnp.array(self.env_spec.healthy_z_min, dtype=jnp.float32)
        healthy_z_max = jnp.array(self.env_spec.healthy_z_max, dtype=jnp.float32)
        healthy_reward = jnp.array(self.env_spec.healthy_reward, dtype=jnp.float32)
        terminate_when_unhealthy = jnp.array(self.env_spec.terminate_when_unhealthy, dtype=jnp.bool_)

        forward_reward_weight = jnp.array(self.env_spec.forward_reward_weight, dtype=jnp.float32)
        ctrl_cost_weight = jnp.array(self.env_spec.ctrl_cost_weight, dtype=jnp.float32)
        contact_cost_weight = jnp.array(self.env_spec.contact_cost_weight, dtype=jnp.float32)
        contact_cost_max = jnp.array(self.env_spec.contact_cost_max, dtype=jnp.float32)
        include_contact_cost = bool(self.env_spec.include_contact_cost)

        reset_noise_scale = jnp.array(self.env_spec.reset_noise_scale, dtype=jnp.float32)

        obs_variant = self.env_spec.obs_variant
        actuator_force_start = 6 if (obs_variant == "v5" and int(self.model.nv) - int(self.model.nu) == 6) else 0

        C_full = self.chunk_candidates
        E = self.episodes_per_candidate
        B = self.num_envs
        D = self.obs_dim
        A = self.action_dim
        N = self.policy.cfg.N
        nq = int(self.model.nq)
        nv = int(self.model.nv)

        def mass_center_x(data: mjx.Data) -> jnp.ndarray:
            return jnp.sum(body_mass * data.xipos[:, 0]) / body_mass_sum

        def get_obs(data: mjx.Data) -> jnp.ndarray:
            position = data.qpos[2:]
            velocity = data.qvel

            if obs_variant == "v5":
                com_inertia = data.cinert[1:].reshape((-1,))
                com_velocity = data.cvel[1:].reshape((-1,))
                external_contact_forces = data.cfrc_ext[1:].reshape((-1,))
                actuator_forces = data.qfrc_actuator[actuator_force_start:]
            else:
                com_inertia = data.cinert.reshape((-1,))
                com_velocity = data.cvel.reshape((-1,))
                external_contact_forces = data.cfrc_ext.reshape((-1,))
                actuator_forces = data.qfrc_actuator

            obs = jnp.concatenate(
                (
                    position,
                    velocity,
                    com_inertia,
                    com_velocity,
                    actuator_forces,
                    external_contact_forces,
                ),
                axis=0,
            )
            return obs.astype(jnp.float32)

        def step_env(data: mjx.Data, action: jnp.ndarray) -> Tuple[mjx.Data, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            x_before = mass_center_x(data)
            data = data.replace(ctrl=action)

            data = jax.lax.fori_loop(0, frame_skip, lambda _, d: mjx.step(self.mjx_model, d), data)

            x_after = mass_center_x(data)
            x_vel = (x_after - x_before) / dt

            z = data.qpos[2]
            is_healthy = (z > healthy_z_min) & (z < healthy_z_max)
            terminated = terminate_when_unhealthy & (~is_healthy)

            healthy_term = jnp.where(is_healthy | terminate_when_unhealthy, healthy_reward, 0.0)
            forward_reward = forward_reward_weight * x_vel
            ctrl_cost = ctrl_cost_weight * jnp.sum(jnp.square(action))

            if include_contact_cost:
                contact_cost = contact_cost_weight * jnp.minimum(
                    jnp.sum(jnp.square(data.cfrc_ext)),
                    contact_cost_max,
                )
            else:
                contact_cost = jnp.array(0.0, dtype=jnp.float32)

            reward = forward_reward + healthy_term - ctrl_cost - contact_cost
            obs = get_obs(data)
            return data, obs, reward.astype(jnp.float32), terminated

        def reset_one(key: jnp.ndarray) -> Tuple[mjx.Data, jnp.ndarray]:
            k1, k2 = jax.random.split(key)

            qpos = init_qpos + jax.random.uniform(
                k1,
                shape=(nq,),
                minval=-reset_noise_scale,
                maxval=reset_noise_scale,
                dtype=jnp.float32,
            )
            qvel = init_qvel + jax.random.uniform(
                k2,
                shape=(nv,),
                minval=-reset_noise_scale,
                maxval=reset_noise_scale,
                dtype=jnp.float32,
            )

            data = mjx.make_data(self.mjx_model)
            data = data.replace(qpos=qpos, qvel=qvel)
            data = mjx.forward(self.mjx_model, data)
            obs = get_obs(data)
            return data, obs

        def reset_batch(reset_key: jnp.ndarray) -> Tuple[mjx.Data, jnp.ndarray]:
            keys = jax.random.split(reset_key, B)
            return jax.vmap(reset_one)(keys)

        def evaluate_chunk_impl(
            chunk_theta: jnp.ndarray,
            reset_key: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            cand_U, cand_V, cand_Wa, cand_ba = self.policy.split_theta(chunk_theta)

            data, obs_flat = reset_batch(reset_key)
            obs = obs_flat.reshape((C_full, E, D))

            h = jnp.zeros((C_full, E, N), dtype=jnp.float32)
            returns = jnp.zeros((C_full, E), dtype=jnp.float32)
            lengths = jnp.zeros((C_full, E), dtype=jnp.int32)
            alive = jnp.ones((C_full, E), dtype=jnp.bool_)
            exploded_by_candidate = jnp.zeros((C_full,), dtype=jnp.bool_)

            def cond_fn(carry):
                step_idx = carry[0]
                alive = carry[6]
                return (step_idx < self.rollout_steps) & jnp.any(alive)

            def body_fn(carry):
                step_idx, h, data, obs, returns, lengths, alive, exploded_by_candidate, eff_steps, loop_steps = carry
                active_count = jnp.sum(alive.astype(jnp.int32))

                alive_mask = alive[..., None].astype(jnp.float32)
                h_live = h * alive_mask

                h_next, action, exploded = self.policy.policy_step(
                    h_live,
                    obs,
                    cand_U,
                    cand_V,
                    cand_Wa,
                    cand_ba,
                )

                exploded_by_candidate_next = exploded_by_candidate | jnp.any(exploded, axis=1)
                h_next = jnp.where(exploded[..., None], 0.0, h_next)

                action_flat = action.reshape((B, A))
                data_next, obs_next_flat, reward_flat, terminated_flat = jax.vmap(step_env)(
                    data,
                    action_flat,
                )

                obs_next = obs_next_flat.reshape((C_full, E, D))
                reward = reward_flat.reshape((C_full, E))
                terminated = terminated_flat.reshape((C_full, E))

                returns_next = returns + jnp.where(alive, reward, 0.0)
                newly_done = alive & terminated
                lengths_next = jnp.where(newly_done, step_idx + 1, lengths)
                alive_next = alive & (~terminated)

                return (
                    step_idx + 1,
                    h_next,
                    data_next,
                    obs_next,
                    returns_next,
                    lengths_next,
                    alive_next,
                    exploded_by_candidate_next,
                    eff_steps + active_count,
                    loop_steps + jnp.array(1, dtype=jnp.int32),
                )

            carry0 = (
                jnp.array(0, dtype=jnp.int32),
                h,
                data,
                obs,
                returns,
                lengths,
                alive,
                exploded_by_candidate,
                jnp.array(0, dtype=jnp.int32),
                jnp.array(0, dtype=jnp.int32),
            )
            carry_f = jax.lax.while_loop(cond_fn, body_fn, carry0)

            _, _, _, _, returns_f, lengths_f, alive_f, exploded_f, effective_env_steps, loop_steps = carry_f
            lengths_f = jnp.where(alive_f, self.rollout_steps, lengths_f)

            returns_c = jnp.mean(returns_f, axis=1)
            lengths_c = jnp.mean(lengths_f.astype(jnp.float32), axis=1)
            actual_env_steps = loop_steps * B

            return returns_c, lengths_c, exploded_f, effective_env_steps, actual_env_steps, loop_steps

        self._evaluate_chunk_jit = jax.jit(evaluate_chunk_impl)

    def close(self) -> None:
        return

    def evaluate_chunk(
        self,
        chunk_theta: jnp.ndarray,
        seed_base: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Evaluate up to chunk_candidates parameters.

        Returns:
          returns: [C]
          mean_lengths: [C]
          exploded: [C] bool
          eval stats dict
        """
        C_active = int(chunk_theta.shape[0])
        C_full = self.chunk_candidates

        if C_active > C_full:
            raise RuntimeError(f"Chunk too large: {C_active} > {C_full}")

        if C_active < C_full:
            pad_count = C_full - C_active
            pad = jnp.repeat(chunk_theta[-1:, :], repeats=pad_count, axis=0)
            theta_eval = jnp.concatenate([chunk_theta, pad], axis=0)
        else:
            theta_eval = chunk_theta

        t0 = time.perf_counter()
        reset_key = jax.random.PRNGKey(int(seed_base))
        r, l, e, eff_steps, act_steps, loop_steps = self._evaluate_chunk_jit(theta_eval, reset_key)

        r_np = np.asarray(r[:C_active], dtype=np.float64)
        l_np = np.asarray(l[:C_active], dtype=np.float64)
        e_np = np.asarray(e[:C_active], dtype=bool)
        eff_steps_i = int(np.asarray(eff_steps))
        act_steps_i = int(np.asarray(act_steps))
        loop_steps_i = int(np.asarray(loop_steps))
        wall_s = time.perf_counter() - t0

        stats = make_eval_stats()
        stats["chunks"] = 1
        stats["loop_steps"] = loop_steps_i
        stats["effective_env_steps"] = eff_steps_i
        stats["actual_env_steps"] = act_steps_i
        stats["wall_s"] = wall_s
        return r_np, l_np, e_np, stats

    def evaluate_population(
        self,
        population_theta: jnp.ndarray,
        seed_base: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        num_candidates = int(population_theta.shape[0])
        returns = np.zeros(num_candidates, dtype=np.float64)
        lengths = np.zeros(num_candidates, dtype=np.float64)
        exploded = np.zeros(num_candidates, dtype=bool)

        total_stats = make_eval_stats()

        for chunk_idx, start in enumerate(range(0, num_candidates, self.chunk_candidates)):
            end = min(start + self.chunk_candidates, num_candidates)
            chunk = population_theta[start:end]

            chunk_seed = int(seed_base + chunk_idx * 100_003)
            r, l, e, stats = self.evaluate_chunk(chunk, chunk_seed)
            returns[start:end] = r
            lengths[start:end] = l
            exploded[start:end] = e
            merge_eval_stats(total_stats, stats)

        return returns, lengths, exploded, total_stats


def save_checkpoint(
    path: str,
    iteration: int,
    theta: Any,
    args: argparse.Namespace,
    extra: Dict[str, float],
) -> None:
    if isinstance(theta, torch.Tensor):
        theta_np = theta.detach().cpu().numpy().astype(np.float32)
    else:
        theta_np = np.asarray(theta, dtype=np.float32)

    payload = {
        "iter": int(iteration),
        "theta": theta_np,
        "args": vars(args),
        "extra": extra,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def get_gpu_stats() -> Dict[str, float]:
    out = {
        "gpu_total_gb": float("nan"),
        "gpu_free_gb": float("nan"),
        "gpu_used_frac": float("nan"),
        "gpu_peak_alloc_gb": float("nan"),
        "gpu_peak_reserved_gb": float("nan"),
        "gpu_peak_reserved_frac": float("nan"),
    }

    try:
        dev = jax.devices()[0]
    except Exception:
        return out

    if dev.platform != "gpu":
        return out

    try:
        mem = dev.memory_stats()
    except Exception:
        mem = None

    if not mem:
        return out

    total_b = mem.get("bytes_limit", None)
    used_b = mem.get("bytes_in_use", None)
    peak_b = mem.get("peak_bytes_in_use", None)

    if total_b is None:
        return out

    total_b = float(total_b)
    out["gpu_total_gb"] = total_b / 1e9

    if used_b is not None:
        used_b = float(used_b)
        out["gpu_free_gb"] = (total_b - used_b) / 1e9
        out["gpu_used_frac"] = used_b / max(total_b, 1.0)

    if peak_b is not None:
        peak_b = float(peak_b)
        out["gpu_peak_alloc_gb"] = peak_b / 1e9
        out["gpu_peak_reserved_gb"] = peak_b / 1e9
        out["gpu_peak_reserved_frac"] = peak_b / max(total_b, 1.0)

    return out


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def enable_jax_compilation_cache(cache_dir: str) -> str:
    cache_dir_abs = os.path.abspath(cache_dir)
    os.makedirs(cache_dir_abs, exist_ok=True)

    enabled = False
    errors: List[str] = []

    try:
        from jax.experimental import compilation_cache as cc  # type: ignore

        set_cache_dir = getattr(cc, "set_cache_dir", None)
        if set_cache_dir is None:
            nested_cc = getattr(cc, "compilation_cache", None)
            if nested_cc is not None:
                set_cache_dir = getattr(nested_cc, "set_cache_dir", None)

        if set_cache_dir is not None:
            set_cache_dir(cache_dir_abs)
            enabled = True
    except Exception as exc:
        errors.append(str(exc))

    if not enabled:
        try:
            from jax.experimental.compilation_cache import compilation_cache as cc  # type: ignore

            cc.set_cache_dir(cache_dir_abs)
            enabled = True
        except Exception as exc:
            errors.append(str(exc))

    if enabled:
        print(f"[compile] persistent JAX cache: {cache_dir_abs}", flush=True)
    else:
        print(
            f"[compile] warning: failed to enable persistent JAX cache for {cache_dir_abs}: {errors}",
            flush=True,
        )
    return cache_dir_abs


def build_compile_key(
    args: argparse.Namespace,
    env_id: str,
    xml_path: str,
    obs_dim: int,
    action_dim: int,
    pairs: int,
    chunk: int,
) -> str:
    dev = jax.devices()[0]
    payload = {
        "env_id": env_id,
        "xml_sha256": file_sha256(xml_path),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "hidden": int(args.hidden),
        "rank": int(args.rank),
        "pairs": int(pairs),
        "candidate_chunk": int(chunk),
        "episodes_per_candidate": int(args.episodes_per_candidate),
        "rollout_steps": int(args.rollout_steps),
        "jax_version": str(jax.__version__),
        "mujoco_version": str(mujoco.__version__),
        "platform": str(dev.platform),
        "device_kind": str(getattr(dev, "device_kind", "")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perturbative RNN RL (Humanoid, MJX)")

    parser.add_argument("--env_candidates", type=str, default="Humanoid-v5,Humanoid-v4")
    parser.add_argument("--xml_path", type=str, default="", help="Path to humanoid.xml")

    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--k_in", type=int, default=50)
    parser.add_argument("--leak", type=float, default=0.2)
    parser.add_argument("--h_clip", type=float, default=80.0)
    parser.add_argument("--w0_std", type=float, default=1.0)
    parser.add_argument("--win_std", type=float, default=0.15)

    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--pairs", type=int, default=FIXED_PAIRS)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--theta_lr", type=float, default=0.01)

    parser.add_argument("--episodes_per_candidate", type=int, default=2)
    parser.add_argument("--candidate_chunk", type=int, default=FIXED_CANDIDATE_CHUNK)
    parser.add_argument("--rollout_steps", type=int, default=500)

    parser.add_argument("--init_action_std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--torch_num_threads",
        type=int,
        default=4,
        help="CPU threads used by Torch (reservoir init + optimizer math).",
    )
    parser.add_argument(
        "--compile_cache_dir",
        type=str,
        default=DEFAULT_JAX_COMPILE_CACHE_DIR,
        help="Persistent JAX compilation cache directory.",
    )

    parser.add_argument("--results_root", type=str, default="humanoid/results")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--checkpoint_every", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.pairs < 1:
        raise RuntimeError("--pairs must be >= 1")
    if args.candidate_chunk < 1:
        raise RuntimeError("--candidate_chunk must be >= 1")
    if args.episodes_per_candidate < 1:
        raise RuntimeError("--episodes_per_candidate must be >= 1")

    if args.pairs != FIXED_PAIRS:
        print(f"[config] overriding --pairs={args.pairs} with fixed value {FIXED_PAIRS}", flush=True)
    if args.candidate_chunk != FIXED_CANDIDATE_CHUNK:
        print(
            f"[config] overriding --candidate_chunk={args.candidate_chunk} with fixed value {FIXED_CANDIDATE_CHUNK}",
            flush=True,
        )
    args.pairs = FIXED_PAIRS
    args.candidate_chunk = FIXED_CANDIDATE_CHUNK

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.torch_num_threads) > 0:
        torch.set_num_threads(int(args.torch_num_threads))
    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(args.seed)

    devices = jax.devices()
    print("JAX devices:", [f"{d.platform}:{d.id}" for d in devices], flush=True)

    env_candidates = [x.strip() for x in args.env_candidates.split(",") if x.strip()]
    env_id, env_spec = resolve_env_spec(env_candidates)
    xml_path = resolve_humanoid_xml(args.xml_path)

    host_model = mujoco.MjModel.from_xml_path(xml_path)
    obs_dim = infer_obs_dim(env_spec, host_model)
    action_dim = int(host_model.nu)

    ctrlrange = np.asarray(host_model.actuator_ctrlrange, dtype=np.float32)
    if ctrlrange.shape != (action_dim, 2):
        raise RuntimeError(
            f"Unexpected actuator_ctrlrange shape {ctrlrange.shape}; expected ({action_dim}, 2)"
        )
    action_low = ctrlrange[:, 0]
    action_high = ctrlrange[:, 1]

    print(f"Using environment spec: {env_id}", flush=True)
    print(f"MuJoCo XML: {xml_path}", flush=True)
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}", flush=True)

    compile_cache_dir = enable_jax_compilation_cache(args.compile_cache_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = f"{timestamp}_{args.run_name}" if args.run_name else timestamp
    run_dir = os.path.join(args.results_root, run_label)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    metrics_dir = os.path.join(run_dir, "metrics")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    save_config(args, os.path.join(run_dir, "config.json"))

    logger = CSVLogger(
        filepath=os.path.join(run_dir, "train_log.csv"),
        fieldnames=[
            "iter",
            "base_return",
            "cand_mean",
            "cand_std",
            "cand_min",
            "cand_median",
            "cand_max",
            "bad_frac",
            "theta_norm",
            "sigma",
            "pairs_used",
            "chunk_used",
            "mean_ep_len",
            "env_steps",
            "actual_env_steps",
            "env_slot_util",
            "cand_per_sec",
            "gpu_used_frac",
            "gpu_peak_reserved_frac",
            "gpu_peak_reserved_gb",
            "fps",
            "elapsed_min",
        ],
    )

    cfg = ReservoirConfig(
        N=args.hidden,
        D=obs_dim,
        A=action_dim,
        rank=args.rank,
        leak=args.leak,
        h_clip=args.h_clip,
        k_in=args.k_in,
        w0_std=args.w0_std,
        win_std=args.win_std,
    )

    policy = ReservoirPolicy(cfg, action_low=action_low, action_high=action_high)

    theta = torch.zeros((policy.theta_dim,), dtype=torch.float32, device="cpu")
    theta.requires_grad_(True)

    # Initialize action head with small random values for non-degenerate startup.
    n, r, a = cfg.N, cfg.rank, cfg.A
    uv = 2 * n * r
    wa = a * n
    with torch.no_grad():
        theta[uv : uv + wa] = args.init_action_std * torch.randn((wa,), generator=torch_gen, dtype=torch.float32)
        theta[uv + wa : uv + wa + a] = 0.0
    theta_optimizer = torch.optim.Adam([theta], lr=args.theta_lr)

    def build_evaluator(chunk_candidates: int) -> CandidateEvaluator:
        return CandidateEvaluator(
            env_id=env_id,
            xml_path=xml_path,
            env_spec=env_spec,
            obs_dim=obs_dim,
            action_dim=action_dim,
            chunk_candidates=chunk_candidates,
            episodes_per_candidate=args.episodes_per_candidate,
            rollout_steps=args.rollout_steps,
            policy=policy,
        )

    current_pairs = FIXED_PAIRS
    current_chunk = FIXED_CANDIDATE_CHUNK

    evaluator = build_evaluator(current_chunk)
    print(
        f"MJX evaluator: chunk={current_chunk}, "
        f"episodes/candidate={args.episodes_per_candidate}, envs={evaluator.num_envs}, pairs={current_pairs}",
        flush=True,
    )
    compile_key = build_compile_key(
        args=args,
        env_id=env_id,
        xml_path=xml_path,
        obs_dim=obs_dim,
        action_dim=action_dim,
        pairs=current_pairs,
        chunk=current_chunk,
    )
    warmup_marker = os.path.join(compile_cache_dir, f"warmup_{compile_key}.json")
    if os.path.exists(warmup_marker):
        print(f"[compile] warmup already completed for fixed shape (marker: {warmup_marker})", flush=True)
    else:
        print("[compile] first-run warmup for fixed shape...", flush=True)
        t_compile0 = time.perf_counter()
        theta_jnp = torch_to_jnp(theta)
        evaluator.evaluate_population(theta_jnp[None, :], seed_base=int(args.seed + 4242))
        warmup_wall_s = time.perf_counter() - t_compile0
        with open(warmup_marker, "w") as f:
            json.dump(
                {
                    "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    "pairs": current_pairs,
                    "chunk": current_chunk,
                    "warmup_wall_s": warmup_wall_s,
                },
                f,
                indent=2,
            )
        print(f"[compile] warmup complete in {warmup_wall_s:.1f}s", flush=True)

    best_base_return = -float("inf")
    start = time.time()

    try:

        for iteration in range(1, args.iters + 1):
            iter_t0 = time.time()
            seed_it = 10_000_000 + args.seed * 1_000_000 + iteration * 1000

            K = int(current_pairs)
            iter_pairs = int(K)
            iter_chunk = int(current_chunk)
            theta_dim = int(theta.numel())

            theta_view = theta.detach().unsqueeze(0)
            epsilon = torch.randn((K, theta_dim), generator=torch_gen, dtype=torch.float32)
            pop_theta_t = torch.cat(
                [
                    theta_view + args.sigma * epsilon,
                    theta_view - args.sigma * epsilon,
                ],
                dim=0,
            )
            pop_theta = torch_to_jnp(pop_theta_t)

            cand_returns_np, cand_lengths_np, cand_exploded_np, eval_stats = evaluator.evaluate_population(
                pop_theta,
                seed_it,
            )

            cand_returns = jnp.asarray(cand_returns_np, dtype=jnp.float32)
            fitness = rank_transform(cand_returns, maximize=True)
            fitness_t = torch.from_numpy(np.asarray(fitness, dtype=np.float32))
            w_pos = fitness_t[:K, None]
            w_neg = fitness_t[K:, None]
            grad_est = ((w_pos - w_neg) * epsilon).mean(dim=0)
            theta_optimizer.zero_grad(set_to_none=True)
            theta.grad = -grad_est
            theta_optimizer.step()

            base_return = float("nan")
            if iteration % args.log_every == 0 or iteration == 1:
                theta_jnp = torch_to_jnp(theta)
                base_ret_np, _, _, base_stats = evaluator.evaluate_population(
                    theta_jnp[None, :],
                    seed_it + 777,
                )
                base_return = float(base_ret_np[0])
                merge_eval_stats(eval_stats, base_stats)

            elapsed = time.time() - start
            iter_elapsed = time.time() - iter_t0

            effective_env_steps = int(eval_stats["effective_env_steps"])
            actual_env_steps = int(eval_stats["actual_env_steps"])
            fps = float(effective_env_steps / max(iter_elapsed, 1e-6))
            actual_fps = float(actual_env_steps / max(iter_elapsed, 1e-6))
            env_slot_util = float(effective_env_steps / max(actual_env_steps, 1))

            cand_mean = float(np.mean(cand_returns_np))
            cand_std = float(np.std(cand_returns_np))
            cand_min = float(np.min(cand_returns_np))
            cand_med = float(np.median(cand_returns_np))
            cand_max = float(np.max(cand_returns_np))
            bad_frac = float(np.mean(cand_exploded_np.astype(np.float32)))
            mean_ep_len = float(np.mean(cand_lengths_np))
            theta_norm = float(theta.detach().norm().item())

            cand_per_sec = float((2 * K) / max(float(eval_stats["wall_s"]), 1e-9))

            gpu = get_gpu_stats()
            gpu_peak_reserved_frac = float(gpu["gpu_peak_reserved_frac"])
            gpu_peak_reserved_gb = float(gpu["gpu_peak_reserved_gb"])
            gpu_used_frac = float(gpu["gpu_used_frac"])

            if iteration % args.log_every == 0 or iteration == 1:
                np.savez_compressed(
                    os.path.join(metrics_dir, f"candidate_metrics_iter_{iteration:05d}.npz"),
                    returns=cand_returns_np,
                    lengths=cand_lengths_np,
                    exploded=cand_exploded_np.astype(np.int8),
                    eval_wall_s=float(eval_stats["wall_s"]),
                    effective_env_steps=effective_env_steps,
                    actual_env_steps=actual_env_steps,
                    env_slot_util=env_slot_util,
                )

            print(
                f"iter {iteration:5d} | base {base_return:9.2f} | "
                f"cand mean/min/med/max {cand_mean:8.2f}/{cand_min:8.2f}/{cand_med:8.2f}/{cand_max:8.2f} | "
                f"bad_frac {bad_frac:.3f} | len {mean_ep_len:6.1f} | "
                f"|theta| {theta_norm:8.2f} | pairs/chunk {iter_pairs}/{iter_chunk} | "
                f"eff_fps {fps:9.1f} | elapsed {elapsed/60.0:7.1f}m",
                flush=True,
            )

            print(
                f"  profile | cand/s {cand_per_sec:8.1f} | "
                f"env_util {env_slot_util*100:5.1f}% ({effective_env_steps}/{actual_env_steps} active slots) | "
                f"actual_fps {actual_fps:9.1f}",
                flush=True,
            )

            if np.isfinite(gpu_used_frac):
                print(
                    f"  gpu     | used {gpu_used_frac*100:5.1f}% | peak_reserved {gpu_peak_reserved_gb:6.2f} GB "
                    f"({gpu_peak_reserved_frac*100:5.1f}% of total)",
                    flush=True,
                )

            logger.log(
                {
                    "iter": iteration,
                    "base_return": base_return,
                    "cand_mean": cand_mean,
                    "cand_std": cand_std,
                    "cand_min": cand_min,
                    "cand_median": cand_med,
                    "cand_max": cand_max,
                    "bad_frac": bad_frac,
                    "theta_norm": theta_norm,
                    "sigma": args.sigma,
                    "pairs_used": iter_pairs,
                    "chunk_used": iter_chunk,
                    "mean_ep_len": mean_ep_len,
                    "env_steps": effective_env_steps,
                    "actual_env_steps": actual_env_steps,
                    "env_slot_util": env_slot_util,
                    "cand_per_sec": cand_per_sec,
                    "gpu_used_frac": gpu_used_frac,
                    "gpu_peak_reserved_frac": gpu_peak_reserved_frac,
                    "gpu_peak_reserved_gb": gpu_peak_reserved_gb,
                    "fps": fps,
                    "elapsed_min": elapsed / 60.0,
                }
            )

            save_checkpoint(
                os.path.join(ckpt_dir, "checkpoint_latest.pkl"),
                iteration,
                theta,
                args,
                {
                    "base_return": base_return,
                    "cand_mean": cand_mean,
                    "cand_max": cand_max,
                    "bad_frac": bad_frac,
                    "pairs_used": float(iter_pairs),
                    "chunk_used": float(iter_chunk),
                },
            )

            if iteration % args.checkpoint_every == 0:
                save_checkpoint(
                    os.path.join(ckpt_dir, f"checkpoint_iter_{iteration:05d}.pkl"),
                    iteration,
                    theta,
                    args,
                    {
                        "base_return": base_return,
                        "cand_mean": cand_mean,
                        "cand_max": cand_max,
                        "bad_frac": bad_frac,
                        "pairs_used": float(iter_pairs),
                        "chunk_used": float(iter_chunk),
                    },
                )

            if np.isfinite(base_return) and base_return > best_base_return:
                best_base_return = base_return
                save_checkpoint(
                    os.path.join(ckpt_dir, "checkpoint_best.pkl"),
                    iteration,
                    theta,
                    args,
                    {
                        "base_return": base_return,
                        "cand_mean": cand_mean,
                        "cand_max": cand_max,
                        "bad_frac": bad_frac,
                        "pairs_used": float(iter_pairs),
                        "chunk_used": float(iter_chunk),
                    },
                )

        summary = {
            "env_id": env_id,
            "xml_path": xml_path,
            "iters": args.iters,
            "pairs_initial": args.pairs,
            "pairs_final": current_pairs,
            "episodes_per_candidate": args.episodes_per_candidate,
            "candidate_chunk_initial": args.candidate_chunk,
            "candidate_chunk_final": current_chunk,
            "best_base_return": best_base_return,
            "elapsed_min": (time.time() - start) / 60.0,
        }
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    finally:
        evaluator.close()
        logger.close()


if __name__ == "__main__":
    main()
