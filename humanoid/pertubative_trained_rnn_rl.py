"""
Perturbative ES-trained reservoir policy for MuJoCo Humanoid using MJX.

This is a GPU-oriented migration of the previous Gym VectorEnv + PyTorch version.
Rollouts are evaluated with batched MJX physics and JAX-jitted policy inference.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np

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
        "policy_s": 0.0,
        "env_s": 0.0,
        "tensor_s": 0.0,
        "chunks": 0,
        "loop_steps": 0,
        "effective_env_steps": 0,
        "actual_env_steps": 0,
    }


def merge_eval_stats(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for key in ("wall_s", "policy_s", "env_s", "tensor_s"):
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

        rng = np.random.default_rng(1234)
        self.Win = jnp.asarray(
            rng.standard_normal(size=(cfg.N, cfg.D), dtype=np.float32) * cfg.win_std,
            dtype=jnp.float32,
        )
        self.b = jnp.zeros((cfg.N,), dtype=jnp.float32)
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

        rng = np.random.default_rng(1234)
        rows = np.repeat(np.arange(N, dtype=np.int32), k_in)
        cols = rng.integers(0, N, size=N * k_in, endpoint=False, dtype=np.int32)
        scale = w_std / math.sqrt(float(k_in))
        vals = rng.standard_normal(size=N * k_in).astype(np.float32) * scale

        dense = np.zeros((N, N), dtype=np.float32)
        np.add.at(dense, (rows, cols), vals)

        dt = time.time() - t0
        print(f"  [Reservoir] W0 generated in {dt:.2f}s", flush=True)
        return jnp.asarray(dense, dtype=jnp.float32)

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


class AdamState(NamedTuple):
    m: jnp.ndarray
    v: jnp.ndarray
    t: jnp.ndarray


def adam_init(theta: jnp.ndarray) -> AdamState:
    return AdamState(
        m=jnp.zeros_like(theta),
        v=jnp.zeros_like(theta),
        t=jnp.array(0, dtype=jnp.int32),
    )


def adam_step(
    theta: jnp.ndarray,
    grad: jnp.ndarray,
    state: AdamState,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[jnp.ndarray, AdamState]:
    t = state.t + 1
    m = beta1 * state.m + (1.0 - beta1) * grad
    v = beta2 * state.v + (1.0 - beta2) * (grad * grad)

    t_f = t.astype(jnp.float32)
    m_hat = m / (1.0 - jnp.power(beta1, t_f))
    v_hat = v / (1.0 - jnp.power(beta2, t_f))

    theta_next = theta - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return theta_next, AdamState(m=m, v=v, t=t)


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
        if int(self.model.ntendon) > 0:
            raise RuntimeError(
                "MJX does not support tendons for this stack. "
                f"Loaded XML has ntendon={int(self.model.ntendon)}: {xml_path}. "
                "Use tendonless humanoid XML (e.g. humanoid/humanoid.xml in this repo)."
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
        cfrc_rows_target = (int(self.model.nbody) - 1) if obs_variant == "v5" else int(self.model.nbody)
        cfrc_size_target = cfrc_rows_target * 6
        actuator_size_target = int(self.model.nu) if obs_variant == "v5" else int(self.model.nv)
        qfrc_full_size = max(int(self.model.nv), actuator_force_start + int(self.model.nu))

        C_full = self.chunk_candidates
        E = self.episodes_per_candidate
        B = self.num_envs
        D = self.obs_dim
        A = self.action_dim
        N = self.policy.cfg.N
        nq = int(self.model.nq)
        nv = int(self.model.nv)

        sample_data = mjx.make_data(self.mjx_model)
        has_qfrc_actuator = hasattr(sample_data, "qfrc_actuator")
        has_cfrc_ext = hasattr(sample_data, "cfrc_ext")
        zeros_qfrc_full = jnp.zeros((qfrc_full_size,), dtype=jnp.float32)
        zeros_actuator = jnp.zeros((actuator_size_target,), dtype=jnp.float32)
        zeros_cfrc = jnp.zeros((cfrc_size_target,), dtype=jnp.float32)

        def mass_center_x(data: mjx.Data) -> jnp.ndarray:
            return jnp.sum(body_mass * data.xipos[:, 0]) / body_mass_sum

        def get_obs(data: mjx.Data) -> jnp.ndarray:
            position = data.qpos[2:]
            velocity = data.qvel

            if has_qfrc_actuator:
                qfrc_full = data.qfrc_actuator
            else:
                qfrc_full = zeros_qfrc_full

            if obs_variant == "v5":
                com_inertia = data.cinert[1:].reshape((-1,))
                com_velocity = data.cvel[1:].reshape((-1,))
                if has_cfrc_ext:
                    external_contact_forces = data.cfrc_ext[1:].reshape((-1,))
                else:
                    external_contact_forces = zeros_cfrc
                actuator_forces = qfrc_full[actuator_force_start : actuator_force_start + actuator_size_target]
            else:
                com_inertia = data.cinert.reshape((-1,))
                com_velocity = data.cvel.reshape((-1,))
                if has_cfrc_ext:
                    external_contact_forces = data.cfrc_ext.reshape((-1,))
                else:
                    external_contact_forces = zeros_cfrc
                actuator_forces = qfrc_full[:actuator_size_target]

            if actuator_forces.shape[0] != actuator_size_target:
                actuator_forces = zeros_actuator
            if external_contact_forces.shape[0] != cfrc_size_target:
                external_contact_forces = zeros_cfrc

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

            if include_contact_cost and has_cfrc_ext:
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

            def scan_step(carry, step_idx):
                h, data, obs, returns, lengths, alive, exploded_by_candidate = carry
                active_count = jnp.sum(alive.astype(jnp.int32))

                def active_branch(_):
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
                        h_next,
                        data_next,
                        obs_next,
                        returns_next,
                        lengths_next,
                        alive_next,
                        exploded_by_candidate_next,
                    )

                carry_next = jax.lax.cond(active_count > 0, active_branch, lambda _: carry, operand=None)
                loop_inc = jnp.where(active_count > 0, 1, 0).astype(jnp.int32)
                return carry_next, (active_count, loop_inc)

            carry0 = (h, data, obs, returns, lengths, alive, exploded_by_candidate)
            carry_f, (active_counts, loop_incs) = jax.lax.scan(
                scan_step,
                carry0,
                jnp.arange(self.rollout_steps, dtype=jnp.int32),
            )

            _, _, _, returns_f, lengths_f, alive_f, exploded_f = carry_f
            lengths_f = jnp.where(alive_f, self.rollout_steps, lengths_f)

            returns_c = jnp.mean(returns_f, axis=1)
            lengths_c = jnp.mean(lengths_f.astype(jnp.float32), axis=1)
            effective_env_steps = jnp.sum(active_counts)
            loop_steps = jnp.sum(loop_incs)
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
    theta: jnp.ndarray,
    args: argparse.Namespace,
    extra: Dict[str, float],
) -> None:
    payload = {
        "iter": int(iteration),
        "theta": np.asarray(theta, dtype=np.float32),
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


def _round_to_step(value: int, step: int, min_value: int, max_value: int) -> int:
    step = max(1, int(step))
    value = int(round(float(value) / float(step)) * step)
    value = max(int(min_value), value)
    value = min(int(max_value), value)
    return int(value)


def run_dry_compile_check(
    evaluator: CandidateEvaluator,
    theta: jnp.ndarray,
    seed_base: int,
    repeats: int,
) -> Dict[str, Any]:
    repeats = max(1, int(repeats))
    C = int(evaluator.chunk_candidates)
    theta_batch = jnp.repeat(theta[None, :], repeats=C, axis=0)

    entries: List[Dict[str, float]] = []
    for i in range(repeats):
        t0 = time.perf_counter()
        _, _, _, stats = evaluator.evaluate_chunk(theta_batch, seed_base=seed_base + i * 17)
        wall = time.perf_counter() - t0
        eff = int(stats["effective_env_steps"])
        fps = float(eff / max(wall, 1e-9))
        entries.append(
            {
                "run_idx": float(i + 1),
                "wall_s": float(wall),
                "effective_env_steps": float(eff),
                "effective_fps": float(fps),
            }
        )

    gpu = get_gpu_stats()
    return {"runs": entries, "gpu": gpu}


def propose_autotuned_batch_sizes(
    pairs: int,
    chunk: int,
    iter_elapsed: float,
    gpu_used_frac: float,
    gpu_peak_reserved_frac: float,
    args: argparse.Namespace,
) -> Tuple[int, int, List[str]]:
    notes: List[str] = []

    target_iter = max(float(args.headroom_target_iter_sec), 1e-6)
    min_scale = max(0.1, float(args.autotune_min_scale))
    max_scale = max(min_scale, float(args.autotune_max_scale))

    time_scale = target_iter / max(iter_elapsed, 1e-6)
    time_scale = float(np.clip(time_scale, min_scale, max_scale))
    notes.append(f"time_scale={time_scale:.2f}")

    mem_scale = 1.0
    if np.isfinite(gpu_peak_reserved_frac) and gpu_peak_reserved_frac > 1e-6:
        mem_scale = float(args.autotune_peak_reserved_target) / float(gpu_peak_reserved_frac)
        mem_scale = float(np.clip(mem_scale, min_scale, max_scale))
        notes.append(f"mem_scale={mem_scale:.2f}")
    else:
        notes.append("mem_scale=n/a")

    util_scale = 1.0
    if np.isfinite(gpu_used_frac):
        low = float(args.autotune_gpu_used_target) * 0.70
        high = min(0.995, float(args.autotune_gpu_used_target) * 1.10)
        if gpu_used_frac < low:
            util_scale = 1.15
        elif gpu_used_frac > high:
            util_scale = 0.85
        notes.append(f"util_scale={util_scale:.2f}")
    else:
        notes.append("util_scale=n/a")

    pair_scale = time_scale * util_scale * math.sqrt(max(1e-6, mem_scale))
    pair_scale = float(np.clip(pair_scale, min_scale, max_scale))

    chunk_scale = mem_scale
    if iter_elapsed > target_iter * 1.15:
        chunk_scale = min(chunk_scale, 1.0)
    elif iter_elapsed < target_iter * 0.75 and np.isfinite(gpu_used_frac):
        if gpu_used_frac < float(args.autotune_gpu_used_target):
            chunk_scale = max(chunk_scale, 1.10)
    chunk_scale = float(np.clip(chunk_scale, min_scale, max_scale))

    next_pairs = _round_to_step(
        int(round(float(pairs) * pair_scale)),
        step=int(args.autotune_pairs_step),
        min_value=1,
        max_value=max(1, int(args.autotune_pairs_cap)),
    )
    next_chunk = _round_to_step(
        int(round(float(chunk) * chunk_scale)),
        step=int(args.autotune_chunk_step),
        min_value=1,
        max_value=max(1, int(args.autotune_chunk_cap)),
    )

    # Keep chunk bounded by total candidate count to avoid needless over-padding.
    next_chunk = min(next_chunk, max(1, 2 * next_pairs))

    if abs(next_pairs - pairs) < max(int(args.autotune_pairs_step), int(0.05 * max(1, pairs))):
        next_pairs = int(pairs)
    if abs(next_chunk - chunk) < max(int(args.autotune_chunk_step), int(0.05 * max(1, chunk))):
        next_chunk = int(chunk)

    notes.append(f"pair_scale={pair_scale:.2f}")
    notes.append(f"chunk_scale={chunk_scale:.2f}")
    return int(next_pairs), int(next_chunk), notes


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
    parser.add_argument("--pairs", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--theta_lr", type=float, default=0.01)

    parser.add_argument("--episodes_per_candidate", type=int, default=2)
    parser.add_argument("--candidate_chunk", type=int, default=64)
    parser.add_argument("--rollout_steps", type=int, default=500)

    parser.add_argument("--init_action_std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--torch_num_threads",
        type=int,
        default=4,
        help="Deprecated compatibility flag; ignored in MJX mode.",
    )
    parser.add_argument(
        "--dry_run_compile",
        action="store_true",
        help="Compile and benchmark one evaluator chunk, then exit.",
    )
    parser.add_argument(
        "--dry_run_repeats",
        type=int,
        default=2,
        help="Number of dry-run evaluator calls. First call includes JIT compile.",
    )

    parser.add_argument(
        "--autotune_warmup_iters",
        type=int,
        default=0,
        help="If >0, autotune pairs/chunk during the first N iterations.",
    )
    parser.add_argument("--autotune_pairs_cap", type=int, default=32768)
    parser.add_argument("--autotune_chunk_cap", type=int, default=1024)
    parser.add_argument("--autotune_pairs_step", type=int, default=64)
    parser.add_argument("--autotune_chunk_step", type=int, default=8)
    parser.add_argument("--autotune_gpu_used_target", type=float, default=0.90)
    parser.add_argument("--autotune_peak_reserved_target", type=float, default=0.88)
    parser.add_argument("--autotune_min_scale", type=float, default=0.70)
    parser.add_argument("--autotune_max_scale", type=float, default=1.80)

    parser.add_argument("--results_root", type=str, default="humanoid/results")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument(
        "--headroom_target_iter_sec",
        type=float,
        default=600.0,
        help="Target iteration duration used to estimate a larger pairs count.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.pairs < 1:
        raise RuntimeError("--pairs must be >= 1")
    if args.candidate_chunk < 1:
        raise RuntimeError("--candidate_chunk must be >= 1")
    if args.episodes_per_candidate < 1:
        raise RuntimeError("--episodes_per_candidate must be >= 1")

    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

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
            "autotuned",
            "mean_ep_len",
            "env_steps",
            "actual_env_steps",
            "env_slot_util",
            "eval_policy_share",
            "eval_env_share",
            "cand_per_sec",
            "gpu_used_frac",
            "gpu_peak_reserved_frac",
            "gpu_peak_reserved_gb",
            "suggest_pairs",
            "suggest_chunk",
            "suggest_envs",
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

    theta = jnp.zeros((policy.theta_dim,), dtype=jnp.float32)

    # Initialize action head with small random values for non-degenerate startup.
    n, r, a = cfg.N, cfg.rank, cfg.A
    uv = 2 * n * r
    wa = a * n
    rng, k_action = jax.random.split(rng)
    theta = theta.at[uv : uv + wa].set(
        args.init_action_std * jax.random.normal(k_action, shape=(wa,), dtype=jnp.float32)
    )
    theta = theta.at[uv + wa : uv + wa + a].set(jnp.zeros((a,), dtype=jnp.float32))

    adam_state = adam_init(theta)

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

    current_pairs = int(args.pairs)
    current_chunk = int(args.candidate_chunk)
    evaluator = build_evaluator(current_chunk)

    print(
        f"MJX evaluator: chunk={current_chunk}, "
        f"episodes/candidate={args.episodes_per_candidate}, envs={evaluator.num_envs}",
        flush=True,
    )
    if args.autotune_warmup_iters > 0:
        print(
            "Autotune enabled: "
            f"warmup_iters={args.autotune_warmup_iters}, "
            f"pairs_cap={args.autotune_pairs_cap}, chunk_cap={args.autotune_chunk_cap}",
            flush=True,
        )

    best_base_return = -float("inf")
    start = time.time()
    chunk_autotune_updates = 0

    try:
        if args.dry_run_compile:
            print(
                f"Dry-run compile mode: chunk={current_chunk}, repeats={max(1, int(args.dry_run_repeats))}",
                flush=True,
            )
            dry = run_dry_compile_check(
                evaluator=evaluator,
                theta=theta,
                seed_base=args.seed + 123_456,
                repeats=args.dry_run_repeats,
            )
            for entry in dry["runs"]:
                print(
                    f"  dry_run {int(entry['run_idx'])}: wall={entry['wall_s']:.2f}s | "
                    f"eff_steps={int(entry['effective_env_steps'])} | eff_fps={entry['effective_fps']:.1f}",
                    flush=True,
                )
            gpu = dry["gpu"]
            if np.isfinite(float(gpu["gpu_used_frac"])):
                print(
                    f"  dry_run gpu: used={float(gpu['gpu_used_frac'])*100:.1f}% | "
                    f"peak_reserved={float(gpu['gpu_peak_reserved_gb']):.2f}GB",
                    flush=True,
                )
            with open(os.path.join(run_dir, "dry_run_compile.json"), "w") as f:
                json.dump(dry, f, indent=2)
            return

        for iteration in range(1, args.iters + 1):
            iter_t0 = time.time()
            seed_it = 10_000_000 + args.seed * 1_000_000 + iteration * 1000

            K = int(current_pairs)
            iter_pairs = int(K)
            iter_chunk = int(current_chunk)
            theta_dim = int(theta.shape[0])

            rng, eps_key = jax.random.split(rng)
            epsilon = jax.random.normal((eps_key), shape=(K, theta_dim), dtype=jnp.float32)
            theta_view = theta[None, :]
            pop_theta = jnp.concatenate(
                [
                    theta_view + args.sigma * epsilon,
                    theta_view - args.sigma * epsilon,
                ],
                axis=0,
            )

            cand_returns_np, cand_lengths_np, cand_exploded_np, eval_stats = evaluator.evaluate_population(
                pop_theta,
                seed_it,
            )

            cand_returns = jnp.asarray(cand_returns_np, dtype=jnp.float32)
            fitness = rank_transform(cand_returns, maximize=True)
            w_pos = fitness[:K]
            w_neg = fitness[K:]

            grad_est = ((w_pos[:, None] - w_neg[:, None]) * epsilon).mean(axis=0)
            theta, adam_state = adam_step(theta, grad=-grad_est, state=adam_state, lr=args.theta_lr)

            base_return = float("nan")
            if iteration % args.log_every == 0 or iteration == 1:
                base_ret_np, _, _, base_stats = evaluator.evaluate_population(
                    theta[None, :],
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
            theta_norm = float(np.linalg.norm(np.asarray(theta)))

            timed_total = float(eval_stats["policy_s"] + eval_stats["env_s"] + eval_stats["tensor_s"])
            if timed_total > 0.0:
                eval_policy_share = float(eval_stats["policy_s"] / timed_total)
                eval_env_share = float(eval_stats["env_s"] / timed_total)
            else:
                eval_policy_share = float("nan")
                eval_env_share = float("nan")

            cand_per_sec = float((2 * K) / max(float(eval_stats["wall_s"]), 1e-9))

            gpu = get_gpu_stats()
            gpu_peak_reserved_frac = float(gpu["gpu_peak_reserved_frac"])
            gpu_peak_reserved_gb = float(gpu["gpu_peak_reserved_gb"])
            gpu_used_frac = float(gpu["gpu_used_frac"])

            if np.isfinite(gpu_peak_reserved_frac) and gpu_peak_reserved_frac > 0:
                mem_scale = max(1.0, min(4.0, 0.85 / gpu_peak_reserved_frac))
            else:
                mem_scale = 1.0

            suggest_chunk = int(max(iter_chunk, round(iter_chunk * mem_scale)))
            suggest_pairs = int(
                max(iter_pairs, round(iter_pairs * (args.headroom_target_iter_sec / max(iter_elapsed, 1e-9))))
            )
            suggest_envs = int(suggest_chunk * args.episodes_per_candidate)
            autotuned = 0
            autotune_message = ""

            if iteration % args.log_every == 0 or iteration == 1:
                np.savez_compressed(
                    os.path.join(metrics_dir, f"candidate_metrics_iter_{iteration:05d}.npz"),
                    returns=cand_returns_np,
                    lengths=cand_lengths_np,
                    exploded=cand_exploded_np.astype(np.int8),
                    eval_wall_s=float(eval_stats["wall_s"]),
                    eval_policy_s=float(eval_stats["policy_s"]),
                    eval_env_s=float(eval_stats["env_s"]),
                    eval_tensor_s=float(eval_stats["tensor_s"]),
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

            if np.isfinite(eval_policy_share) and np.isfinite(eval_env_share):
                split_tensor = max(0.0, 1.0 - eval_policy_share - eval_env_share)
                print(
                    f"  profile | cand/s {cand_per_sec:8.1f} | eval split policy/env/tensor "
                    f"{eval_policy_share*100:5.1f}/{eval_env_share*100:5.1f}/{split_tensor*100:5.1f}% | "
                    f"env_util {env_slot_util*100:5.1f}% ({effective_env_steps}/{actual_env_steps} active slots) | "
                    f"actual_fps {actual_fps:9.1f}",
                    flush=True,
                )
            else:
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

            print(
                f"  headroom| suggest next-run pairs ~{suggest_pairs} "
                f"(target {args.headroom_target_iter_sec:.0f}s/iter), "
                f"candidate_chunk ~{suggest_chunk}, envs ~{suggest_envs}",
                flush=True,
            )

            if args.autotune_warmup_iters > 0 and iteration <= args.autotune_warmup_iters:
                next_pairs, next_chunk, notes = propose_autotuned_batch_sizes(
                    pairs=current_pairs,
                    chunk=current_chunk,
                    iter_elapsed=iter_elapsed,
                    gpu_used_frac=gpu_used_frac,
                    gpu_peak_reserved_frac=gpu_peak_reserved_frac,
                    args=args,
                )
                if next_pairs != current_pairs or next_chunk != current_chunk:
                    prev_pairs = current_pairs
                    prev_chunk = current_chunk
                    blocked_chunk_change = False
                    current_pairs = int(next_pairs)
                    if next_chunk != current_chunk:
                        if chunk_autotune_updates < 1:
                            evaluator.close()
                            current_chunk = int(next_chunk)
                            evaluator = build_evaluator(current_chunk)
                            chunk_autotune_updates += 1
                        else:
                            blocked_chunk_change = True
                            next_chunk = current_chunk
                    autotuned = int((current_pairs != prev_pairs) or (current_chunk != prev_chunk))
                    autotune_message = (
                        f"autotune update pairs {prev_pairs}->{current_pairs}, "
                        f"chunk {prev_chunk}->{current_chunk}"
                    )
                    if blocked_chunk_change:
                        autotune_message += " (chunk retune locked after first rebuild)"
                else:
                    autotune_message = "autotune kept current pairs/chunk"
                print(f"  autotune| {autotune_message} | {'; '.join(notes)}", flush=True)

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
                    "autotuned": autotuned,
                    "mean_ep_len": mean_ep_len,
                    "env_steps": effective_env_steps,
                    "actual_env_steps": actual_env_steps,
                    "env_slot_util": env_slot_util,
                    "eval_policy_share": eval_policy_share,
                    "eval_env_share": eval_env_share,
                    "cand_per_sec": cand_per_sec,
                    "gpu_used_frac": gpu_used_frac,
                    "gpu_peak_reserved_frac": gpu_peak_reserved_frac,
                    "gpu_peak_reserved_gb": gpu_peak_reserved_gb,
                    "suggest_pairs": suggest_pairs,
                    "suggest_chunk": suggest_chunk,
                    "suggest_envs": suggest_envs,
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
