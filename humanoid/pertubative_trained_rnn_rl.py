"""
Pertubative Trained RNN for Mujoco Humanoid.

This script adapts the perturbative ES training setup from Moving-MNIST to
continuous-control RL on Humanoid. The recurrent core is a fixed reservoir plus
trainable low-rank adapter (U, V). We optimize policy parameters with
antithetic Evolution Strategies (no teacher forcing and no warmup).

The rollout evaluator runs many CPU MuJoCo environments in parallel while
batching policy inference for candidate chunks on GPU.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import functools
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

try:
    import gymnasium as gym
    from gymnasium.spaces import Box
    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
except Exception as exc:
    raise RuntimeError(
        "gymnasium + mujoco are required. Install with: pip install 'gymnasium[mujoco]'"
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


def rank_transform(values: torch.Tensor, maximize: bool = True) -> torch.Tensor:
    """
    Rank-based fitness shaping.
    Higher rank -> larger positive weight (when maximize=True).
    """
    n = len(values)
    if n <= 1:
        return torch.zeros_like(values)

    ranks = torch.empty_like(values, dtype=torch.float32)
    order = torch.argsort(values)
    ranks[order] = torch.arange(n, dtype=torch.float32, device=values.device)

    if maximize:
        w = ranks / float(n - 1) - 0.5
    else:
        w = -(ranks / float(n - 1) - 0.5)

    w = w - w.mean()
    return w


def build_env(env_id: str):
    return gym.make(env_id)


def resolve_env_id(candidates: List[str]) -> Tuple[str, int, int, np.ndarray, np.ndarray]:
    last_exc = None
    for env_id in candidates:
        env_id = env_id.strip()
        if not env_id:
            continue
        try:
            env = gym.make(env_id)
            obs_space = env.observation_space
            act_space = env.action_space

            if not isinstance(obs_space, Box):
                raise RuntimeError(f"{env_id} observation space must be Box, got {type(obs_space)}")
            if not isinstance(act_space, Box):
                raise RuntimeError(f"{env_id} action space must be Box, got {type(act_space)}")

            obs_dim = int(np.prod(obs_space.shape))
            action_dim = int(np.prod(act_space.shape))
            action_low = np.asarray(act_space.low, dtype=np.float32).reshape(-1)
            action_high = np.asarray(act_space.high, dtype=np.float32).reshape(-1)
            env.close()
            return env_id, obs_dim, action_dim, action_low, action_high
        except Exception as exc:  # pragma: no cover
            last_exc = exc

    raise RuntimeError(f"Unable to create any env from {candidates}. Last error: {last_exc}")


class ReservoirPolicy(nn.Module):
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
        device: torch.device,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.Win = nn.Parameter(
            torch.randn(cfg.N, cfg.D, device=device, dtype=torch.float32) * cfg.win_std,
            requires_grad=False,
        )
        self.b = nn.Parameter(torch.zeros(cfg.N, device=device, dtype=torch.float32), requires_grad=False)
        self.W0 = self._make_sparse_w0(cfg.N, cfg.k_in, device=device, w_std=cfg.w0_std)

        low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        high = torch.as_tensor(action_high, dtype=torch.float32, device=device)
        self.register_buffer("action_scale", 0.5 * (high - low))
        self.register_buffer("action_bias", 0.5 * (high + low))

    @property
    def theta_dim(self) -> int:
        n, r, a = self.cfg.N, self.cfg.rank, self.cfg.A
        return 2 * n * r + a * n + a

    @staticmethod
    def _make_sparse_w0(N: int, k_in: int, device: torch.device, w_std: float) -> torch.Tensor:
        print(f"  [Reservoir] Generating sparse W0 ({N}x{N})...", flush=True)
        t0 = time.time()

        g = torch.Generator(device="cpu")
        g.manual_seed(1234)

        rows: List[torch.Tensor] = []
        cols: List[torch.Tensor] = []
        vals: List[torch.Tensor] = []
        scale = w_std / math.sqrt(float(k_in))

        for i in range(N):
            c = torch.randint(0, N, (k_in,), generator=g)
            v = torch.randn((k_in,), generator=g) * scale
            rows.append(torch.full((k_in,), i, dtype=torch.long))
            cols.append(c.to(torch.long))
            vals.append(v)

        row = torch.cat(rows)
        col = torch.cat(cols)
        val = torch.cat(vals).to(torch.float32)

        idx = torch.stack([row, col], dim=0)
        W0 = torch.sparse_coo_tensor(idx, val, (N, N), device=device)
        W0 = W0.coalesce().to_dense()

        dt = time.time() - t0
        print(f"  [Reservoir] W0 generated in {dt:.2f}s", flush=True)
        return W0

    def split_theta(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        theta shape: [C, theta_dim]
        returns:
          U: [C, N, R]
          V: [C, N, R]
          Wa: [C, A, N]
          ba: [C, A]
        """
        C = theta.shape[0]
        n, r, a = self.cfg.N, self.cfg.rank, self.cfg.A

        uv = 2 * n * r
        wa = a * n

        U = theta[:, : n * r].view(C, n, r)
        V = theta[:, n * r : uv].view(C, n, r)
        Wa = theta[:, uv : uv + wa].view(C, a, n)
        ba = theta[:, uv + wa : uv + wa + a].view(C, a)
        return U, V, Wa, ba

    @torch.no_grad()
    def policy_step(
        self,
        h: torch.Tensor,
        obs: torch.Tensor,
        cand_U: torch.Tensor,
        cand_V: torch.Tensor,
        cand_Wa: torch.Tensor,
        cand_ba: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        h_flat = h.view(C * E, N)
        obs_flat = obs.view(C * E, D)

        rec0 = h_flat @ self.W0.t()
        rec0 = rec0.view(C, E, N)

        low_tmp = torch.bmm(h, cand_V)
        low = torch.bmm(low_tmp, cand_U.transpose(1, 2))

        inp = obs_flat @ self.Win.t()
        inp = inp.view(C, E, N)

        pre = rec0 + low + inp + self.b
        nh = torch.tanh(pre)
        next_h = (1.0 - self.cfg.leak) * h + self.cfg.leak * nh

        exploded = (~torch.isfinite(next_h)).any(dim=2) | (next_h.abs().amax(dim=2) > self.cfg.h_clip)
        if exploded.any():
            safe = torch.nan_to_num(next_h, nan=0.0, posinf=0.0, neginf=0.0)
            next_h = safe
            mask = (~exploded).unsqueeze(-1).to(next_h.dtype)
            next_h = next_h * mask

        act_pre = torch.einsum("cen,can->cea", next_h, cand_Wa) + cand_ba.unsqueeze(1)
        action = torch.tanh(act_pre)
        action = action * self.action_scale.view(1, 1, -1) + self.action_bias.view(1, 1, -1)

        return next_h, action, exploded


class CandidateEvaluator:
    def __init__(
        self,
        env_id: str,
        obs_dim: int,
        action_dim: int,
        chunk_candidates: int,
        episodes_per_candidate: int,
        rollout_steps: int,
        use_async_envs: bool,
        policy: ReservoirPolicy,
        device: torch.device,
    ):
        self.env_id = env_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_candidates = chunk_candidates
        self.episodes_per_candidate = episodes_per_candidate
        self.rollout_steps = rollout_steps
        self.use_async_envs = use_async_envs
        self.policy = policy
        self.device = device

        self.num_envs = self.chunk_candidates * self.episodes_per_candidate

        env_fns = [functools.partial(build_env, self.env_id) for _ in range(self.num_envs)]
        if self.use_async_envs:
            self.vec_env = AsyncVectorEnv(env_fns, shared_memory=False)
        else:
            self.vec_env = SyncVectorEnv(env_fns)

    def close(self) -> None:
        self.vec_env.close()

    @torch.no_grad()
    def evaluate_chunk(
        self,
        chunk_theta: torch.Tensor,
        seed_base: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Evaluate up to chunk_candidates parameters.

        Returns:
          returns: [C]
          mean_lengths: [C]
          exploded: [C] bool
          env_steps: total environment steps consumed by this chunk
        """
        C_active = chunk_theta.shape[0]
        C_full = self.chunk_candidates
        E = self.episodes_per_candidate

        if C_active > C_full:
            raise RuntimeError(f"Chunk too large: {C_active} > {C_full}")

        if C_active < C_full:
            pad_count = C_full - C_active
            pad = chunk_theta[-1:].repeat(pad_count, 1)
            theta_eval = torch.cat([chunk_theta, pad], dim=0)
        else:
            theta_eval = chunk_theta

        cand_U, cand_V, cand_Wa, cand_ba = self.policy.split_theta(theta_eval)

        seeds = [int(seed_base + i) for i in range(self.num_envs)]
        obs, _ = self.vec_env.reset(seed=seeds)

        h = torch.zeros(
            (C_full, E, self.policy.cfg.N),
            dtype=torch.float32,
            device=self.device,
        )

        returns = np.zeros(self.num_envs, dtype=np.float64)
        lengths = np.zeros(self.num_envs, dtype=np.int32)
        alive = np.ones(self.num_envs, dtype=bool)
        exploded_by_candidate = np.zeros(C_full, dtype=bool)

        stats = make_eval_stats()
        chunk_t0 = time.perf_counter()

        for step in range(self.rollout_steps):
            active_count = int(alive.sum())
            if active_count == 0:
                break

            stats["loop_steps"] += 1
            stats["effective_env_steps"] += active_count

            t_tensor_0 = time.perf_counter()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(C_full, E, self.obs_dim)

            alive_mask = torch.as_tensor(alive, device=self.device).view(C_full, E, 1).to(torch.float32)
            h = h * alive_mask
            stats["tensor_s"] += time.perf_counter() - t_tensor_0

            t_policy_0 = time.perf_counter()
            h, action_t, exploded_t = self.policy.policy_step(h, obs_t, cand_U, cand_V, cand_Wa, cand_ba)

            if exploded_t.any():
                exploded_np = exploded_t.detach().cpu().numpy()
                exploded_by_candidate |= exploded_np.any(axis=1)
                keep_mask = torch.as_tensor(~exploded_np, device=self.device).view(C_full, E, 1).to(torch.float32)
                h = h * keep_mask
            stats["policy_s"] += time.perf_counter() - t_policy_0

            action_np = action_t.view(self.num_envs, self.action_dim).detach().cpu().numpy()
            t_env_0 = time.perf_counter()
            obs, reward, terminated, truncated, _ = self.vec_env.step(action_np)
            stats["env_s"] += time.perf_counter() - t_env_0

            done = np.logical_or(terminated, truncated)
            returns[alive] += reward[alive]

            newly_done = np.logical_and(alive, done)
            if newly_done.any():
                lengths[newly_done] = step + 1
                alive[newly_done] = False

        lengths[alive] = self.rollout_steps

        returns_c = returns.reshape(C_full, E).mean(axis=1)
        lengths_c = lengths.reshape(C_full, E).mean(axis=1)

        stats["actual_env_steps"] = int(stats["loop_steps"]) * self.num_envs
        stats["chunks"] = 1
        stats["wall_s"] = time.perf_counter() - chunk_t0

        return (
            returns_c[:C_active].copy(),
            lengths_c[:C_active].copy(),
            exploded_by_candidate[:C_active].copy(),
            stats,
        )

    @torch.no_grad()
    def evaluate_population(
        self,
        population_theta: torch.Tensor,
        seed_base: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        num_candidates = population_theta.shape[0]
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
    theta: torch.Tensor,
    args: argparse.Namespace,
    extra: Dict[str, float],
) -> None:
    payload = {
        "iter": iteration,
        "theta": theta.detach().cpu(),
        "args": vars(args),
        "extra": extra,
    }
    torch.save(payload, path)


def get_gpu_stats(device: torch.device) -> Dict[str, float]:
    out = {
        "gpu_total_gb": float("nan"),
        "gpu_free_gb": float("nan"),
        "gpu_used_frac": float("nan"),
        "gpu_peak_alloc_gb": float("nan"),
        "gpu_peak_reserved_gb": float("nan"),
        "gpu_peak_reserved_frac": float("nan"),
    }
    if device.type != "cuda":
        return out

    free_b, total_b = torch.cuda.mem_get_info(device)
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    out["gpu_total_gb"] = total_b / 1e9
    out["gpu_free_gb"] = free_b / 1e9
    out["gpu_used_frac"] = 1.0 - (free_b / max(total_b, 1))
    out["gpu_peak_alloc_gb"] = peak_alloc / 1e9
    out["gpu_peak_reserved_gb"] = peak_reserved / 1e9
    out["gpu_peak_reserved_frac"] = peak_reserved / max(total_b, 1)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pertubative RNN RL (Humanoid)")

    parser.add_argument("--env_candidates", type=str, default="Humanoid-v5,Humanoid-v4")
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
    parser.add_argument("--sync_env", action="store_true", help="Use SyncVectorEnv for debugging")

    parser.add_argument("--init_action_std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch_num_threads", type=int, default=4)

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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.torch_num_threads))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    env_candidates = [x.strip() for x in args.env_candidates.split(",") if x.strip()]
    env_id, obs_dim, action_dim, action_low, action_high = resolve_env_id(env_candidates)
    print(f"Using environment: {env_id}", flush=True)
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

    policy = ReservoirPolicy(cfg, action_low=action_low, action_high=action_high, device=device).to(device)
    policy.eval()

    theta = nn.Parameter(torch.zeros(policy.theta_dim, dtype=torch.float32, device=device))

    # Initialize action head portion with small random values for non-degenerate startup.
    with torch.no_grad():
        n, r, a = cfg.N, cfg.rank, cfg.A
        uv = 2 * n * r
        wa = a * n
        theta[uv : uv + wa].normal_(mean=0.0, std=args.init_action_std)
        theta[uv + wa : uv + wa + a].zero_()

    optimizer = torch.optim.Adam([theta], lr=args.theta_lr)

    evaluator = CandidateEvaluator(
        env_id=env_id,
        obs_dim=obs_dim,
        action_dim=action_dim,
        chunk_candidates=args.candidate_chunk,
        episodes_per_candidate=args.episodes_per_candidate,
        rollout_steps=args.rollout_steps,
        use_async_envs=not args.sync_env,
        policy=policy,
        device=device,
    )

    print(
        f"Rollout evaluator: chunk={args.candidate_chunk}, "
        f"episodes/candidate={args.episodes_per_candidate}, envs={evaluator.num_envs}, "
        f"async={not args.sync_env}",
        flush=True,
    )

    best_base_return = -float("inf")
    start = time.time()

    try:
        for iteration in range(1, args.iters + 1):
            iter_t0 = time.time()
            seed_it = 10_000_000 + args.seed * 1_000_000 + iteration * 1000

            K = args.pairs
            theta_dim = theta.numel()

            epsilon = torch.randn((K, theta_dim), dtype=torch.float32, device=device)
            theta_view = theta.detach().unsqueeze(0)
            pop_theta = torch.cat(
                [
                    theta_view + args.sigma * epsilon,
                    theta_view - args.sigma * epsilon,
                ],
                dim=0,
            )

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            cand_returns_np, cand_lengths_np, cand_exploded_np, eval_stats = evaluator.evaluate_population(
                pop_theta, seed_it
            )

            cand_returns = torch.as_tensor(cand_returns_np, dtype=torch.float32, device=device)
            fitness = rank_transform(cand_returns, maximize=True)
            w_pos = fitness[:K]
            w_neg = fitness[K:]

            grad_est = (w_pos.view(-1, 1) * epsilon - w_neg.view(-1, 1) * epsilon).mean(dim=0)

            optimizer.zero_grad(set_to_none=True)
            theta.grad = -grad_est
            optimizer.step()

            base_return = float("nan")
            if iteration % args.log_every == 0 or iteration == 1:
                base_ret_np, _, _, base_stats = evaluator.evaluate_population(
                    theta.detach().unsqueeze(0),
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

            timed_total = float(eval_stats["policy_s"] + eval_stats["env_s"] + eval_stats["tensor_s"])
            eval_policy_share = float(eval_stats["policy_s"] / max(timed_total, 1e-9))
            eval_env_share = float(eval_stats["env_s"] / max(timed_total, 1e-9))
            cand_per_sec = float((2 * K) / max(float(eval_stats["wall_s"]), 1e-9))

            gpu = get_gpu_stats(device)
            gpu_peak_reserved_frac = float(gpu["gpu_peak_reserved_frac"])
            gpu_peak_reserved_gb = float(gpu["gpu_peak_reserved_gb"])
            gpu_used_frac = float(gpu["gpu_used_frac"])

            if np.isfinite(gpu_peak_reserved_frac) and gpu_peak_reserved_frac > 0:
                mem_scale = max(1.0, min(4.0, 0.85 / gpu_peak_reserved_frac))
            else:
                mem_scale = 1.0

            if eval_env_share > 0.75:
                suggest_chunk = args.candidate_chunk
            else:
                suggest_chunk = int(max(args.candidate_chunk, round(args.candidate_chunk * mem_scale)))

            suggest_pairs = int(max(args.pairs, round(args.pairs * (args.headroom_target_iter_sec / max(iter_elapsed, 1e-9)))))
            suggest_envs = int(suggest_chunk * args.episodes_per_candidate)

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
                f"|theta| {theta_norm:8.2f} | eff_fps {fps:9.1f} | elapsed {elapsed/60.0:7.1f}m",
                flush=True,
            )
            print(
                f"  profile | cand/s {cand_per_sec:8.1f} | eval split policy/env/tensor "
                f"{eval_policy_share*100:5.1f}/{eval_env_share*100:5.1f}/{(1.0-eval_policy_share-eval_env_share)*100:5.1f}% | "
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
                os.path.join(ckpt_dir, "checkpoint_latest.pt"),
                iteration,
                theta,
                args,
                {
                    "base_return": base_return,
                    "cand_mean": cand_mean,
                    "cand_max": cand_max,
                    "bad_frac": bad_frac,
                },
            )

            if iteration % args.checkpoint_every == 0:
                save_checkpoint(
                    os.path.join(ckpt_dir, f"checkpoint_iter_{iteration:05d}.pt"),
                    iteration,
                    theta,
                    args,
                    {
                        "base_return": base_return,
                        "cand_mean": cand_mean,
                        "cand_max": cand_max,
                        "bad_frac": bad_frac,
                    },
                )

            if np.isfinite(base_return) and base_return > best_base_return:
                best_base_return = base_return
                save_checkpoint(
                    os.path.join(ckpt_dir, "checkpoint_best.pt"),
                    iteration,
                    theta,
                    args,
                    {
                        "base_return": base_return,
                        "cand_mean": cand_mean,
                        "cand_max": cand_max,
                        "bad_frac": bad_frac,
                    },
                )

        summary = {
            "env_id": env_id,
            "iters": args.iters,
            "pairs": args.pairs,
            "episodes_per_candidate": args.episodes_per_candidate,
            "candidate_chunk": args.candidate_chunk,
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
