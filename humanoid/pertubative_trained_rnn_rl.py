import argparse
import datetime
import os
import time
import jax
import jax.numpy as jnp
import numpy as np

from env import resolve_env_spec, resolve_humanoid_xml, MJXHumanoidEnv
from model import ReservoirConfig, ReservoirPolicy
from utils import (
    CSVLogger, save_config, enable_jax_compilation_cache, 
    build_compile_key, get_gpu_stats, save_checkpoint
)

def rank_transform(values: jnp.ndarray) -> jnp.ndarray:
    n = values.shape[0]
    order = jnp.argsort(values)
    ranks = jnp.zeros(n).at[order].set(jnp.arange(n, dtype=jnp.float32))
    w = ranks / (n - 1.0) - 0.5
    return w - jnp.mean(w)

def main():
    parser = argparse.ArgumentParser(description="Clean Perturbative RNN RL (MJX)")
    parser.add_argument("--env_candidates", type=str, default="Humanoid-v5,Humanoid-v4")
    parser.add_argument("--xml_path", type=str, default="")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--pairs", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--theta_lr", type=float, default=0.01)
    parser.add_argument("--episodes_per_candidate", type=int, default=1)
    parser.add_argument("--rollout_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--results_root", type=str, default="humanoid/results")
    args = parser.parse_args()

    # Setup
    key = jax.random.PRNGKey(args.seed)
    env_id, env_spec = resolve_env_spec(args.env_candidates)
    xml_path = resolve_humanoid_xml(args.xml_path)
    env = MJXHumanoidEnv(xml_path, env_spec)
    
    cfg = ReservoirConfig(N=args.hidden, D=env.obs_dim, A=env.action_dim, rank=args.rank)
    policy = ReservoirPolicy(cfg, (jnp.array(env.model.actuator_ctrlrange[:, 0]), jnp.array(env.model.actuator_ctrlrange[:, 1])))
    
    # Initialize theta
    theta = jnp.zeros(policy.theta_dim)
    # Small random init for action head (using same logic as before)
    n, r, a = cfg.N, cfg.rank, cfg.A
    uv, wa = 2 * n * r, a * n
    head_key, loop_key = jax.random.split(key)
    theta = theta.at[uv : uv + wa].set(0.02 * jax.random.normal(head_key, (wa,)))

    # Adam state for JAX (manual implementation for simplicity)
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # Directories & Logging
    run_dir = os.path.join(args.results_root, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    save_config(args, os.path.join(run_dir, "config.json"))
    logger = CSVLogger(os.path.join(run_dir, "train_log.csv"), ["iter", "base_return", "cand_mean", "fps", "elapsed_min"])
    enable_jax_compilation_cache("humanoid/jax_compile_cache")

    def rollout_fn(theta_pop, rollout_key):
        """Rollout a population of policies."""
        K = theta_pop.shape[0]
        E = args.episodes_per_candidate
        
        # Reset all envs
        reset_keys = jax.random.split(rollout_key, K * E)
        data, obs = jax.vmap(env.reset_one)(reset_keys)
        
        # Reshape for [K, E, ...]
        obs = obs.reshape((K, E, -1))
        h = jnp.zeros((K, E, cfg.N))
        returns = jnp.zeros((K, E))
        done = jnp.zeros((K, E), dtype=bool)

        def step_fn(i, carry):
            h, data, obs, returns, done = carry
            # Vectorized policy step across K and E
            h_next, action = policy.batched_rollout(h, obs, theta_pop)
            
            # Step environments
            action_flat = action.reshape((K * E, -1))
            data_next, obs_next, reward, terminated = jax.vmap(env.step)(data, action_flat)
            
            # Update state
            obs_next = obs_next.reshape((K, E, -1))
            reward = reward.reshape((K, E))
            terminated = terminated.reshape((K, E))
            
            # Only accumulate reward if not done
            returns += reward * (~done)
            done |= terminated
            return h_next, data_next, obs_next, returns, done

        final_carry = jax.lax.fori_loop(0, args.rollout_steps, step_fn, (h, data, obs, returns, done))
        return jnp.mean(final_carry[3], axis=1) # Mean over episodes

    jit_rollout = jax.jit(rollout_fn)

    print(f"Starting ES loop on {env_id}...")
    start_time = time.time()
    
    for it in range(1, args.iters + 1):
        it_start = time.time()
        loop_key, eps_key, rollout_key = jax.random.split(loop_key, 3)
        
        # Sample perturbations
        epsilon = jax.random.normal(eps_key, (args.pairs, policy.theta_dim))
        pop_theta = jnp.concatenate([theta + args.sigma * epsilon, theta - args.sigma * epsilon], axis=0)
        
        # Evaluate population
        cand_returns = jit_rollout(pop_theta, rollout_key)
        
        # Gradient Estimation
        fitness = rank_transform(cand_returns)
        w_pos, w_neg = fitness[:args.pairs, None], fitness[args.pairs:, None]
        grad = -jnp.mean((w_pos - w_neg) * epsilon, axis=0) # Negative for gradient ascent

        # Adam Update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1**it)
        v_hat = v / (1 - beta2**it)
        theta -= args.theta_lr * m_hat / (jnp.sqrt(v_hat) + eps)

        # Logging
        if it % args.log_every == 0 or it == 1:
            base_ret = float(jit_rollout(theta[None, :], rollout_key)[0])
            elapsed = (time.time() - start_time) / 60
            fps = (args.pairs * 2 * args.episodes_per_candidate * args.rollout_steps) / (time.time() - it_start)
            print(f"Iter {it:3d} | Base Ret: {base_ret:8.2f} | Cand Mean: {jnp.mean(cand_returns):8.2f} | FPS: {fps:6.0f} | {elapsed:4.1f}m")
            logger.log({"iter": it, "base_return": base_ret, "cand_mean": float(jnp.mean(cand_returns)), "fps": fps, "elapsed_min": elapsed})
            save_checkpoint(os.path.join(run_dir, "checkpoint_latest.pkl"), it, theta, args, {"base_return": base_ret})

    logger.close()

if __name__ == "__main__":
    main()
