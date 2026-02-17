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
    parser.add_argument("--k_in", type=int, default=50)
    parser.add_argument("--leak", type=float, default=0.2)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--pairs", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--theta_lr", type=float, default=0.01)
    parser.add_argument("--episodes_per_candidate", type=int, default=1)
    parser.add_argument("--candidate_chunk", type=int, default=256)
    parser.add_argument("--rollout_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--results_root", type=str, default="humanoid/results")
    args = parser.parse_args()

    # Setup
    key = jax.random.PRNGKey(args.seed)
    env_id, env_spec = resolve_env_spec(args.env_candidates)
    xml_path = resolve_humanoid_xml(args.xml_path)
    env = MJXHumanoidEnv(xml_path, env_spec)
    
    cfg = ReservoirConfig(
        N=args.hidden, 
        D=env.obs_dim, 
        A=env.action_dim, 
        rank=args.rank, 
        k_in=args.k_in, 
        leak=args.leak
    )
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

    # Compile a single-chunk rollout function
    def rollout_chunk(chunk_theta, chunk_key):
        """Rollout a single chunk of candidates."""
        K_chunk = chunk_theta.shape[0]
        E = args.episodes_per_candidate
        
        # Reset envs for this chunk
        reset_keys = jax.random.split(chunk_key, K_chunk * E)
        data, obs = jax.vmap(env.reset_one)(reset_keys)
        
        # Reshape for [K_chunk, E, ...]
        obs = obs.reshape((K_chunk, E, -1))
        h = jnp.zeros((K_chunk, E, cfg.N))
        returns = jnp.zeros((K_chunk, E))
        done = jnp.zeros((K_chunk, E), dtype=bool)

        def step_fn(carry, _):
            h, data, obs, returns, done = carry
            h_next, action = policy.batched_rollout(h, obs, chunk_theta)
            
            action_flat = action.reshape((K_chunk * E, -1))
            data_next, obs_next, reward, terminated = jax.vmap(env.step)(data, action_flat)
            
            obs_next = obs_next.reshape((K_chunk, E, -1))
            reward = reward.reshape((K_chunk, E))
            terminated = terminated.reshape((K_chunk, E))
            
            returns += reward * (~done)
            done |= terminated
            return (h_next, data_next, obs_next, returns, done), None

        # Use scan for time loop to prevent unrolling
        final_carry, _ = jax.lax.scan(step_fn, (h, data, obs, returns, done), None, length=args.rollout_steps)
        return jnp.mean(final_carry[3], axis=1)

    jit_rollout_chunk = jax.jit(rollout_chunk)

    def rollout_fn(theta_pop, rollout_key):
        """Process population in chunks using a Python loop."""
        K = theta_pop.shape[0]
        chunk_size = args.candidate_chunk if args.candidate_chunk > 0 else K
        
        if K % chunk_size != 0:
            chunk_size = K
        
        num_chunks = K // chunk_size
        theta_chunks = theta_pop.reshape((num_chunks, chunk_size, -1))
        keys = jax.random.split(rollout_key, num_chunks)
        
        all_results = []
        for i in range(num_chunks):
            if i == 0: 
                print(f"  [Chunk {i+1}/{num_chunks}] Configuring & Compiling... (may take ~2 mins)", flush=True)
                t0 = time.time()
            
            res = jit_rollout_chunk(theta_chunks[i], keys[i])
            res.block_until_ready()
            
            if i == 0:
                print(f"  [Chunk {i+1}/{num_chunks}] Finished first chunk in {time.time()-t0:.1f}s", flush=True)
            
            all_results.append(res)
            
        return jnp.concatenate(all_results, axis=0)
    # jit_rollout = jax.jit(rollout_fn) # Not needed as we manually JIT chunks
    
    print(f"Starting ES loop on {env_id}...", flush=True)
    start_time = time.time()
    
    for it in range(1, args.iters + 1):
        it_start = time.time()
        loop_key, eps_key, rollout_key = jax.random.split(loop_key, 3)
        
        # Sample perturbations
        epsilon = jax.random.normal(eps_key, (args.pairs, policy.theta_dim))
        pop_theta = jnp.concatenate([theta + args.sigma * epsilon, theta - args.sigma * epsilon], axis=0)
        
        # Evaluate population
        cand_returns = rollout_fn(pop_theta, rollout_key)
        
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

        # Logging & Checkpointing
        if it % args.log_every == 0 or it == 1:
            base_ret = float(rollout_fn(theta[None, :], rollout_key)[0])
            elapsed = (time.time() - start_time) / 60
            fps = (args.pairs * 2 * args.episodes_per_candidate * args.rollout_steps) / (time.time() - it_start)
            print(f"Iter {it:3d} | Base Ret: {base_ret:8.2f} | Cand Mean: {jnp.mean(cand_returns):8.2f} | FPS: {fps:6.0f} | {elapsed:4.1f}m", flush=True)
            logger.log({"iter": it, "base_return": base_ret, "cand_mean": float(jnp.mean(cand_returns)), "fps": fps, "elapsed_min": elapsed})
        
        if it % args.checkpoint_every == 0 or it == args.iters:
            save_checkpoint(os.path.join(run_dir, "checkpoint_latest.pkl"), it, theta, args, {"base_return": base_ret if 'base_ret' in locals() else -1.0})

    logger.close()

if __name__ == "__main__":
    main()
