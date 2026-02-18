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
    
    # ML Researcher Stats Headers
    logger_fields = [
        "iter", "base_return", "max_return", "cand_mean", 
        "survival_rate", "u_norm", "v_norm", "wa_norm",
        "update_ratio", "fps", "elapsed_min"
    ]
    logger = CSVLogger(os.path.join(run_dir, "train_log.csv"), logger_fields)
    enable_jax_compilation_cache("humanoid/jax_compile_cache")

    print(f"Logging results to: {run_dir}", flush=True)

    def rollout_fn(theta_pop, rollout_key):
        """Rollout a population of policies, potentially in chunks."""
        K = theta_pop.shape[0]
        E = args.episodes_per_candidate
        chunk_size = args.candidate_chunk if args.candidate_chunk > 0 else K
        num_chunks = K // chunk_size
        
        # Ensure K is divisible by chunk_size for simplicity 
        if K % chunk_size != 0:
            chunk_size = K
            num_chunks = 1

        theta_pop = theta_pop.reshape((num_chunks, chunk_size, -1))
        keys = jax.random.split(rollout_key, num_chunks)

        def chunk_step(chunk_theta, chunk_key):
            # Reset all envs in chunk
            reset_keys = jax.random.split(chunk_key, chunk_size * E)
            data, obs = jax.vmap(env.reset_one)(reset_keys)
            
            # Reshape for [chunk_size, E, ...]
            obs = obs.reshape((chunk_size, E, -1))
            h = jnp.zeros((chunk_size, E, cfg.N))
            returns = jnp.zeros((chunk_size, E))
            done = jnp.zeros((chunk_size, E), dtype=bool)

            def step_fn(carry, _):
                h, data, obs, returns, done = carry
                h_next, action = policy.batched_rollout(h, obs, chunk_theta)
                
                action_flat = action.reshape((chunk_size * E, -1))
                data_next, obs_next, reward, terminated = jax.vmap(env.step)(data, action_flat)
                
                obs_next = obs_next.reshape((chunk_size, E, -1))
                reward = reward.reshape((chunk_size, E))
                terminated = terminated.reshape((chunk_size, E))
                
                returns += reward * (~done)
                done |= terminated
                return (h_next, data_next, obs_next, returns, done), done

            # We capture 'done' history to calculate survival rate
            final_carry, done_history = jax.lax.scan(step_fn, (h, data, obs, returns, done), None, length=args.rollout_steps)
            
            # Survival: did they reach the end without done=True?
            # final_carry[4] is the final 'done' state
            survived = jnp.logical_not(final_carry[4]) # [chunk_size, E]
            return jnp.mean(final_carry[3], axis=1), jnp.mean(survived.astype(jnp.float32))

        # Process chunks sequentially using scan to save memory
        _, results = jax.lax.scan(lambda _, x: (None, chunk_step(x[0], x[1])), None, (theta_pop, keys))
        chunk_returns, chunk_survivals = results
        return chunk_returns.reshape((K,)), jnp.mean(chunk_survivals)

    jit_rollout = jax.jit(rollout_fn)

    def train_step(state, rollout_key, eps_key):
        theta, m, v, it = state
        
        # Sample perturbations
        epsilon = jax.random.normal(eps_key, (args.pairs, policy.theta_dim))
        pop_theta = jnp.concatenate([theta + args.sigma * epsilon, theta - args.sigma * epsilon], axis=0)
        
        # Evaluate population
        cand_returns, survival_rate = rollout_fn(pop_theta, rollout_key)
        
        # Gradient Estimation
        fitness = rank_transform(cand_returns)
        w_pos, w_neg = fitness[:args.pairs, None], fitness[args.pairs:, None]
        grad = -jnp.mean((w_pos - w_neg) * epsilon, axis=0)

        # Adam Update
        m_next = beta1 * m + (1 - beta1) * grad
        v_next = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m_next / (1 - beta1 ** (it + 1))
        v_hat = v_next / (1 - beta2 ** (it + 1))
        
        delta = args.theta_lr * m_hat / (jnp.sqrt(v_hat) + eps)
        theta_next = theta - delta
        
        stats = {
            "cand_mean": jnp.mean(cand_returns),
            "max_return": jnp.max(cand_returns),
            "survival_rate": survival_rate,
            "grad_norm": jnp.linalg.norm(grad),
            "update_ratio": jnp.linalg.norm(delta) / (jnp.linalg.norm(theta) + 1e-9)
        }
        
        return (theta_next, m_next, v_next, it + 1), stats

    jit_train_step = jax.jit(train_step)

    print(f"Starting ES loop on {env_id}...", flush=True)
    start_time = time.time()
    
    state = (theta, m, v, 0)
    
    for it in range(1, args.iters + 1):
        it_start = time.time()
        loop_key, eps_key, rollout_key = jax.random.split(loop_key, 3)
        
        # Execute fused training step on GPU
        state, stats = jit_train_step(state, rollout_key, eps_key)
        theta, m, v, _ = state
        
        # Stats for logging
        if it % args.log_every == 0 or it == 1:
            # We use the jit_rollout for the base return check
            base_ret_arr, _ = jit_rollout(theta[None, :], rollout_key)
            base_ret = float(base_ret_arr[0])
            
            elapsed = (time.time() - start_time) / 60
            fps = (args.pairs * 2 * args.episodes_per_candidate * args.rollout_steps) / (time.time() - it_start)
            
            u, v_params, wa, ba = policy.split_theta(theta)
            un, vn, wan = jnp.linalg.norm(u), jnp.linalg.norm(v_params), jnp.linalg.norm(wa)

            print(f"Iter {it:3d} | Base: {base_ret:7.1f} | Max: {float(stats['max_return']):7.1f} | Surv: {float(stats['survival_rate']):5.2f} | FPS: {fps:6.0f} | Ratio: {float(stats['update_ratio']):.2e} | {elapsed:4.1f}m", flush=True)
            
            logger.log({
                "iter": it, 
                "base_return": base_ret, 
                "max_return": float(stats["max_return"]),
                "cand_mean": float(stats["cand_mean"]), 
                "survival_rate": float(stats["survival_rate"]),
                "u_norm": float(un),
                "v_norm": float(vn),
                "wa_norm": float(wan),
                "update_ratio": float(stats["update_ratio"]),
                "fps": fps, 
                "elapsed_min": elapsed
            })
        
        if it % args.checkpoint_every == 0 or it == args.iters:
            ckpt_name = f"checkpoint_it_{it:04d}.pkl"
            ckpt_path = os.path.join(run_dir, ckpt_name)
            save_checkpoint(ckpt_path, it, theta, args, {
                "base_return": base_ret if 'base_ret' in locals() else -1.0,
                "max_return": float(stats["max_return"]),
                "survival_rate": float(stats["survival_rate"])
            })
            print(f"  [Checkpoint] Saved to {ckpt_path}", flush=True)

    logger.close()

if __name__ == "__main__":
    main()
