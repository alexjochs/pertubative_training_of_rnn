import argparse
import datetime
import os
import time
import jax
import jax.numpy as jnp
import numpy as np

from env import resolve_env_spec, resolve_humanoid_xml, MJXHumanoidEnv
from model import ReservoirConfig, ReservoirPolicy, sample_hyperparams, generate_reservoir_params
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
    parser.add_argument("--num_candidates", type=int, default=1024, help="Number of reservoirs to search over initially.")
    parser.add_argument("--search_burst_iters", type=int, default=20, help="Number of ES iterations per halving round.")
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
    
    # Generate population properties
    hp = sample_hyperparams(key, args.num_candidates)
    res_pop = generate_reservoir_params(cfg, key, hp)
    
    # Initialize theta
    theta_pop = jnp.zeros((args.num_candidates, policy.theta_dim))
    n, r, a = cfg.N, cfg.rank, cfg.A
    uv, wa = 2 * n * r, a * n
    head_key, loop_key = jax.random.split(key)
    theta_pop = theta_pop.at[:, uv : uv + wa].set(0.02 * jax.random.normal(head_key, (args.num_candidates, wa)))

    # Adam state for JAX
    m_pop = jnp.zeros_like(theta_pop)
    v_pop = jnp.zeros_like(theta_pop)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # Directories & Logging
    run_dir = os.path.join(args.results_root, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    save_config(args, os.path.join(run_dir, "config.json"))
    
    # ML Researcher Stats Headers
    logger_fields = [
        "stage", "active_candidates", "iter", "base_return", "max_return", "cand_mean", 
        "survival_rate", "u_norm", "v_norm", "wa_norm",
        "update_ratio", "fps", "elapsed_min"
    ]
    logger = CSVLogger(os.path.join(run_dir, "train_log.csv"), logger_fields)
    enable_jax_compilation_cache("humanoid/jax_compile_cache")

    print(f"Logging results to: {run_dir}", flush=True)

    def build_functions(active_candidates, active_pairs):
        def rollout_fn(flat_theta, flat_res, rollout_key):
            K = flat_theta.shape[0]
            E = args.episodes_per_candidate
            chunk_size = args.candidate_chunk if args.candidate_chunk <= K else K
            
            if K % chunk_size != 0:
                chunk_size = K
            num_chunks = K // chunk_size

            flat_theta = flat_theta.reshape((num_chunks, chunk_size, -1))
            chunked_res = jax.tree.map(lambda x: x.reshape((num_chunks, chunk_size, *x.shape[1:])), flat_res)
            keys = jax.random.split(rollout_key, num_chunks)

            def chunk_step(chunk_theta, chunk_res_params, chunk_key):
                reset_keys = jax.random.split(chunk_key, chunk_size * E)
                data, obs = jax.vmap(env.reset_one)(reset_keys)
                
                obs = obs.reshape((chunk_size, E, -1))
                h = jnp.zeros((chunk_size, E, cfg.N))
                returns = jnp.zeros((chunk_size, E))
                done = jnp.zeros((chunk_size, E), dtype=bool)

                def step_fn(carry, _):
                    h, data, obs, returns, done = carry
                    h_next, action = policy.batched_rollout(h, obs, chunk_theta, chunk_res_params)
                    
                    action_flat = action.reshape((chunk_size * E, -1))
                    data_next, obs_next, reward, terminated = jax.vmap(env.step)(data, action_flat)
                    
                    obs_next = obs_next.reshape((chunk_size, E, -1))
                    reward = reward.reshape((chunk_size, E))
                    terminated = terminated.reshape((chunk_size, E))
                    
                    returns += reward * (~done)
                    done |= terminated
                    return (h_next, data_next, obs_next, returns, done), done

                final_carry, _ = jax.lax.scan(step_fn, (h, data, obs, returns, done), None, length=args.rollout_steps)
                survived = jnp.logical_not(final_carry[4]) 
                return jnp.mean(final_carry[3], axis=1), jnp.mean(survived.astype(jnp.float32))

            _, results = jax.lax.scan(lambda _, x: (None, chunk_step(x[0], x[1], x[2])), None, (flat_theta, chunked_res, keys))
            chunk_returns, chunk_survivals = results
            return chunk_returns.reshape((K,)), jnp.mean(chunk_survivals)

        return jax.jit(rollout_fn)

    start_time = time.time()
    
    current_candidates = args.num_candidates
    search_pairs = 32 # Keep pairs low during search to avoid OOM with large candidate pool
    stage = 1

    while current_candidates >= 1:
        if current_candidates > 1:
            stage_iters = args.search_burst_iters
            active_pairs = search_pairs
            # First round freeze UV
            update_uv = (current_candidates < args.num_candidates)
            print(f"\n--- Search Stage {stage}: {current_candidates} Candidates | {stage_iters} Iters | Update UV: {update_uv} ---", flush=True)
        else:
            stage_iters = args.iters
            active_pairs = args.pairs
            update_uv = True
            print(f"\n--- Main Phase: 1 Candidate | {stage_iters} Iters | Update UV: True ---", flush=True)

        uv_mask = jnp.ones(policy.theta_dim)
        if not update_uv:
            uv_mask = uv_mask.at[:uv].set(0.0)

        # Build execution functions for this shape
        jit_rollout = build_functions(current_candidates, active_pairs)
        
        @jax.jit
        def train_step(theta, m, v, it, res_params, rollout_key, eps_key):
            # Sample perturbations [C, P, D]
            epsilon = jax.random.normal(eps_key, (current_candidates, active_pairs, policy.theta_dim))
            epsilon = epsilon * uv_mask
            
            # pop_theta [C, 2P, D]
            pop_theta_inner = jnp.concatenate([theta[:, None, :] + args.sigma * epsilon, 
                                         theta[:, None, :] - args.sigma * epsilon], axis=1)
            
            flat_theta = pop_theta_inner.reshape((current_candidates * 2 * active_pairs, policy.theta_dim))
            
            flat_res_params = jax.tree.map(lambda x: jnp.repeat(x, 2 * active_pairs, axis=0), res_params)
            
            cand_returns, survival_rate = jit_rollout(flat_theta, flat_res_params, rollout_key)
            cand_returns = cand_returns.reshape((current_candidates, 2 * active_pairs))
            
            # Gradient Estimation
            fitness = jax.vmap(rank_transform)(cand_returns)
            w_pos, w_neg = fitness[:, :active_pairs], fitness[:, active_pairs:]
            grad = -jnp.sum((w_pos - w_neg)[..., None] * epsilon, axis=1) / active_pairs

            # Adam Update
            m_next = beta1 * m + (1 - beta1) * grad
            v_next = beta2 * v + (1 - beta2) * (grad**2)
            
            # For iteration correction, we want iter to be global per candidate or per stage.
            # Using stage iteration simplifies things.
            m_hat = m_next / (1 - beta1 ** it)
            v_hat = v_next / (1 - beta2 ** it)
            
            delta = args.theta_lr * m_hat / (jnp.sqrt(v_hat) + eps)
            theta_next = theta - delta
            
            stats = {
                "cand_mean": jnp.mean(cand_returns),
                "max_return": jnp.max(cand_returns),
                "survival_rate": survival_rate,
                "update_ratio": jnp.linalg.norm(delta) / (jnp.linalg.norm(theta) + 1e-9)
            }
            
            return theta_next, m_next, v_next, stats
            
        @jax.jit
        def eval_base_candidates(theta, res_params, rollout_key):
            return jit_rollout(theta, res_params, rollout_key)

        for it in range(1, stage_iters + 1):
            it_start = time.time()
            loop_key, eps_key, rollout_key = jax.random.split(loop_key, 3)
            
            theta_pop, m_pop, v_pop, stats = train_step(theta_pop, m_pop, v_pop, it, res_pop, rollout_key, eps_key)
            
            if it % args.log_every == 0 or it == 1:
                base_ret_arr, _ = eval_base_candidates(theta_pop, res_pop, rollout_key)
                
                # Report metrics across all active candidates
                best_base_ret = float(jnp.max(base_ret_arr))
                avg_base_ret = float(jnp.mean(base_ret_arr))
                
                elapsed = (time.time() - start_time) / 60
                fps = (current_candidates * active_pairs * 2 * args.episodes_per_candidate * args.rollout_steps) / (time.time() - it_start)
                
                # Take norms of the best candidate to monitor scale
                best_idx = jnp.argmax(base_ret_arr)
                u, v_params, wa, ba = policy.split_theta(theta_pop[best_idx])
                un, vn, wan = jnp.linalg.norm(u), jnp.linalg.norm(v_params), jnp.linalg.norm(wa)

                print(f"Iter {it:3d} | Best Base: {best_base_ret:7.1f} | Avg Base: {avg_base_ret:7.1f} | Max Perturb: {float(stats['max_return']):7.1f} | Surv: {float(stats['survival_rate']):5.2f} | FPS: {fps:6.0f} | Ratio: {float(stats['update_ratio']):.2e} | {elapsed:4.1f}m", flush=True)
                
                logger.log({
                    "stage": stage,
                    "active_candidates": current_candidates,
                    "iter": it, 
                    "base_return": best_base_ret, 
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
            
            if current_candidates == 1 and (it % args.checkpoint_every == 0 or it == stage_iters):
                ckpt_name = f"checkpoint_it_{it:04d}.pkl"
                ckpt_path = os.path.join(run_dir, ckpt_name)
                save_checkpoint(ckpt_path, it, theta_pop[0], args, {
                    "base_return": best_base_ret if 'best_base_ret' in locals() else -1.0,
                    "max_return": float(stats["max_return"]),
                    "survival_rate": float(stats["survival_rate"])
                })

        # Halving Step
        if current_candidates > 1:
            loop_key, eval_key = jax.random.split(loop_key)
            returns, _ = eval_base_candidates(theta_pop, res_pop, eval_key)
            
            keep_n = current_candidates // 2
            top_idx = jnp.argsort(returns)[-keep_n:]
            
            print(f"Halving from {current_candidates} to {keep_n}. Reward threshold: {float(returns[top_idx[0]]):.2f}", flush=True)
            
            theta_pop = theta_pop[top_idx]
            m_pop = m_pop[top_idx]
            v_pop = v_pop[top_idx]
            res_pop = jax.tree.map(lambda x: x[top_idx], res_pop)
            
            current_candidates = keep_n
            stage += 1
        else:
            break

    logger.close()

if __name__ == "__main__":
    main()
