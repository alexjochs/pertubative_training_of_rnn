from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp

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

def sample_hyperparams(key: jax.random.PRNGKey, num_candidates: int) -> dict:
    # 1024 candidates, sweep leak (log uniform), win_std (log uniform), and w0_std (log uniform)
    k1, k2, k3 = jax.random.split(key, 3)
    
    # leak in [0.01, 1.0] -> log(0.01) to log(1.0)
    leak = jnp.exp(jax.random.uniform(k1, (num_candidates,), minval=jnp.log(0.01), maxval=jnp.log(1.0)))
    
    # win_std in [0.01, 2.0] -> log(0.01) to log(2.0)
    win_std = jnp.exp(jax.random.uniform(k2, (num_candidates,), minval=jnp.log(0.01), maxval=jnp.log(2.0)))
    
    # w0_std in [0.1, 5.0] -> log(0.1) to log(5.0)
    w0_std = jnp.exp(jax.random.uniform(k3, (num_candidates,), minval=jnp.log(0.1), maxval=jnp.log(5.0)))
    
    return {"leak": leak, "win_std": win_std, "w0_std": w0_std}

def generate_reservoir_params(cfg: ReservoirConfig, key: jax.random.PRNGKey, hyperparams: dict) -> dict:
    leak_arr = hyperparams["leak"]
    win_std_arr = hyperparams["win_std"]
    w0_std_arr = hyperparams["w0_std"]
    num_cand = leak_arr.shape[0]
    
    keys = jax.random.split(key, num_cand)
    
    def make_res(k, win_std, w0_std):
        k1, k2 = jax.random.split(k)
        Win_T = (jax.random.normal(k1, (cfg.N, cfg.D)) * win_std).T
        W0 = jax.random.normal(k2, (cfg.N, cfg.N)) * (w0_std / (cfg.N**0.5))
        mask = jax.random.uniform(k2, (cfg.N, cfg.N)) < (cfg.k_in / cfg.N)
        return Win_T, (W0 * mask).T
        
    Win_T_pop, W0_T_pop = jax.vmap(make_res)(keys, win_std_arr, w0_std_arr)
    return {"leak": leak_arr[:, None], "Win_T": Win_T_pop, "W0_T": W0_T_pop}


class ReservoirPolicy:
    def __init__(self, cfg: ReservoirConfig, action_range: Tuple[jnp.ndarray, jnp.ndarray]):
        self.cfg = cfg
        self.low, self.high = action_range
        self.action_scale = 0.5 * (self.high - self.low)
        self.action_bias = 0.5 * (self.high + self.low)
        
        # We define a single b for all, though we could sweep it too
        self.b = jnp.zeros(cfg.N)

    @property
    def theta_dim(self) -> int:
        n, r, a = self.cfg.N, self.cfg.rank, self.cfg.A
        return 2 * n * r + a * n + a

    def split_theta(self, theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        n, r, a = self.cfg.N, self.cfg.rank, self.cfg.A
        uv = 2 * n * r
        wa = a * n
        U = theta[: n * r].reshape((n, r))
        V = theta[n * r : uv].reshape((n, r))
        Wa = theta[uv : uv + wa].reshape((a, n))
        ba = theta[uv + wa :].reshape((a))
        return U, V, Wa, ba

    def step(self, h: jnp.ndarray, obs: jnp.ndarray, theta: jnp.ndarray, res_params: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        U, V, Wa, ba = self.split_theta(theta)
        
        # RNN Dynamics
        rec0 = h @ res_params["W0_T"]
        low = (h @ V) @ U.T
        inp = obs @ res_params["Win_T"]
        
        pre = rec0 + low + inp + self.b
        h_next = (1.0 - res_params["leak"]) * h + res_params["leak"] * jnp.tanh(pre)
        
        # Check for explosion/NaN
        exploded = jnp.any(jnp.logical_not(jnp.isfinite(h_next))) | (jnp.max(jnp.abs(h_next)) > self.cfg.h_clip)
        h_next = jnp.where(exploded, jnp.zeros_like(h_next), h_next)
        
        # Action head
        act_pre = h_next @ Wa.T + ba
        action = jnp.tanh(act_pre) * self.action_scale + self.action_bias
        
        return h_next, action

    def batched_rollout(self, h, obs, population_theta, pop_res_params):
        """Vectorized across population and environments."""
        return jax.vmap(
            jax.vmap(self.step, in_axes=(0, 0, None, None)), 
            in_axes=(0, 0, 0, 0)
        )(h, obs, population_theta, pop_res_params)
