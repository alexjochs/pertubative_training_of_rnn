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

class ReservoirPolicy:
    def __init__(self, cfg: ReservoirConfig, action_range: Tuple[jnp.ndarray, jnp.ndarray]):
        self.cfg = cfg
        self.low, self.high = action_range
        self.action_scale = 0.5 * (self.high - self.low)
        self.action_bias = 0.5 * (self.high + self.low)
        
        # Initialize fixed parameters
        key = jax.random.PRNGKey(1234)
        k1, k2 = jax.random.split(key)
        self.Win_T = (jax.random.normal(k1, (cfg.N, cfg.D)) * cfg.win_std).T
        self.b = jnp.zeros(cfg.N)
        self.W0_T = self._make_sparse_w0(cfg.N, cfg.k_in, cfg.w0_std).T

    @property
    def theta_dim(self) -> int:
        n, r, a = self.cfg.N, self.cfg.rank, self.cfg.A
        return 2 * n * r + a * n + a

    def _make_sparse_w0(self, N, k_in, w_std):
        key = jax.random.PRNGKey(1234)
        scale = w_std / (k_in**0.5)
        # Simple dense representation for now, as MJX is dense anyway
        # For true sparsity, we'd use jax.experimental.sparse but dense is faster on GPU for N=1024
        W0 = jax.random.normal(key, (N, N)) * (w_std / (N**0.5))
        # Mask to k_in connections per neuron
        mask = jax.random.uniform(key, (N, N)) < (k_in / N)
        return W0 * mask

    def split_theta(self, theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        n, r, a = self.cfg.N, self.cfg.rank, self.cfg.A
        uv = 2 * n * r
        wa = a * n
        U = theta[: n * r].reshape((n, r))
        V = theta[n * r : uv].reshape((n, r))
        Wa = theta[uv : uv + wa].reshape((a, n))
        ba = theta[uv + wa :].reshape((a))
        return U, V, Wa, ba

    def step(self, h: jnp.ndarray, obs: jnp.ndarray, theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        U, V, Wa, ba = self.split_theta(theta)
        
        # RNN Dynamics
        rec0 = h @ self.W0_T
        low = (h @ V) @ U.T
        inp = obs @ self.Win_T
        
        pre = rec0 + low + inp + self.b
        h_next = (1.0 - self.cfg.leak) * h + self.cfg.leak * jnp.tanh(pre)
        
        # Check for explosion/NaN
        exploded = jnp.any(jnp.logical_not(jnp.isfinite(h_next))) | (jnp.max(jnp.abs(h_next)) > self.cfg.h_clip)
        h_next = jnp.where(exploded, jnp.zeros_like(h_next), h_next)
        
        # Action head
        act_pre = h_next @ Wa.T + ba
        action = jnp.tanh(act_pre) * self.action_scale + self.action_bias
        
        return h_next, action

    def batched_rollout(self, h, obs, population_theta):
        """Vectorized across population and environments."""
        return jax.vmap(jax.vmap(self.step, in_axes=(0, 0, None)), in_axes=(0, 0, 0))(h, obs, population_theta)
