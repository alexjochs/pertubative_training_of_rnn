from dataclasses import dataclass
import os
from typing import Tuple, List
import numpy as np
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp

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

def resolve_env_spec(candidates: str) -> Tuple[str, HumanoidSpec]:
    for env_id in candidates.split(","):
        env_id = env_id.strip()
        name = env_id.lower()
        if "v5" in name: return env_id, HumanoidSpec(obs_variant="v5", include_contact_cost=True)
        if "v4" in name: return env_id, HumanoidSpec(obs_variant="v4", include_contact_cost=False)
        if "v3" in name or "v2" in name or name == "humanoid":
            return env_id, HumanoidSpec(obs_variant="v4", include_contact_cost=True)
    raise RuntimeError(f"Unknown Humanoid variant: {candidates}")

def resolve_humanoid_xml(xml_path: str) -> str:
    candidates = [xml_path, os.environ.get("HUMANOID_XML", ""), "humanoid/humanoid.xml", "humanoid.xml"]
    for p in candidates:
        if p and os.path.exists(p): return os.path.abspath(p)
    raise RuntimeError("Could not locate humanoid.xml")

def infer_obs_dim(spec: HumanoidSpec, model: mujoco.MjModel) -> int:
    nq, nv, nbody, nu = model.nq, model.nv, model.nbody, model.nu
    if spec.obs_variant == "v5":
        return (nq - 2) + nv + (nbody - 1) * 10 + (nbody - 1) * 6 + nu + (nbody - 1) * 6
    return (nq - 2) + nv + nbody * 10 + nbody * 6 + nv + nbody * 6

class MJXHumanoidEnv:
    def __init__(self, xml_path: str, spec: HumanoidSpec):
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.mjx_model = mjx.put_model(self.model)
        self.obs_dim = infer_obs_dim(spec, self.model)
        self.action_dim = self.model.nu
        
        # Precompute constants
        self.body_mass = jnp.array(self.model.body_mass)
        self.total_mass = jnp.sum(self.body_mass)
        self.dt = self.model.opt.timestep * spec.frame_skip
        
        self.init_qpos = jnp.array(self.model.key_qpos[0] if self.model.nkey > 0 else self.model.qpos0)
        self.init_qvel = jnp.zeros(self.model.nv)

    def get_obs(self, data: mjx.Data) -> jnp.ndarray:
        position = data.qpos[2:]
        velocity = data.qvel
        if self.spec.obs_variant == "v5":
            com_inertia = data.cinert[1:].ravel()
            com_velocity = data.cvel[1:].ravel()
            external_contact_forces = data.cfrc_ext[1:].ravel()
            actuator_forces = data.qfrc_actuator[(-self.action_dim):]
        else:
            com_inertia = data.cinert.ravel()
            com_velocity = data.cvel.ravel()
            external_contact_forces = data.cfrc_ext.ravel()
            actuator_forces = data.qfrc_actuator
        return jnp.concatenate([position, velocity, com_inertia, com_velocity, actuator_forces, external_contact_forces])

    def step(self, data: mjx.Data, action: jnp.ndarray) -> Tuple[mjx.Data, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x_before = jnp.sum(self.body_mass * data.xipos[:, 0]) / self.total_mass
        data = data.replace(ctrl=action)
        data = jax.lax.fori_loop(0, self.spec.frame_skip, lambda _, d: mjx.step(self.mjx_model, d), data)
        x_after = jnp.sum(self.body_mass * data.xipos[:, 0]) / self.total_mass
        
        x_vel = (x_after - x_before) / self.dt
        z = data.qpos[2]
        
        is_healthy = (z > self.spec.healthy_z_min) & (z < self.spec.healthy_z_max)
        terminated = self.spec.terminate_when_unhealthy & (~is_healthy)
        
        reward = (self.spec.forward_reward_weight * x_vel + 
                  jnp.where(is_healthy, self.spec.healthy_reward, 0.0) -
                  self.spec.ctrl_cost_weight * jnp.sum(jnp.square(action)))
        
        if self.spec.include_contact_cost:
            contact_cost = self.spec.contact_cost_weight * jnp.minimum(jnp.sum(jnp.square(data.cfrc_ext)), self.spec.contact_cost_max)
            reward -= contact_cost
            
        return data, self.get_obs(data), reward, terminated

    def reset_one(self, key: jnp.ndarray) -> Tuple[mjx.Data, jnp.ndarray]:
        k1, k2 = jax.random.split(key)
        noise = self.spec.reset_noise_scale
        qpos = self.init_qpos + jax.random.uniform(k1, (self.model.nq,), minval=-noise, maxval=noise)
        qvel = self.init_qvel + jax.random.uniform(k2, (self.model.nv,), minval=-noise, maxval=noise)
        data = mjx.make_data(self.mjx_model).replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)
        return data, self.get_obs(data)
