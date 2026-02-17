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
        
        # Explicitly place model on GPU (device 0) to avoid any CPU fallback ambiguity
        try:
            device = jax.devices("gpu")[0]
        except:
            device = jax.devices()[0]

        # Put model on device and CLAMP solver iterations to effectively "time-limit" the physics
        # This prevents "bad" states from consuming infinite compute (hanging the kernel)
        mjx_model = mjx.put_model(self.model, device=device)
        self.mjx_model = mjx_model.replace(
            opt=mjx_model.opt.replace(
                iterations=4,     # Newton solver steps (usually converges in 2-5)
                ls_iterations=4   # Line search iterations
            )
        )
        
        self.obs_dim = infer_obs_dim(spec, self.model)
        self.action_dim = self.model.nu
        
        # Precompute constants (ensure they are on device)
        self.body_mass = jax.device_put(jnp.array(self.model.body_mass), device)
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
        # Check for pre-existing NaNs or action NaNs to prevent solver hang
        bad_state = jnp.any(jnp.logical_not(jnp.isfinite(data.qpos))) | \
                    jnp.any(jnp.logical_not(jnp.isfinite(data.qvel))) | \
                    jnp.any(jnp.logical_not(jnp.isfinite(action)))
        
        # If state is bad, we must skip mjx.step entirely to avoid GPU hangs
        # We branch on 'bad_state'
        
        def safe_step(d_in):
            d = d_in.replace(ctrl=action)
            return jax.lax.fori_loop(0, self.spec.frame_skip, lambda _, x: mjx.step(self.mjx_model, x), d)
            
        def unsafe_step(d_in):
            return d_in # Identity op, no physics
            
        # Execute physics only if safe
        data_next = jax.lax.cond(bad_state, unsafe_step, safe_step, data)

        # Compute rewards/obs based on data_next
        x_before = jnp.sum(self.body_mass * data.xipos[:, 0]) / self.total_mass
        x_after = jnp.sum(self.body_mass * data_next.xipos[:, 0]) / self.total_mass
        x_vel = (x_after - x_before) / self.dt
        
        z = data_next.qpos[2]
        is_healthy = (z > self.spec.healthy_z_min) & (z < self.spec.healthy_z_max)
        terminated = (self.spec.terminate_when_unhealthy & (~is_healthy)) | bad_state
        
        # Zero reward if bad_state
        reward = (self.spec.forward_reward_weight * x_vel + 
                  jnp.where(is_healthy, self.spec.healthy_reward, 0.0) -
                  self.spec.ctrl_cost_weight * jnp.sum(jnp.square(action)))
        
        if self.spec.include_contact_cost:
             contact_cost = self.spec.contact_cost_weight * jnp.minimum(jnp.sum(jnp.square(data_next.cfrc_ext)), self.spec.contact_cost_max)
             reward -= contact_cost

        reward = jnp.where(bad_state, 0.0, reward)
        
        return data_next, self.get_obs(data_next), reward, terminated

    def reset_one(self, key: jnp.ndarray) -> Tuple[mjx.Data, jnp.ndarray]:
        k1, k2 = jax.random.split(key)
        noise = self.spec.reset_noise_scale
        qpos = self.init_qpos + jax.random.uniform(k1, (self.model.nq,), minval=-noise, maxval=noise)
        qvel = self.init_qvel + jax.random.uniform(k2, (self.model.nv,), minval=-noise, maxval=noise)
        data = mjx.make_data(self.mjx_model).replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)
        return data, self.get_obs(data)
