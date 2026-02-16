import csv
import json
import os
import hashlib
import time
import datetime
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp

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

def save_config(args: Any, filepath: str) -> None:
    with open(filepath, "w") as f:
        json.dump(vars(args), f, indent=2)

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

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
        if dev.platform != "gpu": return out
        mem = dev.memory_stats()
        if not mem: return out
        total_b = float(mem.get("bytes_limit", 0))
        used_b = float(mem.get("bytes_in_use", 0))
        peak_b = float(mem.get("peak_bytes_in_use", 0))
        out["gpu_total_gb"] = total_b / 1e9
        out["gpu_free_gb"] = (total_b - used_b) / 1e9
        out["gpu_used_frac"] = used_b / max(total_b, 1.0)
        out["gpu_peak_alloc_gb"] = peak_b / 1e9
        out["gpu_peak_reserved_gb"] = peak_b / 1e9
        out["gpu_peak_reserved_frac"] = peak_b / max(total_b, 1.0)
    except:
        pass
    return out

def enable_jax_compilation_cache(cache_dir: str) -> str:
    cache_dir_abs = os.path.abspath(cache_dir)
    os.makedirs(cache_dir_abs, exist_ok=True)
    try:
        from jax.experimental import compilation_cache as cc
        cc.set_cache_dir(cache_dir_abs)
    except:
        try:
            from jax.experimental.compilation_cache import compilation_cache as cc
            cc.set_cache_dir(cache_dir_abs)
        except Exception as e:
            print(f"[compile] warning: failed to enable JAX cache: {e}")
            return cache_dir_abs
    print(f"[compile] persistent JAX cache: {cache_dir_abs}")
    return cache_dir_abs

def build_compile_key(args: Any, env_id: str, xml_path: str, obs_dim: int, action_dim: int) -> str:
    dev = jax.devices()[0]
    payload = {
        "env_id": env_id,
        "xml_sha256": file_sha256(xml_path),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "hidden": int(args.hidden),
        "rank": int(args.rank),
        "episodes_per_candidate": int(args.episodes_per_candidate),
        "rollout_steps": int(args.rollout_steps),
        "platform": str(dev.platform),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

def save_checkpoint(path: str, iteration: int, theta: jnp.ndarray, args: Any, extra: Dict[str, Any]) -> None:
    payload = {
        "iter": int(iteration),
        "theta": np.array(theta),
        "args": vars(args),
        "extra": extra,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
