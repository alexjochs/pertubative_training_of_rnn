#!/usr/bin/env python3
"""
Clean, minimal ES training loop for:
Moving MNIST -> frozen CNN encoder -> sparse RNN reservoir with low-rank adapter -> linear decoder
Train adapter (U,V) with antithetic ES across multiple GPUs.
Train decoder (Wout) with standard Adam on GPU0 (few steps/iter) using the current (unperturbed) adapter.

Tested design assumptions:
- Single multi-GPU node (8xH200 target)
- Uses torch.multiprocessing spawn; no torch.distributed required
- Workers sync by receiving *indices* into the standard MovingMNIST dataset
"""

from __future__ import annotations

import math

import time
import math

import time
import os
import argparse
import json
import csv
import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision
from torchvision import transforms


# -----------------------------
# Frozen CNN encoder
# -----------------------------
class PretrainedResNet18Encoder(nn.Module):
    """
    ResNet18 (pretrained on ImageNet) adapted for:
    1. 1-channel input (summing RGB weights of first conv).
    2. 128-d output embedding (replacing fc layer).
    """
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        # Load pretrained resnet
        resnet = torchvision.models.resnet18(weights="DEFAULT")
        
        # 1. Adapt first conv to 1 channel
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Sum the weights across the RGB channel dimension to preserve activations
        # old_weight: [64, 3, 7, 7] -> sum(1, keepdim=True) -> [64, 1, 7, 7]
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
        
        resnet.conv1 = new_conv
        
        # 2. Remove classification head (fc) and replace with projection
        # ResNet18 penultimate features are 512-d
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Output: [B, 512, 1, 1]
        self.projection = nn.Linear(512, emb_dim)
        
        # Freeze everything initially
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.projection.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,64,64]
        h = self.backbone(x)           # [B, 512, 1, 1]
        h = h.view(h.size(0), -1)      # [B, 512]
        z = self.projection(h)         # [B, 128]
        return z


# -----------------------------
# Sparse reservoir + low-rank adapter
# -----------------------------
@dataclass
class ReservoirConfig:
    N: int = 2000
    D: int = 128
    rank: int = 32
    leak: float = 0.3
    h_clip: float = 50.0
    k_in: int = 50
    w0_std: float = 1.0
    win_std: float = 0.2


class Reservoir(nn.Module):
    """
    h_{t+1} = (1-leak) h_t + leak * tanh( (W0 + U V^T) h_t + Win z_t + b )
    W0 sparse COO, Win dense.
    """
    def __init__(self, cfg: ReservoirConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Dense input projection
        self.Win = nn.Parameter(torch.randn(cfg.N, cfg.D, device=device) * cfg.win_std, requires_grad=False)
        self.b = nn.Parameter(torch.zeros(cfg.N, device=device), requires_grad=False)

        # Low-rank adapter parameters (train via ES externally)
        self.U = nn.Parameter(torch.zeros(cfg.N, cfg.rank, device=device), requires_grad=False)
        self.V = nn.Parameter(torch.zeros(cfg.N, cfg.rank, device=device), requires_grad=False)

        # Sparse recurrent base matrix W0
        self.W0 = self._make_sparse_w0(cfg.N, cfg.k_in, device=device, w_std=cfg.w0_std)

    @staticmethod
    def _make_sparse_w0(N: int, k_in: int, device: torch.device, w_std: float) -> torch.Tensor:
        """
        Fixed in-degree approximation of Erdos–Renyi:
        For each row i, choose k_in random cols.
        """
        g = torch.Generator(device="cpu")
        g.manual_seed(1234)

        rows = []
        cols = []
        vals = []

        scale = (w_std / math.sqrt(k_in))
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
        W0 = W0.coalesce()
        return W0

    def set_adapter_from_theta(self, theta: torch.Tensor) -> None:
        """
        theta: 1D CPU or GPU tensor containing [U.flatten(), V.flatten()]
        """
        N, r = self.cfg.N, self.cfg.rank
        assert theta.numel() == 2 * N * r
        th = theta.to(self.device, non_blocking=True)
        U = th[: N * r].view(N, r)
        V = th[N * r :].view(N, r)
        self.U.data.copy_(U)
        self.V.data.copy_(V)

    def step(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Single recurrent step:
        h_{t+1} = (1-alpha)*h_t + alpha * tanh( W0*h_t + U*V^T*h_t + Win*z_t + b )
        
        Args:
            h: Hidden state [B, N]
            z: Input embedding [B, D]
        """
        # Recurrent sparse: (W0 @ h^T)^T
        rec0 = torch.sparse.mm(self.W0, h.t()).t()  # [B,N]

        # Low-rank: (U V^T) h = U (V^T h)
        # h @ V -> [B,r], then @ U^T -> [B,N]
        low = (h @ self.V) @ self.U.t()

        inp = z @ self.Win.t()  # [B,N]
        pre = rec0 + low + inp + self.b

        nh = torch.tanh(pre)
        h_next = (1.0 - self.cfg.leak) * h + self.cfg.leak * nh

        return h_next

    @torch.no_grad()
    def rollout_loss(self, z_seq: torch.Tensor, decoder: nn.Linear, warmup: int) -> torch.Tensor:
        """
        Autoregressive rollout loss.
        z_seq: [B,T,D] on device
        warmup: Number of steps to strict teacher forcing.
        
        Steps 0..warmup-1: Feed z_t, update h. No loss.
        Steps warmup..T-1: Predict z_{t}, feed BACK into network as z_t for next step. Calculate loss match.
        """
        B, T, D = z_seq.shape
        N = self.cfg.N
        h = torch.zeros((B, N), device=self.device, dtype=torch.float32)

        losses = []
        
        # Current input for the step
        # Initially, it's the first frame from the sequence
        z_in = z_seq[:, 0] # t=0

        for t in range(T - 1):
            # Update state with current input
            h = self.step(h, z_in)
            
            # Predict next frame embedding
            pred = decoder(h) # prediction for t+1
            
            # Target is always the true next frame
            target = z_seq[:, t + 1]

            if t < warmup:
                # WARMUP PHASE: Teacher Forcing.
                # Next input is the GROUND TRUTH for t+1
                z_in = target
            else:
                # ROLLOUT PHASE: Autoregressive.
                # Next input is the PREDICTION we just made
                z_in = pred
                
                # We calculate loss only during the rollout phase
                losses.append(F.mse_loss(pred, target, reduction="mean"))

            # early terminate explosion
            if not torch.isfinite(h).all() or h.abs().max() > self.cfg.h_clip:
                return torch.tensor(1e6, device=self.device)

        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device)
            
        return torch.stack(losses).mean()


# -----------------------------
# Worker process for candidate evaluation
# -----------------------------
@dataclass
class WorkerJob:
    indices: torch.Tensor  # Indices into the dataset [B]
    theta: torch.Tensor    # CPU float32
    decoder_w: torch.Tensor # CPU float32, [D,N]
    decoder_b: torch.Tensor # CPU float32, [D]
    T: int
    warmup: int


def worker_main(rank: int, args: argparse.Namespace, inq: mp.Queue, outq: mp.Queue) -> None:
    # Handle CPU-only case
    if args.gpus == 0:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

    # Standard Moving MNIST
    # The dataset typically returns [T, 1, 64, 64] images in range [0, 255] or [0, 1] depending on version.
    # torchvision MovingMNIST:
    # "The data is given in the form of a tensor of shape [20, 1, 64, 64]" per item?
    # Inspecting typical usage: It is often [20, 1, 64, 64] byte tensor.
    # We will need to check transform.
    transform = transforms.Lambda(lambda x: x.float() / 255.0)
    mm = torchvision.datasets.MovingMNIST(
        root=args.data_root,
        split=None,
        download=True,
        transform=transform
    )

    enc = PretrainedResNet18Encoder(emb_dim=args.emb_dim).to(device)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    cfg = ReservoirConfig(
        N=args.hidden,
        D=args.emb_dim,
        rank=args.rank,
        leak=args.leak,
        h_clip=args.h_clip,
        k_in=args.k_in,
        w0_std=args.w0_std,
        win_std=args.win_std,
    )
    res = Reservoir(cfg, device=device).to(device)
    res.eval()

    # Decoder lives in worker only for scoring (weights copied from master per job)
    decoder = nn.Linear(cfg.N, cfg.D, bias=True).to(device)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    while True:
        job = inq.get()
        if job is None:
            break
        
        # Set adapter + decoder
        res.set_adapter_from_theta(job.theta)

        decoder.weight.data.copy_(job.decoder_w.to(device, non_blocking=True))
        decoder.bias.data.copy_(job.decoder_b.to(device, non_blocking=True))

        # Load batch from indices
        # mm[i] returns a tuple/tensor. torchvision MovingMNIST __getitem__ returns tensor [T, 1, 64, 64]
        # We need to stack them -> [B, T, 1, 64, 64]
        indices = job.indices.tolist()
        batch_list = [mm[i] for i in indices] 
        frames = torch.stack(batch_list).to(device, non_blocking=True) # [B, T, 1, 64, 64]
        
        # If T is specified and less than dataset T, slice it
        if job.T < frames.shape[1]:
            frames = frames[:, :job.T]

        # Encode
        B, T, _, H, W = frames.shape
        x = frames.reshape(B * T, 1, H, W)
        
        with torch.no_grad():
            z = enc(x).view(B, T, -1)  # [B,T,D]
            loss = res.rollout_loss(z, decoder, warmup=job.warmup).detach().float().cpu()

        outq.put(loss.item())


# -----------------------------
# Master ES loop
# -----------------------------
def make_decoder(N: int, D: int, device: torch.device) -> nn.Linear:
    dec = nn.Linear(N, D, bias=True).to(device)
    return dec


def es_rank_weights(losses: torch.Tensor) -> torch.Tensor:
    M = losses.numel()
    order = torch.argsort(losses)  # ascending (best first)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(M, dtype=torch.float32)
    w = -(ranks / (M - 1) - 0.5)
    w = w - w.mean()
    return w


class CSVLogger:
    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.file = open(filepath, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.file.flush()

    def log(self, data):
        self.writer.writerow(data)
        self.file.flush()

    def close(self):
        self.file.close()


def save_config(args: argparse.Namespace, filepath: str):
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)


def visualize_batch(z_seq: torch.Tensor, decoder: nn.Linear, res: Reservoir, warmup: int, save_path: str):
    """
    Runs a forward pass on the FIRST item in the batch and saves a comparison heatmap.
    """
    plt.switch_backend('Agg')
    
    # Take first element: [T, D]
    z_gt = z_seq[0].detach().cpu()
    T, D = z_gt.shape
    
    # Run model in inference mode
    h = torch.zeros((1, res.cfg.N), device=res.device)
    z_pred_list = []
    
    # Initial input
    z_in = z_gt[0:1].to(res.device) # [1,D]
    
    with torch.no_grad():
        for t in range(T - 1):
            h = res.step(h, z_in)
            pred = decoder(h) # [1,D]
            z_pred_list.append(pred.cpu())
            
            if t < warmup:
                z_in = z_gt[t+1:t+2].to(res.device) # Teacher forcing
            else:
                z_in = pred # Autoregressive
    
    # Stack predictions: [T-1, D]
    z_pred = torch.cat(z_pred_list, dim=0)
    
    # GT for t=1..T
    z_target = z_gt[1:]
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # GT Heatmap
    im0 = axes[0].imshow(z_target.T, aspect='auto', cmap='viridis')
    axes[0].set_title("Ground Truth Embedding (z_t)")
    axes[0].set_ylabel("Dimension")
    plt.colorbar(im0, ax=axes[0])
    
    # Pred Heatmap
    im1 = axes[1].imshow(z_pred.T, aspect='auto', cmap='viridis')
    axes[1].set_title(f"Predicted Embedding (Warmup={warmup})")
    axes[1].set_ylabel("Dimension")
    axes[1].set_xlabel("Time Step (t)")
    
    # Draw warmup line
    axes[1].axvline(x=warmup, color='red', linestyle='--', linewidth=2, label='End of Warmup')
    axes[1].legend()
    
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--gpus", type=int, default=8, help="Number of GPUs. Set to 0 for CPU mode.")
    ap.add_argument("--hidden", type=int, default=2000)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--k_in", type=int, default=50)

    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=15, help="Number of steps for teacher-forced warmup before autoregressive rollout")
    ap.add_argument("--batch", type=int, default=16)
    
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--pairs", type=int, default=32, help="number of antithetic pairs per ES iter")
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--theta_lr", type=float, default=0.02)
    ap.add_argument("--dec_lr", type=float, default=1e-3)
    ap.add_argument("--dec_steps", type=int, default=1, help="decoder Adam steps per iter on unperturbed theta")

    ap.add_argument("--leak", type=float, default=0.3)
    ap.add_argument("--h_clip", type=float, default=50.0)
    ap.add_argument("--w0_std", type=float, default=1.0)
    ap.add_argument("--win_std", type=float, default=0.2)

    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Logging to {results_dir}")

    # Save config
    save_config(args, os.path.join(results_dir, "config.json"))

    # Init CSV Logger
    csv_logger = CSVLogger(
        os.path.join(results_dir, "train_log.csv"),
        fieldnames=["iter", "base_loss", "min_loss", "med_loss", "bad_frac", "theta_norm", "h_norm", "elapsed_min"]
    )

    # Manual Seed
    torch.manual_seed(args.seed)
    
    # Device handling
    if args.gpus == 0:
        dev0 = torch.device("cpu")
        print("Running in CPU mode (0 GPUs specified)")
    else:
        assert torch.cuda.is_available()
        torch.cuda.set_device(0)
        dev0 = torch.device("cuda:0")

    # Load master dataset to get length and handle download
    transform = transforms.Lambda(lambda x: x.float() / 255.0)
    mm0 = torchvision.datasets.MovingMNIST(
        root=args.data_root,
        split=None,
        download=True,
        transform=transform
    )
    dataset_len = len(mm0)
    print(f"Dataset loaded. Size: {dataset_len}")

    # Workers
    ctx = mp.get_context("spawn")
    inqs: List[mp.Queue] = [ctx.Queue(maxsize=8) for _ in range(max(1, args.gpus))]
    outqs: List[mp.Queue] = [ctx.Queue(maxsize=8) for _ in range(max(1, args.gpus))]
    procs: List[mp.Process] = []

    # If 0 GPUs, we verify logic with 1 worker on CPU
    num_workers = args.gpus if args.gpus > 0 else 1
    
    for r in range(num_workers):
        p = ctx.Process(target=worker_main, args=(r, args, inqs[r], outqs[r]))
        p.daemon = True
        p.start()
        procs.append(p)

    # Master components
    N, D, rnk = args.hidden, args.emb_dim, args.rank
    theta_dim = 2 * N * rnk

    torch.manual_seed(args.seed)
    theta = torch.zeros(theta_dim, dtype=torch.float32)
    theta_opt = torch.optim.Adam([theta], lr=args.theta_lr)

    decoder = make_decoder(N, D, device=dev0)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=args.dec_lr)

    enc0 = PretrainedResNet18Encoder(emb_dim=D).to(dev0)
    enc0.eval()
    for p in enc0.parameters():
        p.requires_grad_(False)

    cfg0 = ReservoirConfig(
        N=N, D=D, rank=rnk, leak=args.leak, h_clip=args.h_clip,
        k_in=args.k_in, w0_std=args.w0_std, win_std=args.win_std
    )
    res0 = Reservoir(cfg0, device=dev0).to(dev0)
    res0.eval()

    def get_batch_indices(seed: int, batch_size: int) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(seed)
        return torch.randint(0, dataset_len, (batch_size,), generator=g)

    def train_decoder_on_batch(seed: int) -> float:
        indices = get_batch_indices(seed, args.batch)
        batch_list = [mm0[i] for i in indices]
        frames = torch.stack(batch_list).to(dev0, non_blocking=True) 
        
        if args.T < frames.shape[1]:
            frames = frames[:, :args.T]
            
        B, T, _, H, W = frames.shape
        x = frames.reshape(B * T, 1, H, W)
        with torch.no_grad():
            z = enc0(x).view(B, T, -1)

        res0.set_adapter_from_theta(theta)

        decoder.train()
        for _ in range(args.dec_steps):
            dec_opt.zero_grad(set_to_none=True)
            h = torch.zeros((B, N), device=dev0)
            loss_steps = []
            for t in range(T - 1):
                zt = z[:, t]
                with torch.no_grad():
                    h = res0.step(h, zt)
                    if not torch.isfinite(h).all() or h.abs().max() > args.h_clip:
                        return 1e6
                pred = decoder(h)
                loss_steps.append(F.mse_loss(pred, z[:, t + 1], reduction="mean"))
            loss = torch.stack(loss_steps).mean()
            loss.backward()
            dec_opt.step()

        decoder.eval()
        return float(loss.detach().cpu().item())

    # ES loop
    base_seed = 10_000_000 + args.seed * 1_000_000
    t0 = time.time()

    try:
        for it in range(1, args.iters + 1):
            seed_it = base_seed + it

            # 1) Train decoder on unperturbed theta
            base_loss = train_decoder_on_batch(seed_it)

            dec_w_cpu = decoder.weight.detach().float().cpu().contiguous()
            dec_b_cpu = decoder.bias.detach().float().cpu().contiguous()

            # 2) Sample antithetic pairs
            K = args.pairs
            eps = torch.randn((K, theta_dim), dtype=torch.float32)
            sigma = args.sigma

            losses = torch.empty((2 * K,), dtype=torch.float32)
            jobs: List[Tuple[int, torch.Tensor]] = []
            for i in range(K):
                jobs.append((+1, eps[i]))
                jobs.append((-1, eps[i]))

            # Dispatch
            # SAME Indices for all candidates in this iteration
            indices = get_batch_indices(seed_it, args.batch)

            for j, (sgn, e) in enumerate(jobs):
                cand = theta + (sgn * sigma) * e
                job = WorkerJob(
                    indices=indices,
                    theta=cand,
                    decoder_w=dec_w_cpu,
                    decoder_b=dec_b_cpu,
                    T=args.T,
                    warmup=args.warmup,
                )
                inqs[j % num_workers].put(job)

            # Collect
            for j in range(len(jobs)):
                loss_j = outqs[j % num_workers].get()
                losses[j] = float(loss_j)

            # 3) Update
            w = es_rank_weights(losses)
            signed_eps = torch.empty((2 * K, theta_dim), dtype=torch.float32)
            for i in range(K):
                signed_eps[2 * i + 0] = +eps[i]
                signed_eps[2 * i + 1] = -eps[i]

            g_hat = (w.view(-1, 1) * signed_eps).mean(dim=0) / max(sigma, 1e-8)

            theta_opt.zero_grad(set_to_none=True)
            theta.grad = g_hat
            theta_opt.step()

            if it % args.log_every == 0:
                frac_bad = float((losses >= 1e5).float().mean().item())
                dt = time.time() - t0
                
                # Get h_norm (approximation from one batch on master)
                # We'll run a quick forward pass on a small batch for logging/viz
                viz_indices = get_batch_indices(seed_it, 1) # Just 1 for viz
                viz_batch = [mm0[i] for i in viz_indices]
                viz_frames = torch.stack(viz_batch).to(dev0, non_blocking=True)
                if args.T < viz_frames.shape[1]: viz_frames = viz_frames[:, :args.T]
                with torch.no_grad():
                    # Encode
                    B_v, T_v, _, H_v, W_v = viz_frames.shape
                    viz_x = viz_frames.reshape(B_v * T_v, 1, H_v, W_v)
                    viz_z = enc0(viz_x).view(B_v, T_v, -1)
                    
                    # Update adapter
                    res0.set_adapter_from_theta(theta)
                    
                    # Run viz + tracking
                    # Just calculate h_norm of the final state
                    h_temp = torch.zeros((1, N), device=dev0)
                    for t_ in range(args.T-1):
                        h_temp = res0.step(h_temp, viz_z[:, t_])
                    h_norm = h_temp.norm().item()
                    
                    # Save Heatmap
                    viz_path = os.path.join(results_dir, f"viz_iter_{it:04d}.png")
                    visualize_batch(viz_z, decoder, res0, args.warmup, viz_path)

                print(
                    f"iter {it:6d}  base_loss {base_loss:10.6f}  "
                    f"cand_loss[min/med] {losses.min().item():.6f}/{losses.median().item():.6f}  "
                    f"bad_frac {frac_bad:.2f}  "
                    f"|theta| {theta.norm().item():.3f}  "
                    f"|h| {h_norm:.2f}  "
                    f"elapsed {dt/60:.1f}m",
                    flush=True
                )
                
                # Log to CSV
                csv_logger.log({
                    "iter": it,
                    "base_loss": base_loss,
                    "min_loss": losses.min().item(),
                    "med_loss": losses.median().item(),
                    "bad_frac": frac_bad,
                    "theta_norm": theta.norm().item(),
                    "h_norm": h_norm,
                    "elapsed_min": dt/60
                })
                
                # Save Checkpoint
                ckpt_path = os.path.join(results_dir, "checkpoint_latest.pt")
                torch.save({
                    "iter": it,
                    "theta": theta,
                    "decoder": decoder.state_dict(),
                    "theta_opt": theta_opt.state_dict(),
                    "dec_opt": dec_opt.state_dict(),
                    "args": vars(args)
                }, ckpt_path)

    finally:
        for q in inqs:
            q.put(None)
        for p in procs:
            p.join(timeout=5)


if __name__ == "__main__":
    main()