#!/usr/bin/env python3
"""
Perturbative Trained RNN (Single-Process GPU Optimized)

Optimizations:
1. Pre-compute embeddings for the entire dataset (Encoder is frozen).
2. Remove multiprocessing overhead.
3. Run ES loop on GPU.
"""

from __future__ import annotations

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
import torchvision
from torchvision import transforms


@dataclass
class ReservoirConfig:
    N: int = 2000          # Reservoir size
    D: int = 128           # Input embedding dimension
    rank: int = 32         # Rank of the learned perturbation (U, V)
    leak: float = 0.1      # Leak rate (alpha)
    h_clip: float = 50.0   # Clip value for hidden state
    k_in: int = 10         # Sparsity of W0
    w0_std: float = 1.0    # Spectral radius controls
    win_std: float = 0.2   # Input scaling


class PretrainedResNet18Encoder(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        resnet = torchvision.models.resnet18(weights="DEFAULT")
        
        # 1. Adapt first conv to 1 channel
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        resnet.conv1 = new_conv
        
        # 2. Remove classification head (fc) and replace with projection
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Output: [B, 512, 1, 1]
        self.projection = nn.Linear(512, emb_dim)
        
        # Freeze everything
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.projection.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 64, 64]
        feat = self.backbone(x)      # [B, 512, 1, 1]
        feat = feat.view(feat.size(0), -1) # [B, 512]
        out = self.projection(feat)  # [B, D]
        return out


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
        print(f"  [Reservoir] Generating sparse W0 ({N}x{N})...", flush=True)
        t0 = time.time()
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
        dt = time.time() - t0
        print(f"  [Reservoir] W0 generated in {dt:.2f}s.", flush=True)
        return W0.to_dense()

    def set_adapter_from_theta(self, theta: torch.Tensor) -> None:
        N, r = self.cfg.N, self.cfg.rank
        assert theta.numel() == 2 * N * r
        th = theta.to(self.device, non_blocking=True)
        U = th[: N * r].view(N, r)
        V = th[N * r :].view(N, r)
        self.U.data.copy_(U)
        self.V.data.copy_(V)

    def step(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Recurrent dense: h @ W0.t()
        # W0 is [N, N]. h is [B, N].
        rec0 = h @ self.W0.t()

        # Low-rank: (U V^T) h = U (V^T h)
        low = (h @ self.V) @ self.U.t()

        inp = z @ self.Win.t()  # [B,N]
        pre = rec0 + low + inp + self.b

        nh = torch.tanh(pre)
        h_next = (1.0 - self.cfg.leak) * h + self.cfg.leak * nh

        return h_next

    @torch.no_grad()
    def rollout_loss(self, z_seq: torch.Tensor, decoder: nn.Linear, warmup: int) -> torch.Tensor:
        B, T, D = z_seq.shape
        N = self.cfg.N
        h = torch.zeros((B, N), device=self.device, dtype=torch.float32)

        losses = []
        z_in = z_seq[:, 0] # t=0

        for t in range(T - 1):
            h = self.step(h, z_in)
            pred = decoder(h)
            target = z_seq[:, t + 1]

            if t < warmup:
                z_in = target
            else:
                z_in = pred
                losses.append(F.mse_loss(pred, target, reduction="mean"))

            if not torch.isfinite(h).all() or h.abs().max() > self.cfg.h_clip:
                return torch.tensor(1e6, device=self.device)

        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device)
            
        return torch.stack(losses).mean()


def make_decoder(N: int, D: int, device: torch.device) -> nn.Linear:
    decoder = nn.Linear(N, D, bias=True).to(device)
    # Using Xavier init
    nn.init.xavier_normal_(decoder.weight)
    nn.init.zeros_(decoder.bias)
    return decoder


def rank_transform(raw_losses: torch.Tensor) -> torch.Tensor:
    M = len(raw_losses)
    ranks = torch.empty_like(raw_losses)
    order = torch.argsort(raw_losses)
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
    plt.switch_backend('Agg')
    z_gt = z_seq[0].detach().cpu()
    T, D = z_gt.shape
    
    h = torch.zeros((1, res.cfg.N), device=res.device)
    z_pred_list = []
    z_in = z_gt[0:1].to(res.device)
    
    with torch.no_grad():
        for t in range(T - 1):
            h = res.step(h, z_in)
            pred = decoder(h)
            z_pred_list.append(pred.cpu())
            if t < warmup:
                z_in = z_gt[t+1:t+2].to(res.device)
            else:
                z_in = pred
    
    z_pred = torch.cat(z_pred_list, dim=0)
    z_target = z_gt[1:]
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    im0 = axes[0].imshow(z_target.T, aspect='auto', cmap='viridis')
    axes[0].set_title("Ground Truth Embedding (z_t)")
    axes[0].set_ylabel("Dimension")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(z_pred.T, aspect='auto', cmap='viridis')
    axes[1].set_title(f"Predicted Embedding (Warmup={warmup})")
    axes[1].set_ylabel("Dimension")
    axes[1].set_xlabel("Time Step (t)")
    axes[1].axvline(x=warmup, color='red', linestyle='--', linewidth=2, label='End of Warmup')
    axes[1].legend()
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def transform_cifar_norm(x):
    return x.float() / 255.0

def precompute_embeddings(args, dev0):
    print("Pre-computing embeddings for the entire dataset...", flush=True)
    t0 = time.time()
    
    # transform = transforms.Lambda(lambda x: x.float() / 255.0)
    mm0 = torchvision.datasets.MovingMNIST(
        root=args.data_root,
        split=None,
        download=True,
        transform=transform_cifar_norm
    )
    
    # Create loader
    # Using a large batch size for inference speed
    loader = torch.utils.data.DataLoader(mm0, batch_size=64, shuffle=False, num_workers=4)
    encoder = PretrainedResNet18Encoder(emb_dim=args.emb_dim).to(dev0)
    encoder.eval()

    all_embeddings = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # batch: [B, T, 1, 64, 64]
            if args.T < batch.shape[1]:
                batch = batch[:, :args.T]
            
            batch = batch.to(dev0)
            B, T, _, H, W = batch.shape
            x = batch.reshape(B * T, 1, H, W)
            z = encoder(x).view(B, T, -1) # [B, T, D]
            all_embeddings.append(z.cpu()) # Store on CPU to avoid OOM if huge, or GPU if small?
            if i % 10 == 0:
                print(f"  Encoded batch {i}/{len(loader)}", flush=True)

    # Concatenate all
    full_z = torch.cat(all_embeddings, dim=0) # [N_total, T, D]
    dt = time.time() - t0
    print(f"Pre-computation complete. Shape: {full_z.shape}. Time: {dt:.2f}s", flush=True)
    
    # Move to GPU if it fits? 
    # 10000 * 20 * 128 * 4 bytes = ~100 MB. Fits easily.
    return full_z.to(dev0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--gpus", type=int, default=1, help="Ignored in single process mode, kept for compat")
    ap.add_argument("--hidden", type=int, default=2000)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--k_in", type=int, default=50)

    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--theta_lr", type=float, default=0.01)
    
    ap.add_argument("--dec_lr", type=float, default=0.001)
    ap.add_argument("--dec_steps", type=int, default=1)

    ap.add_argument("--leak", type=float, default=0.1)
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
    save_config(args, os.path.join(results_dir, "config.json"))

    csv_logger = CSVLogger(
        os.path.join(results_dir, "train_log.csv"),
        fieldnames=["iter", "base_loss", "min_loss", "med_loss", "bad_frac", "theta_norm", "h_norm", "elapsed_min"]
    )

    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        dev0 = torch.device("cuda:0")
    else:
        dev0 = torch.device("cpu")
    print(f"Device: {dev0}")

    # 1. Pre-compute embeddings
    dataset_z = precompute_embeddings(args, dev0)
    dataset_len = len(dataset_z)

    # 2. Setup Models
    N, D, rnk = args.hidden, args.emb_dim, args.rank
    theta_dim = 2 * N * rnk

    theta = torch.zeros(theta_dim, dtype=torch.float32, device=dev0) # Keep theta on GPU now
    # theta_opt = torch.optim.Adam([theta], lr=args.theta_lr) # We do manual update for ES usually, or use Adam?
    # Original code used Adam for Theta.
    theta_opt = torch.optim.Adam([theta], lr=args.theta_lr)

    decoder = make_decoder(N, D, device=dev0)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=args.dec_lr)

    cfg0 = ReservoirConfig(
        N=N, D=D, rank=rnk, leak=args.leak, h_clip=args.h_clip,
        k_in=args.k_in, w0_std=args.w0_std, win_std=args.win_std
    )
    res0 = Reservoir(cfg0, device=dev0).to(dev0)
    res0.eval()

    def get_batch(seed: int, batch_size: int) -> torch.Tensor:
        g = torch.Generator(device=dev0)
        g.manual_seed(seed)
        indices = torch.randint(0, dataset_len, (batch_size,), generator=g, device=dev0)
        return dataset_z[indices] # [B, T, D]

    def train_decoder_on_batch(seed: int, z_batch: torch.Tensor) -> float:
        # z_batch: [B, T, D]
        # print("    [train_decoder] Start...", flush=True)
        B, T, _ = z_batch.shape
        res0.set_adapter_from_theta(theta)

        decoder.train()
        for _ in range(args.dec_steps):
            dec_opt.zero_grad(set_to_none=True)
            h = torch.zeros((B, N), device=dev0)
            loss_steps = []
            
            # Use teacher forcing for decoder training or rollout?
            # Usually teacher forcing is better for training the readout
            for t in range(T - 1):
                zt = z_batch[:, t]
                with torch.no_grad():
                    h = res0.step(h, zt)
                pred = decoder(h)
                loss_steps.append(F.mse_loss(pred, z_batch[:, t + 1], reduction="mean"))
            
            loss = torch.stack(loss_steps).mean()
            loss.backward()
            dec_opt.step()

        decoder.eval()
        return float(loss.detach().cpu().item())

    # 3. ES Loop
    base_seed = 10_000_000 + args.seed * 1_000_000
    print("Main: Starting ES loop...", flush=True)
    t0 = time.time()

    for it in range(1, args.iters + 1):
        seed_it = base_seed + it
        
        if it % 1 == 0:
            print(f"Main: Starting iter {it}...", flush=True)
            
        # A. Get Batch for this iteration
        # We use the same batch for base and all candidates to reduce variance
        z_batch = get_batch(seed_it, args.batch)

        # B. Train Decoder / Get Base Loss
        # print("  [Main] Training decoder...", flush=True)
        base_loss = train_decoder_on_batch(seed_it, z_batch)
        # print(f"  [Main] Base loss: {base_loss:.4f}", flush=True)

        # C. Sample Antithetic Pairs
        K = args.pairs
        # Generate noise directly on GPU
        eps = torch.randn((K, theta_dim), device=dev0, dtype=torch.float32)
        
        candidates = []
        for sign in [1.0, -1.0]:
            for k in range(K):
                cand = theta + sign * args.sigma * eps[k]
                candidates.append(cand)
        
        # D. Evaluate Candidates
        # Sequential evaluation on GPU is fast enough because of pre-computation
        # We can also batch this if we really wanted to, but let's keep it simple first
        cand_losses = []
        
        # We can reuse the same z_batch for all candidates
        # To vectorize this: stack the batch? 
        # But we change weights (U, V).
        # Vectorizing across weights is hard in standard PyTorch without vmap.
        # So we loop. But since 'step' is just matrix multiplies and z is pre-loaded, it's fast.
        
        for i, cand in enumerate(candidates):
            res0.set_adapter_from_theta(cand)
            loss = res0.rollout_loss(z_batch, decoder, args.warmup)
            cand_losses.append(loss)
            # if i % 10 == 0:
            #    print(f"    [Main] Cand {i}/{len(candidates)} evaluated.", flush=True)
        
        # print("  [Main] Candidates evaluated.", flush=True)
        cand_losses = torch.stack(cand_losses)
        
        # E. Update Theta
        # Reconstruct pairs
        losses_pos = cand_losses[:K]
        losses_neg = cand_losses[K:]
        
        # Fitness shaping (using all losses together)
        w = rank_transform(cand_losses)
        w_pos = w[:K]
        w_neg = w[K:]
        
        # Gradient estimate
        # g = (w_pos - w_neg) * eps / (2 * sigma) ... roughly
        # Actually standard ES: sum(w_i * eps_i)
        
        # eps corresponds to pos. -eps corresponds to neg.
        # grad ~ sum( w_pos[k] * eps[k] + w_neg[k] * (-eps[k]) )
        grad_est = (w_pos.view(-1, 1) * eps - w_neg.view(-1, 1) * eps).mean(dim=0)
        # This simplifies to (w_pos - w_neg) * eps
        
        # Adam Step
        theta.grad = -grad_est # We want to minimize loss, so we move opposite to gradient of loss (which is ascent on fitness?)
        # Wait, rank transform: low loss -> high rank -> high weight. 
        # So we want to ASCEND the weighted average.
        # Adam minimizes. So we set grad = -grad_est.
        
        theta_opt.step()
        theta_opt.zero_grad()
        
        # Logging
        if it % args.log_every == 0:
            dt = time.time() - t0
            frac_bad = float((cand_losses >= 1e5).float().mean().item())
            
            # Visualization
            res0.set_adapter_from_theta(theta)
            # Visualize first item of current batch
            viz_path = os.path.join(results_dir, f"viz_iter_{it:04d}.png")
            visualize_batch(z_batch, decoder, res0, args.warmup, viz_path)
            
            # Simple H norm check
            with torch.no_grad():
                h_dummy = torch.zeros((1, N), device=dev0)
                h_dummy = res0.step(h_dummy, z_batch[0,0:1])
                h_norm = h_dummy.norm().item()

            print(
                f"iter {it:6d}  base_loss {base_loss:10.6f}  "
                f"cand_loss[min/med] {cand_losses.min().item():.6f}/{cand_losses.median().item():.6f}  "
                f"bad_frac {frac_bad:.2f}  "
                f"|theta| {theta.norm().item():.3f}  "
                f"|h| {h_norm:.2f}  "
                f"elapsed {dt/60:.1f}m",
                flush=True
            )
            
            csv_logger.log({
                "iter": it,
                "base_loss": base_loss,
                "min_loss": cand_losses.min().item(),
                "med_loss": cand_losses.median().item(),
                "bad_frac": frac_bad,
                "theta_norm": theta.norm().item(),
                "h_norm": h_norm,
                "elapsed_min": dt/60
            })
            
            ckpt_path = os.path.join(results_dir, "checkpoint_latest.pt")
            torch.save({
                "iter": it,
                "theta": theta,
                "decoder": decoder.state_dict(),
                "args": vars(args)
            }, ckpt_path)

if __name__ == "__main__":
    main()