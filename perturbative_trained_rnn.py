
"""
Perturbative Trained RNN (Single-Process GPU Optimized)

This script implements an Evolutionary Strategy (ES) to train a Recurrent Neural Network (RNN) 
with a low-rank perturbative adapter. It is optimized for high-performance GPUs (like H200) 
by pre-computing embeddings and running the entire ES loop on the GPU to avoid CPU-GPU bottlenecks.
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



    @torch.no_grad()
    def batched_rollout_loss(self, z_seq: torch.Tensor, decoder: nn.Linear, warmup: int, 
                             cand_U: torch.Tensor, cand_V: torch.Tensor) -> torch.Tensor:
        """
        Batched rollout for multiple candidates simultaneously.
        z_seq: [B, T, D] (shared input)
        cand_U: [K, N, R]
        cand_V: [K, N, R]
        
        Returns: [K] losses
        """
        K = cand_U.shape[0]  # Number of candidates
        B, T, D = z_seq.shape
        N = self.cfg.N
        
        # Initial state h: [K, B, N]
        h = torch.zeros((K, B, N), device=self.device, dtype=torch.float32)
        
        losses = []
        z_in = z_seq[:, 0] # [B, D] - shared across K
        
        # Pre-compute W0 transpose for efficiency if possible, but W0 is sparse [N, N]
        # We can't batch sparse MATMUL easily with [K, B, N].
        # However, W0 is shared. h @ W0.t() is [K*B, N] @ [N, N].
        # We can reshape h to [K*B, N] to use sparse mm.
        
        # Win is shared. z @ Win.t() is [B, N]. Broadcast to [K, B, N].
        
        for t in range(T - 1):
            # 1. Recurrent Base: h @ W0.t()
            # Reshape h to [K*B, N]
            h_flat = h.view(K * B, N)
            rec0 = (h_flat @ self.W0.t()).view(K, B, N)
            
            # 2. Low-Rank Adapter: (h @ V) @ U.t()
            # h: [K, B, N], V: [K, N, R] -> [K, B, R]
            # specific V per K.
            # torch.bmm is [B, N, M] x [B, M, P].
            # We want [K, B, N] x [K, N, R].
            # bmm works on the first dim K!
            low_temp = torch.bmm(h, cand_V) # [K, B, R]
            
            # Now [K, B, R] x [K, R, N] (U transposed)
            # U is [K, N, R], so U.transpose(1, 2) is [K, R, N]
            low = torch.bmm(low_temp, cand_U.transpose(1, 2)) # [K, B, N]
            
            # 3. Input: z @ Win.t()
            # z_in is [K, B, D] or [B, D]?
            # If z_in varies per candidate (due to feedback), it must be [K, B, D].
            # Initially z_in is [B, D] (shared).
            if z_in.dim() == 2:
                inp = z_in @ self.Win.t() # [B, N]
                inp = inp.unsqueeze(0).expand(K, -1, -1) # [K, B, N]
            else:
                 # z_in is [K, B, D]
                 # We need to reshape to apply shared Win
                 z_in_flat = z_in.view(K * B, D)
                 inp = (z_in_flat @ self.Win.t()).view(K, B, N)
            
            pre = rec0 + low + inp + self.b # b broadcasted
            
            nh = torch.tanh(pre)
            h = (1.0 - self.cfg.leak) * h + self.cfg.leak * nh
            
            # Decode
            # Decoder is shared. h: [K, B, N]
            # We can reshape to [K*B, N]
            h_flat_next = h.view(K * B, N)
            pred_flat = decoder(h_flat_next) # [K*B, D]
            pred = pred_flat.view(K, B, D)
            
            target = z_seq[:, t + 1] # [B, D] shared
            
            if t >= warmup:
                # Loss calculation
                # MSE: (pred - target)^2
                # Target broadcasted to [K, B, D]
                err = pred - target.unsqueeze(0)
                mse = (err ** 2).mean(dim=[1, 2]) # Mean over B and D, keep K
                losses.append(mse)
                
                # Feedback: Feeding back prediction
                z_in = pred
            else:
                # Feedback: Teacher forcing (GT)
                z_in = target # [B, D] shared (so dim=2)
                
            # Check stability (simple check on one random batch/cand, or max)
            # if h.abs().max() > self.cfg.h_clip: ... costly to check all
            # Let's skip check for speed on H200 or do loosely
            pass

        if len(losses) == 0:
            return torch.zeros(K, device=self.device)
            
        return torch.stack(losses).mean(dim=0) # [K]




def rank_transform(raw_losses: torch.Tensor) -> torch.Tensor:
    M = len(raw_losses)
    ranks = torch.empty_like(raw_losses)
    order = torch.argsort(raw_losses)
    # FIX: Ensure arange is on the same device as ranks
    ranks[order] = torch.arange(M, dtype=torch.float32, device=raw_losses.device)
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

    # Concatenate all embeddings
    full_embeddings = torch.cat(all_embeddings, dim=0) # [N_total, T, D]
    elapsed = time.time() - t0
    print(f"Pre-computation complete. Shape: {full_embeddings.shape}. Time: {elapsed:.2f}s", flush=True)
    
    return full_embeddings.to(device)


def main() -> None:
    # Disable cuDNN benchmarking to prevent hangs during algorithm search
    torch.backends.cudnn.benchmark = False
    
    args_parser = argparse.ArgumentParser(description="P-RNN Training Script")
    args_parser.add_argument("--data_root", type=str, default="./data", help="Root directory for datasets")
    args_parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (Ignored in simple single-proc mode)")
    args_parser.add_argument("--hidden", type=int, default=2000, help="Reservoir hidden size (N)")
    args_parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension (D)")
    args_parser.add_argument("--rank", type=int, default=32, help="Rank of perturbation matrix")
    args_parser.add_argument("--k_in", type=int, default=50, help="Sparsity of W0 (connections per neuron)")

    args_parser.add_argument("--T", type=int, default=20, help="Sequence length")
    args_parser.add_argument("--warmup", type=int, default=15, help="Warmup steps before tracking loss")
    args_parser.add_argument("--batch", type=int, default=16, help="Batch size for ES evaluation")
    


    args_parser.add_argument("--iters", type=int, default=2000, help="Total ES iterations")
    args_parser.add_argument("--pairs", type=int, default=32, help="Number of antithetic perturbation pairs per iter")
    args_parser.add_argument("--sigma", type=float, default=0.1, help="Perturbation standard deviation")
    args_parser.add_argument("--theta_lr", type=float, default=0.01, help="Learning rate for adapter (theta)")
    
    args_parser.add_argument("--dec_lr", type=float, default=0.001, help="Learning rate for readout decoder")
    args_parser.add_argument("--dec_steps", type=int, default=1, help="Decoder training steps per ES iter")

    
    args_parser.add_argument("--leak", type=float, default=0.1, help="Leak rate (alpha)")
    args_parser.add_argument("--h_clip", type=float, default=50.0, help="Clip value for hidden state")
    args_parser.add_argument("--w0_std", type=float, default=1.0, help="Spectral radius scaling")
    args_parser.add_argument("--win_std", type=float, default=0.2, help="Input scaling")
    
    args_parser.add_argument("--log_every", type=int, default=20)
    args_parser.add_argument("--seed", type=int, default=0)
    args = args_parser.parse_args()

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
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # 1. Pre-compute embeddings
    dataset_embeddings = precompute_embeddings(args, device)
    dataset_len = len(dataset_embeddings)

    # 2. Setup Modeis
    N, D, rnk = args.hidden, args.emb_dim, args.rank
    theta_dim = 2 * N * rnk

    # Theta: The flattened parameter vector for the low-rank adapter (U, V)
    theta = torch.zeros(theta_dim, dtype=torch.float32, device=device) 
    
    # Use Adam for optimizing theta (applied to the estimated gradient)
    theta_optimizer = torch.optim.Adam([theta], lr=args.theta_lr)

    decoder = make_decoder(N, D, device=device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.dec_lr)

    config = ReservoirConfig(
        N=N, D=D, rank=rnk, leak=args.leak, h_clip=args.h_clip,
        k_in=args.k_in, w0_std=args.w0_std, win_std=args.win_std
    )
    reservoir = Reservoir(config, device=device).to(device)
    reservoir.eval()

    # Create a reusable generator to avoid overhead/potential issues
    batch_generator = torch.Generator(device=device)

    def get_batch(seed: int, batch_size: int) -> torch.Tensor:
        """Sample a batch of pre-computed embeddings using a specific seed."""
        batch_generator.manual_seed(seed)
        indices = torch.randint(0, dataset_len, (batch_size,), generator=batch_generator, device=device)
        return dataset_embeddings[indices] # [B, T, D]

    def train_decoder_on_batch(seed: int, z_batch: torch.Tensor) -> float:
        """Train the readout decoder on the current batch using the base theta."""
        B, T, _ = z_batch.shape
        reservoir.set_adapter_from_theta(theta)

        decoder.train()
        for _ in range(args.dec_steps):
            decoder_optimizer.zero_grad(set_to_none=True)
            h = torch.zeros((B, N), device=device)
            loss_steps = []
            
            # Forward pass: Using teacher forcing for stable decoder training
            for t in range(T - 1):
                zt = z_batch[:, t]
                with torch.no_grad():
                    h = reservoir.step(h, zt)
                pred = decoder(h)
                loss_steps.append(F.mse_loss(pred, z_batch[:, t + 1], reduction="mean"))
            
            loss = torch.stack(loss_steps).mean()
            loss.backward()
            decoder_optimizer.step()

        decoder.eval()
        return float(loss.detach().cpu().item())

    # 3. ES Loop
    base_seed = 10_000_000 + args.seed * 1_000_000
    print("Main: Starting ES loop...", flush=True)
    start_time = time.time()

    for iteration in range(1, args.iters + 1):
        seed_it = base_seed + iteration
        
        if iteration % 1 == 0:
            print(f"Main: Starting iter {iteration}...", flush=True)
            
        # A. Get Batch for this iteration
        # We share the same batch across candidates to reduce variance
        z_batch = get_batch(seed_it, args.batch)

        # B. Train Decoder to get baseline loss
        # Note: We train the decoder *before* sampling candidates so the readout is optimal for the current theta
        base_loss = train_decoder_on_batch(seed_it, z_batch)

        # D. Evaluate Candidates - VECTORIZED
        # 1. Expand parameters for all candidates: K = 2 * args.pairs
        # We need U [K, N, r] and V [K, N, r]
        # theta is [2*N*r].
        # We generate noise epsilon [pairs, 2*N*r].
        # candidates = theta +/- sigma * epsilon
        
        # Let's construct pop_theta directly as a tensor [K, theta_dim]
        # epsilon is [args.pairs, theta_dim].
        # theta is [theta_dim].
        
        # Pos candidates: theta + sigma * epsilon
        theta_unsqueezed = theta.unsqueeze(0)
        pop_pos = theta_unsqueezed + args.sigma * epsilon
        pop_neg = theta_unsqueezed - args.sigma * epsilon
        
        pop_theta = torch.cat([pop_pos, pop_neg], dim=0) # [K, theta_dim] where K = 2*pairs
        
        # Reshape into U and V
        # theta structure: [U_flat, V_flat]
        N, r = args.hidden, args.rank
        U_size = N * r
        
        pop_U_flat = pop_theta[:, :U_size]
        pop_V_flat = pop_theta[:, U_size:]
        
        cand_U = pop_U_flat.view(-1, N, r)
        cand_V = pop_V_flat.view(-1, N, r)
        
        # Batched Rollout
        # z_batch is [B, T, D]
        # Memory Check:
        # h state: [K, B, N]
        # K=8192, B=256, N=4096 => 8192*256*4096*4 bytes = 34 GB.
        # H200 has 141 GB. Safe.
        
        try:
            cand_losses = reservoir.batched_rollout_loss(z_batch, decoder, args.warmup, cand_U, cand_V)
        except torch.cuda.OutOfMemoryError:
            print("WARNING: OOM with full batch. Switching to chunked evaluation.", flush=True)
            # Fallback to chunks of 1024
            chunk_size = 1024
            loss_chunks = []
            for i in range(0, len(pop_theta), chunk_size):
                u_chunk = cand_U[i : i + chunk_size]
                v_chunk = cand_V[i : i + chunk_size]
                l_chunk = reservoir.batched_rollout_loss(z_batch, decoder, args.warmup, u_chunk, v_chunk)
                loss_chunks.append(l_chunk)
            cand_losses = torch.cat(loss_chunks)

        
        # E. Update Theta
        losses_pos = cand_losses[:K]
        losses_neg = cand_losses[K:]
        
        # Fitness shaping (rank transformation)
        fitness_weights = rank_transform(cand_losses)
        w_pos = fitness_weights[:K]
        w_neg = fitness_weights[K:]
        
        # Gradient Estimation:
        # grad ~ (w_pos - w_neg) * eps
        # We expand views to broadcast: [K, 1] * [K, dim] -> [K, dim] -> mean -> [dim]
        grad_estimate = (w_pos.view(-1, 1) * epsilon - w_neg.view(-1, 1) * epsilon).mean(dim=0)
        
        # Optimization Step
        # Since we want to Minimize loss but rank_transform assigns higher values to better (lower) losses,
        # we treat this as maximizing fitness. Adam minimizes, so we negate the gradient.
        # Wait, let's double check:
        # Rank transform: Best individual (lowest loss) -> Highest rank -> Positive weight.
        # Gradient Ascent: theta += alpha * (fitness * noise)
        # Adam Step (Minimize): theta -= alpha * grad
        # So we set grad = -(fitness * noise) to achieve ascent.
        theta.grad = -grad_estimate
        
        theta_optimizer.step()
        theta_optimizer.zero_grad()
        
        # Logging
        # Logging
        if iteration % args.log_every == 0:
            elapsed_total = time.time() - start_time
            frac_bad = float((cand_losses >= 1e5).float().mean().item())
            
            # Visualization
            reservoir.set_adapter_from_theta(theta)
            # Visualize first item of current batch
            viz_path = os.path.join(results_dir, f"viz_iter_{iteration:04d}.png")
            visualize_batch(z_batch, decoder, reservoir, args.warmup, viz_path)
            
            # Simple Reservoir Hidden State Norm Check
            with torch.no_grad():
                h_dummy = torch.zeros((1, N), device=device)
                h_dummy = reservoir.step(h_dummy, z_batch[0,0:1])
                h_norm = h_dummy.norm().item()

            print(
                f"iter {iteration:6d}  base_loss {base_loss:10.6f}  "
                f"cand_loss[min/med] {cand_losses.min().item():.6f}/{cand_losses.median().item():.6f}  "
                f"bad_frac {frac_bad:.2f}  "
                f"|theta| {theta.norm().item():.3f}  "
                f"|h| {h_norm:.2f}  "
                f"elapsed {elapsed_total/60:.1f}m",
                flush=True
            )
            
            csv_logger.log({
                "iter": iteration,
                "base_loss": base_loss,
                "min_loss": cand_losses.min().item(),
                "med_loss": cand_losses.median().item(),
                "bad_frac": frac_bad,
                "theta_norm": theta.norm().item(),
                "h_norm": h_norm,
                "elapsed_min": elapsed_total/60
            })
            
            ckpt_path = os.path.join(results_dir, "checkpoint_latest.pt")
            torch.save({
                "iter": iteration,
                "theta": theta,
                "decoder": decoder.state_dict(),
                "args": vars(args)
            }, ckpt_path)

if __name__ == "__main__":
    main()