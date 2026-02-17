
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


class MNISTDecoder(nn.Module):
    """
    Deconvolutional decoder to map 128-d embeddings back to 64x64 images.
    """
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 512 * 4 * 4) # Output for deconv
        )
        
        self.deconv = nn.Sequential(
            # [B, 512, 4, 4]
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # [B, 256, 8, 8]
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # [B, 128, 16, 16]
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 32, 32]
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),   # [B, 1, 64, 64]
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D]
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        x = self.deconv(x)
        return x


class PretrainedResNet18Encoder(nn.Module):
    """
    ResNet18 encoder adapted for single-channel input and custom embedding dimension.
    The classification head is replaced with a linear projection.
    All parameters are frozen.
    """
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
    Reservoir with a sparse recurrent base (W0) and a low-rank adapter (U, V).
    Dynamics: h_{t+1} = (1-leak)*h_t + leak * tanh(W0 h_t + U V^T h_t + W_in z_t + b)
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
        # 1. Recurrent base: W0 (sparse) @ h
        rec0 = h @ self.W0.t()

        # 2. Low-rank adapter: U @ (V^T @ h)
        low = (h @ self.V) @ self.U.t()

        # 3. Input projection
        inp = z @ self.Win.t()  # [B,N]

        pre = rec0 + low + inp + self.b
        nh = torch.tanh(pre)
        return (1.0 - self.cfg.leak) * h + self.cfg.leak * nh

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
        
        # Memory Optimization:
        # Pre-allocate buffer for activations to avoid creating new [K, B, N] tensors (32GB) per step.
        pre_buffer = torch.empty((K, B, N), device=self.device, dtype=torch.float32)

        losses = []
        z_in = z_seq[:, 0] # [B, D] - shared across K
        
        for t in range(T - 1):
            # 1. Initialize pre-activation with Input term
            if z_in.dim() == 2:
                # Shared input [B, D] -> [B, N]
                inp_small = z_in @ self.Win.t()
                # Expand and copy to buffer (Broadcasting)
                pre_buffer.copy_(inp_small.unsqueeze(0))
            else:
                # Individual input [K, B, D] -> [KB, N]
                # Direct MM into buffer view
                z_in_flat = z_in.view(K * B, D)
                torch.mm(z_in_flat, self.Win.t(), out=pre_buffer.view(K * B, N))
            
            # 2. Add Bias
            pre_buffer.add_(self.b)
            
            # 3. Add Recurrent Base: h @ W0.t()
            # W0 is dense. h is [K, B, N].
            h_flat = h.view(K * B, N)
            pre_flat = pre_buffer.view(K * B, N)
            pre_flat.addmm_(h_flat, self.W0.t())
            
            # 4. Add Low-Rank Adapter: (h @ V) @ U.t()
            # low_temp = h @ V -> [K, B, R] (Smaller tensor)
            low_temp = torch.bmm(h, cand_V)
            
            # pre += low_temp @ U.t()
            pre_buffer.baddbmm_(low_temp, cand_U.transpose(1, 2))

            # 5. Activation (In-place)
            # pre_buffer becomes 'nh'
            pre_buffer.tanh_()
            
            # 6. Update h in-place
            # h = (1 - leak) * h + leak * nh
            h.mul_(1.0 - self.cfg.leak)
            h.add_(pre_buffer, alpha=self.cfg.leak)
            
            # Decode
            # Decoder is shared. h: [K, B, N]
            h_flat_next = h.view(K * B, N)
            pred_flat = decoder(h_flat_next) # [K*B, D]
            pred = pred_flat.view(K, B, D)
            
            target = z_seq[:, t + 1] # [B, D] shared
            
            if t >= warmup:
                # Loss calculation
                # MSE: (pred - target)^2
                err = pred - target.unsqueeze(0)
                mse = (err ** 2).mean(dim=[1, 2]) # Mean over B and D, keep K
                losses.append(mse)
                
                # Feedback: Feeding back prediction
                z_in = pred
            else:
                # Feedback: Teacher forcing (GT)
                z_in = target # [B, D]
            
        if len(losses) == 0:
            return torch.zeros(K, device=self.device)
            
        return torch.stack(losses).mean(dim=0) # [K]





def make_decoder(N: int, D: int, device: torch.device) -> nn.Linear:
    decoder = nn.Linear(N, D, bias=True).to(device)
    # Using Xavier init
    nn.init.xavier_normal_(decoder.weight)
    nn.init.zeros_(decoder.bias)
    return decoder


def rank_transform(raw_losses: torch.Tensor) -> torch.Tensor:
    """
    Transform losses into rank-based fitness weights.
    Lower loss -> Higher rank -> Higher positive weight.
    Returns:
        Tensor of shape [M], centered so that mean is 0.
    """
    M = len(raw_losses)
    ranks = torch.empty_like(raw_losses)
    order = torch.argsort(raw_losses)
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


def train_image_decoder(args, z_all: torch.Tensor, x_all: torch.ByteTensor, device: torch.device):
    """
    Train (or load) MNSITDecoder to reconstruction images from embeddings.
    """
    ckpt_path = os.path.join(args.data_root, "image_decoder_ckpt.pt")
    
    decoder = MNISTDecoder(emb_dim=args.emb_dim).to(device)
    
    if os.path.exists(ckpt_path):
        print(f"Loading pre-trained image decoder from {ckpt_path}...", flush=True)
        # Verify if dimensions match? For now assume yes.
        try:
            decoder.load_state_dict(torch.load(ckpt_path, map_location=device))
            decoder.eval()
            return decoder
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Retraining...", flush=True)
    
    print("Training image decoder (z -> x)...", flush=True)
    decoder.train()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    criterion = nn.BCELoss() # Images are [0,1]
    
    # Flatten sequences for training: [N_seq * T, ...]
    N_seq, T, D = z_all.shape
    
    # We can train on a subset or full dataset. 
    # With 100 epochs on full dataset (200k images), it might take a while.
    # The user said "~100 epochs".
    
    batch_size = 256
    n_samples = N_seq * T
    indices = torch.arange(n_samples)
    
    # Flatten inputs
    z_flat = z_all.view(-1, D)
    x_flat = x_all.view(-1, 1, 64, 64) # ByteTensor
    
    epochs = args.dec_train_epochs
    t0 = time.time()
    
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i : i + batch_size]
            
            z_batch = z_flat[batch_idx]
            x_batch = x_flat[batch_idx].to(device).float() / 255.0
            
            recon = decoder(z_batch)
            loss = criterion(recon, x_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        avg_loss = total_loss / batches
        if (epoch + 1) % 10 == 0:
            dt = time.time() - t0
            print(f"  Decoder Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Elapsed: {dt:.1f}s", flush=True)
            
    print(f"Image decoder trained. Saving to {ckpt_path}...", flush=True)
    torch.save(decoder.state_dict(), ckpt_path)
    decoder.eval()
    return decoder


def visualize_batch(z_seq: torch.Tensor, decoder: nn.Linear, image_decoder: MNISTDecoder, 
                    res: Reservoir, warmup: int, save_path: str):
    plt.switch_backend('Agg')
    z_gt_seq = z_seq[0].detach().cpu() # [T, D]
    T, D = z_gt_seq.shape
    
    # 1. Run Reservoir to get Pred Z
    h = torch.zeros((1, res.cfg.N), device=res.device)
    z_pred_list = []
    z_in = z_gt_seq[0:1].to(res.device)
    
    with torch.no_grad():
        for t in range(T - 1):
            h = res.step(h, z_in)
            pred = decoder(h)
            z_pred_list.append(pred.cpu())
            if t < warmup:
                z_in = z_gt_seq[t+1:t+2].to(res.device)
            else:
                z_in = pred
    
    # z_pred_list has T-1 entries (from t=1 to T-1). t=0 is initial input.
    # Prepend first frame from GT for nice visualization scaling? 
    # Or just align t=1..T-1. 
    # Let's align with GT z_target which typically is t=1..T-1 or t=0..T depending on perspective.
    # In loop: target is z_seq[:, t+1]. So z_pred aligns with z_gt[1:].
    
    z_pred = torch.cat(z_pred_list, dim=0) # [T-1, D]
    z_target = z_gt_seq[1:] # [T-1, D]
    
    # 2. Reconstruct Images
    # We need to run image_decoder on Z
    # z_target: [T-1, D]
    # z_pred: [T-1, D]
    with torch.no_grad():
        x_target = image_decoder(z_target.to(res.device)).cpu() # [T-1, 1, 64, 64]
        x_pred = image_decoder(z_pred.to(res.device)).cpu()     # [T-1, 1, 64, 64]

    # 3. Plotting
    # We want 4 rows: GT Image, GT Z, Pred Image, Pred Z
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # Row 0: GT Images
    # Concatenate images horizontally
    # x_target: [Time, 1, H, W] -> [H, Time*W]
    x_gt_grid = torchvision.utils.make_grid(x_target, nrow=T-1, padding=2, pad_value=0.5)
    axes[0].imshow(x_gt_grid.permute(1, 2, 0).numpy(), cmap='gray')
    axes[0].set_title("Ground Truth Reconstructed Images")
    axes[0].axis('off')
    
    # Row 1: GT Z
    im1 = axes[1].imshow(z_target.T, aspect='auto', cmap='viridis')
    axes[1].set_title("Ground Truth Embedding (z_t)")
    axes[1].set_ylabel("Dimension")
    plt.colorbar(im1, ax=axes[1])
    
    # Row 2: Pred Images
    x_pred_grid = torchvision.utils.make_grid(x_pred, nrow=T-1, padding=2, pad_value=0.5)
    axes[2].imshow(x_pred_grid.permute(1, 2, 0).numpy(), cmap='gray')
    axes[2].set_title(f"Predicted Reconstructed Images (Warmup={warmup})")
    axes[2].axis('off')

    # Row 3: Pred Z
    im3 = axes[3].imshow(z_pred.T, aspect='auto', cmap='viridis')
    axes[3].set_title(f"Predicted Embedding")
    axes[3].set_ylabel("Dimension")
    axes[3].set_xlabel("Time Step (t)")
    
    # Add vertical line for warmup on Z plots
    # Time axis is 0..T-2. Warmup happens at step `warmup`. 
    # Loop index t goes 0..T-2. t < warmup is forced.
    # So boundary is at x = warmup - 0.5?
    # t=warmup-1 is last forced step. t=warmup is first free step.
    # We plot z_pred which matches z_gt[1:].
    # Index 0 corresponds to t=1.
    # Index warmup-1 corresponds to t=warmup. (The first predicted step).
    # So the line should be at warmup - 0.5 roughly.
    
    axes[1].axvline(x=warmup - 0.5, color='red', linestyle='--', linewidth=2)
    axes[3].axvline(x=warmup - 0.5, color='red', linestyle='--', linewidth=2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def transform_image_norm(x):
    """Normalize image to [0, 1]."""
    return x.float() / 255.0

def precompute_embeddings(args, device):
    print("Pre-computing embeddings for the entire dataset...", flush=True)
    t0 = time.time()
    
    mm0 = torchvision.datasets.MovingMNIST(
        root=args.data_root,
        split=None,
        download=True,
        transform=transform_image_norm
    )
    
    # Create loader
    # Using a large batch size for inference speed
    loader = torch.utils.data.DataLoader(mm0, batch_size=64, shuffle=False, num_workers=4)
    encoder = PretrainedResNet18Encoder(emb_dim=args.emb_dim).to(device)
    encoder.eval()

    all_embeddings = []
    all_images = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # batch: [B, T, 1, 64, 64]
            if args.T < batch.shape[1]:
                batch = batch[:, :args.T]
            
            # Store original images as ByteTensor to save RAM (0-255)
            # Flatten B and T later
            B, T, C, H, W = batch.shape
            # batch is uint8 or float? dataset transform says float/255.
            # But the dataset loader might have applied it.
            # Convert back to uint8 for storage if it was float.
            # Ah, the transform in precompute uses transform_image_norm which floats and div 255.
            # We want to store compact.
            # Actually, let's just keep them as uint8 before transform? 
            # But `loader` applies transform.
            # Let's revert: (batch * 255).to(torch.uint8)
            imgs_byte = (batch * 255).clamp(0, 255).to(torch.uint8)
            all_images.append(imgs_byte)
            
            batch = batch.to(device)
            x = batch.reshape(B * T, 1, H, W)
            z = encoder(x).view(B, T, -1) # [B, T, D]
            all_embeddings.append(z.cpu()) 
            if i % 10 == 0:
                print(f"  Encoded batch {i}/{len(loader)}", flush=True)
            
            if args.limit_data > 0 and i >= args.limit_data - 1:
                print(f"  Reached limit of {args.limit_data} batches. Stopping.", flush=True)
                break

    # Concatenate all embeddings
    full_embeddings = torch.cat(all_embeddings, dim=0) # [N_total, T, D]
    full_images = torch.cat(all_images, dim=0)         # [N_total, T, 1, 64, 64] (uint8)
    
    elapsed = time.time() - t0
    print(f"Pre-computation complete. Z: {full_embeddings.shape}, X: {full_images.shape}. Time: {elapsed:.2f}s", flush=True)
    
    # Return both. Keep images on CPU until needed (1.6GB). Embeddings to device (small).
    return full_embeddings.to(device), full_images


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
    args_parser.add_argument("--dec_train_epochs", type=int, default=100, help="Epochs to pre-train image decoder")
    args_parser.add_argument("--limit_data", type=int, default=0, help="Limit number of batches for pre-computation (0=no limit)")

    
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
    dataset_embeddings, dataset_images = precompute_embeddings(args, device)
    dataset_len = len(dataset_embeddings)

    # 1.5 Train Image Decoder (for visualization)
    image_decoder = train_image_decoder(args, dataset_embeddings, dataset_images, device)

    # 2. Setup Models
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
        # Generate noise: epsilon [pairs, 2*N*r]
        K = args.pairs
        epsilon = torch.randn((K, theta_dim), device=device, dtype=torch.float32)
        
        # Create candidates: theta +/- sigma * epsilon
        theta_unsqueezed = theta.unsqueeze(0)
        pop_theta = torch.cat([
            theta_unsqueezed + args.sigma * epsilon,
            theta_unsqueezed - args.sigma * epsilon
        ], dim=0) # [2*K, theta_dim]
        
        
        # Free memory of intermediate tensors
        # (Variables are now temporary in the cat call, so they are freed automatically)
        
        # Reshape into U and V
        # theta structure: [U_flat, V_flat]
        N, r = args.hidden, args.rank
        U_size = N * r
        
        pop_U_flat = pop_theta[:, :U_size]
        pop_V_flat = pop_theta[:, U_size:]
        
        cand_U = pop_U_flat.view(-1, N, r)
        cand_V = pop_V_flat.view(-1, N, r)
        
        # Batched Rollout
        cand_losses = reservoir.batched_rollout_loss(z_batch, decoder, args.warmup, cand_U, cand_V)


        
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
        # Maximize fitness (rank-based weights) via Gradient Ascent.
        # Adam minimizes, so we negate the estimated gradient:
        # grad_estimate ~ (w_pos - w_neg) * epsilon
        # theta_new = theta - lr * (-grad_estimate) = theta + lr * grad_estimate
        theta.grad = -grad_estimate
        
        theta_optimizer.step()
        theta_optimizer.zero_grad()
        
        # Logging
        if iteration % args.log_every == 0:
            elapsed_total = time.time() - start_time
            frac_bad = float((cand_losses >= 1e5).float().mean().item())
            
            # Visualization
            reservoir.set_adapter_from_theta(theta)
            # Visualize first item of current batch
            viz_path = os.path.join(results_dir, f"viz_iter_{iteration:04d}.png")
            visualize_batch(z_batch, decoder, image_decoder, reservoir, args.warmup, viz_path)
            
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