#!/usr/bin/env python3
"""
Visualization script for Perturbative RNN.
Goals:
1. Show input data (Moving MNIST frames).
2. Show model structure + what is trained vs frozen.
3. Show reservoir dynamics (hidden state activity).
"""


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from perturbative_trained_rnn import PretrainedResNet18Encoder, Reservoir, ReservoirConfig, make_decoder

def print_section(title):
    print(f"\n{'='*40}")
    print(f" {title}")
    print(f"{'='*40}")

def inspect_data():
    print_section("1. INPUT DATA (Moving MNIST)")
    
    # Load dataset
    transform = transforms.Lambda(lambda x: x.float() / 255.0)
    data_root = "./data"
    print(f"Loading MovingMNIST form {data_root}...")
    try:
        mm = torchvision.datasets.MovingMNIST(
            root=data_root, download=True, transform=transform
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Get one sample: tensor [20, 1, 64, 64]
    seq_len = 20
    sample = mm[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Value range: [{sample.min():.2f}, {sample.max():.2f}]")
    
    # Save a visualization of the sequence
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10): # Show first 10 frames
        ax = axes[i]
        frame = sample[i, 0].numpy()
        ax.imshow(frame, cmap='gray')
        ax.axis('off')
        ax.set_title(f"t={i}")
    
    plt.suptitle("First 10 frames of Moving MNIST Sample 0")
    plt.tight_layout()
    plt.savefig("data_viz_frames.png")
    print("Saved 'data_viz_frames.png' - Please open this file to see the inputs.")

def inspect_model_structure():
    print_section("2. MODEL STRUCTURE & TRAINING STATUS")
    
    device = torch.device("cpu")
    
    # 1. Encoder (Matches training script logic: explicitly freeze)
    enc = PretrainedResNet18Encoder(emb_dim=128).to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    
    # 2. Reservoir
    cfg = ReservoirConfig(N=500, D=128, rank=16) # Smaller for viz
    res = Reservoir(cfg, device=device).to(device)
    
    # 3. Decoder
    dec = make_decoder(N=500, D=128, device=device)
    
    print(f"Arch: Input(64x64) -> Encoder -> z(128) -> Reservoir(N={cfg.N}) -> h -> Decoder -> z_pred(128)")
    print("-" * 75)
    print(f"{'Module':<20} | {'Parameter Name':<20} | {'Shape':<15} | {'TRAINABLE?':<15}")
    print("-" * 75)
    
    def walk_model(model, prefix):
        for name, param in model.named_parameters():
            status = "YES (GRAD)" if param.requires_grad else "NO (FROZEN)"
            
            # Special check for ES parameters
            # U and V are nn.Parameter(..., requires_grad=False)
            if "Reservoir" in prefix and name in ["U", "V"]:
                status = "YES (ES)"
            
            print(f"{prefix:<20} | {name:<20} | {str(list(param.shape)):<15} | {status:<15}")

    walk_model(enc, "Encoder (CNN)")
    walk_model(res, "Reservoir (RNN)")
    walk_model(dec, "Decoder (Readout)")
    print("-" * 75)
    print("NOTE: 'YES (ES)' means trained via Evolution Strategies.")
    print("NOTE: 'YES (GRAD)' means trained via Standard Backprop (Analytic readout).")

def inspect_dynamics():
    print_section("3. RESERVOIR DYNAMICS (Forward Pass)")
    
    device = torch.device("cpu")
    
    # Setup model
    transform = transforms.Lambda(lambda x: x.float() / 255.0)
    mm = torchvision.datasets.MovingMNIST(root="./data", download=True, transform=transform)
    sample = mm[0].unsqueeze(0) # [1, 20, 1, 64, 64]
    
    enc = PretrainedResNet18Encoder(emb_dim=128)
    cfg = ReservoirConfig(N=500, D=128, leak=0.3)
    res = Reservoir(cfg, device=device)
    
    # Run forward
    T = sample.shape[1]
    B = 1
    
    # Encode
    B_in, T_in, C, H, W = sample.shape
    x = sample.view(B_in*T_in, C, H, W)
    with torch.no_grad():
        z = enc(x).view(B_in, T_in, -1) # [1, 20, 128]
    
    # Recurse
    h = torch.zeros((B, cfg.N))
    h_norms = []
    neuron_activity = [] # Track first 5 neurons
    
    for t in range(T):
        zt = z[:, t]
        h = res.step(h, zt)
        h_norms.append(h.norm().item())
        neuron_activity.append(h[0, :5].detach().numpy())
    
    print(f"Encoded Sequence Shape: {z.shape}")
    print(f"Final Hidden State Norm: {h_norms[-1]:.2f}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(h_norms, marker='o')
    ax1.set_title("Reservoir State Norm ||h_t|| over Time")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Norm")
    ax1.grid(True)
    
    ax2.plot(neuron_activity)
    ax2.set_title("Activity of First 5 Neurons")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Activation (tanh)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("model_dynamics.png")
    print("Saved 'model_dynamics.png' - Visualizing reservoir stability.")

if __name__ == "__main__":
    try:
        inspect_data()
        inspect_model_structure()
        inspect_dynamics()
        print("\nVisualization complete!")
    except ImportError:
        print("Please run this in the environment where torch/torchvision/matplotlib are installed.")
