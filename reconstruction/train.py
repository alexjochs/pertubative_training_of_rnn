import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import time

from reconstruction.model import MNISTDecoder
# We need the encoder from the main script to generate embeddings
from perturbative_trained_rnn import PretrainedResNet18Encoder

def train_reconstruction():
    print("Starting reconstruction training...")
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # Dataset setup
    data_root = "./data"
    transform = transforms.Lambda(lambda x: x.float() / 255.0)
    dataset = torchvision.datasets.MovingMNIST(
        root=data_root, split=None, download=True, transform=transform
    )
    
    # We only take the first frame for training the reconstruction?
    # Or we can treat all frames as independent samples.
    # Total samples: 10000 * 20 = 200,000
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Models
    emb_dim = 128
    encoder = PretrainedResNet18Encoder(emb_dim=emb_dim).to(device)
    encoder.eval() # Keep frozen
    
    decoder = MNISTDecoder(emb_dim=emb_dim).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    # Training loop
    epochs = 10
    os.makedirs("reconstruction/checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        total_loss = 0
        t0 = time.time()
        for i, batch in enumerate(loader):
            # batch: [64, 20, 1, 64, 64]
            # Flatten B and T
            B, T, C, H, W = batch.shape
            images = batch.view(B * T, C, H, W).to(device)
            
            # 1. Encode
            with torch.no_grad():
                z = encoder(images)
            
            # 2. Decode
            recon = decoder(z)
            
            # 3. Loss
            loss = criterion(recon, images)
            
            # 4. Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i}/{len(loader)} | Loss: {loss.item():.6f}")
                print(f"  GT Range: [{images.min().item():.3f}, {images.max().item():.3f}]")
                print(f"  Recon Range: [{recon.min().item():.3f}, {recon.max().item():.3f}]")
                
        avg_loss = total_loss / len(loader)
        dt = time.time() - t0
        print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.6f} | Time: {dt:.2f}s")
        
        # Save visualization
        save_reconstruction_viz(images[:8], recon[:8], f"reconstruction/recon_epoch_{epoch+1}.png")
        
    # Save the model
    torch.save(decoder.state_dict(), "reconstruction/decoder.pt")
    print("Training finished and model saved.")

def save_reconstruction_viz(gt, recon, path):
    gt = gt.detach().cpu()
    recon = recon.detach().cpu()
    
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(gt[i, 0], cmap='gray')
        axes[0, i].set_title("GT")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(recon[i, 0], cmap='gray')
        axes[1, i].set_title("Recon")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

if __name__ == "__main__":
    train_reconstruction()
