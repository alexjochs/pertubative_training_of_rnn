import torch
import torchvision
from torchvision import transforms
from perturbative_trained_rnn import PretrainedResNet18Encoder

def check_embeddings():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Checking embeddings on {device}...")
    
    transform = transforms.Lambda(lambda x: x.float() / 255.0)
    dataset = torchvision.datasets.MovingMNIST(root="./data", split=None, download=True, transform=transform)
    
    encoder = PretrainedResNet18Encoder(emb_dim=128).to(device)
    encoder.eval()
    
    with torch.no_grad():
        # Get two different sequences
        s1 = dataset[0].to(device) # [20, 1, 64, 64]
        s2 = dataset[1].to(device)
        
        z1 = encoder(s1[0:1]) # Frame 0 of seq 0
        z2 = encoder(s1[1:2]) # Frame 1 of seq 0
        z3 = encoder(s2[0:1]) # Frame 0 of seq 1
        
        print(f"z1 (Seq 0, t=0) mean: {z1.mean().item():.4f}, std: {z1.std().item():.4f}, norm: {z1.norm().item():.4f}")
        print(f"z2 (Seq 0, t=1) mean: {z2.mean().item():.4f}, std: {z2.std().item():.4f}, norm: {z2.norm().item():.4f}")
        print(f"z3 (Seq 1, t=0) mean: {z3.mean().item():.4f}, std: {z3.std().item():.4f}, norm: {z3.norm().item():.4f}")
        
        diff12 = (z1 - z2).norm().item()
        diff13 = (z1 - z3).norm().item()
        
        print(f"Norm diff (t=0 vs t=1): {diff12:.4f}")
        print(f"Norm diff (Seq 0 vs Seq 1): {diff13:.4f}")
        
        # Check if all values are same
        print(f"z1 unique vals: {len(torch.unique(z1.round(decimals=4)))}")

if __name__ == "__main__":
    check_embeddings()
