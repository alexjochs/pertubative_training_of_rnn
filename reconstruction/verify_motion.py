import torch
import torchvision
import matplotlib.pyplot as plt
import os

def verify_motion():
    print("Verifying Moving MNIST motion...")
    
    data_root = "./data"
    os.makedirs(data_root, exist_ok=True)
    
    # Load dataset
    transform = torchvision.transforms.Lambda(lambda x: x.float() / 255.0)
    dataset = torchvision.datasets.MovingMNIST(
        root=data_root, split=None, download=True, transform=transform
    )
    
    # Get a sample: [20, 1, 64, 64]
    sample = dataset[0]
    seq_len = sample.shape[0]
    
    # Create a grid of frames to show motion
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    axes = axes.flatten()
    
    for i in range(seq_len):
        ax = axes[i]
        ax.imshow(sample[i, 0], cmap='gray')
        ax.set_title(f"t={i}")
        ax.axis('off')
    
    plt.suptitle("Moving MNIST Sequence (20 frames)")
    plt.tight_layout()
    save_path = "reconstruction/motion_verification.png"
    os.makedirs("reconstruction", exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved motion verification plot to {save_path}")

if __name__ == "__main__":
    verify_motion()
