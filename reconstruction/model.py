import torch
import torch.nn as nn

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
