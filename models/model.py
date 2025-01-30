import torch
import torch.nn as nn

class InterpolationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (input: 6 channels = 2 frames)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),  # (B, 64, H, W)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # Halves spatial dims
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (B, 128, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # Halves again
            nn.Conv2d(128, 256, kernel_size=3, padding=1) # (B, 256, H/4, W/4)
        )
        # Decoder (output: 3 channels = interpolated frame)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Doubles dims
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # Doubles again
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Final adjustment to 3 channels
            nn.Sigmoid()
        )

    def forward(self, frame1, frame2):
        x = torch.cat([frame1, frame2], dim=1)  # Concatenate along channels
        x = self.encoder(x)
        x = self.decoder(x)
        return x