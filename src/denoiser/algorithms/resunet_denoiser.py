# Residual U-Net CNN denoising algorithm

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseDenoiser


def _build_resunet(in_channels: int, base_channels: int):
    import torch
    import torch.nn as nn

    class _ResidualBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()

            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(x + self.block(x))


    class _DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()

            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    class ResidualUNet(nn.Module):
        def __init__(self, channels: int, features: int) -> None:
            super().__init__()

            self.enc1 = _DoubleConv(channels, features)
            self.res1 = _ResidualBlock(features)
            self.pool1 = nn.MaxPool2d(2)

            self.enc2 = _DoubleConv(features, features * 2)
            self.res2 = _ResidualBlock(features * 2)
            self.pool2 = nn.MaxPool2d(2)

            self.bottleneck = _DoubleConv(features * 2, features * 4)

            self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
            self.dec2 = _DoubleConv(features * 4, features * 2)

            self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
            self.dec1 = _DoubleConv(features * 2, features)

            self.out_conv = nn.Conv2d(features, channels, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            e1 = self.res1(self.enc1(x))
            e2 = self.res2(self.enc2(self.pool1(e1)))
            b = self.bottleneck(self.pool2(e2))

            d2 = self.up2(b)
            d2 = torch.cat([d2, e2], dim=1)
            d2 = self.dec2(d2)

            d1 = self.up1(d2)
            d1 = torch.cat([d1, e1], dim=1)
            d1 = self.dec1(d1)

            noise_residual = self.out_conv(d1)
            return torch.clamp(x - noise_residual, 0.0, 1.0)

    return ResidualUNet(in_channels, base_channels)


class ResUNetDenoiser(BaseDenoiser):
    # Initialise the Residual U-Net denoiser with model path and architecture/device settings
    def __init__(
        self,
        model_path: Optional[str] = None,
        base_channels: int = 32,
        device: str = 'auto',
    ) -> None:
        super().__init__(model_path=model_path, base_channels=base_channels, device=device)
        self.model_path = model_path
        self.base_channels = base_channels
        self.device = device

        self._model = None
        self._model_channels: Optional[int] = None

    # Resolve runtime device
    def _resolve_device(self):
        import torch

        if self.device != 'auto':
            return torch.device(self.device)

        if torch.cuda.is_available():
            return torch.device('cuda')

        return torch.device('cpu')

    # Load model weights once for a given number of channels
    def _load_model(self, channels: int):
        import torch

        if self.model_path is None:
            raise ValueError(
                "ResUNetDenoiser requires '--model-path' with pretrained weights. "
                "Provide a .pth/.pt checkpoint when using algorithm 'resunet'."
            )

        if self._model is not None and self._model_channels == channels:
            return self._model

        model = _build_resunet(in_channels=channels, base_channels=self.base_channels)
        checkpoint_path = Path(self.model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        if isinstance(state_dict, dict):
            state_dict = {
                (key[7:] if key.startswith('module.') else key): value
                for key, value in state_dict.items()
            }

        model.load_state_dict(state_dict)
        model.to(self._resolve_device())
        model.eval()

        self._model = model
        self._model_channels = channels
        return model

    # Denoise an image using a pretrained Residual U-Net model
    def denoise(self, noisy_image: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        # Ensure float range [0, 1]
        if noisy_image.dtype not in (np.float32, np.float64):
            noisy_image = noisy_image.astype(np.float32) / 255.0

        noisy_image = np.clip(noisy_image, 0.0, 1.0).astype(np.float32)

        # Convert to BCHW tensor
        if noisy_image.ndim == 2:
            tensor = torch.from_numpy(noisy_image).unsqueeze(0).unsqueeze(0)
            channels = 1
            is_grayscale = True
        elif noisy_image.ndim == 3:
            tensor = torch.from_numpy(np.transpose(noisy_image, (2, 0, 1))).unsqueeze(0)
            channels = noisy_image.shape[2]
            is_grayscale = False
        else:
            raise ValueError(f"Unsupported image shape: {noisy_image.shape}")

        model = self._load_model(channels=channels)
        device = self._resolve_device()
        tensor = tensor.to(device)

        # Pad to match two downsampling stages (factor 4)
        _, _, height, width = tensor.shape
        pad_h = (4 - (height % 4)) % 4
        pad_w = (4 - (width % 4)) % 4
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            output = model(tensor)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :height, :width]

        output_np = output.squeeze(0).detach().cpu().numpy()
        if is_grayscale:
            denoised = output_np.squeeze(0)
        else:
            denoised = np.transpose(output_np, (1, 2, 0))

        return np.clip(denoised, 0.0, 1.0)