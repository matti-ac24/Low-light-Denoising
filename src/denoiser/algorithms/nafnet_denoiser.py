# NAFNet denoising algorithm

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseDenoiser


def _build_nafnet(
    in_channels: int,
    base_channels: int,
    num_blocks: int = 2,
    middle_blocks: int = 4,
    expansion: int = 2,
):
    """Build the NAFNet model used by the denoiser wrapper."""
    import torch
    import torch.nn as nn

    class _LayerNorm2d(nn.Module):
        def __init__(self, channels: int, eps: float = 1e-6) -> None:
            """Initialize the object with the provided settings."""
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the forward pass for this block."""
            mean = x.mean(dim=1, keepdim=True)
            variance = x.var(dim=1, keepdim=True, unbiased=False)
            normalized = (x - mean) / torch.sqrt(variance + self.eps)
            return normalized * self.weight + self.bias

    class _SimpleGate(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the forward pass for this block."""
            first_half, second_half = torch.chunk(x, 2, dim=1)
            return first_half * second_half

    class _NAFBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            """Initialize the object with the provided settings."""
            super().__init__()

            hidden_channels = channels * expansion
            expanded_channels = hidden_channels * 2

            self.norm1 = _LayerNorm2d(channels)
            self.pw_conv1 = nn.Conv2d(channels, expanded_channels, kernel_size=1)
            self.dw_conv1 = nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=3,
                padding=1,
                groups=expanded_channels,
            )
            self.gate = _SimpleGate()
            self.pw_conv2 = nn.Conv2d(hidden_channels, channels, kernel_size=1)

            self.norm2 = _LayerNorm2d(channels)
            self.pw_conv3 = nn.Conv2d(channels, expanded_channels, kernel_size=1)
            self.dw_conv2 = nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=3,
                padding=1,
                groups=expanded_channels,
            )
            self.pw_conv4 = nn.Conv2d(hidden_channels, channels, kernel_size=1)

            self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the forward pass for this block."""
            residual = x
            x = self.norm1(x)
            x = self.pw_conv1(x)
            x = self.dw_conv1(x)
            x = self.gate(x)
            x = self.pw_conv2(x)
            x = residual + x * self.beta

            residual = x
            x = self.norm2(x)
            x = self.pw_conv3(x)
            x = self.dw_conv2(x)
            x = self.gate(x)
            x = self.pw_conv4(x)
            return residual + x * self.gamma

    class _ResidualGroup(nn.Module):
        def __init__(self, channels: int, block_count: int) -> None:
            """Initialize the object with the provided settings."""
            super().__init__()
            self.blocks = nn.Sequential(*[_NAFBlock(channels) for _ in range(block_count)])
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the forward pass for this block."""
            return x + self.conv(self.blocks(x))

    class NAFNet(nn.Module):
        def __init__(self, channels: int, features: int) -> None:
            """Initialize the object with the provided settings."""
            super().__init__()

            self.intro = nn.Conv2d(channels, features, kernel_size=3, padding=1)

            self.enc1 = _ResidualGroup(features, num_blocks)
            self.down1 = nn.Conv2d(features, features * 2, kernel_size=2, stride=2)

            self.enc2 = _ResidualGroup(features * 2, num_blocks)
            self.down2 = nn.Conv2d(features * 2, features * 4, kernel_size=2, stride=2)

            self.middle = _ResidualGroup(features * 4, middle_blocks)

            self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
            self.dec2 = _ResidualGroup(features * 4, num_blocks)
            self.dec2_reduce = nn.Conv2d(features * 4, features * 2, kernel_size=1)

            self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
            self.dec1 = _ResidualGroup(features * 2, num_blocks)
            self.dec1_reduce = nn.Conv2d(features * 2, features, kernel_size=1)

            self.out_conv = nn.Conv2d(features, channels, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the forward pass for this block."""
            shallow = self.intro(x)

            enc1 = self.enc1(shallow)
            enc2 = self.enc2(self.down1(enc1))
            middle = self.middle(self.down2(enc2))

            dec2 = self.up2(middle)
            dec2 = torch.cat([dec2, enc2], dim=1)
            dec2 = self.dec2_reduce(self.dec2(dec2))

            dec1 = self.up1(dec2)
            dec1 = torch.cat([dec1, enc1], dim=1)
            dec1 = self.dec1_reduce(self.dec1(dec1))

            residual = self.out_conv(dec1)
            return torch.clamp(x - residual, 0.0, 1.0)

    return NAFNet(in_channels, base_channels)


class NAFNetDenoiser(BaseDenoiser):
    def __init__(
        self,
        base_channels: int = 32,
        num_blocks: int = 2,
        middle_blocks: int = 4,
        expansion: int = 2,
        device: str = 'auto',
        show_architecture: bool = False,
        model_path: Optional[str] = None,
    ) -> None:
        # Resolve model path: allow override (e.g., real-world pretrained weights)
        """Initialize the object with the provided settings."""
        if model_path is None:
            resolved_path = Path(__file__).resolve().parents[3] / 'models' / 'weights' / 'nafnet.pth'
        else:
            resolved_path = Path(model_path)
        super().__init__(
            model_path=str(resolved_path),
            base_channels=base_channels,
            num_blocks=num_blocks,
            middle_blocks=middle_blocks,
            expansion=expansion,
            device=device,
            show_architecture=show_architecture,
        )
        self.model_path = resolved_path
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.middle_blocks = middle_blocks
        self.expansion = expansion
        self.device = device
        self.show_architecture = show_architecture

        self._model = None
        self._model_channels: Optional[int] = None
        self._architecture_printed = False

    def _infer_checkpoint_channels(self, state_dict: dict) -> int:
        """Infer the channel count from a checkpoint state dict."""
        intro_key = 'intro.weight'
        out_key = 'out_conv.weight'

        if intro_key not in state_dict or out_key not in state_dict:
            raise KeyError(
                "Checkpoint missing required keys for channel inference: "
                f"'{intro_key}' and '{out_key}'"
            )

        in_channels = int(state_dict[intro_key].shape[1])
        out_channels = int(state_dict[out_key].shape[0])
        if in_channels != out_channels:
            raise ValueError(
                "Checkpoint appears inconsistent: "
                f"in_channels={in_channels}, out_channels={out_channels}"
            )

        return in_channels

    def _resolve_device(self):
        """Resolve the best available device for inference."""
        import torch

        if self.device != 'auto':
            return torch.device(self.device)

        if torch.cuda.is_available():
            return torch.device('cuda')

        return torch.device('cpu')

    def _load_model(self, channels: int):
        """Load the model checkpoint and move it to the target device."""
        import torch

        if self._model is not None and self._model_channels == channels:
            return self._model

        checkpoint_path = self.model_path
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

        model_channels = self._infer_checkpoint_channels(state_dict)
        base_channels = int(checkpoint.get('base_channels', self.base_channels)) if isinstance(checkpoint, dict) else self.base_channels
        num_blocks = int(checkpoint.get('num_blocks', self.num_blocks)) if isinstance(checkpoint, dict) else self.num_blocks
        middle_blocks = int(checkpoint.get('middle_blocks', self.middle_blocks)) if isinstance(checkpoint, dict) else self.middle_blocks
        expansion = int(checkpoint.get('expansion', self.expansion)) if isinstance(checkpoint, dict) else self.expansion

        model = _build_nafnet(
            in_channels=model_channels,
            base_channels=base_channels,
            num_blocks=num_blocks,
            middle_blocks=middle_blocks,
            expansion=expansion,
        )

        model.load_state_dict(state_dict)
        model.to(self._resolve_device())
        model.eval()

        if self.show_architecture and not self._architecture_printed:
            print(f"\nArchitecture for {self.__class__.__name__}:\n{model}\n")
            self._architecture_printed = True

        self._model = model
        self._model_channels = model_channels
        return model

    def denoise(self, noisy_image: np.ndarray) -> np.ndarray:
        """Denoise the provided image and return the result."""
        import torch
        import torch.nn.functional as F

        if noisy_image.dtype not in (np.float32, np.float64):
            noisy_image = noisy_image.astype(np.float32) / 255.0

        noisy_image = np.clip(noisy_image, 0.0, 1.0).astype(np.float32)

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
        model_channels = self._model_channels if self._model_channels is not None else channels
        device = self._resolve_device()
        tensor = tensor.to(device)

        if channels != model_channels:
            if channels == 1 and model_channels == 3:
                tensor = tensor.repeat(1, 3, 1, 1)
            elif channels == 3 and model_channels == 1:
                tensor = tensor.mean(dim=1, keepdim=True)
            else:
                raise ValueError(
                    f"Unsupported channel conversion from {channels} to {model_channels}"
                )

        # Ensures the input height/width are multiples of 8 by reflect-padding the right/bottom edges
        def _forward_with_padding(input_tensor: torch.Tensor) -> torch.Tensor:
            """Perform the forward with padding helper step."""
            _, _, in_h, in_w = input_tensor.shape
            pad_h = (4 - (in_h % 4)) % 4
            pad_w = (4 - (in_w % 4)) % 4
            if pad_h > 0 or pad_w > 0:
                input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')

            pred = model(input_tensor)
            if pad_h > 0 or pad_w > 0:
                pred = pred[:, :, :in_h, :in_w]
            return pred

        # Breaks large images into overlapping tiles to fit GPU memory and blends overlaps
        def _tiled_inference(input_tensor: torch.Tensor, tile_size: int = 512, overlap: int = 32) -> torch.Tensor:
            """Perform the tiled inference helper step."""
            _, _, full_h, full_w = input_tensor.shape
            if full_h <= tile_size and full_w <= tile_size:
                return _forward_with_padding(input_tensor)

            stride = max(1, tile_size - overlap)
            ys = list(range(0, max(full_h - tile_size, 0) + 1, stride))
            xs = list(range(0, max(full_w - tile_size, 0) + 1, stride))

            last_y = max(full_h - tile_size, 0)
            last_x = max(full_w - tile_size, 0)
            if not ys or ys[-1] != last_y:
                ys.append(last_y)
            if not xs or xs[-1] != last_x:
                xs.append(last_x)

            output = torch.zeros_like(input_tensor)
            weight = torch.zeros_like(input_tensor)

            for y in ys:
                for x in xs:
                    tile = input_tensor[:, :, y:y + tile_size, x:x + tile_size]
                    pred = _forward_with_padding(tile)
                    tile_h, tile_w = tile.shape[-2:]
                    output[:, :, y:y + tile_h, x:x + tile_w] += pred
                    weight[:, :, y:y + tile_h, x:x + tile_w] += 1.0

            return output / weight.clamp_min(1e-6)

        with torch.no_grad():
            output = _tiled_inference(tensor)

        if channels != model_channels:
            if channels == 1 and model_channels == 3:
                output = output.mean(dim=1, keepdim=True)
            elif channels == 3 and model_channels == 1:
                output = output.repeat(1, 3, 1, 1)

        output_np = output.squeeze(0).detach().cpu().numpy()
        if is_grayscale:
            denoised = output_np.squeeze(0)
        else:
            denoised = np.transpose(output_np, (1, 2, 0))

        return np.clip(denoised, 0.0, 1.0)