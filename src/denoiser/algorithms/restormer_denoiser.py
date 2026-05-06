# Restormer denoising algorithm

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseDenoiser


def _build_restormer(
    in_channels: int,
    base_channels: int = 32,
    num_blocks: tuple[int, int, int, int] = (2, 2, 4, 2),
    heads: tuple[int, int, int, int] = (1, 2, 4, 8),
    ffn_expansion: float = 2.66,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _LayerNorm2d(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias)
            return x.permute(0, 3, 1, 2)

    class _FeedForward(nn.Module):
        def __init__(self, dim: int, expansion: float) -> None:
            super().__init__()
            hidden = int(dim * expansion)
            self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False)
            self.dwconv = nn.Conv2d(
                hidden * 2,
                hidden * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden * 2,
                bias=False,
            )
            self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.project_in(x)
            x1, x2 = self.dwconv(x).chunk(2, dim=1)
            x = torch.nn.functional.gelu(x1) * x2
            return self.project_out(x)

    class _Attention(nn.Module):
        def __init__(self, dim: int, num_heads: int) -> None:
            super().__init__()
            if dim % num_heads != 0:
                raise ValueError(f'dim={dim} must be divisible by num_heads={num_heads}')

            self.num_heads = num_heads
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
            self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
            self.qkv_dwconv = nn.Conv2d(
                dim * 3,
                dim * 3,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=dim * 3,
                bias=False,
            )
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, h, w = x.shape
            qkv = self.qkv_dwconv(self.qkv(x))
            q, k, v = qkv.chunk(3, dim=1)

            head_dim = c // self.num_heads
            q = q.view(b, self.num_heads, head_dim, h * w)
            k = k.view(b, self.num_heads, head_dim, h * w)
            v = v.view(b, self.num_heads, head_dim, h * w)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)

            out = attn @ v
            out = out.view(b, c, h, w)
            return self.project_out(out)

    class _TransformerBlock(nn.Module):
        def __init__(self, dim: int, num_heads: int, expansion: float) -> None:
            super().__init__()
            self.norm1 = _LayerNorm2d(dim)
            self.attn = _Attention(dim, num_heads)
            self.norm2 = _LayerNorm2d(dim)
            self.ffn = _FeedForward(dim, expansion)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
            return x

    class _OverlapPatchEmbed(nn.Module):
        def __init__(self, inp_channels: int, dim: int) -> None:
            super().__init__()
            self.proj = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    class _Downsample(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.body = nn.Conv2d(channels, channels // 2, kernel_size=3, stride=1, padding=1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pixel_unshuffle(self.body(x), 2)

    class _Upsample(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.body = nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pixel_shuffle(self.body(x), 2)

    class Restormer(nn.Module):
        def __init__(
            self,
            channels: int,
            dim: int,
            blocks: tuple[int, int, int, int],
            stage_heads: tuple[int, int, int, int],
            expansion: float,
        ) -> None:
            super().__init__()

            self.patch_embed = _OverlapPatchEmbed(channels, dim)

            self.encoder_level1 = nn.Sequential(
                *[_TransformerBlock(dim, stage_heads[0], expansion) for _ in range(blocks[0])]
            )
            self.down1_2 = _Downsample(dim)

            self.encoder_level2 = nn.Sequential(
                *[_TransformerBlock(dim * 2, stage_heads[1], expansion) for _ in range(blocks[1])]
            )
            self.down2_3 = _Downsample(dim * 2)

            self.encoder_level3 = nn.Sequential(
                *[_TransformerBlock(dim * 4, stage_heads[2], expansion) for _ in range(blocks[2])]
            )
            self.down3_4 = _Downsample(dim * 4)

            self.latent = nn.Sequential(
                *[_TransformerBlock(dim * 8, stage_heads[3], expansion) for _ in range(blocks[3])]
            )

            self.up4_3 = _Upsample(dim * 8)
            self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=False)
            self.decoder_level3 = nn.Sequential(
                *[_TransformerBlock(dim * 4, stage_heads[2], expansion) for _ in range(blocks[2])]
            )

            self.up3_2 = _Upsample(dim * 4)
            self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=False)
            self.decoder_level2 = nn.Sequential(
                *[_TransformerBlock(dim * 2, stage_heads[1], expansion) for _ in range(blocks[1])]
            )

            self.up2_1 = _Upsample(dim * 2)
            self.decoder_level1 = nn.Sequential(
                *[_TransformerBlock(dim * 2, stage_heads[0], expansion) for _ in range(blocks[0])]
            )
            self.refinement = nn.Sequential(
                _TransformerBlock(dim * 2, stage_heads[0], expansion),
                _TransformerBlock(dim * 2, stage_heads[0], expansion),
            )

            self.output = nn.Conv2d(dim * 2, channels, kernel_size=3, stride=1, padding=1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            inp = x

            level1 = self.encoder_level1(self.patch_embed(inp))
            level2 = self.encoder_level2(self.down1_2(level1))
            level3 = self.encoder_level3(self.down2_3(level2))
            latent = self.latent(self.down3_4(level3))

            level3_dec = self.up4_3(latent)
            level3_dec = torch.cat([level3_dec, level3], dim=1)
            level3_dec = self.decoder_level3(self.reduce_chan_level3(level3_dec))

            level2_dec = self.up3_2(level3_dec)
            level2_dec = torch.cat([level2_dec, level2], dim=1)
            level2_dec = self.decoder_level2(self.reduce_chan_level2(level2_dec))

            level1_dec = self.up2_1(level2_dec)
            level1_dec = torch.cat([level1_dec, level1], dim=1)
            level1_dec = self.decoder_level1(level1_dec)
            level1_dec = self.refinement(level1_dec)

            residual = self.output(level1_dec)
            return torch.clamp(inp - residual, 0.0, 1.0)

    if len(num_blocks) != 4:
        raise ValueError('num_blocks must contain 4 values, one per stage')
    if len(heads) != 4:
        raise ValueError('heads must contain 4 values, one per stage')

    return Restormer(
        channels=in_channels,
        dim=base_channels,
        blocks=num_blocks,
        stage_heads=heads,
        expansion=ffn_expansion,
    )


class RestormerDenoiser(BaseDenoiser):
    def __init__(
        self,
        base_channels: int = 32,
        num_blocks: tuple[int, int, int, int] = (2, 2, 4, 2),
        heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        ffn_expansion: float = 2.66,
        device: str = 'auto',
        show_architecture: bool = False,
        model_path: Optional[str] = None,
    ) -> None:
        # Resolve model path: allow override (e.g., real-world pretrained weights)
        if model_path is None:
            resolved_path = Path(__file__).resolve().parents[3] / 'models' / 'weights' / 'restormer.pth'
        else:
            resolved_path = Path(model_path)
        super().__init__(
            model_path=str(resolved_path),
            base_channels=base_channels,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion=ffn_expansion,
            device=device,
            show_architecture=show_architecture,
        )
        self.model_path = resolved_path
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.heads = heads
        self.ffn_expansion = ffn_expansion
        self.device = device
        self.show_architecture = show_architecture

        self._model = None
        self._model_channels: Optional[int] = None
        self._architecture_printed = False

    def _infer_checkpoint_channels(self, state_dict: dict) -> int:
        in_key = 'patch_embed.proj.weight'
        out_key = 'output.weight'

        if in_key not in state_dict or out_key not in state_dict:
            raise KeyError(
                "Checkpoint missing required keys for channel inference: "
                f"'{in_key}' and '{out_key}'"
            )

        in_channels = int(state_dict[in_key].shape[1])
        out_channels = int(state_dict[out_key].shape[0])
        if in_channels != out_channels:
            raise ValueError(
                'Checkpoint appears inconsistent: '
                f'in_channels={in_channels}, out_channels={out_channels}'
            )

        return in_channels

    def _resolve_device(self):
        import torch

        if self.device != 'auto':
            return torch.device(self.device)

        if torch.cuda.is_available():
            return torch.device('cuda')

        return torch.device('cpu')

    def _load_model(self, channels: int):
        import torch

        if self._model is not None and self._model_channels == channels:
            return self._model

        checkpoint_path = self.model_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Model file not found: {checkpoint_path}')

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
        num_blocks = tuple(checkpoint.get('num_blocks', self.num_blocks)) if isinstance(checkpoint, dict) else self.num_blocks
        heads = tuple(checkpoint.get('heads', self.heads)) if isinstance(checkpoint, dict) else self.heads
        ffn_expansion = float(checkpoint.get('ffn_expansion', self.ffn_expansion)) if isinstance(checkpoint, dict) else self.ffn_expansion

        model = _build_restormer(
            in_channels=model_channels,
            base_channels=base_channels,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion=ffn_expansion,
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
            raise ValueError(f'Unsupported image shape: {noisy_image.shape}')

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
                    f'Unsupported channel conversion from {channels} to {model_channels}'
                )

        _, _, height, width = tensor.shape
        pad_h = (8 - (height % 8)) % 8
        pad_w = (8 - (width % 8)) % 8
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            output = model(tensor)

        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :height, :width]

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