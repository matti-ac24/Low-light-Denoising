from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from skimage import io, img_as_float
from skimage.color import rgb2gray
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from denoiser.algorithms.nafnet_denoiser import _build_nafnet


class NoisyPatchDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        patch_size: int,
        patches_per_image: int,
        sigma_min: float,
        sigma_max: float,
        channels: int,
    ) -> None:
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.channels = channels

        self.images = [self._load_image(path) for path in self.image_paths]

    def _load_image(self, image_path: Path) -> np.ndarray:
        image = img_as_float(io.imread(image_path)).astype(np.float32)

        if self.channels == 1:
            if image.ndim == 3:
                image = rgb2gray(image).astype(np.float32)
            if image.ndim != 2:
                raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")
            image = np.expand_dims(image, axis=0)
        else:
            if image.ndim == 2:
                image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=-1)
            if image.ndim != 3 or image.shape[2] < 3:
                raise ValueError(f"Unsupported image shape for RGB conversion: {image.shape}")
            image = image[:, :, :3]
            image = np.transpose(image, (2, 0, 1))

        return np.clip(image, 0.0, 1.0)

    def __len__(self) -> int:
        return len(self.images) * self.patches_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index % len(self.images)]
        channels, height, width = image.shape

        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"Image is smaller than patch size {self.patch_size}: got {height}x{width}"
            )

        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)

        clean_patch = image[:, top:top + self.patch_size, left:left + self.patch_size]
        clean_tensor = torch.from_numpy(clean_patch).float()

        sigma_value = random.uniform(self.sigma_min, self.sigma_max)
        noise = torch.randn_like(clean_tensor) * sigma_value
        noisy_tensor = torch.clamp(clean_tensor + noise, 0.0, 1.0)

        return noisy_tensor, clean_tensor


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != 'auto':
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def resolve_sigma_range(args: argparse.Namespace) -> tuple[float, float]:
    if args.sigma is not None:
        if args.sigma < 0 or args.sigma > 1:
            raise ValueError('--sigma must be in [0, 1]')
        return float(args.sigma), float(args.sigma)

    values = [item.strip() for item in args.sigma_range.split(',') if item.strip()]
    if len(values) != 2:
        raise ValueError('--sigma-range must be in the format min,max')

    sigma_min = float(values[0])
    sigma_max = float(values[1])

    if sigma_min < 0 or sigma_max < 0 or sigma_min > 1 or sigma_max > 1:
        raise ValueError('--sigma-range values must be in [0, 1]')
    if sigma_max < sigma_min:
        raise ValueError('--sigma-range max must be >= min')

    return sigma_min, sigma_max


def describe_model(model: torch.nn.Module, channels: int, patch_size: int, device: torch.device) -> str:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    leaf_layers = sum(1 for module in model.modules() if len(list(module.children())) == 0)

    with torch.no_grad():
        example = torch.zeros(1, channels, patch_size, patch_size, device=device)
        output = model(example)

    lines = [
        f'Model: {model.__class__.__name__}',
        f'Input shape: {(1, channels, patch_size, patch_size)}',
        f'Output shape: {tuple(output.shape)}',
        f'Total parameters: {total_params:,}',
        f'Trainable parameters: {trainable_params:,}',
        f'Leaf layers: {leaf_layers}',
        '',
        'Architecture:',
        str(model),
    ]
    return '\n'.join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train NAFNet denoiser')

    parser.add_argument('--dataset-path', type=str, required=True, help='Path to clean images folder')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--patch-size', type=int, default=128, help='Random patch size')
    parser.add_argument('--patches-per-image', type=int, default=32, help='Patches sampled per image per epoch')
    parser.add_argument(
        '--sigma',
        type=float,
        default=None,
        help='Fixed noise std-dev in [0, 1] (overrides --sigma-range)',
    )
    parser.add_argument(
        '--sigma-range',
        type=str,
        default='0.02,0.20',
        help='Noise range as min,max when sampling per patch (default: 0.02,0.20)',
    )
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--channels', type=int, choices=[1, 3], default=3, help='Model input channels')
    parser.add_argument('--base-channels', type=int, default=32, help='NAFNet base feature channels')
    parser.add_argument('--num-blocks', type=int, default=2, help='Residual blocks per encoder/decoder stage')
    parser.add_argument('--middle-blocks', type=int, default=4, help='Residual blocks in the bottleneck')
    parser.add_argument('--expansion', type=int, default=2, help='Feed-forward expansion factor')
    parser.add_argument('--device', type=str, default='auto', help='auto | cpu | cuda')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--save-path',
        type=str,
        default=str((Path(__file__).resolve().parents[1] / 'weights' / 'nafnet.pth')),
        help='Output checkpoint path',
    )
    parser.add_argument(
        '--initial-weights',
        type=str,
        default=None,
        help='Path to pre-trained weights checkpoint for transfer learning',
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sigma_min, sigma_max = resolve_sigma_range(args)

    dataset_dir = Path(args.dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_dir}")

    image_paths = sorted(
        [
            path
            for path in dataset_dir.rglob('*')
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        ]
    )

    if not image_paths:
        raise ValueError(f"No supported image files found in: {dataset_dir}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Found {len(image_paths)} training image(s)")
    if sigma_min == sigma_max:
        print(f"Training with fixed sigma: {sigma_min:.4f}")
    else:
        print(f"Training with sigma sampled per patch from [{sigma_min:.4f}, {sigma_max:.4f}]")

    dataset = NoisyPatchDataset(
        image_paths=image_paths,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        channels=args.channels,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = _build_nafnet(
        in_channels=args.channels,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
        middle_blocks=args.middle_blocks,
        expansion=args.expansion,
    ).to(device)

    # Load pre-trained weights if provided (transfer learning)
    if args.initial_weights:
        checkpoint = torch.load(args.initial_weights, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded pre-trained weights from: {args.initial_weights}')

    model_description = describe_model(model, channels=args.channels, patch_size=args.patch_size, device=device)
    print('\n' + model_description + '\n')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.L1Loss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch:03d}/{args.epochs:03d}",
            unit='batch',
            leave=False,
        )

        for noisy_batch, clean_batch in progress:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)

            optimizer.zero_grad()
            denoised_batch = model(noisy_batch)
            loss = criterion(denoised_batch, clean_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(batch_loss=f"{loss.item():.6f}")

        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch:03d}/{args.epochs:03d} - L1 Loss: {avg_loss:.6f}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'state_dict': model.state_dict(),
        'channels': args.channels,
        'base_channels': args.base_channels,
        'num_blocks': args.num_blocks,
        'middle_blocks': args.middle_blocks,
        'expansion': args.expansion,
        'sigma': args.sigma,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'sigma_range': [sigma_min, sigma_max],
        'patch_size': args.patch_size,
        'epochs': args.epochs,
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to: {save_path}")

    architecture_path = save_path.with_name(f"{save_path.stem}_architecture.txt")
    architecture_path.write_text(model_description + '\n', encoding='utf-8')
    print(f"Saved architecture summary to: {architecture_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())