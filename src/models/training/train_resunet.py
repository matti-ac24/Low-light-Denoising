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


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from denoiser.algorithms.resunet_denoiser import _build_resunet


class NoisyPatchDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        patch_size: int,
        patches_per_image: int,
        sigma: float,
        channels: int,
    ) -> None:
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.sigma = sigma
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

        noise = torch.randn_like(clean_tensor) * self.sigma
        noisy_tensor = torch.clamp(clean_tensor + noise, 0.0, 1.0)

        return noisy_tensor, clean_tensor


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != 'auto':
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Residual U-Net denoiser')

    parser.add_argument('--dataset-path', type=str, required=True, help='Path to clean images folder')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--patch-size', type=int, default=128, help='Random patch size')
    parser.add_argument('--patches-per-image', type=int, default=32, help='Patches sampled per image per epoch')
    parser.add_argument('--sigma', type=float, default=0.1, help='Noise std-dev in [0, 1]')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--channels', type=int, choices=[1, 3], default=3, help='Model input channels')
    parser.add_argument('--base-channels', type=int, default=32, help='ResUNet base feature channels')
    parser.add_argument('--device', type=str, default='auto', help="auto | cpu | cuda")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--save-path',
        type=str,
        default=str((Path(__file__).resolve().parents[1] / 'weights' / 'resunet.pth')),
        help='Output checkpoint path',
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    dataset = NoisyPatchDataset(
        image_paths=image_paths,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        sigma=args.sigma,
        channels=args.channels,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = _build_resunet(in_channels=args.channels, base_channels=args.base_channels).to(device)
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
        'sigma': args.sigma,
        'patch_size': args.patch_size,
        'epochs': args.epochs,
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to: {save_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
