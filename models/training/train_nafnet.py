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


SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _load_image(image_path: Path, channels: int) -> np.ndarray:
    image = img_as_float(io.imread(image_path)).astype(np.float32)

    if channels == 1:
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


def _collect_paired_images(split_dir: Path) -> list[tuple[Path, Path]]:
    noisy_files: dict[str, Path] = {}
    gt_files: dict[str, Path] = {}

    for path in split_dir.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue

        upper_name = path.name.upper()
        if upper_name.startswith('NOISY'):
            noisy_files[path.name[5:]] = path
        elif upper_name.startswith('GT'):
            gt_files[path.name[2:]] = path

    paired_keys = sorted(noisy_files.keys() & gt_files.keys())
    pairs = [(noisy_files[key], gt_files[key]) for key in paired_keys]

    missing_noisy = sorted(gt_files.keys() - noisy_files.keys())
    missing_gt = sorted(noisy_files.keys() - gt_files.keys())
    if missing_noisy:
        print(f"Warning: {len(missing_noisy)} GT file(s) have no matching NOISY pair in {split_dir}")
    if missing_gt:
        print(f"Warning: {len(missing_gt)} NOISY file(s) have no matching GT pair in {split_dir}")

    return pairs


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

        # Lazy load images on demand; keep a tiny cache to limit RAM
        self._image_cache: dict[int, np.ndarray] = {}

    def _load_image(self, image_path: Path) -> np.ndarray:
        return _load_image(image_path, self.channels)

    def __len__(self) -> int:
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_idx = index % len(self.image_paths)
        if image_idx not in self._image_cache:
            # load and cache
            self._image_cache[image_idx] = self._load_image(self.image_paths[image_idx])
            # keep cache small (max 2 images), but never evict the one we just loaded
            if len(self._image_cache) > 2:
                # remove the oldest key (smallest numeric key that isn't the one we just added)
                candidates = [k for k in self._image_cache.keys() if k != image_idx]
                if candidates:
                    oldest = min(candidates)
                    try:
                        del self._image_cache[oldest]
                    except KeyError:
                        pass

        image = self._image_cache[image_idx]
        channels, height, width = image.shape

        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"Image is smaller than patch size {self.patch_size}: got {height}x{width}"
            )

        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)

        clean_patch = image[:, top:top + self.patch_size, left:left + self.patch_size]
        clean_tensor = torch.from_numpy(clean_patch).float()

        # For real-world paired data we should not add synthetic noise; sigma range will be 0 when using paired_data
        sigma_value = random.uniform(self.sigma_min, self.sigma_max)
        noise = torch.randn_like(clean_tensor) * sigma_value
        noisy_tensor = torch.clamp(clean_tensor + noise, 0.0, 1.0)

        return noisy_tensor, clean_tensor


class PairedPatchDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        patch_size: int,
        patches_per_image: int,
        channels: int,
        random_crop: bool = True,
    ) -> None:
        self.split_dir = split_dir
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.channels = channels
        self.random_crop = random_crop
        self.pairs = _collect_paired_images(split_dir)

        if not self.pairs:
            raise ValueError(f'No paired NOISY/GT images found in: {split_dir}')

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_path, gt_path = self.pairs[index % len(self.pairs)]
        noisy = _load_image(noisy_path, self.channels)
        clean = _load_image(gt_path, self.channels)

        if noisy.shape != clean.shape:
            raise ValueError(f'Mismatched pair shapes for {noisy_path.name}: {noisy.shape} vs {clean.shape}')

        _, height, width = clean.shape
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(f'Image is smaller than patch size {self.patch_size}: got {height}x{width}')

        if self.random_crop:
            top = random.randint(0, height - self.patch_size)
            left = random.randint(0, width - self.patch_size)
        else:
            top = (height - self.patch_size) // 2
            left = (width - self.patch_size) // 2

        noisy_patch = noisy[:, top:top + self.patch_size, left:left + self.patch_size]
        clean_patch = clean[:, top:top + self.patch_size, left:left + self.patch_size]

        return torch.from_numpy(noisy_patch).float(), torch.from_numpy(clean_patch).float()


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


def evaluate_loss(model: torch.nn.Module, dataloader: DataLoader, criterion: torch.nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for noisy_batch, clean_batch in dataloader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            denoised_batch = model(noisy_batch)
            total_loss += criterion(denoised_batch, clean_batch).item()

    return total_loss / max(1, len(dataloader))


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

    dataset_dir = Path(args.dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f'Dataset path not found: {dataset_dir}')

    train_split_dir = dataset_dir / 'train'
    test_split_dir = dataset_dir / 'test'
    paired_data = train_split_dir.exists() and test_split_dir.exists()

    sigma_min, sigma_max = (0.0, 0.0) if paired_data else resolve_sigma_range(args)

    device = resolve_device(args.device)
    print(f'Using device: {device}')

    if paired_data:
        train_dataset = PairedPatchDataset(
            split_dir=train_split_dir,
            patch_size=args.patch_size,
            patches_per_image=args.patches_per_image,
            channels=args.channels,
            random_crop=True,
        )
        val_dataset = PairedPatchDataset(
            split_dir=test_split_dir,
            patch_size=args.patch_size,
            patches_per_image=1,
            channels=args.channels,
            random_crop=False,
        )

        print('Using paired real-world data')
        print(f'Train split: {train_split_dir}')
        print(f'Validation split: {test_split_dir}')
        print(f'Found {len(train_dataset.pairs)} paired training image(s)')
        print(f'Found {len(val_dataset.pairs)} paired validation image(s)')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        image_paths = sorted(
            [
                path
                for path in dataset_dir.rglob('*')
                if path.suffix.lower() in SUPPORTED_EXTS
            ]
        )

        if not image_paths:
            raise ValueError(f'No supported image files found in: {dataset_dir}')

        print(f'Found {len(image_paths)} training image(s)')
        if sigma_min == sigma_max:
            print(f'Training with fixed sigma: {sigma_min:.4f}')
        else:
            print(f'Training with sigma sampled per patch from [{sigma_min:.4f}, {sigma_max:.4f}]')

        train_dataset = NoisyPatchDataset(
            image_paths=image_paths,
            patch_size=args.patch_size,
            patches_per_image=args.patches_per_image,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            channels=args.channels,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = None

    model = _build_nafnet(
        in_channels=args.channels,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
        middle_blocks=args.middle_blocks,
        expansion=args.expansion,
    ).to(device)

    if args.initial_weights:
        checkpoint = torch.load(args.initial_weights, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        model.load_state_dict(state_dict)
        print(f'Loaded pre-trained weights from: {args.initial_weights}')

    model_description = describe_model(model, channels=args.channels, patch_size=args.patch_size, device=device)
    print('\n' + model_description + '\n')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.L1Loss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f'Epoch {epoch:03d}/{args.epochs:03d}',
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
            progress.set_postfix(batch_loss=f'{loss.item():.6f}')

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f'Epoch {epoch:03d}/{args.epochs:03d} - L1 Loss: {avg_loss:.6f}')

        if val_loader is not None:
            val_loss = evaluate_loss(model, val_loader, criterion, device)
            print(f'Epoch {epoch:03d}/{args.epochs:03d} - Validation L1 Loss: {val_loss:.6f}')

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
        'paired_data': paired_data,
        'train_split_name': 'train',
        'val_split_name': 'test',
    }
    torch.save(checkpoint, save_path)
    print(f'Saved checkpoint to: {save_path}')

    architecture_path = save_path.with_name(f'{save_path.stem}_architecture.txt')
    architecture_path.write_text(model_description + '\n', encoding='utf-8')
    print(f'Saved architecture summary to: {architecture_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())