from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
import sys
from pathlib import Path


SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
SIDD_PAIR_RE = re.compile(r'^(?P<scene>\d+)_(?P<tag>NOISY|GT)_(?P<rest>.+)$', re.IGNORECASE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare SIDD Medium for scene-level training and evaluation splits.'
    )
    parser.add_argument(
        '--source-root',
        type=str,
        required=True,
        help='Path to the raw SIDD Medium root that contains Data/ and Scene_Instances.txt',
    )
    parser.add_argument(
        '--output-root',
        type=str,
        required=True,
        help='Path where the prepared train/test dataset will be written',
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Fraction of scene instances to place in the training split',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed used before shuffling scene instances',
    )
    parser.add_argument(
        '--link-mode',
        choices=['symlink', 'hardlink', 'copy'],
        default='symlink',
        help='How files are materialized in the prepared dataset',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Remove the output directory first if it already exists',
    )
    return parser.parse_args(argv)


def _load_scene_names(source_root: Path) -> list[str]:
    scene_list_file = source_root / 'Scene_Instances.txt'
    if scene_list_file.exists():
        scene_names = [line.strip() for line in scene_list_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        if scene_names:
            return scene_names

    data_dir = source_root / 'Data'
    if not data_dir.exists():
        raise FileNotFoundError(f'Data directory not found: {data_dir}')

    scene_names = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    if not scene_names:
        raise ValueError(f'No scene directories found in: {data_dir}')
    return scene_names


def _discover_kaggle_source_root() -> Path:
    kaggle_input_root = Path('/kaggle/input')
    if not kaggle_input_root.exists():
        raise FileNotFoundError('Kaggle input directory not found at /kaggle/input')

    candidates: list[Path] = []
    for root in sorted(kaggle_input_root.iterdir()):
        if not root.is_dir():
            continue
        if (root / 'Data').exists() and (root / 'Scene_Instances.txt').exists():
            candidates.append(root)
            continue

        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / 'Data').exists() and (child / 'Scene_Instances.txt').exists():
                candidates.append(child)

    if not candidates:
        raise FileNotFoundError(
            'Could not auto-detect the SIDD Medium root under /kaggle/input. '
            'Set source_root manually and call prepare_sidd_medium(...).'
        )

    if len(candidates) > 1:
        print('Warning: multiple SIDD-like roots found under /kaggle/input; using the first one:')
        for candidate in candidates:
            print(f'  - {candidate}')

    return candidates[0]


def _in_notebook() -> bool:
    try:
        shell = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return False
    return shell is not None


def _split_scene_names(scene_names: list[str], train_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError('--train-ratio must be strictly between 0 and 1')

    rng = random.Random(seed)
    shuffled = scene_names[:]
    rng.shuffle(shuffled)

    train_count = int(round(len(shuffled) * train_ratio))
    train_count = max(1, min(len(shuffled) - 1, train_count))

    return shuffled[:train_count], shuffled[train_count:]


def _collect_scene_pairs(scene_dir: Path) -> list[tuple[Path, Path, str]]:
    """Collect NOISY/GT pairs using SIDD-specific naming patterns (e.g., 0078_GT_SRGB_010.PNG)."""
    noisy_files: dict[str, Path] = {}
    gt_files: dict[str, Path] = {}

    for path in sorted(scene_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue

        match = SIDD_PAIR_RE.match(path.name)
        if not match:
            continue

        key = f"{match.group('scene')}_{match.group('rest')}"
        tag = match.group('tag').upper()

        if tag == 'NOISY':
            noisy_files[key] = path
        elif tag == 'GT':
            gt_files[key] = path

    shared_keys = sorted(noisy_files.keys() & gt_files.keys())
    if not shared_keys:
        raise ValueError(
            f'No paired NOISY/GT PNGs found in: {scene_dir}. '
            f'Expected SIDD naming like: 0078_GT_SRGB_010.PNG, 0078_NOISY_SRGB_010.PNG'
        )

    missing_noisy = sorted(gt_files.keys() - noisy_files.keys())
    missing_gt = sorted(noisy_files.keys() - gt_files.keys())
    if missing_noisy:
        print(f'Warning: {len(missing_noisy)} GT file(s) have no matching NOISY file in {scene_dir.name}')
    if missing_gt:
        print(f'Warning: {len(missing_gt)} NOISY file(s) have no matching GT file in {scene_dir.name}')

    return [(noisy_files[key], gt_files[key], key) for key in shared_keys]


def _ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f'Output directory already exists: {path}. Re-run with --overwrite to replace it.'
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _materialize_file(source: Path, destination: Path, link_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if link_mode == 'copy':
        shutil.copy2(source, destination)
        return

    if link_mode == 'hardlink':
        os.link(source, destination)
        return

    relative_target = os.path.relpath(source, start=destination.parent)
    destination.symlink_to(relative_target)


def _prepare_split(
    split_name: str,
    scene_names: list[str],
    source_data_dir: Path,
    output_root: Path,
    link_mode: str,
    manifest_rows: list[dict[str, str]],
) -> None:
    split_root = output_root / split_name
    paired_root = split_root
    clean_root = split_root / 'clean'
    noisy_root = split_root / 'noisy'

    paired_root.mkdir(parents=True, exist_ok=True)
    clean_root.mkdir(parents=True, exist_ok=True)
    noisy_root.mkdir(parents=True, exist_ok=True)

    for scene_name in scene_names:
        scene_dir = source_data_dir / scene_name
        if not scene_dir.exists():
            raise FileNotFoundError(f'Scene directory not found: {scene_dir}')

        pairs = _collect_scene_pairs(scene_dir)
        scene_output_dir = paired_root / scene_name
        scene_output_dir.mkdir(parents=True, exist_ok=True)

        for noisy_path, gt_path, pair_key in pairs:
            common_name = f'{scene_name}__{pair_key}'
            noisy_ext = noisy_path.suffix.lower()
            gt_ext = gt_path.suffix.lower()
            if noisy_ext != gt_ext:
                raise ValueError(
                    f'Paired files use different extensions in {scene_dir.name}: {noisy_path.name} vs {gt_path.name}'
                )

            # Rename to match trainer expectations: files must start with NOISY_ or GT_
            paired_noisy_name = f'NOISY_{pair_key}{noisy_ext}'
            paired_gt_name = f'GT_{pair_key}{gt_ext}'
            clean_name = f'{common_name}{gt_ext}'

            _materialize_file(noisy_path, scene_output_dir / paired_noisy_name, link_mode)
            _materialize_file(gt_path, scene_output_dir / paired_gt_name, link_mode)

            _materialize_file(gt_path, clean_root / clean_name, link_mode)
            _materialize_file(noisy_path, noisy_root / clean_name, link_mode)

            manifest_rows.append(
                {
                    'split': split_name,
                    'scene_name': scene_name,
                    'pair_name': clean_name,
                    'source_noisy': str(noisy_path),
                    'source_gt': str(gt_path),
                    'paired_noisy': str(scene_output_dir / paired_noisy_name),
                    'paired_gt': str(scene_output_dir / paired_gt_name),
                    'clean_path': str(clean_root / clean_name),
                    'noisy_path': str(noisy_root / clean_name),
                }
            )


def prepare_sidd_medium(
    source_root: str | Path,
    output_root: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    link_mode: str = 'symlink',
    overwrite: bool = False,
) -> Path:
    """Prepare SIDD Medium dataset with scene-level 80/20 split and trainer-compatible output structure."""
    source_root_path = Path(source_root).expanduser().resolve()
    output_root_path = Path(output_root).expanduser().resolve()

    if not source_root_path.exists():
        raise FileNotFoundError(f'Source root not found: {source_root_path}')

    data_dir = source_root_path / 'Data'
    if not data_dir.exists():
        raise FileNotFoundError(f'Data directory not found: {data_dir}')

    _ensure_empty_dir(output_root_path, overwrite=overwrite)

    scene_names = _load_scene_names(source_root_path)
    train_scenes, test_scenes = _split_scene_names(scene_names, train_ratio, seed)

    manifest_rows: list[dict[str, str]] = []
    _prepare_split('train', train_scenes, data_dir, output_root_path, link_mode, manifest_rows)
    _prepare_split('test', test_scenes, data_dir, output_root_path, link_mode, manifest_rows)

    manifest_path = output_root_path / 'split_manifest.csv'
    with manifest_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'split',
                'scene_name',
                'pair_name',
                'source_noisy',
                'source_gt',
                'paired_noisy',
                'paired_gt',
                'clean_path',
                'noisy_path',
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f'Prepared SIDD Medium split at: {output_root_path}')
    print(f'Training scenes: {len(train_scenes)}')
    print(f'Test scenes: {len(test_scenes)}')
    print(f'Paired rows written: {len(manifest_rows)}')
    print(f'Manifest: {manifest_path}')
    print()
    print('Output structure (ready for NAFNet and Restormer trainers):')
    print('  train/')
    print('    <scene_id>/           # e.g., 0001_001_S6_00100_00060_3200_L/')
    print('      NOISY_SRGB_*.PNG')
    print('      GT_SRGB_*.PNG')
    print('    clean/               # Also generated for real-world evaluation')
    print('    noisy/')
    print('  test/')
    print('Use the root directory with the existing trainers, and use <root>/test for real-world evaluation.')

    return output_root_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prepare_sidd_medium(
        source_root=args.source_root,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )

    return 0


if __name__ == '__main__':
    if any(arg.startswith('--source-root') or arg.startswith('--output-root') for arg in sys.argv[1:]):
        raise SystemExit(main())
    if _in_notebook():
        source_root = _discover_kaggle_source_root()
        output_root = Path('/kaggle/working/sidd_medium_prepared')
        prepare_sidd_medium(
            source_root=source_root,
            output_root=output_root,
            train_ratio=0.8,
            seed=42,
            link_mode='symlink',
            overwrite=True,
        )
        print()
        print('✓ Ready to train! Next cell example:')
        print()
        print('# For NAFNet:')
        print('!python /kaggle/working/models/training/train_nafnet.py \\')
        print('    --dataset-path /kaggle/working/sidd_medium_prepared \\')
        print('    --epochs 20 \\')
        print('    --batch-size 8 \\')
        print('    --patch-size 128 \\')
        print('    --device auto \\')
        print('    --initial-weights /kaggle/working/models/weights/nafnet.pth')
        print()
        print('# For Restormer:')
        print('!python /kaggle/working/models/training/train_restormer.py \\')
        print('    --dataset-path /kaggle/working/sidd_medium_prepared \\')
        print('    --epochs 20 \\')
        print('    --batch-size 8 \\')
        print('    --patch-size 128 \\')
        print('    --device auto \\')
        print('    --initial-weights /kaggle/working/models/weights/restormer.pth')
    else:
        print('Run this script with --source-root and --output-root, or execute it inside a Kaggle notebook cell.')
