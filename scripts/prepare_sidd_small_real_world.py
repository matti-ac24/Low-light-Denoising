from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / 'data' / 'SIDD_Small_sRGB_Only' / 'Data'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'SIDD_small_real_world' / 'test'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Reorganize SIDD Small scene folders into clean/noisy pairs for real-world evaluation.'
    )
    parser.add_argument(
        '--source-dir',
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help='Path to SIDD Small Data directory containing scene folders.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Destination directory where clean/ and noisy/ will be created.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Remove output directory first if it already exists.',
    )
    return parser.parse_args()


def _ensure_output_layout(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    clean_dir = output_dir / 'clean'
    noisy_dir = output_dir / 'noisy'

    if output_dir.exists() and not overwrite:
        if any(clean_dir.glob('*')) or any(noisy_dir.glob('*')):
            raise FileExistsError(
                f'Output directory already contains files: {output_dir}. '
                'Use --overwrite to replace it.'
            )

    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    return clean_dir, noisy_dir


def _collect_scene_pairs(scene_dir: Path) -> list[tuple[Path, Path, str]]:
    gt_map: dict[str, Path] = {}
    noisy_map: dict[str, Path] = {}

    for img_path in sorted(scene_dir.glob('*.PNG')):
        name = img_path.name.upper()
        if name.startswith('GT_SRGB_'):
            key = name.replace('GT_SRGB_', '')
            gt_map[key] = img_path
        elif name.startswith('NOISY_SRGB_'):
            key = name.replace('NOISY_SRGB_', '')
            noisy_map[key] = img_path

    shared_keys = sorted(gt_map.keys() & noisy_map.keys())
    return [(gt_map[key], noisy_map[key], key) for key in shared_keys]


def reorganize_sidd_small(source_dir: Path, output_dir: Path, overwrite: bool = False) -> Path:
    if not source_dir.exists():
        raise FileNotFoundError(f'Source directory not found: {source_dir}')

    clean_dir, noisy_dir = _ensure_output_layout(output_dir, overwrite)

    manifest_rows: list[dict[str, str]] = []
    copied_pairs = 0

    for scene_dir in sorted(source_dir.iterdir()):
        if not scene_dir.is_dir():
            continue

        pairs = _collect_scene_pairs(scene_dir)
        if not pairs:
            continue

        for gt_path, noisy_path, key in pairs:
            stem = Path(key).stem
            out_name = f'{scene_dir.name}_{stem}.png'

            clean_out = clean_dir / out_name
            noisy_out = noisy_dir / out_name

            shutil.copy2(gt_path, clean_out)
            shutil.copy2(noisy_path, noisy_out)

            manifest_rows.append(
                {
                    'scene': scene_dir.name,
                    'key': key,
                    'source_gt': str(gt_path),
                    'source_noisy': str(noisy_path),
                    'clean_out': str(clean_out),
                    'noisy_out': str(noisy_out),
                }
            )
            copied_pairs += 1

    if copied_pairs == 0:
        raise ValueError(f'No GT/NOISY pairs found under: {source_dir}')

    manifest_path = output_dir / 'manifest.csv'
    with open(manifest_path, 'w', newline='') as f:
        fieldnames = ['scene', 'key', 'source_gt', 'source_noisy', 'clean_out', 'noisy_out']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f'Prepared {copied_pairs} paired images at: {output_dir}')
    print(f'Manifest saved to: {manifest_path}')

    return output_dir


def main() -> int:
    args = parse_args()
    reorganize_sidd_small(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
