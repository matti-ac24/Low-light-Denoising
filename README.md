# Low-light Image Denoising Framework

A modular framework for evaluating and comparing image denoising algorithms. It currently includes BM3D, Non-Local Means, ResUNet, NAFNet, and Restormer.

## Features

- Multiple algorithms: BM3D, NL-Means, ResUNet, NAFNet, and Restormer
- Single-algorithm evaluation and multi-algorithm comparison
- PSNR and SSIM metrics
- Automatic plots and CSV exports
- Synthetic, real-world paired, and built-in test datasets
- Pretrained CNN-based denoisers for ResUNet and NAFNet

## Quick Start

### 1. Setup Environment

```bash
conda env create -f dependencies/environment.yaml
conda activate denoising
```

Or install manually:

```bash
pip install numpy matplotlib scikit-image bm3d torch tqdm
```

### 2. Run a First Test

```bash
cd src
python -m denoiser --test bm3d
```

## Usage

### Test a Single Algorithm

```bash
python -m denoiser --test bm3d
python -m denoiser --test nl-means
python -m denoiser --test resunet --device cpu
python -m denoiser --test nafnet --device cpu
```

### Compare Multiple Algorithms

```bash
python -m denoiser --test bm3d nl-means
python -m denoiser --test bm3d nl-means resunet nafnet --device cpu
python -m denoiser --test bm3d nl-means restormer --device cpu
```

### Dataset Modes

Built-in test images:

```bash
python -m denoiser --test bm3d
```

Synthetic noise on clean images:

```bash
python -m denoiser --synthetic bm3d --sigma 0.15
```

Real-world paired images:

```bash
python -m denoiser --real-world bm3d
```

Note: the CLI default for `--real-world` is the prepared SIDD small test set at `data/SIDD_small_real_world/test`. See "Real-world evaluation and SIDD preparation" below for how to prepare or override this.

The prepared real-world folder is generated from the raw SIDD Small source tree:

- `data/SIDD_Small_sRGB_Only/` is the original SIDD Small dataset source. It is not read by the runtime CLI directly; it is used by the preparation script below.
- `data/SIDD_small_real_world/` is the flattened runtime layout consumed by `--real-world`.

The synthetic benchmark folders follow the same pattern:

- `data/benchmark/clean/` is the runtime input used by `--synthetic` and the training scripts.
- `data/external/BSDS500-master/` is the raw BSDS500 source tree kept for reference and dataset preparation; the code reads benchmark images from the prepared `data/benchmark/clean/` layout, not from this raw external copy.

### Algorithm Parameters

BM3D:

```bash
python -m denoiser --test bm3d --sigma 0.15
```

NL-Means:

```bash
python -m denoiser --test nl-means --patch-size 7 --patch-distance 8 --sigma 0.12
```

ResUNet:

```bash
python -m denoiser --test resunet --base-channels 32 --device cpu
python -m denoiser --test resunet --device cpu --show-architecture
```

NAFNet:

```bash
python -m denoiser --test nafnet --base-channels 32 --device cpu
python -m denoiser --test nafnet --device cpu --show-architecture
```

Restormer:

```bash
python -m denoiser --test restormer --base-channels 32 --device cpu
python -m denoiser --test restormer --device cpu --show-architecture
```

### Comparison Mode

```bash
python -m denoiser --synthetic --compare bm3d nl-means resunet nafnet \
    --output results/compare
```

This writes to a canonical comparison folder (algorithm order does not matter), for example:

`results/compare/bm3d_vs_nafnet_vs_nl-means_vs_resunet/synthetic/`

If you only want to rebuild the representative side-by-side image from already saved comparison outputs, use `--rep-only`:

```bash
python -m denoiser --real-world --compare nafnet restormer \
    --rep-only \
    --rep-seed 19 \
    --verbose
```

What it does:
- `--rep-only` skips the denoising/evaluation pass and rebuilds only the representative comparison image from outputs already present under the comparison folder.
- `--rep-seed 19` makes the image selection deterministic. Changing the seed changes which cached image is chosen as the representative example.

### Sigma Range Mode

```bash
python -m denoiser --synthetic --compare bm3d nl-means resunet nafnet \
    --sigma-range 0.05,0.2,0.05 \
    --output results/sigma_range_compare
```

## CLI Flags Reference

### Dataset Type

- `--test` - Use built-in test images
- `--synthetic` - Add synthetic noise to images from `./data/benchmark/clean/test`
- `--real-world` - Use real-world noisy/clean pairs from `data/SIDD_small_real_world/test` by default. To evaluate against a different prepared real-world test set, pass `--dataset-path /path/to/your/test`.

### Algorithm Selection

- `algorithm(s)` - One or more: `bm3d`, `nl-means`, `nlmeans`, `nafnet`, `resunet`, `res-unet`, `residual-unet`, `restormer`, `restorer`

### Dataset Configuration

- `--sigma FLOAT` - Noise level for synthetic datasets
- `--sigma-range start,end,step` - Evaluate a sigma sweep on synthetic data

### Algorithm Parameters

- `--patch-size INT` - Patch size for NL-Means
- `--patch-distance INT` - Patch search distance for NL-Means
- `--base-channels INT` - Base channels for ResUNet, NAFNet, and Restormer
- `--device STR` - Runtime device for ResUNet, NAFNet, and Restormer (`auto`, `cpu`, `cuda`)
- `--show-architecture` - Print loaded ResUNet/NAFNet/Restormer architecture in terminal during inference

### Output Options

- `--output DIR` - Optional base directory for comparison and sigma-range results. For comparison mode, outputs are written under `<output>/<algorithm-combination>/<dataset-type>/`. Without `--output`, default base is `<project-root>/results/compare`.
- `--show-plot` - Display plots interactively
- `--verbose` - Show detailed output
- `--rep-only` - Rebuild only the representative comparison image from already saved outputs.
- `--rep-seed INT` - Seed used to choose which cached image becomes the representative example.

Comparison mode now processes the full available dataset for the selected dataset type and always saves results under `images/`, `metrics/`, and `plots/` inside the comparison folder.

## Training

### Train ResUNet

```bash
python models/training/train_resunet.py \
    --dataset-path ./data/benchmark/clean/train \
    --sigma-range 0.02,0.20 \
    --device cpu \
    --save-path ./models/weights/resunet.pth
```

Notes:
- Use `--sigma 0.1` for fixed-noise training.
- Use `--sigma-range min,max` to sample sigma per patch (default: `0.02,0.20`).
- The script prints architecture details in terminal and saves `resunet_architecture.txt` near the checkpoint.

### Train NAFNet

```bash
python models/training/train_nafnet.py \
    --dataset-path ./data/benchmark/clean/train \
    --sigma-range 0.02,0.20 \
    --device cpu \
    --save-path ./models/weights/nafnet.pth
```

Notes:
- Use `--sigma 0.1` for fixed-noise training.
- Use `--sigma-range min,max` to sample sigma per patch (default: `0.02,0.20`).
- The script prints architecture details in terminal and saves `nafnet_architecture.txt` near the checkpoint.

## Output Files

Single-algorithm runs write to `results/single/<algorithm>/<dataset-type>/`.
Comparison runs write to:

`results/compare/<algorithm-combination>/<dataset-type>/`

or (if `--output DIR` is passed):

`<output>/<algorithm-combination>/<dataset-type>/`

`<algorithm-combination>` is canonical and order-insensitive (algorithms are normalized and sorted), e.g.:

`bm3d_vs_nlmeans`

Each comparison dataset folder contains:

- `images/`
- `metrics/`
- `plots/`

Comparison runs also maintain a `cache/` subtree under the comparison output directory. This cache mirrors the single-run layout so the CLI can reuse already computed per-image results instead of rerunning denoising. It is primarily there to make comparison runs faster and to keep them usable when the comparison output location is the only writable path. If the cache is removed, it will be rebuilt on the next comparison run.

Typical generated files include:

- `metrics/*.csv`
- `plots/*.png`

Comparison-mode outputs always include:

- `metrics/comparison_summary.csv`
- `plots/comparison_<algorithm1>_vs_<algorithm2>_....png`

Sigma-range comparison outputs include:

- `metrics/sigma_range_summary.csv`
- `plots/psnr_vs_sigma_range.png`
- `plots/ssim_vs_sigma_range.png`

Single-algorithm image outputs:

- For `test` and `synthetic` datasets:
    - `images/noisy/*_noisy*.png`
    - `images/denoised/*_denoised*.png`
- For `real-world` datasets:
    - `images/*_denoised*.png`

## Project Structure

```text
Low-light-Denoising/
├── models/
│   ├── training/
│   │   ├── train_resunet.py
│   │   └── train_nafnet.py
│   └── weights/
├── src/
│   └── denoiser/
│       ├── __main__.py
│       ├── evaluator.py
│       ├── algorithms/
│       │   ├── bm3d_denoiser.py
│       │   ├── nl_means_denoiser.py
│       │   ├── resunet_denoiser.py
│       │   ├── nafnet_denoiser.py
│       │   └── restormer_denoiser.py
│       ├── datasets/
│       └── utils/
├── data/
├── dependencies/
└── README.md
```

## Programmatic Usage

```python
from denoiser.algorithms import BM3DDenoiser, NLMeansDenoiser, ResUNetDenoiser, NAFNetDenoiser
from denoiser.datasets import get_dataset_loader
from denoiser.evaluator import ComparisonEvaluator

loader = get_dataset_loader('test', noise_sigma=0.1)

bm3d = BM3DDenoiser(sigma_psd=0.1)
nlmeans = NLMeansDenoiser(sigma=0.1)
resunet = ResUNetDenoiser(device='cpu')
nafnet = NAFNetDenoiser(device='cpu')

comparison = ComparisonEvaluator([bm3d, nlmeans, resunet, nafnet], loader, verbose=True)
results = comparison.evaluate_all()
```

## Real-world evaluation and SIDD preparation

This repository expects a prepared real‑world paired dataset layout with `clean/` and `noisy/` subdirectories. The provided helper script prepares SIDD Small into that layout:

```bash
python scripts/prepare_sidd_small_real_world.py  # writes to data/SIDD_small_real_world/test by default
```

What the script does:
- Scans the SIDD Small scene folders for matching `GT_SRGB_*` and `NOISY_SRGB_*` images.
- Copies paired files into `data/SIDD_small_real_world/test/clean` and `/noisy` and writes `manifest.csv`.
- `data/SIDD_Small_sRGB_Only/Data` is the raw source tree; the CLI does not read it directly.

Defaults and CLI behavior:
- The CLI `--real-world` option uses the default path `data/SIDD_small_real_world/test` unless overridden with `--dataset-path`.
- If you prepared SIDD Medium elsewhere (e.g., on Kaggle), run the CLI with `--dataset-path /path/to/sidd_medium_root/test` to evaluate on that dataset instead of SIDD Small. The SIDD Medium data itself is not bundled in this repo because it is typically prepared externally for storage and compute reasons.

Benchmark data layout notes:
- `data/benchmark/clean/test` is the default synthetic inference input used by the CLI.
- `data/benchmark/clean/train` and `data/benchmark/clean/val` are used when you point the training scripts at those splits.
- `data/benchmark/clean/one_image` is a convenience subset for quick checks and examples.
- `data/external/BSDS500-master` is the raw BSDS500 source copy kept alongside the prepared benchmark data for reference.

Using your Kaggle-trained checkpoints:
- The denoiser constructors accept a `model_path` argument programmatically. Example:

```python
from denoiser.algorithms import NAFNetDenoiser
from denoiser.datasets import get_dataset_loader
from denoiser.evaluator import Evaluator

algo = NAFNetDenoiser(model_path='/path/to/your/kaggle_checkpoint.pth', device='auto')
loader = get_dataset_loader('real-world', dataset_path='data/SIDD_small_real_world/test')
evaluator = Evaluator(algo, loader, verbose=True)
results = evaluator.evaluate()
```

- Alternatively, copy your checkpoint into the repo weights folder and name it to match the CLI defaults so the CLI picks it automatically:
    - `models/weights/nafnet_real_world.pth` for NAFNet (or `nafnet.pth` for non-real-world default)
    - `models/weights/restormer_real_world.pth` for Restormer

Training scripts note:
- The training scripts detect paired real‑world data when `train/` and `test/` splits exist and set synthetic noise to zero during training (paired training uses `PairedPatchDataset`).

## Troubleshooting

- `Module not found`: run commands from `src/` or set `PYTHONPATH=src`
- `BM3D errors`: ensure `bm3d` is installed
- `ResUNet errors`: ensure `./models/weights/resunet.pth` exists
 - `NAFNet errors`: ensure a checkpoint exists in `./models/weights/` (for real‑world evaluation the CLI looks for `nafnet_real_world.pth`; `nafnet.pth` is also supported).
 - `Restormer errors`: ensure a checkpoint exists in `./models/weights/` (for real‑world evaluation the CLI looks for `restormer_real_world.pth`; `restormer.pth` is also supported).
- `No images loaded`: verify the dataset paths and file extensions

## Adding New Algorithms

To add a new algorithm, create a module in `src/denoiser/algorithms/`, inherit from `BaseDenoiser`, implement `denoise`, and register it in `src/denoiser/algorithms/__init__.py`.
