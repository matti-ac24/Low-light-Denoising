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
    --max-images 1 --sample-seed 42
```

This writes to a canonical comparison folder (algorithm order does not matter), for example:

`results/compare/bm3d_vs_nafnet_vs_nl-means_vs_resunet/synthetic/`

### Sigma Range Mode

```bash
python -m denoiser --synthetic --compare bm3d nl-means resunet nafnet \
    --max-images 1 --sample-seed 42 \
    --sigma-range 0.05,0.2,0.05 \
    --output results/sigma_range_compare
```

## CLI Flags Reference

### Dataset Type

- `--test` - Use built-in test images
- `--synthetic` - Add synthetic noise to images from `./data/benchmark/clean/test`
- `--real-world` - Use real-world noisy/clean pairs from `./data/demo_pair`

### Algorithm Selection

- `algorithm(s)` - One or more: `bm3d`, `nl-means`, `nlmeans`, `nafnet`, `resunet`, `res-unet`, `residual-unet`, `restormer`, `restorer`

### Dataset Configuration

- `--sigma FLOAT` - Noise level for synthetic datasets
- `--max-images INT` - Process up to N randomly selected images
- `--sample-seed INT` - Deterministic image sampling seed
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
- `--show-images` - Display noisy vs denoised images
- `--num-display INT` - Number of images to display with `--show-images`
- `--verbose` - Show detailed output

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

## Troubleshooting

- `Module not found`: run commands from `src/` or set `PYTHONPATH=src`
- `BM3D errors`: ensure `bm3d` is installed
- `ResUNet errors`: ensure `./models/weights/resunet.pth` exists
- `NAFNet errors`: ensure `./models/weights/nafnet.pth` exists
- `No images loaded`: verify the dataset paths and file extensions

## Adding New Algorithms

To add a new algorithm, create a module in `src/denoiser/algorithms/`, inherit from `BaseDenoiser`, implement `denoise`, and register it in `src/denoiser/algorithms/__init__.py`.
