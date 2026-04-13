# Low-light Image Denoising Framework

A modular framework for evaluating and comparing image denoising algorithms. Test different algorithms, visualize results, and compare performance metrics with ease.

## Features

✨ **Multiple Algorithms**: BM3D, Non-Local Means (NL-Means), and Residual U-Net (ResUNet)  
📊 **Performance Visualization**: Automatic plot generation for PSNR, MSE, and SSIM  
⚖️ **Algorithm Comparison**: Side-by-side comparison of multiple algorithms  
📉 **Sigma Range Curves**: Line plots for PSNR vs σ and SSIM vs σ across algorithms  
📈 **Quality Metrics**: PSNR, MSE, and SSIM evaluation  
🎯 **Flexible Datasets**: Built-in test images, synthetic noise, or real-world paired images  
💾 **Export Results**: CSV metrics and denoised images

## Quick Start

### 1. Setup Environment

```bash
# Create and activate conda environment
conda env create -f dependencies/environment.yaml
conda activate denoising

# Or install manually
pip install numpy matplotlib scikit-image bm3d torch tqdm
```

### 2. Run Your First Test

```bash
cd src
python -m denoiser --test bm3d --plot
```

This will:
- Run BM3D on 3 built-in test images
- Display performance metrics (PSNR, MSE, SSIM)
- Generate a comparison plot showing noisy vs denoised results

## Usage

### Basic Commands

**Test a single algorithm:**
```bash
python -m denoiser --test bm3d
python -m denoiser --test nl-means
python -m denoiser --test resunet --device cpu
```

**Compare multiple algorithms:**
```bash
python -m denoiser --test bm3d nl-means --plot
python -m denoiser --test bm3d nl-means resunet --device cpu --plot
```

**Single algorithm output location:**
```bash
python -m denoiser --test bm3d --plot
```

Single-mode output is always written under `results/single/<algorithm>/`.

### Dataset Options

**Built-in test images** (camera, astronaut, text):
```bash
python -m denoiser --test bm3d
```

**Synthetic noise** (add noise to clean images):
```bash
python -m denoiser --synthetic bm3d --sigma 0.15
```

**Real-world paired images** (clean/noisy pairs):
```bash
python -m denoiser --real-world bm3d
```

### Algorithm Parameters

**BM3D:**
```bash
python -m denoiser --test bm3d --sigma 0.15
```

**NL-Means with custom parameters:**
```bash
python -m denoiser --test nl-means --patch-size 7 --patch-distance 8 --sigma 0.12
```

**ResUNet with pretrained weights:**
```bash
python -m denoiser --test resunet --base-channels 32 --device cpu
```

### Comparison Mode

Compare BM3D and NL-Means on the same dataset:

```bash
# Explicit comparison
python -m denoiser --test --compare bm3d nl-means --output results/comparison --plot

# Implicit (specifying multiple algorithms automatically enables comparison)
python -m denoiser --test bm3d nl-means --output results/comparison --plot

# Fast comparison on one automatically selected image
python -m denoiser --synthetic --compare bm3d nl-means resunet \
    --single-image --sample-seed 42 \
    --output ../results/compare_single_image --plot

# Compare across multiple sigma values and generate line plots
python -m denoiser --synthetic --compare bm3d nl-means resunet \
    --single-image --sample-seed 42 \
    --sigma-range 0.05,0.2,0.05 \
    --output ../results/sigma_range_compare --plot
```

## CLI Flags Reference

### Required: Dataset Type (choose one)
- `--test` - Use built-in test images
- `--synthetic` - Add synthetic noise to images from `./data/benchmark/clean/test`
- `--real-world` - Use real-world noisy/clean pairs from `./data/demo_pair`

### Algorithm Selection
- `algorithm(s)` - One or more: `bm3d`, `nl-means`, `nlmeans`, `resunet`, `res-unet`, `residual-unet`

### Dataset Configuration
- `--sigma FLOAT` - Noise level (default: 0.1)
- `--single-image` - Use one automatically selected image from dataset (fast runs)
- `--sample-seed INT` - Seed for deterministic image sampling with `--single-image` (default: 42)
- `--sigma-range start,end,step` - Run synthetic evaluation across a sigma range and generate range plots

### Algorithm Parameters
- `--patch-size INT` - Patch size for NL-Means (default: 5)
- `--patch-distance INT` - Patch search distance for NL-Means (default: 6)
- `--base-channels INT` - Base channels for ResUNet (default: 32)
- `--device STR` - Runtime device for ResUNet (`auto`, `cpu`, `cuda`)

### Output Options
- `--output DIR` - Save comparison and sigma-range results to a directory
- `--plot` - Generate performance plots
- `--show-plot` - Display plots interactively
- `--show-images` - Display side-by-side comparison of noisy and denoised images
- `--num-display INT` - Number of random images to display with `--show-images` (default: 3)
- `--verbose` - Show detailed verbose output

### Comparison Mode
- `--compare` - Enable comparison mode (optional if multiple algorithms specified)

### Sigma Range Notes
- `--sigma-range` works with `--synthetic` datasets
- `--sigma-range` format is `start,end,step` (example: `0.05,0.2,0.05`)
- During sigma-range runs, dataset noise and BM3D/NL-Means sigma are updated per step
- For reproducible single-image ranges, use `--single-image --sample-seed <seed>`

## Output Files

### Single Algorithm Mode
```
results/
└── single/
    └── bm3d/
        ├── metrics/
        │   └── metrics_sigma_0p1.csv
        ├── plots/
        │   └── metrics_plot_bm3d_sigma_0p1.png
        └── images/
            ├── camera_denoised_sigma_0p1.png
            ├── astronaut_denoised_sigma_0p1.png
            └── text_denoised_sigma_0p1.png
```

### Comparison Mode
```
results/
└── comparison/
    ├── comparison_summary.csv
    └── comparison_bm3d_vs_nl_means.png
```

### Sigma Range Mode (Single Algorithm)
```
results/
└── single/
    └── bm3d/
        ├── metrics/
        │   └── sigma_range_summary.csv
        └── plots/
            ├── psnr_vs_sigma_range.png
            └── ssim_vs_sigma_range.png
```

## Example Workflows

### 1. Quick Algorithm Test
```bash
# Test BM3D and see results
python -m denoiser --test bm3d --plot
```

### 2. Compare Algorithms
```bash
# Compare BM3D vs NL-Means and save everything
python -m denoiser --test bm3d nl-means \
    --output results/comparison \
    --plot
```

### 3. Custom Noise Level Evaluation
```bash
# Synthetic run uses ./data/benchmark/clean/test internally
python -m denoiser --synthetic bm3d \
    --sigma 0.2 \
    --plot
```

### 4. Batch Processing (Verbose Mode)
```bash
# Run comparison with detailed logs
python -m denoiser --test bm3d nl-means \
    --output results/batch_run \
    --plot --verbose
```

### 5. Visual Image Comparison
```bash
# Display side-by-side comparison of noisy and denoised images
python -m denoiser --test bm3d --show-images

# Compare multiple algorithms visually
python -m denoiser --test bm3d nl-means --show-images --num-display 2
```

### 6. Train ResUNet (AWGN benchmark setup)
```bash
# Train using clean benchmark split (noise is added synthetically during training)
python models/training/train_resunet.py \
    --dataset-path ./data/benchmark/clean/train \
    --sigma 0.1 \
    --device cpu \
    --save-path ./models/weights/resunet.pth
```

### 7. Fast One-Image Comparison
```bash
# Quickly compare algorithms on one sampled image
python -m denoiser --synthetic --compare bm3d nl-means resunet \
    --single-image --sample-seed 42 \
    --output ../results/compare_single_image --plot
```

### 8. PSNR/SSIM vs Sigma Curves
```bash
# Generate two line graphs: PSNR vs sigma and SSIM vs sigma
python -m denoiser --synthetic --compare bm3d nl-means resunet \
    --single-image --sample-seed 42 \
    --sigma-range 0.05,0.2,0.05 \
    --output ../results/sigma_range_compare --plot
```

## Performance Metrics

The framework calculates three quality metrics:

- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better. Typical range: 20-40 dB
- **MSE** (Mean Squared Error): Lower is better. Range: 0-1
- **SSIM** (Structural Similarity Index): Higher is better. Range: 0-1

All metrics compare the denoised image against the clean reference image.

## Project Structure

```
Low-light-Denoising/
├── models/
│   ├── training/
│   │   └── train_resunet.py
│   └── weights/
├── src/
│   ├── denoiser/              # Main package
│   │   ├── __init__.py
│   │   ├── __main__.py        # CLI entry point
│   │   ├── evaluator.py       # Evaluation and comparison logic
│   │   ├── algorithms/        # Denoising algorithms
│   │   │   ├── bm3d_denoiser.py
│   │   │   ├── nl_means_denoiser.py
│   │   │   └── resunet_denoiser.py
│   │   ├── datasets/          # Dataset loaders
│   │   │   └── loader.py
│   │   └── utils/            # Metrics and utilities
│   │       ├── metrics.py
│   │       └── noise.py
├── data/                     # Your datasets
├── dependencies/
│   └── environment.yaml      # Conda environment
└── README.md
```

## Adding Your Own Images

### For Synthetic Noise Testing
The CLI uses a fixed synthetic directory: `./data/benchmark/clean/test`.
Place clean images there:
```bash
mkdir -p data/benchmark/clean/test
cp your_images/*.png data/benchmark/clean/test/

python -m denoiser --synthetic bm3d --plot
```

### For Real-World Paired Images
The CLI uses a fixed real-world directory: `./data/demo_pair`.
Organize as clean/noisy pairs:
```bash
mkdir -p data/demo_pair/clean data/demo_pair/noisy
# Place matching clean and noisy images in respective folders

python -m denoiser --real-world bm3d --plot
```

## Troubleshooting

**"Module not found" error:**
- Activate conda environment: `conda activate denoising`
- Or install dependencies: `pip install numpy matplotlib scikit-image bm3d torch tqdm`
- Make sure you run CLI commands from the `src` folder (or set `PYTHONPATH` to `src`)

**BM3D errors:**
- Check you have `bm3d` installed: `pip install bm3d`
- Try updating: `pip install -U bm3d`

**ResUNet errors:**
- Ensure `torch` is installed in the same environment used to run commands
- Ensure model weights exist at `./models/weights/resunet.pth`

**No images loaded:**
- For synthetic mode, check images under `./data/benchmark/clean/test`
- For real-world mode, check `./data/demo_pair/clean` and `./data/demo_pair/noisy`
- Ensure images are in PNG or JPG format

## Programmatic Usage

Beyond the CLI, you can use the package directly in Python scripts:

### Single Algorithm Evaluation

```python
from denoiser.algorithms import BM3DDenoiser
from denoiser.datasets import get_dataset_loader
from denoiser.evaluator import Evaluator

# Load dataset
loader = get_dataset_loader('test', noise_sigma=0.1)

# Initialize algorithm
algorithm = BM3DDenoiser(sigma_psd=0.1)

# Create evaluator and run
evaluator = Evaluator(algorithm, loader, verbose=True)
results = evaluator.evaluate()

# Save results and generate plots
evaluator.save_results('output/', save_images=True)
evaluator.plot_results('output/', show_plot=True)
evaluator.show_image_comparison(num_images=3)
```

### Algorithm Comparison

```python
from denoiser.algorithms import BM3DDenoiser, NLMeansDenoiser, ResUNetDenoiser
from denoiser.datasets import get_dataset_loader
from denoiser.evaluator import ComparisonEvaluator

# Load dataset
loader = get_dataset_loader('test', noise_sigma=0.1)

# Initialize algorithms
bm3d = BM3DDenoiser(sigma_psd=0.1)
nlmeans = NLMeansDenoiser(sigma=0.1)
resunet = ResUNetDenoiser(device='cpu')

# Create comparison evaluator
comparison = ComparisonEvaluator([bm3d, nlmeans, resunet], loader, verbose=True)

# Run comparison
all_results = comparison.evaluate_all()

# Save and display results
comparison.save_comparison_summary('output/')
comparison.plot_comparison('output/', show_plot=True)
comparison.show_image_comparison(num_images=2)
```

## Adding New Algorithms

To extend the framework with your own denoising algorithm:

### Step 1: Create Algorithm File

Create a new file in `src/denoiser/algorithms/` (e.g., `cnn_denoiser.py`):

```python
from .base import BaseDenoiser
import numpy as np

class CNNDenoiser(BaseDenoiser):
    """CNN-based denoising algorithm."""

    def __init__(self, model_path: str = None, **kwargs) -> None:
        super().__init__(model_path=model_path, **kwargs)
        self.model_path = model_path
        # Load your model here

    def denoise(self, noisy_image: np.ndarray) -> np.ndarray:
        """Apply CNN denoising to image."""
        # Implement your algorithm
        denoised = ...  # Your implementation
        return denoised
```

### Step 2: Register Algorithm

Add to `src/denoiser/algorithms/__init__.py`:

```python
from .cnn_denoiser import CNNDenoiser

ALGORITHMS = {
    'bm3d': BM3DDenoiser,
    'nl-means': NLMeansDenoiser,
    'cnn': CNNDenoiser,  # Your new algorithm
}
```

### Step 3: Use Your Algorithm

```bash
python -m denoiser --test cnn
```

**Requirements for custom algorithms:**
- Inherit from `BaseDenoiser` class
- Implement `denoise(noisy_image)` method that takes and returns a NumPy array
- Images are in float format (0-1 range)
- Support both grayscale (H×W) and RGB (H×W×3) images

## Algorithm Comparison Results

Typical performance on test images:

| Algorithm | Avg PSNR Improvement | Speed (per image) | Best For |
|-----------|---------------------|-------------------|----------|
| BM3D      | ~9-10 dB           | 5-15 seconds     | Maximum quality |
| NL-Means  | ~8 dB              | 0.2-0.7 seconds  | Speed/quality balance |
| ResUNet   | Depends on training | Fast inference   | Learned denoising patterns |
