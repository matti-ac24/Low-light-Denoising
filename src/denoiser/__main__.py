# A Command-line interface for the denoiser package

import sys
import argparse
import csv
import shutil
from pathlib import Path

from .algorithms import get_algorithm, ALGORITHMS
from .datasets import get_dataset_loader
from .evaluator import Evaluator, ComparisonEvaluator


DEFAULT_SYNTHETIC_DATASET_REL_PATH = Path('data/benchmark/clean/test')
DEFAULT_REAL_WORLD_DATASET_REL_PATH = Path('data/demo_pair')
DEFAULT_COMPARISON_OUTPUT_REL_PATH = Path('results/compare')


def format_sigma_for_path(sigma: float) -> str:

    formatted = f"{sigma:.4f}".rstrip('0').rstrip('.')
    return formatted.replace('.', 'p')


def format_algorithm_for_path(name: str) -> str:

    return name.strip().lower().replace(' ', '-').replace('_', '-')


def format_sigma_suffix_for_metrics(sigma: float | None) -> str:

    if sigma is None:
        return ''
    formatted = f"{sigma:.4f}".rstrip('0').rstrip('.').replace('.', 'p')
    return f"_sigma_{formatted}"


def get_single_output_dir(project_root: Path, algorithm_display_name: str, dataset_type: str) -> Path:

    return (
        project_root
        / 'results'
        / 'single'
        / format_algorithm_for_path(algorithm_display_name)
        / dataset_type
    )


def get_single_metrics_csv_path(
    project_root: Path,
    algorithm_display_name: str,
    dataset_type: str,
    sigma: float | None,
) -> Path:

    output_dir = get_single_output_dir(project_root, algorithm_display_name, dataset_type)
    sigma_suffix = format_sigma_suffix_for_metrics(sigma)
    return output_dir / 'metrics' / f'metrics{sigma_suffix}.csv'


def load_cached_single_results(metrics_csv_path: Path, image_names: list[str]) -> list[dict] | None:

    import csv

    if not metrics_csv_path.exists():
        return None

    with open(metrics_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rows_by_image = {row['image']: row for row in rows if 'image' in row}
    if any(name not in rows_by_image for name in image_names):
        return None

    cached_results: list[dict] = []
    for image_name in image_names:
        row = rows_by_image[image_name]
        cached_results.append(
            {
                'name': image_name,
                'processing_time': float(row.get('processing_time', 0.0)),
                'noisy_metrics': {
                    'psnr': float(row['noisy_psnr']),
                    'ssim': float(row['noisy_ssim']),
                },
                'denoised_metrics': {
                    'psnr': float(row['denoised_psnr']),
                    'ssim': float(row['denoised_ssim']),
                },
            }
        )

    return cached_results


def write_results_metrics_csv(results: list[dict], metrics_csv_path: Path) -> None:

    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_csv_path, 'w', newline='') as f:
        fieldnames = [
            'image',
            'processing_time',
            'noisy_psnr',
            'denoised_psnr',
            'psnr_improvement',
            'noisy_ssim',
            'denoised_ssim',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    'image': result['name'],
                    'processing_time': result['processing_time'],
                    'noisy_psnr': result['noisy_metrics']['psnr'],
                    'denoised_psnr': result['denoised_metrics']['psnr'],
                    'psnr_improvement': result['denoised_metrics']['psnr'] - result['noisy_metrics']['psnr'],
                    'noisy_ssim': result['noisy_metrics']['ssim'],
                    'denoised_ssim': result['denoised_metrics']['ssim'],
                }
            )


def mirror_single_results_to_comparison_dir(
    single_output_dir: Path,
    comparison_output_dir: Path | None,
    results: list[dict],
    sigma: float | None,
) -> None:

    if comparison_output_dir is None:
        return

    comparison_cache_dir = comparison_output_dir / 'cache'
    comparison_cache_dir.mkdir(parents=True, exist_ok=True)

    algorithm_name = single_output_dir.parent.name
    dataset_type = single_output_dir.name
    target_dir = comparison_cache_dir / algorithm_name / dataset_type
    try:
        shutil.copytree(single_output_dir, target_dir, dirs_exist_ok=True)
    except Exception:
        metrics_dir = target_dir / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        sigma_suffix = format_sigma_suffix_for_metrics(sigma)
        write_results_metrics_csv(results, metrics_dir / f'metrics{sigma_suffix}.csv')


def evaluate_or_reuse_algorithm_results(
    algorithm: object,
    dataset_loader: object,
    args: argparse.Namespace,
    project_root: Path,
    dataset_type: str,
    sigma: float | None,
    max_images: int | None,
    comparison_output_dir: Path | None = None,
) -> list[dict]:

    images = dataset_loader.load_images()
    image_names = [item['name'] for item in images]

    metrics_csv_path = get_single_metrics_csv_path(
        project_root,
        algorithm.name,
        dataset_type,
        sigma,
    )

    if not args.show_images:
        cached_results = load_cached_single_results(metrics_csv_path, image_names)
        if cached_results is not None:
            if args.verbose:
                print(f"Reusing cached metrics for {algorithm.name} from: {metrics_csv_path}")
            mirror_single_results_to_comparison_dir(
                single_output_dir=get_single_output_dir(project_root, algorithm.name, dataset_type),
                comparison_output_dir=comparison_output_dir,
                results=cached_results,
                sigma=sigma,
            )
            return cached_results

    evaluator = Evaluator(
        algorithm=algorithm,
        dataset_loader=dataset_loader,
        verbose=args.verbose,
        show_progress=(
            not args.verbose
            and dataset_type in ['synthetic', 'real-world']
            and max_images is None
        ),
    )
    results = evaluator.evaluate()

    single_output_dir = get_single_output_dir(project_root, algorithm.name, dataset_type)
    save_noisy_images = dataset_type in ['test', 'synthetic']
    evaluator.save_results(
        single_output_dir,
        save_images=True,
        sigma=sigma,
        save_noisy_images=save_noisy_images,
        split_image_dirs=save_noisy_images,
    )
    mirror_single_results_to_comparison_dir(
        single_output_dir=single_output_dir,
        comparison_output_dir=comparison_output_dir,
        results=results,
        sigma=sigma,
    )

    if args.verbose:
        print(f"Cached single-mode results for {algorithm.name} at: {single_output_dir}")

    return results


def build_comparison_folder_name(algorithm_names: list[str]) -> str:

    normalized = sorted(format_algorithm_for_path(name) for name in algorithm_names)
    return '_vs_'.join(normalized)


def resolve_comparison_output_dir(
    project_root: Path,
    output_arg: str | None,
    dataset_type: str,
    algorithm_names: list[str],
) -> Path:

    base_output_dir = Path(output_arg) if output_arg else project_root / DEFAULT_COMPARISON_OUTPUT_REL_PATH
    comparison_dir = base_output_dir / build_comparison_folder_name(algorithm_names) / dataset_type

    for sub_dir in ['images', 'metrics', 'plots']:
        (comparison_dir / sub_dir).mkdir(parents=True, exist_ok=True)

    return comparison_dir


def parse_sigma_range(sigma_range_arg: str) -> list[float]:

    values = [item.strip() for item in sigma_range_arg.split(',') if item.strip()]
    if len(values) != 3:
        raise ValueError("--sigma-range must be in the format: start,end,step")

    start = float(values[0])
    end = float(values[1])
    step = float(values[2])

    if start <= 0 or end <= 0:
        raise ValueError("Start and end values in --sigma-range must be > 0")
    if step <= 0:
        raise ValueError("Step value in --sigma-range must be > 0")
    if end < start:
        raise ValueError("End value in --sigma-range must be >= start value")

    sigmas: list[float] = []
    current = start
    epsilon = step * 1e-9
    while current <= end + epsilon:
        sigmas.append(round(current, 10))
        current += step

    if not sigmas:
        raise ValueError("--sigma-range produced no sigma values")

    return sigmas


def resolve_max_images(args: argparse.Namespace) -> int | None:
    if args.max_images is None:
        return None

    if args.max_images < 1:
        raise ValueError("--max-images must be >= 1")

    return args.max_images


def run_sigma_range(
    args: argparse.Namespace,
    project_root: Path,
    dataset_type: str,
    dataset_path: str,
    algorithms_list: list[str],
    is_comparison: bool,
    max_images: int | None,
    output_dir: Path,
) -> int:

    import csv
    import numpy as np
    import matplotlib.pyplot as plt

    sigma_values = parse_sigma_range(args.sigma_range)

    if dataset_type != 'synthetic':
        raise ValueError("--sigma-range is supported only with --synthetic datasets")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / 'metrics'
    plots_dir = output_dir / 'plots'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"\nRunning sigma range with values: {sigma_values}")

    algo_metrics: dict[str, dict[str, list[float]]] = {}
    for algo_name in algorithms_list:
        algorithm_class = get_algorithm(algo_name)
        algo_params = build_algorithm_params(algo_name, args)
        if 'sigma' in algo_params:
            algo_params['sigma'] = sigma_values[0]
        if 'sigma_psd' in algo_params:
            algo_params['sigma_psd'] = sigma_values[0]
        display_name = algorithm_class(**algo_params).name
        if display_name not in algo_metrics:
            algo_metrics[display_name] = {'psnr': [], 'ssim': []}

    for sigma in sigma_values:
        if args.verbose:
            print(f"\n{'-'*70}")
            print(f"Sigma range step: σ={sigma:.3f}")
            print(f"{'-'*70}")

        dataset_loader = get_dataset_loader(
            dataset_type=dataset_type,
            dataset_path=dataset_path,
            noise_sigma=sigma,
            max_images=max_images,
            sample_seed=args.sample_seed,
        )

        if is_comparison:
            algorithms = []
            for algo_name in algorithms_list:
                algorithm_class = get_algorithm(algo_name)
                algo_params = build_algorithm_params(algo_name, args)
                if 'sigma' in algo_params:
                    algo_params['sigma'] = sigma
                if 'sigma_psd' in algo_params:
                    algo_params['sigma_psd'] = sigma
                algorithms.append(algorithm_class(**algo_params))

            all_results: dict[str, list[dict]] = {}
            for algorithm in algorithms:
                all_results[algorithm.name] = evaluate_or_reuse_algorithm_results(
                    algorithm=algorithm,
                    dataset_loader=dataset_loader,
                    args=args,
                    project_root=project_root,
                    dataset_type=dataset_type,
                    sigma=sigma,
                    max_images=max_images,
                    comparison_output_dir=output_dir,
                )

            for algo_display_name, results in all_results.items():
                denoised_psnr = [r['denoised_metrics']['psnr'] for r in results]
                denoised_ssim = [r['denoised_metrics']['ssim'] for r in results]
                algo_metrics[algo_display_name]['psnr'].append(float(np.mean(denoised_psnr)))
                algo_metrics[algo_display_name]['ssim'].append(float(np.mean(denoised_ssim)))
        else:
            algorithm_name = algorithms_list[0]
            algorithm_class = get_algorithm(algorithm_name)
            algo_params = build_algorithm_params(algorithm_name, args)
            if 'sigma' in algo_params:
                algo_params['sigma'] = sigma
            if 'sigma_psd' in algo_params:
                algo_params['sigma_psd'] = sigma
            algorithm = algorithm_class(**algo_params)

            evaluator = Evaluator(
                algorithm=algorithm,
                dataset_loader=dataset_loader,
                verbose=args.verbose,
                show_progress=(
                    not args.verbose
                    and dataset_type in ['synthetic', 'real-world']
                    and max_images is None
                ),
            )
            results = evaluator.evaluate()

            denoised_psnr = [r['denoised_metrics']['psnr'] for r in results]
            denoised_ssim = [r['denoised_metrics']['ssim'] for r in results]
            algo_metrics[algorithm.name]['psnr'].append(float(np.mean(denoised_psnr)))
            algo_metrics[algorithm.name]['ssim'].append(float(np.mean(denoised_ssim)))

    summary_csv_path = metrics_dir / 'sigma_range_summary.csv'
    with open(summary_csv_path, 'w', newline='') as f:
        fieldnames = ['sigma', 'algorithm', 'avg_psnr', 'avg_ssim']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, sigma in enumerate(sigma_values):
            for algo_name, metrics in algo_metrics.items():
                writer.writerow(
                    {
                        'sigma': sigma,
                        'algorithm': algo_name,
                        'avg_psnr': metrics['psnr'][idx],
                        'avg_ssim': metrics['ssim'][idx],
                    }
                )

    # PSNR vs sigma line graph
    plt.figure(figsize=(8, 5))
    for algo_name, metrics in algo_metrics.items():
        plt.plot(sigma_values, metrics['psnr'], marker='o', linewidth=2, label=algo_name)
    plt.xlabel('Sigma (noise std)')
    plt.ylabel('Average PSNR (dB)')
    plt.title('PSNR vs Sigma')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    psnr_plot_path = plots_dir / 'psnr_vs_sigma_range.png'
    plt.savefig(psnr_plot_path, dpi=150, bbox_inches='tight')
    if args.show_plot:
        plt.show()
    else:
        plt.close()

    # SSIM vs sigma line graph
    plt.figure(figsize=(8, 5))
    for algo_name, metrics in algo_metrics.items():
        plt.plot(sigma_values, metrics['ssim'], marker='o', linewidth=2, label=algo_name)
    plt.xlabel('Sigma (noise std)')
    plt.ylabel('Average SSIM')
    plt.title('SSIM vs Sigma')
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    ssim_plot_path = plots_dir / 'ssim_vs_sigma_range.png'
    plt.savefig(ssim_plot_path, dpi=150, bbox_inches='tight')
    if args.show_plot:
        plt.show()
    else:
        plt.close()

    print(f"\nSigma range complete!")
    print(f"Results saved to: {output_dir}")
    if args.verbose:
        print(f"  - Summary CSV: {summary_csv_path}")
        print(f"  - PSNR plot: {psnr_plot_path}")
        print(f"  - SSIM plot: {ssim_plot_path}")

    return 0

# Define all the command-line argument options
def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description='Modular Image Denoising Evaluation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Run BM3D on built-in test images
  python -m denoiser --test bm3d
  
  # Run NL-Means on synthetic noisy images
    python -m denoiser --synthetic nl-means

    # Run Residual U-Net CNN with pretrained weights
        python -m denoiser --test resunet --device cpu

    # Run NAFNet CNN with pretrained weights
        python -m denoiser --test nafnet --device cpu
  
  # Compare algorithms side-by-side
        python -m denoiser --test --compare bm3d nl-means --output results/compare
  
  # Run BM3D on real-world paired images
    python -m denoiser --real-world bm3d
  
  # Customize noise level and save results
    python -m denoiser --synthetic bm3d --sigma 0.15
        """
    )
    
    # Dataset type (mutually exclusive)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        '--test',
        action='store_true',
        help='Use built-in test images (no dataset path required)'
    )
    dataset_group.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic dataset from ./data/benchmark/clean/test (adds noise)'
    )
    dataset_group.add_argument(
        '--real-world',
        action='store_true',
        help='Use real-world paired dataset from ./data/demo_pair (clean/noisy pairs)'
    )
    
    # Comparison mode
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Enable comparison mode to evaluate multiple algorithms'
    )
    
    # Algorithm selection
    parser.add_argument(
        'algorithm',
        type=str,
        nargs='+',
        choices=list(ALGORITHMS.keys()),
        help='Denoising algorithm(s) to use (can specify multiple for comparison)'
    )
    
    parser.add_argument(
        '--sigma',
        type=float,
        default=0.1,
        help='Noise standard deviation for synthetic datasets (default: 0.1)'
    )

    parser.add_argument(
        '--sigma-range',
        type=str,
        default=None,
        help='Sigma range as start,end,step (e.g. 0.05,0.2,0.05)'
    )

    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Process up to N randomly selected images from the dataset'
    )

    parser.add_argument(
        '--sample-seed',
        type=int,
        default=42,
        help='Random seed used for image sampling (default: 42)'
    )
    
    # Algorithm parameters
    parser.add_argument(
        '--patch-size',
        type=int,
        default=5,
        help='Patch size for NL-Means (default: 5)'
    )
    
    parser.add_argument(
        '--patch-distance',
        type=int,
        default=6,
        help='Patch search distance for NL-Means (default: 6)'
    )

    parser.add_argument(
        '--base-channels',
        type=int,
        default=32,
        help='Base feature channels for ResUNet, NAFNet, and Restormer architectures (default: 32)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help="Runtime device for CNN inference: 'auto', 'cpu', or 'cuda'"
    )

    parser.add_argument(
        '--show-architecture',
        action='store_true',
        help='Print loaded ResUNet/NAFNet/Restormer architecture in terminal during inference'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for comparison and sigma-range results (ignored in single mode; comparison default: <project root>/results/compare)'
    )
    
    parser.add_argument(
        '--show-plot',
        action='store_true',
        help='Display generated plots interactively'
    )
    
    parser.add_argument(
        '--show-images',
        action='store_true',
        help='Display side-by-side comparison of noisy and denoised images'
    )
    
    parser.add_argument(
        '--num-display',
        type=int,
        default=3,
        help='Number of random images to display with --show-images (default: 3)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed verbose output'
    )
    
    return parser.parse_args()

# Build a parameter dictionary for a specific algorithm
def build_algorithm_params(algorithm_name: str, args: argparse.Namespace) -> dict:

    if algorithm_name in ['nl-means', 'nlmeans']:
        return {
            'patch_size': args.patch_size,
            'patch_distance': args.patch_distance,
            'sigma': args.sigma
        }
    if algorithm_name in ['resunet', 'res-unet', 'residual-unet']:
        return {
            'base_channels': args.base_channels,
            'device': args.device,
            'show_architecture': args.show_architecture,
        }
    if algorithm_name == 'nafnet':
        return {
            'base_channels': args.base_channels,
            'device': args.device,
            'show_architecture': args.show_architecture,
        }
    if algorithm_name in ['restormer', 'restorer']:
        return {
            'base_channels': args.base_channels,
            'device': args.device,
            'show_architecture': args.show_architecture,
        }
    else:  # BM3D and other algorithms
        return {'sigma_psd': args.sigma}

# The main entry point for CLI
def main() -> int:
    
    args = parse_args()
    
    # Determine dataset type
    if args.test:
        dataset_type = 'test'
    elif args.synthetic:
        dataset_type = 'synthetic'
    elif args.real_world:
        dataset_type = 'real-world'
    else:
        print("Error: No dataset type specified")
        sys.exit(1)
    
    project_root = Path(__file__).resolve().parents[2]
    synthetic_dataset_path = project_root / DEFAULT_SYNTHETIC_DATASET_REL_PATH
    real_world_dataset_path = project_root / DEFAULT_REAL_WORLD_DATASET_REL_PATH

    if dataset_type == 'synthetic':
        dataset_path: str | None = str(synthetic_dataset_path)
    elif dataset_type == 'real-world':
        dataset_path = str(real_world_dataset_path)
    else:
        dataset_path = None

    if dataset_path and args.verbose:
        print(f"Using dataset path: {dataset_path}")
    
    try:
        if dataset_type == 'test' and args.max_images is not None:
            raise ValueError("--max-images is not supported with --test")

        # Check if comparison mode (multiple algorithms or --compare flag)
        algorithms_list = args.algorithm if isinstance(args.algorithm, list) else [args.algorithm]
        is_comparison = args.compare or len(algorithms_list) > 1
        max_images = resolve_max_images(args)

        if args.sigma_range:
            if dataset_type != 'synthetic':
                raise ValueError("--sigma-range can only be used with --synthetic")

            if is_comparison:
                display_names: list[str] = []
                for algo_name in algorithms_list:
                    algorithm_class = get_algorithm(algo_name)
                    algorithm = algorithm_class(**build_algorithm_params(algo_name, args))
                    display_names.append(algorithm.name)

                output_dir = resolve_comparison_output_dir(
                    project_root,
                    args.output,
                    dataset_type,
                    display_names,
                )
            else:
                algorithm_name = algorithms_list[0]
                algorithm_class = get_algorithm(algorithm_name)
                algorithm = algorithm_class(**build_algorithm_params(algorithm_name, args))
                output_dir = (
                    project_root
                    / 'results'
                    / 'single'
                    / format_algorithm_for_path(algorithm.name)
                    / dataset_type
                )
                if args.output and args.verbose:
                    print(f"Note: --output is ignored in single mode. Using: {output_dir}")

            return run_sigma_range(
                args,
                project_root,
                dataset_type,
                dataset_path,
                algorithms_list,
                is_comparison,
                max_images,
                output_dir,
            )

        # Load dataset
        dataset_loader = get_dataset_loader(
            dataset_type=dataset_type,
            dataset_path=dataset_path,
            noise_sigma=args.sigma,
            max_images=max_images,
            sample_seed=args.sample_seed,
        )
        
        if is_comparison:
            # Comparison mode: multiple algorithms
            algorithms = []
            for algo_name in algorithms_list:
                algorithm_class = get_algorithm(algo_name)
                algo_params = build_algorithm_params(algo_name, args)
                algorithms.append(algorithm_class(**algo_params))
                
            display_names = [algorithm.name for algorithm in algorithms]
            output_dir = resolve_comparison_output_dir(
                project_root,
                args.output,
                dataset_type,
                display_names,
            )

            all_results: dict[str, list[dict]] = {}
            for algorithm in algorithms:
                all_results[algorithm.name] = evaluate_or_reuse_algorithm_results(
                    algorithm=algorithm,
                    dataset_loader=dataset_loader,
                    args=args,
                    project_root=project_root,
                    dataset_type=dataset_type,
                    sigma=args.sigma,
                    max_images=max_images,
                    comparison_output_dir=output_dir,
                )

            comparison_evaluator = ComparisonEvaluator(
                algorithms=algorithms,
                dataset_loader=dataset_loader,
                verbose=args.verbose,
            )
            comparison_evaluator.all_results = all_results
            
            # Show image comparisons if requested
            if args.show_images:
                comparison_evaluator.show_image_comparison(num_images=args.num_display)

            display_names = [algorithm.name for algorithm in algorithms]
            output_dir = resolve_comparison_output_dir(
                project_root,
                args.output,
                dataset_type,
                display_names,
            )
            comparison_evaluator.save_comparison_summary(output_dir)
            comparison_evaluator.plot_comparison_from_csv(
                output_dir / 'metrics' / 'comparison_summary.csv',
                output_dir,
                show_plot=args.show_plot,
            )
            print(f"\nComparison results saved to: {output_dir}")
        
        else:
            # Single algorithm mode
            algorithm_name = algorithms_list[0]
            algorithm_class = get_algorithm(algorithm_name)
            algo_params = build_algorithm_params(algorithm_name, args)
            algorithm = algorithm_class(**algo_params)
            
            # Create evaluator
            evaluator = Evaluator(
                algorithm=algorithm,
                dataset_loader=dataset_loader,
                verbose=args.verbose,
                show_progress=(
                    not args.verbose
                    and dataset_type in ['synthetic', 'real-world']
                    and max_images is None
                ),
            )
            
            # Run evaluation
            evaluator.evaluate()
            
            # Show image comparisons if requested
            if args.show_images:
                evaluator.show_image_comparison(num_images=args.num_display)

            output_dir = (
                project_root
                / 'results'
                / 'single'
                / format_algorithm_for_path(algorithm.name)
                / dataset_type
            )

            if args.output and args.verbose:
                print(f"Note: --output is ignored in single mode. Using: {output_dir}")

            save_noisy_images = dataset_type in ['test', 'synthetic']
            evaluator.save_results(
                output_dir,
                save_images=True,
                sigma=args.sigma,
                save_noisy_images=save_noisy_images,
                split_image_dirs=save_noisy_images,
            )
            print(f"\nResults saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
