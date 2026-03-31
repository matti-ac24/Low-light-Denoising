# A Command-line interface for the denoiser package

import sys
import argparse
from pathlib import Path

from .algorithms import get_algorithm, ALGORITHMS
from .datasets import get_dataset_loader
from .evaluator import Evaluator, ComparisonEvaluator


def parse_sigma_sweep(sigma_sweep_arg: str) -> list[float]:

    values = [item.strip() for item in sigma_sweep_arg.split(',') if item.strip()]
    if not values:
        raise ValueError("--sigma-sweep must contain at least one sigma value")

    sigmas = []
    for value in values:
        sigma = float(value)
        if sigma <= 0:
            raise ValueError("Sigma values in --sigma-sweep must be > 0")
        sigmas.append(sigma)

    return sigmas


def run_sigma_sweep(
    args: argparse.Namespace,
    dataset_type: str,
    algorithms_list: list[str],
    is_comparison: bool,
) -> int:

    import csv
    import numpy as np
    import matplotlib.pyplot as plt

    sigma_values = parse_sigma_sweep(args.sigma_sweep)

    if dataset_type != 'synthetic':
        raise ValueError("--sigma-sweep is supported only with --synthetic datasets")

    output_dir = Path(args.output) if args.output else Path('results/sigma_sweep')
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"\nRunning sigma sweep for values: {sigma_values}")

    algo_metrics: dict[str, dict[str, list[float]]] = {}
    for algo_name in algorithms_list:
        algorithm_class = get_algorithm(algo_name)
        display_name = algorithm_class(**build_algorithm_params(algo_name, args)).name
        if display_name not in algo_metrics:
            algo_metrics[display_name] = {'psnr': [], 'ssim': []}

    for sigma in sigma_values:
        if args.verbose:
            print(f"\n{'-'*70}")
            print(f"Sigma sweep step: σ={sigma:.3f}")
            print(f"{'-'*70}")

        dataset_loader = get_dataset_loader(
            dataset_type=dataset_type,
            dataset_path=args.dataset_path,
            noise_sigma=sigma,
            max_images=1 if args.single_image else None,
            sample_seed=args.sample_seed,
        )

        if is_comparison:
            algorithms = []
            for algo_name in algorithms_list:
                algorithm_class = get_algorithm(algo_name)
                algo_params = build_algorithm_params(algo_name, args)
                algorithms.append(algorithm_class(**algo_params))

            comparison_evaluator = ComparisonEvaluator(
                algorithms=algorithms,
                dataset_loader=dataset_loader,
                verbose=args.verbose,
            )
            all_results = comparison_evaluator.evaluate_all()

            for algo_display_name, results in all_results.items():
                denoised_psnr = [r['denoised_metrics']['psnr'] for r in results]
                denoised_ssim = [r['denoised_metrics']['ssim'] for r in results]
                algo_metrics[algo_display_name]['psnr'].append(float(np.mean(denoised_psnr)))
                algo_metrics[algo_display_name]['ssim'].append(float(np.mean(denoised_ssim)))
        else:
            algorithm_name = algorithms_list[0]
            algorithm_class = get_algorithm(algorithm_name)
            algo_params = build_algorithm_params(algorithm_name, args)
            algorithm = algorithm_class(**algo_params)

            evaluator = Evaluator(
                algorithm=algorithm,
                dataset_loader=dataset_loader,
                verbose=args.verbose,
            )
            results = evaluator.evaluate()

            denoised_psnr = [r['denoised_metrics']['psnr'] for r in results]
            denoised_ssim = [r['denoised_metrics']['ssim'] for r in results]
            algo_metrics[algorithm.name]['psnr'].append(float(np.mean(denoised_psnr)))
            algo_metrics[algorithm.name]['ssim'].append(float(np.mean(denoised_ssim)))

    summary_csv_path = output_dir / 'sigma_sweep_summary.csv'
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
    psnr_plot_path = output_dir / 'psnr_vs_sigma.png'
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
    ssim_plot_path = output_dir / 'ssim_vs_sigma.png'
    plt.savefig(ssim_plot_path, dpi=150, bbox_inches='tight')
    if args.show_plot:
        plt.show()
    else:
        plt.close()

    print(f"\nSigma sweep complete!")
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
  python -m denoiser --synthetic nl-means --dataset-path ./datasets/images

    # Run Residual U-Net CNN with pretrained weights
    python -m denoiser --test resunet --model-path ./models/resunet.pth --device cpu
  
  # Compare algorithms side-by-side
  python -m denoiser --test --compare bm3d nl-means --output results/comparison --plot
  
  # Run BM3D on real-world paired images
  python -m denoiser --real-world bm3d --dataset-path ./datasets/paired
  
  # Customize noise level and save results
  python -m denoiser --synthetic bm3d --sigma 0.15 --output results/bm3d_sigma015/
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
        help='Use synthetic dataset (add noise to clean images)'
    )
    dataset_group.add_argument(
        '--real-world',
        action='store_true',
        help='Use real-world paired dataset (clean/noisy pairs)'
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
    
    # Dataset options
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Path to dataset directory (required for --synthetic and --real-world)'
    )
    
    parser.add_argument(
        '--sigma',
        type=float,
        default=0.1,
        help='Noise standard deviation for synthetic datasets (default: 0.1)'
    )

    parser.add_argument(
        '--sigma-sweep',
        type=str,
        default=None,
        help='Comma-separated sigma values for sweep plots (e.g. 0.05,0.1,0.15)'
    )

    parser.add_argument(
        '--single-image',
        action='store_true',
        help='Automatically select one random image from the dataset for a fast run'
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
        '--model-path',
        type=str,
        default=None,
        help='Path to pretrained CNN weights (.pt/.pth) for ResUNet'
    )

    parser.add_argument(
        '--base-channels',
        type=int,
        default=32,
        help='Base feature channels for ResUNet architecture (default: 32)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help="Runtime device for CNN inference: 'auto', 'cpu', or 'cuda'"
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results (optional)'
    )
    
    parser.add_argument(
        '--save-images',
        action='store_true',
        help='Save denoised images to output directory'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate and save performance plots'
    )
    
    parser.add_argument(
        '--show-plot',
        action='store_true',
        help='Display plots interactively (requires --plot)'
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
            'model_path': args.model_path,
            'base_channels': args.base_channels,
            'device': args.device,
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
    
    # Validate dataset path
    if dataset_type in ['synthetic', 'real-world'] and not args.dataset_path:
        print(f"Error: --dataset-path required for --{dataset_type} datasets")
        sys.exit(1)
    
    try:
        # Check if comparison mode (multiple algorithms or --compare flag)
        algorithms_list = args.algorithm if isinstance(args.algorithm, list) else [args.algorithm]
        is_comparison = args.compare or len(algorithms_list) > 1

        if args.sigma_sweep:
            return run_sigma_sweep(args, dataset_type, algorithms_list, is_comparison)

        # Load dataset
        dataset_loader = get_dataset_loader(
            dataset_type=dataset_type,
            dataset_path=args.dataset_path,
            noise_sigma=args.sigma,
            max_images=1 if args.single_image else None,
            sample_seed=args.sample_seed,
        )
        
        if is_comparison:
            # Comparison mode: multiple algorithms
            algorithms = []
            for algo_name in algorithms_list:
                algorithm_class = get_algorithm(algo_name)
                algo_params = build_algorithm_params(algo_name, args)
                algorithms.append(algorithm_class(**algo_params))
            
            # Create comparison evaluator
            comparison_evaluator = ComparisonEvaluator(
                algorithms=algorithms,
                dataset_loader=dataset_loader,
                verbose=args.verbose
            )
            
            # Run comparison evaluation
            all_results = comparison_evaluator.evaluate_all()
            
            # Show image comparisons if requested
            if args.show_images:
                comparison_evaluator.show_image_comparison(num_images=args.num_display)
            
            # Save and plot results if output directory specified
            if args.output:
                output_dir = Path(args.output)
                comparison_evaluator.save_comparison_summary(output_dir)
                print(f"\nComparison results saved to: {output_dir}")
                
                # Generate comparison plots if requested
                if args.plot:
                    comparison_evaluator.plot_comparison(output_dir, show_plot=args.show_plot)
            elif args.plot:
                # If --plot is specified without --output, create a default output directory
                output_dir = Path('results/comparison')
                comparison_evaluator.plot_comparison(output_dir, show_plot=args.show_plot)
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
                verbose=args.verbose
            )
            
            # Run evaluation
            results = evaluator.evaluate()
            
            # Show image comparisons if requested
            if args.show_images:
                evaluator.show_image_comparison(num_images=args.num_display)
            
            # Save results if output directory specified
            if args.output:
                output_dir = Path(args.output)
                evaluator.save_results(output_dir, save_images=args.save_images)
                print(f"\nResults saved to: {output_dir}")
                
                # Generate plots if requested
                if args.plot:
                    evaluator.plot_results(output_dir, show_plot=args.show_plot)
            elif args.plot:
                # If --plot is specified without --output, create a default output directory
                output_dir = Path('results')
                evaluator.plot_results(output_dir, show_plot=args.show_plot)
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
