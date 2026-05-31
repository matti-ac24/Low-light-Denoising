# Evaluator module for running denoising experiments

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Union

from .utils.metrics import calculate_all_metrics

# Evaluator for a single denoising algorithm
class Evaluator:

    @staticmethod
    def _format_sigma_suffix(sigma: float | None) -> str:
        """Format the sigma value for filenames and directories."""
        if sigma is None:
            return ''
        formatted = f"{sigma:.4f}".rstrip('0').rstrip('.').replace('.', 'p')
        return f"_sigma_{formatted}"
    
    # Initialise the evaluator with algorithm, dataset loader, and verbosity settings
    def __init__(
        self,
        algorithm: Any,
        dataset_loader: Any,
        verbose: bool = True,
        show_progress: bool = False,
        stream_output_dir: Union[str, Path, None] = None,
        stream_sigma: float | None = None,
        stream_save_noisy_images: bool = False,
        stream_split_image_dirs: bool = False,
    ) -> None:

        """Initialize the object with the provided settings."""
        self.algorithm = algorithm
        self.dataset_loader = dataset_loader
        self.verbose = verbose
        self.show_progress = show_progress
        self.results = []
        self.stream_output_dir = Path(stream_output_dir) if stream_output_dir is not None else None
        self.stream_sigma = stream_sigma
        self.stream_save_noisy_images = stream_save_noisy_images
        self.stream_split_image_dirs = stream_split_image_dirs

    def _save_result_images_streaming(self, result: dict[str, Any]) -> None:

        """Save the current image outputs as they are produced."""
        if self.stream_output_dir is None:
            return

        try:
            from skimage import io

            images_dir = self.stream_output_dir / 'images'
            images_dir.mkdir(parents=True, exist_ok=True)

            denoised_dir = images_dir / 'denoised' if self.stream_split_image_dirs else images_dir
            noisy_dir = images_dir / 'noisy' if self.stream_split_image_dirs else images_dir
            denoised_dir.mkdir(parents=True, exist_ok=True)
            if self.stream_save_noisy_images:
                noisy_dir.mkdir(parents=True, exist_ok=True)

            sigma_suffix = self._format_sigma_suffix(self.stream_sigma)
            denoised_path = denoised_dir / f"{result['name']}_denoised{sigma_suffix}.png"
            denoised_uint8 = (np.clip(result['denoised'], 0, 1) * 255).astype(np.uint8)
            io.imsave(denoised_path, denoised_uint8)

            if self.stream_save_noisy_images:
                noisy_path = noisy_dir / f"{result['name']}_noisy{sigma_suffix}.png"
                noisy_uint8 = (np.clip(result['noisy'], 0, 1) * 255).astype(np.uint8)
                io.imsave(noisy_path, noisy_uint8)
        except Exception as e:
            print(f"Warning: Could not stream-save image '{result.get('name', 'unknown')}': {e}")
    
    # Run evaluation on the dataset and return its results
    def evaluate(self) -> list[dict[str, Any]]:

        """Evaluate the model or results using the provided inputs."""
        if self.verbose:
            print(f"\n{'='*70}", flush=True)
            print(f"Denoising Evaluation", flush=True)
            print(f"{'='*70}", flush=True)
            print(f"Algorithm: {self.algorithm.name}", flush=True)
            print(f"Dataset Type: {self.dataset_loader.dataset_type}", flush=True)
            if self.dataset_loader.dataset_type == 'synthetic':
                print(f"Noise Level (σ): {self.dataset_loader.noise_sigma:.3f}", flush=True)
            print(f"{'='*70}\n", flush=True)
        
        # Load images
        if self.verbose:
            print("Loading dataset...", flush=True)
        
        images = self.dataset_loader.load_images()
        
        if not images:
            print("Error: No images loaded!", flush=True)
            return []
        
        if self.verbose:
            print(f"Loaded {len(images)} image(s)\n", flush=True)
        
        # Process each image
        self.results = []

        use_progress_bar = (
            self.show_progress
            and not self.verbose
            and self.dataset_loader.dataset_type in ['synthetic', 'real-world']
        )

        iterable = enumerate(images, 1)
        if use_progress_bar:
            try:
                from tqdm.auto import tqdm

                iterable = tqdm(
                    iterable,
                    total=len(images),
                    desc=f"{self.algorithm.name}",
                    unit='img',
                )
            except Exception:
                # Fall back to regular iteration if tqdm is unavailable.
                iterable = enumerate(images, 1)

        for idx, image_data in iterable:
            if self.verbose:
                print(f"[{idx}/{len(images)}] Processing: {image_data['name']}", flush=True)
            
            result = self._process_image(image_data)

            if self.stream_output_dir is not None:
                self._save_result_images_streaming(result)
                # Keep only lightweight metrics in memory to prevent OOM on large datasets.
                result.pop('clean', None)
                result.pop('noisy', None)
                result.pop('denoised', None)

            self.results.append(result)
            
            if self.verbose:
                self._print_result(result)
        
        # Print summary
        if self.verbose and len(self.results) > 1:
            self._print_summary()

        if self.stream_output_dir is not None:
            # Persist CSV metrics after streaming image writes.
            self.save_results(
                self.stream_output_dir,
                save_images=False,
                sigma=self.stream_sigma,
                save_noisy_images=self.stream_save_noisy_images,
                split_image_dirs=self.stream_split_image_dirs,
            )
        
        return self.results
    
    # Process a single image and return its metrics
    def _process_image(self, image_data: dict[str, Any]) -> dict[str, Any]:

        """Process a single image through the denoiser pipeline."""
        if 'clean' in image_data and 'noisy' in image_data:
            clean = image_data['clean']
            noisy = image_data['noisy']
        else:
            from skimage import io, img_as_float

            clean = img_as_float(io.imread(image_data['clean_path']))
            noisy = img_as_float(io.imread(image_data['noisy_path']))
        name = image_data['name']
        
        # Measure denoising time
        start_time = time.time()
        denoised = self.algorithm.denoise(noisy)
        processing_time = time.time() - start_time
        
        # Calculate metrics
        metrics_noisy = calculate_all_metrics(clean, noisy)
        metrics_denoised = calculate_all_metrics(clean, denoised)
        
        return {
            'name': name,
            'processing_time': processing_time,
            'noisy_metrics': metrics_noisy,
            'denoised_metrics': metrics_denoised,
            'clean': clean,
            'noisy': noisy,
            'denoised': denoised
        }
    
    # Print results for a single image
    def _print_result(self, result: dict[str, Any]) -> None:

        """Print a single image result summary."""
        print(f"  Processing time: {result['processing_time']:.3f}s")
        
        # Noisy image metrics
        noisy_metrics = result['noisy_metrics']
        print(f"  Noisy image:     PSNR={noisy_metrics['psnr']:.2f} dB, SSIM={noisy_metrics['ssim']:.4f}")
        
        # Denoised image metrics
        denoised_metrics = result['denoised_metrics']
        print(f"  Denoised image:  PSNR={denoised_metrics['psnr']:.2f} dB, SSIM={denoised_metrics['ssim']:.4f}")
        
        # Improvement
        psnr_improvement = denoised_metrics['psnr'] - noisy_metrics['psnr']
        print(f"  Improvement:     ΔPSNR={psnr_improvement:+.2f} dB", flush=True)
        print(flush=True)
    
    # Print summary statistics across all images
    def _print_summary(self) -> None:

        """Print the aggregated evaluation summary."""
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")
        
        # Calculate averages

        #PSNR
        avg_time = np.mean([r['processing_time'] for r in self.results])
        avg_noisy_psnr = np.mean([r['noisy_metrics']['psnr'] for r in self.results])
        avg_denoised_psnr = np.mean([r['denoised_metrics']['psnr'] for r in self.results])
        avg_psnr_improvement = avg_denoised_psnr - avg_noisy_psnr
        
        print(f"Images processed: {len(self.results)}", flush=True)
        print(f"Average processing time: {avg_time:.3f}s", flush=True)
        print(f"Average noisy PSNR: {avg_noisy_psnr:.2f} dB", flush=True)
        print(f"Average denoised PSNR: {avg_denoised_psnr:.2f} dB", flush=True)
        print(f"Average PSNR improvement: {avg_psnr_improvement:+.2f} dB", flush=True)
        
        #SSIM
        avg_noisy_ssim = np.mean([r['noisy_metrics']['ssim'] for r in self.results])
        avg_denoised_ssim = np.mean([r['denoised_metrics']['ssim'] for r in self.results])
        print(f"Average noisy SSIM: {avg_noisy_ssim:.4f}", flush=True)
        print(f"Average denoised SSIM: {avg_denoised_ssim:.4f}", flush=True)
            
        print(f"{'='*70}\n", flush=True)
    
    # Generate and save performance plots for the evaluation results
    def plot_results(self, output_dir: Union[str, Path], show_plot: bool = False, sigma: float | None = None) -> None:

        """Plot the evaluation results for a single algorithm."""
        if not self.results:
            print("Warning: No results to plot!")
            return
        
        output_dir = Path(output_dir)
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        image_names = [r['name'] for r in self.results]
        noisy_psnr = [r['noisy_metrics']['psnr'] for r in self.results]
        denoised_psnr = [r['denoised_metrics']['psnr'] for r in self.results]
        noisy_ssim = [r['noisy_metrics']['ssim'] for r in self.results]
        denoised_ssim = [r['denoised_metrics']['ssim'] for r in self.results]
        
        # Create figure with subplots
        n_plots = 2
        _ , axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        
        x = np.arange(len(image_names))
        width = 0.35
        
        # Plot PSNR
        ax = axes[0]
        ax.bar(x - width/2, noisy_psnr, width, label='Noisy', alpha=0.8, color='#e74c3c')
        ax.bar(x + width/2, denoised_psnr, width, label='Denoised', alpha=0.8, color='#2ecc71')
        ax.set_xlabel('Image', fontsize=11)
        ax.set_ylabel('PSNR (dB)', fontsize=11)
        ax.set_title(f'PSNR Comparison - {self.algorithm.name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(image_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot SSIM
        ax = axes[1]
        ax.bar(x - width/2, noisy_ssim, width, label='Noisy', alpha=0.8, color='#e74c3c')
        ax.bar(x + width/2, denoised_ssim, width, label='Denoised', alpha=0.8, color='#2ecc71')
        ax.set_xlabel('Image', fontsize=11)
        ax.set_ylabel('SSIM', fontsize=11)
        ax.set_title(f'SSIM Comparison - {self.algorithm.name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(image_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save plot
        sigma_suffix = self._format_sigma_suffix(sigma)
        plot_path = plots_dir / f'metrics_plot_{self.algorithm.name.replace(" ", "_").lower()}{sigma_suffix}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        if self.verbose:
            print(f"Plot saved to: {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    # Save evaluation results and optionally images to disk.
    def save_results(
        self,
        output_dir: Union[str, Path],
        save_images: bool = True,
        sigma: float | None = None,
        save_noisy_images: bool = False,
        split_image_dirs: bool = False,
    ) -> None:

        """Save the evaluation results and derived artifacts."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = output_dir / 'metrics'
        metrics_dir.mkdir(exist_ok=True)
        
        # Save metrics to CSV
        import csv
        
        sigma_suffix = self._format_sigma_suffix(sigma)
        csv_path = metrics_dir / f'metrics{sigma_suffix}.csv'
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['image', 'processing_time', 'noisy_psnr', 'denoised_psnr', 
                         'psnr_improvement', 'noisy_ssim', 'denoised_ssim']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'image': result['name'],
                    'processing_time': result['processing_time'],
                    'noisy_psnr': result['noisy_metrics']['psnr'],
                    'denoised_psnr': result['denoised_metrics']['psnr'],
                    'psnr_improvement': result['denoised_metrics']['psnr'] - result['noisy_metrics']['psnr'],
                    'noisy_ssim': result['noisy_metrics']['ssim'],
                    'denoised_ssim': result['denoised_metrics']['ssim']
                }                
                
                writer.writerow(row)
        
        if self.verbose:
            print(f"Metrics saved to: {csv_path}")
        
        # Save output images
        if save_images:
            try:
                from skimage import io
                
                images_dir = output_dir / 'images'
                images_dir.mkdir(exist_ok=True)

                denoised_dir = images_dir / 'denoised' if split_image_dirs else images_dir
                noisy_dir = images_dir / 'noisy' if split_image_dirs else images_dir
                denoised_dir.mkdir(exist_ok=True)
                if save_noisy_images:
                    noisy_dir.mkdir(exist_ok=True)
                
                for result in self.results:
                    img_path = denoised_dir / f"{result['name']}_denoised{sigma_suffix}.png"
                    # Convert to uint8 for saving
                    img_uint8 = (np.clip(result['denoised'], 0, 1) * 255).astype(np.uint8)
                    io.imsave(img_path, img_uint8)

                    if save_noisy_images:
                        noisy_path = noisy_dir / f"{result['name']}_noisy{sigma_suffix}.png"
                        noisy_uint8 = (np.clip(result['noisy'], 0, 1) * 255).astype(np.uint8)
                        io.imsave(noisy_path, noisy_uint8)
                
                if self.verbose:
                    print(f"Denoised images saved to: {denoised_dir}")
                    if save_noisy_images:
                        print(f"Noisy images saved to: {noisy_dir}")
            except Exception as e:
                print(f"Warning: Could not save images: {e}")
    
# Evaluator for comparing multiple denoising algorithms
class ComparisonEvaluator:

    # Initialise comparison evaluator with multiple algorithms and dataset loader
    def __init__(self, algorithms: list[Any], dataset_loader: Any, verbose: bool = True) -> None:

        """Initialize the object with the provided settings."""
        self.algorithms = algorithms
        self.dataset_loader = dataset_loader
        self.verbose = verbose
        self.all_results = {}  # Dictionary mapping algorithm name to results
    
    # Run evaluation for all algorithms and return combined results
    def evaluate_all(self) -> dict[str, list[dict[str, Any]]]:

        """Evaluate every configured algorithm on the dataset."""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Alogrithm Comparison")
            print(f"{'='*70}")
            print(f"Algorithms: {', '.join([algo.name for algo in self.algorithms])}")
            print(f"Dataset Type: {self.dataset_loader.dataset_type}")
            if self.dataset_loader.dataset_type == 'synthetic':
                print(f"Noise Level (σ): {self.dataset_loader.noise_sigma:.3f}")
            print(f"{'='*70}\n")
        
        # Run evaluation for each algorithm
        for algorithm in self.algorithms:
            evaluator = Evaluator(algorithm, self.dataset_loader, verbose=self.verbose)
            results = evaluator.evaluate()
            self.all_results[algorithm.name] = results
        
        return self.all_results

    @staticmethod
    def _load_comparison_summary_csv(csv_path: Union[str, Path]) -> dict[str, Any]:
        """Load a saved comparison summary from CSV."""
        import csv

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Comparison summary CSV not found: {csv_path}")

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError(f"Comparison summary CSV is empty: {csv_path}")

        fieldnames = reader.fieldnames or []
        algorithm_names: list[str] = []
        for field in fieldnames:
            if field in {'image', 'noisy_psnr', 'noisy_ssim'}:
                continue
            if field.endswith('_psnr'):
                algorithm_names.append(field[:-5])

        image_names = [row['image'] for row in rows]

        metrics_data: dict[str, dict[str, list[float]]] = {}
        for algo_name in algorithm_names:
            metrics_data[algo_name] = {
                'psnr': [],
                'ssim': [],
            }

        noisy_metrics = {
            'psnr': [],
            'ssim': [],
        }

        for row in rows:
            if 'noisy_psnr' in row and row['noisy_psnr'] not in (None, ''):
                noisy_metrics['psnr'].append(float(row['noisy_psnr']))
                noisy_metrics['ssim'].append(float(row['noisy_ssim']))
            else:
                noisy_metrics['psnr'].append(float('nan'))
                noisy_metrics['ssim'].append(float('nan'))

            for algo_name in algorithm_names:
                metrics_data[algo_name]['psnr'].append(float(row[f'{algo_name}_psnr']))
                metrics_data[algo_name]['ssim'].append(float(row[f'{algo_name}_ssim']))

        return {
            'image_names': image_names,
            'algorithm_names': algorithm_names,
            'metrics_data': metrics_data,
            'noisy_metrics': noisy_metrics,
        }

    def _plot_comparison_from_data(
        self,
        image_names: list[str],
        algorithm_names: list[str],
        metrics_data: dict[str, dict[str, list[float]]],
        noisy_metrics: dict[str, list[float]],
        output_dir: Union[str, Path],
        show_plot: bool = False,
    ) -> None:

        """Plot comparison metrics from in-memory data."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)

        _, axes = plt.subplots(1, 2, figsize=(14, 6))

        box_colors = plt.cm.Set2(np.linspace(0, 1, max(1, len(algorithm_names))))

        # PSNR distribution boxplot
        ax = axes[0]
        psnr_data = [metrics_data[algo_name]['psnr'] for algo_name in algorithm_names]
        psnr_plot = ax.boxplot(
            psnr_data,
            labels=algorithm_names,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            medianprops={'color': '#2c3e50', 'linewidth': 1.8},
            meanprops={'color': '#e74c3c', 'linewidth': 1.8},
        )
        for patch, color in zip(psnr_plot['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title('PSNR Distribution by Algorithm', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # SSIM distribution boxplot
        ax = axes[1]
        ssim_data = [metrics_data[algo_name]['ssim'] for algo_name in algorithm_names]
        ssim_plot = ax.boxplot(
            ssim_data,
            labels=algorithm_names,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            medianprops={'color': '#2c3e50', 'linewidth': 1.8},
            meanprops={'color': '#e74c3c', 'linewidth': 1.8},
        )
        for patch, color in zip(ssim_plot['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_ylabel('SSIM', fontsize=12)
        ax.set_title('SSIM Distribution by Algorithm', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])

        if noisy_metrics.get('psnr'):
            valid_noisy_psnr = [v for v in noisy_metrics['psnr'] if not np.isnan(v)]
            if valid_noisy_psnr:
                axes[0].axhline(
                    np.median(valid_noisy_psnr),
                    linestyle='--',
                    linewidth=1.4,
                    color='#7f8c8d',
                    label='Noisy median',
                )
                axes[0].legend(loc='best')

        if noisy_metrics.get('ssim'):
            valid_noisy_ssim = [v for v in noisy_metrics['ssim'] if not np.isnan(v)]
            if valid_noisy_ssim:
                axes[1].axhline(
                    np.median(valid_noisy_ssim),
                    linestyle='--',
                    linewidth=1.4,
                    color='#7f8c8d',
                    label='Noisy median',
                )
                axes[1].legend(loc='best')

        plt.tight_layout()

        algo_names = '_vs_'.join([name.replace(' ', '_').lower() for name in algorithm_names])
        plot_path = plots_dir / f'comparison_{algo_names}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')

        if self.verbose:
            print(f"\nComparison plot saved to: {plot_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()
    
    # Generate comparison plots for all algorithms
    def plot_comparison(self, output_dir: Union[str, Path], show_plot: bool = False) -> None:

        """Plot the comparison results for the configured algorithms."""
        if not self.all_results:
            print("Warning: No results to plot!")
            return
        first_algo = list(self.all_results.keys())[0]
        image_names = [r['name'] for r in self.all_results[first_algo]]
        algorithm_names = list(self.all_results.keys())

        metrics_data: dict[str, dict[str, list[float]]] = {}
        for algo_name, results in self.all_results.items():
            metrics_data[algo_name] = {
                'psnr': [r['denoised_metrics']['psnr'] for r in results],
                'ssim': [r['denoised_metrics']['ssim'] for r in results],
            }

        noisy_metrics = {
            'psnr': [r['noisy_metrics']['psnr'] for r in self.all_results[first_algo]],
            'ssim': [r['noisy_metrics']['ssim'] for r in self.all_results[first_algo]],
        }

        self._plot_comparison_from_data(
            image_names=image_names,
            algorithm_names=algorithm_names,
            metrics_data=metrics_data,
            noisy_metrics=noisy_metrics,
            output_dir=output_dir,
            show_plot=show_plot,
        )

    def plot_comparison_from_csv(
        self,
        summary_csv_path: Union[str, Path],
        output_dir: Union[str, Path],
        show_plot: bool = False,
    ) -> None:

        """Plot the comparison results from a CSV summary."""
        data = self._load_comparison_summary_csv(summary_csv_path)
        self._plot_comparison_from_data(
            image_names=data['image_names'],
            algorithm_names=data['algorithm_names'],
            metrics_data=data['metrics_data'],
            noisy_metrics=data['noisy_metrics'],
            output_dir=output_dir,
            show_plot=show_plot,
        )
    
    # Save a comparison summary CSV with all algorithms' metrics
    def save_comparison_summary(self, output_dir: Union[str, Path]) -> None:

        """Save the comparison summary to disk."""
        if not self.all_results:
            print("Warning: No results to save!")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = output_dir / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        import csv
        
        csv_path = metrics_dir / 'comparison_summary.csv'
        
        # Get image names from first algorithm
        first_algo = list(self.all_results.keys())[0]
        image_names = [r['name'] for r in self.all_results[first_algo]]
        
        with open(csv_path, 'w', newline='') as f:
            # Build fieldnames
            fieldnames = ['image', 'noisy_psnr', 'noisy_ssim']
            for algo_name in self.all_results.keys():
                fieldnames.extend([
                    f'{algo_name}_psnr',
                    f'{algo_name}_ssim'
                ])
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write data for each image
            for idx, img_name in enumerate(image_names):
                row = {
                    'image': img_name,
                    'noisy_psnr': self.all_results[first_algo][idx]['noisy_metrics']['psnr'],
                    'noisy_ssim': self.all_results[first_algo][idx]['noisy_metrics']['ssim'],
                }
                
                for algo_name, results in self.all_results.items():
                    result = results[idx]
                    row[f'{algo_name}_psnr'] = result['denoised_metrics']['psnr']
                    row[f'{algo_name}_ssim'] = result['denoised_metrics']['ssim']         
                
                writer.writerow(row)
        
        if self.verbose:
            print(f"Comparison summary saved to: {csv_path}")
    
