# Evaluator module for running denoising experiments

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Union

from .utils.metrics import calculate_all_metrics

# Evaluator for a single denoising algorithm
class Evaluator:
    
    # Initialise the evaluator with algorithm, dataset loader, and verbosity settings
    def __init__(self, algorithm: Any, dataset_loader: Any, verbose: bool = True) -> None:

        self.algorithm = algorithm
        self.dataset_loader = dataset_loader
        self.verbose = verbose
        self.results = []
    
    # Run evaluation on the dataset and return its results
    def evaluate(self) -> List[Dict[str, Any]]:

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Denoising Evaluation")
            print(f"{'='*70}")
            print(f"Algorithm: {self.algorithm.name}")
            print(f"Dataset Type: {self.dataset_loader.dataset_type}")
            if self.dataset_loader.dataset_type == 'synthetic':
                print(f"Noise Level (σ): {self.dataset_loader.noise_sigma:.3f}")
            print(f"{'='*70}\n")
        
        # Load images
        if self.verbose:
            print("Loading dataset...")
        
        images = self.dataset_loader.load_images()
        
        if not images:
            print("Error: No images loaded!")
            return []
        
        if self.verbose:
            print(f"Loaded {len(images)} image(s)\n")
        
        # Process each image
        self.results = []
        
        for idx, image_data in enumerate(images, 1):
            if self.verbose:
                print(f"[{idx}/{len(images)}] Processing: {image_data['name']}")
            
            result = self._process_image(image_data)
            self.results.append(result)
            
            if self.verbose:
                self._print_result(result)
        
        # Print summary
        if self.verbose and len(self.results) > 1:
            self._print_summary()
        
        return self.results
    
    # Process a single image and return its metrics
    def _process_image(self, image_data: Dict[str, Any]) -> Dict[str, Any]:

        clean = image_data['clean']
        noisy = image_data['noisy']
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
    def _print_result(self, result: Dict[str, Any]) -> None:

        print(f"  Processing time: {result['processing_time']:.3f}s")
        
        # Noisy image metrics
        noisy_metrics = result['noisy_metrics']
        print(f"  Noisy image:     PSNR={noisy_metrics['psnr']:.2f} dB, MSE={noisy_metrics['mse']:.6f}", end='')
        print(f", SSIM={noisy_metrics['ssim']:.4f}")
        
        # Denoised image metrics
        denoised_metrics = result['denoised_metrics']
        print(f"  Denoised image:  PSNR={denoised_metrics['psnr']:.2f} dB, MSE={denoised_metrics['mse']:.6f}", end='')
        print(f", SSIM={denoised_metrics['ssim']:.4f}")
        
        # Improvement
        psnr_improvement = denoised_metrics['psnr'] - noisy_metrics['psnr']
        print(f"  Improvement:     ΔPSNR={psnr_improvement:+.2f} dB")
        print()
    
    # Print summary statistics across all images
    def _print_summary(self) -> None:

        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")
        
        # Calculate averages

        #PSNR
        avg_time = np.mean([r['processing_time'] for r in self.results])
        avg_noisy_psnr = np.mean([r['noisy_metrics']['psnr'] for r in self.results])
        avg_denoised_psnr = np.mean([r['denoised_metrics']['psnr'] for r in self.results])
        avg_psnr_improvement = avg_denoised_psnr - avg_noisy_psnr
        
        print(f"Images processed: {len(self.results)}")
        print(f"Average processing time: {avg_time:.3f}s")
        print(f"Average noisy PSNR: {avg_noisy_psnr:.2f} dB")
        print(f"Average denoised PSNR: {avg_denoised_psnr:.2f} dB")
        print(f"Average PSNR improvement: {avg_psnr_improvement:+.2f} dB")
        
        #SSIM
        avg_noisy_ssim = np.mean([r['noisy_metrics']['ssim'] for r in self.results])
        avg_denoised_ssim = np.mean([r['denoised_metrics']['ssim'] for r in self.results])
        print(f"Average noisy SSIM: {avg_noisy_ssim:.4f}")
        print(f"Average denoised SSIM: {avg_denoised_ssim:.4f}")
            
        print(f"{'='*70}\n")
    
    # Generate and save performance plots for the evaluation results
    def plot_results(self, output_dir: Union[str, Path], show_plot: bool = False) -> None:

        if not self.results:
            print("Warning: No results to plot!")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        image_names = [r['name'] for r in self.results]
        noisy_psnr = [r['noisy_metrics']['psnr'] for r in self.results]
        denoised_psnr = [r['denoised_metrics']['psnr'] for r in self.results]
        noisy_mse = [r['noisy_metrics']['mse'] for r in self.results]
        denoised_mse = [r['denoised_metrics']['mse'] for r in self.results]
        noisy_ssim = [r['noisy_metrics']['ssim'] for r in self.results]
        denoised_ssim = [r['denoised_metrics']['ssim'] for r in self.results]
        
        # Create figure with subplots
        n_plots = 3
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
        
        # Plot MSE
        ax = axes[1]
        ax.bar(x - width/2, noisy_mse, width, label='Noisy', alpha=0.8, color='#e74c3c')
        ax.bar(x + width/2, denoised_mse, width, label='Denoised', alpha=0.8, color='#2ecc71')
        ax.set_xlabel('Image', fontsize=11)
        ax.set_ylabel('MSE', fontsize=11)
        ax.set_title(f'MSE Comparison - {self.algorithm.name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(image_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot SSIM
        ax = axes[2]
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
        plot_path = output_dir / f'metrics_plot_{self.algorithm.name.replace(" ", "_").lower()}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        if self.verbose:
            print(f"Plot saved to: {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    # Save evaluation results and optionally images to disk.
    def save_results(self, output_dir: Union[str, Path], save_images: bool = True) -> None:

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics to CSV
        import csv
        
        csv_path = output_dir / 'metrics.csv'
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['image', 'processing_time', 'noisy_psnr', 'denoised_psnr', 
                         'psnr_improvement', 'noisy_mse', 'denoised_mse', 'noisy_ssim', 'denoised_ssim']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'image': result['name'],
                    'processing_time': result['processing_time'],
                    'noisy_psnr': result['noisy_metrics']['psnr'],
                    'denoised_psnr': result['denoised_metrics']['psnr'],
                    'psnr_improvement': result['denoised_metrics']['psnr'] - result['noisy_metrics']['psnr'],
                    'noisy_mse': result['noisy_metrics']['mse'],
                    'denoised_mse': result['denoised_metrics']['mse'],
                    'noisy_ssim': result['noisy_metrics']['ssim'],
                    'denoised_ssim': result['denoised_metrics']['ssim']
                }                
                
                writer.writerow(row)
        
        if self.verbose:
            print(f"Metrics saved to: {csv_path}")
        
        # Save denoised images
        if save_images:
            try:
                from skimage import io
                
                images_dir = output_dir / 'denoised_images'
                images_dir.mkdir(exist_ok=True)
                
                for result in self.results:
                    img_path = images_dir / f"{result['name']}_denoised.png"
                    # Convert to uint8 for saving
                    img_uint8 = (result['denoised'] * 255).astype(np.uint8)
                    io.imsave(img_path, img_uint8)
                
                if self.verbose:
                    print(f"Images saved to: {images_dir}")
            except Exception as e:
                print(f"Warning: Could not save images: {e}")
    
    # Display side-by-side comparison of random noisy and denoised images
    def show_image_comparison(self, num_images: int = 3) -> None:

        if not self.results:
            print("Warning: No results to display!")
            return
        
        # Select random subset of images
        num_to_show = min(num_images, len(self.results))
        indices = np.random.choice(len(self.results), size=num_to_show, replace=False)
        selected_results = [self.results[i] for i in indices]
        
        # Create figure with subplots
        fig, axes = plt.subplots(num_to_show, 3, figsize=(15, 5 * num_to_show))
        
        # Handle single image case
        if num_to_show == 1:
            axes = axes.reshape(1, -1)
        
        for idx, result in enumerate(selected_results):
            # Get images
            clean = result['clean']
            noisy = result['noisy']
            denoised = result['denoised']
            
            # Get metrics
            noisy_psnr = result['noisy_metrics']['psnr']
            denoised_psnr = result['denoised_metrics']['psnr']
            improvement = denoised_psnr - noisy_psnr
            
            # Display clean image
            axes[idx, 0].imshow(clean, cmap='gray' if clean.ndim == 2 else None)
            axes[idx, 0].set_title(f"Clean: {result['name']}", fontsize=12, fontweight='bold')
            axes[idx, 0].axis('off')
            
            # Display noisy image
            axes[idx, 1].imshow(noisy, cmap='gray' if noisy.ndim == 2 else None)
            axes[idx, 1].set_title(f"Noisy (PSNR: {noisy_psnr:.2f} dB)", fontsize=12)
            axes[idx, 1].axis('off')
            
            # Display denoised image
            axes[idx, 2].imshow(denoised, cmap='gray' if denoised.ndim == 2 else None)
            axes[idx, 2].set_title(f"Denoised (PSNR: {denoised_psnr:.2f} dB, Δ{improvement:+.2f} dB)", fontsize=12, color='green')
            axes[idx, 2].axis('off')
        
        plt.suptitle(f"Image Quality Comparison - {self.algorithm.name}", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()


# Evaluator for comparing multiple denoising algorithms
class ComparisonEvaluator:

    # Initialise comparison evaluator with multiple algorithms and dataset loader
    def __init__(self, algorithms: List[Any], dataset_loader: Any, verbose: bool = True) -> None:

        self.algorithms = algorithms
        self.dataset_loader = dataset_loader
        self.verbose = verbose
        self.all_results = {}  # Dictionary mapping algorithm name to results
    
    # Run evaluation for all algorithms and return combined results
    def evaluate_all(self) -> Dict[str, List[Dict[str, Any]]]:

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
    
    # Generate comparison plots for all algorithms
    def plot_comparison(self, output_dir: Union[str, Path], show_plot: bool = False) -> None:

        if not self.all_results:
            print("Warning: No results to plot!")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get image names from the first algorithm's results
        first_algo = list(self.all_results.keys())[0]
        image_names = [r['name'] for r in self.all_results[first_algo]]
        num_images = len(image_names)
        
        # Create figure with subplots for each metric
        n_metrics = 3
        _ , axes = plt.subplots(1, n_metrics, figsize=(7*n_metrics, 6))
        
        # Prepare data for each algorithm
        x = np.arange(num_images)
        width = 0.8 / (len(self.algorithms) + 1)  # +1 for noisy baseline
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.algorithms) + 1))
        
        # Extract metrics for all algorithms
        metrics_data = {}
        for algo_name, results in self.all_results.items():
            metrics_data[algo_name] = {
                'psnr': [r['denoised_metrics']['psnr'] for r in results],
                'mse': [r['denoised_metrics']['mse'] for r in results],
                'ssim': [r['denoised_metrics']['ssim'] for r in results]
            }
        
        # Get noisy baseline from first algorithm
        noisy_psnr = [r['noisy_metrics']['psnr'] for r in self.all_results[first_algo]]
        noisy_mse = [r['noisy_metrics']['mse'] for r in self.all_results[first_algo]]
        noisy_ssim = [r['noisy_metrics']['ssim'] for r in self.all_results[first_algo]]
        
        # Plot PSNR comparison
        ax = axes[0]
        offset = -(len(self.algorithms)) * width / 2
        ax.bar(x + offset, noisy_psnr, width, label='Noisy', alpha=0.7, color=colors[0])
        offset += width
        
        for idx, (algo_name, data) in enumerate(metrics_data.items()):
            ax.bar(x + offset, data['psnr'], width, label=algo_name, alpha=0.8, color=colors[idx + 1])
            offset += width
        
        ax.set_xlabel('Image', fontsize=12)
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title('PSNR Comparison Across Algorithms', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(image_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot MSE comparison
        ax = axes[1]
        offset = -(len(self.algorithms)) * width / 2
        ax.bar(x + offset, noisy_mse, width, label='Noisy', alpha=0.7, color=colors[0])
        offset += width
        
        for idx, (algo_name, data) in enumerate(metrics_data.items()):
            ax.bar(x + offset, data['mse'], width, label=algo_name, alpha=0.8, color=colors[idx + 1])
            offset += width
        
        ax.set_xlabel('Image', fontsize=12)
        ax.set_ylabel('MSE', fontsize=12)
        ax.set_title('MSE Comparison Across Algorithms', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(image_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot SSIM comparison
        ax = axes[2]
        offset = -(len(self.algorithms)) * width / 2
        ax.bar(x + offset, noisy_ssim, width, label='Noisy', alpha=0.7, color=colors[0])
        offset += width
        
        for idx, (algo_name, data) in enumerate(metrics_data.items()):
            ax.bar(x + offset, data['ssim'], width, label=algo_name, alpha=0.8, color=colors[idx + 1])
            offset += width
        
        ax.set_xlabel('Image', fontsize=12)
        ax.set_ylabel('SSIM', fontsize=12)
        ax.set_title('SSIM Comparison Across Algorithms', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(image_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
            
        plt.tight_layout()
        
        # Save plot
        algo_names = '_vs_'.join([a.name.replace(' ', '_').lower() for a in self.algorithms])
        plot_path = output_dir / f'comparison_{algo_names}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        if self.verbose:
            print(f"\nComparison plot saved to: {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    # Save a comparison summary CSV with all algorithms' metrics
    def save_comparison_summary(self, output_dir: Union[str, Path]) -> None:

        if not self.all_results:
            print("Warning: No results to save!")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import csv
        
        csv_path = output_dir / 'comparison_summary.csv'
        
        # Get image names from first algorithm
        first_algo = list(self.all_results.keys())[0]
        image_names = [r['name'] for r in self.all_results[first_algo]]
        
        with open(csv_path, 'w', newline='') as f:
            # Build fieldnames
            fieldnames = ['image']
            for algo_name in self.all_results.keys():
                fieldnames.extend([
                    f'{algo_name}_psnr',
                    f'{algo_name}_mse',
                    f'{algo_name}_ssim'
                ])
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write data for each image
            for idx, img_name in enumerate(image_names):
                row = {'image': img_name}
                
                for algo_name, results in self.all_results.items():
                    result = results[idx]
                    row[f'{algo_name}_psnr'] = result['denoised_metrics']['psnr']
                    row[f'{algo_name}_mse'] = result['denoised_metrics']['mse']
                    row[f'{algo_name}_ssim'] = result['denoised_metrics']['ssim']         
                
                writer.writerow(row)
        
        if self.verbose:
            print(f"Comparison summary saved to: {csv_path}")
    
    # Display side-by-side comparison of random images across all algorithms
    def show_image_comparison(self, num_images: int = 3) -> None:

        if not self.all_results:
            print("Warning: No results to display!")
            return
        
        # Get first algorithm's results to determine image count
        first_algo = list(self.all_results.keys())[0]
        available_images = len(self.all_results[first_algo])
        
        # Select random subset of images
        num_to_show = min(num_images, available_images)
        indices = np.random.choice(available_images, size=num_to_show, replace=False)
        
        # Create figure: rows = images, columns = algorithms + 2 (clean + noisy)
        num_cols = len(self.algorithms) + 2
        fig, axes = plt.subplots(num_to_show, num_cols, figsize=(5 * num_cols, 5 * num_to_show))
        
        # Handle single image case
        if num_to_show == 1:
            axes = axes.reshape(1, -1)
        
        for row_idx, img_idx in enumerate(indices):
            # Get data from first algorithm
            first_result = self.all_results[first_algo][img_idx]
            clean = first_result['clean']
            noisy = first_result['noisy']
            noisy_psnr = first_result['noisy_metrics']['psnr']
            
            # Display clean image
            axes[row_idx, 0].imshow(clean, cmap='gray' if clean.ndim == 2 else None)
            axes[row_idx, 0].set_title(f"Clean: {first_result['name']}", fontsize=11, fontweight='bold')
            axes[row_idx, 0].axis('off')
            
            # Display noisy image
            axes[row_idx, 1].imshow(noisy, cmap='gray' if noisy.ndim == 2 else None)
            axes[row_idx, 1].set_title(f"Noisy\n(PSNR: {noisy_psnr:.2f} dB)", fontsize=11)
            axes[row_idx, 1].axis('off')
            
            # Display denoised images for each algorithm
            for col_idx, algo_name in enumerate(self.all_results.keys()):
                result = self.all_results[algo_name][img_idx]
                denoised = result['denoised']
                denoised_psnr = result['denoised_metrics']['psnr']
                improvement = denoised_psnr - noisy_psnr
                
                axes[row_idx, col_idx + 2].imshow(denoised, cmap='gray' if denoised.ndim == 2 else None)
                axes[row_idx, col_idx + 2].set_title(f"{algo_name}\n(PSNR: {denoised_psnr:.2f} dB, Δ{improvement:+.2f} dB)", 
                                                      fontsize=11, color='green')
                axes[row_idx, col_idx + 2].axis('off')
        
        plt.suptitle("Algorithm Comparison - Image Quality", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()
