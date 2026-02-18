"""
Add AWGN (Additive White Gaussian Noise) to an image
and visualising the results using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.util import random_noise
import os

def add_awgn(image, mean=0, sigma=0.1):
    """
    Add Additive White Gaussian Noise to an image.
    
    Parameters:
    -----------
    image : ndarray
        Input image (should be in float format [0, 1])
    mean : float
        Mean of the Gaussian noise (default: 0)
    sigma : float
        Standard deviation of the Gaussian noise (default: 0.1)
    
    Returns:
    --------
    noisy_image : ndarray
        Image with added Gaussian noise
    """
    # Generate Gaussian noise
    noise = np.random.normal(mean, sigma, image.shape)
    
    # Add noise to the image
    noisy_image = image + noise
    
    # Clip values to [0, 1] range
    noisy_image = np.clip(noisy_image, 0, 1)
    
    return noisy_image


def visualise_noise_comparison(original, noisy, sigma):
    """
    Visualise original and noisy images side by side.
    
    Parameters:
    -----------
    original : ndarray
        Original image
    noisy : ndarray
        Noisy image
    sigma : float
        Noise standard deviation (for title)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if len(original.shape) == 2:  # Grayscale
        axes[0].imshow(original, cmap='gray', vmin=0, vmax=1)
    else:  # Color
        axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Noisy image
    if len(noisy.shape) == 2:  # Grayscale
        axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=1)
    else:  # Color
        axes[1].imshow(noisy)
    axes[1].set_title(f'Noisy Image (σ={sigma})', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Noise (difference)
    noise_diff = np.abs(noisy - original)
    if len(noise_diff.shape) == 2:  # Grayscale
        axes[2].imshow(noise_diff, cmap='hot', vmin=0, vmax=0.5)
    else:  # Color
        axes[2].imshow(np.mean(noise_diff, axis=2), cmap='hot', vmin=0, vmax=0.5)
    axes[2].set_title('Absolute Noise Difference', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to demonstrate AWGN addition."""

    print("AWGN (Additive White Gaussian Noise) Demonstration")
    
    # Load a standard test image (grayscale)
    print("\nLoading standard test image (camera)...")
    image = img_as_float(data.camera())
    
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Set noise parameters
    noise_mean = 0.0
    noise_sigma = 0.1  # Standard deviation (try 0.05, 0.1, 0.15, 0.2 for different noise levels)
    
    print(f"\nAdding AWGN with:")
    print(f"  Mean (μ): {noise_mean}")
    print(f"  Std Dev (σ): {noise_sigma}")
    
    # Add noise
    noisy_image = add_awgn(image, mean=noise_mean, sigma=noise_sigma)
    
    # Calculate metrics
    mse = np.mean((image - noisy_image) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    
    print(f"\nNoise metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    
    # Visualise
    print("\nVisualising results...")
    visualise_noise_comparison(image, noisy_image, noise_sigma)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
