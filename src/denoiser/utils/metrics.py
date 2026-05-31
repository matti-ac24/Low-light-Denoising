# All image quality metrics for the evaluation process

from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np

# Calculate the Peak Signal-to-Noise Ratio between two images
def calculate_psnr(image1: np.ndarray, image2: np.ndarray, data_range: float = 1.0) -> float:

    """Compute the PSNR between two images."""
    return peak_signal_noise_ratio(image1, image2, data_range=data_range)

#Calculate the Structural Similarity Index between two images
def calculate_ssim(image1: np.ndarray, image2: np.ndarray, data_range: float = 1.0, multichannel: Optional[bool] = None) -> float:
    
    # Auto-detect multichannel
    """Compute the SSIM between two images."""
    if multichannel is None:
        multichannel = (image1.ndim == 3)
    
    return structural_similarity(
        image1, image2, 
        data_range=data_range,
        channel_axis=-1 if multichannel else None
    )

# Calculate all available metrics (PSNR, SSIM)
def calculate_all_metrics(reference: np.ndarray, denoised: np.ndarray, data_range: float = 1.0) -> dict[str, float]:

    """Compute all supported quality metrics for a pair of images."""
    metrics = {
        'psnr': calculate_psnr(reference, denoised, data_range=data_range),
        'ssim': calculate_ssim(reference, denoised, data_range=data_range)
    }
    
    return metrics
