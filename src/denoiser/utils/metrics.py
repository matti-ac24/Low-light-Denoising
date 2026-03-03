# All image quality metrics for the evaluation process

from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np

# Calculate the Mean Squared Error between two images
def calculate_mse(image1: np.ndarray, image2: np.ndarray) -> float:

    return np.mean((image1 - image2) ** 2)

# Calculate the Peak Signal-to-Noise Ratio between two images
def calculate_psnr(image1: np.ndarray, image2: np.ndarray, data_range: float = 1.0) -> float:

    return peak_signal_noise_ratio(image1, image2, data_range=data_range)

#Calculate the Structural Similarity Index between two images
def calculate_ssim(image1: np.ndarray, image2: np.ndarray, data_range: float = 1.0, multichannel: Optional[bool] = None) -> float:
    
    # Auto-detect multichannel
    if multichannel is None:
        multichannel = (image1.ndim == 3)
    
    return structural_similarity(
        image1, image2, 
        data_range=data_range,
        channel_axis=-1 if multichannel else None
    )

# Calculate all available metrics (MSE, PSNR, SSIM)
def calculate_all_metrics(reference: np.ndarray, denoised: np.ndarray, data_range: float = 1.0) -> dict[str, float]:

    metrics = {
        'mse': calculate_mse(reference, denoised),
        'psnr': calculate_psnr(reference, denoised, data_range=data_range),
        'ssim': calculate_ssim(reference, denoised, data_range=data_range)
    }
    
    return metrics
