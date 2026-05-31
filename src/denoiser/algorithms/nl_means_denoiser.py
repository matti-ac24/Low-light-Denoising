# Non-Local Means denoising algorithm

from typing import Optional
from .base import BaseDenoiser
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import img_as_float
import numpy as np

# A class which implements the Non-Local Means denoising algorithm
class NLMeansDenoiser(BaseDenoiser):
    
    # Initialise NL-Means denoiser with filtering parameters
    def __init__(self, h: Optional[float] = None, patch_size: int = 5, patch_distance: int = 6, 
                 fast_mode: bool = True, sigma: Optional[float] = None) -> None:
        
        """Initialize the object with the provided settings."""
        super().__init__(h=h, patch_size=patch_size, patch_distance=patch_distance,
                        fast_mode=fast_mode, sigma=sigma)
        self.h = h
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.fast_mode = fast_mode
        self.sigma = sigma
    
    # Denoise an image using the Non-Local Means algorithm
    def denoise(self, noisy_image: np.ndarray) -> np.ndarray:

        # Ensure float format
        """Denoise the provided image and return the result."""
        noisy_image = img_as_float(noisy_image)
        
        # Estimate sigma if not provided
        if self.sigma is None:
            sigma_estimated = estimate_sigma(noisy_image, channel_axis=-1 if noisy_image.ndim == 3 else None)
        else:
            sigma_estimated = self.sigma
        
        # Calculate h if not provided (rule of thumb: h ~= sigma)
        h = self.h if self.h is not None else sigma_estimated
        
        # Apply NL-Means denoising
        denoised = denoise_nl_means(
            noisy_image,
            h=h,
            patch_size=self.patch_size,
            patch_distance=self.patch_distance,
            fast_mode=self.fast_mode,
            channel_axis=-1 if noisy_image.ndim == 3 else None
        )
        
        # Ensure the output is in valid range
        denoised = np.clip(denoised, 0, 1)
        
        return denoised
