# BM3D (Block-Matching and 3D filtering) denoising algorithm

from .base import BaseDenoiser
import numpy as np
from bm3d import bm3d, BM3DStages

# A class which implements the BM3D denoising algorithm
class BM3DDenoiser(BaseDenoiser):
    
    # Initialise the BM3D denoiser with noise level and processing stage
    def __init__(self, sigma_psd: float = 0.1, stage_arg: BM3DStages = BM3DStages.ALL_STAGES) -> None:

        super().__init__(sigma_psd=sigma_psd, stage_arg=stage_arg)
        self.sigma_psd = sigma_psd
        self.stage_arg = stage_arg
    
    # Denoise an image using the BM3D algorithm
    def denoise(self, noisy_image: np.ndarray) -> np.ndarray:

        # BM3D expects images in float format [0, 1]
        if noisy_image.dtype != np.float32 and noisy_image.dtype != np.float64:
            noisy_image = noisy_image.astype(np.float64)
        
        # Apply BM3D (stage_arg should be a BM3DStages enum, not a string)
        denoised = bm3d(noisy_image, sigma_psd=self.sigma_psd, stage_arg=self.stage_arg)
        
        # Ensure the output is in valid range
        denoised = np.clip(denoised, 0, 1)
        
        return denoised
