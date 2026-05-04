import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_ROOT))

from denoiser.algorithms.nafnet_denoiser import NAFNetDenoiser
from skimage import io
import numpy as np

def denoise_with_tiling(denoiser, image, tile_size=480, overlap=32):
    """Denoise large image by tiling with overlap blending."""
    h, w = image.shape[:2]
    result = np.zeros_like(image)
    weights = np.zeros(h if image.ndim == 2 else (h, w), dtype=np.float32)
    
    stride = tile_size - 2 * overlap
    
    for y in range(0, h - overlap, stride):
        for x in range(0, w - overlap, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            
            tile = image[y_start:y_end, x_start:x_end]
            denoised_tile = denoiser.denoise(tile)
            
            result[y_start:y_end, x_start:x_end] += denoised_tile
            if image.ndim == 2:
                weights[y_start:y_end, x_start:x_end] += 1
            else:
                weights[y_start:y_end, x_start:x_end] += 1
    
    # Normalize by overlap weights
    if image.ndim == 3:
        result = result / weights[:, :, np.newaxis]
    else:
        result = result / weights
    
    return np.clip(result, 0, 1)

# Load image
noisy_img = io.imread('./data/received_966790769428179.jpeg')
noisy_img = noisy_img.astype(np.float32) / 255.0

# Denoise with tiling
denoiser = NAFNetDenoiser()
denoised_img = denoise_with_tiling(denoiser, noisy_img, tile_size=480, overlap=32)

# Save result
io.imsave('output_denoised.png', (denoised_img * 255).astype(np.uint8))