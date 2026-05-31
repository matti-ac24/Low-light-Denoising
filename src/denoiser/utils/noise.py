# Some different types of noise that can be added to the images

from typing import Optional
from skimage.util import random_noise
import numpy as np

# Add Additive White Gaussian Noise to an image - Usually appears from electronic interference
def add_awgn(image: np.ndarray, mean: float = 0, sigma: float = 0.1, seed: Optional[int] = None) -> np.ndarray:

    """Add Gaussian noise to the image."""
    rng = np.random.default_rng(seed) if seed is not None else None
    return random_noise(image, mode='gaussian', mean=mean, var=sigma**2, rng=rng, clip=True)

# Add Poisson (shot) noise to an image - Usually appears from photon variations in low-light conditions
def add_poisson_noise(image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:

    """Add Poisson noise to the image."""
    rng = np.random.default_rng(seed) if seed is not None else None
    return random_noise(image, mode='poisson', rng=rng, clip=True)

# Add salt and pepper noise to an image - Usually appears due to transmission errors
def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.05, salt_vs_pepper: float = 0.5, seed: Optional[int] = None) -> np.ndarray:

    """Add salt-and-pepper noise to the image."""
    rng = np.random.default_rng(seed) if seed is not None else None
    return random_noise(image, mode='s&p', amount=amount, salt_vs_pepper=salt_vs_pepper, rng=rng, clip=True)
