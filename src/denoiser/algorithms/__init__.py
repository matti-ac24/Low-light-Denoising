# A module for handling different types of denoising algorithms

from typing import Type
from .base import BaseDenoiser
from .bm3d_denoiser import BM3DDenoiser
from .nl_means_denoiser import NLMeansDenoiser
from .nafnet_denoiser import NAFNetDenoiser
from .resunet_denoiser import ResUNetDenoiser
from .restormer_denoiser import RestormerDenoiser

# Registry of available algorithms
ALGORITHMS = {
    'bm3d': BM3DDenoiser,
    'nl-means': NLMeansDenoiser,
    'nlmeans': NLMeansDenoiser,  # Alias without hyphen
    'nafnet': NAFNetDenoiser,
    'resunet': ResUNetDenoiser,
    'res-unet': ResUNetDenoiser,
    'residual-unet': ResUNetDenoiser,
    'restormer': RestormerDenoiser,
}

# Get the denoising algorithm class by name
def get_algorithm(name: str) -> Type[BaseDenoiser]:

    name_lower = name.lower()
    if name_lower not in ALGORITHMS:
        available = ', '.join(ALGORITHMS.keys())
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")
    
    return ALGORITHMS[name_lower]

__all__ = ['BaseDenoiser', 'BM3DDenoiser', 'NLMeansDenoiser', 'NAFNetDenoiser', 'ResUNetDenoiser', 'RestormerDenoiser', 'get_algorithm', 'ALGORITHMS']
