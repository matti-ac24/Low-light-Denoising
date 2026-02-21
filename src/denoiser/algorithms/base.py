# Base class for all denoising algorithms

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

# An Abstract Base Class for denoising algorithms
class BaseDenoiser(ABC):

    # Initialise the denoiser with optional parameters
    def __init__(self, **kwargs: Any) -> None:
        self.params = kwargs
    
    # Denoise any image and return the denoised result
    @abstractmethod
    def denoise(self, noisy_image: np.ndarray) -> np.ndarray:
        pass
    
    # Return a string representation of the denoiser instance
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"
    
    # Return the algorithm name (without 'Denoiser' at the end)
    @property
    def name(self) -> str:
        return self.__class__.__name__.replace('Denoiser', '')
