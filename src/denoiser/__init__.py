"""
Denoiser Package - A modular framework for image denoising algorithms.

This package provides:
- Multiple denoising algorithms (BM3D, NL-Means, CNN, etc.)
- Support for different dataset types (real-world, synthetic)
- Evaluation metrics and utilities
- Flexible CLI interface
"""

from .algorithms import get_algorithm
from .evaluator import Evaluator, ComparisonEvaluator

# Decides which functions are imported when the wildcard '*' is used
__all__ = ['get_algorithm', 'Evaluator', 'ComparisonEvaluator']
