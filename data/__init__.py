"""
Data package for S&P 500 Prediction Project

This package contains:
- Data pipeline for downloading and processing financial data
- Technical indicator calculations
- Data preprocessing utilities
- Feature engineering functions
"""

from .data_pipeline import SP500DataPipeline

__version__ = "0.1.0"
__author__ = "Ismail Moudden"
__email__ = "ismail.moudden1@gmail.com"

__all__ = [
    "SP500DataPipeline"
]
