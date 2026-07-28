"""Reusable tooling around EfficientAD training outputs.

This package intentionally lives outside ``EfficientAD-main`` so the upstream
implementation can stay untouched.
"""

from .artifacts import ModelArtifacts
from .predictor import EfficientADPredictor, FeatureAnalysis, Prediction

__all__ = [
    "EfficientADPredictor",
    "FeatureAnalysis",
    "ModelArtifacts",
    "Prediction",
]
