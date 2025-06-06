"""Hurricane forecasting models.

This module contains implementations of various deep learning models
for hurricane track and intensity prediction.
"""

from .base import BaseHurricaneModel, ModelConfig
from .hurricane_cnn import HurricaneCNNTransformer
from .losses import (
    HurricaneTrackLoss,
    PhysicsInformedHurricaneLoss,
    EnsembleLoss
)

# Import foundation models when available
try:
    from .graphcast import GraphCastHurricane
except ImportError:
    GraphCastHurricane = None

try:
    from .pangu import PanguWeatherHurricane
except ImportError:
    PanguWeatherHurricane = None

try:
    from .ensemble import HurricaneEnsembleModel
except ImportError:
    HurricaneEnsembleModel = None

__all__ = [
    # Base classes
    "BaseHurricaneModel",
    "ModelConfig",
    
    # Models
    "HurricaneCNNTransformer",
    "GraphCastHurricane",
    "PanguWeatherHurricane",
    "HurricaneEnsembleModel",
    
    # Loss functions
    "HurricaneTrackLoss",
    "PhysicsInformedHurricaneLoss",
    "EnsembleLoss",
]
