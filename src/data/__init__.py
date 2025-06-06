"""Data loading and processing modules for hurricane forecasting."""

from .loaders import (
    HURDAT2Loader,
    IBTrACSLoader,
    ERA5Loader,
    HurricaneDataPipeline
)

# Import processors and validators when they're implemented
try:
    from .processors import (
        HurricanePreprocessor,
        ERA5Preprocessor,
        normalize_track_data,
        create_track_features
    )
except ImportError:
    HurricanePreprocessor = None
    ERA5Preprocessor = None
    normalize_track_data = None
    create_track_features = None

try:
    from .validators import (
        HurricaneDataValidator,
        validate_track_continuity,
        validate_intensity_physics
    )
except ImportError:
    HurricaneDataValidator = None
    validate_track_continuity = None
    validate_intensity_physics = None

__all__ = [
    # Loaders
    "HURDAT2Loader",
    "IBTrACSLoader",
    "ERA5Loader",
    "HurricaneDataPipeline",
    
    # Processors
    "HurricanePreprocessor",
    "ERA5Preprocessor",
    "normalize_track_data",
    "create_track_features",
    
    # Validators
    "HurricaneDataValidator",
    "validate_track_continuity",
    "validate_intensity_physics",
]
