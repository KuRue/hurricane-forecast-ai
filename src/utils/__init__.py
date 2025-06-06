"""Utility modules for hurricane forecasting system."""

from .config import (
    get_config,
    update_config,
    save_config,
    config_to_dict,
    ExperimentConfig,
    get_data_dir,
    get_model_dir,
    get_checkpoint_dir,
    get_device,
    get_seed
)

from .logging import (
    setup_logging,
    get_logger,
    ExperimentLogger,
    ProgressLogger,
    log_separator,
    log_dict,
    log_gpu_info,
    console
)

# Import other utilities when implemented
try:
    from .metrics import (
        calculate_track_error,
        calculate_intensity_error,
        calculate_skill_score
    )
except ImportError:
    calculate_track_error = None
    calculate_intensity_error = None
    calculate_skill_score = None

try:
    from .visualization import (
        plot_hurricane_track,
        plot_intensity_forecast,
        create_forecast_animation
    )
except ImportError:
    plot_hurricane_track = None
    plot_intensity_forecast = None
    create_forecast_animation = None

__all__ = [
    # Config utilities
    "get_config",
    "update_config", 
    "save_config",
    "config_to_dict",
    "ExperimentConfig",
    "get_data_dir",
    "get_model_dir",
    "get_checkpoint_dir",
    "get_device",
    "get_seed",
    
    # Logging utilities
    "setup_logging",
    "get_logger",
    "ExperimentLogger",
    "ProgressLogger",
    "log_separator",
    "log_dict",
    "log_gpu_info",
    "console",
    
    # Metrics
    "calculate_track_error",
    "calculate_intensity_error",
    "calculate_skill_score",
    
    # Visualization
    "plot_hurricane_track",
    "plot_intensity_forecast",
    "create_forecast_animation",
]
