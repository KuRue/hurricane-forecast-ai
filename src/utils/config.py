"""Configuration management utilities using Hydra and OmegaConf."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from loguru import logger


class ConfigManager:
    """Singleton configuration manager."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def load_config(
        self,
        config_path: Optional[Union[str, Path]] = None,
        overrides: Optional[list] = None
    ) -> DictConfig:
        """Load configuration from file.
        
        Args:
            config_path: Path to config file or config name
            overrides: List of config overrides in Hydra format
            
        Returns:
            Loaded configuration
        """
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # Determine config directory and name
        if config_path is None:
            config_dir = Path(__file__).parent.parent.parent / "configs"
            config_name = "default_config"
        elif isinstance(config_path, (str, Path)) and Path(config_path).exists():
            config_path = Path(config_path)
            config_dir = config_path.parent
            config_name = config_path.stem
        else:
            config_dir = Path(__file__).parent.parent.parent / "configs"
            config_name = str(config_path)
        
        # Initialize Hydra
        with initialize_config_dir(
            version_base=None,
            config_dir=str(config_dir.absolute())
        ):
            cfg = compose(
                config_name=f"{config_name}.yaml",
                overrides=overrides or []
            )
        
        # Store config
        self._config = cfg
        
        # Log configuration
        logger.debug(f"Loaded configuration from {config_dir}/{config_name}.yaml")
        if overrides:
            logger.debug(f"Applied overrides: {overrides}")
            
        return cfg
    
    @property
    def config(self) -> Optional[DictConfig]:
        """Get current configuration."""
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Dot-separated key (e.g., 'model.name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
            
        try:
            return OmegaConf.select(self._config, key)
        except Exception:
            return default
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration values.
        
        Args:
            updates: Dictionary of updates
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
            
        self._config = OmegaConf.merge(self._config, updates)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            OmegaConf.save(self._config, f)
            
        logger.info(f"Saved configuration to {path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
            
        return OmegaConf.to_container(self._config, resolve=True)


# Global configuration manager instance
_config_manager = ConfigManager()


def get_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[list] = None,
    reload: bool = False
) -> DictConfig:
    """Get configuration.
    
    Args:
        config_path: Path to config file or config name
        overrides: List of config overrides
        reload: Force reload configuration
        
    Returns:
        Configuration object
    """
    if _config_manager.config is None or reload:
        _config_manager.load_config(config_path, overrides)
    
    return _config_manager.config


def update_config(updates: Dict[str, Any]) -> None:
    """Update configuration values.
    
    Args:
        updates: Dictionary of updates
    """
    _config_manager.update(updates)


def save_config(path: Union[str, Path]) -> None:
    """Save current configuration.
    
    Args:
        path: Path to save configuration
    """
    _config_manager.save(path)


def config_to_dict() -> Dict[str, Any]:
    """Get configuration as dictionary.
    
    Returns:
        Configuration dictionary
    """
    return _config_manager.to_dict()


class ExperimentConfig:
    """Helper class for experiment configuration."""
    
    def __init__(self, base_config: Optional[Union[str, Path]] = None):
        """Initialize experiment configuration.
        
        Args:
            base_config: Base configuration file
        """
        self.base_config = get_config(base_config)
        self.overrides = []
        
    def override(self, **kwargs) -> 'ExperimentConfig':
        """Add configuration overrides.
        
        Args:
            **kwargs: Key-value pairs to override
            
        Returns:
            Self for chaining
        """
        for key, value in kwargs.items():
            self.overrides.append(f"{key}={value}")
        return self
    
    def override_model(self, **kwargs) -> 'ExperimentConfig':
        """Override model configuration.
        
        Args:
            **kwargs: Model configuration overrides
            
        Returns:
            Self for chaining
        """
        for key, value in kwargs.items():
            self.overrides.append(f"model.{key}={value}")
        return self
    
    def override_training(self, **kwargs) -> 'ExperimentConfig':
        """Override training configuration.
        
        Args:
            **kwargs: Training configuration overrides
            
        Returns:
            Self for chaining
        """
        for key, value in kwargs.items():
            self.overrides.append(f"training.{key}={value}")
        return self
    
    def override_data(self, **kwargs) -> 'ExperimentConfig':
        """Override data configuration.
        
        Args:
            **kwargs: Data configuration overrides
            
        Returns:
            Self for chaining
        """
        for key, value in kwargs.items():
            self.overrides.append(f"data.{key}={value}")
        return self
    
    def load(self) -> DictConfig:
        """Load configuration with overrides.
        
        Returns:
            Final configuration
        """
        return get_config(
            config_path=self.base_config,
            overrides=self.overrides,
            reload=True
        )


# Utility functions for common configuration tasks
def get_data_dir() -> Path:
    """Get data directory from configuration.
    
    Returns:
        Path to data directory
    """
    config = get_config()
    return Path(config.data.root_dir).expanduser()


def get_model_dir() -> Path:
    """Get model directory from configuration.
    
    Returns:
        Path to model directory
    """
    return get_data_dir() / "models"


def get_checkpoint_dir() -> Path:
    """Get checkpoint directory from configuration.
    
    Returns:
        Path to checkpoint directory
    """
    return get_data_dir() / "checkpoints"


def get_device() -> str:
    """Get device from configuration.
    
    Returns:
        Device string (cuda or cpu)
    """
    config = get_config()
    return config.project.device


def get_seed() -> int:
    """Get random seed from configuration.
    
    Returns:
        Random seed
    """
    config = get_config()
    return config.project.seed
