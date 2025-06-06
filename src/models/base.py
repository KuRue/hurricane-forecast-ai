"""Base classes for hurricane forecasting models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from ..utils.config import get_config


@dataclass
class ModelConfig:
    """Configuration for hurricane models."""
    
    # Model architecture
    model_name: str = "hurricane_cnn"
    input_features: int = 13
    hidden_dim: int = 256
    num_layers: int = 6
    dropout: float = 0.1
    
    # Input/output settings
    sequence_length: int = 8  # 48 hours (6-hourly data)
    forecast_length: int = 20  # 120 hours
    
    # Training settings
    learning_rate: float = 5e-5
    batch_size: int = 32
    gradient_clip: float = 1.0
    
    # Device settings
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Output settings
    predict_track: bool = True
    predict_intensity: bool = True
    predict_size: bool = False
    
    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> "ModelConfig":
        """Create ModelConfig from configuration file."""
        config = get_config(config_path)
        
        return cls(
            model_name=config.model.name,
            hidden_dim=config.model.hurricane_cnn.d_model,
            num_layers=config.model.hurricane_cnn.n_encoder_layers,
            dropout=config.model.hurricane_cnn.dropout,
            sequence_length=config.inference.time_step * 8,  # Convert to steps
            forecast_length=config.inference.forecast_hours // 6,  # 6-hourly
            learning_rate=config.training.optimizer.lr,
            batch_size=config.data.pipeline.batch_size,
            gradient_clip=config.training.gradient_clip_val,
            device=config.project.device,
            mixed_precision=config.project.mixed_precision,
            predict_track=config.model.hurricane_cnn.predict_track,
            predict_intensity=config.model.hurricane_cnn.predict_intensity,
            predict_size=config.model.hurricane_cnn.predict_size
        )


class BaseHurricaneModel(nn.Module, ABC):
    """Base class for all hurricane forecasting models."""
    
    def __init__(self, config: ModelConfig):
        """Initialize base model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Track training state
        self._is_trained = False
        self._training_history = []
        
        logger.info(f"Initialized {self.__class__.__name__} on {self.device}")
    
    @abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            inputs: Input tensor of shape (batch, seq_len, features)
            masks: Optional attention masks
            
        Returns:
            Dictionary with model outputs
        """
        pass
    
    def predict(
        self,
        inputs: Union[torch.Tensor, np.ndarray],
        return_uncertainty: bool = False
    ) -> Dict[str, np.ndarray]:
        """Make predictions with the model.
        
        Args:
            inputs: Input data
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary with predictions
        """
        self.eval()
        
        # Convert to tensor if needed
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        
        # Add batch dimension if needed
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        
        # Move to device
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.forward(inputs)
        
        # Convert to numpy
        predictions = {}
        for key, value in outputs.items():
            predictions[key] = value.cpu().numpy()
        
        # Add uncertainty if requested
        if return_uncertainty and hasattr(self, "estimate_uncertainty"):
            predictions["uncertainty"] = self.estimate_uncertainty(inputs)
        
        return predictions
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "is_trained": self._is_trained,
            "training_history": self._training_history,
            "model_class": self.__class__.__name__
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved model checkpoint to {path}")
    
    def load(self, path: Union[str, Path], strict: bool = True) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            strict: Whether to strictly enforce state dict matching
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load state dict
        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        
        # Load metadata
        self._is_trained = checkpoint.get("is_trained", True)
        self._training_history = checkpoint.get("training_history", [])
        
        logger.info(f"Loaded model checkpoint from {path}")
    
    def count_parameters(self) -> int:
        """Count trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning."""
        # Override in subclasses to implement specific freezing logic
        pass
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to save memory."""
        self.config.gradient_checkpointing = True
        logger.info("Enabled gradient checkpointing")
    
    def get_optimizer(self, **kwargs) -> torch.optim.Optimizer:
        """Get optimizer for training.
        
        Args:
            **kwargs: Additional optimizer arguments
            
        Returns:
            Optimizer instance
        """
        # Merge with default config
        lr = kwargs.pop("lr", self.config.learning_rate)
        weight_decay = kwargs.pop("weight_decay", 0.01)
        
        # Get parameter groups
        param_groups = self._get_parameter_groups()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        
        return optimizer
    
    def _get_parameter_groups(self) -> List[Dict]:
        """Get parameter groups for optimizer.
        
        Returns:
            List of parameter groups
        """
        # Default: single group with all parameters
        return [{"params": self.parameters()}]
    
    def estimate_memory_usage(self, batch_size: int = 1) -> Dict[str, float]:
        """Estimate memory usage for given batch size.
        
        Args:
            batch_size: Batch size to estimate for
            
        Returns:
            Dictionary with memory estimates in MB
        """
        # Parameter memory
        param_memory = sum(
            p.numel() * p.element_size() for p in self.parameters()
        ) / 1024**2
        
        # Estimate activation memory (rough approximation)
        seq_len = self.config.sequence_length
        features = self.config.input_features
        hidden = self.config.hidden_dim
        
        # Input tensor
        input_memory = (
            batch_size * seq_len * features * 4  # float32
        ) / 1024**2
        
        # Hidden states (approximate)
        hidden_memory = (
            batch_size * seq_len * hidden * 4 * self.config.num_layers
        ) / 1024**2
        
        # Gradient memory (approximate)
        gradient_memory = param_memory * 2  # Params + gradients
        
        return {
            "parameters": param_memory,
            "input": input_memory,
            "activations": hidden_memory,
            "gradients": gradient_memory,
            "total": param_memory + input_memory + hidden_memory + gradient_memory
        }


class HurricaneModelMixin:
    """Mixin class for hurricane-specific functionality."""
    
    @staticmethod
    def encode_cyclical_features(
        timestamps: torch.Tensor,
        max_val: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode cyclical features (e.g., hour of day, day of year).
        
        Args:
            timestamps: Timestamp values
            max_val: Maximum value (24 for hours, 365 for days)
            
        Returns:
            Tuple of (sin_encoding, cos_encoding)
        """
        angle = 2 * np.pi * timestamps / max_val
        return torch.sin(angle), torch.cos(angle)
    
    @staticmethod
    def haversine_distance(
        lat1: torch.Tensor,
        lon1: torch.Tensor,
        lat2: torch.Tensor,
        lon2: torch.Tensor
    ) -> torch.Tensor:
        """Calculate haversine distance between coordinates.
        
        Args:
            lat1, lon1: First coordinates
            lat2, lon2: Second coordinates
            
        Returns:
            Distance in kilometers
        """
        R = 6371.0  # Earth's radius in km
        
        # Convert to radians
        lat1_rad = torch.deg2rad(lat1)
        lat2_rad = torch.deg2rad(lat2)
        dlat = torch.deg2rad(lat2 - lat1)
        dlon = torch.deg2rad(lon2 - lon1)
        
        # Haversine formula
        a = (
            torch.sin(dlat / 2) ** 2 +
            torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
        )
        c = 2 * torch.arcsin(torch.sqrt(a))
        
        return R * c
    
    @staticmethod
    def calculate_track_errors(
        pred_tracks: torch.Tensor,
        true_tracks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Calculate various track error metrics.
        
        Args:
            pred_tracks: Predicted tracks (batch, time, 2)
            true_tracks: True tracks (batch, time, 2)
            
        Returns:
            Dictionary with error metrics
        """
        # Extract lat/lon
        pred_lat, pred_lon = pred_tracks[..., 0], pred_tracks[..., 1]
        true_lat, true_lon = true_tracks[..., 0], true_tracks[..., 1]
        
        # Track errors (great circle distance)
        track_errors = HurricaneModelMixin.haversine_distance(
            pred_lat, pred_lon, true_lat, true_lon
        )
        
        # Along-track and cross-track errors
        # (Simplified - proper calculation requires track direction)
        lat_errors = pred_lat - true_lat
        lon_errors = pred_lon - true_lon
        
        return {
            "track_error": track_errors,
            "latitude_error": lat_errors,
            "longitude_error": lon_errors,
            "mean_track_error": track_errors.mean(),
            "max_track_error": track_errors.max()
        }
