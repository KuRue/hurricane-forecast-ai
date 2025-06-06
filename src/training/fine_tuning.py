"""Fine-tuning utilities for pre-trained weather models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from loguru import logger

from ..models.base import BaseHurricaneModel
from ..models.graphcast import GraphCastHurricane
from ..models.pangu import PanguWeatherHurricane
from .optimization import create_optimizer, create_scheduler
from .trainer import HurricaneTrainer, TrainingConfig


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning."""
    
    # Model settings
    base_model: str = "graphcast"  # graphcast, pangu, or custom checkpoint
    checkpoint_path: Optional[str] = None
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Training settings
    learning_rate: float = 1e-5
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Freezing strategy
    freeze_backbone: bool = True
    unfreeze_after_epochs: int = 5
    freeze_bn: bool = True
    
    # Layer-wise learning rates
    use_layerwise_lr: bool = True
    lr_decay_factor: float = 0.8
    
    # Hurricane-specific settings
    auxiliary_loss_weight: float = 0.1
    use_physics_constraints: bool = True


class FineTuner:
    """Base fine-tuning class for weather models."""
    
    def __init__(
        self,
        model: BaseHurricaneModel,
        config: FineTuningConfig
    ):
        """Initialize fine-tuner.
        
        Args:
            model: Model to fine-tune
            config: Fine-tuning configuration
        """
        self.model = model
        self.config = config
        
        # Apply initial freezing
        if config.freeze_backbone:
            self._freeze_backbone()
        
        # Set up layer groups for different learning rates
        if config.use_layerwise_lr:
            self.param_groups = self._get_layerwise_param_groups()
        
        logger.info(
            f"Initialized FineTuner for {model.__class__.__name__} "
            f"with {self._count_trainable_params():,} trainable parameters"
        )
    
    def _freeze_backbone(self) -> None:
        """Freeze backbone model parameters."""
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze hurricane-specific components
        modules_to_unfreeze = [
            'hurricane_encoder',
            'track_head',
            'intensity_head',
            'track_predictor',
            'intensity_predictor',
            'uncertainty_net'
        ]
        
        for name, module in self.model.named_modules():
            if any(target in name for target in modules_to_unfreeze):
                for param in module.parameters():
                    param.requires_grad = True
                logger.debug(f"Unfroze module: {name}")
        
        # Freeze batch norm if requested
        if self.config.freeze_bn:
            self._freeze_bn()
    
    def _freeze_bn(self) -> None:
        """Freeze batch normalization layers."""
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
    
    def _unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        
        logger.info("Unfroze all model parameters")
        
        # Keep BN frozen if requested
        if self.config.freeze_bn:
            self._freeze_bn()
    
    def _get_layerwise_param_groups(self) -> List[Dict]:
        """Get parameter groups with layer-wise learning rates."""
        param_groups = []
        
        # Identify layer depth for each parameter
        layer_params = {}
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Determine layer depth (simplified)
            depth = name.count('.')
            
            if depth not in layer_params:
                layer_params[depth] = []
            
            layer_params[depth].append(param)
        
        # Create parameter groups with decaying learning rates
        base_lr = self.config.learning_rate
        
        for depth in sorted(layer_params.keys()):
            lr = base_lr * (self.config.lr_decay_factor ** depth)
            
            param_groups.append({
                'params': layer_params[depth],
                'lr': lr,
                'name': f'layer_{depth}'
            })
            
            logger.debug(f"Layer {depth}: {len(layer_params[depth])} params, lr={lr:.2e}")
        
        return param_groups
    
    def _count_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer for fine-tuning."""
        if self.config.use_layerwise_lr and hasattr(self, 'param_groups'):
            # Use layer-wise parameter groups
            optimizer = torch.optim.AdamW(
                self.param_groups,
                weight_decay=self.config.weight_decay
            )
        else:
            # Standard optimizer
            optimizer = create_optimizer(
                self.model,
                optimizer_name="AdamW",
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        return optimizer
    
    def on_epoch_end(self, epoch: int) -> None:
        """Called at the end of each epoch.
        
        Args:
            epoch: Current epoch number
        """
        # Unfreeze backbone after specified epochs
        if (self.config.freeze_backbone and 
            epoch == self.config.unfreeze_after_epochs):
            logger.info(f"Unfreezing backbone at epoch {epoch}")
            self._unfreeze_backbone()


class LoRAFineTuner(FineTuner):
    """Fine-tuner using LoRA (Low-Rank Adaptation)."""
    
    def __init__(
        self,
        model: BaseHurricaneModel,
        config: FineTuningConfig
    ):
        """Initialize LoRA fine-tuner.
        
        Args:
            model: Model to fine-tune
            config: Fine-tuning configuration
        """
        # Apply LoRA before parent initialization
        self._apply_lora(model, config)
        
        super().__init__(model, config)
    
    def _apply_lora(self, model: BaseHurricaneModel, config: FineTuningConfig) -> None:
        """Apply LoRA to the model."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # Default target modules if not specified
            if config.lora_target_modules is None:
                # Common attention/linear layer patterns
                config.lora_target_modules = [
                    "q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "down_proj", "up_proj",
                    "linear1", "linear2",
                    "self_attn.out_proj"
                ]
            
            # Create LoRA configuration
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            
            # Apply LoRA
            self.model = get_peft_model(model, lora_config)
            
            # Print LoRA summary
            self.model.print_trainable_parameters()
            
        except ImportError:
            logger.warning("PEFT not installed, using standard fine-tuning")
            config.use_lora = False


def create_fine_tuning_config(
    base_model: str,
    hurricane_data_size: int,
    **kwargs
) -> FineTuningConfig:
    """Create fine-tuning configuration based on data size and model.
    
    Args:
        base_model: Base model name
        hurricane_data_size: Number of hurricane samples
        **kwargs: Additional configuration options
        
    Returns:
        Fine-tuning configuration
    """
    # Adaptive configuration based on data size
    if hurricane_data_size < 100:
        # Very limited data - aggressive regularization
        config = FineTuningConfig(
            base_model=base_model,
            use_lora=True,
            lora_r=8,
            learning_rate=5e-6,
            num_epochs=10,
            freeze_backbone=True,
            unfreeze_after_epochs=1000,  # Never unfreeze
            weight_decay=0.1
        )
    elif hurricane_data_size < 1000:
        # Limited data - moderate settings
        config = FineTuningConfig(
            base_model=base_model,
            use_lora=True,
            lora_r=16,
            learning_rate=1e-5,
            num_epochs=20,
            freeze_backbone=True,
            unfreeze_after_epochs=10,
            weight_decay=0.05
        )
    else:
        # Sufficient data - full fine-tuning possible
        config = FineTuningConfig(
            base_model=base_model,
            use_lora=False,
            learning_rate=5e-5,
            num_epochs=50,
            freeze_backbone=True,
            unfreeze_after_epochs=5,
            weight_decay=0.01
        )
    
    # Update with provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


class ModelAdapter:
    """Adapt pre-trained models for hurricane forecasting."""
    
    @staticmethod
    def adapt_graphcast(
        checkpoint_path: str,
        config: FineTuningConfig
    ) -> GraphCastHurricane:
        """Adapt GraphCast for hurricane forecasting.
        
        Args:
            checkpoint_path: Path to GraphCast checkpoint
            config: Fine-tuning configuration
            
        Returns:
            Adapted GraphCast model
        """
        # Load base model
        model = GraphCastHurricane.from_pretrained(
            checkpoint_path,
            use_lora=config.use_lora
        )
        
        # Add hurricane-specific modifications
        model = ModelAdapter._add_hurricane_heads(model)
        
        return model
    
    @staticmethod
    def adapt_pangu(
        checkpoint_path: str,
        config: FineTuningConfig
    ) -> PanguWeatherHurricane:
        """Adapt Pangu-Weather for hurricane forecasting.
        
        Args:
            checkpoint_path: Path to Pangu checkpoint
            config: Fine-tuning configuration
            
        Returns:
            Adapted Pangu model
        """
        # Load base model
        model = PanguWeatherHurricane(
            config=config,
            onnx_path=checkpoint_path
        )
        
        # Add modifications
        model = ModelAdapter._add_hurricane_heads(model)
        
        return model
    
    @staticmethod
    def _add_hurricane_heads(model: BaseHurricaneModel) -> BaseHurricaneModel:
        """Add hurricane-specific prediction heads.
        
        Args:
            model: Base model
            
        Returns:
            Model with additional heads
        """
        # Check if heads already exist
        if hasattr(model, 'track_head'):
            return model
        
        # Get hidden dimension
        hidden_dim = 256  # Default
        if hasattr(model, 'config'):
            hidden_dim = model.config.hidden_dim
        
        # Add prediction heads if not present
        if not hasattr(model, 'track_head'):
            model.track_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 2)
            )
        
        if not hasattr(model, 'intensity_head'):
            model.intensity_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 2)
            )
        
        if not hasattr(model, 'uncertainty_head'):
            model.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 2)
            )
        
        logger.info("Added hurricane-specific prediction heads")
        
        return model


def fine_tune_model(
    model: BaseHurricaneModel,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    config: Optional[FineTuningConfig] = None,
    training_config: Optional[TrainingConfig] = None
) -> BaseHurricaneModel:
    """Fine-tune a model for hurricane forecasting.
    
    Args:
        model: Model to fine-tune
        train_dataloader: Training data
        val_dataloader: Validation data
        config: Fine-tuning configuration
        training_config: Training configuration
        
    Returns:
        Fine-tuned model
    """
    if config is None:
        config = FineTuningConfig()
    
    if training_config is None:
        training_config = TrainingConfig(
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )
    
    # Create fine-tuner
    if config.use_lora:
        fine_tuner = LoRAFineTuner(model, config)
    else:
        fine_tuner = FineTuner(model, config)
    
    # Create trainer
    trainer = HurricaneTrainer(
        model=fine_tuner.model,
        config=training_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    
    # Add fine-tuning callback
    class FineTuningCallback:
        def on_epoch_end(self, trainer, metrics):
            fine_tuner.on_epoch_end(trainer.current_epoch)
    
    trainer.callback_handler.callbacks.append(FineTuningCallback())
    
    # Train
    trainer.train()
    
    return fine_tuner.model
