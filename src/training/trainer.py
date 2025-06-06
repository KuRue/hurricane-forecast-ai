"""Main trainer class for hurricane forecasting models."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
import wandb

from ..models.base import BaseHurricaneModel
from ..models.losses import get_loss_function
from ..utils import ExperimentLogger, ProgressLogger, get_config
from .callbacks import CallbackHandler
from .optimization import create_optimizer, create_scheduler


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Validation
    val_check_interval: float = 0.25  # Fraction of epoch
    val_batch_size: int = 64
    
    # Loss configuration
    loss_type: str = "physics"  # Options: track, intensity, physics, ensemble
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingWarmRestarts"
    warmup_steps: int = 1000
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3
    save_every_n_epochs: int = 5
    
    # Logging
    log_every_n_steps: int = 10
    use_tensorboard: bool = True
    use_wandb: bool = False
    experiment_name: str = "hurricane_forecast"
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val/track_error"
    early_stopping_mode: str = "min"
    
    # Device
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> "TrainingConfig":
        """Create TrainingConfig from configuration file."""
        config = get_config(config_path)
        
        return cls(
            num_epochs=config.training.max_epochs,
            batch_size=config.data.pipeline.batch_size,
            learning_rate=config.training.optimizer.lr,
            weight_decay=config.training.optimizer.weight_decay,
            gradient_clip_val=config.training.gradient_clip_val,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            loss_type=config.training.get("loss_type", "physics"),
            use_amp=config.inference.use_mixed_precision,
            checkpoint_dir=str(Path(config.data.root_dir) / "checkpoints"),
            device=config.project.device,
            num_workers=config.data.pipeline.num_workers,
            pin_memory=config.data.pipeline.pin_memory
        )


class HurricaneTrainer:
    """Trainer for hurricane forecasting models."""
    
    def __init__(
        self,
        model: BaseHurricaneModel,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        callbacks: Optional[List] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            callbacks: List of callbacks
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Move model to device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize loss function
        self.loss_fn = get_loss_function(config.loss_type, **config.loss_kwargs)
        
        # Initialize optimizer and scheduler
        self.optimizer = create_optimizer(
            self.model,
            optimizer_name=config.optimizer,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_name=config.scheduler,
            warmup_steps=config.warmup_steps,
            num_epochs=config.num_epochs,
            steps_per_epoch=len(train_dataloader)
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        
        # Initialize callbacks
        self.callback_handler = CallbackHandler(callbacks or [])
        
        # Logging
        self.experiment_logger = ExperimentLogger(
            experiment_name=config.experiment_name,
            use_mlflow=config.use_tensorboard,
            use_wandb=config.use_wandb
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf') if config.early_stopping_mode == 'min' else float('-inf')
        
        logger.info(
            f"Initialized HurricaneTrainer with {self.model.count_parameters():,} "
            f"parameters on {self.device}"
        )
    
    def train(self) -> Dict[str, float]:
        """Run full training loop.
        
        Returns:
            Dictionary with final metrics
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        # Callback: on_train_begin
        self.callback_handler.on_train_begin(self)
        
        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                
                # Train epoch
                train_metrics = self._train_epoch()
                
                # Validation
                val_metrics = {}
                if self.val_dataloader is not None:
                    if (epoch + 1) % int(1 / self.config.val_check_interval) == 0:
                        val_metrics = self._validate()
                
                # Combine metrics
                metrics = {**train_metrics, **val_metrics}
                
                # Log metrics
                self.experiment_logger.log_metrics(metrics, step=epoch)
                
                # Callback: on_epoch_end
                self.callback_handler.on_epoch_end(self, metrics)
                
                # Check early stopping
                if self._check_early_stopping(metrics):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_every_n_epochs == 0:
                    self._save_checkpoint(metrics)
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Callback: on_train_end
            self.callback_handler.on_train_end(self)
            
            # Finish logging
            self.experiment_logger.finish()
        
        logger.info("Training completed")
        return metrics
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        # Initialize metrics
        epoch_losses = []
        epoch_metrics = {}
        
        # Progress bar
        progress = ProgressLogger(
            len(self.train_dataloader),
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}"
        )
        
        # Callback: on_epoch_begin
        self.callback_handler.on_epoch_begin(self)
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Callback: on_batch_begin
            self.callback_handler.on_batch_begin(self, batch_idx)
            
            # Forward pass
            loss, metrics = self._train_step(batch)
            
            # Backward pass
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps
            
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
            
            # Update metrics
            epoch_losses.append(loss.item())
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value.item() if torch.is_tensor(value) else value)
            
            # Update progress
            progress.update(
                loss=loss.item(),
                lr=self.optimizer.param_groups[0]['lr']
            )
            
            # Log step metrics
            if (batch_idx + 1) % self.config.log_every_n_steps == 0:
                step_metrics = {
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr']
                }
                self.experiment_logger.log_metrics(step_metrics, step=self.global_step)
            
            # Callback: on_batch_end
            self.callback_handler.on_batch_end(self, batch_idx, loss.item())
            
            self.global_step += 1
        
        progress.finish()
        
        # Compute epoch metrics
        epoch_summary = {
            'train/loss': sum(epoch_losses) / len(epoch_losses)
        }
        
        for key, values in epoch_metrics.items():
            epoch_summary[f'train/{key}'] = sum(values) / len(values)
        
        return epoch_summary
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Mixed precision context
        with autocast(enabled=self.config.use_amp):
            # Forward pass
            predictions = self.model(
                batch['inputs'],
                batch.get('reanalysis_maps')
            )
            
            # Compute loss
            targets = {
                'track': batch['target_track'],
                'wind': batch.get('target_wind'),
                'pressure': batch.get('target_pressure')
            }
            
            loss_dict = self.loss_fn(predictions, targets, batch.get('reanalysis_data'))
            
            if isinstance(loss_dict, dict):
                loss = loss_dict['total']
                metrics = {k: v for k, v in loss_dict.items() if k != 'total'}
            else:
                loss = loss_dict
                metrics = {}
        
        return loss, metrics
    
    def _validate(self) -> Dict[str, float]:
        """Run validation.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        val_losses = []
        val_metrics = {}
        
        logger.info("Running validation...")
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                batch = self._batch_to_device(batch)
                
                # Forward pass
                with autocast(enabled=self.config.use_amp):
                    predictions = self.model(
                        batch['inputs'],
                        batch.get('reanalysis_maps')
                    )
                    
                    # Compute loss
                    targets = {
                        'track': batch['target_track'],
                        'wind': batch.get('target_wind'),
                        'pressure': batch.get('target_pressure')
                    }
                    
                    loss_dict = self.loss_fn(predictions, targets)
                    
                    if isinstance(loss_dict, dict):
                        loss = loss_dict['total']
                        metrics = {k: v for k, v in loss_dict.items() if k != 'total'}
                    else:
                        loss = loss_dict
                        metrics = {}
                
                val_losses.append(loss.item())
                
                for key, value in metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = []
                    val_metrics[key].append(value.item() if torch.is_tensor(value) else value)
        
        # Compute validation summary
        val_summary = {
            'val/loss': sum(val_losses) / len(val_losses)
        }
        
        for key, values in val_metrics.items():
            val_summary[f'val/{key}'] = sum(values) / len(values)
        
        logger.info(f"Validation loss: {val_summary['val/loss']:.4f}")
        
        return val_summary
    
    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device.
        
        Args:
            batch: Batch dictionary
            
        Returns:
            Batch on device
        """
        device_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """Check if early stopping should trigger.
        
        Args:
            metrics: Current metrics
            
        Returns:
            True if should stop
        """
        if self.config.early_stopping_patience <= 0:
            return False
        
        metric_key = self.config.early_stopping_metric
        if metric_key not in metrics:
            return False
        
        current_metric = metrics[metric_key]
        
        # Check if improved
        if self.config.early_stopping_mode == 'min':
            improved = current_metric < self.best_metric
        else:
            improved = current_metric > self.best_metric
        
        if improved:
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter = getattr(self, 'patience_counter', 0) + 1
        
        return self.patience_counter >= self.config.early_stopping_patience
    
    def _save_checkpoint(self, metrics: Dict[str, float]) -> None:
        """Save model checkpoint.
        
        Args:
            metrics: Current metrics
        """
        checkpoint_path = Path(self.config.checkpoint_dir) / f"epoch_{self.current_epoch + 1}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'best_metric': self.best_metric
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Log as artifact
        self.experiment_logger.log_artifact(checkpoint_path, "model")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch + 1}")
