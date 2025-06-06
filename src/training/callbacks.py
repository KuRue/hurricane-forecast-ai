"""Callback system for training monitoring and control."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import psutil
import GPUtil


class Callback(ABC):
    """Base callback class."""
    
    @abstractmethod
    def on_train_begin(self, trainer) -> None:
        """Called at the beginning of training."""
        pass
    
    @abstractmethod
    def on_train_end(self, trainer) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer) -> None:
        """Called at the beginning of an epoch."""
        pass
    
    def on_epoch_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Called at the end of an epoch."""
        pass
    
    def on_batch_begin(self, trainer, batch_idx: int) -> None:
        """Called at the beginning of a batch."""
        pass
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float) -> None:
        """Called at the end of a batch."""
        pass


class CallbackHandler:
    """Manages multiple callbacks."""
    
    def __init__(self, callbacks: List[Callback]):
        """Initialize callback handler.
        
        Args:
            callbacks: List of callbacks
        """
        self.callbacks = callbacks
    
    def on_train_begin(self, trainer) -> None:
        """Called at the beginning of training."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer) -> None:
        """Called at the end of training."""
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, trainer) -> None:
        """Called at the beginning of an epoch."""
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer)
    
    def on_epoch_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Called at the end of an epoch."""
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, metrics)
    
    def on_batch_begin(self, trainer, batch_idx: int) -> None:
        """Called at the beginning of a batch."""
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch_idx)
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float) -> None:
        """Called at the end of a batch."""
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, loss)


class EarlyStopping(Callback):
    """Early stopping callback."""
    
    def __init__(
        self,
        monitor: str = "val/loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0001
    ):
        """Initialize early stopping.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait
            mode: "min" or "max"
            min_delta: Minimum change to qualify as improvement
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_score = None
        self.counter = 0
        self.stopped_epoch = 0
        
    def on_train_begin(self, trainer) -> None:
        """Reset state at training start."""
        self.best_score = None
        self.counter = 0
        self.stopped_epoch = 0
    
    def on_train_end(self, trainer) -> None:
        """Log if early stopping triggered."""
        if self.stopped_epoch > 0:
            logger.info(f"Early stopping triggered at epoch {self.stopped_epoch}")
    
    def on_epoch_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Check if should stop."""
        if self.monitor not in metrics:
            return
        
        score = metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = score
        else:
            if self.mode == "min":
                improved = score < self.best_score - self.min_delta
            else:
                improved = score > self.best_score + self.min_delta
            
            if improved:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                
                if self.counter >= self.patience:
                    self.stopped_epoch = trainer.current_epoch + 1
                    trainer.should_stop = True
                    logger.info(
                        f"Early stopping: {self.monitor} has not improved "
                        f"for {self.patience} epochs"
                    )


class ModelCheckpoint(Callback):
    """Save model checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = "val/loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
        filename_format: str = "epoch={epoch}-{monitor:.4f}"
    ):
        """Initialize model checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: "min" or "max"
            save_top_k: Number of best models to keep
            save_last: Whether to save last checkpoint
            filename_format: Filename format string
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.filename_format = filename_format
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_k_models = []
        
    def on_train_begin(self, trainer) -> None:
        """Initialize checkpoint tracking."""
        self.best_k_models = []
    
    def on_epoch_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Save checkpoint if needed."""
        epoch = trainer.current_epoch
        
        # Save last checkpoint
        if self.save_last:
            last_path = self.checkpoint_dir / "last.pt"
            self._save_checkpoint(trainer, last_path, metrics)
        
        # Check if should save based on metric
        if self.monitor in metrics:
            score = metrics[self.monitor]
            
            # Format filename
            monitor_str = f"{self.monitor.replace('/', '_')}={score:.4f}"
            filename = self.filename_format.format(
                epoch=epoch,
                monitor=score,
                **metrics
            )
            checkpoint_path = self.checkpoint_dir / f"{filename}.pt"
            
            # Check if should save
            should_save = False
            if len(self.best_k_models) < self.save_top_k:
                should_save = True
            else:
                worst_score, worst_path = self.best_k_models[-1]
                if self.mode == "min":
                    should_save = score < worst_score
                else:
                    should_save = score > worst_score
            
            if should_save:
                # Save checkpoint
                self._save_checkpoint(trainer, checkpoint_path, metrics)
                
                # Update best k models
                self.best_k_models.append((score, checkpoint_path))
                self.best_k_models.sort(
                    key=lambda x: x[0],
                    reverse=(self.mode == "max")
                )
                
                # Remove worst model if exceeds k
                if len(self.best_k_models) > self.save_top_k:
                    _, path_to_remove = self.best_k_models.pop()
                    if path_to_remove.exists():
                        path_to_remove.unlink()
                        logger.info(f"Removed checkpoint: {path_to_remove}")
    
    def _save_checkpoint(self, trainer, path: Path, metrics: Dict[str, float]) -> None:
        """Save checkpoint to file."""
        checkpoint = {
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'metrics': metrics,
            'config': trainer.config,
            'best_metric': getattr(trainer, 'best_metric', None)
        }
        
        if trainer.scheduler:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
        
        if hasattr(trainer, 'scaler') and trainer.scaler:
            checkpoint['scaler_state_dict'] = trainer.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")


class LearningRateMonitor(Callback):
    """Monitor and log learning rate."""
    
    def __init__(self, log_every_n_steps: int = 100):
        """Initialize LR monitor.
        
        Args:
            log_every_n_steps: Logging frequency
        """
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float) -> None:
        """Log learning rate."""
        self.step_count += 1
        
        if self.step_count % self.log_every_n_steps == 0:
            # Get learning rates for all param groups
            lrs = [group['lr'] for group in trainer.optimizer.param_groups]
            
            if len(lrs) == 1:
                logger.info(f"Learning rate: {lrs[0]:.2e}")
            else:
                for i, lr in enumerate(lrs):
                    logger.info(f"Learning rate (group {i}): {lr:.2e}")


class MemoryMonitor(Callback):
    """Monitor GPU and system memory usage."""
    
    def __init__(self, log_every_n_steps: int = 100):
        """Initialize memory monitor.
        
        Args:
            log_every_n_steps: Logging frequency
        """
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float) -> None:
        """Log memory usage."""
        self.step_count += 1
        
        if self.step_count % self.log_every_n_steps == 0:
            # GPU memory
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    
                    gpu = GPUtil.getGPUs()[i]
                    gpu_util = gpu.memoryUtil * 100
                    
                    logger.info(
                        f"GPU {i} memory: {allocated:.1f}/{reserved:.1f} GB allocated/reserved "
                        f"({gpu_util:.1f}% utilization)"
                    )
            
            # System memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory()
            ram_used = ram.used / 1024**3
            ram_total = ram.total / 1024**3
            ram_percent = ram.percent
            
            logger.info(
                f"System: CPU {cpu_percent:.1f}%, "
                f"RAM {ram_used:.1f}/{ram_total:.1f} GB ({ram_percent:.1f}%)"
            )


class TensorBoardLogger(Callback):
    """Log metrics and visualizations to TensorBoard."""
    
    def __init__(
        self,
        log_dir: str,
        comment: str = "",
        log_every_n_steps: int = 10
    ):
        """Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            comment: Comment to add to run name
            log_every_n_steps: Logging frequency
        """
        self.log_dir = Path(log_dir)
        self.comment = comment
        self.log_every_n_steps = log_every_n_steps
        self.writer = None
        self.step_count = 0
    
    def on_train_begin(self, trainer) -> None:
        """Initialize TensorBoard writer."""
        run_name = f"{trainer.config.experiment_name}_{self.comment}"
        self.writer = SummaryWriter(
            log_dir=self.log_dir / run_name,
            comment=self.comment
        )
        
        # Log model graph if possible
        try:
            dummy_input = torch.randn(
                1,
                trainer.config.sequence_length,
                trainer.config.input_features
            ).to(trainer.device)
            self.writer.add_graph(trainer.model, dummy_input)
        except Exception as e:
            logger.warning(f"Could not log model graph: {e}")
    
    def on_train_end(self, trainer) -> None:
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float) -> None:
        """Log batch metrics."""
        self.step_count += 1
        
        if self.step_count % self.log_every_n_steps == 0:
            # Log loss
            self.writer.add_scalar('train/batch_loss', loss, self.step_count)
            
            # Log learning rate
            lr = trainer.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/learning_rate', lr, self.step_count)
    
    def on_epoch_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Log epoch metrics."""
        epoch = trainer.current_epoch
        
        # Log all metrics
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, epoch)
        
        # Log parameter histograms
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'params/{name}', param, epoch)
                self.writer.add_histogram(f'grads/{name}', param.grad, epoch)
        
        # Log weight statistics
        total_norm = 0
        for param in trainer.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.writer.add_scalar('train/gradient_norm', total_norm, epoch)


class GradientClippingMonitor(Callback):
    """Monitor gradient clipping."""
    
    def __init__(self, log_every_n_steps: int = 100):
        """Initialize gradient clipping monitor.
        
        Args:
            log_every_n_steps: Logging frequency
        """
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
        self.clip_counts = []
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float) -> None:
        """Check if gradients were clipped."""
        self.step_count += 1
        
        # Calculate gradient norm
        total_norm = 0
        for param in trainer.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Check if clipped
        was_clipped = total_norm > trainer.config.gradient_clip_val
        self.clip_counts.append(was_clipped)
        
        if self.step_count % self.log_every_n_steps == 0:
            # Calculate clip rate
            recent_clips = self.clip_counts[-self.log_every_n_steps:]
            clip_rate = sum(recent_clips) / len(recent_clips)
            
            logger.info(
                f"Gradient norm: {total_norm:.2f}, "
                f"Clip rate: {clip_rate:.1%}"
            )
