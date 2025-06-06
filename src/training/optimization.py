"""Memory optimization utilities for efficient training on single GPUs."""

import gc
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
from loguru import logger


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "AdamW",
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    **kwargs
) -> torch.optim.Optimizer:
    """Create optimizer with appropriate settings.
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    # Get parameter groups
    param_groups = get_parameter_groups(model, weight_decay)
    
    # Create optimizer
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=kwargs.get('nesterov', True)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    logger.info(f"Created {optimizer_name} optimizer with lr={lr}")
    return optimizer


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01
) -> List[Dict[str, Any]]:
    """Get parameter groups with different settings.
    
    Args:
        model: Model
        weight_decay: Default weight decay
        
    Returns:
        List of parameter groups
    """
    # Separate parameters that should not have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Don't apply weight decay to bias and normalization parameters
        if 'bias' in name or 'norm' in name or 'bn' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # Add LoRA parameters if present
    lora_params = [p for n, p in model.named_parameters() if 'lora' in n and p.requires_grad]
    if lora_params:
        param_groups.append({
            'params': lora_params,
            'weight_decay': 0.0,
            'lr': 2e-4  # Higher LR for LoRA
        })
    
    return param_groups


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "CosineAnnealingWarmRestarts",
    warmup_steps: int = 1000,
    num_epochs: int = 100,
    steps_per_epoch: int = 100,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_name: Name of scheduler
        warmup_steps: Number of warmup steps
        num_epochs: Total number of epochs
        steps_per_epoch: Steps per epoch
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance or None
    """
    total_steps = num_epochs * steps_per_epoch
    
    if scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 10),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('eta_min', 1e-7)
        )
    elif scheduler_name == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'] * 10,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos'
        )
    elif scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            min_lr=kwargs.get('min_lr', 1e-7)
        )
    else:
        scheduler = None
        logger.warning(f"Unknown scheduler: {scheduler_name}, using constant LR")
    
    # Add warmup if specified
    if warmup_steps > 0 and scheduler is not None:
        scheduler = WarmupScheduler(scheduler, warmup_steps)
    
    return scheduler


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Warmup wrapper for learning rate schedulers."""
    
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        warmup_steps: int
    ):
        """Initialize warmup scheduler.
        
        Args:
            scheduler: Base scheduler
            warmup_steps: Number of warmup steps
        """
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        # Initialize parent
        super().__init__(scheduler.optimizer, last_epoch=-1)
    
    def get_lr(self) -> List[float]:
        """Get learning rates."""
        if self.step_count < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.step_count / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Use base scheduler
            return self.scheduler.get_lr()
    
    def step(self, epoch: Optional[int] = None) -> None:
        """Step scheduler."""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            super().step(epoch)
        else:
            self.scheduler.step(epoch)


class GradientAccumulator:
    """Gradient accumulation for larger effective batch sizes."""
    
    def __init__(
        self,
        model: nn.Module,
        accumulation_steps: int = 1,
        max_norm: float = 1.0
    ):
        """Initialize gradient accumulator.
        
        Args:
            model: Model
            accumulation_steps: Number of accumulation steps
            max_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.max_norm = max_norm
        self.step_count = 0
    
    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass with accumulation.
        
        Args:
            loss: Loss to backpropagate
        """
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.step_count += 1
    
    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Optimizer step if accumulated enough.
        
        Args:
            optimizer: Optimizer
            
        Returns:
            True if optimizer step was taken
        """
        if self.step_count % self.accumulation_steps == 0:
            # Clip gradients
            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_norm
                )
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            return True
        
        return False


class MemoryEfficientTrainer:
    """Memory-efficient training utilities."""
    
    def __init__(
        self,
        model: nn.Module,
        use_gradient_checkpointing: bool = True,
        use_cpu_offload: bool = False,
        use_mixed_precision: bool = True
    ):
        """Initialize memory-efficient trainer.
        
        Args:
            model: Model to train
            use_gradient_checkpointing: Whether to use gradient checkpointing
            use_cpu_offload: Whether to offload to CPU
            use_mixed_precision: Whether to use mixed precision
        """
        self.model = model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_cpu_offload = use_cpu_offload
        self.use_mixed_precision = use_mixed_precision
        
        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            self.enable_gradient_checkpointing()
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if use_mixed_precision else None
        
        # CPU offload state
        self.cpu_state = {} if use_cpu_offload else None
    
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing in the model."""
        if hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()
        else:
            logger.warning("Model does not support gradient checkpointing")
    
    @contextmanager
    def forward_context(self):
        """Context for memory-efficient forward pass."""
        # Mixed precision context
        if self.use_mixed_precision:
            with autocast():
                yield
        else:
            yield
    
    def backward(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        gradient_accumulation_steps: int = 1,
        step_idx: int = 0
    ) -> None:
        """Memory-efficient backward pass.
        
        Args:
            loss: Loss to backpropagate
            optimizer: Optimizer
            gradient_accumulation_steps: Number of accumulation steps
            step_idx: Current step index
        """
        # Scale loss
        scaled_loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Offload gradients to CPU if enabled
        if self.use_cpu_offload and (step_idx + 1) % gradient_accumulation_steps != 0:
            self._offload_gradients()
    
    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        max_norm: float = 1.0
    ) -> None:
        """Memory-efficient optimizer step.
        
        Args:
            optimizer: Optimizer
            max_norm: Maximum gradient norm
        """
        # Restore gradients from CPU if needed
        if self.use_cpu_offload:
            self._restore_gradients()
        
        # Gradient clipping
        if self.scaler:
            self.scaler.unscale_(optimizer)
        
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm
            )
        
        # Optimizer step
        if self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
        
        # Clear CPU state
        if self.use_cpu_offload:
            self.cpu_state.clear()
    
    def _offload_gradients(self) -> None:
        """Offload gradients to CPU memory."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.cpu_state[name] = param.grad.cpu()
                param.grad = None
    
    def _restore_gradients(self) -> None:
        """Restore gradients from CPU memory."""
        for name, param in self.model.named_parameters():
            if name in self.cpu_state:
                if param.grad is None:
                    param.grad = self.cpu_state[name].cuda()
                else:
                    param.grad += self.cpu_state[name].cuda()
    
    @staticmethod
    def optimize_model_for_inference(model: nn.Module) -> nn.Module:
        """Optimize model for inference.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        model.eval()
        
        # Fuse batch norm if possible
        if hasattr(torch.nn.utils, 'fusion'):
            model = torch.nn.utils.fusion.fuse_conv_bn_eval(model)
        
        # Enable inference mode optimizations
        for module in model.modules():
            if hasattr(module, 'inference_mode'):
                module.inference_mode = True
        
        return model
    
    @staticmethod
    def estimate_memory_usage(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_size: int = 1
    ) -> Dict[str, float]:
        """Estimate memory usage for model.
        
        Args:
            model: Model
            input_shape: Input shape (without batch dimension)
            batch_size: Batch size
            
        Returns:
            Memory usage estimates in MB
        """
        # Model parameters
        param_memory = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / 1024**2
        
        # Estimate activations (rough approximation)
        dummy_input = torch.randn(batch_size, *input_shape, device='cuda')
        
        # Hook to track activation sizes
        activation_memory = 0
        handles = []
        
        def hook_fn(module, input, output):
            nonlocal activation_memory
            if torch.is_tensor(output):
                activation_memory += output.numel() * output.element_size() / 1024**2
        
        # Register hooks
        for module in model.modules():
            handles.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Gradient memory (approximate)
        gradient_memory = param_memory * 2  # Gradients + optimizer state
        
        return {
            'parameters': param_memory,
            'activations': activation_memory,
            'gradients': gradient_memory,
            'total': param_memory + activation_memory + gradient_memory
        }


def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    torch.cuda.empty_cache()
    
    # Log memory status
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.debug(
                f"GPU {i} after clearing: "
                f"{allocated:.1f} GB allocated, "
                f"{reserved:.1f} GB reserved"
            )
