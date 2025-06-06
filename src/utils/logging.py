"""Logging utilities for the hurricane forecast system."""

import sys
import logging
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

from .config import get_config, get_data_dir


# Global console for rich output
console = Console()


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages and redirect to loguru."""
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a logging record."""
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    serialize: bool = False,
    colorize: bool = True,
    include_timestamp: bool = True
) -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        serialize: Whether to serialize logs to JSON
        colorize: Whether to colorize console output
        include_timestamp: Whether to include timestamp in logs
    """
    # Get config
    config = get_config()
    
    # Use config level if not specified
    if level is None:
        level = config.logging.level
    
    # Remove existing handlers
    logger.remove()
    
    # Console handler format
    if include_timestamp:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    else:
        console_format = (
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=colorize,
        serialize=serialize
    )
    
    # Add file handler if specified
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File format (always include timestamp)
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )
        
        logger.add(
            log_path,
            format=file_format,
            level=level,
            rotation="100 MB",
            retention="1 month",
            compression="zip",
            serialize=serialize
        )
        
        logger.info(f"Logging to file: {log_path}")
    
    # Set up default log directory
    log_dir = get_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Add rotating file handler for all logs
    logger.add(
        log_dir / "hurricane_forecast_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="DEBUG",  # Capture all logs in file
        rotation="1 day",
        retention="30 days",
        compression="zip",
        serialize=serialize
    )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Set levels for third-party loggers
    for logger_name in ["urllib3", "matplotlib", "PIL", "h5py"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logger.info(f"Logging configured at {level} level")


def get_logger(name: Optional[str] = None) -> logger:
    """Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


class ExperimentLogger:
    """Logger for ML experiments with MLflow/W&B integration."""
    
    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        use_mlflow: bool = True,
        use_wandb: bool = False
    ):
        """Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Name of the specific run
            use_mlflow: Whether to use MLflow
            use_wandb: Whether to use Weights & Biases
        """
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        
        # Set up experiment tracking
        if self.use_mlflow:
            self._setup_mlflow()
        
        if self.use_wandb:
            self._setup_wandb()
            
    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        try:
            import mlflow
            
            config = get_config()
            mlflow.set_tracking_uri(config.mlflow.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=self.run_name)
            
            # Log configuration
            mlflow.log_params({"config": config})
            
            logger.info(f"MLflow tracking started: {self.experiment_name}/{self.run_name}")
            
        except ImportError:
            logger.warning("MLflow not installed, skipping MLflow tracking")
            self.use_mlflow = False
            
    def _setup_wandb(self):
        """Set up Weights & Biases tracking."""
        try:
            import wandb
            
            config = get_config()
            wandb.init(
                project=config.wandb.project,
                name=self.run_name,
                config=config,
                mode=config.wandb.mode
            )
            
            logger.info(f"W&B tracking started: {config.wandb.project}/{self.run_name}")
            
        except ImportError:
            logger.warning("wandb not installed, skipping W&B tracking")
            self.use_wandb = False
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to experiment tracking.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        # Log to console
        logger.info(f"Metrics at step {step}: {metrics}")
        
        # Log to MLflow
        if self.use_mlflow:
            import mlflow
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        
        # Log to W&B
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)
    
    def log_params(self, params: dict):
        """Log parameters to experiment tracking.
        
        Args:
            params: Dictionary of parameters
        """
        logger.info(f"Parameters: {params}")
        
        if self.use_mlflow:
            import mlflow
            mlflow.log_params(params)
            
        if self.use_wandb:
            import wandb
            wandb.config.update(params)
    
    def log_artifact(self, path: Union[str, Path], artifact_type: str = "model"):
        """Log artifact to experiment tracking.
        
        Args:
            path: Path to artifact
            artifact_type: Type of artifact
        """
        path = Path(path)
        logger.info(f"Logging {artifact_type} artifact: {path}")
        
        if self.use_mlflow:
            import mlflow
            if artifact_type == "model" and path.suffix in [".pt", ".pth"]:
                mlflow.pytorch.log_model(pytorch_model=path, artifact_path="model")
            else:
                mlflow.log_artifact(str(path))
                
        if self.use_wandb:
            import wandb
            wandb.save(str(path))
    
    def finish(self):
        """Finish experiment tracking."""
        if self.use_mlflow:
            import mlflow
            mlflow.end_run()
            
        if self.use_wandb:
            import wandb
            wandb.finish()
            
        logger.info("Experiment tracking finished")


class ProgressLogger:
    """Logger for tracking training progress."""
    
    def __init__(self, total_steps: int, desc: str = "Training"):
        """Initialize progress logger.
        
        Args:
            total_steps: Total number of steps
            desc: Description for progress bar
        """
        self.total_steps = total_steps
        self.desc = desc
        self.current_step = 0
        self.start_time = datetime.now()
        
        # Try to use rich progress bar
        try:
            from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
            
            self.progress = Progress(
                SpinnerColumn(),
                "[progress.description]{task.description}",
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                console=console
            )
            self.task = self.progress.add_task(desc, total=total_steps)
            self.progress.start()
            self.use_rich = True
            
        except ImportError:
            self.use_rich = False
            logger.info(f"Starting {desc} with {total_steps} steps")
    
    def update(self, n: int = 1, **kwargs):
        """Update progress.
        
        Args:
            n: Number of steps to advance
            **kwargs: Additional info to log
        """
        self.current_step += n
        
        if self.use_rich:
            self.progress.update(self.task, advance=n)
            
            # Log additional info
            if kwargs:
                info_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                   for k, v in kwargs.items())
                self.progress.console.print(f"[cyan]{info_str}[/cyan]")
        else:
            # Calculate progress
            progress = self.current_step / self.total_steps
            elapsed = datetime.now() - self.start_time
            eta = elapsed * (1 - progress) / progress if progress > 0 else elapsed
            
            # Log progress
            info_str = f"Step {self.current_step}/{self.total_steps} ({progress:.1%})"
            if kwargs:
                info_str += " - " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                            for k, v in kwargs.items())
            info_str += f" - ETA: {eta}"
            
            logger.info(info_str)
    
    def finish(self):
        """Finish progress tracking."""
        if self.use_rich:
            self.progress.stop()
            
        elapsed = datetime.now() - self.start_time
        logger.info(f"Completed {self.desc} in {elapsed}")


# Convenience functions
def log_separator(title: str = "", char: str = "=", length: int = 80):
    """Log a separator line.
    
    Args:
        title: Optional title to center in separator
        char: Character to use for separator
        length: Length of separator
    """
    if title:
        padding = (length - len(title) - 2) // 2
        line = f"{char * padding} {title} {char * padding}"
        if len(line) < length:
            line += char * (length - len(line))
    else:
        line = char * length
        
    logger.info(line)


def log_dict(data: dict, title: str = "Configuration"):
    """Log a dictionary in a formatted way.
    
    Args:
        data: Dictionary to log
        title: Title for the log
    """
    log_separator(title)
    for key, value in data.items():
        logger.info(f"{key}: {value}")
    log_separator()


def log_gpu_info():
    """Log GPU information."""
    try:
        import torch
        
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}")
                logger.info(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
                logger.info(f"  Compute capability: {props.major}.{props.minor}")
        else:
            logger.warning("CUDA not available, using CPU")
            
    except ImportError:
        logger.warning("PyTorch not installed, cannot check GPU info")
