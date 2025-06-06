#!/usr/bin/env python3
"""Main training script for hurricane forecasting models.

This script provides a command-line interface for training various
hurricane forecasting models with different configurations.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.loaders import HURDAT2Loader
from src.models import (
    HurricaneCNNTransformer,
    GraphCastHurricane,
    PanguWeatherHurricane,
    HurricaneEnsembleModel
)
from src.models.losses import get_loss_function
from src.training import (
    HurricaneTrainer,
    TrainingConfig,
    create_data_loaders,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    MemoryMonitor,
    TensorBoardLogger,
    FineTuningConfig,
    fine_tune_model
)
from src.utils import setup_logging, get_config, log_gpu_info


def get_storm_splits(
    years: List[int],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_intensity: int = 64
) -> Dict[str, List[str]]:
    """Split storms into train/val/test sets.
    
    Args:
        years: Years to include
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        min_intensity: Minimum intensity threshold
        
    Returns:
        Dictionary with storm ID splits
    """
    # Load HURDAT2 data
    loader = HURDAT2Loader()
    storms_df, tracks_df = loader.load()
    
    # Filter storms by year and intensity
    all_storms = []
    
    for year in years:
        year_storms = storms_df[storms_df['year'] == year]
        
        for _, storm in year_storms.iterrows():
            storm_track = tracks_df[tracks_df['storm_id'] == storm['storm_id']]
            
            # Check if reached minimum intensity
            if storm_track['max_wind'].max() >= min_intensity:
                all_storms.append(storm['storm_id'])
    
    logger.info(f"Found {len(all_storms)} storms meeting criteria")
    
    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(all_storms)
    
    n_val = int(len(all_storms) * val_ratio)
    n_test = int(len(all_storms) * test_ratio)
    n_train = len(all_storms) - n_val - n_test
    
    splits = {
        'train': all_storms[:n_train],
        'val': all_storms[n_train:n_train + n_val],
        'test': all_storms[n_train + n_val:]
    }
    
    logger.info(f"Split sizes - Train: {len(splits['train'])}, "
                f"Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    return splits


def create_model(config: Dict) -> torch.nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Model instance
    """
    model_name = config.model.name
    
    if model_name == "hurricane_cnn":
        model = HurricaneCNNTransformer(config.model)
        
    elif model_name == "graphcast":
        model = GraphCastHurricane(
            config.model,
            checkpoint_path=config.model.graphcast.checkpoint_path
        )
        
    elif model_name == "pangu":
        model = PanguWeatherHurricane(
            config.model,
            model_type=config.model.pangu.model_type,
            onnx_path=config.model.pangu.checkpoint_path
        )
        
    elif model_name == "ensemble":
        model = HurricaneEnsembleModel(
            config.model,
            models=config.model.ensemble.include_models,
            ensemble_size=config.model.ensemble.size
        )
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    logger.info(f"Created {model_name} model with {model.count_parameters():,} parameters")
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train hurricane forecasting models"
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['hurricane_cnn', 'graphcast', 'pangu', 'ensemble'],
        help='Model to train (overrides config)'
    )
    
    # Data settings
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        help='Years to include in training'
    )
    parser.add_argument(
        '--min-intensity',
        type=int,
        default=64,
        help='Minimum hurricane intensity (knots)'
    )
    
    # Training settings
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--loss',
        type=str,
        choices=['track', 'intensity', 'physics', 'ensemble'],
        help='Loss function (overrides config)'
    )
    
    # Fine-tuning
    parser.add_argument(
        '--fine-tune',
        action='store_true',
        help='Fine-tune pre-trained model'
    )
    parser.add_argument(
        '--use-lora',
        action='store_true',
        help='Use LoRA for fine-tuning'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to pre-trained checkpoint'
    )
    
    # Other settings
    parser.add_argument(
        '--name',
        type=str,
        help='Experiment name'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode (small dataset)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run (no training)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="DEBUG" if args.debug else "INFO")
    logger.info("Hurricane Forecast Model Training")
    logger.info("=" * 50)
    
    # Load configuration
    config = get_config(args.config)
    
    # Override configuration with command-line arguments
    if args.model:
        config.model.name = args.model
    if args.epochs:
        config.training.max_epochs = args.epochs
    if args.batch_size:
        config.data.pipeline.batch_size = args.batch_size
    if args.lr:
        config.training.optimizer.lr = args.lr
    if args.loss:
        config.training.loss_type = args.loss
    if args.name:
        config.experiment_name = args.name
    
    # Log configuration
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Epochs: {config.training.max_epochs}")
    logger.info(f"Batch size: {config.data.pipeline.batch_size}")
    logger.info(f"Learning rate: {config.training.optimizer.lr}")
    logger.info(f"Loss: {config.training.loss_type}")
    
    # Check GPU
    log_gpu_info()
    
    if not torch.cuda.is_available():
        logger.warning("GPU not available, training will be slow!")
        if not args.debug:
            response = input("Continue without GPU? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
    
    # Get data splits
    years = args.years or config.data.training_years
    
    if args.debug:
        # Use fewer years for debugging
        years = years[:2]
        config.data.pipeline.batch_size = 2
        config.training.max_epochs = 2
    
    splits = get_storm_splits(
        years=years,
        min_intensity=args.min_intensity
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        config=config,
        train_storms=splits['train'],
        val_storms=splits['val'],
        test_storms=splits['test'],
        num_workers=config.data.pipeline.num_workers
    )
    
    if args.dry_run:
        logger.info("Dry run mode - exiting")
        return
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from: {args.checkpoint}")
        model.load(args.checkpoint)
    
    # Set up fine-tuning if requested
    if args.fine_tune:
        fine_tune_config = FineTuningConfig(
            base_model=config.model.name,
            use_lora=args.use_lora,
            learning_rate=config.training.optimizer.lr,
            num_epochs=config.training.max_epochs
        )
        
        # Fine-tune model
        logger.info("Starting fine-tuning...")
        model = fine_tune_model(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=fine_tune_config
        )
        
    else:
        # Standard training
        training_config = TrainingConfig.from_config(config)
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor=config.training.early_stopping.monitor,
                patience=config.training.early_stopping.patience,
                mode=config.training.early_stopping.mode
            ),
            ModelCheckpoint(
                checkpoint_dir=training_config.checkpoint_dir,
                monitor=config.training.checkpoint.monitor,
                mode=config.training.checkpoint.mode,
                save_top_k=config.training.checkpoint.save_top_k
            ),
            LearningRateMonitor(log_every_n_steps=50),
            MemoryMonitor(log_every_n_steps=100)
        ]
        
        if training_config.use_tensorboard:
            callbacks.append(
                TensorBoardLogger(
                    log_dir=Path(config.data.root_dir) / "logs",
                    comment=config.model.name
                )
            )
        
        # Create trainer
        trainer = HurricaneTrainer(
            model=model,
            config=training_config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            callbacks=callbacks
        )
        
        # Train model
        logger.info("Starting training...")
        try:
            metrics = trainer.train()
            
            # Log final metrics
            logger.info("Training completed!")
            logger.info("Final metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")
            
            # Save final model
            final_path = Path(training_config.checkpoint_dir) / "final_model.pt"
            model.save(final_path)
            logger.info(f"Saved final model to: {final_path}")
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
