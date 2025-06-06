"""Dataset classes for hurricane model training."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from loguru import logger

from ..data.loaders import HurricaneDataPipeline
from ..data.processors import HurricanePreprocessor, ERA5Preprocessor
from ..data.validators import validate_training_data
from ..utils.config import get_config


class HurricaneDataset(Dataset):
    """Basic hurricane dataset for track and intensity prediction."""
    
    def __init__(
        self,
        storm_ids: List[str],
        sequence_length: int = 8,
        forecast_length: int = 20,
        data_source: str = "hurdat2",
        include_era5: bool = False,
        transform: Optional[callable] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize hurricane dataset.
        
        Args:
            storm_ids: List of storm IDs to include
            sequence_length: Input sequence length (time steps)
            forecast_length: Forecast length (time steps)
            data_source: Data source ("hurdat2" or "ibtracs")
            include_era5: Whether to include ERA5 reanalysis data
            transform: Optional data transformation
            cache_dir: Directory to cache processed data
        """
        self.storm_ids = storm_ids
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length
        self.data_source = data_source
        self.include_era5 = include_era5
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize data pipeline and preprocessor
        self.pipeline = HurricaneDataPipeline()
        self.preprocessor = HurricanePreprocessor()
        
        if self.include_era5:
            self.era5_preprocessor = ERA5Preprocessor()
        
        # Load and prepare data
        self._prepare_dataset()
        
        logger.info(
            f"Initialized HurricaneDataset with {len(self.samples)} samples "
            f"from {len(self.storm_ids)} storms"
        )
    
    def _prepare_dataset(self):
        """Prepare dataset by loading and processing all storms."""
        self.samples = []
        
        for storm_id in self.storm_ids:
            try:
                # Check cache first
                if self.cache_dir and self._is_cached(storm_id):
                    sample = self._load_from_cache(storm_id)
                else:
                    # Load storm data
                    storm_data = self.pipeline.load_hurricane_for_training(
                        storm_id=storm_id,
                        source=self.data_source,
                        include_era5=self.include_era5
                    )
                    
                    # Validate data
                    if not validate_training_data(storm_data):
                        logger.warning(f"Skipping storm {storm_id} due to validation failure")
                        continue
                    
                    # Process track data
                    track_df = storm_data['track']
                    track_normalized = self.preprocessor.normalize_track_data(track_df)
                    track_features = self.preprocessor.create_track_features(track_normalized)
                    
                    # Create sequences
                    inputs, targets = self.preprocessor.prepare_sequences(
                        track_features,
                        sequence_length=self.sequence_length,
                        forecast_length=self.forecast_length
                    )
                    
                    # Process ERA5 data if available
                    era5_patches = None
                    if self.include_era5 and 'era5' in storm_data:
                        era5_patches = self._process_era5_data(
                            storm_data['era5'],
                            track_features
                        )
                    
                    # Create sample
                    sample = {
                        'storm_id': storm_id,
                        'inputs': inputs,
                        'targets': targets,
                        'track_features': track_features,
                        'era5_patches': era5_patches
                    }
                    
                    # Cache if enabled
                    if self.cache_dir:
                        self._save_to_cache(storm_id, sample)
                
                # Add sequences from this storm
                for i in range(len(sample['inputs'])):
                    self.samples.append({
                        'storm_id': storm_id,
                        'sequence_idx': i,
                        'inputs': sample['inputs'][i],
                        'targets': sample['targets'][i],
                        'era5': sample['era5_patches'][i] if sample['era5_patches'] is not None else None
                    })
                    
            except Exception as e:
                logger.error(f"Error processing storm {storm_id}: {e}")
    
    def _process_era5_data(
        self,
        era5_data: xr.Dataset,
        track_features: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Process ERA5 data for the storm track.
        
        Args:
            era5_data: ERA5 dataset
            track_features: Hurricane track features
            
        Returns:
            Processed ERA5 patches
        """
        try:
            patches = []
            
            for _, point in track_features.iterrows():
                # Extract patch around hurricane center
                patch = self.era5_preprocessor.extract_patches(
                    era5_data,
                    center_lat=point['latitude'],
                    center_lon=point['longitude'],
                    patch_size=25.0
                )
                
                # Compute derived fields
                patch = self.era5_preprocessor.compute_derived_fields(patch)
                
                # Normalize
                patch = self.era5_preprocessor.normalize_fields(patch, fit=True)
                
                # Convert to tensor
                patch_tensor = self.era5_preprocessor.to_tensor(patch)
                patches.append(patch_tensor)
            
            return torch.stack(patches).numpy()
            
        except Exception as e:
            logger.warning(f"Error processing ERA5 data: {e}")
            return None
    
    def _is_cached(self, storm_id: str) -> bool:
        """Check if storm data is cached."""
        cache_file = self.cache_dir / f"{storm_id}.pt"
        return cache_file.exists()
    
    def _load_from_cache(self, storm_id: str) -> Dict:
        """Load processed storm data from cache."""
        cache_file = self.cache_dir / f"{storm_id}.pt"
        return torch.load(cache_file)
    
    def _save_to_cache(self, storm_id: str, data: Dict):
        """Save processed storm data to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{storm_id}.pt"
        torch.save(data, cache_file)
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample data
        """
        sample = self.samples[idx]
        
        # Convert to tensors
        item = {
            'inputs': torch.from_numpy(sample['inputs']).float(),
            'target_track': torch.from_numpy(sample['targets']).float(),
            'storm_id': sample['storm_id']
        }
        
        # Add ERA5 data if available
        if sample['era5'] is not None:
            item['reanalysis_maps'] = torch.from_numpy(sample['era5']).float()
        
        # Extract additional targets
        # Assuming input features include wind and pressure
        if sample['inputs'].shape[-1] >= 4:
            # Extract wind and pressure from future targets
            # This is simplified - actual implementation would be more sophisticated
            item['target_wind'] = item['inputs'][:, 2]  # Placeholder
            item['target_pressure'] = item['inputs'][:, 3]  # Placeholder
        
        # Apply transform if provided
        if self.transform:
            item = self.transform(item)
        
        return item


class HurricaneSequenceDataset(Dataset):
    """Advanced dataset with full sequence handling and augmentation."""
    
    def __init__(
        self,
        storm_ids: List[str],
        config: Dict,
        mode: str = "train",
        augment: bool = True
    ):
        """Initialize sequence dataset.
        
        Args:
            storm_ids: List of storm IDs
            config: Dataset configuration
            mode: Dataset mode (train/val/test)
            augment: Whether to apply data augmentation
        """
        self.storm_ids = storm_ids
        self.config = config
        self.mode = mode
        self.augment = augment and mode == "train"
        
        # Load configuration
        self.sequence_length = config.get('sequence_length', 8)
        self.forecast_length = config.get('forecast_length', 20)
        self.stride = config.get('stride', 1 if mode == "train" else 4)
        
        # Initialize components
        self.pipeline = HurricaneDataPipeline()
        self.preprocessor = HurricanePreprocessor()
        
        # Prepare sequences
        self._prepare_sequences()
        
        logger.info(
            f"Initialized HurricaneSequenceDataset ({mode}) with "
            f"{len(self.sequences)} sequences from {len(self.storm_ids)} storms"
        )
    
    def _prepare_sequences(self):
        """Prepare all sequences from storms."""
        self.sequences = []
        self.storm_data = {}
        
        for storm_id in self.storm_ids:
            try:
                # Load storm
                storm_data = self.pipeline.load_hurricane_for_training(
                    storm_id=storm_id,
                    source='hurdat2',
                    include_era5=True
                )
                
                # Process track
                track_df = storm_data['track']
                track_normalized = self.preprocessor.normalize_track_data(track_df)
                track_features = self.preprocessor.create_track_features(track_normalized)
                
                # Store processed data
                self.storm_data[storm_id] = {
                    'track_features': track_features,
                    'era5': storm_data.get('era5')
                }
                
                # Create sequence indices
                total_length = len(track_features)
                sequence_total = self.sequence_length + self.forecast_length
                
                for start_idx in range(0, total_length - sequence_total + 1, self.stride):
                    self.sequences.append({
                        'storm_id': storm_id,
                        'start_idx': start_idx,
                        'end_idx': start_idx + sequence_total
                    })
                    
            except Exception as e:
                logger.error(f"Error processing storm {storm_id}: {e}")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample data
        """
        seq_info = self.sequences[idx]
        storm_id = seq_info['storm_id']
        start_idx = seq_info['start_idx']
        end_idx = seq_info['end_idx']
        
        # Get storm data
        storm_data = self.storm_data[storm_id]
        track_features = storm_data['track_features']
        
        # Extract sequence
        sequence = track_features.iloc[start_idx:end_idx]
        
        # Split into input and target
        input_seq = sequence.iloc[:self.sequence_length]
        target_seq = sequence.iloc[self.sequence_length:]
        
        # Prepare features
        feature_cols = self.preprocessor.feature_names
        input_features = input_seq[feature_cols].values
        
        # Prepare targets
        target_track = target_seq[['latitude', 'longitude']].values
        target_wind = target_seq['max_wind'].values if 'max_wind' in target_seq else None
        target_pressure = target_seq['min_pressure'].values if 'min_pressure' in target_seq else None
        
        # Apply augmentation if enabled
        if self.augment:
            input_features, target_track = self._augment_sequence(
                input_features, target_track
            )
        
        # Convert to tensors
        item = {
            'inputs': torch.from_numpy(input_features).float(),
            'target_track': torch.from_numpy(target_track).float(),
            'storm_id': storm_id,
            'timestamp': input_seq['timestamp'].iloc[0]
        }
        
        if target_wind is not None:
            item['target_wind'] = torch.from_numpy(target_wind).float()
        
        if target_pressure is not None:
            item['target_pressure'] = torch.from_numpy(target_pressure).float()
        
        # Add ERA5 data if available
        if storm_data['era5'] is not None:
            # Extract ERA5 patches for the sequence
            # This is simplified - actual implementation would properly align times
            item['reanalysis_maps'] = torch.randn(
                self.sequence_length, 9, 100, 100
            )  # Placeholder
        
        return item
    
    def _augment_sequence(
        self,
        inputs: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to sequence.
        
        Args:
            inputs: Input features
            targets: Target values
            
        Returns:
            Augmented inputs and targets
        """
        # Random rotation (small angle)
        if np.random.random() < 0.5:
            angle = np.random.uniform(-5, 5)  # degrees
            # Apply rotation to track coordinates
            # Simplified - actual implementation would properly rotate coordinates
            rotation_noise = np.random.randn(*targets.shape) * 0.1
            targets = targets + rotation_noise
        
        # Random time shift
        if np.random.random() < 0.3:
            # Shift temporal features slightly
            if inputs.shape[-1] > 10:  # Assuming temporal features are at specific indices
                inputs[:, 8:10] += np.random.randn(inputs.shape[0], 2) * 0.05
        
        # Random intensity perturbation
        if np.random.random() < 0.5:
            # Perturb intensity-related features
            if inputs.shape[-1] > 4:
                inputs[:, 2:4] += np.random.randn(inputs.shape[0], 2) * 0.02
        
        return inputs, targets


def create_data_loaders(
    config: Union[Dict, str],
    train_storms: List[str],
    val_storms: Optional[List[str]] = None,
    test_storms: Optional[List[str]] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Create data loaders for training.
    
    Args:
        config: Configuration dict or path
        train_storms: List of training storm IDs
        val_storms: List of validation storm IDs
        test_storms: List of test storm IDs
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if isinstance(config, str):
        config = get_config(config)
    
    # Create datasets
    train_dataset = HurricaneSequenceDataset(
        storm_ids=train_storms,
        config=config.data,
        mode='train',
        augment=True
    )
    
    val_dataset = None
    if val_storms:
        val_dataset = HurricaneSequenceDataset(
            storm_ids=val_storms,
            config=config.data,
            mode='val',
            augment=False
        )
    
    test_dataset = None
    if test_storms:
        test_dataset = HurricaneSequenceDataset(
            storm_ids=test_storms,
            config=config.data,
            mode='test',
            augment=False
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.pipeline.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.data.pipeline.pin_memory,
        drop_last=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.pipeline.batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.data.pipeline.pin_memory
        )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.pipeline.batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.data.pipeline.pin_memory
        )
    
    logger.info(
        f"Created data loaders: "
        f"train={len(train_loader)} batches, "
        f"val={len(val_loader) if val_loader else 0} batches, "
        f"test={len(test_loader) if test_loader else 0} batches"
    )
    
    return train_loader, val_loader, test_loader
