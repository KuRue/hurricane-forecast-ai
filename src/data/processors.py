"""Data preprocessing utilities for hurricane forecasting."""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate, signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from loguru import logger

from ..utils.config import get_config


class HurricanePreprocessor:
    """Preprocess hurricane track data for model training."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().data
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def normalize_track_data(self, track_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize hurricane track data.
        
        Args:
            track_df: DataFrame with hurricane track
            
        Returns:
            Normalized DataFrame
        """
        # Create a copy
        df = track_df.copy()
        
        # Convert timestamps to numeric features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['year'] = df['timestamp'].dt.year
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Handle missing values
        # Fill wind radii with 0 (no data means no extent)
        wind_radii_cols = [col for col in df.columns if 'kt' in col]
        df[wind_radii_cols] = df[wind_radii_cols].fillna(0)
        
        # Interpolate missing pressure values
        if 'min_pressure' in df.columns:
            df['min_pressure'] = df['min_pressure'].interpolate(method='linear')
            # Fill remaining NaNs with typical values
            df['min_pressure'] = df['min_pressure'].fillna(1013)
        
        # Ensure max_wind has no NaNs
        if 'max_wind' in df.columns:
            df['max_wind'] = df['max_wind'].interpolate(method='linear')
            df['max_wind'] = df['max_wind'].fillna(df['max_wind'].min())
        
        # Convert wind speed units if needed (ensure knots)
        # HURDAT2 is already in knots, but IBTrACS might need conversion
        
        return df
    
    def create_track_features(self, track_df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from track data.
        
        Args:
            track_df: Normalized track DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = track_df.copy()
        
        # Motion features
        df['lat_change'] = df['latitude'].diff()
        df['lon_change'] = df['longitude'].diff()
        
        # Forward fill the first NaN values
        df['lat_change'] = df['lat_change'].fillna(0)
        df['lon_change'] = df['lon_change'].fillna(0)
        
        # Storm motion speed and direction
        df['storm_speed'] = np.sqrt(
            df['lat_change']**2 + df['lon_change']**2
        ) * 60  # Convert to nautical miles (1 degree ~ 60 nm)
        
        df['storm_direction'] = np.arctan2(
            df['lon_change'], df['lat_change']
        ) * 180 / np.pi
        
        # Intensity change
        if 'max_wind' in df.columns:
            df['intensity_change'] = df['max_wind'].diff().fillna(0)
            df['rapid_intensification'] = (df['intensity_change'] >= 30).astype(int)
        
        # Pressure gradient
        if 'min_pressure' in df.columns:
            df['pressure_change'] = df['min_pressure'].diff().fillna(0)
            df['pressure_gradient'] = df['pressure_change'] / 6  # Per hour (6-hourly data)
        
        # Size metrics from wind radii
        wind_radii_34 = ['ne_34kt', 'se_34kt', 'sw_34kt', 'nw_34kt']
        if all(col in df.columns for col in wind_radii_34):
            df['size_34kt'] = df[wind_radii_34].mean(axis=1)
            df['asymmetry_34kt'] = df[wind_radii_34].std(axis=1)
        
        # Environmental pressure deficit
        if 'min_pressure' in df.columns:
            df['pressure_deficit'] = 1013 - df['min_pressure']
        
        # Coriolis parameter
        df['coriolis'] = 2 * 7.2921e-5 * np.sin(np.radians(df['latitude']))
        
        # Track curvature (requires at least 3 points)
        if len(df) >= 3:
            df['curvature'] = self._calculate_curvature(
                df['latitude'].values,
                df['longitude'].values
            )
        else:
            df['curvature'] = 0
        
        return df
    
    def _calculate_curvature(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Calculate track curvature.
        
        Args:
            lats: Latitude array
            lons: Longitude array
            
        Returns:
            Curvature array
        """
        # Convert to Cartesian coordinates
        x = lons * np.cos(np.radians(lats))
        y = lats
        
        # Calculate first and second derivatives
        if len(x) < 3:
            return np.zeros_like(x)
        
        # Use gradient for derivatives
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**(3/2)
        
        # Avoid division by zero
        curvature = np.zeros_like(numerator)
        mask = denominator > 1e-10
        curvature[mask] = numerator[mask] / denominator[mask]
        
        return curvature
    
    def prepare_sequences(
        self,
        track_df: pd.DataFrame,
        sequence_length: int = 8,
        forecast_length: int = 20,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for time series prediction.
        
        Args:
            track_df: Preprocessed track DataFrame
            sequence_length: Number of input time steps
            forecast_length: Number of output time steps
            stride: Stride for sliding window
            
        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        # Select features for model input
        feature_cols = [
            'latitude', 'longitude', 'max_wind', 'min_pressure',
            'storm_speed', 'storm_direction', 'intensity_change',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'coriolis', 'pressure_deficit'
        ]
        
        # Add wind radii if available
        for col in ['size_34kt', 'asymmetry_34kt']:
            if col in track_df.columns:
                feature_cols.append(col)
        
        # Filter to available columns
        available_features = [col for col in feature_cols if col in track_df.columns]
        self.feature_names = available_features
        
        # Extract feature matrix
        features = track_df[available_features].values
        
        # Create sequences
        total_length = sequence_length + forecast_length
        sequences = []
        
        for i in range(0, len(features) - total_length + 1, stride):
            sequence = features[i:i + total_length]
            sequences.append(sequence)
        
        if not sequences:
            raise ValueError(
                f"Track too short for sequence_length={sequence_length} "
                f"and forecast_length={forecast_length}"
            )
        
        sequences = np.array(sequences)
        
        # Split into input and target
        inputs = sequences[:, :sequence_length, :]
        targets = sequences[:, sequence_length:, :2]  # Only lat/lon for targets
        
        return inputs, targets
    
    def fit_scaler(self, tracks: List[pd.DataFrame]) -> None:
        """Fit scaler on multiple tracks.
        
        Args:
            tracks: List of track DataFrames
        """
        # Collect all features
        all_features = []
        
        for track in tracks:
            track = self.normalize_track_data(track)
            track = self.create_track_features(track)
            
            if self.feature_names:
                features = track[self.feature_names].values
                all_features.append(features)
        
        # Fit scaler
        if all_features:
            all_features = np.vstack(all_features)
            self.scaler.fit(all_features)
            logger.info(f"Fitted scaler on {len(all_features)} samples")
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler.
        
        Args:
            features: Feature array
            
        Returns:
            Scaled features
        """
        return self.scaler.transform(features)


class ERA5Preprocessor:
    """Preprocess ERA5 reanalysis data for hurricane modeling."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize ERA5 preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().data.era5
        self.normalizer = {}
        
    def extract_patches(
        self,
        era5_data: xr.Dataset,
        center_lat: float,
        center_lon: float,
        patch_size: float = 25.0,
        variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        """Extract a patch around hurricane center.
        
        Args:
            era5_data: ERA5 dataset
            center_lat: Hurricane center latitude
            center_lon: Hurricane center longitude
            patch_size: Size of patch in degrees
            variables: Variables to extract
            
        Returns:
            Extracted patch
        """
        if variables is None:
            variables = self.config.variables
        
        # Define bounds
        lat_min = center_lat - patch_size / 2
        lat_max = center_lat + patch_size / 2
        lon_min = center_lon - patch_size / 2
        lon_max = center_lon + patch_size / 2
        
        # Handle longitude wrapping
        if lon_min < -180:
            lon_min += 360
        if lon_max > 180:
            lon_max -= 360
        
        # Extract patch
        if lon_min < lon_max:
            patch = era5_data.sel(
                latitude=slice(lat_max, lat_min),  # ERA5 latitude is descending
                longitude=slice(lon_min, lon_max)
            )
        else:
            # Handle dateline crossing
            patch1 = era5_data.sel(
                latitude=slice(lat_max, lat_min),
                longitude=slice(lon_min, 180)
            )
            patch2 = era5_data.sel(
                latitude=slice(lat_max, lat_min),
                longitude=slice(-180, lon_max)
            )
            patch = xr.concat([patch1, patch2], dim='longitude')
        
        # Select variables
        available_vars = [var for var in variables if var in patch.data_vars]
        patch = patch[available_vars]
        
        # Add metadata
        patch.attrs['center_lat'] = center_lat
        patch.attrs['center_lon'] = center_lon
        patch.attrs['patch_size'] = patch_size
        
        return patch
    
    def compute_derived_fields(self, era5_patch: xr.Dataset) -> xr.Dataset:
        """Compute derived meteorological fields.
        
        Args:
            era5_patch: ERA5 patch data
            
        Returns:
            Dataset with additional derived fields
        """
        patch = era5_patch.copy()
        
        # Wind speed and direction
        if 'u10' in patch and 'v10' in patch:
            patch['wind_speed'] = np.sqrt(patch['u10']**2 + patch['v10']**2)
            patch['wind_direction'] = np.arctan2(patch['v10'], patch['u10']) * 180 / np.pi
        
        # Relative vorticity at 10m
        if 'u10' in patch and 'v10' in patch:
            # Calculate vorticity using finite differences
            dx = np.gradient(patch.longitude) * 111000 * np.cos(np.radians(patch.latitude))
            dy = np.gradient(patch.latitude) * 111000
            
            dvdx = patch['v10'].differentiate('longitude') / dx[np.newaxis, :]
            dudy = patch['u10'].differentiate('latitude') / dy[:, np.newaxis]
            
            patch['vorticity_10m'] = dvdx - dudy
        
        # Wind shear (if we have pressure level data)
        if 'u' in patch and 'v' in patch and 'level' in patch.dims:
            # 200-850 hPa shear
            if 200 in patch.level and 850 in patch.level:
                u_200 = patch['u'].sel(level=200)
                v_200 = patch['v'].sel(level=200)
                u_850 = patch['u'].sel(level=850)
                v_850 = patch['v'].sel(level=850)
                
                patch['shear_u'] = u_200 - u_850
                patch['shear_v'] = v_200 - v_850
                patch['shear_magnitude'] = np.sqrt(
                    patch['shear_u']**2 + patch['shear_v']**2
                )
        
        # Convergence at 10m
        if 'u10' in patch and 'v10' in patch:
            dudx = patch['u10'].differentiate('longitude') / dx[np.newaxis, :]
            dvdy = patch['v10'].differentiate('latitude') / dy[:, np.newaxis]
            patch['convergence_10m'] = -(dudx + dvdy)
        
        # Temperature anomaly
        if 't2m' in patch:
            patch['t2m_anomaly'] = patch['t2m'] - patch['t2m'].mean()
        
        return patch
    
    def normalize_fields(
        self,
        era5_patch: xr.Dataset,
        fit: bool = False
    ) -> xr.Dataset:
        """Normalize ERA5 fields.
        
        Args:
            era5_patch: ERA5 patch data
            fit: Whether to fit normalizer
            
        Returns:
            Normalized dataset
        """
        normalized = era5_patch.copy()
        
        for var in era5_patch.data_vars:
            data = era5_patch[var].values
            
            if fit:
                # Fit normalizer
                if var not in self.normalizer:
                    self.normalizer[var] = {
                        'mean': np.nanmean(data),
                        'std': np.nanstd(data)
                    }
            
            # Apply normalization
            if var in self.normalizer:
                mean = self.normalizer[var]['mean']
                std = self.normalizer[var]['std']
                normalized[var] = (data - mean) / (std + 1e-8)
        
        return normalized
    
    def create_multi_scale_features(
        self,
        era5_patch: xr.Dataset,
        scales: List[int] = [1, 2, 4]
    ) -> Dict[int, xr.Dataset]:
        """Create multi-scale representations of ERA5 data.
        
        Args:
            era5_patch: ERA5 patch data
            scales: List of downsampling scales
            
        Returns:
            Dictionary of datasets at different scales
        """
        multi_scale = {}
        
        for scale in scales:
            if scale == 1:
                multi_scale[scale] = era5_patch
            else:
                # Downsample using coarsen
                coarsened = era5_patch.coarsen(
                    latitude=scale,
                    longitude=scale,
                    boundary='trim'
                ).mean()
                
                multi_scale[scale] = coarsened
        
        return multi_scale
    
    def to_tensor(
        self,
        era5_patch: xr.Dataset,
        variables: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Convert ERA5 patch to PyTorch tensor.
        
        Args:
            era5_patch: ERA5 patch data
            variables: Variables to include
            
        Returns:
            Tensor of shape (C, H, W)
        """
        if variables is None:
            variables = list(era5_patch.data_vars)
        
        # Stack variables
        arrays = []
        for var in variables:
            if var in era5_patch:
                arr = era5_patch[var].values
                if arr.ndim == 2:
                    arrays.append(arr)
                elif arr.ndim == 3:
                    # If there's a time dimension, take the first time
                    arrays.append(arr[0])
        
        # Stack into tensor
        if arrays:
            tensor = np.stack(arrays, axis=0)
            return torch.from_numpy(tensor).float()
        else:
            raise ValueError("No valid variables found in dataset")


def normalize_track_data(track_df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to normalize track data.
    
    Args:
        track_df: Hurricane track DataFrame
        
    Returns:
        Normalized DataFrame
    """
    preprocessor = HurricanePreprocessor()
    return preprocessor.normalize_track_data(track_df)


def create_track_features(track_df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to create track features.
    
    Args:
        track_df: Hurricane track DataFrame
        
    Returns:
        DataFrame with features
    """
    preprocessor = HurricanePreprocessor()
    normalized = preprocessor.normalize_track_data(track_df)
    return preprocessor.create_track_features(normalized)
