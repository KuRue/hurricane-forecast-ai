"""Data validation utilities for hurricane forecasting."""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from loguru import logger

from ..utils.config import get_config


class HurricaneDataValidator:
    """Validate hurricane track and intensity data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().data
        
        # Physical constraints
        self.max_valid_wind = 200  # knots (highest recorded ~185)
        self.min_valid_pressure = 870  # mb (lowest recorded ~870)
        self.max_valid_pressure = 1020  # mb
        self.max_speed_of_motion = 70  # knots
        self.max_lat = 60  # Tropical cyclones rare above 60°
        self.min_lat = -60
        
    def validate_track(self, track_df: pd.DataFrame) -> Dict[str, Union[bool, List[str]]]:
        """Validate hurricane track data.
        
        Args:
            track_df: Hurricane track DataFrame
            
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required columns
        required_cols = ['timestamp', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in track_df.columns]
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"Missing required columns: {missing_cols}")
            return results
        
        # Check data types
        try:
            track_df['timestamp'] = pd.to_datetime(track_df['timestamp'])
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Invalid timestamp format: {e}")
        
        # Validate coordinates
        lat_issues = self._validate_latitude(track_df['latitude'])
        lon_issues = self._validate_longitude(track_df['longitude'])
        
        results['errors'].extend(lat_issues['errors'])
        results['warnings'].extend(lat_issues['warnings'])
        results['errors'].extend(lon_issues['errors'])
        results['warnings'].extend(lon_issues['warnings'])
        
        # Validate physical constraints
        if 'max_wind' in track_df.columns:
            wind_issues = self._validate_wind_speed(track_df['max_wind'])
            results['errors'].extend(wind_issues['errors'])
            results['warnings'].extend(wind_issues['warnings'])
        
        if 'min_pressure' in track_df.columns:
            pressure_issues = self._validate_pressure(track_df['min_pressure'])
            results['errors'].extend(pressure_issues['errors'])
            results['warnings'].extend(pressure_issues['warnings'])
        
        # Validate track continuity
        continuity_issues = self._validate_track_continuity(track_df)
        results['errors'].extend(continuity_issues['errors'])
        results['warnings'].extend(continuity_issues['warnings'])
        
        # Validate intensity physics
        if 'max_wind' in track_df.columns and 'min_pressure' in track_df.columns:
            physics_issues = self._validate_intensity_physics(
                track_df['max_wind'],
                track_df['min_pressure']
            )
            results['warnings'].extend(physics_issues['warnings'])
        
        # Update valid flag
        results['valid'] = len(results['errors']) == 0
        
        return results
    
    def _validate_latitude(self, latitudes: pd.Series) -> Dict[str, List[str]]:
        """Validate latitude values."""
        issues = {'errors': [], 'warnings': []}
        
        # Check range
        if latitudes.min() < -90 or latitudes.max() > 90:
            issues['errors'].append("Latitude values outside valid range [-90, 90]")
        
        # Check for tropical cyclone plausibility
        if latitudes.max() > self.max_lat:
            issues['warnings'].append(
                f"Latitude > {self.max_lat}°: Unusual for tropical cyclones"
            )
        
        # Check for NaN values
        if latitudes.isna().any():
            issues['errors'].append("Missing latitude values detected")
        
        return issues
    
    def _validate_longitude(self, longitudes: pd.Series) -> Dict[str, List[str]]:
        """Validate longitude values."""
        issues = {'errors': [], 'warnings': []}
        
        # Check range
        if longitudes.min() < -180 or longitudes.max() > 180:
            issues['errors'].append("Longitude values outside valid range [-180, 180]")
        
        # Check for NaN values
        if longitudes.isna().any():
            issues['errors'].append("Missing longitude values detected")
        
        # Check for large jumps (possible dateline issues)
        lon_diff = longitudes.diff().abs()
        if (lon_diff > 180).any():
            issues['warnings'].append(
                "Large longitude jumps detected - possible dateline crossing"
            )
        
        return issues
    
    def _validate_wind_speed(self, wind_speeds: pd.Series) -> Dict[str, List[str]]:
        """Validate wind speed values."""
        issues = {'errors': [], 'warnings': []}
        
        # Remove NaN for validation
        valid_winds = wind_speeds.dropna()
        
        if len(valid_winds) == 0:
            issues['warnings'].append("No valid wind speed data")
            return issues
        
        # Check range
        if valid_winds.min() < 0:
            issues['errors'].append("Negative wind speeds detected")
        
        if valid_winds.max() > self.max_valid_wind:
            issues['errors'].append(
                f"Wind speed > {self.max_valid_wind} knots: Exceeds physical limits"
            )
        
        # Check for unrealistic jumps
        wind_changes = valid_winds.diff().abs()
        max_change = wind_changes.max()
        if max_change > 50:  # 50 knot change in 6 hours is extreme
            issues['warnings'].append(
                f"Large wind speed change detected: {max_change:.1f} knots"
            )
        
        return issues
    
    def _validate_pressure(self, pressures: pd.Series) -> Dict[str, List[str]]:
        """Validate pressure values."""
        issues = {'errors': [], 'warnings': []}
        
        # Remove NaN for validation
        valid_pressures = pressures.dropna()
        
        if len(valid_pressures) == 0:
            issues['warnings'].append("No valid pressure data")
            return issues
        
        # Check range
        if valid_pressures.min() < self.min_valid_pressure:
            issues['errors'].append(
                f"Pressure < {self.min_valid_pressure} mb: Below physical limits"
            )
        
        if valid_pressures.max() > self.max_valid_pressure:
            issues['warnings'].append(
                f"Pressure > {self.max_valid_pressure} mb: Unusual for tropical cyclone"
            )
        
        # Check for unrealistic jumps
        pressure_changes = valid_pressures.diff().abs()
        max_change = pressure_changes.max()
        if max_change > 50:  # 50 mb change in 6 hours is extreme
            issues['warnings'].append(
                f"Large pressure change detected: {max_change:.1f} mb"
            )
        
        return issues
    
    def _validate_track_continuity(self, track_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate track continuity and motion."""
        issues = {'errors': [], 'warnings': []}
        
        # Check time intervals
        time_diffs = track_df['timestamp'].diff()
        expected_interval = pd.Timedelta(hours=6)  # Standard for HURDAT2
        
        # Allow for some variation
        min_interval = pd.Timedelta(hours=1)
        max_interval = pd.Timedelta(hours=12)
        
        irregular_intervals = time_diffs[
            (time_diffs < min_interval) | (time_diffs > max_interval)
        ].dropna()
        
        if len(irregular_intervals) > 0:
            issues['warnings'].append(
                f"Irregular time intervals detected: {len(irregular_intervals)} points"
            )
        
        # Check storm motion speed
        lat_diff = track_df['latitude'].diff()
        lon_diff = track_df['longitude'].diff()
        time_hours = time_diffs.dt.total_seconds() / 3600
        
        # Calculate distance (approximate)
        distances = np.sqrt(
            (lat_diff * 60)**2 + 
            (lon_diff * 60 * np.cos(np.radians(track_df['latitude'])))**2
        )
        
        speeds = distances / time_hours
        valid_speeds = speeds.dropna()
        
        if len(valid_speeds) > 0 and valid_speeds.max() > self.max_speed_of_motion:
            issues['warnings'].append(
                f"Unrealistic storm motion speed: {valid_speeds.max():.1f} knots"
            )
        
        # Check for stationary periods
        stationary_threshold = 1  # knot
        stationary_points = (valid_speeds < stationary_threshold).sum()
        if stationary_points > len(valid_speeds) * 0.3:
            issues['warnings'].append(
                f"Storm appears stationary for {stationary_points} points"
            )
        
        return issues
    
    def _validate_intensity_physics(
        self,
        wind_speeds: pd.Series,
        pressures: pd.Series
    ) -> Dict[str, List[str]]:
        """Validate wind-pressure relationship."""
        issues = {'errors': [], 'warnings': []}
        
        # Remove NaN values
        mask = wind_speeds.notna() & pressures.notna()
        winds = wind_speeds[mask]
        pressures = pressures[mask]
        
        if len(winds) < 3:
            return issues
        
        # Check wind-pressure relationship
        # Empirical relationship: V_max ≈ k * sqrt(ΔP)
        # where ΔP = environmental_pressure - central_pressure
        environmental_pressure = 1013  # mb
        pressure_deficit = environmental_pressure - pressures
        
        # Avoid negative values
        pressure_deficit = pressure_deficit.clip(lower=0)
        
        # Expected relationship (Atkinson-Holliday)
        expected_winds = 6.3 * np.sqrt(pressure_deficit)
        
        # Calculate residuals
        residuals = winds - expected_winds
        
        # Check for large deviations
        large_residuals = np.abs(residuals) > 30  # 30 knot tolerance
        if large_residuals.any():
            issues['warnings'].append(
                f"Wind-pressure relationship violations: {large_residuals.sum()} points"
            )
        
        # Check correlation
        if len(winds) > 10:
            correlation = np.corrcoef(winds, -pressures)[0, 1]
            if correlation < 0.7:
                issues['warnings'].append(
                    f"Weak wind-pressure correlation: {correlation:.2f}"
                )
        
        return issues


def validate_track_continuity(track_df: pd.DataFrame) -> bool:
    """Validate track continuity.
    
    Args:
        track_df: Hurricane track DataFrame
        
    Returns:
        True if track is continuous
    """
    validator = HurricaneDataValidator()
    results = validator.validate_track(track_df)
    return results['valid']


def validate_intensity_physics(
    wind_speeds: pd.Series,
    pressures: pd.Series
) -> bool:
    """Validate intensity physics.
    
    Args:
        wind_speeds: Wind speed series
        pressures: Pressure series
        
    Returns:
        True if physics are valid
    """
    validator = HurricaneDataValidator()
    issues = validator._validate_intensity_physics(wind_speeds, pressures)
    return len(issues['warnings']) == 0


class ERA5DataValidator:
    """Validate ERA5 reanalysis data."""
    
    def __init__(self):
        """Initialize ERA5 validator."""
        self.expected_variables = [
            'u10', 'v10', 'msl', 't2m', 'sst'
        ]
        
    def validate_era5_patch(
        self,
        era5_patch: xr.Dataset
    ) -> Dict[str, Union[bool, List[str]]]:
        """Validate ERA5 patch data.
        
        Args:
            era5_patch: ERA5 patch dataset
            
        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check dimensions
        required_dims = ['latitude', 'longitude']
        missing_dims = [dim for dim in required_dims if dim not in era5_patch.dims]
        if missing_dims:
            results['valid'] = False
            results['errors'].append(f"Missing dimensions: {missing_dims}")
        
        # Check variables
        missing_vars = [
            var for var in self.expected_variables 
            if var not in era5_patch.data_vars
        ]
        if missing_vars:
            results['warnings'].append(f"Missing expected variables: {missing_vars}")
        
        # Check for NaN values
        for var in era5_patch.data_vars:
            nan_count = era5_patch[var].isnull().sum().item()
            if nan_count > 0:
                total_size = era5_patch[var].size
                nan_percent = (nan_count / total_size) * 100
                
                if nan_percent > 50:
                    results['errors'].append(
                        f"Variable {var} has {nan_percent:.1f}% missing values"
                    )
                elif nan_percent > 10:
                    results['warnings'].append(
                        f"Variable {var} has {nan_percent:.1f}% missing values"
                    )
        
        # Validate physical ranges
        range_checks = {
            'u10': (-50, 50),  # m/s
            'v10': (-50, 50),  # m/s  
            't2m': (200, 350),  # K
            'msl': (85000, 105000),  # Pa
            'sst': (270, 310)  # K
        }
        
        for var, (vmin, vmax) in range_checks.items():
            if var in era5_patch.data_vars:
                data = era5_patch[var]
                if data.min() < vmin or data.max() > vmax:
                    results['warnings'].append(
                        f"Variable {var} outside expected range [{vmin}, {vmax}]"
                    )
        
        results['valid'] = len(results['errors']) == 0
        
        return results


def validate_training_data(
    hurricane_data: Dict[str, Union[pd.DataFrame, xr.Dataset]]
) -> bool:
    """Validate combined hurricane training data.
    
    Args:
        hurricane_data: Dictionary with 'track' and optionally 'era5' data
        
    Returns:
        True if all data is valid
    """
    # Validate track
    track_validator = HurricaneDataValidator()
    track_results = track_validator.validate_track(hurricane_data['track'])
    
    if not track_results['valid']:
        logger.error(f"Track validation failed: {track_results['errors']}")
        return False
    
    if track_results['warnings']:
        for warning in track_results['warnings']:
            logger.warning(f"Track validation warning: {warning}")
    
    # Validate ERA5 if present
    if 'era5' in hurricane_data:
        era5_validator = ERA5DataValidator()
        era5_results = era5_validator.validate_era5_patch(hurricane_data['era5'])
        
        if not era5_results['valid']:
            logger.error(f"ERA5 validation failed: {era5_results['errors']}")
            return False
        
        if era5_results['warnings']:
            for warning in era5_results['warnings']:
                logger.warning(f"ERA5 validation warning: {warning}")
    
    return True
