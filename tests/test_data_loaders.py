"""Unit tests for data loaders."""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

from src.data.loaders import HURDAT2Loader, IBTrACSLoader, ERA5Loader, HurricaneDataPipeline
from src.data.processors import HurricanePreprocessor, ERA5Preprocessor
from src.data.validators import HurricaneDataValidator, validate_training_data


class TestHURDAT2Loader:
    """Test HURDAT2 data loader."""
    
    @pytest.fixture
    def sample_hurdat2_data(self):
        """Create sample HURDAT2 format data."""
        data = """AL092023,                LEE,     55,
20230905, 0000,  , TD, 14.8N,  42.0W,  30, 1008,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20230905, 0600,  , TD, 14.9N,  42.9W,  30, 1008,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20230905, 1200,  , TD, 15.0N,  43.8W,  35, 1007,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20230905, 1800,  , TS, 15.0N,  44.8W,  40, 1006,   40,   40,   30,   30,    0,    0,    0,    0,    0,    0,    0,    0,
20230906, 0000,  , TS, 15.0N,  45.8W,  45, 1003,   50,   50,   40,   40,    0,    0,    0,    0,    0,    0,    0,    0,
AL102023,             MARGOT,     10,
20230907, 1200,  , TD, 16.5N,  36.2W,  30, 1009,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20230907, 1800,  , TD, 17.0N,  37.5W,  35, 1008,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,"""
        return data
    
    @pytest.fixture
    def temp_hurdat2_file(self, sample_hurdat2_data):
        """Create temporary HURDAT2 file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_hurdat2_data)
            temp_path = Path(f.name)
        
        yield temp_path
        
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    
    def test_load_hurdat2(self, temp_hurdat2_file):
        """Test loading HURDAT2 data."""
        loader = HURDAT2Loader(data_path=temp_hurdat2_file)
        storms_df, tracks_df = loader.load()
        
        # Check storms
        assert len(storms_df) == 2
        assert storms_df.iloc[0]['storm_id'] == 'AL092023'
        assert storms_df.iloc[0]['name'] == 'LEE'
        assert storms_df.iloc[1]['storm_id'] == 'AL102023'
        assert storms_df.iloc[1]['name'] == 'MARGOT'
        
        # Check tracks
        assert len(tracks_df) == 7  # 5 for Lee + 2 for Margot
        assert tracks_df.iloc[0]['storm_id'] == 'AL092023'
        assert tracks_df.iloc[0]['latitude'] == 14.8
        assert tracks_df.iloc[0]['longitude'] == -42.0
        assert tracks_df.iloc[0]['max_wind'] == 30
    
    def test_parse_coordinates(self):
        """Test coordinate parsing."""
        loader = HURDAT2Loader()
        
        assert loader._parse_latitude('25.4N') == 25.4
        assert loader._parse_latitude('25.4S') == -25.4
        assert loader._parse_longitude('71.4W') == -71.4
        assert loader._parse_longitude('71.4E') == 71.4
    
    def test_get_storm(self, temp_hurdat2_file):
        """Test getting specific storm."""
        loader = HURDAT2Loader(data_path=temp_hurdat2_file)
        loader.load()
        
        lee_track = loader.get_storm('AL092023')
        assert len(lee_track) == 5
        assert all(lee_track['storm_id'] == 'AL092023')
        
        margot_track = loader.get_storm('AL102023')
        assert len(margot_track) == 2
        assert all(margot_track['storm_id'] == 'AL102023')
    
    def test_get_storms_by_year(self, temp_hurdat2_file):
        """Test filtering storms by year."""
        loader = HURDAT2Loader(data_path=temp_hurdat2_file)
        loader.load()
        
        storms_2023 = loader.get_storms_by_year(2023)
        assert len(storms_2023) == 2
        assert all(storms_2023['year'] == 2023)


class TestHurricanePreprocessor:
    """Test hurricane data preprocessing."""
    
    @pytest.fixture
    def sample_track(self):
        """Create sample hurricane track."""
        dates = pd.date_range('2023-09-05', periods=10, freq='6H')
        track = pd.DataFrame({
            'storm_id': ['AL092023'] * 10,
            'timestamp': dates,
            'latitude': np.linspace(15.0, 25.0, 10),
            'longitude': np.linspace(-45.0, -70.0, 10),
            'max_wind': [30, 35, 40, 50, 65, 80, 85, 75, 65, 50],
            'min_pressure': [1008, 1006, 1003, 998, 985, 970, 968, 975, 985, 995],
            'ne_34kt': [0, 0, 40, 50, 60, 80, 90, 80, 60, 40],
            'se_34kt': [0, 0, 40, 50, 60, 80, 90, 80, 60, 40],
            'sw_34kt': [0, 0, 30, 40, 50, 70, 80, 70, 50, 30],
            'nw_34kt': [0, 0, 30, 40, 50, 70, 80, 70, 50, 30]
        })
        return track
    
    def test_normalize_track_data(self, sample_track):
        """Test track data normalization."""
        preprocessor = HurricanePreprocessor()
        normalized = preprocessor.normalize_track_data(sample_track)
        
        # Check new columns
        assert 'hour_sin' in normalized.columns
        assert 'hour_cos' in normalized.columns
        assert 'day_sin' in normalized.columns
        assert 'day_cos' in normalized.columns
        
        # Check cyclical encoding
        assert -1 <= normalized['hour_sin'].max() <= 1
        assert -1 <= normalized['hour_cos'].max() <= 1
        
        # Check no NaN in wind radii
        wind_cols = [col for col in normalized.columns if 'kt' in col]
        for col in wind_cols:
            assert not normalized[col].isna().any()
    
    def test_create_track_features(self, sample_track):
        """Test feature creation."""
        preprocessor = HurricanePreprocessor()
        normalized = preprocessor.normalize_track_data(sample_track)
        features = preprocessor.create_track_features(normalized)
        
        # Check motion features
        assert 'storm_speed' in features.columns
        assert 'storm_direction' in features.columns
        assert 'intensity_change' in features.columns
        assert 'rapid_intensification' in features.columns
        
        # Check physics features
        assert 'pressure_deficit' in features.columns
        assert 'coriolis' in features.columns
        
        # Check rapid intensification detection
        # From 50 to 65 knots is 15 knot increase (not RI)
        # From 65 to 80 knots is 15 knot increase (not RI)
        assert features['rapid_intensification'].sum() == 0
    
    def test_prepare_sequences(self, sample_track):
        """Test sequence preparation."""
        preprocessor = HurricanePreprocessor()
        normalized = preprocessor.normalize_track_data(sample_track)
        features = preprocessor.create_track_features(normalized)
        
        sequence_length = 4
        forecast_length = 2
        
        inputs, targets = preprocessor.prepare_sequences(
            features,
            sequence_length=sequence_length,
            forecast_length=forecast_length
        )
        
        # Check shapes
        expected_samples = len(features) - (sequence_length + forecast_length) + 1
        assert inputs.shape[0] == expected_samples
        assert inputs.shape[1] == sequence_length
        assert targets.shape[0] == expected_samples
        assert targets.shape[1] == forecast_length
        assert targets.shape[2] == 2  # Only lat/lon for targets
    
    def test_sequence_too_short(self, sample_track):
        """Test error handling for short tracks."""
        preprocessor = HurricanePreprocessor()
        short_track = sample_track.iloc[:5]  # Only 5 points
        
        normalized = preprocessor.normalize_track_data(short_track)
        features = preprocessor.create_track_features(normalized)
        
        with pytest.raises(ValueError, match="Track too short"):
            preprocessor.prepare_sequences(
                features,
                sequence_length=4,
                forecast_length=4  # Total 8 > 5 available
            )


class TestHurricaneDataValidator:
    """Test data validation."""
    
    @pytest.fixture
    def valid_track(self):
        """Create valid hurricane track."""
        dates = pd.date_range('2023-09-05', periods=10, freq='6H')
        track = pd.DataFrame({
            'timestamp': dates,
            'latitude': np.linspace(15.0, 25.0, 10),
            'longitude': np.linspace(-45.0, -70.0, 10),
            'max_wind': np.linspace(30, 120, 10),
            'min_pressure': np.linspace(1005, 950, 10)
        })
        return track
    
    @pytest.fixture
    def invalid_track(self):
        """Create invalid hurricane track."""
        dates = pd.date_range('2023-09-05', periods=5, freq='6H')
        track = pd.DataFrame({
            'timestamp': dates,
            'latitude': [15.0, 95.0, 20.0, np.nan, 25.0],  # Invalid lat
            'longitude': [-45.0, -50.0, -55.0, -60.0, -65.0],
            'max_wind': [30, -10, 250, 80, 90],  # Invalid winds
            'min_pressure': [1005, 850, 980, 970, 1050]  # Invalid pressures
        })
        return track
    
    def test_validate_valid_track(self, valid_track):
        """Test validation of valid track."""
        validator = HurricaneDataValidator()
        results = validator.validate_track(valid_track)
        
        assert results['valid'] == True
        assert len(results['errors']) == 0
    
    def test_validate_invalid_track(self, invalid_track):
        """Test validation of invalid track."""
        validator = HurricaneDataValidator()
        results = validator.validate_track(invalid_track)
        
        assert results['valid'] == False
        assert len(results['errors']) > 0
        
        # Check specific errors
        error_messages = ' '.join(results['errors'])
        assert 'Latitude values outside valid range' in error_messages
        assert 'Missing latitude values' in error_messages
        assert 'Negative wind speeds' in error_messages
        assert 'Wind speed > 200 knots' in error_messages
    
    def test_validate_intensity_physics(self):
        """Test wind-pressure relationship validation."""
        validator = HurricaneDataValidator()
        
        # Create physically consistent data
        pressures = pd.Series([1005, 990, 970, 950, 940])
        pressure_deficit = 1013 - pressures
        expected_winds = 6.3 * np.sqrt(pressure_deficit)
        
        # Add some noise but stay within tolerance
        winds = expected_winds + np.random.normal(0, 10, len(expected_winds))
        
        issues = validator._validate_intensity_physics(winds, pressures)
        assert len(issues['warnings']) == 0
        
        # Create inconsistent data
        bad_winds = pd.Series([30, 150, 40, 160, 50])  # Random values
        issues = validator._validate_intensity_physics(bad_winds, pressures)
        assert len(issues['warnings']) > 0


class TestERA5Preprocessor:
    """Test ERA5 data preprocessing."""
    
    @pytest.fixture
    def sample_era5_data(self):
        """Create sample ERA5 dataset."""
        # Create coordinates
        lats = np.linspace(30, 10, 21)
        lons = np.linspace(-80, -60, 21)
        time = pd.date_range('2023-09-05', periods=4, freq='6H')
        
        # Create data
        shape = (len(time), len(lats), len(lons))
        
        ds = xr.Dataset({
            'u10': xr.DataArray(
                np.random.randn(*shape) * 10,
                dims=['time', 'latitude', 'longitude'],
                coords={'time': time, 'latitude': lats, 'longitude': lons}
            ),
            'v10': xr.DataArray(
                np.random.randn(*shape) * 10,
                dims=['time', 'latitude', 'longitude'],
                coords={'time': time, 'latitude': lats, 'longitude': lons}
            ),
            'msl': xr.DataArray(
                np.random.randn(*shape) * 100 + 101300,
                dims=['time', 'latitude', 'longitude'],
                coords={'time': time, 'latitude': lats, 'longitude': lons}
            ),
            't2m': xr.DataArray(
                np.random.randn(*shape) * 5 + 298,
                dims=['time', 'latitude', 'longitude'],
                coords={'time': time, 'latitude': lats, 'longitude': lons}
            )
        })
        
        return ds
    
    def test_extract_patches(self, sample_era5_data):
        """Test patch extraction."""
        preprocessor = ERA5Preprocessor()
        
        center_lat = 20.0
        center_lon = -70.0
        patch_size = 10.0
        
        patch = preprocessor.extract_patches(
            sample_era5_data,
            center_lat,
            center_lon,
            patch_size
        )
        
        # Check patch bounds
        assert patch.latitude.max() <= center_lat + patch_size / 2
        assert patch.latitude.min() >= center_lat - patch_size / 2
        assert patch.longitude.max() <= center_lon + patch_size / 2
        assert patch.longitude.min() >= center_lon - patch_size / 2
        
        # Check metadata
        assert patch.attrs['center_lat'] == center_lat
        assert patch.attrs['center_lon'] == center_lon
        assert patch.attrs['patch_size'] == patch_size
    
    def test_compute_derived_fields(self, sample_era5_data):
        """Test derived field computation."""
        preprocessor = ERA5Preprocessor()
        
        # Extract a patch first
        patch = preprocessor.extract_patches(
            sample_era5_data,
            center_lat=20.0,
            center_lon=-70.0,
            patch_size=10.0
        )
        
        # Compute derived fields
        enhanced = preprocessor.compute_derived_fields(patch)
        
        # Check new fields
        assert 'wind_speed' in enhanced
        assert 'wind_direction' in enhanced
        assert 'vorticity_10m' in enhanced
        assert 'convergence_10m' in enhanced
        
        # Check wind speed calculation
        expected_speed = np.sqrt(patch['u10']**2 + patch['v10']**2)
        np.testing.assert_allclose(
            enhanced['wind_speed'].values,
            expected_speed.values,
            rtol=1e-5
        )
    
    def test_to_tensor(self, sample_era5_data):
        """Test conversion to tensor."""
        preprocessor = ERA5Preprocessor()
        
        patch = preprocessor.extract_patches(
            sample_era5_data.isel(time=0),  # Single time
            center_lat=20.0,
            center_lon=-70.0,
            patch_size=10.0
        )
        
        tensor = preprocessor.to_tensor(patch, variables=['u10', 'v10', 'msl'])
        
        # Check shape
        assert tensor.shape[0] == 3  # 3 variables
        assert tensor.shape[1] == patch.dims['latitude']
        assert tensor.shape[2] == patch.dims['longitude']
        
        # Check type
        assert tensor.dtype == torch.float32


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_hurricane_data_pipeline(self, temp_hurdat2_file):
        """Test full data pipeline."""
        # This would require more setup including ERA5 data
        # For now, test basic pipeline initialization
        pipeline = HurricaneDataPipeline()
        
        assert pipeline.hurdat2 is not None
        assert pipeline.ibtracs is not None
        assert pipeline.era5 is not None
    
    def test_validate_training_data(self):
        """Test training data validation."""
        # Create sample training data
        dates = pd.date_range('2023-09-05', periods=20, freq='6H')
        track = pd.DataFrame({
            'timestamp': dates,
            'latitude': np.linspace(15.0, 25.0, 20),
            'longitude': np.linspace(-45.0, -70.0, 20),
            'max_wind': np.linspace(30, 120, 20),
            'min_pressure': np.linspace(1005, 950, 20)
        })
        
        training_data = {'track': track}
        
        # Validate
        is_valid = validate_training_data(training_data)
        assert is_valid == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
