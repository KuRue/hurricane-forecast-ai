"""Data loaders for hurricane forecasting system.

This module implements data loaders for:
- HURDAT2: NOAA's hurricane database
- IBTrACS: International Best Track Archive
- ERA5: ECMWF reanalysis data
"""

import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
from loguru import logger
import requests
import geopandas as gpd
from shapely.geometry import Point

from ..utils.config import get_config


class HURDAT2Loader:
    """Load and process HURDAT2 hurricane database.
    
    HURDAT2 contains Atlantic hurricane data from 1851-present.
    Format: https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-atl-1851-2021.pdf
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize HURDAT2 loader.
        
        Args:
            data_path: Path to HURDAT2 file. If None, downloads latest.
        """
        self.data_path = data_path or self._get_default_path()
        self.storms_df = None
        self.tracks_df = None
        
    def _get_default_path(self) -> Path:
        """Get default HURDAT2 data path."""
        config = get_config()
        return Path(config.data.hurdat2_path)
    
    def download_latest(self, force: bool = False) -> Path:
        """Download latest HURDAT2 data from NOAA.
        
        Args:
            force: Force re-download even if file exists
            
        Returns:
            Path to downloaded file
        """
        url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"
        output_path = self.data_path
        
        if output_path.exists() and not force:
            logger.info(f"HURDAT2 data already exists at {output_path}")
            return output_path
            
        logger.info(f"Downloading HURDAT2 data from {url}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"Downloaded HURDAT2 data to {output_path}")
        return output_path
    
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load HURDAT2 data into DataFrames.
        
        Returns:
            Tuple of (storms_df, tracks_df)
        """
        if not self.data_path.exists():
            self.download_latest()
            
        logger.info(f"Loading HURDAT2 data from {self.data_path}")
        
        storms = []
        tracks = []
        
        with open(self.data_path, 'r') as f:
            current_storm = None
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Header line for new storm
                if line.startswith('AL') or line.startswith('EP'):
                    parts = line.split(',')
                    storm_id = parts[0].strip()
                    storm_name = parts[1].strip()
                    storm_entries = int(parts[2].strip())
                    
                    current_storm = {
                        'storm_id': storm_id,
                        'name': storm_name,
                        'num_entries': storm_entries
                    }
                    storms.append(current_storm)
                    
                else:
                    # Track data line
                    parts = [p.strip() for p in line.split(',')]
                    
                    # Parse date and time
                    date_str = parts[0]
                    time_str = parts[1]
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    hour = int(time_str[:2])
                    minute = int(time_str[2:]) if len(time_str) > 2 else 0
                    
                    timestamp = datetime(year, month, day, hour, minute)
                    
                    # Parse track data
                    track_point = {
                        'storm_id': current_storm['storm_id'],
                        'timestamp': timestamp,
                        'record_identifier': parts[2],
                        'system_status': parts[3],
                        'latitude': self._parse_latitude(parts[4]),
                        'longitude': self._parse_longitude(parts[5]),
                        'max_wind': int(parts[6]) if parts[6] else np.nan,
                        'min_pressure': int(parts[7]) if parts[7] != '-999' else np.nan,
                        'ne_34kt': int(parts[8]) if parts[8] != '-999' else np.nan,
                        'se_34kt': int(parts[9]) if parts[9] != '-999' else np.nan,
                        'sw_34kt': int(parts[10]) if parts[10] != '-999' else np.nan,
                        'nw_34kt': int(parts[11]) if parts[11] != '-999' else np.nan,
                    }
                    
                    # Add additional wind radii if available
                    if len(parts) > 12:
                        track_point.update({
                            'ne_50kt': int(parts[12]) if parts[12] != '-999' else np.nan,
                            'se_50kt': int(parts[13]) if parts[13] != '-999' else np.nan,
                            'sw_50kt': int(parts[14]) if parts[14] != '-999' else np.nan,
                            'nw_50kt': int(parts[15]) if parts[15] != '-999' else np.nan,
                        })
                        
                    if len(parts) > 16:
                        track_point.update({
                            'ne_64kt': int(parts[16]) if parts[16] != '-999' else np.nan,
                            'se_64kt': int(parts[17]) if parts[17] != '-999' else np.nan,
                            'sw_64kt': int(parts[18]) if parts[18] != '-999' else np.nan,
                            'nw_64kt': int(parts[19]) if parts[19] != '-999' else np.nan,
                        })
                    
                    tracks.append(track_point)
        
        self.storms_df = pd.DataFrame(storms)
        self.tracks_df = pd.DataFrame(tracks)
        
        # Add year column to storms
        self.storms_df['year'] = self.storms_df['storm_id'].str[4:8].astype(int)
        
        logger.info(f"Loaded {len(self.storms_df)} storms with {len(self.tracks_df)} track points")
        
        return self.storms_df, self.tracks_df
    
    def _parse_latitude(self, lat_str: str) -> float:
        """Parse latitude from HURDAT2 format (e.g., '25.4N')."""
        value = float(lat_str[:-1])
        if lat_str.endswith('S'):
            value = -value
        return value
    
    def _parse_longitude(self, lon_str: str) -> float:
        """Parse longitude from HURDAT2 format (e.g., '71.4W')."""
        value = float(lon_str[:-1])
        if lon_str.endswith('W'):
            value = -value
        return value
    
    def get_storm(self, storm_id: str) -> pd.DataFrame:
        """Get track data for a specific storm.
        
        Args:
            storm_id: Storm identifier (e.g., 'AL052023')
            
        Returns:
            DataFrame with storm track
        """
        if self.tracks_df is None:
            self.load()
            
        return self.tracks_df[self.tracks_df['storm_id'] == storm_id].copy()
    
    def get_storms_by_year(self, year: int) -> pd.DataFrame:
        """Get all storms from a specific year.
        
        Args:
            year: Year to filter
            
        Returns:
            DataFrame with storms from that year
        """
        if self.storms_df is None:
            self.load()
            
        return self.storms_df[self.storms_df['year'] == year].copy()


class IBTrACSLoader:
    """Load and process IBTrACS global tropical cyclone database.
    
    IBTrACS contains global tropical cyclone data from multiple agencies.
    """
    
    def __init__(self, data_path: Optional[str] = None, basin: str = 'ALL'):
        """Initialize IBTrACS loader.
        
        Args:
            data_path: Path to IBTrACS NetCDF file
            basin: Basin to load ('ALL', 'NA', 'EP', 'WP', etc.)
        """
        self.data_path = data_path or self._get_default_path()
        self.basin = basin
        self.data = None
        
    def _get_default_path(self) -> Path:
        """Get default IBTrACS data path."""
        config = get_config()
        return Path(config.data.ibtracs_path)
    
    def download_latest(self, force: bool = False) -> Path:
        """Download latest IBTrACS data from NOAA.
        
        Args:
            force: Force re-download even if file exists
            
        Returns:
            Path to downloaded file
        """
        basin_str = self.basin.lower() if self.basin != 'ALL' else 'ALL'
        url = f"https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/netcdf/IBTrACS.{basin_str}.v04r00.nc"
        
        output_path = self.data_path
        
        if output_path.exists() and not force:
            logger.info(f"IBTrACS data already exists at {output_path}")
            return output_path
            
        logger.info(f"Downloading IBTrACS data from {url}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        from tqdm import tqdm
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading IBTrACS') as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
        logger.info(f"Downloaded IBTrACS data to {output_path}")
        return output_path
    
    def load(self) -> xr.Dataset:
        """Load IBTrACS NetCDF data.
        
        Returns:
            xarray Dataset with IBTrACS data
        """
        if not self.data_path.exists():
            self.download_latest()
            
        logger.info(f"Loading IBTrACS data from {self.data_path}")
        
        self.data = xr.open_dataset(self.data_path)
        
        # Decode storm names
        if 'name' in self.data.variables:
            self.data['name'] = self.data['name'].astype(str)
            
        logger.info(f"Loaded IBTrACS data with {len(self.data.storm)} storms")
        
        return self.data
    
    def get_storm(self, storm_id: str) -> xr.Dataset:
        """Get data for a specific storm.
        
        Args:
            storm_id: Storm SID (e.g., '2023239N10316')
            
        Returns:
            Dataset with storm data
        """
        if self.data is None:
            self.load()
            
        # Find storm index
        storm_mask = self.data.sid == storm_id
        storm_idx = np.where(storm_mask)[0]
        
        if len(storm_idx) == 0:
            raise ValueError(f"Storm {storm_id} not found")
            
        return self.data.isel(storm=storm_idx[0])
    
    def to_dataframe(self, storm_id: Optional[str] = None) -> pd.DataFrame:
        """Convert IBTrACS data to pandas DataFrame.
        
        Args:
            storm_id: Specific storm to convert, or None for all
            
        Returns:
            DataFrame with storm track data
        """
        if self.data is None:
            self.load()
            
        if storm_id:
            storm_data = self.get_storm(storm_id)
            df = storm_data.to_dataframe()
        else:
            df = self.data.to_dataframe()
            
        # Clean up the dataframe
        df = df.reset_index()
        
        # Convert time coordinates
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
            
        return df


class ERA5Loader:
    """Load ERA5 reanalysis data from ECMWF.
    
    ERA5 provides hourly atmospheric reanalysis data on a 0.25° grid.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize ERA5 loader.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self._get_default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CDS API client
        try:
            self.client = cdsapi.Client()
        except Exception as e:
            logger.warning(f"CDS API client initialization failed: {e}")
            logger.warning("Please set up ~/.cdsapirc with your credentials")
            self.client = None
            
    def _get_default_cache_dir(self) -> Path:
        """Get default cache directory."""
        config = get_config()
        return Path(config.data.era5_cache_dir)
    
    def download_hurricane_period(
        self,
        start_date: datetime,
        end_date: datetime,
        bbox: Tuple[float, float, float, float],
        variables: Optional[List[str]] = None
    ) -> Path:
        """Download ERA5 data for a hurricane period.
        
        Args:
            start_date: Start of period
            end_date: End of period
            bbox: Bounding box (north, west, south, east)
            variables: List of variables to download
            
        Returns:
            Path to downloaded NetCDF file
        """
        if self.client is None:
            raise RuntimeError("CDS API client not initialized")
            
        if variables is None:
            variables = [
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                'mean_sea_level_pressure',
                '2m_temperature',
                'sea_surface_temperature',
                'total_precipitation',
                'convective_available_potential_energy',
                'geopotential',
                'relative_humidity',
                'temperature',
                'u_component_of_wind',
                'v_component_of_wind',
                'vertical_velocity',
                'vorticity'
            ]
        
        # Create filename based on request
        filename = (
            f"era5_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_"
            f"{bbox[0]:.1f}_{bbox[1]:.1f}_{bbox[2]:.1f}_{bbox[3]:.1f}.nc"
        )
        output_path = self.cache_dir / filename
        
        if output_path.exists():
            logger.info(f"Using cached ERA5 data: {output_path}")
            return output_path
            
        logger.info(f"Downloading ERA5 data for {start_date} to {end_date}")
        
        # Prepare request
        request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variables,
            'year': list(set([
                str(year) 
                for year in range(start_date.year, end_date.year + 1)
            ])),
            'month': [f"{m:02d}" for m in range(1, 13)],
            'day': [f"{d:02d}" for d in range(1, 32)],
            'time': [f"{h:02d}:00" for h in range(24)],
            'area': bbox,  # North, West, South, East
        }
        
        # Handle pressure levels
        if any('pressure' in var or 'geopotential' in var for var in variables):
            request['pressure_level'] = [
                '1000', '975', '950', '925', '900', '875', '850', '825',
                '800', '775', '750', '700', '650', '600', '550', '500',
                '450', '400', '350', '300', '250', '200', '150', '100'
            ]
            
        # Download data
        self.client.retrieve(
            'reanalysis-era5-pressure-levels' if 'pressure_level' in request else 'reanalysis-era5-single-levels',
            request,
            str(output_path)
        )
        
        logger.info(f"Downloaded ERA5 data to {output_path}")
        return output_path
    
    def extract_hurricane_patches(
        self,
        track_df: pd.DataFrame,
        patch_size: float = 25.0,
        variables: Optional[List[str]] = None,
        lead_time_hours: int = 6,
        lag_time_hours: int = 24
    ) -> xr.Dataset:
        """Extract ERA5 patches around hurricane track.
        
        Args:
            track_df: DataFrame with hurricane track (needs lat, lon, timestamp columns)
            patch_size: Size of patch in degrees
            variables: Variables to extract
            lead_time_hours: Hours after storm passage to include
            lag_time_hours: Hours before storm passage to include
            
        Returns:
            Dataset with ERA5 patches
        """
        # Determine spatial and temporal bounds
        lat_min = track_df['latitude'].min() - patch_size / 2
        lat_max = track_df['latitude'].max() + patch_size / 2
        lon_min = track_df['longitude'].min() - patch_size / 2
        lon_max = track_df['longitude'].max() + patch_size / 2
        
        time_min = track_df['timestamp'].min() - timedelta(hours=lag_time_hours)
        time_max = track_df['timestamp'].max() + timedelta(hours=lead_time_hours)
        
        # Download data for the period
        era5_path = self.download_hurricane_period(
            start_date=time_min,
            end_date=time_max,
            bbox=(lat_max, lon_min, lat_min, lon_max),
            variables=variables
        )
        
        # Load and extract patches
        era5_data = xr.open_dataset(era5_path)
        
        patches = []
        for _, point in track_df.iterrows():
            # Extract patch around storm center
            patch = era5_data.sel(
                latitude=slice(
                    point['latitude'] + patch_size / 2,
                    point['latitude'] - patch_size / 2
                ),
                longitude=slice(
                    point['longitude'] - patch_size / 2,
                    point['longitude'] + patch_size / 2
                ),
                time=point['timestamp'],
                method='nearest'
            )
            
            # Add storm information
            patch = patch.assign_coords({
                'storm_lat': point['latitude'],
                'storm_lon': point['longitude'],
                'storm_time': point['timestamp']
            })
            
            patches.append(patch)
            
        # Combine patches
        combined = xr.concat(patches, dim='track_time')
        
        return combined
    
    def load_cached_data(self, filepath: Union[str, Path]) -> xr.Dataset:
        """Load cached ERA5 data.
        
        Args:
            filepath: Path to NetCDF file
            
        Returns:
            ERA5 dataset
        """
        return xr.open_dataset(filepath)


class HurricaneDataPipeline:
    """Main data pipeline combining all data sources."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize data pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        
        # Initialize data loaders
        self.hurdat2 = HURDAT2Loader()
        self.ibtracs = IBTrACSLoader()
        self.era5 = ERA5Loader()
        
        # Cache for loaded data
        self._cache = {}
        
    def load_hurricane_for_training(
        self,
        storm_id: str,
        source: str = 'hurdat2',
        include_era5: bool = True,
        patch_size: float = 25.0
    ) -> Dict[str, Union[pd.DataFrame, xr.Dataset]]:
        """Load complete hurricane data for training.
        
        Args:
            storm_id: Storm identifier
            source: Data source ('hurdat2' or 'ibtracs')
            include_era5: Whether to include ERA5 reanalysis data
            patch_size: Size of ERA5 patches in degrees
            
        Returns:
            Dictionary with track data and optional ERA5 data
        """
        logger.info(f"Loading hurricane {storm_id} from {source}")
        
        # Load track data
        if source == 'hurdat2':
            track_df = self.hurdat2.get_storm(storm_id)
        elif source == 'ibtracs':
            track_df = self.ibtracs.to_dataframe(storm_id)
        else:
            raise ValueError(f"Unknown source: {source}")
            
        result = {'track': track_df}
        
        # Load ERA5 data if requested
        if include_era5:
            logger.info(f"Extracting ERA5 patches for {storm_id}")
            era5_patches = self.era5.extract_hurricane_patches(
                track_df,
                patch_size=patch_size
            )
            result['era5'] = era5_patches
            
        return result
    
    def prepare_training_dataset(
        self,
        years: List[int],
        min_intensity: int = 64,  # Hurricane strength
        source: str = 'hurdat2'
    ) -> pd.DataFrame:
        """Prepare dataset of hurricanes for training.
        
        Args:
            years: List of years to include
            min_intensity: Minimum max wind speed (knots)
            source: Data source
            
        Returns:
            DataFrame with hurricane metadata
        """
        all_storms = []
        
        for year in years:
            logger.info(f"Processing year {year}")
            
            if source == 'hurdat2':
                # Get storms from this year
                year_storms = self.hurdat2.get_storms_by_year(year)
                
                for _, storm in year_storms.iterrows():
                    # Get track data
                    track = self.hurdat2.get_storm(storm['storm_id'])
                    
                    # Check if it reached hurricane strength
                    if track['max_wind'].max() >= min_intensity:
                        storm_info = {
                            'storm_id': storm['storm_id'],
                            'name': storm['name'],
                            'year': year,
                            'max_intensity': track['max_wind'].max(),
                            'min_pressure': track['min_pressure'].min(),
                            'duration_hours': len(track) * 6,  # 6-hourly data
                            'track_length': len(track)
                        }
                        all_storms.append(storm_info)
                        
        storms_df = pd.DataFrame(all_storms)
        logger.info(f"Found {len(storms_df)} hurricanes meeting criteria")
        
        return storms_df
