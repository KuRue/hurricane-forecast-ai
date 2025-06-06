#!/usr/bin/env python3
"""Setup script to download required data and models for hurricane forecasting.

This script downloads:
- HURDAT2 hurricane database
- IBTrACS global tropical cyclone data
- ERA5 sample data (full download requires CDS API key)
- Pre-trained model weights (GraphCast, Pangu-Weather)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional
import requests
import zipfile
import tarfile
from tqdm import tqdm
import yaml
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level
    )


def download_file(url: str, output_path: Path, description: str) -> bool:
    """Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save file
        description: Description for progress bar
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
        logger.info(f"Downloaded {description} to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {description}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def setup_directories(config_path: Optional[Path] = None) -> dict:
    """Create necessary directories based on configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary of directory paths
    """
    # Load configuration
    if config_path is None:
        config_path = project_root / "configs" / "default_config.yaml"
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get data root directory
    data_root = Path(config['data']['root_dir']).expanduser()
    
    # Create directory structure
    directories = {
        'root': data_root,
        'hurdat2': data_root / 'hurdat2',
        'ibtracs': data_root / 'ibtracs',
        'era5': data_root / 'era5',
        'models': data_root / 'models',
        'graphcast': data_root / 'models' / 'graphcast',
        'pangu': data_root / 'models' / 'pangu',
        'checkpoints': data_root / 'checkpoints',
        'cache': data_root / 'cache',
        'logs': data_root / 'logs',
        'mlruns': data_root / 'mlruns'
    }
    
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {path}")
        
    logger.info(f"Data directory structure created at {data_root}")
    
    return directories


def download_hurdat2(data_dir: Path, force: bool = False) -> bool:
    """Download HURDAT2 Atlantic hurricane database.
    
    Args:
        data_dir: Directory to save data
        force: Force re-download
        
    Returns:
        True if successful
    """
    url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"
    output_path = data_dir / "hurdat2.txt"
    
    if output_path.exists() and not force:
        logger.info("HURDAT2 data already exists, skipping download")
        return True
        
    return download_file(url, output_path, "HURDAT2 Atlantic hurricane database")


def download_ibtracs(data_dir: Path, force: bool = False) -> bool:
    """Download IBTrACS global tropical cyclone database.
    
    Args:
        data_dir: Directory to save data
        force: Force re-download
        
    Returns:
        True if successful
    """
    # Download NetCDF version (smaller than CSV)
    url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/netcdf/IBTrACS.ALL.v04r00.nc"
    output_path = data_dir / "IBTrACS.ALL.v04r00.nc"
    
    if output_path.exists() and not force:
        logger.info("IBTrACS data already exists, skipping download")
        return True
        
    return download_file(url, output_path, "IBTrACS global cyclone database")


def setup_era5_credentials() -> bool:
    """Set up CDS API credentials for ERA5 download.
    
    Returns:
        True if credentials are set up
    """
    cdsapi_path = Path.home() / ".cdsapirc"
    
    if cdsapi_path.exists():
        logger.info("CDS API credentials already configured")
        return True
        
    logger.warning("CDS API credentials not found")
    logger.info("To download ERA5 data, you need to:")
    logger.info("1. Register at https://cds.climate.copernicus.eu/user/register")
    logger.info("2. Get your API key from https://cds.climate.copernicus.eu/api-how-to")
    logger.info("3. Create ~/.cdsapirc with:")
    logger.info("   url: https://cds.climate.copernicus.eu/api/v2")
    logger.info("   key: YOUR_UID:YOUR_API_KEY")
    
    response = input("\nDo you have CDS API credentials to set up now? (y/n): ")
    
    if response.lower() == 'y':
        uid = input("Enter your UID: ")
        api_key = input("Enter your API key: ")
        
        with open(cdsapi_path, 'w') as f:
            f.write("url: https://cds.climate.copernicus.eu/api/v2\n")
            f.write(f"key: {uid}:{api_key}\n")
            
        os.chmod(cdsapi_path, 0o600)
        logger.info("CDS API credentials saved")
        return True
        
    return False


def download_era5_sample(data_dir: Path, force: bool = False) -> bool:
    """Download sample ERA5 data for testing.
    
    Args:
        data_dir: Directory to save data
        force: Force re-download
        
    Returns:
        True if successful
    """
    output_path = data_dir / "era5_sample_2023_hurricane_lee.nc"
    
    if output_path.exists() and not force:
        logger.info("ERA5 sample data already exists, skipping download")
        return True
    
    # Check if CDS API is configured
    try:
        import cdsapi
        c = cdsapi.Client()
        
        logger.info("Downloading ERA5 sample data (Hurricane Lee 2023)...")
        
        # Download a small sample around Hurricane Lee's track
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    'mean_sea_level_pressure',
                    'sea_surface_temperature',
                ],
                'year': '2023',
                'month': '09',
                'day': ['05', '06', '07', '08', '09', '10'],
                'time': ['00:00', '06:00', '12:00', '18:00'],
                'area': [40, -80, 10, -40],  # North, West, South, East
            },
            str(output_path)
        )
        
        logger.info(f"Downloaded ERA5 sample data to {output_path}")
        return True
        
    except ImportError:
        logger.error("cdsapi not installed. Install with: pip install cdsapi")
        return False
    except Exception as e:
        logger.error(f"Failed to download ERA5 sample: {e}")
        logger.info("Make sure your CDS API credentials are configured")
        return False


def download_graphcast_weights(model_dir: Path, force: bool = False) -> bool:
    """Download GraphCast model weights.
    
    Args:
        model_dir: Directory to save model
        force: Force re-download
        
    Returns:
        True if successful
    """
    logger.info("GraphCast model download instructions:")
    logger.info("GraphCast weights must be downloaded manually due to license restrictions")
    logger.info("1. Visit: https://github.com/deepmind/graphcast")
    logger.info("2. Follow their instructions to download model weights")
    logger.info(f"3. Place the weights in: {model_dir}")
    logger.info("4. Expected files: params.npz, config.json")
    
    # Create placeholder files for development
    params_path = model_dir / "params.npz"
    config_path = model_dir / "config.json"
    
    if not params_path.exists():
        logger.info("Creating GraphCast placeholder files for development...")
        import numpy as np
        import json
        
        # Create dummy params file
        np.savez(params_path, dummy=np.array([1, 2, 3]))
        
        # Create config file
        config = {
            "model_type": "graphcast",
            "resolution": 0.25,
            "variables": 227,
            "pressure_levels": 37
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.warning("Created placeholder GraphCast files. Replace with real weights for production use.")
    
    return True


def download_pangu_weights(model_dir: Path, force: bool = False) -> bool:
    """Download Pangu-Weather model weights.
    
    Args:
        model_dir: Directory to save model
        force: Force re-download
        
    Returns:
        True if successful
    """
    logger.info("Pangu-Weather model download instructions:")
    logger.info("1. Visit: https://github.com/198808xc/Pangu-Weather")
    logger.info("2. Follow instructions to download ONNX model files")
    logger.info(f"3. Place the models in: {model_dir}")
    logger.info("4. Expected files: pangu_weather_24.onnx, pangu_weather_6.onnx, etc.")
    
    # Create placeholder for development
    onnx_path = model_dir / "pangu_weather_24.onnx"
    
    if not onnx_path.exists():
        logger.info("Creating Pangu-Weather placeholder file for development...")
        
        # Create a minimal valid ONNX file
        try:
            import onnx
            from onnx import helper, TensorProto
            
            # Create a dummy model
            input_tensor = helper.make_tensor_value_info(
                'input', TensorProto.FLOAT, [1, 69, 721, 1440]
            )
            output_tensor = helper.make_tensor_value_info(
                'output', TensorProto.FLOAT, [1, 69, 721, 1440]
            )
            
            node = helper.make_node(
                'Identity',
                inputs=['input'],
                outputs=['output']
            )
            
            graph = helper.make_graph(
                [node],
                'pangu_placeholder',
                [input_tensor],
                [output_tensor]
            )
            
            model = helper.make_model(graph)
            onnx.save(model, str(onnx_path))
            
            logger.warning("Created placeholder Pangu-Weather file. Replace with real model for production use.")
            
        except ImportError:
            logger.error("ONNX not installed. Install with: pip install onnx")
            onnx_path.touch()  # Create empty file
            
    return True


def verify_installation(dirs: dict) -> bool:
    """Verify that all required files are present.
    
    Args:
        dirs: Dictionary of directories
        
    Returns:
        True if all files present
    """
    logger.info("\nVerifying installation...")
    
    required_files = {
        'HURDAT2': dirs['hurdat2'] / 'hurdat2.txt',
        'IBTrACS': dirs['ibtracs'] / 'IBTrACS.ALL.v04r00.nc',
        'GraphCast params': dirs['graphcast'] / 'params.npz',
        'GraphCast config': dirs['graphcast'] / 'config.json',
        'Pangu-Weather model': dirs['pangu'] / 'pangu_weather_24.onnx',
    }
    
    optional_files = {
        'ERA5 sample': dirs['era5'] / 'era5_sample_2023_hurricane_lee.nc',
        'CDS API config': Path.home() / '.cdsapirc',
    }
    
    all_good = True
    
    logger.info("\nRequired files:")
    for name, path in required_files.items():
        if path.exists():
            logger.info(f"✓ {name}: {path}")
        else:
            logger.error(f"✗ {name}: {path} (MISSING)")
            all_good = False
            
    logger.info("\nOptional files:")
    for name, path in optional_files.items():
        if path.exists():
            logger.info(f"✓ {name}: {path}")
        else:
            logger.warning(f"○ {name}: {path} (not configured)")
            
    return all_good


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Set up data and models for hurricane forecasting"
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--download-era5',
        action='store_true',
        help='Download ERA5 sample data'
    )
    parser.add_argument(
        '--download-models',
        action='store_true',
        help='Download model weights'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download of existing files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    logger.info("Hurricane Forecast AI - Data Setup")
    logger.info("=" * 50)
    
    # Create directories
    dirs = setup_directories(args.config)
    
    # Download hurricane databases
    logger.info("\nDownloading hurricane databases...")
    success = True
    
    if not download_hurdat2(dirs['hurdat2'], args.force):
        success = False
        
    if not download_ibtracs(dirs['ibtracs'], args.force):
        success = False
    
    # Set up ERA5 if requested
    if args.download_era5:
        logger.info("\nSetting up ERA5 data access...")
        if setup_era5_credentials():
            if not download_era5_sample(dirs['era5'], args.force):
                logger.warning("Failed to download ERA5 sample data")
        else:
            logger.warning("Skipping ERA5 download (no credentials)")
    
    # Download models if requested
    if args.download_models:
        logger.info("\nSetting up model weights...")
        download_graphcast_weights(dirs['graphcast'], args.force)
        download_pangu_weights(dirs['pangu'], args.force)
    
    # Verify installation
    if verify_installation(dirs):
        logger.info("\n✓ Setup completed successfully!")
        logger.info(f"Data directory: {dirs['root']}")
        logger.info("\nNext steps:")
        logger.info("1. Replace placeholder model files with real weights")
        logger.info("2. Configure ERA5 access if needed")
        logger.info("3. Run training with: python scripts/train_model.py")
        return 0
    else:
        logger.error("\n✗ Setup incomplete. Please check missing files.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
