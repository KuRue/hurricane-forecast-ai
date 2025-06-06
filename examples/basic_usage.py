#!/usr/bin/env python3
"""Basic usage example for Hurricane Forecast AI system.

This script demonstrates:
1. Loading hurricane data
2. Preprocessing tracks
3. Validating data
4. Basic visualization
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from loguru import logger

from src.data.loaders import HURDAT2Loader, HurricaneDataPipeline
from src.data.processors import HurricanePreprocessor
from src.data.validators import HurricaneDataValidator
from src.utils import setup_logging, get_config
from src.utils.visualization import plot_hurricane_track, plot_intensity_forecast


def main():
    """Run basic usage example."""
    # Setup logging
    setup_logging(level="INFO")
    logger.info("Hurricane Forecast AI - Basic Usage Example")
    
    # Load configuration
    config = get_config()
    logger.info(f"Data directory: {config.data.root_dir}")
    
    # 1. Load hurricane data
    logger.info("\n1. Loading hurricane data...")
    hurdat2_loader = HURDAT2Loader()
    
    try:
        storms_df, tracks_df = hurdat2_loader.load()
        logger.info(f"Loaded {len(storms_df)} storms with {len(tracks_df)} track points")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.info("Run 'python scripts/setup_data.py' to download data")
        return
    
    # 2. Select a specific hurricane
    # Let's use Hurricane Ida (2021) as an example
    storm_id = "AL092021"
    logger.info(f"\n2. Analyzing Hurricane Ida ({storm_id})...")
    
    ida_track = hurdat2_loader.get_storm(storm_id)
    logger.info(f"Track points: {len(ida_track)}")
    logger.info(f"Duration: {ida_track['timestamp'].min()} to {ida_track['timestamp'].max()}")
    logger.info(f"Peak intensity: {ida_track['max_wind'].max()} knots")
    
    # 3. Preprocess the track data
    logger.info("\n3. Preprocessing track data...")
    preprocessor = HurricanePreprocessor()
    
    # Normalize and create features
    ida_normalized = preprocessor.normalize_track_data(ida_track)
    ida_features = preprocessor.create_track_features(ida_normalized)
    
    # Show new features created
    original_cols = set(ida_track.columns)
    new_cols = set(ida_features.columns) - original_cols
    logger.info(f"Created {len(new_cols)} new features:")
    for col in sorted(new_cols)[:5]:  # Show first 5
        logger.info(f"  - {col}")
    logger.info("  ...")
    
    # 4. Validate the data
    logger.info("\n4. Validating hurricane data...")
    validator = HurricaneDataValidator()
    validation_results = validator.validate_track(ida_track)
    
    if validation_results['valid']:
        logger.info("✓ Data validation passed")
    else:
        logger.error("✗ Data validation failed")
        for error in validation_results['errors']:
            logger.error(f"  - {error}")
    
    if validation_results['warnings']:
        logger.warning("Validation warnings:")
        for warning in validation_results['warnings']:
            logger.warning(f"  - {warning}")
    
    # 5. Create training sequences
    logger.info("\n5. Creating training sequences...")
    try:
        sequence_length = 8  # 48 hours of history
        forecast_length = 20  # 120 hours forecast
        
        inputs, targets = preprocessor.prepare_sequences(
            ida_features,
            sequence_length=sequence_length,
            forecast_length=forecast_length
        )
        
        logger.info(f"Input sequences shape: {inputs.shape}")
        logger.info(f"Target sequences shape: {targets.shape}")
        logger.info(f"Number of training samples: {len(inputs)}")
        
    except ValueError as e:
        logger.warning(f"Could not create sequences: {e}")
    
    # 6. Visualize the hurricane track
    logger.info("\n6. Creating visualizations...")
    
    # Plot hurricane track
    fig1 = plot_hurricane_track(
        ida_track,
        title="Hurricane Ida (2021) Track",
        show_intensity=True,
        figsize=(12, 8)
    )
    plt.savefig("ida_track.png", dpi=150, bbox_inches='tight')
    logger.info("Saved track visualization to ida_track.png")
    
    # Plot intensity over time
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Wind speed
    ax1.plot(ida_track['timestamp'], ida_track['max_wind'], 'b-', linewidth=2)
    ax1.set_ylabel('Maximum Wind (knots)')
    ax1.set_title('Hurricane Ida Intensity')
    ax1.grid(True, alpha=0.3)
    
    # Add category thresholds
    ax1.axhline(34, color='gray', linestyle=':', alpha=0.5, label='TS')
    ax1.axhline(64, color='gray', linestyle=':', alpha=0.5, label='Cat 1')
    ax1.axhline(83, color='gray', linestyle=':', alpha=0.5, label='Cat 2')
    ax1.axhline(96, color='gray', linestyle=':', alpha=0.5, label='Cat 3')
    ax1.axhline(113, color='gray', linestyle=':', alpha=0.5, label='Cat 4')
    ax1.axhline(137, color='gray', linestyle=':', alpha=0.5, label='Cat 5')
    ax1.legend(loc='upper left')
    
    # Pressure
    ax2.plot(ida_track['timestamp'], ida_track['min_pressure'], 'r-', linewidth=2)
    ax2.set_ylabel('Minimum Pressure (mb)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig("ida_intensity.png", dpi=150, bbox_inches='tight')
    logger.info("Saved intensity plot to ida_intensity.png")
    
    # 7. Analyze multiple hurricanes
    logger.info("\n7. Analyzing recent major hurricanes...")
    
    # Get all Category 4+ hurricanes from 2020-2023
    recent_years = [2020, 2021, 2022, 2023]
    major_hurricanes = []
    
    for year in recent_years:
        year_storms = hurdat2_loader.get_storms_by_year(year)
        
        for _, storm in year_storms.iterrows():
            storm_track = hurdat2_loader.get_storm(storm['storm_id'])
            max_intensity = storm_track['max_wind'].max()
            
            if max_intensity >= 113:  # Category 4+
                major_hurricanes.append({
                    'storm_id': storm['storm_id'],
                    'name': storm['name'],
                    'year': year,
                    'max_wind': max_intensity,
                    'min_pressure': storm_track['min_pressure'].min()
                })
    
    logger.info(f"Found {len(major_hurricanes)} Category 4+ hurricanes in {recent_years}")
    
    # Display summary
    if major_hurricanes:
        logger.info("\nMajor hurricanes summary:")
        for h in sorted(major_hurricanes, key=lambda x: x['max_wind'], reverse=True)[:5]:
            logger.info(
                f"  {h['name']} ({h['year']}): "
                f"{h['max_wind']} kt, {h['min_pressure']} mb"
            )
    
    # 8. Data pipeline example
    logger.info("\n8. Using the data pipeline...")
    pipeline = HurricaneDataPipeline()
    
    # Load complete hurricane data
    hurricane_data = pipeline.load_hurricane_for_training(
        storm_id=storm_id,
        source='hurdat2',
        include_era5=False  # Set to True if ERA5 data is available
    )
    
    logger.info(f"Loaded data keys: {list(hurricane_data.keys())}")
    logger.info(f"Track shape: {hurricane_data['track'].shape}")
    
    logger.info("\n✓ Basic usage example completed successfully!")
    logger.info("Check ida_track.png and ida_intensity.png for visualizations")
    
    # Close plots
    plt.close('all')


if __name__ == "__main__":
    main()
