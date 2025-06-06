"""Visualization utilities for hurricane forecasting."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from IPython.display import HTML
import warnings

from ..utils.config import get_config


# Saffir-Simpson scale colors
SAFFIR_SIMPSON_COLORS = {
    "TD": "#5EBAFF",     # Tropical Depression (blue)
    "TS": "#00FAF4",     # Tropical Storm (cyan)
    "C1": "#FFFFCC",     # Category 1 (light yellow)
    "C2": "#FFE775",     # Category 2 (yellow)
    "C3": "#FFC140",     # Category 3 (orange)
    "C4": "#FF8F20",     # Category 4 (dark orange)
    "C5": "#FF6060",     # Category 5 (red)
}


def get_storm_category(wind_speed: float) -> str:
    """Get storm category from wind speed.
    
    Args:
        wind_speed: Maximum sustained wind speed in knots
        
    Returns:
        Storm category string
    """
    if wind_speed < 34:
        return "TD"
    elif wind_speed < 64:
        return "TS"
    elif wind_speed < 83:
        return "C1"
    elif wind_speed < 96:
        return "C2"
    elif wind_speed < 113:
        return "C3"
    elif wind_speed < 137:
        return "C4"
    else:
        return "C5"


def plot_hurricane_track(
    track_df: pd.DataFrame,
    title: Optional[str] = None,
    show_intensity: bool = True,
    extent: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot hurricane track on a map.
    
    Args:
        track_df: Hurricane track DataFrame
        title: Plot title
        show_intensity: Whether to color by intensity
        extent: Map extent [west, east, south, north]
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, alpha=0.5)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, alpha=0.5)
    ax.add_feature(cfeature.STATES, alpha=0.3)
    
    # Set extent
    if extent is None:
        # Auto extent with padding
        lat_range = track_df['latitude'].max() - track_df['latitude'].min()
        lon_range = track_df['longitude'].max() - track_df['longitude'].min()
        padding = max(lat_range, lon_range) * 0.2
        
        extent = [
            track_df['longitude'].min() - padding,
            track_df['longitude'].max() + padding,
            track_df['latitude'].min() - padding,
            track_df['latitude'].max() + padding
        ]
    
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    if show_intensity and 'max_wind' in track_df.columns:
        # Plot track colored by intensity
        for i in range(len(track_df) - 1):
            start_point = track_df.iloc[i]
            end_point = track_df.iloc[i + 1]
            
            # Get color based on intensity
            wind_speed = start_point['max_wind']
            category = get_storm_category(wind_speed)
            color = SAFFIR_SIMPSON_COLORS[category]
            
            ax.plot(
                [start_point['longitude'], end_point['longitude']],
                [start_point['latitude'], end_point['latitude']],
                color=color,
                linewidth=3,
                transform=ccrs.PlateCarree(),
                alpha=0.8
            )
        
        # Add points
        for _, point in track_df.iterrows():
            category = get_storm_category(point['max_wind'])
            ax.scatter(
                point['longitude'],
                point['latitude'],
                color=SAFFIR_SIMPSON_COLORS[category],
                s=50,
                edgecolor='black',
                linewidth=0.5,
                transform=ccrs.PlateCarree(),
                zorder=10
            )
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=SAFFIR_SIMPSON_COLORS["TD"], label='TD (<34 kt)'),
            Patch(facecolor=SAFFIR_SIMPSON_COLORS["TS"], label='TS (34-63 kt)'),
            Patch(facecolor=SAFFIR_SIMPSON_COLORS["C1"], label='Cat 1 (64-82 kt)'),
            Patch(facecolor=SAFFIR_SIMPSON_COLORS["C2"], label='Cat 2 (83-95 kt)'),
            Patch(facecolor=SAFFIR_SIMPSON_COLORS["C3"], label='Cat 3 (96-112 kt)'),
            Patch(facecolor=SAFFIR_SIMPSON_COLORS["C4"], label='Cat 4 (113-136 kt)'),
            Patch(facecolor=SAFFIR_SIMPSON_COLORS["C5"], label='Cat 5 (≥137 kt)')
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=8)
    else:
        # Simple track
        ax.plot(
            track_df['longitude'],
            track_df['latitude'],
            'r-',
            linewidth=2,
            transform=ccrs.PlateCarree()
        )
        ax.scatter(
            track_df['longitude'],
            track_df['latitude'],
            c='red',
            s=30,
            transform=ccrs.PlateCarree()
        )
    
    # Add title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_intensity_forecast(
    observed: pd.DataFrame,
    forecast: pd.DataFrame,
    models: Optional[Dict[str, pd.DataFrame]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot intensity forecast comparison.
    
    Args:
        observed: Observed track with intensity
        forecast: Forecast track with intensity
        models: Dictionary of other model forecasts
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot wind speed
    ax1.plot(
        observed['timestamp'],
        observed['max_wind'],
        'k-',
        linewidth=2,
        label='Observed'
    )
    ax1.plot(
        forecast['timestamp'],
        forecast['max_wind'],
        'r--',
        linewidth=2,
        label='Forecast'
    )
    
    if models:
        for name, model_forecast in models.items():
            ax1.plot(
                model_forecast['timestamp'],
                model_forecast['max_wind'],
                '--',
                linewidth=1.5,
                label=name,
                alpha=0.7
            )
    
    # Add category thresholds
    ax1.axhline(34, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(64, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(83, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(96, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(113, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(137, color='gray', linestyle=':', alpha=0.5)
    
    ax1.set_ylabel('Maximum Wind (knots)', fontsize=12)
    ax1.set_title('Wind Speed Forecast', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot pressure
    if 'min_pressure' in observed.columns:
        ax2.plot(
            observed['timestamp'],
            observed['min_pressure'],
            'k-',
            linewidth=2,
            label='Observed'
        )
        
        if 'min_pressure' in forecast.columns:
            ax2.plot(
                forecast['timestamp'],
                forecast['min_pressure'],
                'r--',
                linewidth=2,
                label='Forecast'
            )
        
        ax2.set_ylabel('Minimum Pressure (mb)', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_title('Pressure Forecast', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()  # Lower pressure = stronger storm
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_forecast_animation(
    track_history: pd.DataFrame,
    forecast_tracks: List[pd.DataFrame],
    interval: int = 200,
    save_path: Optional[str] = None
) -> animation.FuncAnimation:
    """Create animated forecast visualization.
    
    Args:
        track_history: Historical track data
        forecast_tracks: List of forecast tracks (ensemble members)
        interval: Animation interval in milliseconds
        save_path: Path to save animation
        
    Returns:
        Matplotlib animation
    """
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, alpha=0.5)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, alpha=0.5)
    
    # Set extent
    all_lons = [track_history['longitude'].values]
    all_lats = [track_history['latitude'].values]
    
    for forecast in forecast_tracks:
        all_lons.append(forecast['longitude'].values)
        all_lats.append(forecast['latitude'].values)
    
    all_lons = np.concatenate(all_lons)
    all_lats = np.concatenate(all_lats)
    
    extent = [
        all_lons.min() - 5,
        all_lons.max() + 5,
        all_lats.min() - 5,
        all_lats.max() + 5
    ]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Initialize plot elements
    history_line, = ax.plot([], [], 'k-', linewidth=2, transform=ccrs.PlateCarree())
    history_point, = ax.plot([], [], 'ko', markersize=8, transform=ccrs.PlateCarree())
    
    forecast_lines = []
    for _ in forecast_tracks:
        line, = ax.plot([], [], 'r-', alpha=0.3, linewidth=1, transform=ccrs.PlateCarree())
        forecast_lines.append(line)
    
    # Animation function
    def animate(frame):
        # Update history
        if frame < len(track_history):
            history_line.set_data(
                track_history['longitude'][:frame+1],
                track_history['latitude'][:frame+1]
            )
            history_point.set_data(
                [track_history['longitude'].iloc[frame]],
                [track_history['latitude'].iloc[frame]]
            )
        
        # Update forecasts
        forecast_start = max(0, frame - len(track_history) + 1)
        for i, (line, forecast) in enumerate(zip(forecast_lines, forecast_tracks)):
            if forecast_start > 0 and forecast_start < len(forecast):
                line.set_data(
                    forecast['longitude'][:forecast_start+1],
                    forecast['latitude'][:forecast_start+1]
                )
        
        return [history_line, history_point] + forecast_lines
    
    # Create animation
    total_frames = len(track_history) + max(len(f) for f in forecast_tracks)
    anim = animation.FuncAnimation(
        fig, animate,
        frames=total_frames,
        interval=interval,
        blit=True
    )
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=5)
        else:
            anim.save(save_path, writer='ffmpeg', fps=5)
    
    return anim


def plot_era5_field(
    era5_data: xr.Dataset,
    variable: str,
    time_idx: int = 0,
    level: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'RdBu_r',
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot ERA5 atmospheric field.
    
    Args:
        era5_data: ERA5 dataset
        variable: Variable to plot
        time_idx: Time index to plot
        level: Pressure level (if applicable)
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Select data
    data = era5_data[variable].isel(time=time_idx)
    if level is not None and 'level' in data.dims:
        data = data.sel(level=level, method='nearest')
    
    # Plot data
    im = ax.contourf(
        era5_data.longitude,
        era5_data.latitude,
        data,
        levels=20,
        cmap=cmap,
        transform=ccrs.PlateCarree()
    )
    
    # Add contour lines
    cs = ax.contour(
        era5_data.longitude,
        era5_data.latitude,
        data,
        levels=10,
        colors='black',
        linewidths=0.5,
        alpha=0.5,
        transform=ccrs.PlateCarree()
    )
    
    # Add features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, alpha=0.5)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
    cbar.set_label(f'{variable}', rotation=270, labelpad=20)
    
    # Title
    time_str = pd.Timestamp(era5_data.time[time_idx].values).strftime('%Y-%m-%d %H:%M')
    title = f'{variable} at {time_str}'
    if level is not None:
        title += f' ({level} hPa)'
    ax.set_title(title, fontsize=14)
    
    # Grid
    ax.gridlines(draw_labels=True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_forecast_ensemble(
    ensemble_tracks: List[pd.DataFrame],
    observed: Optional[pd.DataFrame] = None,
    show_spread: bool = True,
    percentiles: List[int] = [10, 25, 75, 90],
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot ensemble forecast with uncertainty.
    
    Args:
        ensemble_tracks: List of ensemble member tracks
        observed: Observed track (if available)
        show_spread: Whether to show ensemble spread
        percentiles: Percentiles for uncertainty bands
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, alpha=0.5)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, alpha=0.5)
    ax.add_feature(cfeature.STATES, alpha=0.3)
    
    # Plot ensemble members
    for i, track in enumerate(ensemble_tracks):
        ax.plot(
            track['longitude'],
            track['latitude'],
            'gray',
            alpha=0.2,
            linewidth=0.5,
            transform=ccrs.PlateCarree()
        )
    
    # Calculate ensemble statistics
    forecast_times = ensemble_tracks[0]['timestamp'].unique()
    mean_lons = []
    mean_lats = []
    
    percentile_data = {p: {'lons': [], 'lats': []} for p in percentiles}
    
    for time in forecast_times:
        lons = []
        lats = []
        
        for track in ensemble_tracks:
            time_data = track[track['timestamp'] == time]
            if len(time_data) > 0:
                lons.append(time_data['longitude'].iloc[0])
                lats.append(time_data['latitude'].iloc[0])
        
        if lons:
            mean_lons.append(np.mean(lons))
            mean_lats.append(np.mean(lats))
            
            # Calculate percentiles
            for p in percentiles:
                percentile_data[p]['lons'].append(np.percentile(lons, p))
                percentile_data[p]['lats'].append(np.percentile(lats, p))
    
    # Plot mean track
    ax.plot(
        mean_lons,
        mean_lats,
        'r-',
        linewidth=3,
        label='Ensemble Mean',
        transform=ccrs.PlateCarree()
    )
    
    # Plot uncertainty bands
    if show_spread and len(percentiles) >= 4:
        from matplotlib.patches import Polygon
        
        # Create polygon for each time step
        for i in range(len(mean_lons)):
            if i < len(percentile_data[percentiles[0]]['lons']):
                # Inner band (25-75 percentile)
                inner_poly = Polygon([
                    (percentile_data[percentiles[1]]['lons'][i], percentile_data[percentiles[1]]['lats'][i]),
                    (percentile_data[percentiles[2]]['lons'][i], percentile_data[percentiles[1]]['lats'][i]),
                    (percentile_data[percentiles[2]]['lons'][i], percentile_data[percentiles[2]]['lats'][i]),
                    (percentile_data[percentiles[1]]['lons'][i], percentile_data[percentiles[2]]['lats'][i])
                ], alpha=0.3, facecolor='red', transform=ccrs.PlateCarree())
                ax.add_patch(inner_poly)
    
    # Plot observed track
    if observed is not None:
        ax.plot(
            observed['longitude'],
            observed['latitude'],
            'k-',
            linewidth=2,
            label='Observed',
            transform=ccrs.PlateCarree()
        )
    
    # Set extent
    all_lons = mean_lons.copy()
    all_lats = mean_lats.copy()
    
    if observed is not None:
        all_lons.extend(observed['longitude'].values)
        all_lats.extend(observed['latitude'].values)
    
    extent = [
        min(all_lons) - 5,
        max(all_lons) + 5,
        min(all_lats) - 5,
        max(all_lats) + 5
    ]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add legend and title
    ax.legend(loc='best')
    ax.set_title('Ensemble Hurricane Forecast', fontsize=16, fontweight='bold')
    
    # Grid
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
