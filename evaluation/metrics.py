"""Evaluation metrics for hurricane forecasting models."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from loguru import logger


def haversine_distance(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Calculate haversine distance between coordinates.
    
    Args:
        lat1, lon1: First coordinates
        lat2, lon2: Second coordinates
        
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    
    # Earth's radius in km
    R = 6371.0
    return R * c


def calculate_track_errors(
    predictions: np.ndarray,
    observations: np.ndarray,
    forecast_hours: Optional[List[int]] = None
) -> Dict[str, Union[float, np.ndarray]]:
    """Calculate track forecast errors.
    
    Args:
        predictions: Predicted tracks (samples, time, 2)
        observations: Observed tracks (samples, time, 2)
        forecast_hours: Specific forecast hours to evaluate
        
    Returns:
        Dictionary with error metrics
    """
    if predictions.shape != observations.shape:
        raise ValueError("Predictions and observations must have same shape")
    
    # Extract coordinates
    pred_lat = predictions[..., 0]
    pred_lon = predictions[..., 1]
    obs_lat = observations[..., 0]
    obs_lon = observations[..., 1]
    
    # Calculate track errors
    track_errors = haversine_distance(pred_lat, pred_lon, obs_lat, obs_lon)
    
    # Convert to nautical miles
    track_errors_nm = track_errors * 0.539957
    
    # Calculate statistics
    results = {
        'mean_track_error_km': np.mean(track_errors),
        'mean_track_error_nm': np.mean(track_errors_nm),
        'median_track_error_km': np.median(track_errors),
        'median_track_error_nm': np.median(track_errors_nm),
        'std_track_error_km': np.std(track_errors),
        'max_track_error_km': np.max(track_errors),
        'track_errors_by_time': np.mean(track_errors, axis=0)
    }
    
    # Calculate errors at specific forecast hours if provided
    if forecast_hours is not None:
        # Assuming 6-hourly data
        time_indices = [h // 6 for h in forecast_hours]
        
        for hour, idx in zip(forecast_hours, time_indices):
            if idx < track_errors.shape[1]:
                results[f'track_error_{hour}h_km'] = np.mean(track_errors[:, idx])
                results[f'track_error_{hour}h_nm'] = np.mean(track_errors_nm[:, idx])
    
    # Calculate along-track and cross-track errors
    along_track, cross_track = calculate_along_cross_track_errors(
        predictions, observations
    )
    
    results['mean_along_track_error_km'] = np.mean(np.abs(along_track))
    results['mean_cross_track_error_km'] = np.mean(np.abs(cross_track))
    
    return results


def calculate_along_cross_track_errors(
    predictions: np.ndarray,
    observations: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate along-track and cross-track errors.
    
    Args:
        predictions: Predicted tracks
        observations: Observed tracks
        
    Returns:
        Tuple of (along_track_errors, cross_track_errors)
    """
    # Calculate storm motion vector
    obs_motion = observations[:, 1:] - observations[:, :-1]
    
    # Calculate error vector
    errors = predictions[:, :-1] - observations[:, :-1]
    
    # Project error onto motion vector (along-track)
    # and perpendicular to motion (cross-track)
    along_track = []
    cross_track = []
    
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            error = errors[i, j]
            motion = obs_motion[i, j]
            
            # Skip if no motion
            motion_mag = np.linalg.norm(motion)
            if motion_mag < 1e-6:
                along_track.append(0)
                cross_track.append(np.linalg.norm(error))
                continue
            
            # Normalize motion vector
            motion_unit = motion / motion_mag
            
            # Project error
            along = np.dot(error, motion_unit)
            cross = np.linalg.norm(error - along * motion_unit)
            
            along_track.append(along)
            cross_track.append(cross)
    
    along_track = np.array(along_track).reshape(errors.shape[0], errors.shape[1])
    cross_track = np.array(cross_track).reshape(errors.shape[0], errors.shape[1])
    
    # Convert to km
    along_track *= 111.0  # Approximate km per degree
    cross_track *= 111.0
    
    return along_track, cross_track


def calculate_intensity_errors(
    pred_wind: np.ndarray,
    obs_wind: np.ndarray,
    pred_pressure: Optional[np.ndarray] = None,
    obs_pressure: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate intensity forecast errors.
    
    Args:
        pred_wind: Predicted wind speeds (knots)
        obs_wind: Observed wind speeds
        pred_pressure: Predicted pressures (mb)
        obs_pressure: Observed pressures
        
    Returns:
        Dictionary with intensity error metrics
    """
    results = {}
    
    # Wind speed errors
    wind_mae = mean_absolute_error(obs_wind.flatten(), pred_wind.flatten())
    wind_rmse = np.sqrt(mean_squared_error(obs_wind.flatten(), pred_wind.flatten()))
    wind_bias = np.mean(pred_wind - obs_wind)
    
    results.update({
        'wind_mae_kt': wind_mae,
        'wind_rmse_kt': wind_rmse,
        'wind_bias_kt': wind_bias,
        'wind_relative_error': wind_mae / np.mean(obs_wind)
    })
    
    # Pressure errors if provided
    if pred_pressure is not None and obs_pressure is not None:
        pressure_mae = mean_absolute_error(obs_pressure.flatten(), pred_pressure.flatten())
        pressure_rmse = np.sqrt(mean_squared_error(obs_pressure.flatten(), pred_pressure.flatten()))
        pressure_bias = np.mean(pred_pressure - obs_pressure)
        
        results.update({
            'pressure_mae_mb': pressure_mae,
            'pressure_rmse_mb': pressure_rmse,
            'pressure_bias_mb': pressure_bias
        })
    
    # Category accuracy
    pred_categories = wind_to_category(pred_wind)
    obs_categories = wind_to_category(obs_wind)
    
    category_accuracy = np.mean(pred_categories == obs_categories)
    results['category_accuracy'] = category_accuracy
    
    # Rapid intensification detection
    if len(pred_wind.shape) > 1:  # Time series
        pred_ri = detect_rapid_intensification(pred_wind)
        obs_ri = detect_rapid_intensification(obs_wind)
        
        ri_pod = probability_of_detection(pred_ri, obs_ri)
        ri_far = false_alarm_ratio(pred_ri, obs_ri)
        
        results.update({
            'ri_pod': ri_pod,
            'ri_far': ri_far,
            'ri_csi': critical_success_index(pred_ri, obs_ri)
        })
    
    return results


def wind_to_category(wind_speed: np.ndarray) -> np.ndarray:
    """Convert wind speed to Saffir-Simpson category.
    
    Args:
        wind_speed: Wind speed in knots
        
    Returns:
        Category array (0=TD/TS, 1-5=Hurricane categories)
    """
    categories = np.zeros_like(wind_speed, dtype=int)
    
    categories[wind_speed >= 34] = 0  # TS
    categories[wind_speed >= 64] = 1  # Cat 1
    categories[wind_speed >= 83] = 2  # Cat 2
    categories[wind_speed >= 96] = 3  # Cat 3
    categories[wind_speed >= 113] = 4  # Cat 4
    categories[wind_speed >= 137] = 5  # Cat 5
    
    return categories


def detect_rapid_intensification(wind_speed: np.ndarray) -> np.ndarray:
    """Detect rapid intensification events.
    
    Rapid intensification is defined as 30+ knot increase in 24 hours.
    
    Args:
        wind_speed: Wind speed time series
        
    Returns:
        Boolean array indicating RI events
    """
    if len(wind_speed.shape) == 1:
        wind_speed = wind_speed.reshape(1, -1)
    
    # Calculate 24-hour changes (assuming 6-hourly data)
    if wind_speed.shape[1] >= 5:
        changes_24h = wind_speed[:, 4:] - wind_speed[:, :-4]
        ri_events = changes_24h >= 30
        
        # Pad to match original length
        ri_full = np.zeros_like(wind_speed, dtype=bool)
        ri_full[:, 4:] = ri_events
    else:
        ri_full = np.zeros_like(wind_speed, dtype=bool)
    
    return ri_full


def probability_of_detection(predictions: np.ndarray, observations: np.ndarray) -> float:
    """Calculate probability of detection (POD).
    
    POD = hits / (hits + misses)
    
    Args:
        predictions: Binary predictions
        observations: Binary observations
        
    Returns:
        POD score
    """
    hits = np.sum(predictions & observations)
    misses = np.sum(~predictions & observations)
    
    if hits + misses == 0:
        return 0.0
    
    return hits / (hits + misses)


def false_alarm_ratio(predictions: np.ndarray, observations: np.ndarray) -> float:
    """Calculate false alarm ratio (FAR).
    
    FAR = false_alarms / (hits + false_alarms)
    
    Args:
        predictions: Binary predictions
        observations: Binary observations
        
    Returns:
        FAR score
    """
    hits = np.sum(predictions & observations)
    false_alarms = np.sum(predictions & ~observations)
    
    if hits + false_alarms == 0:
        return 0.0
    
    return false_alarms / (hits + false_alarms)


def critical_success_index(predictions: np.ndarray, observations: np.ndarray) -> float:
    """Calculate critical success index (CSI).
    
    CSI = hits / (hits + misses + false_alarms)
    
    Args:
        predictions: Binary predictions
        observations: Binary observations
        
    Returns:
        CSI score
    """
    hits = np.sum(predictions & observations)
    misses = np.sum(~predictions & observations)
    false_alarms = np.sum(predictions & ~observations)
    
    if hits + misses + false_alarms == 0:
        return 0.0
    
    return hits / (hits + misses + false_alarms)


def calculate_skill_score(
    model_errors: np.ndarray,
    baseline_errors: np.ndarray
) -> float:
    """Calculate skill score relative to baseline.
    
    Skill = 1 - (model_error / baseline_error)
    
    Args:
        model_errors: Model forecast errors
        baseline_errors: Baseline forecast errors
        
    Returns:
        Skill score
    """
    model_mae = np.mean(np.abs(model_errors))
    baseline_mae = np.mean(np.abs(baseline_errors))
    
    if baseline_mae == 0:
        return 0.0
    
    return 1 - (model_mae / baseline_mae)


def calculate_cliper_baseline(
    initial_position: np.ndarray,
    initial_motion: np.ndarray,
    forecast_hours: int
) -> np.ndarray:
    """Calculate CLIPER baseline forecast.
    
    CLIPER (CLImatology and PERsistence) is a simple baseline
    that extrapolates current motion.
    
    Args:
        initial_position: Initial position (lat, lon)
        initial_motion: Initial motion vector (dlat/dt, dlon/dt)
        forecast_hours: Forecast length in hours
        
    Returns:
        Forecast track
    """
    # Create time steps (6-hourly)
    time_steps = forecast_hours // 6
    
    # Linear extrapolation
    forecast = np.zeros((time_steps, 2))
    
    for t in range(time_steps):
        hours = (t + 1) * 6
        forecast[t] = initial_position + initial_motion * hours
    
    return forecast


def evaluate_ensemble_forecast(
    ensemble_predictions: List[np.ndarray],
    observations: np.ndarray,
    confidence_levels: List[float] = [0.5, 0.9]
) -> Dict[str, float]:
    """Evaluate ensemble forecast performance.
    
    Args:
        ensemble_predictions: List of ensemble member predictions
        observations: Observed values
        confidence_levels: Confidence levels to evaluate
        
    Returns:
        Dictionary with ensemble metrics
    """
    ensemble_array = np.stack(ensemble_predictions)
    
    # Ensemble mean
    ensemble_mean = np.mean(ensemble_array, axis=0)
    
    # Track errors for ensemble mean
    mean_errors = calculate_track_errors(ensemble_mean, observations)
    
    results = {f'ensemble_mean_{k}': v for k, v in mean_errors.items()}
    
    # Ensemble spread
    ensemble_std = np.std(ensemble_array, axis=0)
    results['ensemble_spread_mean'] = np.mean(ensemble_std)
    
    # Reliability - is spread calibrated to error?
    actual_errors = haversine_distance(
        ensemble_mean[..., 0], ensemble_mean[..., 1],
        observations[..., 0], observations[..., 1]
    )
    
    # Spread-error correlation
    spread_error_corr = np.corrcoef(
        ensemble_std[..., 0].flatten(),
        actual_errors.flatten()
    )[0, 1]
    
    results['spread_error_correlation'] = spread_error_corr
    
    # Rank histogram (reliability)
    ranks = []
    for i in range(observations.shape[0]):
        for j in range(observations.shape[1]):
            obs = observations[i, j, 0]  # Latitude
            ensemble_vals = ensemble_array[:, i, j, 0]
            rank = np.sum(ensemble_vals <= obs)
            ranks.append(rank)
    
    # Chi-square test for uniformity
    rank_counts, _ = np.histogram(ranks, bins=len(ensemble_predictions) + 1)
    expected_count = len(ranks) / (len(ensemble_predictions) + 1)
    chi2_stat = np.sum((rank_counts - expected_count)**2 / expected_count)
    
    results['rank_histogram_chi2'] = chi2_stat
    
    # Coverage for confidence intervals
    for conf_level in confidence_levels:
        lower_percentile = (1 - conf_level) / 2
        upper_percentile = 1 - lower_percentile
        
        lower_bound = np.percentile(ensemble_array, lower_percentile * 100, axis=0)
        upper_bound = np.percentile(ensemble_array, upper_percentile * 100, axis=0)
        
        # Check coverage
        in_bounds = (observations >= lower_bound) & (observations <= upper_bound)
        coverage = np.mean(in_bounds)
        
        results[f'coverage_{int(conf_level*100)}pct'] = coverage
    
    return results


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to use
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        forecast_hours: List[int] = [24, 48, 72, 96, 120]
    ) -> Dict[str, float]:
        """Evaluate model on a dataset.
        
        Args:
            dataloader: Data loader
            forecast_hours: Forecast hours to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        all_observations = []
        all_wind_pred = []
        all_wind_obs = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                inputs = batch['inputs'].to(self.device)
                targets = batch['target_track'].to(self.device)
                
                # Get predictions
                outputs = self.model(inputs)
                
                # Collect predictions and observations
                all_predictions.append(outputs['track'].cpu().numpy())
                all_observations.append(targets.cpu().numpy())
                
                if 'wind' in outputs and 'target_wind' in batch:
                    all_wind_pred.append(outputs['wind'].cpu().numpy())
                    all_wind_obs.append(batch['target_wind'].cpu().numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)
        observations = np.concatenate(all_observations, axis=0)
        
        # Calculate track errors
        track_metrics = calculate_track_errors(
            predictions, observations, forecast_hours
        )
        
        # Calculate intensity errors if available
        intensity_metrics = {}
        if all_wind_pred and all_wind_obs:
            wind_pred = np.concatenate(all_wind_pred, axis=0)
            wind_obs = np.concatenate(all_wind_obs, axis=0)
            
            intensity_metrics = calculate_intensity_errors(
                wind_pred, wind_obs
            )
        
        # Combine metrics
        metrics = {**track_metrics, **intensity_metrics}
        
        return metrics
    
    def compare_with_baselines(
        self,
        dataloader: torch.utils.data.DataLoader,
        baseline_models: Dict[str, torch.nn.Module]
    ) -> pd.DataFrame:
        """Compare model with baseline models.
        
        Args:
            dataloader: Data loader
            baseline_models: Dictionary of baseline models
            
        Returns:
            DataFrame with comparison results
        """
        results = {}
        
        # Evaluate main model
        logger.info("Evaluating main model...")
        results['Model'] = self.evaluate_dataset(dataloader)
        
        # Evaluate baselines
        for name, baseline in baseline_models.items():
            logger.info(f"Evaluating {name}...")
            evaluator = ModelEvaluator(baseline, self.device)
            results[name] = evaluator.evaluate_dataset(dataloader)
        
        # Create comparison DataFrame
        df = pd.DataFrame(results).T
        
        # Add improvement percentages
        for col in df.columns:
            if 'error' in col:
                baseline_val = df.loc['CLIPER', col] if 'CLIPER' in df.index else df.iloc[1][col]
                df[f'{col}_improvement'] = (baseline_val - df[col]) / baseline_val * 100
        
        return df
