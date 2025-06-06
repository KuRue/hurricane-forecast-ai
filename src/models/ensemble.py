"""Ensemble model for hurricane forecasting.

This module implements an ensemble system that combines predictions from
multiple models (GraphCast, Pangu-Weather, CNN-Transformer) with advanced
memory optimization for single GPU deployment.
"""

import gc
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from .base import BaseHurricaneModel, ModelConfig
from .graphcast import GraphCastHurricane
from .hurricane_cnn import HurricaneCNNTransformer
from .pangu import PanguWeatherHurricane


class HurricaneEnsembleModel(BaseHurricaneModel):
    """Ensemble model combining multiple hurricane forecasting approaches."""
    
    def __init__(
        self,
        config: ModelConfig,
        models: Optional[List[str]] = None,
        ensemble_size: int = 50,
        use_memory_optimization: bool = True
    ):
        """Initialize ensemble model.
        
        Args:
            config: Model configuration
            models: List of model names to include
            ensemble_size: Number of ensemble members
            use_memory_optimization: Whether to use memory optimization
        """
        super().__init__(config)
        
        self.ensemble_size = ensemble_size
        self.use_memory_optimization = use_memory_optimization
        
        # Default models
        if models is None:
            models = ["hurricane_cnn", "graphcast", "pangu"]
        self.model_names = models
        
        # Initialize component models
        self._init_models()
        
        # Ensemble combination weights (learnable)
        self.model_weights = nn.Parameter(
            torch.ones(len(self.models)) / len(self.models)
        )
        
        # Uncertainty estimation network
        self.uncertainty_net = nn.Sequential(
            nn.Linear(len(self.models) * 4, 128),  # 4 features per model
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # uncertainty for track and intensity
        )
        
        # Perturbation generator for ensemble members
        self.perturbation_scale = 0.01
        
        logger.info(
            f"Initialized HurricaneEnsembleModel with {len(self.models)} models "
            f"and {self.ensemble_size} ensemble members"
        )
    
    def _init_models(self):
        """Initialize component models."""
        self.models = nn.ModuleDict()
        
        for model_name in self.model_names:
            if model_name == "hurricane_cnn":
                self.models[model_name] = HurricaneCNNTransformer(self.config)
            elif model_name == "graphcast":
                self.models[model_name] = GraphCastHurricane(self.config)
            elif model_name == "pangu":
                self.models[model_name] = PanguWeatherHurricane(self.config)
            else:
                logger.warning(f"Unknown model: {model_name}")
    
    def generate_perturbations(
        self,
        initial_conditions: torch.Tensor,
        num_perturbations: int
    ) -> torch.Tensor:
        """Generate initial condition perturbations for ensemble.
        
        Args:
            initial_conditions: Base initial conditions
            num_perturbations: Number of perturbations to generate
            
        Returns:
            Perturbed initial conditions
        """
        batch_size, seq_len, features = initial_conditions.shape
        device = initial_conditions.device
        
        # Generate perturbations
        perturbations = []
        
        for i in range(num_perturbations):
            # Different perturbation strategies
            if i < num_perturbations // 3:
                # Gaussian noise
                noise = torch.randn_like(initial_conditions) * self.perturbation_scale
                perturbed = initial_conditions + noise
                
            elif i < 2 * num_perturbations // 3:
                # Bred vector perturbations (simplified)
                # In practice, this would use bred vectors from previous forecasts
                direction = torch.randn_like(initial_conditions)
                direction = direction / torch.norm(direction, dim=-1, keepdim=True)
                magnitude = torch.rand(batch_size, seq_len, 1).to(device) * self.perturbation_scale
                perturbed = initial_conditions + direction * magnitude
                
            else:
                # Physics-based perturbations
                # Perturb specific variables more (e.g., pressure, wind)
                noise = torch.zeros_like(initial_conditions)
                # Assume first two features are lat/lon, next two are wind/pressure
                noise[..., 2:4] = torch.randn(batch_size, seq_len, 2).to(device) * self.perturbation_scale * 2
                perturbed = initial_conditions + noise
            
            perturbations.append(perturbed)
        
        return torch.stack(perturbations, dim=1)  # (batch, num_pert, seq, features)
    
    def forward(
        self,
        inputs: torch.Tensor,
        reanalysis_data: Optional[torch.Tensor] = None,
        return_all_members: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble.
        
        Args:
            inputs: Input features
            reanalysis_data: Optional reanalysis data
            return_all_members: Whether to return all member predictions
            
        Returns:
            Dictionary with ensemble predictions
        """
        batch_size = inputs.shape[0]
        device = inputs.device
        
        # Generate ensemble members
        if self.training:
            # During training, use smaller ensemble
            num_members = min(10, self.ensemble_size)
        else:
            num_members = self.ensemble_size
        
        # Memory-optimized ensemble generation
        if self.use_memory_optimization:
            ensemble_outputs = self._memory_optimized_ensemble(
                inputs, reanalysis_data, num_members
            )
        else:
            ensemble_outputs = self._standard_ensemble(
                inputs, reanalysis_data, num_members
            )
        
        # Combine ensemble predictions
        combined_output = self._combine_ensemble(ensemble_outputs)
        
        # Estimate uncertainty
        uncertainty = self._estimate_uncertainty(ensemble_outputs)
        combined_output['uncertainty'] = uncertainty
        
        if return_all_members:
            combined_output['all_members'] = ensemble_outputs
        
        return combined_output
    
    def _memory_optimized_ensemble(
        self,
        inputs: torch.Tensor,
        reanalysis_data: Optional[torch.Tensor],
        num_members: int
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate ensemble with memory optimization."""
        ensemble_outputs = []
        
        # Process in smaller batches to save memory
        batch_size = 8  # Process 8 members at a time
        
        # Generate perturbations
        perturbations = self.generate_perturbations(inputs, num_members)
        
        for i in range(0, num_members, batch_size):
            batch_end = min(i + batch_size, num_members)
            batch_perturbations = perturbations[:, i:batch_end]
            
            # Run models on batch
            batch_outputs = []
            
            for j in range(batch_perturbations.shape[1]):
                member_input = batch_perturbations[:, j]
                member_outputs = {}
                
                # Run each model
                for model_name, model in self.models.items():
                    # Use gradient checkpointing if available
                    if hasattr(model, 'gradient_checkpointing') and self.training:
                        model.gradient_checkpointing = True
                    
                    # Get model output
                    with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                        if model_name == "hurricane_cnn":
                            output = model(member_input, reanalysis_data)
                        else:
                            output = model(member_input)
                    
                    # Store only essential outputs
                    member_outputs[model_name] = {
                        'track': output.get('track').detach(),
                        'wind': output.get('wind', torch.zeros_like(output['track'][..., 0])).detach(),
                        'pressure': output.get('pressure', torch.zeros_like(output['track'][..., 0])).detach()
                    }
                
                batch_outputs.append(member_outputs)
            
            # Move to CPU to free GPU memory
            for output in batch_outputs:
                for model_name in output:
                    for key in output[model_name]:
                        output[model_name][key] = output[model_name][key].cpu()
            
            ensemble_outputs.extend(batch_outputs)
            
            # Clear GPU cache
            if self.use_memory_optimization:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Move back to GPU for final processing
        for output in ensemble_outputs:
            for model_name in output:
                for key in output[model_name]:
                    output[model_name][key] = output[model_name][key].to(inputs.device)
        
        return ensemble_outputs
    
    def _standard_ensemble(
        self,
        inputs: torch.Tensor,
        reanalysis_data: Optional[torch.Tensor],
        num_members: int
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate ensemble without memory optimization."""
        ensemble_outputs = []
        
        # Generate all perturbations at once
        perturbations = self.generate_perturbations(inputs, num_members)
        
        for i in range(num_members):
            member_input = perturbations[:, i]
            member_outputs = {}
            
            # Run each model
            for model_name, model in self.models.items():
                if model_name == "hurricane_cnn":
                    output = model(member_input, reanalysis_data)
                else:
                    output = model(member_input)
                
                member_outputs[model_name] = output
            
            ensemble_outputs.append(member_outputs)
        
        return ensemble_outputs
    
    def _combine_ensemble(
        self,
        ensemble_outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Combine ensemble member predictions."""
        # Get device from first output
        device = ensemble_outputs[0][self.model_names[0]]['track'].device
        
        # Collect predictions by model
        model_predictions = {model: [] for model in self.model_names}
        
        for member_output in ensemble_outputs:
            for model_name in self.model_names:
                model_predictions[model_name].append(member_output[model_name])
        
        # Compute weighted model means
        weighted_tracks = []
        weighted_winds = []
        weighted_pressures = []
        
        # Normalize weights
        weights = F.softmax(self.model_weights, dim=0)
        
        for i, model_name in enumerate(self.model_names):
            # Stack predictions for this model
            model_tracks = torch.stack([
                pred['track'] for pred in model_predictions[model_name]
            ])
            model_winds = torch.stack([
                pred['wind'] for pred in model_predictions[model_name]
            ])
            model_pressures = torch.stack([
                pred['pressure'] for pred in model_predictions[model_name]
            ])
            
            # Compute model ensemble mean
            model_track_mean = model_tracks.mean(dim=0)
            model_wind_mean = model_winds.mean(dim=0)
            model_pressure_mean = model_pressures.mean(dim=0)
            
            # Apply model weight
            weighted_tracks.append(weights[i] * model_track_mean)
            weighted_winds.append(weights[i] * model_wind_mean)
            weighted_pressures.append(weights[i] * model_pressure_mean)
        
        # Combine weighted predictions
        combined_track = torch.stack(weighted_tracks).sum(dim=0)
        combined_wind = torch.stack(weighted_winds).sum(dim=0)
        combined_pressure = torch.stack(weighted_pressures).sum(dim=0)
        
        # Also compute ensemble statistics
        all_tracks = []
        all_winds = []
        all_pressures = []
        
        for member_output in ensemble_outputs:
            # Average across models for each member
            member_track = torch.stack([
                member_output[model]['track'] for model in self.model_names
            ]).mean(dim=0)
            member_wind = torch.stack([
                member_output[model]['wind'] for model in self.model_names
            ]).mean(dim=0)
            member_pressure = torch.stack([
                member_output[model]['pressure'] for model in self.model_names
            ]).mean(dim=0)
            
            all_tracks.append(member_track)
            all_winds.append(member_wind)
            all_pressures.append(member_pressure)
        
        all_tracks = torch.stack(all_tracks)
        all_winds = torch.stack(all_winds)
        all_pressures = torch.stack(all_pressures)
        
        # Compute percentiles for uncertainty bounds
        track_percentiles = torch.quantile(
            all_tracks,
            torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]).to(device),
            dim=0
        )
        
        return {
            'track': combined_track,
            'wind': combined_wind,
            'pressure': combined_pressure,
            'track_mean': all_tracks.mean(dim=0),
            'track_std': all_tracks.std(dim=0),
            'track_percentiles': track_percentiles,
            'wind_mean': all_winds.mean(dim=0),
            'wind_std': all_winds.std(dim=0),
            'pressure_mean': all_pressures.mean(dim=0),
            'pressure_std': all_pressures.std(dim=0),
            'model_weights': weights
        }
    
    def _estimate_uncertainty(
        self,
        ensemble_outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Estimate prediction uncertainty from ensemble spread."""
        # Compute ensemble statistics for uncertainty estimation
        track_spreads = []
        intensity_spreads = []
        
        for model_name in self.model_names:
            model_tracks = torch.stack([
                output[model_name]['track'] 
                for output in ensemble_outputs
            ])
            model_winds = torch.stack([
                output[model_name]['wind'] 
                for output in ensemble_outputs
            ])
            
            # Compute spread (standard deviation)
            track_spread = model_tracks.std(dim=0).mean(dim=-1)  # Average lat/lon spread
            wind_spread = model_winds.std(dim=0)
            
            track_spreads.append(track_spread)
            intensity_spreads.append(wind_spread)
        
        # Stack spreads
        track_spread_features = torch.stack(track_spreads, dim=-1)
        intensity_spread_features = torch.stack(intensity_spreads, dim=-1)
        
        # Combine features
        uncertainty_features = torch.cat([
            track_spread_features,
            intensity_spread_features,
            track_spread_features.mean(dim=-1, keepdim=True),
            intensity_spread_features.mean(dim=-1, keepdim=True)
        ], dim=-1)
        
        # Estimate uncertainty through neural network
        uncertainty = self.uncertainty_net(uncertainty_features)
        
        return {
            'track_uncertainty': torch.exp(uncertainty[..., 0]),  # Exponential to ensure positive
            'intensity_uncertainty': torch.exp(uncertainty[..., 1]),
            'ensemble_spread': track_spread_features.mean(dim=-1)
        }
    
    def calibrate_uncertainty(
        self,
        predictions: torch.Tensor,
        observations: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """Calibrate uncertainty estimates using observations.
        
        Args:
            predictions: Ensemble predictions
            observations: Ground truth observations
            uncertainties: Predicted uncertainties
            
        Returns:
            Calibration loss
        """
        # Compute actual errors
        errors = torch.norm(predictions - observations, dim=-1)
        
        # Negative log-likelihood assuming Gaussian distribution
        nll = torch.log(uncertainties + 1e-6) + \
              errors ** 2 / (2 * uncertainties ** 2 + 1e-6)
        
        # Add calibration term to encourage well-calibrated uncertainties
        # Uncertainty should match root mean square error
        rmse = torch.sqrt((errors ** 2).mean())
        calibration_loss = F.mse_loss(uncertainties.mean(), rmse)
        
        return nll.mean() + 0.1 * calibration_loss
    
    def get_confidence_bounds(
        self,
        predictions: Dict[str, torch.Tensor],
        confidence_level: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """Get confidence bounds for predictions.
        
        Args:
            predictions: Ensemble predictions
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary with confidence bounds
        """
        # Calculate percentiles for confidence bounds
        alpha = (1 - confidence_level) / 2
        lower_percentile = alpha
        upper_percentile = 1 - alpha
        
        if 'track_percentiles' in predictions:
            percentiles = predictions['track_percentiles']
            # Interpolate to get exact confidence bounds
            lower_idx = int(lower_percentile * (len(percentiles) - 1))
            upper_idx = int(upper_percentile * (len(percentiles) - 1))
            
            return {
                'track_lower': percentiles[lower_idx],
                'track_upper': percentiles[upper_idx],
                'track_median': percentiles[len(percentiles) // 2]
            }
        else:
            # Use Gaussian assumption
            track_mean = predictions['track_mean']
            track_std = predictions['track_std']
            
            z_score = 1.96 if confidence_level == 0.95 else 2.58
            
            return {
                'track_lower': track_mean - z_score * track_std,
                'track_upper': track_mean + z_score * track_std,
                'track_median': track_mean
            }
    
    def adaptive_ensemble_size(
        self,
        forecast_difficulty: float
    ) -> int:
        """Adaptively determine ensemble size based on forecast difficulty.
        
        Args:
            forecast_difficulty: Estimated difficulty (0-1)
            
        Returns:
            Recommended ensemble size
        """
        # More members for difficult forecasts
        min_size = 20
        max_size = self.ensemble_size
        
        adaptive_size = int(min_size + (max_size - min_size) * forecast_difficulty)
        
        # Round to nearest 10 for efficiency
        return (adaptive_size // 10) * 10
