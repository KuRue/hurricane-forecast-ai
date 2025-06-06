"""Loss functions for hurricane forecasting models.

Includes standard prediction losses and physics-informed constraints.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger


class HurricaneTrackLoss(nn.Module):
    """Loss function for hurricane track prediction."""
    
    def __init__(
        self,
        use_weighted_loss: bool = True,
        max_weight: float = 5.0
    ):
        """Initialize track loss.
        
        Args:
            use_weighted_loss: Whether to weight loss by forecast hour
            max_weight: Maximum weight for late forecast hours
        """
        super().__init__()
        self.use_weighted_loss = use_weighted_loss
        self.max_weight = max_weight
    
    def forward(
        self,
        pred_track: torch.Tensor,
        true_track: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate track loss.
        
        Args:
            pred_track: Predicted track (batch, time, 2)
            true_track: True track (batch, time, 2)
            mask: Optional mask for valid time steps
            
        Returns:
            Track loss
        """
        # Calculate haversine distance
        pred_lat, pred_lon = pred_track[..., 0], pred_track[..., 1]
        true_lat, true_lon = true_track[..., 0], true_track[..., 1]
        
        # Convert to radians
        pred_lat_rad = torch.deg2rad(pred_lat)
        true_lat_rad = torch.deg2rad(true_lat)
        dlat = torch.deg2rad(true_lat - pred_lat)
        dlon = torch.deg2rad(true_lon - pred_lon)
        
        # Haversine formula
        a = (
            torch.sin(dlat / 2) ** 2 +
            torch.cos(pred_lat_rad) * torch.cos(true_lat_rad) * 
            torch.sin(dlon / 2) ** 2
        )
        c = 2 * torch.arcsin(torch.sqrt(torch.clamp(a, 0, 1)))
        
        # Distance in km
        R = 6371.0
        distances = R * c
        
        # Apply mask if provided
        if mask is not None:
            distances = distances * mask
        
        # Apply time weighting if requested
        if self.use_weighted_loss:
            time_steps = distances.shape[1]
            weights = torch.linspace(1.0, self.max_weight, time_steps).to(distances.device)
            weights = weights.unsqueeze(0)  # Add batch dimension
            distances = distances * weights
        
        # Mean loss
        if mask is not None:
            loss = distances.sum() / mask.sum()
        else:
            loss = distances.mean()
        
        return loss


class IntensityLoss(nn.Module):
    """Loss function for intensity prediction (wind speed and pressure)."""
    
    def __init__(
        self,
        wind_weight: float = 1.0,
        pressure_weight: float = 0.5,
        use_relative_error: bool = True
    ):
        """Initialize intensity loss.
        
        Args:
            wind_weight: Weight for wind speed loss
            pressure_weight: Weight for pressure loss
            use_relative_error: Whether to use relative error
        """
        super().__init__()
        self.wind_weight = wind_weight
        self.pressure_weight = pressure_weight
        self.use_relative_error = use_relative_error
    
    def forward(
        self,
        pred_wind: torch.Tensor,
        true_wind: torch.Tensor,
        pred_pressure: Optional[torch.Tensor] = None,
        true_pressure: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate intensity loss.
        
        Args:
            pred_wind: Predicted wind speed
            true_wind: True wind speed
            pred_pressure: Predicted pressure (optional)
            true_pressure: True pressure (optional)
            mask: Optional mask
            
        Returns:
            Intensity loss
        """
        # Wind speed loss
        if self.use_relative_error:
            # Relative error (percentage)
            wind_error = torch.abs(pred_wind - true_wind) / (true_wind + 1e-6)
        else:
            # Absolute error
            wind_error = F.mse_loss(pred_wind, true_wind, reduction='none')
        
        if mask is not None:
            wind_error = wind_error * mask
            wind_loss = wind_error.sum() / mask.sum()
        else:
            wind_loss = wind_error.mean()
        
        total_loss = self.wind_weight * wind_loss
        
        # Pressure loss if provided
        if pred_pressure is not None and true_pressure is not None:
            if self.use_relative_error:
                # For pressure, smaller is stronger, so use absolute difference
                pressure_error = torch.abs(pred_pressure - true_pressure)
            else:
                pressure_error = F.mse_loss(pred_pressure, true_pressure, reduction='none')
            
            if mask is not None:
                pressure_error = pressure_error * mask
                pressure_loss = pressure_error.sum() / mask.sum()
            else:
                pressure_loss = pressure_error.mean()
            
            total_loss += self.pressure_weight * pressure_loss
        
        return total_loss


class PhysicsInformedHurricaneLoss(nn.Module):
    """Physics-informed loss function incorporating atmospheric constraints."""
    
    def __init__(
        self,
        track_weight: float = 1.0,
        intensity_weight: float = 0.5,
        physics_weight: float = 0.1,
        gradient_weight: float = 0.05,
        consistency_weight: float = 0.05
    ):
        """Initialize physics-informed loss.
        
        Args:
            track_weight: Weight for track prediction loss
            intensity_weight: Weight for intensity prediction loss
            physics_weight: Weight for physics constraints
            gradient_weight: Weight for gradient wind balance
            consistency_weight: Weight for wind-pressure consistency
        """
        super().__init__()
        
        # Component losses
        self.track_loss = HurricaneTrackLoss(use_weighted_loss=True)
        self.intensity_loss = IntensityLoss()
        
        # Loss weights
        self.track_weight = track_weight
        self.intensity_weight = intensity_weight
        self.physics_weight = physics_weight
        self.gradient_weight = gradient_weight
        self.consistency_weight = consistency_weight
        
        # Physical constants
        self.omega = 7.2921e-5  # Earth's angular velocity (rad/s)
        self.g = 9.81  # Gravitational acceleration (m/s²)
        
        logger.info("Initialized physics-informed hurricane loss")
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        reanalysis_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Calculate physics-informed loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            reanalysis_data: Optional atmospheric data for physics constraints
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # Track loss
        if 'track' in predictions and 'track' in targets:
            losses['track'] = self.track_loss(
                predictions['track'],
                targets['track']
            )
        
        # Intensity loss
        if 'wind' in predictions and 'wind' in targets:
            losses['intensity'] = self.intensity_loss(
                predictions.get('wind'),
                targets.get('wind'),
                predictions.get('pressure'),
                targets.get('pressure')
            )
        
        # Physics constraints
        if self.physics_weight > 0:
            physics_losses = self._compute_physics_constraints(
                predictions,
                reanalysis_data
            )
            losses.update(physics_losses)
        
        # Combine losses
        total_loss = torch.tensor(0.0, device=predictions['track'].device)
        
        if 'track' in losses:
            total_loss += self.track_weight * losses['track']
        
        if 'intensity' in losses:
            total_loss += self.intensity_weight * losses['intensity']
        
        if 'gradient_wind' in losses:
            total_loss += self.gradient_weight * losses['gradient_wind']
        
        if 'wind_pressure' in losses:
            total_loss += self.consistency_weight * losses['wind_pressure']
        
        if 'motion_consistency' in losses:
            total_loss += self.physics_weight * losses['motion_consistency']
        
        losses['total'] = total_loss
        
        return losses
    
    def _compute_physics_constraints(
        self,
        predictions: Dict[str, torch.Tensor],
        reanalysis_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute physics-based constraint losses.
        
        Args:
            predictions: Model predictions
            reanalysis_data: Atmospheric data
            
        Returns:
            Dictionary with physics constraint losses
        """
        physics_losses = {}
        
        # Gradient wind balance
        if 'track' in predictions and reanalysis_data is not None:
            gradient_loss = self._gradient_wind_balance(
                predictions['track'],
                reanalysis_data.get('pressure_field')
            )
            if gradient_loss is not None:
                physics_losses['gradient_wind'] = gradient_loss
        
        # Wind-pressure relationship
        if 'wind' in predictions and 'pressure' in predictions:
            wp_loss = self._wind_pressure_relationship(
                predictions['wind'],
                predictions['pressure']
            )
            physics_losses['wind_pressure'] = wp_loss
        
        # Motion consistency
        if 'track' in predictions:
            motion_loss = self._motion_consistency(predictions['track'])
            physics_losses['motion_consistency'] = motion_loss
        
        return physics_losses
    
    def _gradient_wind_balance(
        self,
        track: torch.Tensor,
        pressure_field: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Compute gradient wind balance constraint.
        
        Args:
            track: Hurricane track (batch, time, 2)
            pressure_field: Pressure field around hurricane
            
        Returns:
            Gradient wind balance loss
        """
        if pressure_field is None:
            return None
        
        # Extract positions
        lat = track[..., 0]
        
        # Coriolis parameter
        f = 2 * self.omega * torch.sin(torch.deg2rad(lat))
        
        # Compute pressure gradient (simplified)
        # In practice, this would involve proper gradient calculation
        # from the pressure field
        dp_dr = torch.ones_like(lat) * 100  # Placeholder
        
        # Gradient wind equation: V² / r + fV - (1/ρ) * dp/dr = 0
        # Simplified constraint
        constraint = f * 10  # Simplified placeholder
        
        return F.mse_loss(constraint, torch.zeros_like(constraint))
    
    def _wind_pressure_relationship(
        self,
        wind: torch.Tensor,
        pressure: torch.Tensor
    ) -> torch.Tensor:
        """Compute wind-pressure relationship constraint.
        
        Uses empirical relationship: V_max ≈ k * sqrt(ΔP)
        where ΔP = environmental_pressure - central_pressure
        
        Args:
            wind: Wind speed predictions (knots)
            pressure: Pressure predictions (mb)
            
        Returns:
            Wind-pressure consistency loss
        """
        # Environmental pressure (assumed)
        p_env = 1013.0
        
        # Pressure deficit
        pressure_deficit = torch.clamp(p_env - pressure, min=0)
        
        # Empirical relationship (Atkinson-Holliday)
        # V_max = 6.3 * sqrt(ΔP) for Atlantic hurricanes
        expected_wind = 6.3 * torch.sqrt(pressure_deficit)
        
        # Allow some tolerance
        wind_error = torch.abs(wind - expected_wind)
        
        # Use Huber loss for robustness
        return F.smooth_l1_loss(wind, expected_wind)
    
    def _motion_consistency(self, track: torch.Tensor) -> torch.Tensor:
        """Compute motion consistency constraint.
        
        Ensures smooth, physically plausible hurricane motion.
        
        Args:
            track: Hurricane track (batch, time, 2)
            
        Returns:
            Motion consistency loss
        """
        # Compute velocities
        dt = 6.0  # 6-hour time steps
        velocities = (track[:, 1:] - track[:, :-1]) / dt
        
        # Compute accelerations
        accelerations = (velocities[:, 1:] - velocities[:, :-1]) / dt
        
        # Penalize large accelerations (smooth motion)
        max_acceleration = 0.5  # degrees per hour²
        excess_acceleration = F.relu(
            torch.abs(accelerations) - max_acceleration
        )
        
        # Also penalize unrealistic speeds
        speeds = torch.norm(velocities, dim=-1)
        max_speed = 70 / 60  # 70 knots in degrees/hour
        excess_speed = F.relu(speeds - max_speed)
        
        return excess_acceleration.mean() + 0.1 * excess_speed.mean()


class EnsembleLoss(nn.Module):
    """Loss function for ensemble predictions with uncertainty."""
    
    def __init__(
        self,
        base_loss: nn.Module,
        uncertainty_weight: float = 0.1,
        diversity_weight: float = 0.05
    ):
        """Initialize ensemble loss.
        
        Args:
            base_loss: Base loss function for predictions
            uncertainty_weight: Weight for uncertainty calibration
            diversity_weight: Weight for ensemble diversity
        """
        super().__init__()
        self.base_loss = base_loss
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
    
    def forward(
        self,
        ensemble_predictions: List[Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
        return_member_losses: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Calculate ensemble loss.
        
        Args:
            ensemble_predictions: List of predictions from ensemble members
            targets: Ground truth values
            return_member_losses: Whether to return individual member losses
            
        Returns:
            Dictionary with loss components
        """
        num_members = len(ensemble_predictions)
        device = ensemble_predictions[0]['track'].device
        
        # Calculate individual member losses
        member_losses = []
        for pred in ensemble_predictions:
            loss = self.base_loss(pred, targets)
            member_losses.append(loss)
        
        # Stack member predictions
        track_preds = torch.stack(
            [pred['track'] for pred in ensemble_predictions],
            dim=0
        )  # (members, batch, time, 2)
        
        # Calculate ensemble mean
        ensemble_mean = track_preds.mean(dim=0)
        
        # Calculate ensemble variance
        ensemble_var = track_preds.var(dim=0)
        
        # Main loss: ensemble mean prediction
        mean_loss = self.base_loss(
            {'track': ensemble_mean},
            targets
        )
        
        # Uncertainty calibration loss
        # Penalize if variance doesn't match actual error
        actual_error = torch.norm(
            ensemble_mean - targets['track'],
            dim=-1
        )
        predicted_std = torch.sqrt(ensemble_var.sum(dim=-1))
        
        # Negative log-likelihood assuming Gaussian
        nll_loss = torch.log(predicted_std + 1e-6) + \
                   actual_error ** 2 / (2 * predicted_std ** 2 + 1e-6)
        
        # Diversity loss - encourage diverse predictions
        if self.diversity_weight > 0:
            # Pairwise distances between predictions
            diversity_loss = 0
            for i in range(num_members):
                for j in range(i + 1, num_members):
                    dist = torch.norm(
                        track_preds[i] - track_preds[j],
                        dim=-1
                    ).mean()
                    diversity_loss -= dist  # Negative to encourage diversity
            
            diversity_loss /= (num_members * (num_members - 1) / 2)
        else:
            diversity_loss = torch.tensor(0.0, device=device)
        
        # Combine losses
        total_loss = mean_loss
        
        if isinstance(total_loss, dict):
            total_loss = total_loss.get('total', total_loss.get('track'))
        
        total_loss += self.uncertainty_weight * nll_loss.mean()
        total_loss += self.diversity_weight * diversity_loss
        
        losses = {
            'total': total_loss,
            'mean_prediction': mean_loss,
            'uncertainty': nll_loss.mean(),
            'diversity': diversity_loss
        }
        
        if return_member_losses:
            losses['member_losses'] = member_losses
        
        return losses


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """Get loss function by name.
    
    Args:
        loss_type: Type of loss function
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function instance
    """
    loss_functions = {
        'track': HurricaneTrackLoss,
        'intensity': IntensityLoss,
        'physics': PhysicsInformedHurricaneLoss,
        'ensemble': EnsembleLoss
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss_functions[loss_type](**kwargs)
