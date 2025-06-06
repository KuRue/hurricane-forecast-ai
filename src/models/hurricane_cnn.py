"""CNN-Transformer hybrid model for hurricane track prediction.

This model combines CNN encoders for spatial feature extraction from
reanalysis data with Transformer decoders for temporal modeling.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger

from .base import BaseHurricaneModel, HurricaneModelMixin, ModelConfig


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_pooling: bool = True
    ):
        """Initialize convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding
            use_pooling: Whether to use max pooling
        """
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        self.use_pooling = use_pooling
        
        if use_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        
        if self.use_pooling:
            x = self.pool(x)
        
        return x


class CNNEncoder(nn.Module):
    """CNN encoder for spatial feature extraction from reanalysis maps."""
    
    def __init__(
        self,
        input_channels: int = 9,  # Default ERA5 variables
        channels: List[int] = [64, 128, 256, 512],
        output_dim: int = 256
    ):
        """Initialize CNN encoder.
        
        Args:
            input_channels: Number of input channels (weather variables)
            channels: Channel sizes for each layer
            output_dim: Output dimension
        """
        super().__init__()
        
        # Build convolutional layers
        layers = []
        in_channels = input_channels
        
        for i, out_channels in enumerate(channels):
            layers.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    use_pooling=(i < len(channels) - 1)
                )
            )
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(channels[-1], output_dim)
        
        # Spatial attention (optional)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels[-1], 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Tuple of (features, attention_weights)
        """
        # Extract convolutional features
        conv_features = self.conv_layers(x)
        
        # Apply spatial attention
        attention = self.spatial_attention(conv_features)
        attended_features = conv_features * attention
        
        # Global pooling
        pooled = self.global_pool(attended_features)
        pooled = pooled.squeeze(-1).squeeze(-1)
        
        # Project to output dimension
        features = self.projection(pooled)
        
        return features, attention


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        return x + self.pe[:, :x.size(1)]


class HurricaneCNNTransformer(BaseHurricaneModel, HurricaneModelMixin):
    """CNN-Transformer hybrid model for hurricane forecasting."""
    
    def __init__(self, config: ModelConfig):
        """Initialize model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # CNN encoder for spatial features (if using reanalysis data)
        self.use_cnn = hasattr(config, 'use_reanalysis') and config.use_reanalysis
        if self.use_cnn:
            self.cnn_encoder = CNNEncoder(
                input_channels=9,  # ERA5 variables
                output_dim=config.hidden_dim
            )
        
        # Input projection for track features
        self.input_projection = nn.Linear(
            config.input_features,
            config.hidden_dim
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            config.hidden_dim,
            max_len=config.sequence_length + config.forecast_length
        )
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=config.hidden_dim,
            nhead=8,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        # Output heads
        self.output_heads = nn.ModuleDict()
        
        if config.predict_track:
            self.output_heads['track'] = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, 2)  # lat, lon
            )
        
        if config.predict_intensity:
            self.output_heads['intensity'] = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, 2)  # wind, pressure
            )
        
        if config.predict_size:
            self.output_heads['size'] = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, 4)  # wind radii
            )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"Initialized HurricaneCNNTransformer with "
            f"{self.count_parameters():,} parameters"
        )
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        track_features: torch.Tensor,
        reanalysis_maps: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            track_features: Track features (batch, seq_len, features)
            reanalysis_maps: Optional reanalysis maps (batch, seq_len, channels, H, W)
            src_mask: Source sequence mask
            tgt_mask: Target sequence mask
            
        Returns:
            Dictionary with model outputs
        """
        batch_size, seq_len, _ = track_features.shape
        
        # Project track features
        track_embedded = self.input_projection(track_features)
        
        # Process reanalysis maps if available
        if self.use_cnn and reanalysis_maps is not None:
            # Reshape to process all time steps
            b, t, c, h, w = reanalysis_maps.shape
            maps_reshaped = rearrange(reanalysis_maps, 'b t c h w -> (b t) c h w')
            
            # Extract spatial features
            spatial_features, attention_maps = self.cnn_encoder(maps_reshaped)
            spatial_features = rearrange(
                spatial_features, '(b t) d -> b t d', b=b, t=t
            )
            
            # Combine with track features
            embedded = track_embedded + spatial_features
        else:
            embedded = track_embedded
            attention_maps = None
        
        # Add positional encoding
        embedded = self.positional_encoding(embedded)
        
        # Split into encoder and decoder sequences
        encoder_seq = embedded[:, :self.config.sequence_length]
        
        # Create decoder input (shifted right)
        decoder_seq = torch.zeros(
            batch_size,
            self.config.forecast_length,
            self.config.hidden_dim,
            device=embedded.device
        )
        
        # Transformer forward pass
        transformer_out = self.transformer(
            src=encoder_seq,
            tgt=decoder_seq,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        
        # Apply output heads
        outputs = {}
        
        if 'track' in self.output_heads:
            outputs['track'] = self.output_heads['track'](transformer_out)
        
        if 'intensity' in self.output_heads:
            intensity = self.output_heads['intensity'](transformer_out)
            outputs['wind'] = intensity[..., 0]
            outputs['pressure'] = intensity[..., 1]
        
        if 'size' in self.output_heads:
            outputs['size'] = self.output_heads['size'](transformer_out)
        
        # Add attention maps if available
        if attention_maps is not None:
            outputs['attention_maps'] = attention_maps
        
        return outputs
    
    def generate_forecast(
        self,
        initial_conditions: torch.Tensor,
        steps: int,
        reanalysis_maps: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate multi-step forecast autoregressively.
        
        Args:
            initial_conditions: Initial track conditions
            steps: Number of forecast steps
            reanalysis_maps: Optional reanalysis data
            
        Returns:
            Dictionary with forecasts
        """
        self.eval()
        
        with torch.no_grad():
            # Initialize with input sequence
            current_track = initial_conditions.clone()
            forecasts = {'track': [], 'wind': [], 'pressure': []}
            
            for step in range(steps):
                # Make single step prediction
                outputs = self.forward(
                    current_track,
                    reanalysis_maps[:, :current_track.size(1)]
                    if reanalysis_maps is not None else None
                )
                
                # Extract predictions for last time step
                if 'track' in outputs:
                    next_pos = outputs['track'][:, -1:]
                    forecasts['track'].append(next_pos)
                    
                    # Update current track
                    current_track = torch.cat([
                        current_track[:, 1:],
                        torch.cat([
                            next_pos,
                            outputs.get('wind', torch.zeros_like(next_pos[..., :1]))[:, -1:],
                            outputs.get('pressure', torch.zeros_like(next_pos[..., :1]))[:, -1:]
                        ], dim=-1)
                    ], dim=1)
                
                if 'wind' in outputs:
                    forecasts['wind'].append(outputs['wind'][:, -1:])
                
                if 'pressure' in outputs:
                    forecasts['pressure'].append(outputs['pressure'][:, -1:])
            
            # Stack forecasts
            for key in forecasts:
                if forecasts[key]:
                    forecasts[key] = torch.cat(forecasts[key], dim=1)
        
        return forecasts
    
    def _get_parameter_groups(self) -> List[Dict]:
        """Get parameter groups for differential learning rates."""
        # Separate CNN and transformer parameters
        if self.use_cnn:
            cnn_params = list(self.cnn_encoder.parameters())
            transformer_params = list(self.transformer.parameters())
            head_params = list(self.output_heads.parameters())
            
            return [
                {"params": cnn_params, "lr": self.config.learning_rate * 0.1},
                {"params": transformer_params, "lr": self.config.learning_rate},
                {"params": head_params, "lr": self.config.learning_rate * 2}
            ]
        else:
            return super()._get_parameter_groups()


class LightweightHurricaneModel(BaseHurricaneModel):
    """Lightweight model for quick experiments and baseline comparisons."""
    
    def __init__(self, config: ModelConfig):
        """Initialize lightweight model."""
        super().__init__(config)
        
        # Simple LSTM-based model
        self.lstm = nn.LSTM(
            input_size=config.input_features,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, 2)  # lat, lon
        
        logger.info(
            f"Initialized LightweightHurricaneModel with "
            f"{self.count_parameters():,} parameters"
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # LSTM forward
        lstm_out, _ = self.lstm(inputs)
        
        # Project to track coordinates
        track_pred = self.output_projection(lstm_out)
        
        return {"track": track_pred}
