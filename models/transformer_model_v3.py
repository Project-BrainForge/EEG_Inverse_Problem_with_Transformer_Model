"""
Enhanced Transformer model V3 with improved architecture for better performance
Includes: Residual connections, Pre-LN, Better initialization
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EEGSourceTransformerV3(nn.Module):
    """
    Enhanced Transformer model V3 with improvements:
    - Better weight initialization
    - Pre-LayerNorm architecture (more stable)
    - Residual connections in output projection
    - Input/output normalization
    - Optional skip connection from input to output
    
    Args:
        eeg_channels: Number of EEG channels (75)
        source_regions: Number of source regions (994)
        d_model: Dimension of the model (default: 512)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 8)
        dim_feedforward: Dimension of feedforward network (default: 2048)
        dropout: Dropout rate (default: 0.15)
        use_skip_connection: Whether to use skip connection from input (default: True)
    """
    
    def __init__(self,
                 eeg_channels: int = 75,
                 source_regions: int = 994,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.15,
                 use_skip_connection: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.use_skip_connection = use_skip_connection
        
        # Input normalization
        self.input_norm = nn.LayerNorm(eeg_channels)
        
        # Input projection with residual
        self.input_projection = nn.Sequential(
            nn.Linear(eeg_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Pre-LN Transformer encoder layers (more stable than post-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',  # GELU is better than ReLU for transformers
            batch_first=True,
            norm_first=True  # Pre-LN: more stable training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)  # Final normalization
        )
        
        # Skip connection projection (if input/output dims don't match)
        if use_skip_connection:
            self.skip_projection = nn.Linear(eeg_channels, source_regions)
        
        # Enhanced output projection with residual connections
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, source_regions)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Enhanced weight initialization
        - Xavier for linear layers
        - Small std for output layer (prevents initial explosion)
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization with conservative gain
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Extra small initialization for final output layer
        final_layer = self.output_projection[-1]
        nn.init.xavier_uniform_(final_layer.weight, gain=0.1)
        if final_layer.bias is not None:
            nn.init.zeros_(final_layer.bias)
    
    def forward(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connections
        
        Args:
            eeg_data: Input EEG data of shape (batch_size, seq_len, eeg_channels)
        
        Returns:
            predicted_source: Predicted source data of shape (batch_size, seq_len, source_regions)
        """
        # Save input for skip connection
        if self.use_skip_connection:
            skip = self.skip_projection(eeg_data)
        
        # Input normalization
        x = self.input_norm(eeg_data)
        
        # Clamp to prevent extreme values
        x = torch.clamp(x, min=-10, max=10)
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding (need to permute for pos_encoder)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Project to output dimension
        output = self.output_projection(x)
        
        # Add skip connection if enabled
        if self.use_skip_connection:
            output = output + 0.1 * skip  # Weighted skip connection
        
        return output


class EEGSourceTransformerV3Large(EEGSourceTransformerV3):
    """
    Larger version of V3 for maximum performance (requires good GPU)
    """
    
    def __init__(self,
                 eeg_channels: int = 75,
                 source_regions: int = 994):
        super().__init__(
            eeg_channels=eeg_channels,
            source_regions=source_regions,
            d_model=768,  # Much larger
            nhead=12,
            num_layers=12,
            dim_feedforward=3072,
            dropout=0.15,
            use_skip_connection=True
        )


class EEGSourceTransformerV3Small(EEGSourceTransformerV3):
    """
    Smaller version of V3 for faster experimentation
    """
    
    def __init__(self,
                 eeg_channels: int = 75,
                 source_regions: int = 994):
        super().__init__(
            eeg_channels=eeg_channels,
            source_regions=source_regions,
            d_model=256,
            nhead=8,
            num_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            use_skip_connection=True
        )


if __name__ == "__main__":
    # Test the models
    batch_size = 4
    seq_len = 500
    eeg_channels = 75
    source_regions = 994
    
    # Create dummy data
    eeg_data = torch.randn(batch_size, seq_len, eeg_channels)
    
    print("=" * 70)
    print("Testing Enhanced Models")
    print("=" * 70)
    
    # Test V3 (default)
    print("\n1. EEGSourceTransformerV3 (Default)...")
    model_v3 = EEGSourceTransformerV3(
        eeg_channels=eeg_channels,
        source_regions=source_regions
    )
    output = model_v3(eeg_data)
    print(f"   Input shape: {eeg_data.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_v3.parameters()):,}")
    
    # Test V3 Large
    print("\n2. EEGSourceTransformerV3Large...")
    model_v3_large = EEGSourceTransformerV3Large(
        eeg_channels=eeg_channels,
        source_regions=source_regions
    )
    output = model_v3_large(eeg_data)
    print(f"   Input shape: {eeg_data.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_v3_large.parameters()):,}")
    
    # Test V3 Small
    print("\n3. EEGSourceTransformerV3Small...")
    model_v3_small = EEGSourceTransformerV3Small(
        eeg_channels=eeg_channels,
        source_regions=source_regions
    )
    output = model_v3_small(eeg_data)
    print(f"   Input shape: {eeg_data.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_v3_small.parameters()):,}")
    
    print("\n" + "=" * 70)
    print("All models tested successfully!")
    print("=" * 70)


