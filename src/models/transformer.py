"""
Transformer model for EEG source localization
Maps EEG signals (500x75) to brain source activity (500x994)
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EEGTransformer(nn.Module):
    """
    Transformer model for EEG source localization
    
    Architecture:
    1. Linear projection from EEG channels to model dimension
    2. Positional encoding
    3. Transformer encoder layers
    4. Linear projection to output brain regions
    """
    
    def __init__(
        self,
        input_channels=75,      # EEG channels
        output_channels=994,    # Brain regions
        d_model=256,            # Model dimension
        nhead=8,                # Number of attention heads
        num_layers=6,           # Number of transformer layers
        dim_feedforward=1024,   # Feedforward dimension
        dropout=0.1,
        max_seq_len=500
    ):
        super(EEGTransformer, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.d_model = d_model
        
        # Input projection: (batch, seq_len, input_channels) -> (batch, seq_len, d_model)
        self.input_projection = nn.Linear(input_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection: (batch, seq_len, d_model) -> (batch, seq_len, output_channels)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_channels)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_channels)
               e.g., (batch_size, 500, 75)
        
        Returns:
            output: Output tensor of shape (batch_size, seq_len, output_channels)
                    e.g., (batch_size, 500, 994)
        """
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Output projection
        output = self.output_projection(x)  # (batch, seq_len, output_channels)
        
        return output


class EEGTransformerEncoderDecoder(nn.Module):
    """
    Encoder-Decoder Transformer for EEG source localization
    
    This architecture uses both encoder and decoder for potentially better performance
    """
    
    def __init__(
        self,
        input_channels=75,
        output_channels=994,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=500
    ):
        super(EEGTransformerEncoderDecoder, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_channels, d_model)
        
        # Target embedding (for decoder input during training)
        self.target_projection = nn.Linear(output_channels, d_model)
        
        # Positional encodings
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_channels)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt=None):
        """
        Forward pass
        
        Args:
            src: Source tensor (batch_size, seq_len, input_channels)
            tgt: Target tensor (batch_size, seq_len, output_channels) - optional for inference
        
        Returns:
            output: Predicted tensor (batch_size, seq_len, output_channels)
        """
        # Encode source
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        
        # Prepare target for decoder
        if tgt is None:
            # During inference, use zeros as initial decoder input
            tgt = torch.zeros(src.size(0), src.size(1), self.d_model, device=src.device)
        else:
            tgt = self.target_projection(tgt)
        
        tgt = self.pos_decoder(tgt)
        
        # Transformer forward
        output = self.transformer(src, tgt)
        
        # Project to output space
        output = self.output_projection(output)
        
        return output


def create_model(model_type='encoder', **kwargs):
    """
    Factory function to create transformer models
    
    Args:
        model_type: 'encoder' or 'encoder_decoder'
        **kwargs: Model hyperparameters
    
    Returns:
        model: PyTorch model
    """
    if model_type == 'encoder':
        return EEGTransformer(**kwargs)
    elif model_type == 'encoder_decoder':
        return EEGTransformerEncoderDecoder(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the model
    batch_size = 8
    seq_len = 500
    input_channels = 75
    output_channels = 994
    
    # Create model
    model = EEGTransformer(
        input_channels=input_channels,
        output_channels=output_channels,
        d_model=256,
        nhead=8,
        num_layers=6
    )
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_channels)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

