"""
Transformer model for EEG to Source localization
Using a Transformer encoder-decoder architecture optimized for time series regression
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EEGSourceTransformer(nn.Module):
    """
    Transformer model for EEG to Source localization
    
    Architecture:
    - Input projection: Maps EEG channels (75) to model dimension
    - Transformer Encoder: Processes EEG time series
    - Transformer Decoder: Generates source predictions
    - Output projection: Maps to source dimensions (994)
    
    Args:
        eeg_channels: Number of EEG channels (75)
        source_regions: Number of source regions (994)
        d_model: Dimension of the model (default: 256)
        nhead: Number of attention heads (default: 8)
        num_encoder_layers: Number of encoder layers (default: 6)
        num_decoder_layers: Number of decoder layers (default: 6)
        dim_feedforward: Dimension of feedforward network (default: 1024)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self,
                 eeg_channels: int = 75,
                 source_regions: int = 994,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.eeg_channels = eeg_channels
        self.source_regions = source_regions
        
        # Input projection layers
        self.eeg_embedding = nn.Linear(eeg_channels, d_model)
        self.source_embedding = nn.Linear(source_regions, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # (seq_len, batch, feature)
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, source_regions)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, eeg_data: torch.Tensor, target_source: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            eeg_data: Input EEG data of shape (batch_size, seq_len, eeg_channels)
            target_source: Target source data of shape (batch_size, seq_len, source_regions)
                          Used during training for teacher forcing
        
        Returns:
            predicted_source: Predicted source data of shape (batch_size, seq_len, source_regions)
        """
        batch_size, seq_len, _ = eeg_data.shape
        
        # Project EEG data to model dimension
        # (batch_size, seq_len, eeg_channels) -> (batch_size, seq_len, d_model)
        eeg_embedded = self.eeg_embedding(eeg_data)
        
        # Change to (seq_len, batch_size, d_model) for transformer
        eeg_embedded = eeg_embedded.permute(1, 0, 2)
        
        # Add positional encoding
        eeg_encoded = self.pos_encoder(eeg_embedded)
        
        # During training, use teacher forcing with target
        # During inference, use autoregressive generation
        if target_source is not None:
            # Training mode: teacher forcing
            source_embedded = self.source_embedding(target_source)
            source_embedded = source_embedded.permute(1, 0, 2)
            source_encoded = self.pos_encoder(source_embedded)
            
            # Generate causal mask for decoder
            tgt_mask = self.transformer.generate_square_subsequent_mask(seq_len).to(eeg_data.device)
            
            # Transformer forward
            transformer_output = self.transformer(
                src=eeg_encoded,
                tgt=source_encoded,
                tgt_mask=tgt_mask
            )
        else:
            # Inference mode: use encoder-only approach for simplicity
            # We'll use the encoder output directly
            memory = self.transformer.encoder(eeg_encoded)
            
            # Initialize decoder input with zeros
            tgt = torch.zeros(seq_len, batch_size, self.d_model).to(eeg_data.device)
            tgt = self.pos_encoder(tgt)
            
            # Decode
            transformer_output = self.transformer.decoder(tgt, memory)
        
        # Change back to (batch_size, seq_len, d_model)
        transformer_output = transformer_output.permute(1, 0, 2)
        
        # Project to source dimensions
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, source_regions)
        predicted_source = self.output_projection(transformer_output)
        
        return predicted_source


class EEGSourceTransformerV2(nn.Module):
    """
    Simplified Transformer model using encoder-only architecture
    This is more suitable for regression tasks where we predict all timesteps at once
    
    Args:
        eeg_channels: Number of EEG channels (75)
        source_regions: Number of source regions (994)
        d_model: Dimension of the model (default: 256)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 6)
        dim_feedforward: Dimension of feedforward network (default: 1024)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self,
                 eeg_channels: int = 75,
                 source_regions: int = 994,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(eeg_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, source_regions)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            eeg_data: Input EEG data of shape (batch_size, seq_len, eeg_channels)
        
        Returns:
            predicted_source: Predicted source data of shape (batch_size, seq_len, source_regions)
        """
        # Project input to model dimension
        # (batch_size, seq_len, eeg_channels) -> (batch_size, seq_len, d_model)
        x = self.input_projection(eeg_data)
        
        # Add positional encoding (need to permute for pos_encoder)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Project to output dimension
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, source_regions)
        output = self.output_projection(x)
        
        return output


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    seq_len = 500
    eeg_channels = 75
    source_regions = 994
    
    # Create dummy data
    eeg_data = torch.randn(batch_size, seq_len, eeg_channels)
    
    # Test V2 model (simpler, recommended)
    print("Testing EEGSourceTransformerV2...")
    model_v2 = EEGSourceTransformerV2(
        eeg_channels=eeg_channels,
        source_regions=source_regions,
        d_model=256,
        nhead=8,
        num_layers=6
    )
    
    output = model_v2(eeg_data)
    print(f"Input shape: {eeg_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model_v2.parameters()):,}")

