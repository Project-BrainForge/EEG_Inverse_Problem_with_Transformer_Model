"""
Hybrid CNN-Transformer Model for EEG Source Localization
Separates CNN spatial feature extraction from transformer temporal modeling.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from .cnn_encoder import SpatialCNN, SpatialCNNLight
    from .transformer import EEGTransformer
    from ..utils.topological_converter import EEGTopologicalConverter
except ImportError:
    from models.cnn_encoder import SpatialCNN, SpatialCNNLight
    from models.transformer import EEGTransformer
    from utils.topological_converter import EEGTopologicalConverter


class CNNTransformerHybrid(nn.Module):
    """
    Hybrid model combining CNN spatial feature extraction with Transformer.
    
    Architecture:
        1. Raw EEG (batch, seq_len, channels) 
        2. → Topological Converter → (batch, seq_len, H, W)
        3. → CNN Encoder → (batch, seq_len, d_model)
        4. → Transformer → (batch, seq_len, output_channels)
    
    This separation allows:
    - Independent development/testing of CNN and Transformer
    - Easy swapping of CNN architectures
    - Better code organization and reusability
    
    Parameters
    ----------
    eeg_channels : int
        Number of EEG input channels (default: 75)
    output_channels : int
        Number of output brain regions (default: 994)
    d_model : int
        Transformer embedding dimension (default: 256)
    nhead : int
        Number of transformer attention heads (default: 8)
    num_layers : int
        Number of transformer encoder layers (default: 6)
    dim_feedforward : int
        Transformer feedforward dimension (default: 1024)
    dropout : float
        Dropout rate (default: 0.1)
    topo_image_size : int
        Size of topological maps (default: 64)
    electrode_file : str
        Path to electrode configuration (default: 'anatomy/electrode_75.mat')
    cnn_channels : list
        CNN channel progression (default: [32, 64, 128])
    cnn_kernel_size : int
        CNN kernel size (default: 3)
    cnn_type : str
        CNN type: 'standard' or 'light' (default: 'standard')
    """
    
    def __init__(
        self,
        eeg_channels=75,
        output_channels=994,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=500,
        topo_image_size=64,
        electrode_file='anatomy/electrode_75.mat',
        cnn_channels=[32, 64, 128],
        cnn_kernel_size=3,
        cnn_type='standard'
    ):
        super(CNNTransformerHybrid, self).__init__()
        
        self.eeg_channels = eeg_channels
        self.output_channels = output_channels
        self.d_model = d_model
        
        # Component 1: Topological Converter
        print(f"Initializing topological converter ({topo_image_size}x{topo_image_size})...")
        self.topo_converter = EEGTopologicalConverter(
            electrode_file=electrode_file,
            image_size=(topo_image_size, topo_image_size),
            sphere='auto',
            normalize=True
        )
        
        # Component 2: CNN Spatial Encoder
        print(f"Initializing CNN encoder (type: {cnn_type})...")
        if cnn_type == 'light':
            self.cnn_encoder = SpatialCNNLight(
                image_size=topo_image_size,
                d_model=d_model,
                channels=cnn_channels[:2],
                kernel_size=cnn_kernel_size,
                dropout=dropout
            )
        else:
            self.cnn_encoder = SpatialCNN(
                image_size=topo_image_size,
                d_model=d_model,
                channels=cnn_channels,
                kernel_size=cnn_kernel_size,
                dropout=dropout
            )
        
        # Component 3: Transformer (without input projection)
        print("Initializing transformer encoder...")
        self.transformer = EEGTransformer(
            input_channels=eeg_channels,  # Not used, but kept for compatibility
            output_channels=output_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_cnn_encoder=False  # Transformer uses its own linear projection
        )
        
        # Replace transformer's input projection with identity (CNN already projects to d_model)
        self.transformer.input_projection = nn.Identity()
        
        print("✓ Hybrid model initialized successfully")
    
    def forward(self, x):
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input EEG tensor of shape (batch_size, seq_len, eeg_channels)
               e.g., (8, 500, 75)
        
        Returns:
            output: Predicted source tensor of shape (batch_size, seq_len, output_channels)
                    e.g., (8, 500, 994)
        """
        # Step 1: Convert to topological maps
        # (batch, seq_len, channels) -> (batch, seq_len, H, W)
        x_topo = self.topo_converter.to_torch(x, device=x.device, verbose=False)
        
        # Step 2: CNN spatial feature extraction
        # (batch, seq_len, H, W) -> (batch, seq_len, d_model)
        x_features = self.cnn_encoder(x_topo)
        
        # Step 3: Transformer temporal modeling
        # (batch, seq_len, d_model) -> (batch, seq_len, output_channels)
        output = self.transformer(x_features)
        
        return output
    
    def get_intermediate_features(self, x):
        """
        Get intermediate features for visualization/analysis.
        
        Args:
            x: Input EEG tensor of shape (batch_size, seq_len, eeg_channels)
        
        Returns:
            dict with keys:
                - 'topological': Topological maps (batch, seq_len, H, W)
                - 'cnn_features': CNN features (batch, seq_len, d_model)
                - 'output': Final predictions (batch, seq_len, output_channels)
        """
        # Get topological maps
        x_topo = self.topo_converter.to_torch(x, device=x.device, verbose=False)
        
        # Get CNN features
        x_features = self.cnn_encoder(x_topo)
        
        # Get final output
        output = self.transformer(x_features)
        
        return {
            'topological': x_topo,
            'cnn_features': x_features,
            'output': output
        }


def create_hybrid_model(config):
    """
    Factory function to create hybrid model from config.
    
    Args:
        config: Configuration object with model hyperparameters
    
    Returns:
        CNNTransformerHybrid model
    """
    return CNNTransformerHybrid(
        eeg_channels=config.EEG_CHANNELS,
        output_channels=config.SOURCE_REGIONS,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        max_seq_len=config.SEQ_LEN,
        topo_image_size=config.TOPO_IMAGE_SIZE,
        electrode_file=config.ELECTRODE_FILE,
        cnn_channels=config.CNN_CHANNELS,
        cnn_kernel_size=config.CNN_KERNEL_SIZE,
        cnn_type=config.CNN_TYPE
    )


if __name__ == "__main__":
    # Test the hybrid model
    print("="*60)
    print("Testing CNN-Transformer Hybrid Model")
    print("="*60)
    
    batch_size = 4
    seq_len = 500
    eeg_channels = 75
    output_channels = 994
    
    # Create model
    model = CNNTransformerHybrid(
        eeg_channels=eeg_channels,
        output_channels=output_channels,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=seq_len,
        topo_image_size=64,
        electrode_file='../../anatomy/electrode_75.mat',
        cnn_channels=[32, 64, 128],
        cnn_kernel_size=3,
        cnn_type='standard'
    )
    
    print(f"\n{'='*60}")
    print("Model Statistics")
    print("="*60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    cnn_params = sum(p.numel() for p in model.cnn_encoder.parameters())
    transformer_params = sum(p.numel() for p in model.transformer.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"  - CNN encoder: {cnn_params:,} ({cnn_params/total_params*100:.1f}%)")
    print(f"  - Transformer: {transformer_params:,} ({transformer_params/total_params*100:.1f}%)")
    
    # Test forward pass
    print(f"\n{'='*60}")
    print("Testing Forward Pass")
    print("="*60)
    
    x = torch.randn(batch_size, seq_len, eeg_channels)
    print(f"Input shape: {x.shape}")
    
    import time
    start = time.time()
    output = model(x)
    elapsed = time.time() - start
    
    print(f"Output shape: {output.shape}")
    print(f"Forward pass time: {elapsed*1000:.2f} ms")
    print(f"Throughput: {batch_size/elapsed:.2f} samples/sec")
    
    # Test intermediate features
    print(f"\n{'='*60}")
    print("Testing Intermediate Features")
    print("="*60)
    
    features = model.get_intermediate_features(x)
    print(f"Topological maps: {features['topological'].shape}")
    print(f"CNN features: {features['cnn_features'].shape}")
    print(f"Output: {features['output'].shape}")
    
    print(f"\n{'='*60}")
    print("✓ All tests passed!")
    print("="*60)
