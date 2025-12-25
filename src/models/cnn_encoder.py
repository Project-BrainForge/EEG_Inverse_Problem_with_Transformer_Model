"""
CNN Encoder for Spatial EEG Feature Extraction
Processes 2D topological EEG maps to extract spatial features for transformer input.
"""

import torch
import torch.nn as nn


class SpatialCNN(nn.Module):
    """
    CNN encoder for processing topological EEG maps.
    
    Processes each timestep's 2D spatial map independently and extracts features
    that are then fed into the transformer encoder.
    
    Architecture:
        Input: (batch, time, height, width) - e.g., (batch, 500, 64, 64)
        Process each timestep through CNN blocks
        Output: (batch, time, d_model) - e.g., (batch, 500, 256)
    
    Parameters
    ----------
    image_size : int or tuple
        Size of input images (height, width) (default: (64, 64))
    d_model : int
        Output feature dimension (default: 256)
    channels : list
        Number of channels in each conv layer (default: [32, 64, 128])
    kernel_size : int
        Kernel size for conv layers (default: 3)
    dropout : float
        Dropout rate (default: 0.1)
    """
    
    def __init__(self, 
                 image_size=(64, 64),
                 d_model=256,
                 channels=[32, 64, 128],
                 kernel_size=3,
                 dropout=0.1):
        super(SpatialCNN, self).__init__()
        
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        
        self.image_size = image_size
        self.d_model = d_model
        self.channels = channels
        
        # Build CNN layers
        layers = []
        in_channels = 1  # Grayscale input
        
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                         padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=dropout)
            ])
            in_channels = out_channels
        
        self.cnn_layers = nn.Sequential(*layers)
        
        # Calculate spatial dimensions after pooling
        # After each MaxPool2d(2, 2), dimensions are halved
        num_pools = len(channels)
        h_out = image_size[0] // (2 ** num_pools)
        w_out = image_size[1] // (2 ** num_pools)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection to d_model
        self.projection = nn.Sequential(
            nn.Linear(channels[-1], d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, time, height, width)
               e.g., (8, 500, 64, 64)
        
        Returns:
            features: Output tensor of shape (batch_size, time, d_model)
                     e.g., (8, 500, 256)
        """
        batch_size, time_steps, height, width = x.shape
        
        # Reshape to process all timesteps together
        # (batch, time, H, W) -> (batch*time, 1, H, W)
        x = x.reshape(batch_size * time_steps, 1, height, width)
        
        # Pass through CNN layers
        x = self.cnn_layers(x)  # (batch*time, channels[-1], H', W')
        
        # Global average pooling
        x = self.global_pool(x)  # (batch*time, channels[-1], 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (batch*time, channels[-1])
        
        # Project to d_model
        x = self.projection(x)  # (batch*time, d_model)
        
        # Reshape back to (batch, time, d_model)
        x = x.reshape(batch_size, time_steps, self.d_model)
        
        return x


class SpatialCNNLight(nn.Module):
    """
    Lightweight CNN encoder for smaller models.
    
    Uses fewer parameters while maintaining spatial feature extraction capability.
    """
    
    def __init__(self,
                 image_size=(64, 64),
                 d_model=256,
                 channels=[32, 64],
                 kernel_size=3,
                 dropout=0.1):
        super(SpatialCNNLight, self).__init__()
        
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        
        self.image_size = image_size
        self.d_model = d_model
        
        # Depthwise separable convolutions for efficiency
        layers = []
        in_channels = 1
        
        for out_channels in channels:
            # Depthwise
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=kernel_size//2, groups=in_channels, bias=False))
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.ReLU(inplace=True))
            
            # Pointwise
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout2d(p=dropout))
            
            in_channels = out_channels
        
        self.cnn_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Simpler projection
        self.projection = nn.Sequential(
            nn.Linear(channels[-1], d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, time, height, width)
        
        Returns:
            features: Output tensor of shape (batch_size, time, d_model)
        """
        batch_size, time_steps, height, width = x.shape
        
        # Reshape: (batch, time, H, W) -> (batch*time, 1, H, W)
        x = x.reshape(batch_size * time_steps, 1, height, width)
        
        # CNN processing
        x = self.cnn_layers(x)
        x = self.global_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        
        # Project to d_model
        x = self.projection(x)
        
        # Reshape back: (batch*time, d_model) -> (batch, time, d_model)
        x = x.reshape(batch_size, time_steps, self.d_model)
        
        return x


def test_cnn_encoder():
    """Test the CNN encoder modules."""
    print("="*60)
    print("Testing CNN Encoder Modules")
    print("="*60)
    
    batch_size = 8
    time_steps = 500
    image_size = (64, 64)
    d_model = 256
    
    # Create dummy input
    x = torch.randn(batch_size, time_steps, *image_size)
    print(f"\nInput shape: {x.shape}")
    print(f"Input size: {x.element_size() * x.nelement() / 1024**2:.2f} MB")
    
    # Test standard CNN
    print("\n" + "-"*60)
    print("Test 1: Standard SpatialCNN")
    print("-"*60)
    
    model = SpatialCNN(
        image_size=image_size,
        d_model=d_model,
        channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output size: {output.element_size() * output.nelement() / 1024**2:.2f} MB")
    
    assert output.shape == (batch_size, time_steps, d_model), \
        f"Expected shape ({batch_size}, {time_steps}, {d_model}), got {output.shape}"
    
    # Test lightweight CNN
    print("\n" + "-"*60)
    print("Test 2: Lightweight SpatialCNNLight")
    print("-"*60)
    
    model_light = SpatialCNNLight(
        image_size=image_size,
        d_model=d_model,
        channels=[32, 64],
        kernel_size=3,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model_light.parameters()):,}")
    
    output_light = model_light(x)
    print(f"Output shape: {output_light.shape}")
    print(f"Output size: {output_light.element_size() * output_light.nelement() / 1024**2:.2f} MB")
    
    assert output_light.shape == (batch_size, time_steps, d_model), \
        f"Expected shape ({batch_size}, {time_steps}, {d_model}), got {output_light.shape}"
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_cnn_encoder()
