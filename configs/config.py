"""
Configuration file for EEG Source Localization Transformer
"""
import torch


class Config:
    """Configuration class for training and model parameters"""
    
    # Data parameters
    DATA_DIR = "dataset_with_label"
    EEG_CHANNELS = 75
    SOURCE_REGIONS = 994
    SEQ_LEN = 500
    
    # Model parameters
    D_MODEL = 256  # Dimension of the model
    NHEAD = 8  # Number of attention heads
    NUM_LAYERS = 6  # Number of transformer layers
    DIM_FEEDFORWARD = 1024  # Dimension of feedforward network
    DROPOUT = 0.1  # Dropout rate
    
    # Training parameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-5  # Reduced from 1e-4 for better stability
    WEIGHT_DECAY = 1e-5
    WARMUP_EPOCHS = 5
    
    # Data split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    # TEST_SPLIT = 0.1 (automatically calculated)
    
    # Training settings
    NUM_WORKERS = 4
    PIN_MEMORY = True
    NORMALIZE = True
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Checkpoint and logging
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    SAVE_EVERY = 5  # Save checkpoint every N epochs
    LOG_EVERY = 10  # Log every N batches
    
    # Early stopping
    PATIENCE = 15  # Number of epochs to wait for improvement
    MIN_DELTA = 1e-4  # Minimum change to qualify as improvement
    
    # Loss function
    LOSS_FN = "mse"  # Options: "mse", "mae", "huber"
    
    # Gradient clipping
    CLIP_GRAD_NORM = 1.0
    
    # Mixed precision training
    USE_AMP = True  # Automatic Mixed Precision
    
    # Resume training
    RESUME_CHECKPOINT = None  # Path to checkpoint to resume from
    
    @classmethod
    def display(cls):
        """Display all configuration parameters"""
        print("=" * 50)
        print("Configuration Parameters")
        print("=" * 50)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key}: {value}")
        print("=" * 50)


if __name__ == "__main__":
    Config.display()

