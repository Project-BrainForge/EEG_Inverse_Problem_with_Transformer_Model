"""
Configuration file for EEG Source Localization Transformer
"""
import torch


class Config:
    """Configuration class for training and model parameters"""
    
    # Data parameters
    USE_METADATA_LOADER = True  # If True, use metadata files; if False, use dynamic generation
    
    # For metadata-based loading (USE_METADATA_LOADER = True)
    TRAIN_METADATA_PATH = "source/train_sample_source.mat"
    TEST_METADATA_PATH = "source/test_sample_source.mat"
    NMM_SPIKES_DIR = "source/nmm_spikes"  # Directory with a0/, a1/, etc. folders
    FWD_MATRIX_PATH = "anatomy/leadfield_75_20k.mat"  # Path to forward matrix file
    TRAIN_DATASET_LEN = 100  # None = use all samples from metadata
    TEST_DATASET_LEN = 50   # None = use all samples from metadata
    
    # For dynamic generation (USE_METADATA_LOADER = False)
    DATA_DIR = "."  # Root directory containing 'source/nmm_spikes'
    DATASET_LEN = 1000  # Number of samples to generate
    NUM_SOURCES = 2  # Number of active sources per sample
    PATCH_SIZE = 20  # Size of each source patch
    SNR_RANGE = (0, 30)  # SNR range in dB
    
    # Model parameters
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
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    WARMUP_EPOCHS = 5
    
    # Data split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    # TEST_SPLIT = 0.1 (automatically calculated)
    
    # Training settings
    # Note: Set NUM_WORKERS = 0 on Windows to avoid multiprocessing issues
    # On Linux/Mac, you can use 4-8 workers for faster loading
    NUM_WORKERS = 0  # Use 0 for Windows, 4-8 for Linux/Mac
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

    # Model selection and topological converter / CNN params
    MODEL_TYPE = 'hybrid'  # 'transformer' or 'hybrid'
    TOPO_IMAGE_SIZE = 64
    ELECTRODE_FILE = 'anatomy/electrode_75.mat'
    CNN_PARAMS = {"channels": [32, 64, 128], "kernel_size": 3, "dropout": 0.1}
    CNN_OUT_DIM = None  # If None, uses D_MODEL
    
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

