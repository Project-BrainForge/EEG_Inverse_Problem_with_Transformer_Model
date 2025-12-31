"""
Conservative Configuration for Stable Training
Use this if you're getting "Warning: very large gradient norm"
"""
import torch


class ConfigConservative:
    """
    Conservative configuration for stable training
    Designed to prevent large gradients and ensure stability
    """
    
    # Data parameters
    DATA_DIR = "dataset_with_label"
    EEG_CHANNELS = 75
    SOURCE_REGIONS = 994
    SEQ_LEN = 500
    
    # Model parameters - More conservative
    D_MODEL = 384  # Moderate size
    NHEAD = 8
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 1536
    DROPOUT = 0.2  # Higher dropout for regularization
    
    # Training parameters - VERY CONSERVATIVE
    BATCH_SIZE = 12
    NUM_EPOCHS = 200
    LEARNING_RATE = 3e-5  # Much lower than optimized (was 1e-4)
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 15  # Longer warmup
    
    # Advanced learning rate schedule
    USE_COSINE_ANNEALING = True
    T_0 = 25  # Longer period
    T_MULT = 2
    ETA_MIN = 1e-7  # Lower minimum
    
    # Data split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    
    # Training settings
    NUM_WORKERS = 6
    PIN_MEMORY = True
    NORMALIZE = True
    PREFETCH_FACTOR = 2
    PERSISTENT_WORKERS = True
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # GPU optimization
    CUDNN_BENCHMARK = True
    TF32_ALLOW = True
    
    # Checkpoint and logging
    CHECKPOINT_DIR = "checkpoints_conservative"
    LOG_DIR = "logs_conservative"
    SAVE_EVERY = 10
    LOG_EVERY = 5
    SAVE_BEST_ONLY = False
    
    # Early stopping - More patient
    PATIENCE = 40
    MIN_DELTA = 1e-5
    
    # Loss function
    LOSS_FN = "mse"
    
    # Gradient clipping - MORE AGGRESSIVE
    CLIP_GRAD_NORM = 0.5  # Much lower (was 2.0)
    
    # Mixed precision training
    USE_AMP = True
    
    # Model EMA
    USE_EMA = True
    EMA_DECAY = 0.999
    
    # Label smoothing
    USE_LABEL_SMOOTHING = False
    LABEL_SMOOTHING_FACTOR = 0.1
    
    # Resume training
    RESUME_CHECKPOINT = None
    
    @classmethod
    def display(cls):
        """Display all configuration parameters"""
        print("=" * 70)
        print("Conservative Configuration Parameters")
        print("=" * 70)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key:30} : {value}")
        print("=" * 70)
    
    @classmethod
    def verify_gpu(cls):
        """Verify GPU availability"""
        if torch.cuda.is_available():
            print("\n" + "=" * 70)
            print("GPU INFORMATION")
            print("=" * 70)
            print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA Version: {torch.version.cuda}")
            print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
            
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ Total GPU Memory: {total_mem:.2f} GB")
            print("=" * 70)
            return True
        else:
            print("\n" + "=" * 70)
            print("WARNING: GPU not available, will use CPU")
            print("=" * 70)
            return False


if __name__ == "__main__":
    ConfigConservative.verify_gpu()
    print("\n")
    ConfigConservative.display()


