"""
GPU-Optimized Configuration for Better Performance
This config is designed to leverage GPU and improve prediction results
"""
import torch


class ConfigGPUOptimized:
    """Optimized configuration for GPU training with better performance"""
    
    # Data parameters
    DATA_DIR = "dataset_with_label"
    EEG_CHANNELS = 75
    SOURCE_REGIONS = 994
    SEQ_LEN = 500
    
    # Model parameters - INCREASED for better capacity
    D_MODEL = 512  # Increased from 256 (GPU can handle this)
    NHEAD = 8  # Keep at 8 (D_MODEL must be divisible by NHEAD)
    NUM_LAYERS = 8  # Increased from 6 for better representation
    DIM_FEEDFORWARD = 2048  # Increased from 1024
    DROPOUT = 0.15  # Slightly increased for regularization
    
    # Training parameters - OPTIMIZED for GPU
    BATCH_SIZE = 16  # Increased from 8 (GPU can handle larger batches)
    NUM_EPOCHS = 200  # Increased for better convergence
    LEARNING_RATE = 1e-4  # Increased back (we have better stability now)
    WEIGHT_DECAY = 1e-4  # Increased for better regularization
    WARMUP_EPOCHS = 10  # Increased warmup period
    
    # Advanced learning rate schedule
    USE_COSINE_ANNEALING = True  # Use cosine annealing with warm restarts
    T_0 = 20  # Period for first restart (in epochs)
    T_MULT = 2  # Factor to increase period after each restart
    ETA_MIN = 1e-6  # Minimum learning rate
    
    # Data split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    # TEST_SPLIT = 0.1 (automatically calculated)
    
    # Training settings - OPTIMIZED for GPU
    NUM_WORKERS = 8  # Increased for faster data loading on GPU
    PIN_MEMORY = True  # Essential for GPU training
    NORMALIZE = True
    PREFETCH_FACTOR = 2  # Prefetch batches for faster loading
    PERSISTENT_WORKERS = True  # Keep workers alive between epochs
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # GPU optimization
    CUDNN_BENCHMARK = True  # Enable cuDNN auto-tuner
    TF32_ALLOW = True  # Allow TF32 on Ampere GPUs for better performance
    
    # Checkpoint and logging
    CHECKPOINT_DIR = "checkpoints_gpu_optimized"
    LOG_DIR = "logs_gpu_optimized"
    SAVE_EVERY = 10  # Save less frequently (we have more epochs)
    LOG_EVERY = 5  # Log more frequently
    SAVE_BEST_ONLY = False  # Save periodic checkpoints too
    
    # Early stopping - More patient for better convergence
    PATIENCE = 30  # Increased from 15
    MIN_DELTA = 1e-5  # More sensitive to improvements
    
    # Loss function
    LOSS_FN = "mse"  # Options: "mse", "mae", "huber"
    
    # Gradient clipping - Can be more aggressive with better initialization
    CLIP_GRAD_NORM = 2.0  # Increased from 1.0
    
    # Mixed precision training - Essential for GPU performance
    USE_AMP = True  # Automatic Mixed Precision (faster on GPU)
    
    # Label smoothing (optional, for better generalization)
    USE_LABEL_SMOOTHING = False
    LABEL_SMOOTHING_FACTOR = 0.1
    
    # Resume training
    RESUME_CHECKPOINT = None  # Path to checkpoint to resume from
    
    # Model EMA (Exponential Moving Average) for more stable predictions
    USE_EMA = True
    EMA_DECAY = 0.999
    
    @classmethod
    def display(cls):
        """Display all configuration parameters"""
        print("=" * 70)
        print("GPU-Optimized Configuration Parameters")
        print("=" * 70)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key:30} : {value}")
        print("=" * 70)
    
    @classmethod
    def verify_gpu(cls):
        """Verify GPU availability and print info"""
        if torch.cuda.is_available():
            print("\n" + "=" * 70)
            print("GPU INFORMATION")
            print("=" * 70)
            print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA Version: {torch.version.cuda}")
            print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
            print(f"✓ Current GPU: {torch.cuda.current_device()}")
            
            # Memory info
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ Total GPU Memory: {total_mem:.2f} GB")
            
            # Check for TF32 support
            if hasattr(torch.backends.cuda, 'matmul'):
                print(f"✓ TF32 Support: Available on Ampere+ GPUs")
            
            print("=" * 70)
            return True
        else:
            print("\n" + "=" * 70)
            print("WARNING: GPU not available, will use CPU")
            print("For best performance, ensure CUDA is properly installed")
            print("=" * 70)
            return False


class ConfigBalanced:
    """Balanced configuration - between CPU and GPU optimized
    Use this if GPU memory is limited or for faster experimentation"""
    
    # Data parameters
    DATA_DIR = "dataset_with_label"
    EEG_CHANNELS = 75
    SOURCE_REGIONS = 994
    SEQ_LEN = 500
    
    # Model parameters - BALANCED
    D_MODEL = 384  # Between 256 and 512
    NHEAD = 8
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 1536  # Between 1024 and 2048
    DROPOUT = 0.12
    
    # Training parameters
    BATCH_SIZE = 12
    NUM_EPOCHS = 150
    LEARNING_RATE = 7e-5
    WEIGHT_DECAY = 5e-5
    WARMUP_EPOCHS = 7
    
    # Advanced learning rate schedule
    USE_COSINE_ANNEALING = True
    T_0 = 15
    T_MULT = 2
    ETA_MIN = 1e-6
    
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
    CHECKPOINT_DIR = "checkpoints_balanced"
    LOG_DIR = "logs_balanced"
    SAVE_EVERY = 10
    LOG_EVERY = 5
    SAVE_BEST_ONLY = False
    
    # Early stopping
    PATIENCE = 25
    MIN_DELTA = 5e-5
    
    # Loss function
    LOSS_FN = "mse"
    
    # Gradient clipping
    CLIP_GRAD_NORM = 1.5
    
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
        print("Balanced Configuration Parameters")
        print("=" * 70)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key:30} : {value}")
        print("=" * 70)
    
    @classmethod
    def verify_gpu(cls):
        """Verify GPU availability"""
        return ConfigGPUOptimized.verify_gpu()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Available Configurations")
    print("=" * 70)
    print("1. ConfigGPUOptimized - Maximum performance (requires good GPU)")
    print("2. ConfigBalanced - Good balance (moderate GPU requirements)")
    print("=" * 70)
    
    ConfigGPUOptimized.verify_gpu()
    print("\n")
    ConfigGPUOptimized.display()


