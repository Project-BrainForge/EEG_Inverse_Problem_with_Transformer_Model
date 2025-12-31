#!/bin/bash
# Quick Start Script for GPU-Optimized Training
# This script guides you through improved training for better results

echo "========================================================================"
echo "EEG Source Localization - GPU-Optimized Training"
echo "========================================================================"

# Check if CUDA is available
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
if [ $? -ne 0 ]; then
    echo "WARNING: CUDA not available! Will use CPU (much slower)"
    echo "To use GPU, ensure PyTorch with CUDA is installed"
    read -p "Continue with CPU? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ GPU Available"
    python -c "from configs.config_gpu_optimized import ConfigGPUOptimized; ConfigGPUOptimized.verify_gpu()"
fi

echo ""
echo "========================================================================"
echo "Training Configuration Options"
echo "========================================================================"
echo "1. Balanced     - Good performance, moderate GPU (4-6GB), ~1-2 hours"
echo "2. Optimized    - Best performance, good GPU (6-8GB), ~2-4 hours"
echo "3. Quick Test   - Fast validation, 10 epochs, ~5-10 minutes"
echo "4. Custom       - Specify your own parameters"
echo "========================================================================"
echo ""

read -p "Select option (1-4): " option

case $option in
    1)
        echo "Starting Balanced Configuration..."
        python train_optimized.py \
            --config balanced \
            --data_dir ./dataset_with_label \
            --epochs 150
        ;;
    2)
        echo "Starting Optimized Configuration (Best Results)..."
        python train_optimized.py \
            --config optimized \
            --data_dir ./dataset_with_label \
            --epochs 200
        ;;
    3)
        echo "Starting Quick Test (10 epochs)..."
        python train_optimized.py \
            --config balanced \
            --data_dir ./dataset_with_label \
            --epochs 10
        ;;
    4)
        echo ""
        read -p "Config (balanced/optimized): " config
        read -p "Number of epochs: " epochs
        read -p "Batch size (leave empty for default): " batch_size
        read -p "Learning rate (leave empty for default): " lr
        
        cmd="python train_optimized.py --config $config --data_dir ./dataset_with_label --epochs $epochs"
        
        if [ ! -z "$batch_size" ]; then
            cmd="$cmd --batch_size $batch_size"
        fi
        
        if [ ! -z "$lr" ]; then
            cmd="$cmd --lr $lr"
        fi
        
        echo "Running: $cmd"
        eval $cmd
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "Training Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Validate checkpoints:"
echo "   python -m utils.checkpoint_validator --all"
echo ""
echo "2. View training curves:"
echo "   tensorboard --logdir logs_gpu_optimized"
echo ""
echo "3. Test on real data:"
echo "   python eval_real.py --checkpoint checkpoints_gpu_optimized/best_model.pt --data_dir source --subjects VEP"
echo ""
echo "========================================================================"


