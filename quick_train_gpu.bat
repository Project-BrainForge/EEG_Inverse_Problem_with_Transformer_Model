@echo off
REM Quick Start Script for GPU-Optimized Training (Windows)
REM This script guides you through improved training for better results

echo ========================================================================
echo EEG Source Localization - GPU-Optimized Training
echo ========================================================================

REM Check if CUDA is available
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo WARNING: CUDA not available! Will use CPU ^(much slower^)
    echo To use GPU, ensure PyTorch with CUDA is installed
    set /p continue="Continue with CPU? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
) else (
    echo [92m✓ GPU Available[0m
    python -c "from configs.config_gpu_optimized import ConfigGPUOptimized; ConfigGPUOptimized.verify_gpu()"
)

echo.
echo ========================================================================
echo Training Configuration Options
echo ========================================================================
echo 1. Balanced     - Good performance, moderate GPU (4-6GB), ~1-2 hours
echo 2. Optimized    - Best performance, good GPU (6-8GB), ~2-4 hours
echo 3. Quick Test   - Fast validation, 10 epochs, ~5-10 minutes
echo 4. Custom       - Specify your own parameters
echo ========================================================================
echo.

set /p option="Select option (1-4): "

if "%option%"=="1" (
    echo Starting Balanced Configuration...
    python train_optimized.py --config balanced --data_dir ./dataset_with_label --epochs 150
) else if "%option%"=="2" (
    echo Starting Optimized Configuration ^(Best Results^)...
    python train_optimized.py --config optimized --data_dir ./dataset_with_label --epochs 200
) else if "%option%"=="3" (
    echo Starting Quick Test ^(10 epochs^)...
    python train_optimized.py --config balanced --data_dir ./dataset_with_label --epochs 10
) else if "%option%"=="4" (
    echo.
    set /p config="Config (balanced/optimized): "
    set /p epochs="Number of epochs: "
    set /p batch_size="Batch size (leave empty for default): "
    set /p lr="Learning rate (leave empty for default): "
    
    set cmd=python train_optimized.py --config %config% --data_dir ./dataset_with_label --epochs %epochs%
    
    if not "%batch_size%"=="" set cmd=%cmd% --batch_size %batch_size%
    if not "%lr%"=="" set cmd=%cmd% --lr %lr%
    
    echo Running: %cmd%
    %cmd%
) else (
    echo Invalid option
    exit /b 1
)

echo.
echo ========================================================================
echo Training Complete!
echo ========================================================================
echo.
echo Next steps:
echo 1. Validate checkpoints:
echo    python -m utils.checkpoint_validator --all
echo.
echo 2. View training curves:
echo    tensorboard --logdir logs_gpu_optimized
echo.
echo 3. Test on real data:
echo    python eval_real.py --checkpoint checkpoints_gpu_optimized/best_model.pt --data_dir source --subjects VEP
echo.
echo ========================================================================
pause

