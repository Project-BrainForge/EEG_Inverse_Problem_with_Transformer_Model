#!/bin/bash

# Quick Start Script for EEG Source Localization Transformer

echo "==============================================="
echo "EEG Source Localization Transformer Quick Start"
echo "==============================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""

# Test dataset loading
echo "Testing dataset loading..."
cd src
python -c "from data.dataset import EEGSourceDataset; print('✓ Dataset module OK')"

# Test model creation
echo "Testing model creation..."
python -c "from models.transformer import create_model; model = create_model(); print('✓ Model creation OK')"

echo ""
echo "==============================================="
echo "Ready to train! Run the following commands:"
echo "==============================================="
echo ""
echo "1. Train the model:"
echo "   cd src"
echo "   python train.py --config ../configs/config.yaml"
echo ""
echo "2. Monitor training (in a new terminal):"
echo "   tensorboard --logdir logs"
echo ""
echo "3. Evaluate the model:"
echo "   cd src"
echo "   python evaluate.py --config ../configs/config.yaml --checkpoint ../checkpoints/best_model.pth"
echo ""
echo "4. Run inference:"
echo "   cd src"
echo "   python inference.py --config ../configs/config.yaml --checkpoint ../checkpoints/best_model.pth --input ../dataset_with_label/sample_00000.mat"
echo ""
