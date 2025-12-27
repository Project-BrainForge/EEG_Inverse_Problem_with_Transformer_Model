"""
Quick diagnostic script to check the MAT file contents
"""
from scipy.io import loadmat
import numpy as np
from pathlib import Path

mat_file = Path("../../source/VEP/transformer_predictions_best_model.mat")

print(f"Loading: {mat_file}")
print("=" * 60)

try:
    data = loadmat(str(mat_file))
    
    print("Keys in MAT file:")
    for key in data.keys():
        if not key.startswith('__'):
            print(f"  - {key}: {type(data[key])}, shape: {data[key].shape if hasattr(data[key], 'shape') else 'N/A'}")
    
    print("\n" + "=" * 60)
    
    if 'all_out' in data:
        predictions = data['all_out']
        print(f"\nPredictions shape: {predictions.shape}")
        print(f"Predictions dtype: {predictions.dtype}")
        print(f"Min: {np.min(predictions):.6f}")
        print(f"Max: {np.max(predictions):.6f}")
        print(f"Mean: {np.mean(predictions):.6f}")
    
    if 'file_names' in data:
        print(f"\nfile_names type: {type(data['file_names'])}")
        print(f"file_names shape: {data['file_names'].shape}")
        print(f"file_names dtype: {data['file_names'].dtype}")
        print(f"First few entries:")
        
        file_names = data['file_names']
        for i in range(min(3, len(file_names))):
            print(f"  [{i}]: {file_names[i]}, type: {type(file_names[i])}")
            if isinstance(file_names[i], np.ndarray):
                print(f"       shape: {file_names[i].shape}, content: {file_names[i]}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: File loaded correctly")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

