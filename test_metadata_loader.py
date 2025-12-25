"""
Test script for the metadata-based spike data loader
Tests loading from train_sample_source1.mat and test_sample_source1.mat
"""
import os
import numpy as np
from utils.loader import SpikeEEGMetadataDataset, create_dataloaders_from_metadata, load_mat_file


def test_metadata_dataset():
    """Test the SpikeEEGMetadataDataset class"""
    print("=" * 70)
    print("Testing SpikeEEGMetadataDataset")
    print("=" * 70)
    
    # Check if files exist
    train_metadata = "source/train_sample_source1.mat"
    fwd_matrix_path = "anatomy/leadfield_75_20k.mat"
    
    if not os.path.exists(train_metadata):
        print(f"ERROR: Training metadata file not found: {train_metadata}")
        print("Please ensure the file exists at the specified location.")
        return False
    
    if not os.path.exists(fwd_matrix_path):
        print(f"ERROR: Forward matrix file not found: {fwd_matrix_path}")
        print("Please ensure the file exists at the specified location.")
        return False
    
    try:
        # Load forward matrix
        print(f"\nLoading forward matrix from {fwd_matrix_path}...")
        fwd_data = load_mat_file(fwd_matrix_path)
        
        # Try to find forward matrix
        fwd = None
        for key in ['fwd', 'forward', 'leadfield', 'L']:
            if key in fwd_data:
                fwd = fwd_data[key]
                print(f"Found forward matrix with key '{key}', shape: {fwd.shape}")
                break
        
        if fwd is None:
            print("Available keys in forward matrix file:")
            for key in fwd_data.keys():
                if not key.startswith('__'):
                    val = fwd_data[key]
                    if isinstance(val, np.ndarray):
                        print(f"  - {key}: shape {val.shape}")
            return False
        
        # Transpose if needed (should be 75 x 994)
        if fwd.shape[0] == 994 and fwd.shape[1] == 75:
            fwd = fwd.T
            print(f"Transposed forward matrix to shape: {fwd.shape}")
        
        # Create dataset
        print(f"\nCreating dataset from {train_metadata}...")
        dataset = SpikeEEGMetadataDataset(
            metadata_path=train_metadata,
            fwd=fwd,
            nmm_spikes_dir="source/nmm_spikes",
            dataset_len=10,  # Test with just 10 samples
            normalize=True
        )
        
        print(f"\nDataset created successfully!")
        print(f"Dataset length: {len(dataset)}")
        
        # Test loading a few samples
        print("\nTesting sample loading...")
        for i in range(min(3, len(dataset))):
            print(f"\nLoading sample {i}...")
            eeg_data, source_data = dataset[i]
            
            print(f"  EEG data shape: {eeg_data.shape}")
            print(f"  Source data shape: {source_data.shape}")
            print(f"  EEG range: [{eeg_data.min():.4f}, {eeg_data.max():.4f}]")
            print(f"  Source range: [{source_data.min():.4f}, {source_data.max():.4f}]")
            print(f"  Active sources: {(source_data.sum(dim=0) > 0).sum().item()} regions")
        
        # Verify shapes
        assert eeg_data.shape == (500, 75), f"Expected EEG shape (500, 75), got {eeg_data.shape}"
        assert source_data.shape == (500, 994), f"Expected source shape (500, 994), got {source_data.shape}"
        
        print("\n✓ Metadata dataset test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Metadata dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_dataloader():
    """Test the create_dataloaders_from_metadata function"""
    print("\n" + "=" * 70)
    print("Testing create_dataloaders_from_metadata")
    print("=" * 70)
    
    train_metadata = "source/train_sample_source1.mat"
    test_metadata = "source/test_sample_source1.mat"
    fwd_matrix_path = "anatomy/leadfield_75_20k.mat"
    
    # Check if files exist
    if not os.path.exists(train_metadata):
        print(f"ERROR: {train_metadata} not found!")
        return False
    
    if not os.path.exists(test_metadata):
        print(f"ERROR: {test_metadata} not found!")
        return False
    
    if not os.path.exists(fwd_matrix_path):
        print(f"ERROR: {fwd_matrix_path} not found!")
        return False
    
    try:
        print("\nCreating dataloaders...")
        train_loader, val_loader, test_loader, _ = create_dataloaders_from_metadata(
            train_metadata_path=train_metadata,
            test_metadata_path=test_metadata,
            fwd_matrix_path=fwd_matrix_path,
            batch_size=4,
            val_split=0.2,
            num_workers=0,  # Use 0 for testing
            nmm_spikes_dir="source/nmm_spikes",
            train_dataset_len=20,  # Small dataset for testing
            test_dataset_len=10,
        )
        
        print(f"\nDataloaders created successfully!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test loading batches
        print("\nTesting batch loading...")
        
        print("\n1. Training batch:")
        for eeg_batch, source_batch in train_loader:
            print(f"   EEG batch shape: {eeg_batch.shape}")
            print(f"   Source batch shape: {source_batch.shape}")
            print(f"   EEG range: [{eeg_batch.min():.4f}, {eeg_batch.max():.4f}]")
            print(f"   Source range: [{source_batch.min():.4f}, {source_batch.max():.4f}]")
            break
        
        print("\n2. Validation batch:")
        for eeg_batch, source_batch in val_loader:
            print(f"   EEG batch shape: {eeg_batch.shape}")
            print(f"   Source batch shape: {source_batch.shape}")
            break
        
        print("\n3. Test batch:")
        for eeg_batch, source_batch in test_loader:
            print(f"   EEG batch shape: {eeg_batch.shape}")
            print(f"   Source batch shape: {source_batch.shape}")
            break
        
        print("\n✓ Dataloader test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_structure():
    """Check if the required data structure exists"""
    print("=" * 70)
    print("Checking Data Structure")
    print("=" * 70)
    
    required_files = [
        "source/train_sample_source1.mat",
        "source/test_sample_source1.mat",
        "anatomy/leadfield_75_20k.mat",
    ]
    
    required_dirs = [
        "source/nmm_spikes",
        "source/nmm_spikes/a0",
        "source/nmm_spikes/a1",
    ]
    
    all_ok = True
    
    print("\nChecking required files:")
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
        if not exists:
            all_ok = False
    
    print("\nChecking required directories:")
    for dir_path in required_dirs:
        exists = os.path.exists(dir_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}")
        if exists and dir_path.startswith("source/nmm_spikes/a"):
            # Count files in directory
            import glob
            mat_files = glob.glob(os.path.join(dir_path, "*.mat"))
            print(f"      ({len(mat_files)} .mat files)")
        if not exists:
            all_ok = False
    
    if all_ok:
        print("\n✓ All required files and directories found!")
    else:
        print("\n✗ Some required files or directories are missing.")
        print("\nPlease ensure you have:")
        print("  1. Metadata files: train_sample_source1.mat, test_sample_source1.mat")
        print("  2. Forward matrix: anatomy/leadfield_75_20k.mat")
        print("  3. NMM spike data: source/nmm_spikes/a0/, a1/, etc.")
    
    return all_ok


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("METADATA-BASED LOADER TEST SUITE")
    print("=" * 70)
    
    # First check data structure
    structure_ok = check_data_structure()
    
    if not structure_ok:
        print("\n⚠️ Data structure check failed. Please fix the issues above.")
        exit(1)
    
    # Run tests
    test1_passed = test_metadata_dataset()
    test2_passed = test_metadata_dataloader()
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Data structure check: {'✓ PASSED' if structure_ok else '✗ FAILED'}")
    print(f"Dataset test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Dataloader test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    
    if structure_ok and test1_passed and test2_passed:
        print("\n🎉 All tests passed!")
        print("\nYou can now train the model using:")
        print("  python train.py")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

