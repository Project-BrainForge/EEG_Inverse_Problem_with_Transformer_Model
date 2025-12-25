"""
Test script for the spike data loader
"""
import os
import numpy as np
from utils.loader import SpikeEEGDataset, create_dataloaders_from_spikes

def test_spike_dataset():
    """Test the SpikeEEGDataset class"""
    print("=" * 60)
    print("Testing SpikeEEGDataset")
    print("=" * 60)
    
    # Create a dummy forward matrix for testing
    fwd = np.random.randn(75, 994).astype(np.float32)
    
    # Check if nmm_spikes directory exists
    data_root = "."
    if not os.path.exists(os.path.join(data_root, "source", "nmm_spikes")):
        print("ERROR: source/nmm_spikes directory not found!")
        print("Please create the directory structure:")
        print("  source/nmm_spikes/a0/*.mat")
        print("  source/nmm_spikes/a1/*.mat")
        print("  ...")
        print("  source/nmm_spikes/a993/*.mat")
        return False
    
    try:
        # Create dataset
        dataset = SpikeEEGDataset(
            data_root=data_root,
            fwd=fwd,
            num_sources=2,
            patch_size=20,
            dataset_len=10,  # Small dataset for testing
            snr_range=(0, 30),
            normalize=True
        )
        
        print(f"\nDataset created successfully!")
        print(f"Dataset length: {len(dataset)}")
        print(f"Number of regions with data: {len(dataset.available_regions)}")
        
        # Test loading a sample
        print("\nLoading sample 0...")
        eeg_data, source_data = dataset[0]
        
        print(f"EEG data shape: {eeg_data.shape}")
        print(f"Source data shape: {source_data.shape}")
        print(f"EEG data range: [{eeg_data.min():.4f}, {eeg_data.max():.4f}]")
        print(f"Source data range: [{source_data.min():.4f}, {source_data.max():.4f}]")
        
        # Check data types
        assert eeg_data.shape == (500, 75), f"Expected EEG shape (500, 75), got {eeg_data.shape}"
        assert source_data.shape == (500, 994), f"Expected source shape (500, 994), got {source_data.shape}"
        
        print("\n✓ Dataset test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """Test the create_dataloaders_from_spikes function"""
    print("\n" + "=" * 60)
    print("Testing create_dataloaders_from_spikes")
    print("=" * 60)
    
    # Check if forward matrix exists
    fwd_path = "forward_matrix.mat"
    if not os.path.exists(fwd_path):
        print(f"WARNING: Forward matrix file '{fwd_path}' not found!")
        print("Creating a dummy forward matrix for testing...")
        from scipy.io import savemat
        dummy_fwd = np.random.randn(75, 994).astype(np.float32)
        savemat(fwd_path, {'fwd': dummy_fwd})
        print(f"Dummy forward matrix saved to {fwd_path}")
    
    try:
        train_loader, val_loader, test_loader, _ = create_dataloaders_from_spikes(
            data_root=".",
            fwd_matrix_path=fwd_path,
            batch_size=4,
            train_split=0.8,
            val_split=0.1,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            dataset_len=20,  # Small dataset for testing
            num_sources=2,
            patch_size=20,
            snr_range=(0, 30)
        )
        
        print(f"\nDataloaders created successfully!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test loading a batch
        print("\nLoading first training batch...")
        for eeg_batch, source_batch in train_loader:
            print(f"EEG batch shape: {eeg_batch.shape}")
            print(f"Source batch shape: {source_batch.shape}")
            break
        
        print("\n✓ Dataloader test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SPIKE DATA LOADER TEST SUITE")
    print("=" * 60)
    
    test1_passed = test_spike_dataset()
    test2_passed = test_dataloader()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Dataset test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Dataloader test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

