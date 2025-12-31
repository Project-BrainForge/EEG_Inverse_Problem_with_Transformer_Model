"""
Utility to clean up invalid checkpoints
"""
import os
import sys
from .checkpoint_validator import validate_checkpoint


def cleanup_invalid_checkpoints(checkpoint_dir="checkpoints", dry_run=True):
    """
    Remove invalid checkpoint files
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        dry_run: If True, only show what would be deleted without actually deleting
    """
    if not os.path.exists(checkpoint_dir):
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.endswith('.pt') and not f.startswith('best_model')]
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Scanning {len(checkpoint_files)} checkpoint(s) in {checkpoint_dir}")
    if dry_run:
        print("DRY RUN MODE - No files will be deleted")
    print('='*60)
    
    invalid_files = []
    
    for ckpt_file in sorted(checkpoint_files):
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        is_valid = validate_checkpoint(ckpt_path, verbose=False)
        
        if not is_valid:
            invalid_files.append((ckpt_file, ckpt_path))
            status = "Would delete" if dry_run else "Deleting"
            print(f"{status}: {ckpt_file}")
    
    if not invalid_files:
        print("\n✓ All checkpoints are valid - nothing to clean up")
        return
    
    print(f"\n{'='*60}")
    print(f"Found {len(invalid_files)} invalid checkpoint(s)")
    print('='*60)
    
    if dry_run:
        print("\nTo actually delete these files, run:")
        print(f"  python -m utils.cleanup_checkpoints --no-dry-run")
    else:
        print("\nDeleting invalid checkpoints...")
        deleted_count = 0
        for ckpt_file, ckpt_path in invalid_files:
            try:
                os.remove(ckpt_path)
                print(f"  ✓ Deleted: {ckpt_file}")
                deleted_count += 1
            except Exception as e:
                print(f"  ✗ Failed to delete {ckpt_file}: {e}")
        
        print(f"\n{'='*60}")
        print(f"Successfully deleted {deleted_count}/{len(invalid_files)} invalid checkpoint(s)")
        print('='*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up invalid checkpoint files')
    parser.add_argument('--dir', type=str, default='checkpoints', 
                       help='Directory containing checkpoints')
    parser.add_argument('--no-dry-run', action='store_true', 
                       help='Actually delete files (default is dry run)')
    
    args = parser.parse_args()
    
    cleanup_invalid_checkpoints(args.dir, dry_run=not args.no_dry_run)

