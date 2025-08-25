"""
Quick start script for Two-Stage GNN preprocessing and training
"""

import os
import sys
import subprocess
import argparse

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=False)
        print(f"✅ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Quick start for Two-Stage GNN')
    parser.add_argument('--data_dir', default='../cifs', help='Directory with CIF files')
    parser.add_argument('--dataset_csv', default='../dataset.csv', help='Dataset CSV file')
    parser.add_argument('--test_only', action='store_true', help='Only test preprocessing')
    parser.add_argument('--skip_preprocess', action='store_true', help='Skip preprocessing step')
    parser.add_argument('--max_files', type=int, default=None, help='Limit number of files')
    
    args = parser.parse_args()
    
    print("Two-Stage GNN Quick Start")
    print("=" * 40)
    print(f"Data directory: {args.data_dir}")
    print(f"Dataset CSV: {args.dataset_csv}")
    print(f"Test only: {args.test_only}")
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        print("Please ensure CIF files are in the correct location")
        return
    
    # Count CIF files
    cif_files = [f for f in os.listdir(args.data_dir) if f.endswith('.cif')]
    print(f"Found {len(cif_files)} CIF files")
    
    if len(cif_files) == 0:
        print("❌ No CIF files found!")
        return
    
    success = True
    
    if args.test_only:
        # Test preprocessing only
        success = run_command(
            "python preprocess_data.py --test",
            "Testing preprocessing on a few files"
        )
        
    elif not args.skip_preprocess:
        # Step 1: Preprocess data
        preprocess_cmd = f"python preprocess_data.py --data_dir {args.data_dir} --dataset_csv {args.dataset_csv}"
        if args.max_files:
            preprocess_cmd += f" --max_files {args.max_files}"
        
        success = run_command(
            preprocess_cmd,
            "Preprocessing dataset"
        )
        
        if success:
            # Analyze preprocessing results
            run_command(
                "python preprocess_data.py --analyze",
                "Analyzing preprocessing results"
            )
    
    if success and not args.test_only:
        # Step 2: Test model architecture
        success = run_command(
            "python test_model.py",
            "Testing model architecture"
        )
        
        if success:
            # Step 3: Train model (short test)
            train_cmd = "python train.py --use_preprocessed --epochs 5 --batch_size 4"
            success = run_command(
                train_cmd,
                "Testing training (5 epochs)"
            )
            
            if success:
                print(f"\n🎉 Quick start completed successfully!")
                print(f"\nNext steps:")
                print(f"1. Run full training: python train.py --use_preprocessed --epochs 100")
                print(f"2. Evaluate model: python evaluate.py --model_path checkpoints/best_model.pth --use_preprocessed")
            else:
                print(f"\n❌ Training test failed")
        else:
            print(f"\n❌ Model test failed")
    
    if not success:
        print(f"\n❌ Quick start failed at some step")
        print(f"Check the error messages above")

if __name__ == "__main__":
    main()
