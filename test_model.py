"""
Example usage and testing script for the Two-Stage GNN
"""

import os
import sys
import torch
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import PolyhedralDataset, create_batches
from model import TwoStageGNN, count_parameters


def test_data_processing():
    """Test data processing pipeline"""
    print("Testing Data Processing Pipeline")
    print("=" * 40)
    
    # Setup paths (adjust as needed)
    data_dir = "../cifs"
    dataset_csv = "../dataset.csv"
    
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} not found")
        print("Please ensure CIF files are available")
        return False
    
    # Create dataset
    dataset = PolyhedralDataset(data_dir, dataset_csv)
    
    # Get available CIF files
    cif_files = [f for f in os.listdir(data_dir) if f.endswith('.cif')]
    if not cif_files:
        print("No CIF files found in data directory")
        return False
    
    print(f"Found {len(cif_files)} CIF files")
    
    # Process a few files for testing
    test_files = cif_files[:3]  # Test with first 3 files
    print(f"Testing with files: {test_files}")
    
    processed_data = dataset.load_data(test_files)
    
    if not processed_data:
        print("Failed to process any CIF files")
        return False
    
    print(f"Successfully processed {len(processed_data)} structures")
    
    # Show details for first structure
    first_structure = processed_data[0]
    print(f"\nFirst structure details:")
    print(f"  CIF file: {first_structure['cif_filename']}")
    print(f"  Number of polyhedra: {first_structure['num_polyhedra']}")
    print(f"  Intra-poly graphs: {len(first_structure['intra_poly_graphs'])}")
    
    # Test batching
    batches = create_batches(processed_data, batch_size=2)
    print(f"Created {len(batches)} batches")
    
    if batches:
        first_batch = batches[0]
        print(f"First batch contains {len(first_batch['batch_sizes'])} crystals")
        print(f"Batch sizes: {first_batch['batch_sizes']}")
    
    return True


def test_model_architecture():
    """Test model architecture and forward pass"""
    print("\nTesting Model Architecture")
    print("=" * 40)
    
    # Model parameters
    atom_input_dim = 89 + 3  # Element features + atomic properties
    
    # Create model
    model = TwoStageGNN(
        atom_input_dim=atom_input_dim,
        poly_hidden_dim=64,  # Smaller for testing
        inter_hidden_dim=64,
        output_dim=1,
        intra_layers=2,
        inter_layers=2,
        pooling='attention'
    )
    
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    
    # Create dummy intra-polyhedral graphs
    from torch_geometric.data import Data, Batch
    
    # Dummy polyhedron 1: 5 atoms (tetrahedral + center)
    poly1_x = torch.randn(5, atom_input_dim)
    poly1_edges = torch.tensor([[0,1,0,2,0,3,0,4,1,2,1,3,1,4,2,3,2,4,3,4],
                               [1,0,2,0,3,0,4,0,2,1,3,1,4,1,3,2,4,2,4,3]], dtype=torch.long)
    poly1 = Data(x=poly1_x, edge_index=poly1_edges, poly_idx=0)
    
    # Dummy polyhedron 2: 6 atoms (octahedral + center)  
    poly2_x = torch.randn(6, atom_input_dim)
    poly2_edges = torch.tensor([[0,1,0,2,0,3,0,4,0,5,1,2,1,3,1,4,1,5,2,3,2,4,2,5,3,4,3,5,4,5],
                               [1,0,2,0,3,0,4,0,5,0,2,1,3,1,4,1,5,1,3,2,4,2,5,2,4,3,5,3,5,4]], dtype=torch.long)
    poly2 = Data(x=poly2_x, edge_index=poly2_edges, poly_idx=1)
    
    # Batch intra-poly graphs
    intra_batch = Batch.from_data_list([poly1, poly2])
    
    # Dummy inter-polyhedral graph
    inter_x = torch.randn(2, atom_input_dim + 1)  # Initial polyhedron features
    inter_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
    inter_edge_attr = torch.tensor([[0, 0, 1, 0], [0, 0, 1, 0]], dtype=torch.float32)  # Edge-sharing
    inter_graph = Data(x=inter_x, edge_index=inter_edge_index, edge_attr=inter_edge_attr)
    
    # Create batch data
    batch_data = {
        'intra_batch': intra_batch,
        'inter_graphs': [inter_graph],
        'batch_sizes': [2],  # 2 polyhedra in this crystal
        'cif_filenames': ['test.cif'],
        'target_properties': [None]
    }
    
    # Test forward pass
    try:
        model.eval()
        with torch.no_grad():
            predictions = model(batch_data)
        
        print(f"Forward pass successful!")
        print(f"Output shape: {predictions.shape}")
        print(f"Prediction value: {predictions.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_setup():
    """Test training setup (without actual training)"""
    print("\nTesting Training Setup")
    print("=" * 40)
    
    from train import Trainer
    
    # Create model
    atom_input_dim = 89 + 3
    model = TwoStageGNN(
        atom_input_dim=atom_input_dim,
        poly_hidden_dim=32,  # Very small for testing
        inter_hidden_dim=32,
        output_dim=1,
        intra_layers=1,
        inter_layers=1,
        pooling='mean'  # Simpler pooling for testing
    )
    
    # Create trainer
    trainer = Trainer(model, device='cpu')  # Force CPU for testing
    trainer.setup_training(learning_rate=0.01)
    
    print("Trainer setup successful!")
    print(f"Optimizer: {type(trainer.optimizer).__name__}")
    print(f"Criterion: {type(trainer.criterion).__name__}")
    print(f"Scheduler: {type(trainer.scheduler).__name__}")
    
    return True


def main():
    """Run all tests"""
    print("Two-Stage GNN Testing Suite")
    print("=" * 50)
    
    tests = [
        ("Data Processing", test_data_processing),
        ("Model Architecture", test_model_architecture), 
        ("Training Setup", test_training_setup)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("-" * 30)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        print("\nAll tests passed! The Two-Stage GNN is ready to use.")
        print("\nNext steps:")
        print("1. Ensure your CIF files are in the ../cifs directory")
        print("2. Run: python train.py --epochs 10 --batch_size 4  (for a quick test)")
        print("3. Run: python evaluate.py --model_path checkpoints/best_model.pth")
    else:
        print(f"\n{len(tests) - passed} tests failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
