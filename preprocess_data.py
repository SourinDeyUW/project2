"""
Dataset preprocessing script for Two-Stage GNN
Pre-processes CIF files to speed up training by caching expensive computations
"""

import os
import sys
import pickle
from tqdm import tqdm
import sys
import numpy as np
from datetime import datetime

# Import required functions from current directory
try:
    from extract_polyhedra_analysis import extract_polyhedra_from_cif
    from draw_graphs import draw_polyhedron_sharing_graph
    print("Successfully imported required functions")
except ImportError as e:
    print(f"Warning: Could not import functions: {e}")
    print("Make sure extract_polyhedra_analysis.py and draw_graphs.py are accessible")
    # Define dummy function to avoid NameError
    def extract_polyhedra_from_cif(*args, **kwargs):
        return None

from data_loader import PolyhedralDataProcessor

def preprocess_dataset(data_dir, dataset_csv, output_file='processed_data.pkl', 
                      force_reprocess=False, max_files=None):
    """
    Pre-process all CIF files and save to pickle for fast loading during training.
    
    Args:
        data_dir: Directory containing CIF files
        dataset_csv: CSV file with structure metadata
        output_file: Output pickle file path
        force_reprocess: Whether to reprocess even if output exists
        max_files: Maximum number of files to process (for testing)
    """
    
    if os.path.exists(output_file) and not force_reprocess:
        print(f"Processed data already exists at {output_file}")
        print("Use --force to reprocess, or delete the file to reprocess")
        return
    
    print("Starting dataset preprocessing...")
    print("This may take a while for large datasets...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize processor
    print("Initializing PolyhedralDataProcessor...")
    processor = PolyhedralDataProcessor()
    
    # Get list of CIF files (recursively search subdirectories)
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found!")
        return
        
    cif_files = []
    cif_paths = []
    
    # Walk through all subdirectories to find CIF files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.cif'):
                cif_files.append(file)
                cif_paths.append(os.path.join(root, file))
    
    print(f"Searched in {data_dir} and found CIF files in subdirectories")
    
    if max_files:
        cif_files = cif_files[:max_files]
        cif_paths = cif_paths[:max_files]
        print(f"Processing first {max_files} files for testing")
    
    print(f"Found {len(cif_files)} CIF files to process")
    
    if len(cif_files) == 0:
        print("No CIF files found! Check your data directory.")
        return
    
    # Process each file
    processed_data = []
    failed_files = []
    processing_times = []
    
    for i, (cif_file, cif_path) in enumerate(tqdm(zip(cif_files, cif_paths), desc="Processing CIF files", total=len(cif_files))):
        start_time = datetime.now()
        
        try:
            # Process the structure using full path
            data = processor.process_cif_file(cif_path)
            
            if data is not None:
                processed_data.append({
                    'filename': cif_file,
                    'data': data
                })
                
                processing_time = (datetime.now() - start_time).total_seconds()
                processing_times.append(processing_time)
                
                # Progress update every 10 files
                if (i + 1) % 10 == 0:
                    avg_time = np.mean(processing_times[-10:])
                    print(f"Processed {i+1}/{len(cif_files)} files. Avg time: {avg_time:.2f}s/file")
            else:
                failed_files.append(cif_file)
                print(f"Failed to process {cif_file}: No valid polyhedra found")
                
        except Exception as e:
            print(f"Failed to process {cif_file}: {str(e)}")
            failed_files.append(cif_file)
    
    # Processing summary
    print(f"\nProcessing Summary:")
    print(f"==================")
    print(f"Successfully processed: {len(processed_data)} structures")
    print(f"Failed to process: {len(failed_files)} structures")
    print(f"Success rate: {len(processed_data)/len(cif_files)*100:.1f}%")
    
    if processing_times:
        print(f"Average processing time: {np.mean(processing_times):.2f}s per file")
        print(f"Total processing time: {sum(processing_times):.1f}s ({sum(processing_times)/60:.1f} minutes)")
    
    if failed_files:
        print(f"\nFirst 10 failed files: {failed_files[:10]}")
    
    if len(processed_data) == 0:
        print("No structures were successfully processed!")
        return
    
    # Save to pickle
    print(f"\nSaving processed data to {output_file}...")
    
    save_data = {
        'processed_data': processed_data,
        'failed_files': failed_files,
        'metadata': {
            'total_files': len(cif_files),
            'successful': len(processed_data),
            'failed': len(failed_files),
            'processing_date': datetime.now().isoformat(),
            'data_dir': data_dir,
            'dataset_csv': dataset_csv,
            'avg_processing_time': np.mean(processing_times) if processing_times else 0
        }
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(save_data, f)
    
    file_size_mb = os.path.getsize(output_file) / (1024*1024)
    print(f"Preprocessing complete! Data saved to {output_file}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Average size per structure: {file_size_mb/len(processed_data):.3f} MB")

def analyze_preprocessing_results(processed_file='processed_data.pkl'):
    """Analyze the preprocessing results"""
    
    if not os.path.exists(processed_file):
        print(f"Preprocessed file {processed_file} not found!")
        return
    
    print(f"Loading preprocessed data from {processed_file}...")
    
    with open(processed_file, 'rb') as f:
        data = pickle.load(f)
    
    processed_data = data['processed_data']
    metadata = data['metadata']
    
    print(f"\nPreprocessing Analysis:")
    print(f"======================")
    print(f"Processing date: {metadata.get('processing_date', 'Unknown')}")
    print(f"Data directory: {metadata.get('data_dir', 'Unknown')}")
    print(f"Total files: {metadata['total_files']}")
    print(f"Successfully processed: {metadata['successful']}")
    print(f"Failed: {metadata['failed']}")
    print(f"Success rate: {metadata['successful']/metadata['total_files']*100:.1f}%")
    print(f"Average processing time: {metadata.get('avg_processing_time', 0):.2f}s per file")
    
    # Analyze structure sizes
    polyhedra_counts = []
    atom_counts = []
    connection_counts = []
    
    for item in processed_data:
        data_item = item['data']
        
        # Count polyhedra
        num_polyhedra = data_item['num_polyhedra']
        polyhedra_counts.append(num_polyhedra)
        
        # Count total atoms in intra-poly graphs
        total_atoms = 0
        for graph in data_item['intra_poly_graphs']:
            total_atoms += graph.x.size(0)
        atom_counts.append(total_atoms)
        
        # Count connections
        if data_item['inter_poly_graph'].edge_index.size(1) > 0:
            num_connections = data_item['inter_poly_graph'].edge_index.size(1) // 2  # Undirected edges
        else:
            num_connections = 0
        connection_counts.append(num_connections)
    
    print(f"\nStructure Statistics:")
    print(f"====================")
    print(f"Polyhedra per structure:")
    print(f"  Min: {min(polyhedra_counts)}")
    print(f"  Max: {max(polyhedra_counts)}")
    print(f"  Average: {np.mean(polyhedra_counts):.1f}")
    print(f"  Median: {np.median(polyhedra_counts):.1f}")
    
    print(f"\nAtoms per structure:")
    print(f"  Min: {min(atom_counts)}")
    print(f"  Max: {max(atom_counts)}")
    print(f"  Average: {np.mean(atom_counts):.1f}")
    print(f"  Median: {np.median(atom_counts):.1f}")
    
    print(f"\nConnections per structure:")
    print(f"  Min: {min(connection_counts)}")
    print(f"  Max: {max(connection_counts)}")
    print(f"  Average: {np.mean(connection_counts):.1f}")
    print(f"  Median: {np.median(connection_counts):.1f}")
    
    # Memory usage estimate
    file_size_mb = os.path.getsize(processed_file) / (1024*1024)
    print(f"\nMemory Usage:")
    print(f"=============")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Average per structure: {file_size_mb/len(processed_data):.3f} MB")
    print(f"Estimated RAM for full dataset: {file_size_mb:.1f} MB")

def test_preprocessing(data_dir='../cifs', max_files=3):
    """Test preprocessing on a few files"""
    print("Testing preprocessing on a few files...")
    
    # Convert to absolute path to avoid path issues
    data_dir = os.path.abspath(data_dir)
    test_output = 'test_processed_data.pkl'
    
    preprocess_dataset(
        data_dir=data_dir,
        dataset_csv='../dataset.csv',
        output_file=test_output,
        force_reprocess=True,
        max_files=max_files
    )
    
    if os.path.exists(test_output):
        print("\nAnalyzing test results...")
        analyze_preprocessing_results(test_output)
        
        # Clean up test file
        os.remove(test_output)
        print(f"\nTest complete! Removed {test_output}")
    else:
        print("Test failed - no output file created")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess crystal structure dataset for Two-Stage GNN')
    parser.add_argument('--data_dir', default='../cifs', 
                       help='Directory containing CIF files')
    parser.add_argument('--dataset_csv', default='../dataset.csv', 
                       help='CSV file with structure metadata')
    parser.add_argument('--output', default='processed_data.pkl', 
                       help='Output pickle file')
    parser.add_argument('--force', action='store_true', 
                       help='Force reprocessing even if output exists')
    parser.add_argument('--analyze', action='store_true', 
                       help='Analyze existing preprocessed data')
    parser.add_argument('--test', action='store_true', 
                       help='Test preprocessing on a few files')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    if args.test:
        test_preprocessing(args.data_dir, max_files=3)
    elif args.analyze:
        analyze_preprocessing_results(args.output)
    else:
        preprocess_dataset(
            data_dir=args.data_dir,
            dataset_csv=args.dataset_csv,
            output_file=args.output,
            force_reprocess=args.force,
            max_files=args.max_files
        )
