"""
Script to view and analyze preprocessed pickle files
"""

import pickle
import sys
import os
from pprint import pprint

def view_pkl_file(pkl_file):
    """
    Load and display contents of a pickle file
    
    Args:
        pkl_file: Path to the pickle file
    """
    if not os.path.exists(pkl_file):
        print(f"File not found: {pkl_file}")
        return
    
    print(f"Loading pickle file: {pkl_file}")
    print("=" * 50)
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"File size: {os.path.getsize(pkl_file) / (1024*1024):.2f} MB")
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Dictionary keys: {list(data.keys())}")
            
            if 'metadata' in data:
                print("\nMetadata:")
                pprint(data['metadata'])
            
            if 'processed_data' in data:
                structures = data['processed_data']
                print(f"\nNumber of structures: {len(structures)}")
                
                if len(structures) > 0:
                    print("\n" + "="*60)
                    print("FIRST STRUCTURE DETAILS:")
                    print("="*60)
                    first_structure = structures[0]
                    
                    for key, value in first_structure.items():
                        print(f"\n{key.upper()}:")
                        if key == 'polyhedra_data' and isinstance(value, list):
                            print(f"  Number of polyhedra: {len(value)}")
                            for i, poly in enumerate(value[:3]):  # Show first 3 polyhedra
                                print(f"  Polyhedron {i+1}:")
                                if isinstance(poly, dict):
                                    for k, v in poly.items():
                                        if isinstance(v, list) and len(v) > 5:
                                            print(f"    {k}: List with {len(v)} items")
                                        else:
                                            print(f"    {k}: {v}")
                                else:
                                    print(f"    {poly}")
                            if len(value) > 3:
                                print(f"  ... and {len(value)-3} more polyhedra")
                                
                        elif key == 'graph_data' and isinstance(value, dict):
                            print(f"  Graph nodes: {len(value.get('nodes', []))}")
                            print(f"  Graph edges: {len(value.get('edges', []))}")
                            if 'nodes' in value and len(value['nodes']) > 0:
                                print(f"  First node: {value['nodes'][0]}")
                            if 'edges' in value and len(value['edges']) > 0:
                                print(f"  First edge: {value['edges'][0]}")
                                
                        elif isinstance(value, dict):
                            print(f"  Dictionary with {len(value)} keys:")
                            for k, v in list(value.items())[:5]:  # Show first 5 items
                                if isinstance(v, (list, tuple)) and len(v) > 5:
                                    print(f"    {k}: {type(v).__name__} with {len(v)} items")
                                else:
                                    print(f"    {k}: {v}")
                            if len(value) > 5:
                                print(f"    ... and {len(value)-5} more items")
                                
                        elif isinstance(value, (list, tuple)):
                            print(f"  {type(value).__name__} with {len(value)} items")
                            if len(value) > 0:
                                print(f"  First item: {value[0]}")
                                
                        else:
                            print(f"  {value}")
                
                # Show statistics for all structures
                print("\n" + "="*60)
                print("STATISTICS FOR ALL STRUCTURES:")
                print("="*60)
                
                polyhedra_counts = []
                atom_counts = []
                filenames = []
                
                for s in structures:
                    if 'polyhedra_data' in s:
                        polyhedra_counts.append(len(s['polyhedra_data']))
                    if 'num_atoms' in s:
                        atom_counts.append(s['num_atoms'])
                    if 'filename' in s:
                        filenames.append(s['filename'])
                
                if polyhedra_counts:
                    print(f"Polyhedra per structure:")
                    print(f"  Min: {min(polyhedra_counts)}, Max: {max(polyhedra_counts)}, Avg: {sum(polyhedra_counts)/len(polyhedra_counts):.1f}")
                
                if atom_counts:
                    print(f"Atoms per structure:")
                    print(f"  Min: {min(atom_counts)}, Max: {max(atom_counts)}, Avg: {sum(atom_counts)/len(atom_counts):.1f}")
                
                print(f"\nProcessed files:")
                for i, filename in enumerate(filenames[:10]):  # Show first 10 filenames
                    poly_count = polyhedra_counts[i] if i < len(polyhedra_counts) else "N/A"
                    atom_count = atom_counts[i] if i < len(atom_counts) else "N/A"
                    print(f"  {filename}: {poly_count} polyhedra, {atom_count} atoms")
                if len(filenames) > 10:
                    print(f"  ... and {len(filenames)-10} more files")
                    
                # Ask if user wants to see a specific structure
                print(f"\nEnter structure number (1-{len(structures)}) to view details, or 'q' to quit:")
                choice = input().strip()
                if choice.isdigit() and 1 <= int(choice) <= len(structures):
                    idx = int(choice) - 1
                    print(f"\n" + "="*60)
                    print(f"STRUCTURE {choice} DETAILS:")
                    print("="*60)
                    structure = structures[idx]
                    pprint(structure)
        
        elif isinstance(data, list):
            print(f"List with {len(data)} items")
            if len(data) > 0:
                print(f"First item type: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"First item keys: {list(data[0].keys())}")
        
        else:
            print("Data content:")
            pprint(data)
            
    except Exception as e:
        print(f"Error loading pickle file: {e}")

def main():
    if len(sys.argv) < 2:
        # List available pickle files in current directory
        pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
        if pkl_files:
            print("Available pickle files:")
            for i, f in enumerate(pkl_files, 1):
                size_mb = os.path.getsize(f) / (1024*1024)
                print(f"  {i}. {f} ({size_mb:.2f} MB)")
            
            choice = input("\nEnter file number or filename: ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(pkl_files):
                pkl_file = pkl_files[int(choice) - 1]
            else:
                pkl_file = choice
        else:
            print("No pickle files found in current directory.")
            print("Usage: python view_pkl.py <pickle_file>")
            return
    else:
        pkl_file = sys.argv[1]
    
    view_pkl_file(pkl_file)

if __name__ == "__main__":
    main()
