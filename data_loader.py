"""
Data preprocessing and loading for Two-Stage GNN
Converts CIF files to the required graph format with polyhedra information
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from collections import defaultdict, Counter
import pickle

# Add parent directory to import existing functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from extract_polyhedra_analysis import extract_polyhedra_from_cif
    from draw_graphs import get_polyhedron_sharing_pairs_verbose
except ImportError:
    try:
        from polyhedra_pipeline import extract_polyhedra_from_cif
        from draw_graphs import get_polyhedron_sharing_pairs_verbose
    except ImportError:
        print("Warning: Could not import functions from parent directory")
        print("Make sure extract_polyhedra_analysis.py and draw_graphs.py are accessible")

from pymatgen.core import Element
from pymatgen.core.structure import Structure


class PolyhedralDataProcessor:
    """
    Processes CIF files into the two-stage graph format required for the GNN
    """
    
    def __init__(self):
        # Element features for atomic nodes
        self.element_features = self._setup_element_features()
        
    def _setup_element_features(self):
        """Setup atomic features for all elements"""
        # Common elements in materials science
        elements = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'
        ]
        
        features = {}
        for i, elem_str in enumerate(elements):
            try:
                elem = Element(elem_str)
                # Create feature vector: [one_hot_encoding, electronegativity, atomic_radius, atomic_mass]
                one_hot = [0] * len(elements)
                one_hot[i] = 1
                
                # Get atomic properties (with fallback values)
                electronegativity = getattr(elem, 'X', 1.0) or 1.0
                atomic_radius = getattr(elem, 'atomic_radius', 1.0) or 1.0
                atomic_mass = getattr(elem, 'atomic_mass', 1.0) or 1.0
                
                # Normalize properties
                electronegativity = electronegativity / 4.0  # Max ~4.0
                atomic_radius = atomic_radius / 3.0  # Max ~3.0 Angstrom
                atomic_mass = atomic_mass / 250.0  # Max ~250 amu
                
                feature_vector = one_hot + [electronegativity, atomic_radius, atomic_mass]
                features[elem_str] = feature_vector
                
            except Exception as e:
                print(f"Warning: Could not process element {elem_str}: {e}")
                # Fallback feature vector
                one_hot = [0] * len(elements)
                one_hot[i] = 1
                features[elem_str] = one_hot + [0.0, 0.0, 0.0]
        
        return features
    
    def process_cif_file(self, cif_filename, target_property=None):
        """
        Process a single CIF file into the two-stage graph format
        
        Args:
            cif_filename: Path to CIF file
            target_property: Target property value (optional)
            
        Returns:
            dict containing processed data for the two-stage GNN
        """
        try:
            # Step 1: Extract polyhedra from CIF
            # Handle both full paths and filename+folder combinations
            if os.path.isabs(cif_filename):
                # Full path provided - split into directory and filename
                cifs_folder = os.path.dirname(cif_filename)
                cif_name = os.path.basename(cif_filename)
                polyhedra_counts, cn_counts, polyhedra_data = extract_polyhedra_from_cif(cif_name, cifs_folder)
            else:
                # Relative filename provided - use default cifs folder
                polyhedra_counts, cn_counts, polyhedra_data = extract_polyhedra_from_cif(cif_filename)
            
            if not polyhedra_data:
                print(f"No polyhedra found in {cif_filename}")
                return None
                
            print(f"Found {len(polyhedra_data)} polyhedra in {cif_filename}")
            
            # Step 2: Create intra-polyhedral graphs (Stage 1 input)
            intra_poly_graphs = []
            polyhedron_info = []
            
            for poly_idx, poly_data in enumerate(polyhedra_data):
                # Create subgraph for this polyhedron
                poly_graph = self._create_polyhedron_subgraph(poly_data, poly_idx)
                
                if poly_graph is not None:
                    intra_poly_graphs.append(poly_graph)
                    polyhedron_info.append({
                        'poly_idx': poly_idx,
                        'center_element': poly_data['element'],
                        'coordination_number': poly_data['cn'],
                        'center_site_index': poly_data.get('center_index', poly_idx),
                        'polyhedron_type': poly_data.get('type', 'Unknown')
                    })
            
            if not intra_poly_graphs:
                print(f"No valid polyhedron graphs created for {cif_filename}")
                return None
            
            # Step 3: Analyze inter-polyhedral connectivity
            face_tri_pairs, face_quad_pairs, edge_pairs, point_pairs = get_polyhedron_sharing_pairs_verbose(polyhedra_data)
            
            # Step 4: Create inter-polyhedral graph (Stage 2 input)
            inter_poly_graph = self._create_inter_polyhedron_graph(
                polyhedron_info, face_tri_pairs, face_quad_pairs, edge_pairs, point_pairs
            )
            
            # Step 5: Package data
            processed_data = {
                'cif_filename': cif_filename,
                'intra_poly_graphs': intra_poly_graphs,
                'inter_poly_graph': inter_poly_graph,
                'polyhedron_info': polyhedron_info,
                'target_property': target_property,
                'num_polyhedra': len(polyhedra_data)
            }
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing {cif_filename}: {e}")
            return None
    
    def _create_polyhedron_subgraph(self, poly_data, poly_idx):
        """Create a subgraph for a single polyhedron"""
        try:
            # Extract element information
            center_element = poly_data['element']
            neighbor_elements = poly_data['ve_elements']
            
            # Collect all elements (center + neighbors)
            all_elements = [center_element] + neighbor_elements
            
            # Create node features
            node_features = []
            for element in all_elements:
                if element in self.element_features:
                    features = self.element_features[element]
                else:
                    # Fallback for unknown elements
                    features = [0] * (len(next(iter(self.element_features.values()))))
                    print(f"Warning: Unknown element {element}, using zero features")
                
                node_features.append(features)
            
            node_features = torch.tensor(node_features, dtype=torch.float32)
            
            # Create edges (fully connected within polyhedron)
            num_atoms = len(all_elements)
            edge_index = []
            
            for i in range(num_atoms):
                for j in range(num_atoms):
                    if i != j:  # No self-loops
                        edge_index.append([i, j])
            
            if not edge_index:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            # Create PyTorch Geometric Data object
            poly_graph = Data(
                x=node_features,
                edge_index=edge_index,
                poly_idx=poly_idx,
                center_atom_idx=0,  # Center atom is always first
                num_atoms=num_atoms
            )
            
            return poly_graph
            
        except Exception as e:
            print(f"Error creating polyhedron subgraph: {e}")
            return None
    
    def _create_inter_polyhedron_graph(self, polyhedron_info, face_tri_pairs, face_quad_pairs, edge_pairs, point_pairs):
        """Create the inter-polyhedron connectivity graph"""
        try:
            num_polyhedra = len(polyhedron_info)
            
            # Initialize connectivity matrix
            connectivity_matrix = defaultdict(list)
            
            # Process different types of connections
            # Face-sharing (triangular faces)
            for pair, shared_atoms in face_tri_pairs.items():
                poly1, poly2 = pair
                if poly1 < num_polyhedra and poly2 < num_polyhedra:
                    connectivity_matrix[(poly1, poly2)].append('face_tri')
                    connectivity_matrix[(poly2, poly1)].append('face_tri')
            
            # Face-sharing (quadrilateral faces)
            for pair, shared_atoms in face_quad_pairs.items():
                poly1, poly2 = pair
                if poly1 < num_polyhedra and poly2 < num_polyhedra:
                    connectivity_matrix[(poly1, poly2)].append('face_quad')
                    connectivity_matrix[(poly2, poly1)].append('face_quad')
            
            # Edge-sharing
            for pair, shared_atoms in edge_pairs.items():
                poly1, poly2 = pair
                if poly1 < num_polyhedra and poly2 < num_polyhedra:
                    connectivity_matrix[(poly1, poly2)].append('edge')
                    connectivity_matrix[(poly2, poly1)].append('edge')
            
            # Point-sharing (corner-sharing)
            for pair, shared_atoms in point_pairs.items():
                poly1, poly2 = pair
                if poly1 < num_polyhedra and poly2 < num_polyhedra:
                    connectivity_matrix[(poly1, poly2)].append('point')
                    connectivity_matrix[(poly2, poly1)].append('point')
            
            # Create edge list and edge features
            edge_index = []
            edge_attr = []
            
            for (poly1, poly2), connection_types in connectivity_matrix.items():
                edge_index.append([poly1, poly2])
                
                # Create edge feature vector [face_tri, face_quad, edge, point]
                edge_features = [0, 0, 0, 0]
                for conn_type in connection_types:
                    if conn_type == 'face_tri':
                        edge_features[0] = 1
                    elif conn_type == 'face_quad':
                        edge_features[1] = 1
                    elif conn_type == 'edge':
                        edge_features[2] = 1
                    elif conn_type == 'point':
                        edge_features[3] = 1
                
                edge_attr.append(edge_features)
            
            # Convert to tensors
            if edge_index:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 4), dtype=torch.float32)
            
            # Initial node features for polyhedra (will be replaced by Stage 1 output)
            # For now, create placeholder features based on center atom
            node_features = []
            for poly_info in polyhedron_info:
                element = poly_info['center_element']
                cn = poly_info['coordination_number']
                
                # Simple initial features: center atom features + coordination number
                if element in self.element_features:
                    elem_features = self.element_features[element]
                else:
                    elem_features = [0] * (len(next(iter(self.element_features.values()))))
                
                # Add coordination number as additional feature
                poly_features = elem_features + [cn / 12.0]  # Normalize CN
                node_features.append(poly_features)
            
            node_features = torch.tensor(node_features, dtype=torch.float32)
            
            # Create inter-polyhedron graph
            inter_graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_polyhedra=num_polyhedra
            )
            
            return inter_graph
            
        except Exception as e:
            print(f"Error creating inter-polyhedron graph: {e}")
            return None


class PolyhedralDataset:
    """
    Dataset class for loading and batching polyhedral crystal data
    """
    
    def __init__(self, data_dir, dataset_csv=None):
        self.data_dir = data_dir
        self.processed_data = []
        self.processor = PolyhedralDataProcessor()
        
        # Load target properties if CSV is provided
        self.properties = {}
        if dataset_csv and os.path.exists(dataset_csv):
            df = pd.read_csv(dataset_csv)
            for _, row in df.iterrows():
                mp_id = row['mp_id']
                if isinstance(mp_id, str) and mp_id.startswith('cifs/'):
                    cif_filename = mp_id.replace('cifs/', '')
                    # You can add any target property here
                    # For now, using a placeholder
                    self.properties[cif_filename] = {
                        'reduced_formula': row.get('reduced_formula', 'Unknown'),
                        'prototype': row.get('prototype', 'Unknown')
                    }
    
    def __len__(self):
        """Return the number of processed structures"""
        return len(self.processed_data)
    
    def load_data(self, cif_files=None):
        """Load and process CIF files"""
        if cif_files is None:
            # Load all CIF files in the directory
            cif_files = [f for f in os.listdir(self.data_dir) if f.endswith('.cif')]
        
        print(f"Loading {len(cif_files)} CIF files...")
        
        for cif_file in cif_files:
            cif_path = os.path.join(self.data_dir, cif_file)
            
            # Get target property if available
            target_property = self.properties.get(cif_file, None)
            
            processed_data = self.processor.process_cif_file(cif_path, target_property)
            
            if processed_data is not None:
                self.processed_data.append(processed_data)
                print(f"Successfully processed {cif_file}")
            else:
                print(f"Failed to process {cif_file}")
        
        print(f"Successfully loaded {len(self.processed_data)} structures")
        return self.processed_data
    
    def load_preprocessed_data(self, processed_file='processed_data.pkl'):
        """
        Load pre-processed data from pickle file for fast training.
        
        Args:
            processed_file: Path to preprocessed pickle file
            
        Returns:
            List of processed structure data
        """
        print(f"Loading preprocessed data from {processed_file}")
        
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Preprocessed file {processed_file} not found!")
        
        with open(processed_file, 'rb') as f:
            data = pickle.load(f)
        
        processed_items = data['processed_data']
        metadata = data['metadata']
        
        print(f"Loaded {metadata['successful']} preprocessed structures")
        print(f"Success rate: {metadata['successful']/metadata['total_files']*100:.1f}%")
        print(f"Processing date: {metadata.get('processing_date', 'Unknown')}")
        
        # Extract just the data part
        self.processed_data = [item['data'] for item in processed_items]
        
        return self.processed_data
    
    def create_dataloaders_from_preprocessed(self, processed_file='processed_data.pkl', 
                                           train_split=0.8, batch_size=8, shuffle=True):
        """
        Create train/val dataloaders from preprocessed data.
        
        Args:
            processed_file: Path to preprocessed pickle file
            train_split: Fraction of data for training
            batch_size: Batch size for dataloaders
            shuffle: Whether to shuffle the data
            
        Returns:
            train_batches, val_batches (lists of batches)
        """
        # Load preprocessed data
        all_data = self.load_preprocessed_data(processed_file)
        
        if len(all_data) == 0:
            raise ValueError("No data loaded from preprocessed file!")
        
        # Split into train/val
        n_train = int(len(all_data) * train_split)
        
        if shuffle:
            import random
            random.seed(42)  # For reproducibility
            random.shuffle(all_data)
        
        train_data = all_data[:n_train]
        val_data = all_data[n_train:]
        
        print(f"Training set: {len(train_data)} structures")
        print(f"Validation set: {len(val_data)} structures")
        
        # Create batches
        train_batches = create_batches(train_data, batch_size)
        val_batches = create_batches(val_data, batch_size)
        
        print(f"Created {len(train_batches)} training batches")
        print(f"Created {len(val_batches)} validation batches")
        
        return train_batches, val_batches
    
    def save_processed_data(self, save_path):
        """Save processed data to disk"""
        with open(save_path, 'wb') as f:
            pickle.dump(self.processed_data, f)
        print(f"Saved processed data to {save_path}")
    
    def load_processed_data(self, load_path):
        """Load previously processed data"""
        with open(load_path, 'rb') as f:
            self.processed_data = pickle.load(f)
        print(f"Loaded {len(self.processed_data)} processed structures from {load_path}")
        return self.processed_data


def create_batches(processed_data, batch_size=32):
    """
    Create batches for training
    Note: Due to the two-stage nature, we need custom batching
    """
    batches = []
    
    for i in range(0, len(processed_data), batch_size):
        batch_data = processed_data[i:i+batch_size]
        
        # Collect all intra-poly graphs for this batch
        all_intra_graphs = []
        batch_sizes = []  # Track how many polyhedra per crystal
        
        for data in batch_data:
            intra_graphs = data['intra_poly_graphs']
            all_intra_graphs.extend(intra_graphs)
            batch_sizes.append(len(intra_graphs))
        
        # Batch intra-poly graphs
        if all_intra_graphs:
            intra_batch = Batch.from_data_list(all_intra_graphs)
        else:
            intra_batch = None
        
        # Collect inter-poly graphs
        inter_graphs = [data['inter_poly_graph'] for data in batch_data]
        
        batch = {
            'intra_batch': intra_batch,
            'inter_graphs': inter_graphs,
            'batch_sizes': batch_sizes,
            'cif_filenames': [data['cif_filename'] for data in batch_data],
            'target_properties': [data['target_property'] for data in batch_data]
        }
        
        batches.append(batch)
    
    return batches


if __name__ == "__main__":
    # Example usage
    data_dir = "../cifs"  # Adjust path as needed
    dataset_csv = "../dataset.csv"  # Adjust path as needed
    
    # Create dataset
    dataset = PolyhedralDataset(data_dir, dataset_csv)
    
    # Load a few files for testing
    test_files = ['mp-757245.cif', 'mp-757246.cif']  # Adjust as needed
    processed_data = dataset.load_data(test_files)
    
    # Create batches
    batches = create_batches(processed_data, batch_size=2)
    
    print(f"Created {len(batches)} batches")
    if batches:
        print(f"First batch contains {len(batches[0]['batch_sizes'])} crystals")
        print(f"Batch sizes: {batches[0]['batch_sizes']}")
