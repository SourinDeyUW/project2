"""
Final Training Script for Two-Stage GNN
- Combines all pickle files in the directory
- Removes duplicates based on material ID
- Uses band gap as target property
- Comprehensive training with validation
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import glob
from tqdm import tqdm

from data_loader import PolyhedralDataset, create_batches
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Batch


class SimplifiedTwoStageGNN(nn.Module):
    """
    Simplified Two-Stage GNN for demo purposes
    """
    
    def __init__(self, atom_input_dim=89, hidden_dim=64, output_dim=1):
        super(SimplifiedTwoStageGNN, self).__init__()
        
        self.atom_input_dim = atom_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Stage 1: Intra-polyhedral GNN (process atoms within polyhedra)
        self.intra_conv1 = GCNConv(atom_input_dim, hidden_dim)
        self.intra_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.intra_conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Stage 2: Inter-polyhedral GNN (process polyhedra interactions)
        self.inter_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.inter_conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Final prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        print(f"Model created with {self.count_parameters():,} parameters")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, batch_data):
        """Forward pass through the two-stage GNN"""
        intra_batch = batch_data['intra_batch']
        inter_graphs = batch_data['inter_graphs']
        batch_sizes = batch_data['batch_sizes']
        
        # Stage 1: Process intra-polyhedral graphs
        x = intra_batch.x
        edge_index = intra_batch.edge_index
        batch = intra_batch.batch
        
        # Apply intra-polyhedral convolutions
        x = torch.relu(self.intra_conv1(x, edge_index))
        x = torch.relu(self.intra_conv2(x, edge_index))
        x = torch.relu(self.intra_conv3(x, edge_index))
        
        # Pool to get polyhedron representations
        poly_representations = global_mean_pool(x, batch)
        
        # Stage 2: Process each crystal's inter-polyhedral graph
        crystal_predictions = []
        poly_start = 0
        
        for crystal_idx, num_polyhedra in enumerate(batch_sizes):
            # Get polyhedron representations for this crystal
            poly_end = poly_start + num_polyhedra
            crystal_poly_reps = poly_representations[poly_start:poly_end]
            
            # Get inter-polyhedral graph
            inter_graph = inter_graphs[crystal_idx]
            
            # Apply inter-polyhedral convolutions
            inter_x = crystal_poly_reps
            inter_edge_index = inter_graph.edge_index
            
            if inter_edge_index.size(1) > 0:  # Only if there are edges
                inter_x = torch.relu(self.inter_conv1(inter_x, inter_edge_index))
                inter_x = torch.relu(self.inter_conv2(inter_x, inter_edge_index))
            
            # Pool to get crystal representation
            crystal_rep = global_mean_pool(inter_x, torch.zeros(inter_x.size(0), dtype=torch.long, device=inter_x.device))
            
            # Make prediction
            prediction = self.predictor(crystal_rep)
            crystal_predictions.append(prediction)
            
            poly_start = poly_end
        
        # Stack predictions
        predictions = torch.cat(crystal_predictions, dim=0)
        return predictions


class ComprehensiveTrainer:
    """
    Comprehensive trainer with band gap targets
    """
    
    def __init__(self, model, band_gap_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.band_gap_data = band_gap_data
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        print(f"Using device: {device}")
        print(f"Band gap data loaded for {len(band_gap_data)} materials")
    
    def _extract_band_gap_targets(self, batch):
        """Extract band gap targets for batch"""
        targets = []
        
        for cif_filename in batch['cif_filenames']:
            # Extract material ID from CIF filename
            material_id = cif_filename.split('\\')[-1].split('.')[0]
            
            # Get band gap value (should always exist since we pre-filtered)
            band_gap = self.band_gap_data.get(material_id, None)
            
            if band_gap is not None:
                targets.append(band_gap)
            else:
                raise ValueError(f"❌ Missing band gap for {material_id} - this should not happen after filtering!")
        
        return torch.tensor(targets, dtype=torch.float32, device=self.device)
    
    def train_epoch(self, train_batches):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        #
        for batch_idx, batch in enumerate(train_batches):
            # Move data to device
            batch = self._move_batch_to_device(batch)
            
            # Extract band gap targets
            targets = self._extract_band_gap_targets(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            
            # Compute loss
            loss = self.criterion(predictions.squeeze(), targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f'  Batch {batch_idx+1}/{len(train_batches)}, Loss: {loss.item():.4f}')
        
        return total_loss / len(train_batches)
    
    def validate_epoch(self, val_batches):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_batches:
                # Move data to device
                batch = self._move_batch_to_device(batch)
                
                # Extract band gap targets
                targets = self._extract_band_gap_targets(batch)
                
                # Forward pass
                predictions = self.model(batch)
                
                # Compute loss
                loss = self.criterion(predictions.squeeze(), targets)
                total_loss += loss.item()
                
                # Store for metrics
                all_predictions.extend(predictions.squeeze().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        r2 = r2_score(all_targets, all_predictions)
        
        return total_loss / len(val_batches), mae, rmse, r2
    
    def _move_batch_to_device(self, batch):
        """Move batch data to device"""
        if batch['intra_batch'] is not None:
            batch['intra_batch'] = batch['intra_batch'].to(self.device)
        
        for i, inter_graph in enumerate(batch['inter_graphs']):
            batch['inter_graphs'][i] = inter_graph.to(self.device)
        
        return batch
    
    def train(self, train_batches, val_batches, num_epochs=50, save_dir='checkpoints'):
        """Main training loop"""
        os.makedirs(save_dir, exist_ok=True)
        
        train_losses = []
        val_losses = []
        val_maes = []
        val_rmses = []
        val_r2s = []
        best_val_loss = float('inf')
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training batches: {len(train_batches)}")
        print(f"Validation batches: {len(val_batches)}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(train_batches)
            train_losses.append(train_loss)
            
            # Validate
            val_loss, mae, rmse, r2 = self.validate_epoch(val_batches)
            val_losses.append(val_loss)
            val_maes.append(mae)
            val_rmses.append(rmse)
            val_r2s.append(r2)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, MAE: {mae:.4f} eV, RMSE: {rmse:.4f} eV, R²: {r2:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(os.path.join(save_dir, 'best_model.pth'), epoch, val_loss, mae, rmse, r2)
                print(f"💾 New best model saved! Val Loss: {val_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'), epoch, val_loss, mae, rmse, r2)
        
        # Plot final results
        self._plot_results(train_losses, val_losses, val_maes, val_rmses, val_r2s, save_dir)
        
        print("\n✅ Training completed successfully!")
        return train_losses, val_losses, val_maes, val_rmses, val_r2s
    
    def save_model(self, filepath, epoch, val_loss, mae, rmse, r2):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'model_config': {
                'atom_input_dim': self.model.atom_input_dim,
                'hidden_dim': self.model.hidden_dim,
                'output_dim': self.model.output_dim
            }
        }
        torch.save(checkpoint, filepath)
    
    def _plot_results(self, train_losses, val_losses, val_maes, val_rmses, val_r2s, save_dir):
        """Plot comprehensive training results"""
        plt.figure(figsize=(20, 8))
        
        # Loss curves
        plt.subplot(2, 3, 1)
        plt.plot(train_losses, label='Train Loss', marker='o', alpha=0.7)
        plt.plot(val_losses, label='Val Loss', marker='s', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # MAE
        plt.subplot(2, 3, 2)
        plt.plot(val_maes, label='MAE', marker='d', color='green', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error (eV)')
        plt.title('Validation MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # RMSE
        plt.subplot(2, 3, 3)
        plt.plot(val_rmses, label='RMSE', marker='^', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Root Mean Square Error (eV)')
        plt.title('Validation RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # R²
        plt.subplot(2, 3, 4)
        plt.plot(val_r2s, label='R²', marker='*', color='purple', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('R² Score')
        plt.title('Validation R²')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Combined metrics
        plt.subplot(2, 3, 5)
        plt.plot(val_maes, label='MAE', alpha=0.7)
        plt.plot(val_rmses, label='RMSE', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Error (eV)')
        plt.title('Error Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Final summary
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.8, f"Final Results:", fontsize=14, fontweight='bold')
        plt.text(0.1, 0.7, f"MAE: {val_maes[-1]:.4f} eV", fontsize=12)
        plt.text(0.1, 0.6, f"RMSE: {val_rmses[-1]:.4f} eV", fontsize=12)
        plt.text(0.1, 0.5, f"R²: {val_r2s[-1]:.4f}", fontsize=12)
        plt.text(0.1, 0.4, f"Best MAE: {min(val_maes):.4f} eV", fontsize=12)
        plt.text(0.1, 0.3, f"Best RMSE: {min(val_rmses):.4f} eV", fontsize=12)
        plt.text(0.1, 0.2, f"Best R²: {max(val_r2s):.4f}", fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Final Summary')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=300, bbox_inches='tight')
        plt.show()


def load_band_gap_data_for_oxides(oxide_dir="oxide_cifs"):
    """Load band gap data for oxide materials by matching filenames"""
    try:
        # First try to load from existing CSV
        band_gap_data = load_band_gap_data_csv("data/mp_formulas.csv")
        
        if not band_gap_data:
            print("❌ No band gap data found in CSV")
            return {}
        
        # Get all material IDs from CIF files
        import glob
        cif_files = glob.glob(f"{oxide_dir}/*.cif")
        oxide_material_ids = set()
        
        for cif_file in cif_files:
            material_id = os.path.basename(cif_file).replace('.cif', '')
            oxide_material_ids.add(material_id)
        
        print(f"🔍 Found {len(cif_files)} oxide CIF files")
        print(f"📊 Available band gap data for {len(band_gap_data)} total materials")
        
        # Filter band gap data to only include oxide materials
        oxide_band_gap_data = {}
        for material_id in oxide_material_ids:
            if material_id in band_gap_data:
                oxide_band_gap_data[material_id] = band_gap_data[material_id]
        
        print(f"✅ Matched band gap data for {len(oxide_band_gap_data)} oxide materials")
        return oxide_band_gap_data
        
    except Exception as e:
        print(f"❌ Error loading band gap data for oxides: {e}")
        return {}


def load_band_gap_data_csv(csv_path="data/mp_formulas.csv"):
    """Load band gap data from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        band_gap_dict = dict(zip(df['material_id'], df['band_gap']))
        print(f"✅ Loaded band gap data for {len(band_gap_dict)} materials from CSV")
        return band_gap_dict
    except FileNotFoundError:
        print(f"❌ Error: {csv_path} not found!")
        return {}
    except Exception as e:
        print(f"❌ Error loading band gap data: {e}")
        return {}


def process_oxide_cifs(oxide_dir="oxide_cifs", max_files=None, save_processed=True, processed_file="processed_oxide_data.pkl"):
    """Process oxide CIF files directly with option to save processed data"""
    import glob
    from pymatgen.io.cif import CifParser
    from pymatgen.analysis.local_env import CrystalNN
    from tqdm import tqdm
    import pickle
    from datetime import datetime
    
    # Check if processed file already exists
    if save_processed and os.path.exists(processed_file):
        print(f"🔍 Found existing processed data file: {processed_file}")
        response = input("Load existing data? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return load_processed_oxide_data(processed_file)
    
    cif_files = glob.glob(f"{oxide_dir}/*.cif")
    
    if not cif_files:
        print(f"❌ No CIF files found in {oxide_dir}")
        return []
    
    print(f"📁 Found {len(cif_files)} oxide CIF files")
    
    # Limit number of files if specified
    if max_files is not None and len(cif_files) > max_files:
        print(f"🔄 Processing first {max_files} files for demo")
        cif_files = cif_files[:max_files]
    else:
        print(f"🔄 Processing all {len(cif_files)} oxide CIF files")
    
    all_data = []
    failed_count = 0
    
    for cif_file in tqdm(cif_files, desc="Processing oxide CIFs"):
        try:
            # Convert to absolute path
            abs_cif_path = os.path.abspath(cif_file)
            
            # Create polyhedral data using the processor directly
            polyhedral_data = create_polyhedral_data_from_structure(None, abs_cif_path)
            
            if polyhedral_data is not None:
                all_data.append(polyhedral_data)
            else:
                failed_count += 1
                
        except Exception as e:
            failed_count += 1
            continue
    
    print(f"\n📈 Summary:")
    print(f"  Successfully processed: {len(all_data)} structures")
    print(f"  Failed to process: {failed_count} structures")
    
    # Save processed data if requested
    if save_processed and all_data:
        save_processed_oxide_data(all_data, processed_file, len(cif_files), len(all_data))
    
    return all_data


def save_processed_oxide_data(processed_data, save_path, total_files, successful_files):
    """Save processed oxide data to pickle file"""
    try:
        from datetime import datetime
        
        # Create metadata
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'total_files': total_files,
            'successful': successful_files,
            'failed': total_files - successful_files,
            'success_rate': successful_files / total_files * 100,
            'data_type': 'oxide_structures',
            'source': 'Materials Project binary/ternary metal oxides'
        }
        
        # Package data
        save_data = {
            'metadata': metadata,
            'processed_data': processed_data
        }
        
        # Save to pickle
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"\n💾 Processed oxide data saved to: {save_path}")
        print(f"   📊 Structures: {successful_files}")
        print(f"   📈 Success rate: {metadata['success_rate']:.1f}%")
        print(f"   🗓️  Date: {metadata['processing_date']}")
        
    except Exception as e:
        print(f"❌ Error saving processed data: {e}")


def load_processed_oxide_data(file_path):
    """Load processed oxide data from pickle file"""
    try:
        print(f"📖 Loading processed oxide data from: {file_path}")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        metadata = data['metadata']
        processed_data = data['processed_data']
        
        print(f"✅ Loaded {len(processed_data)} processed oxide structures")
        print(f"   📊 Original processing: {metadata['successful']}/{metadata['total_files']} structures")
        print(f"   📈 Success rate: {metadata['success_rate']:.1f}%")
        print(f"   🗓️  Processing date: {metadata.get('processing_date', 'Unknown')}")
        print(f"   🔬 Data type: {metadata.get('data_type', 'Unknown')}")
        
        return processed_data
        
    except Exception as e:
        print(f"❌ Error loading processed data: {e}")
        return []


def create_polyhedral_data_from_structure(structure, cif_filename):
    """Create polyhedral data from pymatgen structure matching expected format"""
    try:
        from data_loader import PolyhedralDataProcessor
        
        # Use the existing PolyhedralDataProcessor to properly process the structure
        processor = PolyhedralDataProcessor()
        result = processor.process_cif_file(cif_filename, target_property=None)
        
        if result is not None:
            return result
        else:
            return None
        
    except Exception as e:
        print(f"Error processing {cif_filename}: {e}")
        return None


def combine_pickle_files(pickle_pattern="processed_data_*.pkl"):
    """
    Combine all pickle files and remove duplicates
    """
    print("🔍 Searching for pickle files...")
    pickle_files = glob.glob(pickle_pattern)
    
    if not pickle_files:
        print(f"❌ No pickle files found matching pattern: {pickle_pattern}")
        return None
    
    print(f"📂 Found {len(pickle_files)} pickle files: {pickle_files}")
    
    all_data = []
    seen_materials = set()
    total_files = 0
    successful_files = 0
    
    for pkl_file in sorted(pickle_files):
        print(f"📖 Loading {pkl_file}...")
        
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            metadata = data['metadata']
            processed_items = data['processed_data']
            
            print(f"  📊 {pkl_file}: {metadata['successful']}/{metadata['total_files']} structures")
            total_files += metadata['total_files']
            
            # Add unique structures
            for item in processed_items:
                structure_data = item['data']
                cif_filename = structure_data['cif_filename']
                material_id = cif_filename.split('\\')[-1].split('.')[0]
                
                if material_id not in seen_materials:
                    all_data.append(structure_data)
                    seen_materials.add(material_id)
                    successful_files += 1
                else:
                    print(f"  ⚠️  Duplicate found: {material_id}")
            
        except Exception as e:
            print(f"  ❌ Error loading {pkl_file}: {e}")
    
    print(f"\n📈 Summary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Unique structures: {successful_files}")
    print(f"  Duplicates removed: {total_files - successful_files}")
    
    return all_data


def main():
    print("🚀 Comprehensive Two-Stage GNN Training")
    print("=" * 60)
    
    # Step 1: Load band gap data for oxide materials
    print("\n📊 Step 1: Loading band gap data for oxide materials...")
    band_gap_data = load_band_gap_data_for_oxides("oxide_cifs")
    if not band_gap_data:
        print("❌ Cannot proceed without band gap data!")
        return
    
    # Step 2: Process oxide CIF files directly (with option to save/load processed data)
    print("\n📂 Step 2: Processing oxide CIF files...")
    all_structures = process_oxide_cifs("oxide_cifs", max_files=None, save_processed=True)  # Use all files and save processed data
    if not all_structures:
        print("❌ No data loaded from oxide CIF files!")
        return
    
    # Step 3: Filter structures with band gap data (strict filtering)
    print("\n🎯 Step 3: Filtering structures with band gap data...")
    structures_with_targets = []
    missing_targets = 0
    invalid_bandgaps = 0
    
    for structure in all_structures:
        cif_filename = structure['cif_filename']
        material_id = cif_filename.split('\\')[-1].split('.')[0]
        material_id=material_id.split('/')[-1]
        if material_id in band_gap_data:
            band_gap_value = band_gap_data[material_id]
            
            # Additional validation: ensure band gap is a valid number
            if isinstance(band_gap_value, (int, float)) and not np.isnan(band_gap_value):
                structures_with_targets.append(structure)
            else:
                invalid_bandgaps += 1
                print(f"⚠️  Invalid band gap for {material_id}: {band_gap_value}")
        else:
            missing_targets += 1
    
    print(f"✅ Structures with valid band gap data: {len(structures_with_targets)}")
    print(f"❌ Structures missing band gap data: {missing_targets}")
    print(f"⚠️  Structures with invalid band gap values: {invalid_bandgaps}")
    
    if len(structures_with_targets) < 10:
        print("❌ Not enough structures with valid band gap data for training!")
        return
    
    # Step 4: Create train/validation split
    print("\n📈 Step 4: Creating train/validation split...")
    train_data, val_data = train_test_split(
        structures_with_targets, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )
    
    print(f"Training set: {len(train_data)} structures")
    print(f"Validation set: {len(val_data)} structures")
    
    # Step 5: Create batches
    print("\n🔄 Step 5: Creating batches...")
    batch_size = min(16, len(train_data) // 20)  # Larger batch size for more data
    train_batches = create_batches(train_data, batch_size)
    val_batches = create_batches(val_data, batch_size)
    
    print(f"Training batches: {len(train_batches)} (batch size: {batch_size})")
    print(f"Validation batches: {len(val_batches)}")
    
    # Step 6: Create model
    print("\n🧠 Step 6: Creating model...")
    atom_input_dim = 89  # Element features from preprocessing
    model = SimplifiedTwoStageGNN(
        atom_input_dim=atom_input_dim,
        hidden_dim=128,  # Larger model for better performance
        output_dim=1
    )
    
    # Step 7: Train model
    print("\n🏋️ Step 7: Training model...")
    trainer = ComprehensiveTrainer(model, band_gap_data)
    
    try:
        train_losses, val_losses, val_maes, val_rmses, val_r2s = trainer.train(
            train_batches, val_batches, num_epochs=50  # More epochs for full dataset
        )
        
        print("\n🎉 Final Results:")
        print(f"   Final Train Loss: {train_losses[-1]:.4f}")
        print(f"   Final Val Loss: {val_losses[-1]:.4f}")
        print(f"   Final MAE: {val_maes[-1]:.4f} eV")
        print(f"   Final RMSE: {val_rmses[-1]:.4f} eV")
        print(f"   Final R²: {val_r2s[-1]:.4f}")
        print(f"   Best MAE: {min(val_maes):.4f} eV")
        print(f"   Best RMSE: {min(val_rmses):.4f} eV")
        print(f"   Best R²: {max(val_r2s):.4f}")
        print(f"   🎯 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
