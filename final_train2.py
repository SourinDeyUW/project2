"""
Final Training Script for Two-Stage GNN (No Pooling after Step 1)
- Step 1: Intra-polyhedral message passing (NO pooling at the end)
- Step 2: Inter-polyhedral message passing (WITH pooling at the end)
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
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch

class TwoStageGNN_NoPoolStep1(nn.Module):
    """
    Two-Stage GNN: 
    Step 1: Intra-polyhedral MP (NO pooling at end)
    Step 2: Additional MP within polyhedra (WITH pooling at end)
    """
    def __init__(self, atom_input_dim=89, hidden_dim=128, output_dim=1):
        super().__init__()
        self.atom_input_dim = atom_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Step 1: First intra-polyhedral MP layers
        self.intra_conv1 = GCNConv(atom_input_dim, hidden_dim)
        self.intra_bn1 = nn.BatchNorm1d(hidden_dim)
        self.intra_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.intra_bn2 = nn.BatchNorm1d(hidden_dim)
        self.intra_conv3 = GCNConv(hidden_dim, hidden_dim)
        self.intra_bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Step 2: Additional MP within polyhedra (deeper processing)
        self.poly_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.poly_bn1 = nn.BatchNorm1d(hidden_dim)
        self.poly_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.poly_bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Prediction head
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
        intra_batch = batch_data['intra_batch']
        batch_sizes = batch_data['batch_sizes']
        
        x = intra_batch.x
        edge_index = intra_batch.edge_index
        batch = intra_batch.batch
        
        # Step 1: Initial intra-polyhedral MP (NO pooling)
        x = torch.relu(self.intra_bn1(self.intra_conv1(x, edge_index)))
        x = torch.relu(self.intra_bn2(self.intra_conv2(x, edge_index)))
        x = torch.relu(self.intra_bn3(self.intra_conv3(x, edge_index)))
        
        # Step 2: Additional MP within polyhedra (still using same graph structure)
        x = torch.relu(self.poly_bn1(self.poly_conv1(x, edge_index)))
        x = torch.relu(self.poly_bn2(self.poly_conv2(x, edge_index)))
        
        # Pool atoms to polyhedra, then polyhedra to crystals
        poly_representations = global_mean_pool(x, batch)
        
        # Pool polyhedra to crystals for final prediction
        crystal_predictions = []
        poly_start = 0
        
        for crystal_idx, num_polyhedra in enumerate(batch_sizes):
            poly_end = poly_start + num_polyhedra
            crystal_poly_reps = poly_representations[poly_start:poly_end]
            
            # Pool all polyhedra in this crystal
            crystal_rep = torch.mean(crystal_poly_reps, dim=0, keepdim=True)
            prediction = self.predictor(crystal_rep)
            crystal_predictions.append(prediction)
            
            poly_start = poly_end
        
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
            # Extract material ID from CIF filename - handle both Windows and Unix paths
            material_id = os.path.basename(cif_filename).replace('.cif', '')
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


def main():
    print("🚀 Two-Stage GNN Training (No Pool Step 1)")
    print("=" * 60)
    
    # Step 1: Load band gap data for oxide materials
    print("\n📊 Step 1: Loading band gap data for oxide materials...")
    band_gap_data = load_band_gap_data_for_oxides("oxide_cifs")
    if not band_gap_data:
        print("❌ Cannot proceed without band gap data!")
        return
    
    # Step 2: Load processed oxide data
    print("\n📂 Step 2: Loading processed oxide data...")
    processed_file = "processed_oxide_data.pkl"
    if os.path.exists(processed_file):
        all_structures = load_processed_oxide_data(processed_file)
    else:
        print("❌ Processed data file not found!")
        return
    
    if not all_structures:
        print("❌ No data loaded!")
        return
    
    # Step 3: Filter structures with band gap data
    print("\n🎯 Step 3: Filtering structures with band gap data...")
    structures_with_targets = []
    missing_targets = 0
    
    for structure in all_structures:
        cif_filename = structure['cif_filename']
        material_id = os.path.basename(cif_filename).replace('.cif', '')
        if material_id in band_gap_data:
            band_gap_value = band_gap_data[material_id]
            if isinstance(band_gap_value, (int, float)) and not np.isnan(band_gap_value):
                structures_with_targets.append(structure)
            else:
                missing_targets += 1
        else:
            missing_targets += 1
    
    print(f"✅ Structures with valid band gap data: {len(structures_with_targets)}")
    print(f"❌ Structures missing band gap data: {missing_targets}")
    
    if len(structures_with_targets) < 10:
        print("❌ Not enough structures with valid band gap data for training!")
        return
    
    # Step 4: Create train/validation split (use full dataset)
    print("\n📈 Step 4: Creating train/validation split...")
    # Use all available structures for comprehensive training
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
    batch_size = 16  # Larger batch size for full dataset
    train_batches = create_batches(train_data, batch_size)
    val_batches = create_batches(val_data, batch_size)
    
    print(f"Training batches: {len(train_batches)} (batch size: {batch_size})")
    print(f"Validation batches: {len(val_batches)}")
    
    # Step 6: Create model (No Pool Step 1)
    print("\n🧠 Step 6: Creating model (No Pool Step 1)...")
    atom_input_dim = 89
    model = TwoStageGNN_NoPoolStep1(
        atom_input_dim=atom_input_dim,
        hidden_dim=128,
        output_dim=1
    )
    
    # Step 7: Train model
    print("\n🏋️ Step 7: Training model...")
    trainer = ComprehensiveTrainer(model, band_gap_data)
    
    try:
        train_losses, val_losses, val_maes, val_rmses, val_r2s = trainer.train(
            train_batches, val_batches, num_epochs=50  # Reduced epochs for full dataset
        )
        
        print("\n🎉 Final Results:")
        print(f"   Final Train Loss: {train_losses[-1]:.4f}")
        print(f"   Final Val Loss: {val_losses[-1]:.4f}")
        print(f"   Final MAE: {val_maes[-1]:.4f} eV")
        print(f"   Final RMSE: {val_rmses[-1]:.4f} eV")
        print(f"   Final R²: {val_r2s[-1]:.4f}")
        print(f"   🎯 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
