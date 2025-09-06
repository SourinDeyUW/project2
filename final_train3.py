"""
Dual Graph GNN Architecture for Metal Oxides
============================================

This implementation uses two separate message passing networks:
1. Atomistic Graph MP: Global atomic connectivity with message passing
2. Polyhedral Graph MP: Polyhedron-level connectivity with message passing
3. Concatenation: Combine embeddings from both graphs
4. Final FNN: Feedforward neural network for band gap prediction

The architecture leverages both ato    # Load dataset
    dataset = OxideDataset(config['pickle_file'], config['csv_file'])
    
    # Create train/validation/test split
    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(
        indices, 
        train_size=config['train_split'], 
        random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=0.5,  # Split remaining data equally between val and test
        random_state=42
    )
    
    # Filter out samples without band gap data
    valid_train_indices = []
    valid_val_indices = []
    valid_test_indices = []
    
    for idx in train_indices:
        item = dataset.get_item(idx)
        if item['target'].item() >= 0:  # Include all band gaps >= 0
            valid_train_indices.append(idx)
    
    for idx in val_indices:
        item = dataset.get_item(idx)
        if item['target'].item() >= 0:
            valid_val_indices.append(idx)
            
    for idx in test_indices:
        item = dataset.get_item(idx)
        if item['target'].item() >= 0:
            valid_test_indices.append(idx)
    
    print(f"📊 Train samples: {len(valid_train_indices)}")
    print(f"📊 Validation samples: {len(valid_val_indices)}")
    print(f"📊 Test samples: {len(valid_test_indices)}")
    
    if len(valid_train_indices) == 0:
        print("❌ No valid training samples found!")
        returnl and polyhedral-level information
for improved metal oxide property prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected, add_self_loops
import numpy as np
import pandas as pd
import pickle
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm


class AtomisticGraphGNN(nn.Module):
    """
    Message Passing Network for Atomistic Graph
    Processes global atomic connectivity
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(AtomisticGraphGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers for atomistic message passing
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1))
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Global pooling
        self.global_pool = global_mean_pool
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch):
        """
        Forward pass for atomistic graph
        
        Args:
            x: Atomic features [num_atoms, input_dim]
            edge_index: Atomic connectivity [2, num_edges]
            batch: Batch indices for atoms
            
        Returns:
            Graph-level representation [batch_size, output_dim]
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        
        # Apply GNN layers with residual connections
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h_new = gnn_layer(h, edge_index)
            h_new = norm(h_new)
            h = F.relu(h_new) + h if i > 0 else F.relu(h_new)  # Skip residual for first layer
            h = self.dropout(h)
        
        # Global pooling to get graph-level representation
        graph_repr = self.global_pool(h, batch)
        
        # Output projection
        graph_repr = self.output_proj(graph_repr)
        
        return graph_repr


class PolyhedralGraphGNN(nn.Module):
    """
    Message Passing Network for Polyhedral Graph
    Processes polyhedron-level connectivity with edge attributes
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim=4, num_layers=3):
        super(PolyhedralGraphGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Edge-conditioned GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(EdgeConditionedConv(hidden_dim, hidden_dim, edge_dim))
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Global pooling
        self.global_pool = global_mean_pool
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass for polyhedral graph
        
        Args:
            x: Polyhedral features [num_polyhedra, input_dim]
            edge_index: Polyhedral connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            batch: Batch indices for polyhedra
            
        Returns:
            Graph-level representation [batch_size, output_dim]
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        
        # Apply edge-conditioned GNN layers
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h_new = gnn_layer(h, edge_index, edge_attr)
            h_new = norm(h_new)
            h = F.relu(h_new) + h if i > 0 else F.relu(h_new)  # Skip residual for first layer
            h = self.dropout(h)
        
        # Global pooling to get graph-level representation
        graph_repr = self.global_pool(h, batch)
        
        # Output projection
        graph_repr = self.output_proj(graph_repr)
        
        return graph_repr


class EdgeConditionedConv(nn.Module):
    """
    Edge-conditioned convolution layer for polyhedral graphs
    """
    
    def __init__(self, input_dim, output_dim, edge_dim):
        super(EdgeConditionedConv, self).__init__()
        
        # Message function incorporating edge attributes
        self.message_net = nn.Sequential(
            nn.Linear(input_dim * 2 + edge_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # Update function
        self.update_net = nn.Sequential(
            nn.Linear(input_dim + output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x, edge_index, edge_attr):
        """Forward pass with edge conditioning"""
        row, col = edge_index
        
        # Create messages with edge conditioning
        messages = self.message_net(torch.cat([x[row], x[col], edge_attr], dim=1))
        
        # Aggregate messages (sum aggregation)
        aggr_messages = torch.zeros_like(x)
        aggr_messages.index_add_(0, row, messages)
        
        # Update node features
        updated_x = self.update_net(torch.cat([x, aggr_messages], dim=1))
        
        return updated_x


class DualGraphGNN(nn.Module):
    """
    Dual Graph GNN combining Atomistic and Polyhedral representations
    """
    
    def __init__(self, atomistic_input_dim, polyhedral_input_dim, hidden_dim=128, 
                 embedding_dim=64, num_layers=3, edge_dim=4):
        super(DualGraphGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Atomistic graph encoder
        self.atomistic_gnn = AtomisticGraphGNN(
            input_dim=atomistic_input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers
        )
        
        # Polyhedral graph encoder
        self.polyhedral_gnn = PolyhedralGraphGNN(
            input_dim=polyhedral_input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            edge_dim=edge_dim,
            num_layers=num_layers
        )
        
        # Final feedforward network
        combined_dim = embedding_dim * 2  # Concatenated embeddings
        self.final_fnn = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, atomistic_data, polyhedral_data):
        """
        Forward pass through dual graph architecture
        
        Args:
            atomistic_data: Dict with 'x', 'edge_index', 'batch'
            polyhedral_data: Dict with 'x', 'edge_index', 'edge_attr', 'batch'
            
        Returns:
            Predicted band gap values [batch_size, 1]
        """
        # Get atomistic graph embedding
        atomistic_embedding = self.atomistic_gnn(
            atomistic_data['x'],
            atomistic_data['edge_index'],
            atomistic_data['batch']
        )
        
        # Get polyhedral graph embedding
        polyhedral_embedding = self.polyhedral_gnn(
            polyhedral_data['x'],
            polyhedral_data['edge_index'],
            polyhedral_data['edge_attr'],
            polyhedral_data['batch']
        )
        
        # Concatenate embeddings
        combined_embedding = torch.cat([atomistic_embedding, polyhedral_embedding], dim=1)
        
        # Final prediction
        prediction = self.final_fnn(combined_embedding)
        
        return prediction


class OxideDataset:
    """
    Dataset class for metal oxide structures with dual graph representation
    """
    
    def __init__(self, pickle_file, csv_file=None):
        """
        Initialize dataset
        
        Args:
            pickle_file: Path to processed oxide data pickle file
            csv_file: Path to CSV file with band gap values (optional)
        """
        self.pickle_file = pickle_file
        self.csv_file = csv_file
        self.data = None
        self.band_gap_dict = {}
        
        self._load_data()
        if csv_file:
            self._load_band_gaps()
    
    def _load_data(self):
        """Load processed oxide data from pickle file"""
        with open(self.pickle_file, 'rb') as f:
            self.data = pickle.load(f)
        print(f"✅ Loaded {len(self.data['processed_data'])} oxide structures")
    
    def _load_band_gaps(self):
        """Load band gap values from CSV file"""
        if os.path.exists(self.csv_file):
            df = pd.read_csv(self.csv_file)
            for _, row in df.iterrows():
                self.band_gap_dict[row['material_id']] = row['band_gap']
            print(f"✅ Loaded band gaps for {len(self.band_gap_dict)} materials")
        else:
            print(f"❌ CSV file not found: {self.csv_file}")
    
    def _create_atomistic_graph(self, intra_poly_graphs):
        """
        Create global atomistic graph from intra-polyhedral graphs
        
        Args:
            intra_poly_graphs: List of polyhedral atomic graphs
            
        Returns:
            Global atomistic graph with connectivity
        """
        # Combine all atomic features
        all_atomic_features = []
        all_edges = []
        atom_offset = 0
        
        for poly_graph in intra_poly_graphs:
            # Add atomic features
            all_atomic_features.append(poly_graph.x)
            
            # Add edges with offset
            poly_edges = poly_graph.edge_index + atom_offset
            all_edges.append(poly_edges)
            
            atom_offset += poly_graph.x.shape[0]
        
        # Concatenate all features and edges
        if all_atomic_features:
            combined_features = torch.cat(all_atomic_features, dim=0)
            if all_edges:
                combined_edges = torch.cat(all_edges, dim=1)
            else:
                # If no edges, create self-loops
                num_atoms = combined_features.shape[0]
                combined_edges = torch.stack([torch.arange(num_atoms), torch.arange(num_atoms)])
        else:
            # Empty structure
            combined_features = torch.zeros((1, 89))  # Default atomic feature size
            combined_edges = torch.zeros((2, 0), dtype=torch.long)
        
        # Add inter-polyhedral atomic connections (simplified)
        # For now, we'll add connections between centroids of nearby polyhedra
        # This is a simplified approach - could be enhanced with actual atomic distances
        
        return combined_features, combined_edges
    
    def get_item(self, idx):
        """
        Get a single data item with dual graph representation
        
        Args:
            idx: Index of the structure
            
        Returns:
            Dict with atomistic and polyhedral graph data
        """
        item = self.data['processed_data'][idx]
        
        # Extract material ID for band gap lookup
        material_id = os.path.basename(item['cif_filename']).split('.')[0]
        band_gap = self.band_gap_dict.get(material_id, 0.0)  # Default to 0 if not found
        
        # Create atomistic graph
        atomic_features, atomic_edges = self._create_atomistic_graph(item['intra_poly_graphs'])
        
        # Get polyhedral graph
        poly_graph = item['inter_poly_graph']
        
        return {
            'atomistic': {
                'x': atomic_features,
                'edge_index': atomic_edges,
            },
            'polyhedral': {
                'x': poly_graph.x,
                'edge_index': poly_graph.edge_index,
                'edge_attr': poly_graph.edge_attr if hasattr(poly_graph, 'edge_attr') and poly_graph.edge_attr is not None else torch.zeros((poly_graph.edge_index.shape[1], 4)),
            },
            'target': torch.tensor([band_gap], dtype=torch.float32),
            'material_id': material_id
        }
    
    def __len__(self):
        return len(self.data['processed_data'])


def collate_dual_graphs(batch):
    """
    Custom collate function for dual graph batching
    """
    atomistic_graphs = []
    polyhedral_graphs = []
    targets = []
    material_ids = []
    
    for item in batch:
        # Atomistic graph
        atomistic_graphs.append(Data(
            x=item['atomistic']['x'],
            edge_index=item['atomistic']['edge_index']
        ))
        
        # Polyhedral graph
        polyhedral_graphs.append(Data(
            x=item['polyhedral']['x'],
            edge_index=item['polyhedral']['edge_index'],
            edge_attr=item['polyhedral']['edge_attr']
        ))
        
        targets.append(item['target'])
        material_ids.append(item['material_id'])
    
    # Batch graphs using PyTorch Geometric's batching
    atomistic_batch = Batch.from_data_list(atomistic_graphs)
    polyhedral_batch = Batch.from_data_list(polyhedral_graphs)
    targets = torch.cat(targets, dim=0)
    
    return {
        'atomistic': {
            'x': atomistic_batch.x,
            'edge_index': atomistic_batch.edge_index,
            'batch': atomistic_batch.batch
        },
        'polyhedral': {
            'x': polyhedral_batch.x,
            'edge_index': polyhedral_batch.edge_index,
            'edge_attr': polyhedral_batch.edge_attr,
            'batch': polyhedral_batch.batch
        },
        'target': targets,
        'material_ids': material_ids
    }


def train_dual_graph_model():
    """
    Main training function for dual graph GNN
    """
    # Configuration
    config = {
        'pickle_file': 'processed_oxide_data.pkl',
        'csv_file': 'data/mp_formulas.csv',
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'hidden_dim': 128,
        'embedding_dim': 64,
        'num_layers': 3,
        'train_split': 0.8,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print(f"🚀 Starting Dual Graph GNN Training")
    print(f"📱 Device: {config['device']}")
    
    # Load dataset
    dataset = OxideDataset(config['pickle_file'], config['csv_file'])
    
    # Use entire dataset for training
    all_indices = list(range(len(dataset)))
    
    print(f"📊 Total samples: {len(all_indices)}")
    print(f"� Training on entire dataset")
    
    # Create data loaders with custom dataset wrapper
    class DualGraphDataset(torch.utils.data.Dataset):
        def __init__(self, indices, dataset):
            self.indices = indices
            self.dataset = dataset
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            return self.dataset.get_item(self.indices[idx])
    
    train_dataset_wrapper = DualGraphDataset(valid_train_indices, dataset)
    val_dataset_wrapper = DualGraphDataset(valid_val_indices, dataset)
    test_dataset_wrapper = DualGraphDataset(valid_test_indices, dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset_wrapper,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_dual_graphs
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset_wrapper,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_dual_graphs
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset_wrapper,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_dual_graphs
    )
    
    # Get feature dimensions from first sample
    sample = dataset.get_item(0)
    atomistic_input_dim = sample['atomistic']['x'].shape[1]
    polyhedral_input_dim = sample['polyhedral']['x'].shape[1]
    
    print(f"🔢 Atomistic feature dim: {atomistic_input_dim}")
    print(f"🔢 Polyhedral feature dim: {polyhedral_input_dim}")
    
    # Initialize model
    model = DualGraphGNN(
        atomistic_input_dim=atomistic_input_dim,
        polyhedral_input_dim=polyhedral_input_dim,
        hidden_dim=config['hidden_dim'],
        embedding_dim=config['embedding_dim'],
        num_layers=config['num_layers']
    ).to(config['device'])
    
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}")
        for batch in train_pbar:
            # Move to device
            atomistic_data = {k: v.to(config['device']) for k, v in batch['atomistic'].items()}
            polyhedral_data = {k: v.to(config['device']) for k, v in batch['polyhedral'].items()}
            targets = batch['target'].to(config['device'])
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(atomistic_data, polyhedral_data)
            loss = criterion(predictions.squeeze(), targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                atomistic_data = {k: v.to(config['device']) for k, v in batch['atomistic'].items()}
                polyhedral_data = {k: v.to(config['device']) for k, v in batch['polyhedral'].items()}
                targets = batch['target'].to(config['device'])
                
                predictions = model(atomistic_data, polyhedral_data)
                loss = criterion(predictions.squeeze(), targets)
                
                val_loss += loss.item()
                val_preds.extend(predictions.squeeze().cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate validation metrics
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_r2 = r2_score(val_targets, val_preds)
        
        print(f"Epoch {epoch+1:3d}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MAE: {val_mae:.4f}, Val R²: {val_r2:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'val_loss': avg_val_loss
            }, 'checkpoints/dual_graph_best_model.pth')
            print(f"💾 Saved best model (Val Loss: {avg_val_loss:.4f})")
        
        scheduler.step(avg_val_loss)
    
    # Final test evaluation
    print(f"\n🧪 Final Test Evaluation:")
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            atomistic_data = {k: v.to(config['device']) for k, v in batch['atomistic'].items()}
            polyhedral_data = {k: v.to(config['device']) for k, v in batch['polyhedral'].items()}
            targets = batch['target'].to(config['device'])
            
            predictions = model(atomistic_data, polyhedral_data)
            loss = criterion(predictions.squeeze(), targets)
            
            test_loss += loss.item()
            test_preds.extend(predictions.squeeze().cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    test_r2 = r2_score(test_targets, test_preds)
    
    print(f"📊 Test Results:")
    print(f"   Test Loss: {avg_test_loss:.4f}")
    print(f"   Test MAE: {test_mae:.4f} eV")
    print(f"   Test RMSE: {test_rmse:.4f} eV")
    print(f"   Test R²: {test_r2:.4f}")
    
    # Plot training curves and results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.scatter(val_targets, val_preds, alpha=0.6, label='Validation')
    plt.plot([min(val_targets), max(val_targets)], [min(val_targets), max(val_targets)], 'r--')
    plt.xlabel('True Band Gap (eV)')
    plt.ylabel('Predicted Band Gap (eV)')
    plt.title(f'Validation Results\nR² = {val_r2:.3f}, MAE = {val_mae:.3f} eV')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.scatter(test_targets, test_preds, alpha=0.6, color='green', label='Test')
    plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--')
    plt.xlabel('True Band Gap (eV)')
    plt.ylabel('Predicted Band Gap (eV)')
    plt.title(f'Test Results\nR² = {test_r2:.3f}, MAE = {test_mae:.3f} eV')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dual_graph_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"🎉 Training completed! Best validation loss: {best_val_loss:.4f}")
    print(f"📊 Final test metrics - MAE: {test_mae:.4f} eV, RMSE: {test_rmse:.4f} eV, R²: {test_r2:.4f}")


if __name__ == "__main__":
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Run training
    train_dual_graph_model()
