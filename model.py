"""
Two-Stage GNN Architecture Implementation

This module implements the hierarchical two-stage GNN described in the blueprint:
1. Stage 1: Intra-polyhedral encoding (atom-level message passing within polyhedra)
2. Stage 2: Inter-polyhedral message passing (polyhedron-level message passing)
3. Final prediction head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_mean, scatter_add, scatter_max


class IntraPolyhedralGNN(nn.Module):
    """
    Stage 1: Intra-polyhedral encoding
    Processes atoms within each polyhedron to create polyhedron-level representations
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, pooling='attention'):
        super(IntraPolyhedralGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers for intra-polyhedral message passing
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            else:
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
        
        # Pooling mechanism
        if pooling == 'attention':
            self.attention_weights = nn.Linear(hidden_dim, 1)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Normalization and dropout
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index, batch, poly_batch_indices):
        """
        Forward pass for intra-polyhedral encoding
        
        Args:
            x: Node features [num_atoms, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices for atoms
            poly_batch_indices: Indices mapping atoms to polyhedra
            
        Returns:
            Polyhedron representations [num_polyhedra, output_dim]
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        
        # Apply GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            h_new = gnn_layer(h, edge_index)
            h_new = self.layer_norms[i](h_new)
            h = F.relu(h_new) + h  # Residual connection
            h = self.dropout(h)
        
        # Pool atoms to get polyhedron representations
        if self.pooling == 'mean':
            poly_repr = scatter_mean(h, poly_batch_indices, dim=0)
        elif self.pooling == 'sum':
            poly_repr = scatter_add(h, poly_batch_indices, dim=0)
        elif self.pooling == 'max':
            poly_repr, _ = scatter_max(h, poly_batch_indices, dim=0)
        elif self.pooling == 'attention':
            # Attention-based pooling
            attention_scores = self.attention_weights(h)  # [num_atoms, 1]
            attention_weights = F.softmax(attention_scores, dim=0)
            
            # Apply attention weights and pool
            weighted_features = h * attention_weights
            poly_repr = scatter_add(weighted_features, poly_batch_indices, dim=0)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Output projection
        poly_repr = self.output_proj(poly_repr)
        
        return poly_repr


class EdgeConditionedConv(MessagePassing):
    """
    Edge-conditioned convolution for inter-polyhedral message passing
    Incorporates edge attributes (connection types) into message passing
    """
    
    def __init__(self, input_dim, output_dim, edge_dim):
        super(EdgeConditionedConv, self).__init__(aggr='add')
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        
        # Message function
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
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            
        Returns:
            Updated node features [num_nodes, output_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        """
        Create messages between connected nodes
        
        Args:
            x_i: Features of target nodes [num_edges, input_dim]
            x_j: Features of source nodes [num_edges, input_dim]
            edge_attr: Edge attributes [num_edges, edge_dim]
            
        Returns:
            Messages [num_edges, output_dim]
        """
        # Concatenate source, target, and edge features
        message_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        message = self.message_net(message_input)
        return message
    
    def update(self, aggr_out, x):
        """
        Update node features
        
        Args:
            aggr_out: Aggregated messages [num_nodes, output_dim]
            x: Original node features [num_nodes, input_dim]
            
        Returns:
            Updated node features [num_nodes, output_dim]
        """
        update_input = torch.cat([x, aggr_out], dim=1)
        updated_x = self.update_net(update_input)
        return updated_x


class InterPolyhedralGNN(nn.Module):
    """
    Stage 2: Inter-polyhedral message passing
    Processes polyhedron-level graph to capture global crystal structure
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim=4, num_layers=3):
        super(InterPolyhedralGNN, self).__init__()
        
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
            if i == 0:
                self.gnn_layers.append(EdgeConditionedConv(hidden_dim, hidden_dim, edge_dim))
            else:
                self.gnn_layers.append(EdgeConditionedConv(hidden_dim, hidden_dim, edge_dim))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Normalization and dropout
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for inter-polyhedral message passing
        
        Args:
            x: Polyhedron features [num_polyhedra, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            
        Returns:
            Updated polyhedron features [num_polyhedra, output_dim]
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        
        # Apply edge-conditioned GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            h_new = gnn_layer(h, edge_index, edge_attr)
            h_new = self.layer_norms[i](h_new)
            h = F.relu(h_new) + h  # Residual connection
            h = self.dropout(h)
        
        # Output projection
        h = self.output_proj(h)
        
        return h


class GlobalAttentionPooling(nn.Module):
    """
    Global attention pooling for crystal-level representation
    """
    
    def __init__(self, input_dim):
        super(GlobalAttentionPooling, self).__init__()
        
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
    
    def forward(self, x, batch):
        """
        Global attention pooling
        
        Args:
            x: Node features [num_nodes, input_dim]
            batch: Batch indices [num_nodes]
            
        Returns:
            Graph-level features [batch_size, input_dim]
        """
        # Compute attention scores
        attention_scores = self.attention_net(x)  # [num_nodes, 1]
        
        # Apply softmax per graph in batch
        attention_weights = torch.zeros_like(attention_scores)
        for i in range(batch.max().item() + 1):
            mask = (batch == i)
            if mask.sum() > 0:
                attention_weights[mask] = F.softmax(attention_scores[mask], dim=0)
        
        # Weighted sum
        weighted_features = x * attention_weights
        graph_features = scatter_add(weighted_features, batch, dim=0)
        
        return graph_features


class TwoStageGNN(nn.Module):
    """
    Complete two-stage GNN architecture
    """
    
    def __init__(self, 
                 atom_input_dim,
                 poly_hidden_dim=128,
                 inter_hidden_dim=128,
                 output_dim=1,
                 intra_layers=3,
                 inter_layers=3,
                 pooling='attention'):
        super(TwoStageGNN, self).__init__()
        
        self.atom_input_dim = atom_input_dim
        self.poly_hidden_dim = poly_hidden_dim
        self.inter_hidden_dim = inter_hidden_dim
        self.output_dim = output_dim
        
        # Stage 1: Intra-polyhedral GNN
        self.intra_gnn = IntraPolyhedralGNN(
            input_dim=atom_input_dim,
            hidden_dim=poly_hidden_dim,
            output_dim=poly_hidden_dim,
            num_layers=intra_layers,
            pooling=pooling
        )
        
        # Stage 2: Inter-polyhedral GNN
        self.inter_gnn = InterPolyhedralGNN(
            input_dim=poly_hidden_dim,
            hidden_dim=inter_hidden_dim,
            output_dim=inter_hidden_dim,
            edge_dim=4,  # [face_tri, face_quad, edge, point]
            num_layers=inter_layers
        )
        
        # Global pooling
        self.global_pooling = GlobalAttentionPooling(inter_hidden_dim)
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(inter_hidden_dim, inter_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(inter_hidden_dim // 2, inter_hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(inter_hidden_dim // 4, output_dim)
        )
    
    def forward(self, batch_data):
        """
        Forward pass through the complete two-stage architecture
        
        Args:
            batch_data: Dictionary containing:
                - intra_batch: Batched intra-polyhedral graphs
                - inter_graphs: List of inter-polyhedral graphs
                - batch_sizes: Number of polyhedra per crystal
                
        Returns:
            Predictions for each crystal in the batch
        """
        intra_batch = batch_data['intra_batch']
        inter_graphs = batch_data['inter_graphs']
        batch_sizes = batch_data['batch_sizes']
        
        predictions = []
        
        # Process each crystal in the batch
        poly_start_idx = 0
        
        for crystal_idx, num_polyhedra in enumerate(batch_sizes):
            # Get polyhedra for this crystal
            poly_end_idx = poly_start_idx + num_polyhedra
            
            # Create mapping from atoms to polyhedra for this crystal
            poly_batch_indices = []
            atom_start = 0
            
            for poly_idx in range(poly_start_idx, poly_end_idx):
                # Find atoms belonging to this polyhedron
                poly_mask = (intra_batch.poly_idx == poly_idx)
                num_atoms_in_poly = poly_mask.sum().item()
                
                # Map these atoms to the relative polyhedron index
                poly_batch_indices.extend([poly_idx - poly_start_idx] * num_atoms_in_poly)
            
            poly_batch_indices = torch.tensor(poly_batch_indices, device=intra_batch.x.device)
            
            # Stage 1: Intra-polyhedral encoding
            # Get atoms for this crystal
            crystal_atom_mask = torch.zeros(intra_batch.x.size(0), dtype=torch.bool, device=intra_batch.x.device)
            for poly_idx in range(poly_start_idx, poly_end_idx):
                crystal_atom_mask |= (intra_batch.poly_idx == poly_idx)
            
            crystal_atoms_x = intra_batch.x[crystal_atom_mask]
            
            # Adjust edge indices for this crystal
            crystal_edge_mask = crystal_atom_mask[intra_batch.edge_index[0]] & crystal_atom_mask[intra_batch.edge_index[1]]
            crystal_edge_index = intra_batch.edge_index[:, crystal_edge_mask]
            
            # Remap edge indices to local indices
            atom_idx_mapping = torch.zeros(intra_batch.x.size(0), dtype=torch.long, device=intra_batch.x.device)
            atom_idx_mapping[crystal_atom_mask] = torch.arange(crystal_atoms_x.size(0), device=intra_batch.x.device)
            crystal_edge_index = atom_idx_mapping[crystal_edge_index]
            
            # Apply Stage 1 GNN
            poly_representations = self.intra_gnn(
                x=crystal_atoms_x,
                edge_index=crystal_edge_index,
                batch=None,  # Single crystal
                poly_batch_indices=poly_batch_indices
            )
            
            # Stage 2: Inter-polyhedral message passing
            inter_graph = inter_graphs[crystal_idx]
            
            # Replace initial node features with Stage 1 output
            updated_poly_features = self.inter_gnn(
                x=poly_representations,
                edge_index=inter_graph.edge_index,
                edge_attr=inter_graph.edge_attr
            )
            
            # Global pooling to get crystal representation
            crystal_batch = torch.zeros(updated_poly_features.size(0), dtype=torch.long, device=updated_poly_features.device)
            crystal_representation = self.global_pooling(updated_poly_features, crystal_batch)
            
            # Final prediction
            prediction = self.prediction_head(crystal_representation)
            predictions.append(prediction)
            
            poly_start_idx = poly_end_idx
        
        # Stack predictions
        predictions = torch.cat(predictions, dim=0)
        
        return predictions


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Two-Stage GNN Architecture...")
    
    # Model parameters
    atom_input_dim = 89 + 3  # 86 elements + 3 atomic properties
    model = TwoStageGNN(
        atom_input_dim=atom_input_dim,
        poly_hidden_dim=128,
        inter_hidden_dim=128,
        output_dim=1,
        intra_layers=3,
        inter_layers=3,
        pooling='attention'
    )
    
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    print("\nModel architecture:")
    print(model)
