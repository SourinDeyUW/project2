# Two-Stage GNN for Crystal Property Prediction

A hierarchical Graph Neural Network implementation for predicting crystal properties from polyhedral connectivity patterns. This implementation follows the architectural blueprint you specified with two stages of message passing.

## Architecture Overview

The model implements a hierarchical two-stage approach:

### Stage 1: Intra-Polyhedral Encoding
- **Input**: Subgraphs representing individual polyhedra (atoms as nodes, bonds as edges)
- **Processing**: Graph Attention Networks (GAT) for message passing within each polyhedron
- **Pooling**: Attention-based pooling to create polyhedron-level representations
- **Output**: Fixed-size vector for each polyhedron

### Stage 2: Inter-Polyhedral Message Passing
- **Input**: Crystal-level graph where nodes are polyhedra from Stage 1
- **Edge Features**: Connection types (face-sharing, edge-sharing, point-sharing)
- **Processing**: Edge-Conditioned Convolution incorporating connection types
- **Output**: Context-aware polyhedron representations

### Final Prediction
- **Global Pooling**: Attention-based aggregation to crystal-level representation
- **Prediction Head**: MLP for final property prediction

## Installation

1. Install PyTorch and PyTorch Geometric:
```bash
# For CUDA 11.7 (adjust based on your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install torch-geometric
```

2. Install other requirements:
```bash
pip install -r requirements.txt
```

3. Ensure you have access to the parent directory files:
- view_polyhedra.py (for polyhedra extraction)
- draw_graphs.py (for connectivity analysis)
- CIF files in ../cifs/ directory
- dataset.csv for structure metadata

## Usage

### Data Processing and Training

1. **Process data and train the model**:
```bash
python train.py --data_dir ../cifs --dataset_csv ../dataset.csv --epochs 100 --batch_size 8
```

2. **Training with custom parameters**:
```bash
python train.py --data_dir ../cifs --dataset_csv ../dataset.csv --epochs 200 --batch_size 16 --lr 0.0005 --hidden_dim 256
```

### Evaluation

1. **Evaluate on test set**:
```bash
python evaluate.py --model_path checkpoints/best_model.pth --data_dir ../cifs --dataset_csv ../dataset.csv
```

2. **Analyze a single structure**:
```bash
python evaluate.py --model_path checkpoints/best_model.pth --data_dir ../cifs --analyze_single mp-757245.cif
```

## File Structure

```
two_stage_gnn/
├── data_loader.py          # Data processing and loading
├── model.py               # Two-stage GNN architecture
├── train.py               # Training script
├── evaluate.py            # Evaluation and inference
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── checkpoints/          # Saved model checkpoints
├── runs/                 # TensorBoard logs
└── processed_data.pkl    # Cached processed data
```

## Model Components

### IntraPolyhedralGNN
- Processes atoms within individual polyhedra
- Uses Graph Attention Networks for message passing
- Supports multiple pooling strategies (attention, mean, sum, max)
- Residual connections and layer normalization

### InterPolyhedralGNN  
- Processes polyhedron-level connectivity
- Edge-Conditioned Convolution layers
- Incorporates connection type features
- Learns global crystal structure patterns

### EdgeConditionedConv
- Custom message passing layer
- Conditions messages on edge attributes (connection types)
- Learns different interaction patterns for different connection types

### TwoStageGNN
- Complete end-to-end architecture
- Integrates both stages with global pooling
- Flexible prediction head for various properties

## Key Features

1. **Hierarchical Architecture**: Two-level message passing (atom → polyhedron → crystal)

2. **Connection-Aware**: Explicitly models different polyhedron connection types:
   - Face-sharing (triangular and quadrilateral)
   - Edge-sharing  
   - Point-sharing (corner-sharing)

3. **Attention Mechanisms**: 
   - Intra-polyhedral attention for atom aggregation
   - Global attention for crystal-level pooling

4. **Flexible Design**:
   - Configurable hidden dimensions
   - Multiple pooling strategies
   - Residual connections and normalization

5. **Comprehensive Training**:
   - Learning rate scheduling
   - Gradient clipping
   - TensorBoard logging
   - Checkpoint saving

## Training Features

- **Data Augmentation**: Handles variable-size structures
- **Batch Processing**: Custom batching for hierarchical data
- **Monitoring**: TensorBoard integration for loss and metrics tracking
- **Checkpointing**: Automatic saving of best models
- **Validation**: Built-in validation loop with multiple metrics

## Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)  
- R² Score
- Prediction vs. target scatter plots

## Example Output

```
Analyzing structure: mp-757245.cif
==================================================
Number of polyhedra: 9

Polyhedra information:
  Polyhedron 0: Cr (CN: 4)
  Polyhedron 1: Cr (CN: 4)
  ...

Connectivity:
  Number of connections: 26
  Face-tri connections: 0
  Face-quad connections: 0
  Edge connections: 8
  Point connections: 18

Predicted property: 2.3456
```

## Customization

### Adding New Target Properties

Modify the `_extract_targets()` method in `train.py`:

```python
def _extract_targets(self, batch):
    targets = []
    for properties in batch['target_properties']:
        if properties and 'your_property' in properties:
            targets.append(properties['your_property'])
        else:
            targets.append(0.0)  # Default value
    return torch.tensor(targets, device=self.device)
```

### Adjusting Model Architecture

Modify parameters in model creation:

```python
model = TwoStageGNN(
    atom_input_dim=atom_input_dim,
    poly_hidden_dim=256,        # Increase for more capacity
    inter_hidden_dim=256,       # Increase for more capacity  
    output_dim=1,               # Change for multi-target
    intra_layers=4,             # More layers for complex polyhedra
    inter_layers=4,             # More layers for complex connectivity
    pooling='attention'         # 'mean', 'sum', 'max', 'attention'
)
```

## Performance Tips

1. **GPU Usage**: The model automatically uses CUDA if available
2. **Batch Size**: Start with smaller batches (4-8) for large structures
3. **Memory**: Monitor GPU memory usage with complex structures
4. **Preprocessing**: Save processed data to avoid recomputation

## Troubleshooting

1. **CUDA Errors**: Ensure PyTorch and PyTorch Geometric versions match your CUDA version
2. **Memory Issues**: Reduce batch size or model dimensions
3. **Data Loading**: Verify CIF files and dependency imports from parent directory
4. **Training Instability**: Try lower learning rates or gradient clipping

## Model Interpretability

The hierarchical design provides natural interpretability:
- Stage 1 representations reveal polyhedron-specific features
- Stage 2 representations capture connectivity patterns
- Attention weights show which polyhedra/atoms are most important

## Future Extensions

1. **Multi-task Learning**: Predict multiple properties simultaneously
2. **Transfer Learning**: Pre-train on large datasets, fine-tune on specific properties
3. **Uncertainty Quantification**: Add dropout or ensemble methods
4. **Graph Augmentation**: Synthetic connectivity patterns for data augmentation
