"""
Inference and evaluation script for the Two-Stage GNN
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse

from model import TwoStageGNN
from data_loader import PolyhedralDataset, create_batches
from train import Trainer


class Evaluator:
    """
    Evaluation class for the Two-Stage GNN
    """
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
        print(f"Using device: {device}")
    
    def _load_model(self, model_path):
        """Load trained model from checkpoint"""
        # Model parameters (should match training)
        atom_input_dim = 89 + 3  # Element features + atomic properties
        model = TwoStageGNN(
            atom_input_dim=atom_input_dim,
            poly_hidden_dim=128,
            inter_hidden_dim=128,
            output_dim=1,
            intra_layers=3,
            inter_layers=3,
            pooling='attention'
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def predict_single(self, processed_data):
        """Make prediction for a single crystal structure"""
        # Create batch with single item
        batch = create_batches([processed_data], batch_size=1)[0]
        
        # Move to device
        if batch['intra_batch'] is not None:
            batch['intra_batch'] = batch['intra_batch'].to(self.device)
        
        for i, inter_graph in enumerate(batch['inter_graphs']):
            batch['inter_graphs'][i] = inter_graph.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(batch)
            return prediction.cpu().numpy()[0]
    
    def predict_batch(self, processed_data_list):
        """Make predictions for a batch of crystal structures"""
        predictions = []
        
        # Process in batches
        batches = create_batches(processed_data_list, batch_size=8)
        
        for batch in batches:
            # Move to device
            if batch['intra_batch'] is not None:
                batch['intra_batch'] = batch['intra_batch'].to(self.device)
            
            for i, inter_graph in enumerate(batch['inter_graphs']):
                batch['inter_graphs'][i] = inter_graph.to(self.device)
            
            # Make predictions
            with torch.no_grad():
                batch_predictions = self.model(batch)
                predictions.extend(batch_predictions.cpu().numpy())
        
        return np.array(predictions)
    
    def evaluate_dataset(self, processed_data_list, targets=None):
        """Evaluate model on a dataset"""
        print(f"Evaluating on {len(processed_data_list)} structures...")
        
        # Make predictions
        predictions = self.predict_batch(processed_data_list)
        
        results = {
            'cif_filenames': [data['cif_filename'] for data in processed_data_list],
            'predictions': predictions.flatten(),
            'num_polyhedra': [data['num_polyhedra'] for data in processed_data_list]
        }
        
        # Add target properties if available
        if targets is not None:
            results['targets'] = targets
            
            # Calculate metrics
            mae = mean_absolute_error(targets, predictions)
            rmse = np.sqrt(mean_squared_error(targets, predictions))
            r2 = r2_score(targets, predictions)
            
            print(f"Evaluation Metrics:")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R²: {r2:.4f}")
            
            results['metrics'] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            # Plot predictions vs targets
            self.plot_predictions(targets, predictions)
        
        return results
    
    def plot_predictions(self, targets, predictions, save_path=None):
        """Plot predictions vs targets"""
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, predictions, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs True Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² to plot
        r2 = r2_score(targets, predictions)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_structure(self, processed_data):
        """Analyze a single structure and provide detailed output"""
        cif_filename = processed_data['cif_filename']
        
        print(f"\nAnalyzing structure: {cif_filename}")
        print("=" * 50)
        
        # Basic structure info
        print(f"Number of polyhedra: {processed_data['num_polyhedra']}")
        
        # Polyhedra details
        print("\nPolyhedra information:")
        for i, poly_info in enumerate(processed_data['polyhedron_info']):
            print(f"  Polyhedron {i}: {poly_info['center_element']} "
                  f"(CN: {poly_info['coordination_number']})")
        
        # Connectivity analysis
        inter_graph = processed_data['inter_poly_graph']
        num_edges = inter_graph.edge_index.size(1)
        
        print(f"\nConnectivity:")
        print(f"  Number of connections: {num_edges}")
        
        if num_edges > 0:
            edge_types = inter_graph.edge_attr.sum(dim=0)
            connection_types = ['Face-tri', 'Face-quad', 'Edge', 'Point']
            for i, conn_type in enumerate(connection_types):
                print(f"  {conn_type} connections: {int(edge_types[i])}")
        
        # Make prediction
        prediction = self.predict_single(processed_data)
        print(f"\nPredicted property: {prediction:.4f}")
        
        return prediction


def main():
    parser = argparse.ArgumentParser(description='Evaluate Two-Stage GNN')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='../cifs',
                       help='Directory containing CIF files')
    parser.add_argument('--dataset_csv', type=str, default='../dataset.csv',
                       help='CSV file with structure metadata')
    parser.add_argument('--use_preprocessed', action='store_true',
                       help='Use preprocessed data instead of processing on-the-fly')
    parser.add_argument('--processed_data', type=str, default='processed_data.pkl',
                       help='Path to processed data file')
    parser.add_argument('--output_file', type=str, default='predictions.csv',
                       help='Output file for predictions')
    parser.add_argument('--analyze_single', type=str, default=None,
                       help='Analyze a single CIF file')
    
    args = parser.parse_args()
    
    print("Two-Stage GNN Evaluation")
    print("=" * 50)
    
    # Load evaluator
    evaluator = Evaluator(args.model_path)
    
    if args.analyze_single:
        # Analyze single structure
        print(f"Analyzing single structure: {args.analyze_single}")
        
        # Process the CIF file
        dataset = PolyhedralDataset(args.data_dir, args.dataset_csv)
        processed_data = dataset.processor.process_cif_file(
            os.path.join(args.data_dir, args.analyze_single)
        )
        
        if processed_data:
            prediction = evaluator.analyze_structure(processed_data)
        else:
            print(f"Failed to process {args.analyze_single}")
    
    else:
        # Evaluate on dataset
        if args.use_preprocessed:
            if not os.path.exists(args.processed_data):
                print(f"Preprocessed data file not found: {args.processed_data}")
                print("Please run: python preprocess_data.py first")
                return
            
            # Load processed data
            dataset = PolyhedralDataset(args.data_dir, args.dataset_csv)
            processed_data_list = dataset.load_preprocessed_data(args.processed_data)
            
        else:
            print("Processing data on-the-fly...")
            dataset = PolyhedralDataset(args.data_dir, args.dataset_csv)
            
            # Get CIF files and process
            cif_files = [f for f in os.listdir(args.data_dir) if f.endswith('.cif')]
            processed_data_list = dataset.load_data(cif_files[:10])  # Limit for testing
        
        if not processed_data_list:
            print("No data could be processed!")
            return
        
        # Evaluate
        results = evaluator.evaluate_dataset(processed_data_list)
        
        # Save results
        df = pd.DataFrame({
            'cif_filename': results['cif_filenames'],
            'prediction': results['predictions'],
            'num_polyhedra': results['num_polyhedra']
        })
        
        df.to_csv(args.output_file, index=False)
        print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
