"""
Training script for the Two-Stage GNN
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import json

from model import TwoStageGNN, count_parameters
from data_loader import PolyhedralDataset, create_batches


class Trainer:
    """
    Training class for the Two-Stage GNN
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Training components
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        
        # Logging
        self.writer = None
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        print(f"Using device: {device}")
        print(f"Model has {count_parameters(model):,} trainable parameters")
    
    def setup_training(self, learning_rate=0.001, weight_decay=1e-5):
        """Setup optimizer, criterion, and scheduler"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.MSELoss()
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
    
    def setup_logging(self, log_dir='runs'):
        """Setup tensorboard logging"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(log_dir, f'two_stage_gnn_{timestamp}')
        self.writer = SummaryWriter(log_path)
        print(f"Logging to: {log_path}")
    
    def train_epoch(self, train_batches):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_batches)
        
        for batch_idx, batch in enumerate(train_batches):
            # Move data to device
            batch = self._move_batch_to_device(batch)
            
            # Extract targets (you'll need to implement this based on your target property)
            targets = self._extract_targets(batch)
            if targets is None:
                continue
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            
            # Compute loss
            loss = self.criterion(predictions.squeeze(), targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate_epoch(self, val_batches):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for batch in val_batches:
                # Move data to device
                batch = self._move_batch_to_device(batch)
                
                # Extract targets
                targets = self._extract_targets(batch)
                if targets is None:
                    continue
                
                # Forward pass
                predictions = self.model(batch)
                
                # Compute loss
                loss = self.criterion(predictions.squeeze(), targets)
                total_loss += loss.item()
                
                # Store predictions and targets for metrics
                predictions_list.append(predictions.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_batches) if len(val_batches) > 0 else 0.0
        
        # Compute additional metrics
        if predictions_list:
            all_predictions = np.concatenate(predictions_list)
            all_targets = np.concatenate(targets_list)
            
            mae = mean_absolute_error(all_targets, all_predictions)
            rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
            r2 = r2_score(all_targets, all_predictions)
            
            return avg_loss, mae, rmse, r2
        
        return avg_loss, 0.0, 0.0, 0.0
    
    def train(self, train_batches, val_batches, num_epochs=100, save_dir='checkpoints'):
        """Main training loop"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training batches: {len(train_batches)}")
        print(f"Validation batches: {len(val_batches)}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(train_batches)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, mae, rmse, r2 = self.validate_epoch(val_batches)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.writer.add_scalar('Metrics/MAE', mae, epoch)
                self.writer.add_scalar('Metrics/RMSE', rmse, epoch)
                self.writer.add_scalar('Metrics/R2', r2, epoch)
                self.writer.add_scalar('Learning_Rate', 
                                     self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(os.path.join(save_dir, 'best_model.pth'), epoch)
                print(f"New best model saved! Val Loss: {val_loss:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'), epoch)
        
        print("\nTraining completed!")
        self.plot_training_curves(save_dir)
    
    def _move_batch_to_device(self, batch):
        """Move batch data to device"""
        if batch['intra_batch'] is not None:
            batch['intra_batch'] = batch['intra_batch'].to(self.device)
        
        for i, inter_graph in enumerate(batch['inter_graphs']):
            batch['inter_graphs'][i] = inter_graph.to(self.device)
        
        return batch
    
    def _extract_targets(self, batch):
        """Extract target values from batch (implement based on your target property)"""
        # This is a placeholder - you'll need to implement based on your specific target
        # For now, return random targets for demonstration
        batch_size = len(batch['cif_filenames'])
        
        # You could extract from target_properties or load from external data
        # For demonstration, using random values
        targets = torch.randn(batch_size, device=self.device)
        
        return targets
    
    def save_checkpoint(self, filepath, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def plot_training_curves(self, save_dir):
        """Plot and save training curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()


def load_and_prepare_data(data_dir, dataset_csv, processed_data_path=None, 
                         use_preprocessed=False, test_size=0.2, batch_size=8):
    """Load and prepare data for training"""
    
    dataset = PolyhedralDataset(data_dir, dataset_csv)
    
    if use_preprocessed:
        if not processed_data_path or not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Preprocessed file {processed_data_path} not found! "
                                  "Run: python preprocess_data.py first")
        
        print("Using preprocessed data for fast training...")
        train_batches, val_batches = dataset.create_dataloaders_from_preprocessed(
            processed_data_path, train_split=1-test_size, batch_size=batch_size
        )
        
    else:
        print("Processing data on-the-fly (slower)...")
        
        # Get list of CIF files
        cif_files = [f for f in os.listdir(data_dir) if f.endswith('.cif')]
        print(f"Found {len(cif_files)} CIF files")
        
        if len(cif_files) == 0:
            raise ValueError("No CIF files found in data directory!")
        
        # Process all files (or limit for testing)
        processed_data = dataset.load_data(cif_files[:10])  # Limit for testing
        
        if not processed_data:
            raise ValueError("No data could be processed!")
        
        # Save processed data for future use
        if processed_data_path:
            dataset.save_processed_data(processed_data_path)
        
        # Split data manually
        train_data, val_data = train_test_split(processed_data, test_size=test_size, random_state=42)
        
        print(f"Train set: {len(train_data)} crystals")
        print(f"Validation set: {len(val_data)} crystals")
        
        # Create batches
        train_batches = create_batches(train_data, batch_size)
        val_batches = create_batches(val_data, batch_size)
    
    return train_batches, val_batches


def main():
    parser = argparse.ArgumentParser(description='Train Two-Stage GNN')
    parser.add_argument('--data_dir', type=str, default='../cifs', 
                       help='Directory containing CIF files')
    parser.add_argument('--dataset_csv', type=str, default='../dataset.csv',
                       help='CSV file with structure metadata')
    parser.add_argument('--use_preprocessed', action='store_true',
                       help='Use preprocessed data instead of processing on-the-fly')
    parser.add_argument('--processed_data', type=str, default='processed_data.pkl',
                       help='Path to save/load processed data')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension size')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs',
                       help='Directory for tensorboard logs')
    
    args = parser.parse_args()
    
    print("Two-Stage GNN Training")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Dataset CSV: {args.dataset_csv}")
    print(f"Use preprocessed: {args.use_preprocessed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden dimension: {args.hidden_dim}")
    
    # Load and prepare data
    try:
        train_batches, val_batches = load_and_prepare_data(
            args.data_dir, args.dataset_csv, args.processed_data,
            args.use_preprocessed, batch_size=args.batch_size
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create batches if not using preprocessed data
    if not args.use_preprocessed:
        train_batches = create_batches(train_data, args.batch_size)
        val_batches = create_batches(val_data, args.batch_size)
    
    # Create model
    atom_input_dim = 89 + 3  # Element features + atomic properties
    model = TwoStageGNN(
        atom_input_dim=atom_input_dim,
        poly_hidden_dim=args.hidden_dim,
        inter_hidden_dim=args.hidden_dim,
        output_dim=1,
        intra_layers=3,
        inter_layers=3,
        pooling='attention'
    )
    
    # Setup trainer
    trainer = Trainer(model)
    trainer.setup_training(learning_rate=args.lr)
    trainer.setup_logging(args.log_dir)
    
    # Train model
    trainer.train(train_batches, val_batches, args.epochs, args.save_dir)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
