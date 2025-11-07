"""
Flower Federated Learning Client
Represents one institution in the federated setup
"""

import flwr as fl
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from federated_learning_model import DiabetesModelTrainer, DiabetesDataset


class DiabetesFlowerClient(fl.client.NumPyClient):
    """Flower client for federated learning"""
    
    def __init__(
        self, 
        institution_name: str,
        trainer: DiabetesModelTrainer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        local_epochs: int = 5
    ):
        self.institution_name = institution_name
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        
        print(f"[{institution_name}] Flower client initialized")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current local model parameters"""
        print(f"[{self.institution_name}] Sending parameters to server")
        return self.trainer.get_parameters()
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model on local data
        
        Args:
            parameters: Global model parameters from server
            config: Configuration dictionary
        
        Returns:
            Updated parameters, number of examples, metrics
        """
        print(f"\n[{self.institution_name}] Starting local training (Round {config.get('round', '?')})")
        
        # Set global parameters
        self.trainer.set_parameters(parameters)
        
        # Train locally
        history = self.trainer.train(
            self.train_loader, 
            self.val_loader,
            epochs=self.local_epochs,
            verbose=False
        )
        
        # Get updated parameters
        updated_parameters = self.trainer.get_parameters()
        
        # Calculate number of training examples
        num_examples = len(self.train_loader.dataset)
        
        # Prepare metrics
        metrics = {
            "train_loss": float(history['train_loss'][-1]),
            "train_accuracy": float(history['train_acc'][-1]),
        }
        
        if history['val_metrics']:
            val_metrics = history['val_metrics'][-1]
            metrics.update({
                "val_loss": float(val_metrics['loss']),
                "val_accuracy": float(val_metrics['accuracy']),
                "val_f1": float(val_metrics['f1']),
                "val_auc": float(val_metrics['auc'])
            })
        
        print(f"[{self.institution_name}] Training complete")
        print(f"  Train Loss: {metrics['train_loss']:.4f}, Train Acc: {metrics['train_accuracy']:.4f}")
        if 'val_accuracy' in metrics:
            print(f"  Val Acc: {metrics['val_accuracy']:.4f}, Val F1: {metrics['val_f1']:.4f}")
        
        return updated_parameters, num_examples, metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local data
        
        Args:
            parameters: Global model parameters
            config: Configuration dictionary
        
        Returns:
            Loss, number of examples, metrics
        """
        print(f"[{self.institution_name}] Evaluating global model...")
        
        # Set parameters
        self.trainer.set_parameters(parameters)
        
        # Evaluate
        metrics = self.trainer.evaluate(self.val_loader)
        
        # Number of validation examples
        num_examples = len(self.val_loader.dataset)
        
        print(f"[{self.institution_name}] Evaluation complete")
        print(f"  Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Return loss and metrics
        return float(metrics['loss']), num_examples, metrics


def load_institution_data(institution_name: str, data_path: str = "data/processed"):
    """
    Load preprocessed data for an institution
    
    Returns:
        features, labels as numpy arrays
    """
    print(f"Loading data for {institution_name}...")
    
    file_path = f"{data_path}/{institution_name}_ml_data.parquet"
    
    try:
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df)} records from {file_path}")
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using synthetic data.")
        # Create synthetic data
        n_samples = 1000
        input_dim = 10
        features = np.random.randn(n_samples, input_dim).astype(np.float32)
        labels = np.random.randint(0, 2, n_samples)
        return features, labels
    
    # Extract features and labels
    features = np.array([np.array(f) for f in df['features']])
    labels = df['hypo_risk'].values
    
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    return features, labels


def create_client_fn(institution_name: str, input_dim: int, local_epochs: int = 5):
    """
    Factory function to create Flower client
    """
    def client_fn(cid: str) -> fl.client.Client:
        # Load data
        features, labels = load_institution_data(institution_name)
        
        # Calculate class weights for imbalanced data
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        class_weights = [total / (len(unique) * count) for count in counts]
        print(f"[{institution_name}] Class distribution: {dict(zip(unique, counts))}")
        print(f"[{institution_name}] Class weights: {class_weights}")
        
        # Create dataloaders
        dataset = DiabetesDataset(features, labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create trainer with class weights
        trainer = DiabetesModelTrainer(
            input_dim=input_dim,
            hidden_dims=[64, 32, 16],
            learning_rate=0.001,
            class_weights=class_weights
        )
        
        # Create and return client
        return DiabetesFlowerClient(
            institution_name=institution_name,
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            local_epochs=local_epochs
        ).to_client()
    
    return client_fn


if __name__ == "__main__":
    import sys
    
    # Get institution name from command line
    institution_name = sys.argv[1] if len(sys.argv) > 1 else "institution_1"
    
    print(f"Starting Flower client for {institution_name}")
    
    # Load data to determine input dimension
    features, labels = load_institution_data(institution_name)
    input_dim = features.shape[1]
    
    # Create dataloaders
    dataset = DiabetesDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create trainer
    trainer = DiabetesModelTrainer(
        input_dim=input_dim,
        hidden_dims=[64, 32, 16],
        learning_rate=0.001
    )
    
    # Create client
    client = DiabetesFlowerClient(
        institution_name=institution_name,
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        local_epochs=5
    )
    
    # Start client
    print(f"\nConnecting to Flower server...")
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client
    )