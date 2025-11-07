"""
Federated Learning Model for Diabetes Prediction
Using PyTorch and Flower Framework
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from typing import List, Tuple, Dict


class DiabetesDataset(Dataset):
    """PyTorch Dataset for Diabetes Data"""
    
    def __init__(self, features, labels):
        """
        Args:
            features: numpy array or list of feature vectors
            labels: numpy array or list of labels
        """
        # Handle sparse vectors from PySpark
        if hasattr(features[0], 'toArray'):
            features = np.array([f.toArray() for f in features])
        elif isinstance(features, pd.Series):
            features = np.array([np.array(f) for f in features])
        else:
            features = np.array(features)
        
        labels = np.array(labels)
        
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DiabetesNN(nn.Module):
    """Neural Network for Diabetes Risk Prediction"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], dropout_rate=0.3):
        super(DiabetesNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 2))  # Binary classification
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DiabetesModelTrainer:
    """Trainer for Diabetes Prediction Model"""
    
    def __init__(
        self, 
        input_dim, 
        hidden_dims=[64, 32, 16],
        learning_rate=0.001,
        device=None
    ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DiabetesNN(input_dim, hidden_dims).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print(f"Model initialized on {self.device}")
        print(f"Model architecture: Input({input_dim}) -> {hidden_dims} -> Output(2)")
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in dataloader:
            features, labels = features.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for features, labels in dataloader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # Calculate metrics
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, zero_division=0),
            'recall': recall_score(all_labels, all_predictions, zero_division=0),
            'f1': f1_score(all_labels, all_predictions, zero_division=0)
        }
        
        try:
            metrics['auc'] = roc_auc_score(all_labels, all_probabilities)
        except:
            metrics['auc'] = 0.0
        
        return metrics
    
    def train(self, train_loader, val_loader=None, epochs=10, verbose=True):
        """Train model for multiple epochs"""
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_metrics': []
        }
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validate
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                history['val_metrics'].append(val_metrics)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}")
                    print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                    print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                          f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            else:
                if verbose and epoch % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        return history
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as numpy arrays"""
        return [param.cpu().detach().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from numpy arrays"""
        params_dict = zip(self.model.parameters(), parameters)
        for param, new_param in params_dict:
            param.data = torch.from_numpy(new_param).to(self.device)
    
    def save_model(self, filepath):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")


def prepare_dataloaders(features, labels, batch_size=32, train_split=0.8):
    """
    Prepare training and validation dataloaders
    """
    # Create dataset
    dataset = DiabetesDataset(features, labels)
    
    # Split into train and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    print(f"Data split: Train={train_size}, Val={val_size}")
    
    return train_loader, val_loader


def federated_averaging(model_weights: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Perform Federated Averaging (FedAvg)
    
    Args:
        model_weights: List of model parameters from each institution
    
    Returns:
        Averaged model parameters
    """
    if not model_weights:
        return []
    
    # Initialize with zeros
    avg_weights = [np.zeros_like(w) for w in model_weights[0]]
    
    # Sum all weights
    for weights in model_weights:
        for i, w in enumerate(weights):
            avg_weights[i] += w
    
    # Average
    num_models = len(model_weights)
    avg_weights = [w / num_models for w in avg_weights]
    
    print(f"Averaged parameters from {num_models} institutions")
    
    return avg_weights


if __name__ == "__main__":
    # Test with dummy data
    print("Testing Diabetes Model...")
    
    # Create dummy data
    n_samples = 1000
    input_dim = 10
    
    features = np.random.randn(n_samples, input_dim).astype(np.float32)
    labels = np.random.randint(0, 2, n_samples)
    
    # Prepare dataloaders
    train_loader, val_loader = prepare_dataloaders(features, labels, batch_size=32)
    
    # Initialize trainer
    trainer = DiabetesModelTrainer(input_dim=input_dim, hidden_dims=[32, 16])
    
    # Train
    print("\nTraining model...")
    history = trainer.train(train_loader, val_loader, epochs=5)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = trainer.evaluate(val_loader)
    print(f"Validation Metrics: {metrics}")
    
    print("\nModel test completed successfully!")