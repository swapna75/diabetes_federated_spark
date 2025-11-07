"""
Flower Federated Learning Server
Coordinates federated training across multiple institutions
"""

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from typing import List, Tuple, Optional, Dict
import numpy as np
from federated_learning_model import DiabetesModelTrainer
import torch
import json
from datetime import datetime


class FederatedServer:
    """Federated Learning Server using Flower"""
    
    def __init__(
        self,
        input_dim: int,
        num_rounds: int = 10,
        min_clients: int = 2,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0
    ):
        self.input_dim = input_dim
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        
        # Initialize global model
        self.global_trainer = DiabetesModelTrainer(
            input_dim=input_dim,
            hidden_dims=[64, 32, 16],
            learning_rate=0.001
        )
        
        # Metrics storage
        self.history = {
            'rounds': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'val_auc': []
        }
        
        print("="*60)
        print("Federated Learning Server Initialized")
        print("="*60)
        print(f"Input dimension: {input_dim}")
        print(f"Number of rounds: {num_rounds}")
        print(f"Minimum clients: {min_clients}")
        print("="*60)
    
    def get_initial_parameters(self):
        """Get initial model parameters"""
        return self.global_trainer.get_parameters()
    
    def weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """
        Aggregate metrics from multiple clients using weighted average
        """
        # Initialize accumulators
        total_examples = sum([num_examples for num_examples, _ in metrics])
        
        aggregated = {}
        
        # Get all metric keys from first client
        if metrics:
            metric_keys = metrics[0][1].keys()
            
            for key in metric_keys:
                weighted_sum = sum([
                    num_examples * m[key] 
                    for num_examples, m in metrics 
                    if key in m
                ])
                aggregated[key] = weighted_sum / total_examples
        
        return aggregated
    
    def fit_config(self, server_round: int) -> Dict:
        """Return training configuration for each round"""
        config = {
            "round": server_round,
            "local_epochs": 5,
            "batch_size": 32,
        }
        return config
    
    def evaluate_config(self, server_round: int) -> Dict:
        """Return evaluation configuration"""
        return {"round": server_round}
    
    def create_strategy(self):
        """Create federated learning strategy"""
        
        strategy = FedAvg(
            fraction_fit=self.fraction_fit,
            fraction_evaluate=self.fraction_evaluate,
            min_fit_clients=self.min_fit_clients,
            min_evaluate_clients=self.min_evaluate_clients,
            min_available_clients=self.min_clients,
            initial_parameters=fl.common.ndarrays_to_parameters(self.get_initial_parameters()),
            fit_metrics_aggregation_fn=self.weighted_average,
            evaluate_metrics_aggregation_fn=self.weighted_average,
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
        )
        
        return strategy
    
    def start_server(self, server_address: str = "0.0.0.0:8080"):
        """Start the federated learning server"""
        
        print(f"\nStarting Flower server at {server_address}")
        print(f"Waiting for {self.min_clients} clients to connect...\n")
        
        # Create strategy
        strategy = self.create_strategy()
        
        # Start server
        history = fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )
        
        print("\n" + "="*60)
        print("Federated Learning Complete!")
        print("="*60)
        
        # Process and save history
        self.process_history(history)
        
        return history
    
    def process_history(self, history):
        """Process and save training history"""
        print("\nProcessing training history...")
        
        # Extract metrics from History object
        if hasattr(history, 'metrics_distributed_fit'):
            # Get metrics from distributed fit
            metrics_fit = history.metrics_distributed_fit
            
            # Process each metric
            for metric_name in ['train_loss', 'train_accuracy', 'val_loss', 
                               'val_accuracy', 'val_f1', 'val_auc']:
                if metric_name in metrics_fit:
                    for round_num, value in metrics_fit[metric_name]:
                        if len(self.history['rounds']) < round_num:
                            self.history['rounds'].append(round_num)
                        if len(self.history[metric_name]) < round_num:
                            self.history[metric_name].append(float(value))
        
        # Also get evaluation metrics
        if hasattr(history, 'metrics_distributed'):
            metrics_eval = history.metrics_distributed
            for metric_name in ['accuracy', 'f1', 'auc']:
                if metric_name in metrics_eval:
                    for i, (round_num, value) in enumerate(metrics_eval[metric_name]):
                        if i < len(self.history['val_accuracy']):
                            continue
                        if metric_name == 'accuracy':
                            self.history['val_accuracy'].append(float(value))
                        elif metric_name == 'f1':
                            self.history['val_f1'].append(float(value))
                        elif metric_name == 'auc':
                            self.history['val_auc'].append(float(value))
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/fl_history_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"History saved to {output_file}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        
        if self.history['rounds']:
            print(f"Total Rounds: {len(self.history['rounds'])}")
            
            if self.history['val_accuracy']:
                print(f"Final Validation Accuracy: {self.history['val_accuracy'][-1]:.4f}")
                print(f"Final Validation F1: {self.history['val_f1'][-1]:.4f}")
                print(f"Final Validation AUC: {self.history['val_auc'][-1]:.4f}")
                
                best_acc_round = np.argmax(self.history['val_accuracy']) + 1
                best_acc = max(self.history['val_accuracy'])
                print(f"Best Validation Accuracy: {best_acc:.4f} (Round {best_acc_round})")
        
        print("="*60)


def run_simulation(
    num_institutions: int = 3,
    input_dim: int = 10,
    num_rounds: int = 10
):
    """
    Run federated learning simulation
    This simulates the complete FL process
    """
    from fl_client import create_client_fn
    
    print("="*60)
    print("Starting Federated Learning Simulation")
    print("="*60)
    print(f"Number of institutions: {num_institutions}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Input dimension: {input_dim}")
    print("="*60)
    
    # Create server
    server = FederatedServer(
        input_dim=input_dim,
        num_rounds=num_rounds,
        min_clients=num_institutions,
        min_fit_clients=num_institutions,
        min_evaluate_clients=num_institutions
    )
    
    # Create client functions (Ray uses 0-indexed client IDs)
    client_fns = {}
    for i in range(num_institutions):
        institution_name = f"institution_{i+1}"
        client_fns[str(i)] = create_client_fn(institution_name, input_dim, local_epochs=5)
    
    # Create strategy
    strategy = server.create_strategy()
    
    # Run simulation
    print("\nStarting simulation...")
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: client_fns[cid](cid),
        num_clients=num_institutions,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )
    
    print("\n" + "="*60)
    print("Simulation Complete!")
    print("="*60)
    
    # Process history
    server.process_history(history)
    
    return history, server


if __name__ == "__main__":
    import os
    import sys
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Configuration
    NUM_INSTITUTIONS = 3
    INPUT_DIM = 10  # Should match the number of features
    NUM_ROUNDS = 10
    
    # Check if running in simulation mode
    simulation_mode = "--simulate" in sys.argv
    
    if simulation_mode:
        print("Running in SIMULATION mode\n")
        history, server = run_simulation(
            num_institutions=NUM_INSTITUTIONS,
            input_dim=INPUT_DIM,
            num_rounds=NUM_ROUNDS
        )
    else:
        print("Running in SERVER mode")
        print("Clients should be started separately using fl_client.py\n")
        
        # Create and start server
        server = FederatedServer(
            input_dim=INPUT_DIM,
            num_rounds=NUM_ROUNDS,
            min_clients=NUM_INSTITUTIONS,
            min_fit_clients=NUM_INSTITUTIONS,
            min_evaluate_clients=NUM_INSTITUTIONS
        )
        
        history = server.start_server(server_address="0.0.0.0:8080")
    
    print("\nFederated Learning Server finished successfully!")