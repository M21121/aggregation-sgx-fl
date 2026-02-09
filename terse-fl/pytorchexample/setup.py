#!/usr/bin/env python3
"""Setup TERSE cryptographic material before FL training."""

import subprocess
import os

def setup_terse(n_clients: int, n_rounds: int, model_params: int):
    """
    Initialize TERSE for federated learning.

    Args:
        n_clients: Number of FL clients
        n_rounds: Number of training rounds
        model_params: Total number of model parameters (flattened)
    """
    os.makedirs("data", exist_ok=True)

    # Each round encrypts model_params values
    # We need timestamps for: n_rounds * num_param_tensors
    n_timestamps = n_rounds * 10  # Adjust based on your model's param count
    vector_dim = model_params // n_timestamps + 1

    print(f"Setting up TERSE for {n_clients} clients, {n_timestamps} timestamps")

    # Run client setup
    subprocess.run([
        "./setup_clients", 
        str(n_clients), 
        str(n_timestamps), 
        str(vector_dim)
    ], check=True)

    # Run trusted setup
    subprocess.run([
        "./setup_trusted", 
        str(n_clients), 
        str(n_timestamps), 
        str(vector_dim)
    ], check=True)

    print("TERSE setup complete!")


if __name__ == "__main__":
    from pytorchexample.task import Net

    # Count model parameters
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} parameters")

    setup_terse(
        n_clients=10,
        n_rounds=5,
        model_params=total_params
    )
