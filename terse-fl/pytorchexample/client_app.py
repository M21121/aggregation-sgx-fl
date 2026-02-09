"""TERSE-FL Client App with encrypted model updates."""

import os
import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from pytorchexample.task import Net, load_data, train, test
from pytorchexample.terse_utils import (
    get_terse_module,
    quantize_parameters,
    DATA_DIR,
    PROJECT_ROOT,
)


class TERSEFlowerClient(NumPyClient):
    """Flower client that encrypts model updates with TERSE."""

    def __init__(
        self,
        partition_id: int,
        num_partitions: int,
        batch_size: int,
        local_epochs: int,
        learning_rate: float,
        scale_factor: int,
        plaintext_modulus: int,
        vector_dim: int,
    ):
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.scale_factor = scale_factor
        self.plaintext_modulus = plaintext_modulus
        self.vector_dim = vector_dim

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainloader, self.valloader = load_data(
            partition_id, num_partitions, batch_size
        )

        # Initialize TERSE client
        # Must chdir to PROJECT_ROOT because C++ uses relative "data/" path
        terse_py = get_terse_module()
        params_file = str(DATA_DIR / "params.bin")

        original_cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            self.terse_client = terse_py.TERSEClient(params_file, partition_id)
        finally:
            os.chdir(original_cwd)

    def fit(self, parameters, config):
        """Train model and return encrypted update."""
        model = Net()
        self.set_parameters(model, parameters)
        model.to(self.device)

        train_loss = train(
            model,
            self.trainloader,
            self.local_epochs,
            self.learning_rate,
            self.device,
        )

        # Server sends server_round starting at 1
        server_round = int(config.get("server_round", 1))

        encrypted_chunks = self.get_encrypted_parameters(model, server_round)

        return encrypted_chunks, len(self.trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        """Evaluate model on local validation data."""
        model = Net()
        self.set_parameters(model, parameters)
        model.to(self.device)

        loss, accuracy = test(model, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

    def set_parameters(self, model: Net, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

    def get_encrypted_parameters(self, model: Net, server_round: int):
        """Extract parameters, flatten, quantize, and encrypt."""
        all_params = []
        for param in model.parameters():
            all_params.append(param.detach().cpu().numpy().flatten())
        flat_params = np.concatenate(all_params)

        padded_len = ((len(flat_params) + self.vector_dim - 1) // self.vector_dim) * self.vector_dim
        padded_params = np.zeros(padded_len, dtype=np.float32)
        padded_params[: len(flat_params)] = flat_params

        quantized = quantize_parameters(
            padded_params, self.scale_factor, self.plaintext_modulus
        )

        encrypted_chunks = []
        n_chunks = padded_len // self.vector_dim

        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.vector_dim
            end = start + self.vector_dim
            chunk = quantized[start:end]

            # IMPORTANT: round starts at 1; timestamps start at 0
            timestamp = (server_round - 1) * n_chunks + chunk_idx

            encrypted = self.terse_client.encrypt_vector(chunk, timestamp)
            encrypted_chunks.append(encrypted)

        return encrypted_chunks


def client_fn(context: Context):
    """Create a TERSEFlowerClient instance."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    scale_factor = context.run_config["scale-factor"]
    plaintext_modulus = context.run_config["plaintext-modulus"]
    vector_dim = context.run_config["vector-dim"]

    return TERSEFlowerClient(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        scale_factor=scale_factor,
        plaintext_modulus=plaintext_modulus,
        vector_dim=vector_dim,
    ).to_client()


app = ClientApp(client_fn=client_fn)
