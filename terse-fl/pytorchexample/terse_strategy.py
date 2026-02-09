"""Custom FedAvg strategy using TERSE secure aggregation."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from flwr.server.strategy import FedAvg
from flwr.common import (
    Parameters,
    FitRes,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

import sys
sys.path.insert(0, ".")  # Adjust path to find terse_py
import terse_py


class TERSEFedAvg(FedAvg):
    """FedAvg with TERSE secure aggregation."""

    def __init__(
        self,
        n_clients: int,
        n_rounds: int,
        params_file: str = "data/params.bin",
        server_key_file: str = "data/server_key.bin",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_clients = n_clients
        self.current_round = 0

        # Initialize TERSE components
        self.terse_server = terse_py.TERSEServer(params_file)
        self.terse_trusted = terse_py.TERSETrusted(params_file, server_key_file)

        # Track vector dimension (set on first aggregation)
        self.vector_dim = None

def aggregate_fit(self, server_round, results, failures):
    if failures:
        raise RuntimeError(f"Aggregation aborted due to failures: {failures}")

    round_offset = (server_round - 1) * self.n_chunks
    chunk_aggregates = []

    for chunk_idx in range(self.n_chunks):
        timestamp = round_offset + chunk_idx
        client_ciphertexts = [
            parameters_to_ndarrays(res.parameters)[chunk_idx]
            for _, res in results
        ]

        aggregate_ct = self.terse_server.aggregate_ciphertexts(
            client_ciphertexts,
            timestamp=timestamp,
        )
        self.terse_server.save_aggregate(aggregate_ct, timestamp)

        decrypted = self.terse_trusted.decrypt_aggregate(
            stream_idx=timestamp,
            vector_dim=aggregate_ct.size,
        )
        chunk_aggregates.append(np.array(decrypted, dtype=np.float32))

    aggregated_parameters = ndarrays_to_parameters(chunk_aggregates)
    metrics = {}
    return aggregated_parameters, metrics
