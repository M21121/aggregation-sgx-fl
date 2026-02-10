"""TERSE-FL Server App with secure aggregation."""

import os
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Context

from pytorchexample.task import Net, load_centralized_dataset, test, count_parameters
from pytorchexample.terse_utils import (
    get_terse_module,
    run_terse_setup,
    dequantize_parameters,
    DATA_DIR,
    PROJECT_ROOT,
)

from pytorchexample.enclave_ipc import PersistentTrustedRound


class TERSEFedAvg(FedAvg):
    """FedAvg strategy with TERSE secure aggregation (decrypt in SGX)."""

    def __init__(
        self,
        n_clients: int,
        vector_dim: int,
        scale_factor: int,
        plaintext_modulus: int,
        model_param_count: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_clients = n_clients
        self.vector_dim = vector_dim
        self.scale_factor = scale_factor
        self.plaintext_modulus = plaintext_modulus
        self.model_param_count = model_param_count

        padded_len = ((model_param_count + vector_dim - 1) // vector_dim) * vector_dim
        self.n_chunks = padded_len // vector_dim
        self.padded_len = padded_len

        terse_py = get_terse_module()
        params_file = str(DATA_DIR / "params.bin")
        self.terse_server = terse_py.TERSEServer(params_file)

        # Start ONE enclave process and keep it alive.
        # server_fn() already does os.chdir(PROJECT_ROOT), but we still pass cwd explicitly.
        self._enclave = PersistentTrustedRound(
            project_root=PROJECT_ROOT,
            cwd=PROJECT_ROOT,
            stderr_path=DATA_DIR / "trusted_round.stderr.log",
        )

    def _load_decrypted_chunk(self, timestamp: int) -> np.ndarray:
        p = DATA_DIR / f"decrypted_{timestamp}.bin"
        raw = p.read_bytes()
        arr = np.frombuffer(raw, dtype=np.uint32, count=self.vector_dim)

        if arr.size != self.vector_dim:
            raise ValueError(
                f"Bad decrypted size for ts={timestamp}: got {arr.size}, expected {self.vector_dim}"
            )
        return arr

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        all_client_chunks: List[List[np.ndarray]] = []
        for _client, fit_res in results:
            encrypted_chunks = parameters_to_ndarrays(fit_res.parameters)
            all_client_chunks.append(encrypted_chunks)

        aggregated_flat = np.zeros(self.padded_len, dtype=np.float32)

        # timestamp = (server_round - 1) * n_chunks + chunk_idx
        start_ts = (server_round - 1) * self.n_chunks

        # 1) Aggregate and save encrypted aggregates for all chunks (untrusted)
        for chunk_idx in range(self.n_chunks):
            client_ciphertexts = [
                np.ascontiguousarray(client_chunks[chunk_idx], dtype=np.uint64)
                for client_chunks in all_client_chunks
            ]

            timestamp = start_ts + chunk_idx

            aggregate_ct = self.terse_server.aggregate_ciphertexts(client_ciphertexts, timestamp)
            self.terse_server.save_aggregate(aggregate_ct, timestamp)

        # 2) Trusted decrypt: reuse the persistent enclave (no restart)
        self._enclave.decrypt_round(
            start_ts=start_ts,
            n_chunks=self.n_chunks,
            vector_dim=self.vector_dim,
        )

        # 3) Load decrypted chunks and finish aggregation (untrusted post-processing)
        for chunk_idx in range(self.n_chunks):
            timestamp = start_ts + chunk_idx

            decrypted = self._load_decrypted_chunk(timestamp)

            chunk_float = dequantize_parameters(
                decrypted, self.scale_factor, self.plaintext_modulus
            )
            chunk_float = chunk_float / len(results)

            start = chunk_idx * self.vector_dim
            end = start + self.vector_dim
            aggregated_flat[start:end] = chunk_float

        aggregated_flat = aggregated_flat[: self.model_param_count]
        aggregated_params = self._unflatten_parameters(aggregated_flat)

        return ndarrays_to_parameters(aggregated_params), {}

    def _unflatten_parameters(self, flat_params: np.ndarray) -> List[np.ndarray]:
        model = Net()
        params: List[np.ndarray] = []
        offset = 0

        for param in model.parameters():
            shape = param.shape
            numel = param.numel()
            param_flat = flat_params[offset : offset + numel]
            params.append(param_flat.reshape(shape))
            offset += numel

        return params


def server_fn(context: Context) -> ServerAppComponents:
    os.chdir(PROJECT_ROOT)

    num_rounds = context.run_config["num-server-rounds"]
    n_clients = context.run_config["num-clients"]
    vector_dim = context.run_config["vector-dim"]
    scale_factor = context.run_config["scale-factor"]
    plaintext_modulus = context.run_config["plaintext-modulus"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    model = Net()
    model_param_count = count_parameters(model)

    padded_len = ((model_param_count + vector_dim - 1) // vector_dim) * vector_dim
    n_chunks = padded_len // vector_dim
    n_timestamps = num_rounds * n_chunks

    print(f"[TERSE Server] Model has {model_param_count} parameters")
    print(f"[TERSE Server] Using {n_chunks} chunks per round, {n_timestamps} total timestamps")

    run_terse_setup(n_clients, n_timestamps, vector_dim)

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        return {"server_round": server_round}

    strategy = TERSEFedAvg(
        n_clients=n_clients,
        vector_dim=vector_dim,
        scale_factor=scale_factor,
        plaintext_modulus=plaintext_modulus,
        model_param_count=model_param_count,
        fraction_fit=1.0,
        fraction_evaluate=fraction_evaluate,
        initial_parameters=ndarrays_to_parameters(
            [p.detach().cpu().numpy() for p in model.parameters()]
        ),
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=fit_config,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


def get_evaluate_fn():
    def evaluate(server_round: int, parameters: List[np.ndarray], config: Dict[str, Scalar]):
        model = Net()

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        testloader = load_centralized_dataset()
        loss, accuracy = test(model, testloader, device)

        print(f"[Round {server_round}] Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        return loss, {"accuracy": accuracy}

    return evaluate


app = ServerApp(server_fn=server_fn)
