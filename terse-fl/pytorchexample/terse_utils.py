"""TERSE utility functions and automatic setup."""

import os
import sys
import subprocess
from pathlib import Path


# Paths - use resolve() to get absolute paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Go up to main project root
DATA_DIR = PROJECT_ROOT / "data"
PYTHON_DIR = PROJECT_ROOT / "python"
SETUP_CLIENTS_BIN = PROJECT_ROOT / "setup_clients"
SETUP_TRUSTED_BIN = PROJECT_ROOT / "setup_trusted"

# Add python directory to path for terse_py module
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

# Quantization parameters
PLAINTEXT_MODULUS = 65537
SCALE_FACTOR = 1000


def get_terse_module():
    """Import and return the terse_py module."""
    try:
        import terse_py
        return terse_py
    except ImportError as e:
        raise ImportError(
            f"Failed to import terse_py. Make sure you've run 'make python' "
            f"in the project root. Error: {e}\n"
            f"Expected location: {PYTHON_DIR}"
        )


def is_setup_complete(n_clients: int, n_timestamps: int, vector_dim: int) -> bool:
    """Check if TERSE setup has already been completed with matching parameters."""
    required_files = [
        DATA_DIR / "params.bin",
        DATA_DIR / "server_key.bin",
        DATA_DIR / "n_clients.txt",
        DATA_DIR / "n_timestamps.txt",
        DATA_DIR / "vector_dim.txt",
    ]

    # Check all required files exist
    for f in required_files:
        if not f.exists():
            return False

    # Check client precompute files
    for i in range(n_clients):
        if not (DATA_DIR / f"client_precompute_{i}.bin").exists():
            return False

    # Verify parameters match
    try:
        with open(DATA_DIR / "n_clients.txt") as f:
            saved_clients = int(f.read().strip())
        with open(DATA_DIR / "n_timestamps.txt") as f:
            saved_timestamps = int(f.read().strip())
        with open(DATA_DIR / "vector_dim.txt") as f:
            saved_vector_dim = int(f.read().strip())

        return (saved_clients >= n_clients and 
                saved_timestamps >= n_timestamps and 
                saved_vector_dim >= vector_dim)
    except Exception:
        return False


def run_terse_setup(n_clients: int, n_timestamps: int, vector_dim: int) -> None:
    """Run TERSE setup if not already done."""

    if is_setup_complete(n_clients, n_timestamps, vector_dim):
        print(f"[TERSE] Setup already complete for {n_clients} clients, "
              f"{n_timestamps} timestamps, vector_dim={vector_dim}")
        return

    print(f"[TERSE] Running setup for {n_clients} clients, "
          f"{n_timestamps} timestamps, vector_dim={vector_dim}...")

    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)

    # Check binaries exist
    if not SETUP_CLIENTS_BIN.exists():
        raise FileNotFoundError(
            f"setup_clients binary not found at {SETUP_CLIENTS_BIN}. "
            f"Run 'make' in the project root first."
        )
    if not SETUP_TRUSTED_BIN.exists():
        raise FileNotFoundError(
            f"setup_trusted binary not found at {SETUP_TRUSTED_BIN}. "
            f"Run 'make' in the project root first."
        )

    # Run client setup
    print("[TERSE] Running setup_clients...")
    result = subprocess.run(
        [str(SETUP_CLIENTS_BIN), str(n_clients), str(n_timestamps), str(vector_dim)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"setup_clients failed: {result.stderr}")
    print(result.stdout)

    # Run trusted setup
    print("[TERSE] Running setup_trusted...")
    result = subprocess.run(
        [str(SETUP_TRUSTED_BIN), str(n_clients), str(n_timestamps), str(vector_dim)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"setup_trusted failed: {result.stderr}")
    print(result.stdout)

    print("[TERSE] Setup complete!")


def quantize_parameters(flat_array, scale_factor: int, plaintext_mod: int):
    """Convert float array to quantized uint32 for encryption."""
    import numpy as np

    # Scale to integer range
    scaled = (flat_array * scale_factor).astype(np.int64)

    # Map to positive range [0, plaintext_mod)
    quantized = np.mod(scaled, plaintext_mod).astype(np.uint32)

    return quantized


def dequantize_parameters(quantized_array, scale_factor: int, plaintext_mod: int):
    """Convert quantized uint32 back to float array."""
    import numpy as np

    values = quantized_array.astype(np.int64)

    # Handle wrap-around for negative values
    half_mod = plaintext_mod // 2
    values = np.where(values > half_mod, values - plaintext_mod, values)

    # Scale back to float
    return values.astype(np.float32) / scale_factor
