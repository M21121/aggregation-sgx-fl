#!/usr/bin/env python3
import numpy as np
import sys
import os

# Ensure the module is in the path
sys.path.insert(0, os.getcwd())

try:
    import terse_py
    print("✓ Successfully imported terse_py module")
except ImportError as e:
    print(f"✗ Failed to import terse_py: {e}")
    sys.exit(1)

def test_basic_encryption_decryption():
    """Test basic encrypt -> aggregate -> decrypt flow"""
    print("\n=== Test 1: Basic Encryption/Decryption ===")

    # Setup parameters
    n_clients = 3
    n_timestamps = 2
    vector_dim = 10

    print(f"Clients: {n_clients}, Timestamps: {n_timestamps}, Vector dim: {vector_dim}")

    # Run C++ setup first
    print("\nRunning setup (C++)...")
    os.system(f"./setup_clients {n_clients} {n_timestamps} {vector_dim}")
    os.system(f"./setup_trusted {n_clients} {n_timestamps} {vector_dim}")

    # Initialize Python wrappers
    print("\nInitializing Python wrappers...")
    try:
        clients = [terse_py.TERSEClient("data/params.bin", i) 
                   for i in range(n_clients)]
        server = terse_py.TERSEServer("data/params.bin")
        trusted = terse_py.TERSETrusted("data/params.bin", "data/server_key.bin")
        print("✓ All wrappers initialized")
    except Exception as e:
        print(f"✗ Failed to initialize wrappers: {e}")
        return False

    # Test encryption
    print("\nTesting encryption...")
    timestamp = 0
    test_data = [
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint32),
        np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.uint32),
        np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=np.uint32)
    ]

    expected_sum = test_data[0] + test_data[1] + test_data[2]
    print(f"Expected sum: {expected_sum}")

    try:
        encrypted = [client.encrypt_vector(data, timestamp) 
                     for client, data in zip(clients, test_data)]
        print(f"✓ Encrypted {len(encrypted)} vectors")
        print(f"  Ciphertext shape: {encrypted[0].shape}")
        print(f"  Ciphertext dtype: {encrypted[0].dtype}")
    except Exception as e:
        print(f"✗ Encryption failed: {e}")
        return False

    # Test aggregation
    print("\nTesting aggregation...")
    try:
        aggregate = server.aggregate_ciphertexts(encrypted, timestamp)
        print(f"✓ Aggregated ciphertexts")
        print(f"  Aggregate shape: {aggregate.shape}")

        server.save_aggregate(aggregate, timestamp)
        print("✓ Saved aggregate to disk")
    except Exception as e:
        print(f"✗ Aggregation failed: {e}")
        return False

    # Test decryption
    print("\nTesting decryption...")
    try:
        decrypted = trusted.decrypt_aggregate(timestamp, vector_dim)
        print(f"✓ Decrypted aggregate")
        print(f"  Decrypted: {decrypted}")
        print(f"  Expected:  {expected_sum}")

        if np.array_equal(decrypted, expected_sum):
            print("✓ Decryption matches expected sum!")
            return True
        else:
            print("✗ Decryption mismatch!")
            print(f"  Difference: {decrypted - expected_sum}")
            return False
    except Exception as e:
        print(f"✗ Decryption failed: {e}")
        return False

def test_multiple_timestamps():
    """Test multiple rounds of encryption"""
    print("\n=== Test 2: Multiple Timestamps ===")

    n_clients = 2
    n_timestamps = 5
    vector_dim = 4

    print(f"Testing {n_timestamps} timestamps...")

    os.system(f"./setup_clients {n_clients} {n_timestamps} {vector_dim}")
    os.system(f"./setup_trusted {n_clients} {n_timestamps} {vector_dim}")

    clients = [terse_py.TERSEClient("data/params.bin", i) 
               for i in range(n_clients)]
    server = terse_py.TERSEServer("data/params.bin")
    trusted = terse_py.TERSETrusted("data/params.bin", "data/server_key.bin")

    all_passed = True

    for ts in range(n_timestamps):
        # Generate random data
        data = [np.random.randint(0, 100, vector_dim, dtype=np.uint32) 
                for _ in range(n_clients)]
        expected = sum(data)

        # Encrypt, aggregate, decrypt
        encrypted = [client.encrypt_vector(d, ts) 
                     for client, d in zip(clients, data)]
        aggregate = server.aggregate_ciphertexts(encrypted, ts)
        server.save_aggregate(aggregate, ts)
        decrypted = trusted.decrypt_aggregate(ts, vector_dim)

        if np.array_equal(decrypted, expected):
            print(f"  Timestamp {ts}: ✓")
        else:
            print(f"  Timestamp {ts}: ✗ (diff: {np.max(np.abs(decrypted - expected))})")
            all_passed = False

    return all_passed

def test_large_vector():
    """Test with larger vectors"""
    print("\n=== Test 3: Large Vector ===")

    n_clients = 5
    n_timestamps = 1
    vector_dim = 1000

    print(f"Testing vector dimension: {vector_dim}")

    os.system(f"./setup_clients {n_clients} {n_timestamps} {vector_dim}")
    os.system(f"./setup_trusted {n_clients} {n_timestamps} {vector_dim}")

    clients = [terse_py.TERSEClient("data/params.bin", i) 
               for i in range(n_clients)]
    server = terse_py.TERSEServer("data/params.bin")
    trusted = terse_py.TERSETrusted("data/params.bin", "data/server_key.bin")

    # Generate data
    data = [np.random.randint(0, 1000, vector_dim, dtype=np.uint32) 
            for _ in range(n_clients)]
    expected = sum(data)

    # Process
    encrypted = [client.encrypt_vector(d, 0) 
                 for client, d in zip(clients, data)]
    aggregate = server.aggregate_ciphertexts(encrypted, 0)
    server.save_aggregate(aggregate, 0)
    decrypted = trusted.decrypt_aggregate(0, vector_dim)

    matches = np.array_equal(decrypted, expected)
    if matches:
        print(f"✓ Large vector test passed")
    else:
        max_diff = np.max(np.abs(decrypted - expected))
        print(f"✗ Large vector test failed (max diff: {max_diff})")

    return matches

def test_error_handling():
    """Test error conditions"""
    print("\n=== Test 4: Error Handling ===")

    # Setup minimal system
    os.system("./setup_clients 1 1 1")
    os.system("./setup_trusted 1 1 1")

    client = terse_py.TERSEClient("data/params.bin", 0)

    # Test 1: Invalid timestamp
    print("Testing invalid timestamp...")
    try:
        data = np.array([1], dtype=np.uint32)
        client.encrypt_vector(data, 999)  # Out of bounds
        print("✗ Should have raised exception")
        return False
    except Exception as e:
        print(f"✓ Correctly raised exception: {type(e).__name__}")

    # Test 2: Wrong array size
    print("Testing wrong array size...")
    try:
        data = np.array([1, 2, 3], dtype=np.uint32)  # Expected size 1
        client.encrypt_vector(data, 0)
        print("✗ Should have raised exception")
        return False
    except Exception as e:
        print(f"✓ Correctly raised exception: {type(e).__name__}")

    # Test 3: Value exceeding plaintext modulus
    print("Testing value exceeding modulus...")
    try:
        # Plain modulus is 65537, so this should fail
        data = np.array([100000], dtype=np.uint32)
        client.encrypt_vector(data, 0)
        print("✗ Should have raised exception")
        return False
    except Exception as e:
        print(f"✓ Correctly raised exception: {type(e).__name__}")

    return True

if __name__ == "__main__":
    print("=" * 60)
    print("TERSE Python Bindings Test Suite")
    print("=" * 60)

    # Clean up before tests
    os.system("rm -rf data && mkdir -p data")

    results = []

    results.append(("Basic Encryption/Decryption", test_basic_encryption_decryption()))
    results.append(("Multiple Timestamps", test_multiple_timestamps()))
    results.append(("Large Vector", test_large_vector()))
    results.append(("Error Handling", test_error_handling()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")

    all_passed = all(result[1] for result in results)
    print("=" * 60)

    if all_passed:
        print("All tests passed! 🎉")
        sys.exit(0)
    else:
        print("Some tests failed.")
        sys.exit(1)
