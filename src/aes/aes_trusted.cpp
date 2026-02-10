// aes_trusted.cpp 
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <openssl/evp.h>
#include "common/dp_mechanisms.h"

using namespace std;
using namespace std::chrono;

struct CiphertextData {
    size_t client_idx;
    vector<uint8_t> ciphertext;
};

vector<CiphertextData> load_ciphertexts_untrusted(size_t ts_idx, size_t n_clients) {
    vector<CiphertextData> ciphertexts;
    ciphertexts.reserve(n_clients);

    for (size_t client_idx = 0; client_idx < n_clients; client_idx++) {
        string ct_file = "data/aes_ct_" + to_string(client_idx) + 
                       "_" + to_string(ts_idx) + ".bin";
        ifstream ct_in(ct_file, ios::binary);
        if (!ct_in) throw runtime_error("Failed to open " + ct_file);

        size_t ct_size;
        ct_in.read(reinterpret_cast<char*>(&ct_size), sizeof(ct_size));

        CiphertextData ct_data;
        ct_data.client_idx = client_idx;
        ct_data.ciphertext.resize(ct_size);
        ct_in.read(reinterpret_cast<char*>(ct_data.ciphertext.data()), ct_size);
        ct_in.close();

        ciphertexts.push_back(move(ct_data));
    }

    return ciphertexts;
}

vector<uint64_t> aes_decrypt(const vector<uint8_t>& ciphertext, 
                             const vector<uint8_t>& key, 
                             size_t vector_dim) {
    if (ciphertext.size() < 16) {
        throw runtime_error("Ciphertext too short");
    }

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) throw runtime_error("Failed to create cipher context");

    vector<uint8_t> iv(ciphertext.begin(), ciphertext.begin() + 16);
    const uint8_t* ct_data = ciphertext.data() + 16;
    size_t ct_len = ciphertext.size() - 16;

    if (EVP_DecryptInit_ex(ctx, EVP_aes_128_cbc(), nullptr, key.data(), iv.data()) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        throw runtime_error("Failed to initialize decryption");
    }

    vector<uint8_t> plaintext_bytes(ct_len + EVP_CIPHER_block_size(EVP_aes_128_cbc()));
    int out_len1 = 0, out_len2 = 0;

    if (EVP_DecryptUpdate(ctx, plaintext_bytes.data(), &out_len1, ct_data, ct_len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        throw runtime_error("Decryption failed");
    }

    if (EVP_DecryptFinal_ex(ctx, plaintext_bytes.data() + out_len1, &out_len2) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        throw runtime_error("Decryption finalization failed");
    }

    EVP_CIPHER_CTX_free(ctx);

    size_t total_bytes = out_len1 + out_len2;
    if (total_bytes < vector_dim * sizeof(uint64_t)) {
        throw runtime_error("Decrypted data too short");
    }

    vector<uint64_t> plaintext(vector_dim);
    memcpy(plaintext.data(), plaintext_bytes.data(), vector_dim * sizeof(uint64_t));

    return plaintext;
}

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 5) {
        cerr << "Usage: " << argv[0] << " <n_timestamps> [epsilon] [dp_mechanism] [delta]" << endl;
        cerr << "  dp_mechanism: none (default), laplace, gaussian" << endl;
        cerr << "  delta: required for gaussian mechanism (e.g., 1e-5)" << endl;
        return 1;
    }

    size_t n_timestamps = stoull(argv[1]);
    double epsilon = (argc >= 3) ? stod(argv[2]) : 0.0;
    string dp_mechanism = (argc >= 4) ? string(argv[3]) : "none";
    double delta = (argc >= 5) ? stod(argv[4]) : 0.0;

    bool enable_dp = (dp_mechanism != "none");

    if (enable_dp && epsilon <= 0) {
        cerr << "Error: epsilon must be positive when DP is enabled" << endl;
        return 1;
    }

    if (dp_mechanism == "gaussian" && delta <= 0) {
        cerr << "Error: delta must be positive for Gaussian mechanism" << endl;
        return 1;
    }

    ifstream meta("data/aes_metadata.txt");
    if (!meta) throw runtime_error("Failed to open metadata file");

    size_t n_clients, n_ts, vector_dim;
    meta >> n_clients >> n_ts >> vector_dim;
    meta.close();

    if (n_ts != n_timestamps) {
        throw runtime_error("Timestamp count mismatch");
    }

    cout << "=== AES Decryption ===" << endl;
    if (enable_dp) {
        cout << "Differential Privacy: ENABLED" << endl;
        cout << "  Mechanism: " << dp_mechanism << endl;
        cout << "  Epsilon: " << epsilon << endl;
        if (dp_mechanism == "gaussian") {
            cout << "  Delta: " << delta << endl;
        }
        cout << "  Sensitivity: 65536" << endl;
        cout << "  Total privacy budget: " << (epsilon * n_timestamps) << endl;
    } else {
        cout << "Differential Privacy: DISABLED" << endl;
    }

    cout << "\n=== One-Time Setup: Key Provisioning ===" << endl;
    cout << "Loading " << n_clients << " client keys into memory..." << endl;

    auto key_provision_start = high_resolution_clock::now();

    unordered_map<size_t, vector<uint8_t>> key_table;
    key_table.reserve(n_clients);

    for (size_t client_idx = 0; client_idx < n_clients; client_idx++) {
        string key_file = "data/aes_key_" + to_string(client_idx) + ".bin";
        ifstream key_in(key_file, ios::binary);
        if (!key_in) throw runtime_error("Failed to open " + key_file);

        vector<uint8_t> key(16);
        key_in.read(reinterpret_cast<char*>(key.data()), 16);
        key_in.close();

        key_table[client_idx] = key;
    }

    auto key_provision_end = high_resolution_clock::now();
    double key_provision_ms = 
        duration_cast<nanoseconds>(key_provision_end - key_provision_start).count() / 1e6;

    cout << "Key provisioning time (one-time): " << key_provision_ms << " ms" << endl;
    cout << "Average per key: " << (key_provision_ms / n_clients) << " ms" << endl;

    cout << "\n=== Runtime: Decryption and Aggregation ===" << endl;
    cout << "Processing " << n_timestamps << " timestamps..." << endl;

    double total_untrusted_io_ms = 0;
    double total_trusted_decrypt_aggregate_ms = 0;
    double total_dp_noise_ms = 0;
    bool all_passed = true;

    for (size_t ts_idx = 0; ts_idx < n_timestamps; ts_idx++) {
        // UNTRUSTED: Load ciphertexts
        auto io_start = high_resolution_clock::now();
        vector<CiphertextData> ciphertexts = load_ciphertexts_untrusted(ts_idx, n_clients);
        auto io_end = high_resolution_clock::now();
        double io_time_ms = duration_cast<nanoseconds>(io_end - io_start).count() / 1e6;
        total_untrusted_io_ms += io_time_ms;

        // TRUSTED: Decrypt and aggregate
        auto trusted_start = high_resolution_clock::now();

        vector<uint64_t> aggregate(vector_dim, 0);

        for (const auto& ct_data : ciphertexts) {
            vector<uint64_t> plaintext = aes_decrypt(ct_data.ciphertext, 
                                                     key_table[ct_data.client_idx], 
                                                     vector_dim);

            for (size_t i = 0; i < vector_dim; i++) {
                aggregate[i] += plaintext[i];
            }
        }

        auto trusted_end = high_resolution_clock::now();
        double trusted_time_ms = duration_cast<nanoseconds>(trusted_end - trusted_start).count() / 1e6;
        total_trusted_decrypt_aggregate_ms += trusted_time_ms;

        // TRUSTED: Apply differential privacy if enabled
        vector<double> final_result;
        if (enable_dp) {
            auto dp_start = high_resolution_clock::now();

            // Convert to uint32_t for DP functions
            vector<uint32_t> aggregate_u32(vector_dim);
            for (size_t i = 0; i < vector_dim; i++) {
                aggregate_u32[i] = static_cast<uint32_t>(aggregate[i] & 0xFFFFFFFF);
            }

            uint32_t sensitivity = 65536;  // Maximum value per coordinate in AES

            if (dp_mechanism == "laplace") {
                final_result = add_laplace_noise(aggregate_u32, epsilon, sensitivity);
            } else if (dp_mechanism == "gaussian") {
                final_result = add_gaussian_noise(aggregate_u32, epsilon, delta, sensitivity);
            } else {
                cerr << "Unknown DP mechanism: " << dp_mechanism << endl;
                return 1;
            }

            auto dp_end = high_resolution_clock::now();
            double dp_time_ms = duration_cast<nanoseconds>(dp_end - dp_start).count() / 1e6;
            total_dp_noise_ms += dp_time_ms;

            // Save noisy results
            string noisy_file = "data/aes_noisy_sum_" + to_string(ts_idx) + ".txt";
            ofstream noisy_out(noisy_file);
            for (size_t i = 0; i < vector_dim; i++) {
                noisy_out << final_result[i];
                if (i + 1 < vector_dim) noisy_out << ' ';
            }
            noisy_out << endl;
            noisy_out.close();
        }

        // Verify (only if DP is disabled)
        if (!enable_dp) {
            string sum_file = "data/aes_expected_sum_" + to_string(ts_idx) + ".txt";
            ifstream sum_in(sum_file);
            if (!sum_in) throw runtime_error("Failed to open " + sum_file);

            for (size_t i = 0; i < vector_dim; i++) {
                uint64_t expected;
                sum_in >> expected;
                if (aggregate[i] != expected) {
                    cerr << "Timestamp " << ts_idx << " coord " << i 
                         << " FAILED: Expected " << expected 
                         << ", Got " << aggregate[i] << endl;
                    all_passed = false;
                }
            }
            sum_in.close();
        }
    }

    cout << "\n=== Performance Results ===" << endl;
    cout << "One-time key provisioning: " << key_provision_ms << " ms" << endl;
    cout << "\nPer-Timestamp Breakdown:" << endl;
    cout << "  Untrusted I/O (avg): " << (total_untrusted_io_ms / n_timestamps) << " ms" << endl;
    cout << "  Trusted decrypt+aggregate (avg): " << (total_trusted_decrypt_aggregate_ms / n_timestamps) << " ms" << endl;
    if (enable_dp) {
        cout << "  DP noise addition (avg): " << (total_dp_noise_ms / n_timestamps) << " ms" << endl;
    }
    cout << "  Total per timestamp: " << ((total_untrusted_io_ms + total_trusted_decrypt_aggregate_ms + total_dp_noise_ms) / n_timestamps) << " ms" << endl;

    cout << "\nTotal Times:" << endl;
    cout << "  Untrusted I/O: " << total_untrusted_io_ms << " ms" << endl;
    cout << "  Trusted operations: " << total_trusted_decrypt_aggregate_ms << " ms" << endl;
    if (enable_dp) {
        cout << "  DP noise addition: " << total_dp_noise_ms << " ms" << endl;
    }
    cout << "  Combined runtime: " << (total_untrusted_io_ms + total_trusted_decrypt_aggregate_ms + total_dp_noise_ms) << " ms" << endl;

    cout << "\nAmortized (including setup): " 
         << ((key_provision_ms + total_untrusted_io_ms + total_trusted_decrypt_aggregate_ms + total_dp_noise_ms) / n_timestamps) 
         << " ms per timestamp" << endl;

    if (enable_dp) {
        cout << "\nDifferential Privacy Summary:" << endl;
        cout << "  Noisy results saved to data/aes_noisy_sum_*.txt" << endl;
        cout << "  Privacy guarantee: " << dp_mechanism << "-DP" << endl;
        cout << "  Per-timestamp epsilon: " << epsilon << endl;
        cout << "  Total privacy cost: " << (epsilon * n_timestamps) << endl;
        if (dp_mechanism == "gaussian") {
            cout << "  Delta: " << delta << endl;
        }
    } else {
        cout << "\nVerification: " << (all_passed ? "ALL PASSED" : "SOME FAILED") << endl;
    }

    return (enable_dp || all_passed) ? 0 : 1;
}
