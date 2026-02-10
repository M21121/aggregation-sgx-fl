// aes_client.cpp
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <chrono>
#include <numeric>
#include <array>
#include <filesystem>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <cstring>

using namespace std;
using namespace std::chrono;

namespace {

const size_t AES_KEY_BYTES = 16;
const size_t MASTER_KEY_BYTES = 32;
const unsigned KDF_ITERATIONS = 10000;
const char* MASTER_KEY_FILE = "data/aes_master_key.bin";

void ensure_data_dir() {
    filesystem::create_directories("data");
}

vector<uint8_t> load_or_create_master_key() {
    ensure_data_dir();
    vector<uint8_t> master(MASTER_KEY_BYTES);

    ifstream in(MASTER_KEY_FILE, ios::binary);
    if (in.good()) {
        in.read(reinterpret_cast<char*>(master.data()), master.size());
        if (static_cast<size_t>(in.gcount()) != master.size()) {
            throw runtime_error("Master key file corrupted");
        }
        return master;
    }

    if (RAND_bytes(master.data(), master.size()) != 1) {
        throw runtime_error("Failed to generate master key");
    }

    ofstream out(MASTER_KEY_FILE, ios::binary);
    if (!out) {
        throw runtime_error("Failed to write master key file");
    }
    out.write(reinterpret_cast<const char*>(master.data()), master.size());
    return master;
}

vector<uint8_t> derive_client_key(const vector<uint8_t>& master_key, size_t client_idx) {
    vector<uint8_t> key(AES_KEY_BYTES);

    array<uint8_t, 16> salt{};
    uint64_t idx_le = htole64(static_cast<uint64_t>(client_idx));
    memcpy(salt.data(), &idx_le, sizeof(idx_le));
    const string info = "AES-Client-" + to_string(client_idx);

    vector<uint8_t> salted_info(salt.begin(), salt.end());
    salted_info.insert(salted_info.end(), info.begin(), info.end());

    if (PKCS5_PBKDF2_HMAC(reinterpret_cast<const char*>(master_key.data()),
                          master_key.size(),
                          salted_info.data(),
                          salted_info.size(),
                          KDF_ITERATIONS,
                          EVP_sha256(),
                          key.size(),
                          key.data()) != 1) {
        throw runtime_error("KDF failed for client " + to_string(client_idx));
    }

    return key;
}

} // namespace

vector<uint8_t> aes_encrypt(const vector<uint64_t>& plaintext, const vector<uint8_t>& key) {
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        throw runtime_error("Failed to create cipher context");
    }

    vector<uint8_t> iv(16);
    if (RAND_bytes(iv.data(), 16) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        throw runtime_error("Failed to generate IV");
    }

    if (EVP_EncryptInit_ex(ctx, EVP_aes_128_cbc(), nullptr, key.data(), iv.data()) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        throw runtime_error("Failed to initialize encryption");
    }

    const uint8_t* in = reinterpret_cast<const uint8_t*>(plaintext.data());
    size_t in_len = plaintext.size() * sizeof(uint64_t);

    vector<uint8_t> ciphertext(in_len + EVP_CIPHER_block_size(EVP_aes_128_cbc()));
    int out_len1 = 0, out_len2 = 0;

    if (EVP_EncryptUpdate(ctx, ciphertext.data(), &out_len1, in, in_len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        throw runtime_error("Encryption failed");
    }

    if (EVP_EncryptFinal_ex(ctx, ciphertext.data() + out_len1, &out_len2) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        throw runtime_error("Encryption finalization failed");
    }

    EVP_CIPHER_CTX_free(ctx);

    vector<uint8_t> result;
    result.reserve(16 + out_len1 + out_len2);
    result.insert(result.end(), iv.begin(), iv.end());
    result.insert(result.end(), ciphertext.begin(), ciphertext.begin() + out_len1 + out_len2);

    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <n_clients> <n_timestamps> <vector_dim>" << endl;
        return 1;
    }

    size_t n_clients = stoull(argv[1]);
    size_t n_timestamps = stoull(argv[2]);
    size_t vector_dim = stoull(argv[3]);

    mt19937_64 rng(random_device{}());
    uniform_int_distribution<uint32_t> dist(0, 65536);

    vector<vector<uint64_t>> expected_sums(n_timestamps, vector<uint64_t>(vector_dim, 0));

    vector<double> per_client_keygen_times(n_clients);
    auto total_keygen_start = high_resolution_clock::now();

    vector<uint8_t> master_key = load_or_create_master_key();

    for (size_t client_idx = 0; client_idx < n_clients; client_idx++) {
        auto client_start = high_resolution_clock::now();

        vector<uint8_t> key = derive_client_key(master_key, client_idx);

        auto client_end = high_resolution_clock::now();
        per_client_keygen_times[client_idx] =
            duration_cast<nanoseconds>(client_end - client_start).count() / 1e6;

        string key_file = "data/aes_key_" + to_string(client_idx) + ".bin";
        ofstream key_out(key_file, ios::binary);
        if (!key_out) {
            throw runtime_error("Failed to open " + key_file);
        }
        key_out.write(reinterpret_cast<const char*>(key.data()), key.size());
    }

    auto total_keygen_end = high_resolution_clock::now();

    double avg_keygen_ms = accumulate(per_client_keygen_times.begin(),
                                      per_client_keygen_times.end(), 0.0) / n_clients;
    double total_keygen_wall_clock_ms =
        duration_cast<nanoseconds>(total_keygen_end - total_keygen_start).count() / 1e6;

    cout << "AES key generation time (per user, averaged): " << avg_keygen_ms << " ms" << endl;
    cout << "AES key generation time (wall clock): " << total_keygen_wall_clock_ms << " ms" << endl;

    vector<double> per_client_encrypt_times(n_clients, 0.0);

    for (size_t client_idx = 0; client_idx < n_clients; client_idx++) {
        string key_file = "data/aes_key_" + to_string(client_idx) + ".bin";
        ifstream key_in(key_file, ios::binary);
        vector<uint8_t> key(AES_KEY_BYTES);
        key_in.read(reinterpret_cast<char*>(key.data()), key.size());
        key_in.close();

        for (size_t ts_idx = 0; ts_idx < n_timestamps; ts_idx++) {
            vector<uint64_t> plaintext(vector_dim);
            for (size_t i = 0; i < vector_dim; i++) {
                plaintext[i] = dist(rng);
                expected_sums[ts_idx][i] += plaintext[i];
            }

            auto enc_start = high_resolution_clock::now();
            vector<uint8_t> ciphertext = aes_encrypt(plaintext, key);
            auto enc_end = high_resolution_clock::now();

            per_client_encrypt_times[client_idx] +=
                duration_cast<nanoseconds>(enc_end - enc_start).count() / 1e6;

            string ct_file = "data/aes_ct_" + to_string(client_idx) +
                             "_" + to_string(ts_idx) + ".bin";
            ofstream ct_out(ct_file, ios::binary);
            if (!ct_out) {
                throw runtime_error("Failed to open " + ct_file);
            }
            size_t ct_size = ciphertext.size();
            ct_out.write(reinterpret_cast<const char*>(&ct_size), sizeof(ct_size));
            ct_out.write(reinterpret_cast<const char*>(ciphertext.data()), ct_size);
        }
    }

    double total_encrypt_ms = accumulate(per_client_encrypt_times.begin(),
                                         per_client_encrypt_times.end(), 0.0);
    double avg_per_client_encrypt_ms = total_encrypt_ms / n_clients;
    double encrypt_per_vector_ms = avg_per_client_encrypt_ms / n_timestamps;

    for (size_t ts_idx = 0; ts_idx < n_timestamps; ts_idx++) {
        string sum_file = "data/aes_expected_sum_" + to_string(ts_idx) + ".txt";
        ofstream out(sum_file);
        if (!out) {
            throw runtime_error("Failed to open " + sum_file);
        }
        for (size_t i = 0; i < vector_dim; i++) {
            out << expected_sums[ts_idx][i];
            if (i + 1 < vector_dim) out << ' ';
        }
        out << endl;
    }

    ofstream meta("data/aes_metadata.txt");
    meta << n_clients << " " << n_timestamps << " " << vector_dim << endl;
    meta.close();

    cout << "AES encryption time (per vector, averaged): " << encrypt_per_vector_ms << " ms" << endl;

    return 0;
}
