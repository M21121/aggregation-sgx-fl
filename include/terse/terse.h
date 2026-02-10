// terse.h
#ifndef TERSE_H
#define TERSE_H

#include "pke/openfhe.h"
#include <vector>
#include <memory>
#include <cstdint>
#include <numeric>
#include <cmath>
#include <fstream>
#include <string>

using namespace lbcrypto;
using namespace std;

struct TimestampSplit {
    uint64_t theta;
    size_t tau;
    TimestampSplit(uint64_t t, size_t ta) : theta(t), tau(ta) {}
};

class TERSEParams {
public:
    uint32_t poly_modulus_degree;
    uint64_t plain_modulus;
    uint64_t cipher_modulus;
    uint32_t mult_depth;
    SecurityLevel sec_level;
    double error_stddev;

    TERSEParams(uint32_t N = 8192, uint64_t t = 65537, uint32_t depth = 1,
                SecurityLevel sec = HEStd_128_classic, double sigma = 3.2);

    bool validate_parameters(size_t n_clients) const;
    uint64_t compute_secure_modulus(size_t n_clients) const;

    void save(const string& filename) const;
    static TERSEParams load(const string& filename);
};

class TERSEClient {
public:
    DCRTPoly secret_key;
    vector<NativeInteger> precomputed_p;

    TERSEClient() = default;
};

class TERSEServer {
public:
    DCRTPoly server_key;
    vector<NativeInteger> precomputed_p_prime;

    TERSEServer() = default;
};

class TERSESystem {
private:
    TERSEParams params;
    CryptoContext<DCRTPoly> context;
    shared_ptr<ILDCRTParams<BigInteger>> element_params;
    size_t n_clients;
    double error_stddev;

    int64_t sample_gaussian_error() const;

public:
    TERSESystem(TERSEParams p);

    static TimestampSplit parse_timestamp(uint64_t timestamp, uint32_t poly_degree);

    vector<TERSEClient> generate_client_keys(size_t n_clients);
    TERSEServer generate_server_key(const vector<TERSEClient>& clients);

    DCRTPoly generate_A_theta(uint64_t theta);
    void save_A_theta(const DCRTPoly& A_theta, const string& filename) const;
    DCRTPoly load_A_theta(const string& filename) const;
    
    void precompute_client(TERSEClient& client, const DCRTPoly& A_theta, size_t tau);
    void precompute_server(TERSEServer& server, const DCRTPoly& A_theta, size_t tau);

    // Batch precomputation methods
    void precompute_client_batch(TERSEClient& client, const DCRTPoly& A_theta);
    void precompute_server_batch(TERSEServer& server, const DCRTPoly& A_theta);

    NativeInteger encrypt(const TERSEClient& client, uint32_t plaintext, size_t timestamp_idx);
    uint64_t aggregate(const TERSEServer& server, const vector<NativeInteger>& ciphertexts,
                       size_t timestamp_idx);

    CryptoContext<DCRTPoly> get_context() const { return context; }
    const TERSEParams& get_params() const { return params; }
    size_t get_n_clients() const { return n_clients; }

    void save_server_key(const TERSEServer& server, const string& filename) const;
    TERSEServer load_server_key(const string& filename);

    void save_client_keys(const vector<TERSEClient>& clients, const string& filename) const;
    vector<TERSEClient> load_client_keys(const string& filename);

    void save_ciphertexts(const vector<NativeInteger>& cts, const string& filename) const;
    vector<NativeInteger> load_ciphertexts(const string& filename) const;

    void save_ciphertext_matrix(const vector<vector<NativeInteger>>& cts,
                                const string& filename) const;
    vector<vector<NativeInteger>> load_ciphertext_matrix(const string& filename) const;

    void save_aggregate_result(const NativeInteger& result, const string& filename) const;
    NativeInteger load_aggregate_result(const string& filename) const;

    void save_aggregate_vector(const vector<NativeInteger>& result,
                               const string& filename) const;
    vector<NativeInteger> load_aggregate_vector(const string& filename) const;
};

#endif
