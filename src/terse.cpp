// terse.cpp - Add batch methods
#include "terse.h"
#include <random>
#include <stdexcept>
#include <openssl/sha.h>
#include <numeric>
#include <cmath>

TERSEParams::TERSEParams(uint32_t N, uint64_t t, uint32_t depth,
                         SecurityLevel sec, double sigma)
    : poly_modulus_degree(N), plain_modulus(t), mult_depth(depth),
      sec_level(sec), error_stddev(sigma) {
    cipher_modulus = 0;
}

uint64_t TERSEParams::compute_secure_modulus(size_t n_clients) const {
    double log2_bound = std::log2(3.0) + std::log2(static_cast<double>(n_clients))
                        + std::log2(static_cast<double>(plain_modulus));

    uint32_t required_bits = static_cast<uint32_t>(std::ceil(log2_bound)) + 20;

    uint64_t q = (1ULL << required_bits) + 1;

    while (std::gcd(q, plain_modulus) != 1) {
        q += 2;
    }

    return q;
}

bool TERSEParams::validate_parameters(size_t n_clients) const {
    if (cipher_modulus == 0) {
        throw std::runtime_error("Cipher modulus not initialized");
    }

    if (std::gcd(cipher_modulus, plain_modulus) != 1) {
        throw std::runtime_error("q and t must be coprime");
    }

    double log2_3nt = std::log2(3.0) + std::log2(static_cast<double>(n_clients))
                      + std::log2(static_cast<double>(plain_modulus));
    double log2_q = std::log2(static_cast<double>(cipher_modulus));

    if (log2_3nt >= log2_q) {
        throw std::runtime_error(
            "Security constraint violated: log2(3*n*t) = " +
            std::to_string(log2_3nt) + " >= log2(q) = " + std::to_string(log2_q)
        );
    }

    return true;
}

void TERSEParams::save(const string& filename) const {
    ofstream out(filename, ios::binary);
    out.write(reinterpret_cast<const char*>(&poly_modulus_degree), sizeof(poly_modulus_degree));
    out.write(reinterpret_cast<const char*>(&plain_modulus), sizeof(plain_modulus));
    out.write(reinterpret_cast<const char*>(&cipher_modulus), sizeof(cipher_modulus));
    out.write(reinterpret_cast<const char*>(&mult_depth), sizeof(mult_depth));
    out.write(reinterpret_cast<const char*>(&sec_level), sizeof(sec_level));
    out.write(reinterpret_cast<const char*>(&error_stddev), sizeof(error_stddev));
    out.close();
}

TERSEParams TERSEParams::load(const string& filename) {
    ifstream in(filename, ios::binary);
    uint32_t N, depth;
    uint64_t t, q;
    SecurityLevel sec;
    double sigma;

    in.read(reinterpret_cast<char*>(&N), sizeof(N));
    in.read(reinterpret_cast<char*>(&t), sizeof(t));
    in.read(reinterpret_cast<char*>(&q), sizeof(q));
    in.read(reinterpret_cast<char*>(&depth), sizeof(depth));
    in.read(reinterpret_cast<char*>(&sec), sizeof(sec));
    in.read(reinterpret_cast<char*>(&sigma), sizeof(sigma));
    in.close();

    TERSEParams params(N, t, depth, sec, sigma);
    params.cipher_modulus = q;
    return params;
}

TERSESystem::TERSESystem(TERSEParams p)
    : params(p), n_clients(0), error_stddev(p.error_stddev) {

    CCParams<CryptoContextBFVRNS> parameters;
    parameters.SetPlaintextModulus(params.plain_modulus);
    parameters.SetMultiplicativeDepth(params.mult_depth);
    parameters.SetSecurityLevel(params.sec_level);
    parameters.SetRingDim(params.poly_modulus_degree);

    context = GenCryptoContext(parameters);
    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);

    element_params = context->GetCryptoParameters()->GetElementParams();
}

int64_t TERSESystem::sample_gaussian_error() const {
    static thread_local std::mt19937_64 gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, error_stddev);

    double sample = dist(gen);
    return static_cast<int64_t>(std::round(sample));
}

TimestampSplit TERSESystem::parse_timestamp(uint64_t timestamp, uint32_t poly_degree) {
    uint32_t lsb_bits = 0;
    uint32_t temp = poly_degree;
    while (temp > 1) {
        lsb_bits++;
        temp >>= 1;
    }

    size_t tau = timestamp & ((1ULL << lsb_bits) - 1);
    uint64_t theta = timestamp >> lsb_bits;

    return TimestampSplit(theta, tau);
}

vector<TERSEClient> TERSESystem::generate_client_keys(size_t n) {
    n_clients = n;

    params.cipher_modulus = params.compute_secure_modulus(n_clients);
    params.validate_parameters(n_clients);

    vector<TERSEClient> clients(n);

    DiscreteUniformGeneratorImpl<NativeVector> dug;
    dug.SetModulus(element_params->GetModulus());

    for (size_t i = 0; i < n; i++) {
        clients[i].secret_key = DCRTPoly(dug, element_params, Format::EVALUATION);
    }

    return clients;
}

TERSEServer TERSESystem::generate_server_key(const vector<TERSEClient>& clients) {
    TERSEServer server;

    server.server_key = DCRTPoly(element_params, Format::EVALUATION, true);

    for (const auto& client : clients) {
        server.server_key += client.secret_key;
    }

    server.server_key = server.server_key.Negate();

    return server;
}

DCRTPoly TERSESystem::generate_A_theta(uint64_t theta) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(&theta), sizeof(theta), hash);

    uint64_t seed = *reinterpret_cast<uint64_t*>(hash);
    mt19937_64 rng(seed);

    DiscreteUniformGeneratorImpl<NativeVector> dug;
    dug.SetModulus(element_params->GetModulus());

    return DCRTPoly(dug, element_params, Format::EVALUATION);
}

void TERSESystem::save_A_theta(const DCRTPoly& A_theta, const string& filename) const {
    ofstream out(filename, ios::binary);
    if (!out) {
        throw runtime_error("Failed to open " + filename);
    }

    // Save in coefficient form for consistency
    DCRTPoly A_copy = A_theta;
    A_copy.SetFormat(Format::COEFFICIENT);

    size_t num_towers = A_copy.GetNumOfElements();
    out.write(reinterpret_cast<const char*>(&num_towers), sizeof(num_towers));

    for (size_t i = 0; i < num_towers; i++) {
        const auto& tower = A_copy.GetElementAtIndex(i);
        const auto& values = tower.GetValues();
        size_t n_values = values.GetLength();
        out.write(reinterpret_cast<const char*>(&n_values), sizeof(n_values));

        // Save modulus for verification
        uint64_t modulus = tower.GetModulus().ConvertToInt();
        out.write(reinterpret_cast<const char*>(&modulus), sizeof(modulus));

        // Save coefficients
        for (size_t j = 0; j < n_values; j++) {
            uint64_t val = values[j].ConvertToInt();
            out.write(reinterpret_cast<const char*>(&val), sizeof(val));
        }
    }
    out.close();
}

DCRTPoly TERSESystem::load_A_theta(const string& filename) const {
    ifstream in(filename, ios::binary);
    if (!in) {
        throw runtime_error("Failed to open " + filename);
    }

    size_t num_towers;
    in.read(reinterpret_cast<char*>(&num_towers), sizeof(num_towers));

    DCRTPoly A_theta(element_params, Format::COEFFICIENT, true);

    for (size_t i = 0; i < num_towers; i++) {
        size_t n_values;
        in.read(reinterpret_cast<char*>(&n_values), sizeof(n_values));

        uint64_t modulus;
        in.read(reinterpret_cast<char*>(&modulus), sizeof(modulus));

        NativeVector values(n_values, NativeInteger(modulus));
        for (size_t j = 0; j < n_values; j++) {
            uint64_t val;
            in.read(reinterpret_cast<char*>(&val), sizeof(val));
            values[j] = NativeInteger(val);
        }

        NativePoly tower(element_params->GetParams()[i], Format::COEFFICIENT, true);
        tower.SetValues(values, Format::COEFFICIENT);
        A_theta.SetElementAtIndex(i, tower);
    }

    in.close();
    A_theta.SetFormat(Format::EVALUATION);
    return A_theta;
}


void TERSESystem::precompute_client(TERSEClient& client, const DCRTPoly& A_theta, size_t tau) {
    DCRTPoly product = A_theta * client.secret_key;

    product.SetFormat(Format::COEFFICIENT);

    const auto& first_tower = product.GetElementAtIndex(0);
    NativeInteger p_i = first_tower.GetValues()[tau];

    client.precomputed_p.push_back(p_i);
}

void TERSESystem::precompute_server(TERSEServer& server,
                                    const DCRTPoly& A_theta, size_t tau) {
    DCRTPoly product = A_theta * server.server_key;

    product.SetFormat(Format::COEFFICIENT);

    const auto& first_tower = product.GetElementAtIndex(0);
    NativeInteger p_prime = first_tower.GetValues()[tau];

    server.precomputed_p_prime.push_back(p_prime);
}

// NEW: Batch precomputation for client
void TERSESystem::precompute_client_batch(TERSEClient& client, const DCRTPoly& A_theta) {
    DCRTPoly product = A_theta * client.secret_key;
    product.SetFormat(Format::COEFFICIENT);

    const auto& first_tower = product.GetElementAtIndex(0);
    const auto& values = first_tower.GetValues();

    // Extract all N coefficients at once
    for (size_t tau = 0; tau < values.GetLength(); tau++) {
        client.precomputed_p.push_back(values[tau]);
    }
}

// NEW: Batch precomputation for server
void TERSESystem::precompute_server_batch(TERSEServer& server, const DCRTPoly& A_theta) {
    DCRTPoly product = A_theta * server.server_key;
    product.SetFormat(Format::COEFFICIENT);

    const auto& first_tower = product.GetElementAtIndex(0);
    const auto& values = first_tower.GetValues();

    // Extract all N coefficients at once
    for (size_t tau = 0; tau < values.GetLength(); tau++) {
        server.precomputed_p_prime.push_back(values[tau]);
    }
}

NativeInteger TERSESystem::encrypt(const TERSEClient& client, uint32_t plaintext,
                                   size_t timestamp_idx) {
    if (plaintext >= params.plain_modulus) {
        throw std::runtime_error("Plaintext exceeds plaintext modulus");
    }

    if (timestamp_idx >= client.precomputed_p.size()) {
        throw std::runtime_error("Timestamp index out of bounds");
    }

    NativeInteger q_mod = element_params->GetParams()[0]->GetModulus();

    int64_t e_i = sample_gaussian_error();

    int64_t combined = static_cast<int64_t>(params.plain_modulus) * e_i + 
                       static_cast<int64_t>(plaintext);

    NativeInteger result = client.precomputed_p[timestamp_idx];

    if (combined >= 0) {
        result = result.ModAdd(NativeInteger(static_cast<uint64_t>(combined)), q_mod);
    } else {
        result = result.ModSub(NativeInteger(static_cast<uint64_t>(-combined)), q_mod);
    }

    return result;
}

uint64_t TERSESystem::aggregate(const TERSEServer& server,
                                const vector<NativeInteger>& ciphertexts,
                                size_t timestamp_idx) {
    if (timestamp_idx >= server.precomputed_p_prime.size()) {
        throw std::runtime_error("Timestamp index out of bounds");
    }

    NativeInteger q_mod = element_params->GetParams()[0]->GetModulus();
    uint64_t t = params.plain_modulus;

    NativeInteger sum = server.precomputed_p_prime[timestamp_idx];

    for (const auto& ct : ciphertexts) {
        sum = sum.ModAdd(ct, q_mod);
    }

    uint64_t raw = sum.ConvertToInt();
    uint64_t q_val = q_mod.ConvertToInt();

    int64_t signed_val;
    if (raw > q_val / 2) {
        signed_val = static_cast<int64_t>(raw) - static_cast<int64_t>(q_val);
    } else {
        signed_val = static_cast<int64_t>(raw);
    }

    int64_t result = signed_val % static_cast<int64_t>(t);
    if (result < 0) {
        result += static_cast<int64_t>(t);
    }

    return static_cast<uint64_t>(result);
}

void TERSESystem::save_server_key(const TERSEServer& server, const string& filename) const {
    ofstream out(filename, ios::binary);

    size_t n_precomputed = server.precomputed_p_prime.size();
    out.write(reinterpret_cast<const char*>(&n_precomputed), sizeof(n_precomputed));

    for (const auto& p : server.precomputed_p_prime) {
        uint64_t val = p.ConvertToInt();
        out.write(reinterpret_cast<const char*>(&val), sizeof(val));
    }
    out.close();
}

TERSEServer TERSESystem::load_server_key(const string& filename) {
    ifstream in(filename, ios::binary);
    TERSEServer server;

    size_t n_precomputed;
    in.read(reinterpret_cast<char*>(&n_precomputed), sizeof(n_precomputed));

    server.precomputed_p_prime.resize(n_precomputed);
    for (size_t i = 0; i < n_precomputed; i++) {
        uint64_t val;
        in.read(reinterpret_cast<char*>(&val), sizeof(val));
        server.precomputed_p_prime[i] = NativeInteger(val);
    }
    in.close();
    return server;
}

void TERSESystem::save_ciphertexts(const vector<NativeInteger>& cts, const string& filename) const {
    ofstream out(filename, ios::binary);
    size_t n = cts.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));

    for (const auto& ct : cts) {
        uint64_t val = ct.ConvertToInt();
        out.write(reinterpret_cast<const char*>(&val), sizeof(val));
    }
    out.close();
}

vector<NativeInteger> TERSESystem::load_ciphertexts(const string& filename) const {
    ifstream in(filename, ios::binary);
    size_t n;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));

    vector<NativeInteger> cts(n);
    for (size_t i = 0; i < n; i++) {
        uint64_t val;
        in.read(reinterpret_cast<char*>(&val), sizeof(val));
        cts[i] = NativeInteger(val);
    }
    in.close();
    return cts;
}

void TERSESystem::save_ciphertext_matrix(const vector<vector<NativeInteger>>& cts,
                                         const string& filename) const {
    ofstream out(filename, ios::binary);

    size_t n_rows = cts.size();
    size_t n_cols = n_rows ? cts[0].size() : 0;
    out.write(reinterpret_cast<const char*>(&n_rows), sizeof(n_rows));
    out.write(reinterpret_cast<const char*>(&n_cols), sizeof(n_cols));

    vector<uint64_t> buffer;
    buffer.reserve(n_cols);

    for (const auto& row : cts) {
        buffer.clear();
        for (const auto& ct : row) {
            buffer.push_back(ct.ConvertToInt());
        }
        out.write(reinterpret_cast<const char*>(buffer.data()), 
                  buffer.size() * sizeof(uint64_t));
    }
    out.close();
}

vector<vector<NativeInteger>> TERSESystem::load_ciphertext_matrix(const string& filename) const {
    ifstream in(filename, ios::binary);
    size_t n_rows = 0, n_cols = 0;
    in.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
    in.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));

    vector<vector<NativeInteger>> cts(n_rows, vector<NativeInteger>(n_cols));

    vector<uint64_t> buffer(n_cols);

    for (size_t r = 0; r < n_rows; r++) {
        in.read(reinterpret_cast<char*>(buffer.data()), 
                buffer.size() * sizeof(uint64_t));
        for (size_t c = 0; c < n_cols; c++) {
            cts[r][c] = NativeInteger(buffer[c]);
        }
    }
    in.close();
    return cts;
}

void TERSESystem::save_aggregate_result(const NativeInteger& result, const string& filename) const {
    ofstream out(filename, ios::binary);
    uint64_t val = result.ConvertToInt();
    out.write(reinterpret_cast<const char*>(&val), sizeof(val));
    out.close();
}

NativeInteger TERSESystem::load_aggregate_result(const string& filename) const {
    ifstream in(filename, ios::binary);
    uint64_t val;
    in.read(reinterpret_cast<char*>(&val), sizeof(val));
    in.close();
    return NativeInteger(val);
}

void TERSESystem::save_aggregate_vector(const vector<NativeInteger>& result,
                                        const string& filename) const {
    ofstream out(filename, ios::binary);
    size_t n = result.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));

    vector<uint64_t> buffer;
    buffer.reserve(n);
    for (const auto& val : result) {
        buffer.push_back(val.ConvertToInt());
    }
    out.write(reinterpret_cast<const char*>(buffer.data()), 
              buffer.size() * sizeof(uint64_t));
    out.close();
}

vector<NativeInteger> TERSESystem::load_aggregate_vector(const string& filename) const {
    ifstream in(filename, ios::binary);
    size_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));

    vector<uint64_t> buffer(n);
    in.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(uint64_t));

    vector<NativeInteger> vals(n);
    for (size_t i = 0; i < n; i++) {
        vals[i] = NativeInteger(buffer[i]);
    }
    in.close();
    return vals;
}

void TERSESystem::save_client_keys(const vector<TERSEClient>& clients, const string& filename) const {
    ofstream out(filename, ios::binary);
    if (!out) {
        throw runtime_error("Failed to open " + filename);
    }

    size_t n_clients = clients.size();
    out.write(reinterpret_cast<const char*>(&n_clients), sizeof(n_clients));

    for (const auto& client : clients) {
        DCRTPoly sk_copy = client.secret_key;
        sk_copy.SetFormat(Format::COEFFICIENT);

        size_t num_towers = sk_copy.GetNumOfElements();
        out.write(reinterpret_cast<const char*>(&num_towers), sizeof(num_towers));

        for (size_t i = 0; i < num_towers; i++) {
            const auto& tower = sk_copy.GetElementAtIndex(i);
            const auto& values = tower.GetValues();
            size_t n_values = values.GetLength();
            out.write(reinterpret_cast<const char*>(&n_values), sizeof(n_values));

            for (size_t j = 0; j < n_values; j++) {
                uint64_t val = values[j].ConvertToInt();
                out.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        }
    }
    out.close();
}

vector<TERSEClient> TERSESystem::load_client_keys(const string& filename) {
    ifstream in(filename, ios::binary);
    if (!in) {
        throw runtime_error("Failed to open " + filename);
    }

    size_t n_clients;
    in.read(reinterpret_cast<char*>(&n_clients), sizeof(n_clients));

    vector<TERSEClient> clients(n_clients);

    for (size_t c = 0; c < n_clients; c++) {
        size_t num_towers;
        in.read(reinterpret_cast<char*>(&num_towers), sizeof(num_towers));

        clients[c].secret_key = DCRTPoly(element_params, Format::COEFFICIENT, true);

        for (size_t i = 0; i < num_towers; i++) {
            size_t n_values;
            in.read(reinterpret_cast<char*>(&n_values), sizeof(n_values));

            NativeVector values(n_values, element_params->GetParams()[i]->GetModulus());
            for (size_t j = 0; j < n_values; j++) {
                uint64_t val;
                in.read(reinterpret_cast<char*>(&val), sizeof(val));
                values[j] = NativeInteger(val);
            }

            NativePoly tower(element_params->GetParams()[i], Format::COEFFICIENT, true);
            tower.SetValues(values, Format::COEFFICIENT);
            clients[c].secret_key.SetElementAtIndex(i, tower);
        }

        clients[c].secret_key.SetFormat(Format::EVALUATION);
    }

    in.close();
    return clients;
}
