// server.cpp
#include "terse/terse.h"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <numeric>

using namespace std;
using namespace std::chrono;

static TERSEParams load_params() {
    return TERSEParams::load("data/params.bin");
}

static size_t load_size_from_file(const string& filename) {
    ifstream in(filename);
    if (!in) throw runtime_error("Failed to open " + filename);
    size_t value;
    in >> value;
    return value;
}

static vector<vector<vector<NativeInteger>>> load_all_ciphertexts(
    TERSESystem& system, size_t n_clients) {

    vector<vector<vector<NativeInteger>>> all_cts(n_clients);

    for (size_t i = 0; i < n_clients; i++) {
        string filename = "data/ciphertexts_client_" + to_string(i) + ".bin";
        all_cts[i] = system.load_ciphertext_matrix(filename);
    }

    return all_cts;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <n_clients> <n_timestamps> <vector_dim>" << endl;
        return 1;
    }

    size_t n_clients = stoull(argv[1]);
    size_t n_timestamps = stoull(argv[2]);
    size_t vector_dim = stoull(argv[3]);

    TERSEParams params = load_params();
    TERSESystem system(params);

    size_t expected_n_clients = load_size_from_file("data/n_clients.txt");
    size_t expected_n_timestamps = load_size_from_file("data/n_timestamps.txt");
    size_t expected_vector_dim = load_size_from_file("data/vector_dim.txt");

    if (n_clients > expected_n_clients) {
        cerr << "Error: Requested " << n_clients << " clients, but only "
             << expected_n_clients << " were set up" << endl;
        return 1;
    }

    if (n_timestamps > expected_n_timestamps || vector_dim > expected_vector_dim) {
        cerr << "Error: Requested dimensions exceed setup parameters" << endl;
        return 1;
    }

    NativeInteger q_mod = system.get_context()->GetCryptoParameters()
                          ->GetElementParams()->GetParams()[0]->GetModulus();
    uint64_t q_val = q_mod.ConvertToInt();

    cout << "=== TERSE Server Aggregation ===" << endl;
    cout << "Loading ciphertexts from " << n_clients << " clients..." << flush;

    auto load_ct_start = high_resolution_clock::now();
    vector<vector<vector<NativeInteger>>> all_ciphertexts = load_all_ciphertexts(system, n_clients);
    auto load_ct_end = high_resolution_clock::now();
    double load_ct_ms = duration_cast<nanoseconds>(load_ct_end - load_ct_start).count() / 1e6;

    cout << " Done (" << load_ct_ms << " ms)" << endl;

    vector<double> per_aggregation_times;
    per_aggregation_times.reserve(n_timestamps);

    cout << "\nAggregating ciphertexts (untrusted - no decryption)..." << endl;
    auto agg_total_start = high_resolution_clock::now();

    for (size_t ts = 0; ts < n_timestamps; ts++) {
        // Progress update
        if (n_timestamps >= 1000) {
            size_t progress_interval = max(1UL, n_timestamps / 100);
            if (ts % progress_interval == 0 || ts == n_timestamps - 1) {
                double percent = (100.0 * ts) / n_timestamps;
                auto current_time = high_resolution_clock::now();
                double elapsed_ms = duration_cast<milliseconds>(current_time - agg_total_start).count();
                double rate = (ts > 0) ? (ts / (elapsed_ms / 1000.0)) : 0;

                cout << "\r  Progress: " << fixed << setprecision(1) << percent << "% "
                     << "(" << ts << "/" << n_timestamps << " timestamps, "
                     << static_cast<int>(rate) << " ts/sec)" << flush;
            }
        }

        auto agg_start = high_resolution_clock::now();

        // PUBLIC AGGREGATION: Just sum ciphertexts (no p')
        // ŷ_ts,j ← Σᵢ cᵢ,ts,j mod q
        vector<NativeInteger> aggregate(vector_dim);

        for (size_t coord = 0; coord < vector_dim; coord++) {
            __uint128_t sum = 0;

            for (size_t client_idx = 0; client_idx < n_clients; client_idx++) {
                sum += all_ciphertexts[client_idx][ts][coord].ConvertToInt();
            }

            uint64_t reduced = static_cast<uint64_t>(sum % q_val);
            aggregate[coord] = NativeInteger(reduced);
        }

        // Save encrypted aggregate for trusted party
        string agg_file = "data/encrypted_aggregate_" + to_string(ts) + ".bin";
        system.save_aggregate_vector(aggregate, agg_file);

        auto agg_end = high_resolution_clock::now();
        double agg_time_ms = duration_cast<nanoseconds>(agg_end - agg_start).count() / 1e6;
        per_aggregation_times.push_back(agg_time_ms);
    }

    auto agg_total_end = high_resolution_clock::now();
    double agg_total_ms = duration_cast<nanoseconds>(agg_total_end - agg_total_start).count() / 1e6;

    cout << "\r" << string(80, ' ') << "\r";  // Clear progress line

    double avg_per_agg_ms = accumulate(per_aggregation_times.begin(), 
                                       per_aggregation_times.end(), 0.0) / per_aggregation_times.size();

    cout << "\n=== Aggregation Results ===" << endl;
    cout << "Total aggregation time: " << agg_total_ms << " ms" << endl;
    cout << "Aggregation time (per timestamp, averaged): " << avg_per_agg_ms << " ms" << endl;
    cout << "Throughput: " << (n_timestamps * 1000.0) / agg_total_ms << " aggregations/second" << endl;

    cout << "\nEncrypted aggregates saved to ./data/encrypted_aggregate_*.bin" << endl;
    cout << "Run trusted.cpp to decrypt results" << endl;

    return 0;
}
