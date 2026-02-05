// client.cpp
#include "terse.h"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <numeric>
#include <random>
#include <omp.h>

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

static vector<vector<NativeInteger>> load_client_precomputes(size_t n_clients, size_t expected_streams) {
    vector<vector<NativeInteger>> precomputes(n_clients);

    for (size_t idx = 0; idx < n_clients; idx++) {
        string filename = "data/client_precompute_" + to_string(idx) + ".bin";
        ifstream in(filename, ios::binary);
        if (!in) throw runtime_error("Failed to open " + filename);

        size_t n_entries;
        in.read(reinterpret_cast<char*>(&n_entries), sizeof(n_entries));

        if (n_entries != expected_streams) {
            throw runtime_error("Client " + to_string(idx) + " precompute size mismatch");
        }

        vector<uint64_t> buffer(n_entries);
        in.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(uint64_t));

        precomputes[idx].resize(n_entries);
        for (size_t i = 0; i < n_entries; i++) {
            precomputes[idx][i] = NativeInteger(buffer[i]);
        }
    }

    return precomputes;
}

int main(int argc, char* argv[]) {
    omp_set_num_threads(1);

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

    size_t total_streams = n_timestamps * vector_dim;
    size_t expected_streams = expected_n_timestamps * expected_vector_dim;

    if (total_streams > expected_streams) {
        cerr << "Error: Requested " << total_streams << " streams, but only "
             << expected_streams << " were precomputed" << endl;
        return 1;
    }

    cout << "=== TERSE Client Encryption ===" << endl;
    cout << "Loading precomputed values for " << n_clients << " clients..." << flush;

    auto load_start = high_resolution_clock::now();
    vector<vector<NativeInteger>> client_precomputes = load_client_precomputes(n_clients, expected_streams);
    auto load_end = high_resolution_clock::now();
    double load_time_ms = duration_cast<nanoseconds>(load_end - load_start).count() / 1e6;

    cout << " Done (" << load_time_ms << " ms)" << endl;

    random_device rd;
    mt19937_64 gen(rd());
    uniform_int_distribution<uint32_t> plaintext_dist(0, params.plain_modulus - 1);

    vector<vector<vector<NativeInteger>>> all_ciphertexts(n_clients);
    for (size_t i = 0; i < n_clients; i++) {
        all_ciphertexts[i].resize(n_timestamps, vector<NativeInteger>(vector_dim));
    }

    vector<vector<uint64_t>> expected_sums(n_timestamps, vector<uint64_t>(vector_dim, 0));
    vector<double> per_vector_times;
    per_vector_times.reserve(n_clients * n_timestamps);

    cout << "\nEncrypting data..." << endl;
    auto encrypt_total_start = high_resolution_clock::now();

    for (size_t client_idx = 0; client_idx < n_clients; client_idx++) {
        if (n_clients > 1) {
            cout << "\rClient " << (client_idx + 1) << "/" << n_clients << flush;
        }

        TERSEClient client;
        client.precomputed_p = client_precomputes[client_idx];

        for (size_t ts = 0; ts < n_timestamps; ts++) {
            if (n_clients == 1 && n_timestamps >= 10000) {
                size_t progress_interval = max(1UL, n_timestamps / 100);
                if (ts % progress_interval == 0 || ts == n_timestamps - 1) {
                    double percent = (100.0 * ts) / n_timestamps;
                    auto current_time = high_resolution_clock::now();
                    double elapsed_ms = duration_cast<milliseconds>(current_time - encrypt_total_start).count();
                    double rate = (ts > 0) ? (ts / (elapsed_ms / 1000.0)) : 0;

                    cout << "\r  Progress: " << fixed << setprecision(1) << percent << "% "
                         << "(" << ts << "/" << n_timestamps << " timestamps, "
                         << static_cast<int>(rate) << " ts/sec)" << flush;
                }
            }

            auto vec_start = high_resolution_clock::now();

            for (size_t dim = 0; dim < vector_dim; dim++) {
                uint32_t plaintext = plaintext_dist(gen);
                size_t stream_idx = ts * vector_dim + dim;

                NativeInteger ct = system.encrypt(client, plaintext, stream_idx);
                all_ciphertexts[client_idx][ts][dim] = ct;

                expected_sums[ts][dim] = (expected_sums[ts][dim] + plaintext) % params.plain_modulus;
            }

            auto vec_end = high_resolution_clock::now();
            double vec_time_ms = duration_cast<nanoseconds>(vec_end - vec_start).count() / 1e6;
            per_vector_times.push_back(vec_time_ms);
        }
    }

    auto encrypt_total_end = high_resolution_clock::now();
    double encrypt_total_ms = duration_cast<nanoseconds>(encrypt_total_end - encrypt_total_start).count() / 1e6;

    cout << "\r" << string(80, ' ') << "\r";

    double avg_per_vector_ms = accumulate(per_vector_times.begin(), per_vector_times.end(), 0.0) / per_vector_times.size();

    cout << "\n=== Encryption Results ===" << endl;
    cout << "Total encryption time: " << encrypt_total_ms << " ms" << endl;
    cout << "Encryption time (per vector, averaged): " << avg_per_vector_ms << " ms" << endl;
    cout << "Encryption time (per client, total): " << encrypt_total_ms / n_clients << " ms" << endl;
    cout << "Throughput: " << (n_clients * n_timestamps * vector_dim * 1000.0) / encrypt_total_ms << " encryptions/second" << endl;

    cout << "\nSaving ciphertexts..." << flush;
    auto save_start = high_resolution_clock::now();

    for (size_t client_idx = 0; client_idx < n_clients; client_idx++) {
        if (n_clients > 10) {
            cout << "\r  Saving client " << (client_idx + 1) << "/" << n_clients << flush;
        }

        string filename = "data/ciphertexts_client_" + to_string(client_idx) + ".bin";
        system.save_ciphertext_matrix(all_ciphertexts[client_idx], filename);
    }

    auto save_end = high_resolution_clock::now();
    double save_time_ms = duration_cast<nanoseconds>(save_end - save_start).count() / 1e6;

    cout << "\r" << string(80, ' ') << "\r";
    cout << "Ciphertext saving time: " << save_time_ms << " ms" << endl;

    cout << "\nSaving expected sums..." << flush;
    auto sum_save_start = high_resolution_clock::now();

    for (size_t ts = 0; ts < n_timestamps; ts++) {
        string sum_file = "data/expected_sum_" + to_string(ts) + ".txt";
        ofstream sum_out(sum_file);
        if (!sum_out) throw runtime_error("Failed to open " + sum_file);

        for (size_t dim = 0; dim < vector_dim; dim++) {
            sum_out << expected_sums[ts][dim];
            if (dim + 1 < vector_dim) sum_out << ' ';
        }
        sum_out << endl;
    }

    auto sum_save_end = high_resolution_clock::now();
    double sum_save_ms = duration_cast<nanoseconds>(sum_save_end - sum_save_start).count() / 1e6;
    cout << " Done (" << sum_save_ms << " ms)" << endl;

    cout << "\nCiphertexts and expected sums saved to ./data" << endl;

    return 0;
}
