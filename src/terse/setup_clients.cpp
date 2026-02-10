#include "terse/terse.h"
#include <omp.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>

using namespace std;
using namespace std::chrono;

static void ensure_data_dir() {
    filesystem::create_directories("data");
}

static void save_size_to_file(const string& filename, size_t value) {
    ofstream out(filename);
    if (!out) {
        throw runtime_error("Failed to open " + filename);
    }
    out << value << endl;
}

static void save_client_precomputes(const vector<TERSEClient>& clients,
                                    size_t expected_streams) {
    for (size_t idx = 0; idx < clients.size(); idx++) {
        if (clients[idx].precomputed_p.size() != expected_streams) {
            throw runtime_error("Client " + to_string(idx) + " precomputation size mismatch");
        }

        string filename = "data/client_precompute_" + to_string(idx) + ".bin";
        ofstream out(filename, ios::binary);
        if (!out) {
            throw runtime_error("Failed to open " + filename);
        }

        size_t n_entries = clients[idx].precomputed_p.size();
        out.write(reinterpret_cast<const char*>(&n_entries), sizeof(n_entries));

        vector<uint64_t> buffer(n_entries);
        for (size_t i = 0; i < n_entries; i++) {
            buffer[i] = clients[idx].precomputed_p[i].ConvertToInt();
        }
        out.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(uint64_t));
    }
}

int main(int argc, char* argv[]) {
    omp_set_num_threads(1);

    if (argc < 3 || argc > 4) {
        cerr << "Usage: " << argv[0] << " <n_clients> <n_timestamps> [vector_dim]" << endl;
        return 1;
    }

    size_t n_clients = stoull(argv[1]);
    size_t n_timestamps = stoull(argv[2]);
    size_t vector_dim = (argc == 4) ? stoull(argv[3]) : 1;

    if (n_clients == 0 || n_timestamps == 0 || vector_dim == 0) {
        cerr << "All arguments must be positive" << endl;
        return 1;
    }

    ensure_data_dir();

    TERSEParams params(4096, 65537, 0, HEStd_128_classic, 3.2);
    TERSESystem system(params);

    vector<TERSEClient> clients(n_clients);
    vector<double> per_client_keygen_times(n_clients);

    DiscreteUniformGeneratorImpl<NativeVector> dug;
    dug.SetModulus(system.get_context()->GetCryptoParameters()->GetElementParams()->GetModulus());

    cout << "=== TERSE Client Setup ===" << endl;
    cout << "Generating keys for " << n_clients << " clients..." << endl;

    auto keygen_start = high_resolution_clock::now();
    for (size_t i = 0; i < n_clients; i++) {
        auto client_start = high_resolution_clock::now();
        clients[i].secret_key = DCRTPoly(
            dug,
            system.get_context()->GetCryptoParameters()->GetElementParams(),
            Format::EVALUATION);
        auto client_end = high_resolution_clock::now();
        per_client_keygen_times[i] =
            duration_cast<nanoseconds>(client_end - client_start).count() / 1e6;
    }
    auto keygen_end = high_resolution_clock::now();

    double avg_client_keygen_ms =
        accumulate(per_client_keygen_times.begin(), per_client_keygen_times.end(), 0.0) / n_clients;
    double total_keygen_ms =
        duration_cast<nanoseconds>(keygen_end - keygen_start).count() / 1e6;

    NativeInteger q_mod = system.get_context()->GetCryptoParameters()
                              ->GetElementParams()->GetParams()[0]
                              ->GetModulus();
    params.cipher_modulus = q_mod.ConvertToInt();
    params.save("data/params.bin");

    cout << "Client key generation time (per user, averaged): " << avg_client_keygen_ms << " ms" << endl;
    cout << "Key generation time (total): " << total_keygen_ms << " ms" << endl;

    size_t total_streams = n_timestamps * vector_dim;
    size_t n_theta = (total_streams + params.poly_modulus_degree - 1) / params.poly_modulus_degree;

    double a_theta_generation_ms = 0;
    double single_client_precompute_ms = 0;

    cout << "\n=== Client Precomputation Phase ===" << endl;
    cout << "Total streams: " << total_streams << endl;
    cout << "Poly modulus degree: " << params.poly_modulus_degree << endl;
    cout << "Number of theta values: " << n_theta << endl;

    auto precomp_total_start = high_resolution_clock::now();

    for (uint64_t theta = 0; theta < n_theta; theta++) {
        auto theta_start = high_resolution_clock::now();
        DCRTPoly A_theta = system.generate_A_theta(theta);
        auto theta_end = high_resolution_clock::now();
        double theta_ms = duration_cast<nanoseconds>(theta_end - theta_start).count() / 1e6;
        a_theta_generation_ms += theta_ms;

        // Save A_theta to disk for trusted setup
        string a_theta_file = "data/A_theta_" + to_string(theta) + ".bin";
        system.save_A_theta(A_theta, a_theta_file);

        if (theta < 5 || theta == n_theta - 1) {
            cout << "Theta " << theta << " (A_theta generated in " << theta_ms << " ms)" << endl;
        }

        auto one_client_start = high_resolution_clock::now();
        system.precompute_client_batch(clients[0], A_theta);
        auto one_client_end = high_resolution_clock::now();
        single_client_precompute_ms +=
            duration_cast<nanoseconds>(one_client_end - one_client_start).count() / 1e6;

        for (size_t i = 1; i < n_clients; i++) {
            system.precompute_client_batch(clients[i], A_theta);
        }
    }

    for (auto& client : clients) {
        client.precomputed_p.resize(total_streams);
    }

    auto precomp_total_end = high_resolution_clock::now();
    double precompute_total_ms =
        duration_cast<nanoseconds>(precomp_total_end - precomp_total_start).count() / 1e6;

    cout << "\nA_theta generated " << n_theta << " times" << endl;
    cout << "A_theta generation time (total): " << a_theta_generation_ms << " ms" << endl;
    cout << "Client precomputation time (per client for one theta batch): "
         << single_client_precompute_ms << " ms" << endl;
    cout << "Client precomputation wall clock time: " << precompute_total_ms << " ms" << endl;

    system.save_client_keys(clients, "data/client_keys.bin");
    save_client_precomputes(clients, total_streams);

    save_size_to_file("data/n_clients.txt", n_clients);
    save_size_to_file("data/n_timestamps.txt", n_timestamps);
    save_size_to_file("data/vector_dim.txt", vector_dim);

    cout << "\nClient-side artifacts saved to ./data" << endl;
    cout << "Run setup_trusted next to build server materials." << endl;
    return 0;
}
