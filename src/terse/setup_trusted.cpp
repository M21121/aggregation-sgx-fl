#include "terse/terse.h"
#include <chrono>
#include <filesystem>
#include <iostream>

using namespace std;
using namespace std::chrono;

static size_t load_size_from_file(const string& filename) {
    ifstream in(filename);
    if (!in) {
        throw runtime_error("Failed to open " + filename);
    }
    size_t value = 0;
    in >> value;
    return value;
}

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        cerr << "Usage: " << argv[0] << " <n_clients> <n_timestamps> [vector_dim]" << endl;
        return 1;
    }

    size_t requested_clients = stoull(argv[1]);
    size_t requested_timestamps = stoull(argv[2]);
    size_t requested_vector_dim = (argc == 4) ? stoull(argv[3]) : 1;

    TERSEParams params = TERSEParams::load("data/params.bin");
    TERSESystem system(params);

    size_t configured_clients = load_size_from_file("data/n_clients.txt");
    size_t configured_timestamps = load_size_from_file("data/n_timestamps.txt");
    size_t configured_vector_dim = load_size_from_file("data/vector_dim.txt");

    if (requested_clients > configured_clients ||
        requested_timestamps > configured_timestamps ||
        requested_vector_dim > configured_vector_dim) {
        cerr << "Requested dimensions exceed client-setup artifacts" << endl;
        return 1;
    }

    cout << "=== TERSE Trusted Setup ===" << endl;
    cout << "Loading client keys..." << endl;
    vector<TERSEClient> clients = system.load_client_keys("data/client_keys.bin");
    if (clients.size() != configured_clients) {
        throw runtime_error("Client key count mismatch");
    }

    cout << "Generating server key..." << endl;
    auto keygen_start = high_resolution_clock::now();
    TERSEServer server = system.generate_server_key(clients);
    auto keygen_end = high_resolution_clock::now();
    double server_key_ms =
        duration_cast<nanoseconds>(keygen_end - keygen_start).count() / 1e6;
    cout << "Server key generation time: " << server_key_ms << " ms" << endl;

    size_t total_streams = requested_timestamps * requested_vector_dim;
    size_t n_theta =
        (total_streams + params.poly_modulus_degree - 1) / params.poly_modulus_degree;

    cout << "\n=== Server Precomputation Phase ===" << endl;
    cout << "Total streams: " << total_streams << endl;
    cout << "Number of theta values: " << n_theta << endl;

    double a_theta_loading_ms = 0;
    double server_precompute_ms = 0;

    auto precomp_total_start = high_resolution_clock::now();

    for (uint64_t theta = 0; theta < n_theta; theta++) {
        // Load A_theta from disk instead of generating
        auto theta_start = high_resolution_clock::now();
        string a_theta_file = "data/A_theta_" + to_string(theta) + ".bin";
        DCRTPoly A_theta = system.load_A_theta(a_theta_file);
        auto theta_end = high_resolution_clock::now();
        double theta_ms = duration_cast<nanoseconds>(theta_end - theta_start).count() / 1e6;
        a_theta_loading_ms += theta_ms;

        if (theta < 5 || theta == n_theta - 1) {
            cout << "Theta " << theta << " (A_theta loaded in " << theta_ms << " ms)" << endl;
        }

        auto server_start = high_resolution_clock::now();
        system.precompute_server_batch(server, A_theta);
        auto server_end = high_resolution_clock::now();
        server_precompute_ms +=
            duration_cast<nanoseconds>(server_end - server_start).count() / 1e6;
    }

    server.precomputed_p_prime.resize(total_streams);

    auto precomp_total_end = high_resolution_clock::now();
    double precompute_total_ms =
        duration_cast<nanoseconds>(precomp_total_end - precomp_total_start).count() / 1e6;

    cout << "\nA_theta loading time (total): " << a_theta_loading_ms << " ms" << endl;
    cout << "Server precompute time (total): " << server_precompute_ms << " ms" << endl;
    cout << "Server precompute wall clock time: " << precompute_total_ms << " ms" << endl;

    system.save_server_key(server, "data/server_key.bin");
    cout << "\nServer key saved to data/server_key.bin" << endl;

    // Clean up A_theta files
    cout << "\nCleaning up A_theta files..." << endl;
    size_t cleaned = 0;
    for (uint64_t theta = 0; theta < n_theta; theta++) {
        string a_theta_file = "data/A_theta_" + to_string(theta) + ".bin";
        try {
            if (filesystem::remove(a_theta_file)) {
                cleaned++;
            }
        } catch (const filesystem::filesystem_error& e) {
            cerr << "Warning: Unable to delete " << a_theta_file << " (" << e.what() << ")" << endl;
        }
    }
    cout << "Cleaned up " << cleaned << " A_theta files." << endl;

    // Delete client keys after trusted setup
    try {
        filesystem::remove("data/client_keys.bin");
        cout << "Deleted data/client_keys.bin after trusted setup." << endl;
    } catch (const filesystem::filesystem_error& e) {
        cerr << "Warning: Unable to delete client_keys.bin (" << e.what() << ")" << endl;
    }

    cout << "\nTrusted setup complete." << endl;
    return 0;
}
