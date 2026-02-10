#include "terse/terse.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

struct AggregatedCiphertext {
    vector<NativeInteger> ciphertext;
};

static AggregatedCiphertext load_aggregated_ciphertext_untrusted(size_t ts_idx,
                                                                 size_t expected_vector_dim) {
    string agg_file = "data/encrypted_aggregate_" + to_string(ts_idx) + ".bin";
    ifstream agg_in(agg_file, ios::binary);
    if (!agg_in) {
        throw runtime_error("Failed to open " + agg_file);
    }

    size_t vector_dim_in_file = 0;
    agg_in.read(reinterpret_cast<char*>(&vector_dim_in_file), sizeof(vector_dim_in_file));
    if (!agg_in) {
        throw runtime_error("Failed reading vector_dim from " + agg_file);
    }

    if (vector_dim_in_file != expected_vector_dim) {
        throw runtime_error(
            "Vector dim mismatch in " + agg_file +
            " (file: " + to_string(vector_dim_in_file) +
            ", expected: " + to_string(expected_vector_dim) + ")"
        );
    }

    AggregatedCiphertext agg_ct;
    agg_ct.ciphertext.resize(expected_vector_dim);

    vector<uint64_t> buffer(expected_vector_dim);
    agg_in.read(reinterpret_cast<char*>(buffer.data()),
                expected_vector_dim * sizeof(uint64_t));
    if (!agg_in) {
        throw runtime_error("Failed reading ciphertext payload from " + agg_file);
    }

    for (size_t i = 0; i < expected_vector_dim; i++) {
        agg_ct.ciphertext[i] = NativeInteger(buffer[i]);
    }

    return agg_ct;
}

static void save_decrypted_sum(size_t ts_idx, const vector<uint32_t>& decrypted_sum) {
    string out_file = "data/decrypted_" + to_string(ts_idx) + ".bin";
    ofstream out(out_file, ios::binary);
    if (!out) {
        throw runtime_error("Failed to open " + out_file + " for writing");
    }

    out.write(reinterpret_cast<const char*>(decrypted_sum.data()),
              decrypted_sum.size() * sizeof(uint32_t));
    if (!out) {
        throw runtime_error("Failed writing " + out_file);
    }
}

static void decrypt_range(
    size_t start_ts,
    size_t n_chunks,
    size_t vector_dim,
    TERSESystem& system,
    TERSEServer& server,
    const TERSEParams& params
) {
    NativeInteger q_mod = system.get_context()->GetCryptoParameters()
                          ->GetElementParams()->GetParams()[0]->GetModulus();
    uint64_t q_val = q_mod.ConvertToInt();
    uint64_t t = params.plain_modulus;

    for (size_t k = 0; k < n_chunks; k++) {
        size_t ts_idx = start_ts + k;

        AggregatedCiphertext agg_ct = load_aggregated_ciphertext_untrusted(ts_idx, vector_dim);

        vector<uint32_t> decrypted_sum(vector_dim);
        for (size_t coord = 0; coord < vector_dim; coord++) {
            size_t stream_idx = ts_idx * vector_dim + coord;

            NativeInteger sum = agg_ct.ciphertext[coord].ModAdd(
                server.precomputed_p_prime[stream_idx], q_mod);

            uint64_t raw = sum.ConvertToInt();

            int64_t signed_val = (raw > q_val / 2)
                ? (static_cast<int64_t>(raw) - static_cast<int64_t>(q_val))
                : static_cast<int64_t>(raw);

            int64_t result = signed_val % static_cast<int64_t>(t);
            if (result < 0) result += static_cast<int64_t>(t);

            decrypted_sum[coord] = static_cast<uint32_t>(result);
        }

        save_decrypted_sum(ts_idx, decrypted_sum);
    }
}

int main(int argc, char* argv[]) {
    // Make stdin/stdout responsive in persistent mode.
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Load once per enclave lifetime.
    TERSEParams params = TERSEParams::load("data/params.bin");
    TERSESystem system(params);
    TERSEServer server = system.load_server_key("data/server_key.bin");

    // Backward-compatible one-shot mode:
    // trusted_round <start_ts> <n_chunks> <vector_dim>
    if (argc == 4) {
        size_t start_ts   = stoull(argv[1]);
        size_t n_chunks   = stoull(argv[2]);
        size_t vector_dim = stoull(argv[3]);

        decrypt_range(start_ts, n_chunks, vector_dim, system, server, params);
        return 0;
    }

    // Persistent service mode (no argv):
    // Commands on stdin:
    // - DECRYPT <start_ts> <n_chunks> <vector_dim>
    // - QUIT
    //
    // Responses on stdout:
    // - OK
    // - ERR <message>
    string cmd;
    while (cin >> cmd) {
        if (cmd == "QUIT") {
            cout << "OK\n" << std::flush;
            return 0;
        }

        if (cmd == "DECRYPT") {
            size_t start_ts = 0;
            size_t n_chunks = 0;
            size_t vector_dim = 0;

            if (!(cin >> start_ts >> n_chunks >> vector_dim)) {
                cout << "ERR bad_args\n" << std::flush;

                // Recover stream state.
                cin.clear();
                string rest;
                getline(cin, rest);
                continue;
            }

            try {
                decrypt_range(start_ts, n_chunks, vector_dim, system, server, params);
                cout << "OK\n" << std::flush;
            } catch (const exception& e) {
                cout << "ERR " << e.what() << "\n" << std::flush;
            }
            continue;
        }

        cout << "ERR unknown_command\n" << std::flush;
    }

    return 0;
}
