#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "terse.h"

namespace py = pybind11;

class TERSEPythonWrapper {
private:
    TERSESystem* system;
    TERSEClient* client;
    size_t client_idx;

public:
    TERSEPythonWrapper(const std::string& params_file, size_t idx) 
        : client_idx(idx) {
        TERSEParams params = TERSEParams::load(params_file);
        system = new TERSESystem(params);
        client = new TERSEClient();

        // Load precomputed values for this client
        std::string precomp_file = "data/client_precompute_" + 
                                   std::to_string(idx) + ".bin";
        std::ifstream in(precomp_file, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to open " + precomp_file);
        }

        size_t n_entries;
        in.read(reinterpret_cast<char*>(&n_entries), sizeof(n_entries));

        std::vector<uint64_t> buffer(n_entries);
        in.read(reinterpret_cast<char*>(buffer.data()), 
                buffer.size() * sizeof(uint64_t));

        client->precomputed_p.resize(n_entries);
        for (size_t i = 0; i < n_entries; i++) {
            client->precomputed_p[i] = NativeInteger(buffer[i]);
        }
    }

    ~TERSEPythonWrapper() {
        delete client;
        delete system;
    }

    py::array_t<uint64_t> encrypt_vector(py::array_t<uint32_t> plaintext, 
                                          size_t timestamp) {
        auto buf = plaintext.request();

        // Validate dtype
        if (buf.format != py::format_descriptor<uint32_t>::format()) {
            throw std::runtime_error("Input must be uint32 array");
        }

        uint32_t* ptr = static_cast<uint32_t*>(buf.ptr);
        size_t vector_dim = buf.size;

        std::vector<uint64_t> ciphertexts(vector_dim);

        for (size_t i = 0; i < vector_dim; i++) {
            size_t stream_idx = timestamp * vector_dim + i;
            NativeInteger ct = system->encrypt(*client, ptr[i], stream_idx);
            ciphertexts[i] = ct.ConvertToInt();
        }

        return py::array_t<uint64_t>(ciphertexts.size(), ciphertexts.data());
    }
};

class TERSEServerWrapper {
private:
    TERSESystem* system;

public:
    TERSEServerWrapper(const std::string& params_file) {
        TERSEParams params = TERSEParams::load(params_file);
        system = new TERSESystem(params);
    }

    ~TERSEServerWrapper() {
        delete system;
    }

    py::array_t<uint64_t> aggregate_ciphertexts(
        std::vector<py::array_t<uint64_t>> client_ciphertexts,
        size_t timestamp) {

        if (client_ciphertexts.empty()) {
            throw std::runtime_error("No ciphertexts to aggregate");
        }

        auto buf0 = client_ciphertexts[0].request();
        size_t vector_dim = buf0.size;

        NativeInteger q_mod = system->get_context()->GetCryptoParameters()
                              ->GetElementParams()->GetParams()[0]->GetModulus();
        uint64_t q_val = q_mod.ConvertToInt();

        std::vector<uint64_t> aggregate(vector_dim, 0);

        for (const auto& ct_array : client_ciphertexts) {
            auto buf = ct_array.request();
            uint64_t* ptr = static_cast<uint64_t*>(buf.ptr);

            for (size_t i = 0; i < vector_dim; i++) {
                __uint128_t sum = static_cast<__uint128_t>(aggregate[i]) + 
                                  static_cast<__uint128_t>(ptr[i]);
                aggregate[i] = static_cast<uint64_t>(sum % q_val);
            }
        }

        return py::array_t<uint64_t>(aggregate.size(), aggregate.data());
    }

    void save_aggregate(py::array_t<uint64_t> aggregate, size_t timestamp) {
        auto buf = aggregate.request();
        uint64_t* ptr = static_cast<uint64_t*>(buf.ptr);
        size_t vector_dim = buf.size;

        std::vector<NativeInteger> agg_vec(vector_dim);
        for (size_t i = 0; i < vector_dim; i++) {
            agg_vec[i] = NativeInteger(ptr[i]);
        }

        std::string filename = "data/encrypted_aggregate_" + 
                               std::to_string(timestamp) + ".bin";
        system->save_aggregate_vector(agg_vec, filename);
    }
};

class TERSETrustedWrapper {
private:
    TERSESystem* system;
    TERSEServer* server;

public:
    TERSETrustedWrapper(const std::string& params_file, 
                        const std::string& server_key_file) {
        TERSEParams params = TERSEParams::load(params_file);
        system = new TERSESystem(params);
        server = new TERSEServer(system->load_server_key(server_key_file));
    }

    ~TERSETrustedWrapper() {
        delete server;
        delete system;
    }

    py::array_t<uint32_t> decrypt_aggregate(size_t timestamp, 
                                             size_t vector_dim) {
        std::string agg_file = "data/encrypted_aggregate_" + 
                               std::to_string(timestamp) + ".bin";
        std::vector<NativeInteger> agg_ct = 
            system->load_aggregate_vector(agg_file);

        NativeInteger q_mod = system->get_context()->GetCryptoParameters()
                              ->GetElementParams()->GetParams()[0]->GetModulus();
        uint64_t q_val = q_mod.ConvertToInt();
        uint64_t t = system->get_params().plain_modulus;

        std::vector<uint32_t> decrypted(vector_dim);

        for (size_t i = 0; i < vector_dim; i++) {
            size_t stream_idx = timestamp * vector_dim + i;

            NativeInteger sum = agg_ct[i].ModAdd(
                server->precomputed_p_prime[stream_idx], q_mod);

            uint64_t raw = sum.ConvertToInt();

            int64_t signed_val;
            if (raw > q_val / 2) {
                signed_val = static_cast<int64_t>(raw) - 
                             static_cast<int64_t>(q_val);
            } else {
                signed_val = static_cast<int64_t>(raw);
            }

            int64_t result = signed_val % static_cast<int64_t>(t);
            if (result < 0) {
                result += static_cast<int64_t>(t);
            }

            decrypted[i] = static_cast<uint32_t>(result);
        }

        return py::array_t<uint32_t>(decrypted.size(), decrypted.data());
    }
};

PYBIND11_MODULE(terse_py, m) {
    m.doc() = "TERSE secure aggregation Python bindings";

    py::class_<TERSEPythonWrapper>(m, "TERSEClient")
        .def(py::init<const std::string&, size_t>(),
             py::arg("params_file"), py::arg("client_idx"))
        .def("encrypt_vector", &TERSEPythonWrapper::encrypt_vector,
             py::arg("plaintext"), py::arg("timestamp"));

    py::class_<TERSEServerWrapper>(m, "TERSEServer")
        .def(py::init<const std::string&>(), py::arg("params_file"))
        .def("aggregate_ciphertexts", 
             &TERSEServerWrapper::aggregate_ciphertexts,
             py::arg("client_ciphertexts"), py::arg("timestamp"))
        .def("save_aggregate", &TERSEServerWrapper::save_aggregate,
             py::arg("aggregate"), py::arg("timestamp"));

    py::class_<TERSETrustedWrapper>(m, "TERSETrusted")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("params_file"), py::arg("server_key_file"))
        .def("decrypt_aggregate", &TERSETrustedWrapper::decrypt_aggregate,
             py::arg("timestamp"), py::arg("vector_dim"));
}
