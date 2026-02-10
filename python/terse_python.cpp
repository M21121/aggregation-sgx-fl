#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "terse/terse.h"

namespace py = pybind11;

// Force-cast helper: accepts any array-like, returns contiguous 1D uint64 view/copy
static py::array_t<uint64_t, py::array::c_style | py::array::forcecast>
as_uint64_1d(const py::handle &h, const char *name) {
    py::array arr = py::array::ensure(h);
    if (!arr) {
        throw std::runtime_error(std::string(name) + " must be a NumPy array");
    }

    auto casted =
        py::array_t<uint64_t, py::array::c_style | py::array::forcecast>::ensure(arr);
    if (!casted) {
        throw std::runtime_error(std::string(name) + " could not be cast to uint64");
    }

    auto buf = casted.request();
    if (buf.ndim != 1) {
        throw std::runtime_error(std::string(name) + " must be a 1D array");
    }

    return casted;
}

// Force-cast helper: accepts any array-like, returns contiguous 1D uint32 view/copy
static py::array_t<uint32_t, py::array::c_style | py::array::forcecast>
as_uint32_1d(const py::handle &h, const char *name) {
    py::array arr = py::array::ensure(h);
    if (!arr) {
        throw std::runtime_error(std::string(name) + " must be a NumPy array");
    }

    auto casted =
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast>::ensure(arr);
    if (!casted) {
        throw std::runtime_error(std::string(name) + " could not be cast to uint32");
    }

    auto buf = casted.request();
    if (buf.ndim != 1) {
        throw std::runtime_error(std::string(name) + " must be a 1D array");
    }

    return casted;
}

class TERSEPythonWrapper {
private:
    TERSESystem* system;
    TERSEClient* client;
    size_t client_idx;

public:
    TERSEPythonWrapper(const std::string& params_file, size_t idx)
        : system(nullptr), client(nullptr), client_idx(idx) {
        TERSEParams params = TERSEParams::load(params_file);
        system = new TERSESystem(params);
        client = new TERSEClient();

        std::string precomp_file = "data/client_precompute_" + std::to_string(idx) + ".bin";
        std::ifstream in(precomp_file, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to open " + precomp_file);
        }

        size_t n_entries = 0;
        in.read(reinterpret_cast<char*>(&n_entries), sizeof(n_entries));
        if (!in) {
            throw std::runtime_error("Failed to read n_entries from " + precomp_file);
        }

        std::vector<uint64_t> buffer(n_entries);
        in.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(uint64_t));
        if (!in) {
            throw std::runtime_error("Failed to read precompute buffer from " + precomp_file);
        }

        client->precomputed_p.resize(n_entries);
        for (size_t i = 0; i < n_entries; i++) {
            client->precomputed_p[i] = NativeInteger(buffer[i]);
        }
    }

    ~TERSEPythonWrapper() {
        delete client;
        delete system;
    }

    py::array_t<uint64_t> encrypt_vector(py::array plaintext, size_t timestamp) {
        auto pt = as_uint32_1d(plaintext, "plaintext");
        auto buf = pt.request();

        uint32_t* ptr = static_cast<uint32_t*>(buf.ptr);
        size_t vector_dim = static_cast<size_t>(buf.size);

        // Return NumPy-owned buffer (safe lifetime)
        py::array_t<uint64_t> out(vector_dim);
        auto out_view = out.mutable_unchecked<1>();

        for (size_t i = 0; i < vector_dim; i++) {
            size_t stream_idx = timestamp * vector_dim + i;
            NativeInteger ct = system->encrypt(*client, ptr[i], stream_idx);
            out_view(i) = ct.ConvertToInt();
        }

        return out;
    }
};

class TERSEServerWrapper {
private:
    TERSESystem* system;

public:
    TERSEServerWrapper(const std::string& params_file)
        : system(nullptr) {
        TERSEParams params = TERSEParams::load(params_file);
        system = new TERSESystem(params);
    }

    ~TERSEServerWrapper() {
        delete system;
    }

    py::array_t<uint64_t> aggregate_ciphertexts(py::list client_ciphertexts, size_t /*timestamp*/) {
        if (client_ciphertexts.empty()) {
            throw std::runtime_error("No ciphertexts to aggregate");
        }

        // Force-cast first to get vector_dim
        auto first = as_uint64_1d(client_ciphertexts[0], "client_ciphertexts[0]");
        auto buf0 = first.request();
        size_t vector_dim = static_cast<size_t>(buf0.size);

        NativeInteger q_mod = system->get_context()->GetCryptoParameters()
                                  ->GetElementParams()->GetParams()[0]->GetModulus();
        uint64_t q_val = q_mod.ConvertToInt();

        py::array_t<uint64_t> out(vector_dim);
        auto out_view = out.mutable_unchecked<1>();

        for (size_t i = 0; i < vector_dim; i++) {
            out_view(i) = 0;
        }

        for (py::handle h : client_ciphertexts) {
            auto ct = as_uint64_1d(h, "client_ciphertexts[i]");
            auto buf = ct.request();
            if (static_cast<size_t>(buf.size) != vector_dim) {
                throw std::runtime_error("All ciphertext arrays must have identical length");
            }

            uint64_t* ptr = static_cast<uint64_t*>(buf.ptr);
            for (size_t i = 0; i < vector_dim; i++) {
                __uint128_t sum =
                    static_cast<__uint128_t>(out_view(i)) + static_cast<__uint128_t>(ptr[i]);
                out_view(i) = static_cast<uint64_t>(sum % q_val);
            }
        }

        return out;
    }

    void save_aggregate(py::array aggregate, size_t timestamp) {
        auto agg = as_uint64_1d(aggregate, "aggregate");
        auto buf = agg.request();

        uint64_t* ptr = static_cast<uint64_t*>(buf.ptr);
        size_t vector_dim = static_cast<size_t>(buf.size);

        std::vector<NativeInteger> agg_vec(vector_dim);
        for (size_t i = 0; i < vector_dim; i++) {
            agg_vec[i] = NativeInteger(ptr[i]);
        }

        std::string filename = "data/encrypted_aggregate_" + std::to_string(timestamp) + ".bin";
        system->save_aggregate_vector(agg_vec, filename);
    }
};

class TERSETrustedWrapper {
private:
    TERSESystem* system;
    TERSEServer* server;

public:
    TERSETrustedWrapper(const std::string& params_file,
                        const std::string& server_key_file)
        : system(nullptr), server(nullptr) {
        TERSEParams params = TERSEParams::load(params_file);
        system = new TERSESystem(params);
        server = new TERSEServer(system->load_server_key(server_key_file));
    }

    ~TERSETrustedWrapper() {
        delete server;
        delete system;
    }

    py::array_t<uint32_t> decrypt_aggregate(size_t timestamp, size_t vector_dim) {
        std::string agg_file = "data/encrypted_aggregate_" + std::to_string(timestamp) + ".bin";
        std::vector<NativeInteger> agg_ct = system->load_aggregate_vector(agg_file);

        if (agg_ct.size() != vector_dim) {
            throw std::runtime_error(
                "Aggregate length mismatch for timestamp " + std::to_string(timestamp) +
                ": expected " + std::to_string(vector_dim) +
                ", got " + std::to_string(agg_ct.size())
            );
        }

        NativeInteger q_mod = system->get_context()->GetCryptoParameters()
                                  ->GetElementParams()->GetParams()[0]->GetModulus();
        uint64_t q_val = q_mod.ConvertToInt();
        uint64_t t = system->get_params().plain_modulus;

        py::array_t<uint32_t> out(vector_dim);
        auto out_view = out.mutable_unchecked<1>();

        for (size_t i = 0; i < vector_dim; i++) {
            size_t stream_idx = timestamp * vector_dim + i;

            if (stream_idx >= server->precomputed_p_prime.size()) {
                throw std::runtime_error(
                    "stream_idx out of bounds: " + std::to_string(stream_idx) +
                    " >= " + std::to_string(server->precomputed_p_prime.size())
                );
            }

            NativeInteger sum = agg_ct[i].ModAdd(server->precomputed_p_prime[stream_idx], q_mod);
            uint64_t raw = sum.ConvertToInt();

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

            out_view(i) = static_cast<uint32_t>(result);
        }

        return out;
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
        .def("aggregate_ciphertexts", &TERSEServerWrapper::aggregate_ciphertexts,
             py::arg("client_ciphertexts"), py::arg("timestamp"))
        .def("save_aggregate", &TERSEServerWrapper::save_aggregate,
             py::arg("aggregate"), py::arg("timestamp"));

    py::class_<TERSETrustedWrapper>(m, "TERSETrusted")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("params_file"), py::arg("server_key_file"))
        .def("decrypt_aggregate", &TERSETrustedWrapper::decrypt_aggregate,
             py::arg("timestamp"), py::arg("vector_dim"));
}
