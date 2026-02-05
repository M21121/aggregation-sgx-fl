CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -march=native -pthread -fopenmp \
           -Wno-unused-parameter -Wno-missing-field-initializers -fPIC
INCLUDES = -I/usr/local/include/openfhe -I/usr/local/include/openfhe/third-party/include \
           -I/usr/local/include/openfhe/core -I/usr/local/include/openfhe/pke \
           -I/usr/local/include/openfhe/binfhe -Isrc
LDFLAGS = -L/usr/local/lib
LIBS = -lOPENFHEpke -lOPENFHEcore -lssl -lcrypto -lgomp

# Directories
SRC_DIR = src
PYTHON_DIR = python
SGX_DIR = sgx
DATA_DIR = data

# Python binding configuration
PYTHON_INCLUDE = $(shell python3 -m pybind11 --includes)
PYTHON_EXT_SUFFIX = $(shell python3-config --extension-suffix)
PYTHON_TARGET = terse_py$(PYTHON_EXT_SUFFIX)


# Object files
TERSE_OBJECTS = $(SRC_DIR)/terse.o
TARGETS = client server trusted setup_clients setup_trusted aes_client aes_trusted
PYTHON_TARGET = $(PYTHON_DIR)/terse_py$(PYTHON_EXT_SUFFIX)

.PHONY: all sgx python clean cleanall help

all: $(TARGETS)

# TERSE core object
$(SRC_DIR)/terse.o: $(SRC_DIR)/terse.cpp $(SRC_DIR)/terse.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# TERSE binaries
client: $(TERSE_OBJECTS) $(SRC_DIR)/client.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(SRC_DIR)/client.o: $(SRC_DIR)/client.cpp $(SRC_DIR)/terse.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

server: $(TERSE_OBJECTS) $(SRC_DIR)/server.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(SRC_DIR)/server.o: $(SRC_DIR)/server.cpp $(SRC_DIR)/terse.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

trusted: $(TERSE_OBJECTS) $(SRC_DIR)/trusted.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(SRC_DIR)/trusted.o: $(SRC_DIR)/trusted.cpp $(SRC_DIR)/terse.h $(SRC_DIR)/dp_mechanisms.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

setup_clients: $(TERSE_OBJECTS) $(SRC_DIR)/setup_clients.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(SRC_DIR)/setup_clients.o: $(SRC_DIR)/setup_clients.cpp $(SRC_DIR)/terse.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

setup_trusted: $(TERSE_OBJECTS) $(SRC_DIR)/setup_trusted.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(SRC_DIR)/setup_trusted.o: $(SRC_DIR)/setup_trusted.cpp $(SRC_DIR)/terse.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# AES binaries
aes_client: $(SRC_DIR)/aes_client.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

$(SRC_DIR)/aes_client.o: $(SRC_DIR)/aes_client.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

aes_trusted: $(SRC_DIR)/aes_trusted.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

$(SRC_DIR)/aes_trusted.o: $(SRC_DIR)/aes_trusted.cpp $(SRC_DIR)/dp_mechanisms.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Python bindings
python: $(PYTHON_TARGET)

$(PYTHON_TARGET): $(PYTHON_DIR)/terse_python.cpp $(TERSE_OBJECTS)
	$(CXX) -shared -fPIC $(CXXFLAGS) $(PYTHON_INCLUDE) $(INCLUDES) \
		$< $(TERSE_OBJECTS) -o $@ $(LDFLAGS) $(LIBS)

# SGX targets
sgx: $(SGX_DIR)/trusted.manifest.sgx $(SGX_DIR)/aes_trusted.manifest.sgx $(SGX_DIR)/setup_trusted.manifest.sgx

$(SGX_DIR)/trusted.manifest: $(SGX_DIR)/trusted.manifest.template trusted
	gramine-manifest -Dlog_level=error -Darch_libdir=/lib/x86_64-linux-gnu \
		-Dexecdir=$(shell pwd) $< $@

$(SGX_DIR)/trusted.manifest.sgx: $(SGX_DIR)/trusted.manifest trusted
	gramine-sgx-sign --manifest $< --output $@

$(SGX_DIR)/aes_trusted.manifest: $(SGX_DIR)/aes_trusted.manifest.template aes_trusted
	gramine-manifest -Dlog_level=error -Darch_libdir=/lib/x86_64-linux-gnu \
		-Dexecdir=$(shell pwd) $< $@

$(SGX_DIR)/aes_trusted.manifest.sgx: $(SGX_DIR)/aes_trusted.manifest aes_trusted
	gramine-sgx-sign --manifest $< --output $@

$(SGX_DIR)/setup_trusted.manifest: $(SGX_DIR)/setup_trusted.manifest.template setup_trusted
	gramine-manifest -Dlog_level=error -Darch_libdir=/lib/x86_64-linux-gnu \
		-Dexecdir=$(shell pwd) $< $@

$(SGX_DIR)/setup_trusted.manifest.sgx: $(SGX_DIR)/setup_trusted.manifest setup_trusted
	gramine-sgx-sign --manifest $< --output $@

# Clean targets
clean:
	rm -f $(SRC_DIR)/*.o $(TARGETS) $(PYTHON_TARGET)
	rm -f $(SGX_DIR)/*.manifest $(SGX_DIR)/*.manifest.sgx $(SGX_DIR)/*.sig $(SGX_DIR)/*.token

cleanall: clean
	rm -rf $(DATA_DIR)/*

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build all C++ binaries (default)"
	@echo "  python       - Build Python bindings (terse_py module)"
	@echo "  sgx          - Build SGX manifests"
	@echo "  clean        - Remove build artifacts"
	@echo "  cleanall     - Remove build artifacts and data directory"
	@echo ""
	@echo "Individual targets:"
	@echo "  client, server, trusted, setup_clients, setup_trusted"
	@echo "  aes_client, aes_trusted"
