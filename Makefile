# =========================
# Compiler / linker config
# =========================
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -march=native -pthread -fopenmp \
           -Wno-unused-parameter -Wno-missing-field-initializers -fPIC

# After refactor, headers live under ./include (e.g., include/terse/terse.h, include/common/dp_mechanisms.h)
INCLUDES = -Iinclude \
           -I/usr/local/include/openfhe -I/usr/local/include/openfhe/third-party/include \
           -I/usr/local/include/openfhe/core -I/usr/local/include/openfhe/pke \
           -I/usr/local/include/openfhe/binfhe

LDFLAGS = -L/usr/local/lib
LIBS = -lOPENFHEpke -lOPENFHEcore -lssl -lcrypto -lgomp

# =========================
# Directories
# =========================
TERSE_SRC_DIR = src/terse
AES_SRC_DIR   = src/aes
PYTHON_DIR    = python
SGX_DIR       = sgx
DATA_DIR      = data
OBJDIR        = build

# =========================
# Python binding configuration
# =========================
PYTHON_INCLUDE    = $(shell python3 -m pybind11 --includes)
PYTHON_EXT_SUFFIX = $(shell python3-config --extension-suffix)
PYTHON_TARGET     = $(PYTHON_DIR)/terse_py$(PYTHON_EXT_SUFFIX)

# =========================
# Targets
# =========================
TARGETS = client server trusted trusted_round setup_clients setup_trusted aes_client aes_trusted


.PHONY: all core full python sgx

# Build just the native executables (fast dev loop)
core: $(TARGETS)

# Ship-ready default
all: core python sgx


# =========================
# Object files (TERSE)
# =========================
TERSE_CORE_OBJ         = $(OBJDIR)/terse_terse.o
TERSE_CLIENT_OBJ       = $(OBJDIR)/terse_client.o
TERSE_SERVER_OBJ       = $(OBJDIR)/terse_server.o
TERSE_TRUSTED_OBJ      = $(OBJDIR)/terse_trusted.o
TERSE_TRUSTED_ROUND_OBJ= $(OBJDIR)/terse_trusted_round.o
TERSE_SETUP_CLIENTS_OBJ= $(OBJDIR)/terse_setup_clients.o
TERSE_SETUP_TRUSTED_OBJ= $(OBJDIR)/terse_setup_trusted.o

# =========================
# Object files (AES)
# =========================
AES_CLIENT_OBJ  = $(OBJDIR)/aes_client.o
AES_TRUSTED_OBJ = $(OBJDIR)/aes_trusted.o

# =========================
# Build directory
# =========================
$(OBJDIR):
	mkdir -p $(OBJDIR)

# =========================
# Compile rules
# =========================
# TERSE objects get a "terse_" prefix to avoid name collisions with AES objects.
$(OBJDIR)/terse_%.o: $(TERSE_SRC_DIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# AES objects keep their filename-based object name (aes_client.o, aes_trusted.o, ...)
$(OBJDIR)/%.o: $(AES_SRC_DIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# =========================
# Link rules (TERSE binaries)
# =========================
client: $(TERSE_CORE_OBJ) $(TERSE_CLIENT_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

server: $(TERSE_CORE_OBJ) $(TERSE_SERVER_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

trusted: $(TERSE_CORE_OBJ) $(TERSE_TRUSTED_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

trusted_round: $(TERSE_CORE_OBJ) $(TERSE_TRUSTED_ROUND_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

setup_clients: $(TERSE_CORE_OBJ) $(TERSE_SETUP_CLIENTS_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

setup_trusted: $(TERSE_CORE_OBJ) $(TERSE_SETUP_TRUSTED_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

# =========================
# Link rules (AES binaries)
# =========================
aes_client: $(AES_CLIENT_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

aes_trusted: $(AES_TRUSTED_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

# =========================
# Python bindings
# =========================
python: $(PYTHON_TARGET)

# Note: terse_python.cpp should now include "terse/terse.h"
$(PYTHON_TARGET): $(PYTHON_DIR)/terse_python.cpp $(TERSE_CORE_OBJ) | $(OBJDIR)
	$(CXX) -shared -fPIC $(CXXFLAGS) $(PYTHON_INCLUDE) $(INCLUDES) \
		$< $(TERSE_CORE_OBJ) -o $@ $(LDFLAGS) $(LIBS)

# =========================
# SGX targets (unchanged, assumes binaries are at repo root)
# =========================
sgx: $(SGX_DIR)/trusted.manifest.sgx \
     $(SGX_DIR)/trusted_round.manifest.sgx \
     $(SGX_DIR)/aes_trusted.manifest.sgx \
     $(SGX_DIR)/setup_trusted.manifest.sgx

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

$(SGX_DIR)/trusted_round.manifest: $(SGX_DIR)/trusted_round.manifest.template trusted_round
	gramine-manifest -Dlog_level=error -Darch_libdir=/lib/x86_64-linux-gnu \
		-Dexecdir=$(shell pwd) $< $@

$(SGX_DIR)/trusted_round.manifest.sgx: $(SGX_DIR)/trusted_round.manifest trusted_round
	gramine-sgx-sign --manifest $< --output $@

# =========================
# Clean
# =========================
clean:
	rm -rf $(OBJDIR)
	rm -f $(TARGETS) $(PYTHON_TARGET)
	rm -f $(SGX_DIR)/*.manifest $(SGX_DIR)/*.manifest.sgx $(SGX_DIR)/*.sig $(SGX_DIR)/*.token

cleanall: clean
	rm -rf $(DATA_DIR)/*

# =========================
# Help
# =========================
help:
	@echo "Available targets:"
	@echo "  all          - Build all C++ binaries (default)"
	@echo "  python       - Build Python bindings (terse_py module)"
	@echo "  sgx          - Build SGX manifests"
	@echo "  clean        - Remove build artifacts"
	@echo "  cleanall     - Remove build artifacts and data directory contents"
	@echo ""
	@echo "Individual targets:"
	@echo "  client, server, trusted, trusted_round, setup_clients, setup_trusted"
	@echo "  aes_client, aes_trusted"
