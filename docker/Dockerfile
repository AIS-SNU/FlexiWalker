# Start from CUDA base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Update and install essential tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        wget \
        curl \
        ca-certificates \
        software-properties-common \
        libgflags-dev \
        zlib1g-dev \
        libzstd-dev \
        libxml2-dev \
        libedit-dev \
        libffi-dev \
        clang \
        python3 \
        python3-pip

# LLVM version (match your use)
ARG LLVM_VERSION=17

# Add LLVM repo and install clang+llvm toolchain
RUN wget https://apt.llvm.org/llvm-snapshot.gpg.key && \
    apt-key add llvm-snapshot.gpg.key && \
    add-apt-repository "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-${LLVM_VERSION} main" && \
    apt-get update && \
    apt-get install -y \
        llvm-${LLVM_VERSION} \
        llvm-${LLVM_VERSION}-dev \
        llvm-${LLVM_VERSION}-tools \
        clang-${LLVM_VERSION} \
        clang-tools-${LLVM_VERSION} \
        clang-tidy-${LLVM_VERSION} \
        clang-format-${LLVM_VERSION} \
        lldb-${LLVM_VERSION} \
        lld-${LLVM_VERSION} \
        libclang-${LLVM_VERSION}-dev \
        libclang-cpp${LLVM_VERSION}-dev \
        libjsoncpp-dev

# JSON
RUN apt-get install -y nlohmann-json3-dev

# Symlinks for convenience
RUN ln -s /usr/bin/clang-${LLVM_VERSION} /usr/local/bin/clang && \
    ln -s /usr/bin/clang++-${LLVM_VERSION} /usr/local/bin/clang++ && \
    ln -s /usr/bin/opt-${LLVM_VERSION} /usr/local/bin/opt && \
    ln -s /usr/bin/llc-${LLVM_VERSION} /usr/local/bin/llc && \
    ln -s /usr/bin/llvm-dis-${LLVM_VERSION} /usr/local/bin/llvm-dis && \
    ln -s /usr/bin/llvm-as-${LLVM_VERSION} /usr/local/bin/llvm-as && \
    ln -s /usr/bin/llvm-config-${LLVM_VERSION} /usr/local/bin/llvm-config

# Python requirements for codegen and tests
RUN pip3 install --no-cache-dir jinja2 numpy pytest pytest-xdist

# Set working directory
WORKDIR /flexiwalker

#git setting
RUN git config --global --add safe.directory /flexiwalker