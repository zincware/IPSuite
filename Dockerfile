# Base image
ARG PYTHON_VERSION="3.11"
FROM python:${PYTHON_VERSION}

# Build arguments for dynamic version configuration
ARG MAKE_JOBS="8"
ARG CP2K_VERSION="v2024.3"
ARG PACKMOL_VERSION="20.15.3"
ARG GROMACS_VERSION="2024.4"

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/tools/cp2k/exe/local:/opt/tools/packmol-${PACKMOL_VERSION}:/opt/tools/gromacs-${GROMACS_VERSION}/build/bin:$PATH"


# Update and install essential packages
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    gfortran build-essential zip cmake-data cmake curl wget git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install CP2K
WORKDIR /opt/tools
RUN git clone -b ${CP2K_VERSION} --recursive https://github.com/cp2k/cp2k.git cp2k && \
    cd cp2k/tools/toolchain && ./install_cp2k_toolchain.sh --with-openmpi && \
    cp /opt/tools/cp2k/tools/toolchain/install/arch/* /opt/tools/cp2k/arch/ && \
    cd /opt/tools/cp2k && \
    bash -c "source /opt/tools/cp2k/tools/toolchain/install/setup && make -j ${MAKE_JOBS} ARCH=local VERSION='ssmp sdbg psmp pdbg'" && \
    echo "source /opt/tools/cp2k/tools/toolchain/install/setup" >> /etc/bash.bashrc

# Install PACKMOL
WORKDIR /opt/tools
RUN wget https://github.com/m3g/packmol/archive/refs/tags/v${PACKMOL_VERSION}.tar.gz && \
    tar -xzvf v${PACKMOL_VERSION}.tar.gz && \
    cd packmol-${PACKMOL_VERSION} && make && \
    rm /opt/tools/v${PACKMOL_VERSION}.tar.gz

# Install GROMACS
WORKDIR /opt/tools
RUN wget https://ftp.gromacs.org/gromacs/gromacs-${GROMACS_VERSION}.tar.gz && \
    tar xfz gromacs-${GROMACS_VERSION}.tar.gz && \
    cd gromacs-${GROMACS_VERSION} && \
    mkdir build && cd build && \
    cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON && \
    make -j ${MAKE_JOBS} && make check && make install && \
    rm /opt/tools/gromacs-${GROMACS_VERSION}.tar.gz

# Install additional Python packages
RUN pip install dvc-s3 jax[cuda12] MDAnalysis pyedr apax

# Install IPSuite
WORKDIR /opt/tools/ipsuite
COPY ./ ./
RUN pip install .

# Create a working directory
RUN mkdir -m 1777 /work
RUN git config --global --add safe.directory /work

WORKDIR /work

ENTRYPOINT [ "/bin/bash" ]
