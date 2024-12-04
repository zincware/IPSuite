ARG MAKE_JOBS="4"

FROM python:3.11

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/tools/cp2k/exe/local:/opt/tools/packmol-20.15.3:/opt/tools/gromacs-2024.4/build/bin:$PATH"

# Update and install essential packages
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    gfortran build-essential zip cmake-data cmake curl wget git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install CP2K
WORKDIR /opt/tools
RUN git clone -b v2024.1 --recursive https://github.com/cp2k/cp2k.git cp2k && \
    cd cp2k/tools/toolchain && ./install_cp2k_toolchain.sh --with-openmpi && \
    cp /opt/tools/cp2k/tools/toolchain/install/arch/* /opt/tools/cp2k/arch/ && \
    cd /opt/tools/cp2k && \
    bash -c "source /opt/tools/cp2k/tools/toolchain/install/setup && make -j ${MAKE_JOBS} ARCH=local VERSION='ssmp sdbg psmp pdbg'" && \
    echo "source /opt/tools/cp2k/tools/toolchain/install/setup" >> /etc/bash.bashrc

# Install PACKMOL
WORKDIR /opt/tools
RUN wget https://github.com/m3g/packmol/archive/refs/tags/v20.15.3.tar.gz && \
    tar -xzvf v20.15.3.tar.gz && \
    cd packmol-20.15.3 && make && \
    rm /opt/tools/v20.15.3.tar.gz

# Install GROMACS
WORKDIR /opt/tools
RUN wget https://ftp.gromacs.org/gromacs/gromacs-2024.4.tar.gz && \
    tar xfz gromacs-2024.4.tar.gz && \
    cd gromacs-2024.4 && \
    mkdir build && cd build && \
    cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON && \
    make -j ${MAKE_JOBS} && make check && make install && \
    rm /opt/tools/gromacs-2024.4.tar.gz


# Install additional Python packages
RUN pip install dvc-s3 jax[cuda12] MDAnalysis pyedr apax


# Install IPSuite
WORKDIR /opt/tools/ipsuite
COPY ./ ./
RUN pip install .

# Create a working directory
RUN mkdir -m 1777 /work
WORKDIR /work

ENTRYPOINT [ "/bin/bash" ]
