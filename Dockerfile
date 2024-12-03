ARG MAKE_JOBS="4"

FROM python:3.11

RUN apt update -y
RUN apt install -y gfortran build-essential zip cmake-data

# # Install CP2k
WORKDIR /opt/tools
RUN git clone -b v2024.1 --recursive https://github.com/cp2k/cp2k.git cp2k
WORKDIR /opt/tools/cp2k/tools/toolchain
RUN ./install_cp2k_toolchain.sh --with-openmpi
RUN cp /opt/tools/cp2k/tools/toolchain/install/arch/* /opt/tools/cp2k/arch/
WORKDIR /opt/tools/cp2k
RUN bash -c "source /opt/tools/cp2k/tools/toolchain/install/setup && make -j ${MAKE_JOBS} ARCH=local VERSION='ssmp sdbg psmp pdbg'"

# Install PACKMOL
WORKDIR /opt/tools
RUN wget https://github.com/m3g/packmol/archive/refs/tags/v20.15.3.tar.gz
RUN tar -xzvf v20.15.3.tar.gz
WORKDIR /opt/tools/packmol-20.15.3
RUN make

# Install GROMACS
RUN apt install -y cmake
WORKDIR /opt/tools
RUN wget https://ftp.gromacs.org/gromacs/gromacs-2024.4.tar.gz
RUN tar xfz gromacs-2024.4.tar.gz
WORKDIR /opt/tools/gromacs-2024.4/build
RUN cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON
RUN make -j ${MAKE_JOBS}
RUN make check
RUN make install

# Cleanup files
RUN rm /opt/tools/gromacs-2024.4.tar.gz
RUN rm /opt/tools/v20.15.3.tar.gz

# Change the default shell to login shell
SHELL ["/bin/bash", "--login", "-c"]

# Install Node.js and npm from NodeSource
RUN curl -fsSL https://bun.sh/install | bash

# Install CML globally
RUN bun install -g @dvcorg/cml

# Install IPSuite
WORKDIR /opt/tools/ipsuite
COPY ./ ./
RUN pip install .

# Install apax
WORKDIR /opt/tools
RUN git clone https://github.com/apax-hub/apax
WORKDIR /opt/tools/apax
RUN pip install .

# Install Additional packages
RUN pip install dvc-s3 jax[cuda12] MDAnalysis pyedr

RUN echo "source /opt/tools/cp2k/tools/toolchain/install/setup" >> /etc/bash.bashrc
RUN echo "export PATH=/opt/tools/cp2k/exe/local:\$PATH" >> /etc/bash.bashrc
RUN echo "export PATH=/opt/tools/packmol-20.15.3:\$PATH" >> /etc/bash.bashrc
RUN echo "export PATH=/opt/tools/gromacs-2024.4/build/bin:\$PATH" >> /etc/bash.bashrc


# Create a work directory
RUN mkdir -m 1777 /work
WORKDIR /work

CMD ["/bin/bash", "--login", "-c"]
