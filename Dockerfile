FROM python:3.11
RUN apt update -y
RUN apt install -y gfortran build-essential zip cmake-data

# Install Node.js and npm from NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
RUN apt install -y nodejs

# RUN apt install -y build-essential git curl wget zip unzip gfortran gcc libxrender1 pkg-config cmake-data zlib1g-dev

# # Install CP2k
WORKDIR /opt/tools
RUN git clone -b v2024.1 --recursive https://github.com/cp2k/cp2k.git cp2k
WORKDIR /opt/tools/cp2k/tools/toolchain
RUN ./install_cp2k_toolchain.sh --with-openmpi
RUN cp /opt/tools/cp2k/tools/toolchain/install/arch/* /opt/tools/cp2k/arch/
WORKDIR /opt/tools/cp2k
RUN bash -c "source /opt/tools/cp2k/tools/toolchain/install/setup && make -j 192 ARCH=local VERSION='ssmp sdbg psmp pdbg'"

# Install PACKMOL
WORKDIR /opt/tools
RUN wget https://github.com/m3g/packmol/archive/refs/tags/v20.15.0.tar.gz
RUN tar -xzvf v20.15.0.tar.gz
WORKDIR /opt/tools/packmol-20.15.0
RUN make

# Install GROMACS
RUN apt install -y cmake
WORKDIR /opt/tools
RUN wget https://ftp.gromacs.org/gromacs/gromacs-2024.2.tar.gz
RUN tar xfz gromacs-2024.2.tar.gz
WORKDIR /opt/tools/gromacs-2024.2/build
RUN cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON
RUN make -j 192
RUN make check
RUN make install

# Cleanup files
RUN rm /opt/tools/gromacs-2024.2.tar.gz
RUN rm /opt/tools/v20.15.0.tar.gz

# Install CML globally
RUN npm install -g @dvcorg/cml

# Install IPSuite
WORKDIR /opt/tools
RUN git clone https://github.com/zincware/ipsuite
WORKDIR /opt/tools/ipsuite
RUN pip install .

# Install apax
WORKDIR /opt/tools
RUN git clone https://github.com/apax-hub/apax
WORKDIR /opt/tools/apax
RUN pip install .

# Install Additional packages
RUN pip install dvc-s3 jax[cuda12] MDAnalysis pyedr

# Add environment setup to /etc/profile.d
RUN echo "source /opt/tools/cp2k/tools/toolchain/install/setup" >> /opt/setup.sh && \
    echo "export PATH=/opt/tools/cp2k/exe/local:\$PATH" >> /opt/setup.sh && \
    echo "export PATH=/opt/tools/packmol-20.15.0:\$PATH" >> /opt/setup.sh && \
    echo "export PATH=/opt/tools/gromacs-2024.2/build/bin:\$PATH" >> /opt/setup.sh

# Ensure /etc/profile.d/setup_env.sh is readable by all users
RUN chmod 777 /opt/setup.sh

# Set the default shell to bash
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Create a work directory
RUN mkdir -m 1777 /work
WORKDIR /work

CMD [ "/bin/bash" ]
