FROM python:3.11
RUN apt update && apt install -y git

# RUN python3 -m pip config set global.break-system-packages true

COPY . /workspace/ipsuite
WORKDIR /workspace/ipsuite

RUN pip install .[comparison,gap,apax,mace]
RUN pip install --upgrade torch torchvision torchaudio
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install git+https://github.com/PythonFZ/torch-dftd.git@patch-2
RUN pip install dvc-s3
RUN pip install --no-cache-dir notebook jupyterlab

WORKDIR /app
