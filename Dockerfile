FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
RUN conda install git

COPY . /workspace/ipsuite

WORKDIR /workspace/ipsuite
RUN pip install .[comparison,gap,nequip,apax,allegro,mace]
RUN pip install --upgrade torch --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install git+https://github.com/PythonFZ/torch-dftd.git@patch-2
RUN pip install dvc-s3

COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

WORKDIR /app
ENTRYPOINT ["/workspace/ipsuite/entrypoint.sh"]
