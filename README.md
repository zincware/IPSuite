![PyTest](https://github.com/zincware/IPSuite/actions/workflows/tests.yaml/badge.svg)
[![ZnTrack](https://img.shields.io/badge/Powered%20by-ZnTrack-%23007CB0)](https://zntrack.readthedocs.io/en/latest/)
[![zincware](https://img.shields.io/badge/Powered%20by-zincware-darkcyan)](https://github.com/zincware)
[![DOI](https://zenodo.org/badge/608256065.svg)](https://zenodo.org/doi/10.5281/zenodo.10034314)
[![Documentation Status](https://readthedocs.org/projects/ipsuite/badge/?version=latest)](https://ipsuite.readthedocs.io/en/latest/?badge=latest)

# IPS - The Inter Atomic Potential Suite

![Logo](https://raw.githubusercontent.com/zincware/IPSuite/main/misc/IPS_logo.png)

IPS provides you with tools to generate Machine Learned Interatomic Potentials.
You can find the documentation at https://ipsuite.readthedocs.io

Install the package to get started or check out an interactive notebook
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zincware/IPSuite/HEAD)

```python
pip install ipsuite
```

Examples can be found at:

- https://dagshub.com/PythonFZ/IPS-Examples/src/intro/main.ipynb
- https://dagshub.com/PythonFZ/IPS-Examples/src/graph/main.ipynb
- https://dagshub.com/PythonFZ/IPS-Examples/src/modify_graph/main.ipynb

# Docker Image

You can use IPSuite directly from within docker by calling it e.g. like:

```sh
docker run -it -v "$(pwd):/app" --gpus all pythonf/ipsuite dvc repro
docker run -it -v "$(pwd):/app" --gpus all pythonf/ipsuite python
docker run -it -v "$(pwd):/app" --gpus all pythonf/ipsuite zntrack list
```
