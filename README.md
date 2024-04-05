![PyTest](https://github.com/zincware/IPSuite/actions/workflows/tests.yaml/badge.svg)
[![ZnTrack](https://img.shields.io/badge/Powered%20by-ZnTrack-%23007CB0)](https://zntrack.readthedocs.io/en/latest/)
[![zincware](https://img.shields.io/badge/Powered%20by-zincware-darkcyan)](https://github.com/zincware)
[![DOI](https://img.shields.io/badge/DOI-10.1021/acs.jpcb.3c07187-red)](https://pubs.acs.org/doi/10.1021/acs.jpcb.3c07187)
# IPS - The Inter Atomic Potential Suite

![Logo](https://raw.githubusercontent.com/zincware/IPSuite/main/misc/IPS_logo.png)

IPS provides you with tools to generate Machine Learned Interatomic Potentials.

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

# References

If you use IPSuite in your research and find it helpful please consider citing us.

```bibtex
@article{zillsCollaborationMachineLearnedPotentials2024,
  title = {Collaboration on {{Machine-Learned Potentials}} with {{IPSuite}}: {{A Modular Framework}} for {{Learning-on-the-Fly}}},
  shorttitle = {Collaboration on {{Machine-Learned Potentials}} with {{IPSuite}}},
  author = {Zills, Fabian and Schäfer, Moritz René and Segreto, Nico and Kästner, Johannes and Holm, Christian and Tovey, Samuel},
  date = {2024-04-03},
  journaltitle = {The Journal of Physical Chemistry B},
  shortjournal = {J. Phys. Chem. B},
  publisher = {American Chemical Society},
  issn = {1520-6106},
  doi = {10.1021/acs.jpcb.3c07187},
}

@misc{zillsZnTrackDataCode2024,
  title = {{{ZnTrack}} -- {{Data}} as {{Code}}},
  author = {Zills, Fabian and Sch{\"a}fer, Moritz and Tovey, Samuel and K{\"a}stner, Johannes and Holm, Christian},
  year = {2024},
  eprint={2401.10603},
  archivePrefix={arXiv},
}
```
