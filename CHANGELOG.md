# Changelog

## 0.1.2

### Node I/O Changes

- add `img_selection = zntrack.outs_path(zntrack.nwd / "selection.png")` to each
  `ConfigurationSelection` Node
- add `memory = zntrack.params(1000)` to `ConfigurationComparison`
- add `threshold: float = zntrack.params(None)` to `KernelSelection`
- add `reduction_axis = zntrack.params` and
  `dim_reduction: str = zntrack.params` to `ThresholdSelection`
- add `nl_skin: float = params(0.5)` and
  `transformations: dict[str, dict] = params(None)` to the `ApaxEnsemble`
- add `BatchKernelSelection`
- add `seed: int = params()` to `ASEMD` and `wrap: bool` to wrap the coordinates
  during the simulation
- add `pbc: bool` to `Packmol` to fix PBC not supported by PACKMOL
