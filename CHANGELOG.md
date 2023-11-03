# Changelog

## 0.1.2

### Node I/O Changes

- add `img_selection = zntrack.outs_path(zntrack.nwd / "selection.png")` to each
  `ConfigurationSelection` Node
- add `memory = zntrack.params(1000)` to `ConfigurationComparison`
- add `threshold: float = zntrack.params(None)` to `KernelSelection`
