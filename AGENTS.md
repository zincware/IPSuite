# IPSuite Development Guidelines

> **Audience**: LLM-driven engineering agents and human developers

IPSuite is a collection of Nodes (https://zntrack.readthedocs.io) for atomistic simulations.
Nodes are connected via attributes allowing the design of comprehensive workflows from MLIP training to production simulation.


## Required Development Workflow

**CRITICAL**: Always run these commands in sequence before committing:

```bash
uv sync                              # Install dependencies
uvx pre-commit run --all-files       # Linting and Code formatting
uv run pytest                        # Run full test suite
```

**All three must pass** - this is enforced by CI.

## Repository Structure
| Path              | Purpose                                                |
|-------------------|--------------------------------------------------------|
| `ipsuite/`        | Source code for the IPSuite library                    |
| `tests/`          | Unit and integration tests for the IPSuite library     |
| `docs/`           | Documentation for the IPSuite library                  |

## Imports
All imports are made available through the `lazy-loader` from `__init__.pyi`.

### Code Standards

- Python ≥ 3.10 with full type annotations
- Follow existing patterns and maintain consistency
- **Prioritize readable, understandable code** - clarity over cleverness
- Avoid obfuscated or confusing patterns even if they're shorter
- Each feature needs corresponding tests


### Docstrings
- Docstrings are written in the numpy style
- Each Node must have a docstring describing its functionality.
- Docstrings must be concise and clear.
- Docstrings should not include unnecessary information or verbosity. E.g. A Node that runs a molecular dynamics simulation should not explain what molecular dynamics is. Expect the user to have expert knowledge in the field.
- Each Node should have an example that can be tested. `project` and `ips` are available. This can look like
```
Examples
--------
>>> with project:
...     methanol_conformers = ips.Smiles2Conformers(smiles="CO", numConfs=5)
>>> project.repro()
>>> frames = methanol_conformers.frames
>>> print(f"Generated {len(frames)} conformers.")
Generated 5 conformers.
```
