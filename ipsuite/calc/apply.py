import os
import pathlib

import ase
import h5py
import numpy as np
import znh5md
import zntrack
from laufband import Laufband

# from ase.calculators.calculator import all_properties
from ipsuite.abc import NodeWithCalculator
from ipsuite.utils.ase_sim import freeze_copy_atoms


class ApplyCalculator(zntrack.Node):
    """
    Apply a calculator to a list of atoms objects and store the results in a H5MD file.

    Parameters
    ----------
    data : list[ase.Atoms]
        List of atoms objects to calculate.
    model : NodeWithCalculator
        Node providing the calculator object to apply to the data.
    frames_path : pathlib.Path, optional
        Path to the H5MD file where the results will be stored.
    dump_rate : int, optional
        If specified, the results will be dumped to the H5MD file
        every `dump_rate` frames. If None, all frames will be
        dumped at once at the end of the calculation.
    model_outs : pathlib.Path, optional
        Path to the directory where the model outputs will be stored.
        Defaults to a subdirectory named "model" in the current working directory.
    additive : bool, optional
        If True, adds the new calculator results to existing calculations.
        If False (default), replaces any existing calculator results.
        This is useful for adding corrections (e.g., D3 dispersion) to existing models.

    Laufband Configuration
    ----------------------
    This node can use Laufband for auto-checkpointing and parallel execution.
    To enable Laufband features, you can use the following environment variable:

    .. code-block:: bash

        # Enable LAUFBAND
        export LAUFBAND_DISABLE="0"

        # Maximum number of retries for unsuccessful jobs
        export LAUFBAND_MAX_DIED_RETRIES="3".

        # optional, but recommended for identifying dead jobs
        export LAUFBAND_HEARTBEAT_TIMEOUT=$((runtime_seconds))

        # optional, can be used to identify the job
        export LAUFBAND_IDENTIFIER=${SLURM_JOB_ID}
    """

    data: list[ase.Atoms] = zntrack.deps()
    model: NodeWithCalculator = zntrack.deps()
    dump_rate: int | None = zntrack.params(None)
    additive: bool = zntrack.params(False)

    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "frames.h5")
    model_outs: pathlib.Path = zntrack.outs_path(zntrack.nwd / "model")
    implemented_properties: list[str] = zntrack.params(
        default_factory=lambda: ["energy", "forces"]
    )

    def run(self):
        self.model_outs.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "dummy.txt").write_text("Thank you for using IPSuite!")
        frames = []
        io = znh5md.IO(self.frames_path)

        worker = Laufband(
            self.data,
            com=self.model_outs / "laufband.sqlite",
            lock_path=self.model_outs / "laufband.lock",
            disable=os.environ.get("LAUFBAND_DISABLE", "1") == "1",
        )
        # by default, we disable laufband for better performance

        calc_dir = self.model_outs / f"{worker.identifier}"
        calc_dir.mkdir(parents=True, exist_ok=True)

        calc = self.model.get_calculator(directory=calc_dir)

        for atoms in worker:
            # Store original results if in additive mode
            original_results = {}
            if self.additive and atoms.calc is not None:
                original_results = getattr(atoms.calc, "results", {}).copy()
            
            # Apply new calculator
            atoms.calc = calc
            
            # Calculate new results
            if "energy" in self.implemented_properties:
                atoms.get_potential_energy()
            if "forces" in self.implemented_properties:
                atoms.get_forces()
            if "stress" in self.implemented_properties:
                atoms.get_stress()
            
            # Add original results to new results if in additive mode
            if self.additive and original_results:
                new_results = atoms.calc.results.copy()
                for key, original_value in original_results.items():
                    if key in new_results:
                        try:
                            # Add the values if they are numeric
                            atoms.calc.results[key] = original_value + new_results[key]
                        except (TypeError, ValueError):
                            # If addition fails, keep the new value
                            pass
                    else:
                        # If key not in new results, keep original value
                        atoms.calc.results[key] = original_value
                        
            frames.append(freeze_copy_atoms(atoms))
            if self.dump_rate is not None:
                if len(frames) % self.dump_rate == 0:
                    with worker.lock:
                        io.extend(frames)
                        frames = []

        with worker.lock:
            io.extend(frames)

    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "rb") as f:
            with h5py.File(f) as file:
                return list(znh5md.IO(file_handle=file))
