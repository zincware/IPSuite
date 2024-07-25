import functools
import logging
import pathlib
import typing

import ase.io
import h5py
import yaml
import znh5md
import zntrack.utils

from ipsuite.base import ProcessSingleAtom
from ipsuite.models import Apax
from ipsuite.utils.helpers import check_duplicate_keys

log = logging.getLogger(__name__)


class ApaxJaxMD(ProcessSingleAtom):
    """Class to run a more performant JaxMD simulation with a apax Model.

    Attributes
    ----------
    model: ApaxModel
        model to use for the simulation
    repeat: float
        number of repeats
    md_parameter: dict
        parameter for the MD simulation
    md_parameter_file: str
        path to the MD simulation parameter file
    """

    model: Apax = zntrack.deps()
    repeat = zntrack.params(None)

    md_parameter: dict = zntrack.params(None)
    md_parameter_file: str = zntrack.params_path(None)

    sim_dir: pathlib.Path = zntrack.outs_path(zntrack.nwd / "md")
    init_struc_dir: pathlib.Path = zntrack.outs_path(
        zntrack.nwd / "initial_structure.extxyz"
    )

    def post_init(self):
        if not self.state.loaded:
            dict_supplied = self.md_parameter is None
            file_supplied = self.md_parameter_file is None

            if dict_supplied and file_supplied:
                raise TypeError("Please specify either an input dict or a file")
            elif not dict_supplied and not file_supplied:
                raise TypeError(
                    "Can not train apax model without a parameter dict or file"
                )
            else:
                log.info(
                    "Please keep track of the parameter file with git, just like the"
                    " params.yaml"
                )
        # TODO introduce apax base class
        # if not isinstance(self.model, Apax):
        #     raise TypeError(
        #         "Performing simulations with JaxMD requires a apax model Node"
        #     )

    def _handle_parameter_file(self):
        if self.md_parameter_file:
            md_parameter_file_content = pathlib.Path(self.md_parameter_file).read_text()
            self.md_parameter = yaml.safe_load(md_parameter_file_content)

        custom_parameters = {
            "sim_dir": self.sim_dir.as_posix(),
            "initial_structure": self.init_struc_dir.as_posix(),
        }
        check_duplicate_keys(custom_parameters, self.md_parameter, log)
        self.md_parameter.update(custom_parameters)

    def run(self):
        """Primary method to run which executes all steps of the model training"""
        from apax.md.nvt import run_md

        self._handle_parameter_file()
        atoms = self.get_data()
        if self.repeat is not None:
            atoms = atoms.repeat(self.repeat)
        ase.io.write(self.init_struc_dir.as_posix(), atoms)

        self.model._handle_parameter_file()
        run_md(self.model._parameter, self.md_parameter)

    @functools.cached_property
    def atoms(self) -> typing.List[ase.Atoms]:
        with self.state.fs.open(self.sim_dir / "md.h5", "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]
