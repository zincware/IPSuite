import importlib.resources as pkg_resources
import pathlib
import subprocess

import ase
import ase.io
import yaml
import zntrack
from jinja2 import Environment, FileSystemLoader

from ipsuite import base, static_data


class LammpsSimulator(base.ProcessSingleAtom):
    """Can perform LAMMPS Simulations.

    Parameters
    ----------
        lmp_exe: str
            This is the name or path of the LAMMPS executable. Either path
            to executable, "lmp" or "lamp_<machine>".
            See https://docs.lammps.org/Run_basics.html for more information
        lmp_params: str
            Path to file. To be able to change parameters with DVC and not
            have to change them manually in the input script, a params file in yaml
            format and corresponding template file must be provided.
        lmp_template: str
            Path to file. In combination with the params file this will
            be the input script for the LAMMPS simulation
        skiprun: bool, optional
            Whether to skip running LAMMPS or not, by default False

    Returns
    -------
    None
        This function does not return anything. Instead, it creates a LAMMPS input
        script based on the specified template and parameter files, runs the
        LAMMPS simulation using the specified executable, and saves the simulation
        output to the specified directory.

    """

    """A few remarks for future programmers:
    -If there is the error, that NPT.lammpstraj does not exist:
        that means there is an error in the inputscript.
        look at log.lammps in the nwd to see what is wrong
    - dont forget to "dvc init"...
    """
    lmp_directory: str = zntrack.outs_path(zntrack.nwd / "lammps")
    lmp_exe: str = zntrack.meta.Text("lmp_serial")
    skiprun: bool = False

    # outputs
    dump_file = zntrack.outs_path(zntrack.nwd / "NPT.lammpstraj")
    log_file = zntrack.outs_path(zntrack.nwd / "NPT.log")

    lmp_params: str = zntrack.params_path()
    lmp_template: str = zntrack.deps_path()

    def get_atoms(self):
        # look where to get the input_trajectory (either ase.Atoms or file)

        ase.io.write(self.lmp_directory / "atoms.xyz", self.get_data())
        self.atoms_file = (
            pathlib.Path(self.lmp_directory / "atoms.xyz").resolve().as_posix()
        )

    def fill_atoms_with_life(self):
        # Give LAMMPS more information about the Atoms provided.
        # (e.g. Mass or Type (LAMMPS specific)).
        # This Function has to be executed after get_atoms has been executed,
        # otherwise there might not be a xyz file to read.
        # Charges have to be set by Hand in the LAMMPS-inputscript-Template.
        data = ase.io.read(self.atoms_file)

        # Atomic Number
        self.atomic_numbers = data.get_atomic_numbers()
        # Atomic Mass
        self.atomic_masses = data.get_masses()
        # Atom Symbol
        self.atomic_symbols = data.get_chemical_symbols()

        i = 1
        atom_map = {}
        for k in range(len(self.atomic_numbers)):
            if self.atomic_numbers[k] not in atom_map:
                atom_map[self.atomic_numbers[k]] = (
                    i,
                    self.atomic_masses[k],
                    self.atomic_symbols[k],
                )
                i += 1
        self.atomic_type = [atom_map[num][0] for num in self.atomic_numbers]
        self.atomic_masses = [tup[1] for tup in list(atom_map.values())]
        self.atomic_symbols = [tup[2] for tup in list(atom_map.values())]

    def create_input_script(self):
        # Get parameter from yaml:
        with pathlib.Path(self.lmp_params).open("r") as stream:
            params = yaml.safe_load(stream)

        # Resolve paths for input files
        input_dict = {}
        input_dict["log_file"] = self.log_file.resolve().as_posix()
        input_dict["dump_file"] = self.dump_file.resolve().as_posix()
        input_dict["input_trajectory"] = self.atoms_file

        for key in params["sim_parameters"]:
            input_dict[key] = params["sim_parameters"][key]

        # Fill input dict with information about the atoms
        # (all infos are gathered from the xyz file except charges)
        input_dict["atomic_type"] = self.atomic_type
        input_dict["atomic_masses"] = self.atomic_masses
        input_dict["atomic_symbols"] = self.atomic_symbols

        # Get template
        loader = FileSystemLoader(".")
        env = Environment(loader=loader)  # , trim_blocks=True, lstrip_blocks=True)
        template = env.get_template(self.lmp_template)

        # Render Template
        self.lmp_input_script = template.render(input_dict)
        with pathlib.Path(f"{self.lmp_directory}/input.script").open("w") as file:
            file.write(self.lmp_input_script)  # write input script to output directory

    def run(self):
        self.lmp_directory.mkdir(exist_ok=True)  # create output directory
        self.get_atoms()
        self.fill_atoms_with_life()
        self.create_input_script()
        if self.skiprun:
            print("Skipping simulation ...")
            cmd = [self.lmp_exe, "-sr-in", "input.script"]
        else:
            print("Simulating ...")
            cmd = [self.lmp_exe, "-in", "input.script"]

        subprocess.run(
            cmd,
            cwd=self.lmp_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    @classmethod
    def get_template(
        self, filename: str = "lmp.jinja2", name="lammps_npt.jinja2"
    ) -> None:
        """Get the template file.

        Parameters
        ----------
        filename: str, optional
            The name where to write the template to.
        name: str, optional
            The name of the template, by default "lammps_npt.jinja2"

        Returns
        -------
        """
        content = pkg_resources.read_text(static_data, name)
        with pathlib.Path(filename).open("w") as file:
            file.write(content)

        if name == "lammps_npt.jinja2":
            params = {
                "temperature": 300,
                "pressure": 1.0,
                "barostat_factor": 500,
                "steps": 1000,
                "timestep": 0.001,
                "dump_interval": 100,
                "thermo_interval": 100,
            }
        else:
            raise NotImplementedError("Only lammps_npt.jinja2 is implemented")

        with pathlib.Path(filename).with_suffix(".yaml").open("w") as file:
            yaml.dump({"sim_parameters": params}, file)

    @property
    def atoms(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.dump_file, mode="r") as file:
            return list(ase.io.iread(file, format="lammps-dump-text"))
