"""Configuration comparison base."""

import contextlib
import dataclasses
import pathlib
import typing
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import znjson
import zntrack
from dscribe.descriptors import SOAP
from tqdm import trange

from ipsuite import base, utils


def convert_to_df(similarities: typing.List) -> pd.DataFrame:
    """Convert similarities to pd.DataFrame to save as zn.plots.

    Parameters
    ----------
    similarities: typing.List
        contains similarities
    Returns
    -------
    df: pd.DataFrame
        contains a pd.Dataframe with the similarity.
    """
    df = pd.DataFrame({"similarities": similarities})
    df.index.name = "configuration_index"
    return df


@dataclass
class SOAPParameter:
    """Dataclass to store SOAP parameter used for representation.

    Attributes
    ----------
    r_cut: float
        cutoff radius of the soap descriptor in Angstrom
    n_max: int
        number of radial basis functions
    l_max: int
        maximum degree of spherical harmonics
    n_jobs: int
        number of parallel jobs to instantiate
    sigma: float
        The standard deviation of the gaussians used to expand the atomic density.
    rbf: str
        The radial basis functions to use
    weighting: dict
        Contains the options which control the weighting of the atomic density.
    """

    r_cut: float = 9.0
    n_max: int = 7
    l_max: int = 7
    n_jobs: int = -1
    sigma: float = 1.0
    rbf: str = "gto"
    weighting: dict = None


class SOAPParameterConverter(znjson.ConverterBase):
    """Converter class to encode and decode dictionaries and dataclasses."""

    level = 100
    representation = "soap_parameter_dataclass"
    instance = SOAPParameter

    def encode(self, obj: SOAPParameter) -> dict:
        """Encode dataclass to dictionary."""
        return dataclasses.asdict(obj)

    def decode(self, value: dict) -> SOAPParameter:
        """DEcode dictionary to dataclass."""
        return SOAPParameter(**value)


znjson.config.register(SOAPParameterConverter)


def create_dataset(file: h5py.File, data, soap: SOAP, name: str):
    """Create an entry in the HDF5 dataset."""
    file.create_dataset(
        name,
        (
            len(data),
            len(data[0]),
            soap.get_number_of_features(),
        ),
    )


def write_dataset(
    file: h5py.File,
    data,
    name: str,
    soap: SOAP,
    n_jobs,
    desc: str,
    disable_tqdm: bool = False,
):
    """Write data to HDF5 dataset."""
    with trange((len(data) - 1), desc=desc, leave=True, disable=disable_tqdm) as pbar:
        for max_index, atoms in enumerate(data):
            file[name][max_index] = soap.create(atoms, n_jobs=n_jobs)
            pbar.update(1)


class ConfigurationComparison(base.IPSNode):
    """Base of comparison methods to compare atomic configurations.

    Attributes
    ----------
    reference: typing.Union[utils.helpers.UNION_ATOMS_OR_ATOMS_LST,
     utils.types.SupportsAtoms]
        reference configurations to compare analyte to
    analyte: typing.Union[
        utils.helpers.UNION_ATOMS_OR_ATOMS_LST, utils.types.SupportsAtoms
    ]
        analyte comparison to compare with reference
    similarities: zn.plots()
        in the end a csv file to save computed maximal similarities
    soap: typing.Union[dict, SOAPParameter]
        parameter to use for the SOAP descriptor
    result: typing.List[typing.List[float]]
        result of the comparison, all similarity computations
    node_name: str
        name of the node used within the dvc graph
    compile_with_jit: bool
        choose if kernel should be compiled with jit or not.
    memory: int
            How far back to look in the MMK vector.
    """

    reference: base.protocol.HasOrIsAtoms = zntrack.deps()
    analyte: base.protocol.HasOrIsAtoms = zntrack.deps()
    memory: int = zntrack.params(100)
    similarities = zntrack.zn.plots()
    soap: typing.Union[dict, SOAPParameter] = zntrack.zn.params(SOAPParameter())
    result: typing.List[float] = zntrack.zn.outs()

    _name_ = "ConfigurationComparison"
    use_jit: bool = zntrack.meta.Text(True)

    def __init__(
        self,
        reference=None,
        analyte=None,
        soap: dict = None,
        use_jit: bool = True,
        **kwargs
    ):
        """Initialize the ConfigurationComparison node.

        Parameters
        ----------
        reference: typing.Union[utils.helpers.UNION_ATOMS_OR_ATOMS_LST,
                    utils.types.SupportsAtoms]
            reference configurations to compare analyte to
        analyte: typing.Union[utils.helpers.UNION_ATOMS_OR_ATOMS_LST,
                utils.types.SupportsAtoms]
            analyte comparison to compare with reference (If reference is None, analyte
             will be compared to itself)
        similarities: zn.plots()
            In the end a csv file to save computed maximal similarities
        soap: dict
            Parameter to use for the SOAP descriptor.
        use_jit: bool
            use jit compilation.
        kwargs: dict
            additional keyword arguments
        """
        super().__init__(**kwargs)
        if soap is None:
            soap = {}
        if reference is None:
            self.reference = None
        else:
            self.reference = utils.helpers.get_deps_if_node(reference, "atoms")
        self.analyte = utils.helpers.get_deps_if_node(analyte, "atoms")
        if not self.state.loaded:
            self.soap = SOAPParameter(**soap)

        self.soap_file = self.nwd / "soap_representation.hdf5"

        # remove "soap_reference" from HDF5, do not write "soap_analyte"
        self.load_analyte = False
        self.remove_database = True
        self.disable_tqdm = False
        self.use_jit = use_jit

    def save_representation(self):
        """Save the SOAP descriptor representation as hdf5 file to save RAM.

        It will create SOAP descriptor for each configurations
         and save them ordered in a hdf5 file.
        """
        species = [int(x) for x in set(self.analyte[0].get_atomic_numbers())]
        _soap = SOAP(
            species=species,
            periodic=False,  # any(self.analyte[0].pbc),
            r_cut=self.soap.r_cut,
            n_max=self.soap.n_max,
            l_max=self.soap.l_max,
            sigma=self.soap.sigma,
            rbf=self.soap.rbf,
            weighting=self.soap.weighting,
        )
        if self.reference is None:
            with h5py.File(self.soap_file, "w") as representation_file:
                create_dataset(
                    file=representation_file, data=self.analyte, soap=_soap, name="soap"
                )
                write_dataset(
                    file=representation_file,
                    data=self.analyte,
                    name="soap",
                    soap=_soap,
                    n_jobs=self.soap.n_jobs,
                    desc="Writing SOAP",
                    disable_tqdm=self.disable_tqdm,
                )
        else:
            with h5py.File(self.soap_file, "a") as representation_file:
                create_dataset(
                    file=representation_file,
                    data=self.reference,
                    soap=_soap,
                    name="soap_reference",
                )
                write_dataset(
                    file=representation_file,
                    data=self.reference,
                    name="soap_reference",
                    soap=_soap,
                    n_jobs=self.soap.n_jobs,
                    desc="Writing SOAP reference",
                    disable_tqdm=self.disable_tqdm,
                )

                if not self.load_analyte:
                    create_dataset(
                        file=representation_file,
                        data=self.analyte,
                        soap=_soap,
                        name="soap_analyte",
                    )

                    write_dataset(
                        file=representation_file,
                        data=self.analyte,
                        name="soap_analyte",
                        soap=_soap,
                        n_jobs=self.soap.n_jobs,
                        desc="Writing SOAP analyte",
                        disable_tqdm=self.disable_tqdm,
                    )

    def _save_plots(self, max_index, interval: int = 1000):
        """Save the ZnTrack plots at regular intervals."""
        if max_index % interval == 0:
            self.similarities = convert_to_df(self.result)
            type(self).similarities.save(self)

    def unlink_database(self):
        """Remove the database."""
        if pathlib.Path(self.soap_file).is_file():
            pathlib.Path(self.soap_file).unlink()

    def run(self):
        """Run the configuration comparison.

        Use the chosen comparison method to compute the similarity between
        configurations and save the result as a csv file.
        """
        self.result = []
        self.save_representation()
        if self.reference is None:
            with h5py.File(self.soap_file, "r") as representation_file:
                with trange(
                    (len(self.analyte) - 1),
                    desc="Comparing",
                    leave=True,
                    disable=self.disable_tqdm,
                ) as pbar:
                    for max_index, _atoms in enumerate(self.analyte):
                        if max_index == 0:
                            continue
                        if max_index <= self.memory:
                            reference_soap = representation_file["soap"][:max_index]
                        else:
                            reference_soap = representation_file["soap"][
                                max_index - self.memory : max_index
                            ]
                        analyte_soap = representation_file["soap"][max_index]
                        comparison = self.compare(reference_soap, analyte_soap)
                        self.result.append(float(comparison.numpy()))
                        self._save_plots(max_index)
                        pbar.update(1)
        else:
            with h5py.File(self.soap_file, "r") as representation_file:
                with trange(
                    (len(self.analyte)),
                    desc="Comparing",
                    leave=True,
                    disable=self.disable_tqdm,
                ) as pbar:
                    for max_index, _atoms in enumerate(self.analyte):
                        if max_index <= self.memory:
                            reference_soap = representation_file["soap_reference"][
                                :max_index
                            ]
                        else:
                            reference_soap = representation_file["soap_reference"][
                                max_index - self.memory : max_index
                            ]
                        analyte_soap = representation_file["soap_analyte"][max_index]
                        comparison = self.compare(reference_soap, analyte_soap)
                        self.result.append(float(comparison.numpy()))
                        self._save_plots(max_index)
                        pbar.update(1)
        self.similarities = convert_to_df(self.result)
        with h5py.File(self.soap_file, "a") as representation_file:
            with contextlib.suppress(KeyError):
                del representation_file["soap_reference"]
        if self.remove_database:
            self.unlink_database()

    def compare(self, reference: np.ndarray, analyte: np.ndarray) -> tf.Tensor:
        """Actual comparison method to use for similarity computation.

        Parameters
        ----------
        reference: np.ndarray
            reference representations to compare of shape (configuration, atoms, x)
        analyte: np.ndarray
            one representation to compare with the reference of shape (atoms, x).

        Returns
        -------
        maximum: tf.Tensor
            Similarity between analyte and reference.
        """
        raise NotImplementedError
