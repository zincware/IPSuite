"""ipsuite helper modules."""

import contextlib
from logging import Logger

import typing_extensions as tyex
import znflow
from zntrack import Node


def setup_ase():
    """Add uncertainty keys to ASE all properties."""
    from ase.calculators.calculator import all_properties

    for val in [
        "forces_uncertainty",
        "energy_uncertainty",
        "stress_uncertainty",
        "node_energy",
    ]:
        if val not in all_properties:
            all_properties.append(val)


@tyex.deprecated(
    "It is recommended to pass the attribute directly, instead of giving a 'zntrack.Node'"
    " instance."
)
def get_deps_if_node(obj, attribute: str):
    """Apply getdeps if obj is subclass/instance of a Node.

    Parameters
    ----------
    obj: any
        Any object that is either a Node or not.
    attribute: str
        Name of the attribute to get.

    Returns
    -------
    Either the requested attribute if obj is a Node.
    Otherwise, it will return the obj itself.

    """
    if isinstance(obj, (list, tuple)):
        return [get_deps_if_node(x, attribute) for x in obj]
    with contextlib.suppress(TypeError):
        if issubclass(obj, Node):
            return obj @ attribute  # TODO attribute access should also work, right?
    if isinstance(obj, znflow.Connection):
        if obj.attribute is None:
            if obj.item is not None:
                raise ValueError("Cannot get attribute of item.")
            return znflow.Connection(obj.instance, attribute)
    return obj @ attribute if isinstance(obj, Node) else obj


def check_duplicate_keys(dict_a: dict, dict_b: dict, log: Logger) -> None:
    """Check if a key of dict_a is present in dict_b and then log a warning."""
    for key in dict_a:
        if key in dict_b:
            log.warning(
                f"Found <{key}> in given config file. Please be aware that <{key}>"
                " will be overwritten by MLSuite!"
            )
