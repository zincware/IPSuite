import pytest
import tensorflow as tf
from dscribe.descriptors import SOAP

from ipsuite.configuration_comparison import REMatch


@pytest.fixture
def soap_representation(atoms_list):
    soap = SOAP(species=[6, 8], rcut=7, nmax=4, lmax=4)
    return soap.create(atoms_list)


def test_compare(soap_representation):
    soap_representation = soap_representation
    reference = soap_representation[: (len(soap_representation) - 1)]
    analyte = soap_representation[-1]
    kernel = REMatch()
    comparison = kernel.compare(reference, analyte)
    assert isinstance(comparison, tf.Tensor)
    assert 0 <= comparison.numpy() <= 1.0
