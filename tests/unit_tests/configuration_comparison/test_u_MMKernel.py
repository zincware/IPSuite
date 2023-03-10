import pytest
import tensorflow as tf
from dscribe.descriptors import SOAP

from ipsuite.configuration_comparison import MMKernel


@pytest.fixture
def soap_representation(atoms_list):
    soap = SOAP(species=[6, 8], rcut=7, nmax=4, lmax=4)
    return soap.create(atoms_list)


def test_compare(soap_representation):
    soap_representation = soap_representation
    reference = soap_representation[: (len(soap_representation) - 1)]
    analyte = soap_representation[-1]
    kernel = MMKernel()
    comparison = kernel.compare(reference, analyte)
    assert isinstance(comparison, tf.Tensor)
    assert 0 <= comparison.numpy() <= 1.0


def test_old_method_v_new():
    def old_mmk(reference, analyte):
        best_match = []
        assert len(reference.shape) == 3  # check (configuration, atoms, x) is fulfilled
        for config in reference:
            covariance = tf.einsum(
                "ij,jk->ik",
                tf.linalg.normalize(config, axis=1)[0],
                tf.transpose(tf.linalg.normalize(analyte, axis=1)[0]),
            )
            best_match.append(tf.math.reduce_max(covariance, axis=1))
            minimum = tf.math.reduce_min(best_match, axis=1)
        maximum = tf.math.reduce_max(minimum)
        return maximum

    kernel = MMKernel()

    for _ in range(100):
        reference = tf.random.normal((64, 100, 32))
        analyte = tf.random.normal((100, 32))

        # small uncertainty due to XLA and different order of summation
        # in einsum / transpose
        assert (
            abs(old_mmk(reference, analyte) - kernel.compare(reference, analyte)) < 1e-6
        )
