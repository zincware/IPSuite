"""Minimum Membership Kernel (MMKernel) module."""

import numpy as np
import tensorflow as tf

from ipsuite.configuration_comparison import ConfigurationComparison


@tf.function(jit_compile=False)
def _compare(reference: np.ndarray, analyte: np.ndarray) -> float:
    """Compute Minimum Membership Kernel.

    Compute Minimum Membership Kernel between list of descriptor representation
    and one specific representation.

    Parameters
    ----------
    reference: np.ndarray
        reference representations to compare of shape (configuration, atoms, x)
    analyte: np.ndarray
        one representation to compare with the reference of shape (atoms, x)

    Returns
    -------
    maximum: float
        minimal membership kernel between list of descriptor representation and one
        specific representation
    """
    nd_covariance = tf.einsum(
        "bij,kj->bik",  # add configuration dimension, include transpose in einsum
        tf.linalg.normalize(reference, axis=2)[0],
        tf.linalg.normalize(analyte, axis=1)[0],
    )
    best_match = tf.math.reduce_max(nd_covariance, axis=2)
    minimum = tf.math.reduce_min(best_match, axis=1)
    return tf.math.reduce_max(minimum)


@tf.function(jit_compile=True)
def _jit_compare(reference: np.ndarray, analyte: np.ndarray) -> float:
    return _compare(reference=reference, analyte=analyte)


class MMKernel(ConfigurationComparison):
    """Minimum Membership Kernel Node."""

    def compare(self, reference: np.ndarray, analyte: np.ndarray) -> float:
        """Compute Minimum Membership Kernel.

        Compute Minimum Membership Kernel between list of descriptor representation
        and one specific representation.

        Parameters
        ----------
        reference: np.ndarray
            reference representations to compare of shape (configuration, atoms, x)
        analyte: np.ndarray
            one representation to compare with the reference of shape (atoms, x)

        Returns
        -------
        maximum: float
            minimal membership kernel between list of descriptor representation and one
            specific representation
        """
        if self.use_jit:
            return _jit_compare(reference=reference, analyte=analyte)

        else:
            return _compare(reference=reference, analyte=analyte)
