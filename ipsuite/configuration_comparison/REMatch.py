"""REMatch kernel Node."""

import numpy as np
import tensorflow as tf
import zntrack
from dscribe.kernels import REMatchKernel
from sklearn.preprocessing import normalize

from ipsuite.configuration_comparison import ConfigurationComparison


class REMatch(ConfigurationComparison):
    """REMatch kernel implementation.

    Attributes
    ----------
    metric: str
        The pairwise metric used for calculating the local similarity. Accepts any of
        the sklearn pairwise metric strings (e.g. “linear”, “rbf”, “laplacian”,
        “polynomial”) or a custom callable. A callable should accept two arguments and
        the keyword arguments passed to this object as kernel_params, and should return
        a floating point number.
    alpha: float
        Parameter controlling the entropic penalty. Values close to zero approach the
        best-match solution and values towards infinity approach the average kernel.
    threshold: float
        Convergence threshold used in the Sinkhorn-algorithm.

    """

    metric: str = zntrack.zn.params("linear")
    alpha: float = zntrack.zn.params(1.0)
    threshold: float = zntrack.zn.params(1e-6)

    def _post_init_(self):
        """Initialise the REMatchKernel instance."""
        self.re = REMatchKernel(
            metric=self.metric, alpha=self.alpha, threshold=self.threshold
        )

    def compare(self, reference: np.ndarray, analyte: np.ndarray) -> float:
        """Compare configurations to each other using the REMatch kernel.

        This will find the maximum similarity of a configuration compared to a set of
        configurations using the Regularized Entropy Kernel implemented by dscribe.

        Parameters
        ----------
        reference : np.ndarray
            Set of configurations in descriptor representation to compare with analyte
        analyte : np.ndarray
            Descriptor representation to compare with reference set

        Returns
        -------
        float
            Maximum similarity

        """
        rematches = []
        for ref in reference:
            rematch = self.re.create([normalize(analyte), normalize(ref)])
            rematches.append(rematch[0][1])
        return tf.math.reduce_max(np.array(rematches))
