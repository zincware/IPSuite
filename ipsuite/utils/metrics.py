"""Utils for computing metrics."""
import numpy as np


def calculate_l_p_norm(y_true: np.ndarray, y_pred: np.ndarray, p: int = 2):
    r"""Calculate the mean of the l_p_norm of given data.

    .. math::
        l_{p} = \frac{1}{N} \sum_{i}^{N} \left| x_{i} - x_{i}^{\text{true}}\right|^{p}.

    Parameters
    ----------
    y_true: np.ndarray
        reference values with shape (b,)
    y_pred: np.ndarray
        values to compare reference with shape (b,)
    p: int
        order of the Lp norm
    Returns
    -------
    mean: float
        mean of the lp norm of the given data.
    """
    distance = np.abs(y_true - y_pred)
    return (np.sum(distance**p) ** (1 / p)) / len(y_true)


def maximum_error(y_true, y_pred):
    """Calculate the maximum error."""
    return np.max(np.abs(y_true - y_pred))


def mean_squared_error(data_a, data_b):
    """Calculate the mean squared error between data_ids."""
    return np.mean((np.array(data_a) - np.array(data_b)) ** 2)


def root_mean_squared_error(data_a, data_b):
    """Calculate the root mean squared error between data_ids."""
    return np.sqrt(mean_squared_error(data_a, data_b))


def mean_absolute_error(data_a, data_b):
    """Calculate the mean absolute error."""
    return np.mean(np.abs(data_a - data_b))


def relative_rmse(true, pred):
    """Calculate the relative root mean squared error between data_ids."""
    numerator = np.sum((np.array(true) - np.array(pred)) ** 2)
    denominator = np.sum((np.array(true) - np.mean(true)) ** 2)
    return np.sqrt(numerator / denominator)


def get_u_vecs(vector):
    """Get unit vectors from a vector array."""
    return vector / np.linalg.norm(vector, axis=-1)[:, None]


def get_angles(vec1, vec2) -> np.ndarray:
    """Compute the angle between two vectors."""
    u_vec1 = get_u_vecs(vec1)
    u_vec2 = get_u_vecs(vec2)
    return np.rad2deg(
        np.arccos(np.clip(np.einsum("ix, ix -> i", u_vec1, u_vec2), -1.0, 1.0))
    )


def get_full_metrics(true: np.ndarray, prediction: np.ndarray) -> dict:
    """Calculate metrics for a given true and predicted value."""
    return {
        "rmse": root_mean_squared_error(true, prediction),
        "mse": mean_squared_error(true, prediction),
        "mae": mean_absolute_error(true, prediction),
        "max": maximum_error(true, prediction),
        "lp4": calculate_l_p_norm(true, prediction, p=4),
        "rrmse": relative_rmse(true, prediction),
        # "pearsonr": pearsonr(true, prediction)[0],
    }
