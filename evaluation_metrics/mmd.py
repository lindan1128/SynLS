
import numpy as np
import numpy.typing as npt
import typing as T
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.types.core import TensorLike

EPSILON = 1e-18


class TSFeatureWiseScaler:
    """
    Feature-wise scaler for time series data.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        assert len(feature_range) == 2, "Feature range must be a tuple of two elements (min, max)"
        self.min_v, self.max_v = feature_range
        self.mins = None
        self.maxs = None

    def fit(self, X: np.ndarray) -> 'TSFeatureWiseScaler':
        """
        Fit the scaler on the data.

        Parameters:
            X (np.ndarray): Time series data of shape (N, T, D).

        Returns:
            TSFeatureWiseScaler: self
        """
        self.mins = np.min(X, axis=(0, 1))
        self.maxs = np.max(X, axis=(0, 1))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data.

        Parameters:
            X (np.ndarray): Time series data to scale.

        Returns:
            np.ndarray: Scaled time series data.
        """
        return (X - self.mins) / (self.maxs - self.mins + EPSILON) * (self.max_v - self.min_v) + self.min_v

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the scaled data.

        Parameters:
            X (np.ndarray): Scaled time series data to invert.

        Returns:
            np.ndarray: Original scale time series data.
        """
        return (X - self.min_v) / (self.max_v - self.min_v) * (self.maxs - self.mins + EPSILON) + self.mins

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters:
            X (np.ndarray): Time series data to fit and transform.

        Returns:
            np.ndarray: Scaled time series data.
        """
        return self.fit(X).transform(X)

def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

# Usage example
if __name__ == "__main__":
    # Load and preprocess data
    scaler = TSFeatureWiseScaler()
    real = np.load('real.npy')
    real = scaler.fit_transform(real)
    real = real.reshape(real.shape[0], real.shape[1]*real.shape[2])

    gen = np.load('diffusion.npy')
    gen = gen.reshape(gen.shape[0], gen.shape[1]*gen.shape[2])

    # Calculate MMD
    print(mmd_rbf(real, gen))
