##### The source of some functions come from tsgm package #####

import abc
import antropy
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


Tensor = T.Union[tf.Tensor, npt.NDArray]
OptTensor = T.Optional[Tensor]

class DatasetProperties:
    """
    Stores the properties of a dataset. Along with dimensions it can store properties of the covariates.
    """
    def __init__(self, N: int, D: int, T: int, variables: T.Optional[T.List] = None) -> None:
        """
        :param N: The number of samples.
        :type N: int
        :param D: The number of dimensions.
        :type data: int
        :param T: The number of timestemps.
        :type statistics: list
        :param variables: The properties of each covariate.
        :type variables: list
        """
        self.N = N
        self.D = D
        self.T = T
        self._variables = variables
        assert variables is None or self.D == len(variables)

class Dataset(DatasetProperties):
    """
    Wrapper for time-series datasets. Additional information is stored in `metadata` field.
    """
    def __init__(self, x: Tensor, y: Tensor, metadata: T.Optional[T.Dict] = None):
        """
        :param x: The matrix of time series with dimensions NxDxT
        :type x: tsgm.types.Tensor
        :param y: The lables of a time series.
        :type y: tsgm.types.Tensor
        :param metadata: Additional info for the dataset.
        :type statistics: typing.Optional[typing.Dict]
        """
        self._x = x
        self._y = y
        assert self._y is None or self._x.shape[0] == self._y.shape[0]

        self._metadata = metadata or {}
        self._graph = self._metadata.get("graph")
        super().__init__(N=self._x.shape[0], D=self._x.shape[1], T=self._x.shape[2])

    @property
    def X(self) -> Tensor:
        """
        Returns the time series tensor in format: n_samples x seq_len x feat_dim.
        """
        return self._x

    @property
    def y(self) -> Tensor:
        """
        Returns labels tensor.
        """
        return self._y

    @property
    def Xy(self) -> tuple:
        """
        Returns a tuple of a time series tensor and labels tensor.
        """
        return self._x, self._y

    @property
    def Xy_concat(self) -> Tensor:
        """
        Returns a concatenated time series and labels in a tensor.
        Output shape is n_sample x seq_len x feat_dim + y_dim
        """
        if self._y is None:
            return self._x
        elif len(self._y.shape) == 1:
            return np.concatenate((self._x, np.repeat(self._y[:, None, None], self._x.shape[1], axis=1)), axis=2)
        elif len(self._y.shape) == 2:
            if self._y.shape[1] == 1:
                return np.concatenate((self._x, np.repeat(self._y[:, :, None], self._x.shape[1], axis=1)), axis=2)
            elif self._y.shape[1] == self._x.shape[1]:
                return np.concatenate((self._x, self._y[:, :, None]), axis=2)
            else:
                return np.concatenate((self._x, np.repeat(self._y[:, None, :], self._x.shape[1], axis=1)), axis=2)
        else:
            raise ValueError("X & y are not compatible for Xy_concat operation")

    def _compatible(self, other_ds: "Dataset") -> bool:
        if self.X.shape[1:] == other_ds.X.shape[1:]:
            return self.y is None and other_ds.y is None or self.y.shape[1:] == other_ds.y.shape[1:]
        else:
            return False

    def _merge_meta(self, other_meta: dict) -> dict:
        return {**self._metadata, **other_meta}

    def _concatenate_dataset(self, other_ds: "Dataset") -> "Dataset":
        assert self._compatible(other_ds)
        return Dataset(
            np.concatenate((self.X, other_ds.X), axis=0),
            np.concatenate((self.y, other_ds.y), axis=0) if self.y is not None else None,
            self._merge_meta(other_ds._metadata)
        )

    def __add__(self, other_ds: "Dataset") -> "Dataset":
        """
        Returns a concatenated time series and labels in a tensor.
        Output shape is n_sample x seq_len x feat_dim + y_dim
        """
        assert self._compatible(other_ds)
        logger.warning("Operator '+' concatenates dataset objects")
        return self._concatenate_dataset(other_ds)

    def __or__(self, other_ds: "Dataset") -> "Dataset":
        return self._concatenate_dataset(other_ds)

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the time series in the dataset.
        """
        return self.X.shape

    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def seq_len(self) -> int:
        """
        Returns the length of sequences in the dataset.
        """
        return self.X.shape[1]

    @property
    def feat_dim(self) -> int:
        """
        Returns the size of feature dimension in the time series.
        """
        return self.X.shape[2]

    @property
    def output_dim(self) -> int:
        """
        Returns the number of classes in the dataset.
        """
        output_dim = len(set(self.y))
        if output_dim > len(self.y) * 0.5:
            logger.warning("either the number of classes if huge or it is not a classification dataset")
        return len(set(self.y))

DatasetOrTensor = T.Union[Dataset, Tensor]

def _spectral_entropy_per_feature(X: TensorLike) -> TensorLike:
    return antropy.spectral_entropy(X.ravel(), sf=1, method='welch', normalize=True)

def _spectral_entropy_per_sample(X: TensorLike) -> TensorLike:
    if len(X.shape) == 1:
        X = X[:, None]
    return np.apply_along_axis(_spectral_entropy_per_feature, 0, X)

def _spectral_entropy_sum(X: TensorLike) -> TensorLike:
    return np.apply_along_axis(_spectral_entropy_per_sample, 1, X)

def _dataset_or_tensor_to_tensor(D1: DatasetOrTensor) -> Tensor:
    if isinstance(D1, Dataset):
        return D1.X
    else:
        return D1

class Metric(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> float:
        pass

class EntropyMetric(Metric):
    """
    Calculates the spectral entropy of a dataset or tensor.

    This metric measures the randomness or disorder in a dataset or tensor
    using spectral entropy, which is a measure of the distribution of energy
    in the frequency domain.

    Args:
        d (tsgm.dataset.DatasetOrTensor): The input dataset or tensor.

    Returns:
        float: The computed spectral entropy.

    """
    def __call__(self, d: DatasetOrTensor) -> float:
        """
        Calculate the spectral entropy of the input dataset or tensor.

        Args:
            d (tsgm.dataset.DatasetOrTensor): The input dataset or tensor.

        Returns:
            float: The computed spectral entropy.
        """
        X = _dataset_or_tensor_to_tensor(d)
        return np.sum(_spectral_entropy_sum(X), axis=None)


# Usage example
if __name__ == "__main__":
    # Load and preprocess data
    scaler = TSFeatureWiseScaler()
    real = np.load('real.npy')
    real = scaler.fit_transform(real)
    real1 = real[:, :, 0]
    real2 = real[:, :, 1]

    gen = np.load('diffusion.npy')
    gen1 = gen[:, :, 0]
    gen2 = gen[:, :, 1]

    # Calculate spectral entropy
    spec_entropy = EntropyMetric()
    print("Spectral Entropy Real1:", spec_entropy(real1))
    print("Spectral Entropy Gen1:", spec_entropy(gen1))
    print("Spectral Entropy Real2:", spec_entropy(real2))
    print("Spectral Entropy Gen2:", spec_entropy(gen2))
