import math
import numpy as np
import typing as T
import sklearn
import sklearn.manifold
import tensorflow as tf
import numpy.typing as npt

from tensorflow import keras
from tensorflow.python.types.core import TensorLike

Tensor = T.Union[tf.Tensor, npt.NDArray]
OptTensor = T.Optional[Tensor]


EPS = 1e-18

class TSFeatureScaler:
    """Global min-max scaler: [0,1] -> [-1,1]"""

    def __init__(self) -> None:
        self.min_val = None
        self.max_val = None

    def fit(self, X: TensorLike) -> "TSFeatureScaler":
        self.min_val = np.min(X)
        self.max_val = np.max(X)
        return self

    def transform(self, X: TensorLike) -> TensorLike:
        X_scaled = (X - self.min_val) / (self.max_val - self.min_val + EPS)
        return 2.0 * X_scaled - 1.0

    def inverse_transform(self, X: TensorLike) -> TensorLike:
        X_scaled = (X + 1.0) / 2.0
        return X_scaled * (self.max_val - self.min_val + EPS) + self.min_val

    def fit_transform(self, X: TensorLike) -> TensorLike:
        return self.fit(X).transform(X)

    def get_range(self) -> T.Tuple[float, float]:
        return (self.min_val, self.max_val)


class TSFeatureWiseScaler():
    def __init__(self, feature_range: T.Tuple[float, float] = (0, 1)) -> None:
        assert len(feature_range) == 2

        self._min_v, self._max_v = feature_range

    # X: N x T x D
    def fit(self, X: TensorLike) -> "TSFeatureWiseScaler":
        D = X.shape[2]
        self.mins = np.zeros(D)
        self.maxs = np.zeros(D)

        for i in range(D):
            self.mins[i] = np.min(X[:, :, i])
            self.maxs[i] = np.max(X[:, :, i])

        return self

    def transform(self, X: TensorLike) -> TensorLike:
        return ((X - self.mins) / (self.maxs - self.mins + EPS)) * (self._max_v - self._min_v) + self._min_v

    def inverse_transform(self, X: TensorLike) -> TensorLike:
        X -= self._min_v
        X /= self._max_v - self._min_v
        X *= (self.maxs - self.mins + EPS)
        X += self.mins
        return X

    def fit_transform(self, X: TensorLike) -> TensorLike:
        self.fit(X)
        return self.transform(X)
 

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.99): # beta_end=0.99
    betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 0, 0.999)
    return betas


def reconstruction_loss_by_axis(original: tf.Tensor, reconstructed: tf.Tensor, axis: int = 0) -> tf.Tensor:
    """Reconstruction loss: axis=0 sum of squared diffs, axis=1 features MSE, axis=2 time MSE."""
    if axis == 0:
        return tf.reduce_sum(tf.math.squared_difference(original, reconstructed))
    else:
        return tf.losses.mean_squared_error(tf.reduce_mean(original, axis=axis), tf.reduce_mean(reconstructed, axis=axis))


def gen_sine_dataset(N: int, T: int, D: int, max_value: int = 10) -> npt.NDArray:
    result = []
    for i in range(N):
        result.append([])
        a = np.random.random() * max_value
        shift = np.random.random() * max_value + 1
        ts = np.arange(0, T, 1)
        for d in range(1, D + 1):
            result[-1].append((a * np.sin((d + 3) * ts / 25. + shift)).T)

    return np.transpose(np.array(result), [0, 2, 1])


def gen_sine_vs_const_dataset(N: int, T: int, D: int, max_value: int = 10, const: int = 0) -> T.Tuple[TensorLike, TensorLike]:
    result_X, result_y = [], []
    for i in range(N):
        scales = np.random.random(D) * max_value
        consts = np.random.random(D) * const
        shifts = np.random.random(D) * 2
        alpha = np.random.random()
        if np.random.random() < 0.5:
            times = np.repeat(np.arange(0, T, 1)[:, None], D, axis=1) / 10
            result_X.append(np.sin(alpha * times + shifts) * scales)
            result_y.append(0)
        else:
            result_X.append(np.tile(consts, (T, 1)))
            result_y.append(1)
    return np.array(result_X), np.array(result_y)
