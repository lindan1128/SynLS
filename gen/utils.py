import math
import numpy as np
import typing as T
import tensorflow as tf
import numpy.typing as npt
from tensorflow.python.types.core import TensorLike

Tensor = T.Union[tf.Tensor, npt.NDArray]
OptTensor = T.Optional[Tensor]

EPS = 1e-18
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

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.99):
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