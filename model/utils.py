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
    """Global time series scaler that scales all features to [0,1] then normalizes to [-1,1]"""
    
    def __init__(self) -> None:
        self.min_val = None
        self.max_val = None
        
    def fit(self, X: TensorLike) -> "TSFeatureScaler":
        """
        Fit scaler to data
        
        Args:
            X: Input tensor of shape [N, T, D] 
               (N: samples, T: timesteps, D: features)
        """
        # 计算整个数据集的全局最大最小值
        self.min_val = np.min(X)
        self.max_val = np.max(X)
        return self
    
    def transform(self, X: TensorLike) -> TensorLike:
        """
        Transform data in two steps:
        1. Scale to [0,1] using min-max scaling
        2. Normalize to [-1,1]
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler must be fitted before transform")
            
        # 1. 缩放到[0,1]
        X_scaled = (X - self.min_val) / (self.max_val - self.min_val + EPS)
        
        # 2. 归一化到[-1,1]
        X_normalized = 2.0 * X_scaled - 1.0
        
        return X_normalized
    
    def inverse_transform(self, X: TensorLike) -> TensorLike:
        """
        Inverse transform data:
        1. From [-1,1] back to [0,1]
        2. From [0,1] back to original range
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler must be fitted before inverse_transform")
            
        # 1. 从[-1,1]转回[0,1]
        X_scaled = (X + 1.0) / 2.0
        
        # 2. 从[0,1]转回原始范围
        X_original = X_scaled * (self.max_val - self.min_val + EPS) + self.min_val
        
        return X_original
    
    def fit_transform(self, X: TensorLike) -> TensorLike:
        """Fit to data, then transform it"""
        return self.fit(X).transform(X)
    
    def get_range(self) -> T.Tuple[float, float]:
        """获取原始数据的范围"""
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler must be fitted first")
        return (self.min_val, self.max_val)


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
    """
    Calculate the reconstruction loss based on a specified axis.

    This function computes the reconstruction loss between the original data and
    the reconstructed data along a specified axis. The loss can be computed in
    two ways depending on the chosen axis:

    - When `axis` is 0, it computes the loss as the sum of squared differences
      between the original and reconstructed data for all elements.
    - When `axis` is 1 or 2, it computes the mean squared error (MSE) between the
      mean values along the chosen axis for the original and reconstructed data.

    Parameters:
    ----------
    original : tf.Tensor
        The original data tensor.

    reconstructed : tf.Tensor
        The reconstructed data tensor, typically produced by an autoencoder.

    axis : int, optional (default=0)
        The axis along which to compute the reconstruction loss:
        - 0: All elements (sum of squared differences).
        - 1: Along features (MSE).
        - 2: Along time steps (MSE).

    Returns:
    -------
    tf.Tensor
        The computed reconstruction loss as a TensorFlow tensor.
    Notes:
    ------
    - This function is commonly used in the context of autoencoders and other
      reconstruction-based models to assess the quality of the reconstruction.
    - The choice of `axis` determines how the loss is calculated, and it should
      align with the data's structure.
    """

    # axis=0 all (sum of squared diffs)
    # axis=1 features (MSE)
    # axis=2 times (MSE)
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
