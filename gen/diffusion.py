import math
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from gen.utils import linear_beta_schedule, cosine_beta_schedule
import matplotlib.pyplot as plt

class GaussianDiffusion:

    def __init__(
            self,
            beta_schedule='cosine',
            timesteps=10,
            clip_min=-1.0,
            clip_max=1.0,
    ):

        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)
        self.sqrt_recip_alphas = tf.constant(np.sqrt(1. / alphas), dtype=tf.float32)

        self.sqrt_alphas_cumprod = tf.constant(np.sqrt(self.alphas_cumprod), dtype=tf.float32)
        self.sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1.0 - self.alphas_cumprod), dtype=tf.float32)
        self.log_one_minus_alphas_cumprod = tf.constant(np.log(1. - alphas_cumprod), dtype=tf.float32)
        self.sqrt_recip_alphas_cumprod = tf.constant(np.sqrt(1. / alphas_cumprod), dtype=tf.float32)
        self.sqrt_recipm1_alphas_cumprod = tf.constant(np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float32)
        self.posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

        self.posterior_log_variance_clipped = tf.constant(
            np.log(np.maximum(self.posterior_variance, 1e-20)), dtype=tf.float32
        )

        self.posterior_mean_coef1 = tf.constant(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

        self.posterior_mean_coef2 = tf.constant(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

    def _extract(self, a, t, x_shape):
        """Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

        Args:
            a: Tensor to extract from
            t: Timestep for which the coefficients are to be extracted
            x_shape: Shape of the current batched samples
        """
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1])

    def q_sample(self, x_start, t):
        """Diffuse the data.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
            noise: Gaussian noise to be added at the current timestep
        Returns:
            Diffused samples at timestep `t`
        """

        x_start_shape = tf.shape(x_start)
        samp = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        noise = tf.random.normal(shape=tf.shape(x_start), dtype='float32')
        weight_noise = 0.1 * self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape) * noise
        diffused_sample = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start + 0.1 * self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start_shape) * noise

        return samp, weight_noise, diffused_sample

    def predict_start_from_noise(self, x_t, t, noise):
        x_t_shape = tf.shape(x_t)
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
                - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        x_t_shape = tf.shape(x_t)
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=False):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=False):
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )
        noise = tf.random.normal(shape=tf.shape(x), dtype=x.dtype)
        nonzero_mask = tf.reshape(
            1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1]
        )
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise #* 0.1

class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        # temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply

# Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )

def build_encoder_time(embed_dim=16, num_heads=2, ff_dim=32):
    def apply(inputs):
        x, t = inputs
        position_embedding_layer = layers.Embedding(x.shape[1], embed_dim)
        pos_encoding = position_embedding_layer(tf.range(x.shape[1]))
        embeddings = x + pos_encoding + t

        # Encoder blocks
        for _ in range(2):  # Repeat twice
            # Multi-head self-attention mechanism
            attention_output, attention_score = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
                embeddings, embeddings, return_attention_scores=True)

            # Add residual connection and layer normalization
            x = layers.Add()([embeddings, attention_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)

            # Feed-forward network
            ff_output = layers.Dense(ff_dim, activation="relu")(x)
            ff_output = layers.Dense(embed_dim)(ff_output)

            # Add residual connection and layer normalization
            x = layers.Add()([x, ff_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)

        return x, attention_score

    return apply

def build_encoder_variales(embed_dim=16, num_heads=2, ff_dim=32):
    def apply(inputs):
        x, t = inputs
        x = layers.Conv1D(16, kernel_size=3, padding='same')(x)
        embeddings = x + t

        # Encoder blocks
        for _ in range(2):  # Repeat twice
            # Multi-head self-attention mechanism
            attention_output, attention_score = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
                embeddings, embeddings, return_attention_scores=True)

            # Add residual connection and layer normalization
            x = layers.Add()([embeddings, attention_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)

            # Feed-forward network
            ff_output = layers.Dense(ff_dim, activation="relu")(x)
            ff_output = layers.Dense(embed_dim)(ff_output)

            # Add residual connection and layer normalization
            x = layers.Add()([x, ff_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)

        return x, attention_score

    return apply

def build_decoder(embed_dim=16, num_heads=2, ff_dim=32):
    def apply(inputs):
        encoder_outputs, t = inputs
        position_embedding_layer = layers.Embedding(encoder_outputs.shape[1], embed_dim)
        pos_encoding = position_embedding_layer(tf.range(encoder_outputs.shape[1]))
        dec_embeddings = encoder_outputs + pos_encoding + t

        # Decoder blocks
        dec_output = dec_embeddings
        for _ in range(2):  # Repeat twice

            # Multi-head attention over encoder outputs
            attention2_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
                dec_output, encoder_outputs)

            # Add residual connection and layer normalization
            dec_output = layers.Add()([dec_output, attention2_output])
            dec_output = layers.LayerNormalization(epsilon=1e-6)(dec_output)

            # Feed-forward network
            ff_output = layers.Dense(ff_dim, activation="relu")(dec_output)
            ff_output = layers.Dense(embed_dim)(ff_output)

            # Add residual connection and layer normalization
            dec_output = layers.Add()([dec_output, ff_output])
            dec_output = layers.LayerNormalization(epsilon=1e-6)(dec_output)

        return dec_output

    return apply

def build_model(time_len, fea_num):
    x = x_input = layers.Input(shape=(time_len, fea_num), name='x_input')
    x = layers.Conv1D(16, kernel_size=3, padding='same')(x)

    time_input = keras.Input(shape=(1,), name="time_input")
    temb = TimeEmbedding(dim=16)(time_input)

    encoder_time = build_encoder_time()
    encoder_time_outputs, att_time = encoder_time([x, temb])

    encoder_variales = build_encoder_variales()
    encoder_variales_outputs, att_variables = encoder_variales([tf.transpose(x_input, perm=[0, 2, 1]), temb])
    encoder_variales_outputs = tf.transpose(encoder_variales_outputs, perm=[0, 2, 1])
    encoder_variales_outputs = layers.Dense(x_input.shape[1])(encoder_variales_outputs)
    encoder_variales_outputs = tf.transpose(encoder_variales_outputs, perm=[0, 2, 1])

    encoder_outputs = encoder_time_outputs + encoder_variales_outputs

    decoder = build_decoder()
    decoder_outputs = decoder([encoder_outputs, temb])

    x = layers.Conv1D(x_input.shape[2], kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0))(decoder_outputs)

    return keras.Model(inputs=[x_input, time_input], outputs=[x, att_time, att_variables])

class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, timesteps, gdf_util, data, ema=0.999):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = 10
        self.gdf_util = GaussianDiffusion()
        self.ema = ema
        self.data = data

    def train_step(self, data):
        # 1. Get the batch size
        batch_size = tf.shape(data)[0]

        # 2. Sample timesteps uniformly
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int32
        )

        with tf.GradientTape() as tape:
            # 3. Diffuse the images with noise
            _, noise, ts_t = self.gdf_util.q_sample(data, t)

            # 4. Pass the diffused images and time steps to the network
            pred_noise, _, _ = self.network([ts_t, t], training=True)

            # 5. Calculate the loss
            loss = self.loss(noise, pred_noise)

        # 6. Get the gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 7. Update the weights of the network
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 8. Updates the weight values for the network with EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 9. Return loss values
        return {"loss": loss}

    def generate_ts(self, num_ts=16):
        # 1. Randomly sample noise (starting point for reverse process)
        samples = tf.random.normal(
                    shape=(num_ts, tf.shape(self.data)[1], tf.shape(self.data)[2]), dtype=tf.float32
                )

        # batch_size = tf.shape(self.data)[0]
        # indices = tf.random.uniform(shape=(num_ts,), minval=0, maxval=batch_size, dtype=tf.int32)
        # samples = tf.gather(self.data, indices)
        # samples = tf.cast(samples, tf.float32)

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            tt = tf.cast(tf.fill(num_ts, i), dtype=tf.int32)
            pred_noise, _, _ = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=num_ts
            )
            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt, clip_denoised=False
            )
        # 2. Return generated samples
        return samples

    def plot_ts(
            self, num_rows=1, num_cols=8
    ):
        """Utility to plot images using the diffusion model during training."""
        generated_samples = self.generate_ts(num_ts=num_rows * num_cols).numpy()
        generated_samples = generated_samples.reshape(num_rows, num_cols, self.data.shape[1], self.data.shape[2])
        fig = plt.figure(figsize=(8, 1), constrained_layout=True)
        gs = fig.add_gridspec(num_rows, num_cols)
        for n_row in range(num_rows):
            for n_col in range(num_cols):
                f_ax = fig.add_subplot(gs[n_row, n_col])
                f_ax.plot((generated_samples[n_row, n_col]))
                f_ax.axis("on")
        plt.show()


