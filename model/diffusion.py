import math
import shutil
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model.utils import linear_beta_schedule, cosine_beta_schedule
import os
from model.utils import TSFeatureScaler

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
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1])

    def q_sample(self, x_start, t):
        x_start_shape = tf.shape(x_start)
        samp = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        noise = tf.random.normal(shape=tf.shape(x_start), dtype='float32')
        weight_noise = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape) * noise
        diffused_sample = x_start + weight_noise 
        diffused_sample = tf.clip_by_value(diffused_sample , -0.99, 0.99)
        weight_noise = diffused_sample - x_start
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
        variance_term = tf.exp(0.5 * model_log_variance)
        noise = tf.random.normal(shape=tf.shape(x), dtype=x.dtype)
        nonzero_mask = tf.reshape(
        1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1]
    )
        noise_term = variance_term * nonzero_mask * noise
        sample = model_mean + noise_term
        return sample

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
        return temb
    return apply

def kernel_init(scale):

    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )

def pairwise_transformer_encoder(x, time_emb, d_model=16, n_heads=2):

    input_shape = x.shape
    time_len = input_shape[1]
    x_transposed = tf.transpose(x, perm=[0, 2, 1])
    x_proj = layers.Dense(d_model)(x_transposed)
    time_emb = tf.expand_dims(time_emb, 1)
    time_emb = tf.tile(time_emb, [1, tf.shape(x_proj)[1], 1])
    x = x_proj + time_emb
    for _ in range(2):
        x = transformer_encoder_layer(x, d_model, n_heads)
    x = layers.Dense(time_len)(x)
    x = tf.transpose(x, perm=[0, 2, 1])
    x = layers.Dense(d_model)(x)
    return x

def time_transformer_encoder(x, time_emb, d_model=16, n_heads=2):

    pos_emb = get_positional_embedding(tf.shape(x)[1], d_model)
    x = layers.Dense(d_model)(x)
    time_emb = tf.expand_dims(time_emb, 1)
    time_emb = tf.tile(time_emb, [1, tf.shape(x)[1], 1])
    x = x + pos_emb + time_emb
    for _ in range(2):
        x = transformer_encoder_layer(x, d_model, n_heads)
    return x

def transformer_encoder_layer(x, d_model, n_heads):

    attention_output = layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=d_model // n_heads
    )(x, x)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    ffn_output = layers.Dense(d_model * 4, activation='relu')(x)
    ffn_output = layers.Dense(d_model)(ffn_output)
    x = layers.Add()([x, ffn_output])
    x = layers.LayerNormalization()(x)
    return x

def decoder_module(encoded, time_emb):

    time_emb = tf.expand_dims(time_emb, 1)
    time_emb = tf.tile(time_emb, [1, tf.shape(encoded)[1], 1])
    x = layers.Concatenate(axis=-1)([encoded, time_emb])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    return x

def get_time_embedding(timesteps, embedding_dim):

    timesteps = tf.expand_dims(timesteps, -1)
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    emb = tf.cast(timesteps, dtype=tf.float32) * emb[None, :]
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
    if embedding_dim % 2 == 1:
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    return emb

def get_positional_embedding(sequence_length, embedding_dim):

    positions = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
    dimensions = tf.range(0, embedding_dim, 2, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000.0, (2 * dimensions) / tf.cast(embedding_dim, tf.float32))
    angle_rads = positions * angle_rates
    pos_encoding = tf.concat(
        [tf.sin(angle_rads), tf.cos(angle_rads)],
        axis=-1
    )
    if embedding_dim % 2 == 1:
        pos_encoding = tf.pad(pos_encoding, [[0, 0], [0, 1]])
    pos_encoding = tf.expand_dims(pos_encoding, 0)
    return pos_encoding

def build_model(time_len, fea_num, d_model=16, n_heads=2, encoder_type='dual'):

    print(f"\nBuilding model with encoder type: {encoder_type}")
    print(f"Input shape: time_len={time_len}, features={fea_num}, d_model={d_model}")
    x_input = layers.Input(shape=(time_len, fea_num))
    time_input = layers.Input(shape=())
    time_emb = get_time_embedding(time_input, d_model)
    encoded_features = []

    if encoder_type in ['time', 'dual']:
        print("→ Using Time Transformer Encoder")

        time_encoded = time_transformer_encoder(
            x_input,
            time_emb,
            d_model=d_model,
            n_heads=n_heads
        )
        print(f"  Time encoder output shape: {time_encoded.shape}")
        encoded_features.append(time_encoded)

    if encoder_type in ['pairwise', 'dual']:
        print("→ Using Pairwise Correlation Encoder")

        pairwise_encoded = pairwise_transformer_encoder(
            x_input,
            time_emb,
            d_model=d_model,
            n_heads=n_heads
        )
        print(f"  Pairwise encoder output shape: {pairwise_encoded.shape}")
        encoded_features.append(pairwise_encoded)

    if encoder_type == 'dual':
        print("→ Combining both encoders")
        encoded = layers.Concatenate(axis=-1)(encoded_features)
        print(f"  Combined shape before projection: {encoded.shape}")
        encoded = layers.Dense(d_model)(encoded)
        print(f"  Final encoded shape after projection: {encoded.shape}")
    else:
        encoded = encoded_features[0]
        print(f"→ Using single encoder output shape: {encoded.shape}")

    if encoder_type != 'dual':
        print("→ Adding residual connection")
        encoded = layers.Add()([encoded, layers.Dense(d_model)(x_input)])

    decoded = decoder_module(encoded, time_emb)
    print(f"→ Decoder output shape: {decoded.shape}")

    output = layers.Dense(fea_num)(decoded)
    print(f"→ Final output shape: {output.shape}")
    print("Model building completed!\n")

    return keras.Model(inputs=[x_input, time_input], outputs=output)

class DiffusionModel(keras.Model):

    def __init__(self, network, ema_network, timesteps, gdf_util, data, ema=0.999):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.data = data 
        self.ema = ema
        
    def train_step(self, data):

        batch_size = tf.shape(data)[0]

        t = tf.random.uniform(
            minval=0, 
            maxval=self.timesteps, 
            shape=(batch_size,), 
            dtype=tf.int32
        )

        old_weights = [tf.identity(w) for w in self.network.trainable_weights]
        
        with tf.GradientTape() as tape:
            _, noise, x_t = self.gdf_util.q_sample(data, t)
            pred_noise = self.network([x_t, t], training=True)
            loss = self.loss(noise, pred_noise)
        
        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        new_weights = self.network.trainable_weights
        weight_changes = []
        for old_w, new_w in zip(old_weights, new_weights):
            diff = tf.reduce_max(tf.abs(old_w - new_w))
            weight_changes.append(diff)
        max_change = tf.reduce_max(weight_changes)

        return {
        "loss": loss,
        "weight_max_change": max_change,
        "has_weight_changed": max_change > 0
    }

    def check_noise_levels(self):

        x_0 = tf.cast(self.data, tf.float32)  
        
        print("\n=== Noise Level Analysis ===")
        print(f"Using all {len(self.data)} samples")
        print("Checking noise levels at different timesteps:")

        timesteps_to_check = [
            0,  
            self.timesteps//4,  
            self.timesteps//2,  
            3*self.timesteps//4,  
            self.timesteps-1 
        ]
        
        for t in timesteps_to_check:

            sqrt_alphas = tf.sqrt(self.gdf_util.alphas_cumprod[t])
            sqrt_one_minus_alphas = tf.sqrt(1 - self.gdf_util.alphas_cumprod[t])

            _, noise, x_t = self.gdf_util.q_sample(x_0, tf.fill([len(x_0)], t))
            pred_noise = self.ema_network.predict([x_t, tf.fill([len(x_0)], t)], verbose=0)

            print(f"\nTimestep {t}:")
            print(f"Signal scaling factor (sqrt_alphas): {sqrt_alphas:.4f}")
            print(f"Noise scaling factor (sqrt_1-alphas): {sqrt_one_minus_alphas:.4f}")
            print(f"Original data range: [{tf.reduce_min(x_0):.4f}, {tf.reduce_max(x_0):.4f}]")
            print(f"Noisy data range: [{tf.reduce_min(x_t):.4f}, {tf.reduce_max(x_t):.4f}]")
            print(f"Added noise - Mean: {tf.reduce_mean(noise):.4f}, Std: {tf.math.reduce_std(noise):.4f}")
            print(f"Scaled noise - Mean: {tf.reduce_mean(sqrt_one_minus_alphas * noise):.4f}, Std: {tf.math.reduce_std(sqrt_one_minus_alphas * noise):.4f}")
            print(f"Predicted noise - Mean: {tf.reduce_mean(pred_noise):.4f}, Std: {tf.math.reduce_std(pred_noise):.4f}")
            print(f"Noise prediction error: {tf.reduce_mean(tf.abs(noise - pred_noise)):.4f}")

    def generate_ts(self, num_ts=16):
       
        if num_ts > len(self.data):
            indices = tf.random.uniform(
                shape=[num_ts],
                minval=0,
                maxval=len(self.data),
                dtype=tf.int32
            )
            initial_samples = tf.cast(
            tf.gather(self.data, indices),
            tf.float32
        )
        else:
            initial_samples = self.data

        _, _, samples = self.gdf_util.q_sample(initial_samples, tf.fill([num_ts], self.timesteps-1))

        for i in reversed(range(0, self.timesteps)):
            tt = tf.fill([num_ts], i)
            pred_noise = self.ema_network.predict([samples, tt], verbose=0, batch_size=num_ts
            )
            samples = self.gdf_util.p_sample(pred_noise, samples, tt, clip_denoised=False
            )

        return samples

