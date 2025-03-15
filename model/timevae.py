import os, warnings
warnings.filterwarnings('ignore')

from abc import ABC, abstractmethod
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D,  Flatten, Dense, Conv1DTranspose, Reshape, Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.backend import random_normal

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class BaseVariationalAutoencoder(Model, ABC):
    def __init__(self,
                 seq_len,
                 feat_dim,
                 latent_dim,
                 reconstruction_wt=3.0,
                 **kwargs):
        super(BaseVariationalAutoencoder, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

        self.encoder = None
        self.decoder = None

    def call(self, X):
        z_mean, _, _ = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        if len(x_decoded.shape) == 1: x_decoded = x_decoded.reshape((1, -1))
        return x_decoded

    def get_num_trainable_variables(self):
        trainableParams = int(np.sum([np.prod(v.get_shape()) for v in self.trainable_weights]))
        nonTrainableParams = int(np.sum([np.prod(v.get_shape()) for v in self.non_trainable_weights]))
        totalParams = trainableParams + nonTrainableParams
        return trainableParams, nonTrainableParams, totalParams

    def get_prior_samples(self, num_samples):
        Z = np.random.randn(num_samples, self.latent_dim)
        samples = self.decoder.predict(Z)
        return samples

    def get_prior_samples_given_Z(self, Z):
        samples = self.decoder.predict(Z)
        return samples

    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def _get_reconstruction_loss(self, X, X_recons):
        def get_reconst_loss_by_axis(X, X_c, axis):
            x_r = tf.reduce_mean(X, axis=axis)
            x_c_r = tf.reduce_mean(X_recons, axis=axis)
            err = tf.math.squared_difference(x_r, x_c_r)
            loss = tf.reduce_sum(err)
            return loss

        # overall
        err = tf.math.squared_difference(X, X_recons)
        reconst_loss = tf.reduce_sum(err)

        reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[2])  # by time axis
        # reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[1])    # by feature axis
        return reconst_loss

    def train_step(self, X):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(X)

            reconstruction = self.decoder(z)

            reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
            # kl_loss = kl_loss / self.latent_dim

            total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, X):
        z_mean, z_log_var, z = self.encoder(X)
        reconstruction = self.decoder(z)
        reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
        # kl_loss = kl_loss / self.latent_dim

        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def save_weights(self, model_dir, file_pref):
        encoder_wts = self.encoder.get_weights()
        decoder_wts = self.decoder.get_weights()
        joblib.dump(encoder_wts, os.path.join(model_dir, f'{file_pref}encoder_wts.h5'))
        joblib.dump(decoder_wts, os.path.join(model_dir, f'{file_pref}decoder_wts.h5'))

    def load_weights(self, model_dir, file_pref):
        encoder_wts = joblib.load(os.path.join(model_dir, f'{file_pref}encoder_wts.h5'))
        decoder_wts = joblib.load(os.path.join(model_dir, f'{file_pref}decoder_wts.h5'))

        self.encoder.set_weights(encoder_wts)
        self.decoder.set_weights(decoder_wts)

    def save(self, model_dir, file_pref):
        self.save_weights(model_dir, file_pref)
        dict_params = {

            'seq_len': self.seq_len,
            'feat_dim': self.feat_dim,
            'latent_dim': self.latent_dim,
            'reconstruction_wt': self.reconstruction_wt,
            'hidden_layer_sizes': self.hidden_layer_sizes,
        }
        params_file = os.path.join(model_dir, f'{file_pref}parameters.pkl')
        joblib.dump(dict_params, params_file)

class TimeVAE(BaseVariationalAutoencoder):

    def __init__(self, hidden_layer_sizes, trend_poly=0, num_gen_seas=0, custom_seas=None,
                 use_scaler=False, use_residual_conn=True, **kwargs):
        '''
            hidden_layer_sizes: list of number of filters in convolutional layers in encoder and residual connection of decoder.
            trend_poly: integer for number of orders for trend component. e.g. setting trend_poly = 2 will include linear and quadratic term.
            num_gen_seas: Number of sine-waves to use to model seasonalities. Each sine wae will have its own amplitude, frequency and phase.
            custom_seas: list of tuples of (num_seasons, len_per_season).
                num_seasons: number of seasons per cycle.
                len_per_season: number of epochs (time-steps) per season.
            use_residual_conn: boolean value indicating whether to use a residual connection for reconstruction in addition to
            trend, generic and custom seasonalities.
        '''

        super(TimeVAE, self).__init__(**kwargs)

        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = trend_poly
        self.num_gen_seas = num_gen_seas
        self.custom_seas = custom_seas
        self.use_scaler = use_scaler
        self.use_residual_conn = use_residual_conn
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

    def _get_encoder(self):
        encoder_inputs = Input(shape=(self.seq_len, self.feat_dim), name='encoder_input')
        x = encoder_inputs
        for i, num_filters in enumerate(self.hidden_layer_sizes):
            x = Conv1D(
                filters=num_filters,
                kernel_size=3,
                strides=2,
                activation='relu',
                padding='same',
                name=f'enc_conv_{i}')(x)

        x = Flatten(name='enc_flatten')(x)

        # save the dimensionality of this last dense layer before the hidden state layer. We need it in the decoder.
        self.encoder_last_dense_dim = x.get_shape()[-1]

        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)

        encoder_output = Sampling()([z_mean, z_log_var])
        self.encoder_output = encoder_output

        encoder = Model(encoder_inputs, [z_mean, z_log_var, encoder_output], name="encoder")
        return encoder

    def _get_decoder(self):
        decoder_inputs = Input(shape=(int(self.latent_dim)), name='decoder_input')

        outputs = None
        outputs = self.level_model(decoder_inputs)

        # trend polynomials
        if self.trend_poly is not None and self.trend_poly > 0:
            trend_vals = self.trend_model(decoder_inputs)
            outputs = trend_vals if outputs is None else outputs + trend_vals

            # # generic seasonalities
        # if self.num_gen_seas is not None and self.num_gen_seas > 0:
        #     gen_seas_vals, freq, phase, amplitude = self.generic_seasonal_model(decoder_inputs)
        #     # gen_seas_vals = self.generic_seasonal_model2(decoder_inputs)
        #     outputs = gen_seas_vals if outputs is None else outputs + gen_seas_vals

        # custom seasons
        if self.custom_seas is not None and len(self.custom_seas) > 0:
            cust_seas_vals = self.custom_seasonal_model(decoder_inputs)
            outputs = cust_seas_vals if outputs is None else outputs + cust_seas_vals

        if self.use_residual_conn:
            residuals = self._get_decoder_residual(decoder_inputs)
            outputs = residuals if outputs is None else outputs + residuals

        if self.use_scaler and outputs is not None:
            scale = self.scale_model(decoder_inputs)
            outputs *= scale

        # outputs = Activation(activation='sigmoid')(outputs)

        if outputs is None:
            raise Exception('''Error: No decoder model to use. 
            You must use one or more of:
            trend, generic seasonality(ies), custom seasonality(ies), and/or residual connection. ''')

        decoder = Model(decoder_inputs, [outputs], name="decoder")
        return decoder

    def level_model(self, z):
        level_params = Dense(self.feat_dim, name="level_params", activation='relu')(z)
        level_params = Dense(self.feat_dim, name="level_params2")(level_params)
        level_params = Reshape(target_shape=(1, self.feat_dim))(level_params)  # shape: (N, 1, D)

        ones_tensor = tf.ones(shape=[1, self.seq_len, 1], dtype=tf.float32)  # shape: (1, T, D)

        level_vals = level_params * ones_tensor
        return level_vals

    def scale_model(self, z):
        scale_params = Dense(self.feat_dim, name="scale_params", activation='relu')(z)
        scale_params = Dense(self.feat_dim, name="scale_params2")(scale_params)
        scale_params = Reshape(target_shape=(1, self.feat_dim))(scale_params)  # shape: (N, 1, D)

        scale_vals = tf.repeat(scale_params, repeats=self.seq_len, axis=1)  # shape: (N, T, D)
        return scale_vals

    def trend_model(self, z):
        trend_params = Dense(self.feat_dim * self.trend_poly, name="trend_params", activation='relu')(z)
        trend_params = Dense(self.feat_dim * self.trend_poly, name="trend_params2")(trend_params)
        trend_params = Reshape(target_shape=(self.feat_dim, self.trend_poly))(trend_params)  # shape: N x D x P

        lin_space = K.arange(0, float(self.seq_len), 1) / self.seq_len  # shape of lin_space : 1d tensor of length T
        poly_space = K.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], axis=0)  # shape: P x T

        trend_vals = K.dot(trend_params, poly_space)  # shape (N, D, T)
        trend_vals = tf.transpose(trend_vals, perm=[0, 2, 1])  # shape: (N, T, D)
        trend_vals = K.cast(trend_vals, tf.float32)
        return trend_vals

    def custom_seasonal_model(self, z):

        N = tf.shape(z)[0]
        ones_tensor = tf.ones(shape=[N, self.feat_dim, self.seq_len], dtype=tf.int32)

        all_seas_vals = []
        for i, season_tup in enumerate(self.custom_seas):
            num_seasons, len_per_season = season_tup

            season_params = Dense(self.feat_dim * num_seasons, name=f"season_params_{i}")(z)  # shape: (N, D * S)
            season_params = Reshape(target_shape=(self.feat_dim, num_seasons))(season_params)  # shape: (N, D, S)

            season_indexes_over_time = self._get_season_indexes_over_seq(num_seasons, len_per_season)  # shape: (T, )

            dim2_idxes = ones_tensor * tf.reshape(season_indexes_over_time, shape=(1, 1, -1))  # shape: (1, 1, T)

            season_vals = tf.gather(season_params, dim2_idxes, batch_dims=-1)  # shape (N, D, T)

            all_seas_vals.append(season_vals)

        all_seas_vals = K.stack(all_seas_vals, axis=-1)  # shape: (N, D, T, S)
        all_seas_vals = tf.reduce_sum(all_seas_vals, axis=-1)  # shape (N, D, T)
        all_seas_vals = tf.transpose(all_seas_vals, perm=[0, 2, 1])  # shape (N, T, D)
        return all_seas_vals

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        curr_len = 0
        season_idx = []
        curr_idx = 0
        while curr_len < self.seq_len:
            reps = len_per_season if curr_len + len_per_season <= self.seq_len else self.seq_len - curr_len
            season_idx.extend([curr_idx] * reps)
            curr_idx += 1
            if curr_idx == num_seasons: curr_idx = 0
            curr_len += reps
        return season_idx

    def generic_seasonal_model(self, z):

        freq = Dense(self.feat_dim * self.num_gen_seas, name="g_season_freq", activation='sigmoid')(z)
        freq = Reshape(target_shape=(1, self.feat_dim, self.num_gen_seas))(freq)  # shape: (N, 1, D, S)

        phase = Dense(self.feat_dim * self.num_gen_seas, name="g_season_phase")(z)
        phase = Reshape(target_shape=(1, self.feat_dim, self.num_gen_seas))(phase)  # shape: (N, 1, D, S)

        amplitude = Dense(self.feat_dim * self.num_gen_seas, name="g_season_amplitude")(z)
        amplitude = Reshape(target_shape=(1, self.feat_dim, self.num_gen_seas))(amplitude)  # shape: (N, 1, D, S)

        lin_space = K.arange(0, float(self.seq_len), 1) / self.seq_len  # shape of lin_space : 1d tensor of length T
        lin_space = tf.reshape(lin_space, shape=(1, self.seq_len, 1, 1))  # shape: 1, T, 1, 1

        seas_vals = amplitude * K.sin(2. * np.pi * freq * lin_space + phase)  # shape: N, T, D, S
        seas_vals = tf.math.reduce_sum(seas_vals, axis=-1)  # shape: N, T, D

        return seas_vals

    def generic_seasonal_model2(self, z):

        season_params = Dense(self.feat_dim * self.num_gen_seas, name="g_season_params")(z)
        season_params = Reshape(target_shape=(self.feat_dim, self.num_gen_seas))(season_params)  # shape: (D, S)

        p = self.num_gen_seas
        p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)

        ls = K.arange(0, float(self.seq_len), 1) / self.seq_len  # shape of ls : 1d tensor of length T

        s1 = K.stack([K.cos(2 * np.pi * i * ls) for i in range(p1)], axis=0)
        s2 = K.stack([K.sin(2 * np.pi * i * ls) for i in range(p2)], axis=0)
        if p == 1:
            s = s2
        else:
            s = K.concatenate([s1, s2], axis=0)
        s = K.cast(s, np.float32)

        seas_vals = K.dot(season_params, s, name='g_seasonal_vals')
        seas_vals = tf.transpose(seas_vals, perm=[0, 2, 1])  # shape: (N, T, D)
        seas_vals = K.cast(seas_vals, np.float32)
        print('seas_vals shape', tf.shape(seas_vals))

        return seas_vals

    def _get_decoder_residual(self, x):

        x = Dense(self.encoder_last_dense_dim, name="dec_dense", activation='relu')(x)
        x = Reshape(target_shape=(-1, self.hidden_layer_sizes[-1]), name="dec_reshape")(x)

        for i, num_filters in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            x = Conv1DTranspose(
                filters=num_filters,
                kernel_size=3,
                strides=2,
                padding='same',
                activation='relu',
                name=f'dec_deconv_{i}')(x)

        # last de-convolution
        x = Conv1DTranspose(
            filters=self.feat_dim,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            name=f'dec_deconv__{i + 1}')(x)

        x = Flatten(name='dec_flatten')(x)
        x = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final")(x)
        residuals = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)
        return residuals

    def save(self, model_dir, file_pref):

        super().save_weights(model_dir, file_pref)
        dict_params = {
            'seq_len': self.seq_len,
            'feat_dim': self.feat_dim,
            'latent_dim': self.latent_dim,
            'reconstruction_wt': self.reconstruction_wt,

            'hidden_layer_sizes': self.hidden_layer_sizes,
            'trend_poly': self.trend_poly,
            'num_gen_seas': self.num_gen_seas,
            'custom_seas': self.custom_seas,
            'use_scaler': self.use_scaler,
            'use_residual_conn': self.use_residual_conn,
        }
        params_file = os.path.join(model_dir, f'{file_pref}parameters.pkl')
        joblib.dump(dict_params, params_file)

    @staticmethod
    def load(model_dir, file_pref):
        params_file = os.path.join(model_dir, f'{file_pref}parameters.pkl')
        dict_params = joblib.load(params_file)

        vae_model = TimeVAE(**dict_params)

        vae_model.load_weights(model_dir, file_pref)

        vae_model.compile(optimizer=Adam())

        return vae_model