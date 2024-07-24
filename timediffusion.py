#!/usr/bin/env python
# coding: utf-8

import os, warnings
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

from gen.utils import TSFeatureWiseScaler
from gen.diffusion import build_model, GaussianDiffusion, DiffusionModel

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disabling gpu usage because my cuda is corrupted, needs to be fixed.

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        required=True,
                        type=str,
                        help='Path to the data file. Please provide the absolute path.')
    parser.add_argument('--step',
                        required=True,
                        type=str,
                        help='Time step for diffusion.')
    parser.add_argument('--epoch',
                        required=True,
                        type=int,
                        help='The number of training epoch.')
    parser.add_argument('--batch_size',
                        required=True,
                        type=int,
                        help='Training batch size.')
    parser.add_argument('--new_num',
                        required=True,
                        type=int,
                        help='The number of generating new samples.')
    args = parser.parse_args()

    path = args.path
    step = args.step
    epoch = args.epoch
    batch_size = args.batch_size
    new_num = args.new_num

    print('Loading data...')
    scaler = TSFeatureWiseScaler()
    data = np.load(path)
    data = scaler.fit_transform(data)
    print('data shape:', data.shape)

    print('Building model...')
    network = build_model(time_len=data.shape[1], fea_num=data.shape[2])
    ema_network = build_model(time_len=data.shape[1], fea_num=data.shape[2])
    ema_network.set_weights(network.get_weights())  # Initially the weights are the same
    noise_util = GaussianDiffusion()
    model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    timesteps=step,
    gdf_util=noise_util,
    data=data
    )

    print('Compiling model...')
    model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    )

    print('Fitting model...')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./cp.ckpt',
                                                 save_weights_only=True,
                                                 verbose=1)
    model.fit(
    data,
    epochs=epoch,
    batch_size=batch_size,
    callbacks=[cp_callback]
    )

    print('Generating samples...')
    new_samples = model.generate_ts(new_num).numpy()
    new_samples = scaler.fit_transform(new_samples)
    print('new data shape:', new_samples.shape)
    np.save('new_samples.npy', new_samples)


if __name__ == "__main__":
    main()