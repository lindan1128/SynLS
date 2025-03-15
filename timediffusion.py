import os, warnings
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

from model.utils import TSFeatureScaler
from model.diffusion import build_model, GaussianDiffusion, DiffusionModel

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disabling gpu usage because my cuda is corrupted, needs to be fixed.

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        required=True,
                        type=str,
                        help='Path to the data file. Please provide the absolute path.')
    parser.add_argument('--step',
                        type=int,
                        default=10,
                        help='diffusion steps')
    parser.add_argument('--epoch',
                        required=True,
                        type=int,
                        help='The number of training epoch.')
    parser.add_argument('--batch_size',
                        required=True,
                        type=int,
                        help='Training batch size.')
    parser.add_argument('--new_num',
                        type=int,
                        help='The number of generating new samples. If not specified, will use the same number as input data.')
    parser.add_argument('--encoder_type',
                        type=str,
                        default='time',
                        choices=['time', 'pairwise', 'dual'],
                        help='Encoder type: time (Time Transformer only), pairwise (Pairwise Correlation only), or dual (both)')
    parser.add_argument('--output',
                        type=str,
                        default='new_samples.npy',
                        help='Output file name for generated samples (default: new_samples.npy)')
    args = parser.parse_args()

    path = args.path
    step = int(args.step)
    epoch = args.epoch
    batch_size = args.batch_size
    output_file = args.output

    print('Loading data...')
    scaler = TSFeatureScaler()
    data = np.load(path)
    data = scaler.fit_transform(data)
    print('data shape:', data.shape)

    print("Checking for invalid values:")
    print(f"NaN values: {np.isnan(data).any()}")
    print(f"Inf values: {np.isinf(data).any()}")

    print('Building model...')
    network = build_model(
        time_len=data.shape[1], 
        fea_num=data.shape[2],
        d_model=16,
        n_heads=2,
        encoder_type=args.encoder_type
    )
    ema_network = build_model(
        time_len=data.shape[1], 
        fea_num=data.shape[2],
        d_model=16,
        n_heads=2,
        encoder_type=args.encoder_type
    )
    ema_network.set_weights(network.get_weights())  
    noise_util = GaussianDiffusion(timesteps=step)
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
    metrics=['mse'] 
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./cp.ckpt',
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5
        )
    ]

    print('Fitting model...')
    model.fit(
    data,
    epochs=epoch,
    batch_size=batch_size,
    callbacks=callbacks
    )

    print('Generating samples...')
    new_num = args.new_num if args.new_num is not None else data.shape[0]
    new_samples = model.generate_ts(new_num).numpy()
    new_samples = scaler.fit_transform(new_samples)
    np.save(output_file, new_samples)
    print('new data shape:', new_samples.shape)
    print(f'Saving generated samples to {args.output}')

if __name__ == "__main__":
    main()