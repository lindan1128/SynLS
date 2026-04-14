import os, warnings
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from GAN.utils import TSFeatureScaler
from GAN.diffusion import build_model, GaussianDiffusion, DiffusionModel

import time

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
                        default='dual',
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

    start_time = time.perf_counter()

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
    print('new data shape:', new_samples.shape)
    print(f'Saving generated samples to {args.output}')
    np.save(args.output, new_samples)
    new_samples = new_samples + 1

    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time} seconds")

    print('Generating comparison plot...')
    data = data + 1
    real_mean = np.mean(data, axis=0)
    real_std = np.std(data, axis=0)
    gen_mean = np.mean(new_samples, axis=0)
    gen_std = np.std(new_samples, axis=0)

    time_steps = np.arange(data.shape[1])
    n_features = data.shape[2]

    real_color = '#808080'
    gen_color = '#98749E'

    fig, axes = plt.subplots(n_features, 1, figsize=(5, 1.5*n_features))
    if n_features == 1:
        axes = [axes]

    for feature_idx, ax in enumerate(axes):
        ax.plot(time_steps, real_mean[:, feature_idx],
                color=real_color, linewidth=0.8, label='Real')
        ax.fill_between(time_steps,
                       real_mean[:, feature_idx] - real_std[:, feature_idx],
                       real_mean[:, feature_idx] + real_std[:, feature_idx],
                       color=real_color, alpha=0.2)

        ax.plot(time_steps, gen_mean[:, feature_idx],
                color=gen_color, linewidth=0.8, label='Synthetic')
        ax.fill_between(time_steps,
                       gen_mean[:, feature_idx] - gen_std[:, feature_idx],
                       gen_mean[:, feature_idx] + gen_std[:, feature_idx],
                       color=gen_color, alpha=0.2)

        ax.set_ylabel(f'Feature {feature_idx+1}', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time Steps', fontsize=8)
    plt.tight_layout()
    plot_path = args.output.replace('.npy', '_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {plot_path}")
    
    print("\nStatistical Comparison:")
    for feature_idx in range(n_features):
        print(f"\nFeature {feature_idx+1}:")
        print(f"Real data - Mean: {real_mean[:, feature_idx].mean():.3f}, "
              f"Std: {real_std[:, feature_idx].mean():.3f}")
        print(f"Generated data - Mean: {gen_mean[:, feature_idx].mean():.3f}, "
              f"Std: {gen_std[:, feature_idx].mean():.3f}")

if __name__ == "__main__":
    main()