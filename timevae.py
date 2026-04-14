import os, warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import time
import numpy as np
from GAN.utils import TSFeatureWiseScaler
from GAN.timevae import TimeVAE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                    required=True,
                    type=str,
                    help='Path to the data file. Please provide the absolute path.')
    parser.add_argument('--epoch',
                    required=True,
                    type=int,
                    help='The number of training epochs.')
    parser.add_argument('--batch_size',
                    required=True,
                    type=int,
                    help='Training batch size.')
    parser.add_argument('--new_num',
                    type=int,
                    help='The number of generating new samples. If not specified, will use the same number as input data.')
    parser.add_argument('--output',
                    type=str,
                    default='timevae_samples.npy',
                    help='Output file name for generated samples (default: timevae_samples.npy)')
    parser.add_argument('--patience',
                    type=int,
                    default=10,
                    help='Number of epochs with no improvement after which training will be stopped (default: 10)')
    parser.add_argument('--min_delta',
                    type=float,
                    default=1e-4,
                    help='Minimum change in loss to qualify as an improvement (default: 0.0001)')
    
    args = parser.parse_args()
    
    print('Loading data...')
    scaler = TSFeatureWiseScaler()
    data = np.load(args.path)
    data = scaler.fit_transform(data)
    print('data shape:', data.shape)

    new_num = args.new_num if args.new_num is not None else data.shape[0] * 5

    early_stopping = EarlyStopping(
        monitor='loss',
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    start_time = time.perf_counter()
    
    model = TimeVAE(
        seq_len=data.shape[1],
        feat_dim=data.shape[2],
        latent_dim=6,
        hidden_layer_sizes=[16, 16],
        reconstruction_wt=3.0,
        use_residual_conn=True
    )
    
    print('Training model...')
    model.compile(optimizer=Adam())
    history = model.fit(
        data,
        batch_size=args.batch_size,
        epochs=args.epoch,
        shuffle=True,
        verbose=1,
        callbacks=[early_stopping]
    )
    stopped_epoch = early_stopping.stopped_epoch
    if stopped_epoch > 0:
        print(f'\nEarly stopping triggered at epoch {stopped_epoch + 1}')

    print('Generating samples...')
    if new_num <= data.shape[0]:
        generated_samples = model.predict(data[:new_num])
    else:
        generated_samples = []
        remaining = new_num
        while remaining > 0:
            batch_size = min(remaining, data.shape[0])
            batch_samples = model.predict(data[:batch_size])
            generated_samples.append(batch_samples)
            remaining -= batch_size
        generated_samples = np.concatenate(generated_samples, axis=0)
    
    print('Generated data shape:', generated_samples.shape)
    
    print(f'Saving generated samples to {args.output}')
    np.save(args.output, generated_samples)

    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time} seconds")
    
    print('Generating comparison plot...')
    real_mean = np.mean(data, axis=0)
    real_std = np.std(data, axis=0)
    gen_mean = np.mean(generated_samples, axis=0)
    gen_std = np.std(generated_samples, axis=0)

    time_steps = np.arange(data.shape[1])
    n_features = data.shape[2]
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features))
    if n_features == 1:
        axes = [axes]
    
    for feature_idx, ax in enumerate(axes):
        ax.plot(time_steps, real_mean[:, feature_idx],
                'b-', label='Real Mean', linewidth=2)
        ax.fill_between(time_steps,
                       real_mean[:, feature_idx] - real_std[:, feature_idx],
                       real_mean[:, feature_idx] + real_std[:, feature_idx],
                       color='blue', alpha=0.2, label='Real ±1 SD')

        ax.plot(time_steps, gen_mean[:, feature_idx],
                'r--', label='Generated Mean', linewidth=2)
        ax.fill_between(time_steps,
                       gen_mean[:, feature_idx] - gen_std[:, feature_idx],
                       gen_mean[:, feature_idx] + gen_std[:, feature_idx],
                       color='red', alpha=0.2, label='Generated ±1 SD')
        
        ax.set_title(f'Feature {feature_idx+1}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
    print("Saved comparison plot to comparison_plot.png")
    
    print("\nStatistical Comparison:")
    for feature_idx in range(n_features):
        print(f"\nFeature {feature_idx+1}:")
        print(f"Real data - Mean: {real_mean[:, feature_idx].mean():.3f}, "
              f"Std: {real_std[:, feature_idx].mean():.3f}")
        print(f"Generated data - Mean: {gen_mean[:, feature_idx].mean():.3f}, "
              f"Std: {gen_std[:, feature_idx].mean():.3f}")
    
    print('Done!')

if __name__ == "__main__":
    main()
