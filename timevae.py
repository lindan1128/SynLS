import os, warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import time
import numpy as np
from model.utils import TSFeatureWiseScaler
from model.timevae import TimeVAE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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

    new_num = args.new_num if args.new_num is not None else data.shape[0]

    early_stopping = EarlyStopping(
        monitor='loss',
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    model = TimeVAE(
        seq_len=data.shape[1],
        feat_dim=data.shape[2],
        latent_dim=16,
        hidden_layer_sizes=[32, 64],
        reconstruction_wt=3.0,
        use_residual_conn=True
    )

    print('Training model...')
    print(f'Early stopping patience: {args.patience}, min_delta: {args.min_delta}')
    start_time = time.time()
    model.compile(optimizer=Adam())
    history = model.fit(
        data,
        batch_size=args.batch_size,
        epochs=args.epoch,
        shuffle=True,
        verbose=1,
        callbacks=[early_stopping]
    )
    end_time = time.time()
    training_duration = end_time - start_time

    stopped_epoch = early_stopping.stopped_epoch
    if stopped_epoch > 0:
        print(f'\nEarly stopping triggered at epoch {stopped_epoch + 1}')
    print(f'Training completed in {training_duration:.2f} seconds')

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
    np.save(args.output, generated_samples)
    print('Generated data shape:', generated_samples.shape)
    print(f'Saving generated samples to {args.output}')

if __name__ == "__main__":
    main()
