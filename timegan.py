import argparse
import os, warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
import numpy as np
from model.utils import TSFeatureWiseScaler
from model.timegan import TimeGAN
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
                    default='timegan_samples.npy',
                    help='Output file name for generated samples (default: timegan_samples.npy)')
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

    model = TimeGAN(
        seq_len=data.shape[1],
        module="gru",
        hidden_dim=24,
        n_features=data.shape[2],
        n_layers=3,
        batch_size=args.batch_size,
        gamma=1.0)

    print('Training model...')
    print(f'Early stopping patience: {args.patience}, min_delta: {args.min_delta}')
    start_time = time.time()
    model.compile()
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
    generated_samples = model.generate(n_samples=new_num)
    np.save(args.output, generated_samples)
    print('Generated data shape:', generated_samples.shape)
    print(f'Saving generated samples to {args.output}')

if __name__ == "__main__":
    main()

