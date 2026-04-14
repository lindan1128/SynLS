"""
Preprocess Versluijs et al. dataset (demo6)
Tri-axial accelerometer (10Hz) from 38 Beef Cows via Nofence virtual fence collar.
Features: xcal, ycal, zcal (3 features)
Window size: 50 points (5.0 seconds at 10Hz)
Behaviors: Resting, Rumination, Moving, Grazing, Others
Output: per-behavior .npy arrays of shape (N, 50, 3)
"""

import os
import numpy as np
import pandas as pd
import argparse


def process_cow_group(group, window_size=50, step_size=10):
    """Extract behavior-specific windows from a single cow's data."""
    behavior_cols = {
        'resting': ('Res', ['Rum', 'Mov', 'Gra']),
        'rumination': ('Rum', ['Res', 'Mov', 'Gra']),
        'moving': ('Mov', ['Res', 'Rum', 'Gra']),
        'grazing': ('Gra', ['Res', 'Rum', 'Mov']),
        'others': ('other_be', ['Res', 'Rum', 'Mov', 'Gra']),
    }

    results = {k: [] for k in behavior_cols}
    acc_cols = ['xcal', 'ycal', 'zcal']

    for start in range(0, len(group) - window_size + 1, step_size):
        window = group.iloc[start:start + window_size]
        for behavior, (pos_col, neg_cols) in behavior_cols.items():
            if window[pos_col].eq(1).any() and window[neg_cols].isna().all().all():
                results[behavior].append(window[acc_cols].values)

    return {k: np.array(v) if v else np.empty((0, window_size, 3)) for k, v in results.items()}


def apply_moving_average(arrays, window_size=3):
    """Apply rolling mean smoothing."""
    n, m, f = arrays.shape
    smoothed = np.zeros_like(arrays)
    for i in range(n):
        for j in range(f):
            series = pd.Series(arrays[i, :, j])
            smoothed[i, :, j] = series.rolling(window=window_size, min_periods=1).mean().values
    return smoothed


def main():
    parser = argparse.ArgumentParser(description='Preprocess Versluijs et al. dataset')
    parser.add_argument('--input', type=str,
                        default='/Users/lindan/Dropbox/PhD/Projects/PLF/GAN/Data/demo6/dataset.csv',
                        help='Path to raw dataset CSV')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/lindan/Dropbox/PhD/Projects/PLF/GAN/Data/demo6',
                        help='Output directory for .npy files')
    args = parser.parse_args()

    # Load data
    print('Loading data...')
    df = pd.read_csv(args.input, low_memory=False)

    # Process per cow
    print('Extracting behavior windows (window=30, step=10)...')
    all_behaviors = {k: [] for k in ['resting', 'rumination', 'moving', 'grazing', 'others']}

    for serial, group in df.groupby('serial'):
        results = process_cow_group(group)
        for behavior, data in results.items():
            if data.shape[0] > 0:
                all_behaviors[behavior].append(data)

    # Concatenate
    for behavior in all_behaviors:
        if all_behaviors[behavior]:
            all_behaviors[behavior] = np.concatenate(all_behaviors[behavior], axis=0)
        else:
            all_behaviors[behavior] = np.empty((0, 50, 3))

    for behavior, data in all_behaviors.items():
        print(f'  {behavior}: {data.shape}')

    # Apply moving average and save
    os.makedirs(args.output_dir, exist_ok=True)
    print('Applying moving average smoothing (window=3) and saving...')
    for behavior, data in all_behaviors.items():
        if data.shape[0] > 0:
            smoothed = apply_moving_average(data)
        else:
            smoothed = data
        output_path = f'{args.output_dir}/{behavior}.npy'
        np.save(output_path, smoothed)
        print(f'  Saved {output_path}')

    print('Done.')


if __name__ == '__main__':
    main()
