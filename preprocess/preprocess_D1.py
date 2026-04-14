"""
Preprocess D1 dataset (Lin et al.)
Low-resolution daily activity and rumination data from 185 Holstein-Friesian cows.
Features: Activity, Rumination (2 features)
Sequence length: 14 days
Health events: Calving, Abort (Estrus), Disease (Met/Mast/DA/LAME)
Output: health/calving/abort/disease arrays of shape (N, 14, 2)
"""

import os
import numpy as np
import pandas as pd
import argparse


def expand_labels_and_remove_post_event(df, labels, event_columns):
    """Expand event labels to neighboring days and remove post-event days.

    Per SI Supplementary Methods 1:
      - Label 12 days before and 1 day after the event.
      - Post-event exclusion: 6 days for Disease/Calving, 2 days for Estrus.
    """
    estrus_labels = {'Abort'}  # Estrus in D1
    for label in labels:
        for id_value in df['ID'].unique():
            cow_data = df[df['ID'] == id_value]
            positive_dates = cow_data[cow_data[label] == 1]['Date']

            for date in positive_dates:
                # Mark 12 days before and 1 day after as positive
                mark_dates = [date + pd.Timedelta(days=d) for d in range(-12, 2)]
                df.loc[(df['ID'] == id_value) &
                       (df['Date'].isin(mark_dates)), label] = 1

                # Post-event exclusion: 2 days for Estrus, 6 days for Disease/Calving
                delete_start = date + pd.Timedelta(days=1)
                if label in estrus_labels:
                    delete_end = delete_start + pd.Timedelta(days=2)
                else:
                    delete_end = delete_start + pd.Timedelta(days=6)
                df = df.drop(df[(df['ID'] == id_value) &
                                (df['Date'] > delete_start) &
                                (df['Date'] <= delete_end)].index)
    return df


def find_sequences(df, condition, event_columns, features, seq_len=14):
    """Extract time series sequences for a given condition."""
    sequences = []

    for _, group in df.groupby('ID'):
        group = group.sort_values('Date').reset_index(drop=True)
        length = len(group)

        if condition == 'health':
            for i in range(0, length, seq_len):
                if (i + seq_len <= length and
                        group.iloc[i:i + seq_len][event_columns].isna().all(axis=1).all()):
                    sequences.append(group.iloc[i:i + seq_len][features].values)

        elif condition in event_columns:
            indices = group.index[group[condition] == 1].tolist()
            if indices:
                last_index = indices[-1]
                start_index = max(last_index - seq_len + 1, 0)
                if last_index - start_index == seq_len - 1:
                    sequences.append(group.iloc[start_index:last_index + 1][features].values)

        else:  # disease: any of the disease columns
            disease_cols = event_columns[2:]  # Met, Mast, DA, LAME
            condition_indices = group.index[group[disease_cols].notna().any(axis=1)].tolist()
            if condition_indices:
                last_index = condition_indices[-1]
                start_index = max(last_index - seq_len + 1, 0)
                if last_index - start_index == seq_len - 1:
                    sequences.append(group.iloc[start_index:last_index + 1][features].values)

    return np.array(sequences) if sequences else np.empty((0, seq_len, len(features)))


def apply_moving_average(arrays, window_size=3):
    """Apply rolling mean smoothing to time series arrays."""
    n, m, f = arrays.shape
    smoothed = np.zeros_like(arrays)
    for i in range(n):
        for j in range(f):
            series = pd.Series(arrays[i, :, j])
            smoothed[i, :, j] = series.rolling(window=window_size, min_periods=1).mean().values
    return smoothed


def main():
    parser = argparse.ArgumentParser(description='Preprocess D1 dataset')
    parser.add_argument('--input', type=str,
                        default='/Users/lindan/Dropbox/PhD/Projects/PLF/GAN/Data/demo1/dataset.csv',
                        help='Path to raw dataset CSV')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/lindan/Dropbox/PhD/Projects/PLF/GAN/Data/demo1',
                        help='Output directory for .npy files')
    parser.add_argument('--health_sample_size', type=int, default=3940,
                        help='Number of health sequences to subsample')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load data
    print('Loading data...')
    df = pd.read_csv(args.input)
    df['Date'] = pd.to_datetime(df['Date'])

    labels = ['Calving', 'Abort', 'Met', 'Mast', 'DA', 'LAME']
    event_columns = labels
    features = ['Activity', 'Rumination']

    # Expand labels and remove post-event days
    print('Expanding labels and removing post-event days...')
    df = expand_labels_and_remove_post_event(df, labels, event_columns)

    # Extract sequences
    print('Extracting sequences...')
    health = find_sequences(df, 'health', event_columns, features)
    calving = find_sequences(df, 'Calving', event_columns, features)
    abort = find_sequences(df, 'Abort', event_columns, features)
    disease = find_sequences(df, 'disease', event_columns, features)

    print(f'  health: {health.shape}, calving: {calving.shape}, '
          f'abort: {abort.shape}, disease: {disease.shape}')

    # Subsample health sequences
    if health.shape[0] > args.health_sample_size:
        indices = np.random.choice(health.shape[0], args.health_sample_size, replace=False)
        health = health[indices]
        print(f'  health subsampled to: {health.shape}')

    # Apply moving average smoothing
    print('Applying moving average smoothing (window=3)...')
    health = apply_moving_average(health)
    calving = apply_moving_average(calving)
    abort = apply_moving_average(abort)
    disease = apply_moving_average(disease)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Saving to {args.output_dir}/')
    np.save(f'{args.output_dir}/health_arrays_corrected.npy', health)
    np.save(f'{args.output_dir}/calving_arrays_corrected.npy', calving)
    np.save(f'{args.output_dir}/abort_arrays_corrected.npy', abort)
    np.save(f'{args.output_dir}/disease_arrays_corrected.npy', disease)
    print('Done.')


if __name__ == '__main__':
    main()
