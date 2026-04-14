"""
Preprocess D4 dataset (Wanger_2 et al. - INRAE Herbipole, Dec 2014 - Dec 2015)
High-resolution hourly data from 300 Holstein-Friesian cows.
Data directory: demo3
Features: IN_ALLEYS, REST, EAT (3 features)
Sequence length: 96 hours
Health events: Calving, Oestrus (Estrus), Disease (LAME/MAST/Other_disease)
Output: health/calving/estrus/disease arrays of shape (N, 96, 3)

Note: Same format as D3 but different raw data file (dataset4.tab)
      and requires hour handling (Hour 24 -> 0, date shifted).
"""

import os
import numpy as np
import pandas as pd
import argparse


def expand_labels_hourly(df, labels):
    """Expand event labels using rolling windows and remove post-event hours."""
    for label in labels:
        for id_value in df['ID'].unique():
            cow_data = df[df['ID'] == id_value].sort_values(by='DateTime')
            cow_data['temp'] = cow_data[label].rolling(12, min_periods=1).sum()
            df.loc[cow_data.index, 'temp'] = cow_data['temp']

            positive_datetimes = cow_data[cow_data['temp'] == 12]['DateTime']
            if positive_datetimes.empty:
                continue

            if label != 'Oestrus':
                start_time = positive_datetimes.tolist()[0] - pd.Timedelta(hours=12) - pd.Timedelta(hours=48)
                end_time = positive_datetimes.tolist()[-1] + pd.Timedelta(hours=24)
                df.loc[(df['ID'] == id_value) &
                       (df['DateTime'] > start_time) &
                       (df['DateTime'] <= end_time), label] = 1

                delete_start = positive_datetimes.tolist()[-1] + pd.Timedelta(hours=24)
                delete_end = delete_start + pd.Timedelta(hours=144)
                df = df.drop(df[(df['ID'] == id_value) &
                                (df['DateTime'] > delete_start) &
                                (df['DateTime'] <= delete_end)].index)
            else:
                start_time = positive_datetimes.tolist()[0] - pd.Timedelta(hours=12) - pd.Timedelta(hours=24)
                end_time = positive_datetimes.tolist()[-1] + pd.Timedelta(hours=24)
                df.loc[(df['ID'] == id_value) &
                       (df['DateTime'] > start_time) &
                       (df['DateTime'] <= end_time), label] = 1

                # Post-event exclusion: 48h for Estrus (per SI)
                delete_start = positive_datetimes.tolist()[-1] + pd.Timedelta(hours=24)
                delete_end = delete_start + pd.Timedelta(hours=48)
                df = df.drop(df[(df['ID'] == id_value) &
                                (df['DateTime'] > delete_start) &
                                (df['DateTime'] <= delete_end)].index)

    return df


def find_sequences_hourly(df, condition, event_columns, features, seq_len=96):
    """Extract time series sequences for a given condition (hourly data)."""
    sequences = []

    for _, group in df.groupby('ID'):
        group = group.sort_values('DateTime').reset_index(drop=True)
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

        else:  # disease
            disease_cols = event_columns[2:]
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
    parser = argparse.ArgumentParser(description='Preprocess D4 dataset (Wanger_2 et al.)')
    parser.add_argument('--input', type=str,
                        default='/Users/lindan/Dropbox/PhD/Projects/PLF/GAN/Data/demo3/dataset4.tab',
                        help='Path to raw dataset')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/lindan/Dropbox/PhD/Projects/PLF/GAN/Data/demo3',
                        help='Output directory for .npy files')
    parser.add_argument('--health_sample_size', type=int, default=10000,
                        help='Number of health sequences to subsample')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load data
    print('Loading data...')
    df = pd.read_csv(args.input, sep='\t')
    df['Hour'] = df['Hour'].replace(24, 0)
    df['Date'] = pd.to_datetime(df['Date'])
    df.loc[df['Hour'] == 0, 'Date'] = df.loc[df['Hour'] == 0, 'Date'] + pd.Timedelta(days=1)
    df['DateTime'] = pd.to_datetime(
        df['Date'].dt.strftime('%Y-%m-%d ') + df['Hour'].astype(str) + ':00:00')

    labels = ['Calving', 'Oestrus', 'MAST', 'LAME', 'Other_disease']
    event_columns = labels
    features = ['IN_ALLEYS', 'REST', 'EAT']

    # Expand labels
    print('Expanding labels...')
    df = expand_labels_hourly(df, labels)

    # Replace 0 with NaN for event columns
    df[event_columns] = df[event_columns].replace(0, np.nan)

    # Extract sequences (96h)
    print('Extracting 96h sequences...')
    health = find_sequences_hourly(df, 'health', event_columns, features, seq_len=96)
    calving = find_sequences_hourly(df, 'Calving', event_columns, features, seq_len=96)
    estrus = find_sequences_hourly(df, 'Oestrus', event_columns, features, seq_len=96)
    disease = find_sequences_hourly(df, 'disease', event_columns, features, seq_len=96)

    print(f'  health: {health.shape}, calving: {calving.shape}, '
          f'estrus: {estrus.shape}, disease: {disease.shape}')

    # Subsample health
    if health.shape[0] > args.health_sample_size:
        indices = np.random.choice(health.shape[0], args.health_sample_size, replace=False)
        health = health[indices]
        print(f'  health subsampled to: {health.shape}')

    # Apply moving average
    print('Applying moving average smoothing (window=3)...')
    health = apply_moving_average(health)
    calving = apply_moving_average(calving)
    estrus = apply_moving_average(estrus)
    disease = apply_moving_average(disease)

    # Save 96h arrays
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Saving 96h arrays to {args.output_dir}/')
    np.save(f'{args.output_dir}/health_arrays_corrected_96.npy', health)
    np.save(f'{args.output_dir}/calving_arrays_corrected_96.npy', calving)
    np.save(f'{args.output_dir}/abort_arrays_corrected_96.npy', estrus)
    np.save(f'{args.output_dir}/disease_arrays_corrected_96.npy', disease)
    print('Done.')


if __name__ == '__main__':
    main()
