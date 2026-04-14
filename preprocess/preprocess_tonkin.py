"""
Preprocess Tonkin et al. dataset (SPHERE)
Tri-axial accelerometer (20Hz) from wrist-worn sensors on 10 participants.
Features: x, y, z (3 features)
Window size: 60 points (3.0 seconds at 20Hz)
Activities: bent, kneel, lie, sit, squat, stand
Train/Test split: samples 00001-00010 (train), 00011-00019 (test).

Pipeline:
  1. Split raw acceleration data into per-activity CSVs using activity labels
  2. Find continuous segments (max gap <= 1s)
  3. Slice segments into non-overlapping windows of 60 points
  4. Save train/test .npy files per activity
"""

import numpy as np
import pandas as pd
import os
import argparse
from glob import glob


def step1_split_by_activity(data_dir):
    """Split raw accelerometer CSVs into per-activity CSVs."""
    acc_files = glob(f'{data_dir}/*_acceleration.csv')

    for acc_file in acc_files:
        activity_file = acc_file.replace('_acceleration.csv', '_activity.csv')
        if not os.path.exists(activity_file):
            print(f'Warning: No activity file for {acc_file}')
            continue

        acc_df = pd.read_csv(acc_file)
        activity_df = pd.read_csv(activity_file)

        actions = ['p_bent', 'p_kneel', 'p_lie', 'p_sit', 'p_squat', 'p_stand']

        for action in actions:
            action_seconds = activity_df[activity_df[action] != 0]['t']
            if len(action_seconds) == 0:
                continue

            action_data = []
            for second in action_seconds:
                second_data = acc_df[(acc_df['t'] >= second) & (acc_df['t'] < second + 1)]
                if len(second_data) > 0:
                    action_data.append(second_data)

            if action_data:
                action_data = pd.concat(action_data, axis=0)
                action_name = action[2:]  # remove 'p_' prefix
                output_file = acc_file.replace('_acceleration.csv', f'_{action_name}.csv')
                action_data.to_csv(output_file, index=False)

        print(f'  Split {os.path.basename(acc_file)}')


def find_continuous_segments(df, max_gap=1.0, min_length=60):
    """Find continuous time segments with gap <= max_gap seconds."""
    segments = []
    current_segment = [df.index[0]]

    for i in range(1, len(df)):
        time_gap = df.iloc[i]['t'] - df.iloc[i - 1]['t']
        if time_gap <= max_gap:
            current_segment.append(df.index[i])
        else:
            if len(current_segment) >= min_length:
                segments.append(current_segment)
            current_segment = [df.index[i]]

    if len(current_segment) >= min_length:
        segments.append(current_segment)

    return segments


def segment_timeseries(data, window_size=60):
    """Slice data into non-overlapping windows."""
    n_windows = len(data) // window_size
    windows = []
    for i in range(n_windows):
        start = i * window_size
        windows.append(data[start:start + window_size])
    return np.array(windows) if windows else np.array([])


def step2_create_npy(out_dir, split_dir, split_name, window_size=60):
    """Create .npy files from per-activity CSVs for a given train/test split."""
    actions = ['bent', 'kneel', 'lie', 'sit', 'squat', 'stand']

    for action in actions:
        pattern = f'{split_dir}/*_{action}.csv'
        files = glob(pattern)

        if not files:
            print(f'  No files for {action} ({split_name})')
            continue

        all_windows = []
        for file in files:
            df = pd.read_csv(file)
            segments = find_continuous_segments(df, max_gap=1.0, min_length=window_size)

            for segment in segments:
                segment_data = df.iloc[segment][['x', 'y', 'z']].values
                windows = segment_timeseries(segment_data, window_size=window_size)
                if len(windows) > 0:
                    all_windows.append(windows)

        if all_windows:
            all_windows = np.vstack(all_windows)
            output_file = f'{out_dir}/tonkin_{action}_{split_name}.npy'
            np.save(output_file, all_windows)
            print(f'  Saved {output_file}, shape: {all_windows.shape}')
        else:
            print(f'  No valid windows for {action} ({split_name})')


def main():
    parser = argparse.ArgumentParser(description='Preprocess Tonkin et al. dataset')
    parser.add_argument('--data_dir', type=str,
                        default='/Users/lindan/Dropbox/PhD/Projects/PLF/GAN/Data/tonkin',
                        help='Root data directory for Tonkin dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for .npy files (default: same as data_dir)')
    parser.add_argument('--window_size', type=int, default=60,
                        help='Window size in data points (default: 60)')
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.output_dir if args.output_dir else data_dir
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Split raw data by activity
    print('Step 1: Splitting accelerometer data by activity...')
    step1_split_by_activity(data_dir)

    # Step 2: Create .npy files from all per-activity CSVs (no train/test split)
    print('\nStep 2: Creating .npy files (all data)...')
    step2_create_npy(out_dir, data_dir, 'all', window_size=args.window_size)

    print('\nDone.')


if __name__ == '__main__':
    main()
