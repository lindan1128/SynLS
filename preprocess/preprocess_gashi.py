"""
Preprocess Gashi et al. dataset (WEEE)
Tri-axial accelerometer (~52Hz) from Muse S headband on 17 human participants.
Features: acc_x, acc_y, acc_z (3 features)
Window size: 156 points (3.0 seconds)
Activities: sit, stand, cycle, run
Train/Test split: P01-P10 (train), P11-P17 (test) as defined by original authors.

Pipeline:
  1. Parse Study_Information.csv for activity time boundaries
  2. Segment each participant's accelerometer CSV into per-activity CSVs
  3. Slice per-activity CSVs into non-overlapping windows of 156 points
  4. Save train/test .npy files per activity
"""

import numpy as np
import pandas as pd
import os
import argparse
from glob import glob
from datetime import datetime
import pytz


def convert_to_timestamp(time_str):
    """Convert local time string to UTC timestamp."""
    local_dt = datetime.strptime(time_str, '%m/%d/%y %H:%M')
    local_tz = pytz.timezone('Europe/London')
    local_dt = local_tz.localize(local_dt)
    utc_dt = local_dt.astimezone(pytz.UTC)
    return utc_dt.timestamp()


def step1_split_by_activity(data_dir):
    """Split raw accelerometer CSVs into per-activity CSVs."""
    info_df = pd.read_csv(f'{data_dir}/Study_Information.csv')

    time_columns = ['Start_Sit', 'Start_Stand', 'Start_Cycle1', 'Start_Run1', 'Start_Run2']
    for col in time_columns:
        info_df[col] = info_df[col].apply(convert_to_timestamp)

    action_periods = [
        ('sit', 'Start_Sit', 'Start_Stand'),
        ('stand', 'Start_Stand', 'Start_Cycle1'),
        ('cycle', 'Start_Cycle1', 'Start_Run1'),
        ('run', 'Start_Run1', 'Start_Run2'),
    ]

    for _, row in info_df.iterrows():
        participant = row['Participant']
        acc_file = f'{data_dir}/{participant}_acc.csv'
        if not os.path.exists(acc_file):
            print(f'Warning: {acc_file} not found, skipping')
            continue

        acc_df = pd.read_csv(acc_file)
        acc_df.columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z']

        output_dir = f'{data_dir}/processed/{participant}'
        os.makedirs(output_dir, exist_ok=True)

        for action, start_col, end_col in action_periods:
            start_time = row[start_col]
            end_time = row[end_col]

            action_data = acc_df[
                (acc_df['timestamp'] >= start_time) &
                (acc_df['timestamp'] <= end_time)
            ].copy()

            output_file = f'{output_dir}/{participant}_{action}.csv'
            action_data.to_csv(output_file, index=False)

        print(f'  Split {participant} into per-activity CSVs')


def segment_timeseries(data, window_size=156):
    """Slice time series into non-overlapping windows, excluding windows with NaN."""
    n_samples = len(data)
    n_windows = n_samples // window_size

    windows = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window = data[start:end]
        if not pd.isna(window).any():
            windows.append(window)

    return np.array(windows) if windows else np.array([])


def step2_create_npy(out_dir, split_dir, split_name, window_size=156):
    """Create .npy files from per-activity CSVs for a given train/test split."""
    actions = ['sit', 'stand', 'cycle', 'run']

    for action in actions:
        pattern = f'{split_dir}/*_{action}.csv'
        files = glob(pattern)

        all_windows = []
        for file in files:
            df = pd.read_csv(file)
            data = df[['acc_x', 'acc_y', 'acc_z']].values
            windows = segment_timeseries(data, window_size=window_size)
            if len(windows) > 0:
                all_windows.append(windows)

        if all_windows:
            all_windows = np.vstack(all_windows)
            output_file = f'{out_dir}/gashi_{action}_{split_name}.npy'
            np.save(output_file, all_windows)
            print(f'  Saved {output_file}, shape: {all_windows.shape}')
        else:
            print(f'  No valid windows for {action} ({split_name})')


def main():
    parser = argparse.ArgumentParser(description='Preprocess Gashi et al. dataset')
    parser.add_argument('--data_dir', type=str,
                        default='/Users/lindan/Dropbox/PhD/Projects/PLF/GAN/Data/gashi',
                        help='Root data directory for Gashi dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for .npy files (default: same as data_dir)')
    parser.add_argument('--window_size', type=int, default=156,
                        help='Window size in data points (default: 156 = 3s at 52Hz)')
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.output_dir if args.output_dir else data_dir
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Split raw acc CSVs by activity
    print('Step 1: Splitting accelerometer data by activity...')
    step1_split_by_activity(data_dir)

    # Step 2: Create .npy files from all per-activity CSVs (no train/test split)
    print('\nStep 2: Creating .npy files (all data)...')
    # Collect from both Train and Test directories
    import tempfile, shutil
    all_dir = os.path.join(tempfile.gettempdir(), 'gashi_all')
    if os.path.exists(all_dir):
        shutil.rmtree(all_dir)
    os.makedirs(all_dir)
    for sub in ['Train', 'Test']:
        src = os.path.join(data_dir, sub)
        if os.path.exists(src):
            for f in glob(os.path.join(src, '*.csv')):
                shutil.copy2(f, all_dir)
    step2_create_npy(out_dir, all_dir, 'all', window_size=args.window_size)
    shutil.rmtree(all_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
