"""
Figure 3D: Overall health events prediction (instance-based, 3-fold CV x 3 repeats)
For each dataset x event:
  1. 3-fold split on the event (minor) class
  2. For each fold, generate synthetic minor class data using SynLS
  3. Train 2-layer BiLSTM with augmentation ratios: 0%, 50%, 100%, 150%, 200%, 300%, 100%-only-synthetic
  4. Evaluate AUROC and Recall on real test data
Results saved to fig3/tables/fig3d_{D1..D4}_{event}.csv

Usage:
    cd /Users/lindan/Dropbox/PhD/Projects/PLF/GAN/code/fig3
    python fig3d_utility_overall.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--base_dir', type=str,
                     default='/Users/lindan/Dropbox/PhD/Projects/PLF/GAN',
                     help='Base project directory (absolute or relative)')
_parser.add_argument('--no_gpu', action='store_true',
                     help='Disable GPU (default: use GPU if available)')
_args = _parser.parse_args()
if _args.no_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
BASE = _args.base_dir

sys.path.insert(0, os.path.join(BASE, 'code_github'))
from GAN.utils import TSFeatureScaler
from GAN.diffusion import build_model, GaussianDiffusion, DiffusionModel

dataset_dir = os.path.join(BASE, 'code_github', 'dataset')
tables_dir  = os.path.join(BASE, 'code', 'fig3', 'tables')
os.makedirs(tables_dir, exist_ok=True)


def create_lstm_classifier(input_shape):
    """2-layer BiLSTM with FC head (matching paper description)."""
    model = Sequential([
        Bidirectional(LSTM(20, return_sequences=True), input_shape=input_shape),
        Bidirectional(LSTM(20)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def generate_synthetic(data, num_samples, step=10, epoch=40, batch_size=32):
    """Generate synthetic data using SynLS (dual encoder)."""
    scaler = TSFeatureScaler()
    data_scaled = scaler.fit_transform(data)

    network = build_model(
        time_len=data.shape[1], fea_num=data.shape[2],
        d_model=16, n_heads=2, encoder_type='time')
    ema_network = build_model(
        time_len=data.shape[1], fea_num=data.shape[2],
        d_model=16, n_heads=2, encoder_type='time')
    ema_network.set_weights(network.get_weights())

    noise_util = GaussianDiffusion(timesteps=step)
    model = DiffusionModel(
        network=network, ema_network=ema_network,
        timesteps=step, gdf_util=noise_util, data=data_scaled)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=Adam(learning_rate=1e-4), metrics=['mse'])

    model.fit(data_scaled, epochs=epoch, batch_size=batch_size, verbose=0)

    new_samples = model.generate_ts(num_samples).numpy()
    new_samples = scaler.fit_transform(new_samples)
    return new_samples


def balance_classes(X, y):
    """Apply RandomOverSampler to balance classes."""
    ros = RandomOverSampler(random_state=42)
    X_2d = X.reshape(X.shape[0], -1)
    X_resampled_2d, y_resampled = ros.fit_resample(X_2d, y)
    X_resampled = X_resampled_2d.reshape(-1, X.shape[1], X.shape[2])
    return X_resampled, y_resampled


def evaluate(y_true, y_pred, y_pred_proba):
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = 0.5
    return acc, sens, spec, f1, auc


def run_utility_overall(ds_name, event_name, real_event_path, normal_path):
    """Run 3-fold CV with different augmentation ratios."""
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}, Event: {event_name}")

    event_data = np.load(real_event_path)
    normal_data = np.load(normal_path)

    scaler = TSFeatureScaler()
    event_scaled = scaler.fit_transform(event_data)
    normal_scaled = scaler.fit_transform(normal_data)

    n_event = len(event_scaled)
    n_normal = len(normal_scaled)
    print(f"  Event samples: {n_event}, Normal samples: {n_normal}")

    proportions = [0, 0.5, 1.0, 1.5, 2.0, 3.0]
    prop_names = ['all real(baseline)', 'real+50%Syn', 'real+100%Syn',
                  'real+150%Syn', 'real+200%Syn', 'real+300%Syn', 'all Syn']

    all_results = []

    for rep in range(1, 6):
        kf = KFold(n_splits=3, shuffle=True, random_state=42 + rep)
        for fold, (train_idx, test_idx) in enumerate(kf.split(event_scaled), 1):
            print(f"\n  Repeat {rep}/5, Fold {fold}/3")
            event_train = event_scaled[train_idx]
            event_test = event_scaled[test_idx]

            # Split normal correspondingly
            normal_kf = KFold(n_splits=3, shuffle=True, random_state=42 + rep)
            for f_i, (n_train_idx, n_test_idx) in enumerate(normal_kf.split(normal_scaled), 1):
                if f_i == fold:
                    normal_train = normal_scaled[n_train_idx]
                    normal_test = normal_scaled[n_test_idx]
                    break

            # Generate synthetic event data for this fold
            print(f"    Generating synthetic data from {len(event_train)} train samples...")
            max_syn = int(len(event_train) * 3.0)
            syn_event = generate_synthetic(event_data[train_idx], max_syn)

            # Test set (always real)
            X_test = np.concatenate([event_test, normal_test])
            y_test = np.concatenate([np.ones(len(event_test)), np.zeros(len(normal_test))])

            # Different augmentation ratios
            for p_idx, prop in enumerate(proportions):
                n_syn = int(len(event_train) * prop)
                if n_syn > 0:
                    syn_subset = syn_event[:n_syn]
                    X_train = np.concatenate([event_train, syn_subset, normal_train])
                    y_train = np.concatenate([np.ones(len(event_train) + n_syn),
                                              np.zeros(len(normal_train))])
                else:
                    X_train = np.concatenate([event_train, normal_train])
                    y_train = np.concatenate([np.ones(len(event_train)),
                                              np.zeros(len(normal_train))])

                X_train_bal, y_train_bal = balance_classes(X_train, y_train)
                model = create_lstm_classifier((X_train_bal.shape[1], X_train_bal.shape[2]))
                es = EarlyStopping(monitor='loss', patience=10, min_delta=1e-4, verbose=0)
                model.fit(X_train_bal, y_train_bal, epochs=50, batch_size=32,
                          callbacks=[es], verbose=0)

                y_proba = model.predict(X_test, verbose=0).ravel()
                y_pred = (y_proba > 0.5).astype(int)
                acc, sens, spec, f1, auc = evaluate(y_test, y_pred, y_proba)

                all_results.append({
                    'Repeat': rep, 'Fold': fold, 'Proportion': prop_names[p_idx],
                    'Accuracy': acc, 'Sensitivity': sens,
                    'Specificity': spec, 'F1': f1, 'AUC': auc
                })
                print(f"    {prop_names[p_idx]:25s} AUC={auc:.3f} Recall={sens:.3f}")

            # All-synthetic only (no real event data)
            syn_only = syn_event[:int(len(event_train) * 3.0)]
            X_train = np.concatenate([syn_only, normal_train])
            y_train = np.concatenate([np.ones(len(syn_only)), np.zeros(len(normal_train))])

            X_train_bal, y_train_bal = balance_classes(X_train, y_train)
            model = create_lstm_classifier((X_train_bal.shape[1], X_train_bal.shape[2]))
            es = EarlyStopping(monitor='loss', patience=10, min_delta=1e-4, verbose=0)
            model.fit(X_train_bal, y_train_bal, epochs=50, batch_size=32,
                      callbacks=[es], verbose=0)

            y_proba = model.predict(X_test, verbose=0).ravel()
            y_pred = (y_proba > 0.5).astype(int)
            acc, sens, spec, f1, auc = evaluate(y_test, y_pred, y_proba)

            all_results.append({
                'Repeat': rep, 'Fold': fold, 'Proportion': 'all Syn',
                'Accuracy': acc, 'Sensitivity': sens,
                'Specificity': spec, 'F1': f1, 'AUC': auc
            })
            print(f"    {'all Syn':25s} AUC={auc:.3f} Recall={sens:.3f}")

    # Save results
    out_csv = os.path.join(tables_dir, f"fig3d_{ds_name}_{event_name}.csv")
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")


def main():
    datasets = {
        'D1': {'normal': 'normal.npy', 'events': ['disease', 'estrus', 'calving']},
        'D2': {'normal': 'normal.npy', 'events': ['disease', 'estrus', 'calving']},
        'D3': {'normal': 'normal.npy', 'events': ['disease', 'estrus', 'calving']},
        'D4': {'normal': 'normal.npy', 'events': ['disease', 'estrus', 'calving']},
    }

    for ds_name, cfg in datasets.items():
        normal_path = os.path.join(dataset_dir, ds_name, cfg['normal'])
        for event_name in cfg['events']:
            event_path = os.path.join(dataset_dir, ds_name, f"{event_name}.npy")
            run_utility_overall(ds_name, event_name, event_path, normal_path)

    print("\n\nAll Figure 3D results saved.")


if __name__ == "__main__":
    main()
