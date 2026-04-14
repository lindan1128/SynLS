"""
Figure 3E: Future health events prediction (chronological split 60/40)
For each dataset x event:
  1. Chronological split: first 60% as historical, last 40% as prospective
  2. Generate synthetic data from historical event data using SynLS
  3. Train 2-layer BiLSTM with augmentation ratios: 0%, 50%, 100%, 150%, 200%, 300%, 100%-only-synthetic
  4. Evaluate AUROC and Recall on prospective (future) data
Results saved to fig3/tables/fig3e_{D1..D4}_{event}.csv

Note: D1 and D3 may not have calving events in the prospective portion.

Usage:
    cd /Users/lindan/Dropbox/PhD/Projects/PLF/GAN/code/fig3
    python fig3e_utility_future.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
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
    """2-layer BiLSTM with FC head."""
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


def run_utility_future(ds_name, event_name, real_event_path, normal_path):
    """Run chronological 60/40 split with different augmentation ratios."""
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}, Event: {event_name}")

    event_data = np.load(real_event_path)
    normal_data = np.load(normal_path)

    scaler = TSFeatureScaler()

    # Chronological split: first 60% historical, last 40% prospective
    n_event = len(event_data)
    n_normal = len(normal_data)
    split_event = int(n_event * 0.6)
    split_normal = int(n_normal * 0.6)

    event_train_raw = event_data[:split_event]
    event_test_raw = event_data[split_event:]
    normal_train_raw = normal_data[:split_normal]
    normal_test_raw = normal_data[split_normal:]

    print(f"  Event: {n_event} total -> train={len(event_train_raw)}, test={len(event_test_raw)}")
    print(f"  Normal: {n_normal} total -> train={len(normal_train_raw)}, test={len(normal_test_raw)}")

    if len(event_test_raw) == 0:
        print(f"  SKIP: No event samples in prospective portion.")
        return

    # Scale
    event_train = scaler.fit_transform(event_train_raw)
    event_test = scaler.fit_transform(event_test_raw)
    normal_train = scaler.fit_transform(normal_train_raw)
    normal_test = scaler.fit_transform(normal_test_raw)

    # Generate synthetic event data from historical
    print(f"  Generating synthetic data from {len(event_train_raw)} historical samples...")
    max_syn = int(len(event_train) * 3.0)
    if max_syn == 0:
        print(f"  SKIP: Not enough historical event data to generate synthetic.")
        return
    syn_event = generate_synthetic(event_train_raw, max(max_syn, 1))

    # Test set (always real prospective data)
    X_test = np.concatenate([event_test, normal_test])
    y_test = np.concatenate([np.ones(len(event_test)), np.zeros(len(normal_test))])

    proportions = [0, 0.5, 1.0, 1.5, 2.0, 3.0]
    prop_names = ['all real(baseline)', 'real+50%Syn', 'real+100%Syn',
                  'real+150%Syn', 'real+200%Syn', 'real+300%Syn', 'all Syn']

    all_results = []

    # Run 5 repeats for stability
    for repeat in range(1, 6):
        print(f"\n  Repeat {repeat}/5")

        for p_idx, prop in enumerate(proportions):
            n_syn = int(len(event_train) * prop)
            if n_syn > 0 and n_syn <= len(syn_event):
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
                'Fold': repeat, 'Proportion': prop_names[p_idx],
                'Accuracy': acc, 'Sensitivity': sens,
                'Specificity': spec, 'F1': f1, 'AUC': auc
            })
            print(f"    {prop_names[p_idx]:25s} AUC={auc:.3f} Recall={sens:.3f}")

        # All-synthetic only
        n_syn_all = int(len(event_train) * 3.0)
        if n_syn_all > 0 and n_syn_all <= len(syn_event):
            syn_only = syn_event[:n_syn_all]
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
        else:
            acc, sens, spec, f1, auc = 0.5, 0, 0, 0, 0.5

        all_results.append({
            'Fold': repeat, 'Proportion': 'all Syn',
            'Accuracy': acc, 'Sensitivity': sens,
            'Specificity': spec, 'F1': f1, 'AUC': auc
        })
        print(f"    {'all Syn':25s} AUC={auc:.3f} Recall={sens:.3f}")

    # Save
    out_csv = os.path.join(tables_dir, f"fig3e_{ds_name}_{event_name}.csv")
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
            run_utility_future(ds_name, event_name, event_path, normal_path)

    print("\n\nAll Figure 3E results saved.")


if __name__ == "__main__":
    main()
