"""
Figure 3C: Post-hoc discriminator (real vs synthetic)
Runs 2-layer LSTM and CNN classifiers with 5-fold CV x 3 repeats to distinguish
real from synthetic time series for all 4 datasets x 3 events.
Results saved to fig3/tables/ as CSV files.

Usage:
    python fig3c_posthoc.py
    python fig3c_posthoc.py --base_dir . --no_gpu
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import argparse
import os, warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_parser = argparse.ArgumentParser()
_parser.add_argument('--base_dir', type=str,
                     default='/Users/lindan/Dropbox/PhD/Projects/PLF/GAN',
                     help='Base project directory')
_parser.add_argument('--no_gpu', action='store_true',
                     help='Disable GPU')
_args = _parser.parse_args()
if _args.no_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
BASE = _args.base_dir

dataset_dir = os.path.join(BASE, 'code_github', 'dataset')
tables_dir  = os.path.join(BASE, 'code', 'fig3', 'tables')
os.makedirs(tables_dir, exist_ok=True)

EPS = 1e-18

class TSFeatureScaler:
    def __init__(self):
        self.min_val = None
        self.max_val = None
    def fit(self, X):
        self.min_val = np.min(X)
        self.max_val = np.max(X)
        return self
    def transform(self, X):
        X_scaled = (X - self.min_val) / (self.max_val - self.min_val + EPS)
        return 2.0 * X_scaled - 1.0
    def fit_transform(self, X):
        return self.fit(X).transform(X)


def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(20, return_sequences=True, input_shape=input_shape),
        LSTM(20),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def run_posthoc(real_path, synth_path, output_csv, patience=10):
    print(f"\n{'='*60}")
    print(f"Real:  {real_path}")
    print(f"Synth: {synth_path}")

    data1 = np.load(real_path)
    data2 = np.load(synth_path)

    scaler = TSFeatureScaler()
    data1_scaled = scaler.fit_transform(data1)
    data2_scaled = scaler.fit_transform(data2)

    print(f"Real shape: {data1_scaled.shape}, Synth shape: {data2_scaled.shape}")

    labels = np.concatenate([np.ones(len(data1_scaled)), np.zeros(len(data2_scaled))])
    data = np.concatenate([data1_scaled, data2_scaled])

    all_results = []

    # 3-fold CV x 5 repeats
    for rep in range(1, 6):
        kf = KFold(n_splits=3, shuffle=True, random_state=42 + rep)

        for model_type in ['LSTM', 'CNN']:
            for fold, (train_idx, val_idx) in enumerate(kf.split(data), 1):
                X_train, X_val = data[train_idx], data[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]

                if model_type == 'LSTM':
                    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
                else:
                    model = create_cnn_model((X_train.shape[1], X_train.shape[2]))

                early_stopping = EarlyStopping(
                    monitor='loss', patience=patience, min_delta=1e-4,
                    verbose=0, mode='min', restore_best_weights=False
                )

                model.fit(X_train, y_train, epochs=50, batch_size=32,
                          callbacks=[early_stopping], verbose=0)

                y_pred_proba = model.predict(X_val, verbose=0).ravel()
                y_pred = (y_pred_proba > 0.5).astype(int)

                acc = accuracy_score(y_val, y_pred)
                sens = recall_score(y_val, y_pred, zero_division=0)
                spec = recall_score(y_val, y_pred, pos_label=0, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                try:
                    auc = roc_auc_score(y_val, y_pred_proba)
                except ValueError:
                    auc = 0.5

                all_results.append({
                    'Repeat': rep, 'Model': model_type, 'Fold': fold,
                    'Accuracy': acc, 'Sensitivity': sens,
                    'Specificity': spec, 'F1': f1, 'AUC': auc
                })
                print(f"  Rep{rep} {model_type} Fold{fold}: Acc={acc:.3f} |0.5-Acc|={abs(0.5-acc):.3f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")


def main():
    datasets = ['D1', 'D2', 'D3', 'D4']
    events = [
        ('disease', 'disease.npy', 'disease_synls_time.npy'),
        ('estrus',  'estrus.npy',  'estrus_synls_time.npy'),
        ('calving', 'calving.npy', 'calving_synls_time.npy'),
    ]

    for ds in datasets:
        for event_name, real_file, synth_file in events:
            real_path  = os.path.join(dataset_dir, ds, real_file)
            synth_path = os.path.join(dataset_dir, ds, synth_file)
            output_csv = os.path.join(tables_dir, f"{ds}_{event_name}_posthoc.csv")
            run_posthoc(real_path, synth_path, output_csv)

    print(f"\nAll posthoc results saved to {tables_dir}/")


if __name__ == "__main__":
    main()
