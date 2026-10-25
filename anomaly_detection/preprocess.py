#!/usr/bin/env python3
"""
Data Preprocessing and Windowing Pipeline for FPGA Anomaly Detection:
- Enforces strict recording-level train/val/test split (no data leakage).
- Standardizes feature schema (ax, ay, az, gx, gy, gz).
- Calculates Z-score statistics (mean, std) exclusively from training recordings.
- Generates sliding window segments with configurable window size and stride.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import yaml

TARGET_CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]

def load_single_csv(file_path: str) -> np.ndarray:
    """Load a CSV, standardize column names, and return array of shape (T, 6)."""
    df = pd.read_csv(file_path)
    rename_map = {
        'ax_g': 'ax', 'ay_g': 'ay', 'az_g': 'az',
        'gx_dps': 'gx', 'gy_dps': 'gy', 'gz_dps': 'gz'
    }
    df = df.rename(columns=rename_map)
    
    missing_cols = [c for c in TARGET_CHANNELS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"File {file_path} missing required channels: {missing_cols}")
        
    data = df[TARGET_CHANNELS].values.astype(np.float32)
    # Check NaN / Inf
    if np.isnan(data).any() or np.isinf(data).any():
        print(f"Warning: Found NaN/Inf in {file_path}, interpolating/clipping...")
        df_clean = df[TARGET_CHANNELS].interpolate(method='linear').bfill().ffill()
        data = df_clean.values.astype(np.float32)
        
    return data

def extract_windows(data: np.ndarray, label: int, window_size: int, stride: int):
    """
    Segment (T, num_channels) continuous sensor data into sliding windows.
    Returns:
        windows: np.ndarray of shape (N, num_channels, window_size)
        labels: np.ndarray of shape (N,)
    """
    T, num_channels = data.shape
    if T < window_size:
        return np.empty((0, num_channels, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)
        
    # Vectorized / strided extraction
    num_windows = (T - window_size) // stride + 1
    windows = np.zeros((num_windows, num_channels, window_size), dtype=np.float32)
    
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        # Shape: (num_channels, window_size)
        windows[i] = data[start:end, :].T
        
    labels = np.full((num_windows,), fill_value=label, dtype=np.int64)
    return windows, labels

class DatasetPreprocessor:
    def __init__(self, config_path: str = "anomaly_detection/config.yaml"):
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
            
        self.data_dir = self.config.get("data", {}).get("raw_dir", "dataset")
        self.split_cfg = self.config.get("split", {})
        self.stats_file = self.config.get("data", {}).get("stats_file", "checkpoints/norm_stats.json")
        self.means = None
        self.stds = None

    def compute_train_statistics(self) -> dict:
        """Calculate mean and std strictly on training recordings."""
        train_norm_files = [os.path.join(self.data_dir, p) for p in self.split_cfg.get("train_normal", [])]
        train_abnorm_files = [os.path.join(self.data_dir, p) for p in self.split_cfg.get("train_abnormal", [])]
        all_train_files = train_norm_files + train_abnorm_files
        
        all_train_data = []
        for f in all_train_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Training file not found: {f}")
            data = load_single_csv(f)
            all_train_data.append(data)
            
        concat_train = np.vstack(all_train_data)
        means = np.mean(concat_train, axis=0) # shape (6,)
        stds = np.std(concat_train, axis=0)   # shape (6,)
        
        # Guard against zero-variance
        stds = np.where(stds < 1e-8, 1.0, stds)
        
        self.means = means.astype(np.float32)
        self.stds = stds.astype(np.float32)
        
        stats = {
            "channels": TARGET_CHANNELS,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "train_sample_count": int(concat_train.shape[0]),
            "train_files": all_train_files
        }
        
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
        with open(self.stats_file, "w") as f:
            json.dump(stats, f, indent=2)
            
        print(f"Computed & saved training statistics to {self.stats_file}:")
        for ch, m, s in zip(TARGET_CHANNELS, self.means, self.stds):
            print(f"  {ch:>6}: mean = {m:+.4f}, std = {s:.4f}")
            
        return stats

    def load_statistics(self):
        """Load cached normalization statistics."""
        if os.path.exists(self.stats_file):
            with open(self.stats_file, "r") as f:
                stats = json.load(f)
            self.means = np.array(stats["means"], dtype=np.float32)
            self.stds = np.array(stats["stds"], dtype=np.float32)
        else:
            self.compute_train_statistics()

    def apply_normalization(self, data: np.ndarray) -> np.ndarray:
        """Apply z = (x - mu) / sigma."""
        if self.means is None or self.stds is None:
            self.load_statistics()
        return (data - self.means) / self.stds

    def prepare_split(self, split_name: str, window_size: int = 64, stride: int = 32, normalize: bool = True):
        """
        Extract windows for a given split (train, val, test).
        Returns:
            X: np.ndarray of shape (N, 6, window_size)
            y: np.ndarray of shape (N,)
        """
        if normalize and (self.means is None or self.stds is None):
            self.load_statistics()
            
        if split_name == "train":
            norm_rel_files = self.split_cfg.get("train_normal", [])
            abnorm_rel_files = self.split_cfg.get("train_abnormal", [])
        elif split_name in ("val", "valid"):
            norm_rel_files = self.split_cfg.get("val_normal", [])
            abnorm_rel_files = self.split_cfg.get("val_abnormal", [])
        elif split_name == "test":
            norm_rel_files = self.split_cfg.get("test_normal", [])
            abnorm_rel_files = self.split_cfg.get("test_abnormal", [])
        else:
            raise ValueError(f"Unknown split name: {split_name}")
            
        all_X = []
        all_y = []
        
        # Normal recordings (Class 0)
        for rel in norm_rel_files:
            f = os.path.join(self.data_dir, rel)
            raw = load_single_csv(f)
            data = self.apply_normalization(raw) if normalize else raw
            wins, labs = extract_windows(data, label=0, window_size=window_size, stride=stride)
            if len(wins) > 0:
                all_X.append(wins)
                all_y.append(labs)
                
        # Abnormal recordings (Class 1)
        for rel in abnorm_rel_files:
            f = os.path.join(self.data_dir, rel)
            raw = load_single_csv(f)
            data = self.apply_normalization(raw) if normalize else raw
            wins, labs = extract_windows(data, label=1, window_size=window_size, stride=stride)
            if len(wins) > 0:
                all_X.append(wins)
                all_y.append(labs)
                
        if not all_X:
            return np.empty((0, 6, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)
            
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        return X, y

def main():
    parser = argparse.ArgumentParser(description="Preprocess IMU dataset")
    parser.add_argument("--config", type=str, default="anomaly_detection/config.yaml")
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    args = parser.parse_args()
    
    prep = DatasetPreprocessor(args.config)
    prep.compute_train_statistics()
    
    for s in ["train", "val", "test"]:
        X, y = prep.prepare_split(s, window_size=args.window_size, stride=args.stride, normalize=True)
        c0 = int(np.sum(y == 0))
        c1 = int(np.sum(y == 1))
        print(f"Split '{s:<5}': shape {str(X.shape):<20} | Normal (0): {c0:<5} | Abnormal (1): {c1:<5} | Total: {len(y)}")

if __name__ == "__main__":
    main()
