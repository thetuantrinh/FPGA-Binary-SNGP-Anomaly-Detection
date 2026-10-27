#!/usr/bin/env python3
"""
Utilities for FPGA Anomaly Detection Pipeline:
- Hardware/CUDA initialization & policy enforcement
- Metric calculation (Acc, Prec, Rec, F1, Spec, Sens, ROC-AUC, PR-AUC, FPR, FNR)
- Model complexity analysis (Params, FP32/INT8 size, MACs, activations)
- Plotting (Confusion Matrix, ROC, PR curves, Learning curves)
"""

import os
import sys
import random
import socket
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    """Ensure end-to-end reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def init_hardware(require_cuda: bool = True) -> torch.device:
    """
    Check and print hardware environment according to HPC policy.
    Terminates immediately if CUDA is required but unavailable.
    """
    host = socket.gethostname()
    cuda_avail = torch.cuda.is_available()
    
    print("=" * 70)
    print("HARDWARE & COMPUTE ENVIRONMENT REPORT")
    print(f"Hostname:       {host}")
    print(f"CUDA Available: {cuda_avail}")
    
    if cuda_avail:
        cuda_ver = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        reserved_mem = torch.cuda.memory_reserved(0) / (1024 ** 2)
        allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 2)
        free_mem = total_mem - allocated_mem
        
        print(f"CUDA Version:   {cuda_ver}")
        print(f"GPU Name:       {gpu_name}")
        print(f"GPU Count:      {gpu_count}")
        print(f"GPU Memory:     {total_mem:.0f} MiB Total | {free_mem:.0f} MiB Free")
        print(f"Active Device:  cuda:0")
        print("=" * 70)
        return torch.device("cuda:0")
    else:
        print("=" * 70)
        if require_cuda:
            print("ERROR: HPC Policy §2 Violation! CUDA GPU is required for deep learning workloads.")
            print("CPU fallback is strictly forbidden on compute tasks.")
            sys.exit(1)
        print("Warning: Running on CPU (evaluation / unit test only).")
        return torch.device("cpu")

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute full suite of binary anomaly detection metrics:
    - Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score
    - ROC-AUC, PR-AUC
    - False Positive Rate (FPR), False Negative Rate (FNR)
    - Confusion Matrix (TN, FP, FN, TP)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0)) # Sensitivity
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0 # Specificity
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0
        
    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "sensitivity": rec,
        "specificity": spec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "fnr": fnr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": cm.tolist()
    }

def compute_model_complexity(model: torch.nn.Module, input_shape: tuple) -> dict:
    """
    Analyze model complexity for FPGA & TinyML deployment:
    - Trainable parameter count
    - FP32 model size (KB)
    - INT8 estimated model size (KB)
    - Theoretical MACs / FLOPs
    - Peak intermediate activation size (KB)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Model storage size
    fp32_size_kb = (total_params * 4) / 1024.0
    int8_size_kb = (total_params * 1) / 1024.0
    
    # Measure MACs and intermediate activation sizes via forward hook
    activations_bytes = 0
    macs = 0
    hooks = []
    
    def hook_fn(module, inp, outp):
        nonlocal activations_bytes, macs
        if isinstance(outp, torch.Tensor):
            activations_bytes = max(activations_bytes, outp.numel() * 4)
            
        if isinstance(module, torch.nn.Linear):
            # MACs = in_features * out_features
            macs += module.in_features * module.out_features
        elif isinstance(module, torch.nn.Conv1d):
            # MACs = out_channels * in_channels * kernel_size * output_length / groups
            out_len = outp.shape[-1]
            macs += (module.out_channels * (module.in_channels // module.groups) * 
                     module.kernel_size[0] * out_len)

    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
            hooks.append(m.register_forward_hook(hook_fn))
            
    # Dummy forward pass
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        _ = model(dummy_input)
        
    for h in hooks:
        h.remove()
        
    if was_training:
        model.train()
        
    return {
        "num_params": total_params,
        "trainable_params": trainable_params,
        "fp32_size_kb": fp32_size_kb,
        "int8_size_kb": int8_size_kb,
        "macs": int(macs),
        "flops": int(macs * 2),
        "peak_activation_kb": activations_bytes / 1024.0,
        "input_shape": list(input_shape)
    }

def plot_confusion_matrix(cm: list, out_path: str, title: str = "Confusion Matrix"):
    """Plot and save publication-quality confusion matrix."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ["Normal (0)", "Abnormal (1)"]
    ax.set(xticks=np.arange(cm_arr.shape[1]),
           yticks=np.arange(cm_arr.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True Class',
           xlabel='Predicted Class')
    
    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, format(cm_arr[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black",
                    fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, roc_path: str, pr_path: str):
    """Plot ROC and PR curves."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(roc_path), exist_ok=True)
    plt.savefig(roc_path, dpi=150)
    plt.close()
    
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color='purple', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall (PR) Curve')
    ax.legend(loc="lower left")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(pr_path), exist_ok=True)
    plt.savefig(pr_path, dpi=150)
    plt.close()

def plot_learning_curves(history: dict, out_path: str):
    """Plot training and validation loss & accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, history["train_loss"], 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, history["val_loss"], 'r-s', label='Val Loss', markersize=4)
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(epochs, history["train_acc"], 'b-o', label='Train Acc', markersize=4)
    ax2.plot(epochs, history["val_acc"], 'r-s', label='Val Acc', markersize=4)
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
