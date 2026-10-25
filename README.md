# FPGA-Binary-SNGP-Anomaly-Detection

**Uncertainty-Aware TinyML Anomaly Detection on FPGA for Edge AI in Industrial IoT**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![FPGA: Intel Cyclone IV](https://img.shields.io/badge/FPGA-Intel%20Cyclone%20IV%20EP4CE22-blue.svg)](https://www.intel.com/)
[![Verification: Bit-Exact](https://img.shields.io/badge/Board%20Replay-2%2C282%2F2%2C282%20Bit--Exact-success.svg)](#hardware-verification)

Official implementation and hardware co-design for the paper:  
**"Binary-SNGP: Uncertainty-Aware TinyML Anomaly Detection on FPGA for Edge AI in Industrial IoT"**  
*The Tuan Trinh, Nguyen Du, Khoa Nguyen Dang, Xuanque Nguyen* (IEEE SISIMPACT-2027)

---

## 🌟 Overview

Edge nodes in the Industrial Internet of Things (IIoT) monitoring rotating equipment often fail silently when sensor orientation shifts or when operational regimes drift from training distributions. Standard softmax outputs are notoriously overconfident on out-of-distribution (OOD) data, while Monte-Carlo dropout and Deep Ensembles multiply hardware latency, parameter memory, and logic footprint.

**Binary-SNGP** provides single-pass, distance-aware predictive uncertainty co-designed for low-cost FPGAs:
1. **1-Bit Convolutional Backbone:** Replaces all convolutional multiplications with additions/subtractions (Layer 1) and XNOR-popcount operations (Layers 2–3). Batch normalization and sign activations fold into a single integer threshold comparator per channel.
2. **Integer Gaussian-Process (GP) Output Layer:** Random Fourier features reformulated into integer arithmetic with $\pm 1$ binary projection weights, quantized phases, and a 1,024-entry cosine ROM.
3. **Conservative Low-Rank Variance Engine:** A rank-16 variance formulation that **provably never underestimates** exact posterior variance (Proposition 1), ensuring fail-safe, cautious monitoring.
4. **Two-Comparator Decision Logic:** The entire inference reduces to two integer comparisons:
   - $\text{Class Decision: } \mathbb{1}[L > 0]$ (normal vs. abnormal)
   - $\text{Uncertainty Decision: } \mathbb{1}[V > T_V]$ (trusted vs. unknown / OOD)
5. **Physical Silicon Verification:** Deployed on an **Intel Cyclone IV EP4CE22** FPGA at **1,971 clock cycles (39.42 µs @ 50 MHz)** per window, verified **bit-exact across 2,282 physical recordings** over JTAG, consuming only **116.5 mW (2.75 µJ per inference)**.

---

## 📊 Key Results at a Glance

### Hardware Metrics (Cyclone IV EP4CE22F17C6 @ 50 MHz)

| Metric | Deployed Model | Orientation-Robust Model | Silicon Available | Utilization |
|---|:---:|:---:|:---:|:---:|
| **Logic Elements (LE)** | 16,367 | **17,810** | 22,320 | **79.8%** (Core = 28.9%) |
| **Dedicated Registers** | 5,450 | **6,590** | 22,320 | **29.5%** |
| **Block Memory Bits** | 15,360 | **15,360** | 608,256 | **2.5%** |
| **Embedded 9-bit Multipliers** | 22 | **22** | 132 | **16.7%** (0 in conv) |
| **Operating Frequency ($F_{\max}$)** | 61.54 MHz | **58.11 MHz** | — | **Slack +2.79 ns @ 50 MHz** |
| **Deterministic Latency** | 1,778 cycles (35.56 µs) | **1,971 cycles (39.42 µs)** | — | **Zero Jitter (Min = Max)** |
| **Duty Cycle (at 64 ms period)** | 0.056% | **0.0616%** | — | **>99.93% Sleep / Idle** |
| **Active System Power** | 116.5 mW | **116.49 mW** | — | Ultra-Low Power |
| **Energy per Inference** | 2.48 µJ | **2.75 µJ** | — | TinyML Budget |
| **Board Replay Match Rate** | 2,282 / 2,282 | **2,282 / 2,282 (100.0%)** | — | **Bit-for-Bit Exact** |
| **Telemetry Bandwidth** | 12 bytes / 64 ms | **12 bytes / 64 ms (187.5 B/s)** | — | **16× Compression vs. Raw** |

### Uncertainty Quality vs. Hardware Cost Comparison

| Method | Passes | Multipliers | Memory Scaling | Cycles @ 50 MHz | Latency | Mean OOD AUROC (Robust) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Deterministic BNN** | 1 | 1 | $1\times$ (1.15 kbit) | 1,313 | 26.26 µs | $0.466 \pm 0.081$ |
| **MC Dropout ($p=0.2$)** | 20 | 1 | $1\times$ + PRNG | 5,073 | 101.46 µs | $0.592 \pm 0.083$ |
| **Deep Ensemble ($M=5$)** | 5 | 1 | $5\times$ (5.75 kbit) | 6,745 | 134.90 µs | $0.567 \pm 0.075$ |
| **Mahalanobis (DDU)** | 1 | 33 | $1\times$ + Cov. Matrix | 1,417 | 28.34 µs | $0.514 \pm 0.083$ |
| **Binary-SNGP (Ours)** | **1** | **22** | **35.9 kbit** (Self-contained) | **1,971** | **39.42 µs** | **$\mathbf{0.810 \pm 0.030}$** |

### Real Fan Validation (DE0-Nano Live Test)

| Experimental Condition | Description / Injected State | Windows Logged | % Abnormal ($L > T_L$) | % Unknown ($V > T_V$) | Decision Reliability |
|---|---|:---:|:---:|:---:|---|
| **Condition A: Healthy 100%** | Full-speed normal rotation | 940 | **0.00%** | **0.11%** | **Zero False Alarms** |
| **Condition B: Healthy Stepped** | Stepped speed (75% and 50%) | 940 | **0.00%** | **1.28%** | **Robust to speed shifts** |
| **Condition C: Abnormal** | Unbalance mass on fan rotor | 940 | **100.00%** | **0.32%** | **100% Fault Detection** |
| **Condition D: Sensor Fault** | Disconnected / zeroed channel | 903 | **0.00%** | **100.00%** | **100% Flagged Unknown** |

---

## 🏗️ Architecture

```
                       ┌─────────────────────────────────────────────────────────┐
                       │                Binary-SNGP Edge Accelerator             │
                       └─────────────────────────────────────────────────────────┘
Raw IMU Readings (500 Hz)
(ADXL345 & ITG3205)
      │
      ▼
┌──────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ Fixed-Point  │ 8-bit│   1-Bit CNN Backbone    │      │    Integer GP Head      │
│ Normalizer   ├─────►│ L1: 6→16  Conv (Add/Sub)├─────►│ D=128 RFF Cosine ROM    │
│ & DC Removal │      │ L2: 16→32 Conv (XNOR-Pop)      │ R=16 Low-Rank Variance  │
└──────────────┘      │ L3: 32→32 Conv (XNOR-Pop)      └────────────┬────────────┘
                      └─────────────────────────┘                   │
                                                                    ▼
                                                       ┌─────────────────────────┐
                                                       │  Two-Comparator Output  │
                                                       │                         │
                                                       │  L > 0   ──► Abnormal?  │
                                                       │  V > T_V ──► Unknown?   │
                                                       └────────────┬────────────┘
                                                                    │
                                                                    ▼
                                                       ┌─────────────────────────┐
                                                       │ 12-Byte IIoT Telemetry  │
                                                       │ [0xA5, flags, seq, L, V]│
                                                       └─────────────────────────┘
```

---

## 📁 Repository Structure

```
.
├── anomaly_detection/          # Data loaders, Z-score normalization, STE binarized layers
├── bsngp/                      # Core Binary-SNGP algorithms & hardware generators
│   ├── golden.py               # Bit-exact integer reference specification (Python)
│   ├── bsngp_model.py          # PyTorch training model, folded BN, low-rank covariance
│   ├── data.py                 # Dataset splits, synthetic OOD generators, co-sim streams
│   ├── orientation.py          # SO(3) 3D rotations, field mounting simulations
│   ├── train_eval.py           # Training pipeline (5 seeds) + rank/K parameter ablations
│   ├── transfer_orientation.py # Orientation-robust model (isotropic norm + DC removal)
│   ├── compare_uncertainty.py  # Benchmarking against MC Dropout, Ensembles, Mahalanobis
│   ├── export_vhdl.py          # Exports PyTorch parameters to synthesizable VHDL packages
│   ├── rtl/                    # Synthesizable VHDL (bsngp_core.vhd, DE0-Nano top, UART)
│   ├── tb/ & sim/              # GHDL bit-exact testbenches and automated validation scripts
│   ├── host/                   # stream_to_board.py: Real-time 500 Hz streaming & JTAG replay
│   ├── quartus/                # Intel Quartus Prime Lite project, pinout, SDC constraints
│   ├── release_robust/         # Production-ready orientation-robust RTL package & params
│   └── export*/                # Exported model weights, parameter arrays, and JSON logs
├── checkpoints/                # Normalization parameters (isotropic and per-axis)
├── dataset/                    # 12 fan IMU recordings (500 Hz 6-axis vibration data)
├── paper/                      # Figure generation scripts matching manuscript.pdf
├── plotneuralnet/              # LaTeX neural network architecture visualization scripts
├── HARDWARE_RUNS_RESULTS.md    # Detailed laboratory execution report from DE0-Nano board
├── requirements.txt            # Python dependencies
└── README.md                   # This document
```

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup

```bash
# Clone the repository
git clone https://github.com/thetuantrinh/FPGA-Binary-SNGP-Anomaly-Detection.git
cd FPGA-Binary-SNGP-Anomaly-Detection

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Bit-Exact RTL Co-Simulation (GHDL)

Verify that the synthesized VHDL architecture matches the Python reference model with **zero bit errors**:

```bash
# Ensure GHDL (>= 4.0) is installed and available in PATH
# 1. Export VHDL parameter packages and co-sim test vectors
python bsngp/export_vhdl.py --src export_robust

# 2. Run automated co-simulation suite
bash bsngp/sim/run_sim.sh
```
*Expected output: `ALL CO-SIMULATIONS PASS (Bit-for-bit exact match)`.*

### 3. Model Training & OOD Benchmarking

To train the models from scratch across 5 random seeds and evaluate on out-of-distribution datasets:

```bash
# 1. Train original Binary-SNGP model and run K / Rank ablations
python bsngp/train_eval.py

# 2. Train orientation-robust variant with SO(3) data augmentation
python bsngp/transfer_orientation.py

# 3. Benchmark all uncertainty baselines (MC Dropout, Ensembles, Mahalanobis)
python bsngp/compare_uncertainty.py
```

### 4. Reproducing Manuscript Figures

```bash
python bsngp/viz_data.py
python paper/make_figures.py
```
Generated high-resolution vector figures are saved to `paper/figures/`.

---

## ⚡ FPGA Deployment & Hardware Execution

### Quartus Prime Compilation

1. Open `bsngp/quartus/bsngp_de0.qpf` in **Intel Quartus Prime Lite** (v21.1 or v25.1).
2. Target Device: **Cyclone IV E `EP4CE22F17C6N`**.
3. Run **Full Compilation** (Synthesis, Fitter, Timing Analyzer, Assembler).
4. Connect the Terasic DE0-Nano board via USB-Blaster and program `bsngp_de0.sof`.

### Real-Time Live Streaming & JTAG Hardware Replay

Stream IMU vibration windows to the FPGA over UART or replay test recordings bit-for-bit over JTAG:

```bash
# Dry-run validation (executes golden reference model locally)
python bsngp/host/stream_to_board.py --csv dataset/valid/normal/2026-09-17_10-46-36.csv --dry-run

# Live streaming over UART (500 Hz continuous feed)
python bsngp/host/stream_to_board.py --port COM3 --csv dataset/valid/abnormal/2026-09-17_10-59-27.csv
```

### Packet Protocol

* **Input Sensor Frame (8 bytes @ 500 Hz):**
  `[0x80, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, chk]` (int8 format)
* **Output Telemetry Packet (12 bytes @ 15.625 Hz / every 64 ms):**
  `[0xA5, flags, seq, L_msb, L_lsb, V_b3, V_b2, V_b1, V_b0, rsv, rsv, xor_chk]`
  * `flags[0]`: Abnormal indicator ($L > T_L$)
  * `flags[1]`: Unknown / OOD indicator ($V > T_V$)
  * `flags[2]`: Hardware buffer overrun warning

---

## 📖 Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@inproceedings{trinh2027binarysngp,
  title     = {Binary-SNGP: Uncertainty-Aware TinyML Anomaly Detection on FPGA for Edge AI in Industrial IoT},
  author    = {Trinh, The Tuan and Du, Nguyen and Dang, Khoa Nguyen and Nguyen, Xuanque},
  booktitle = {Proceedings of the IEEE International Conference on Smart Information Systems and Impact (SISIMPACT)},
  year      = {2027},
  pages     = {1--6},
  doi       = {10.1109/SISIMPACT.2027.TODO}
}
```

---

## 📜 License & Acknowledgments

* **Code License:** [MIT License](LICENSE).
* **Network Visualization:** PlotNeuralNet layer styles are licensed under the MIT License (`plotneuralnet/LICENSE.PlotNeuralNet`).
* **Funding Acknowledgments:** This research is funded by the Vietnam National Foundation for Science and Technology Development (NAFOSTED) under grant number `102.01-2025.100`.
