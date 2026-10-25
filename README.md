# Binary-SNGP: Uncertainty-Aware TinyML Anomaly Detection on FPGA

Code for the paper *"Binary-SNGP: Uncertainty-Aware TinyML Anomaly Detection on FPGA for Edge AI in Industrial IoT"*.

Binary-SNGP pairs a 1-bit convolutional network with a Gaussian-process output layer that has been rewritten for integer hardware:
- **Random features:** ±1 random features with a 1,024-entry cosine ROM.
- **Variance:** a rank-16 variance whose truncation never under-estimates the exact posterior variance.
- **Decision:** two integer comparators, `abnormal = L > T_L` and `unknown = V > T_V`.

It runs on an Intel Cyclone IV (Terasic DE0-Nano) at **1,778 cycles (35.6 µs at 50 MHz) per 64-sample window**. The FPGA output matches the integer reference model bit for bit.

| | |
|---|---|
| Model memory | 35.9 kbit (4,896 binary weights, RFF signs/phases, β, rank-16 **U**, cosine ROM) |
| Multipliers | 19, none in the convolutions |
| Latency | 1,778 cycles; 1,971 with in-core DC removal (orientation-robust model) |
| Verification | GHDL co-simulation 204/204 and 167/167 windows; board replay 2,282/2,282 windows |
| OOD AUROC (5 seeds, same backbone) | Binary-SNGP 0.98 · deep ensemble 0.79 · MC dropout 0.62 · deterministic 0.54 |

## Repository layout

```
anomaly_detection/      data loading, z-score preprocessing, binary layers (STE), helpers
bsngp/
  golden.py             bit-exact integer reference model = hardware specification
  bsngp_model.py        PyTorch training model, BN folding, quantisation, low-rank covariance
  data.py               in-distribution splits, OOD sets, co-simulation stream, demo ROM
  orientation.py        sensor re-mounting: rotations, field-mount simulation, augmentation
  train_eval.py         train (5 seeds) + rank/K ablations, evaluate the integer model, export
  transfer_orientation.py  orientation-robust model (isotropic norm, DC removal, rotation augmentation)
  compare_uncertainty.py   deterministic / MC dropout / deep ensemble / Mahalanobis / Binary-SNGP
  viz_data.py           data for the dataset / feature-space / uncertainty figures
  export_vhdl.py        params.npz -> VHDL packages + co-simulation vectors
  rtl/                  bsngp_core.vhd, DE0-Nano top, UART, generated packages
  tb/  sim/             GHDL testbenches, run_sim.sh, compare_rtl.py, stimulus + expected outputs
  host/                 stream_to_board.py: 500 Hz UART streaming, packet decoding, golden check
  quartus/              EP4CE22F17C6 project, pins, timing constraints
  release_robust/       ready-to-use orientation-robust model (RTL + params + normalisation)
  export*/              results of the paper (JSON / Markdown) and exported integer models
checkpoints/norm_stats.json   z-score statistics of the training recordings
dataset/                (not included) 12 IMU recordings, 500 Hz, t_s, ax..gz; available from the authors on request
paper/make_figures.py   all result figures of the paper
plotneuralnet/          network drawing (PlotNeuralNet layer styles, MIT licence)
```

## Quick start

```bash
pip install -r requirements.txt

# 1. bit-exact RTL co-simulation of the shipped robust model (no GPU needed; GHDL >= 4 on PATH)
python bsngp/export_vhdl.py --src export_robust
bash   bsngp/sim/run_sim.sh            # -> ALL CO-SIMULATIONS PASS

# 2. run the integer model on an IMU recording (CSV with columns t_s, ax, ay, az, gx, gy, gz; g and deg/s, 500 Hz)
python bsngp/host/stream_to_board.py --csv my_recording.csv --dry-run \
       --stats bsngp/export_robust/norm_stats_isotropic.json --params bsngp/export_robust/params.npz

# 3. re-train and re-evaluate (CUDA GPU; needs the recordings in dataset/, split as in anomaly_detection/config.yaml)
python bsngp/train_eval.py             # original model           -> bsngp/export/
python bsngp/transfer_orientation.py   # orientation-robust model -> bsngp/export_robust/
python bsngp/compare_uncertainty.py    # method comparison        -> bsngp/export_compare/

# 4. figures
python bsngp/viz_data.py && python paper/make_figures.py && python plotneuralnet/bsngp_architecture.py
```

The OOD experiments also use public datasets, which are not included. Place them under `online_datasets/` as expected by `anomaly_detection/online_datasets_loader.py`:
- the [UAV-Realistic-Fault-Dataset](https://github.com/tiiuae/UAV-Realistic-Fault-Dataset);
- the [Electric Motor Vibrations Dataset](https://doi.org/10.5281/zenodo.6473455);
- the [Analog Devices CbM datasets](https://github.com/analogdevicesinc/CbM-Datasets).

## Running on the board

See `bsngp/README.md` ("Running on the board") and `bsngp/release_robust/README.md`.

**UART interface:**
- **In:** frames `0x80, ax, ay, az, gx, gy, gz` (int8).
- **Out:** a 12-byte packet `0xA5, flags, seq, L, V, xor`.

## Citation

```bibtex
@inproceedings{binarysngp,
  title     = {Binary-SNGP: Uncertainty-Aware TinyML Anomaly Detection on FPGA for Edge AI in Industrial IoT},
  author    = {TODO},
  booktitle = {TODO},
  year      = {TODO}
}
```

## Licence

MIT, see `LICENSE`. The PlotNeuralNet layer styles in `plotneuralnet/layers` are also MIT-licensed (see `plotneuralnet/LICENSE.PlotNeuralNet`).
