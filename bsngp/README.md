# Binary-SNGP: real-time anomaly detection with uncertainty on the DE0-Nano

A 1-bit binary 1D-CNN with an SNGP-style Gaussian-process output layer. For every 64-sample IMU window it reports:
- **abnormal / normal**, from the GP mean logit `L`;
- **unknown**, from the GP posterior variance `V` (input unlike the training data).

The FPGA computes exactly the integer function in `golden.py`. GHDL co-simulation checks that bit-for-bit.

## Status

| Item | Status |
|---|---|
| Training (5 seeds) | done, `export/results.md` |
| Float model vs integer spec | 0 bit / 0 decision mismatches, all seeds |
| RTL vs golden model (GHDL 4.1) | **PASS**: 204/204 windows (core), 7/7 UART packets (top, UART mode), 31/31 windows (top, ROM mode) |
| Synthesizability (GHDL `--synth`, VHDL-93) | passes, no latches, no warnings |
| Measured latency (RTL) | 1,778 cycles per window = **35.6 µs @ 50 MHz** (window period 64 ms → 0.06 % duty) |
| Quartus fit / Fmax / LE count | **not run** (Quartus is not installed here); see "Open items" |
| Board test | not done (needs Quartus + board) |

## Model (config: `train_eval.py`, exported to `export/params.npz`)

```
x[64][6] int8 ──► L1 binary conv 6→16 (k3 s1)   add/sub only
             ──► L2 binary conv 16→32 (k3 s2)  XNOR+popcount
             ──► L3 binary conv 32→32 (k3 s2)  XNOR+popcount ──► h[32] = Σt bits ∈ [-16,16]
   BN+sign folded into one integer compare per channel (TH, DIR)
GP head (D=128 random features, rank R=16):
   idx_j = (Σc ±h_c · 4 + BR_j) mod 1024 ;  φ_j = COS_LUT[idx_j]         (no multipliers)
   L = Σ BETA_j φ_j + BETA0                                              (logit)
   V = Σ φ_j² − Σ_k ((U_k · φ) >> 7)²                                   (≈ 127²·λ·posterior var)
   abnormal = L > TL ,  unknown = V > TV   (TV = 99th pct of V on validation)
```

How the head differs from textbook SNGP, and why:
- **Random-feature projection:** weights are ±1 and the phase is quantised to 1/1024 of a turn, so the projection is add/subtract plus one ROM read.
- **Covariance:** `(λI+ΦᵀΦ)⁻¹` is applied through its top-R eigen-directions. Dropping directions only removes subtracted terms, so `V` can over-estimate but never under-estimate the variance (fail-safe).
- **Rank:** the R ablation shows R=16–32 beats full rank in integer arithmetic, because int8 rounding of small eigen-directions adds noise.
- **No spectral normalisation:** sign activations make the backbone scale-invariant, so spectral normalisation has no effect. Distance awareness comes from the GP head only.

Memory: 4,896 binary conv weights, 4,096 RFF sign bits, 2,176 int8 head values, and a 1,024×8 cos ROM.
Multipliers: 18 parallel 8×8 (β·φ, φ², 16×U·φ) plus one 16×16 for `Q²`.

## Results (integer model, 5 seeds, mean ± std)

In-distribution test accuracy is 1.000. That says little: normal and abnormal recordings differ about 4× in vibration level, so the in-distribution task is easy.

**OOD AUROC of `V`, with in-distribution = validation recordings:**

| OOD set | B-SNGP `V` (int) | Mahalanobis on h | −abs(logit) |
|---|---|---|---|
| other machine: UAV 6-DoF IMU | 0.998 ± 0.001 | 0.999 | 0.988 |
| other machine: motor (acc only*) | 0.996 ± 0.006 | 0.996 | 0.983 |
| other machine: ADI CbM (acc only*) | 1.000 | 1.000 | 1.000 |
| sensor stuck | 1.000 | 1.000 | 1.000 |
| sensor zero output | 1.000 | 1.000 | 1.000 |
| sensor axis swap | 0.996 ± 0.006 | 0.998 | 0.999 |
| noise burst | 0.953 ± 0.033 | 0.951 | 0.941 |
| spikes | 0.910 ± 0.044 | 0.910 | 0.907 |

\* The loader synthesises the gyro channels of these sets (`diff(acc)*100`), so they are easy, artefact-driven OOD cases.

How to read these numbers:
- **Hardware threshold TV:**
  - On OOD sets it flags 100 % of stuck, zero and ADI windows, 98 % of UAV, 92 % of axis-swap, 64 % of noise and 50 % of spike windows.
  - On the validation recordings it flags 1 % of windows.
- **The test-normal recording is flagged 100 % "unknown".** That file (`valid/normal/2026-09-17_10-57-42.csv`) comes from a different logger at 1 kHz with the sensor almost still, so the GP is correct that it is out of distribution. It also caps the AUROCs measured against the test split (second table in `export/results.md`).
- **The cheap Mahalanobis-on-h baseline matches B-SNGP here.** The GP head's advantage is that its decision is a calibrated probability, which the host computes via `p = σ(L/s / √(1+π/8·var))`, in a single hardware pass. These data do not show a higher AUROC for it.
- **K is a lengthscale-like setting:** K=1–2 is too smooth (poor OOD), and K=8 costs in-distribution accuracy. K=4 was fixed before the sweep.

## Files

| Path | What |
|---|---|
| `golden.py` | integer reference model = hardware spec |
| `bsngp_model.py` | PyTorch training model, BN folding, quantisation, low-rank covariance |
| `data.py` | ID splits, OOD sets (other machines, synthetic sensor faults), co-sim stream, demo ROM stream |
| `train_eval.py` | train 5 seeds + K sweep + rank ablation; evaluates the integer model; writes `export/` |
| `export_vhdl.py` | `params.npz` → `rtl/bsngp_params_pkg.vhd`, `rtl/bsngp_demo_rom_pkg.vhd`, `sim/*.txt` |
| `rtl/bsngp_core.vhd` | accelerator: ring buffer, 3 binary conv layers, GP head |
| `rtl/bsngp_de0_top.vhd` | DE0-Nano top: UART frames in, result packets out, demo ROM, LEDs |
| `rtl/uart_rx.vhd`, `rtl/uart_tx.vhd` | copied from `../de0_nano/` |
| `tb/`, `sim/run_sim.sh`, `sim/compare_rtl.py` | GHDL co-simulation against the golden model |
| `quartus/` | `.qpf/.qsf/.sdc` for EP4CE22F17C6 |
| `host/stream_to_board.py` | PC tool: real-time 500 Hz streaming, packet decoding, live golden check |

## Reproduce

```bash
# from the repository root
python bsngp/train_eval.py              # train + evaluate + export (CUDA GPU)
python bsngp/export_vhdl.py             # generate VHDL + co-simulation vectors
bash   bsngp/sim/run_sim.sh             # bit-exact GHDL co-simulation
```

GHDL 4.x must be on `PATH` (or set `GHDL=/path/to/ghdl`).

## Running on the board

1. Open `quartus/bsngp_de0.qpf` in Quartus Prime Lite (Cyclone IV E is supported), compile, and program over USB-Blaster.
2. **Stand-alone:** set SW0 down. The on-chip ROM replays normal → abnormal → zero-output sensor in a loop at 500 Hz (SW1 up = 50 Hz slow motion).
   - LED0 = abnormal, LED1 = unknown.
   - Expected sequence: normal (LED0 off, LED1 off) → abnormal (LED0 on) → sensor fault (LED0 on, LED1 on). Windows that straddle two segments also light LED1.
3. **Live stream:**
   - Connect a 3.3 V USB-UART adapter: adapter TX → JP1 pin 2 (GPIO_0[0]), adapter RX ← JP1 pin 4 (GPIO_0[1]), GND → JP1 pin 12. Check these pins against your board manual.
   - Set SW0 up and press KEY0, then run:
   ```bash
   python bsngp/host/stream_to_board.py --port /dev/ttyUSB0 --csv dataset/valid/abnormal/2026-09-17_10-59-27.csv --check
   ```
   - `--check` compares every board result with the golden model.
   - `--dry-run` prints the golden results without a board.

**UART protocol:**
- **In:** `0x80, ax, ay, az, gx, gy, gz`, where each value is `clip(round(16·z), −127, 127)` and `z` uses `checkpoints/norm_stats.json`. That is 3.5 kB/s, 30 % of 115200 baud.
- **Out:** a 12-byte packet `0xA5, flags, seq, L(int32 BE), V(int32 BE), xor`.

## Open items

- **Quartus compile:** get real LE / M9K / multiplier counts and Fmax. The design is sequential with roughly 1,778 cycles of work in 3.2 M available, so timing slack at 50 MHz is expected but unmeasured. The widest single-cycle paths are the 32-channel XNOR-popcount + accumulate (L3) and the 32-input add tree in the head.
- **Board measurements:** end-to-end latency, measured with a logic analyzer on the LED0 edge versus the last UART byte, and board power.
- **Data:** a new, independently recorded test set. The current one is a single session, and its test-normal file is atypical.
- **Baselines:** compare against Mahalanobis on the same features in hardware. Its cost is similar (C3² MACs), and on these data its OOD AUROC is the same.

## Mounting-orientation transfer (fix for the field test in `../report_run.pdf` §8)

**Symptom:** on the fan, every window was "abnormal + unknown".

**Cause:** the IMU was re-mounted.
- Gravity now points along +z instead of (−0.07, −0.45, −0.89) g.
- Gyro x reads 0.
- The per-axis normalization amplifies a vibration that moves onto a different axis. Its σ ranges from 10.7 to 47 deg/s depending on the axis.

`transfer_orientation.py` reproduces the failure in software and fixes it using only the existing recordings:
1. **Isotropic normalization:** one σ for the accel axes (0.999 g), one for the gyro axes (29.06 deg/s), zero mean. See `export_robust/norm_stats_isotropic.json`.
2. **Per-window DC removal in the core** (`DC_REMOVE = true`): x ← x − floor(mean₆₄(x)). Gravity and offsets disappear for any mounting. Cost: +193 cycles.
3. **Rotation augmentation:** 7 uniformly random 3-D orientations per training window, with accel and gyro rotated together, plus 30 % gyro-axis dropout. The GP covariance and TV are fitted on the augmented data.

Results (`export_robust/results.md`, integer models, robust = 5 seeds). Held-out windows are the validation normal and test abnormal recordings. "Field mount" means the training gravity rotated onto +z, at 8 unknown yaw angles:

| Condition | Class | Deployed model: abnormal / unknown | Robust model: abnormal / unknown |
|---|---|---|---|
| original mount | normal | 0 % / 0 % | 0 % / 0 % |
| field mount, gyro x dead | normal | **100 % / 99 %** (the board's behaviour) | **0 % / 0.1 %** |
| field mount, gyro x dead | abnormal | 100 % / 54 % | 100 % / 2.4 % |
| random orientation | normal | 58 % / 97 % | 0 % / 0.8 % |
| random orientation | abnormal | 83 % / 56 % | 100 % / 0.6 % |

The cost is weaker OOD detection.
- **Still detected:** stuck sensor (AUROC 0.999), zero output (1.000), UAV (0.94).
- **No longer detected:**
  - noise bursts: 0.44
  - spikes: 0.62
  - acc-only motor data: 0.67

  Once orientation and gain are free, these look like ordinary vibration.

RTL: co-simulation PASS with 167/167, 7/7 and 31/31 windows, including field-mount segments. Latency is 1,971 cycles = 39.4 µs. The deployed model (`DC_REMOVE = false`) still passes on the same core, at 1,778 cycles.

### What to change on the board

1. Generate the VHDL with `python bsngp/export_vhdl.py --src export_robust`. Copy `rtl/bsngp_core.vhd` and `rtl/bsngp_params_pkg.vhd` into the Quartus project. The core's ports are unchanged.
2. Regenerate the IMU input constants A_c, B_c from `export_robust/norm_stats_isotropic.json` instead of `checkpoints/norm_stats.json`, using the same formula: mean 0, σ = 0.9993 g for the accel axes and 29.059 deg/s for the gyro axes.
3. PC tools: pass `--stats bsngp/export_robust/norm_stats_isotropic.json --params bsngp/export_robust/params.npz`.

**Limitation:** this transfer covers orientation, offsets and a dead gyro axis. It cannot know whether the fan on the new mount vibrates as much as the training rig. Check this before relying on it:
- log about 30 s of raw IMU data with the fan healthy at 100 %;
- run `host/stream_to_board.py --csv <log> --dry-run --stats ... --params ...`.

If healthy windows still come out "abnormal", the vibration level differs. In that case a short recording of normal operation on the new mount is needed. It can be used either for a fine-tune or to refit only the GP covariance and the thresholds.

## Comparison with other uncertainty methods (`compare_uncertainty.py`, `export_compare/results.md`)

All methods share the same binary backbone, data, OOD sets and 5 seeds. The mean OOD AUROC is over 7–8 OOD sets.

| Method | Robust setting | Original setting | Passes | Cycles/window |
|---|---|---|---|---|
| Deterministic BNN (entropy) | 0.529 ± 0.183 | 0.537 ± 0.081 | 1 | ~1,350 (est.) |
| MC Dropout, T=20 (MI) | 0.548 ± 0.024 | 0.623 ± 0.083 | 20 | ~5,000 (est.) |
| Deep Ensemble, M=5 (MI) | 0.466 ± 0.116 | 0.793 ± 0.075 | 5 | ~6,700 (est.) |
| Mahalanobis/DDU on the deterministic BNN's h | 0.592 ± 0.165 | 0.514 ± 0.083 | 1 | ~1,420 (est.) |
| **B-SNGP (integer V)** | **0.810 ± 0.030** | **0.982 ± 0.010** | 1 | 1,971 (RTL) |

Caveats:
- **Mahalanobis depends on which backbone's features it uses.** Applied to the B-SNGP backbone's own features it matches B-SNGP (earlier tables). Applied to a backbone trained with a plain linear head it collapses. So part of the gain comes from the GP head shaping the features during training.
- **No method's hyperparameters were tuned.**
- **The ID task is trivial, so ECE is not informative.** B-SNGP's higher ECE (1–5 %) is under-confidence at 100 % accuracy, which comes from the mean-field softening.
- **Costs are estimates** except B-SNGP, which was measured in RTL.
