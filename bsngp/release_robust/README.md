# Binary-SNGP, orientation-robust: VHDL release

Model: `params.npz`, seed 0 from `bsngp/transfer_orientation.py`.
- Isotropic normalization.
- Per-window DC removal in hardware (`DC_REMOVE = true`).
- Trained with rotation augmentation, so it works for any sensor mounting.

Verified in GHDL bit-exact against `golden.py`: 167/167 windows (core), 7/7 UART packets and 31/31 demo-ROM windows. Latency is 1,971 cycles = 39.4 µs at 50 MHz.

| File | Use |
|---|---|
| `rtl/bsngp_core.vhd` | accelerator core (same ports as before) |
| `rtl/bsngp_params_pkg.vhd` | weights and thresholds of this model (generated) |
| `rtl/bsngp_de0_top.vhd`, `rtl/bsngp_demo_rom_pkg.vhd`, `rtl/uart_*.vhd` | reference DE0-Nano top (UART in/out, demo ROM, LEDs) |
| `quartus/` | project for EP4CE22F17C6 using the reference top |
| `norm_stats_isotropic.json` | input normalization: mean 0; σ = 0.9993 g (ax, ay, az), 29.059 deg/s (gx, gy, gz) |
| `host/stream_to_board.py` | PC streaming, packet decoding, golden check (`--dry-run` works without a board) |

## Into your own board design (IMU, fan, LCD)

1. Replace `bsngp_core.vhd` and `bsngp_params_pkg.vhd`. The ports are unchanged.
2. Regenerate the IMU input constants from `norm_stats_isotropic.json`, using the same input code q = clip(round(16·z), −127, 127). The input code formula is the same as before; only the means and σ change.
3. Before relying on the decisions, log about 30 s of healthy fan data and run:
   `python host/stream_to_board.py --csv log.csv --dry-run`
