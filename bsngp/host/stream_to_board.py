#!/usr/bin/env python3
"""
Stream an IMU recording to the DE0-Nano Binary-SNGP accelerator in real time and decode its results.

Frames sent (115200 8N1): 0x80, ax, ay, az, gx, gy, gz   (int8, each in [-127, 127])
Packets received (12 bytes): 0xA5, flags, seq, L(int32 BE), V(int32 BE), xor(bytes 1..10)
  flags bit0 abnormal (L > TL), bit1 unknown/OOD (V > TV), bit2 overrun, bit3 source=UART

Usage (board: SW0 = up/UART, press KEY0 to reset just before streaming so windows align):
  pip install pyserial numpy pandas
  python bsngp/host/stream_to_board.py --port /dev/ttyUSB0 \
      --csv dataset/valid/abnormal/2026-09-17_10-59-27.csv --check
  # Without a board, run the golden model on the same stream:
  python bsngp/host/stream_to_board.py --csv <file> --dry-run
"""

import os
import sys
import json
import math
import time
import argparse
import threading

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BS = os.path.dirname(HERE)
ROOT = os.path.dirname(BS)
sys.path.insert(0, BS)
import golden  # noqa: E402

CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
RENAME = {"ax_g": "ax", "ay_g": "ay", "az_g": "az", "gx_dps": "gx", "gy_dps": "gy", "gz_dps": "gz"}
MEAN_FIELD = math.pi / 8.0


def load_samples(csv_path, stats_path):
    df = pd.read_csv(csv_path).rename(columns=RENAME)
    raw = df[CHANNELS].to_numpy(np.float64)
    st = json.load(open(stats_path))
    z = (raw - np.asarray(st["means"])) / np.asarray(st["stds"])
    return golden.quantize_input(z)


def decode(L, V, logit_scale, ridge):
    """Mean-field SNGP probability from the integer outputs (host-side, for display/logging)."""
    var = max(V, 0) / (127.0 ** 2 * ridge)
    logit = L / logit_scale
    p = 1.0 / (1.0 + math.exp(-logit / math.sqrt(1.0 + MEAN_FIELD * var)))
    return p, var


class Reader(threading.Thread):
    def __init__(self, ser, on_packet):
        super().__init__(daemon=True)
        self.ser, self.on_packet, self.stop = ser, on_packet, False

    def run(self):
        buf = bytearray()
        while not self.stop:
            buf += self.ser.read(64)
            while len(buf) >= 12:
                if buf[0] != 0xA5:
                    del buf[0]
                    continue
                pkt = bytes(buf[:12])
                chk = 0
                for b in pkt[1:11]:
                    chk ^= b
                if chk != pkt[11]:
                    del buf[0]
                    continue
                del buf[:12]
                flags, seq = pkt[1], pkt[2]
                L = int.from_bytes(pkt[3:7], "big", signed=True)
                V = int.from_bytes(pkt[7:11], "big", signed=True)
                self.on_packet(seq, flags, L, V, time.perf_counter())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--stats", default=os.path.join(ROOT, "checkpoints", "norm_stats.json"))
    ap.add_argument("--params", default=os.path.join(BS, "export", "params.npz"))
    ap.add_argument("--rate", type=float, default=500.0, help="samples per second")
    ap.add_argument("--check", action="store_true", help="compare every board result with the golden model")
    ap.add_argument("--dry-run", action="store_true", help="no board: print golden-model results")
    ap.add_argument("--log", default=None, help="CSV log of results")
    args = ap.parse_args()

    x = load_samples(args.csv, args.stats)
    params = golden.load_params(args.params)
    logit_scale, ridge = float(params["logit_scale"]), float(params["ridge"])
    gold = golden.run(params, golden.stream_windows(x))
    print(f"{len(x)} samples -> {len(gold['logit'])} windows; TL={int(params['tl'])} TV={int(params['tv'])}")

    rows = []

    def show(i, flags, L, V, src):
        p, var = decode(L, V, logit_scale, ridge)
        state = ("UNKNOWN " if flags & 2 else "") + ("ABNORMAL" if flags & 1 else "normal")
        msg = f"[{src}] win {i:4d}  {state:17s} p(abn)={p:.3f}  var={var:8.4f}  L={L:+d} V={V:d}"
        if args.check and src == "board" and i < len(gold["logit"]):
            ok = (L == int(gold["logit"][i]) and V == int(gold["var"][i]))
            msg += "  golden " + ("OK" if ok else f"MISMATCH (L={int(gold['logit'][i])}, V={int(gold['var'][i])})")
        print(msg)
        rows.append({"window": i, "flags": flags, "L": L, "V": V, "p_abnormal": p, "var": var})

    if args.dry_run:
        for i in range(len(gold["logit"])):
            flags = int(gold["abnormal"][i]) | (int(gold["unknown"][i]) << 1)
            show(i, flags, int(gold["logit"][i]), int(gold["var"][i]), "golden")
    else:
        import serial
        if args.port is None:
            sys.exit("--port is required unless --dry-run")
        ser = serial.Serial(args.port, args.baud, timeout=0.05)
        count = {"n": 0}

        def on_packet(seq, flags, L, V, t):
            i = count["n"]
            count["n"] += 1
            show(i, flags, L, V, "board")
            if flags & 4:
                print("  warning: board reports overrun")

        reader = Reader(ser, on_packet)
        reader.start()
        period = 1.0 / args.rate
        t0 = time.perf_counter()
        for n, s in enumerate(x):
            frame = bytes([0x80]) + bytes(int(v) & 0xFF for v in s)
            target = t0 + n * period
            while True:
                dt = target - time.perf_counter()
                if dt <= 0:
                    break
                if dt > 0.002:
                    time.sleep(dt - 0.001)
            ser.write(frame)
        time.sleep(0.5)
        reader.stop = True
        reader.join(timeout=1.0)
        ser.close()
        print(f"sent {len(x)} samples in {time.perf_counter() - t0:.2f} s; received {count['n']} results "
              f"(expected {len(gold['logit'])})")

    if args.log:
        pd.DataFrame(rows).to_csv(args.log, index=False)


if __name__ == "__main__":
    main()
