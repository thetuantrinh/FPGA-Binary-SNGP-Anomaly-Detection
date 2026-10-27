#!/usr/bin/env python3
"""Compare RTL simulation output ("logit var abnormal unknown" per line) with the golden model."""
import sys


def load(path):
    with open(path) as f:
        return [tuple(int(x) for x in line.split()) for line in f if line.strip()]


def main():
    rtl_path, exp_path = sys.argv[1], sys.argv[2]
    n_expected = int(sys.argv[3]) if len(sys.argv) > 3 else None
    rtl, exp = load(rtl_path), load(exp_path)
    if n_expected is not None:
        exp = exp[:n_expected]
    bad = [(i, r, e) for i, (r, e) in enumerate(zip(rtl, exp)) if r != e]
    ok = len(rtl) == len(exp) and not bad
    print(f"{rtl_path}: rtl={len(rtl)} expected={len(exp)} mismatches={len(bad)} -> {'PASS' if ok else 'FAIL'}")
    for i, r, e in bad[:10]:
        print(f"  window {i}: rtl={r} golden={e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
