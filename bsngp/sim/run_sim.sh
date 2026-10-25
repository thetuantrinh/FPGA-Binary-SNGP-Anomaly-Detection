#!/usr/bin/env bash
# Bit-exact RTL vs golden-model co-simulation with GHDL.
# Usage (from anywhere):  bash bsngp/sim/run_sim.sh
# Needs GHDL >= 4 on PATH (or GHDL=/path/to/ghdl) and Python 3 with numpy.
set -euo pipefail

GHDL=${GHDL:-ghdl}
PY=${PY:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
BS=$(dirname "$HERE")
WORK=$HERE/work
mkdir -p "$WORK"
cd "$WORK"
rm -f ./*.cf

STD=--std=93c
SRC=(
  "$BS/rtl/bsngp_params_pkg.vhd"
  "$BS/rtl/bsngp_demo_rom_pkg.vhd"
  "$BS/rtl/bsngp_core.vhd"
  "$BS/rtl/uart_rx.vhd"
  "$BS/rtl/uart_tx.vhd"
  "$BS/rtl/bsngp_de0_top.vhd"
  "$BS/tb/tb_bsngp_core.vhd"
  "$BS/tb/tb_bsngp_top.vhd"
)
for f in "${SRC[@]}"; do
  $GHDL -a $STD "$f"
done
# (mcode backend: -r elaborates and runs, so generics are given at run time)

N_EXP=$(wc -l < "$HERE/expected_stream.txt")
N_DEMO=$(wc -l < "$HERE/demo_expected.txt")
STATUS=0

echo "== 1/3 core: full co-simulation stream ($N_EXP windows)"
$GHDL -r $STD tb_bsngp_core -gSTREAM_FILE="$HERE/stream.txt" -gOUT_FILE="$WORK/rtl_stream.txt" -gGAP=80
$PY "$HERE/compare_rtl.py" "$WORK/rtl_stream.txt" "$HERE/expected_stream.txt" || STATUS=1

echo "== 2/3 top level, UART input -> UART result packets (first 256 samples = 7 windows)"
$GHDL -r $STD tb_bsngp_top -gMODE=1 -gSTREAM_FILE="$HERE/stream.txt" -gOUT_FILE="$WORK/rtl_top_uart.txt" \
      -gN_SAMPLES=256 -gN_RESULTS=7
$PY "$HERE/compare_rtl.py" "$WORK/rtl_top_uart.txt" "$HERE/expected_stream.txt" 7 || STATUS=1

echo "== 3/3 top level, stand-alone demo ROM (two loops = $N_DEMO windows)"
$GHDL -r $STD tb_bsngp_top -gMODE=0 -gSTREAM_FILE="$HERE/stream.txt" -gOUT_FILE="$WORK/rtl_top_rom.txt" \
      -gN_RESULTS="$N_DEMO"
$PY "$HERE/compare_rtl.py" "$WORK/rtl_top_rom.txt" "$HERE/demo_expected.txt" || STATUS=1

if [ $STATUS -eq 0 ]; then echo "ALL CO-SIMULATIONS PASS"; else echo "CO-SIMULATION FAILURES"; fi
exit $STATUS
