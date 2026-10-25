# Binary-SNGP results (integer model = what the FPGA computes)

Main config: C=16/32/32, D=128 RFF, K=4, rank R=16, 5 seeds (mean ± std).

ID test accuracy: 1.0000 ± 0.0000. Windows flagged unknown (TV = 99th percentile of validation V): test 12.78%, val 1.04%. The test-normal recording (1 kHz logger, sensor nearly still) is itself atypical, so OOD AUROC is given against both ID references.

## OOD AUROC, ID = validation windows (clean reference)

| OOD set | B-SNGP V (int, R=16) | full-rank float var | Mahalanobis on h | -abs(logit) | FPR95 (B-SNGP) |
|---|---|---|---|---|---|
| adi_cbm_acc_only | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.0% |
| fault_axis_swap | 0.996 ± 0.006 | 0.998 ± 0.002 | 0.998 ± 0.001 | 0.999 ± 0.001 | 1.9% |
| fault_noise_burst | 0.953 ± 0.033 | 0.957 ± 0.031 | 0.951 ± 0.030 | 0.941 ± 0.046 | 20.8% |
| fault_spikes | 0.910 ± 0.044 | 0.918 ± 0.050 | 0.910 ± 0.041 | 0.907 ± 0.057 | 46.3% |
| fault_stuck | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.0% |
| fault_zero_output | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.0% |
| motor_acc_only | 0.996 ± 0.006 | 0.993 ± 0.012 | 0.996 ± 0.007 | 0.983 ± 0.016 | 2.6% |
| uav_6dof | 0.998 ± 0.001 | 0.998 ± 0.001 | 0.999 ± 0.001 | 0.988 ± 0.006 | 0.2% |

## OOD AUROC, ID = test windows (includes the atypical test-normal recording)

| OOD set | B-SNGP V (int, R=16) | full-rank float var | Mahalanobis on h | -abs(logit) | FPR95 (B-SNGP) |
|---|---|---|---|---|---|
| adi_cbm_acc_only | 0.950 ± 0.061 | 0.949 ± 0.062 | 0.973 ± 0.050 | 0.950 ± 0.060 | 5.1% |
| fault_axis_swap | 0.903 ± 0.004 | 0.905 ± 0.005 | 0.906 ± 0.008 | 0.944 ± 0.022 | 13.0% |
| fault_noise_burst | 0.857 ± 0.013 | 0.859 ± 0.012 | 0.857 ± 0.012 | 0.865 ± 0.035 | 20.6% |
| fault_spikes | 0.833 ± 0.018 | 0.838 ± 0.017 | 0.832 ± 0.016 | 0.827 ± 0.025 | 39.8% |
| fault_stuck | 0.962 ± 0.014 | 0.969 ± 0.010 | 0.965 ± 0.014 | 0.948 ± 0.015 | 12.7% |
| fault_zero_output | 0.975 ± 0.051 | 0.975 ± 0.051 | 0.975 ± 0.051 | 0.975 ± 0.051 | 2.5% |
| motor_acc_only | 0.876 ± 0.004 | 0.890 ± 0.023 | 0.881 ± 0.010 | 0.877 ± 0.018 | 12.8% |
| uav_6dof | 0.902 ± 0.026 | 0.906 ± 0.031 | 0.926 ± 0.040 | 0.880 ± 0.012 | 12.7% |

## Fraction of OOD windows flagged unknown by the hardware threshold TV

| OOD set | flagged |
|---|---|
| adi_cbm_acc_only | 100.0% |
| fault_axis_swap | 91.7% |
| fault_noise_burst | 64.2% |
| fault_spikes | 50.4% |
| fault_stuck | 100.0% |
| fault_zero_output | 100.0% |
| motor_acc_only | 94.9% |
| uav_6dof | 97.7% |

## Rank ablation (seed 0, AUROC of integer V vs validation ID)

| R | adi_cbm_acc_only | fault_axis_swap | fault_noise_burst | fault_spikes | fault_stuck | fault_zero_output | motor_acc_only | uav_6dof |
|---|---|---|---|---|---|---|---|---|
| 2 | 1.000 | 1.000 | 0.917 | 0.921 | 1.000 | 1.000 | 0.990 | 0.999 |
| 4 | 1.000 | 0.975 | 0.879 | 0.877 | 1.000 | 1.000 | 0.996 | 0.999 |
| 8 | 1.000 | 0.979 | 0.895 | 0.887 | 1.000 | 1.000 | 0.996 | 0.999 |
| 16 | 1.000 | 0.986 | 0.900 | 0.897 | 1.000 | 1.000 | 0.996 | 1.000 |
| 32 | 1.000 | 0.996 | 0.903 | 0.884 | 1.000 | 1.000 | 0.996 | 1.000 |
| 64 | 1.000 | 0.996 | 0.851 | 0.834 | 1.000 | 1.000 | 0.990 | 1.000 |
| 128 | 1.000 | 0.839 | 0.552 | 0.649 | 1.000 | 1.000 | 0.918 | 0.986 |

## RFF frequency sweep (K = 2^ksh, AUROC vs validation ID; sensitivity only, not used for model selection)

| K | ID acc | adi_cbm_acc_only | fault_axis_swap | fault_noise_burst | fault_spikes | fault_stuck | fault_zero_output | motor_acc_only | uav_6dof |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.278 | 0.812 | 0.890 | 0.762 | 0.638 | 0.159 | 0.469 | 0.591 |
| 2 | 1.000 | 0.709 | 0.917 | 0.941 | 0.904 | 0.975 | 0.988 | 0.672 | 0.984 |
| 4 | 1.000 | 1.000 | 0.993 | 0.951 | 0.926 | 1.000 | 1.000 | 0.999 | 0.998 |
| 8 | 0.923 | 1.000 | 0.997 | 0.950 | 0.934 | 1.000 | 1.000 | 1.000 | 1.000 |

## Hardware operation counts (seed 0)

```
{
  "binary_weight_bits": 4896,
  "rff_weight_bits": 4096,
  "int8_head_values": 2176,
  "layer1_addsub": 18432,
  "layer2_xnor_bits": 49152,
  "layer3_xnor_bits": 49152,
  "head_mults": 2320,
  "rtl_cycles_estimate": 1780
}
```
