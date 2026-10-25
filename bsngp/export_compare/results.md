# Uncertainty methods on the same binary backbone (5 seeds, mean ± std)

ID reference = validation windows; 'unknown' threshold = 99th percentile of the method's score on validation. MC Dropout: T=20, p=0.2. Deep Ensemble: M=5.

## Setting: robust

| Method | Mean OOD AUROC | uav_6dof | motor_acc_only | adi_cbm_acc_only | fault_stuck | fault_zero_output | fault_noise_burst | fault_spikes |
|---|---|---|---|---|---|---|---|---|
| Deterministic BNN (entropy) | 0.529 ± 0.183 | 0.416 | 0.505 | 0.415 | 0.548 | 0.415 | 0.675 | 0.733 |
| MC Dropout (T=20, MI) | 0.548 ± 0.024 | 0.475 | 0.561 | 0.476 | 0.576 | 0.466 | 0.573 | 0.708 |
| Deep Ensemble (M=5, MI) | 0.466 ± 0.116 | 0.273 | 0.504 | 0.272 | 0.478 | 0.272 | 0.727 | 0.739 |
| Mahalanobis / DDU on h | 0.592 ± 0.165 | 0.525 | 0.601 | 0.525 | 0.633 | 0.525 | 0.636 | 0.702 |
| **B-SNGP (ours, integer V)** | 0.810 ± 0.030 | 0.943 | 0.666 | 1.000 | 0.999 | 1.000 | 0.443 | 0.618 |

| Method | ID acc | ID ECE | field-mount normal: abnormal | field-mount normal: unknown | OOD flagged @99% (mean) | cycles/window | passes | weight bits | multipliers |
|---|---|---|---|---|---|---|---|---|---|
| Deterministic BNN (entropy) | 1.000 | 0.00% | 0.0% | 0.7% | 13.5% | 1,349 (estimate) | 1 | 5,160 | 1 |
| MC Dropout (T=20, MI) | 1.000 | 0.00% | 0.0% | 0.9% | 11.9% | 5,073 (estimate) | 20 | 5,160 | 1 |
| Deep Ensemble (M=5, MI) | 1.000 | 0.00% | 0.0% | 0.0% | 14.8% | 6,745 (estimate) | 5 | 25,800 | 1 |
| Mahalanobis / DDU on h | 1.000 | 0.00% | 0.0% | 0.2% | 12.3% | 1,417 (estimate) | 1 | 22,568 | 33 |
| **B-SNGP (ours, integer V)** | 1.000 | 1.06% | 0.0% | 0.1% | 63.0% | 1,971 (measured in RTL simulation) | 1 | 35,872 | 19 |

## Setting: original

| Method | Mean OOD AUROC | uav_6dof | motor_acc_only | adi_cbm_acc_only | fault_stuck | fault_zero_output | fault_noise_burst | fault_axis_swap | fault_spikes |
|---|---|---|---|---|---|---|---|---|---|
| Deterministic BNN (entropy) | 0.537 ± 0.081 | 0.338 | 0.691 | 0.118 | 0.355 | 0.334 | 0.986 | 0.599 | 0.875 |
| MC Dropout (T=20, MI) | 0.623 ± 0.083 | 0.455 | 0.600 | 0.465 | 0.511 | 0.431 | 0.976 | 0.692 | 0.852 |
| Deep Ensemble (M=5, MI) | 0.793 ± 0.075 | 0.748 | 0.962 | 0.633 | 0.547 | 1.000 | 0.995 | 0.577 | 0.880 |
| Mahalanobis / DDU on h | 0.514 ± 0.083 | 0.405 | 0.658 | 0.018 | 0.345 | 0.294 | 0.974 | 0.592 | 0.825 |
| **B-SNGP (ours, integer V)** | 0.982 ± 0.010 | 0.998 | 0.996 | 1.000 | 1.000 | 1.000 | 0.953 | 0.996 | 0.910 |

| Method | ID acc | ID ECE | field-mount normal: abnormal | field-mount normal: unknown | OOD flagged @99% (mean) | cycles/window | passes | weight bits | multipliers |
|---|---|---|---|---|---|---|---|---|---|
| Deterministic BNN (entropy) | 1.000 | 0.06% | 95.9% | 44.0% | 36.4% | 1,156 (estimate) | 1 | 5,160 | 1 |
| MC Dropout (T=20, MI) | 1.000 | 0.06% | 99.6% | 26.4% | 30.5% | 4,880 (estimate) | 20 | 5,160 | 1 |
| Deep Ensemble (M=5, MI) | 1.000 | 0.06% | 99.8% | 83.8% | 67.4% | 5,780 (estimate) | 5 | 25,800 | 1 |
| Mahalanobis / DDU on h | 1.000 | 0.06% | 95.9% | 44.7% | 35.4% | 1,224 (estimate) | 1 | 22,568 | 33 |
| **B-SNGP (ours, integer V)** | 1.000 | 5.21% | 91.1% | 99.4% | 87.4% | 1,778 (measured in RTL simulation) | 1 | 35,872 | 19 |
