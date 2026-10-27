# Orientation transfer: deployed vs orientation-robust Binary-SNGP (integer models)

Held-out windows: validation normal recording, test abnormal recording. Field mounting = training gravity rotated onto +z (DE0-Nano field test), 8 unknown yaw angles; 'gx_dead' also sets gyro x to 0.
Robust model: 5 seeds (mean ± std), 7 random orientations per window, gyro-axis dropout 0.3, DC removal in hardware, isotropic normalisation.

| Condition | Class | Deployed: abnormal | Deployed: unknown | Robust: abnormal | Robust: unknown |
|---|---|---|---|---|---|
| original_mount | normal | 0.0% | 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% |
| field_mount_gx_dead | normal | 100.0% | 98.8% | 0.0 ± 0.0% | 0.1 ± 0.2% |
| field_mount | normal | 100.0% | 97.4% | 0.0 ± 0.0% | 3.1 ± 3.0% |
| random_orientation | normal | 57.8% | 96.8% | 0.0 ± 0.0% | 0.8 ± 0.5% |
| original_mount | abnormal | 100.0% | 0.3% | 100.0 ± 0.0% | 0.9 ± 1.8% |
| field_mount_gx_dead | abnormal | 100.0% | 53.7% | 100.0 ± 0.0% | 2.4 ± 2.9% |
| field_mount | abnormal | 100.0% | 38.8% | 100.0 ± 0.0% | 0.3 ± 0.6% |
| random_orientation | abnormal | 82.9% | 55.8% | 100.0 ± 0.0% | 0.6 ± 0.5% |

Robust model, OOD AUROC of V (ID = orientation-augmented validation windows):

| OOD set | B-SNGP V | Mahalanobis on h | flagged unknown @TV |
|---|---|---|---|
| adi_cbm_acc_only | 1.000 ± 0.000 | 0.983 ± 0.010 | 100.0% |
| fault_noise_burst | 0.443 ± 0.089 | 0.455 ± 0.070 | 0.8% |
| fault_spikes | 0.618 ± 0.063 | 0.626 ± 0.056 | 37.9% |
| fault_stuck | 0.999 ± 0.001 | 0.986 ± 0.008 | 99.8% |
| fault_zero_output | 1.000 ± 0.000 | 0.983 ± 0.010 | 100.0% |
| motor_acc_only | 0.666 ± 0.112 | 0.616 ± 0.099 | 21.5% |
| uav_6dof | 0.943 ± 0.037 | 0.897 ± 0.039 | 81.1% |

Original-mount test accuracy (robust): 1.0000 ± 0.0000
Float vs integer mismatches on field-mount windows (seed 0): {'b3_bit_mismatch_frac': 0.0, 'decision_mismatch_frac': 0.0}
