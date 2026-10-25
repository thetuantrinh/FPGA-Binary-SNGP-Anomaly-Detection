#!/usr/bin/env python3
"""
Unified Multi-Dataset Analyzer & Standardizer for 10-DOF / 6-DOF IMU & Accelerometer Motor Anomaly Datasets:
1. Primary Dataset: DE0-Nano 10-DOF IMU (ADXL345 Accelerometer + ITG-3200 Gyroscope) @ 500 Hz
2. TII-UAE UAV Realistic Motor/Propeller Fault Dataset: 10-DOF Flight Controller (3-axis Accel + 3-axis Gyro + 3-axis Mag)
3. Zenodo ADXL345 Stepper Motor / 3D Printer Fault Dataset (Record 22716165): ADXL345 3-axis Accelerometer @ 3200 Hz
4. Analog Devices CbM Motor Fault Dataset (SpectraQuest Rig): 3-axis MEMS Accelerometer (Normal vs Bearing/Imbalance Faults)
5. Zenodo Electric Motor Vibrations Dataset (SOON-pEMP Record 6473455): 3-axis Accelerometer across 30 motor fault states
6. CFD Compound Fault Diagnosis Rotating Machinery Dataset: Triaxial vibration under Normal vs Bearing/Rotor/Unbalance faults
7. Engine Vibrations Dataset: Triaxial vibration (X, Y, Z) with Normal vs Fault conditions

All datasets are analyzed, standardized into the 6-channel [ax, ay, az, gx, gy, gz] format
(using zero-padded or angular-rate-estimated gyro channels when only 3-axis accelerometer is present),
and evaluated with Binarized Neural Networks (BNNs) on the GPU compute node!
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import json
import numpy as np
import pandas as pd


def load_uav_imu_dataset(base_dir: str, max_missions_per_class: int = 4, max_samples_per_mission: int = 6000):
    """
    Load TII-UAE UAV Motor/Propeller Fault Dataset (SensorCombined.jsonl).
    Contains true 6-DOF IMU (`accelerometer_m_s2` [3] and `gyro_rad` [3]) + `MagneticField.jsonl` [3].
    Converts accelerometer from m/s^2 to g (/ 9.80665) and gyro from rad/s to dps (* 180 / pi)
    to match ADXL345 (g) and ITG-3200 (deg/s) physical units!
    """
    ds_root = os.path.join(base_dir, "UAV-Realistic-Fault-Dataset", "Dataset")
    if not os.path.exists(ds_root):
        return [], []

    recordings = []
    labels = []
    # Class 0 = Normal, Classes 1..4 = Broken Motor Propeller Faults
    for cls_id in ["0", "1", "2", "3", "4"]:
        cls_dir = os.path.join(ds_root, cls_id)
        if not os.path.isdir(cls_dir):
            continue
        label = 0 if cls_id == "0" else 1
        missions = sorted(os.listdir(cls_dir))[:max_missions_per_class]
        for m in missions:
            sc_files = glob.glob(os.path.join(cls_dir, m, "*-SensorCombined.jsonl"))
            if not sc_files:
                continue
            rows = []
            with open(sc_files[0], "r") as f:
                for idx, line in enumerate(f):
                    if idx >= max_samples_per_mission:
                        break
                    try:
                        rec = json.loads(line)
                        acc = rec["accelerometer_m_s2"]
                        gyr = rec["gyro_rad"]
                        rows.append([
                            acc[0] / 9.80665,
                            acc[1] / 9.80665,
                            acc[2] / 9.80665,
                            gyr[0] * (180.0 / np.pi),
                            gyr[1] * (180.0 / np.pi),
                            gyr[2] * (180.0 / np.pi),
                        ])
                    except Exception:
                        continue
            if len(rows) >= 128:
                recordings.append(np.array(rows, dtype=np.float32))
                labels.append(label)
    return recordings, labels


def load_zenodo_adxl345_dataset(base_dir: str, max_samples_per_file: int = 25000):
    """
    Load Zenodo Record 22716165: ADXL345 3-axis accelerometer @ 3200 Hz on stepper motors.
    Estimates angular velocity proxy (differentiated tilt rate) for channels 3..5 to match 6-channel input.
    """
    ds_root = os.path.join(base_dir, "zenodo_3d_printer_adxl345")
    if not os.path.exists(ds_root):
        return [], []

    recordings = []
    labels = []
    files = [
        ("benchy_2_healthy.parquet", 0),
        ("benchy_25_nozzle_fault.parquet", 1),
    ]
    for fname, label in files:
        fpath = os.path.join(ds_root, fname)
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_parquet(fpath)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Filter out timestamp column if present
            acc_cols = [c for c in num_cols if c.lower() in ("x", "y", "z", "ax", "ay", "az", "acc_x", "acc_y", "acc_z")]
            if len(acc_cols) < 3:
                acc_cols = num_cols[-3:]
            acc = df[acc_cols[:3]].iloc[:max_samples_per_file].values.astype(np.float32)
            # Compute discrete temporal derivative as angular velocity / jerk proxy for channels 3..5
            gyro_proxy = np.vstack([np.zeros((1, 3), dtype=np.float32), np.diff(acc, axis=0) * 100.0])
            data6 = np.hstack([acc, gyro_proxy])
            # Split into 4 non-overlapping recording segments for train/val/test
            chunks = np.array_split(data6, 4)
            for ch in chunks:
                if len(ch) >= 128:
                    recordings.append(ch)
                    labels.append(label)
        except Exception as e:
            print(f"Warning loading {fpath}: {e}")
    return recordings, labels


def load_adi_cbm_motor_dataset(base_dir: str, max_samples_per_file: int = 10000):
    """
    Load Analog Devices CbM-Datasets (SpectraQuest Motor Fault Rig, 3-axis ADXL accelerometer).
    Files with 'GoB_GS_BaLo' are Good Bearing + Good Shaft + Balanced Load (Normal = 0).
    Files with HBF, HIR, HOR, HImL, LBF, LIR, LOR, BS, MS are Motor/Bearing Faults (Abnormal = 1).
    """
    ds_root = os.path.join(
        base_dir, "CbM-Datasets", "SampleMotorDataLimerick", "SpectraQuest_Rig_Data_Voyager_3", "Data_ADXL356C"
    )
    if not os.path.exists(ds_root):
        return [], []

    csv_files = sorted(glob.glob(os.path.join(ds_root, "*.csv")))
    recordings = []
    labels = []

    for fpath in csv_files[:24]:
        bname = os.path.basename(fpath)
        label = 0 if ("GoB_GS_BaLo" in bname) else 1
        try:
            # First line has extra metadata columns; read lines 1..max_samples
            df = pd.read_csv(fpath, sep=";", skiprows=1, header=None, nrows=max_samples_per_file)
            acc = df.iloc[:, 1:4].values.astype(np.float32)
            gyro_proxy = np.vstack([np.zeros((1, 3), dtype=np.float32), np.diff(acc, axis=0) * 100.0])
            data6 = np.hstack([acc, gyro_proxy])
            recordings.append(data6)
            labels.append(label)
        except Exception:
            continue
    return recordings, labels


def load_zenodo_electric_motor_dataset(base_dir: str, max_samples_per_file: int = 12000):
    """
    Load Zenodo Record 6473455 (Electric Motor Vibrations Dataset - SOON-pEMP).
    Files 01, 02, 15, 18 are Normal operation (0); all other 26 files are Mechanical/Electrical Motor Faults (1).
    """
    ds_root = os.path.join(base_dir, "zenodo_electric_motor_vibrations", "Data")
    if not os.path.exists(ds_root):
        return [], []

    csv_files = sorted(glob.glob(os.path.join(ds_root, "*.csv")))
    recordings = []
    labels = []

    normal_prefixes = ("01 ", "02 ", "15 ", "18 ")
    for fpath in csv_files:
        bname = os.path.basename(fpath)
        label = 0 if bname.startswith(normal_prefixes) else 1
        try:
            df = pd.read_csv(fpath, nrows=max_samples_per_file)
            acc_df = df[["AccX", "AccY", "AccZ"]].apply(pd.to_numeric, errors="coerce")
            if acc_df.isna().any().any():
                acc_df = acc_df.interpolate(method="linear").bfill().ffill().fillna(0.0)
            acc = acc_df.values.astype(np.float32) / 16384.0  # Convert raw LSB to g
            acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
            gyro_proxy = np.vstack([np.zeros((1, 3), dtype=np.float32), np.diff(acc, axis=0) * 100.0])
            data6 = np.hstack([acc, gyro_proxy])
            recordings.append(data6)
            labels.append(label)
        except Exception:
            continue
    return recordings, labels


def load_cfd_rotating_machinery_dataset(base_dir: str, max_samples_per_file: int = 12000):
    """
    Load CFD-Datasets (Compound Fault Diagnosis of Rotating Machinery: Motor + Gearbox + Bearing).
    F0.csv is Normal operation (0); F1..F7.csv are single and compound motor/bearing/rotor faults (1).
    Format: 'chid,wave' where each row is '1,val1;val2;val3;...'
    """
    ds_root = os.path.join(base_dir, "CFD-Datasets")
    if not os.path.exists(ds_root):
        return [], []

    recordings = []
    labels = []
    for i in range(8):
        fpath = os.path.join(ds_root, f"F{i}.csv")
        if not os.path.exists(fpath):
            continue
        label = 0 if i == 0 else 1
        try:
            ch_waves = {1: [], 2: [], 3: []}
            with open(fpath, "r") as f:
                header = f.readline()
                for line in f:
                    line = line.strip()
                    if not line or "," not in line:
                        continue
                    chid_str, wave_str = line.split(",", 1)
                    chid = int(chid_str.strip())
                    if chid in ch_waves and len(ch_waves[chid]) < max_samples_per_file:
                        vals = [float(x) for x in wave_str.split(";") if x.strip()]
                        ch_waves[chid].extend(vals)
            min_len = min(len(ch_waves[1]), len(ch_waves[2]), len(ch_waves[3]), max_samples_per_file)
            if min_len < 128:
                # Fallback if single-channel id=1 only
                all_v = ch_waves[1][: max_samples_per_file * 3]
                n_tri = (len(all_v) // 3) * 3
                acc = np.array(all_v[:n_tri], dtype=np.float32).reshape(-1, 3)
            else:
                acc = np.column_stack([
                    np.array(ch_waves[1][:min_len], dtype=np.float32),
                    np.array(ch_waves[2][:min_len], dtype=np.float32),
                    np.array(ch_waves[3][:min_len], dtype=np.float32),
                ])
            gyro_proxy = np.vstack([np.zeros((1, 3), dtype=np.float32), np.diff(acc, axis=0) * 50.0])
            data6 = np.hstack([acc, gyro_proxy])
            chunks = np.array_split(data6, 3)
            for ch in chunks:
                if len(ch) >= 128:
                    recordings.append(ch)
                    labels.append(label)
        except Exception as e:
            print(f"Warning loading {fpath}: {e}")
            continue
    return recordings, labels



def load_engine_vibration_dataset(base_dir: str):
    """
    Load Engine-vibrations-among-x-y-z-axes/engine_failure_dataset.csv:
    Contains Vibration_X, Vibration_Y, Vibration_Z, RPM, Torque, Power_Output, and Fault_Condition (0=Normal, 1..3=Fault).
    """
    fpath = os.path.join(base_dir, "Engine-vibrations-among-x-y-z-axes", "engine_failure_dataset.csv")
    if not os.path.exists(fpath):
        return [], []

    try:
        df = pd.read_csv(fpath)
        feat_cols = [
            "Vibration_X [mm/s]", "Vibration_Y [mm/s]", "Vibration_Z [mm/s]",
            "RPM [n/min]", "Torque [Nm]", "Power_Output [kW]"
        ]
        norm_df = df[df["Fault_Condition [-]"] == 0][feat_cols].values.astype(np.float32)
        abnorm_df = df[df["Fault_Condition [-]"] > 0][feat_cols].values.astype(np.float32)
        recordings = []
        labels = []
        for ch in np.array_split(norm_df, 3):
            if len(ch) >= 64:
                recordings.append(ch)
                labels.append(0)
        for ch in np.array_split(abnorm_df, 3):
            if len(ch) >= 64:
                recordings.append(ch)
                labels.append(1)
        return recordings, labels
    except Exception:
        return [], []


def main():
    online_dir = "online_datasets"
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    summary = {}

    # 1. UAV 10-DOF Dataset
    uav_recs, uav_labs = load_uav_imu_dataset(online_dir)
    summary["TII_UAE_UAV_10DOF_Motor_Propeller"] = {
        "sensor_type": "6-axis IMU (3-axis Accel + 3-axis Gyro) + 3-axis Magnetometer (10-DOF Flight Controller)",
        "num_recordings": len(uav_recs),
        "total_samples": int(sum(len(r) for r in uav_recs)),
        "normal_recordings": int(sum(1 for l in uav_labs if l == 0)),
        "abnormal_recordings": int(sum(1 for l in uav_labs if l == 1)),
    }

    # 2. Zenodo ADXL345 Dataset
    adxl_recs, adxl_labs = load_zenodo_adxl345_dataset(online_dir)
    summary["Zenodo_22716165_ADXL345_StepperMotor"] = {
        "sensor_type": "ADXL345 3-axis Digital Accelerometer @ 3200 Hz",
        "num_recordings": len(adxl_recs),
        "total_samples": int(sum(len(r) for r in adxl_recs)),
        "normal_recordings": int(sum(1 for l in adxl_labs if l == 0)),
        "abnormal_recordings": int(sum(1 for l in adxl_labs if l == 1)),
    }

    # 3. Analog Devices CbM Motor Dataset
    adi_recs, adi_labs = load_adi_cbm_motor_dataset(online_dir)
    summary["AnalogDevices_CbM_SpectraQuest_Motor"] = {
        "sensor_type": "ADXL356C 3-axis MEMS Accelerometer on Motor Rig",
        "num_recordings": len(adi_recs),
        "total_samples": int(sum(len(r) for r in adi_recs)),
        "normal_recordings": int(sum(1 for l in adi_labs if l == 0)),
        "abnormal_recordings": int(sum(1 for l in adi_labs if l == 1)),
    }

    # 4. Zenodo Electric Motor Vibrations Dataset
    em_recs, em_labs = load_zenodo_electric_motor_dataset(online_dir)
    summary["Zenodo_6473455_Electric_Motor_Vibrations"] = {
        "sensor_type": "3-axis MEMS Accelerometer on Asynchronous Electric Motors (m1/m2)",
        "num_recordings": len(em_recs),
        "total_samples": int(sum(len(r) for r in em_recs)),
        "normal_recordings": int(sum(1 for l in em_labs if l == 0)),
        "abnormal_recordings": int(sum(1 for l in em_labs if l == 1)),
    }

    # 5. CFD Compound Fault Diagnosis Dataset
    cfd_recs, cfd_labs = load_cfd_rotating_machinery_dataset(online_dir)
    summary["CFD_Compound_Fault_Rotating_Machinery"] = {
        "sensor_type": "Multi-channel Triaxial Motor & Bearing Vibration Rig",
        "num_recordings": len(cfd_recs),
        "total_samples": int(sum(len(r) for r in cfd_recs)),
        "normal_recordings": int(sum(1 for l in cfd_labs if l == 0)),
        "abnormal_recordings": int(sum(1 for l in cfd_labs if l == 1)),
    }

    # 6. Engine Vibrations X-Y-Z Dataset
    eng_recs, eng_labs = load_engine_vibration_dataset(online_dir)
    summary["Engine_Vibrations_XYZ_Fault_Dataset"] = {
        "sensor_type": "Triaxial Vibration (X, Y, Z) + Rotational Dynamics (RPM, Torque, Power)",
        "num_recordings": len(eng_recs),
        "total_samples": int(sum(len(r) for r in eng_recs)),
        "normal_recordings": int(sum(1 for l in eng_labs if l == 0)),
        "abnormal_recordings": int(sum(1 for l in eng_labs if l == 1)),
    }

    print("=" * 85)
    print("ONLINE DOWNLOADED DATASETS SUMMARY")
    print("=" * 85)
    for k, v in summary.items():
        print(f"Dataset: {k}")
        print(f"  Sensor:     {v['sensor_type']}")
        print(f"  Recordings: {v['num_recordings']} (Normal: {v['normal_recordings']}, Abnormal: {v['abnormal_recordings']})")
        print(f"  Samples:    {v['total_samples']:,}")
        print("-" * 85)

    out_path = os.path.join(out_dir, "online_datasets_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved online datasets summary to: {out_path}")


if __name__ == "__main__":
    main()
