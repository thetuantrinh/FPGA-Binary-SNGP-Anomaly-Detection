#!/usr/bin/env python3
"""
Mounting-orientation tools for transferring the recorded dataset to a new sensor mounting.

All functions work on raw physical windows (N, 6, 64): ax, ay, az [g], gx, gy, gz [deg/s].
A rigid re-mounting rotates accelerometer and gyroscope by the same rotation R.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def rotate(raw: np.ndarray, R: np.ndarray) -> np.ndarray:
    """raw: (N, 6, L); R: (N, 3, 3) or (3, 3). Rotates accel and gyro triads together."""
    if R.ndim == 2:
        R = np.broadcast_to(R, (raw.shape[0], 3, 3))
    out = np.empty_like(raw, dtype=np.float64)
    out[:, 0:3] = np.einsum("nij,njt->nit", R, raw[:, 0:3])
    out[:, 3:6] = np.einsum("nij,njt->nit", R, raw[:, 3:6])
    return out


def gravity_direction(raw_train: np.ndarray) -> np.ndarray:
    g = raw_train[:, 0:3].mean(axis=(0, 2))
    return g / np.linalg.norm(g)


def field_mount_rotations(g_train: np.ndarray, g_field=(0.0, 0.0, 1.0), n_yaw=8):
    """
    Rotations that move the training gravity direction onto the field gravity direction
    (DE0-Nano field test: a_z = +0.97 g), for n_yaw unknown rotations about the field gravity axis.
    """
    g_field = np.asarray(g_field, float) / np.linalg.norm(g_field)
    R0, _ = Rotation.align_vectors([g_field], [g_train])
    out = []
    for k in range(n_yaw):
        yaw = Rotation.from_rotvec(g_field * (2 * np.pi * k / n_yaw))
        out.append((yaw * R0).as_matrix())
    return out


def field_sim(raw: np.ndarray, g_train: np.ndarray, dead_gyro_axis=0, n_yaw=8) -> np.ndarray:
    """Recorded windows as they would look in the field mounting (all yaws), with one dead gyro axis."""
    outs = []
    for R in field_mount_rotations(g_train, n_yaw=n_yaw):
        r = rotate(raw, R)
        if dead_gyro_axis is not None:
            r[:, 3 + dead_gyro_axis] = 0.0
        outs.append(r)
    return np.concatenate(outs)


def random_rotation(raw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    R = Rotation.random(len(raw), random_state=int(rng.integers(2 ** 31))).as_matrix()
    return rotate(raw, R)


def augment(raw: np.ndarray, y: np.ndarray, copies: int, p_gyro_drop: float, seed: int = 0):
    """Original + `copies` uniformly random orientations; each copy may lose one gyro axis (reads 0)."""
    rng = np.random.default_rng(seed)
    xs, ys = [raw.astype(np.float64)], [y]
    for _ in range(copies):
        r = random_rotation(raw, rng)
        drop = rng.random(len(r)) < p_gyro_drop
        axis = rng.integers(0, 3, size=len(r))
        r[drop, 3 + axis[drop]] = 0.0
        xs.append(r)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def isotropic_stats(raw_train: np.ndarray):
    """
    One scale for the 3 accel axes and one for the 3 gyro axes (rotation-equivariant), zero mean.
    The static part is removed in hardware per window (DC_REMOVE), so the mean only centres the int8 range.
    """
    sd = raw_train.transpose(1, 0, 2).reshape(6, -1).std(axis=1)
    sa = float(np.sqrt(np.mean(sd[0:3] ** 2)))
    sg = float(np.sqrt(np.mean(sd[3:6] ** 2)))
    return np.zeros(6), np.array([sa, sa, sa, sg, sg, sg])
