"""IMU → optical axis calibration (3-pose geometric method).

Algorithm
---------
We take 3 still poses while reading the IMU quaternion q_i.
In each pose the operator points the telescope to a known world direction:

  Pose 0  optical axis → North   [0, 1, 0]   top → Up   [0, 0, 1]
  Pose 1  optical axis → East    [1, 0, 0]   top → Up   [0, 0, 1]
  Pose 2  optical axis → Zenith  [0, 0, 1]   (top direction unconstrained)

Convention (from pushto_engine.py):
    v_sensor = _qrotate(q, v_world)

So the optical axis expressed in sensor frame for pose i is:
    oz_i = _qrotate(q_i, d_i)

We average oz_i, then use poses 0 and 1 (known top→Up) to recover oy,
do a Gram-Schmidt orthonormalisation, build R_calib and convert to quaternion.

Persistence
-----------
~/.config/rpicamera2/finder/imu_calibration.json
{"q_calib": [w, x, y, z], "poses": [...], "date": "..."}
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Quaternion helpers (duplicated locally so this module has no circular deps)
# ---------------------------------------------------------------------------

def _qinv(q: list | np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qmul(q1, q2) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _qrotate(q, v) -> np.ndarray:
    """Rotate vector v by quaternion q  (v_sensor = _qrotate(q, v_world))."""
    v = np.asarray(v, dtype=float)
    qv = np.array([0.0, *v])
    return _qmul(_qmul(q, qv), _qinv(q))[1:]


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → [w, x, y, z] quaternion (Shepperd method)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return np.array([0.25 / s,
                         (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s,
                         (R[1, 0] - R[0, 1]) * s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s,
                         0.25 * s,
                         (R[0, 1] + R[1, 0]) / s,
                         (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s,
                         (R[0, 1] + R[1, 0]) / s,
                         0.25 * s,
                         (R[1, 2] + R[2, 1]) / s])
    s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[1, 0] - R[0, 1]) / s,
                     (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s,
                     0.25 * s])


# ---------------------------------------------------------------------------
# Pose definitions
# ---------------------------------------------------------------------------

@dataclass
class PoseDef:
    index:       int
    title:       str
    instruction: str
    axis_world:  list[float]   # optical axis direction in ENU world frame
    top_world:   Optional[list[float]]  # camera-top direction in ENU world frame (None = unconstrained)


POSES: list[PoseDef] = [
    PoseDef(
        index       = 0,
        title       = "Pose 1 — Zénith",
        instruction = "Pointez l'axe optique vers le Zénith\n(tube vertical, objectif en haut).",
        axis_world  = [0.0, 0.0, 1.0],   # Up in ENU
        top_world   = None,              # unconstrained
    ),
    PoseDef(
        index       = 1,
        title       = "Pose 2 — Est",
        instruction = "Pointez l'axe optique vers l'Est géographique,\ncaméra à l'horizontale (top → Zénith).",
        axis_world  = [1.0, 0.0, 0.0],   # East in ENU
        top_world   = [0.0, 0.0, 1.0],   # Up
    ),
    PoseDef(
        index       = 2,
        title       = "Pose 3 — Nord ★",
        instruction = "Pointez l'axe optique vers le Nord géographique,\ncaméra à l'horizontale (top → Zénith).\nRestez dans cette position — c'est l'ancre de pointage.",
        axis_world  = [0.0, 1.0, 0.0],   # North in ENU
        top_world   = [0.0, 0.0, 1.0],   # Up
    ),
]


# ---------------------------------------------------------------------------
# Captured snapshot
# ---------------------------------------------------------------------------

@dataclass
class PoseCapture:
    pose_index: int
    q_samples:  list[list[float]] = field(default_factory=list)
    t_capture:  Optional[float]   = None

    @property
    def q_mean(self) -> Optional[np.ndarray]:
        if not self.q_samples:
            return None
        arr = np.array(self.q_samples, dtype=float)
        # Simple average then re-normalise (valid for small angular spread)
        m = arr.mean(axis=0)
        return m / np.linalg.norm(m)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def compute_r_calib(captures: list[PoseCapture]) -> np.ndarray:
    """Given 3 pose captures, return R_calib quaternion [w,x,y,z].

    R_calib is the rotation that maps the IMU sensor frame to the camera
    optical frame, such that:
        v_optical = _qrotate(r_calib, v_sensor)

    Used by PushToEngine.set_r_calib().
    """
    if len(captures) < 3:
        raise ValueError("Need 3 pose captures")

    oz_vecs = []
    oy_vecs = []

    for cap in captures:
        pose = POSES[cap.pose_index]
        q    = cap.q_mean
        if q is None:
            raise ValueError(f"Pose {cap.pose_index} has no samples")

        # optical axis expressed in sensor frame
        oz_i = _qrotate(q, pose.axis_world)
        oz_i /= np.linalg.norm(oz_i)
        oz_vecs.append(oz_i)

        if pose.top_world is not None:
            # camera-top (Up→camera-up) expressed in sensor frame
            oy_i = _qrotate(q, pose.top_world)
            oy_i /= np.linalg.norm(oy_i)
            oy_vecs.append(oy_i)

    # Average optical axis
    oz = np.mean(oz_vecs, axis=0)
    oz /= np.linalg.norm(oz)

    # Average camera-top (from poses with top_world — East and North)
    if not oy_vecs:
        raise ValueError("No constrained top poses available")
    oy_raw = np.mean(oy_vecs, axis=0)

    # Gram-Schmidt: make oy orthogonal to oz
    oy = oy_raw - np.dot(oy_raw, oz) * oz
    norm_oy = np.linalg.norm(oy)
    if norm_oy < 1e-4:
        raise ValueError("oz and oy are nearly parallel — poses too similar")
    oy /= norm_oy

    # Third axis (East in camera frame)
    ox = np.cross(oy, oz)
    ox /= np.linalg.norm(ox)

    # R_so = sensor_from_optical: columns are optical basis vectors in sensor frame
    R_so = np.column_stack([ox, oy, oz])

    # Enforce proper rotation (det = +1)
    if np.linalg.det(R_so) < 0:
        ox  = -ox
        R_so = np.column_stack([ox, oy, oz])

    # r_calib must be R_optical_from_sensor (used as: q_opt = r_calib⊗Δq⊗r_calib⁻¹)
    R_os = R_so.T
    q_calib = _mat_to_quat(R_os)
    q_calib /= np.linalg.norm(q_calib)
    # Canonical form: w >= 0
    if q_calib[0] < 0:
        q_calib = -q_calib
    return q_calib


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_CALIB_PATH = os.path.expanduser(
    "~/.config/rpicamera2/finder/imu_calibration.json"
)


def save_calibration(q_calib: np.ndarray, captures: list[PoseCapture]) -> str:
    os.makedirs(os.path.dirname(_CALIB_PATH), exist_ok=True)
    data = {
        "q_calib": [round(float(v), 7) for v in q_calib],
        "date":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "poses": [
            {
                "index":     cap.pose_index,
                "n_samples": len(cap.q_samples),
                "q_mean":    [round(float(v), 7) for v in (cap.q_mean.tolist() if cap.q_mean is not None else [1,0,0,0])],
            }
            for cap in captures
        ],
    }
    with open(_CALIB_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return _CALIB_PATH


def load_calibration() -> Optional[np.ndarray]:
    """Return [w,x,y,z] or None if no calibration saved."""
    try:
        with open(_CALIB_PATH) as f:
            data = json.load(f)
        q = np.array(data["q_calib"], dtype=float)
        if q.shape == (4,):
            return q
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass
    return None


def load_calib_anchor() -> Optional[np.ndarray]:
    """Return q_north — IMU quaternion when optical axis pointed geographic North.

    This is the q_mean of pose 2 (Nord — last pose) saved by the wizard.
    Because calibration ends on North and is run at session start, this
    quaternion belongs to the current Mahony filter session — no cross-session
    yaw drift.
    Returns None if no calibration file or pose 2 missing.
    """
    try:
        with open(_CALIB_PATH) as f:
            data = json.load(f)
        for pose in data.get("poses", []):
            if pose.get("index") == 2:
                q = np.array(pose["q_mean"], dtype=float)
                if q.shape == (4,):
                    return q
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass
    return None
