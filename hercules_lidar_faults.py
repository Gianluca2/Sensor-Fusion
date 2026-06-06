import math

import numpy as np


SUPPORTED_HERCULES_LIDAR_FAULTS = [
    "laser",
    "photodetector",
    "scanning",
    "optical",
    "window",
    "mounting",
]


def spherical_angles(xyz: np.ndarray):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r = np.linalg.norm(xyz, axis=1)
    r_xy = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    elev = np.arctan2(z, np.maximum(r_xy, 1e-9))
    return r, theta, elev


def severity_params(severity: str):
    if severity == "mild":
        return {
            "range_q": 0.90,
            "drop_base": 0.05,
            "drop_range": 0.20,
            "noise_frac": 0.02,
            "fov_fraction": 0.90,
            "az_step_deg": 1.0,
            "el_step_deg": 0.5,
            "distortion": 0.30,
            "jitter_deg": 0.20,
            "edge_dropout": 0.20,
            "window_fraction": 0.20,
            "window_drop": 0.85,
            "yaw_deg": 1.0,
            "pitch_deg": 0.5,
            "roll_deg": 0.5,
        }
    if severity == "severe":
        return {
            "range_q": 0.60,
            "drop_base": 0.20,
            "drop_range": 0.70,
            "noise_frac": 0.08,
            "fov_fraction": 0.60,
            "az_step_deg": 4.0,
            "el_step_deg": 2.0,
            "distortion": 0.90,
            "jitter_deg": 0.80,
            "edge_dropout": 0.60,
            "window_fraction": 0.50,
            "window_drop": 0.97,
            "yaw_deg": 4.0,
            "pitch_deg": 2.0,
            "roll_deg": 2.0,
        }
    return {
        "range_q": 0.75,
        "drop_base": 0.10,
        "drop_range": 0.40,
        "noise_frac": 0.05,
        "fov_fraction": 0.75,
        "az_step_deg": 2.0,
        "el_step_deg": 1.0,
        "distortion": 0.60,
        "jitter_deg": 0.40,
        "edge_dropout": 0.40,
        "window_fraction": 0.35,
        "window_drop": 0.93,
        "yaw_deg": 2.0,
        "pitch_deg": 1.0,
        "roll_deg": 1.0,
    }


def keep_points(points: np.ndarray, keep: np.ndarray) -> np.ndarray:
    return points[keep].astype(np.float32, copy=True)


def add_noise_points(points: np.ndarray, noise_count: int, rng: np.random.Generator):
    if noise_count <= 0 or len(points) == 0:
        return points

    xyz = points[:, :3]
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    noise_xyz = rng.uniform(mins, maxs, size=(noise_count, 3)).astype(np.float32)

    source_indices = rng.integers(0, len(points), size=noise_count)
    noise = points[source_indices].astype(np.float32, copy=True)
    noise[:, :3] = noise_xyz
    if noise.shape[1] > 3:
        noise[:, 3:] = 0.0
    if noise.shape[1] > 7:
        noise[:, 7] = rng.uniform(0.0, 0.2, size=noise_count).astype(np.float32)

    return np.vstack([points, noise]).astype(np.float32)


def fault_laser(points: np.ndarray, severity: str, rng: np.random.Generator):
    params = severity_params(severity)
    xyz = points[:, :3]
    r = np.linalg.norm(xyz, axis=1)
    threshold = np.quantile(r, params["range_q"])
    return keep_points(points, r <= threshold)


def fault_photodetector(points: np.ndarray, severity: str, rng: np.random.Generator):
    params = severity_params(severity)
    xyz = points[:, :3]
    r = np.linalg.norm(xyz, axis=1)
    r_min = float(r.min())
    r_max = float(r.max())
    r_norm = (r - r_min) / max(r_max - r_min, 1e-6)

    if points.shape[1] > 7:
        intensity = points[:, 7]
    elif points.shape[1] > 3:
        intensity = points[:, 3]
    else:
        intensity = np.ones_like(r_norm)

    i_low, i_high = np.percentile(intensity, [1, 99])
    intensity_norm = np.clip((intensity - i_low) / max(i_high - i_low, 1e-6), 0.0, 1.0)
    drop_probability = params["drop_base"] + params["drop_range"] * r_norm * (1.0 - intensity_norm)
    keep = rng.random(len(points)) > np.clip(drop_probability, 0.0, 0.98)
    kept = keep_points(points, keep)
    return add_noise_points(kept, int(len(kept) * params["noise_frac"]), rng)


def fault_scanning(points: np.ndarray, severity: str, rng: np.random.Generator):
    params = severity_params(severity)
    xyz = points[:, :3]
    r, theta, elev = spherical_angles(xyz)

    range_threshold = np.quantile(r, params["range_q"])
    keep = r <= range_threshold

    theta_center = 0.5 * (float(theta.min()) + float(theta.max()))
    elev_center = 0.5 * (float(elev.min()) + float(elev.max()))
    theta_half = 0.5 * (float(theta.max()) - float(theta.min())) * params["fov_fraction"]
    elev_half = 0.5 * (float(elev.max()) - float(elev.min())) * params["fov_fraction"]

    keep &= np.abs(theta - theta_center) <= theta_half
    keep &= np.abs(elev - elev_center) <= elev_half
    output = keep_points(points, keep)
    if len(output) == 0:
        return output

    r_out, theta_out, elev_out = spherical_angles(output[:, :3])
    theta_step = math.radians(params["az_step_deg"])
    elev_step = math.radians(params["el_step_deg"])
    theta_q = np.round(theta_out / theta_step) * theta_step
    elev_q = np.round(elev_out / elev_step) * elev_step

    output[:, 0] = r_out * np.cos(elev_q) * np.cos(theta_q)
    output[:, 1] = r_out * np.cos(elev_q) * np.sin(theta_q)
    output[:, 2] = r_out * np.sin(elev_q)
    return output.astype(np.float32)


def fault_optical(points: np.ndarray, severity: str, rng: np.random.Generator):
    params = severity_params(severity)
    output = points.astype(np.float32, copy=True)
    xyz = output[:, :3]
    r, theta, elev = spherical_angles(xyz)

    theta_center = 0.5 * (float(theta.min()) + float(theta.max()))
    elev_center = 0.5 * (float(elev.min()) + float(elev.max()))
    theta_half = 0.5 * (float(theta.max()) - float(theta.min())) + 1e-6
    elev_half = 0.5 * (float(elev.max()) - float(elev.min())) + 1e-6

    theta_delta = theta - theta_center
    elev_delta = elev - elev_center
    edge_distance = np.sqrt((theta_delta / theta_half) ** 2 + (elev_delta / elev_half) ** 2)
    edge_distance = np.clip(edge_distance, 0.0, 1.0)

    factor = 1.0 + params["distortion"] * edge_distance**2
    theta_distorted = theta_center + theta_delta * factor
    elev_distorted = elev_center + elev_delta * factor

    jitter = math.radians(params["jitter_deg"])
    theta_distorted += jitter * edge_distance * rng.standard_normal(len(theta))
    elev_distorted += jitter * edge_distance * rng.standard_normal(len(elev))

    drop_probability = params["edge_dropout"] * edge_distance**2
    keep = rng.random(len(output)) > np.clip(drop_probability, 0.0, 0.98)
    output = output[keep]
    r = r[keep]
    theta_distorted = theta_distorted[keep]
    elev_distorted = elev_distorted[keep]
    output[:, 0] = r * np.cos(elev_distorted) * np.cos(theta_distorted)
    output[:, 1] = r * np.cos(elev_distorted) * np.sin(theta_distorted)
    output[:, 2] = r * np.sin(elev_distorted)
    return output.astype(np.float32)


def fault_window(points: np.ndarray, severity: str, rng: np.random.Generator):
    params = severity_params(severity)
    y = points[:, 1]
    y_min = float(y.min())
    y_max = float(y.max())
    width = y_max - y_min
    if width <= 1e-6:
        return points.astype(np.float32, copy=True)

    band_width = params["window_fraction"] * width
    margin = 0.10 * width
    center = rng.uniform(y_min + margin, y_max - margin)
    y_start = center - 0.5 * band_width
    y_end = center + 0.5 * band_width

    inside = (y >= y_start) & (y <= y_end)
    kill = inside & (rng.random(len(points)) < params["window_drop"])
    return keep_points(points, ~kill)


def rotation_matrix(yaw: float, pitch: float, roll: float):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    return rz @ ry @ rx


def fault_mounting(points: np.ndarray, severity: str, rng: np.random.Generator):
    params = severity_params(severity)
    output = points.astype(np.float32, copy=True)
    yaw = math.radians(params["yaw_deg"] * (0.5 + rng.random()) * rng.choice([-1, 1]))
    pitch = math.radians(params["pitch_deg"] * (0.5 + rng.random()) * rng.choice([-1, 1]))
    roll = math.radians(params["roll_deg"] * (0.5 + rng.random()) * rng.choice([-1, 1]))
    output[:, :3] = (rotation_matrix(yaw, pitch, roll) @ output[:, :3].T).T
    return output


def apply_hercules_lidar_fault(
    aeva_points: np.ndarray,
    fault_type: str,
    severity: str = "moderate",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if fault_type not in SUPPORTED_HERCULES_LIDAR_FAULTS:
        raise ValueError(
            f"Unsupported HeRCULES LiDAR fault '{fault_type}'. "
            f"Choose one of: {SUPPORTED_HERCULES_LIDAR_FAULTS}"
        )
    if rng is None:
        rng = np.random.default_rng()
    if len(aeva_points) == 0:
        return aeva_points.astype(np.float32, copy=True)

    if fault_type == "laser":
        return fault_laser(aeva_points, severity, rng)
    if fault_type == "photodetector":
        return fault_photodetector(aeva_points, severity, rng)
    if fault_type == "scanning":
        return fault_scanning(aeva_points, severity, rng)
    if fault_type == "optical":
        return fault_optical(aeva_points, severity, rng)
    if fault_type == "window":
        return fault_window(aeva_points, severity, rng)
    if fault_type == "mounting":
        return fault_mounting(aeva_points, severity, rng)

    raise AssertionError(f"Unhandled fault type: {fault_type}")
