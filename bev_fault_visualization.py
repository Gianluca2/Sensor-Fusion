import numpy as np


def make_input_preview(x: np.ndarray) -> np.ndarray:
    # LiDAR-only preview: red=voxel count, green=max height, blue=point density.
    red = x[2] if x.shape[0] > 2 else np.zeros_like(x[0])
    lidar_height = x[1] if x.shape[0] > 1 else np.zeros_like(x[0])
    lidar_density = x[0]
    rgb = np.stack([red, lidar_height, lidar_density], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def bounding_box(mask: np.ndarray):
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def draw_box(rgb: np.ndarray, box, color):
    output = np.array(rgb, copy=True)
    if box is None:
        return output

    row_start, row_end, col_start, col_end = box
    output[row_start:row_end + 1, col_start, :] = color
    output[row_start:row_end + 1, col_end, :] = color
    output[row_start, col_start:col_end + 1, :] = color
    output[row_end, col_start:col_end + 1, :] = color
    return output


def overlay_masks(rgb: np.ndarray, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    output = np.array(rgb, copy=True)

    actual_bool = actual.astype(bool)
    predicted_bool = predicted.astype(bool)
    overlap = actual_bool & predicted_bool
    actual_only = actual_bool & ~predicted_bool
    predicted_only = predicted_bool & ~actual_bool

    output[actual_only] = [255, 0, 0]
    output[predicted_only] = [0, 0, 255]

    output = draw_box(output, bounding_box(actual_bool), [255, 0, 0])
    output = draw_box(output, bounding_box(predicted_bool), [0, 0, 255])
    output[overlap] = [255, 0, 255]
    return output


def probability_heatmap(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 0.0, 1.0)
    heatmap = np.zeros((*probs.shape, 3), dtype=np.uint8)
    heatmap[..., 0] = np.clip(probs * 255.0, 0, 255).astype(np.uint8)
    heatmap[..., 1] = np.clip((1.0 - np.abs(probs - 0.5) * 2.0) * 180.0, 0, 180).astype(np.uint8)
    heatmap[..., 2] = np.clip((1.0 - probs) * 80.0, 0, 80).astype(np.uint8)
    return heatmap


def mask_iou(predicted: np.ndarray, actual: np.ndarray):
    predicted = predicted.astype(bool)
    actual = actual.astype(bool)
    intersection = np.logical_and(predicted, actual).sum()
    union = np.logical_or(predicted, actual).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)
