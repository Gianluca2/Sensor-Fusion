from pathlib import Path
import argparse
import csv
from datetime import datetime


def infer_sensor(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    suffix = path.suffix.lower()

    if suffix == ".bin":
        if "aeva" in parts:
            return "lidar_aeva"
        if "continentalobject" in parts or "continental_object" in parts:
            return "radar_continental_object"
        if "continental" in parts:
            return "radar_continental"
        return "pointcloud_unknown"

    if suffix in [".png", ".jpg", ".jpeg"]:
        if "stereo_left" in parts or "left" in parts:
            return "camera_left"
        if "stereo_right" in parts or "right" in parts:
            return "camera_right"
        return "image_unknown"

    if suffix in [".txt", ".csv"]:
        if "calibration" in parts:
            return "calibration"
        if "pr_gt" in parts or name.endswith("_gt.txt"):
            return "ground_truth"
        return "metadata"

    return "unknown"


def timestamp_from_filename(path: Path):
    try:
        return int(path.stem)
    except ValueError:
        return None


def collect_frames(data_root: Path):
    extensions = {".bin", ".png", ".jpg", ".jpeg", ".csv", ".txt"}
    rows = []

    for path in data_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        rows.append({
            "sensor": infer_sensor(path),
            "timestamp": timestamp_from_filename(path),
            "extension": path.suffix.lower(),
            "path": str(path),
        })

    rows.sort(key=lambda r: (r["sensor"], r["timestamp"] or -1, r["path"]))
    return rows


def write_index(rows, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sensor", "timestamp", "extension", "path"])
            writer.writeheader()
            writer.writerows(rows)
        return output
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = output.with_name(f"{output.stem}_{timestamp}{output.suffix}")

        with open(fallback, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sensor", "timestamp", "extension", "path"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"Could not overwrite {output}")
        print("It is probably open in Excel, Preview, or locked by OneDrive sync.")
        return fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\Data",
        help="Path to HeRCULES data folder",
    )
    parser.add_argument(
        "--output",
        default=r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\frames_index.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    rows = collect_frames(data_root)
    output = write_index(rows, Path(args.output))

    print(f"Found {len(rows)} files")
    print(f"Wrote {output}")

    counts = {}
    for row in rows:
        counts[row["sensor"]] = counts.get(row["sensor"], 0) + 1

    print("\nFrame counts:")
    for sensor, count in sorted(counts.items()):
        print(f"  {sensor}: {count}")

    print("\nExamples:")
    seen = set()

    for row in rows:
        sensor = row["sensor"]
        if sensor in seen:
            continue

        print(f"  {sensor}: {row['path']}")
        seen.add(sensor)


if __name__ == "__main__":
    main()
