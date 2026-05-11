from pathlib import Path
import argparse
import bisect
import csv


DEFAULT_FRAME_INDEX = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\frames_index.csv"
)
DEFAULT_OUTPUT = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs"
    r"\lidar_radar_matches.csv"
)


def read_frames(index_path: Path, sensor: str):
    frames = []

    with open(index_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sensor"] != sensor:
                continue
            if not row["timestamp"]:
                continue

            frames.append({
                "timestamp": int(row["timestamp"]),
                "path": row["path"],
            })

    frames.sort(key=lambda row: row["timestamp"])
    return frames


def nearest_frame(timestamp: int, candidates, candidate_timestamps):
    insert_at = bisect.bisect_left(candidate_timestamps, timestamp)
    best = None

    for index in (insert_at - 1, insert_at):
        if index < 0 or index >= len(candidates):
            continue

        candidate = candidates[index]
        delta_ns = candidate["timestamp"] - timestamp

        if best is None or abs(delta_ns) < abs(best["delta_ns"]):
            best = {
                "timestamp": candidate["timestamp"],
                "path": candidate["path"],
                "delta_ns": delta_ns,
            }

    return best


def match_lidar_to_radar(lidar_frames, radar_frames, max_delta_ms: float | None):
    radar_timestamps = [row["timestamp"] for row in radar_frames]
    matches = []

    for lidar in lidar_frames:
        radar = nearest_frame(lidar["timestamp"], radar_frames, radar_timestamps)
        if radar is None:
            continue

        delta_ms = radar["delta_ns"] / 1_000_000.0
        if max_delta_ms is not None and abs(delta_ms) > max_delta_ms:
            continue

        matches.append({
            "lidar_timestamp": lidar["timestamp"],
            "radar_timestamp": radar["timestamp"],
            "delta_ns": radar["delta_ns"],
            "delta_ms": f"{delta_ms:.6f}",
            "lidar_path": lidar["path"],
            "radar_path": radar["path"],
        })

    return matches


def write_matches(matches, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "lidar_timestamp",
            "radar_timestamp",
            "delta_ns",
            "delta_ms",
            "lidar_path",
            "radar_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)


def main():
    parser = argparse.ArgumentParser(
        description="Match HeRCULES Aeva LiDAR frames to nearest Continental radar frames."
    )
    parser.add_argument(
        "--frame-index",
        default=DEFAULT_FRAME_INDEX,
        help="CSV produced by list_frames.py.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output CSV containing nearest LiDAR/radar frame pairs.",
    )
    parser.add_argument(
        "--max-delta-ms",
        type=float,
        default=30.0,
        help="Discard matches with absolute timestamp difference above this value.",
    )
    args = parser.parse_args()

    frame_index = Path(args.frame_index)
    lidar_frames = read_frames(frame_index, "lidar_aeva")
    radar_frames = read_frames(frame_index, "radar_continental")

    if not lidar_frames:
        raise ValueError(f"No lidar_aeva frames found in {frame_index}")
    if not radar_frames:
        raise ValueError(f"No radar_continental frames found in {frame_index}")

    matches = match_lidar_to_radar(
        lidar_frames,
        radar_frames,
        max_delta_ms=args.max_delta_ms,
    )
    write_matches(matches, Path(args.output))

    print(f"LiDAR frames: {len(lidar_frames)}")
    print(f"Radar frames: {len(radar_frames)}")
    print(f"Matched pairs: {len(matches)}")
    print(f"Wrote {args.output}")

    if matches:
        deltas = [abs(float(row["delta_ms"])) for row in matches]
        print(f"Delta ms min: {min(deltas):.6f}")
        print(f"Delta ms max: {max(deltas):.6f}")
        print(f"Delta ms mean: {sum(deltas) / len(deltas):.6f}")
        print("\nFirst match:")
        first = matches[0]
        print(f"  LiDAR: {first['lidar_path']}")
        print(f"  Radar: {first['radar_path']}")
        print(f"  delta_ms: {first['delta_ms']}")


if __name__ == "__main__":
    main()
