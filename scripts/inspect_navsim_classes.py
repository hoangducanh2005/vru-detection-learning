"""In toàn bộ annotation name tìm thấy trong các NAVSIM log local."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs", type=Path,
        default=Path(r"D:\navsim_workspace\dataset\navsim_logs\mini"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts: Counter[str] = Counter()
    frame_count = 0
    for log_path in sorted(args.logs.glob("*.pkl")):
        # Đây là pickle nguồn do NAVSIM cung cấp; script chỉ đọc và không ghi cache.
        with log_path.open("rb") as stream:
            frames = pickle.load(stream)
        for frame in frames:
            counts.update(map(str, frame["anns"]["gt_names"]))
            frame_count += 1
    print(f"logs: {args.logs}")
    print(f"frames: {frame_count}")
    print("unique annotation names:")
    for name, count in counts.most_common():
        print(f"  {name:<24} {count}")


if __name__ == "__main__":
    main()
