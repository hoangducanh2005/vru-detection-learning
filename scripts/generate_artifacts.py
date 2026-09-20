"""Sinh toàn bộ artifact từ đúng một NAVSIM front-camera frame.

Script này là điểm vào được khuyến nghị khi muốn so sánh các stage. Cùng một
``sample-index`` được truyền cho GT projection, LiDAR overlay, Phase 1 và Phase 2,
nhờ vậy người học không nhầm các output là những cảnh khác nhau.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import _bootstrap


def run_script(script_name: str, arguments: list[str]) -> None:
    """Chạy một script con và dừng ngay nếu stage đó gặp lỗi."""

    command = [sys.executable, str(_bootstrap.ROOT / "scripts" / script_name), *arguments]
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=_bootstrap.ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("artifacts/valid_samples.json"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    # KHỐI 1: Chọn duy nhất một ảnh nguồn và lưu bản gốc vào artifacts.
    payload = json.loads(args.index.read_text(encoding="utf-8"))
    sample = payload["samples"][args.sample_index]
    input_output = Path(f"artifacts/input_{args.sample_index:03d}.jpg")
    input_output.parent.mkdir(parents=True, exist_ok=True)
    # Không dùng copy2 vì sensor blob có thể mang thuộc tính read-only sang file
    # đích trên Windows, khiến lần chạy tiếp theo không thể ghi đè artifact.
    if input_output.exists():
        input_output.chmod(stat.S_IREAD | stat.S_IWRITE)
    shutil.copyfile(sample["camera_path"], input_output)
    # Dùng log ASCII để tương thích PowerShell cũ có console encoding cp1252.
    print(f"Input duy nhat: {input_output}")
    print(f"NAVSIM token: {sample['token']}")

    # KHỐI 2: Hai sanity-check hình học cùng dùng ảnh nguồn đã chọn.
    common = ["--index", str(args.index), "--sample-index", str(args.sample_index)]
    run_script("visualize_projected_gt.py", common)
    run_script("visualize_sparse_depth.py", common)

    # KHỐI 3: Hai inference stage cũng nhận đúng sample-index đó.
    inference_args = [
        *common,
        "--score-threshold", str(args.score_threshold),
        "--top-k", str(args.top_k),
    ]
    if Path("checkpoints/phase1_detector.pt").exists():
        run_script("infer_detection.py", inference_args)
    else:
        print("Bo qua Phase 1: chua co checkpoints/phase1_detector.pt")
    if Path("checkpoints/phase2_multitask.pt").exists():
        run_script("demo.py", inference_args)
    else:
        print("Bo qua Phase 2: chua co checkpoints/phase2_multitask.pt")


if __name__ == "__main__":
    main()
