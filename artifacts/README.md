# Artifacts của một input duy nhất

Tất cả ảnh trong thư mục này đều được tạo từ cùng một NAVSIM frame:

```text
sample-index: 0
token: 322cc2787c5d59c0
camera: CAM_F0
```

Luồng artifact:

```text
input_000.jpg                     ảnh camera gốc 1920x1080
    |
    +-- projected_gt_000.jpg      ground-truth 3D box -> bbox 2D
    |
    +-- sparse_depth_000.jpg      current LiDAR -> camera overlay
    |
    +-- phase1_detection_000.jpg  output detector ở 768x432
    |
    +-- demo_000.jpg              detection + depth + VRU risk ở 768x432
```

`valid_samples.json` là index metadata cho Dataset, không phải một ảnh output.

Sinh lại toàn bộ artifact từ input trên:

```powershell
python scripts/generate_artifacts.py --sample-index 0
```

Hai output model chỉ là smoke-test sau vài iteration, chưa đại diện accuracy.
