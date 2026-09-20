# Vehicle / VRU Detection Learning

Repository PyTorch nhỏ để học luồng dữ liệu autonomous driving từ một frame NAVSIM thật đến:

1. detector 2D hai lớp `VEHICLE / VRU`;
2. depth camera-Z theo mét, giám sát thưa bằng LiDAR;
3. khoảng cách object và nhãn demo `CRITICAL / NEAR / FAR` cho VRU.

Đây là code học kiến trúc, **không phải safety-grade perception system** và không nhằm đạt SOTA.

## Pipeline

Phase 1:

```text
NAVSIM raw frame (CAM_F0 + 3D annotations)
            |
            +-- 3D box LiDAR frame --lidar2cam + K--> 2D GT box
            |
        Dataset / DataLoader
            |
      RGB [B,3,432,768]
            |
         ResNet34
     C2 / C3 / C4 / C5
            |
            FPN
   feature [B,128,108,192]
            |
  CenterNet detection head
   /          |          \
heatmap     offset       size
[B,2,h,w] [B,2,h,w]  [B,2,h,w]
            |
     VEHICLE / VRU + 2D box
```

Phase 2 dùng cùng feature, không thay detector:

```text
                  FPN stride-4 feature
                    /              \
          Detection Head          Depth Head
          box + class score      64-bin logits
                    \              /
                    object distance
                           |
                 VRU risk threshold demo
                 CRITICAL / NEAR / FAR
```

Không có 8-camera fusion, BEV, IPM, 3D detector, occupancy, planning hay motion prediction.

## Dữ liệu local đã kiểm tra

Mặc định các script đọc:

```text
D:\navsim_workspace\dataset\navsim_logs\mini
D:\navsim_workspace\dataset\sensor_blobs\mini
```

Ngày 2026-09-21, subset local có 2 log, 1.020 frame và cả 1.020 frame đều có `CAM_F0` lẫn `MergedPointCloud`. Class thực tế:

| NAVSIM name | Số annotation | Mapping |
|---|---:|---|
| `vehicle` | 21.765 | `VEHICLE` (0) |
| `pedestrian` | 10.555 | `VRU` (1) |
| `bicycle` | 28 | `VRU` (1) |
| `generic_object` | 55.575 | `IGNORE` |
| `traffic_cone` | 3.471 | `IGNORE` |
| `barrier` | 360 | `IGNORE` |
| `czone_sign` | 29 | `IGNORE` |

Chạy lại thống kê thay vì tin vào bảng:

```powershell
python scripts/inspect_navsim_classes.py
```

**VRU không được xác định bằng kích thước bbox.** Khi tạo nhãn training, VRU đến từ taxonomy ground truth (`pedestrian`, `bicycle`). Model sau đó học feature thị giác tương ứng. Đây là bài toán khác với risk classification: detector trước tiên trả lời “object có phải VRU không?”, rồi logic khoảng cách mới trả lời “VRU này gần tới mức nào?”.

## Phase 0: frame tọa độ và 2D GT

Các convention đã được đối chiếu với NAVSIM source local:

- `navsim/common/enums.py`: box là `[x, y, z, length, width, height, heading]`;
- `navsim/common/dataclasses.py`: `anns.gt_boxes/gt_names` trở thành `Annotations.boxes/names`;
- `navsim/visualization/camera.py`: NAVSIM nghịch đảo `sensor2lidar_rotation` khi đưa LiDAR/annotation sang camera.

Metadata `sensor2lidar` có chiều **camera → LiDAR**:

```text
p_lidar = R_camera_to_lidar @ p_camera + t_camera_to_lidar
```

Ta có điểm/box trong LiDAR frame, nên cần chiều ngược lại:

```text
p_camera = inverse(R_camera_to_lidar) @ (p_lidar - t_camera_to_lidar)
```

Sau đó chỉ giữ `z_camera > 0` và project bằng intrinsic `K`:

```text
[u', v', w'] = K @ [x_camera, y_camera, z_camera]
u = u' / w'
v = v' / w'
```

`src/data/projection.py` tạo 8 corner của 3D box, transform, project, lấy extent 2D, intersect và clip theo ảnh. Index được build bằng:

```powershell
python scripts/build_index.py
python scripts/visualize_projected_gt.py --sample-index 0
```

`artifacts/valid_samples.json` chỉ chứa path, token, calibration và 2D projected labels (khoảng 3,5 MB trên subset hiện tại). Nó **không chứa decoded image hoặc LiDAR array**. `Dataset.__getitem__()` mới đọc JPEG/PCD.

Visualization box và LiDAR overlay là kiểm tra quan trọng nhất cho extrinsic:

```powershell
python scripts/visualize_sparse_depth.py --sample-index 0
```

Kết quả local ở sample 0 có 21.022 LiDAR point trong ảnh và camera-Z dương khoảng 3,03–70,70 m. Nếu point/box không bám cảnh, không nên training; hãy sửa calibration/convention trước.

### Một input duy nhất cho toàn bộ artifacts

Các ảnh minh họa trong `artifacts/` đều xuất phát từ **cùng một ảnh camera trước**:

```text
sample-index: 0
NAVSIM token: 322cc2787c5d59c0
input: artifacts/input_000.jpg
```

```text
input_000.jpg
    ├── projected_gt_000.jpg       3D GT box -> bbox 2D
    ├── sparse_depth_000.jpg       LiDAR -> camera depth overlay
    ├── phase1_detection_000.jpg   detector prediction
    └── demo_000.jpg               detection + depth + VRU risk
```

`input_000.jpg` giữ resolution gốc 1920×1080. Hai output model được resize về
768×432 theo config, nhưng nội dung vẫn là đúng frame đó. Sinh lại toàn bộ bằng:

```powershell
python scripts/generate_artifacts.py --sample-index 0
```

Script truyền cùng một `sample-index` cho mọi stage, tránh vô tình so sánh output
từ các cảnh khác nhau.

## Dataset và CenterNet targets

Ảnh 1920×1080 được resize thành 768×432 bằng `INTER_AREA`. Box được scale bởi `sx, sy`. Nếu dùng intrinsic sau resize, `fx,cx` nhân `sx`, còn `fy,cy` nhân `sy` (`scale_intrinsic`).

Mỗi sample trả:

```python
{
    "image": Tensor[3, 432, 768],
    "targets": {
        "heatmap": Tensor[2, 108, 192],
        "offset": Tensor[2, 108, 192],
        "size": Tensor[2, 108, 192],
        "mask": Tensor[1, 108, 192],
    },
    "raw_boxes": Tensor[N, 4],
    "raw_labels": Tensor[N],
    "meta": {"token": ..., "camera_path": ...},
}
```

Ý nghĩa target:

- **heatmap**: tâm object ở đâu và thuộc lớp nào; Gaussian quanh tâm giúp supervision bớt quá “mỏng”;
- **offset**: tâm thật thường nằm giữa hai cell stride-4, nên head học phần lẻ để sửa quantization;
- **size**: width/height của bbox theo đơn vị feature-map;
- **mask**: L1 chỉ tính đúng tại cell tâm object.

Loss:

```text
L_det = L_heatmap_focal + lambda_offset * L_offset_L1
                          + lambda_size * L_size_L1
```

Decode làm `sigmoid → local max 3x3 → top-K → class/center → offset/size → XYXY`. Local maximum đóng vai trò suppression đơn giản; repo không thêm một NMS phức tạp.

## Backbone và FPN là gì?

**Backbone** ResNet34 biến pixel thành feature ngày càng giàu ngữ nghĩa. `C2` có độ phân giải cao, còn `C5` có receptive field lớn nhưng thô.

**FPN** dùng lateral 1×1 conv để đưa C2–C5 về 128 channel, rồi upsample và cộng từ C5 xuống C2. Output stride-4 vì VRU nhỏ cần spatial detail, nhưng vẫn nhận semantic context từ tầng sâu. Version đầu chỉ detect ở một scale stride-4 để code dễ theo dõi; multi-scale detection là một hướng mở rộng, không nằm trong scope hiện tại.

## Sparse metric depth

`src/data/depth_projection.py` đọc PCD ở `__getitem__`, transform LiDAR → camera bằng đúng extrinsic trên, giữ `z_camera > 0`, project bằng `K` và giữ return gần nhất khi nhiều point rơi vào cùng pixel stride-4.

Depth head không nói trực tiếp “17,3 m”. Với 64 bin trong 1–80 m, nó tạo probability distribution, ví dụ:

```text
10 m: 0.05
15 m: 0.30
20 m: 0.60
25 m: 0.05
```

Metric depth là weighted expectation `sum(probability_i * bin_center_i)`. Ground truth là bin chứa depth LiDAR. Cross entropy **chỉ tính tại pixel có LiDAR mask**; repo không tự nội suy thành dense pseudo-GT.

Multi-task loss:

```text
L_total = L_detection + lambda_depth * L_sparse_depth
```

Depth ở đây là `z_camera` (độ sâu theo optical axis), không phải Euclidean range 3D tới sensor.

## Object distance và near-VRU

Lấy toàn bbox dễ trộn background. `estimate_box_distance` lấy median trong vùng lower-middle: 50% giữa theo chiều ngang và 40% dưới theo chiều dọc; nếu vùng không hợp lệ thì fallback center pixel.

Threshold demo trong config:

```text
VRU distance <= 10 m          -> CRITICAL
10 m < VRU distance <= 20 m   -> NEAR
VRU distance > 20 m           -> FAR
```

Vehicle vẫn có distance nhưng không nhận near-VRU status. Quy tắc này chỉ để minh họa workflow, không có uncertainty, temporal filtering, braking model hay safety validation.

## Cài đặt và chạy

Trong PowerShell, từ root repository:

```powershell
python -m pip install -r requirements.txt

python scripts/inspect_navsim_classes.py
python scripts/build_index.py
python scripts/visualize_projected_gt.py --sample-index 0
python scripts/visualize_sparse_depth.py --sample-index 0

python scripts/test_dataloader.py --batch-size 2
python scripts/test_dataloader.py --batch-size 1 --with-depth

python scripts/train_detection.py --max-samples 50 --epochs 1 --batch-size 2 --device auto
python scripts/infer_detection.py --sample-index 0

python scripts/train_multitask.py --max-samples 50 --epochs 1 --batch-size 2 `
  --load-phase1 checkpoints/phase1_detector.pt --device auto
python scripts/demo.py --sample-index 0

python -m pytest -q
```

Smoke checkpoint đi kèm workspace chỉ được overfit vài update trên một frame để chứng minh backward/save/load chạy. Nó không đại diện accuracy. Với checkpoint chưa train đủ, có thể hạ threshold chỉ để debug decode, nhưng kết quả không nên được diễn giải như detector tốt:

```powershell
python scripts/infer_detection.py --sample-index 0 --score-threshold 0.01 --top-k 20
python scripts/demo.py --sample-index 0 --score-threshold 0.01 --top-k 20
```

`num_workers=0` là mặc định an toàn cho Windows smoke test. Có thể tăng sau khi pipeline ổn định.

## Cấu trúc code

```text
configs/                 taxonomy, resolution, loss/risk thresholds
src/data/                projection, PCD/depth, targets, Dataset
src/model/               ResNet34, FPN, detection/depth heads, full model
src/losses/              detection and sparse depth loss
src/utils/               decode, drawing, distance/risk
scripts/                 inspect/build/visualize/train/infer/demo
tests/                   real-data, tensor/loss and end-to-end tests
artifacts/               small index and rendered sanity/demo images
checkpoints/             generated Phase 1/2 weights
```

## Relation to METEOR

Project lấy cảm hứng ở mức ý tưởng từ image backbone, depth distribution và detection workflow của METEOR. Đây **không phải full METEOR**. Repo cố ý không có:

- 8-camera fusion;
- depth-gated IPM;
- BEV;
- 3D BEV detection;
- occupancy, motion prediction hoặc planning stack.

Code raw PyTorch và một stride-4 head được chọn để người học nhìn thấy từng tensor/loss thay vì bị framework lớn che khuất.

## Possible Phase 3 (chỉ conceptual)

Một phase sau có thể dùng 8 camera với shared backbone, dự đoán depth distribution, depth-gated IPM để lift feature vào BEV, rồi Vehicle/VRU 3D detector. Phase đó cần calibration/visibility/occlusion evaluation kỹ hơn và không được implement trong repository này.

## Known limitations

- Taxonomy phản ánh subset local quan sát được; class mới phải inspect rồi map có chủ đích.
- 2D GT là rectangle bao quanh các corner 3D nhìn thấy, không phải amodal/instance mask 2D được gán tay.
- Object cắt near-plane dùng các corner phía trước, đủ cho demo nhưng không phải polygon clipping đầy đủ.
- Không augmentation, validation split, metric mAP/depth, scheduler, mixed precision hay distributed training.
- Sparse LiDAR có occlusion/misalignment theo thời gian và không supervise mọi pixel.
- Distance lấy từ predicted camera-Z depth; chưa hiệu chỉnh uncertainty và không phải safety distance.
- Checkpoint smoke vài iteration không có giá trị accuracy/generalization.
