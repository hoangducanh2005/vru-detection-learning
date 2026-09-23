# NAVSIM Vehicle / VRU Detection + Metric Depth


Input là current frame của NAVSIM. Output gồm:

- bbox 2D và confidence của `VEHICLE / VRU`;
- metric depth theo trục Z camera;
- khoảng cách object;
- trạng thái `CRITICAL / NEAR / FAR` cho VRU.


## CODE chính 
[src/model/vru_model.py]
Image
  → ResNet34 backbone
  → FPN
  → Detection Head
  → heatmap / offset / size

## Kiến trúc

```text
NAVSIM current frame
    ├── CAM_F0 image
    ├── 3D annotations
    ├── current LiDAR
    └── camera calibration
              |
              v
      Project 3D GT -> bbox 2D
              |
              v
       Dataset / DataLoader
              |
              v
       RGB [B,3,432,768]
              |
              v
          ResNet34
       C2 / C3 / C4 / C5
              |
              v
     FPN [B,128,108,192]
          /           \
         /             \
Detection Head       Depth Head
 heatmap [B,2,H,W]   64 depth bins
 offset  [B,2,H,W]   sparse LiDAR supervision
 size    [B,2,H,W]   expected metric depth
         \             /
          \           /
       bbox + class + confidence
                  +
            object distance
                  |
                  v
       VRU: CRITICAL / NEAR / FAR
```

## Dữ liệu và taxonomy

Cam trước , current LiDAR và current annotations.

```text
vehicle              -> VEHICLE = 0
pedestrian, bicycle  -> VRU = 1
generic_object       -> IGNORE
traffic_cone         -> IGNORE
barrier              -> IGNORE
czone_sign           -> IGNORE
```

VRU được xác định bằng ground-truth taxonomy, không dùng kích thước bbox.

Ảnh và LiDAR được lazy-load trong `Dataset.__getitem__()`. File
`artifacts/valid_samples.json` chỉ chứa path, token, calibration và bbox nhỏ;
không chứa decoded image hoặc point-cloud arrays.

## Ground truth 2D

NAVSIM cung cấp bbox 3D:

```text
[x, y, z, length, width, height, heading]
```

Code tạo 8 đỉnh, đổi từ LiDAR sang camera, project bằng intrinsic `K`, sau đó
clip bbox theo kích thước ảnh.

```text
3D box in LiDAR
      -> lidar2cam
      -> camera XYZ, giữ z > 0
      -> intrinsic K
      -> bbox 2D
```

`projected_gt_000.jpg` là ground truth lấy từ NAVSIM, không phải prediction.

## Detection targets và loss

```text
heatmap  object ở đâu và thuộc class nào
offset   sửa sai số làm tròn tâm ở feature map stride 4
size     width và height của bbox
mask     cell nào được tính regression loss
```

```text
L_detection = L_heatmap_focal
            + lambda_offset * L_offset_L1
            + lambda_size * L_size_L1
```

Decode:

```text
sigmoid -> local maximum -> top-K
        -> class + center + offset + size
        -> bbox XYXY
```

## Depth và khoảng cách

LiDAR được project lên camera để tạo sparse metric-depth target. Depth loss chỉ
tính tại pixel có LiDAR supervision.

Depth head dự đoán xác suất trên 64 depth bins:

```text
metric depth = sum(probability_i * depth_bin_center_i)
```

```text
L_total = L_detection + lambda_depth * L_sparse_depth
```

Distance của object là median depth trong vùng lower-middle của bbox. Risk status
chỉ áp dụng cho VRU:

```text
distance <= 10 m         -> CRITICAL
10 m < distance <= 20 m -> NEAR
distance > 20 m          -> FAR
```

## Output

Tất cả artifact minh họa dùng cùng `input_000.jpg`:

```text
input_000.jpg
    ├── projected_gt_000.jpg       NAVSIM ground truth
    ├── sparse_depth_000.jpg       LiDAR depth overlay
    ├── phase1_detection_000.jpg   detection output
    ├── phase1_detection_000.json  class + bbox + confidence
    ├── demo_000.jpg               detection + depth + risk
    └── demo_000.json              kết quả đầy đủ
```

Một object trong `demo_000.json`:

```json
{
  "class": "VRU",
  "bbox": [230.73, 242.1, 235.85, 247.78],
  "center": [233.29, 244.94],
  "width": 5.12,
  "height": 5.68,
  "corners": [[230.73, 242.1], [235.85, 242.1], [235.85, 247.78], [230.73, 247.78]],
  "confidence": 0.0276,
  "distance_m": 39.08,
  "status": "FAR"
}
```

## Cấu trúc code

```text
configs/       model, loss, taxonomy và risk thresholds
src/data/      projection, targets, Dataset và sparse depth
src/model/     ResNet34, FPN, detection head và depth head
src/losses/    detection loss và depth loss
src/utils/     decode, distance và visualization
scripts/       inspect, visualization, training và inference
tests/         geometry, tensor shape, loss và end-to-end
artifacts/     input, visualization và JSON outputs
```

## Lưu ý

- Checkpoint hiện tại chỉ được train vài iteration để kiểm tra pipeline.
- Confidence và bbox trong demo chưa đại diện accuracy thực tế.
- Depth là `camera-Z`, không phải Euclidean range 3D từ sensor.
- Logic `CRITICAL / NEAR / FAR` chỉ để minh họa, không phải safety logic.
