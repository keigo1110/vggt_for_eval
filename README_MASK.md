# マスク機能の使い方

VGGTとGaussian Splattingのトレーニングで、二値マスクを使用して学習に使用したくないピクセルや領域を指定できます。

## 方法1: VGGTの推論時にマスクを適用

VGGTのREADMEによると、不要なピクセル（反射面、空、水など）は、対応するピクセル値を0または1に設定することでマスクできます。

### 使用方法

```bash
# 画像にマスクを適用
python apply_mask_to_images.py \
    --image_dir examples/kitchen/images \
    --mask_dir examples/kitchen/masks \
    --output_dir examples/kitchen/images_masked \
    --mask_value 0 \
    --mask_suffix "_mask"
```

**マスクファイルの命名規則**:
- 画像ファイル: `00.png`
- マスクファイル: `00_mask.png` (または `00_mask.jpg`)

**マスクの形式**:
- 白（255）= 使用するピクセル
- 黒（0）= 除外するピクセル

マスクを適用した後、通常通りVGGTを実行します：

```bash
python demo_colmap.py --scene_dir examples/kitchen_masked
```

## 方法2: Gaussian Splattingのトレーニング時にマスクを適用

gsplatのトレーニング時にマスクファイルを直接読み込んで使用できます。

### 使用方法

1. **マスクファイルを準備**:
   - マスクディレクトリを作成: `examples/kitchen/masks/`
   - 各画像に対応するマスクファイルを配置
   - 命名規則: `画像名_mask.png` (例: `00.png` → `00_mask.png`)

2. **トレーニングを実行**:
```bash
cd /home/rkmtlab-gdep/Desktop/workspace/gsplat
conda activate vggt
PYTHONPATH=/home/rkmtlab-gdep/Desktop/workspace/gsplat:$PYTHONPATH \
python examples/simple_trainer.py default \
    --data_dir /home/rkmtlab-gdep/Desktop/workspace/vggt/examples/kitchen \
    --mask_dir /home/rkmtlab-gdep/Desktop/workspace/vggt/examples/kitchen/masks \
    --mask_suffix "_mask" \
    --data_factor 1 \
    --result_dir /home/rkmtlab-gdep/Desktop/workspace/vggt/examples/kitchen/gsplat_results \
    --max-steps 5000 \
    --disable-viewer
```

**パラメータ説明**:
- `--mask_dir`: マスクファイルが格納されているディレクトリ
- `--mask_suffix`: マスクファイル名のサフィックス（デフォルト: `_mask`）

**マスクの形式**:
- 白（255以上）= 使用するピクセル（True）
- 黒（127以下）= 除外するピクセル（False）

## マスクファイルの作成方法

### 手動で作成
画像編集ソフト（GIMP、Photoshop、Paint等）で：
1. 元の画像と同じサイズの画像を作成
2. 使用する領域を白（255）で塗りつぶす
3. 除外する領域を黒（0）で塗りつぶす
4. PNG形式で保存

### プログラムで作成
```python
import cv2
import numpy as np

# 画像を読み込み
image = cv2.imread("image.jpg")
h, w = image.shape[:2]

# マスクを作成（例: 中央の領域のみ使用）
mask = np.zeros((h, w), dtype=np.uint8)
center_x, center_y = w // 2, h // 2
radius = min(w, h) // 3
cv2.circle(mask, (center_x, center_y), radius, 255, -1)

# マスクを保存
cv2.imwrite("image_mask.png", mask)
```

## 注意事項

1. **マスクファイルのサイズ**: 画像と同じサイズである必要があります（自動的にリサイズされます）
2. **ファイル名の対応**: マスクファイル名は `画像名 + mask_suffix + 拡張子` である必要があります
3. **マスクが見つからない場合**: その画像はマスクなしで処理されます（全ピクセルが使用されます）
4. **歪み補正**: カメラに歪みがある場合、マスクも自動的に歪み補正されます

## 例

### 例1: 空の領域を除外
```bash
# 空のセグメンテーションマスクを作成（手動または自動）
# マスクファイル: 00_mask.png, 01_mask.png, ...

# VGGTでマスクを適用
python apply_mask_to_images.py \
    --image_dir examples/kitchen/images \
    --mask_dir examples/kitchen/sky_masks \
    --output_dir examples/kitchen/images_masked \
    --mask_value 0

# マスク適用後の画像でCOLMAPを生成
python demo_colmap.py --scene_dir examples/kitchen_masked
```

### 例2: Gaussian Splattingで直接マスクを使用
```bash
# マスクファイルを準備（既に準備済み）
# examples/kitchen/masks/00_mask.png, 01_mask.png, ...

# マスクを使用してトレーニング
python examples/simple_trainer.py default \
    --data_dir examples/kitchen \
    --mask_dir examples/kitchen/masks \
    --mask_suffix "_mask" \
    ...
```

