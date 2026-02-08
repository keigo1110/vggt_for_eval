# 全フレーム処理スクリプト

全フレームを処理し、マスクの白領域と黒領域で異なる処理を行うスクリプトです。

## 機能

1. **白領域（全フレーム点群生成）**
   - 全フレームに対して点群データを生成
   - マスクの白領域のみを使用

2. **黒領域（30フレームごとにGS処理）**
   - 30フレームごと（t=30からスタート）にGaussian Splattingを実行
   - マスクの黒領域のみを使用
   - 評価メトリクス（PSNR、SSIM、LPIPS）を計算

## 使用方法

### 基本的な使用方法

```bash
python process_all_frames.py \
    --mydata_dir mydata \
    --output_dir evaluation_results \
    --start_frame 0 \
    --gs_start_frame 30 \
    --gs_interval 30 \
    --max_steps 5000
```

### 時系列（ベースライン）とselection.json（提案手法）の比較

```bash
python process_all_frames.py \
    --mydata_dir mydata \
    --output_dir evaluation_results \
    --start_frame 0 \
    --gs_start_frame 30 \
    --gs_interval 30 \
    --max_steps 5000 \
    --compare_methods
```

`--compare_methods`フラグを指定すると、時系列入力画像（ベースライン）とselection.jsonで選ばれた入力画像セット（提案手法）の両方を処理し、比較データを取得します。

### パラメータ

- `--mydata_dir`: mydataディレクトリのパス（デフォルト: `mydata`）
- `--output_dir`: 出力ディレクトリのパス（デフォルト: `evaluation_results`）
- `--start_frame`: 開始フレームID（デフォルト: `0`）
- `--end_frame`: 終了フレームID（デフォルト: `selection.json`から自動取得）
- `--gs_interval`: Gaussian Splatting処理の間隔（デフォルト: `30`）
- `--gs_start_frame`: Gaussian Splatting処理の開始フレーム（デフォルト: `30`）
- `--max_steps`: Gaussian Splattingの最大学習ステップ数（デフォルト: `5000`）
- `--skip_white`: 白領域の処理をスキップ
- `--skip_black`: 黒領域の処理をスキップ

### 例

#### 全フレーム処理（テスト用に範囲を限定）

```bash
python process_all_frames.py \
    --mydata_dir mydata \
    --output_dir evaluation_results \
    --start_frame 0 \
    --end_frame 100 \
    --gs_start_frame 30 \
    --gs_interval 30
```

#### 白領域のみ処理

```bash
python process_all_frames.py \
    --mydata_dir mydata \
    --output_dir evaluation_results \
    --skip_black
```

#### 黒領域のみ処理

```bash
python process_all_frames.py \
    --mydata_dir mydata \
    --output_dir evaluation_results \
    --skip_white
```

## 出力ディレクトリ構造

```
evaluation_results/
├── evaluation_results.json          # 評価結果のサマリー
├── white_regions/
│   └── point_clouds/
│       ├── frame_000000/
│       │   ├── images/
│       │   ├── masks/
│       │   └── sparse_white/
│       │       └── points_white.ply
│       ├── frame_000001/
│       └── ...
└── black_regions/
    └── gaussian_splatting/
        ├── frame_000030/
        │   ├── images/
        │   ├── masks/
        │   ├── sparse/              # 黒領域のCOLMAP
        │   └── gsplat_results/
        │       ├── ckpts/
        │       └── ...
        ├── frame_000060/
        └── ...
```

## 処理フロー

### 白領域の処理

1. 各フレーム（t）について：
   - `selection.json`から入力画像セット（n枚）を取得
   - 時系列画像（t - n + 1 枚目〜t枚目）を取得
   - 画像とマスクを準備
   - VGGTでCOLMAPを生成（白領域のみ）
   - 点群ファイル（`points_white.ply`）を保存

### 黒領域の処理

1. 30フレームごと（t=30, 60, 90, ...）について：
   - `selection.json`から入力画像セット（n枚）を取得
   - 時系列画像（t - n + 1 枚目〜t枚目）を取得
   - 画像とマスクを準備
   - VGGTでCOLMAPを生成（黒領域のみ）
   - Gaussian Splattingで学習
   - 評価メトリクス（PSNR、SSIM、LPIPS）を計算

## 評価結果

`evaluation_results.json`には以下の情報が含まれます：

```json
{
  "white_regions": {
    "description": "Point cloud generation for all frames (white mask regions)",
    "results": {
      "0": {
        "status": "success",
        "point_cloud_path": "...",
        "n_images": 8
      },
      ...
    },
    "summary": {
      "total_frames": 3441,
      "successful": 3400,
      "failed": 41
    }
  },
  "black_regions": {
    "description": "Gaussian Splatting training and evaluation every 30 frames (black mask regions)",
    "results": {
      "30": {
        "status": "success",
        "n_images": 8,
        "training_steps": 4999,
        "checkpoint_path": "...",
        "metrics": [...],
        "average_metrics": {
          "PSNR": 25.3,
          "SSIM": 0.85,
          "LPIPS": 0.12
        }
      },
      ...
    },
    "summary": {
      "total_frames": 114,
      "successful": 110,
      "failed": 4,
      "average_metrics": {
        "PSNR": 25.1,
        "SSIM": 0.84,
        "LPIPS": 0.13
      }
    }
  }
}
```

## 注意事項

1. **マスクファイルの命名規則**
   - 画像ファイル: `frame_000000.jpg`
   - マスクファイル: `m0000.jpg`（四桁の数字で対応付け）

2. **処理時間**
   - 白領域の処理は比較的速い（各フレーム数秒）
   - 黒領域の処理は時間がかかる（各フレーム数分〜数十分、学習ステップ数による）
   - `--compare_methods`を使用する場合、処理時間は約2倍になります

3. **メモリ使用量**
   - Gaussian Splattingの学習には十分なGPUメモリが必要です
   - 必要に応じて`--max_steps`を調整してください

4. **エラーハンドリング**
   - 処理が失敗した場合、詳細なエラー情報が`evaluation_results.json`に記録されます
   - エラーログは各フレームの`logs/`ディレクトリに保存されます（`vggt_colmap.log`、`gsplat_training.log`）
   - 個別のフレームでエラーが発生しても、他のフレームの処理は継続されます
   - エラー情報には以下が含まれます：
     - エラーメッセージ
     - エラータイプ
     - スタックトレース（評価段階でのエラーの場合）
     - ログファイルのパス

5. **比較モード**
   - `--compare_methods`を使用すると、時系列（ベースライン）とselection.json（提案手法）の両方を処理します
   - 比較結果は`evaluation_results.json`の`comparison`セクションに保存されます
   - メトリクスの改善量も自動的に計算されます

## 論文用データの生成

評価結果は論文に載せられる形式で出力されます：

- **定量的評価**: PSNR、SSIM、LPIPSの平均値と標準偏差
- **処理統計**: 成功/失敗フレーム数
- **時系列データ**: 各フレームの詳細なメトリクス

結果は`evaluation_results.json`に保存され、後からデータ分析しやすい形式になっています。

