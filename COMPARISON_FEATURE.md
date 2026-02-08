# 比較機能の実装状況

## 実装済み機能

1. **エラー情報の詳細記録**
   - エラーメッセージ、エラータイプ、スタックトレースを記録
   - ログファイル（`vggt_colmap.log`、`gsplat_training.log`）を保存
   - エラー発生段階（preparation、vggt_colmap、gsplat_training、evaluation）を記録

2. **比較機能の準備**
   - `--compare_methods`フラグを追加
   - `save_results`関数に比較結果を保存する機能を実装
   - 時系列（ベースライン）とselection.json（提案手法）の比較結果を計算

## 実装が必要な機能

`process_white_regions_all_frames`と`process_black_regions_gs`関数に`compare_methods`パラメータを追加し、比較処理を実装する必要があります。

現在の実装では、これらの関数は時系列のみを処理しています。比較モードを有効にすると、両方の手法を処理して比較できるようにする必要があります。

## 使用方法

比較機能を使用するには、`--compare_methods`フラグを指定します：

```bash
python process_all_frames.py \
    --mydata_dir mydata \
    --output_dir evaluation_results \
    --compare_methods
```

比較結果は`evaluation_results.json`の`comparison`セクションに保存されます。

