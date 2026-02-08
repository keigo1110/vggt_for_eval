# 時系列データ vs 提案手法の包括的比較

## 概要

このディレクトリには、時系列データ（temporal_baseline）と提案手法（selection_proposed）の包括的な比較評価スクリプトと結果が含まれています。

## 評価の目的

1. **時系列データ（temporal_baseline）**: 1秒ごとの連続フレームを使用したGaussian Splatting
2. **提案手法（selection_proposed）**: 過去視点選定アルゴリズムを使用したGaussian Splatting

両手法を比較し、以下を評価します：
- レンダリング品質（PSNR, SSIM, LPIPS）
- エラー発生率と原因（視点の悪さなど）
- カメラ品質とレンダリング品質の関係
- 時系列での品質変化

## ファイル構成

### スクリプト

1. **`comprehensive_comparison_temporal_vs_proposed.py`**
   - 全フレームの評価を実行
   - 時系列データと提案手法の比較
   - エラー情報の記録
   - カメラ品質の分析

2. **`analyze_temporal_vs_proposed_results.py`**
   - 評価結果の詳細分析
   - エラー分析
   - カメラ品質とレンダリング品質の関係分析
   - 詳細なグラフとレポートの生成

3. **`check_comparison_progress.py`**
   - 処理の進捗確認

### 出力ファイル

評価結果は以下のディレクトリに保存されます：
```
examples/temporal_vs_proposed_comparison/
├── comparison_results.json          # 全評価結果（JSON）
├── comparison_tables.tex            # LaTeXテーブル
├── psnr_comparison.png             # PSNR比較グラフ
├── ssim_comparison.png             # SSIM比較グラフ
├── quality_vs_psnr.png            # カメラ品質 vs PSNR
├── error_analysis.png              # エラー分析
├── detailed_comparison_report.md    # 詳細レポート
└── ... (その他の詳細グラフ)
```

## 使用方法

### 1. 全フレームの評価を実行

```bash
cd /home/rkmtlab-gdep/Desktop/workspace/vggt
conda activate vggt
python comprehensive_comparison_temporal_vs_proposed.py
```

**注意**: 114フレームを処理するため、約5-10分かかります。

### 2. 処理の進捗確認

```bash
python check_comparison_progress.py
```

または、ログファイルを確認：
```bash
tail -f /tmp/comparison_log.txt
```

### 3. 詳細分析の実行

評価が完了したら、詳細分析を実行：

```bash
python analyze_temporal_vs_proposed_results.py
```

## 評価指標

### レンダリング品質

- **PSNR** (Peak Signal-to-Noise Ratio): 画像品質の指標（高いほど良い）
- **SSIM** (Structural Similarity Index): 構造的類似性（0-1、高いほど良い）
- **LPIPS** (Learned Perceptual Image Patch Similarity): 知覚的類似性（低いほど良い）

### カメラ品質スコア

視点の品質を評価する指標：
- カメラ間距離の適切さ
- カメラ間角度の多様性
- 視点配置の多様性

スコアは0-1の範囲で、高いほど良い視点配置を示します。

### エラー分析

以下のエラーを記録：
- チェックポイントが見つからない
- データディレクトリが見つからない
- レンダリングエラー
- その他のエラー

## 結果の解釈

### 成功率

- **成功**: レンダリング品質メトリクスが計算できたフレーム
- **失敗**: エラーが発生したフレーム

### 品質比較

- **平均PSNR**: 全フレームでの平均PSNR値
- **標準偏差**: フレーム間のばらつき
- **カメラ品質スコア**: 視点配置の品質

### エラー分析

エラーの種類と頻度を分析し、視点の悪さが原因かどうかを判断します。

## 論文での使用

### テーブル

`comparison_tables.tex`をLaTeX文書に含める：

```latex
\input{examples/temporal_vs_proposed_comparison/comparison_tables.tex}
```

### 図

以下の図を使用可能：
- `psnr_comparison.png`: PSNRの時系列比較
- `quality_vs_psnr.png`: カメラ品質とPSNRの関係
- `error_analysis.png`: エラー分析

### レポート

`detailed_comparison_report.md`を参照して、詳細な分析結果を確認できます。

## 注意事項

1. **処理時間**: 全114フレームの処理には約5-10分かかります
2. **メモリ**: GPUメモリが必要です（CUDA使用時）
3. **データ構造**: 各フレームディレクトリに`temporal_baseline`と`selection_proposed`サブディレクトリが必要です

## トラブルシューティング

### チェックポイントが見つからない

- `gsplat_results/ckpts/`ディレクトリを確認
- チェックポイントファイルが存在するか確認

### データディレクトリが見つからない

- `sparse/`と`images/`ディレクトリが存在するか確認
- フレームディレクトリの構造を確認

### レンダリングエラー

- GPUメモリが不足している可能性
- カメラパラメータが正しいか確認

## 関連ファイル

- `COMPREHENSIVE_EVALUATION_REPORT.md`: 包括的評価レポート
- `EVALUATION_RESULTS_INDEX.md`: 評価結果のインデックス

