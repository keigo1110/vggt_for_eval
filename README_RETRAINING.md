# Temporal Baseline 再学習ガイド

## 概要

Cooking Datasetの`frame_001650`のTemporal Baselineモデルを再学習しています。

## 問題の原因

元の学習では、ステップ3100で69,664個のGaussianが一度にpruneされ、最終的に3個まで減少してしまいました。これは以下の設定が原因でした：

- `reset_every: 3000` - ステップ3000でopacityがリセット
- `prune_opa: 0.005` - opacityが0.005未満のGaussianがprune
- `pause_refine_after_reset: 0` - リセット後のpruningが即座に実行

## 修正内容

再学習では以下の設定を変更しました：

1. **reset_every**: 3000 → **5000**（リセット間隔を延長）
2. **prune_opa**: 0.005 → **0.01**（pruning閾値を緩和）
3. **pause_refine_after_reset**: 0 → **200**（リセット後200ステップはpruningを停止）
4. **opacities_lr**: 0.05 → **0.03**（opacity学習率を下げて安定化）

## 再学習の実行

```bash
cd /home/rkmtlab-gdep/Desktop/workspace/vggt
conda activate vggt
python retrain_temporal_baseline_cooking.py
```

## 進行状況の確認

```bash
python check_retraining_status.py
```

または、ログファイルを直接確認：

```bash
tail -f evaluation_results_cooking/black_regions/gaussian_splatting/frame_001650/temporal_baseline/logs/gsplat_retraining.log
```

## 完了後の作業

1. 新しいチェックポイントを確認：
   ```bash
   ls -lh evaluation_results_cooking/black_regions/gaussian_splatting/frame_001650/temporal_baseline/gsplat_results/ckpts/
   ```

2. モデルの状態を確認：
   ```bash
   python check_retraining_status.py
   ```

3. **比較図を再生成（Baseline vs Ours）**：
   ```bash
   python regenerate_comparison_after_retraining.py
   ```
   
   このスクリプトは：
   - 再学習したTemporal BaselineとSelection Proposedを同じ条件で比較
   - メトリクスが良い視点を自動選択
   - 論文品質の比較図を生成
   - BaselineとOursの差を明確に示す

## 期待される結果

- Gaussians数: 15,000以上（初期値は15,550）
- 不透明度の最大値: 0.1以上
- ステップ7000まで正常に学習が完了

## バックアップ

- 設定ファイル: `cfg.yml.backup`
- 古いチェックポイント: `ckpts_backup/`

