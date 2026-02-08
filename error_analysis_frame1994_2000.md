# フレーム1994-2000データセットでの学習エラー分析

## 発生したエラー

1. **SIGFPE (Floating Point Exception)** - ステップ3097で発生
2. **RuntimeError: params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout** - ステップ3100以降で発生

## 根本原因

### 1. ステップ3000でのOpacityリセット

設定ファイル（`cfg.yml`）より：
- `reset_every: 3000` - 3000ステップごとにopacityをリセット
- `prune_opa: 0.005` - opacityが0.005未満のGaussianをprune

**ステップ3000での動作**:
- `reset_opa()`が実行され、すべてのGaussianのopacityが`prune_opa * 2.0 = 0.01`にリセットされる
- 同時に、SH degreeが2から3に変更される（`sh_degree_interval: 1000`）

### 2. ステップ3100での大量Pruning

ログより：
```
Step 3100: 109588 GSs pruned. Now having 2 GSs.
```

**原因**:
- リセット後、多くのGaussianのopacityが0.01に設定された
- その後100ステップ（3000-3100）の学習で、多くのGaussianのopacityが`prune_opa`（0.005）未満に低下
- `_prune_gs()`が実行され、109,588個のGaussianがpruneされた
- 残り2個のGaussianのみが残った

### 3. SIGFPEエラーの発生

**原因**:
- 残り2個のGaussianしかない状態で、CUDA kernel（`gsplat::launch_projection_ewa_3dgs_fused_fwd_kernel`）内で：
  - ゼロ除算が発生
  - NaN/Inf値が生成
  - 数値的不安定性により浮動小数点例外が発生

### 4. RuntimeErrorの発生

**原因**:
- 大量のpruningにより、optimizerの状態（`exp_avgs`, `exp_avg_sqs`）とパラメータの数が不整合になった
- 残り2個のGaussianに対して、optimizerの状態が正しく更新されなかった

## 問題の本質

1. **リセット後のopacityが低すぎる**: `prune_opa * 2.0 = 0.01`は、リセット直後に多くのGaussianがpruneされるリスクがある
2. **リセットとpruningのタイミング**: リセット直後（100ステップ後）にpruningが実行され、opacityが十分に回復する前に大量のGaussianが削除された
3. **データセットの特性**: フレーム1994-2000の連続データセットは、マスク適用により点群が限定的で、Gaussianの分布が不安定になりやすい可能性がある

## 推奨される対策

### 1. リセット閾値の調整
```yaml
strategy:
  reset_every: 3000
  prune_opa: 0.01  # 0.005から0.01に増加（より保守的に）
  # または
  # reset_every: 5000  # リセット間隔を延長
```

### 2. Pruning閾値の調整
```yaml
strategy:
  prune_opa: 0.01  # より高い閾値でpruningを抑制
  prune_scale2d: 0.2  # 0.15から0.2に増加
  prune_scale3d: 0.15  # 0.1から0.15に増加
```

### 3. リセット後のPruning停止期間の設定
```yaml
strategy:
  pause_refine_after_reset: 200  # リセット後200ステップはpruningを停止
```

### 4. 最小Gaussian数の保護
- コード修正が必要: pruning後に最小数のGaussian（例: 1000個）を保証する

### 5. 学習率の調整
```yaml
opacities_lr: 0.01  # 0.05から0.01に減少（opacityの変化を緩やかに）
```

## 比較: フレーム2000データセット

フレーム2000データセットでは問題が発生しなかった理由：
- より多くの画像（8枚 vs 7枚）
- より良い点群分布
- リセット後も十分なGaussianが残存

## 結論

主な原因は、**リセット後のopacityが低すぎること**と、**リセット直後のpruningにより大量のGaussianが削除されたこと**です。これにより、残り2個のGaussianしかない状態になり、数値的不安定性とoptimizerの不整合が発生しました。

対策として、pruning閾値を緩和するか、リセット間隔を延長することを推奨します。

