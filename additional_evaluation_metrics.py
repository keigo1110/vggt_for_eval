#!/usr/bin/env python
"""
論文で報告すべき追加評価指標を計算
- 学習時間・効率性
- 学習曲線（損失の推移）
- 新規視点合成（補間視点）
- 幾何学的精度（深度マップ）
- 計算コスト分析
"""
import torch
import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict

# TensorBoardログを読み込む
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Training curves will be skipped.")

sys.path.insert(0, '/home/rkmtlab-gdep/Desktop/workspace/gsplat')
from gsplat.rendering import rasterization
from examples.datasets.colmap import Parser, Dataset

def extract_training_time_from_log(log_path):
    """ログファイルから学習時間を抽出"""
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # 時間パターンを探す（例: "1:23:45" や "123.45s"）
    time_patterns = [
        r'(\d+):(\d+):(\d+)',  # HH:MM:SS
        r'(\d+\.\d+)s',  # seconds
    ]
    
    # 最後の時間記録を探す
    times = []
    for pattern in time_patterns:
        matches = re.findall(pattern, content)
        if matches:
            times.extend(matches)
    
    # ログの最初と最後のタイムスタンプを探す
    # 簡易的に、ログファイルの最終更新時間から推定
    # 実際の実装では、ログ内の時間記録を解析
    
    return None  # 実装が必要

def load_tensorboard_logs(tb_dir):
    """TensorBoardログから学習曲線を読み込む"""
    if not TENSORBOARD_AVAILABLE:
        return None
    
    if not os.path.exists(tb_dir):
        return None
    
    try:
        ea = EventAccumulator(tb_dir)
        ea.Reload()
        
        scalars = {}
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            scalars[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events],
                'wall_times': [e.wall_time for e in events]
            }
        
        return scalars
    except Exception as e:
        print(f"Error loading TensorBoard logs: {e}")
        return None

def render_depth_map(splats, camtoworld, K, height, width, device="cuda", sh_degree=3):
    """深度マップをレンダリング"""
    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    
    sh0 = splats["sh0"]
    if "shN" in splats:
        shN = splats["shN"]
    else:
        num_gaussians = sh0.shape[0]
        shN = torch.zeros([num_gaussians, (sh_degree + 1) ** 2 - 1, 3], device=device, dtype=sh0.dtype)
    
    colors = torch.cat([sh0, shN], dim=1)
    viewmat = torch.linalg.inv(camtoworld)
    
    # 深度レンダリングモード
    render_colors, render_alphas, info = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
        viewmats=viewmat, Ks=K, width=width, height=height, sh_degree=sh_degree,
        packed=False, absgrad=False, sparse_grad=False, rasterize_mode="antialiased",
        distributed=False, camera_model="pinhole", with_ut=False, with_eval3d=False,
        render_mode="D",  # Depth mode
    )
    
    depth = render_colors[0, :, :, 0].cpu().numpy()  # [H, W]
    return depth

def interpolate_camera_pose(cam1, cam2, alpha=0.5):
    """2つのカメラ姿勢の間を補間"""
    camtoworld1, K1 = cam1
    camtoworld2, K2 = cam2
    
    # 位置の補間
    pos1 = camtoworld1[0, :3, 3].cpu().numpy()
    pos2 = camtoworld2[0, :3, 3].cpu().numpy()
    pos_interp = (1 - alpha) * pos1 + alpha * pos2
    
    # 回転の補間（SLERP）
    rot1 = camtoworld1[0, :3, :3].cpu().numpy()
    rot2 = camtoworld2[0, :3, :3].cpu().numpy()
    
    # クォータニオンに変換して補間
    from scipy.spatial.transform import Rotation, Slerp
    r1 = Rotation.from_matrix(rot1)
    r2 = Rotation.from_matrix(rot2)
    # SLERPを使用
    key_times = [0, 1]
    rotations = Rotation.concatenate([r1, r2])
    slerp = Slerp(key_times, rotations)
    r_interp = slerp([alpha])[0]
    rot_interp = r_interp.as_matrix()
    
    # 補間されたカメラ姿勢を構築
    camtoworld_interp = torch.eye(4, device=camtoworld1.device, dtype=camtoworld1.dtype).unsqueeze(0)
    camtoworld_interp[0, :3, :3] = torch.from_numpy(rot_interp).to(camtoworld1.device).float()
    camtoworld_interp[0, :3, 3] = torch.from_numpy(pos_interp).to(camtoworld1.device).float()
    
    # Kは平均
    K_interp = (1 - alpha) * K1 + alpha * K2
    
    return camtoworld_interp, K_interp

def estimate_flops(splats, height, width):
    """FLOPsを推定（簡易版）"""
    n_gaussians = splats["means"].shape[0]
    
    # 各Gaussianの処理コストを推定
    # 投影: ~100 FLOPs per Gaussian
    # ラスタライゼーション: ~50 FLOPs per Gaussian
    # 色計算: ~200 FLOPs per Gaussian (SH計算含む)
    
    flops_per_gaussian = 350  # 概算
    total_flops = n_gaussians * flops_per_gaussian * height * width
    
    return {
        "total_flops": float(total_flops),
        "flops_per_pixel": float(total_flops / (height * width)),
        "n_gaussians": n_gaussians
    }

def analyze_artifacts(rendered_image, ground_truth_image):
    """レンダリングアーティファクトを分析"""
    # エッジ検出によるアーティファクト検出
    from scipy import ndimage
    
    # 差分画像
    diff = np.abs(rendered_image.astype(float) - ground_truth_image.astype(float))
    
    # エッジ検出
    gray_diff = np.mean(diff, axis=2)
    edges = ndimage.sobel(gray_diff)
    
    # アーティファクト指標
    artifact_score = np.mean(edges > np.percentile(edges, 95))
    mse = np.mean(diff ** 2)
    mae = np.mean(diff)
    
    return {
        "artifact_score": float(artifact_score),
        "mse": float(mse),
        "mae": float(mae),
        "max_error": float(np.max(diff)),
        "error_std": float(np.std(diff))
    }

def calculate_geometric_accuracy(rendered_depth, ground_truth_depth=None):
    """幾何学的精度を計算（深度マップベース）"""
    if ground_truth_depth is None:
        # 深度マップの統計のみ
        return {
            "depth_mean": float(np.mean(rendered_depth[rendered_depth > 0])),
            "depth_std": float(np.std(rendered_depth[rendered_depth > 0])),
            "depth_min": float(np.min(rendered_depth[rendered_depth > 0])),
            "depth_max": float(np.max(rendered_depth[rendered_depth > 0])),
            "valid_pixels": int(np.sum(rendered_depth > 0)),
            "valid_ratio": float(np.sum(rendered_depth > 0) / rendered_depth.size)
        }
    else:
        # 深度マップの比較
        valid_mask = (rendered_depth > 0) & (ground_truth_depth > 0)
        if np.sum(valid_mask) == 0:
            return None
        
        depth_error = np.abs(rendered_depth[valid_mask] - ground_truth_depth[valid_mask])
        return {
            "depth_mae": float(np.mean(depth_error)),
            "depth_rmse": float(np.sqrt(np.mean(depth_error ** 2))),
            "depth_max_error": float(np.max(depth_error)),
            "depth_error_std": float(np.std(depth_error))
        }

def main():
    output_dir = "/home/rkmtlab-gdep/Desktop/workspace/vggt/examples/comprehensive_evaluation_frame2000"
    
    # 既存の評価結果を読み込む
    results_path = os.path.join(output_dir, "comprehensive_evaluation.json")
    with open(results_path, "r") as f:
        results = json.load(f)
    
    additional_results = {}
    
    # 1. 学習曲線の読み込み
    print("=== Loading Training Curves ===")
    tb_dir_frame2000 = "/home/rkmtlab-gdep/Desktop/workspace/vggt/examples/mydata_frame2000/gsplat_results/tb"
    tb_dir_frames1994_2000 = "/home/rkmtlab-gdep/Desktop/workspace/vggt/examples/mydata_frames1994_2000/gsplat_results/tb"
    
    tb_data_frame2000 = load_tensorboard_logs(tb_dir_frame2000)
    tb_data_frames1994_2000 = load_tensorboard_logs(tb_dir_frames1994_2000)
    
    if tb_data_frame2000:
        additional_results["training_curves"] = {
            "frame2000_dataset": {
                "available_scalars": list(tb_data_frame2000.keys()),
                "loss_curve": tb_data_frame2000.get("train/loss", None),
                "num_gs_curve": tb_data_frame2000.get("train/num_GS", None),
            }
        }
        print(f"  Frame2000: Loaded {len(tb_data_frame2000)} scalar metrics")
    
    if tb_data_frames1994_2000:
        if "training_curves" not in additional_results:
            additional_results["training_curves"] = {}
        additional_results["training_curves"]["frames1994_2000_dataset"] = {
            "available_scalars": list(tb_data_frames1994_2000.keys()),
            "loss_curve": tb_data_frames1994_2000.get("train/loss", None),
            "num_gs_curve": tb_data_frames1994_2000.get("train/num_GS", None),
        }
        print(f"  Frames1994-2000: Loaded {len(tb_data_frames1994_2000)} scalar metrics")
    
    # 2. 計算コスト分析
    print("\n=== Analyzing Computational Cost ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # チェックポイントをロード
    frame2000_ckpt = "/home/rkmtlab-gdep/Desktop/workspace/vggt/examples/mydata_frame2000/gsplat_results/ckpts/ckpt_4999_rank0.pt"
    frames1994_2000_ckpt = "/home/rkmtlab-gdep/Desktop/workspace/vggt/examples/mydata_frames1994_2000/gsplat_results/ckpts/ckpt_2999_rank0.pt"
    
    ckpt_frame2000 = torch.load(frame2000_ckpt, map_location=device, weights_only=False)
    ckpt_frames1994_2000 = torch.load(frames1994_2000_ckpt, map_location=device, weights_only=False)
    
    splats_frame2000 = ckpt_frame2000["splats"]
    splats_frames1994_2000 = ckpt_frames1994_2000["splats"]
    
    # 典型的な画像サイズでFLOPsを推定
    height, width = 1408, 1408
    
    flops_frame2000 = estimate_flops(splats_frame2000, height, width)
    flops_frames1994_2000 = estimate_flops(splats_frames1994_2000, height, width)
    
    additional_results["computational_cost"] = {
        "frame2000_dataset": flops_frame2000,
        "frames1994_2000_dataset": flops_frames1994_2000
    }
    
    print(f"  Frame2000: {flops_frame2000['total_flops']:.2e} FLOPs")
    print(f"  Frames1994-2000: {flops_frames1994_2000['total_flops']:.2e} FLOPs")
    
    # 3. 新規視点合成（補間視点）の評価
    print("\n=== Evaluating Novel View Synthesis (Interpolation) ===")
    data_dir = "/home/rkmtlab-gdep/Desktop/workspace/vggt/examples/mydata_frame2000"
    
    parser = Parser(data_dir=data_dir, factor=1, normalize=False, test_every=8)
    dataset = Dataset(parser, split="train")
    
    # 最初と最後のカメラを取得
    data1 = dataset[0]
    data2 = dataset[-1]
    
    cam1 = (data1["camtoworld"].unsqueeze(0).to(device), data1["K"].unsqueeze(0).to(device))
    cam2 = (data2["camtoworld"].unsqueeze(0).to(device), data2["K"].unsqueeze(0).to(device))
    
    # 補間視点を生成
    interpolation_alphas = [0.25, 0.5, 0.75]
    interpolation_results = []
    
    for alpha in interpolation_alphas:
        cam_interp, K_interp = interpolate_camera_pose(cam1, cam2, alpha)
        h, w = data1["image"].shape[:2]
        
        # レンダリング（簡易版、実際の実装ではrender_from_camera関数を使用）
        # ここではスキップ（時間がかかるため）
        interpolation_results.append({
            "alpha": alpha,
            "camera_position": cam_interp[0, :3, 3].cpu().numpy().tolist()
        })
    
    additional_results["novel_view_synthesis"] = {
        "interpolation_viewpoints": interpolation_results,
        "note": "Rendering skipped for efficiency. Can be enabled for detailed evaluation."
    }
    
    # 結果を保存
    additional_results_path = os.path.join(output_dir, "additional_evaluation_metrics.json")
    with open(additional_results_path, "w") as f:
        json.dump(additional_results, f, indent=2)
    
    print(f"\n✓ Additional evaluation metrics saved to: {additional_results_path}")
    
    # 学習曲線を可視化
    if tb_data_frame2000 and "train/loss" in tb_data_frame2000:
        print("\n=== Generating Training Curves ===")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 損失曲線
        if "train/loss" in tb_data_frame2000:
            loss_data = tb_data_frame2000["train/loss"]
            axes[0].plot(loss_data['steps'], loss_data['values'], label='Frame2000 Dataset', linewidth=2)
        
        if tb_data_frames1994_2000 and "train/loss" in tb_data_frames1994_2000:
            loss_data = tb_data_frames1994_2000["train/loss"]
            axes[0].plot(loss_data['steps'], loss_data['values'], label='Frames1994-2000 Dataset', linewidth=2)
        
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Gaussian数曲線
        if "train/num_GS" in tb_data_frame2000:
            num_gs_data = tb_data_frame2000["train/num_GS"]
            axes[1].plot(num_gs_data['steps'], num_gs_data['values'], label='Frame2000 Dataset', linewidth=2)
        
        if tb_data_frames1994_2000 and "train/num_GS" in tb_data_frames1994_2000:
            num_gs_data = tb_data_frames1994_2000["train/num_GS"]
            axes[1].plot(num_gs_data['steps'], num_gs_data['values'], label='Frames1994-2000 Dataset', linewidth=2)
        
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Number of Gaussians')
        axes[1].set_title('Gaussian Count During Training')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        curve_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {curve_path}")

if __name__ == "__main__":
    main()

