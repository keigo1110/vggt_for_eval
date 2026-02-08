#!/usr/bin/env python
"""
フレーム2000データセットの包括的な定量的評価スクリプト
論文レベルの評価指標を計算
"""
import torch
import os
import sys
import numpy as np
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# gsplatのパスを追加
sys.path.insert(0, '/home/rkmtlab-gdep/Desktop/workspace/gsplat')

from gsplat.rendering import rasterization
from examples.datasets.colmap import Parser, Dataset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def load_checkpoint(ckpt_path, device="cuda"):
    """チェックポイントをロード"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = ckpt.get("step", "unknown")
    splats = ckpt["splats"]
    for k in splats:
        splats[k] = splats[k].to(device)
    return splats, step

def render_from_camera(splats, camtoworld, K, height, width, device="cuda", sh_degree=3):
    """指定されたカメラからレンダリング"""
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
    
    render_colors, render_alphas, info = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
        viewmats=viewmat, Ks=K, width=width, height=height, sh_degree=sh_degree,
        packed=False, absgrad=False, sparse_grad=False, rasterize_mode="antialiased",
        distributed=False, camera_model="pinhole", with_ut=False, with_eval3d=False,
    )
    
    render_colors = render_colors[0].cpu().numpy()
    render_colors = np.clip(render_colors, 0, 1)
    render_colors = (render_colors * 255).astype(np.uint8)
    return render_colors

def calculate_metrics(rendered_image, ground_truth_image, device="cuda"):
    """PSNR、SSIM、LPIPSを計算"""
    rendered_tensor = torch.from_numpy(rendered_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    gt_tensor = torch.from_numpy(ground_truth_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    rendered_tensor = rendered_tensor.to(device)
    gt_tensor = gt_tensor.to(device)
    
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    
    return {
        "PSNR": psnr(rendered_tensor, gt_tensor).item(),
        "SSIM": ssim(rendered_tensor, gt_tensor).item(),
        "LPIPS": lpips(rendered_tensor, gt_tensor).item()
    }

def analyze_camera_poses(cameras):
    """カメラ位置姿勢の分析"""
    camera_positions = []
    camera_rotations = []
    camera_names = []
    
    for name, camtoworld, K, h, w, _ in cameras:
        # カメラ位置（world座標系でのカメラ中心）
        cam_pos = camtoworld[0, :3, 3].cpu().numpy()
        camera_positions.append(cam_pos)
        
        # カメラの回転行列
        cam_rot = camtoworld[0, :3, :3].cpu().numpy()
        camera_rotations.append(cam_rot)
        camera_names.append(name)
    
    camera_positions = np.array(camera_positions)
    camera_rotations = np.array(camera_rotations)
    
    # カメラ間の距離
    n_cameras = len(camera_positions)
    distances = []
    for i in range(n_cameras):
        for j in range(i + 1, n_cameras):
            dist = np.linalg.norm(camera_positions[i] - camera_positions[j])
            distances.append(dist)
    
    # カメラ間の角度（視線方向の角度）
    angles = []
    for i in range(n_cameras):
        for j in range(i + 1, n_cameras):
            # カメラの視線方向（-Z軸方向）
            view_dir_i = -camera_rotations[i][:, 2]
            view_dir_j = -camera_rotations[j][:, 2]
            cos_angle = np.clip(np.dot(view_dir_i, view_dir_j), -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle)
    
    # カメラの配置範囲
    pos_min = camera_positions.min(axis=0)
    pos_max = camera_positions.max(axis=0)
    pos_range = pos_max - pos_min
    pos_center = (pos_min + pos_max) / 2
    
    # カメラから中心までの距離
    distances_to_center = [np.linalg.norm(pos - pos_center) for pos in camera_positions]
    
    return {
        "n_cameras": n_cameras,
        "camera_positions": camera_positions.tolist(),
        "camera_names": camera_names,
        "mean_distance_between_cameras": float(np.mean(distances)),
        "std_distance_between_cameras": float(np.std(distances)),
        "min_distance_between_cameras": float(np.min(distances)),
        "max_distance_between_cameras": float(np.max(distances)),
        "mean_angle_between_cameras": float(np.mean(angles)),
        "std_angle_between_cameras": float(np.std(angles)),
        "min_angle_between_cameras": float(np.min(angles)),
        "max_angle_between_cameras": float(np.max(angles)),
        "position_range": pos_range.tolist(),
        "position_center": pos_center.tolist(),
        "mean_distance_to_center": float(np.mean(distances_to_center)),
        "std_distance_to_center": float(np.std(distances_to_center)),
    }

def analyze_point_cloud(splats, device="cuda"):
    """点群の分析"""
    means = splats["means"].cpu().numpy()  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"]).cpu().numpy()  # [N]
    scales = torch.exp(splats["scales"]).cpu().numpy()  # [N, 3]
    
    n_points = len(means)
    
    # 点群の空間分布
    pos_min = means.min(axis=0)
    pos_max = means.max(axis=0)
    pos_range = pos_max - pos_min
    pos_center = (pos_min + pos_max) / 2
    
    # 点群の密度（単位体積あたりの点数）
    volume = np.prod(pos_range)
    density = n_points / volume if volume > 0 else 0
    
    # 不透明度の統計
    opacity_mean = float(np.mean(opacities))
    opacity_std = float(np.std(opacities))
    opacity_min = float(np.min(opacities))
    opacity_max = float(np.max(opacities))
    
    # スケールの統計
    scale_mean = float(np.mean(scales))
    scale_std = float(np.std(scales))
    scale_min = float(np.min(scales))
    scale_max = float(np.max(scales))
    
    # 有効な点（不透明度が高い点）の数
    valid_points = (opacities > 0.1).sum()
    valid_ratio = float(valid_points / n_points)
    
    return {
        "n_points": n_points,
        "n_valid_points": int(valid_points),
        "valid_ratio": valid_ratio,
        "position_range": pos_range.tolist(),
        "position_center": pos_center.tolist(),
        "density": float(density),
        "opacity_mean": opacity_mean,
        "opacity_std": opacity_std,
        "opacity_min": opacity_min,
        "opacity_max": opacity_max,
        "scale_mean": scale_mean,
        "scale_std": scale_std,
        "scale_min": scale_min,
        "scale_max": scale_max,
    }

def analyze_coverage(cameras, splats, device="cuda"):
    """カバー領域の分析"""
    means = splats["means"].cpu().numpy()
    opacities = torch.sigmoid(splats["opacities"]).cpu().numpy()
    
    # 各カメラから見える点の数を計算（より正確な判定）
    n_cameras = len(cameras)
    visible_points_per_camera = []
    coverage_ratios = []
    
    for name, camtoworld, K, h, w, _ in cameras:
        # カメラ位置と姿勢
        cam_pos = camtoworld[0, :3, 3].cpu().numpy()
        cam_rot = camtoworld[0, :3, :3].cpu().numpy()
        
        # カメラから各点へのベクトル
        vectors = means - cam_pos[None, :]
        distances = np.linalg.norm(vectors, axis=1)
        
        # カメラの視線方向（-Z軸）
        view_dir = -cam_rot[:, 2]
        
        # 点がカメラの前方にあるか（より緩い条件）
        directions = vectors / (distances[:, None] + 1e-8)
        cos_angles = np.dot(directions, view_dir)
        
        # 前方にある点（cos > 0 は90度以内）
        front_mask = cos_angles > 0
        
        # 不透明度が高い点も考慮
        valid_opacity_mask = opacities > 0.01
        
        # 可視点の判定（前方かつ有効な不透明度）
        visible_mask = front_mask & valid_opacity_mask
        visible_count = visible_mask.sum()
        
        visible_points_per_camera.append(int(visible_count))
        coverage_ratios.append(float(visible_count / len(means)))
    
    # カメラの視野角（FOV）を計算
    fovs = []
    for name, camtoworld, K, h, w, _ in cameras:
        fx = K[0, 0, 0].item()
        fy = K[0, 1, 1].item()
        fov_h = 2 * np.arctan(w / (2 * fx)) * 180 / np.pi
        fov_v = 2 * np.arctan(h / (2 * fy)) * 180 / np.pi
        fovs.append({"horizontal": float(fov_h), "vertical": float(fov_v)})
    
    return {
        "n_cameras": n_cameras,
        "visible_points_per_camera": visible_points_per_camera,
        "mean_visible_points": float(np.mean(visible_points_per_camera)),
        "std_visible_points": float(np.std(visible_points_per_camera)),
        "min_visible_points": int(np.min(visible_points_per_camera)),
        "max_visible_points": int(np.max(visible_points_per_camera)),
        "coverage_ratios": coverage_ratios,
        "mean_coverage_ratio": float(np.mean(coverage_ratios)),
        "std_coverage_ratio": float(np.std(coverage_ratios)),
        "fovs": fovs,
        "mean_fov_horizontal": float(np.mean([f["horizontal"] for f in fovs])),
        "std_fov_horizontal": float(np.std([f["horizontal"] for f in fovs])),
        "mean_fov_vertical": float(np.mean([f["vertical"] for f in fovs])),
        "std_fov_vertical": float(np.std([f["vertical"] for f in fovs])),
    }

def visualize_camera_trajectory(cameras, output_path):
    """カメラ軌跡の可視化"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    positions = []
    names = []
    for name, camtoworld, K, h, w, _ in cameras:
        pos = camtoworld[0, :3, 3].cpu().numpy()
        positions.append(pos)
        names.append(name)
    
    positions = np.array(positions)
    
    # カメラ位置をプロット
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='red', s=100, marker='o', label='Cameras')
    
    # カメラ間の接続線
    for i in range(len(positions) - 1):
        ax.plot([positions[i, 0], positions[i+1, 0]], 
                [positions[i, 1], positions[i+1, 1]], 
                [positions[i, 2], positions[i+1, 2]], 'b--', alpha=0.3)
    
    # カメラの視線方向を表示
    for i, (name, camtoworld, K, h, w, _) in enumerate(cameras):
        pos = camtoworld[0, :3, 3].cpu().numpy()
        rot = camtoworld[0, :3, :3].cpu().numpy()
        view_dir = -rot[:, 2] * 0.1  # スケール調整
        ax.quiver(pos[0], pos[1], pos[2], view_dir[0], view_dir[1], view_dir[2], 
                 color='green', arrow_length_ratio=0.3, alpha=0.5)
        ax.text(pos[0], pos[1], pos[2], name, fontsize=8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_point_cloud(splats, cameras, output_path):
    """点群の可視化"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    means = splats["means"].cpu().numpy()
    opacities = torch.sigmoid(splats["opacities"]).cpu().numpy()
    
    # サンプリングして表示（点が多すぎる場合）
    if len(means) > 50000:
        indices = np.random.choice(len(means), 50000, replace=False)
        means = means[indices]
        opacities = opacities[indices]
    
    # 不透明度で色分け
    scatter = ax.scatter(means[:, 0], means[:, 1], means[:, 2], 
                        c=opacities, cmap='viridis', s=0.1, alpha=0.5)
    plt.colorbar(scatter, ax=ax, label='Opacity')
    
    # カメラ位置をプロット
    for name, camtoworld, K, h, w, _ in cameras:
        pos = camtoworld[0, :3, 3].cpu().numpy()
        ax.scatter(pos[0], pos[1], pos[2], c='red', s=100, marker='^')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud with Camera Positions')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def get_all_cameras(data_dir, device="cuda", target_indices=None):
    """すべてのカメラパラメータを取得"""
    parser = Parser(data_dir=data_dir, factor=1, normalize=False, test_every=8)
    dataset = Dataset(parser, split="train")
    image_names = parser.image_names
    
    cameras = []
    for i, data in enumerate(dataset):
        if target_indices is not None and i not in target_indices:
            continue
        
        dataset_idx = dataset.indices[i]
        image_name = image_names[dataset_idx] if dataset_idx < len(image_names) else f"image_{dataset_idx:02d}.jpg"
        camtoworld = data["camtoworld"].unsqueeze(0).to(device)
        K = data["K"].unsqueeze(0).to(device)
        
        image = data["image"]
        if isinstance(image, torch.Tensor):
            height, width = image.shape[:2]
        else:
            height, width = image.shape[:2]
        
        original_image_path = os.path.join(data_dir, "images", image_name)
        if os.path.exists(original_image_path):
            original_image = imageio.imread(original_image_path)
            if original_image.shape[:2] != (height, width):
                original_image = np.array(Image.fromarray(original_image).resize((width, height), Image.BICUBIC))
        else:
            original_image = None
        
        cameras.append((image_name, camtoworld, K, height, width, original_image))
    
    return cameras

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # パス設定
    frame2000_data_dir = "/home/rkmtlab-gdep/Desktop/workspace/vggt/examples/mydata_frame2000"
    frames1994_2000_data_dir = "/home/rkmtlab-gdep/Desktop/workspace/vggt/examples/mydata_frames1994_2000"
    
    frame2000_ckpt = "/home/rkmtlab-gdep/Desktop/workspace/vggt/examples/mydata_frame2000/gsplat_results/ckpts/ckpt_4999_rank0.pt"
    frames1994_2000_ckpt = "/home/rkmtlab-gdep/Desktop/workspace/vggt/examples/mydata_frames1994_2000/gsplat_results/ckpts/ckpt_2999_rank0.pt"
    
    output_dir = "/home/rkmtlab-gdep/Desktop/workspace/vggt/examples/comprehensive_evaluation_frame2000"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Loading Data ===")
    all_cameras = get_all_cameras(frame2000_data_dir, device, target_indices=None)
    print(f"Loaded {len(all_cameras)} cameras")
    
    # チェックポイントをロード
    print("\n=== Loading Checkpoints ===")
    splats_frame2000, step_frame2000 = load_checkpoint(frame2000_ckpt, device)
    splats_frames1994_2000, step_frames1994_2000 = load_checkpoint(frames1994_2000_ckpt, device)
    
    # 包括的な評価
    results = {
        "dataset_info": {
            "frame2000_dataset": {
                "n_images": len(all_cameras),
                "training_steps": int(step_frame2000) if isinstance(step_frame2000, (int, float)) else str(step_frame2000)
            },
            "frames1994_2000_dataset": {
                "training_steps": int(step_frames1994_2000) if isinstance(step_frames1994_2000, (int, float)) else str(step_frames1994_2000)
            }
        }
    }
    
    # 1. カメラ位置姿勢の分析
    print("\n=== Analyzing Camera Poses ===")
    camera_analysis = analyze_camera_poses(all_cameras)
    results["camera_analysis"] = camera_analysis
    print(f"  Number of cameras: {camera_analysis['n_cameras']}")
    print(f"  Mean distance between cameras: {camera_analysis['mean_distance_between_cameras']:.4f}")
    print(f"  Mean angle between cameras: {camera_analysis['mean_angle_between_cameras']:.2f}°")
    
    # カメラ軌跡の可視化
    visualize_camera_trajectory(all_cameras, os.path.join(output_dir, "camera_trajectory.png"))
    
    # 2. 点群の分析
    print("\n=== Analyzing Point Clouds ===")
    point_cloud_frame2000 = analyze_point_cloud(splats_frame2000, device)
    point_cloud_frames1994_2000 = analyze_point_cloud(splats_frames1994_2000, device)
    results["point_cloud_analysis"] = {
        "frame2000_dataset": point_cloud_frame2000,
        "frames1994_2000_dataset": point_cloud_frames1994_2000
    }
    print(f"  Frame2000 dataset: {point_cloud_frame2000['n_points']} points, density: {point_cloud_frame2000['density']:.2e}")
    print(f"  Frames1994-2000 dataset: {point_cloud_frames1994_2000['n_points']} points, density: {point_cloud_frames1994_2000['density']:.2e}")
    
    # 点群の可視化
    visualize_point_cloud(splats_frame2000, all_cameras, os.path.join(output_dir, "point_cloud_frame2000.png"))
    visualize_point_cloud(splats_frames1994_2000, all_cameras, os.path.join(output_dir, "point_cloud_frames1994_2000.png"))
    
    # 3. カバー領域の分析
    print("\n=== Analyzing Coverage ===")
    coverage_frame2000 = analyze_coverage(all_cameras, splats_frame2000, device)
    coverage_frames1994_2000 = analyze_coverage(all_cameras, splats_frames1994_2000, device)
    results["coverage_analysis"] = {
        "frame2000_dataset": coverage_frame2000,
        "frames1994_2000_dataset": coverage_frames1994_2000
    }
    print(f"  Frame2000 dataset: mean visible points: {coverage_frame2000['mean_visible_points']:.0f}")
    print(f"  Frames1994-2000 dataset: mean visible points: {coverage_frames1994_2000['mean_visible_points']:.0f}")
    
    # 4. レンダリング品質の評価（全視点）
    print("\n=== Evaluating Rendering Quality (All Viewpoints) ===")
    all_metrics = defaultdict(list)
    per_viewpoint_metrics = {}
    
    for idx, (image_name, camtoworld, K, height, width, original_image) in enumerate(all_cameras):
        print(f"  Processing {image_name}...")
        
        # レンダリング
        tic = time.time()
        render_frame2000 = render_from_camera(splats_frame2000, camtoworld, K, height, width, device)
        render_time_frame2000 = time.time() - tic
        
        tic = time.time()
        render_frames1994_2000 = render_from_camera(splats_frames1994_2000, camtoworld, K, height, width, device)
        render_time_frames1994_2000 = time.time() - tic
        
        # 定量的評価
        if original_image is not None:
            metrics_frame2000 = calculate_metrics(render_frame2000, original_image, device)
            metrics_frames1994_2000 = calculate_metrics(render_frames1994_2000, original_image, device)
            
            metrics_frame2000["render_time"] = render_time_frame2000
            metrics_frames1994_2000["render_time"] = render_time_frames1994_2000
            
            per_viewpoint_metrics[image_name] = {
                "frame2000_dataset": metrics_frame2000,
                "frames1994_2000_dataset": metrics_frames1994_2000
            }
            
            for metric_name in ["PSNR", "SSIM", "LPIPS"]:
                all_metrics[f"frame2000_{metric_name}"].append(metrics_frame2000[metric_name])
                all_metrics[f"frames1994_2000_{metric_name}"].append(metrics_frames1994_2000[metric_name])
    
    # 平均値を計算
    avg_metrics = {}
    for key, values in all_metrics.items():
        avg_metrics[key] = float(np.mean(values))
        avg_metrics[f"{key}_std"] = float(np.std(values))
    
    results["rendering_quality"] = {
        "per_viewpoint": per_viewpoint_metrics,
        "average": {
            "frame2000_dataset": {
                "PSNR": avg_metrics.get("frame2000_PSNR", 0),
                "PSNR_std": avg_metrics.get("frame2000_PSNR_std", 0),
                "SSIM": avg_metrics.get("frame2000_SSIM", 0),
                "SSIM_std": avg_metrics.get("frame2000_SSIM_std", 0),
                "LPIPS": avg_metrics.get("frame2000_LPIPS", 0),
                "LPIPS_std": avg_metrics.get("frame2000_LPIPS_std", 0),
            },
            "frames1994_2000_dataset": {
                "PSNR": avg_metrics.get("frames1994_2000_PSNR", 0),
                "PSNR_std": avg_metrics.get("frames1994_2000_PSNR_std", 0),
                "SSIM": avg_metrics.get("frames1994_2000_SSIM", 0),
                "SSIM_std": avg_metrics.get("frames1994_2000_SSIM_std", 0),
                "LPIPS": avg_metrics.get("frames1994_2000_LPIPS", 0),
                "LPIPS_std": avg_metrics.get("frames1994_2000_LPIPS_std", 0),
            }
        }
    }
    
    # 5. モデルサイズとメモリ使用量
    print("\n=== Analyzing Model Size and Memory ===")
    def get_model_size(splats):
        total_params = 0
        for k, v in splats.items():
            total_params += v.numel()
        return total_params
    
    model_size_frame2000 = get_model_size(splats_frame2000)
    model_size_frames1994_2000 = get_model_size(splats_frames1994_2000)
    
    # メモリ使用量（概算）
    memory_frame2000 = model_size_frame2000 * 4 / (1024**3)  # float32 = 4 bytes
    memory_frames1994_2000 = model_size_frames1994_2000 * 4 / (1024**3)
    
    results["model_analysis"] = {
        "frame2000_dataset": {
            "n_gaussians": int(splats_frame2000["means"].shape[0]),
            "total_parameters": int(model_size_frame2000),
            "estimated_memory_gb": float(memory_frame2000),
        },
        "frames1994_2000_dataset": {
            "n_gaussians": int(splats_frames1994_2000["means"].shape[0]),
            "total_parameters": int(model_size_frames1994_2000),
            "estimated_memory_gb": float(memory_frames1994_2000),
        }
    }
    
    # レンダリング速度の統計
    render_times_frame2000 = [m["frame2000_dataset"]["render_time"] for m in per_viewpoint_metrics.values() if "render_time" in m["frame2000_dataset"]]
    render_times_frames1994_2000 = [m["frames1994_2000_dataset"]["render_time"] for m in per_viewpoint_metrics.values() if "render_time" in m["frames1994_2000_dataset"]]
    
    if render_times_frame2000:
        results["rendering_quality"]["average"]["frame2000_dataset"]["mean_render_time"] = float(np.mean(render_times_frame2000))
        results["rendering_quality"]["average"]["frame2000_dataset"]["mean_fps"] = float(1.0 / np.mean(render_times_frame2000))
    if render_times_frames1994_2000:
        results["rendering_quality"]["average"]["frames1994_2000_dataset"]["mean_render_time"] = float(np.mean(render_times_frames1994_2000))
        results["rendering_quality"]["average"]["frames1994_2000_dataset"]["mean_fps"] = float(1.0 / np.mean(render_times_frames1994_2000))
    
    # 結果を保存
    results_path = os.path.join(output_dir, "comprehensive_evaluation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # サマリーを表示
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*80)
    print(f"\nCamera Analysis:")
    print(f"  Number of cameras: {camera_analysis['n_cameras']}")
    print(f"  Mean distance between cameras: {camera_analysis['mean_distance_between_cameras']:.4f} ± {camera_analysis['std_distance_between_cameras']:.4f}")
    print(f"  Mean angle between cameras: {camera_analysis['mean_angle_between_cameras']:.2f}° ± {camera_analysis['std_angle_between_cameras']:.2f}°")
    
    print(f"\nPoint Cloud Analysis:")
    print(f"  Frame2000 Dataset: {point_cloud_frame2000['n_points']} points, density: {point_cloud_frame2000['density']:.2e}")
    print(f"  Frames1994-2000 Dataset: {point_cloud_frames1994_2000['n_points']} points, density: {point_cloud_frames1994_2000['density']:.2e}")
    
    print(f"\nRendering Quality (Average):")
    print(f"  Frame2000 Dataset:")
    print(f"    PSNR: {avg_metrics.get('frame2000_PSNR', 0):.3f} ± {avg_metrics.get('frame2000_PSNR_std', 0):.3f} dB")
    print(f"    SSIM: {avg_metrics.get('frame2000_SSIM', 0):.4f} ± {avg_metrics.get('frame2000_SSIM_std', 0):.4f}")
    print(f"    LPIPS: {avg_metrics.get('frame2000_LPIPS', 0):.4f} ± {avg_metrics.get('frame2000_LPIPS_std', 0):.4f}")
    print(f"  Frames1994-2000 Dataset:")
    print(f"    PSNR: {avg_metrics.get('frames1994_2000_PSNR', 0):.3f} ± {avg_metrics.get('frames1994_2000_PSNR_std', 0):.3f} dB")
    print(f"    SSIM: {avg_metrics.get('frames1994_2000_SSIM', 0):.4f} ± {avg_metrics.get('frames1994_2000_SSIM_std', 0):.4f}")
    print(f"    LPIPS: {avg_metrics.get('frames1994_2000_LPIPS', 0):.4f} ± {avg_metrics.get('frames1994_2000_LPIPS_std', 0):.4f}")
    
    print(f"\n✓ Comprehensive evaluation completed!")
    print(f"  Results saved in: {output_dir}")
    print(f"  JSON: {results_path}")
    print(f"  Visualizations: camera_trajectory.png, point_cloud_*.png")

if __name__ == "__main__":
    main()

