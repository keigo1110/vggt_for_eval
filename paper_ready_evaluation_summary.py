#!/usr/bin/env python
"""
論文用の包括的な評価サマリーを生成
すべての評価指標を統合して、論文で報告可能な形式で出力
"""
import json
import os
from pathlib import Path
import numpy as np

def generate_comprehensive_summary():
    """すべての評価結果を統合して論文用サマリーを生成"""
    
    base_dir = "/home/rkmtlab-gdep/Desktop/workspace/vggt/examples/comprehensive_evaluation_frame2000"
    
    # 既存の評価結果を読み込む
    main_results_path = os.path.join(base_dir, "comprehensive_evaluation.json")
    additional_results_path = os.path.join(base_dir, "additional_evaluation_metrics.json")
    
    with open(main_results_path, "r") as f:
        main_results = json.load(f)
    
    additional_results = {}
    if os.path.exists(additional_results_path):
        with open(additional_results_path, "r") as f:
            additional_results = json.load(f)
    
    # 論文用の包括的なサマリーを生成
    summary = {
        "paper_ready_summary": {}
    }
    
    # 1. データセット情報
    summary["paper_ready_summary"]["dataset"] = {
        "n_images": main_results["dataset_info"]["frame2000_dataset"]["n_images"],
        "training_steps": {
            "frame2000": main_results["dataset_info"]["frame2000_dataset"]["training_steps"],
            "frames1994_2000": main_results["dataset_info"]["frames1994_2000_dataset"]["training_steps"]
        }
    }
    
    # 2. レンダリング品質（主要指標）
    avg_frame2000 = main_results["rendering_quality"]["average"]["frame2000_dataset"]
    avg_frames1994_2000 = main_results["rendering_quality"]["average"]["frames1994_2000_dataset"]
    
    summary["paper_ready_summary"]["rendering_quality"] = {
        "frame2000_dataset": {
            "PSNR_dB": f"{avg_frame2000['PSNR']:.2f} ± {avg_frame2000['PSNR_std']:.2f}",
            "SSIM": f"{avg_frame2000['SSIM']:.4f} ± {avg_frame2000['SSIM_std']:.4f}",
            "LPIPS": f"{avg_frame2000['LPIPS']:.4f} ± {avg_frame2000['LPIPS_std']:.4f}",
            "mean_render_time_ms": avg_frame2000.get("mean_render_time", 0) * 1000,
            "mean_fps": avg_frame2000.get("mean_fps", 0)
        },
        "frames1994_2000_dataset": {
            "PSNR_dB": f"{avg_frames1994_2000['PSNR']:.2f} ± {avg_frames1994_2000['PSNR_std']:.2f}",
            "SSIM": f"{avg_frames1994_2000['SSIM']:.4f} ± {avg_frames1994_2000['SSIM_std']:.4f}",
            "LPIPS": f"{avg_frames1994_2000['LPIPS']:.4f} ± {avg_frames1994_2000['LPIPS_std']:.4f}",
            "mean_render_time_ms": avg_frames1994_2000.get("mean_render_time", 0) * 1000,
            "mean_fps": avg_frames1994_2000.get("mean_fps", 0)
        }
    }
    
    # 3. 点群統計
    pc_frame2000 = main_results["point_cloud_analysis"]["frame2000_dataset"]
    pc_frames1994_2000 = main_results["point_cloud_analysis"]["frames1994_2000_dataset"]
    
    summary["paper_ready_summary"]["point_cloud"] = {
        "frame2000_dataset": {
            "n_gaussians": pc_frame2000["n_points"],
            "density_points_per_unit3": f"{pc_frame2000['density']:.2e}",
            "valid_ratio": f"{pc_frame2000['valid_ratio']:.4f}",
            "opacity_mean_std": f"{pc_frame2000['opacity_mean']:.4f} ± {pc_frame2000['opacity_std']:.4f}",
            "scale_mean_std": f"{pc_frame2000['scale_mean']:.4f} ± {pc_frame2000['scale_std']:.4f}"
        },
        "frames1994_2000_dataset": {
            "n_gaussians": pc_frames1994_2000["n_points"],
            "density_points_per_unit3": f"{pc_frames1994_2000['density']:.2e}",
            "valid_ratio": f"{pc_frames1994_2000['valid_ratio']:.4f}",
            "opacity_mean_std": f"{pc_frames1994_2000['opacity_mean']:.4f} ± {pc_frames1994_2000['opacity_std']:.4f}",
            "scale_mean_std": f"{pc_frames1994_2000['scale_mean']:.4f} ± {pc_frames1994_2000['scale_std']:.4f}"
        }
    }
    
    # 4. カメラ分析
    cam = main_results["camera_analysis"]
    summary["paper_ready_summary"]["camera_setup"] = {
        "n_cameras": cam["n_cameras"],
        "mean_distance_between_cameras": f"{cam['mean_distance_between_cameras']:.4f} ± {cam['std_distance_between_cameras']:.4f}",
        "mean_angle_between_cameras_deg": f"{cam['mean_angle_between_cameras']:.2f} ± {cam['std_angle_between_cameras']:.2f}",
        "mean_fov_horizontal_deg": main_results["coverage_analysis"]["frame2000_dataset"]["mean_fov_horizontal"],
        "mean_fov_vertical_deg": main_results["coverage_analysis"]["frame2000_dataset"]["mean_fov_vertical"]
    }
    
    # 5. モデル効率
    if "model_analysis" in main_results:
        model_frame2000 = main_results["model_analysis"]["frame2000_dataset"]
        model_frames1994_2000 = main_results["model_analysis"]["frames1994_2000_dataset"]
        
        summary["paper_ready_summary"]["model_efficiency"] = {
            "frame2000_dataset": {
                "n_gaussians": model_frame2000["n_gaussians"],
                "total_parameters": model_frame2000["total_parameters"],
                "memory_gb": f"{model_frame2000['estimated_memory_gb']:.3f}",
                "parameters_per_gaussian": model_frame2000["total_parameters"] // model_frame2000["n_gaussians"]
            },
            "frames1994_2000_dataset": {
                "n_gaussians": model_frames1994_2000["n_gaussians"],
                "total_parameters": model_frames1994_2000["total_parameters"],
                "memory_gb": f"{model_frames1994_2000['estimated_memory_gb']:.3f}",
                "parameters_per_gaussian": model_frames1994_2000["total_parameters"] // model_frames1994_2000["n_gaussians"]
            }
        }
    
    # 6. 計算コスト
    if "computational_cost" in additional_results:
        comp_frame2000 = additional_results["computational_cost"]["frame2000_dataset"]
        comp_frames1994_2000 = additional_results["computational_cost"]["frames1994_2000_dataset"]
        
        summary["paper_ready_summary"]["computational_cost"] = {
            "frame2000_dataset": {
                "total_flops": f"{comp_frame2000['total_flops']:.2e}",
                "flops_per_pixel": f"{comp_frame2000['flops_per_pixel']:.2e}"
            },
            "frames1994_2000_dataset": {
                "total_flops": f"{comp_frames1994_2000['total_flops']:.2e}",
                "flops_per_pixel": f"{comp_frames1994_2000['flops_per_pixel']:.2e}"
            }
        }
    
    # 7. 学習効率
    if "training_curves" in additional_results:
        summary["paper_ready_summary"]["training_efficiency"] = {
            "note": "Training curves available in training_curves.png",
            "available_metrics": additional_results["training_curves"].get("frame2000_dataset", {}).get("available_scalars", [])
        }
    
    # 8. 各視点の詳細結果（論文の補足資料用）
    summary["paper_ready_summary"]["per_viewpoint_details"] = {}
    for viewpoint, metrics in main_results["rendering_quality"]["per_viewpoint"].items():
        summary["paper_ready_summary"]["per_viewpoint_details"][viewpoint] = {
            "frame2000": {
                "PSNR": f"{metrics['frame2000_dataset']['PSNR']:.2f}",
                "SSIM": f"{metrics['frame2000_dataset']['SSIM']:.4f}",
                "LPIPS": f"{metrics['frame2000_dataset']['LPIPS']:.4f}"
            },
            "frames1994_2000": {
                "PSNR": f"{metrics['frames1994_2000_dataset']['PSNR']:.2f}",
                "SSIM": f"{metrics['frames1994_2000_dataset']['SSIM']:.4f}",
                "LPIPS": f"{metrics['frames1994_2000_dataset']['LPIPS']:.4f}"
            }
        }
    
    # 結果を保存
    summary_path = os.path.join(base_dir, "paper_ready_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # 人間が読みやすい形式でも出力
    print("\n" + "="*80)
    print("PAPER-READY EVALUATION SUMMARY")
    print("="*80)
    print("\n1. RENDERING QUALITY (Average)")
    print(f"   Frame2000 Dataset:")
    print(f"     PSNR: {avg_frame2000['PSNR']:.2f} ± {avg_frame2000['PSNR_std']:.2f} dB")
    print(f"     SSIM: {avg_frame2000['SSIM']:.4f} ± {avg_frame2000['SSIM_std']:.4f}")
    print(f"     LPIPS: {avg_frame2000['LPIPS']:.4f} ± {avg_frame2000['LPIPS_std']:.4f}")
    if "mean_fps" in avg_frame2000:
        print(f"     Rendering Speed: {avg_frame2000['mean_fps']:.1f} FPS")
    
    print(f"\n   Frames1994-2000 Dataset:")
    print(f"     PSNR: {avg_frames1994_2000['PSNR']:.2f} ± {avg_frames1994_2000['PSNR_std']:.2f} dB")
    print(f"     SSIM: {avg_frames1994_2000['SSIM']:.4f} ± {avg_frames1994_2000['SSIM_std']:.4f}")
    print(f"     LPIPS: {avg_frames1994_2000['LPIPS']:.4f} ± {avg_frames1994_2000['LPIPS_std']:.4f}")
    if "mean_fps" in avg_frames1994_2000:
        print(f"     Rendering Speed: {avg_frames1994_2000['mean_fps']:.1f} FPS")
    
    print("\n2. POINT CLOUD STATISTICS")
    print(f"   Frame2000 Dataset: {pc_frame2000['n_points']:,} Gaussians, Density: {pc_frame2000['density']:.2e}")
    print(f"   Frames1994-2000 Dataset: {pc_frames1994_2000['n_points']:,} Gaussians, Density: {pc_frames1994_2000['density']:.2e}")
    
    print("\n3. MODEL EFFICIENCY")
    if "model_analysis" in main_results:
        print(f"   Frame2000: {main_results['model_analysis']['frame2000_dataset']['n_gaussians']:,} Gaussians, {main_results['model_analysis']['frame2000_dataset']['total_parameters']:,} parameters, {main_results['model_analysis']['frame2000_dataset']['estimated_memory_gb']:.3f} GB")
        print(f"   Frames1994-2000: {main_results['model_analysis']['frames1994_2000_dataset']['n_gaussians']:,} Gaussians, {main_results['model_analysis']['frames1994_2000_dataset']['total_parameters']:,} parameters, {main_results['model_analysis']['frames1994_2000_dataset']['estimated_memory_gb']:.3f} GB")
    
    print("\n4. COMPUTATIONAL COST")
    if "computational_cost" in additional_results:
        print(f"   Frame2000: {additional_results['computational_cost']['frame2000_dataset']['total_flops']:.2e} FLOPs")
        print(f"   Frames1994-2000: {additional_results['computational_cost']['frames1994_2000_dataset']['total_flops']:.2e} FLOPs")
    
    print("\n5. CAMERA SETUP")
    print(f"   {cam['n_cameras']} cameras, Mean distance: {cam['mean_distance_between_cameras']:.4f}, Mean angle: {cam['mean_angle_between_cameras']:.2f}°")
    
    print(f"\n✓ Paper-ready summary saved to: {summary_path}")
    print(f"\nAll evaluation files are in: {base_dir}/")
    print("  - comprehensive_evaluation.json: Full evaluation data")
    print("  - additional_evaluation_metrics.json: Additional metrics")
    print("  - paper_ready_summary.json: Paper-ready summary")
    print("  - evaluation_report.md: Markdown report")
    print("  - evaluation_tables.tex: LaTeX tables")
    print("  - training_curves.png: Training loss curves")
    print("  - camera_trajectory.png: Camera trajectory visualization")
    print("  - point_cloud_*.png: Point cloud visualizations")

if __name__ == "__main__":
    generate_comprehensive_summary()

