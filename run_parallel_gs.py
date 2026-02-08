#!/usr/bin/env python
"""
GPU 2つを使って黒色領域のGS処理を並列実行するスクリプト
"""
import subprocess
import sys
import os
import time
from pathlib import Path

def main():
    # 引数を取得
    mydata_dir = sys.argv[1] if len(sys.argv) > 1 else "mydata"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "evaluation_results_full"
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 7000
    
    # GS処理対象フレームを取得
    import json
    selection_path = Path(mydata_dir) / "selection.json"
    with open(selection_path, 'r') as f:
        selection_data = json.load(f)
    
    max_frame = max(s['latest_frame_id'] for s in selection_data['selections'])
    gs_frames = list(range(30, max_frame + 1, 30))
    
    # フレームを2つのグループに分割
    mid = len(gs_frames) // 2
    frames_gpu0 = gs_frames[:mid]
    frames_gpu1 = gs_frames[mid:]
    
    print("=" * 60)
    print("GPU 2つで並列処理")
    print("=" * 60)
    print(f"Total GS frames: {len(gs_frames)}")
    print(f"GPU 0: {len(frames_gpu0)} frames ({frames_gpu0[0] if frames_gpu0 else 'N/A'} - {frames_gpu0[-1] if frames_gpu0 else 'N/A'})")
    print(f"GPU 1: {len(frames_gpu1)} frames ({frames_gpu1[0] if frames_gpu1 else 'N/A'} - {frames_gpu1[-1] if frames_gpu1 else 'N/A'})")
    print("=" * 60)
    
    # 出力ディレクトリを作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log0 = output_path / "gs_gpu0.log"
    log1 = output_path / "gs_gpu1.log"
    
    # GPU 0用のプロセス
    if frames_gpu0:
        cmd0 = [
            "python", "process_all_frames.py",
            "--mydata_dir", mydata_dir,
            "--output_dir", str(output_path),
            "--max_steps", str(max_steps),
            "--compare_methods",
            "--skip_white",
            "--gs_start_frame", str(frames_gpu0[0]),
            "--end_frame", str(frames_gpu0[-1]),
        ]
        
        env0 = os.environ.copy()
        env0['CUDA_VISIBLE_DEVICES'] = '0'
        
        print(f"\nStarting GPU 0 process (frames {frames_gpu0[0]}-{frames_gpu0[-1]})...")
        with open(log0, 'w') as f0:
            proc0 = subprocess.Popen(cmd0, env=env0, stdout=f0, stderr=subprocess.STDOUT)
        print(f"  PID: {proc0.pid}, log: {log0}")
    else:
        proc0 = None
    
    # GPU 1用のプロセス
    if frames_gpu1:
        cmd1 = [
            "python", "process_all_frames.py",
            "--mydata_dir", mydata_dir,
            "--output_dir", str(output_path),
            "--max_steps", str(max_steps),
            "--compare_methods",
            "--skip_white",
            "--gs_start_frame", str(frames_gpu1[0]),
            "--end_frame", str(frames_gpu1[-1]),
        ]
        
        env1 = os.environ.copy()
        env1['CUDA_VISIBLE_DEVICES'] = '1'
        
        print(f"Starting GPU 1 process (frames {frames_gpu1[0]}-{frames_gpu1[-1]})...")
        with open(log1, 'w') as f1:
            proc1 = subprocess.Popen(cmd1, env=env1, stdout=f1, stderr=subprocess.STDOUT)
        print(f"  PID: {proc1.pid}, log: {log1}")
    else:
        proc1 = None
    
    print(f"\nMonitor progress with:")
    if proc0:
        print(f"  tail -f {log0}")
    if proc1:
        print(f"  tail -f {log1}")
    print("\nWaiting for processes to complete...")
    
    # プロセスの完了を待つ
    if proc0:
        proc0.wait()
        print("GPU 0 process completed!")
    if proc1:
        proc1.wait()
        print("GPU 1 process completed!")
    
    print("\n✓ All processes completed!")

if __name__ == "__main__":
    main()

