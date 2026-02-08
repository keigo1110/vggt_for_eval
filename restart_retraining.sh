#!/bin/bash
cd /home/rkmtlab-gdep/Desktop/workspace/gsplat

FRAME_DIR="/home/rkmtlab-gdep/Desktop/workspace/vggt/evaluation_results_cooking/black_regions/gaussian_splatting/frame_001650/temporal_baseline"
LOG_FILE="${FRAME_DIR}/logs/gsplat_retraining_restart.log"

export PYTHONPATH="/home/rkmtlab-gdep/Desktop/workspace/gsplat:$PYTHONPATH"

python examples/simple_trainer.py default \
    --data_factor 1 \
    --data_dir "${FRAME_DIR}" \
    --result_dir "${FRAME_DIR}/gsplat_results" \
    --max_steps 7000 \
    --render_traj_path none \
    --disable_viewer \
    2>&1 | tee "${LOG_FILE}"
