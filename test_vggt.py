#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import glob
import os

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check if model is already downloaded
    model_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/model.pt")
    if os.path.exists(model_path):
        print(f"Model found at {model_path}")
        file_size = os.path.getsize(model_path) / (1024**3)  # GB
        print(f"Model size: {file_size:.2f} GB")
    else:
        print("Model not found. Will download on first run.")
    
    print("\nInitializing VGGT model...")
    model = VGGT()
    
    if os.path.exists(model_path):
        print("Loading model from cache...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Downloading model from Hugging Face...")
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, map_location=device))
    
    model.eval()
    model = model.to(device)
    print("✓ Model loaded successfully!")
    
    # Test with a small number of images
    image_folder = "examples/kitchen/images/"
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))[:3]  # Use only first 3 images for quick test
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    print(f"\nLoading {len(image_files)} images from {image_folder}...")
    images = load_and_preprocess_images(image_files).to(device)
    print(f"Preprocessed images shape: {images.shape}")
    
    print("\nRunning inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using dtype: {dtype}")
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    
    print("✓ Inference completed successfully!")
    print("\nPrediction keys:", list(predictions.keys()))
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
    
    print("\n✓ VGGT is working correctly!")

if __name__ == "__main__":
    main()

