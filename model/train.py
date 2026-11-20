"""
YOLOv8 Training Script for Pothole Detection
Run this on Google Colab (free T4 GPU) or locally if you have a GPU.

Usage (Colab):
    1. Upload this file and dataset.yaml to Colab
    2. Upload your dataset to Colab or use Roboflow API
    3. Run: python train.py

Usage (Local with GPU):
    python model/train.py
"""

from ultralytics import YOLO
from pathlib import Path
import shutil


def train():
    """Fine-tune YOLOv8n on pothole dataset."""

    print("=" * 60)
    print("  PotholeSight - Model Training")
    print("=" * 60)

    # --- Configuration ---
    MODEL_SIZE = "yolov8n.pt"      # Nano model (fastest, best for CPU laptop)
    DATASET_YAML = "dataset.yaml"   # Dataset config file
    EPOCHS = 50                     # Training epochs (increase if accuracy is low)
    IMG_SIZE = 640                  # Input image size
    BATCH_SIZE = 16                 # Batch size (reduce if GPU runs out of memory)
    PROJECT = "runs"                # Output directory
    NAME = "pothole_detector"       # Experiment name

    # --- Load pretrained model ---
    print(f"\n📦 Loading pretrained model: {MODEL_SIZE}")
    model = YOLO(MODEL_SIZE)

    # --- Train ---
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    print(f"   Dataset: {DATASET_YAML}")
    print(f"   Image size: {IMG_SIZE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print()

    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,

        # Augmentation settings
        augment=True,
        hsv_h=0.015,       # Hue augmentation
        hsv_s=0.7,         # Saturation augmentation
        hsv_v=0.4,         # Value/brightness augmentation
        degrees=10.0,      # Random rotation
        translate=0.1,     # Random translation
        scale=0.5,         # Random scale
        fliplr=0.5,        # Horizontal flip probability
        flipud=0.0,        # No vertical flip (roads don't flip upside down)
        mosaic=1.0,        # Mosaic augmentation
        mixup=0.1,         # Mixup augmentation

        # Training settings
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        patience=10,       # Early stopping patience
        verbose=True,
    )

    # --- Evaluate on test set ---
    print("\n📊 Evaluating on test set...")
    metrics = model.val(data=DATASET_YAML, split="test")

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")
    print("=" * 60)

    # --- Copy best weights to model/weights/ ---
    best_weights = Path(PROJECT) / NAME / "weights" / "best.pt"
    target_dir = Path(__file__).resolve().parent / "weights"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "best.pt"

    if best_weights.exists():
        shutil.copy2(str(best_weights), str(target_path))
        print(f"\n✅ Best weights saved to: {target_path}")
    else:
        print(f"\n⚠️ Could not find best.pt at {best_weights}")

    return results


if __name__ == "__main__":
    train()