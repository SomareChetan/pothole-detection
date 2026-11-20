"""
YOLOv8 Pothole Detection Wrapper
Loads the fine-tuned model and runs inference on images/frames.
"""

from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2
import os


class PotholeDetector:
    """Wrapper around YOLOv8 for pothole detection."""

    def __init__(self, model_path: str = None, confidence: float = 0.5):
        """
        Initialize the detector.

        Args:
            model_path: Path to the trained .pt weights file.
                        If None, looks in model/weights/best.pt
            confidence: Minimum confidence threshold (0.0 - 1.0)
        """
        if model_path is None:
            # Default path relative to project root
            project_root = Path(__file__).resolve().parent.parent.parent
            model_path = project_root / "model" / "weights" / "best.pt"

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at: {model_path}\n"
                f"Please train the model first or place best.pt in model/weights/"
            )

        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Run pothole detection on a single image.

        Args:
            image: BGR numpy array (OpenCV format)

        Returns:
            List of detection dicts, each containing:
                - bbox: (x_center, y_center, width, height) normalized 0-1
                - bbox_pixels: (x1, y1, x2, y2) in pixel coordinates
                - confidence: float 0-1
                - class_name: str ("pothole")
        """
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            verbose=False
        )

        detections = []
        img_h, img_w = image.shape[:2]

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # Pixel coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Normalized coordinates
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h

                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names.get(cls_id, "pothole")

                detections.append({
                    "bbox": (cx, cy, bw, bh),
                    "bbox_pixels": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": round(conf, 4),
                    "class_name": cls_name
                })

        return detections

    def annotate_image(self, image: np.ndarray, detections: list[dict],
                       severities: list[str] = None) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image.

        Args:
            image: BGR numpy array
            detections: List of detection dicts from detect()
            severities: Optional list of severity strings matching detections

        Returns:
            Annotated BGR numpy array
        """
        annotated = image.copy()

        severity_colors = {
            "Low": (0, 200, 0),       # Green
            "Medium": (0, 200, 255),   # Yellow/Orange
            "High": (0, 0, 255),       # Red
        }
        default_color = (255, 255, 0)  # Cyan

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox_pixels"]
            conf = det["confidence"]

            # Determine severity and color
            if severities and i < len(severities):
                severity = severities[i]
                color = severity_colors.get(severity, default_color)
            else:
                severity = None
                color = default_color

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Build label text
            label = f"Pothole {conf:.0%}"
            if severity:
                label += f" | {severity}"

            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - 10),
                (x1 + label_w + 10, y1),
                color, -1
            )

            # Draw label text
            cv2.putText(
                annotated, label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2
            )

        return annotated

    def set_confidence(self, confidence: float):
        """Update the confidence threshold."""
        self.confidence = max(0.0, min(1.0, confidence))