"""
Page 1: Image Detection
Upload a road image and detect potholes with severity and GPS.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import time

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from app.components.detector import PotholeDetector
from app.components.severity import classify_severity
from app.components.gps_extractor import extract_gps_from_image_bytes
from app.components.database import insert_detection
from app.components.map_builder import create_mini_map

try:
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

st.set_page_config(page_title="Image Detection - PotholeSight", layout="wide")

# Sidebar Configuration
with st.sidebar:
    st.markdown("### Detection Settings")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Filters detections below this probability score."
    )
    st.divider()
    st.caption("Lower threshold increases recall on shallow/distant defects.")

st.title("Image Detection")
st.caption("Upload a single road image for surface defect analysis and spatial indexing.")
st.divider()


@st.cache_resource
def load_detector():
    try:
        return PotholeDetector(confidence=0.35)
    except FileNotFoundError:
        return None


uploaded_file = st.file_uploader(
    "Select Image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported formats: JPG, JPEG, PNG, BMP, WEBP"
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_np

    img_h, img_w = image_bgr.shape[:2]

    gps_data = extract_gps_from_image_bytes(file_bytes)
    detector = load_detector()

    if detector is None:
        st.error("Model weights file (best.pt) not found in model/weights/ directory.")
        st.stop()

    detector.set_confidence(confidence)

    with st.spinner("Processing image..."):
        start_time = time.time()
        detections = detector.detect(image_bgr)
        inference_time = (time.time() - start_time) * 1000

    severities = [classify_severity(det["bbox"], img_w, img_h) for det in detections]
    annotated_bgr = detector.annotate_image(image_bgr, detections, severities)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    st.markdown("#### Inspection View")
    col_orig, col_det = st.columns(2)

    with col_orig:
        with st.container(border=True):
            st.caption("Source Input")
            st.image(image, use_container_width=True)

    with col_det:
        with st.container(border=True):
            st.caption("Inference Result")
            st.image(annotated_rgb, use_container_width=True)

    st.divider()

    if detections:
        num_high = severities.count("High")
        num_med = severities.count("Medium")
        num_low = severities.count("Low")

        st.markdown("#### Detection Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Count", len(detections))
        m2.metric("High Severity", num_high)
        m3.metric("Medium Severity", num_med)
        m4.metric("Low Severity", num_low)
        m5.metric("Inference Latency", f"{inference_time:.0f} ms")

        st.markdown("#### Detection Log")
        for i, (det, sev) in enumerate(zip(detections, severities)):
            with st.expander(f"Defect #{i+1} — {sev} Severity ({det['confidence']:.0%} Confidence)"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Confidence:** {det['confidence']:.2%}")
                    st.write(f"**Severity:** {sev}")
                    st.write(f"**Bounding Box:** {det['bbox_pixels']}")
                with c2:
                    x1, y1, x2, y2 = det["bbox_pixels"]
                    crop = image_np[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                    if crop.size > 0:
                        st.image(crop, caption=f"Region Crop #{i+1}", width=220)

        st.divider()
        st.markdown("#### Spatial Coordinates")
        if gps_data:
            st.info(f"Coordinates: {gps_data['latitude']} N, {gps_data['longitude']} E")
            if HAS_FOLIUM:
                mini_map = create_mini_map(
                    gps_data["latitude"],
                    gps_data["longitude"],
                    severities[0] if severities else "High"
                )
                st_folium(mini_map, width=700, height=320)
        else:
            st.caption("No embedded GPS metadata detected in source file.")

        for det, sev in zip(detections, severities):
            insert_detection(
                source_file=uploaded_file.name,
                frame_number=0,
                bbox=det["bbox"],
                confidence=det["confidence"],
                severity=sev,
                latitude=gps_data["latitude"] if gps_data else None,
                longitude=gps_data["longitude"] if gps_data else None,
            )
        st.success(f"{len(detections)} record(s) indexed to database.")

    else:
        st.info("No surface defects detected in this frame.")
        st.metric("Inference Latency", f"{inference_time:.0f} ms")

else:
    st.info("Upload an image file to begin analysis.")