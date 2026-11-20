"""
Page 2: Video Detection
Upload dashcam video footage to detect potholes across frames.
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path
import sys
import time

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from app.components.detector import PotholeDetector
from app.components.severity import classify_severity
from app.components.gps_extractor import parse_gpx_bytes, interpolate_gps_for_frame
from app.components.database import insert_detection
from app.components.map_builder import create_detection_map

try:
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

st.set_page_config(page_title="Video Detection - PotholeSight", layout="wide")

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
    st.markdown("### Processing Controls")
    skip_factor = st.slider(
        "Frame Sampling Rate",
        min_value=1, max_value=30, value=5,
        help="Evaluates 1 out of every N frames to optimize throughput."
    )
    max_frames = st.slider(
        "Max Frame Limit",
        min_value=50, max_value=1000, value=300,
        help="Upper bound on total frames processed per session."
    )

st.title("Video Detection")
st.caption("Batch frame defect extraction and spatial trajectory mapping.")
st.divider()


@st.cache_resource
def load_detector():
    try:
        return PotholeDetector()
    except FileNotFoundError:
        return None


col_video, col_gpx = st.columns([2, 1])

with col_video:
    video_file = st.file_uploader(
        "Select Road Video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Recommended size: Under 50 MB"
    )

with col_gpx:
    gpx_file = st.file_uploader(
        "Select GPX Route (Optional)",
        type=["gpx"],
        help="Synchronizes defect timestamps with GPS coordinates"
    )

if video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    gps_points = []
    if gpx_file is not None:
        try:
            gps_points = parse_gpx_bytes(gpx_file.read())
            st.success(f"Indexed {len(gps_points)} telemetry points from GPX track.")
        except Exception as e:
            st.warning(f"Could not parse GPX track: {e}")

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.caption(
        f"Stream Specs: {total_frames} frames | {fps} FPS | {vid_width}x{vid_height} | "
        f"Duration: {total_frames/fps:.1f}s"
    )

    detector = load_detector()
    if detector is None:
        st.error("Model weights file (best.pt) not found in model/weights/.")
        st.stop()

    detector.set_confidence(confidence)

    if st.button("Start Processing", type="primary", use_container_width=True):
        all_detections = []
        progress_bar = st.progress(0, text="Evaluating frames...")
        preview_slot = st.empty()

        output_dir = project_root / "outputs" / "annotated"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"annotated_{video_file.name}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (vid_width, vid_height))

        frame_idx = 0
        processed_count = 0
        start_time = time.time()

        while cap.isOpened() and processed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = frame.copy()

            if frame_idx % skip_factor == 0:
                dets = detector.detect(frame)
                if dets:
                    severities = [classify_severity(d["bbox"], vid_width, vid_height) for d in dets]
                    annotated_frame = detector.annotate_image(frame, dets, severities)

                    gps = None
                    if gps_points:
                        gps = interpolate_gps_for_frame(frame_idx, total_frames, gps_points)

                    for det, sev in zip(dets, severities):
                        timestamp_sec = frame_idx / fps
                        all_detections.append({
                            "frame": frame_idx,
                            "time": f"{timestamp_sec:.1f}s",
                            "severity": sev,
                            "confidence": det["confidence"],
                            "latitude": gps["latitude"] if gps else None,
                            "longitude": gps["longitude"] if gps else None,
                        })

                        insert_detection(
                            source_file=video_file.name,
                            frame_number=frame_idx,
                            bbox=det["bbox"],
                            confidence=det["confidence"],
                            severity=sev,
                            latitude=gps["latitude"] if gps else None,
                            longitude=gps["longitude"] if gps else None,
                        )

                processed_count += 1

            out_writer.write(annotated_frame)
            progress = min(frame_idx / total_frames, 1.0)
            progress_bar.progress(progress, text=f"Frame {frame_idx}/{total_frames} | Detected Defects: {len(all_detections)}")

            if frame_idx % (skip_factor * 6) == 0:
                preview_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                preview_slot.image(preview_rgb, caption=f"Frame {frame_idx}", use_container_width=True)

            frame_idx += 1

        cap.release()
        out_writer.release()
        elapsed = time.time() - start_time
        progress_bar.progress(1.0, text="Processing completed.")

        st.divider()
        st.markdown("#### Processing Summary")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Defects Identified", len(all_detections))
        r2.metric("Evaluated Frames", processed_count)
        r3.metric("Execution Time", f"{elapsed:.1f}s")
        r4.metric("Throughput", f"{processed_count/elapsed:.1f} FPS")

        if all_detections:
            st.markdown("#### Defect Timeline")
            import pandas as pd
            df = pd.DataFrame(all_detections)
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0, max_value=1, format="%.0%%"
                    ),
                    "severity": st.column_config.TextColumn("Severity"),
                }
            )

            gps_dets = [d for d in all_detections if d.get("latitude")]
            if gps_dets and HAS_FOLIUM:
                st.markdown("#### Route Trajectory Map")
                route_map = create_detection_map(all_detections)
                st_folium(route_map, width=900, height=450)

            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Annotated Video",
                    data=f.read(),
                    file_name=f"annotated_{video_file.name}",
                    mime="video/mp4"
                )
        else:
            st.info("No surface defects detected in this footage.")

    try:
        Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        pass

else:
    st.info("Upload a video file to begin analysis.")