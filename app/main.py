"""
PotholeSight - Automated Road Surface Inspection System
Main Portal
"""

import streamlit as st

st.set_page_config(
    page_title="PotholeSight",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.subheader("PotholeSight")
    st.caption("Road Condition Assessment Platform")
    st.divider()

    st.markdown("**Navigation**")
    st.page_link("pages/1_Image_Detection.py", label="Image Detection")
    st.page_link("pages/2_Video_Detection.py", label="Video Detection")
    st.page_link("pages/3_Analytics.py", label="Analytics")

    st.divider()
    st.caption("Framework: YOLOv8 / Streamlit")

# Page Header
st.title("PotholeSight")
st.caption("Automated visual inspection, defect localization, and spatial mapping system.")
st.divider()

# Section 1: Core Processing Modules
st.subheader("Processing Modules")
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    with st.container(border=True):
        st.markdown("### Image Detection")
        st.write(
            "Inspect high-resolution road photographs with single-frame defect localization, "
            "severity categorization, and automatic EXIF coordinate extraction."
        )
        st.page_link(
            "pages/1_Image_Detection.py",
            label="Open Image Detection",
            use_container_width=True
        )

with col2:
    with st.container(border=True):
        st.markdown("### Video Detection")
        st.write(
            "Execute batch frame evaluation on dashcam footage. Produces an annotated "
            "playback stream synchronized with GPX telemetry for route tracking."
        )
        st.page_link(
            "pages/2_Video_Detection.py",
            label="Open Video Detection",
            use_container_width=True
        )

with col3:
    with st.container(border=True):
        st.markdown("### Analytics")
        st.write(
            "Centralized inspection records featuring interactive spatial maps, "
            "density heatmaps, severity distributions, and exportable PDF reports."
        )
        st.page_link(
            "pages/3_Analytics.py",
            label="Open Analytics",
            use_container_width=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# Section 2: Pipeline Architecture
st.subheader("System Workflow")
s1, s2, s3, s4 = st.columns(4, gap="medium")

with s1:
    with st.container(border=True):
        st.caption("STAGE 01")
        st.markdown("#### Ingestion")
        st.write("Accepts single-frame road photos or dashcam video streams with optional GPX tracks.")

with s2:
    with st.container(border=True):
        st.caption("STAGE 02")
        st.markdown("#### Inference")
        st.write("YOLOv8 convolutional network scans each frame and extracts bounding box coordinates.")

with s3:
    with st.container(border=True):
        st.caption("STAGE 03")
        st.markdown("#### Severity")
        st.write("Relative area ratios classify defect impact into Low, Medium, or High categories.")

with s4:
    with st.container(border=True):
        st.caption("STAGE 04")
        st.markdown("#### Analytics")
        st.write("Defect telemetry persists to the local database for spatial mapping and reporting.")