"""
Page 3: Analytics
Inspection metrics, spatial maps, and report generation.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from app.components.database import (
    get_all_detections, get_detection_stats, get_gps_detections,
    clear_all_detections
)
from app.components.map_builder import create_detection_map, create_heatmap
from app.components.report_generator import get_report_bytes

try:
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

st.set_page_config(page_title="Analytics - PotholeSight", layout="wide")

st.title("Analytics")
st.caption("Inspection aggregation, spatial distributions, and reporting.")
st.divider()

stats = get_detection_stats()
all_detections = get_all_detections()
gps_detections = get_gps_detections()

st.markdown("#### Aggregate Metrics")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Defects", stats["total"])
c2.metric("High Severity", stats["high"])
c3.metric("Medium Severity", stats["medium"])
c4.metric("Low Severity", stats["low"])
c5.metric("Avg Confidence", f"{stats['avg_confidence']:.0%}")

st.divider()

if stats["total"] == 0:
    st.info("No detection records currently logged in database.")
    st.stop()

# Charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### Severity Breakdown")
    severity_data = pd.DataFrame({
        "Severity": ["High", "Medium", "Low"],
        "Count": [stats["high"], stats["medium"], stats["low"]],
    })
    fig_pie = px.pie(
        severity_data,
        values="Count",
        names="Severity",
        color="Severity",
        color_discrete_map={
            "High": "#F44336",
            "Medium": "#FF9800",
            "Low": "#4CAF50"
        },
        hole=0.45
    )
    fig_pie.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_pie, use_container_width=True)

with chart_col2:
    st.markdown("#### Confidence Distribution")
    if all_detections:
        conf_values = [d["confidence"] for d in all_detections]
        fig_hist = px.histogram(
            x=conf_values,
            nbins=20,
            labels={"x": "Confidence Score", "y": "Observation Count"},
            color_discrete_sequence=["#0284c7"]
        )
        fig_hist.update_layout(
            height=340,
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# Spatial Mapping
st.markdown("#### Spatial Distributions")

if gps_detections and HAS_FOLIUM:
    map_tab, heat_tab = st.tabs(["Point Distribution", "Defect Density Heatmap"])

    with map_tab:
        pin_map = create_detection_map(gps_detections)
        st_folium(pin_map, width=900, height=480)

    with heat_tab:
        heat_map = create_heatmap(gps_detections)
        st_folium(heat_map, width=900, height=480)

elif not gps_detections:
    st.caption("No geocoded defect entries registered in the current dataset.")
else:
    st.warning("streamlit-folium is required to render interactive map layers.")

st.divider()

# Records Log
st.markdown("#### Inspection Records")

filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    severity_filter = st.multiselect(
        "Severity Filter",
        options=["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )
with filter_col2:
    source_files = list(set(d.get("source_file", "") for d in all_detections))
    source_filter = st.multiselect(
        "Source File Filter",
        options=source_files,
        default=source_files
    )

filtered = [
    d for d in all_detections
    if d.get("severity") in severity_filter and d.get("source_file") in source_filter
]

if filtered:
    df = pd.DataFrame(filtered)
    display_cols = ["id", "source_file", "severity", "confidence", "latitude", "longitude", "timestamp"]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols]

    st.dataframe(
        df_display,
        use_container_width=True,
        column_config={
            "id": "ID",
            "source_file": "Source",
            "severity": "Severity",
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.0%%"),
            "latitude": st.column_config.NumberColumn("Latitude", format="%.4f"),
            "longitude": st.column_config.NumberColumn("Longitude", format="%.4f"),
            "timestamp": "Timestamp",
        },
        hide_index=True
    )
    st.caption(f"Showing {len(filtered)} of {stats['total']} total entries")
else:
    st.caption("No entries match the active criteria.")

st.divider()

# Reporting & Maintenance
st.markdown("#### Export and Maintenance")
export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    if st.button("Generate PDF Report", use_container_width=True):
        with st.spinner("Generating document..."):
            pdf_bytes = get_report_bytes(stats, all_detections)
        st.download_button(
            label="Download PDF File",
            data=pdf_bytes,
            file_name="pothole_inspection_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

with export_col2:
    if all_detections:
        df_export = pd.DataFrame(all_detections)
        csv_data = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV Dataset",
            data=csv_data,
            file_name="pothole_records.csv",
            mime="text/csv",
            use_container_width=True
        )

with export_col3:
    if st.button("Reset Database Records", use_container_width=True, type="secondary"):
        clear_all_detections()
        st.warning("Database records cleared.")
        st.rerun()