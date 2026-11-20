# PotholeSight - Automated Road Surface Inspection System

An automated computer vision and spatial mapping platform designed for road surface defect localization, severity classification, and geospatial tracking.

---

## Overview

PotholeSight analyzes static road photographs and dashcam video feeds to detect surface defects in real time. Detected anomalies are graded by physical severity (Low, Medium, High) based on relative dimensional area and geolocated using EXIF metadata or synchronized GPX telemetry tracks.

---

## Features

- **Single-Frame Image Detection**: Evaluates uploaded road imagery, annotates localized defects, and extracts embedded EXIF GPS coordinates for map plotting.
- **Batch Video Stream Analysis**: Evaluates recorded dashcam streams frame-by-frame with configurable temporal sampling, producing annotated video exports and chronological detection timelines.
- **Automated Severity Classification**: Dynamically computes defect area-to-frame ratios to classify potholes into Low, Medium, or High severity tiers without extra inference overhead.
- **Geospatial Mapping & Analytics**: Renders interactive spatial distributions and defect density heatmaps using Folium and OpenStreetMap.
- **Inspection Reporting**: Generates structured PDF inspection summaries and CSV records for road maintenance auditing.

---

## System Architecture

```text
[Input Sources] -> (Images / Dashcam Video / GPX Telemetry)
       |
       v
[Preprocessing] -> (Frame extraction, 640x640 normalization)
       |
       v
[Inference Engine] -> (Fine-tuned YOLOv8 Backbone)
       |
       v
[Classification & Telemetry] -> (Area-ratio severity grading + GPS extraction)
       |
       v
[Data & Presentation] -> (SQLite persistence -> Interactive Streamlit UI / Folium Maps / PDF Export)
```

---

## Tech Stack

| Domain | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Deep Learning Backend | PyTorch |
| Web Application | Streamlit |
| Computer Vision | OpenCV |
| Mapping & Spatial Data | Folium, OpenStreetMap |
| Data Analytics & Charts | Plotly, Pandas |
| Report Generation | FPDF2 |
| Local Database | SQLite |

---

## Installation and Local Setup

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/pothole-detection.git
cd pothole-detection
```

### 3. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Application
```bash
python -m streamlit run app/main.py
```
The interface will automatically launch at `http://localhost:8501`.

---

## Project Structure

```text
pothole-detection/
├── app/
│   ├── main.py                  # Main application portal
│   ├── pages/
│   │   ├── 1_Image_Detection.py # Single image evaluation module
│   │   ├── 2_Video_Detection.py # Video batch evaluation module
│   │   └── 3_Analytics.py       # Spatial maps and reporting dashboard
│   └── components/
│       ├── detector.py          # YOLOv8 inference wrapper
│       ├── severity.py          # Relative area severity classifier
│       ├── gps_extractor.py     # EXIF & GPX parsing utilities
│       ├── map_builder.py       # Interactive Folium map utilities
│       ├── report_generator.py  # PDF summary generator
│       └── database.py          # SQLite database interface
├── model/
│   ├── dataset.yaml             # YOLO dataset configuration
│   └── weights/
│       └── best.pt              # Fine-tuned YOLOv8 weights
├── database/
│   └── detections.db            # Auto-generated SQLite storage
├── outputs/                     # Generated annotated videos and PDF reports
├── requirements.txt             # Project dependencies
├── .gitignore                   # Ignored files and directories
└── README.md                    # Project documentation
```

---

## Operational Parameters

- **Default Input Resolution**: 640 x 640
- **Base Architecture**: YOLOv8
- **Severity Thresholds**:
  - Low Severity: Area < 2% of frame
  - Medium Severity: Area between 2% and 6% of frame
  - High Severity: Area >= 6% of frame
- **Recommended Video Input**: H.264 MP4 format under 50 MB