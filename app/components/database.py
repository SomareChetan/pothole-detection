"""
SQLite Database Module
Stores and retrieves pothole detection records.
"""

import sqlite3
from pathlib import Path
from datetime import datetime


# Default database path
DB_DIR = Path(__file__).resolve().parent.parent.parent / "database"
DB_PATH = DB_DIR / "detections.db"


def get_connection():
    """Get a database connection, creating the DB and table if needed."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # Access columns by name
    create_table(conn)
    return conn


def create_table(conn=None):
    """Create the detections table if it doesn't exist."""
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            frame_number INTEGER DEFAULT 0,
            timestamp TEXT NOT NULL,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL,
            confidence REAL,
            severity TEXT,
            latitude REAL,
            longitude REAL,
            image_path TEXT
        )
    """)
    conn.commit()

    if close_after:
        conn.close()


def insert_detection(source_file: str, frame_number: int,
                     bbox: tuple, confidence: float,
                     severity: str, latitude: float = None,
                     longitude: float = None, image_path: str = None):
    """
    Insert a single detection record.

    Args:
        source_file: Original filename
        frame_number: Frame index (0 for images)
        bbox: (x_center, y_center, width, height) normalized
        confidence: Detection confidence score
        severity: "Low", "Medium", or "High"
        latitude: GPS latitude (optional)
        longitude: GPS longitude (optional)
        image_path: Path to saved annotated crop (optional)
    """
    conn = get_connection()
    conn.execute("""
        INSERT INTO detections
        (source_file, frame_number, timestamp, bbox_x, bbox_y, bbox_w, bbox_h,
         confidence, severity, latitude, longitude, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        source_file,
        frame_number,
        datetime.now().isoformat(),
        bbox[0], bbox[1], bbox[2], bbox[3],
        confidence,
        severity,
        latitude,
        longitude,
        image_path
    ))
    conn.commit()
    conn.close()


def get_all_detections() -> list[dict]:
    """Retrieve all detection records."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM detections ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_detections_by_severity(severity: str) -> list[dict]:
    """Retrieve detections filtered by severity."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM detections WHERE severity = ? ORDER BY id DESC",
        (severity,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_detection_stats() -> dict:
    """Get summary statistics."""
    conn = get_connection()

    total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    high = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE severity = 'High'"
    ).fetchone()[0]
    medium = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE severity = 'Medium'"
    ).fetchone()[0]
    low = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE severity = 'Low'"
    ).fetchone()[0]
    avg_conf = conn.execute(
        "SELECT AVG(confidence) FROM detections"
    ).fetchone()[0]

    conn.close()

    return {
        "total": total,
        "high": high,
        "medium": medium,
        "low": low,
        "avg_confidence": round(avg_conf, 4) if avg_conf else 0.0,
    }


def get_gps_detections() -> list[dict]:
    """Retrieve only detections that have GPS coordinates."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM detections WHERE latitude IS NOT NULL AND longitude IS NOT NULL ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def clear_all_detections():
    """Delete all detection records (for testing/reset)."""
    conn = get_connection()
    conn.execute("DELETE FROM detections")
    conn.commit()
    conn.close()