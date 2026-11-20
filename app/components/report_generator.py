"""
PDF Report Generator
Creates downloadable PDF reports of pothole detections.
"""

from fpdf import FPDF
from pathlib import Path
from datetime import datetime


OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "reports"


class PotholeReport(FPDF):
    """Custom PDF report for pothole detections."""

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "PotholeSight - Detection Report", border=False,
                  ln=True, align="C")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                  ln=True, align="C")
        self.ln(5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_report(stats: dict, detections: list[dict],
                     filename: str = None) -> str:
    """
    Generate a PDF report.

    Args:
        stats: Detection statistics dict (total, high, medium, low, avg_confidence)
        detections: List of detection records
        filename: Output filename (auto-generated if None)

    Returns:
        Path to the generated PDF file
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pothole_report_{timestamp}.pdf"

    filepath = OUTPUT_DIR / filename

    pdf = PotholeReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Summary Section ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Summary", ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 11)
    summary_data = [
        ("Total Potholes Detected", str(stats.get("total", 0))),
        ("High Severity", str(stats.get("high", 0))),
        ("Medium Severity", str(stats.get("medium", 0))),
        ("Low Severity", str(stats.get("low", 0))),
        ("Average Confidence", f"{stats.get('avg_confidence', 0):.2%}"),
    ]

    for label, value in summary_data:
        pdf.cell(90, 8, label, border=1)
        pdf.cell(90, 8, value, border=1, ln=True)

    pdf.ln(10)

    # --- Detections Table ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Detection Log", ln=True)
    pdf.ln(3)

    # Table header
    pdf.set_font("Helvetica", "B", 9)
    col_widths = [15, 45, 25, 25, 35, 45]
    headers = ["ID", "Source File", "Severity", "Confidence", "GPS", "Timestamp"]

    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, border=1, align="C")
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 8)
    for det in detections[:50]:  # Limit to 50 entries
        pdf.cell(col_widths[0], 7, str(det.get("id", "")), border=1, align="C")
        pdf.cell(col_widths[1], 7, str(det.get("source_file", ""))[:25], border=1)
        pdf.cell(col_widths[2], 7, str(det.get("severity", "")), border=1, align="C")
        pdf.cell(col_widths[3], 7, f"{det.get('confidence', 0):.2%}", border=1,
                 align="C")

        lat = det.get("latitude")
        lon = det.get("longitude")
        gps_str = f"{lat:.4f}, {lon:.4f}" if lat and lon else "N/A"
        pdf.cell(col_widths[4], 7, gps_str, border=1, align="C")

        ts = det.get("timestamp", "")
        if ts:
            ts = ts[:19]  # Trim to seconds
        pdf.cell(col_widths[5], 7, ts, border=1, align="C")
        pdf.ln()

    # Save
    pdf.output(str(filepath))
    return str(filepath)


def get_report_bytes(stats: dict, detections: list[dict]) -> bytes:
    """
    Generate PDF and return as bytes (for Streamlit download button).

    Args:
        stats: Detection statistics
        detections: Detection records

    Returns:
        PDF file as bytes
    """
    filepath = generate_report(stats, detections)
    with open(filepath, "rb") as f:
        return f.read()