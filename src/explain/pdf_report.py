# src/explain/pdf_report.py
import io
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from PIL import Image

def build_report_pdf(
    original_img,       # numpy array H×W×3 uint8
    saliency_img,       # numpy array H×W×3 uint8
    label: str,
    confidence: float,
    explanation: str,
    probs=None,         # optional: 1D array of class probabilities
    labels=None,        # optional: list of class names
    model_name: str | None = None,
    app_version: str | None = None,
):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # ——— Title & Meta —————————————————————————————————————————————
    title = "Brain Tumor Classification Report"
    c.setTitle(title)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height - 50, title)                                    # centered title

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.setFont("Helvetica", 9)
    meta_lines = []
    if app_version:
        meta_lines.append(f"App version: {app_version}")
    if model_name:
        meta_lines.append(f"Model: {model_name}")
    meta_lines.append(f"Generated: {now}")
    y = height - 55
    for line in meta_lines:
        c.drawRightString(width - 40, y, line)
        y -= 12
    c.setStrokeColor(colors.grey)
    c.setLineWidth(1)
    c.line(40, height - 65, width - 40, height - 65)                                   # separator line

    # ——— Images ————————————————————————————————————————————————————————
    def to_imgbuf(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        buf.seek(0)
        return buf

    orig_buf     = to_imgbuf(original_img)
    saliency_buf = to_imgbuf(saliency_img)
    margin = 40
    img_w  = (width - 3*margin) / 2
    aspect = original_img.shape[0] / original_img.shape[1]
    img_h  = img_w * aspect
    img_y  = height - 100 - img_h

    c.drawImage(ImageReader(orig_buf),     margin,        img_y, img_w, img_h)
    c.drawImage(ImageReader(saliency_buf), 2*margin+img_w, img_y, img_w, img_h)

    # ——— Classification —————————————————————————————————————————————
    text_y = img_y - 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, text_y, f"Prediction: {label} — Predicted probability: {confidence*100:.2f}%")

    # ——— Probability Table —————————————————————————————————————————————
    if probs is not None and labels is not None:
        data = [["Class", "Predicted probability"]]
        for lab, p in sorted(zip(labels, probs), key=lambda x: -x[1]):
            data.append([lab, f"{p:.4f}"])
        table = Table(data, colWidths=[100, 80])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("TEXTCOLOR", (0,0), (-1,0), colors.black),
            ("FONTNAME",  (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN",     (1,1), (-1,-1), "RIGHT"),
            ("GRID",      (0,0), (-1,-1), 0.5, colors.grey),
            ("BOTTOMPADDING", (0,0), (-1,0), 6),
        ]))
        table.wrapOn(c, width, height)
        table.drawOn(c, margin, text_y - 100)
        expl_start_y = text_y - 120 - len(labels)*18
    else:
        expl_start_y = text_y - 40

    # ——— Explanation —————————————————————————————————————————————————
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, expl_start_y, "Explanation:")
    c.setFont("Helvetica", 10)
    lines = simpleSplit(explanation, "Helvetica", 10, width - 2*margin)
    for i, line in enumerate(lines):
        c.drawString(margin, expl_start_y - 14*(i+1), line)

    c.save()
    buffer.seek(0)
    return buffer.read()
