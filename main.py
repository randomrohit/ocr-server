"""
FastAPI OCR Server for Salesforce Integration
Handles: image validation, OpenCV preprocessing, OCR (TrOCR for handwriting,
Tesseract fallback for printed text), field extraction, structured JSON response.
"""

import base64
import io
import re
from typing import Optional

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="Salesforce OCR Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_MB = 8
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG"}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class OCRRequest(BaseModel):
    image_base64: str          # base64 encoded image (no data: prefix)
    file_name: Optional[str] = "document"


class ExtractedFields(BaseModel):
    installed_location_name: Optional[str] = None   # Installed_Location_Name__c
    installation_code: Optional[str] = None          # Installation_Code__c e.g. "00158-C252"
    installation_date: Optional[str] = None          # Installation_Date__c, normalized YYYY-MM-DD
    installation_location: Optional[str] = None       # Installation_Location__c (address)
    model_name: Optional[str] = None                  # used by Apex for Order_Product__c lookup


class OCRResponse(BaseModel):
    success: bool
    raw_text: str = ""
    fields: ExtractedFields = ExtractedFields()
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Preprocessing (OpenCV)
# ---------------------------------------------------------------------------
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Deskew, denoise, binarize for cleaner OCR input."""
    img = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Downscale very large images — free-tier CPU/RAM is limited and
    # fastNlMeansDenoising is slow at high resolution.
    max_dim = 1800
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Lighter denoise (median blur) — much faster than fastNlMeansDenoising,
    # sufficient for phone-photo document noise.
    denoised = cv2.medianBlur(gray, 3)

    # Otsu's threshold — cleaner than adaptive for evenly-lit scanned/photographed forms
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


# ---------------------------------------------------------------------------
# OCR engine (Tesseract — printed labels + handwritten fill-ins on this form)
# ---------------------------------------------------------------------------
def run_tesseract(processed_img: np.ndarray) -> str:
    # --psm 6: uniform block of text. -c tessedit_do_invert=0: skip auto-invert
    # check (saves a pass). --oem 1: LSTM only (skip legacy engine combo pass).
    config = "--psm 6 --oem 1 -c tessedit_do_invert=0"
    return pytesseract.image_to_string(processed_img, config=config, timeout=60)


# ---------------------------------------------------------------------------
# Field extraction — tuned for ALAN Electronic Systems
# "Installation Report-cum-Warranty Certificate" layout ONLY.
# ---------------------------------------------------------------------------
def _normalize_date_ddmmyy(raw: str) -> Optional[str]:
    """Convert DD/MM/YY or DD/MM/YYYY -> YYYY-MM-DD. Indian format assumed."""
    raw = raw.strip().replace(" ", "")
    m = re.match(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})", raw)
    if not m:
        return None
    day, month, year = m.groups()
    if len(year) == 2:
        year = "20" + year
    try:
        day_i, month_i = int(day), int(month)
        if not (1 <= day_i <= 31 and 1 <= month_i <= 12):
            return None
        return f"{year}-{month_i:02d}-{day_i:02d}"
    except ValueError:
        return None


def extract_fields(text: str) -> ExtractedFields:
    fields = ExtractedFields()

    # --- Hospital/Clinic Name -> Installed_Location_Name__c ---
    m = re.search(r"Hospital.{0,15}Clinic.{0,10}Name\s*[:\-]*\s*(.+)", text, re.IGNORECASE)
    if m:
        fields.installed_location_name = m.group(1).strip().strip("_").strip()

    # --- Model Name -> used for Order_Product__c fuzzy lookup ---
    m = re.search(r"Model\s*Name\s*[:\-=]*\s*([A-Za-z0-9][A-Za-z0-9\s\-]{2,40})", text, re.IGNORECASE)
    if m:
        fields.model_name = m.group(1).strip().strip("_").strip()

    # --- Date of Installation -> Installation_Date__c ---
    m = re.search(r"Date\s*of\s*Installation\s*[:\-]?\s*([\d/\-.]{6,10})", text, re.IGNORECASE)
    if m:
        fields.installation_date = _normalize_date_ddmmyy(m.group(1))

    # --- Address -> Installation_Location__c ---
    # Address may wrap onto a second line before the next label (Contact Person).
    m = re.search(
        r"Address\s*[:\-]\s*(.+?)(?=Contact\s*Person|Contact\s*Number|E-?Mail|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if m:
        addr = " ".join(line.strip() for line in m.group(1).splitlines() if line.strip())
        fields.installation_location = addr.strip()

    # --- Installation Code -> Installation_Code__c ---
    # Hologram box shows two stacked values e.g. "00158" and "C 252" -> "00158-C252"
    m = re.search(r"\b(\d{5})\b.{0,15}\bC\s?(\d{3})\b", text, re.IGNORECASE | re.DOTALL)
    if m:
        fields.installation_code = f"{m.group(1)}-C{m.group(2)}"
    else:
        # fallback: look for two separate short tokens near each other
        code_num = re.search(r"\b(\d{5})\b", text)
        code_c = re.search(r"\bC\s?(\d{3})\b", text)
        if code_num and code_c:
            fields.installation_code = f"{code_num.group(1)}-C{code_c.group(1)}"

    return fields


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/ocr/extract", response_model=OCRResponse)
def extract_document(req: OCRRequest):
    import time
    t0 = time.time()
    print(f"[extract] request received")

    # --- validate ---
    try:
        img_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    size_mb = len(img_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

    try:
        pil_image = Image.open(io.BytesIO(img_bytes))
        pil_image.verify()
        pil_image = Image.open(io.BytesIO(img_bytes))  # reopen after verify()
    except Exception:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    if pil_image.format not in ALLOWED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {pil_image.format}")

    print(f"[extract] validated in {time.time()-t0:.1f}s, size={size_mb:.2f}MB, format={pil_image.format}, dims={pil_image.size}")

    # --- preprocess ---
    t1 = time.time()
    processed = preprocess_image(pil_image)
    print(f"[extract] preprocessed in {time.time()-t1:.1f}s")

    # --- OCR ---
    t2 = time.time()
    try:
        raw_text = run_tesseract(processed)
    except RuntimeError as e:
        print(f"[extract] OCR TIMEOUT after {time.time()-t2:.1f}s: {e}")
        return OCRResponse(success=False, error="OCR timed out — image too large/complex for current server resources")
    except Exception as e:
        print(f"[extract] OCR FAILED after {time.time()-t2:.1f}s: {e}")
        return OCRResponse(success=False, error=f"OCR engine failed: {str(e)}")
    print(f"[extract] OCR done in {time.time()-t2:.1f}s, text_len={len(raw_text)}")

    # --- field extraction ---
    fields = extract_fields(raw_text)

    print(f"[extract] TOTAL time {time.time()-t0:.1f}s")
    return OCRResponse(success=True, raw_text=raw_text, fields=fields)


@app.get("/health")
def health_check():
    return {"status": "ok"}
