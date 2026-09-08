import io
import os
import re
import csv
import json
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np

# Specialized libraries with graceful fallback
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None

try:
    import openpyxl
except ImportError:
    openpyxl = None

_OCR_ENGINE = None

def get_ocr_engine():
    """Lazily initializes RapidOCR neural engine for 100% offline text extraction."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_ENGINE = RapidOCR()
        except Exception as e:
            print(f"[DocParser] RapidOCR init note: {e}")
            _OCR_ENGINE = False
    return _OCR_ENGINE if _OCR_ENGINE is not False else None


def extract_ocr_text_from_image(pil_img: Image.Image) -> str:
    """Runs local neural OCR on a PIL image with spatial line clustering."""
    engine = get_ocr_engine()
    if not engine:
        return ""
    try:
        # Convert to RGB numpy array
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img_np = np.array(pil_img)
        
        results, _ = engine(img_np)
        if not results:
            return ""

        # Each result: [ [ [x1,y1],[x2,y2],[x3,y3],[x4,y4] ], text, score ]
        # Sort items vertically, then cluster into lines
        items = []
        for r in results:
            box, text, score = r[0], r[1], float(r[2]) if len(r) > 2 else 1.0
            if text and text.strip():
                top_y = min(pt[1] for pt in box)
                left_x = min(pt[0] for pt in box)
                height = max(pt[1] for pt in box) - top_y
                items.append({
                    "text": text.strip(),
                    "top_y": top_y,
                    "left_x": left_x,
                    "height": max(12, height),
                    "score": score
                })

        if not items:
            return ""

        # Sort primarily by vertical coordinate
        items.sort(key=lambda x: x["top_y"])

        # Cluster items into visual lines
        lines = []
        current_line = [items[0]]
        current_y = items[0]["top_y"]
        current_h = items[0]["height"]

        for it in items[1:]:
            # If within vertical tolerance of the current line
            if abs(it["top_y"] - current_y) <= (current_h * 0.65):
                current_line.append(it)
            else:
                # Flush current line sorted by horizontal X
                current_line.sort(key=lambda x: x["left_x"])
                lines.append("   ".join(c["text"] for c in current_line))
                current_line = [it]
                current_y = it["top_y"]
                current_h = it["height"]

        if current_line:
            current_line.sort(key=lambda x: x["left_x"])
            lines.append("   ".join(c["text"] for c in current_line))

        return "\n".join(lines).strip()
    except Exception as e:
        print(f"[DocParser] Image OCR error: {e}")
        return ""


def extract_raw_pdf_streams(file_bytes: bytes) -> str:
    """Pure-Python fallback to recover readable text strings from raw PDF streams."""
    try:
        found_chunks = []
        tj_simple = re.compile(rb'\(([^()]{2,500})\)\s*(?:Tj|\'|")', re.DOTALL)
        for m in tj_simple.finditer(file_bytes):
            try:
                txt = m.group(1).decode("latin-1", errors="ignore")
                clean = re.sub(r'\\([0-9]{3}|.)', r'\1', txt).strip()
                if clean and re.search(r'[a-zA-Z0-9]{2,}', clean):
                    found_chunks.append(clean)
            except Exception:
                continue

        tj_array = re.compile(rb'\[(.*?)\]\s*TJ', re.DOTALL)
        for m in tj_array.finditer(file_bytes):
            try:
                inner = re.findall(rb'\(([^()]{2,500})\)', m.group(1))
                line_parts = []
                for b in inner:
                    txt = b.decode("latin-1", errors="ignore")
                    clean = re.sub(r'\\([0-9]{3}|.)', r'\1', txt).strip()
                    if clean and re.search(r'[a-zA-Z0-9]', clean):
                        line_parts.append(clean)
                if line_parts:
                    found_chunks.append(" ".join(line_parts))
            except Exception:
                continue

        if found_chunks and len(found_chunks) >= 2:
            return "\n".join(found_chunks).strip()
    except Exception:
        pass
    return ""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts all text from PDF bytes across all pages with page-level structural demarcation and OCR fallback."""
    if not PdfReader:
        return extract_raw_pdf_streams(file_bytes)

    pages_text = []

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        
        # 1. Handle password-encrypted PDFs (many software exports use empty password encryption)
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                pass

        # 2. Extract Interactive Form Fields (Crucial for insurance policies, tax/application forms)
        try:
            fields = reader.get_form_text_fields()
            if fields:
                form_lines = [f"{k}: {v}" for k, v in fields.items() if v and str(v).strip()]
                if form_lines:
                    pages_text.append("--- [Form Fields & Policy Details] ---\n" + "\n".join(form_lines))
        except Exception:
            pass

        ignored_watermark_phrases = {
            "draft", "confidential", "internal use only", "do not distribute",
            "watermark", "sample copy", "for review only"
        }

        for i, page in enumerate(reader.pages):
            page_num = i + 1
            
            # Isolated per-page extraction with layout fallback
            page_text = ""
            try:
                page_text = page.extract_text() or ""
            except Exception:
                try:
                    page_text = page.extract_text(extraction_mode="layout") or ""
                except Exception:
                    page_text = ""

            lines = [l.strip() for l in page_text.splitlines() if l.strip()]
            
            # Filter only genuine solitary watermark stamps
            clean_lines = []
            for line in lines:
                l_lower = line.lower()
                if l_lower in ignored_watermark_phrases:
                    continue
                clean_lines.append(line)

            # Extract any page annotation contents (sticky notes, callouts, textboxes)
            try:
                if "/Annots" in page:
                    for annot in page["/Annots"]:
                        annot_obj = annot.get_object()
                        contents = annot_obj.get("/Contents")
                        if contents and isinstance(contents, str) and contents.strip():
                            clean_lines.append(contents.strip())
            except Exception:
                pass

            extracted_page_content = "\n".join(clean_lines).strip()

            # If page text is empty or very sparse (< 40 chars), check for scanned images on the page
            if len(extracted_page_content) < 40 and hasattr(page, "images") and len(page.images) > 0:
                ocr_page_parts = []
                for img_obj in page.images:
                    try:
                        pil_img = Image.open(io.BytesIO(img_obj.data))
                        ocr_txt = extract_ocr_text_from_image(pil_img)
                        if ocr_txt and len(ocr_txt) > len(extracted_page_content):
                            ocr_page_parts.append(ocr_txt)
                    except Exception as e:
                        print(f"[DocParser] PDF image extraction error on page {page_num}: {e}")
                
                if ocr_page_parts:
                    extracted_page_content = "\n".join(ocr_page_parts).strip()

            if extracted_page_content:
                pages_text.append(f"--- [Page {page_num}] ---\n" + extracted_page_content)

        full_extracted = "\n\n".join(pages_text).strip()
        if full_extracted and len(full_extracted) > 30:
            return full_extracted
    except Exception as e:
        print(f"[DocParser] PDF read error: {e}")

    # Fallback: Recover strings from raw PDF streams directly
    fallback_text = extract_raw_pdf_streams(file_bytes)
    if fallback_text and len(fallback_text) > 30:
        return fallback_text

    return "\n\n".join(pages_text).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extracts text and table rows from Word DOCX bytes."""
    # Method 1: python-docx
    if docx:
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            content = []
            for p in doc.paragraphs:
                if p.text.strip():
                    content.append(p.text.strip())
            for table in doc.tables:
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        content.append(" | ".join(row_cells))
            return "\n\n".join(content).strip()
        except Exception as e:
            print(f"[DocParser] python-docx read error: {e}")

    # Method 2: Zero-dependency XML parsing fallback
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx_zip:
            xml_content = docx_zip.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            text_nodes = tree.findall(".//w:t", namespace)
            return " ".join(node.text for node in text_nodes if node.text)
    except Exception as e:
        print(f"[DocParser] DOCX XML fallback error: {e}")
        return ""


def extract_text_from_excel(file_bytes: bytes, filename: str) -> str:
    """Extracts structured tabular text from Excel (XLSX, XLS) or CSV files."""
    if filename.lower().endswith(".csv"):
        try:
            content = file_bytes.decode("utf-8", errors="ignore")
            reader = csv.reader(io.StringIO(content))
            rows = []
            for i, row in enumerate(reader):
                if row and any(c.strip() for c in row):
                    rows.append(" | ".join(row))
                if i > 250:
                    rows.append("... (additional rows truncated for length)")
                    break
            return "\n".join(rows)
        except Exception as e:
            print(f"[DocParser] CSV read error: {e}")

    if openpyxl:
        try:
            wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
            output = []
            for sheet in wb.sheetnames[:4]:
                ws = wb[sheet]
                output.append(f"--- Sheet: {sheet} ---")
                for r_idx, row in enumerate(ws.iter_rows(values_only=True)):
                    if any(cell is not None for cell in row):
                        row_vals = [str(c).strip() if c is not None else "" for c in row]
                        output.append(" | ".join(row_vals))
                    if r_idx > 200:
                        output.append("... (rows truncated)")
                        break
            return "\n".join(output).strip()
        except Exception as e:
            print(f"[DocParser] openpyxl read error: {e}")

    return ""


def extract_text_from_plain_text(file_bytes: bytes, filename: str = "") -> str:
    """Extracts clean text from TXT, MD, JSON, LOG, XML, YAML, HTML, etc."""
    try:
        raw_content = file_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        raw_content = file_bytes.decode("latin-1", errors="ignore").strip()

    ext = os.path.splitext(filename)[1].lower() if filename else ""
    if ext == ".json":
        try:
            parsed = json.loads(raw_content)
            return json.dumps(parsed, indent=2)
        except Exception:
            return raw_content
    elif ext in [".xml", ".html", ".htm"]:
        clean = re.sub(r"<[^>]+>", " ", raw_content)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean if clean else raw_content

    return raw_content


def extract_universal_document_text(file_bytes: bytes, filename: str) -> str:
    """
    Universal Document Ingestion Gateway for ANY document format:
    PDFs, Word Docs, Excel, CSV, Plain Text, and Image formats (Prescriptions, Policies, Receipts).
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == ".pdf":
        text = extract_text_from_pdf(file_bytes)
        if not text or not text.strip():
            try:
                decoded = file_bytes.decode("utf-8", errors="ignore").strip()
                if len(decoded) > 30 and not decoded.startswith("%PDF"):
                    return decoded
            except Exception:
                pass
        return text
    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
        # 1. First attempt OCR extraction
        try:
            pil_img = Image.open(io.BytesIO(file_bytes))
            ocr_text = extract_ocr_text_from_image(pil_img)
            if ocr_text and len(ocr_text.strip()) > 20:
                return ocr_text
        except Exception as e:
            print(f"[DocParser] Image OCR extraction note: {e}")

        # 2. Check if it is a prescription image
        try:
            from .prescription_analyzer import local_prescription_analyzer
            analysis = local_prescription_analyzer.analyze_prescription(file_bytes, filename)
            reply = analysis.get("reply", "")
            if reply and "No valid pharmaceutical medications detected" not in reply:
                return reply
        except Exception as e:
            print(f"[DocParser] Image prescription analysis note: {e}")

        return f"Document Image: {filename}"
    elif ext in [".docx", ".doc"]:
        return extract_text_from_docx(file_bytes)
    elif ext in [".xlsx", ".xls", ".csv"]:
        return extract_text_from_excel(file_bytes, filename)
    elif ext in [".txt", ".md", ".json", ".log", ".sql", ".xml", ".yaml", ".yml", ".html", ".htm", ".rst"]:
        return extract_text_from_plain_text(file_bytes, filename)
    else:
        # Fallback generic text decoding
        return extract_text_from_plain_text(file_bytes, filename)

