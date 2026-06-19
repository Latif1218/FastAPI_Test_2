import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

from app.config import settings

if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD


def extract_text_from_image(image_path: str, lang: str = None) -> str:
    """Run Tesseract OCR on a single image file."""
    lang = lang or settings.OCR_LANGUAGES
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang=lang)
    return text


def extract_text_from_pdf(pdf_path: str, lang: str = None) -> str:
    """
    Convert each PDF page to an image (via poppler) then run OCR on every page.
    This works for scanned/image-based PDFs as well as normal ones.
    """
    lang = lang or settings.OCR_LANGUAGES
    pages = convert_from_path(pdf_path, dpi=300)

    full_text = []
    for i, page_image in enumerate(pages):
        print(f"[OCR] Processing page {i + 1}/{len(pages)} ...")
        page_text = pytesseract.image_to_string(page_image, lang=lang)
        full_text.append(page_text)

    return "\n\n".join(full_text)


def extract_text(file_path: str, lang: str = None) -> str:
    """
    Entry point: detects file type and routes to the correct extractor.
    Supports: .pdf, .png, .jpg, .jpeg
    """
    ext = os.path.splitext(file_path)[1].lower()

    print(f"[OCR] Starting local extraction for: {file_path}")

    if ext == ".pdf":
        text = extract_text_from_pdf(file_path, lang)
    elif ext in [".png", ".jpg", ".jpeg"]:
        text = extract_text_from_image(file_path, lang)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    print(f"[OCR] Done. Extracted {len(text)} characters.")
    return text


def detect_language_simple(text: str) -> str:
    """
    Very lightweight heuristic language detector based on Unicode ranges.
    Bangla block: U+0980 to U+09FF
    Returns: 'bn', 'en', or 'bn+en' (mixed)
    """
    bangla_chars = sum(1 for ch in text if "\u0980" <= ch <= "\u09FF")
    english_chars = sum(1 for ch in text if ch.isascii() and ch.isalpha())

    if bangla_chars > 0 and english_chars > 0:
        # if both present in meaningful amounts, treat as mixed
        ratio = bangla_chars / max(english_chars, 1)
        if 0.2 < ratio < 5:
            return "bn+en"
    if bangla_chars > english_chars:
        return "bn"
    return "en"