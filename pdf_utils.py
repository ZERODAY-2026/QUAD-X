"""PDF helpers: text extraction (with OCR fallback) and quote highlighting."""

import logging
import re
import shutil

import pymupdf

logger = logging.getLogger(__name__)

OCR_LANGUAGE = "eng"  # e.g. "eng+hin"


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def extract_pages(pdf_bytes: bytes):
    """Return (pages, skipped_page_numbers, ocr_page_numbers)."""
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:
        raise ValueError("This file could not be opened as a PDF.") from error

    with document:
        if document.needs_pass:
            raise ValueError("This PDF is password-protected. Remove the password and try again.")

        pages, skipped, ocr_pages = [], [], []
        for number, page in enumerate(document, start=1):
            text = page.get_text(sort=True)
            if not text.strip():  # no text layer: try OCR
                try:
                    tp = page.get_textpage_ocr(language=OCR_LANGUAGE, dpi=300, full=True)
                    text = page.get_text(sort=True, textpage=tp)
                except Exception as error:
                    logger.warning("OCR failed on page %d: %s", number, error)
                    text = ""
                if text.strip():
                    ocr_pages.append(number)
            if text.strip():
                pages.append({"page": number, "text": text})
            else:
                skipped.append(number)
    return pages, skipped, ocr_pages


def build_document_text(pages: list[dict]) -> str:
    """Join pages with the [Page N] markers legal_analyzer relies on."""
    return "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in pages)


def _token(word: str) -> str:
    return re.sub(r"\W+", "", word).casefold()


def find_quote_rects(page, quote: str) -> list:
    target = [t for t in map(_token, quote.split()) if t]
    if not target:
        return []
    words = [
        (_token(w[4]), pymupdf.Rect(w[:4]), (w[5], w[6]))
        for w in page.get_text("words", sort=True)
    ]
    words = [w for w in words if w[0]]
    tokens = [w[0] for w in words]
    n = len(target)
    for i in range(len(tokens) - n + 1):
        if tokens[i : i + n] == target:
            lines = {}
            for _, rect, key in words[i : i + n]:
                if key in lines:
                    lines[key].include_rect(rect)
                else:
                    lines[key] = pymupdf.Rect(rect)
            return list(lines.values())
    return page.search_for(quote)


def render_source(pdf_bytes: bytes, page_number, quote: str):
    """Return (page_number, PNG bytes) with the quote highlighted, or None."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        candidates = [page_number - 1] if page_number else range(len(doc))
        for index in candidates:
            if not 0 <= index < len(doc):
                continue
            page = doc[index]
            rects = find_quote_rects(page, quote)
            if not rects:
                continue
            page.add_highlight_annot(rects)
            area = pymupdf.Rect(rects[0])
            for r in rects[1:]:
                area.include_rect(r)
            clip = pymupdf.Rect(
                page.rect.x0, max(page.rect.y0, area.y0 - 70),
                page.rect.x1, min(page.rect.y1, area.y1 + 70),
            )
            return index + 1, page.get_pixmap(dpi=130, clip=clip).tobytes("png")
    return None
