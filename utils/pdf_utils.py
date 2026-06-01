"""
PDF utilities: text extraction, page rendering, text search, and highlighting.
"""

import io
import base64
import fitz  # pymupdf


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text content from a PDF file with page markers.
    Detects tables and formats them as markdown tables for LLM comprehension.

    Args:
        pdf_file: File object or bytes from uploaded PDF

    Returns:
        Extracted text as string with page markers
    """
    pdf_bytes = _get_pdf_bytes(pdf_file)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return _fallback_extract(pdf_bytes)

    # Same approach as AIDE-Web's PDF.js: flat text, no table detection.
    # Let the LLM do all the work of identifying tables from raw text.
    text_parts = []
    for page_num, page in enumerate(doc, 1):
        try:
            page_text = page.get_text("text")
            if page_text.strip():
                text_parts.append(f"--- Page {page_num} ---\n{page_text}")
        except Exception as e:
            text_parts.append(f"--- Page {page_num} ---\n[Error: {str(e)}]")

    doc.close()
    return "\n\n".join(text_parts)


def extract_text_by_page(pdf_file) -> list:
    """
    Extract text from each page separately.

    Returns:
        List of dicts: [{'page': 1, 'text': '...'}, ...]
    """
    pdf_bytes = _get_pdf_bytes(pdf_file)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return []

    pages = []
    for page_num, page in enumerate(doc, 1):
        pages.append({
            'page': page_num,
            'text': page.get_text()
        })
    doc.close()
    return pages


def get_pdf_page_count(pdf_file) -> int:
    """Get total number of pages in a PDF."""
    pdf_bytes = _get_pdf_bytes(pdf_file)
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0


def render_page_as_base64(pdf_file, page_num: int, highlights: list = None,
                          zoom: float = 1.5) -> str:
    """
    Render a PDF page as a base64-encoded PNG image.

    Args:
        pdf_file: PDF file object or bytes
        page_num: 1-indexed page number to render
        highlights: List of highlight rects, each as (x0, y0, x1, y1) in page coords
        zoom: Render resolution multiplier

    Returns:
        Base64-encoded PNG image string
    """
    pdf_bytes = _get_pdf_bytes(pdf_file)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return ""

    if page_num < 1 or page_num > len(doc):
        doc.close()
        return ""

    page = doc[page_num - 1]

    # Render page to pixmap at desired zoom
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    # Draw highlights if provided
    if highlights:
        # Use pymupdf highlight annotations directly
        annots = []
        for rect in highlights:
            r = fitz.Rect(rect[0], rect[1], rect[2], rect[3])
            annot = page.add_highlight_annot(r)
            annots.append(annot)

        # Re-render with highlights visible
        pix = page.get_pixmap(matrix=mat)
        img_bytes = _pil_bytes(pix)

        # Remove temporary annotations
        for annot in annots:
            try:
                page.delete_annot(annot)
            except Exception:
                pass
    else:
        img_bytes = _pil_bytes(pix)

    doc.close()
    return base64.b64encode(img_bytes).decode("utf-8")


def _pil_bytes(pix) -> bytes:
    """Convert pymupdf pixmap to PNG bytes via PIL (strips ICC profile, avoids libpng warnings)."""
    from PIL import Image as PILImage
    import io
    img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="PNG", icc_profile=None)
    return buf.getvalue()


def search_text_for_highlight(pdf_file, search_text: str,
                              preferred_page: int = None) -> dict:
    """
    Fast search for text in PDF. Searches preferred page first, then nearby pages,
    then falls back to full search. Expands highlight rects to cover full lines.

    Returns:
        {'found': bool, 'page': int, 'highlights': [(x0,y0,x1,y1),...], 'matched_text': str}
    """
    pdf_bytes = _get_pdf_bytes(pdf_file)

    base = {'found': False, 'page': 1, 'highlights': [], 'matched_text': ''}
    if not search_text or not search_text.strip():
        return base

    search_text = search_text.strip()

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return base

    total = len(doc)
    phrases = _gen_phrases(search_text)

    # ---- Stage 1: Search preferred page ONLY (fast) ----
    if preferred_page and 1 <= preferred_page <= total:
        result = _search_page(doc[preferred_page - 1], phrases)
        if result:
            doc.close()
            result['page'] = preferred_page
            return result

    # ---- Stage 2: Search ±2 pages around preferred ----
    if preferred_page and 1 <= preferred_page <= total:
        for offset in [1, -1, 2, -2]:
            pn = preferred_page - 1 + offset
            if 0 <= pn < total:
                result = _search_page(doc[pn], phrases)
                if result:
                    doc.close()
                    result['page'] = pn + 1
                    return result

    # ---- Stage 3: Search all remaining pages ----
    for pn in range(total):
        if preferred_page and abs(pn - (preferred_page - 1)) <= 2:
            continue  # Already searched
        result = _search_page(doc[pn], phrases)
        if result:
            doc.close()
            result['page'] = pn + 1
            return result

    # ---- Stage 4: Keyword fallback -- limit matches to avoid whole-page highlight ----
    if preferred_page and 1 <= preferred_page <= total:
        import re as _re
        keywords = _re.findall(r'[A-Za-z0-9]+–[A-Za-z0-9]+|[A-Za-z]{4,}|\d{2,}', search_text)
        keywords = [k.strip().rstrip('.,;:!?') for k in keywords if len(k) >= 3]
        keywords = list(dict.fromkeys(keywords))
        keywords.sort(key=len, reverse=True)

        p = doc[preferred_page - 1]
        all_rects = []
        matched_kw = []
        for kw in keywords:
            if len(all_rects) >= 20:
                break
            areas = p.search_for(kw)
            if areas:
                if kw not in matched_kw:
                    matched_kw.append(kw)
                for a in areas:
                    if len(all_rects) < 20:
                        all_rects.append((a.x0, a.y0, a.x1, a.y1))

        if all_rects:
            # Don't highlight — just scroll to page (keyword matches are scattered)
            doc.close()
            return {
                'found': True,
                'page': preferred_page,
                'highlights': [],
                'matched_text': ''
            }


        # Collect matching rects from preferred page
        all_rects = []
        matched_kw = []
        for kw in keywords:
            areas = doc[preferred_page - 1].search_for(kw)
            if areas:
                if kw not in matched_kw:
                    matched_kw.append(kw)
                for a in areas:
                    all_rects.append((a.x0, a.y0, a.x1, a.y1))

        if all_rects:
            grouped = _expand_highlights(all_rects, doc[preferred_page - 1])
            doc.close()
            return {
                'found': True,
                'page': preferred_page,
                'highlights': grouped,
                'matched_text': ' | '.join(matched_kw[:5])
            }

    doc.close()
    if preferred_page and 1 <= preferred_page <= total:
        base['page'] = preferred_page
    return base


def _search_page(page, phrases: list) -> dict:
    """Search a single page with multiple phrases. Returns result dict or None."""
    # Try full phrases first
    for phrase in phrases:
        if len(phrase) < 3:
            continue
        areas = page.search_for(phrase)
        if areas:
            rects = _expand_highlights(areas, page)
            return {'found': True, 'highlights': rects, 'matched_text': phrase}
    return None


def _expand_highlights(areas: list, page) -> list:
    """Expand highlight rects to cover full text lines and merge adjacent ones."""
    if not areas:
        return []

    page_h = page.rect.height
    page_w = page.rect.width

    expanded = []
    for r in areas:
        # Handle both fitz.Rect and tuple
        if hasattr(r, 'y1') and hasattr(r, 'y0'):
            x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
        else:
            x0, y0, x1, y1 = r[0], r[1], r[2], r[3]
        line_h = max(y1 - y0, 8)
        pad_y = line_h * 3.0
        pad_x = 8
        expanded.append((
            max(0, x0 - pad_x),
            max(0, y0 - pad_y),
            min(page_w, x1 + pad_x),
            min(page_h, y1 + pad_y)
        ))

    # Merge nearby rects
    if len(expanded) <= 1:
        return expanded

    expanded.sort(key=lambda r: (r[1], r[0]))
    merged = [expanded[0]]
    for r in expanded[1:]:
        prev = merged[-1]
        # Merge if vertically close and horizontally overlapping
        if r[1] <= prev[3] + line_h * 2:
            merged[-1] = (
                min(prev[0], r[0]),
                min(prev[1], r[1]),
                max(prev[2], r[2]),
                max(prev[3], r[3])
            )
        else:
            merged.append(r)

    return merged


def _gen_phrases(text: str) -> list:
    """Generate prioritized search phrases (longest first, then substrings)."""
    phrases = []
    text = text.strip()
    if not text:
        return phrases

    # 1. Full text (best match)
    if len(text) >= 5:
        phrases.append(text)

    # 2. First sentence / segment (~30-60 chars)
    if len(text) > 30:
        # Split on common delimiters
        for sep in ['. ', '; ', ': ', '? ', '! ']:
            parts = text.split(sep, 1)
            if len(parts[0]) >= 15:
                phrases.append(parts[0].strip())
                break

    # 3. First N words
    words = text.split()
    if len(words) >= 5:
        phrases.append(' '.join(words[:5]))

    # 4. Key phrase (middle chunk)
    if len(text) > 40:
        start = len(text) // 4
        chunk = text[start:start + 50].strip()
        if len(chunk) >= 10:
            phrases.append(chunk)

    # 5. Short signature (first 25 chars)
    if len(text) >= 15:
        phrases.append(text[:25].strip())

    return [p for p in phrases if len(p) >= 3]


def _get_pdf_bytes(pdf_file) -> bytes:
    """Get raw bytes from a PDF file object or bytes."""
    if hasattr(pdf_file, 'read'):
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)
    else:
        pdf_bytes = pdf_file
    return pdf_bytes


def _fallback_extract(pdf_bytes: bytes) -> str:
    """Fallback text extraction using PyPDF2."""
    try:
        import PyPDF2
        pdf_stream = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_stream)

        text_parts = []
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {page_num} ---\n{page_text}")
            except Exception as e:
                text_parts.append(f"--- Page {page_num} ---\n[Error: {str(e)}]")

        return "\n\n".join(text_parts)
    except Exception as e:
        return f"[Failed to extract text: {str(e)}]"
