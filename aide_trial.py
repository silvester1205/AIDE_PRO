"""
AIDE Desktop — PyQt6 desktop application for AI-Assisted Data Extraction.
Two-tab layout: Setup (API + coding form) → Analyze (PDF viewer + field cards).
"""
import sys
import os
import json
import time
import pandas as pd
import io
import base64
import fitz
from datetime import date

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QLabel, QPushButton, QTextEdit, QFrame, QFileDialog,
    QLineEdit, QMessageBox, QProgressBar, QComboBox, QTabWidget,
    QStatusBar, QGroupBox, QFormLayout, QMenuBar, QMenu, QDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint, QSettings, QRect
from PyQt6.QtGui import QPixmap, QImage, QPalette, QColor, QAction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.pdf_utils import (
    render_page_as_base64, search_text_for_highlight,
    get_pdf_page_count, extract_text_from_pdf
)
from utils.llm_client import analyze_with_llm, test_connection, fetch_available_models


# ============================================================
# Workers
# ============================================================

class AnalyzeWorker(QThread):
    finished = pyqtSignal(bool, object)
    def __init__(self, config, prompts, pdf_bytes):
        super().__init__()
        self.config = config
        self.prompts = prompts
        self.pdf_bytes = pdf_bytes
    def run(self):
        pdf_text = extract_text_from_pdf(self.pdf_bytes)
        success, result = analyze_with_llm(self.config, self.prompts, pdf_text)
        self.finished.emit(success, result)


class FetchModelsWorker(QThread):
    finished = pyqtSignal(bool, list, str)
    def __init__(self, config):
        super().__init__()
        self.config = config
    def run(self):
        success, models, msg = fetch_available_models(self.config)
        self.finished.emit(success, models, msg)

class TestConnWorker(QThread):
    finished = pyqtSignal(bool, str)
    def __init__(self, config):
        super().__init__()
        self.config = config
    def run(self):
        success, msg = test_connection(self.config)
        self.finished.emit(success, msg)


# ============================================================
# Selectable PDF label
# ============================================================

class SelectableLabel(QLabel):
    """Label that supports rectangle-drag text selection on PDF pages."""
    selection_done = pyqtSignal(int, QRect)  # page_num, selection rect in image coords

    def __init__(self, page_num: int, parent=None):
        super().__init__(parent)
        self.page_num = page_num
        self._selecting = False
        self._start = QPoint()
        self._end = QPoint()

    def mousePressEvent(self, event):
        if self._enabled and event.button() == Qt.MouseButton.LeftButton:
            self._selecting = True
            self._start = event.pos()
            self._end = event.pos()
            self.update()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._enabled and self._selecting:
            self._end = event.pos()
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._enabled and self._selecting and event.button() == Qt.MouseButton.LeftButton:
            self._selecting = False
            self._end = event.pos()
            self.update()
            rect = QRect(self._start, self._end).normalized()
            if rect.width() > 10 and rect.height() > 10:
                self.selection_done.emit(self.page_num, rect)
            return
        super().mouseReleaseEvent(event)

    def set_select_enabled(self, enabled: bool):
        self._enabled = enabled
        self._selecting = False
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._selecting:
            from PyQt6.QtGui import QPainter, QPen, QColor
            p = QPainter(self)
            p.setPen(QPen(QColor(52, 152, 219), 2, Qt.PenStyle.DashLine))
            p.setBrush(QColor(52, 152, 219, 50))
            p.drawRect(QRect(self._start, self._end).normalized())


# ============================================================
# Field Card
# ============================================================

class FieldCard(QFrame):
    source_clicked = pyqtSignal(int, str, object)  # idx, quote, page
    record_clicked = pyqtSignal(int, str)           # idx, text

    def __init__(self, index: int, prompt: str, parent=None):
        super().__init__(parent)
        self.index = index
        self.prompt = prompt
        self.source_quote = ""
        self.source_page = None
        self.is_recorded = False
        self._is_active = False
        self._setup()

    def _setup(self):
        self.setFrameStyle(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        self.header = QLabel(f"<b>F{self.index + 1}</b>")
        layout.addWidget(self.header)

        prompt_txt = self.prompt[:160] + "…" if len(self.prompt) > 160 else self.prompt
        pl = QLabel(f"{prompt_txt}")
        pl.setWordWrap(True)
        pl.setStyleSheet("color:#666;font-size:13px;")
        layout.addWidget(pl)

        self.editor = QTextEdit()
        self.editor.setPlaceholderText("LLM extracted data...")
        self.editor.setMinimumHeight(120)
        self.editor.setMaximumHeight(200)
        self.editor.setStyleSheet(
            "QTextEdit{border:1px solid #ddd;border-radius:4px;"
            "padding:8px;font-size:14px;background:#fff;}")
        layout.addWidget(self.editor)

        self.src_info = QLabel("")
        self.src_info.setWordWrap(True)
        self.src_info.setStyleSheet("color:#555;font-size:12px;padding:4px 0;")
        self.src_info.hide()
        layout.addWidget(self.src_info)

        bl = QHBoxLayout()
        bl.setSpacing(6)
        self.src_btn = QPushButton("📍 Source")
        self.src_btn.clicked.connect(lambda: self.source_clicked.emit(
            self.index, self.source_quote, self.source_page))
        bl.addWidget(self.src_btn)

        self.rec_btn = QPushButton("✅ Record")
        self.rec_btn.clicked.connect(lambda: self.record_clicked.emit(
            self.index, self.editor.toPlainText()))
        bl.addWidget(self.rec_btn)
        bl.addStretch()
        layout.addLayout(bl)

        self._restyle()

    def _restyle(self):
        if self.is_recorded:
            self.setStyleSheet(
                "FieldCard{border-radius:8px;margin:4px;padding:2px;"
                "border:3px solid #2ecc71;background:#eafaf1;}")
            self.src_btn.setStyleSheet(
                "QPushButton{border-radius:6px;padding:7px 18px;font-size:12px;font-weight:600;"
                "background:#dee2e6;color:#666;border:1px solid #ccc;}"
                "QPushButton:hover{background:#ced4da;}")
            self.rec_btn.setStyleSheet(
                "QPushButton{border-radius:6px;padding:7px 18px;font-size:12px;font-weight:600;"
                "background:#2ecc71;color:#fff;border:none;}"
                "QPushButton:hover{background:#27ae60;}")
        elif self._is_active:
            self.setStyleSheet(
                "FieldCard{border-radius:8px;margin:4px;padding:2px;"
                "border:3px solid #3498db;background:#eaf2f8;}")
            self.src_btn.setStyleSheet(
                "QPushButton{border-radius:6px;padding:7px 18px;font-size:12px;font-weight:600;"
                "background:#3498db;color:#fff;border:none;}"
                "QPushButton:hover{background:#2980b9;}")
            self.rec_btn.setStyleSheet(
                "QPushButton{border-radius:6px;padding:7px 18px;font-size:12px;font-weight:600;"
                "background:#2980b9;color:#fff;border:none;}"
                "QPushButton:hover{background:#1f6ea0;}")
        else:
            self.setStyleSheet(
                "FieldCard{border-radius:8px;margin:4px;padding:2px;"
                "border:3px solid #e0e0e0;background:#fff;}")
            self.src_btn.setStyleSheet(
                "QPushButton{border-radius:6px;padding:7px 18px;font-size:12px;font-weight:600;"
                "background:#f8f9fa;color:#555;border:1px solid #dee2e6;}"
                "QPushButton:hover{background:#e9ecef;}")
            self.rec_btn.setStyleSheet(
                "QPushButton{border-radius:6px;padding:7px 18px;font-size:12px;font-weight:600;"
                "background:#3498db;color:#fff;border:none;}"
                "QPushButton:hover{background:#2980b9;}")

    def set_data(self, response, source_quote, source_page):
        self.editor.setPlainText(response or "")
        self.source_quote = source_quote or ""
        self.source_page = source_page
        if source_quote:
            q = source_quote[:150] + "…" if len(source_quote) > 150 else source_quote
            t = f"📝 <i>{q}</i>"
            if source_page:
                t += f"  |  📄 Page {source_page}"
            self.src_info.setText(t)
            self.src_info.show()

    def toggle_recorded(self):
        self.is_recorded = not self.is_recorded
        self.editor.setReadOnly(self.is_recorded)
        if self.is_recorded:
            self.rec_btn.setText("✔ Done — click to edit")
            self.header.setText(f"<b>✅ F{self.index + 1}</b>")
        else:
            self.rec_btn.setText("✅ Record")
            self.header.setText(f"<b>F{self.index + 1}</b>")
        self._restyle()

    def set_active(self, active: bool):
        self._is_active = active
        self.src_btn.setText("✓ Source" if active else "📍 Source")
        self._restyle()

    def text(self) -> str:
        return self.editor.toPlainText()


# ============================================================
# PDF Viewer
# ============================================================

class PdfViewer(QWidget):
    ZOOM = 1.5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pdf_bytes = None
        self.total_pages = 0
        self.page_widgets = []
        self.page_pixmaps = {}
        self.page_words = {}  # page_num -> [(x0,y0,x1,y1,text), ...]
        self.highlight_page = None
        self._select_mode = False
        self._search_matches = []  # [(page_num, [(x0,y0,x1,y1),...]), ...]
        self._search_idx = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Unified toolbar: select-text + search
        tbar = QHBoxLayout()
        tbar.setContentsMargins(4, 2, 4, 2)

        self.select_btn = QPushButton("📋 Select")
        self.select_btn.setCheckable(True)
        self.select_btn.setStyleSheet(
            "QPushButton{background:#555;color:#ccc;border:1px solid #666;border-radius:4px;"
            "padding:4px 8px;font-size:11px;}"
            "QPushButton:checked{background:#3498db;color:#fff;border-color:#2980b9;}")
        self.select_btn.toggled.connect(self._toggle_select)
        tbar.addWidget(self.select_btn)

        tbar.addStretch()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search")
        self.search_input.returnPressed.connect(self._do_search)
        self.search_input.setStyleSheet(
            "QLineEdit{background:#444;color:#fff;border:1px solid #666;border-radius:4px;"
            "padding:3px 6px;font-size:11px;min-width:120px;max-width:180px;}")
        tbar.addWidget(self.search_input)

        self.search_btn = QPushButton("🔍")
        self.search_btn.setFixedSize(24, 24)
        self.search_btn.setStyleSheet(
            "QPushButton{background:#555;color:#ccc;border:1px solid #666;border-radius:4px;}"
            "QPushButton:hover{background:#666;}")
        self.search_btn.clicked.connect(self._do_search)
        tbar.addWidget(self.search_btn)

        self.search_prev = QPushButton("◀")
        self.search_prev.setFixedSize(24, 24)
        self.search_prev.setEnabled(False)
        self.search_prev.setStyleSheet(
            "QPushButton{background:#555;color:#ccc;border:1px solid #666;border-radius:4px;}"
            "QPushButton:disabled{background:#333;color:#555;}")
        self.search_prev.clicked.connect(self._search_prev)
        tbar.addWidget(self.search_prev)

        self.search_count = QLabel("")
        self.search_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.search_count.setStyleSheet("color:#aaa;font-size:11px;min-width:30px;")
        tbar.addWidget(self.search_count)

        self.search_next = QPushButton("▶")
        self.search_next.setFixedSize(24, 24)
        self.search_next.setEnabled(False)
        self.search_next.setStyleSheet(
            "QPushButton{background:#555;color:#ccc;border:1px solid #666;border-radius:4px;}"
            "QPushButton:disabled{background:#333;color:#555;}")
        self.search_next.clicked.connect(self._search_next)
        tbar.addWidget(self.search_next)

        self.search_clear_btn = QPushButton("✕")
        self.search_clear_btn.setFixedSize(24, 24)
        self.search_clear_btn.setEnabled(False)
        self.search_clear_btn.setStyleSheet(
            "QPushButton{background:#c0392b;color:#fff;border:none;border-radius:4px;}"
            "QPushButton:disabled{background:#333;color:#555;}")
        self.search_clear_btn.clicked.connect(self._clear_search)
        tbar.addWidget(self.search_clear_btn)

        layout.addLayout(tbar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea{background:#525659;border:none;}")

        self.container = QWidget()
        self.container.setStyleSheet("background:#525659;")
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(8, 8, 8, 8)
        self.container_layout.setSpacing(12)
        self.container_layout.addStretch()
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)

        self.placeholder = QLabel("Click 📂 Open PDF above to begin")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color:#999;padding:60px;")
        self.container_layout.insertWidget(0, self.placeholder)

    def _toggle_select(self, checked):
        self._select_mode = checked
        for lbl in self.page_widgets:
            if isinstance(lbl, SelectableLabel):
                lbl.set_select_enabled(checked)
                lbl.setCursor(Qt.CursorShape.CrossCursor if checked else Qt.CursorShape.ArrowCursor)

    def _render_pages(self):
        """Render all pages at fixed zoom."""
        if not self.pdf_bytes:
            return
        self.page_pixmaps.clear()
        self._clear()
        self.placeholder = QLabel("")
        self.placeholder.hide()
        self.container_layout.insertWidget(0, self.placeholder)

        rects = getattr(self, '_last_rects', None)
        hl_page = self.highlight_page if rects else None

        for pn in range(1, self.total_pages + 1):
            r = rects if pn == hl_page else None
            b64 = render_page_as_base64(self.pdf_bytes, pn, highlights=r, zoom=self.ZOOM)
            if b64:
                pm = self._b64_to_pixmap(b64)
                self.page_pixmaps[pn] = pm
                lbl = SelectableLabel(pn)
                lbl.selection_done.connect(self._on_selection)
                lbl.setPixmap(pm)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setStyleSheet("background:#fff;border-radius:4px;")
                lbl.setObjectName(f"pdfpg{pn}")
                self.page_widgets.append(lbl)
                self.container_layout.insertWidget(self.container_layout.count() - 1, lbl)

                tag = QLabel(f"Page {pn}")
                tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
                tag.setStyleSheet("color:#999;font-size:11px;background:#f0f0f0;padding:2px 8px;border-radius:2px;")
                idx = self.container_layout.indexOf(lbl)
                self.container_layout.insertWidget(idx, tag)

            if pn % 5 == 0:
                QApplication.processEvents()

    def load_pdf(self, pdf_bytes: bytes):
        self.pdf_bytes = pdf_bytes
        self.total_pages = get_pdf_page_count(pdf_bytes)
        self.page_pixmaps.clear()
        self.page_words.clear()
        self.highlight_page = None
        self._last_rects = None
        self._clear()
        self.placeholder = QLabel("")
        self.placeholder.hide()
        self.container_layout.insertWidget(0, self.placeholder)

        # Extract word positions for text selection
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for pn in range(1, self.total_pages + 1):
                words = doc[pn - 1].get_text("words")
                self.page_words[pn] = [(w[0], w[1], w[2], w[3], w[4]) for w in words if len(w) > 4]
            doc.close()
        except Exception:
            pass

        self._render_pages()

    def _on_selection(self, page_num: int, rect: QRect):
        """Extract text from selected rectangle region."""
        words = self.page_words.get(page_num, [])
        if not words:
            return

        # Map image coordinates → PDF coordinates
        scale = 1.0 / self.ZOOM
        x0, y0 = rect.left() * scale, rect.top() * scale
        x1, y1 = rect.right() * scale, rect.bottom() * scale

        # Find words within the selection rectangle
        selected = []
        for wx0, wy0, wx1, wy1, text in words:
            # Check if word overlaps with selection
            if not (wx1 < x0 or wx0 > x1 or wy1 < y0 or wy0 > y1):
                selected.append((wy0, wx0, text))

        if not selected:
            return

        # Sort by y then x (reading order), group into lines
        selected.sort()
        lines = []
        current_line = []
        current_y = selected[0][0]
        for wy0, wx0, text in selected:
            if abs(wy0 - current_y) > 5:
                lines.append(" ".join(current_line))
                current_line = [text]
                current_y = wy0
            else:
                current_line.append(text)
        if current_line:
            lines.append(" ".join(current_line))

        result = "\n".join(lines)

        # Show in popup
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Selected text — Page {page_num}")
        dlg.resize(600, 300)
        layout = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setPlainText(result)
        layout.addWidget(te)
        copy_btn = QPushButton("📋 Copy to Clipboard")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(result))
        layout.addWidget(copy_btn)
        dlg.exec()

    def _clear(self):
        while self.container_layout.count() > 1:
            item = self.container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.page_widgets.clear()

    def _do_search(self):
        """Search PDF for text, find all matches, jump to first."""
        text = self.search_input.text().strip()
        if not text or not self.pdf_bytes:
            return

        self._search_matches = []
        self._search_idx = -1
        self.search_count.setText("")
        self.search_prev.setEnabled(False)
        self.search_next.setEnabled(False)

        try:
            doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
            for pn in range(1, self.total_pages + 1):
                areas = doc[pn - 1].search_for(text)
                if areas:
                    rects = []
                    for a in areas:
                        rects.append((a.x0, a.y0, a.x1, a.y1))
                    self._search_matches.append((pn, rects))
            doc.close()
        except Exception:
            pass

        if self._search_matches:
            self.search_next.setEnabled(True)
            self.search_prev.setEnabled(True)
            self.search_clear_btn.setEnabled(True)
            self._search_goto(0)

    def _search_goto(self, idx: int):
        """Go to the idx-th match, highlight and scroll."""
        if idx < 0 or idx >= len(self._search_matches):
            return

        pn, rects = self._search_matches[idx]
        self._search_idx = idx
        self.search_count.setText(f"{idx+1}/{len(self._search_matches)}")

        # Highlight current match in yellow
        b64 = render_page_as_base64(self.pdf_bytes, pn, highlights=rects, zoom=self.ZOOM)
        if b64 and pn - 1 < len(self.page_widgets):
            pm = self._b64_to_pixmap(b64)
            self.page_widgets[pn - 1].setPixmap(pm)

        # Restore clean for other pages with search matches
        for mpn, _ in self._search_matches:
            if mpn != pn and mpn in self.page_pixmaps and mpn - 1 < len(self.page_widgets):
                self.page_widgets[mpn - 1].setPixmap(self.page_pixmaps[mpn])

        # Restore clean for other pages without matches too (but keep source highlights)
        if not hasattr(self, '_last_rects') or not self._last_rects:
            for ppn in range(1, self.total_pages + 1):
                if ppn != pn and ppn in self.page_pixmaps and ppn - 1 < len(self.page_widgets):
                    self.page_widgets[ppn - 1].setPixmap(self.page_pixmaps[ppn])

        # Scroll to the match
        if pn - 1 < len(self.page_widgets):
            self.scroll.ensureWidgetVisible(self.page_widgets[pn - 1], 0, 50)

    def _search_next(self):
        if self._search_matches:
            nxt = (self._search_idx + 1) % len(self._search_matches)
            self._search_goto(nxt)

    def _search_prev(self):
        if self._search_matches:
            prv = (self._search_idx - 1) % len(self._search_matches)
            self._search_goto(prv)

    def _clear_search(self):
        """Clear search highlights and restore pages."""
        self._search_matches = []
        self._search_idx = -1
        self.search_count.setText("")
        self.search_input.clear()
        self.search_prev.setEnabled(False)
        self.search_next.setEnabled(False)
        self.search_clear_btn.setEnabled(False)
        self._clear_highlights()

    def highlight_and_scroll(self, page_num: int, rects: list):
        if page_num < 1 or page_num > self.total_pages or not self.pdf_bytes:
            return
        self.highlight_page = page_num
        self._last_rects = rects

        # Re-render highlighted page
        b64 = render_page_as_base64(self.pdf_bytes, page_num, highlights=rects, zoom=self.ZOOM)
        if b64 and page_num - 1 < len(self.page_widgets):
            self.page_widgets[page_num - 1].setPixmap(self._b64_to_pixmap(b64))

        # Restore other pages
        for pn in range(1, self.total_pages + 1):
            if pn != page_num and pn in self.page_pixmaps and pn - 1 < len(self.page_widgets):
                self.page_widgets[pn - 1].setPixmap(self.page_pixmaps[pn])

        # Scroll to center the highlighted area
        if page_num - 1 < len(self.page_widgets):
            widget = self.page_widgets[page_num - 1]
            # First ensure the page widget is visible
            self.scroll.ensureWidgetVisible(widget, 0, 50)

            # Then fine-tune: scroll to center the highlight rects
            if rects and hasattr(widget, 'pixmap') and widget.pixmap():
                pm = widget.pixmap()
                pm_h = pm.height()
                widget_h = self.scroll.viewport().height()
                # Get the middle y of highlight rects, scaled
                mid_y = sum(r[1] + r[3] for r in rects) / (2 * len(rects))  # average center y in PDF coords
                # Scale to pixmap coords
                mid_px = mid_y * self.ZOOM
                # Map to container coords
                widget_y = widget.mapTo(self.container, QPoint(0, 0)).y()
                target_y = widget_y + mid_px - widget_h // 3
                sb = self.scroll.verticalScrollBar()
                sb.setValue(max(0, int(target_y)))

    def _clear_highlights(self):
        """Clear all highlights, restore clean page images."""
        self.highlight_page = None
        self._last_rects = None
        for pn in range(1, self.total_pages + 1):
            if pn in self.page_pixmaps and pn - 1 < len(self.page_widgets):
                self.page_widgets[pn - 1].setPixmap(self.page_pixmaps[pn])

    def _b64_to_pixmap(self, b64: str) -> QPixmap:
        data = base64.b64decode(b64)
        img = QImage()
        img.loadFromData(data, "PNG")
        return QPixmap.fromImage(img)


# ============================================================
# Setup Tab
# ============================================================

class SetupTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("AIDE", "Desktop")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # -- API Config --
        api_grp = QGroupBox("API Configuration")
        api_form = QFormLayout(api_grp)
        api_form.setSpacing(6)

        saved_endpoint = self.settings.value("api/endpoint", "https://api.openai.com/v1")
        if not saved_endpoint:
            saved_endpoint = "https://api.openai.com/v1"
        self.endpoint = QLineEdit(saved_endpoint)
        self.endpoint.setStyleSheet(
            "QLineEdit{border:1px solid #ddd;border-radius:6px;padding:8px 12px;font-size:13px;}")
        self.endpoint.textChanged.connect(lambda v: self.settings.setValue("api/endpoint", v))
        api_form.addRow("Endpoint URL:", self.endpoint)

        key_row = QHBoxLayout()
        self.apikey = QLineEdit()
        self.apikey.setEchoMode(QLineEdit.EchoMode.Password)
        self.apikey.setPlaceholderText("sk-...")
        saved_key = self.settings.value("api/key", "")
        if saved_key:
            self.apikey.setText(saved_key)
        self.apikey.textChanged.connect(lambda v: self.settings.setValue("api/key", v))
        self.apikey.setStyleSheet(
            "QLineEdit{border:1px solid #ddd;border-radius:6px;padding:8px 12px;font-size:13px;}")
        key_row.addWidget(self.apikey)
        self.test_btn = QPushButton("🧪 Test Connection")
        self.test_btn.setStyleSheet(
            "QPushButton{background:#2ecc71;color:#fff;border:none;border-radius:6px;"
            "padding:8px 16px;font-size:12px;font-weight:600;}"
            "QPushButton:hover{background:#27ae60;}")
        self.test_btn.clicked.connect(self._test)
        key_row.addWidget(self.test_btn)
        api_form.addRow("API Key:", key_row)

        model_row = QHBoxLayout()
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setStyleSheet(
            "QComboBox{border:1px solid #ddd;border-radius:6px;padding:6px 12px;font-size:13px;}")
        saved_model = self.settings.value("api/model", "deepseek-chat")
        self.model.setEditText(saved_model)
        self.model.currentTextChanged.connect(lambda v: self.settings.setValue("api/model", v))
        model_row.addWidget(self.model)
        fetch_btn = QPushButton("🔄 Fetch")
        fetch_btn.setStyleSheet(
            "QPushButton{background:#ecf0f1;color:#333;border:1px solid #ddd;border-radius:6px;"
            "padding:8px 14px;font-size:12px;}"
            "QPushButton:hover{background:#dfe6e9;}")
        fetch_btn.clicked.connect(self._fetch_models)
        model_row.addWidget(fetch_btn)
        api_form.addRow("Model:", model_row)

        self.api_status = QLabel("")
        self.api_status.setWordWrap(True)
        self.api_status.setStyleSheet("font-size:12px;padding:4px 0;")
        api_form.addRow(self.api_status)

        layout.addWidget(api_grp)

        # -- Coding Form --
        form_grp = QGroupBox("Coding Form")
        form_layout = QVBoxLayout(form_grp)

        # Preset templates
        tpl_row = QHBoxLayout()
        tpl_row.addWidget(QLabel("可用模板:"))
        self.template_combo = QComboBox()
        self.template_combo.addItems([
            "None (manual upload)",
            "Study Info (7 fields)",
            "Effect Size (3 fields)",
            "Custom Outcomes (3 fields)",
            "ROB2 (7 fields)",
        ])
        self.template_combo.currentIndexChanged.connect(self._apply_template)
        self.template_combo.setStyleSheet(
            "QComboBox{border:1px solid #ddd;border-radius:6px;padding:6px 12px;font-size:13px;}")
        tpl_row.addWidget(self.template_combo)
        tpl_row.addStretch()
        form_layout.addLayout(tpl_row)

        up_row = QHBoxLayout()
        self.form_btn = QPushButton("📂 Upload Coding Form (.csv / .xlsx)")
        self.form_btn.setStyleSheet(
            "QPushButton{background:#2ecc71;color:#fff;border:none;border-radius:8px;"
            "padding:10px 20px;font-size:13px;font-weight:600;}"
            "QPushButton:hover{background:#27ae60;}")
        self.form_btn.clicked.connect(self._upload_form)
        up_row.addWidget(self.form_btn)

        self.form_clear_btn = QPushButton("✕ Clear")
        self.form_clear_btn.setStyleSheet(
            "QPushButton{background:#f39c12;color:#fff;border:none;border-radius:8px;"
            "padding:10px 14px;font-size:12px;font-weight:600;}"
            "QPushButton:hover{background:#e67e22;}")
        self.form_clear_btn.clicked.connect(self._clear_form)
        up_row.addWidget(self.form_clear_btn)

        up_row.addStretch()
        form_layout.addLayout(up_row)

        self.form_status = QLabel("No coding form loaded")
        self.form_status.setWordWrap(True)
        self.form_status.setStyleSheet("color:#888;")
        form_layout.addWidget(self.form_status)

        self.prompt_list = QTextEdit()
        self.prompt_list.setMaximumHeight(160)
        self.prompt_list.setPlaceholderText("Prompts will appear here. You can edit them directly. One prompt per line.")
        form_layout.addWidget(self.prompt_list)

        self.apply_prompts_btn = QPushButton("✓ Apply Edited Prompts")
        self.apply_prompts_btn.setStyleSheet(
            "QPushButton{background:#3498db;color:#fff;border:none;border-radius:6px;"
            "padding:6px 14px;font-size:12px;}"
            "QPushButton:hover{background:#2980b9;}")
        self.apply_prompts_btn.clicked.connect(self._apply_prompts)
        form_layout.addWidget(self.apply_prompts_btn)

        layout.addWidget(form_grp)
        layout.addStretch()

        # Author info
        author = QLabel(
            '<div style="text-align:center;color:#999;font-size:11px;padding:8px 0;">'
            'AIDE Free · 系统综述数据提取工具<br>'
            '作者小红书：94373279426</div>')
        layout.addWidget(author)

        # Defer auto-load until widget is in parent hierarchy
        QTimer.singleShot(50, self._auto_load_last_form)

    def _auto_load_last_form(self):
        last_form = self.settings.value("form/last_path", "")
        if last_form and os.path.exists(last_form):
            self._load_form(last_form)

    def get_config(self) -> dict:
        return {
            'endpoint': self.endpoint.text().strip(),
            'api_key': self.apikey.text().strip(),
            'model': self.model.currentText().strip()
        }

    def _test(self):
        self.test_btn.setEnabled(False)
        self.api_status.setText("⏳ Testing connection...")
        self.api_status.setStyleSheet("color:#333;font-size:12px;padding:4px 0;")
        self.w = TestConnWorker(self.get_config())
        self.w.finished.connect(self._test_done)
        self.w.start()

    def _test_done(self, ok, msg):
        self.test_btn.setEnabled(True)
        self.api_status.setText(f"{'✅' if ok else '❌'} {msg}")
        self.api_status.setStyleSheet(
            f"color:{'#28a745' if ok else '#dc3545'};font-size:12px;font-weight:bold;padding:4px 0;")

    def _fetch_models(self):
        self.api_status.setText("⏳ Fetching models...")
        self.api_status.setStyleSheet("color:#333;font-size:12px;padding:4px 0;")
        self.w = FetchModelsWorker(self.get_config())
        self.w.finished.connect(self._models_done)
        self.w.start()

    def _models_done(self, ok, models, msg):
        if ok:
            self.model.clear()
            self.model.addItems(models)
        self.api_status.setText(f"{'✅' if ok else '❌'} {msg}")
        self.api_status.setStyleSheet(
            f"color:{'#28a745' if ok else '#dc3545'};font-size:12px;font-weight:bold;padding:4px 0;")

    def _upload_form(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Coding Form", "",
            "Spreadsheets (*.csv *.xls *.xlsx);;All Files (*)")
        if not path:
            return
        self._load_form(path)
        self.settings.setValue("form/last_path", path)

    def _clear_form(self):
        mw = self._mw()
        if mw:
            mw.coding_form_df = None
            mw.prompts = []
        self.form_status.setText("No coding form loaded")
        self.form_status.setStyleSheet("color:#888;")
        self.prompt_list.clear()
        self.settings.setValue("form/last_path", "")

    def _apply_prompts(self):
        """Apply edited prompts from the text area."""
        import pandas as pd
        text = self.prompt_list.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Empty", "No prompts to apply.")
            return
        prompts = []
        for line in text.strip().split('\n'):
            line = line.strip()
            # Remove leading numbers like "1. " or "1. "
            if line and line[0].isdigit() and '. ' in line[:4]:
                line = line.split('. ', 1)[1]
            if line:
                prompts.append(line)
        if not prompts:
            return
        df = pd.DataFrame([prompts])
        mw = self._mw()
        if mw:
            mw.coding_form_df = df
            mw.prompts = prompts
        self.form_status.setText(f"✅ Applied {len(prompts)} prompts")
        self.form_status.setStyleSheet("color:#28a745;")

    def _apply_template(self, idx):
        """Apply a preset template: fill prompts and update coding form."""
        import pandas as pd
        if idx == 0:  # None
            return
        self._custom_outcomes = []
        templates = {
            1: ("Study Info (7 fields)", [
                "First author last name and publication year. Format: 'Smith 2023'. DO NOT include et al, DO NOT use full title like 'Smith et al., 2023, Journal of...'",
                "Trial registration number (e.g., NCT01234567, ChiCTR2000123456). If not reported, write 'Not reported'.",
                "Study design. Examples: 'randomized double-blind placebo-controlled trial', 'open-label multicenter RCT', 'prospective cohort study', 'crossover trial'.",
                "Total sample size with sex breakdown. Format: 'Total N=XXX, Male=XX, Female=XX'. If sex not reported, write total N only. Example: 'Total N=240, Male=132, Female=108'.",
                "Patient age by group. Format: 'Intervention: mean±SD (or median [IQR/range]), Control: mean±SD (or median [IQR/range])'. Example: 'Intervention: 65.4±8.7, Control: 64.8±9.2'.",
                "Key baseline values (NOT including age/sex which are separate fields). List each variable on a SEPARATE LINE. Example:\nBMI: Intervention=27.3±3.1, Control=28.1±3.5\nFEV1%: Intervention=52.3±8.7, Control=51.9±9.0\nSmoking pack-years: Intervention=45.2±22.1, Control=43.8±20.5\nList 5-8 most important disease-specific measures.",
                "Interventions in detail. Format: 'Intervention group: drug name, dose, frequency, route, duration. Control group: same details'. Example: 'Intervention: FSC 250/50mcg 1 inhalation BID for 52 weeks. Control: placebo 1 inhalation BID for 52 weeks'.",
            ]),
            2: ("Effect Size (3 fields)", [
                "First author last name and publication year. Format: 'Smith 2023'.",
                "INTERVENTION group outcomes. Extract ALL outcomes: EFFICACY and SAFETY. Format each line: Outcome name | N | value | Source p.X. One outcome per line. Example:\nFVC | N=120 | 2.45±0.32 | Source p.5\nSGRQ | N=118 | -8.2±2.1 | Source p.6\nExacerbations | N=120 | 28/120 | Source p.7",
                "CONTROL group outcomes. Same format. List in SAME ORDER as intervention. Example:\nFVC | N=118 | 1.12±0.28 | Source p.5\nSGRQ | N=112 | -3.1±1.9 | Source p.6\nExacerbations | N=118 | 45/118 | Source p.7",
            ]),
            3: ("Custom Outcomes (3 fields)", [
                "First author last name and publication year. Format: 'Smith 2023'.",
                None,  # placeholder — filled with user-specified outcomes
                None,  # placeholder — filled with user-specified outcomes
            ]),
            4: ("ROB2 (7 fields)", [
                "First author last name and publication year. Format: 'Smith 2023'.",
                "Domain 1 — Risk of bias arising from randomization. Judgment (Low / Some concerns / High) + key evidence: allocation concealment, sequence generation, baseline balance.",
                "Domain 2 — Risk of bias due to deviations from intended interventions. Judgment + key evidence: blinding of participants/personnel, ITT analysis.",
                "Domain 3 — Risk of bias due to missing outcome data. Judgment + key evidence: dropout rates, reasons, handling of missing data.",
                "Domain 4 — Risk of bias in outcome measurement. Judgment + key evidence: blinding of assessors, objective vs subjective outcomes.",
                "Domain 5 — Risk of bias in selection of reported result. Judgment + key evidence: pre-registration, protocol, selective reporting.",
                "Overall ROB2 judgment based on Domains 1\u20135 above: Low risk (all 5 Low) / Some concerns (at least one Some concerns, no High) / High risk (at least one High). Brief justification.",
            ]),
        }
        if idx not in templates:
            return
        name, prompts = templates[idx]

        # Handle Custom Outcomes template
        if idx == 3 and prompts[1] is None:
            outcomes, ok = QInputDialog.getMultiLineText(self, "Custom Outcomes",
                "Enter outcome names (one per line):\n\nExample:\nFEV1\nSGRQ\nExacerbations\nAdverse events",
                "FEV1\nSGRQ\nExacerbations\nAdverse events")
            if not ok or not outcomes.strip():
                return
            outcome_list = [o.strip() for o in outcomes.strip().split('\n') if o.strip()]
            if not outcome_list:
                return
            self._custom_outcomes = outcome_list
            prompts = [
                "First author last name and publication year. Format: 'Smith 2023'.",
                "INTERVENTION group. Extract ONLY these outcomes (one per line): " + ', '.join(outcome_list) + ". Format: Outcome | N | value | Source p.X.",
                "CONTROL group. Same order as intervention. Extract ONLY: " + ', '.join(outcome_list) + ". Format: Outcome | N | value | Source p.X.",
            ]

        self.prompt_list.setPlainText(
            "\n".join(f"{i+1}. {p}" for i, p in enumerate(prompts)))
        mw = self._mw()
        if not mw:
            return
        df = pd.DataFrame([prompts])
        mw.coding_form_df = df
        mw.prompts = prompts
        self.form_status.setText(f"✅ Template loaded: {name} ({len(prompts)} fields)")
        self.form_status.setStyleSheet("color:#28a745;")

    def _load_form(self, path):
        try:
            import pandas as pd
            if path.endswith('.csv'):
                df = pd.read_csv(path, header=None)
            else:
                df = pd.read_excel(path, header=None)
            df = df.dropna(how='all')
            prompts = df.iloc[0].tolist()
            self.prompt_list.setPlainText(
                "\n".join(f"{i+1}. {str(p)[:120]}" for i, p in enumerate(prompts)))

            mw = self._mw()
            if mw:
                mw.coding_form_df = df
                mw.prompts = prompts
            self.form_status.setText(f"✅ {len(prompts)} fields loaded | {len(df)} rows (incl. header)")
            self.form_status.setStyleSheet("color:#28a745;")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _mw(self):
        w = self.parent()
        while w:
            if isinstance(w, MainWindow):
                return w
            w = w.parent()
        return None


# ============================================================
# Analyze Tab
# ============================================================

class AnalyzeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pdf_path = ""
        self.field_cards = []
        self._active_src = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Splitter: left PDF, right panel
        split = QSplitter(Qt.Orientation.Horizontal)

        self.pdf_viewer = PdfViewer()
        split.addWidget(self.pdf_viewer)

        # Right side
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 0)

        # Top bar
        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 4)

        self.open_pdf_btn = QPushButton("📂 Open PDF")
        self.open_pdf_btn.setStyleSheet(
            "QPushButton{border-radius:8px;padding:10px 18px;font-size:13px;font-weight:600;"
            "border:none;background:#7f8c8d;color:#fff;}"
            "QPushButton:hover{background:#6c7a7d;}")
        self.open_pdf_btn.clicked.connect(self._on_open_pdf)
        top_layout.addWidget(self.open_pdf_btn)

        self.analyze_btn = QPushButton("🚀 Analyze")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet(
            "QPushButton{border-radius:8px;padding:10px 18px;font-size:13px;font-weight:600;"
            "border:none;background:#3498db;color:#fff;}"
            "QPushButton:hover{background:#2980b9;}"
            "QPushButton:disabled{background:#bdc3c7;color:#fff;}")
        self.analyze_btn.clicked.connect(self._analyze)
        top_layout.addWidget(self.analyze_btn)

        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.setStyleSheet(
            "QPushButton{border-radius:8px;padding:10px 18px;font-size:13px;font-weight:600;"
            "border:none;background:#ecf0f1;color:#555;}"
            "QPushButton:hover{background:#dfe6e9;}")
        self.reset_btn.clicked.connect(self._reset)
        top_layout.addWidget(self.reset_btn)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-size:12px;")
        top_layout.addWidget(self.status)
        top_layout.addStretch()
        right_layout.addWidget(top)

        self.progress = QProgressBar()
        self.progress.hide()
        right_layout.addWidget(self.progress)

        # Scrollable cards
        card_scroll = QScrollArea()
        card_scroll.setWidgetResizable(True)
        self.card_container = QWidget()
        self.card_layout = QVBoxLayout(self.card_container)
        self.card_layout.setContentsMargins(4, 4, 4, 8)
        self.card_layout.setSpacing(4)
        self.card_layout.addStretch()
        card_scroll.setWidget(self.card_container)
        right_layout.addWidget(card_scroll, 1)

        # Status bar at bottom
        self.recorded_lbl = QLabel("0 fields recorded")
        self.recorded_lbl.setStyleSheet("color:#888;font-size:12px;")
        right_layout.addWidget(self.recorded_lbl)

        split.addWidget(right)
        split.setStretchFactor(0, 13)
        split.setStretchFactor(1, 7)
        layout.addWidget(split)

    def _mw(self):
        w = self.parent()
        while w:
            if isinstance(w, MainWindow):
                return w
            w = w.parent()
        return None

    def _on_open_pdf(self):
        mw = self._mw()
        if mw:
            mw._open_pdf()

    def _analyze(self):
        mw = self._mw()
        if not mw or not mw.prompts:
            QMessageBox.warning(self, "Missing", "Upload a coding form in Setup first.")
            return
        config = mw.setup_tab.get_config()
        if not config['api_key']:
            QMessageBox.warning(self, "Missing", "Enter API key in Setup first.")
            return
        if not self.pdf_viewer.pdf_bytes:
            QMessageBox.warning(self, "Missing", "Open a PDF via File menu first.")
            return

        # Daily usage limit (Free version)
        s = QSettings("AIDE", "Usage")
        today = str(date.today())
        saved = s.value("date", "")
        if saved != today:
            s.setValue("date", today)
            s.setValue("count", 0)
        used = int(s.value("count", 0))
        if used >= 3:
            reply = QMessageBox.question(self, "今日限额已满",
                "免费版每天最多分析 3 篇 PDF。\n\n"
                "· 明天再来，或获取正式版（不限量）\n"
                "· 联系作者小红书：94373279426",
                QMessageBox.StandardButton.Ok)
            return

        self.analyze_btn.setEnabled(False)
        self.progress.show()
        self.progress.setRange(0, 0)
        self.status.setText("Analyzing...")

        self._worker = AnalyzeWorker(config, mw.prompts, self.pdf_viewer.pdf_bytes)
        self._worker.finished.connect(self._on_result)
        self._worker.start()

    def _on_result(self, ok, result):
        self.progress.hide()
        self.analyze_btn.setEnabled(True)

        if not ok:
            self.status.setText(f"❌ {result}")
            return

        # Extract and display timing breakdown
        timing = result.pop('_timing', None)

        self._clear_cards()
        mw = self._mw()
        for i, (key, data) in enumerate(result.items()):
            card = FieldCard(i, data.get('prompt', ''))
            card.set_data(
                data.get('response', ''),
                data.get('source_quote', ''),
                data.get('source_page', None))
            card.source_clicked.connect(self._on_src)
            card.record_clicked.connect(self._on_rec)
            self.card_layout.insertWidget(self.card_layout.count() - 1, card)
            self.field_cards.append(card)

        # Add row to coding form
        if mw and mw.coding_form_df is not None:
            import pandas as pd
            empty = [""] * len(mw.coding_form_df.columns)
            nr = pd.DataFrame([empty], columns=mw.coding_form_df.columns)
            mw.coding_form_df = pd.concat([mw.coding_form_df, nr], ignore_index=True)
            mw.current_row_idx = len(mw.coding_form_df) - 1

        n = len(result)
        self.recorded_lbl.setText(f"📋 0/{n} fields recorded")


        # Auto-save to history
        if mw and mw.coding_form_df is not None and mw.current_row_idx is not None:
            field_data = []
            recorded = []
            for card in self.field_cards:
                field_data.append({
                    'response': card.editor.toPlainText(),
                    'source_quote': card.source_quote,
                    'source_page': card.source_page,
                })
                recorded.append(card.is_recorded)
            HistoryTab.save_entry(
                getattr(self, 'pdf_path', '') or '',
                mw.prompts, mw.coding_form_df,
                mw.current_row_idx, field_data, recorded)
            mw.history_tab.refresh()
            mw.export_tab.refresh(mw.coding_form_df)

        # Increment daily usage count
        s = QSettings("AIDE", "Usage")
        used = int(s.value("count", 0))
        s.setValue("count", used + 1)
        if timing:
            self.status.setText(f"📋 今日已分析 {used+1}/3 篇 ✅ (LLM {timing['llm']}s)")
        else:
            self.status.setText(f"📋 今日已分析 {used+1}/3 篇 ✅")

    def _clear_cards(self):
        while self.card_layout.count() > 1:
            item = self.card_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.field_cards.clear()
        self._active_src = -1

    def _on_src(self, idx, quote, page):
        try:
            self.status.setText("🔍 Searching...")
            self.pdf_viewer._clear_highlights()
            if 0 <= self._active_src < len(self.field_cards):
                self.field_cards[self._active_src].set_active(False)
            self._active_src = idx
            if 0 <= idx < len(self.field_cards):
                self.field_cards[idx].set_active(True)

            if page is not None:
                try:
                    page = int(page)
                except (ValueError, TypeError):
                    page = None

            if self.pdf_viewer.pdf_bytes and quote:
                # Debug: show quote preview
                self.status.setText(f"🔍 {quote[:60]}...")
                has_multi = '|' in quote
                if has_multi:
                    # Extract outcome keywords and source pages
                    keywords = []
                    source_pages = set()
                    for line in quote.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split('|')
                        kw = parts[0].strip()
                        if len(kw) >= 3:
                            keywords.append(kw)
                        # Parse source page from the last segment
                        for seg in parts:
                            seg = seg.strip().lower()
                            import re
                            m = re.search(r'(?:source\s*)?p\.?\s*(\d+)', seg)
                            if m:
                                source_pages.add(int(m.group(1)))
                    # Use explicit source pages when available
                    if source_pages:
                        if page is None:
                            page = min(source_pages)
                        else:
                            page = min(source_pages)
                    if keywords:
                        import fitz
                        try:
                            doc = fitz.open(stream=self.pdf_viewer.pdf_bytes, filetype="pdf")
                            page_hits = {}  # page_num -> [rects]
                            for kw in keywords:
                                for pn in range(1, self.pdf_viewer.total_pages + 1):
                                    areas = doc[pn - 1].search_for(kw)
                                    if areas:
                                        if pn not in page_hits:
                                            page_hits[pn] = []
                                        for a in areas:
                                            page_hits[pn].append((a.x0, a.y0, a.x1, a.y1))
                            doc.close()
                            if page_hits:
                                # Use the page with most keyword matches
                                best_pn = max(page_hits, key=lambda p: len(set(r[2] for r in page_hits[p])))
                                best_rects = page_hits[best_pn]
                                self.pdf_viewer.highlight_and_scroll(best_pn, best_rects)
                                self.status.setText(f"📍 Page {best_pn} — {len(keywords)} outcomes, {len(page_hits)} pages")
                                return
                        except Exception:
                            pass

                sr = search_text_for_highlight(self.pdf_viewer.pdf_bytes, quote, preferred_page=page)
                if sr['found']:
                    self.pdf_viewer.highlight_and_scroll(sr['page'], sr['highlights'])
                    self.status.setText(f"📍 Page {sr['page']}")
                    return

            if page and 1 <= page <= self.pdf_viewer.total_pages:
                self.pdf_viewer.highlight_and_scroll(page, [])
                self.status.setText(f"📄 Page {page}")
            else:
                self.status.setText("⚠️ No source page")
        except Exception as e:
            QMessageBox.warning(self, "Source Error", f"{e}")

    def _on_rec(self, idx, text):
        if 0 <= idx < len(self.field_cards):
            card = self.field_cards[idx]
            card.toggle_recorded()

            mw = self._mw()
            if mw and mw.coding_form_df is not None and mw.current_row_idx is not None:
                if idx < len(mw.coding_form_df.columns):
                    # Save text if recording, clear if un-recording
                    mw.coding_form_df.at[mw.current_row_idx, mw.coding_form_df.columns[idx]] = (
                        text if card.is_recorded else "")

        done = sum(1 for c in self.field_cards if c.is_recorded)
        total = len(self.field_cards)
        self.recorded_lbl.setText(f"📋 {done}/{total} fields recorded")
        self.status.setText(f"📋 {done}/{total} recorded")

        if done == total:
            self.recorded_lbl.setText(f"✅ All {total} fields recorded — ready to export")
            self.recorded_lbl.setStyleSheet("color:#28a745;font-weight:bold;font-size:12px;")
            self.status.setText("🎉 All done!")
        else:
            self.recorded_lbl.setStyleSheet("color:#888;font-size:12px;")

        # Auto-update history on record toggle
        if mw and mw.coding_form_df is not None and mw.current_row_idx is not None:
            field_data = []
            recorded = []
            for c in self.field_cards:
                field_data.append({
                    'response': c.editor.toPlainText(),
                    'source_quote': c.source_quote,
                    'source_page': c.source_page,
                })
                recorded.append(c.is_recorded)
            HistoryTab.save_entry(
                getattr(self, 'pdf_path', '') or '',
                mw.prompts, mw.coding_form_df,
                mw.current_row_idx, field_data, recorded)
            mw.history_tab.refresh()

    def _reset(self):
        self._clear_cards()
        self.status.setText("")
        self.recorded_lbl.setText("0 fields recorded")
        self.recorded_lbl.setStyleSheet("color:#888;font-size:12px;")
        self.progress.hide()
        mw = self._mw()
        if mw:
            mw.current_row_idx = None

# ============================================================
# Export Tab
# ============================================================
class HistoryTab(QWidget):
    """Track analysis sessions so users can resume after a crash."""
    HISTORY_FILE = os.path.join(os.path.expanduser('~'), '.aide_history.json')

    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_window = lambda: self.window() if isinstance(self.window(), MainWindow) else None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("<b>📋 Analysis History</b>")
        title.setStyleSheet("font-size:18px;color:#333;padding:6px 0;")
        layout.addWidget(title)

        hint = QLabel("Restore a previous session — PDF, extracted data, and recorded fields will be reloaded.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888;font-size:12px;padding:0 0 8px 0;")
        layout.addWidget(hint)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Study", "Date", "Fields", "Recorded", "PDF"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setStyleSheet("QTableWidget{border:1px solid #ddd;border-radius:4px;}")
        layout.addWidget(self.table)

        # Buttons
        bl = QHBoxLayout()
        self.import_btn = QPushButton("↩️ Restore Selected")
        self.import_btn.setStyleSheet(
            "QPushButton{border-radius:6px;padding:8px 20px;font-size:13px;font-weight:600;"
            "background:#3498db;color:#fff;border:none;}"
            "QPushButton:hover{background:#2980b9;}")
        self.import_btn.clicked.connect(self._import_to_table)
        bl.addWidget(self.import_btn)

        self.delete_btn = QPushButton("🗑️ Delete")
        self.delete_btn.setStyleSheet(
            "QPushButton{border-radius:6px;padding:8px 20px;font-size:13px;font-weight:600;"
            "background:#e74c3c;color:#fff;border:none;}"
            "QPushButton:hover{background:#c0392b;}")
        self.delete_btn.clicked.connect(self._delete)
        bl.addWidget(self.delete_btn)

        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setStyleSheet(
            "QPushButton{border-radius:6px;padding:8px 20px;font-size:13px;font-weight:600;"
            "background:#f8f9fa;color:#555;border:1px solid #ccc;}"
            "QPushButton:hover{background:#e9ecef;}")
        self.refresh_btn.clicked.connect(self.refresh)
        bl.addWidget(self.refresh_btn)

        bl.addStretch()
        layout.addLayout(bl)

        # Status
        self.status = QLabel("")
        self.status.setStyleSheet("color:#888;font-size:12px;padding:4px 0;")
        layout.addWidget(self.status)

        self.refresh()

    # ── data model ──
    @staticmethod
    def _history_path():
        return HistoryTab.HISTORY_FILE

    def _load_all(self) -> list:
        try:
            with open(self._history_path(), 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_all(self, entries: list):
        with open(self._history_path(), 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    @staticmethod
    def save_entry(pdf_path: str, prompts: list, df, row_idx: int,
                   field_data: list, recorded: list):
        """
        Save a history entry.  Called after analysis or on record toggle.
        field_data = [(response, source_quote, source_page), ...]
        """
        import pandas as pd
        if df is not None:
            df_json = df.to_json(orient='split', force_ascii=False)
        else:
            df_json = None

        entry = {
            'id': str(int(time.time())),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pdf_path': pdf_path,
            'pdf_name': os.path.basename(pdf_path) if pdf_path else '',
            'prompts': prompts,
            'coding_form': df_json,
            'row_idx': row_idx,
            'field_data': field_data,
            'recorded': recorded,
            'study': str(df.iloc[row_idx, 0]) if df is not None and row_idx is not None and row_idx < len(df) and pd.notna(df.iloc[row_idx, 0]) else '',
        }

        path = HistoryTab.HISTORY_FILE
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
            else:
                entries = []
        except (FileNotFoundError, json.JSONDecodeError):
            entries = []

        # Replace previous entry for same PDF (keep only latest)
        entries = [e for e in entries if e.get('pdf_path') != pdf_path]
        entries.append(entry)

        # Keep max 50 entries
        entries = entries[-50:]

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    # ── UI ──
    def refresh(self):
        """Reload table from history file."""
        entries = self._load_all()
        self.table.setRowCount(len(entries))
        for i, e in enumerate(reversed(entries)):
            self.table.setItem(i, 0, QTableWidgetItem(e.get('study', '')))
            self.table.setItem(i, 1, QTableWidgetItem(e.get('timestamp', '')))
            n_fields = len(e.get('prompts', []))
            n_rec = sum(1 for r in e.get('recorded', []) if r)
            self.table.setItem(i, 2, QTableWidgetItem(str(n_fields)))
            self.table.setItem(i, 3, QTableWidgetItem(f"{n_rec}/{n_fields}"))
            self.table.setItem(i, 4, QTableWidgetItem(e.get('pdf_name', '')))
        self.status.setText(f"{len(entries)} session(s) saved")

    def _restore(self):
        """Restore selected history entry."""
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No Selection", "Select a session to restore first.")
            return
        entries = self._load_all()
        # Table shows reversed order
        idx = len(entries) - 1 - row
        if idx < 0 or idx >= len(entries):
            return
        e = entries[idx]
        mw = self._main_window()
        if not mw:
            return

        # Confirm
        study = e.get('study', 'Unknown')
        ans = QMessageBox.question(
            self, "Restore Session",
            f"Restore analysis for '{study}'?\n"
            f"PDF: {e.get('pdf_name', '')}\n"
            f"Date: {e.get('timestamp', '')}\n\n"
            "Current unsaved data will be lost.")
        if ans != QMessageBox.StandardButton.Yes:
            return

        # 1. Restore coding form
        df_json = e.get('coding_form')
        if df_json:
            import pandas as pd
            try:
                mw.coding_form_df = pd.read_json(df_json, orient='split')
            except Exception:
                mw.coding_form_df = pd.DataFrame([e.get('prompts', [])])
        else:
            mw.coding_form_df = pd.DataFrame([e.get('prompts', [])])
        mw.prompts = e.get('prompts', [])
        mw.current_row_idx = e.get('row_idx')

        # 2. Load PDF
        pdf_path = e.get('pdf_path', '')
        pdf_loaded = False
        if pdf_path and os.path.exists(pdf_path):
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                mw.analyze_tab._reset()
                mw.analyze_tab.pdf_viewer.load_pdf(pdf_bytes)
                mw.analyze_tab.analyze_btn.setEnabled(True)
                pdf_loaded = True
            except Exception:
                pdf_loaded = False

        if not pdf_loaded and pdf_path:
            ans = QMessageBox.question(
                self, "PDF Not Found",
                f"Cannot find:\n{pdf_path}\n\nLocate the PDF file?")
            if ans == QMessageBox.StandardButton.Yes:
                new_path, _ = QFileDialog.getOpenFileName(
                    self, "Locate PDF", "", "PDF Files (*.pdf)")
                if new_path:
                    pdf_path = new_path
                    with open(pdf_path, 'rb') as f:
                        pdf_bytes = f.read()
                    mw.analyze_tab._reset()
                    mw.analyze_tab.pdf_viewer.load_pdf(pdf_bytes)
                    mw.analyze_tab.analyze_btn.setEnabled(True)
                    pdf_loaded = True

        # 3. Rebuild field cards
        recorded = e.get('recorded', [])
        field_data = e.get('field_data', [])
        mw.analyze_tab._clear_cards()
        for i, prompt in enumerate(mw.prompts):
            card = FieldCard(i, prompt)
            if i < len(field_data):
                fd = field_data[i]
                if isinstance(fd, dict):
                    card.set_data(
                        fd.get('response', ''),
                        fd.get('source_quote', ''),
                        fd.get('source_page', None))
                else:
                    card.set_data(str(fd) if fd else '', '', None)
            if i < len(recorded) and recorded[i]:
                card.toggle_recorded()
            card.source_clicked.connect(mw.analyze_tab._on_src)
            card.record_clicked.connect(mw.analyze_tab._on_rec)
            mw.analyze_tab.card_layout.insertWidget(
                mw.analyze_tab.card_layout.count() - 1, card)
            mw.analyze_tab.field_cards.append(card)

        done = sum(1 for r in recorded if r)
        total = len(mw.prompts)
        mw.analyze_tab.recorded_lbl.setText(f"📋 {done}/{total} fields recorded")
        if done == total:
            mw.analyze_tab.recorded_lbl.setText(f"✅ All {total} fields recorded")
            mw.analyze_tab.recorded_lbl.setStyleSheet("color:#28a745;font-weight:bold;font-size:12px;")
        else:
            mw.analyze_tab.recorded_lbl.setStyleSheet("color:#888;font-size:12px;")

        # 4. Switch to Analyze tab
        mw.tabs.setCurrentIndex(1)

        self.status.setText(f"✅ Restored '{study}' — {total} fields, {done} recorded")
        QMessageBox.information(self, "Done", f"Session '{study}' restored.\nSwitch to Analyze tab to review.")

    def _import_to_table(self):
        """Import selected history entries into the export table (multi-select)."""
        selected = self.table.selectedIndexes()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Select session(s) to import first.")
            return
        entries = self._load_all()
        rows = set()
        for idx in selected:
            rows.add(idx.row())
        entry_indices = [len(entries) - 1 - r for r in sorted(rows)]
        selected_entries = [entries[i] for i in entry_indices if 0 <= i < len(entries)]
        if not selected_entries:
            return
        mw = self._main_window()
        if not mw:
            return
        import pandas as pd
        ref_prompts = selected_entries[0].get('prompts', [])
        if not ref_prompts:
            QMessageBox.warning(self, "Error", "Selected entry has no prompts.")
            return
        for e in selected_entries:
            if e.get('prompts', []) != ref_prompts:
                QMessageBox.warning(self, "Template Mismatch",
                    "Cannot merge with a different coding form.")
                return
        n = len(ref_prompts)
        if mw.coding_form_df is not None and len(mw.coding_form_df) > 0:
            existing = [str(mw.coding_form_df.iloc[0, c]) if pd.notna(mw.coding_form_df.iloc[0, c]) else '' for c in range(min(n, len(mw.coding_form_df.columns)))]
            if existing != ref_prompts:
                ans = QMessageBox.question(self, "Different Template",
                    "Replace current table with the new template? Current data will be lost.")
                if ans != QMessageBox.StandardButton.Yes:
                    return
                mw.coding_form_df = pd.DataFrame([ref_prompts])
                mw.current_row_idx = None
                if not hasattr(mw, '_history_id_map'):
                    mw._history_id_map = {}
        else:
            mw.coding_form_df = pd.DataFrame([ref_prompts])
            mw._history_id_map = {}
        mw.prompts = ref_prompts
        if not hasattr(mw, '_history_id_map'):
            mw._history_id_map = {}
        imported = 0
        for e in selected_entries:
            row_data = {}
            for i in range(n):
                fd = e.get('field_data', [])
                val = ''
                if i < len(fd) and isinstance(fd[i], dict):
                    val = fd[i].get('response', '')
                row_data[i] = val
            nr = pd.DataFrame([row_data])
            mw.coding_form_df = pd.concat([mw.coding_form_df, nr], ignore_index=True)
            mw._history_id_map[len(mw.coding_form_df) - 1] = e.get('id', '')
            imported += 1
        mw.current_row_idx = len(mw.coding_form_df) - 1
        self.status.setText(f"Imported {imported} study/studies to export table")
        mw.export_tab.refresh(mw.coding_form_df)

    def _delete(self):
        """Delete selected history entry."""
        row = self.table.currentRow()
        if row < 0:
            return
        entries = self._load_all()
        idx = len(entries) - 1 - row
        if 0 <= idx < len(entries):
            study = entries[idx].get('study', 'Unknown')
            ans = QMessageBox.question(self, "Delete", f"Delete session '{study}'?")
            if ans == QMessageBox.StandardButton.Yes:
                entries.pop(idx)
                self._save_all(entries)
                self.refresh()
                self.status.setText(f"Deleted session '{study}'")


# ============================================================
# Export Tab
# ============================================================



class ExportTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("<h2>📋 Final Coding Form</h2>")
        hdr.addWidget(title)
        hdr.addStretch()

        self.export_xlsx = QPushButton("📥 Export Excel")
        self.export_xlsx.setStyleSheet(
            "QPushButton{border-radius:8px;padding:10px 22px;font-size:13px;font-weight:600;"
            "border:none;background:#3498db;color:#fff;}"
            "QPushButton:hover{background:#2980b9;}")
        self.export_xlsx.clicked.connect(lambda: self._do_export("xlsx"))
        hdr.addWidget(self.export_xlsx)


        self.export_es = QPushButton("📤 Export Effect Size")
        self.export_es.setStyleSheet(
            "QPushButton{border-radius:8px;padding:10px 22px;font-size:13px;font-weight:600;"
            "border:none;background:#9b59b6;color:#fff;}"
            "QPushButton:hover{background:#8e44ad;}")
        self.export_es.clicked.connect(self._export_effect_size)
        hdr.addWidget(self.export_es)

        self.export_rob2 = QPushButton("📤 Export ROB2")
        self.export_rob2.setStyleSheet(
            "QPushButton{border-radius:8px;padding:10px 22px;font-size:13px;font-weight:600;"
            "border:none;background:#e74c3c;color:#fff;}"
            "QPushButton:hover{background:#c0392b;}")
        self.export_rob2.clicked.connect(self._export_rob2)
        hdr.addWidget(self.export_rob2)

        self.export_baseline = QPushButton("📤 Export Baseline")
        self.export_baseline.setStyleSheet(
            "QPushButton{border-radius:8px;padding:10px 22px;font-size:13px;font-weight:600;"
            "border:none;background:#00a86b;color:#fff;}"
            "QPushButton:hover{background:#008c5a;}")
        self.export_baseline.clicked.connect(self._export_baseline)
        hdr.addWidget(self.export_baseline)
        layout.addLayout(hdr)

        # Stats
        self.stats = QLabel("No data yet. Complete the Setup and Analyze steps first.")
        self.stats.setStyleSheet("color:#888;font-size:12px;padding:8px 0;")
        layout.addWidget(self.stats)

        # Table preview
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setStyleSheet(
            "QTableWidget{border:1px solid #ddd;border-radius:4px;gridline-color:#eee;}"
            "QHeaderView::section{background:#f5f5f5;padding:6px 10px;"
            "border:1px solid #ddd;font-weight:bold;}")
        layout.addWidget(self.table, 1)

        # Delete row button
        del_row = QHBoxLayout()
        del_row.addStretch()
        self.delete_btn = QPushButton("🗑️ Delete Selected Row")
        self.delete_btn.setStyleSheet(
            "QPushButton{background:#e74c3c;color:#fff;border:none;"
            "border-radius:6px;padding:8px 16px;font-size:12px;font-weight:600;}"
            "QPushButton:hover{background:#c0392b;}")
        self.delete_btn.clicked.connect(self._delete_row)
        del_row.addWidget(self.delete_btn)
        layout.addLayout(del_row)

    def refresh(self, df, recorded_count=None):
        """Refresh table and stats from DataFrame."""
        if df is None or len(df) == 0:
            self.stats.setText("No data yet.")
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        rows = len(df)
        display_cols = [c for c in df.columns if c != '_history_id']
        show_df = df[display_cols]
        show_cols = len(show_df.columns)

        # Stats
        data_rows = rows - 1  # first row is prompts/header
        if recorded_count is not None:
            self.stats.setText(
                f"📊 <b>{data_rows}</b> studies extracted, "
                f"<b>{recorded_count}</b> fields recorded in current study | "
                f"Total: <b>{rows}</b> rows × <b>{cols}</b> columns"
            )
        else:
            self.stats.setText(
                f"📊 <b>{data_rows}</b> studies | <b>{rows}</b> rows × <b>{cols}</b> columns"
            )
        self.stats.setStyleSheet("color:#333;font-size:12px;padding:8px 0;")

        # Populate table
        self.table.setRowCount(rows)
        self.table.setColumnCount(show_cols)
        headers = []
        for c in range(cols):
            try:
                h = str(show_df.columns[c])
            except Exception:
                h = f"Col {c}"
            headers.append(h)
        self.table.setHorizontalHeaderLabels(headers)

        for r in range(rows):
            for c in range(show_cols):
                try:
                    val = str(show_df.iat[r, c]) if pd.notna(show_df.iat[r, c]) else ""
                except Exception:
                    val = ""
                item = QTableWidgetItem(val)
                if r == 0:
                    item.setBackground(QColor("#fff3cd"))
                elif val:
                    item.setBackground(QColor("#f6fff6"))
                self.table.setItem(r, c, item)


    def _export_effect_size(self):
        """Parse | -separated effect size columns and export as row-per-outcome CSV."""
        mw = self._mw()
        if mw is None or mw.coding_form_df is None or len(mw.coding_form_df) < 2:
            QMessageBox.warning(self, "No Data", "No effect size data to export.")
            return

        df = mw.coding_form_df
        ncols = len(df.columns)
        # Use last 2 columns as intervention and control outcomes
        id_col = max(0, ncols - 3)
        int_col = max(0, ncols - 2)
        ctrl_col = max(0, ncols - 1)

        # Check if data contains | separator
        has_pipe = False
        for i in range(1, min(len(df), 5)):
            for c in [int_col, ctrl_col]:
                val = str(df.iloc[i, c]) if pd.notna(df.iloc[i, c]) else ""
                if '|' in val:
                    has_pipe = True
                    break
        if not has_pipe:
            QMessageBox.warning(self, "No Effect Size Data",
                "This template doesn't contain | -separated effect size data.\n"
                "Use the Effect Size template to export expanded outcomes.")
            return

        rows = []
        for i in range(1, len(df)):  # skip header row
            study = str(df.iloc[i, id_col]) if pd.notna(df.iloc[i, id_col]) else f"Study {i}"
            int_raw = str(df.iloc[i, int_col]) if pd.notna(df.iloc[i, int_col]) else ""
            ctrl_raw = str(df.iloc[i, ctrl_col]) if pd.notna(df.iloc[i, ctrl_col]) else ""

            # Parse intervention outcomes
            for line in int_raw.split('\n'):
                line = line.strip()
                if not line or '|' not in line:
                    continue
                parts = [p.strip() for p in line.split('|')]
                outcome = parts[0]
                n_val = parts[1] if len(parts) > 1 else ""
                value = parts[2] if len(parts) > 2 else ""
                rows.append({'Study': study, 'Outcome': outcome, 'Group': 'Intervention',
                             'N': n_val, 'Value': value})

            # Parse control outcomes
            for line in ctrl_raw.split('\n'):
                line = line.strip()
                if not line or '|' not in line:
                    continue
                parts = [p.strip() for p in line.split('|')]
                outcome = parts[0]
                n_val = parts[1] if len(parts) > 1 else ""
                value = parts[2] if len(parts) > 2 else ""
                rows.append({'Study': study, 'Outcome': outcome, 'Group': 'Control',
                             'N': n_val, 'Value': value})

        if not rows:
            QMessageBox.information(self, "No Data", "No | -separated data found in effect size columns.")
            return

        out = pd.DataFrame(rows)
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Effect Size (expanded)", "effect_size_expanded.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if not path:
            return
        try:
            if path.endswith('.xlsx'):
                out.to_excel(path, index=False, engine='openpyxl')
            else:
                out.to_csv(path, index=False)
            QMessageBox.information(self, "Done", f"Saved {len(rows)} rows to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _plot_rob2(self):
        """Generate ROB2 traffic-light plot in a dialog."""
        mw = self._mw()
        if mw is None or mw.coding_form_df is None or len(mw.coding_form_df) < 2:
            QMessageBox.warning(self, "No Data", "No ROB2 data found.")
            return

        df = mw.coding_form_df
        ncols = len(df.columns)
        if ncols < 6:
            QMessageBox.warning(self, "No ROB2 Data", "Use the ROB2 template first.")
            return

        # Parse ROB2 judgments from each study row (skip header row 0)
        domain_names = ["D1: Randomization", "D2: Deviations", "D3: Missing data",
                        "D4: Measurement", "D5: Selection", "Overall"]
        colors = {"low": "#2ecc71", "some concerns": "#f39c12", "high": "#e74c3c"}
        labels = {"low": "+", "some concerns": "?", "high": "-"}

        studies = []
        judgments = []
        for i in range(1, len(df)):
            study = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else f"Study {len(studies)+1}"
            study = study.strip()
            row_j = []
            for c in range(min(6, ncols)):
                val = str(df.iloc[i, c]).strip().lower() if pd.notna(df.iloc[i, c]) else ""
                # Extract first judgment word
                for kw in ["high risk", "low risk", "some concerns"]:
                    if kw in val:
                        row_j.append(kw)
                        break
                else:
                    row_j.append("")
            if any(j for j in row_j):
                studies.append(study)
                judgments.append(row_j)

        if not studies:
            QMessageBox.warning(self, "No ROB2 Data", "Could not parse ROB2 judgments from data.")
            return

        # Draw plot
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.patches import Rectangle

            fig = Figure(figsize=(10, 1.5 * len(studies) + 1))
            canvas = FigureCanvasQTAgg(fig)
            ax = fig.add_subplot(111)
            ax.axis('off')

            n_rows = len(studies)
            n_cols = 7
            cell_size = 0.6

            for r, (study, row_j) in enumerate(zip(studies, judgments)):
                y = n_rows - 1 - r
                ax.text(-0.5, y, study, va='center', ha='right', fontsize=10, fontweight='bold')
                for c, j in enumerate(row_j):
                    color = colors.get(j, "#ecf0f1")
                    sym = labels.get(j, "")
                    rect = Rectangle((c, y - 0.4), cell_size, cell_size,
                                    facecolor=color, edgecolor='#555', linewidth=1.5)
                    ax.add_patch(rect)
                    if sym:
                        ax.text(c + 0.3, y, sym, va='center', ha='center',
                                fontsize=18, fontweight='bold', color='#fff')

            # Column headers
            for c, name in enumerate(domain_names):
                ax.text(c + 0.3, n_rows + 0.2, name, va='bottom', ha='center',
                        fontsize=8, fontweight='bold', rotation=20)

            ax.set_xlim(-2, n_cols)
            ax.set_ylim(-0.5, n_rows + 1)
            fig.tight_layout()

            dlg = QDialog(self)
            dlg.setWindowTitle("ROB2 Traffic-light Plot")
            dlg.resize(800, 150 * len(studies) + 100)
            layout = QVBoxLayout(dlg)
            layout.addWidget(canvas)

            btn_row = QHBoxLayout()
            save_btn = QPushButton("📥 Save PNG")
            def _save_rob2():
                path, _ = QFileDialog.getSaveFileName(dlg, "Save ROB2 Plot",
                    "rob2_plot", "PNG (*.png);;TIFF (*.tiff *.tif);;SVG (*.svg)")
                if path:
                    fmt = path.rsplit('.', 1)[-1] if '.' in path else 'png'
                    dpi = 300 if fmt in ('tiff', 'tif') else 200
                    fig.savefig(path, dpi=dpi, bbox_inches='tight')
            save_btn.clicked.connect(_save_rob2)
            btn_row.addWidget(save_btn)
            btn_row.addStretch()
            layout.addLayout(btn_row)
            dlg.exec()

        except ImportError:
            QMessageBox.warning(self, "Missing Library",
                "matplotlib is required for ROB2 plots.\nRun: pip install matplotlib")

    def _export_rob2(self):
        """Export ROB2 judgments only (strip evidence text)."""
        import pandas as pd
        import re
        mw = self._mw()
        if mw is None or mw.coding_form_df is None or len(mw.coding_form_df) < 2:
            QMessageBox.warning(self, "No Data", "No ROB2 data found.")
            return

        df = mw.coding_form_df
        ncols = len(df.columns)
        if ncols < 7:
            QMessageBox.warning(self, "No ROB2 Data", "Use the ROB2 (7 fields) template first.")
            return

        rows = []
        for i in range(1, len(df)):
            study = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else f"Study {i}"
            study = study.strip()
            judgments = []
            for c in range(1, min(7, ncols)):
                val = str(df.iloc[i, c]).strip() if pd.notna(df.iloc[i, c]) else ""
                # Extract just the judgment keyword
                val_lower = val.lower()
                if 'high risk' in val_lower:
                    judgments.append('High')
                elif 'some concerns' in val_lower:
                    judgments.append('Some concerns')
                elif 'low risk' in val_lower:
                    judgments.append('Low')
                else:
                    judgments.append('')
            row = {'Study': study}
            for j_name, j_val in zip(['D1', 'D2', 'D3', 'D4', 'D5', 'Overall'], judgments):
                row[j_name] = j_val
            rows.append(row)

        if not rows:
            return

        out = pd.DataFrame(rows)
        path, _ = QFileDialog.getSaveFileName(
            self, "Export ROB2", "rob2_judgments.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if not path:
            return
        try:
            if path.endswith('.xlsx'):
                out.to_excel(path, index=False, engine='openpyxl')
            else:
                out.to_csv(path, index=False)
            QMessageBox.information(self, "Done", f"Saved {len(rows)} studies to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _export_baseline(self):
        """Export Study Info baseline data as a clean transposed table (variables x groups)."""
        import re
        mw = self._mw()
        if mw is None or mw.coding_form_df is None or len(mw.coding_form_df) < 2:
            QMessageBox.warning(self, "No Data", "No Study Info data found.")
            return

        df = mw.coding_form_df
        ncols = len(df.columns)
        if ncols < 6:
            QMessageBox.warning(self, "No Baseline Data", "Use the Study Info (7 fields) template first.")
            return

        reg_col = 1
        design_col = 2
        sex_col = 3
        age_col = 4
        base_col = 5

        rows = []
        for i in range(1, len(df)):
            study = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else f"Study {i}"

            # Trial registration (field 1)
            reg = str(df.iloc[i, reg_col]) if pd.notna(df.iloc[i, reg_col]) else ""
            if reg and reg.strip() and reg.strip().lower() != 'not reported':
                rows.append({'Study': study, 'Variable': 'Trial registration',
                             'Group': 'Info', 'Value': reg.strip()})

            # Study design (field 2)
            design = str(df.iloc[i, design_col]) if pd.notna(df.iloc[i, design_col]) else ""
            if design and design.strip():
                rows.append({'Study': study, 'Variable': 'Study design',
                             'Group': 'Info', 'Value': design.strip()})

            # Parse sex from field 4
            sex_raw = str(df.iloc[i, sex_col]) if pd.notna(df.iloc[i, sex_col]) else ""
            if sex_raw:
                for sex_var in ['Male', 'Female', 'Total N']:
                    m_s = re.search(rf'{re.escape(sex_var)}=(\d[\d.]*)', sex_raw, re.I)
                    if m_s:
                        rows.append({'Study': study, 'Variable': sex_var,
                                     'Group': 'Info', 'Value': m_s.group(1)})

            # Parse age field (field 5): "Group: value, Group: value"
            age_raw = str(df.iloc[i, age_col]) if pd.notna(df.iloc[i, age_col]) else ""
            if age_raw and age_raw.strip():
                for part in age_raw.split(','):
                    part = part.strip()
                    m_age = re.match(r'(.+?):\s*(.+)', part)
                    if m_age:
                        rows.append({'Study': study, 'Variable': 'Age',
                                     'Group': m_age.group(1).strip(),
                                     'Value': m_age.group(2).strip()})

            # Parse baseline values (field 6): "Variable: Group=value, Group=value"
            base_raw = str(df.iloc[i, base_col]) if pd.notna(df.iloc[i, base_col]) else ""
            for line in base_raw.split('\n'):
                line = line.strip()
                if not line or ':' not in line:
                    continue
                var_name = line.split(':', 1)[0].strip()
                rest = line.split(':', 1)[1].strip()
                if not rest:
                    continue
                for pair in re.findall(r'([^,=]+)=([^,]+)', rest):
                    grp = pair[0].strip()
                    val = pair[1].strip()
                    rows.append({'Study': study, 'Variable': var_name,
                                 'Group': grp, 'Value': val})

        if not rows:
            QMessageBox.information(self, "No Data", "No structured baseline data found.\nRe-analyze with the updated Study Info template.")
            return

        out = pd.DataFrame(rows)

        try:
            out = out.pivot_table(index=['Study', 'Variable'], columns='Group',
                                  values='Value', aggfunc='first').reset_index()
            out.columns.name = None
            cols = out.columns.tolist()
            rest_cols = [c for c in cols if c not in ('Study', 'Variable', 'Info')]
            rest_cols.sort()
            ordered = ['Study', 'Variable']
            if 'Info' in cols:
                ordered.append('Info')
            out = out[ordered + rest_cols]

            if 'Info' in cols:
                info_vars = ['Trial registration', 'Study design', 'Total N', 'Male', 'Female']
                out['_sort_grp'] = out['Variable'].apply(
                    lambda x: 1 if x in info_vars else 0)
                out['_sort_order'] = out['Variable'].apply(
                    lambda x: info_vars.index(x) if x in info_vars else 999)
                out = out.sort_values(['_sort_grp', '_sort_order', 'Variable']) \
                         .drop(columns=['_sort_grp', '_sort_order']) \
                         .reset_index(drop=True)
        except Exception:
            pass

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Baseline Table", "baseline_table.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if not path:
            return
        try:
            if path.endswith('.xlsx'):
                out.to_excel(path, index=False, engine='openpyxl')
            else:
                out.to_csv(path, index=False, encoding='utf-8-sig')
            QMessageBox.information(self, "Done", f"Saved baseline table to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _restore_session(self):
        """Restore full session from selected export table row."""
        mw = self._mw()
        if mw is None or mw.coding_form_df is None:
            QMessageBox.warning(self, "No Data", "No data in export table.")
            return
        sel = self.table.currentRow()
        if sel < 0:
            QMessageBox.warning(self, "No Selection", "Select a row to restore.")
            return
        hist_id = None
        if hasattr(mw, '_history_id_map'):
            if sel < len(mw.coding_form_df):
                val = mw.coding_form_df.iloc[sel].get('_history_id')
                if pd.notna(val):
                    hist_id = str(val)
        if not hist_id:
            QMessageBox.information(self, "Cannot Restore",
                "This row has no linked history entry.")
            return
        import json
        hist_path = HistoryTab.HISTORY_FILE
        try:
            with open(hist_path, 'r', encoding='utf-8') as f:
                entries = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            entries = []
        entry = None
        for e in entries:
            if e.get('id') == hist_id:
                entry = e
                break
        if not entry:
            QMessageBox.warning(self, "Not Found", "History entry not found.")
            return
        pdf_path = entry.get('pdf_path', '')
        pdf_loaded = False
        if pdf_path and os.path.exists(pdf_path):
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                mw.analyze_tab._reset()
                mw.analyze_tab.pdf_path = pdf_path
                mw.analyze_tab.pdf_viewer.load_pdf(pdf_bytes)
                mw.analyze_tab.analyze_btn.setEnabled(True)
                pdf_loaded = True
            except Exception:
                pdf_loaded = False
        if not pdf_loaded and pdf_path:
            ans = QMessageBox.question(self, "PDF Not Found", "Locate the PDF file?")
            if ans == QMessageBox.StandardButton.Yes:
                new_path, _ = QFileDialog.getOpenFileName(self, "Locate PDF", "", "PDF Files (*.pdf)")
                if new_path:
                    with open(new_path, 'rb') as f:
                        pdf_bytes = f.read()
                    mw.analyze_tab._reset()
                    mw.analyze_tab.pdf_path = new_path
                    mw.analyze_tab.pdf_viewer.load_pdf(pdf_bytes)
                    mw.analyze_tab.analyze_btn.setEnabled(True)
        recorded = entry.get('recorded', [])
        field_data = entry.get('field_data', [])
        prompts = entry.get('prompts', [])
        mw.prompts = prompts
        mw.analyze_tab._clear_cards()
        for i, prompt in enumerate(prompts):
            card = FieldCard(i, prompt)
            if i < len(field_data) and isinstance(field_data[i], dict):
                fd = field_data[i]
                card.set_data(fd.get('response', ''), fd.get('source_quote', ''), fd.get('source_page', None))
            if i < len(recorded) and recorded[i]:
                card.toggle_recorded()
            card.source_clicked.connect(mw.analyze_tab._on_src)
            card.record_clicked.connect(mw.analyze_tab._on_rec)
            mw.analyze_tab.card_layout.insertWidget(mw.analyze_tab.card_layout.count() - 1, card)
            mw.analyze_tab.field_cards.append(card)
        mw.current_row_idx = None
        done = sum(1 for r in recorded if r)
        total = len(prompts)
        mw.analyze_tab.recorded_lbl.setText(f"Recorded {done}/{total}")
        mw.tabs.setCurrentIndex(1)
        QMessageBox.information(self, "Done", f"Session restored.")



    def _do_export(self, fmt: str):
        mw = self._mw()
        if mw is None or mw.coding_form_df is None:
            QMessageBox.warning(self, "No Data", "Nothing to export yet.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, f"Export {fmt.upper()}", f"coding_results.{fmt}",
            f"{fmt.upper()} Files (*.{fmt})")
        if not path:
            return
        try:
            if fmt == 'xlsx':
                mw.coding_form_df.to_excel(path, index=False, header=False, engine='openpyxl')
            else:
                mw.coding_form_df.to_csv(path, index=False, header=False)
            QMessageBox.information(self, "Done", f"Saved to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _delete_row(self):
        mw = self._mw()
        if mw is None or mw.coding_form_df is None:
            return

        sel = self.table.currentRow()
        if sel <= 0:
            QMessageBox.information(self, "Info",
                                    "Cannot delete the header row (row 0).\nSelect a data row first.")
            return

        reply = QMessageBox.question(self, "Confirm",
                                     f"Delete row {sel}?\nThis cannot be undone.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Remove from DataFrame
        mw.coding_form_df = mw.coding_form_df.drop(mw.coding_form_df.index[sel])
        mw.coding_form_df = mw.coding_form_df.reset_index(drop=True)
        # Adjust current_row_idx
        if mw.current_row_idx is not None:
            if sel < mw.current_row_idx:
                mw.current_row_idx -= 1
            elif sel == mw.current_row_idx:
                mw.current_row_idx = None

        self.refresh(mw.coding_form_df)

    def _mw(self):
        w = self.parent()
        while w:
            if isinstance(w, MainWindow):
                return w
            w = w.parent()
        return None


# ============================================================
# Main Window
# ============================================================
# Plot Tab
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AIDE Free v1")
        self.resize(1600, 950)

        # Shared state
        self.coding_form_df = None
        self.prompts = []
        self.current_row_idx = None

        # Tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.setup_tab = SetupTab()
        self.tabs.addTab(self.setup_tab, "⚙️  Setup")

        self.analyze_tab = AnalyzeTab()
        self.tabs.addTab(self.analyze_tab, "🔍  Analyze")

        self.export_tab = ExportTab()
        self.tabs.addTab(self.export_tab, "📋  Export")

        self.history_tab = HistoryTab()
        self.tabs.addTab(self.history_tab, "📜  History")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.statusBar().showMessage("Ready")

    def _on_tab_changed(self, idx):
        if idx == 1:  # Analyze
            self.analyze_tab.analyze_btn.setEnabled(
                self.pdf_loaded() and bool(self.prompts)
            )
        elif idx == 2:  # Export
            recorded = sum(1 for c in self.analyze_tab.field_cards if c.is_recorded)
            self.export_tab.refresh(self.coding_form_df, recorded_count=recorded)

    def pdf_loaded(self) -> bool:
        return self.analyze_tab.pdf_viewer.pdf_bytes is not None

    def _open_pdf(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open PDF", "", "PDF Files (*.pdf);;All Files (*)")
        if not path:
            return

        self.statusBar().showMessage(f"Loading {os.path.basename(path)}...")
        with open(path, 'rb') as f:
            pdf_bytes = f.read()

        # Clear old analysis results when opening a new PDF
        self.analyze_tab.pdf_path = path
        self.analyze_tab._reset()
        self.analyze_tab.pdf_viewer.load_pdf(pdf_bytes)
        self.analyze_tab.analyze_btn.setEnabled(bool(self.prompts))

        n = self.analyze_tab.pdf_viewer.total_pages
        self.statusBar().showMessage(f"📄 {os.path.basename(path)} — {n} pages")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Modern palette
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(240, 242, 245))
    p.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(248, 249, 250))
    p.setColor(QPalette.ColorRole.Text, QColor(33, 37, 41))
    p.setColor(QPalette.ColorRole.Button, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Highlight, QColor(52, 152, 219))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(p)

    # Global stylesheet
    app.setStyleSheet("""
    QMainWindow { background: #f0f2f5; }
    QTabWidget::pane { border: 1px solid #ddd; border-radius: 8px; background: #fff; }
    QTabBar::tab {
        background: #e9ecef; border: 1px solid #ddd; border-bottom: none;
        border-radius: 8px 8px 0 0; padding: 10px 24px; margin-right: 2px;
        font-size: 13px; font-weight: 600; color: #555;
    }
    QTabBar::tab:selected {
        background: #fff; color: #3498db; border-bottom: 2px solid #3498db;
    }
    QTabBar::tab:hover:!selected { background: #dfe6e9; }
    QGroupBox {
        border: 1px solid #e0e0e0; border-radius: 8px; margin-top: 12px;
        padding: 16px 12px 12px 12px; font-size: 13px; font-weight: 600;
        background: #fff;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
    QStatusBar { background: #fff; border-top: 1px solid #e0e0e0; font-size: 12px; }
    QScrollArea { border: none; }
    QProgressBar { border: none; border-radius: 4px; background: #ecf0f1; text-align: center; }
    QProgressBar::chunk { background: #3498db; border-radius: 4px; }
    """)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
