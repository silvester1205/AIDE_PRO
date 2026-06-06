"""
AIDE Desktop — PyQt6 desktop application for AI-Assisted Data Extraction.
Two-tab layout: Setup (API + coding form) → Analyze (PDF viewer + field cards).
"""
import sys
import os
import io
import json
import time
import base64
import fitz
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QLabel, QPushButton, QTextEdit, QFrame, QFileDialog,
    QLineEdit, QMessageBox, QProgressBar, QComboBox, QTabWidget,
    QStatusBar, QGroupBox, QFormLayout, QMenuBar, QMenu, QDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QInputDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint, QSettings, QRect
from PyQt6.QtGui import QPixmap, QImage, QPalette, QColor, QAction
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.pdf_utils import (
    render_page_as_base64, search_text_for_highlight,
    get_pdf_page_count, extract_text_from_pdf
)
from utils.llm_client import analyze_with_llm, test_connection, fetch_available_models, generate_template


# ============================================================
# Workers
# ============================================================

class AnalyzeWorker(QThread):
    finished = pyqtSignal(bool, object)
    def __init__(self, config, template_fields, pdf_bytes):
        super().__init__()
        self.config = config
        self.template_fields = template_fields
        self.pdf_bytes = pdf_bytes
    def run(self):
        pdf_text = extract_text_from_pdf(self.pdf_bytes)
        success, result = analyze_with_llm(self.config, self.template_fields, pdf_text)
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

class GenerateTemplateWorker(QThread):
    finished = pyqtSignal(bool, object)
    def __init__(self, config, topic):
        super().__init__()
        self.config = config
        self.topic = topic
    def run(self):
        success, result = generate_template(self.config, self.topic)
        self.finished.emit(success, result)


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
    """Field card for displaying a single extracted field.

    For study-level: shows one editor + one Source button.
    For arm-level: shows per-arm editors + per-arm Source buttons.
    """
    source_clicked = pyqtSignal(int, str, object, int)  # card_index, quote, page, arm_index(-1=study)

    def __init__(self, index: int, field_def: dict, parent=None):
        super().__init__(parent)
        self.index = index
        self.field_def = field_def
        self.name = field_def.get('name', f'Field {index+1}')
        self.level = field_def.get('level', 'study')
        self._arm_data = []  # list of dicts: {response, source_quote, source_page}
        self._arm_widgets = []  # list of dicts: {editor, src_btn, ...}
        self._is_recorded = False
        self._is_active = -1  # which arm is active (-1 = none)
        self._setup()

    def _setup(self):
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        # Header
        level_tag = "📄 study" if self.level == 'study' else "🫂 arm"
        self.header = QLabel(f"<b>{self.name}</b>  <span style='color:#888;font-size:11px;'>{level_tag}</span>")
        layout.addWidget(self.header)

        # Prompt
        prompt = self.field_def.get('prompt', self.name)
        prompt_txt = prompt[:160] + "…" if len(prompt) > 160 else prompt
        pl = QLabel(f"<span style='color:#666;font-size:13px;'>{prompt_txt}</span>")
        pl.setWordWrap(True)
        layout.addWidget(pl)

        # Arm-level fields: per-arm editors + source buttons
        if self.level == 'arm':
            self._arm_layout = QVBoxLayout()
            self._arm_layout.setSpacing(4)
            layout.addLayout(self._arm_layout)
            self._no_data_label = QLabel("(arm-level data will appear here)")
            self._no_data_label.setStyleSheet("color:#999;font-size:12px;padding:4px;")
            self._arm_layout.addWidget(self._no_data_label)
        else:
            # Study-level: single editor
            self.editor = QTextEdit()
            self.editor.setPlaceholderText("LLM extracted data...")
            self.editor.setMinimumHeight(80)
            self.editor.setMaximumHeight(150)
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
                self.index, getattr(self, 'source_quote', ''), getattr(self, 'source_page', None), -1))
            bl.addWidget(self.src_btn)
            bl.addStretch()
            layout.addLayout(bl)

        self._restyle()

    def _restyle(self):
        self.setStyleSheet(
            "FieldCard{border-radius:8px;margin:4px;padding:2px;"
            "border:3px solid #e0e0e0;background:#fff;}")

    def set_data(self, response, source_quote, source_page, arm_index=-1):
        """Set data for a field.

        For study-level (arm_index=-1): sets the single editor.
        For arm-level (arm_index>=0): adds/updates an arm item.
        """
        if self.level == 'study':
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
        else:
            if arm_index < 0:
                return
            while len(self._arm_data) <= arm_index:
                self._arm_data.append({'response': '', 'source_quote': '', 'source_page': None})
                self._add_arm_widget(len(self._arm_data) - 1)
            self._arm_data[arm_index]['response'] = response or ''
            self._arm_data[arm_index]['source_quote'] = source_quote or ''
            self._arm_data[arm_index]['source_page'] = source_page
            if arm_index < len(self._arm_widgets):
                w = self._arm_widgets[arm_index]
                w['editor'].setPlainText(response or '')
                if source_quote:
                    q = source_quote[:80] + "…" if len(source_quote) > 80 else source_quote
                    txt = f"📝 {q}"
                    if source_page:
                        txt += f"  |  📄 p.{source_page}"
                    w['src_info'].setText(txt)
                    w['src_info'].show()
            if self._no_data_label and self._no_data_label.isVisible():
                self._no_data_label.hide()

    def _add_arm_widget(self, arm_idx: int):
        if not hasattr(self, '_arm_layout'):
            return
        if self._no_data_label and self._no_data_label.isVisible():
            self._no_data_label.hide()

        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 2, 0, 2)
        row_layout.setSpacing(2)

        hdr = QHBoxLayout()
        label = QLabel(f"<b>Arm {arm_idx + 1}</b>")
        label.setStyleSheet("color:#555;font-size:12px;")
        hdr.addWidget(label)

        src_btn = QPushButton("📍 Source")
        src_btn.setFixedWidth(70)
        src_btn.setStyleSheet(
            "QPushButton{border-radius:4px;padding:3px 8px;font-size:11px;"
            "background:#f8f9fa;color:#555;border:1px solid #dee2e6;}"
            "QPushButton:hover{background:#e9ecef;}")
        src_btn.clicked.connect(lambda checked, ai=arm_idx: self.source_clicked.emit(
            self.index,
            self._arm_data[ai].get('source_quote', '') if ai < len(self._arm_data) else '',
            self._arm_data[ai].get('source_page', None) if ai < len(self._arm_data) else None,
            ai))
        hdr.addWidget(src_btn)

        src_info = QLabel("")
        src_info.setStyleSheet("color:#888;font-size:11px;")
        src_info.hide()
        hdr.addWidget(src_info)
        hdr.addStretch()
        row_layout.addLayout(hdr)

        editor = QTextEdit()
        editor.setPlaceholderText(f"Arm {arm_idx + 1} value...")
        editor.setMaximumHeight(60)
        editor.setStyleSheet(
            "QTextEdit{border:1px solid #ddd;border-radius:4px;"
            "padding:4px 6px;font-size:13px;background:#fff;}")
        row_layout.addWidget(editor)

        self._arm_layout.addWidget(row)
        self._arm_widgets.append({
            'editor': editor,
            'src_btn': src_btn,
            'src_info': src_info,
            'row': row
        })

    def set_active(self, active_idx: int):
        self._is_active = active_idx

    def get_text(self, arm_index=-1) -> str:
        if self.level == 'study':
            return self.editor.toPlainText()
        else:
            texts = []
            for w in self._arm_widgets:
                t = w['editor'].toPlainText().strip()
                if t:
                    texts.append(t)
            return ' / '.join(texts) if texts else ''

    def get_arm_values(self) -> list:
        if self.level != 'arm':
            return []
        return [w['editor'].toPlainText().strip() for w in self._arm_widgets]


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
        self.page_words = {}
        self.highlight_page = None
        self._select_mode = False
        self._search_matches = []
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
        words = self.page_words.get(page_num, [])
        if not words:
            return

        scale = 1.0 / self.ZOOM
        x0, y0 = rect.left() * scale, rect.top() * scale
        x1, y1 = rect.right() * scale, rect.bottom() * scale

        selected = []
        for wx0, wy0, wx1, wy1, text in words:
            if not (wx1 < x0 or wx0 > x1 or wy1 < y0 or wy0 > y1):
                selected.append((wy0, wx0, text))

        if not selected:
            return

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
        if idx < 0 or idx >= len(self._search_matches):
            return

        pn, rects = self._search_matches[idx]
        self._search_idx = idx
        self.search_count.setText(f"{idx+1}/{len(self._search_matches)}")

        b64 = render_page_as_base64(self.pdf_bytes, pn, highlights=rects, zoom=self.ZOOM)
        if b64 and pn - 1 < len(self.page_widgets):
            pm = self._b64_to_pixmap(b64)
            self.page_widgets[pn - 1].setPixmap(pm)

        for mpn, _ in self._search_matches:
            if mpn != pn and mpn in self.page_pixmaps and mpn - 1 < len(self.page_widgets):
                self.page_widgets[mpn - 1].setPixmap(self.page_pixmaps[mpn])

        if not hasattr(self, '_last_rects') or not self._last_rects:
            for ppn in range(1, self.total_pages + 1):
                if ppn != pn and ppn in self.page_pixmaps and ppn - 1 < len(self.page_widgets):
                    self.page_widgets[ppn - 1].setPixmap(self.page_pixmaps[ppn])

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

        b64 = render_page_as_base64(self.pdf_bytes, page_num, highlights=rects, zoom=self.ZOOM)
        if b64 and page_num - 1 < len(self.page_widgets):
            self.page_widgets[page_num - 1].setPixmap(self._b64_to_pixmap(b64))

        for pn in range(1, self.total_pages + 1):
            if pn != page_num and pn in self.page_pixmaps and pn - 1 < len(self.page_widgets):
                self.page_widgets[pn - 1].setPixmap(self.page_pixmaps[pn])

        if page_num - 1 < len(self.page_widgets):
            widget = self.page_widgets[page_num - 1]
            self.scroll.ensureWidgetVisible(widget, 0, 50)

            if rects and hasattr(widget, 'pixmap') and widget.pixmap():
                pm = widget.pixmap()
                pm_h = pm.height()
                widget_h = self.scroll.viewport().height()
                mid_y = sum(r[1] + r[3] for r in rects) / (2 * len(rects))
                mid_px = mid_y * self.ZOOM
                widget_y = widget.mapTo(self.container, QPoint(0, 0)).y()
                target_y = widget_y + mid_px - widget_h // 3
                sb = self.scroll.verticalScrollBar()
                sb.setValue(max(0, int(target_y)))

    def _clear_highlights(self):
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
        self._generated_fields = []
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

        # -- Template Section --
        tpl_grp = QGroupBox("Extraction Template")
        tpl_layout = QVBoxLayout(tpl_grp)

        # Research topic + generate
        topic_row = QHBoxLayout()
        self.topic_input = QTextEdit()
        self.topic_input.setPlaceholderText("Paste research topic or abstract here...\n\nExample: Effects of antimicrobial stewardship on ICU length of stay and mortality: a multicenter RCT")
        self.topic_input.setMaximumHeight(80)
        self.topic_input.setStyleSheet(
            "QTextEdit{border:1px solid #ddd;border-radius:6px;padding:6px;font-size:13px;}")
        topic_row.addWidget(self.topic_input, 1)

        gen_col = QVBoxLayout()
        self.gen_btn = QPushButton("🤖 Generate Template")
        self.gen_btn.setStyleSheet(
            "QPushButton{background:#9b59b6;color:#fff;border:none;border-radius:6px;"
            "padding:10px 20px;font-size:13px;font-weight:600;}"
            "QPushButton:hover{background:#8e44ad;}"
            "QPushButton:disabled{background:#bdc3c7;color:#fff;}")
        self.gen_btn.clicked.connect(self._generate)
        gen_col.addWidget(self.gen_btn)

        self.gen_status = QLabel("")
        self.gen_status.setStyleSheet("font-size:11px;color:#888;")
        gen_col.addWidget(self.gen_status)
        gen_col.addStretch()
        topic_row.addLayout(gen_col)
        tpl_layout.addLayout(topic_row)

        # Field definitions table
        tpl_layout.addWidget(QLabel("Extraction Fields (editable):"))

        self.field_table = QTableWidget()
        self.field_table.setColumnCount(4)
        self.field_table.setHorizontalHeaderLabels(["Field Name", "Prompt / Instruction", "Type", "Level"])
        self.field_table.setAlternatingRowColors(True)
        self.field_table.horizontalHeader().setStretchLastSection(False)
        self.field_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.field_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.field_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.field_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.field_table.verticalHeader().hide()
        self.field_table.setMinimumHeight(180)
        self.field_table.setStyleSheet(
            "QTableWidget{border:1px solid #ddd;border-radius:4px;gridline-color:#eee;}"
            "QHeaderView::section{background:#f5f5f5;padding:4px 8px;border:1px solid #ddd;font-weight:bold;font-size:12px;}")
        tpl_layout.addWidget(self.field_table, 1)

        # Table action buttons
        tbl_actions = QHBoxLayout()
        self.add_field_btn = QPushButton("➕ Add Field")
        self.add_field_btn.setStyleSheet(
            "QPushButton{border-radius:4px;padding:5px 14px;font-size:12px;"
            "background:#3498db;color:#fff;border:none;}"
            "QPushButton:hover{background:#2980b9;}")
        self.add_field_btn.clicked.connect(self._add_field_row)
        tbl_actions.addWidget(self.add_field_btn)

        self.del_field_btn = QPushButton("🗑️ Delete Selected")
        self.del_field_btn.setStyleSheet(
            "QPushButton{border-radius:4px;padding:5px 14px;font-size:12px;"
            "background:#e74c3c;color:#fff;border:none;}"
            "QPushButton:hover{background:#c0392b;}")
        self.del_field_btn.clicked.connect(self._delete_field_row)
        tbl_actions.addWidget(self.del_field_btn)

        tbl_actions.addStretch()

        self.import_csv_btn = QPushButton("📂 Import CSV")
        self.import_csv_btn.setStyleSheet(
            "QPushButton{border-radius:4px;padding:5px 14px;font-size:12px;"
            "background:#f8f9fa;color:#555;border:1px solid #ccc;}"
            "QPushButton:hover{background:#e9ecef;}")
        self.import_csv_btn.clicked.connect(self._import_csv)
        tbl_actions.addWidget(self.import_csv_btn)

        self.export_csv_btn = QPushButton("📤 Export CSV")
        self.export_csv_btn.setStyleSheet(
            "QPushButton{border-radius:4px;padding:5px 14px;font-size:12px;"
            "background:#f8f9fa;color:#555;border:1px solid #ccc;}"
            "QPushButton:hover{background:#e9ecef;}")
        self.export_csv_btn.clicked.connect(self._export_csv)
        tbl_actions.addWidget(self.export_csv_btn)

        tpl_layout.addLayout(tbl_actions)

        # Apply button
        apply_row = QHBoxLayout()
        self.apply_btn = QPushButton("✓ Apply Template")
        self.apply_btn.setStyleSheet(
            "QPushButton{background:#2ecc71;color:#fff;border:none;border-radius:8px;"
            "padding:12px 28px;font-size:14px;font-weight:600;}"
            "QPushButton:hover{background:#27ae60;}"
            "QPushButton:disabled{background:#bdc3c7;color:#fff;}")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_template)
        apply_row.addWidget(self.apply_btn)

        self.tpl_status = QLabel("")
        self.tpl_status.setWordWrap(True)
        self.tpl_status.setStyleSheet("color:#888;font-size:12px;")
        apply_row.addWidget(self.tpl_status)
        apply_row.addStretch()
        tpl_layout.addLayout(apply_row)

        layout.addWidget(tpl_grp, 1)

        # Author info
        author = QLabel(
            '<div style="text-align:center;color:#999;font-size:11px;padding:8px 0;">'
            'AIDE Pro · 系统综述数据提取工具<br>'
            '作者小红书：94373279426</div>')
        layout.addWidget(author)

    # ── helpers ──
    def get_config(self) -> dict:
        return {
            'endpoint': self.endpoint.text().strip(),
            'api_key': self.apikey.text().strip(),
            'model': self.model.currentText().strip()
        }

    def get_fields(self) -> list:
        fields = []
        for r in range(self.field_table.rowCount()):
            name_item = self.field_table.item(r, 0)
            prompt_item = self.field_table.item(r, 1)
            if name_item is None or not name_item.text().strip():
                continue
            name = name_item.text().strip()
            prompt = prompt_item.text().strip() if prompt_item else name
            type_widget = self.field_table.cellWidget(r, 2)
            ftype = type_widget.currentText() if type_widget else 'text'
            level_widget = self.field_table.cellWidget(r, 3)
            level = level_widget.currentText() if level_widget else 'study'
            fields.append({'name': name, 'prompt': prompt, 'type': ftype, 'level': level})
        return fields

    def _populate_table(self, fields: list):
        self.field_table.setRowCount(0)
        for f in fields:
            self._add_field_row(f)

    def _add_field_row(self, field_def=None):
        r = self.field_table.rowCount()
        self.field_table.insertRow(r)
        name = field_def.get('name', '') if field_def else ''
        prompt = field_def.get('prompt', name) if field_def else ''
        ftype = field_def.get('type', 'text') if field_def else 'text'
        level = field_def.get('level', 'study') if field_def else 'study'

        self.field_table.setItem(r, 0, QTableWidgetItem(name))
        self.field_table.setItem(r, 1, QTableWidgetItem(prompt))

        tc = QComboBox()
        tc.addItems(['text', 'integer', 'float', 'boolean', 'categorical'])
        idx = tc.findText(ftype)
        if idx >= 0:
            tc.setCurrentIndex(idx)
        tc.setStyleSheet("QComboBox{font-size:11px;padding:2px;}")
        self.field_table.setCellWidget(r, 2, tc)

        lc = QComboBox()
        lc.addItems(['study', 'arm'])
        idx = lc.findText(level)
        if idx >= 0:
            lc.setCurrentIndex(idx)
        lc.setStyleSheet("QComboBox{font-size:11px;padding:2px;}")
        self.field_table.setCellWidget(r, 3, lc)

        self.apply_btn.setEnabled(True)
        self.tpl_status.setText(f"{self.field_table.rowCount()} field(s) defined — click Apply")

    def _delete_field_row(self):
        sel = self.field_table.currentRow()
        if sel >= 0:
            self.field_table.removeRow(sel)
            n = self.field_table.rowCount()
            if n == 0:
                self.apply_btn.setEnabled(False)
                self.tpl_status.setText("No fields defined")
            else:
                self.tpl_status.setText(f"{n} field(s) defined — click Apply")

    # ── API actions ──
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

    def _generate(self):
        topic = self.topic_input.toPlainText().strip()
        if not topic:
            QMessageBox.warning(self, "Empty", "Enter a research topic or abstract first.")
            return
        config = self.get_config()
        if not config['api_key']:
            QMessageBox.warning(self, "Missing", "Enter API key first.")
            return
        self.gen_btn.setEnabled(False)
        self.gen_status.setText("⏳ Generating template...")
        self.gen_status.setStyleSheet("color:#333;")
        self.w = GenerateTemplateWorker(config, topic)
        self.w.finished.connect(self._generate_done)
        self.w.start()

    def _generate_done(self, ok, result):
        self.gen_btn.setEnabled(True)
        if not ok:
            self.gen_status.setText(f"❌ {result}")
            self.gen_status.setStyleSheet("color:#dc3545;")
            return
        fields = result
        self._generated_fields = fields
        self._populate_table(fields)
        self.gen_status.setText(f"✅ Generated {len(fields)} fields — review and edit below")
        self.gen_status.setStyleSheet("color:#28a745;")

    def _apply_template(self):
        fields = self.get_fields()
        if not fields:
            QMessageBox.warning(self, "Empty", "No fields defined.")
            return
        mw = self._mw()
        if mw:
            mw.template_fields = fields
            mw.prompts = [f.get('prompt', f.get('name', '')) for f in fields]
            field_names = [f['name'] for f in fields]
            mw.coding_form_df = pd.DataFrame([field_names])
        self.tpl_status.setText(f"✅ Applied template with {len(fields)} fields")
        self.tpl_status.setStyleSheet("color:#28a745;")

    # ── CSV import/export ──
    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Template CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
            fields = []
            for _, row in df.iterrows():
                fields.append({
                    'name': row.get('name', row.get('Name', '')),
                    'prompt': row.get('prompt', row.get('Prompt', row.get('name', ''))),
                    'type': row.get('type', row.get('Type', 'text')),
                    'level': row.get('level', row.get('Level', 'study')),
                })
            if fields:
                self._populate_table(fields)
                self.gen_status.setText(f"✅ Imported {len(fields)} fields from CSV")
                self.gen_status.setStyleSheet("color:#28a745;")
            else:
                QMessageBox.warning(self, "Empty", "No fields found in CSV.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to import CSV:\n{e}")

    def _export_csv(self):
        fields = self.get_fields()
        if not fields:
            QMessageBox.warning(self, "Empty", "No fields to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Template CSV", "template_fields.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            df = pd.DataFrame(fields)
            df.to_csv(path, index=False, encoding='utf-8-sig')
            QMessageBox.information(self, "Done", f"Exported {len(fields)} fields to {path}")
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

        split = QSplitter(Qt.Orientation.Horizontal)
        self.pdf_viewer = PdfViewer()
        split.addWidget(self.pdf_viewer)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 0)

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

        self.record_all_btn = QPushButton("✅ Record All")
        self.record_all_btn.setEnabled(False)
        self.record_all_btn.setStyleSheet(
            "QPushButton{border-radius:8px;padding:10px 18px;font-size:13px;font-weight:600;"
            "border:none;background:#2ecc71;color:#fff;}"
            "QPushButton:hover{background:#27ae60;}"
            "QPushButton:disabled{background:#bdc3c7;color:#fff;}")
        self.record_all_btn.clicked.connect(self._on_record_all)
        top_layout.addWidget(self.record_all_btn)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-size:12px;")
        top_layout.addWidget(self.status)
        top_layout.addStretch()
        right_layout.addWidget(top)

        self.progress = QProgressBar()
        self.progress.hide()
        right_layout.addWidget(self.progress)

        card_scroll = QScrollArea()
        card_scroll.setWidgetResizable(True)
        self.card_container = QWidget()
        self.card_layout = QVBoxLayout(self.card_container)
        self.card_layout.setContentsMargins(4, 4, 4, 8)
        self.card_layout.setSpacing(4)
        self.card_layout.addStretch()
        card_scroll.setWidget(self.card_container)
        right_layout.addWidget(card_scroll, 1)

        self.recorded_lbl = QLabel("Ready")
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
        if not mw or not mw.template_fields:
            QMessageBox.warning(self, "Missing", "Apply a template in Setup first.")
            return
        config = mw.setup_tab.get_config()
        if not config['api_key']:
            QMessageBox.warning(self, "Missing", "Enter API key in Setup first.")
            return
        if not self.pdf_viewer.pdf_bytes:
            QMessageBox.warning(self, "Missing", "Open a PDF via File menu first.")
            return

        # Daily limit check
        today = str(date.today())
        usage = QSettings("AIDE", "Usage")
        usage_date = usage.value("trial_date", "")
        count = int(usage.value("trial_count", "0"))
        if usage_date != today:
            usage.setValue("trial_date", today)
            usage.setValue("trial_count", "0")
            count = 0
        if count >= 3:
            QMessageBox.warning(self, "Limit Reached",
                "今日已分析3篇文献（免费版限制）。\n请明天再试，或购买Pro版。")
            return

        self.analyze_btn.setEnabled(False)
        self.progress.show()
        self.progress.setRange(0, 0)
        self.status.setText("Analyzing...")

        self._worker = AnalyzeWorker(config, mw.template_fields, self.pdf_viewer.pdf_bytes)
        self._worker.finished.connect(self._on_result)
        self._worker.start()

    def _on_result(self, ok, result):
        self.progress.hide()
        self.analyze_btn.setEnabled(True)

        if not ok:
            self.status.setText(f"❌ {result}")
            return

        timing = result.pop('_timing', None)

        self._clear_cards()
        mw = self._mw()
        template_fields = mw.template_fields if mw else []

        for i, (key, data) in enumerate(result.items()):
            field_def = template_fields[i] if i < len(template_fields) else {'name': key, 'level': 'study', 'prompt': ''}
            card = FieldCard(i, field_def)
            level = data.get('level', 'study')
            response = data.get('response', '')

            if level == 'arm' and isinstance(response, list):
                arm_sources = data.get('arm_sources', [])
                arm_pages = data.get('arm_pages', [])
                for ai in range(len(response)):
                    arm_resp = response[ai] if ai < len(response) else ''
                    arm_src = arm_sources[ai] if ai < len(arm_sources) else data.get('source', '')
                    arm_page = arm_pages[ai] if ai < len(arm_pages) else data.get('source_page', None)
                    card.set_data(arm_resp, arm_src, arm_page, ai)
            else:
                card.set_data(
                    response if not isinstance(response, list) else ', '.join(str(v) for v in response),
                    data.get('source_quote', ''),
                    data.get('source_page', None))

            card.source_clicked.connect(self._on_src)
            self.card_layout.insertWidget(self.card_layout.count() - 1, card)
            self.field_cards.append(card)

        # Add data rows to coding form (one per arm)
        if mw and mw.coding_form_df is not None and mw.template_fields:
            n_arms = 1
            for key, data in result.items():
                if key == '_timing':
                    continue
                if data.get('level') == 'arm' and isinstance(data.get('response'), list):
                    n_arms = max(n_arms, len(data['response']))

            for ai in range(n_arms):
                row_data = {}
                for fi, f in enumerate(mw.template_fields):
                    fk = f"field_{fi+1}"
                    fd = result.get(fk, {})
                    if fd.get('level') == 'arm' and isinstance(fd.get('response'), list):
                        arms = fd['response']
                        val = arms[ai] if ai < len(arms) else ''
                    else:
                        val = fd.get('response', '') if ai == 0 else ''
                        if isinstance(val, list):
                            val = ', '.join(str(v) for v in val)
                    row_data[fi] = str(val) if val is not None else ''
                nr = pd.DataFrame([row_data])
                mw.coding_form_df = pd.concat([mw.coding_form_df, nr], ignore_index=True)
            mw.current_row_idx = len(mw.coding_form_df) - 1

        n = len(result)
        self.recorded_lbl.setText(f"📋 {n} fields extracted — review, then click Record All")
        self.record_all_btn.setEnabled(True)

        # Increment daily count
        try:
            today = str(date.today())
            usage = QSettings("AIDE", "Usage")
            usage_date = usage.value("trial_date", "")
            count = int(usage.value("trial_count", "0"))
            if usage_date == today:
                usage.setValue("trial_count", str(count + 1))
                mw.statusBar().showMessage(f"今日已分析 {count+1}/3 篇")
        except Exception:
            pass

        if timing:
            self.status.setText(
                f"✅ {n} fields in {timing['total']}s "
                f"(prompt {timing['build']}s | LLM {timing['llm']}s | parse {timing['parse']}s, "
                f"in {timing['input_chars']} chars → out {timing['output_chars']} chars)")
        else:
            self.status.setText(f"✅ {n} fields — review below")
        self.status.setStyleSheet("color:#28a745;")

    def _on_src(self, idx, quote, page, arm_index=-1):
        """Handle source click for both study and arm fields."""
        try:
            self.status.setText("🔍 Searching...")
            self.pdf_viewer._clear_highlights()
            if 0 <= self._active_src < len(self.field_cards):
                self.field_cards[self._active_src].set_active(-1)
            self._active_src = idx
            if 0 <= idx < len(self.field_cards):
                self.field_cards[idx].set_active(arm_index)

            if page is not None:
                try:
                    page = int(page)
                except (ValueError, TypeError):
                    page = None

            if self.pdf_viewer.pdf_bytes and quote:
                self.status.setText(f"🔍 {quote[:60]}...")
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

    def _on_record_all(self):
        """Record all field values to the coding form DataFrame."""
        mw = self._mw()
        if not mw or mw.coding_form_df is None or mw.current_row_idx is None:
            return

        arm_counts = []
        for card in self.field_cards:
            if card.level == 'arm':
                vals = card.get_arm_values()
                arm_counts.append(len(vals))
        n_arms = max(arm_counts) if arm_counts else 1

        row_start = mw.current_row_idx - (n_arms - 1)
        if row_start < 1:
            row_start = 1

        for ai in range(n_arms):
            row_idx = row_start + ai
            if row_idx >= len(mw.coding_form_df):
                break
            fi = 0
            for card in self.field_cards:
                if card.level == 'arm':
                    vals = card.get_arm_values()
                    val = vals[ai] if ai < len(vals) else ''
                else:
                    val = card.get_text() if ai == 0 else ''
                    if ai > 0:
                        prev_row = mw.coding_form_df.iloc[row_start]
                        if fi < len(mw.coding_form_df.columns):
                            val = prev_row.iloc[fi] if fi < len(prev_row) else ''
                if fi < len(mw.coding_form_df.columns):
                    mw.coding_form_df.at[row_idx, mw.coding_form_df.columns[fi]] = str(val) if val else ''
                fi += 1

        self.record_all_btn.setEnabled(False)
        self.recorded_lbl.setText(f"✅ Recorded {len(self.field_cards)} fields ({n_arms} row(s))")
        self.recorded_lbl.setStyleSheet("color:#28a745;font-weight:bold;")
        self.status.setText("🎉 All recorded!")
        self.status.setStyleSheet("color:#28a745;")

        try:
            field_data = []
            for card in self.field_cards:
                if card.level == 'arm':
                    field_data.append({
                        'response': card.get_text(),
                        'arm_values': card.get_arm_values(),
                        'level': 'arm',
                    })
                else:
                    field_data.append({
                        'response': card.get_text(),
                        'level': 'study',
                    })
            HistoryTab.save_entry(
                getattr(self, 'pdf_path', '') or '',
                mw.template_fields, mw.coding_form_df,
                mw.current_row_idx, field_data)
            mw.history_tab.refresh()
        except Exception:
            pass

    def _clear_cards(self):
        while self.card_layout.count() > 1:
            item = self.card_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.field_cards.clear()
        self._active_src = -1

    def _reset(self):
        self._clear_cards()
        self.status.setText("")
        self.recorded_lbl.setText("Ready")
        self.record_all_btn.setEnabled(False)
        self.progress.hide()
        mw = self._mw()
        if mw:
            mw.current_row_idx = None


# ============================================================
# History Tab
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

        hint = QLabel("Select entries and click 'Import to Table' to add their data to the export table.\nThen go to Export tab → select a row → click 'Restore Session' to reload PDF and field cards.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888;font-size:12px;padding:0 0 8px 0;")
        layout.addWidget(hint)

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

        bl = QHBoxLayout()
        self.import_btn = QPushButton("📥 Import to Table")
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

        self.status = QLabel("")
        self.status.setStyleSheet("color:#888;font-size:12px;padding:4px 0;")
        layout.addWidget(self.status)

        self.refresh()

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
    def save_entry(pdf_path: str, template_spec, df, row_idx: int, field_data: list):
        """Save a history entry.
        template_spec: list of template field dicts (new format) or list of prompts (old format)
        """
        if df is not None:
            df_json = df.to_json(orient='split', force_ascii=False)
        else:
            df_json = None

        # Detect: if template_spec is list of dicts with 'level', it's new format
        is_new_format = template_spec and isinstance(template_spec, list) and len(template_spec) > 0 and isinstance(template_spec[0], dict)

        entry = {
            'id': str(int(time.time())),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pdf_path': pdf_path,
            'pdf_name': os.path.basename(pdf_path) if pdf_path else '',
            'template_fields': template_spec if is_new_format else [],
            'prompts': template_spec if not is_new_format else [f.get('prompt', f.get('name', '')) for f in template_spec],
            'coding_form': df_json,
            'row_idx': row_idx,
            'field_data': field_data,
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

        entries = [e for e in entries if e.get('pdf_path') != pdf_path]
        entries.append(entry)
        entries = entries[-50:]

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    def refresh(self):
        entries = self._load_all()
        self.table.setRowCount(len(entries))
        for i, e in enumerate(reversed(entries)):
            self.table.setItem(i, 0, QTableWidgetItem(e.get('study', '')))
            self.table.setItem(i, 1, QTableWidgetItem(e.get('timestamp', '')))
            n_fields = len(e.get('prompts', []))
            self.table.setItem(i, 2, QTableWidgetItem(str(n_fields)))
            self.table.setItem(i, 3, QTableWidgetItem(str(n_fields)))
            self.table.setItem(i, 4, QTableWidgetItem(e.get('pdf_name', '')))
        self.status.setText(f"{len(entries)} session(s) saved")

    def _import_to_table(self):
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

        hdr = QHBoxLayout()
        title = QLabel("<h2>📋 Final Coding Form</h2>")
        hdr.addWidget(title)
        hdr.addStretch()

        self.export_xlsx = QPushButton("📥 Export Excel")
        self.export_xlsx.setStyleSheet(
            "QPushButton{border-radius:8px;padding:10px 22px;font-size:13px;font-weight:600;"
            "border:none;background:#2ecc71;color:#fff;}"
            "QPushButton:hover{background:#27ae60;}")
        self.export_xlsx.clicked.connect(lambda: self._do_export("xlsx"))
        hdr.addWidget(self.export_xlsx)

        self.export_wide = QPushButton("📤 Export Wide")
        self.export_wide.setStyleSheet(
            "QPushButton{border-radius:8px;padding:10px 22px;font-size:13px;font-weight:600;"
            "border:none;background:#9b59b6;color:#fff;}"
            "QPushButton:hover{background:#8e44ad;}")
        self.export_wide.clicked.connect(self._export_wide)
        hdr.addWidget(self.export_wide)

        layout.addLayout(hdr)

        self.stats = QLabel("No data yet. Complete the Setup and Analyze steps first.")
        self.stats.setStyleSheet("color:#888;font-size:12px;padding:8px 0;")
        layout.addWidget(self.stats)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setStyleSheet(
            "QTableWidget{border:1px solid #ddd;border-radius:4px;gridline-color:#eee;}"
            "QHeaderView::section{background:#f5f5f5;padding:6px 10px;"
            "border:1px solid #ddd;font-weight:bold;}")
        layout.addWidget(self.table, 1)

        btn_row = QHBoxLayout()

        restore_btn = QPushButton("↩️ Restore Session")
        restore_btn.setStyleSheet(
            "QPushButton{background:#3498db;color:#fff;border:none;"
            "border-radius:6px;padding:8px 16px;font-size:12px;font-weight:600;}"
            "QPushButton:hover{background:#2980b9;}")
        restore_btn.clicked.connect(self._restore_session)
        btn_row.addWidget(restore_btn)

        btn_row.addStretch()

        self.delete_btn = QPushButton("🗑️ Delete Selected Row")
        self.delete_btn.setStyleSheet(
            "QPushButton{background:#e74c3c;color:#fff;border:none;"
            "border-radius:6px;padding:8px 16px;font-size:12px;font-weight:600;}"
            "QPushButton:hover{background:#c0392b;}")
        self.delete_btn.clicked.connect(self._delete_row)
        btn_row.addWidget(self.delete_btn)

        layout.addLayout(btn_row)

    def refresh(self, df, recorded_count=None):
        if df is None or len(df) == 0:
            self.stats.setText("No data yet.")
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        rows = len(df)
        show_df = df
        show_cols = len(show_df.columns)
        data_rows = rows - 1

        if recorded_count is not None:
            self.stats.setText(
                f"📊 <b>{data_rows}</b> studies extracted, "
                f"<b>{recorded_count}</b> fields recorded in current study | "
                f"Total: <b>{rows}</b> rows × <b>{show_cols}</b> columns")
        else:
            self.stats.setText(
                f"📊 <b>{data_rows}</b> studies | <b>{rows}</b> rows × <b>{show_cols}</b> columns")
        self.stats.setStyleSheet("color:#333;font-size:12px;padding:8px 0;")

        self.table.setRowCount(rows)
        self.table.setColumnCount(show_cols)
        headers = []
        for c in range(show_cols):
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

    def _restore_session(self):
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
            hist_id = mw._history_id_map.get(sel)
        if not hist_id:
            QMessageBox.information(self, "Cannot Restore",
                "This row has no linked history entry.")
            return
        try:
            with open(HistoryTab.HISTORY_FILE, 'r', encoding='utf-8') as f:
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
        field_data = entry.get('field_data', [])
        prompts = entry.get('prompts', [])
        mw.prompts = prompts
        # Try to restore template_fields if available
        template_fields = entry.get('template_fields', None)
        if template_fields:
            mw.template_fields = template_fields
        mw.analyze_tab._clear_cards()
        for i, prompt in enumerate(prompts):
            field_def = template_fields[i] if template_fields and i < len(template_fields) else {'name': prompt, 'level': 'study', 'prompt': prompt}
            card = FieldCard(i, field_def)
            if i < len(field_data) and isinstance(field_data[i], dict):
                fd = field_data[i]
                card.set_data(fd.get('response', ''), fd.get('source_quote', ''), fd.get('source_page', None))
            card.source_clicked.connect(mw.analyze_tab._on_src)
            mw.analyze_tab.card_layout.insertWidget(mw.analyze_tab.card_layout.count() - 1, card)
            mw.analyze_tab.field_cards.append(card)
        mw.current_row_idx = None
        total = len(prompts)
        mw.analyze_tab.recorded_lbl.setText(f"Restored {total} fields")
        mw.tabs.setCurrentIndex(1)

    def _do_export(self, fmt: str):
        mw = self._mw()
        if mw is None or mw.coding_form_df is None:
            QMessageBox.warning(self, "No Data", "Nothing to export yet.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, f"Export Data", f"coding_results.{fmt}",
            "Excel Files (*.xlsx);;CSV Files (*.csv)")
        if not path:
            return
        try:
            actual_fmt = 'csv' if path.endswith('.csv') else 'xlsx'
            if actual_fmt == 'xlsx':
                mw.coding_form_df.to_excel(path, index=False, header=False, engine='openpyxl')
            else:
                mw.coding_form_df.to_csv(path, index=False, header=False, encoding='utf-8-sig')
            QMessageBox.information(self, "Done", f"Saved to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _export_wide(self):
        """Export wide format: one row per study (pivot on arm fields)."""
        mw = self._mw()
        if mw is None or mw.coding_form_df is None or len(mw.coding_form_df) < 2:
            QMessageBox.warning(self, "No Data", "Nothing to export yet.")
            return
        df = mw.coding_form_df
        if len(df) < 2:
            QMessageBox.warning(self, "No Data", "No data rows to export.")
            return
        header = [str(df.iloc[0, c]) if pd.notna(df.iloc[0, c]) else f"Col{c}" for c in range(len(df.columns))]
        data = df.iloc[1:].copy()
        data.columns = header
        first_col = header[0]
        try:
            grouped = data.groupby(first_col, sort=False)
            wide_rows = []
            for study_name, group in grouped:
                row = {first_col: study_name}
                arm_num = 0
                for _, arm_row in group.iterrows():
                    arm_num += 1
                    for col in header[1:]:
                        row[f"{col}_arm{arm_num}"] = arm_row.get(col, '')
                row['_n_arms'] = arm_num
                wide_rows.append(row)
            if not wide_rows:
                QMessageBox.information(self, "No Data", "Could not pivot data.")
                return
            out = pd.DataFrame(wide_rows)
            out = out.drop(columns=['_n_arms'])
        except Exception as e:
            QMessageBox.warning(self, "Pivot Failed", f"Could not pivot: {e}\nExporting raw format.")
            out = data

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Wide Format", "coding_results_wide.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if not path:
            return
        try:
            if path.endswith('.xlsx'):
                out.to_excel(path, index=False, engine='openpyxl')
            else:
                out.to_csv(path, index=False, encoding='utf-8-sig')
            QMessageBox.information(self, "Done", f"Saved wide format ({len(out)} studies) to {path}")
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

        mw.coding_form_df = mw.coding_form_df.drop(mw.coding_form_df.index[sel])
        mw.coding_form_df = mw.coding_form_df.reset_index(drop=True)
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AIDE Free v1")
        self.resize(1600, 950)

        # Shared state
        self.coding_form_df = None
        self.template_fields = []
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
                self.pdf_loaded() and bool(self.template_fields)
            )
        elif idx == 2:  # Export
            self.export_tab.refresh(self.coding_form_df)

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

        self.analyze_tab.pdf_path = path
        self.analyze_tab._reset()
        self.analyze_tab.pdf_viewer.load_pdf(pdf_bytes)
        self.analyze_tab.analyze_btn.setEnabled(bool(self.template_fields))

        n = self.analyze_tab.pdf_viewer.total_pages
        self.statusBar().showMessage(f"📄 {os.path.basename(path)} — {n} pages")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(240, 242, 245))
    p.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(248, 249, 250))
    p.setColor(QPalette.ColorRole.Text, QColor(33, 37, 41))
    p.setColor(QPalette.ColorRole.Button, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Highlight, QColor(52, 152, 219))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(p)

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