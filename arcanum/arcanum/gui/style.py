"""Visual identity for the Arcanum GUI: palette and Qt stylesheet."""

BG = "#0e1017"
PANEL = "#171a24"
PANEL2 = "#1e2230"
LINE = "#2a2e3e"
INK = "#e8e2d4"
MUTED = "#8b8fa3"
GOLD = "#d4af6a"
GOLD_DIM = "#b3934f"
TEAL = "#5ec8b8"
VIOLET = "#a98fd6"
ROSE = "#d67f8f"

CATEGORY_COLORS = {"esoteric": VIOLET, "theological": GOLD, "scientific": TEAL, "syncretic": ROSE}

STYLESHEET = f"""
* {{
    font-family: Georgia, 'Times New Roman', serif;
}}
QMainWindow, QDialog {{
    background: {BG};
}}
QWidget {{
    color: {INK};
    font-size: 14px;
}}
#Sidebar {{
    background: {PANEL};
    border-right: 1px solid {LINE};
}}
#AppTitle {{
    color: {GOLD};
    font-size: 26px;
    letter-spacing: 2px;
}}
#AppTagline {{
    color: {MUTED};
    font-size: 12px;
    letter-spacing: 3px;
    text-transform: uppercase;
}}
#SectionLabel {{
    color: {MUTED};
    font-size: 11px;
    letter-spacing: 2px;
    margin-top: 10px;
}}
QLabel {{
    background: transparent;
    color: {INK};
}}
QCheckBox {{ color: {INK}; }}
QLineEdit, QComboBox, QSpinBox, QListWidget, QTreeWidget,
QPlainTextEdit, QTextBrowser {{
    background: {PANEL2};
    border: 1px solid {LINE};
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: {GOLD_DIM};
    selection-color: {BG};
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {PANEL2};
    border: 1px solid {LINE};
    selection-background-color: {GOLD_DIM};
    selection-color: {BG};
}}
QPushButton {{
    background: {PANEL2};
    border: 1px solid {LINE};
    border-radius: 6px;
    padding: 8px 16px;
}}
QPushButton:hover {{ border-color: {GOLD_DIM}; color: {GOLD}; }}
QPushButton:disabled {{ color: {MUTED}; }}
QPushButton#RunButton {{
    background: {GOLD};
    color: {BG};
    font-size: 15px;
    font-weight: bold;
    padding: 12px;
    border: none;
    border-radius: 8px;
}}
QPushButton#RunButton:hover {{ background: #e2c185; }}
QPushButton#RunButton:disabled {{ background: {LINE}; color: {MUTED}; }}
QPushButton#GhostButton {{
    background: transparent;
    border: 1px solid {LINE};
    color: {MUTED};
}}
QPushButton#GhostButton:hover {{ color: {ROSE}; border-color: {ROSE}; }}
QProgressBar {{
    background: {PANEL2};
    border: 1px solid {LINE};
    border-radius: 6px;
    height: 14px;
    text-align: center;
    color: {INK};
    font-size: 10px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {GOLD_DIM}, stop:1 {GOLD});
    border-radius: 5px;
}}
QTabWidget::pane {{
    border: 1px solid {LINE};
    border-radius: 8px;
    background: {PANEL};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {MUTED};
    padding: 9px 20px;
    border: 1px solid transparent;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {PANEL};
    color: {GOLD};
    border: 1px solid {LINE};
    border-bottom: 1px solid {PANEL};
}}
QTabBar::tab:hover:!selected {{ color: {INK}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {LINE};
    border-radius: 4px;
    background: {PANEL2};
}}
QCheckBox::indicator:checked {{
    background: {GOLD};
    border-color: {GOLD};
}}
QTreeWidget::item {{ padding: 5px 4px; }}
QTreeWidget::item:selected {{ background: {PANEL2}; color: {GOLD}; }}
QHeaderView::section {{
    background: {PANEL};
    color: {MUTED};
    border: none;
    border-bottom: 1px solid {LINE};
    padding: 6px 8px;
    font-size: 11px;
    letter-spacing: 1px;
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{
    background: {BG}; width: 10px; border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background: {LINE}; border-radius: 5px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {GOLD_DIM}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar:horizontal {{
    background: {BG}; height: 10px; border-radius: 5px;
}}
QScrollBar::handle:horizontal {{
    background: {LINE}; border-radius: 5px; min-width: 30px;
}}
QStatusBar {{
    background: {PANEL};
    color: {MUTED};
    border-top: 1px solid {LINE};
}}
QPlainTextEdit, QTextBrowser {{
    font-size: 13px;
}}
QSplitter::handle {{ background: {LINE}; }}
QToolTip {{
    background: {PANEL2};
    color: {INK};
    border: 1px solid {GOLD_DIM};
    padding: 4px;
}}
"""
