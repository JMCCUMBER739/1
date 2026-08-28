"""Background worker thread running the pipeline without blocking the UI."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal

from ..pipeline import PipelineCancelled, PipelineOptions, PipelineResult, run_pipeline


class PipelineWorker(QThread):
    progressed = Signal(int, str)  # percent, message
    finished_ok = Signal(object)  # PipelineResult
    failed = Signal(str)  # traceback text
    cancelled = Signal()

    def __init__(self, options: PipelineOptions, parent=None):
        super().__init__(parent)
        self._options = options
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            result: PipelineResult = run_pipeline(
                self._options,
                progress=lambda pct, msg: self.progressed.emit(pct, msg),
                cancel_check=lambda: self._cancel,
            )
            self.finished_ok.emit(result)
        except PipelineCancelled:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())
