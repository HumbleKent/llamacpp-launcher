"""Shared pytest configuration.

Two safety nets here, both learned the hard way:

1. **A hard per-test timeout** (``pytest-timeout`` via ``timeout`` in
   pyproject.toml).  A Qt modal dialog blocking in a native event loop cannot be
   interrupted by faulthandler, so a plain wall-clock timeout is the only thing
   that reliably stops a wedged suite.

2. **Modals are stubbed out for the whole session.**  ``QMessageBox.exec()``
   spins a native modal loop that waits for a human click; offscreen there is no
   one to click it, so any test that can reach a dialog hangs forever.  Every
   dialog entry point is replaced with a non-blocking stub that records the call,
   so tests can still *assert* that a dialog was raised via ``dialogs``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


@dataclass
class DialogRecorder:
    """Captures calls to QMessageBox helpers instead of showing a modal."""

    calls: List[tuple] = field(default_factory=list)
    #: What a warning/question should "answer".  Defaults to the affirmative /
    #: first button so tests exercise the proceed path unless they opt out.
    answer: Any = None

    def record(self, kind: str, args: tuple) -> Any:
        self.calls.append((kind, args))

    def of_kind(self, kind: str) -> List[tuple]:
        return [args for name, args in self.calls if name == kind]

    def texts(self) -> str:
        return "\n".join(str(a) for _, args in self.calls for a in args if isinstance(a, str))

    def clear(self) -> None:
        self.calls.clear()


@pytest.fixture()
def dialogs(monkeypatch):
    """Replace every blocking QMessageBox helper with a recording stub.

    Returns the recorder; the default ``answer`` is ``Yes`` so that
    confirmation-gated code paths proceed unless a test says otherwise.
    """
    from PyQt6.QtWidgets import QMessageBox

    recorder = DialogRecorder()
    yes = QMessageBox.StandardButton.Yes
    recorder.answer = yes

    def make_stub(kind: str, staticmethod_name: str):
        def stub(*args, **kwargs):
            recorder.record(kind, args)
            if kind == "question":
                return recorder.answer
            return None
        stub.__name__ = staticmethod_name
        return staticmethod(stub)

    for kind, name in (
        ("information", "information"),
        ("warning", "warning"),
        ("critical", "critical"),
        ("question", "question"),
        ("about", "about"),
    ):
        monkeypatch.setattr(QMessageBox, name, make_stub(kind, name))

    return recorder
