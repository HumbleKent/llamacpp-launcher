"""GUI and subprocess tests, including integration against a real llama.cpp build.

Runs with Qt's ``offscreen`` platform plugin, so no display is required.  The
``dialogs`` fixture (see conftest.py) stubs every QMessageBox so a modal can
never block the suite.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import llamacpp_launcher as app_module  # noqa: E402
from llamacpp_launcher import (  # noqa: E402
    KV_CACHE_TYPES,
    BinaryFlavor,
    LlamaConfig,
    LlamaProcess,
    Profile,
    ProfileStore,
    ProcessError,
    RunState,
    build_argv,
)

#: A real llama.cpp build, if this machine has one.  The integration tests skip
#: cleanly when it is absent so the suite stays portable.
REAL_SERVER = r"C:\Users\User\Documents\LocalLLm\llama-server.exe"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def qapp():
    """Qt requires exactly one QApplication per process."""
    app = QApplication.instance() or QApplication(sys.argv)
    app_module.build_app(sys.argv)
    yield app


@pytest.fixture()
def app(qapp):
    return qapp


@pytest.fixture()
def loop():
    """A private event loop that is never exec()'d, only spun on demand."""
    return QEventLoop()


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Keep every test away from the real per-user profile directory."""
    monkeypatch.setattr(app_module.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


@pytest.fixture()
def window(app, isolated_config, dialogs):
    win = app_module.MainWindow(store=ProfileStore(isolated_config / "profiles"))
    yield win
    if win.process.is_active:
        win.process.kill()
    win.close()


def spin(loop: QEventLoop, ms: int) -> None:
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def wait_for(loop: QEventLoop, predicate, timeout_ms: int = 15000) -> bool:
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        if predicate():
            return True
        spin(loop, 50)
    return predicate()


BLOCKER = "import time\nprint('blocking', flush=True)\ntime.sleep(300)\n"
REPEATER = (
    "import sys\n"
    "print('banner: ready', flush=True)\n"
    "sys.stderr.write('stderr line\\n')\n"
    "sys.stderr.flush()\n"
    "sys.stdin.readline()\n"
)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, check=False,
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


# =========================================================================== #
# Subprocess lifecycle
# =========================================================================== #
def test_output_streams_from_both_channels(app, loop):
    proc = LlamaProcess()
    chunks: list[tuple[str, bool]] = []
    proc.output.connect(lambda text, is_err: chunks.append((text, is_err)))

    proc.start(sys.executable, ["-c", REPEATER])
    assert wait_for(loop, lambda: any("banner: ready" in t for t, _ in chunks)), chunks
    assert any("stderr line" in t for t, is_err in chunks if is_err), chunks
    proc.stop()
    assert wait_for(loop, lambda: proc.state is RunState.STOPPED)


def test_graceful_stop_terminates_the_child(app, loop):
    proc = LlamaProcess()
    proc.start(sys.executable, ["-c", BLOCKER])
    assert wait_for(loop, lambda: proc.state is RunState.RUNNING)
    pid = proc.pid
    proc.stop()
    assert wait_for(loop, lambda: proc.state is RunState.STOPPED)
    assert proc.pid == 0
    assert not _pid_alive(pid)


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only: needs SIGTERM catchable/ignorable")
def test_force_kill_terminates_a_stubborn_child(app, loop):
    stubborn = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('stubborn', flush=True)\n"
        "time.sleep(300)\n"
    )
    proc = LlamaProcess(grace_ms=400)
    proc.start(sys.executable, ["-c", stubborn])
    assert wait_for(loop, lambda: proc.state is RunState.RUNNING)
    pid = proc.pid
    proc.stop()
    assert wait_for(loop, lambda: proc.state is RunState.STOPPED, timeout_ms=20000)
    assert not _pid_alive(pid)


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only: needs SIGTERM catchable/ignorable")
def test_kill_during_stopping_still_terminates_the_child(app, loop):
    """Regression: kill() must not bail out while a stop is already in flight.

    A child that ignores SIGTERM sits in STOPPING for the whole grace period; if
    kill() refused to act there, closing the window mid-shutdown would leave the
    process -- and its VRAM -- alive.
    """
    stubborn = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('stubborn', flush=True)\n"
        "time.sleep(300)\n"
    )
    proc = LlamaProcess(grace_ms=60_000)
    proc.start(sys.executable, ["-c", stubborn])
    assert wait_for(loop, lambda: proc.state is RunState.RUNNING)
    pid = proc.pid

    proc.stop()
    assert wait_for(loop, lambda: proc.state is RunState.STOPPING)
    assert proc.is_active

    proc.kill()
    assert not proc.is_active
    assert not _pid_alive(pid)


def test_kill_is_synchronous(app, loop):
    proc = LlamaProcess()
    proc.start(sys.executable, ["-c", BLOCKER])
    assert wait_for(loop, lambda: proc.state is RunState.RUNNING)
    pid = proc.pid
    proc.kill()
    assert not proc.is_active
    assert not _pid_alive(pid)


def test_double_start_is_rejected(app, loop):
    proc = LlamaProcess()
    proc.start(sys.executable, ["-c", BLOCKER])
    assert wait_for(loop, lambda: proc.state is RunState.RUNNING)
    try:
        with pytest.raises(ProcessError):
            proc.start(sys.executable, ["-c", BLOCKER])
    finally:
        proc.kill()
        wait_for(loop, lambda: proc.state is RunState.STOPPED)


def test_missing_executable_raises(app):
    proc = LlamaProcess()
    with pytest.raises(ProcessError):
        proc.start("definitely-not-a-real-binary-xyz", [])
    assert not proc.is_active


def test_crash_is_reported(app, loop, dialogs):
    proc = LlamaProcess()
    failures: list[str] = []
    proc.failed.connect(lambda summary, detail: failures.append(summary))
    proc.start(sys.executable, ["-c", "raise SystemExit(3)"])
    assert wait_for(loop, lambda: bool(failures)), failures
    assert wait_for(loop, lambda: proc.state is RunState.CRASHED)


def test_user_initiated_stop_is_not_reported_as_a_crash(app, loop):
    proc = LlamaProcess()
    failures: list[str] = []
    proc.failed.connect(lambda summary, detail: failures.append(summary))
    proc.start(sys.executable, ["-c", BLOCKER])
    assert wait_for(loop, lambda: proc.state is RunState.RUNNING)
    proc.kill()
    spin(loop, 200)
    assert proc.state is RunState.STOPPED
    assert failures == []


def test_no_child_survives_a_full_session(app, loop):
    """The property that matters: no orphan left holding VRAM and a TCP port."""
    observed: list[int] = []
    proc = LlamaProcess(grace_ms=400)
    proc.state_changed.connect(lambda _state: observed.append(proc.pid))

    for _ in range(3):
        proc.start(sys.executable, ["-c", BLOCKER])
        assert wait_for(loop, lambda: proc.state is RunState.RUNNING)
        proc.stop()
        assert wait_for(loop, lambda: proc.state is RunState.STOPPED, timeout_ms=20000)

    proc.start(sys.executable, ["-c", BLOCKER])
    assert wait_for(loop, lambda: proc.state is RunState.RUNNING)
    proc.kill()

    pids = {pid for pid in observed if pid > 0}
    assert len(pids) == 4, f"expected 4 distinct children, saw {pids}"
    survivors = sorted(pid for pid in pids if _pid_alive(pid))
    assert survivors == [], f"orphaned child processes: {survivors}"


# =========================================================================== #
# GUI: layout and binding
# =========================================================================== #
def test_window_starts_stopped(window):
    assert window.process.state is RunState.STOPPED
    assert window.start_button.isEnabled()
    assert not window.stop_button.isEnabled()
    assert window.form.isEnabled()


def test_four_organised_tabs_exist(window):
    titles = [window.form.tabs.tabText(i) for i in range(window.form.tabs.count())]
    assert len(titles) == 4
    for expected in ("Model", "Hardware", "Server", "Workflow"):
        assert any(expected in t for t in titles), titles


def test_command_preview_reflects_core_parameters(window):
    window.form.binary_edit.setText("llama-server.exe")
    window.form.model_edit.setText("tiny.gguf")
    window.form.ngl.set_value(42)
    window.form.ctx.set_value(32768)
    window.form.threads.set_value(4)

    preview = window.command_view.toPlainText()
    assert "--n-gpu-layers 42" in preview
    assert "--ctx-size 32768" in preview
    assert "--threads 4" in preview
    assert "tiny.gguf" in preview


def test_status_text_matches_state(window, loop):
    window.process.start(sys.executable, ["-c", BLOCKER])
    assert wait_for(loop, lambda: window.process.state is RunState.RUNNING)
    assert window.status._text.text() == "Running"
    assert not window.start_button.isEnabled()
    assert window.stop_button.isEnabled()
    # Parameter *inputs* freeze, but the form itself is not disabled -- the Web
    # UI button lives inside it and must stay gate-able.
    assert not window.form.binary_edit.isEnabled()
    assert not window.form.model_edit.isEnabled()
    assert not window.form.ngl_spin.isEnabled()

    window.process.kill()
    assert wait_for(loop, lambda: window.process.state is RunState.STOPPED)
    assert window.status._text.text() == "Stopped"
    assert window.start_button.isEnabled()
    assert window.form.binary_edit.isEnabled()
    assert window.form.ngl_spin.isEnabled()


def test_inputs_are_frozen_while_running(window, loop):
    window.process.start(sys.executable, ["-c", BLOCKER])
    assert wait_for(loop, lambda: window.process.state is RunState.RUNNING)
    form = window.form
    for widget in (
        form.binary_edit, form.model_edit, form.ctx, form.threads_spin,
        form.kv_combo, form.flash_attn_combo, form.moe_mode_combo,
        form.lan_check, form.port_spin, form.api_key_edit, form.extra_edit,
        form.schema_edit, form.jinja_check, form.ngl_slider,
    ):
        assert not widget.isEnabled(), f"{widget} should be frozen while running"
    window.process.kill()
    assert wait_for(loop, lambda: window.process.state is RunState.STOPPED)


def test_web_button_stays_clickable_while_the_server_runs(window, loop):
    """Regression: the Web UI button used to be re-disabled with the whole form.

    The button is only useful while the server is up, so it must survive the
    input lock that applies during a run.
    """
    window.form.binary_edit.setText("llama-server.exe")
    window.process.start(sys.executable, ["-c", BLOCKER])
    assert wait_for(loop, lambda: window.process.state is RunState.RUNNING)

    window._on_server_ready("http://127.0.0.1:8080")
    assert window.form.web_button.isEnabled(), "must be clickable during a run"
    assert not window.form.binary_edit.isEnabled(), "inputs stay frozen"

    window.process.kill()
    assert wait_for(loop, lambda: window.process.state is RunState.STOPPED)


# -- hardware ---------------------------------------------------------------- #
def test_kv_cache_combo_carries_every_advertised_type(window):
    offered = {window.form.kv_combo.itemData(i) for i in range(window.form.kv_combo.count())}
    assert set(KV_CACHE_TYPES) <= offered
    assert "" in offered  # "model default" must be selectable


def test_gui_kv_and_flash_attention_reach_the_command_preview(window):
    form = window.form
    form.binary_edit.setText("llama-server.exe")
    form.model_edit.setText("m.gguf")
    form.flash_attn_combo.setCurrentIndex(form.flash_attn_combo.findData("on"))
    form.kv_combo.setCurrentIndex(form.kv_combo.findData("q8_0"))

    preview = window.command_view.toPlainText()
    assert "--flash-attn on" in preview
    assert "--cache-type-k q8_0" in preview
    assert "--cache-type-v q8_0" in preview


def test_gui_moe_modes_produce_the_right_flags(window):
    form = window.form
    form.binary_edit.setText("llama-server.exe")
    form.model_edit.setText("m.gguf")

    form.moe_mode_combo.setCurrentIndex(form.moe_mode_combo.findData("all"))
    assert "--cpu-moe" in window.command_view.toPlainText()
    assert not form.moe_spin.isEnabled()

    form.moe_mode_combo.setCurrentIndex(form.moe_mode_combo.findData("layers"))
    assert form.moe_spin.isEnabled(), "choosing the layer mode must enable the count"
    form.moe_spin.setValue(7)
    preview = window.command_view.toPlainText()
    assert "--n-cpu-moe 7" in preview
    assert "--cpu-moe" not in preview


# -- context selector -------------------------------------------------------- #
def test_context_dropdown_offers_1k_to_1m(window):
    combo = window.form.ctx.combo
    labels = [combo.itemText(i) for i in range(combo.count())]
    for expected in ("1k", "2k", "4k", "8k", "32k", "64k", "128k", "256k", "512k", "1M"):
        assert expected in labels, f"{expected} missing from {labels}"
    assert labels[-1] == "Custom…"


def test_context_dropdown_selection_updates_config(window):
    selector = window.form.ctx
    selector.combo.setCurrentIndex(selector.combo.findData(65536))
    assert selector.value() == 65536
    assert selector.spin.isHidden(), "custom field not needed for a preset value"


def test_context_dropdown_falls_back_to_custom_for_odd_values(window):
    """A profile with e.g. 6144 must not be silently rounded to a preset."""
    selector = window.form.ctx
    selector.set_value(6144)
    assert selector.value() == 6144
    assert selector.combo.currentData() is None
    assert not selector.spin.isHidden(), "custom field must appear for 6144"


def test_context_value_round_trips_through_the_form(window):
    window.form.ctx.set_value(262144)
    window.form.apply_to(window.config)
    assert window.config.ctx_size == 262144


def test_spin_boxes_are_wide_enough_for_their_largest_value(window):
    """Regression: a maximumWidth cap narrower than the text clips digits.

    An over-narrow QSpinBox silently renders "1,048,576" as "1,04...", so the
    user cannot read back the value they just set.  Text width is measured
    directly rather than compared against sizeHint(), because in the fontless
    offscreen environment sizeHint() under-reports and would hide the bug.
    """
    for widget, largest in (
        (window.form.ctx.spin, 10_000_000),
        (window.form.port_spin, 65535),
        (window.form.ngl_spin, 1000),
        (window.form.threads_spin, 256),
        (window.form.moe_spin, 1000),
        (window.form.batch_spin, 1_000_000),
    ):
        widget.setValue(largest)
        text_width = widget.fontMetrics().horizontalAdvance(
            widget.locale().toString(widget.value())
        )
        # Room for the digits plus the spin arrows and frame.
        assert widget.maximumWidth() >= text_width + 24, (
            f"{type(widget).__name__} capped at {widget.maximumWidth()}px, but "
            f"{widget.value()} needs ~{text_width}px of digits plus arrows"
        )


# -- network ----------------------------------------------------------------- #
def test_lan_checkbox_forces_host_and_locks_the_field(window):
    form = window.form
    assert not form.lan_check.isChecked()
    assert form.host_edit.isEnabled()

    form.lan_check.setChecked(True)
    assert form.host_edit.text() == "0.0.0.0"
    assert not form.host_edit.isEnabled()
    assert not form.localhost_check.isChecked()
    assert "--host 0.0.0.0" in window.command_view.toPlainText()

    form.lan_check.setChecked(False)
    assert form.host_edit.text() == "127.0.0.1"
    assert form.host_edit.isEnabled()


def test_generate_api_key_fills_the_field_and_clipboard(window):
    window._on_generate_api_key()
    key = window.form.api_key_edit.text()
    assert key.startswith("sk-")
    assert QApplication.clipboard().text() == key
    assert window.form.api_key_edit.echoMode().name == "Normal", "key should be revealed"


def test_exposure_hint_warns_only_when_unauthenticated(window):
    form = window.form
    form.lan_check.setChecked(True)
    assert "No API key" in form.exposure_label.text()

    form.api_key_edit.setText("sk-something")
    assert "authenticate" in form.exposure_label.text()

    form.lan_check.setChecked(False)
    assert "Loopback only" in form.exposure_label.text()


# -- web UI ------------------------------------------------------------------ #
def test_web_button_disabled_until_listening(window):
    form = window.form
    form.binary_edit.setText("llama-server.exe")
    assert not form.web_button.isEnabled()
    assert "Start llama-server" in form.web_hint.text()


def test_web_button_enabled_once_the_server_reports_listening(window, loop):
    form = window.form
    form.binary_edit.setText("llama-server.exe")

    # A real child must be running: the button is gated on the controller's
    # state, and _on_state_changed is a *view* handler that never sets it.
    window.process.start(sys.executable, ["-c", BLOCKER])
    assert wait_for(loop, lambda: window.process.state is RunState.RUNNING)
    assert not form.web_button.isEnabled(), "not ready before the banner"

    window._on_server_ready("http://127.0.0.1:8080")
    assert form.web_button.isEnabled()
    assert window._web_action.isEnabled()
    assert "http://127.0.0.1:8080" in form.web_hint.text()

    window.process.kill()
    assert wait_for(loop, lambda: window.process.state is RunState.STOPPED)


def test_web_button_reports_cli_binary_has_no_ui(window):
    window.form.binary_edit.setText("llama-cli.exe")
    window._refresh_web_button()
    assert not window.form.web_button.isEnabled()
    assert "no HTTP server" in window.form.web_hint.text()


def test_open_web_ui_hands_the_url_to_the_default_browser(window, loop, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(
        app_module.QDesktopServices, "openUrl",
        staticmethod(lambda url: opened.append(url.toString()) or True),
    )
    window.form.binary_edit.setText("llama-server.exe")
    window.process.start(sys.executable, ["-c", BLOCKER])
    assert wait_for(loop, lambda: window.process.state is RunState.RUNNING)

    window._on_server_ready("http://127.0.0.1:8080")
    window._on_open_web_ui()
    assert opened == ["http://127.0.0.1:8080"]

    window.process.kill()
    assert wait_for(loop, lambda: window.process.state is RunState.STOPPED)


def test_open_web_ui_rewrites_wildcard_host_to_loopback(window, loop, monkeypatch):
    """With --host 0.0.0.0 the banner says 0.0.0.0, which a browser cannot open."""
    opened: list[str] = []
    monkeypatch.setattr(
        app_module.QDesktopServices, "openUrl",
        staticmethod(lambda url: opened.append(url.toString()) or True),
    )
    form = window.form
    form.binary_edit.setText("llama-server.exe")
    form.lan_check.setChecked(True)
    form.port_spin.setValue(8080)
    window.process.start(sys.executable, ["-c", BLOCKER])
    assert wait_for(loop, lambda: window.process.state is RunState.RUNNING)

    window._on_server_ready("http://0.0.0.0:8080")
    window._on_open_web_ui()
    assert opened == ["http://127.0.0.1:8080"]

    window.process.kill()
    assert wait_for(loop, lambda: window.process.state is RunState.STOPPED)


def test_open_web_ui_refuses_before_ready(window, monkeypatch, dialogs):
    opened: list[str] = []
    monkeypatch.setattr(
        app_module.QDesktopServices, "openUrl",
        staticmethod(lambda url: opened.append(url.toString()) or True),
    )
    window._on_open_web_ui()
    assert opened == [], "must not open a dead tab"
    assert dialogs.of_kind("information"), "expected an explanation"


def test_auto_open_browser_fires_on_ready(window, loop, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(
        app_module.QDesktopServices, "openUrl",
        staticmethod(lambda url: opened.append(url.toString()) or True),
    )
    form = window.form
    form.binary_edit.setText("llama-server.exe")
    form.auto_open_check.setChecked(True)
    form.apply_to(window.config)
    window.process.start(sys.executable, ["-c", BLOCKER])
    assert wait_for(loop, lambda: window.process.state is RunState.RUNNING)

    window._on_server_ready("http://127.0.0.1:8080")
    assert opened == ["http://127.0.0.1:8080"]

    window.process.kill()
    assert wait_for(loop, lambda: window.process.state is RunState.STOPPED)


# -- workflow ---------------------------------------------------------------- #
def test_jinja_toggle_reaches_the_command(window):
    window.form.model_edit.setText("m.gguf")
    window.form.jinja_check.setChecked(True)
    assert "--jinja" in window.command_view.toPlainText()
    window.form.jinja_check.setChecked(False)
    assert "--no-jinja" in window.command_view.toPlainText()


def test_schema_box_reports_valid_and_invalid_json(window):
    form = window.form
    form.schema_check.setChecked(True)
    form.schema_edit.setPlainText('{"type": "object"}')
    assert "Valid JSON" in form.schema_status.text()

    form.schema_edit.setPlainText("{oops")
    assert "Invalid JSON" in form.schema_status.text()
    assert form.schema_status.text().startswith("✗")


def test_schema_disabled_contributes_nothing(window):
    form = window.form
    form.model_edit.setText("m.gguf")
    form.schema_edit.setPlainText('{"type": "object"}')
    form.schema_check.setChecked(False)
    form.apply_to(window.config)
    assert window.config.json_schema == ""
    assert "--json-schema" not in window.command_view.toPlainText()


def test_format_schema_pretty_prints(window):
    form = window.form
    form.schema_edit.setPlainText('{"type":"object","properties":{"a":{"type":"string"}}}')
    form._format_schema()
    text = form.schema_edit.toPlainText()
    assert "\n" in text
    assert json.loads(text)["type"] == "object"


def test_large_schema_uses_a_file_and_is_cleaned_up(window, isolated_config):
    """A multi-KB schema must not be inlined into the command line."""
    form = window.form
    form.binary_edit.setText("llama-server.exe")
    form.model_edit.setText("m.gguf")
    form.schema_check.setChecked(True)
    form.schema_edit.setPlainText(json.dumps({"type": "object", "pad": "x" * 9000}))
    form.apply_to(window.config)

    path = app_module.plan_schema_file(window.config, isolated_config / "tmp")
    assert path is not None and path.exists()
    assert "--json-schema-file" in app_module.format_command(
        app_module.build_command(window.config, BinaryFlavor.SERVER, path)
    )

    window._schema_file = path
    window._on_process_finished(0, "exited")
    assert not path.exists(), "the temp schema file must not outlive the run"
    assert window._schema_file is None


def test_cleanup_schema_file_is_idempotent(window):
    window._cleanup_schema_file()
    window._cleanup_schema_file()
    assert window._schema_file is None


def test_schema_round_trips_through_a_profile(window):
    form = window.form
    form.schema_check.setChecked(True)
    form.schema_edit.setPlainText('{"type": "object", "title": "Person"}')
    form.apply_to(window.config)

    saved = window.store.save(Profile(name="withschema", config=window.config))
    loaded = window.store.load(saved)
    assert json.loads(loaded.config.json_schema)["title"] == "Person"


# -- profiles ---------------------------------------------------------------- #
def test_save_and_load_preset_round_trip(window):
    window.form.model_edit.setText("models/llama.gguf")
    window.form.ngl.set_value(17)
    window.form.kv_combo.setCurrentIndex(window.form.kv_combo.findData("q8_0"))
    window.form.apply_to(window.config)

    path = window.store.save(Profile(name="roundtrip", config=window.config))
    window.form.ngl.set_value(0)
    window._apply_profile(window.store.load(path))

    assert window.form.ngl.value() == 17
    assert window.config.kv_cache_type == "q8_0"
    assert window.windowTitle().endswith("roundtrip")
    assert window.dirty_label.text() == ""


def test_applying_a_profile_refreshes_the_command_preview(window):
    cfg = LlamaConfig(
        binary_path="llama-server.exe", model_path="m.gguf", ctx_size=131072,
        kv_cache_type="q8_0", moe_mode="layers", moe_layers=4,
    )
    window._apply_profile(Profile(name="p", config=cfg))
    preview = window.command_view.toPlainText()
    assert "--ctx-size 131072" in preview
    assert "--cache-type-k q8_0" in preview
    assert "--n-cpu-moe 4" in preview


# -- validation -------------------------------------------------------------- #
def test_validation_blocks_a_launch_without_a_model(window, dialogs):
    window.form.model_edit.setText("")
    window._on_start()
    assert dialogs.of_kind("warning"), "expected a validation warning"
    assert not window.process.is_active


def test_invalid_schema_blocks_the_launch(window, dialogs):
    form = window.form
    form.model_edit.setText("m.gguf")
    form.binary_edit.setText("llama-server.exe")
    form.schema_check.setChecked(True)
    form.schema_edit.setPlainText("{broken")
    window._on_start()
    assert "JSON schema" in dialogs.texts()
    assert not window.process.is_active


def test_exposed_without_key_requires_confirmation(window, isolated_config, dialogs):
    """Binding to 0.0.0.0 with no key must ask before publishing the model."""
    # Real files on disk so validation gets past the fatal checks and reaches
    # the exposure warning (which is deliberately non-fatal).
    binary = isolated_config / "llama-server.exe"
    binary.write_bytes(b"stub")
    model = isolated_config / "m.gguf"
    model.write_bytes(b"stub")

    form = window.form
    form.binary_edit.setText(str(binary))
    form.model_edit.setText(str(model))
    form.lan_check.setChecked(True)
    form.apply_to(window.config)

    dialogs.answer = QMessageBox.StandardButton.Cancel
    window._on_start()
    assert "0.0.0.0" in dialogs.texts(), dialogs.texts()
    assert not window.process.is_active, "cancelling must not start the server"


def test_exposed_with_key_starts_without_a_confirmation_prompt(
    window, isolated_config, dialogs, loop
):
    """With a key set, binding to the LAN is a legitimate choice, not a warning."""
    binary = isolated_config / "llama-server.exe"
    binary.write_bytes(b"stub")
    model = isolated_config / "m.gguf"
    model.write_bytes(b"stub")

    form = window.form
    form.binary_edit.setText(str(binary))
    form.model_edit.setText(str(model))
    form.lan_check.setChecked(True)
    form.api_key_edit.setText("sk-test-key")
    form.apply_to(window.config)

    window._on_start()
    assert "0.0.0.0" not in dialogs.texts() or "Start anyway" not in dialogs.texts()
    if window.process.is_active:
        window.process.kill()
        wait_for(loop, lambda: window.process.state is RunState.STOPPED)


# -- terminal ---------------------------------------------------------------- #
def test_terminal_is_read_only_and_bounded(window):
    window.terminal.clear()
    window.terminal.append("hello\nworld\n")
    assert window.terminal.view.isReadOnly()
    assert window.terminal.view.maximumBlockCount() > 0
    window.terminal.clear()
    assert window.terminal.text() == ""


def test_terminal_keeps_each_line_separate(window):
    """Regression: an HTML-based append fuses the whole log into one paragraph."""
    window.terminal.clear()
    window.terminal.append("first\nsecond\n")
    window.terminal.append("third\n", is_stderr=True)
    window.terminal.append_notice("a notice")
    lines = window.terminal.text().splitlines()
    assert lines == ["first", "second", "third", "── a notice ──"], lines


# =========================================================================== #
# Integration against the real llama.cpp binary (skipped when absent)
# =========================================================================== #
real_binary = pytest.mark.skipif(
    not os.path.isfile(REAL_SERVER), reason="no llama.cpp build on this machine"
)


@real_binary
def test_real_binary_accepts_every_generated_flag(tmp_path):
    """Prove the argv we generate parses in a real build, not just in theory.

    A bogus model path makes the server exit early, but *after* argument
    parsing -- so a parse failure shows up as "error while handling argument",
    while an accepted command line gets as far as trying to open the GGUF.
    """
    schema_file = tmp_path / "s.schema.json"
    schema_file.write_text('{"type":"object","properties":{"a":{"type":"string"}}}', encoding="utf-8")

    cfg = LlamaConfig(
        binary_path=REAL_SERVER,
        model_path=str(tmp_path / "does-not-exist.gguf"),
        n_gpu_layers=8, ctx_size=4096, threads=4, threads_batch=4, batch_size=2048,
        flash_attn="on", kv_cache_type="q8_0",
        moe_mode="layers", moe_layers=5,
        jinja=True, json_schema=schema_file.read_text(encoding="utf-8"),
        port=8123,
    )
    argv = build_argv(cfg, BinaryFlavor.SERVER, schema_file)
    assert "--json-schema-file" in argv

    result = subprocess.run(
        [REAL_SERVER] + argv, capture_output=True, text=True, timeout=120, check=False
    )
    output = (result.stdout or "") + (result.stderr or "")

    assert "error while handling argument" not in output, output[-2000:]
    assert "invalid argument" not in output.lower()
    # It must have progressed to the model-loading stage.
    assert "GGUF" in output or "model" in output.lower(), output[-2000:]


@real_binary
def test_real_binary_rejects_the_boolean_flag_given_a_value(tmp_path):
    """Documents *why* --cpu-moe and --n-cpu-moe are modelled separately.

    Passing a numeric value to the boolean --cpu-moe is a parse error, which is
    exactly the bug a naive "cmoe takes a number" implementation would ship.
    """
    result = subprocess.run(
        [REAL_SERVER, "--model", str(tmp_path / "x.gguf"), "--cpu-moe", "5"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    assert ("error while handling argument" in output
            or "invalid argument" in output.lower()
            or "expected" in output.lower()), output[-2000:]


@real_binary
def test_real_server_lifecycle_leaves_no_orphan(tmp_path, loop):
    """End-to-end: spawn a real llama-server via LlamaProcess, then kill it.

    A dummy GGUF makes it fail fast either way; the point is that the process is
    tracked and nothing survives.
    """
    model = tmp_path / "dummy.gguf"
    model.write_bytes(b"not a real gguf")

    proc = LlamaProcess(grace_ms=3000)
    cfg = LlamaConfig(
        binary_path=REAL_SERVER, model_path=str(model),
        n_gpu_layers=0, ctx_size=512, threads=2, port=8199,
    )
    argv = build_argv(cfg, BinaryFlavor.SERVER)

    proc.start(REAL_SERVER, argv)
    settled = wait_for(
        loop,
        lambda: proc.state in (RunState.RUNNING, RunState.STOPPED, RunState.CRASHED),
        timeout_ms=60000,
    )
    assert settled, f"server neither started nor exited (state={proc.state})"

    pid = proc.pid
    if proc.is_active:
        proc.kill()
    assert not proc.is_active
    assert not _pid_alive(pid), "leaked a real llama-server process"
