#!/usr/bin/env python3
"""llamacpp Launcher -- a PyQt6 desktop GUI for the llama.cpp command-line tools.

A single-file launcher for ``llama-server`` (HTTP API + web UI) and
``llama-cli`` (interactive prompt).

Flags are version-checked against llama.cpp build 10262.  Where a flag has
changed meaning across releases, the builder emits the form that is valid for
the *detected* binary flavour, and the UI only offers options the selected
binary actually accepts.

Run with::

    python llamacpp_launcher.py

Requires: Python 3.9+ and PyQt6 (``pip install PyQt6``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PyQt6.QtCore import QProcess, QProcessEnvironment, QObject, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QFont,
    QFontDatabase,
    QKeySequence,
    QTextCharFormat,
    QTextCursor,
)
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

APP_NAME = "llamacpp Launcher"
APP_VERSION = "0.2.0"
PROFILE_SCHEMA_VERSION = 2
PROFILE_SUFFIX = ".json"
SCHEMA_SUFFIX = ".schema.json"
APP_DIR_NAME = "llamacpp-launcher"

#: llama.cpp's own limit is roughly 32k characters on Windows for a whole
#: command line; well below that, an inline schema is fine.  Above it we must
#: switch to --json-schema-file or the process fails to spawn.
INLINE_SCHEMA_LIMIT = 8000

#: Values accepted by --cache-type-k / --cache-type-v (build 10262).
KV_CACHE_TYPES = ("f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "f32")

#: Context sizes offered in the dropdown (value, label).
CONTEXT_PRESETS = (
    (512, "512"),
    (1024, "1k"),
    (2048, "2k"),
    (4096, "4k"),
    (8192, "8k"),
    (16384, "16k"),
    (32768, "32k"),
    (65536, "64k"),
    (131072, "128k"),
    (262144, "256k"),
    (524288, "512k"),
    (1048576, "1M"),
)

DEFAULT_PORT = 8080
DEFAULT_GRACE_MS = 4000
MAX_LOG_BLOCKS = 5000


class ConfigError(Exception):
    """Raised for unreadable, malformed or incompatible profile files."""


class ProcessError(Exception):
    """Raised when a launch cannot even be attempted."""


# =========================================================================== #
# Configuration model
# =========================================================================== #
@dataclass
class LlamaConfig:
    """Every launcher-controlled setting for one llama.cpp invocation.

    ``None`` (or an empty string) on an optional field means "omit the flag
    entirely" so llama.cpp keeps its own default.
    """

    # --- executables / files ------------------------------------------------
    binary_path: str = ""
    model_path: str = ""

    # --- core parameters ----------------------------------------------------
    n_gpu_layers: int = 0            # -ngl / --n-gpu-layers
    ctx_size: int = 4096             # -c / --ctx-size
    threads: int = 8                 # -t / --threads
    threads_batch: Optional[int] = None      # -tb
    batch_size: Optional[int] = None         # -b
    mlock: bool = False                      # --mlock
    no_mmap: bool = False                    # --no-mmap

    # --- advanced hardware --------------------------------------------------
    flash_attn: str = "auto"                 # -fa [on|off|auto]
    kv_cache_type: str = ""                  # -ctk / -ctv, "" = model default
    moe_mode: str = "off"                    # off | all | layers
    moe_layers: Optional[int] = None         # -ncmoe N (when mode == "layers")

    # --- network ------------------------------------------------------------
    host: str = "127.0.0.1"                  # --host
    port: int = DEFAULT_PORT                 # --port
    bind_all_interfaces: bool = False        # UI toggle for 0.0.0.0
    api_key: str = ""                        # --api-key

    # --- modern workflows ---------------------------------------------------
    jinja: bool = True                       # --jinja / --no-jinja
    json_schema: str = ""                    # --json-schema / --json-schema-file

    # --- free-form ----------------------------------------------------------
    extra_args: str = ""

    # --- launcher behaviour (never passed to llama.cpp) ---------------------
    auto_scroll: bool = True
    auto_open_browser: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-ready dict, excluding the JSON schema.

        The schema lives in a ``.schema.json`` sidecar (see ProfileStore) because
        it can be many KB and would swamp a profile meant to be read and
        hand-edited.  ``from_dict`` therefore always round-trips it as empty, and
        the store re-attaches it on load.
        """
        data = asdict(self)
        data["json_schema"] = ""
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LlamaConfig":
        """Build a config from a mapping, ignoring unknown keys.

        Unknown keys are dropped rather than raising so a profile written by a
        newer build still loads here.
        """
        if not isinstance(data, dict):
            raise ConfigError("profile 'config' must be a JSON object")
        known = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.normalize()
        return cfg

    def normalize(self) -> None:
        """Coerce and clamp values so a hand-edited profile cannot crash the UI."""
        self.binary_path = str(self.binary_path or "").strip()
        self.model_path = str(self.model_path or "").strip()
        self.host = str(self.host or "127.0.0.1").strip()
        self.api_key = str(self.api_key or "").strip()
        self.extra_args = str(self.extra_args or "").strip()
        self.json_schema = str(self.json_schema or "").strip()

        self.n_gpu_layers = _clamp_int(self.n_gpu_layers, 0, 1000, 0)
        self.ctx_size = _clamp_int(self.ctx_size, 0, 10_000_000, 4096)
        self.threads = _clamp_int(self.threads, 1, 1024, 8)
        self.port = _clamp_int(self.port, 1, 65535, DEFAULT_PORT)
        self.threads_batch = _clamp_int(self.threads_batch, 1, 1024, None)
        self.batch_size = _clamp_int(self.batch_size, 1, 1_000_000, None)
        self.moe_layers = _clamp_int(self.moe_layers, 0, 1000, None)

        self.flash_attn = self.flash_attn if self.flash_attn in ("on", "off", "auto") else "auto"
        self.kv_cache_type = (
            self.kv_cache_type if self.kv_cache_type in KV_CACHE_TYPES else ""
        )
        self.moe_mode = self.moe_mode if self.moe_mode in ("off", "all", "layers") else "off"

        self.mlock = bool(self.mlock)
        self.no_mmap = bool(self.no_mmap)
        self.bind_all_interfaces = bool(self.bind_all_interfaces)
        self.jinja = bool(self.jinja)
        self.auto_scroll = bool(self.auto_scroll)
        self.auto_open_browser = bool(self.auto_open_browser)

        # The bind toggle is the single source of truth for `host`, so the two
        # can never disagree after a hand-edit or a profile load.
        if self.bind_all_interfaces:
            self.host = "0.0.0.0"
        elif self.host == "0.0.0.0":
            self.bind_all_interfaces = True

    def effective_host(self) -> str:
        return "0.0.0.0" if self.bind_all_interfaces else (self.host or "127.0.0.1")

    def schema_object(self) -> Optional[Any]:
        """Return the parsed JSON schema, or ``None`` when unset/invalid."""
        text = self.json_schema.strip()
        if not text or text == "null":
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None


def _clamp_int(value: Any, low: int, high: int, default: Optional[int]) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, ivalue))


# --------------------------------------------------------------------------- #
# Paths and profile persistence
# --------------------------------------------------------------------------- #
def config_dir() -> Path:
    """Per-user directory for application state (profiles, temp schemas)."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / APP_DIR_NAME


def profiles_dir() -> Path:
    return config_dir() / "profiles"


def default_profile_path() -> Path:
    return profiles_dir() / f"default{PROFILE_SUFFIX}"


def safe_profile_name(name: str) -> str:
    """Reduce an arbitrary user string to a filesystem-safe profile stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip(" .")
    return cleaned or "profile"


@dataclass
class Profile:
    """A named :class:`LlamaConfig` plus bookkeeping metadata."""

    name: str = "default"
    schema_version: int = PROFILE_SCHEMA_VERSION
    config: LlamaConfig = field(default_factory=LlamaConfig)

    def to_dict(self) -> Dict[str, Any]:
        # The JSON schema is deliberately *not* embedded: it can be many KB and
        # would swamp a profile meant to be read and hand-edited.
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Profile":
        if not isinstance(data, dict):
            raise ConfigError("profile root must be a JSON object")

        version = data.get("schema_version", 0)
        if not isinstance(version, int):
            raise ConfigError("profile 'schema_version' must be an integer")
        if version > PROFILE_SCHEMA_VERSION:
            raise ConfigError(
                f"profile schema v{version} is newer than this app supports "
                f"(v{PROFILE_SCHEMA_VERSION})"
            )

        # v0 profiles were a bare config object.
        payload = data.get("config", {}) if "config" in data else data
        return cls(
            name=str(data.get("name") or "default"),
            schema_version=version or PROFILE_SCHEMA_VERSION,
            config=LlamaConfig.from_dict(payload),
        )


class ProfileStore:
    """Reads and writes profiles as JSON files inside one directory."""

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.directory = Path(directory) if directory else profiles_dir()

    def ensure_dir(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> Path:
        return self.directory / f"{safe_profile_name(name)}{PROFILE_SUFFIX}"

    def schema_path_for(self, profile_path: Path) -> Path:
        return Path(profile_path).with_suffix(SCHEMA_SUFFIX)

    def save(self, profile: Profile, path: Optional[Path] = None) -> Path:
        """Write *profile* atomically, plus its schema sidecar if present."""
        target = Path(path) if path else self.path_for(profile.name)
        if target.suffix.lower() != PROFILE_SUFFIX:
            target = target.with_suffix(PROFILE_SUFFIX)
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(profile.to_dict(), indent=2, sort_keys=True)
        _atomic_write(target, payload + "\n")
        self._save_schema_sidecar(target, profile.config.json_schema)
        return target

    def _save_schema_sidecar(self, profile_path: Path, schema_text: str) -> None:
        sidecar = self.schema_path_for(profile_path)
        text = (schema_text or "").strip()
        if not text or text == "null":
            if sidecar.exists():
                try:
                    sidecar.unlink()
                except OSError:
                    pass
            return
        _atomic_write(sidecar, text + "\n")

    def load(self, path: Path) -> Profile:
        path = Path(path)
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"could not read {path.name}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path.name} is not valid JSON: {exc}") from exc

        profile = Profile.from_dict(data)
        if not profile.name or profile.name == "default":
            profile.name = path.stem

        sidecar = self.schema_path_for(path)
        if sidecar.is_file():
            try:
                profile.config.json_schema = sidecar.read_text(encoding="utf-8").strip()
            except OSError:
                pass
        return profile

    def list_profiles(self) -> List[Path]:
        if not self.directory.is_dir():
            return []
        return sorted(
            (p for p in self.directory.glob(f"*{PROFILE_SUFFIX}") if p.is_file()),
            key=lambda p: p.name.lower(),
        )

    def delete(self, name: str) -> bool:
        target = self.path_for(name)
        try:
            target.unlink()
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise ConfigError(f"could not delete profile: {exc}") from exc
        sidecar = self.schema_path_for(target)
        if sidecar.exists():
            try:
                sidecar.unlink()
            except OSError:
                pass
        return True


def _atomic_write(target: Path, text: str) -> None:
    """Write via a temp file + os.replace so a crash cannot truncate the target."""
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        raise ConfigError(f"could not write {target.name}: {exc}") from exc
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Argument tokenising / formatting
# --------------------------------------------------------------------------- #
def split_extra_args(text: str) -> List[str]:
    """Split a free-form argument string the way a POSIX shell would.

    Handles single quotes, double quotes and backslash escapes so Windows paths
    such as ``"C:\\Program Files\\m.gguf"`` survive.  Inside double quotes a
    backslash is only special before ``"``, ``\\``, ``$`` or a backtick -- the
    rule bash uses -- which keeps ``"C:\\models"`` literal.

    Deliberately performs **no** globbing or variable expansion: the result is
    handed straight to a process without a shell.
    """
    tokens: List[str] = []
    current: List[str] = []
    quote: Optional[str] = None
    escaped = False
    started = False

    for index, char in enumerate(text):
        following = text[index + 1] if index + 1 < len(text) else ""
        if escaped:
            current.append(char)
            escaped = False
            started = True
        elif char == "\\" and quote != "'":
            if quote == '"' and following not in ('"', "\\", "$", "`"):
                current.append(char)
                started = True
            else:
                escaped = True
        elif quote:
            if char == quote:
                quote = None
            else:
                current.append(char)
            started = True
        elif char in ("'", '"'):
            quote = char
            started = True
        elif char.isspace():
            if started:
                tokens.append("".join(current))
                current = []
                started = False
        else:
            current.append(char)
            started = True

    if escaped:
        current.append("\\")
    if quote:
        raise ConfigError(f"unbalanced {quote} quote in extra arguments")
    if started:
        tokens.append("".join(current))
    return tokens


def quote_for_display(token: str) -> str:
    if token and not any(c in token for c in ' \t"\''):
        return token
    return '"' + token.replace('"', '\\"') + '"'


def format_command(argv: Sequence[str]) -> str:
    return " ".join(quote_for_display(t) for t in argv)


# --------------------------------------------------------------------------- #
# Executable flavour detection
# --------------------------------------------------------------------------- #
class BinaryFlavor(Enum):
    SERVER = "server"   # llama-server / server -> HTTP API + web UI
    CLI = "cli"         # llama-cli / main       -> interactive prompt
    UNKNOWN = "unknown"


_SERVER_STEMS = {"llama-server", "server", "llamacpp-server"}
_CLI_STEMS = {"llama-cli", "main", "llama", "llama-run", "llamacpp"}
_READY_RE = re.compile(r"server is listening on\s+(\S+)", re.IGNORECASE)


def detect_flavor(binary_path: str) -> BinaryFlavor:
    """Infer the executable flavour from its file name."""
    if not binary_path:
        return BinaryFlavor.UNKNOWN
    stem = Path(binary_path).stem.lower()
    if stem in _SERVER_STEMS:
        return BinaryFlavor.SERVER
    if stem in _CLI_STEMS:
        return BinaryFlavor.CLI
    for token, flavor in (("server", BinaryFlavor.SERVER), ("cli", BinaryFlavor.CLI)):
        if re.search(rf"(^|[-_]){token}([-_]|$)", stem):
            return flavor
    return BinaryFlavor.UNKNOWN


def resolve_executable(path: str) -> Optional[str]:
    """Return an absolute executable path, or ``None`` if it is not runnable."""
    if not path or not path.strip():
        return None
    candidate = Path(path.strip().strip('"')).expanduser()
    if candidate.is_file():
        return str(candidate)
    if os.name == "nt" and candidate.suffix == "":
        for suffix in (".exe", ".cmd", ".bat"):
            alt = candidate.with_suffix(suffix)
            if alt.is_file():
                return str(alt)
    return shutil.which(path.strip())


def parse_ready_url(line: str) -> Optional[str]:
    """Extract the API base URL from a llama-server readiness banner."""
    match = _READY_RE.search(line)
    return match.group(1).rstrip("/") if match else None


def browser_url_for(host: str, port: int, detected_url: Optional[str] = None) -> str:
    """Pick a URL a browser can actually open.

    A wildcard bind address (0.0.0.0 / ::) is not a connectable destination, so
    it is rewritten to loopback -- otherwise the Web UI button would open a
    dead address.
    """
    candidate = (detected_url or "").strip()
    if candidate:
        url = candidate if "://" in candidate else f"http://{candidate}"
        return re.sub(r"//(\[::\]|0\.0\.0\.0|::)", "//127.0.0.1", url)
    host = (host or "127.0.0.1").strip()
    if host in ("0.0.0.0", "::", "[::]", ""):
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):  # bare IPv6
        host = f"[{host}]"
    return f"http://{host}:{port}"


# --------------------------------------------------------------------------- #
# Validation and argv assembly
# --------------------------------------------------------------------------- #
@dataclass
class ValidationIssue:
    field: str
    message: str
    fatal: bool = True


def validate(cfg: LlamaConfig, flavor: Optional[BinaryFlavor] = None) -> List[ValidationIssue]:
    """Pre-flight checks run before spawning anything."""
    issues: List[ValidationIssue] = []
    if flavor is None:
        flavor = detect_flavor(cfg.binary_path)

    if not resolve_executable(cfg.binary_path):
        issues.append(ValidationIssue(
            "binary_path", "llama.cpp executable not found -- pick llama-server or llama-cli."
        ))

    if not cfg.model_path:
        issues.append(ValidationIssue("model_path", "No model (.gguf) file selected."))
    elif not Path(cfg.model_path).expanduser().is_file():
        issues.append(ValidationIssue(
            "model_path", f"Model file does not exist:\n{cfg.model_path}"
        ))
    elif Path(cfg.model_path).suffix.lower() != ".gguf":
        issues.append(ValidationIssue(
            "model_path", "Model file is not a .gguf file -- llama.cpp may refuse to load it.",
            fatal=False,
        ))

    if cfg.threads < 1:
        issues.append(ValidationIssue("threads", "CPU threads must be at least 1."))

    if cfg.kv_cache_type and cfg.kv_cache_type not in KV_CACHE_TYPES:
        issues.append(ValidationIssue(
            "kv_cache_type", f"Unsupported KV cache type '{cfg.kv_cache_type}'."
        ))

    if cfg.moe_mode == "layers" and (cfg.moe_layers is None or cfg.moe_layers < 1):
        issues.append(ValidationIssue(
            "moe_layers", "MoE offloading is set to a layer count but no count is given."
        ))

    # JSON schema sanity: catch it here rather than letting llama.cpp abort.
    schema_text = cfg.json_schema.strip()
    if schema_text and schema_text != "null":
        try:
            parsed = json.loads(schema_text)
        except json.JSONDecodeError as exc:
            issues.append(ValidationIssue(
                "json_schema", f"JSON schema is not valid JSON:\n{exc}"
            ))
        else:
            if not isinstance(parsed, (dict, bool)):
                issues.append(ValidationIssue(
                    "json_schema",
                    "JSON schema must be a JSON object (or false/true for none/any).",
                ))

    try:
        split_extra_args(cfg.extra_args)
    except ConfigError as exc:
        issues.append(ValidationIssue("extra_args", str(exc)))

    # Exposure warning: binding to 0.0.0.0 publishes the API to the whole LAN.
    if cfg.bind_all_interfaces or cfg.effective_host() == "0.0.0.0":
        if not cfg.api_key:
            issues.append(ValidationIssue(
                "api_key",
                "The server is bound to 0.0.0.0 (all network interfaces) with no "
                "API key set. Anyone on your network can use this model and will "
                "see your prompts.\n\nSet an API key below, or leave the key field "
                "blank and confirm you accept the risk.",
                fatal=False,
            ))

    if flavor is BinaryFlavor.CLI:
        if cfg.effective_host() != "127.0.0.1" or cfg.port != DEFAULT_PORT:
            issues.append(ValidationIssue(
                "binary_path",
                "This is the CLI binary; --host/--port and the Web UI only apply to "
                "llama-server and will be ignored.",
                fatal=False,
            ))

    return issues


def validate_before_start(cfg: LlamaConfig, flavor: BinaryFlavor) -> List[ValidationIssue]:
    return validate(cfg, flavor)


def build_argv(
    cfg: LlamaConfig,
    flavor: Optional[BinaryFlavor] = None,
    schema_file: Optional[Path] = None,
) -> List[str]:
    """Build the argument vector for one launch.

    ``extra_args`` is appended last so a user can always override an earlier flag
    -- llama.cpp parses repeated options last-wins.

    ``schema_file`` lets a large JSON schema be passed via --json-schema-file,
    which sidesteps the ~32k Windows command-line limit.
    """
    argv: List[str] = []

    if cfg.model_path:
        argv += ["--model", cfg.model_path]

    # --- core ---------------------------------------------------------------
    argv += ["--n-gpu-layers", str(cfg.n_gpu_layers)]
    if cfg.ctx_size and cfg.ctx_size > 0:
        argv += ["--ctx-size", str(cfg.ctx_size)]
    argv += ["--threads", str(cfg.threads)]
    if cfg.threads_batch:
        argv += ["--threads-batch", str(cfg.threads_batch)]
    if cfg.batch_size:
        argv += ["--batch-size", str(cfg.batch_size)]
    if cfg.mlock:
        argv.append("--mlock")
    if cfg.no_mmap:
        argv.append("--no-mmap")

    # --- advanced hardware --------------------------------------------------
    # 'auto' is llama.cpp's own default, so only send an explicit on/off.
    if cfg.flash_attn in ("on", "off"):
        argv += ["--flash-attn", cfg.flash_attn]
    if cfg.kv_cache_type:
        argv += ["--cache-type-k", cfg.kv_cache_type]
        argv += ["--cache-type-v", cfg.kv_cache_type]
    if cfg.moe_mode == "all":
        argv.append("--cpu-moe")             # boolean: ALL MoE weights on CPU
    elif cfg.moe_mode == "layers" and cfg.moe_layers:
        argv += ["--n-cpu-moe", str(cfg.moe_layers)]   # first N layers only

    # --- modern workflows ---------------------------------------------------
    argv.append("--jinja" if cfg.jinja else "--no-jinja")
    schema_text = cfg.json_schema.strip()
    if schema_text and schema_text != "null":
        if schema_file is not None:
            argv += ["--json-schema-file", str(schema_file)]
        else:
            argv += ["--json-schema", schema_text]

    # --- server-only --------------------------------------------------------
    if flavor is not BinaryFlavor.CLI:
        argv += ["--host", cfg.effective_host()]
        if cfg.port:
            argv += ["--port", str(cfg.port)]
        if cfg.api_key:
            argv += ["--api-key", cfg.api_key]

    argv += split_extra_args(cfg.extra_args)
    return argv


def build_command(
    cfg: LlamaConfig,
    flavor: Optional[BinaryFlavor] = None,
    schema_file: Optional[Path] = None,
) -> List[str]:
    program = resolve_executable(cfg.binary_path) or cfg.binary_path or "<llamacpp executable>"
    return [program] + build_argv(cfg, flavor, schema_file)


def plan_schema_file(cfg: LlamaConfig, directory: Path) -> Optional[Path]:
    """Decide whether the schema needs a file, and write it if so.

    Returns ``None`` when the schema is absent or small enough to inline.
    """
    text = cfg.json_schema.strip()
    if not text or text == "null":
        return None
    if len(text) <= INLINE_SCHEMA_LIMIT:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".schema.json", prefix="llamacpp-schema-",
        dir=directory, delete=False, encoding="utf-8",
    )
    try:
        handle.write(text)
    finally:
        handle.close()
    return Path(handle.name)


def generate_api_key() -> str:
    """A URL-safe random key for the HTTP API."""
    import secrets

    return "sk-" + secrets.token_hex(24)


# =========================================================================== #
# Subprocess control
# =========================================================================== #
class RunState(Enum):
    STOPPED = "Stopped"
    STARTING = "Starting"
    RUNNING = "Running"
    STOPPING = "Stopping"
    CRASHED = "Crashed"


class LlamaProcess(QObject):
    """Owns exactly one llama.cpp child process and its teardown policy.

    Teardown is multi-stage: a polite terminate first, then a forced process-tree
    kill after a grace period.  A llama.cpp server can hold gigabytes of VRAM, so
    a process that survives the UI is a real problem, not a cosmetic one.
    """

    output = pyqtSignal(str, bool)          # (text, is_stderr)
    state_changed = pyqtSignal(object)      # RunState
    ready = pyqtSignal(str)                 # detected API URL
    failed = pyqtSignal(str, str)           # (summary, detail)
    finished = pyqtSignal(int, str)         # (exit_code, status_text)

    def __init__(self, parent: Optional[QObject] = None, grace_ms: int = DEFAULT_GRACE_MS) -> None:
        super().__init__(parent)
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self._proc.readyReadStandardOutput.connect(self._drain_stdout)
        self._proc.readyReadStandardError.connect(self._drain_stderr)
        self._proc.started.connect(self._on_started)
        self._proc.errorOccurred.connect(self._on_error)
        self._proc.finished.connect(self._on_finished)

        self._isolate_process_group()

        self._state = RunState.STOPPED
        self._grace_ms = grace_ms
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._force_kill)

        # Set when *we* ask the child to stop, so a kill is not misreported as a
        # crash (a kill always surfaces as CrashExit with a non-zero code).
        self._stop_requested = False
        self._ready_seen = False
        self._program = ""
        self._args: List[str] = []

    # ------------------------------------------------------------- lifecycle #
    def _isolate_process_group(self) -> None:
        """Put the child in its own group/session where the platform allows.

        This lets teardown signal the whole tree, and stops a Ctrl+C aimed at the
        GUI from reaching the model server.
        """
        if os.name == "nt":
            return
        setter = getattr(self._proc, "setChildProcessModifier", None)
        if setter is None:
            return  # PyQt6 < 6.1: teardown falls back to the direct child
        try:
            setter(os.setsid)
        except (AttributeError, TypeError, ValueError):
            pass

    @property
    def state(self) -> RunState:
        return self._state

    @property
    def is_active(self) -> bool:
        """True while a child exists in any phase, including mid-shutdown."""
        return self._state in (RunState.STARTING, RunState.RUNNING, RunState.STOPPING)

    @property
    def can_start(self) -> bool:
        return self._state in (RunState.STOPPED, RunState.CRASHED)

    @property
    def pid(self) -> int:
        return int(self._proc.processId() or 0)

    def _set_state(self, state: RunState) -> None:
        if state is not self._state:
            self._state = state
            self.state_changed.emit(state)

    def start(self, program: str, args: Sequence[str], cwd: Optional[str] = None) -> None:
        """Launch *program* with *args*.

        Raises :class:`ProcessError` if a child is already active -- a
        double-start would orphan the first server on its port and in VRAM.
        """
        if not self.can_start:
            raise ProcessError("A llama.cpp process is already running. Stop it first.")

        program_path = Path(program)
        if not program_path.is_file():
            raise ProcessError(f"Executable not found:\n{program}")

        self._stop_requested = False
        self._ready_seen = False
        self._program = str(program_path)
        self._args = [str(a) for a in args]
        self._proc.setProgram(self._program)
        self._proc.setArguments(self._args)

        # Run from the binary's own directory so side-by-side DLLs / shared
        # libraries (CUDA, oneAPI, Vulkan) resolve.
        work_dir = cwd or str(program_path.parent)
        if work_dir and Path(work_dir).is_dir():
            self._proc.setWorkingDirectory(work_dir)

        env = QProcessEnvironment.systemEnvironment()
        env.insert("LLAMACPP_LAUNCHER", "1")
        self._proc.setProcessEnvironment(env)

        self._set_state(RunState.STARTING)
        self._proc.start()

    def stop(self) -> None:
        """Request a clean shutdown, escalating to a tree kill if needed."""
        if not self.is_active:
            return
        self._stop_requested = True
        self._set_state(RunState.STOPPING)
        self._kill_timer.start(self._grace_ms)
        self._terminate_tree(force=False)

    def kill(self) -> None:
        """Force-kill immediately and reap synchronously (used on window close)."""
        if not self.is_active:
            return
        self._stop_requested = True
        self._kill_timer.stop()
        self._terminate_tree(force=True)
        # Reaping synchronously is what makes kill() safe from closeEvent: when
        # it returns, the child is gone.
        if not self._proc.waitForFinished(3000):
            self._proc.kill()
            self._proc.waitForFinished(2000)

    def _terminate_tree(self, force: bool) -> None:
        pid = self.pid
        if pid <= 0:
            return
        if os.name != "nt":
            pgid = self._pgid_for(pid)
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL if force else signal.SIGTERM)
                    return
                except (ProcessLookupError, PermissionError, OSError):
                    pass  # fall through to the Qt-level path
        if force:
            self._proc.kill()
        else:
            self._proc.terminate()

    @staticmethod
    def _pgid_for(pid: int) -> Optional[int]:
        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError, OSError):
            return None
        # Only signal a group we created; signalling our own would kill the GUI.
        if pgid in (0, os.getpgrp()):
            return None
        return pgid

    def _force_kill(self) -> None:
        """Grace period expired: kill the tree, including grandchildren."""
        if not self.is_active:
            return
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(self.pid), "/T", "/F"],
                    capture_output=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                pass
        if self._proc.state() != QProcess.ProcessState.NotRunning:
            self._terminate_tree(force=True)
        self._proc.kill()

    # ---------------------------------------------------------------- signals #
    def _drain_stdout(self) -> None:
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.output.emit(data, False)
            self._inspect_for_ready(data)

    def _drain_stderr(self) -> None:
        data = bytes(self._proc.readAllStandardError()).decode("utf-8", errors="replace")
        if data:
            self.output.emit(data, True)
            self._inspect_for_ready(data)

    def _inspect_for_ready(self, chunk: str) -> None:
        if self._ready_seen:
            return
        for line in chunk.splitlines():
            url = parse_ready_url(line)
            if url:
                self._ready_seen = True
                self._set_state(RunState.RUNNING)
                self.ready.emit(url)
                return

    def _on_started(self) -> None:
        # The process exists, so RUNNING is accurate; llama-server additionally
        # emits `ready` with the real URL once it is listening.
        if self._state is RunState.STARTING:
            self._set_state(RunState.RUNNING)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._kill_timer.stop()
            self._set_state(RunState.STOPPED)
            self.failed.emit(
                "Could not start llamacpp",
                f"The executable could not be launched:\n{self._program}\n\n"
                "Check that the file is a real executable and that any required "
                "runtime libraries (CUDA/Vulkan) sit next to it.",
            )

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._kill_timer.stop()
        stop_requested = self._stop_requested
        self._stop_requested = False

        # A kill we asked for reports CrashExit with a platform-specific code on
        # every OS (negative on Windows), so this flag -- not the exit status --
        # decides whether the exit was a failure.
        if stop_requested:
            self._set_state(RunState.STOPPED)
            self.finished.emit(exit_code, "stopped")
            return

        if exit_status == QProcess.ExitStatus.CrashExit or exit_code != 0:
            self._set_state(RunState.CRASHED)
            self.failed.emit(
                "llamacpp exited unexpectedly",
                f"Exit code {exit_code} "
                f"({'crashed' if exit_status == QProcess.ExitStatus.CrashExit else 'non-zero exit'}).\n\n"
                "See the terminal pane for the last output.",
            )
        else:
            self._set_state(RunState.STOPPED)
        self.finished.emit(exit_code, "exited")


# =========================================================================== #
# UI: parameter widgets
# =========================================================================== #
def _mouse_wheel_guard(widget: QWidget) -> None:
    """Stop spin boxes/sliders hijacking the scroll area's wheel."""
    widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    widget.wheelEvent = lambda event: event.ignore()  # type: ignore[method-assign]


class _LinkedSlider:
    """Two-way binding between a QSlider and a QSpinBox."""

    def __init__(self, slider: QSlider, spin: QSpinBox, low: int, high: int, on_change) -> None:
        self.slider, self.spin = slider, spin
        self._syncing = False
        self._on_change = on_change
        slider.setRange(low, high)
        spin.setRange(low, high)
        _mouse_wheel_guard(slider)
        _mouse_wheel_guard(spin)
        slider.valueChanged.connect(self._from_slider)
        spin.valueChanged.connect(self._from_spin)

    def _from_slider(self, value: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.spin.setValue(value)
        finally:
            self._syncing = False
        self._on_change()

    def _from_spin(self, value: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.slider.setValue(value)
        finally:
            self._syncing = False
        self._on_change()

    def set_value(self, value: int) -> None:
        """Move both widgets, notifying listeners only if the value changed."""
        before = self.value()
        self._syncing = True
        try:
            self.spin.setValue(value)
            self.slider.setValue(value)
        finally:
            self._syncing = False
        if self.value() != before:
            self._on_change()

    def value(self) -> int:
        return self.spin.value()


class _CappedSpin(QSpinBox):
    """A spin box capped at a width that still fits its widest value.

    The cap is derived from the widest text the range allows (with headroom for
    real fonts and thousands separators), then clamped to a sensible band.  A
    hard-coded cap narrower than the text is what causes digits to be clipped,
    which silently misreports settings like a 1,048,576 context size.
    """

    MIN_WIDTH = 76
    # 150px comfortably fits 8 digits plus the spin arrows; the wider end of the
    # range (a 10,000,000 custom context) must not be clipped.
    MAX_WIDTH = 150

    def __init__(self, low: int, high: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # setRange() applies the width cap through the override below, so the
        # range must actually be set here -- sizing for a range the widget does
        # not have would leave it stuck at Qt's 0..99 default.
        self.setRange(low, high)

    def _apply_cap(self, low: Optional[int] = None, high: Optional[int] = None) -> None:
        low = self.minimum() if low is None else low
        high = self.maximum() if high is None else high
        metrics = self.fontMetrics()
        widest = max(
            metrics.horizontalAdvance(self.locale().toString(value))
            for value in (high, low)
        )
        # Headroom for font substitution and grouping separators.
        needed = int(widest * 1.5) + 26
        self.setMaximumWidth(max(self.MIN_WIDTH, min(self.MAX_WIDTH, needed)))

    def setRange(self, low: int, high: int) -> None:  # noqa: N802 (Qt override)
        super().setRange(low, high)
        if self.fontMetrics().horizontalAdvance("0"):
            self._apply_cap(low, high)


class ContextSizeSelector(QWidget):
    """A dropdown of standard context sizes, plus a Custom escape hatch.

    Requested sizes: 1k, 2k, 4k, 8k ... 1M.  A preset list alone cannot
    represent a value loaded from a profile, so selecting Custom reveals a spin
    box -- otherwise loading a profile with e.g. 6144 would silently round it.
    """

    changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._loading = False
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.combo = QComboBox()
        for value, label in CONTEXT_PRESETS:
            self.combo.addItem(label, value)
        self.combo.addItem("Custom…", None)
        self.combo.setToolTip("Context window (-c / --ctx-size).")
        row.addWidget(self.combo, 1)

        self.spin = _CappedSpin(CONTEXT_PRESETS[0][0], 10_000_000)
        self.spin.setSingleStep(512)
        self.spin.setVisible(False)
        _mouse_wheel_guard(self.spin)
        row.addWidget(self.spin)

        self.combo.currentIndexChanged.connect(self._on_combo)
        self.spin.valueChanged.connect(self._on_spin)
        # Tagged for ParameterForm.set_locked(); the internal widgets are managed
        # by _on_combo, so we do not tag them individually.
        ParameterForm._editable(self)

    def _on_combo(self, _index: int) -> None:
        preset = self.combo.currentData()
        custom = preset is None
        self.spin.setVisible(custom)
        if not custom and not self._loading:
            self.spin.setValue(int(preset))
            self.changed.emit()
        elif custom:
            self.spin.setFocus()

    def _on_spin(self, _value: int) -> None:
        if not self._loading:
            # Keep the label honest when a custom value matches a preset.
            self.changed.emit()

    def value(self) -> int:
        preset = self.combo.currentData()
        return int(preset) if preset is not None else self.spin.value()

    def set_value(self, value: int) -> None:
        self._loading = True
        try:
            index = self.combo.findData(value)
            if index >= 0:
                self.combo.setCurrentIndex(index)
                self.spin.setVisible(False)
            else:
                self.combo.setCurrentIndex(self.combo.count() - 1)  # Custom…
                self.spin.setValue(max(self.spin.minimum(), value))
                self.spin.setVisible(True)
        finally:
            self._loading = False


class ParameterForm(QWidget):
    """All launch parameters, bound to a :class:`LlamaConfig`.

    Advanced options live in a QTabWidget so the common path (model + core
    parameters) stays uncluttered.
    """

    changed = pyqtSignal()
    binary_changed = pyqtSignal()
    request_open_web_ui = pyqtSignal()
    request_generate_api_key = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._loading = False
        #: True while a process is running; drives set_locked().
        self._locked = False
        self._build()

    # ------------------------------------------------------------------- ui #
    @staticmethod
    def _editable(widget: QWidget) -> QWidget:
        """Tag a widget as a launch parameter so set_locked() can freeze it.

        Untagged widgets (the Web UI button, copy/clear helpers) stay clickable
        while the server runs.
        """
        widget.setProperty("editable", True)
        return widget

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_core_tab(), "Model && Core")
        self.tabs.addTab(self._build_hardware_tab(), "Hardware")
        self.tabs.addTab(self._build_server_tab(), "Server && Network")
        self.tabs.addTab(self._build_workflow_tab(), "Workflow")
        root.addWidget(self.tabs, 1)
        self.tabs.setCurrentIndex(0)

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _labelled(text: str) -> QLabel:
        return QLabel(text)

    @staticmethod
    def _row(*widgets: QWidget, stretch: int = 1) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        for index, widget in enumerate(widgets):
            row.addWidget(widget, stretch if index == 0 else 0)
        return container

    def _scrollable(self, inner: QWidget) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(inner)
        return scroll

    @staticmethod
    def _hint(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #8b909a; font-size: 11px;")
        return label

    # -- tab 1: model & core ----------------------------------------------- #
    def _build_core_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        files = QGroupBox("Executable && model")
        form = QFormLayout(files)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.binary_edit = self._editable(QLineEdit())
        self.binary_edit.setPlaceholderText("Path to llama-server.exe or llama-cli.exe")
        self.binary_edit.setClearButtonEnabled(True)
        binary_button = self._editable(QPushButton("Browse…"))
        binary_button.clicked.connect(self._pick_binary)
        form.addRow("llamacpp binary:", self._row(self.binary_edit, binary_button))

        self.flavor_label = QLabel("flavour: unknown")
        self.flavor_label.setStyleSheet("color: #8b909a; font-size: 11px;")
        form.addRow("", self.flavor_label)

        self.model_edit = self._editable(QLineEdit())
        self.model_edit.setPlaceholderText("Path to a .gguf model file")
        self.model_edit.setClearButtonEnabled(True)
        model_button = self._editable(QPushButton("Browse…"))
        model_button.clicked.connect(self._pick_model)
        form.addRow("Model (.gguf):", self._row(self.model_edit, model_button))
        outer.addWidget(files)

        core = QGroupBox("Core parameters")
        grid = QGridLayout(core)
        grid.setColumnStretch(1, 1)

        self.ngl_slider = self._editable(QSlider(Qt.Orientation.Horizontal))
        self.ngl_spin = self._editable(_CappedSpin(0, 1000))
        self.ngl = _LinkedSlider(self.ngl_slider, self.ngl_spin, 0, 1000, self._emit_changed)
        grid.addWidget(QLabel("GPU layers (<code>-ngl</code>):"), 0, 0)
        grid.addWidget(self._row(self.ngl_slider, self.ngl_spin), 0, 1)

        self.ctx = ContextSizeSelector()
        grid.addWidget(QLabel("Context size (<code>-c</code>):"), 1, 0)
        grid.addWidget(self.ctx, 1, 1)

        self.threads_slider = self._editable(QSlider(Qt.Orientation.Horizontal))
        self.threads_spin = self._editable(_CappedSpin(1, 256))
        self.threads = _LinkedSlider(self.threads_slider, self.threads_spin, 1, 256, self._emit_changed)
        grid.addWidget(QLabel("CPU threads (<code>-t</code>):"), 2, 0)
        grid.addWidget(self._row(self.threads_slider, self.threads_spin), 2, 1)

        self.tb_check, self.tb_spin = self._optional_spin("Batch threads (-tb):", 1, 1024)
        grid.addWidget(self.tb_check, 3, 0)
        grid.addWidget(self.tb_spin, 3, 1)

        self.batch_check, self.batch_spin = self._optional_spin("Batch size (-b):", 1, 1_000_000)
        grid.addWidget(self.batch_check, 4, 0)
        grid.addWidget(self.batch_spin, 4, 1)

        self.mlock_check = self._editable(QCheckBox("Lock model in RAM (--mlock)"))
        self.no_mmap_check = self._editable(QCheckBox("Disable mmap (--no-mmap)"))
        for box in (self.mlock_check, self.no_mmap_check):
            box.stateChanged.connect(self._emit_changed)
        flags = QHBoxLayout()
        flags.addWidget(self.mlock_check)
        flags.addWidget(self.no_mmap_check)
        flags.addStretch(1)
        grid.addLayout(flags, 5, 0, 1, 2)
        outer.addWidget(core)
        outer.addStretch(1)

        self.binary_edit.textChanged.connect(self._emit_binary_changed)
        self.model_edit.textChanged.connect(self._emit_changed)
        return self._scrollable(page)

    def _optional_spin(self, label: str, low: int, high: int):
        check = self._editable(QCheckBox(label))
        spin = self._editable(_CappedSpin(low, high))
        spin.setEnabled(False)
        _mouse_wheel_guard(spin)
        check.toggled.connect(self._emit_changed)
        check.toggled.connect(lambda: self.refresh_optional_states())
        spin.valueChanged.connect(self._emit_changed)
        return check, spin

    # -- tab 2: hardware --------------------------------------------------- #
    def _build_hardware_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        accel = QGroupBox("GPU acceleration")
        accel_form = QFormLayout(accel)
        accel_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.flash_attn_combo = self._editable(QComboBox())
        # 'auto' is llama.cpp's default, so it round-trips to "no flag sent".
        for value, label in (
            ("auto", "auto (llama.cpp default)"),
            ("on", "on (-fa on)"),
            ("off", "off (-fa off)"),
        ):
            self.flash_attn_combo.addItem(label, value)
        self.flash_attn_combo.setToolTip(
            "Flash Attention (-fa). Requires a supported GPU backend; on older "
            "GPUs 'on' can fail outright, which is why 'auto' is the default."
        )
        self.flash_attn_combo.currentIndexChanged.connect(self._emit_changed)
        accel_form.addRow("Flash attention:", self.flash_attn_combo)

        self.moe_mode_combo = self._editable(QComboBox())
        for value, label in (
            ("off", "Off — keep MoE weights on GPU"),
            ("all", "All MoE experts on CPU (--cpu-moe)"),
            ("layers", "First N layers on CPU (--n-cpu-moe N)"),
        ):
            self.moe_mode_combo.addItem(label, value)
        self.moe_mode_combo.setToolTip(
            "Mixture-of-Experts offloading. Note these are two *different* "
            "llama.cpp flags: --cpu-moe is a switch for all experts, while "
            "--n-cpu-moe takes a layer count."
        )
        self.moe_mode_combo.currentIndexChanged.connect(self._on_moe_mode)
        accel_form.addRow("MoE offloading:", self.moe_mode_combo)

        self.moe_spin = self._editable(_CappedSpin(1, 1000))
        self.moe_spin.setEnabled(False)
        _mouse_wheel_guard(self.moe_spin)
        self.moe_spin.valueChanged.connect(self._emit_changed)
        accel_form.addRow("MoE layers N:", self.moe_spin)

        accel_form.addRow("", self._hint(
            "MoE offloading only affects Mixture-of-Experts models (Qwen-MoE, "
            "Mixtral, DeepSeek…). It is ignored for dense models."
        ))
        outer.addWidget(accel)

        kv = QGroupBox("KV cache quantisation")
        kv_form = QFormLayout(kv)
        kv_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.kv_combo = self._editable(QComboBox())
        self.kv_combo.addItem("model default (f16)", "")
        for value in KV_CACHE_TYPES:
            self.kv_combo.addItem(f"{value}", value)
        self.kv_combo.setToolTip(
            "Applies to both --cache-type-k and --cache-type-v. Quantising the "
            "KV cache cuts VRAM sharply at long context, at some quality cost. "
            "q8_0 is the usual safe choice; needs Flash Attention for best results."
        )
        self.kv_combo.currentIndexChanged.connect(self._emit_changed)
        kv_form.addRow("KV cache type:", self.kv_combo)
        kv_form.addRow("", self._hint(
            "Sets both K and V. To use different types per side, add an override "
            "in Extra args on the Workflow tab (it is applied last, so it wins)."
        ))
        outer.addWidget(kv)
        outer.addStretch(1)
        return self._scrollable(page)

    def _on_moe_mode(self, _index: int) -> None:
        self.refresh_optional_states()
        self._emit_changed()

    # -- tab 3: server & network ------------------------------------------- #
    def _build_server_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        net = QGroupBox("Network binding")
        net_form = QFormLayout(net)
        net_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.lan_check = self._editable(QCheckBox("Expose on the local network (--host 0.0.0.0)"))
        self.lan_check.setToolTip(
            "Binds every network interface instead of loopback only, so other "
            "devices on your LAN can reach the API and web UI."
        )
        self.lan_check.stateChanged.connect(self._on_lan_toggled)
        net_form.addRow(self.lan_check)

        self.localhost_check = QCheckBox("Loopback only (127.0.0.1)")
        self.localhost_check.setChecked(True)
        self.localhost_check.setEnabled(False)
        self.localhost_check.setToolTip("The inverse of the option above.")
        net_form.addRow(self.localhost_check)

        self.host_edit = self._editable(QLineEdit())
        self.host_edit.setPlaceholderText("127.0.0.1")
        self.host_edit.setMaximumWidth(200)
        self.host_edit.textChanged.connect(self._emit_changed)
        net_form.addRow("Host:", self.host_edit)

        self.port_spin = self._editable(_CappedSpin(1, 65535))
        _mouse_wheel_guard(self.port_spin)
        self.port_spin.valueChanged.connect(self._emit_changed)
        net_form.addRow("Port:", self.port_spin)
        outer.addWidget(net)

        security = QGroupBox("Access control")
        sec_form = QFormLayout(security)
        sec_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.api_key_edit = self._editable(QLineEdit())
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("(empty = no authentication)")
        self.api_key_edit.setClearButtonEnabled(True)
        self.api_key_edit.textChanged.connect(self._emit_changed)
        # Adding or clearing a key changes whether the 0.0.0.0 warning still
        # applies, so the hint has to follow the field, not just the checkbox.
        self.api_key_edit.textChanged.connect(lambda _text: self._refresh_exposure_hint())
        generate_button = self._editable(QPushButton("Generate"))
        generate_button.setToolTip("Create a random API key and put it in the field.")
        generate_button.clicked.connect(self.request_generate_api_key.emit)
        sec_form.addRow("API key (--api-key):", self._row(self.api_key_edit, generate_button))

        self.show_key_check = self._editable(QCheckBox("Show key"))
        self.show_key_check.toggled.connect(
            lambda shown: self.api_key_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if shown else QLineEdit.EchoMode.Password
            )
        )
        sec_form.addRow("", self.show_key_check)

        self.exposure_label = QLabel("")
        self.exposure_label.setWordWrap(True)
        self.exposure_label.setStyleSheet("color: #e0a63c; font-size: 11px;")
        sec_form.addRow("", self.exposure_label)
        outer.addWidget(security)

        web = QGroupBox("Web UI")
        web_layout = QVBoxLayout(web)
        # Deliberately NOT tagged editable: the whole point of this button is to
        # be clickable *while* the server runs, so set_locked() must leave it alone.
        self.web_button = QPushButton("🌐  Open Web UI in browser")
        self.web_button.setMinimumHeight(34)
        self.web_button.setEnabled(False)
        self.web_button.clicked.connect(self.request_open_web_ui.emit)
        web_layout.addWidget(self.web_button)

        self.auto_open_check = self._editable(
            QCheckBox("Open automatically when the server is ready")
        )
        self.auto_open_check.stateChanged.connect(self._emit_changed)
        web_layout.addWidget(self.auto_open_check)

        self.web_hint = QLabel("")
        self.web_hint.setWordWrap(True)
        self.web_hint.setStyleSheet("color: #8b909a; font-size: 11px;")
        web_layout.addWidget(self.web_hint)
        outer.addWidget(web)

        outer.addStretch(1)
        self._refresh_exposure_hint()
        return self._scrollable(page)

    def _on_lan_toggled(self, checked: bool) -> None:
        self.localhost_check.setChecked(not checked)
        self.host_edit.setEnabled(not checked and not self._locked)
        if checked:
            self.host_edit.setText("0.0.0.0")
        elif self.host_edit.text() == "0.0.0.0":
            self.host_edit.setText("127.0.0.1")
        self._refresh_exposure_hint()
        self._emit_changed()

    def _refresh_exposure_hint(self) -> None:
        if self.lan_check.isChecked():
            has_key = bool(self.api_key_edit.text().strip())
            self.exposure_label.setText(
                "⚠  Bound to 0.0.0.0: reachable by every device on your network. "
                + ("An API key is set, so clients must authenticate."
                   if has_key else
                   "No API key is set — anyone on the network can use this model. "
                   "Use ‘Generate’ to create one.")
            )
        else:
            self.exposure_label.setText(
                "Loopback only: reachable from this machine alone. No API key needed."
            )

    # -- tab 4: workflow --------------------------------------------------- #
    def _build_workflow_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        templates = QGroupBox("Chat templates")
        tmpl_layout = QVBoxLayout(templates)
        self.jinja_check = self._editable(QCheckBox("Enable Jinja chat templates (--jinja)"))
        self.jinja_check.setToolTip(
            "llama-server enables Jinja by default, so this stays explicit in the "
            "command line either way (--jinja / --no-jinja)."
        )
        self.jinja_check.stateChanged.connect(self._emit_changed)
        tmpl_layout.addWidget(self.jinja_check)
        tmpl_layout.addWidget(self._hint(
            "Controls how the model's chat template is applied. Disabling it falls "
            "back to the older, more limited template handling."
        ))
        outer.addWidget(templates)

        schema = QGroupBox("Structured output — JSON schema")
        schema_layout = QVBoxLayout(schema)
        self.schema_check = self._editable(QCheckBox("Constrain generation to a JSON schema"))
        self.schema_check.setToolTip(
            "Passed as --json-schema (or --json-schema-file for large schemas, to "
            "avoid the Windows command-line length limit)."
        )
        self.schema_check.stateChanged.connect(self._on_schema_toggled)
        schema_layout.addWidget(self.schema_check)

        self.schema_edit = self._editable(QPlainTextEdit())
        self.schema_edit.setPlaceholderText(
            '{\n'
            '  "type": "object",\n'
            '  "properties": {\n'
            '    "name": { "type": "string" },\n'
            '    "age":  { "type": "integer" }\n'
            '  },\n'
            '  "required": ["name"]\n'
            '}'
        )
        self.schema_edit.setFont(self._mono_font())
        self.schema_edit.setMinimumHeight(160)
        self.schema_edit.textChanged.connect(self._on_schema_text_changed)
        schema_layout.addWidget(self.schema_edit, 1)

        schema_buttons = QHBoxLayout()
        format_button = self._editable(QPushButton("Format / validate"))
        format_button.clicked.connect(self._format_schema)
        schema_buttons.addWidget(format_button)
        load_button = self._editable(QPushButton("Load from file…"))
        load_button.clicked.connect(self._load_schema_file)
        schema_buttons.addWidget(load_button)
        clear_button = self._editable(QPushButton("Clear"))
        clear_button.clicked.connect(lambda: self.schema_edit.setPlainText(""))
        schema_buttons.addWidget(clear_button)
        schema_buttons.addStretch(1)
        schema_layout.addLayout(schema_buttons)

        self.schema_status = QLabel("")
        self.schema_status.setWordWrap(True)
        self.schema_status.setStyleSheet("font-size: 11px; color: #8b909a;")
        schema_layout.addWidget(self.schema_status)
        outer.addWidget(schema, 1)

        extras = QGroupBox("Extra arguments")
        extras_form = QFormLayout(extras)
        self.extra_edit = self._editable(QLineEdit())
        self.extra_edit.setPlaceholderText('e.g. --temp 0.7 --cache-type-k q8_0')
        self.extra_edit.setClearButtonEnabled(True)
        self.extra_edit.textChanged.connect(self._emit_changed)
        extras_form.addRow("Extra args:", self.extra_edit)
        extras_form.addRow("", self._hint(
            "Appended last, so these override every field above. Passed directly to "
            "the process — no shell is involved, so quotes are handled safely."
        ))
        outer.addWidget(extras)
        return self._scrollable(page)

    def _on_schema_toggled(self, checked: bool) -> None:
        self.schema_edit.setEnabled(checked)
        self.schema_edit.setVisible(True)
        self._refresh_schema_status()
        self._emit_changed()

    def _on_schema_text_changed(self) -> None:
        self._refresh_schema_status()
        self._emit_changed()

    def _refresh_schema_status(self) -> None:
        """Live validation feedback so a bad schema never reaches llama.cpp."""
        if not self.schema_check.isChecked():
            self.schema_status.setText("Schema constraint disabled.")
            self.schema_status.setStyleSheet("font-size: 11px; color: #8b909a;")
            return
        text = self.schema_edit.toPlainText().strip()
        if not text or text == "null":
            self.schema_status.setText("Empty schema — no constraint will be sent.")
            self.schema_status.setStyleSheet("font-size: 11px; color: #8b909a;")
            return
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            self.schema_status.setText(f"✗ Invalid JSON: line {exc.lineno}, column {exc.colno} — {exc.msg}")
            self.schema_status.setStyleSheet("font-size: 11px; color: #ff9b9b;")
            return
        if not isinstance(parsed, (dict, bool)):
            self.schema_status.setText("✗ Schema must be a JSON object (or true/false).")
            self.schema_status.setStyleSheet("font-size: 11px; color: #ff9b9b;")
            return
        size = len(text)
        if size > INLINE_SCHEMA_LIMIT:
            self.schema_status.setText(
                f"✓ Valid JSON object ({size:,} chars) — will be passed via "
                f"--json-schema-file to stay under the command-line limit."
            )
        else:
            self.schema_status.setText(f"✓ Valid JSON object ({size:,} chars).")
        self.schema_status.setStyleSheet("font-size: 11px; color: #3ec46d;")

    def _format_schema(self) -> None:
        text = self.schema_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Nothing to format", "The schema box is empty.")
            return
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            QMessageBox.warning(
                self, "Invalid JSON",
                f"Could not parse the schema.\n\nLine {exc.lineno}, column {exc.colno}:\n{exc.msg}",
            )
            return
        self.schema_edit.setPlainText(json.dumps(parsed, indent=2))
        self._refresh_schema_status()

    def _load_schema_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load JSON schema", str(Path.home()), "JSON files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Could not read file", str(exc))
            return
        self.schema_edit.setPlainText(text)
        self.schema_check.setChecked(True)

    # -- file pickers ------------------------------------------------------ #
    def _pick_binary(self) -> None:
        pattern = "Executables (*.exe *.bin *);;All files (*)" if os.name == "nt" else "All files (*)"
        path, _ = QFileDialog.getOpenFileName(
            self, "Select the llamacpp executable", self._start_dir(self.binary_edit.text()), pattern
        )
        if path:
            self.binary_edit.setText(path)

    def _pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a GGUF model", self._start_dir(self.model_edit.text()),
            "GGUF models (*.gguf);;All files (*)",
        )
        if path:
            self.model_edit.setText(path)

    @staticmethod
    def _start_dir(current: str) -> str:
        if current:
            parent = Path(current).expanduser().parent
            if parent.is_dir():
                return str(parent)
        return str(Path.home())

    @staticmethod
    def _mono_font() -> QFont:
        family = "Consolas" if "Consolas" in QFontDatabase.families() else "Monospace"
        return QFont(family, 9)

    # -- signals ----------------------------------------------------------- #
    def _emit_changed(self, *_args: Any) -> None:
        if not self._loading:
            self.changed.emit()

    def _emit_binary_changed(self, *_args: Any) -> None:
        self.flavor_label.setText(f"flavour: {self._flavor_text()}")
        if not self._loading:
            self.binary_changed.emit()
            self.changed.emit()

    def _flavor_text(self) -> str:
        flavor = detect_flavor(self.binary_edit.text())
        if flavor is BinaryFlavor.UNKNOWN:
            return "unknown (server flags will be applied)"
        return flavor.value

    # -- config binding ---------------------------------------------------- #
    def load_config(self, cfg: LlamaConfig) -> None:
        """Push *cfg* into the widgets without emitting change signals."""
        self._loading = True
        try:
            self.binary_edit.setText(cfg.binary_path)
            self.model_edit.setText(cfg.model_path)
            self.ngl.set_value(cfg.n_gpu_layers)
            self.ctx.set_value(cfg.ctx_size)
            self.threads.set_value(cfg.threads)
            self.tb_check.setChecked(cfg.threads_batch is not None)
            if cfg.threads_batch is not None:
                self.tb_spin.setValue(cfg.threads_batch)
            self.batch_check.setChecked(cfg.batch_size is not None)
            if cfg.batch_size is not None:
                self.batch_spin.setValue(cfg.batch_size)
            self.mlock_check.setChecked(cfg.mlock)
            self.no_mmap_check.setChecked(cfg.no_mmap)

            self._set_combo_data(self.flash_attn_combo, cfg.flash_attn or "auto")
            self._set_combo_data(self.moe_mode_combo, cfg.moe_mode or "off")
            if cfg.moe_layers:
                self.moe_spin.setValue(cfg.moe_layers)
            self.moe_spin.setEnabled((cfg.moe_mode or "off") == "layers")
            self._set_combo_data(self.kv_combo, cfg.kv_cache_type or "")

            self.lan_check.setChecked(cfg.bind_all_interfaces)
            self.localhost_check.setChecked(not cfg.bind_all_interfaces)
            self.host_edit.setEnabled(not cfg.bind_all_interfaces)
            self.host_edit.setText(cfg.host)
            self.port_spin.setValue(cfg.port)
            self.api_key_edit.setText(cfg.api_key)

            self.jinja_check.setChecked(cfg.jinja)
            self.schema_edit.setPlainText(cfg.json_schema)
            self.schema_check.setChecked(bool(cfg.json_schema.strip()))
            self.schema_edit.setEnabled(self.schema_check.isChecked())
            self.extra_edit.setText(cfg.extra_args)

            self.auto_open_check.setChecked(cfg.auto_open_browser)
            self.flavor_label.setText(f"flavour: {self._flavor_text()}")
            self._refresh_exposure_hint()
            self._refresh_schema_status()
            self.refresh_optional_states()
        finally:
            self._loading = False

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: Any) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def refresh_optional_states(self) -> None:
        self.tb_spin.setEnabled(self.tb_check.isChecked() and not self._locked)
        self.batch_spin.setEnabled(self.batch_check.isChecked() and not self._locked)
        self.moe_spin.setEnabled(
            self.moe_mode_combo.currentData() == "layers" and not self._locked
        )

    def set_locked(self, locked: bool) -> None:
        """Freeze parameter *inputs* while a process runs.

        Deliberately not ``QWidget.setEnabled(False)`` on the whole form: that
        recursively disables every child, and re-enabling the form afterwards
        would also re-enable the Web UI button while the server is still
        starting -- so the button could never be gated correctly.  Only widgets
        explicitly tagged ``editable`` are touched.
        """
        self._locked = locked
        for widget in self.findChildren(QWidget):
            if widget.property("editable"):
                widget.setEnabled(not locked)

        # Host is disabled by the LAN checkbox, not just by locking.
        if not locked:
            self.host_edit.setEnabled(not self.lan_check.isChecked())
            self.schema_edit.setEnabled(self.schema_check.isChecked())
        # These are derived from their checkboxes, so re-apply the relationship.
        self.refresh_optional_states()

    def apply_to(self, cfg: LlamaConfig) -> LlamaConfig:
        """Copy widget state into *cfg* and return it."""
        cfg.binary_path = self.binary_edit.text().strip()
        cfg.model_path = self.model_edit.text().strip()
        cfg.n_gpu_layers = self.ngl.value()
        cfg.ctx_size = self.ctx.value()
        cfg.threads = self.threads.value()
        cfg.threads_batch = self.tb_spin.value() if self.tb_check.isChecked() else None
        cfg.batch_size = self.batch_spin.value() if self.batch_check.isChecked() else None
        cfg.mlock = self.mlock_check.isChecked()
        cfg.no_mmap = self.no_mmap_check.isChecked()

        cfg.flash_attn = self.flash_attn_combo.currentData() or "auto"
        cfg.moe_mode = self.moe_mode_combo.currentData() or "off"
        cfg.moe_layers = self.moe_spin.value() if cfg.moe_mode == "layers" else None
        cfg.kv_cache_type = self.kv_combo.currentData() or ""

        cfg.bind_all_interfaces = self.lan_check.isChecked()
        cfg.host = "0.0.0.0" if cfg.bind_all_interfaces else (self.host_edit.text().strip() or "127.0.0.1")
        cfg.port = self.port_spin.value()
        cfg.api_key = self.api_key_edit.text().strip()

        cfg.jinja = self.jinja_check.isChecked()
        cfg.json_schema = self.schema_edit.toPlainText().strip() if self.schema_check.isChecked() else ""
        cfg.extra_args = self.extra_edit.text().strip()
        cfg.auto_open_browser = self.auto_open_check.isChecked()

        cfg.normalize()
        return cfg


# =========================================================================== #
# UI: terminal pane
# =========================================================================== #
class TerminalPane(QWidget):
    """Streams process output into a bounded, append-only text view."""

    IMPORTANT_RE = re.compile(
        r"(error|failed|fatal|exception|out of memory|oom|listening|loaded|"
        r"warning|abort|assert)",
        re.IGNORECASE,
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("Output")
        title.setStyleSheet("font-weight: 600;")
        header.addWidget(title)
        self.status_hint = QLabel("")
        self.status_hint.setStyleSheet("color: #8b909a; font-size: 11px;")
        header.addWidget(self.status_hint)
        header.addStretch(1)

        self.autoscroll_check = QCheckBox("Auto-scroll")
        self.autoscroll_check.setChecked(True)
        header.addWidget(self.autoscroll_check)
        self.timestamp_check = QCheckBox("Timestamps")
        header.addWidget(self.timestamp_check)
        copy_button = QPushButton("Copy all")
        copy_button.clicked.connect(self.copy_all)
        header.addWidget(copy_button)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear)
        header.addWidget(clear_button)
        root.addLayout(header)

        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setUndoRedoEnabled(False)
        self.view.setMaximumBlockCount(MAX_LOG_BLOCKS)
        self.view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.view.setFont(ParameterForm._mono_font())
        self.view.setStyleSheet(
            "QPlainTextEdit { background: #12141a; color: #d6dae3;"
            " border: 1px solid #2a2e3a; border-radius: 4px; padding: 6px; }"
        )
        self.view.setPlaceholderText("llamacpp output will stream here once the process starts.")
        root.addWidget(self.view, 1)

    # --------------------------------------------------------------- output #
    def append(self, text: str, is_stderr: bool = False) -> None:
        """Append a chunk of output, preserving lines and colouring stderr.

        ``insertText`` is used rather than ``insertHtml`` on purpose: Qt's HTML
        subset collapses bare newlines, which would fuse the whole log into one
        paragraph.  Formatting is applied afterwards via ``mergeCharFormat``.
        """
        if not text:
            return
        self._keep_at_end()
        cursor = self.view.textCursor()
        cursor.beginEditBlock()
        try:
            for line in text.splitlines():
                self._insert_line(cursor, line, is_stderr)
        finally:
            cursor.endEditBlock()
        self._view_settled()

    def append_notice(self, text: str, color: str = "#7fd4ff") -> None:
        self._keep_at_end()
        cursor = self.view.textCursor()
        cursor.beginEditBlock()
        try:
            self._insert_line(cursor, f"── {text} ──", False, override_color=color)
        finally:
            cursor.endEditBlock()
        self._view_settled()

    def _insert_line(
        self, cursor: QTextCursor, line: str, is_stderr: bool, override_color: Optional[str] = None
    ) -> None:
        cursor.movePosition(QTextCursor.MoveOperation.End)
        prefix = ""
        if self.timestamp_check.isChecked() and line.strip():
            prefix = f"{datetime.now():%H:%M:%S} "
        cursor.insertText(f"{prefix}{line}\n")
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(override_color or self._line_color(line, is_stderr)))
        cursor.mergeCharFormat(fmt)
        cursor.movePosition(QTextCursor.MoveOperation.End)

    @classmethod
    def _line_color(cls, line: str, is_stderr: bool) -> str:
        if cls.IMPORTANT_RE.search(line):
            return "#ffd479" if not is_stderr else "#ff9b9b"
        return "#ff9b9b" if is_stderr else "#d6dae3"

    def _keep_at_end(self) -> None:
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.view.setTextCursor(cursor)

    def _view_settled(self) -> None:
        if self.autoscroll_check.isChecked():
            self.view.moveCursor(QTextCursor.MoveOperation.End)
            self.view.ensureCursorVisible()

    # ------------------------------------------------------------------ api #
    def clear(self) -> None:
        self.view.clear()

    def copy_all(self) -> None:
        QApplication.clipboard().setText(self.view.toPlainText())

    def set_hint(self, text: str) -> None:
        self.status_hint.setText(text)

    def text(self) -> str:
        return self.view.toPlainText()


# =========================================================================== #
# UI: status indicator
# =========================================================================== #
class StatusIndicator(QWidget):
    """A coloured dot plus a label -- the Running/Stopped readout."""

    COLORS = {
        RunState.STOPPED: ("#7a8290", "Stopped"),
        RunState.STARTING: ("#e0a63c", "Starting…"),
        RunState.RUNNING: ("#3ec46d", "Running"),
        RunState.STOPPING: ("#e0a63c", "Stopping…"),
        RunState.CRASHED: ("#e5534b", "Crashed"),
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._dot = QLabel()
        self._dot.setFixedSize(12, 12)
        row.addWidget(self._dot)
        self._text = QLabel("Stopped")
        self._text.setStyleSheet("font-weight: 600;")
        row.addWidget(self._text)
        self._detail = QLabel("")
        self._detail.setStyleSheet("color: #8b909a; font-size: 11px;")
        row.addWidget(self._detail)
        self.set_state(RunState.STOPPED)

    def set_state(self, state: RunState, detail: str = "") -> None:
        color, label = self.COLORS.get(state, ("#7a8290", state.value))
        self._dot.setStyleSheet(
            f"background-color: {color}; border-radius: 6px; min-width: 12px; min-height: 12px;"
        )
        self._text.setText(label)
        self._detail.setText(detail)


# =========================================================================== #
# Main window
# =========================================================================== #
class MainWindow(QMainWindow):
    """Wires the parameter form, the process controller, the log pane and profiles."""

    def __init__(self, store: Optional[ProfileStore] = None) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1240, 820)
        self.setMinimumSize(940, 640)

        self.store = store or ProfileStore()
        self.config = LlamaConfig()
        self.process = LlamaProcess(self)
        self.current_profile: Optional[Profile] = None
        self._dirty = False
        self._last_cfg_path = ""
        self._schema_file: Optional[Path] = None
        self._listening_url: Optional[str] = None

        self._build_ui()
        self._build_menu()
        self._connect_process()
        self._connect_form()

        self._load_startup_profile()
        self._refresh_preview()

    # ------------------------------------------------------------------- ui #
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        outer.addWidget(self._build_profile_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_form_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([520, 660])
        outer.addWidget(splitter, 1)

        outer.addWidget(self._build_control_bar())

    def _build_profile_bar(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Profile:"))
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(220)
        row.addWidget(self.profile_combo)

        for label, slot in (
            ("Load", self._on_load_selected),
            ("Save preset…", self._on_save_preset),
            ("Delete", self._on_delete_selected),
            ("Import…", self._on_import),
            ("Export…", self._on_export),
        ):
            button = QPushButton(label)
            button.clicked.connect(slot)
            row.addWidget(button)
        self.delete_button = row.itemAt(3).widget()

        row.addStretch(1)
        self.dirty_label = QLabel("")
        self.dirty_label.setStyleSheet("color: #e0a63c; font-size: 11px;")
        row.addWidget(self.dirty_label)
        return bar

    def _build_form_panel(self) -> QWidget:
        self.form = ParameterForm()
        self.form.setMinimumWidth(460)
        return self.form

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        column = QVBoxLayout(panel)
        column.setContentsMargins(8, 0, 0, 0)
        column.setSpacing(8)

        header = QHBoxLayout()
        label = QLabel("Command preview")
        label.setStyleSheet("font-weight: 600;")
        header.addWidget(label)
        header.addStretch(1)
        copy_button = QPushButton("Copy")
        copy_button.setToolTip("Copy the exact command line that will run.")
        copy_button.clicked.connect(
            lambda: QApplication.clipboard().setText(self.command_view.toPlainText())
        )
        header.addWidget(copy_button)
        column.addLayout(header)

        self.command_view = QPlainTextEdit()
        self.command_view.setReadOnly(True)
        self.command_view.setFixedHeight(96)
        self.command_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.command_view.setFont(ParameterForm._mono_font())
        self.command_view.setStyleSheet(
            "QPlainTextEdit { background: #1b1e26; color: #9fe8b5;"
            " border: 1px solid #2a2e3a; border-radius: 4px; padding: 6px; }"
        )
        column.addWidget(self.command_view)

        self.terminal = TerminalPane()
        column.addWidget(self.terminal, 1)
        return panel

    def _build_control_bar(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        self.start_button = QPushButton("▶  Start")
        self.start_button.setMinimumHeight(34)
        self.start_button.setMinimumWidth(130)
        self.start_button.setStyleSheet(
            "QPushButton { background:#1f7a45; color:white; font-weight:600;"
            " border-radius:5px; padding:6px 16px; }"
            "QPushButton:disabled { background:#3a3f4a; color:#8b909a; }"
            "QPushButton:hover:enabled { background:#26914f; }"
        )
        self.start_button.clicked.connect(self._on_start)
        row.addWidget(self.start_button)

        self.stop_button = QPushButton("■  Stop")
        self.stop_button.setMinimumHeight(34)
        self.stop_button.setMinimumWidth(130)
        self.stop_button.setStyleSheet(
            "QPushButton { background:#9c3b33; color:white; font-weight:600;"
            " border-radius:5px; padding:6px 16px; }"
            "QPushButton:disabled { background:#3a3f4a; color:#8b909a; }"
            "QPushButton:hover:enabled { background:#b8473e; }"
        )
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop)
        row.addWidget(self.stop_button)

        self.status = StatusIndicator()
        row.addWidget(self.status)

        row.addStretch(1)
        self.uptime_label = QLabel("")
        self.uptime_label.setStyleSheet("color: #8b909a; font-size: 11px;")
        row.addWidget(self.uptime_label)

        self._uptime_timer = QTimer(self)
        self._uptime_timer.setInterval(1000)
        self._uptime_timer.timeout.connect(self._tick_uptime)
        self._started_at: Optional[datetime] = None
        return bar

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        save_action = QAction("&Save preset…", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._on_save_preset)
        file_menu.addAction(save_action)
        load_action = QAction("&Load preset…", self)
        load_action.setShortcut(QKeySequence.StandardKey.Open)
        load_action.triggered.connect(self._on_import)
        file_menu.addAction(load_action)
        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        run_menu = self.menuBar().addMenu("&Run")
        self._start_action = QAction("&Start", self)
        self._start_action.setShortcut("F5")
        self._start_action.triggered.connect(self._on_start)
        run_menu.addAction(self._start_action)
        self._stop_action = QAction("S&top", self)
        self._stop_action.setShortcut("Shift+F5")
        self._stop_action.triggered.connect(self._on_stop)
        run_menu.addAction(self._stop_action)
        run_menu.addSeparator()
        self._web_action = QAction("Open &Web UI", self)
        self._web_action.setShortcut("Ctrl+B")
        self._web_action.setEnabled(False)
        self._web_action.triggered.connect(self._on_open_web_ui)
        run_menu.addAction(self._web_action)

        view_menu = self.menuBar().addMenu("&View")
        clear_action = QAction("&Clear output", self)
        clear_action.triggered.connect(lambda: self.terminal.clear())
        view_menu.addAction(clear_action)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    # --------------------------------------------------------------- wiring #
    def _connect_form(self) -> None:
        self.form.changed.connect(self._on_form_changed)
        self.form.binary_changed.connect(self._on_binary_changed)
        self.form.request_open_web_ui.connect(self._on_open_web_ui)
        self.form.request_generate_api_key.connect(self._on_generate_api_key)

    def _connect_process(self) -> None:
        self.process.output.connect(self.terminal.append)
        self.process.state_changed.connect(self._on_state_changed)
        self.process.ready.connect(self._on_server_ready)
        self.process.failed.connect(self._on_process_failed)
        self.process.finished.connect(self._on_process_finished)

    # ------------------------------------------------------- config handling #
    def _on_form_changed(self) -> None:
        self.form.apply_to(self.config)
        self._mark_dirty(True)
        self._refresh_preview()
        self._refresh_web_button()

    def _on_binary_changed(self) -> None:
        self._refresh_web_button()

    def _refresh_preview(self) -> None:
        flavor = detect_flavor(self.config.binary_path)
        self.command_view.setPlainText(format_command(build_command(self.config, flavor)))

    def _mark_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        self.dirty_label.setText("• unsaved changes" if dirty else "")

    def _refresh_web_button(self) -> None:
        """The Web UI button is only meaningful, and only safe, once listening."""
        is_server = detect_flavor(self.config.binary_path) is not BinaryFlavor.CLI
        running = self.process.state is RunState.RUNNING
        listening = self._listening_url is not None

        enabled = bool(is_server and running and listening)
        self.form.web_button.setEnabled(enabled)
        self._web_action.setEnabled(enabled)

        if not is_server:
            hint = "The selected binary is llama-cli; it has no HTTP server or web UI."
        elif self.process.is_active and not listening:
            hint = "Waiting for the server to report it is listening…"
        elif enabled:
            hint = f"Ready: {self._web_url()}"
        else:
            hint = "Start llama-server to enable the web UI."
        self.form.web_hint.setText(hint)

    def _web_url(self) -> str:
        return browser_url_for(self.config.effective_host(), self.config.port, self._listening_url)

    # ------------------------------------------------------------- profiles #
    def _refresh_profile_list(self) -> None:
        previous = self.profile_combo.currentText()
        self.profile_combo.clear()
        for path in self.store.list_profiles():
            self.profile_combo.addItem(path.stem, str(path))
        if previous:
            index = self.profile_combo.findText(previous)
            if index >= 0:
                self.profile_combo.setCurrentIndex(index)
        self.delete_button.setEnabled(self.profile_combo.count() > 0)

    def _load_startup_profile(self) -> None:
        self._refresh_profile_list()
        candidate = default_profile_path()
        if not candidate.is_file():
            paths = self.store.list_profiles()
            if not paths:
                self.terminal.append_notice(
                    "No profiles yet — pick your llamacpp binary and model to begin."
                )
                return
            candidate = paths[0]
        try:
            self._apply_profile(self.store.load(candidate))
        except ConfigError as exc:
            self.terminal.append_notice(f"Could not load startup profile: {exc}", "#ff9b9b")

    def _apply_profile(self, profile: Profile) -> None:
        self.current_profile = profile
        self.config = profile.config
        self.form.load_config(self.config)
        index = self.profile_combo.findText(profile.name)
        if index >= 0:
            self.profile_combo.setCurrentIndex(index)
        self._mark_dirty(False)
        self._refresh_preview()
        self._refresh_web_button()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION} — {profile.name}")

    def _on_load_selected(self) -> None:
        path = self.profile_combo.currentData()
        if not path:
            QMessageBox.information(self, "No profiles", "There are no saved presets yet.")
            return
        if not self._confirm_discard():
            return
        try:
            self._apply_profile(self.store.load(Path(path)))
        except ConfigError as exc:
            QMessageBox.warning(self, "Could not load preset", str(exc))
            return
        self.terminal.append_notice(f"Loaded preset '{self.current_profile.name}'")

    def _on_save_preset(self) -> None:
        self.form.apply_to(self.config)
        suggested = (
            self.current_profile.name
            if self.current_profile
            else (Path(self.config.model_path).stem or "default")
        )
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:", text=suggested)
        if not ok or not name.strip():
            return
        name = name.strip()
        if self.store.path_for(name).exists():
            confirm = QMessageBox.question(
                self, "Overwrite preset?",
                f"A preset named '{name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
        profile = Profile(name=name, config=self.config)
        try:
            saved = self.store.save(profile)
        except ConfigError as exc:
            QMessageBox.warning(self, "Could not save preset", str(exc))
            return
        self.current_profile = profile
        self._mark_dirty(False)
        self._refresh_profile_list()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION} — {name}")
        self.terminal.append_notice(f"Saved preset to {saved}")

    def _on_delete_selected(self) -> None:
        path = self.profile_combo.currentData()
        if not path:
            return
        name = Path(path).stem
        confirm = QMessageBox.question(
            self, "Delete preset?", f"Delete the preset '{name}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self.store.delete(name)
        except ConfigError as exc:
            QMessageBox.warning(self, "Could not delete preset", str(exc))
            return
        if self.current_profile and self.current_profile.name == name:
            self.current_profile = None
            self._mark_dirty(True)
        self._refresh_profile_list()
        self.terminal.append_notice(f"Deleted preset '{name}'")

    def _on_import(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Import preset", self._last_cfg_path or str(Path.home()),
            f"Preset files (*{PROFILE_SUFFIX});;All files (*)",
        )
        if not path_str:
            return
        self._last_cfg_path = str(Path(path_str).parent)
        try:
            profile = self.store.load(Path(path_str))
        except ConfigError as exc:
            QMessageBox.warning(self, "Could not import preset", str(exc))
            return
        if not self._confirm_discard():
            return
        self._apply_profile(profile)
        self._mark_dirty(True)
        self.terminal.append_notice(f"Imported preset from {path_str}")

    def _on_export(self) -> None:
        self.form.apply_to(self.config)
        name = self.current_profile.name if self.current_profile else "preset"
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Export preset",
            str(Path(self._last_cfg_path or Path.home()) / f"{name}{PROFILE_SUFFIX}"),
            f"Preset files (*{PROFILE_SUFFIX})",
        )
        if not path_str:
            return
        self._last_cfg_path = str(Path(path_str).parent)
        try:
            saved = self.store.save(
                Profile(name=Path(path_str).stem, config=self.config), Path(path_str)
            )
        except ConfigError as exc:
            QMessageBox.warning(self, "Could not export preset", str(exc))
            return
        self.terminal.append_notice(f"Exported preset to {saved}")

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self, "Discard unsaved changes?",
            "The current parameters have unsaved changes. Discard them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------ execution #
    def _on_start(self) -> None:
        self.form.apply_to(self.config)
        if self.process.is_active:
            QMessageBox.information(self, "Already running", "A llamacpp process is already running.")
            return

        flavor = detect_flavor(self.config.binary_path)
        issues = validate(self.config, flavor)
        fatal = [i for i in issues if i.fatal]
        if fatal:
            QMessageBox.warning(
                self, "Cannot start llamacpp",
                "\n\n".join(f"• {i.message}" for i in fatal),
            )
            return

        # Non-fatal issue that needs explicit consent before we publish a model
        # server to the local network.
        for issue in issues:
            if not issue.fatal:
                if issue.field == "api_key":
                    proceed = QMessageBox.warning(
                        self, "Exposed without authentication",
                        issue.message + "\n\nStart anyway?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                        QMessageBox.StandardButton.Cancel,
                    )
                    if proceed != QMessageBox.StandardButton.Yes:
                        return
                else:
                    self.terminal.append_notice(f"Warning: {issue.message}", "#ffd479")

        program = resolve_executable(self.config.binary_path)
        if not program:
            QMessageBox.warning(self, "Cannot start llamacpp", "Executable could not be resolved.")
            return

        try:
            self._schema_file = plan_schema_file(self.config, config_dir() / "tmp")
        except (OSError, ConfigError) as exc:
            QMessageBox.warning(self, "Could not write schema file", str(exc))
            self._schema_file = None
            return

        argv = build_argv(self.config, flavor, self._schema_file)

        self.terminal.append_notice(
            f"Starting {Path(program).name} ({flavor.value}) at {datetime.now():%H:%M:%S}"
        )
        self.terminal.append_notice(format_command([program] + argv), "#9fe8b5")
        if self._schema_file:
            self.terminal.append_notice(
                f"JSON schema written to {self._schema_file} (passed via --json-schema-file)"
            )
        if self.config.bind_all_interfaces:
            self.terminal.append_notice(
                f"Binding to 0.0.0.0:{self.config.port} — reachable from your local network.",
                "#ffd479",
            )

        self._listening_url = None
        try:
            self.process.start(program, argv)
        except ProcessError as exc:
            QMessageBox.warning(self, "Cannot start llamacpp", str(exc))
            self.terminal.append_notice(f"Launch failed: {exc}", "#ff9b9b")
            self._cleanup_schema_file()

    def _on_stop(self) -> None:
        if not self.process.is_active:
            return
        self.terminal.append_notice("Stop requested — terminating llamacpp…", "#ffd479")
        self.process.stop()

    # -- web UI ------------------------------------------------------------ #
    def _on_open_web_ui(self) -> None:
        url = self._web_url()
        if not (self.process.state is RunState.RUNNING and self._listening_url):
            QMessageBox.information(
                self, "Server not ready",
                "The web UI becomes available once llama-server reports that it is "
                "listening. Start the server and wait for the “listening on” line "
                "in the output pane.",
            )
            return
        self.terminal.append_notice(f"Opening {url} in your default browser")
        if not QDesktopServices.openUrl(QUrl(url)):
            QMessageBox.warning(
                self, "Could not open browser",
                f"Qt could not hand the URL to your default browser.\n\nOpen it manually:\n{url}",
            )

    def _on_generate_api_key(self) -> None:
        key = generate_api_key()
        self.form.api_key_edit.setText(key)
        self.form.show_key_check.setChecked(True)
        QApplication.clipboard().setText(key)
        self.terminal.append_notice(
            "Generated a new API key (copied to clipboard). Clients must send "
            "'Authorization: Bearer <key>'."
        )

    # -- process signals --------------------------------------------------- #
    @pyqtSlot(object)
    def _on_state_changed(self, state: RunState) -> None:
        active = state in (RunState.STARTING, RunState.RUNNING, RunState.STOPPING)
        self.start_button.setEnabled(not active)
        self.stop_button.setEnabled(active)
        self._start_action.setEnabled(not active)
        self._stop_action.setEnabled(active)
        self.form.set_locked(active)

        detail = f"pid {self.process.pid}" if active and self.process.pid else ""
        self.status.set_state(state, detail)

        if state is RunState.RUNNING and self._started_at is None:
            self._started_at = datetime.now()
            self._uptime_timer.start()
        elif state in (RunState.STOPPED, RunState.CRASHED):
            self._uptime_timer.stop()
            self._started_at = None
            self.uptime_label.setText("")
            self._listening_url = None
        self._refresh_web_button()

    @pyqtSlot(str)
    def _on_server_ready(self, url: str) -> None:
        self._listening_url = url
        web_url = self._web_url()
        self.terminal.append_notice(f"llama-server is listening on {url}", "#3ec46d")
        self.terminal.set_hint(f"API: {url}")
        self.status.set_state(RunState.RUNNING, f"API {url}")
        self._refresh_web_button()
        if self.config.auto_open_browser:
            self._on_open_web_ui()

    @pyqtSlot(str, str)
    def _on_process_failed(self, summary: str, detail: str) -> None:
        self.terminal.append_notice(f"{summary}: {detail.splitlines()[0]}", "#ff9b9b")
        QMessageBox.warning(self, summary, detail)

    @pyqtSlot(int, str)
    def _on_process_finished(self, code: int, status: str) -> None:
        color = "#3ec46d" if code == 0 else "#ff9b9b"
        self.terminal.append_notice(
            f"llamacpp {status} (exit code {code}) at {datetime.now():%H:%M:%S}", color
        )
        self.terminal.set_hint("")
        self._listening_url = None
        self._cleanup_schema_file()
        self._refresh_web_button()

    def _cleanup_schema_file(self) -> None:
        """Delete the temporary schema file, if one was written.

        Safe to call repeatedly: it always clears the reference.
        """
        path, self._schema_file = self._schema_file, None
        if path is None:
            return
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass

    def _tick_uptime(self) -> None:
        if self._started_at is None:
            return
        elapsed = int((datetime.now() - self._started_at).total_seconds())
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.uptime_label.setText(f"uptime {hours:02d}:{minutes:02d}:{seconds:02d}")

    def _on_about(self) -> None:
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> {APP_VERSION}<br><br>"
            "A PyQt6 front-end for llamacpp's <code>llama-server</code> and "
            "<code>llama-cli</code>.<br><br>"
            f"Profiles are stored as JSON in:<br><code>{self.store.directory}</code>",
        )

    # ------------------------------------------------------------ shutdown #
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt override)
        """Never leave an orphaned model server holding VRAM after the GUI exits."""
        if self.process.is_active:
            answer = QMessageBox.question(
                self, "llamacpp is still running",
                "The model process is still running.\n\nStop it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._uptime_timer.stop()
            self.terminal.append_notice("Shutting down: killing llamacpp…", "#ffd479")
            self.process.kill()

        if self._dirty and self.current_profile is not None:
            answer = QMessageBox.question(
                self, "Save changes before quitting?",
                f"Save changes to preset '{self.current_profile.name}'?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.StandardButton.Save:
                try:
                    self.store.save(Profile(name=self.current_profile.name, config=self.config))
                except ConfigError as exc:
                    QMessageBox.warning(self, "Could not save preset", str(exc))
                    event.ignore()
                    return

        self._cleanup_schema_file()
        event.accept()


# =========================================================================== #
# Bootstrap
# =========================================================================== #
def _install_excepthook() -> None:
    """Show unhandled exceptions in a dialog instead of dying silently.

    A bare traceback on stderr is invisible when the app is launched from a
    desktop shortcut, which makes field reports useless.
    """

    def hook(exc_type, exc_value, exc_tb):
        import traceback

        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        sys.stderr.write(text)
        if QApplication.instance() is not None:
            QMessageBox.critical(
                None, "Unexpected error",
                f"{exc_type.__name__}: {exc_value}\n\n{text[-2000:]}",
            )

    sys.excepthook = hook


def _install_interrupt_polling(app: QApplication) -> QTimer:
    """Let Ctrl+C close the GUI.

    Qt's event loop blocks the interpreter, so Python signal handlers would not
    run until the loop returns; a trivial timer forces regular wake-ups, at
    which point a pending signal is delivered.
    """
    timer = QTimer(app)
    timer.setInterval(250)
    timer.timeout.connect(lambda: None)
    timer.start()
    return timer


def build_app(argv: Optional[Sequence[str]] = None) -> QApplication:
    """Create and configure the QApplication (or configure the existing one)."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(list(sys.argv if argv is None else argv))
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("llamacpp-launcher")

    if getattr(app, "_interrupt_timer", None) is None:
        timer = _install_interrupt_polling(app)
        app._interrupt_timer = timer  # type: ignore[attr-defined]
        app.aboutToQuit.connect(timer.stop)

    _install_excepthook()
    return app


def run(argv: Optional[Sequence[str]] = None) -> int:
    """Create the application, show the window, and return the exit code."""
    app = build_app(argv)
    window = MainWindow()
    window.show()
    return app.exec()


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
