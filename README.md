# llamacpp Launcher

A single-file **PyQt6 desktop GUI** for driving llama.cpp's command-line tools
(`llama-server` and `llama-cli`) â€” GPU offloading, KV-cache quantisation, MoE
offloading, LAN binding and structured-output schemas, without hand-editing long
shell invocations or hunting for the last good set of flags.

Everything lives in **`llamacpp_launcher.py`** (~2,600 lines). No package to
install, no source tree to navigate: run the one file.

---

## Features

The UI is organised into four tabs so the common path stays uncluttered.

### Model & Core
| Control | Flag |
| --- | --- |
| llama.cpp binary + `.gguf` model pickers | (flavour auto-detected from the binary name) |
| GPU layers (slider + spin box) | `-ngl` / `--n-gpu-layers` |
| **Context size dropdown** â€” 512, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, 256k, 512k, 1M, plus **Customâ€¦** | `-c` / `--ctx-size` |
| CPU threads (slider + spin box) | `-t` / `--threads` |
| Batch threads / batch size (optional) | `-tb`, `-b` |
| Lock in RAM / disable mmap | `--mlock`, `--no-mmap` |

*Customâ€¦* exists because a fixed preset list cannot represent a value loaded
from a profile: selecting it reveals a spin box, so a profile holding e.g. 6144
is not silently rounded to 8k.

### Hardware
| Control | Flag |
| --- | --- |
| Flash attention: `auto` / `on` / `off` | `-fa` / `--flash-attn` |
| KV cache quantisation: `f16`, `bf16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0`, `iq4_nl`, `f32` | `-ctk` / `-ctv` (sets both K and V) |
| **MoE offloading**: Off / all experts / first *N* layers | `-cmoe` **or** `-ncmoe N` |

Two details worth calling out, both verified against a real build:

* **`--cpu-moe` and `--n-cpu-moe` are different flags.** `-cmoe`/`--cpu-moe` is a
  *boolean* switch that moves **all** MoE weights to CPU; the numeric form is
  `-ncmoe`/`--n-cpu-moe N`, which moves the first *N* layers' experts. The UI
  offers them as mutually exclusive modes because passing a number to the
  boolean flag is a parse error.
* `-fa auto` is llama.cpp's own default, so selecting *auto* sends no flag at
  all rather than a redundant one.

### Server & Network
| Control | Flag |
| --- | --- |
| **Expose on the local network** â€” binds every interface | `--host 0.0.0.0` |
| Loopback only (the default) | `--host 127.0.0.1` |
| Host / port | `--host`, `--port` |
| API key, with **Generate** (random, copied to clipboard) | `--api-key` |
| **Open Web UI in browser** + auto-open when ready | (no flag â€” opens the detected URL) |

**About `0.0.0.0`.** Binding every interface publishes the API *and the web UI*
to your whole LAN, and modern llama.cpp does **not** require authentication by
default. So the exposure checkbox does three things: it shows a live warning
when no key is set, and starting an unauthenticated exposed server requires
explicit confirmation. With a key set, it starts without nagging.

The **Web UI** button is enabled only once the server actually reports it is
listening (`server is listening on â€¦`), and it rewrites a wildcard bind address
to loopback â€” `http://0.0.0.0:8080` is a bind address, not something a browser
can open, so the button would otherwise open a dead tab.

### Workflow
| Control | Flag |
| --- | --- |
| Jinja chat templates | `--jinja` / `--no-jinja` |
| JSON schema for structured output (with live validation) | `-j` / `-jf` |
| Extra arguments, appended last so they override everything | (verbatim) |

The schema box validates as you type (JSON error with line/column) and has
*Format / validate* and *Load from fileâ€¦*. Schemas over ~8 KB are written to a
temporary file and passed via `--json-schema-file`, because Windows caps a whole
command line at ~32 KB and schemas routinely exceed that; the temp file is
deleted when the process exits.

---

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe llamacpp_launcher.py
#    ...or double-click run.bat
```

Linux/macOS: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
then `.venv/bin/python llamacpp_launcher.py`.

You also need a llama.cpp build â€” point the binary picker at `llama-server` (for
the web UI / API) or `llama-cli` (interactive prompt). The binary's flavour is
detected from its filename, and server-only flags are withheld from the CLI.

---

## Flags, verified

Every flag this app generates was checked against **llama.cpp build 10262
(f26efa02a)**, and an integration test feeds the full generated argument vector
to a real binary and asserts the argument parser accepted it (a rejected flag
surfaces as `error while handling argument`).

One caveat: `--mlock` and `--no-mmap` are **deprecated** in current builds in
favour of `--load-mode {none,mmap,mlock,mmap+mlock,dio}`. They still work and
only log a deprecation warning, and they are kept because they remain valid on
older builds. All the flags added here (`-fa`, `-ctk/-ctv`, `-cmoe`, `-ncmoe`,
`--jinja`, `-j/-jf`, `--host`, `--port`, `--api-key`) are current and
non-deprecated.

---

## Profiles

Presets are plain JSON in the per-user config directory:

| Platform | Location |
| --- | --- |
| Windows | `%APPDATA%\llamacpp-launcher\profiles\` |
| Linux | `$XDG_CONFIG_HOME/llamacpp-launcher/profiles/` |
| macOS | `~/Library/Application Support/llamacpp-launcher/profiles/` |

```json
{
  "schema_version": 2,
  "name": "qwen3-moe-lan",
  "config": {
    "binary_path": "C:/llama.cpp/llama-server.exe",
    "model_path": "D:/models/Qwen3-30B-A3B-Q4_K_M.gguf",
    "n_gpu_layers": 99,
    "ctx_size": 65536,
    "threads": 12,
    "flash_attn": "on",
    "kv_cache_type": "q8_0",
    "moe_mode": "layers",
    "moe_layers": 24,
    "host": "0.0.0.0",
    "port": 8080,
    "bind_all_interfaces": true,
    "api_key": "sk-â€¦",
    "jinja": true,
    "json_schema": ""
  }
}
```

A JSON schema is stored in a **sidecar** file next to the preset
(`mypreset.schema.json`) rather than embedded, so presets stay short and
hand-readable. `Importâ€¦`/`Exportâ€¦` move presets anywhere on disk for sharing.
`default.json` loads automatically at startup. v1 profiles from earlier builds
still load.

---

## How the subprocess handling stays safe

Losing track of a model server is expensive: a mislaid `llama-server` keeps
gigabytes of VRAM allocated and its TCP port bound, and the next launch then
fails with a confusing "address already in use".

1. **Refuses a second launch** â€” `start()` raises if a child is active, so the
   previous server cannot be silently abandoned.
2. **Uses `QProcess`, not `subprocess`** â€” the child is driven by the Qt event
   loop: no reader threads, stdout/stderr read on separate channels so stderr
   can be colourised.
3. **Isolates the child** â€” on POSIX it gets its own session
   (`os.setsid` via `setChildProcessModifier`), so the whole tree can be
   signalled and a Ctrl+C aimed at the GUI never reaches the model server. On
   Windows `taskkill /T /F` handles the tree.
4. **Escalates on shutdown** â€” polite terminate first, then a forced process-group
   kill after a 4 s grace period.
5. **Kills synchronously on exit** â€” `closeEvent` confirms, calls `kill()`, and
   blocks on `waitForFinished`, so nothing survives the GUI. Critically, `kill()`
   still acts while a stop is already in flight; an earlier version returned
   early there and could orphan a process mid-shutdown.
6. **Never uses a shell** â€” the argv list goes straight to the process, and
   `extra_args` is split by an internal quote-aware tokeniser with no globbing or
   variable expansion.
7. **Runs the child from its own directory** so side-by-side CUDA/Vulkan DLLs
   resolve.
8. **Bounds the log buffer** â€” an unbounded terminal pane is a slow memory leak.
9. **Distinguishes a stop from a crash** â€” a killed process reports `CrashExit`
   with a platform-specific code, so a user-requested stop is not reported as a
   failure.

---

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install pytest pytest-timeout
$env:QT_QPA_PLATFORM="offscreen"
.\.venv\Scripts\python.exe -m pytest
```

127 tests: argv construction, profile persistence and URL selection headlessly;
the real subprocess lifecycle against `sys.executable` as a stand-in child
(output streaming, graceful stop, force-killing a `SIGTERM`-ignoring child,
double-start rejection, crash vs. stop reporting, and a no-orphan assertion that
every spawned PID is dead afterwards); GUI state transitions; and integration
tests against a real llama.cpp build when one is present.

Two safety nets in `tests/conftest.py` are deliberate:

* **Modals are stubbed out session-wide.** `QMessageBox.exec()` spins a native
  modal loop waiting for a human click; offscreen there is nobody to click it, so
  any test reaching a dialog hangs forever and cannot be interrupted by
  faulthandler. Every dialog entry point is replaced with a recording stub, so
  tests still assert that a dialog was raised.
* **A hard per-test timeout** (`timeout = 120` in `pyproject.toml`) is the only
  thing that reliably kills a wedge inside a C call.

---

## Known limitations

* No chat/inference client â€” the API URL is opened in a browser instead.
* llama-cli runs interactively; the pane is read-only, so prompts must be driven
  from the server API. A stdin writer would be the natural next step.
* The KV-cache dropdown sets K and V together. Use *Extra args* to set them
  differently (`--cache-type-k q8_0 --cache-type-v f16`) â€” extra args win.
* `--mlock` / `--no-mmap` are deprecated upstream; `--load-mode` is not surfaced.
* No GPU-layer auto-detection from the GGUF header, so `-ngl` is still a guess
  the first time (the log reports how many layers were actually offloaded).
