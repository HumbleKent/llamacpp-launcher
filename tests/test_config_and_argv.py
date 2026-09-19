"""Pure-logic tests: config model, profiles, argv assembly, URL handling.

These import the single-file application module but never construct a QWidget,
so they run with no display and no Qt event loop.
"""
from __future__ import annotations

import json

import pytest

from llamacpp_launcher import (
    CONTEXT_PRESETS,
    KV_CACHE_TYPES,
    BinaryFlavor,
    ConfigError,
    LlamaConfig,
    Profile,
    ProfileStore,
    browser_url_for,
    build_argv,
    build_command,
    detect_flavor,
    generate_api_key,
    parse_ready_url,
    plan_schema_file,
    split_extra_args,
    validate,
)


# --------------------------------------------------------------------------- #
# Flavour detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "filename,expected",
    [
        ("llama-server", BinaryFlavor.SERVER),
        ("llama-server.exe", BinaryFlavor.SERVER),
        ("server.exe", BinaryFlavor.SERVER),
        ("llama-server-cuda12.exe", BinaryFlavor.SERVER),
        ("llama-cli", BinaryFlavor.CLI),
        ("main.exe", BinaryFlavor.CLI),
        ("llamacpp", BinaryFlavor.CLI),
        ("something-else", BinaryFlavor.UNKNOWN),
        ("", BinaryFlavor.UNKNOWN),
    ],
)
def test_detect_flavor(filename, expected):
    assert detect_flavor(filename) is expected


# --------------------------------------------------------------------------- #
# Core argv
# --------------------------------------------------------------------------- #
def _value_of(argv, flag):
    return argv[argv.index(flag) + 1]


def test_argv_contains_core_flags():
    cfg = LlamaConfig(
        binary_path="llama-server.exe", model_path="models/tiny.gguf",
        n_gpu_layers=35, ctx_size=8192, threads=6,
    )
    argv = build_argv(cfg, BinaryFlavor.SERVER)
    assert _value_of(argv, "--model") == "models/tiny.gguf"
    assert _value_of(argv, "--n-gpu-layers") == "35"
    assert _value_of(argv, "--ctx-size") == "8192"
    assert _value_of(argv, "--threads") == "6"


def test_server_flavor_gets_host_and_port():
    cfg = LlamaConfig(model_path="m.gguf", port=9090)
    argv = build_argv(cfg, BinaryFlavor.SERVER)
    assert _value_of(argv, "--host") == "127.0.0.1"
    assert _value_of(argv, "--port") == "9090"


def test_cli_flavor_omits_server_only_flags():
    cfg = LlamaConfig(
        model_path="m.gguf", bind_all_interfaces=True, port=8080, api_key="secret"
    )
    argv = build_argv(cfg, BinaryFlavor.CLI)
    assert "--host" not in argv
    assert "--port" not in argv
    assert "--api-key" not in argv


def test_extra_args_appended_last_so_they_win():
    cfg = LlamaConfig(
        model_path="m.gguf", ctx_size=4096,
        extra_args='--ctx-size 32768 --temp "0.7 1"',
    )
    argv = build_argv(cfg, BinaryFlavor.SERVER)
    assert argv.count("--ctx-size") == 2
    assert argv[-2:] == ["--temp", "0.7 1"]


# --------------------------------------------------------------------------- #
# Advanced hardware flags
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode,flag", [("on", "on"), ("off", "off")])
def test_flash_attention_explicit_values(mode, flag):
    argv = build_argv(LlamaConfig(model_path="m.gguf", flash_attn=mode), BinaryFlavor.SERVER)
    assert _value_of(argv, "--flash-attn") == flag


def test_flash_attention_auto_sends_no_flag():
    """'auto' is llama.cpp's own default, so the flag is omitted entirely."""
    argv = build_argv(LlamaConfig(model_path="m.gguf", flash_attn="auto"), BinaryFlavor.SERVER)
    assert "--flash-attn" not in argv


@pytest.mark.parametrize("cache", ["q8_0", "f16", "q4_0", "bf16"])
def test_kv_cache_quantisation_sets_both_k_and_v(cache):
    argv = build_argv(
        LlamaConfig(model_path="m.gguf", kv_cache_type=cache), BinaryFlavor.SERVER
    )
    assert _value_of(argv, "--cache-type-k") == cache
    assert _value_of(argv, "--cache-type-v") == cache


def test_kv_cache_omitted_when_unset():
    argv = build_argv(LlamaConfig(model_path="m.gguf"), BinaryFlavor.SERVER)
    assert "--cache-type-k" not in argv and "--cache-type-v" not in argv


def test_moe_all_uses_the_boolean_cpu_moe_flag():
    """--cpu-moe takes NO value; --n-cpu-moe takes a count. They are different."""
    argv = build_argv(
        LlamaConfig(model_path="m.gguf", moe_mode="all", moe_layers=99), BinaryFlavor.SERVER
    )
    assert "--cpu-moe" in argv
    assert "--n-cpu-moe" not in argv
    # The boolean flag must not be followed by a stray number.
    assert argv[argv.index("--cpu-moe") + 1] != "99"


def test_moe_layers_uses_n_cpu_moe_with_a_count():
    argv = build_argv(
        LlamaConfig(model_path="m.gguf", moe_mode="layers", moe_layers=12),
        BinaryFlavor.SERVER,
    )
    assert _value_of(argv, "--n-cpu-moe") == "12"
    assert "--cpu-moe" not in argv


def test_moe_off_sends_neither_flag():
    argv = build_argv(LlamaConfig(model_path="m.gguf", moe_mode="off"), BinaryFlavor.SERVER)
    assert "--cpu-moe" not in argv and "--n-cpu-moe" not in argv


# --------------------------------------------------------------------------- #
# Network binding
# --------------------------------------------------------------------------- #
def test_bind_all_interfaces_sets_host_0_0_0_0():
    cfg = LlamaConfig(model_path="m.gguf", bind_all_interfaces=True)
    cfg.normalize()
    argv = build_argv(cfg, BinaryFlavor.SERVER)
    assert _value_of(argv, "--host") == "0.0.0.0"


def test_api_key_passed_to_server():
    cfg = LlamaConfig(model_path="m.gguf", api_key="sk-abc123")
    argv = build_argv(cfg, BinaryFlavor.SERVER)
    assert _value_of(argv, "--api-key") == "sk-abc123"


def test_api_key_omitted_when_empty():
    argv = build_argv(LlamaConfig(model_path="m.gguf"), BinaryFlavor.SERVER)
    assert "--api-key" not in argv


def test_bind_toggle_and_host_field_cannot_disagree():
    """normalize() reconciles the checkbox with the host string in both directions."""
    cfg = LlamaConfig(host="0.0.0.0", bind_all_interfaces=False)
    cfg.normalize()
    assert cfg.bind_all_interfaces is True

    cfg = LlamaConfig(host="192.168.1.5", bind_all_interfaces=True)
    cfg.normalize()
    assert cfg.effective_host() == "0.0.0.0"


def test_localhost_binding_is_the_default():
    assert LlamaConfig().effective_host() == "127.0.0.1"


# --------------------------------------------------------------------------- #
# Web UI URL selection
# --------------------------------------------------------------------------- #
def test_browser_url_rewrites_wildcard_to_loopback():
    """0.0.0.0 is a bind address, not a connectable one."""
    assert browser_url_for("0.0.0.0", 8080) == "http://127.0.0.1:8080"
    assert browser_url_for("0.0.0.0", 9000) == "http://127.0.0.1:9000"


def test_browser_url_prefers_the_detected_listening_address():
    detected = "http://127.0.0.1:8080"
    assert browser_url_for("0.0.0.0", 9999, detected) == detected


def test_browser_url_rewrites_detected_wildcard():
    assert browser_url_for("0.0.0.0", 8080, "http://0.0.0.0:8080") == "http://127.0.0.1:8080"


def test_browser_url_respects_a_specific_host():
    assert browser_url_for("192.168.1.5", 8081) == "http://192.168.1.5:8081"


def test_parse_ready_url():
    line = "main: server is listening on http://127.0.0.1:8080 -- starting the main loop"
    assert parse_ready_url(line) == "http://127.0.0.1:8080"
    assert parse_ready_url("loading model") is None


# --------------------------------------------------------------------------- #
# Workflow flags: jinja + JSON schema
# --------------------------------------------------------------------------- #
def test_jinja_flag_sent_either_way():
    assert "--jinja" in build_argv(LlamaConfig(jinja=True), BinaryFlavor.SERVER)
    assert "--no-jinja" in build_argv(LlamaConfig(jinja=False), BinaryFlavor.SERVER)


def test_json_schema_inline_when_small():
    schema = json.dumps({"type": "object", "properties": {"a": {"type": "string"}}})
    argv = build_argv(
        LlamaConfig(model_path="m.gguf", json_schema=schema), BinaryFlavor.SERVER
    )
    assert _value_of(argv, "--json-schema") == schema
    assert "--json-schema-file" not in argv


def test_json_schema_file_used_when_supplied():
    schema = json.dumps({"type": "object"})
    argv = build_argv(
        LlamaConfig(model_path="m.gguf", json_schema=schema),
        BinaryFlavor.SERVER,
        schema_file="C:/tmp/x.schema.json",
    )
    assert _value_of(argv, "--json-schema-file") == "C:/tmp/x.schema.json"
    assert "--json-schema" not in argv


def test_json_schema_omitted_when_empty():
    argv = build_argv(LlamaConfig(model_path="m.gguf"), BinaryFlavor.SERVER)
    assert "--json-schema" not in argv and "--json-schema-file" not in argv


def test_plan_schema_file_writes_only_for_large_schemas(tmp_path):
    small = LlamaConfig(json_schema='{"type":"object"}')
    assert plan_schema_file(small, tmp_path) is None

    big = LlamaConfig(json_schema=json.dumps({"type": "object", "pad": "x" * 9000}))
    path = plan_schema_file(big, tmp_path)
    assert path is not None and path.exists()
    # The file must contain the schema verbatim so llama.cpp parses it.
    assert json.loads(path.read_text(encoding="utf-8"))["type"] == "object"


def test_plan_schema_file_absent_when_disabled(tmp_path):
    assert plan_schema_file(LlamaConfig(), tmp_path) is None
    assert plan_schema_file(LlamaConfig(json_schema="null"), tmp_path) is None


def test_generate_api_key_is_random_and_url_safe():
    first, second = generate_api_key(), generate_api_key()
    assert first != second
    assert first.startswith("sk-")
    assert len(first) > 20


# --------------------------------------------------------------------------- #
# Context size presets
# --------------------------------------------------------------------------- #
def test_context_presets_span_1k_to_1m():
    values = [value for value, _ in CONTEXT_PRESETS]
    assert values == sorted(values)
    assert 1024 in values
    assert 1048576 in values          # 1M
    labels = dict(CONTEXT_PRESETS)
    assert labels[1048576] == "1M"
    assert labels[1024] == "1k"
    assert labels[32768] == "32k"


def test_context_size_reaches_argv():
    argv = build_argv(LlamaConfig(model_path="m.gguf", ctx_size=1048576), BinaryFlavor.SERVER)
    assert _value_of(argv, "--ctx-size") == "1048576"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validate_flags_missing_binary_and_model():
    issues = validate(LlamaConfig())
    fields = {i.field for i in issues if i.fatal}
    assert "binary_path" in fields and "model_path" in fields


def test_validate_reports_invalid_json_schema_as_fatal():
    cfg = LlamaConfig(model_path="m.gguf", json_schema="{not json")
    issues = [i for i in validate(cfg, BinaryFlavor.SERVER) if i.field == "json_schema"]
    assert issues and issues[0].fatal


def test_validate_accepts_a_valid_schema():
    cfg = LlamaConfig(model_path="m.gguf", json_schema='{"type":"object"}')
    assert not [i for i in validate(cfg, BinaryFlavor.SERVER) if i.field == "json_schema"]


def test_validate_rejects_non_object_schema():
    cfg = LlamaConfig(model_path="m.gguf", json_schema="[1, 2, 3]")
    issues = [i for i in validate(cfg, BinaryFlavor.SERVER) if i.field == "json_schema"]
    assert issues and issues[0].fatal


def test_validate_warns_about_exposed_server_without_a_key():
    cfg = LlamaConfig(model_path="m.gguf", bind_all_interfaces=True)
    issues = [i for i in validate(cfg, BinaryFlavor.SERVER) if i.field == "api_key"]
    assert issues and not issues[0].fatal, "exposure must warn, not block"


def test_validate_no_exposure_warning_when_key_is_set():
    cfg = LlamaConfig(model_path="m.gguf", bind_all_interfaces=True, api_key="sk-x")
    assert not [i for i in validate(cfg, BinaryFlavor.SERVER) if i.field == "api_key"]


def test_validate_no_exposure_warning_on_loopback():
    cfg = LlamaConfig(model_path="m.gguf")
    assert not [i for i in validate(cfg, BinaryFlavor.SERVER) if i.field == "api_key"]


def test_validate_moe_layers_mode_needs_a_count():
    cfg = LlamaConfig(model_path="m.gguf", moe_mode="layers", moe_layers=None)
    issues = [i for i in validate(cfg, BinaryFlavor.SERVER) if i.field == "moe_layers"]
    assert issues and issues[0].fatal


def test_validate_rejects_unknown_kv_cache_type():
    cfg = LlamaConfig(model_path="m.gguf", kv_cache_type="q2_k")
    issues = [i for i in validate(cfg, BinaryFlavor.SERVER) if i.field == "kv_cache_type"]
    assert issues and issues[0].fatal


def test_every_advertised_kv_type_is_accepted():
    for cache in KV_CACHE_TYPES:
        cfg = LlamaConfig(model_path="m.gguf", kv_cache_type=cache)
        assert not [i for i in validate(cfg, BinaryFlavor.SERVER) if i.field == "kv_cache_type"]


# --------------------------------------------------------------------------- #
# Argument splitting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("", []),
        ("   ", []),
        ("--a 1 --b 2", ["--a", "1", "--b", "2"]),
        ('--path "C:\\Program Files\\m.gguf"', ["--path", "C:\\Program Files\\m.gguf"]),
        ("--path 'single quoted'", ["--path", "single quoted"]),
        ("--x a\\ b", ["--x", "a b"]),
        ("--flag", ["--flag"]),
    ],
)
def test_split_extra_args(text, expected):
    assert split_extra_args(text) == expected


def test_split_extra_args_rejects_unbalanced_quotes():
    with pytest.raises(ConfigError):
        split_extra_args('--temp "0.7')


# --------------------------------------------------------------------------- #
# Profile store
# --------------------------------------------------------------------------- #
def _full_config() -> LlamaConfig:
    cfg = LlamaConfig(
        binary_path="llama-server.exe", model_path="m.gguf",
        n_gpu_layers=48, ctx_size=65536, threads=12, threads_batch=8, batch_size=2048,
        mlock=True, no_mmap=False, flash_attn="on", kv_cache_type="q8_0",
        moe_mode="layers", moe_layers=10, host="0.0.0.0", port=9090,
        bind_all_interfaces=True, api_key="sk-test", jinja=True,
        json_schema='{"type":"object"}', extra_args="--temp 0.7",
        auto_open_browser=True,
    )
    cfg.normalize()
    return cfg


def test_profile_round_trip_preserves_every_new_field(tmp_path):
    store = ProfileStore(tmp_path)
    cfg = _full_config()
    saved = store.save(Profile(name="full", config=cfg))
    loaded = store.load(saved).config

    for name in (
        "flash_attn", "kv_cache_type", "moe_mode", "moe_layers",
        "bind_all_interfaces", "api_key", "jinja", "json_schema",
        "auto_open_browser", "ctx_size",
    ):
        assert getattr(loaded, name) == getattr(cfg, name), name


def test_profile_name_is_sanitised(tmp_path):
    saved = ProfileStore(tmp_path).save(Profile(name="my preset", config=LlamaConfig()))
    assert saved.name == "my_preset.json"


def test_schema_goes_to_a_sidecar_not_the_profile(tmp_path):
    """Profiles stay hand-readable: the schema body lives in its own file."""
    store = ProfileStore(tmp_path)
    schema = '{"type":"object","title":"SIDECAR_MARKER"}'
    saved = store.save(Profile(name="p", config=LlamaConfig(json_schema=schema)))

    sidecar = store.schema_path_for(saved)
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text(encoding="utf-8"))["title"] == "SIDECAR_MARKER"

    profile_data = json.loads(saved.read_text(encoding="utf-8"))
    assert profile_data["config"]["json_schema"] == "", "schema body must not be inlined"
    assert "SIDECAR_MARKER" not in saved.read_text(encoding="utf-8")

    assert store.load(saved).config.json_schema == schema


def test_sidecar_removed_when_schema_cleared(tmp_path):
    store = ProfileStore(tmp_path)
    saved = store.save(Profile(name="p", config=LlamaConfig(json_schema='{"a":1}')))
    assert store.schema_path_for(saved).is_file()
    store.save(Profile(name="p", config=LlamaConfig(json_schema="")))
    assert not store.schema_path_for(saved).exists()


def test_delete_removes_the_schema_sidecar(tmp_path):
    store = ProfileStore(tmp_path)
    store.save(Profile(name="p", config=LlamaConfig(json_schema='{"a":1}')))
    assert store.delete("p") is True
    assert list(tmp_path.glob("*")) == []


def test_saved_profile_is_readable_json(tmp_path):
    store = ProfileStore(tmp_path)
    path = store.save(Profile(name="p", config=LlamaConfig(ctx_size=1234)))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["config"]["ctx_size"] == 1234
    assert data["schema_version"] == 2


def test_load_rejects_malformed_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        ProfileStore(tmp_path).load(bad)


def test_load_rejects_future_schema(tmp_path):
    future = tmp_path / "future.json"
    future.write_text(json.dumps({"schema_version": 99, "config": {}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="newer"):
        ProfileStore(tmp_path).load(future)


def test_v1_profile_still_loads(tmp_path):
    """The MVP wrote schema_version 1; those profiles must keep working."""
    old = tmp_path / "old.json"
    old.write_text(
        json.dumps({
            "schema_version": 1, "name": "old",
            "config": {"ctx_size": 8192, "n_gpu_layers": 30, "flash_attn": True},
        }),
        encoding="utf-8",
    )
    cfg = ProfileStore(tmp_path).load(old).config
    assert cfg.ctx_size == 8192
    assert cfg.n_gpu_layers == 30
    # v1 stored flash_attn as a bool; normalize() must not crash on it.
    assert cfg.flash_attn == "auto"


def test_unknown_keys_are_ignored(tmp_path):
    path = tmp_path / "extra.json"
    path.write_text(
        json.dumps({"schema_version": 2, "config": {"ctx_size": 4096, "future_flag": True}}),
        encoding="utf-8",
    )
    profile = ProfileStore(tmp_path).load(path)
    assert profile.config.ctx_size == 4096
    assert not hasattr(profile.config, "future_flag")


def test_hand_edited_garbage_values_are_normalised(tmp_path):
    path = tmp_path / "garbage.json"
    path.write_text(
        json.dumps({
            "schema_version": 2,
            "config": {
                "ctx_size": "not a number", "threads": -5, "port": 999999,
                "flash_attn": "banana", "kv_cache_type": "q2_k", "moe_mode": "sideways",
            },
        }),
        encoding="utf-8",
    )
    cfg = ProfileStore(tmp_path).load(path).config
    assert cfg.ctx_size == 4096      # falls back to the default
    assert cfg.threads == 1          # clamped low
    assert cfg.port == 65535         # clamped high
    assert cfg.flash_attn == "auto"
    assert cfg.kv_cache_type == ""
    assert cfg.moe_mode == "off"


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    store = ProfileStore(tmp_path)
    store.save(Profile(name="p", config=LlamaConfig()))
    assert list(tmp_path.glob("*.tmp")) == []


def test_store_lists_and_deletes(tmp_path):
    store = ProfileStore(tmp_path)
    store.save(Profile(name="a", config=LlamaConfig()))
    store.save(Profile(name="b", config=LlamaConfig()))
    assert [p.stem for p in store.list_profiles()] == ["a", "b"]
    assert store.delete("a") is True
    assert store.delete("a") is False
    assert [p.stem for p in store.list_profiles()] == ["b"]


def test_build_command_resolves_and_prepends_the_program(tmp_path):
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"stub")
    cfg = LlamaConfig(binary_path=str(exe), model_path="m.gguf", ctx_size=1024)
    command = build_command(cfg, BinaryFlavor.SERVER)
    assert command[0] == str(exe)
    assert "--ctx-size" in command
