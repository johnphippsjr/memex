"""Board #1038: an ingest that produces no symbols for a language it had files for
must FAIL, not exit 0.

This is the gate that would have caught the original defect on the first progress
line. The real run saw 4,773 files, wrote 142 symbols with 0 call edges across 734
discarded files, and exited 0 - so every watcher and human read it as success.

Each case below is written so it FAILS against the pre-fix code (which had no
per-extension counters and no exit gate at all).
"""
import pytest

from memex.ingest_history import (
    IngestStats, _CODE_EXTS, assert_not_hollow, capability_report, LANGUAGE_CAPABILITIES,
    is_build_artifact,
)


# --- board #1046: build-artifact exclusion (86% of smokesignals full-history symbols were
#     dist/*.bundle.js garbage before this) --------------------------------------------------

def test_build_artifacts_are_excluded():
    for p in (
        "dist/main.b16695b6.bundle.js",          # the exact smokesignals-web offender
        "dist/vendor.js",
        "node_modules/react/index.js",
        "build/static/js/app.js",
        "src/app.min.js",
        "public/bundle.min.js",
        "packages/x/node_modules/y/z.ts",
        "coverage/lcov-report/x.js",
    ):
        assert is_build_artifact(p), f"{p} should be excluded as a build artifact"


def test_real_source_is_not_excluded():
    for p in (
        "src/app.tsx",
        "src/pages/auth/Register.tsx",
        "lib/dist-utils/helper.ts",              # 'dist-utils' is not 'dist' (exact segment)
        "src/my-vendor.ts",                      # 'my-vendor' is not 'vendor'
        "memex/ingest_history.py",
        "components/Button.jsx",
    ):
        assert not is_build_artifact(p), f"{p} is real source and must NOT be excluded"


def _stats(files_by_ext, symbols_by_ext):
    s = IngestStats()
    s.files_by_ext = dict(files_by_ext)
    s.symbols_by_ext = dict(symbols_by_ext)
    return s


def test_the_real_1038_shape_is_detected():
    """smokesignals-web: thousands of ts/tsx files, zero symbols. Must be flagged."""
    s = _stats({"ts": 367, "tsx": 320, "js": 367}, {"ts": 0, "tsx": 0, "js": 0})
    assert set(s.hollow_extensions()) == {"js", "ts", "tsx"}


def test_a_healthy_run_is_not_flagged():
    """homek8-shaped: python files that did produce symbols."""
    s = _stats({"py": 120}, {"py": 745})
    assert s.hollow_extensions() == []


def test_yaml_with_zero_code_symbols_is_NOT_a_failure():
    """K8S/YAML flows through the resource extractor. Zero CODE symbols there is
    correct output, and alarming on it is how an alert gets muted."""
    s = _stats({"yaml": 400, "yml": 50}, {})
    assert "yaml" not in s.hollow_extensions()
    assert "yml" not in s.hollow_extensions()


def test_a_single_symbol_free_file_is_NOT_a_failure():
    """One .js file of pure constants is honest, not broken. The gate must not
    false-fire on the small case or it will be switched off."""
    s = _stats({"js": 1}, {"js": 0})
    assert s.hollow_extensions() == []


def test_partial_coverage_flags_only_the_dead_language():
    """python fine, typescript dead -> flag ts alone, not the whole run."""
    s = _stats({"py": 40, "ts": 200}, {"py": 300, "ts": 0})
    assert s.hollow_extensions() == ["ts"]


def test_new_extensions_are_in_the_ingest_gate():
    """The SECOND extension gate. treesitter.py's lang_map is not enough: files are
    `continue`d in ingest_history BEFORE the extractor is called, so .tsx never
    reached it. 320 of 775 real files are .tsx."""
    for ext in ("tsx", "jsx", "mjs", "cjs"):
        assert ext in _CODE_EXTS, (
            "%s missing from _CODE_EXTS - those files are skipped before the "
            "extractor ever sees them" % ext
        )


# --- the SHARED gate both runners call (the fix for the dead-code gate) ------

def test_assert_not_hollow_raises_systemexit_3_on_the_real_shape():
    """This is the gate the PRODUCTION runner (ingest_v3.py) will call. It must
    raise SystemExit(3) - not 1 (a crash), not 0 (silent success) - so a hollow run
    fails loudly and distinguishably."""
    s = _stats({"ts": 367, "tsx": 320, "js": 367}, {"ts": 0, "tsx": 0, "js": 0})
    with pytest.raises(SystemExit) as ei:
        assert_not_hollow(s)
    assert ei.value.code == 3


def test_assert_not_hollow_passes_a_healthy_run():
    s = _stats({"tsx": 320, "ts": 449}, {"tsx": 1591, "ts": 1368})
    assert assert_not_hollow(s) == []


def test_capability_report_flags_only_the_hollow_language():
    """The per-language line that makes the failure unmissable. yaml is a
    'resources' capability, so zero CODE symbols there is NOT hollow."""
    s = _stats({"tsx": 320, "py": 40, "yaml": 100}, {"tsx": 0, "py": 745, "yaml": 0})
    rep = capability_report(s)
    assert rep["tsx"]["hollow"] is True
    assert rep["py"]["hollow"] is False
    assert rep["yaml"]["hollow"] is False
    assert rep["yaml"]["capability"] == "resources"
    assert rep["tsx"]["capability"] == "symbols"


def test_capability_table_matches_the_extension_sets():
    """The capability table must not drift from _CODE_EXTS/_K8S_EXTS."""
    for ext in _CODE_EXTS:
        assert LANGUAGE_CAPABILITIES[ext] == "symbols"
